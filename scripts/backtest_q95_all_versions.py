#!/usr/bin/env python3
"""
统一回测: 8个版本 × MSE vs Q95-Widen-30-Top-10

直接调用每个版本的scorer（需已集成Q95），对比head_rank vs composite排名。
对于未集成Q95的scorer版本，手动加载Q95模型做回测。
"""
import sys, os, sqlite3, json, time, joblib
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'
COST = 0.00302
WIDEN_K = 30

VERSION_MAP = {
    'v4.7.5': ('v475', 'V475ProductionScorer', 'v475_production_scorer'),
    'v4.8.1': ('v481', 'V481ProductionScorer', 'v481_production_scorer'),
    'v4.8.2': ('v482', 'V482ProductionScorer', 'v482_production_scorer'),
    'v4.8.3': ('v483', 'V483ProductionScorer', 'v483_production_scorer'),
    'v4.8.4': ('v484', 'V484ProductionScorer', 'v484_production_scorer'),
    'v4.8.5': ('v485', 'V485ProductionScorer', 'v485_production_scorer'),
    'v4.8.6': ('v486', 'V486ProductionScorer', 'v486_production_scorer'),
    'v4.8.7': ('v487', 'V487ProductionScorer', 'v487_production_scorer'),
}


def load_scorer(version):
    vkey, cls_name, mod_name = VERSION_MAP[version]
    mod = __import__(f'ml_models.v39.{mod_name}', fromlist=[cls_name])
    return getattr(mod, cls_name)(model_type='small_data'), vkey


def load_q95(vkey):
    model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / vkey
    q95_files = sorted(model_dir.glob('q95_model_*.pkl'))
    if q95_files:
        try:
            data = joblib.load(q95_files[-1])
            return data['models'].get('10d')
        except:
            pass
    return None


def main():
    versions = ['v4.7.5', 'v4.8.1', 'v4.8.2', 'v4.8.3', 'v4.8.4', 'v4.8.5', 'v4.8.6', 'v4.8.7']

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute('PRAGMA busy_timeout=30000')
    all_dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT trade_date FROM v39_feature_cache "
        "WHERE trade_date >= '2025-01-01' AND trade_date <= '2026-03-27' ORDER BY trade_date"
    ).fetchall()]
    fwd_dates_all = [r[0] for r in conn.execute(
        "SELECT DISTINCT trade_date FROM daily_quotes ORDER BY trade_date"
    ).fetchall()]
    date_idx = {d: i for i, d in enumerate(fwd_dates_all)}

    # Preload features
    sample_dates = all_dates[::5]  # every 5th day (~60 dates)
    ph = ','.join(['?'] * len(sample_dates))
    fc = pd.read_sql(f"SELECT code, trade_date, features_json FROM v39_feature_cache WHERE trade_date IN ({ph})",
                     conn, params=sample_dates)
    prices_df = pd.read_sql(
        "SELECT s.code, dq.trade_date, dq.close FROM daily_quotes dq "
        "JOIN securities s ON s.id=dq.security_id WHERE dq.trade_date >= '2025-01-01'", conn)
    conn.close()

    parsed = fc['features_json'].apply(json.loads)
    features_all = pd.DataFrame(parsed.tolist())
    features_all['code'] = fc['code'].values
    features_all['trade_date'] = fc['trade_date'].values
    pm = {(r['code'], r['trade_date']): r['close'] for _, r in prices_df.iterrows()}

    print(f"Backtesting {len(sample_dates)} dates × {len(versions)} versions")

    all_results = {}

    for version in versions:
        vkey, cls_name, mod_name = VERSION_MAP[version]
        print(f"\n{'='*60}")
        print(f"  {version} ({vkey})")

        scorer, _ = load_scorer(version)
        q95_model = load_q95(vkey)
        feature_cols = scorer.feature_cols or []
        print(f"  Q95: {'loaded' if q95_model else 'MISSING'}")

        strats = {'mse_top10': [], 'q95_widen_top10': []}

        for di, date in enumerate(sample_dates):
            day = features_all[features_all['trade_date'] == date]
            if len(day) < 500:
                continue
            for col in feature_cols:
                if col not in day.columns:
                    day[col] = 0
            X = day[feature_cols].fillna(0).values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            codes = day['code'].values
            n = len(codes)

            # MSE ensemble
            tw = scorer.weights.get('label_10d', {})
            mse_pred = np.zeros(n)
            total_w = 0
            for name, model in scorer.models.get('10d', {}).items():
                try:
                    if name == 'xgb':
                        import xgboost as xgb_lib
                        p = model.predict(xgb_lib.DMatrix(X))
                    else:
                        p = model.predict(X)
                    w = tw.get(name, 0.2)
                    mse_pred += w * p
                    total_w += w
                except:
                    continue
            if total_w > 0:
                mse_pred /= total_w

            mse_rank = rankdata(-mse_pred, method='ordinal')

            # Forward returns
            if date not in date_idx:
                continue
            fi = date_idx[date] + 10
            if fi >= len(fwd_dates_all):
                continue
            fd = fwd_dates_all[fi]
            fwd = np.full(n, np.nan)
            for i, code in enumerate(codes):
                bp, sp = pm.get((code, date)), pm.get((code, fd))
                if bp and sp and bp > 0:
                    fwd[i] = (sp - bp) / bp - 2 * COST

            # MSE Top-10
            idx = np.where(mse_rank <= 10)[0]
            rets = [fwd[i] for i in idx if not np.isnan(fwd[i])]
            if rets:
                strats['mse_top10'].append((date, rets))

            # Q95 Widen-30 → Top-10
            if q95_model:
                q95_pred = q95_model.predict(X)
                pool = np.where(mse_rank <= WIDEN_K)[0]
                if len(pool) >= 3:
                    q95_in_pool = q95_pred[pool]
                    top10 = pool[np.argsort(q95_in_pool)[-10:]]
                    rets_q = [fwd[i] for i in top10 if not np.isnan(fwd[i])]
                    if rets_q:
                        strats['q95_widen_top10'].append((date, rets_q))

        all_results[version] = strats
        if (di + 1) % 20 == 0:
            pass

    # ============================================================
    # Output
    # ============================================================
    def compute_metrics(data_list):
        if not data_list:
            return None
        all_r = [r for _, rets in data_list for r in rets]
        dr = [np.mean(rets) for _, rets in data_list]
        rets = np.array(all_r)
        daily = np.array(dr)
        nav = [1.0]
        for r in daily:
            nav.append(nav[-1] * (1 + r))
        nav = np.array(nav)
        years = len(data_list) * 5 / 244
        annual = (nav[-1] ** (1 / max(years, 0.1)) - 1) if nav[-1] > 0 else -1
        peak = np.maximum.accumulate(nav)
        mdd = np.min((nav - peak) / peak)
        sharpe = np.mean(daily) / max(np.std(daily), 1e-8) * np.sqrt(244 / 5) if len(daily) > 1 else 0
        wr = (rets > 0).mean()
        pf = rets[rets > 0].sum() / max(abs(rets[rets < 0].sum()), 1e-8)
        return {'annual': annual, 'mdd': mdd, 'sharpe': sharpe, 'wr': wr, 'pf': pf, 'n': len(rets)}

    print(f"\n\n{'='*130}")
    print(f"  全期对比: MSE Top-10 vs Q95 Widen-30→Top-10 ({sample_dates[0]} ~ {sample_dates[-1]})")
    print(f"{'='*130}")
    print(f"{'版本':>8} | {'--- MSE Ensemble Top-10 ---':>40} | {'--- Q95 Widen-30 → Top-10 ---':>40} | {'Sharpe提升':>10}")
    print(f"{'':>8} | {'年化':>8} {'Sharpe':>7} {'胜率':>6} {'PF':>6} | {'年化':>8} {'Sharpe':>7} {'胜率':>6} {'PF':>6} |")
    print("-" * 130)

    for version in versions:
        strats = all_results.get(version, {})
        m_mse = compute_metrics(strats.get('mse_top10'))
        m_q95 = compute_metrics(strats.get('q95_widen_top10'))

        if m_mse:
            mse_str = f"{m_mse['annual']:>+7.0%} {m_mse['sharpe']:>7.2f} {m_mse['wr']*100:>5.1f}% {m_mse['pf']:>6.3f}"
        else:
            mse_str = f"{'N/A':>30}"

        if m_q95:
            q95_str = f"{m_q95['annual']:>+7.0%} {m_q95['sharpe']:>7.2f} {m_q95['wr']*100:>5.1f}% {m_q95['pf']:>6.3f}"
            if m_mse:
                delta = f"{(m_q95['sharpe']/m_mse['sharpe']-1)*100:>+.0f}%"
            else:
                delta = "N/A"
        else:
            q95_str = f"{'NO Q95':>30}"
            delta = "N/A"

        print(f"{version:>8} | {mse_str:>40} | {q95_str:>40} | {delta:>10}")

    # Near 2 months
    print(f"\n{'='*130}")
    print(f"  近2月: MSE vs Q95 (>= 2026-01-27)")
    print(f"{'='*130}")
    print(f"{'版本':>8} | {'MSE 10d均收':>12} {'胜率':>6} {'PF':>6} | {'Q95 10d均收':>12} {'胜率':>6} {'PF':>6}")
    print("-" * 80)

    for version in versions:
        strats = all_results.get(version, {})
        parts = []
        for sname in ['mse_top10', 'q95_widen_top10']:
            rec = [(d, rets) for d, rets in strats.get(sname, []) if d >= '2026-01-27']
            if rec:
                rr = np.array([r for _, rets in rec for r in rets])
                rpf = rr[rr > 0].sum() / max(abs(rr[rr < 0].sum()), 1e-8)
                parts.append(f"{rr.mean():>+11.2%} {(rr>0).mean()*100:>5.1f}% {rpf:>6.3f}")
            else:
                parts.append(f"{'N/A':>24}")
        print(f"{version:>8} | {parts[0]} | {parts[1]}")


if __name__ == '__main__':
    main()
