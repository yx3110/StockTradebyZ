#!/usr/bin/env python3
"""
统一回测: 8个模型版本 × 3种策略 (Composite/Consensus/Refiner)

对每个版本:
1. 加载scorer + head refiner
2. 在~40个采样日全市场打分
3. 计算Composite排名、Consensus投票、Refiner概率
4. 匹配10d forward returns
5. 输出全期+近2月对比表

用法:
    python3 scripts/backtest_all_refiners.py
    python3 scripts/backtest_all_refiners.py --versions v4.8.4 v4.8.7
    python3 scripts/backtest_all_refiners.py --n-dates 20  # 快速模式
"""

import sys, os, json, time, argparse, sqlite3, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'
COST = 0.00302
CONSENSUS_TOP_K = 20

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
    cls = getattr(mod, cls_name)
    return cls(model_type='small_data'), vkey


def load_head_refiner(vkey):
    model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / vkey
    hr_files = sorted(model_dir.glob('head_refiner_*.pkl'))
    if hr_files:
        latest = hr_files[-1]
        try:
            data = joblib.load(latest)
            return data
        except Exception:
            pass
    return None


def compute_meta_and_refiner(scorer, hr_data, X, codes, n):
    """Compute consensus + head refiner scores from per-model predictions."""
    model_names = list(scorer.models.get('10d', {}).keys())
    preds = {}
    for name, model in scorer.models['10d'].items():
        try:
            if name == 'xgb':
                import xgboost as xgb_lib
                preds[name] = model.predict(xgb_lib.DMatrix(X))
            else:
                preds[name] = model.predict(X)
        except Exception:
            continue

    if not preds:
        return np.zeros(n), np.zeros(n), np.zeros(n)

    # Rescale rank models
    regression_names = [nm for nm in preds if nm not in ('lgb_rank', 'lgb_listnet')]
    rank_names = [nm for nm in preds if nm in ('lgb_rank', 'lgb_listnet')]
    if regression_names and rank_names:
        reg_means = [np.mean(preds[nm]) for nm in regression_names]
        reg_stds = [max(np.std(preds[nm]), 1e-8) for nm in regression_names]
        t_mean, t_std = np.mean(reg_means), np.mean(reg_stds)
        for rn in rank_names:
            rp = preds[rn]
            rp_std = max(np.std(rp), 1e-8)
            preds[rn] = (rp - np.mean(rp)) / rp_std * t_std + t_mean

    avail = [nm for nm in model_names if nm in preds and len(preds[nm]) == n]
    if not avail:
        return np.zeros(n), np.zeros(n), np.zeros(n)

    pred_matrix = np.column_stack([preds[nm] for nm in avail])
    n_models = pred_matrix.shape[1]

    # Composite (weighted average using scorer weights)
    target_w = scorer.weights.get('label_10d', {})
    composite = np.zeros(n)
    total_w = 0
    for nm in avail:
        w = target_w.get(nm, 0.2)
        composite += w * preds[nm]
        total_w += w
    if total_w > 0:
        composite /= total_w

    # Consensus
    consensus = np.zeros(n)
    for j in range(n_models):
        ranks = rankdata(-pred_matrix[:, j], method='ordinal')
        consensus += (ranks <= CONSENSUS_TOP_K).astype(float)

    # Head refiner
    hr_proba = np.zeros(n)
    if hr_data is not None:
        meta_feature_names = hr_data.get('meta_feature_names', [])
        stock_feature_names = hr_data.get('stock_feature_names', [])

        meta = {}
        for nm in avail:
            meta[f'pred_{nm}'] = preds[nm]
        meta['pred_mean'] = np.mean(pred_matrix, axis=1)
        meta['pred_std'] = np.std(pred_matrix, axis=1)
        meta['pred_min'] = np.min(pred_matrix, axis=1)
        meta['pred_max'] = np.max(pred_matrix, axis=1)
        meta['pred_range'] = meta['pred_max'] - meta['pred_min']
        meta['pred_sharpe'] = meta['pred_mean'] / np.maximum(meta['pred_std'], 1e-8)
        meta['consensus_count'] = consensus
        meta['rank_in_day'] = rankdata(-meta['pred_mean'], method='average') / n
        rank_mat = np.zeros((n, n_models))
        for j in range(n_models):
            rank_mat[:, j] = rankdata(-pred_matrix[:, j], method='average')
        meta['rank_std'] = np.std(rank_mat, axis=1)
        meta['rank_mean'] = np.mean(rank_mat, axis=1)

        meta_arr = np.column_stack([meta.get(col, np.zeros(n)) for col in meta_feature_names])
        stock_arr = np.zeros((n, len(stock_feature_names)))
        X_hr = np.column_stack([meta_arr, stock_arr])
        X_hr = np.nan_to_num(X_hr, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            hr_proba = hr_data['model'].predict(X_hr)
        except Exception:
            pass

    return composite, consensus, hr_proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--versions', nargs='+',
                        default=['v4.7.5', 'v4.8.1', 'v4.8.2', 'v4.8.3', 'v4.8.4', 'v4.8.5', 'v4.8.6', 'v4.8.7'])
    parser.add_argument('--n-dates', type=int, default=40, help='Number of sample dates')
    parser.add_argument('--start-date', default='2025-01-01')
    parser.add_argument('--end-date', default='2026-03-27')
    args = parser.parse_args()

    # Load dates and prices
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute('PRAGMA busy_timeout=30000')
    all_codes = [r[0] for r in conn.execute("SELECT code FROM securities WHERE type='A股'").fetchall()]
    all_trade_dates = [r[0] for r in conn.execute(
        f"SELECT DISTINCT trade_date FROM v39_feature_cache "
        f"WHERE trade_date >= ? AND trade_date <= ? ORDER BY trade_date",
        (args.start_date, args.end_date)
    ).fetchall()]
    fwd_dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT trade_date FROM daily_quotes ORDER BY trade_date"
    ).fetchall()]
    date_idx = {d: i for i, d in enumerate(fwd_dates)}
    prices_df = pd.read_sql(
        f"SELECT s.code, dq.trade_date, dq.close FROM daily_quotes dq "
        f"JOIN securities s ON s.id=dq.security_id WHERE dq.trade_date >= ?",
        conn, params=(args.start_date,)
    )
    conn.close()
    pm = {(r['code'], r['trade_date']): r['close'] for _, r in prices_df.iterrows()}

    step = max(len(all_trade_dates) // args.n_dates, 1)
    sample_dates = all_trade_dates[::step]
    print(f"Backtesting {len(sample_dates)} dates: {sample_dates[0]} ~ {sample_dates[-1]}")
    print(f"Versions: {args.versions}")

    # Pre-load feature cache for sample dates
    conn2 = sqlite3.connect(str(DB_PATH))
    conn2.execute('PRAGMA busy_timeout=30000')
    ph = ','.join(['?'] * len(sample_dates))
    fc_df = pd.read_sql(
        f"SELECT code, trade_date, features_json FROM v39_feature_cache WHERE trade_date IN ({ph})",
        conn2, params=sample_dates
    )
    conn2.close()
    parsed = fc_df['features_json'].apply(json.loads)
    features_all = pd.DataFrame(parsed.tolist())
    features_all['code'] = fc_df['code'].values
    features_all['trade_date'] = fc_df['trade_date'].values
    print(f"Feature cache loaded: {len(features_all)} rows")

    # Backtest each version
    all_results = {}

    for version in args.versions:
        vkey = VERSION_MAP[version][0]
        print(f"\n{'='*60}")
        print(f"  {version} ({vkey})")
        print(f"{'='*60}")

        scorer, _ = load_scorer(version)
        hr_data = load_head_refiner(vkey)
        hr_status = f"AUC={hr_data['test_auc']:.4f}" if hr_data else "NONE"
        print(f"  Head Refiner: {hr_status}")

        feature_cols = scorer.feature_cols or []

        rows = []
        for di, date in enumerate(sample_dates):
            day_mask = features_all['trade_date'] == date
            day_df = features_all[day_mask].copy()
            if len(day_df) < 100:
                continue

            for col in feature_cols:
                if col not in day_df.columns:
                    day_df[col] = 0

            X = day_df[feature_cols].fillna(0).values if feature_cols else day_df.drop(columns=['code', 'trade_date']).fillna(0).values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            codes = day_df['code'].tolist()
            n = len(codes)

            composite, consensus, hr_proba = compute_meta_and_refiner(scorer, hr_data, X, codes, n)

            for i, code in enumerate(codes):
                if date not in date_idx:
                    continue
                fi = date_idx[date] + 10
                if fi >= len(fwd_dates):
                    continue
                bp, sp = pm.get((code, date)), pm.get((code, fwd_dates[fi]))
                if not bp or not sp or bp <= 0:
                    continue
                fwd_10d = (sp - bp) / bp - 2 * COST
                rows.append({
                    'date': date, 'code': code,
                    'composite': composite[i],
                    'consensus': int(consensus[i]),
                    'refiner': float(hr_proba[i]),
                    'fwd_10d': fwd_10d,
                })

            if (di + 1) % 10 == 0:
                print(f"    [{di+1}/{len(sample_dates)}] {date}")

        df = pd.DataFrame(rows)
        if df.empty:
            print(f"  No data for {version}")
            continue

        n_dates = df['date'].nunique()
        df['rank_c'] = df.groupby('date')['composite'].rank(ascending=False, method='first')
        df['rank_r'] = df.groupby('date')['refiner'].rank(ascending=False, method='first')

        all_results[version] = df
        print(f"  {len(df)} rows, {n_dates} dates")

    # ============================================================
    # Output comparison tables
    # ============================================================
    print(f"\n\n{'='*120}")
    print(f"  全期对比 (10d持仓, {sample_dates[0]} ~ {sample_dates[-1]})")
    print(f"{'='*120}")
    print(f"{'版本':>8} | {'Composite Top-10':>30} | {'Consensus>=2':>30} | {'Refiner Top-10':>30}")
    print(f"{'':>8} | {'avg':>8} {'wr':>6} {'PF':>6} {'n':>5} | {'avg':>8} {'wr':>6} {'PF':>6} {'n':>5} | {'avg':>8} {'wr':>6} {'PF':>6} {'n':>5}")
    print("-" * 120)

    for version in args.versions:
        if version not in all_results:
            continue
        df = all_results[version]
        parts = []
        for label, sub in [
            ('C', df[df['rank_c'] <= 10]),
            ('CC', df[df['consensus'] >= 2]),
            ('R', df[df['rank_r'] <= 10]),
        ]:
            v = sub['fwd_10d']
            if len(v) < 5:
                parts.append(f"{'N/A':>8} {'N/A':>6} {'N/A':>6} {'N/A':>5}")
                continue
            pf = v[v > 0].sum() / max(abs(v[v < 0].sum()), 1e-8)
            parts.append(f"{v.mean():>+8.2%} {(v>0).mean()*100:>5.1f}% {pf:>6.3f} {len(v):>5}")
        print(f"{version:>8} | {parts[0]} | {parts[1]} | {parts[2]}")

    # Near 2 months
    print(f"\n{'='*120}")
    print(f"  近2月对比 (2026-01-27 ~)")
    print(f"{'='*120}")
    print(f"{'版本':>8} | {'Composite Top-10':>30} | {'Consensus>=2':>30} | {'Refiner Top-10':>30}")
    print(f"{'':>8} | {'avg':>8} {'wr':>6} {'PF':>6} {'n':>5} | {'avg':>8} {'wr':>6} {'PF':>6} {'n':>5} | {'avg':>8} {'wr':>6} {'PF':>6} {'n':>5}")
    print("-" * 120)

    for version in args.versions:
        if version not in all_results:
            continue
        df = all_results[version]
        recent = df[df['date'] >= '2026-01-27'].copy()
        if recent.empty:
            continue
        recent['rank_c'] = recent.groupby('date')['composite'].rank(ascending=False, method='first')
        recent['rank_r'] = recent.groupby('date')['refiner'].rank(ascending=False, method='first')

        parts = []
        for label, sub in [
            ('C', recent[recent['rank_c'] <= 10]),
            ('CC', recent[recent['consensus'] >= 2]),
            ('R', recent[recent['rank_r'] <= 10]),
        ]:
            v = sub['fwd_10d']
            if len(v) < 3:
                parts.append(f"{'N/A':>8} {'N/A':>6} {'N/A':>6} {'N/A':>5}")
                continue
            pf = v[v > 0].sum() / max(abs(v[v < 0].sum()), 1e-8)
            parts.append(f"{v.mean():>+8.2%} {(v>0).mean()*100:>5.1f}% {pf:>6.3f} {len(v):>5}")
        print(f"{version:>8} | {parts[0]} | {parts[1]} | {parts[2]}")


if __name__ == '__main__':
    main()
