#!/usr/bin/env python3
"""
多模型消融实验: 对V4.4/V4.6/V4.7.3分别测试pure 10d vs mixed weights

用法:
    python3 backtest/ablation_multi_model.py --version v4.4
    python3 backtest/ablation_multi_model.py --version v4.6
    python3 backtest/ablation_multi_model.py --version v4.7.3
"""

import sys, os, json, time, sqlite3, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
DB_PATH = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')


def get_trading_dates(start_date, end_date):
    conn = sqlite3.connect(DB_PATH)
    dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT trade_date FROM v39_feature_cache WHERE trade_date >= ? AND trade_date <= ? ORDER BY trade_date",
        (start_date, end_date)).fetchall()]
    conn.close()
    return dates


def preload_features(dates):
    result = {}
    conn = sqlite3.connect(DB_PATH)
    for i in range(0, len(dates), 50):
        chunk = dates[i:i+50]
        ph = ','.join(['?'] * len(chunk))
        df = pd.read_sql_query(f"""
            SELECT code, trade_date, features_json,
                   market_return_20d, market_return_10d, market_return_5d,
                   market_volatility_20d, market_volatility_10d,
                   market_up_ratio_20d, market_up_ratio_10d,
                   market_drawdown_20d, market_volume_ratio,
                   market_position_20d, market_momentum_20d, market_momentum_5d
            FROM v39_feature_cache WHERE trade_date IN ({ph})
        """, conn, params=chunk)
        if df.empty:
            continue
        parsed = df['features_json'].apply(json.loads)
        features_all = pd.DataFrame(parsed.tolist())
        features_all['code'] = df['code'].values
        features_all['trade_date'] = df['trade_date'].values
        for col in [c for c in df.columns if c.startswith('market_')]:
            features_all[col] = df[col].values
        for date, group in features_all.groupby('trade_date'):
            result[date] = group.drop(columns=['trade_date']).reset_index(drop=True)
    conn.close()
    return result


def preload_forward_returns(dates, hold_days=10):
    conn = sqlite3.connect(DB_PATH)
    all_dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT trade_date FROM daily_quotes ORDER BY trade_date").fetchall()]
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    result = {}
    for date in dates:
        idx = date_to_idx.get(date)
        if idx is None:
            continue
        buy_idx = idx + 1
        sell_idx = buy_idx + hold_days
        if sell_idx >= len(all_dates):
            continue
        buy_date, sell_date = all_dates[buy_idx], all_dates[sell_idx]
        rows = conn.execute("""
            SELECT s.code, q_buy.close, q_sell.close
            FROM daily_quotes q_buy
            JOIN daily_quotes q_sell ON q_buy.security_id = q_sell.security_id
            JOIN securities s ON q_buy.security_id = s.id
            WHERE q_buy.trade_date = ? AND q_sell.trade_date = ?
              AND q_buy.close > 0 AND q_sell.close > 0
        """, (buy_date, sell_date)).fetchall()
        result[date] = {code: (sp - bp) / bp for code, bp, sp in rows}
    conn.close()
    return result


def load_scorer(version):
    if version == 'v4.4':
        from ml_models.v39.v44_production_scorer import V44ProductionScorer
        return V44ProductionScorer()
    elif version == 'v4.6':
        from ml_models.v39.v46_production_scorer import V46ProductionScorer
        return V46ProductionScorer()
    elif version == 'v4.7.3':
        from ml_models.v39.v473_production_scorer import V473ProductionScorer
        return V473ProductionScorer()
    elif version == 'v4.7.5':
        from ml_models.v39.v475_production_scorer import V475ProductionScorer
        return V475ProductionScorer()
    else:
        raise ValueError(f"Unknown version: {version}")


def score_date(scorer, features_df, date):
    """Use scorer's predict_scores_from_preloaded to get predictions"""
    all_codes = features_df['code'].tolist()
    results = scorer.predict_scores_from_preloaded(all_codes, date, features_df)
    codes = []
    preds = {'3d': [], '5d': [], '10d': [], '15d': []}
    for code in all_codes:
        if code in results:
            codes.append(code)
            for t in ['3d', '5d', '10d', '15d']:
                preds[t].append(results[code].get(f'pred_{t}', 0))
    preds = {t: np.array(v) for t, v in preds.items()}
    return codes, preds


def evaluate_config(daily_data, weights, top_n=10):
    ics, top_rets = [], []
    for dd in daily_data:
        preds, actual, valid = dd['predictions'], dd['actual_returns'], dd['valid_mask']
        composite = sum(weights[t] * preds[t] for t in ['3d', '5d', '10d', '15d'])
        if valid.sum() > 20:
            ic, _ = stats.spearmanr(composite[valid], actual[valid])
            ics.append(ic)
        ranked = np.argsort(-composite)
        top_idx = [idx for idx in ranked if valid[idx]][:top_n]
        if top_idx:
            top_rets.append(np.mean(actual[top_idx]))
    if not ics:
        return None
    ics, top_rets = np.array(ics), np.array(top_rets)
    ic_mean = np.mean(ics)
    icir = ic_mean / np.std(ics) if np.std(ics) > 0 else 0
    avg_ret = np.mean(top_rets)
    ann_ret = (1 + avg_ret) ** 24.5 - 1
    sharpe = avg_ret / np.std(top_rets) * np.sqrt(24.5) if np.std(top_rets) > 0 else 0
    return {'ic': ic_mean, 'icir': icir, 'ic_pos': np.mean(ics > 0),
            'avg_ret': avg_ret, 'ann_ret': ann_ret, 'sharpe': sharpe,
            'hit_rate': np.mean(top_rets > 0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', required=True, choices=['v4.4', 'v4.6', 'v4.7.3', 'v4.7.5'])
    parser.add_argument('--start-date', default='2024-01-01')
    parser.add_argument('--end-date', default='2026-02-13')
    parser.add_argument('--top-n', type=int, default=10)
    args = parser.parse_args()

    print(f"=== Ablation: {args.version} ===")
    print(f"Period: {args.start_date} ~ {args.end_date}, Top-{args.top_n}\n")

    scorer = load_scorer(args.version)

    dates = get_trading_dates(args.start_date, args.end_date)
    print(f"\nLoading data ({len(dates)} dates)...")
    sys.stdout.flush()
    features_cache = preload_features(dates)
    fwd_returns = preload_forward_returns(dates)

    # Phase 1: Score all dates (uses scorer's full pipeline incl. bear/isotonic/meta-learner)
    print(f"Scoring...")
    sys.stdout.flush()
    t0 = time.time()
    daily_data = []
    for di, date in enumerate(dates):
        fdf = features_cache.get(date)
        fwd = fwd_returns.get(date)
        if fdf is None or fwd is None or len(fdf) < 100:
            continue
        codes, preds = score_date(scorer, fdf, date)
        if len(codes) < 50:
            continue
        actual = np.array([fwd.get(c, np.nan) for c in codes])
        valid = ~np.isnan(actual)
        if valid.sum() < 50:
            continue
        daily_data.append({
            'date': date, 'codes': codes,
            'predictions': preds, 'actual_returns': actual, 'valid_mask': valid,
        })
        if (di + 1) % 50 == 0:
            print(f"  {di+1}/{len(dates)} ({time.time()-t0:.0f}s)")
            sys.stdout.flush()
    print(f"  Done: {len(daily_data)} valid dates in {time.time()-t0:.0f}s\n")
    sys.stdout.flush()

    # Phase 2: Evaluate different weight configs
    configs = {
        'Pure 3d':          {'3d': 1.0, '5d': 0.0, '10d': 0.0, '15d': 0.0},
        'Pure 5d':          {'3d': 0.0, '5d': 1.0, '10d': 0.0, '15d': 0.0},
        'Pure 10d':         {'3d': 0.0, '5d': 0.0, '10d': 1.0, '15d': 0.0},
        'Pure 15d':         {'3d': 0.0, '5d': 0.0, '10d': 0.0, '15d': 1.0},
        'Default mix':      {'3d': 0.20, '5d': 0.25, '10d': 0.35, '15d': 0.20},
        'Long mix':         {'3d': 0.10, '5d': 0.20, '10d': 0.40, '15d': 0.30},
        '10d+15d':          {'3d': 0.00, '5d': 0.00, '10d': 0.60, '15d': 0.40},
        '5d+10d':           {'3d': 0.00, '5d': 0.30, '10d': 0.70, '15d': 0.00},
        '10d(95)+5d(5)':    {'3d': 0.00, '5d': 0.05, '10d': 0.95, '15d': 0.00},
    }

    print(f"--- {args.version} Weight Ablation Results ---")
    print(f"{'Config':<22} {'IC':>7} {'ICIR':>7} {'IC>0%':>7} {'AvgRet':>8} {'AnnRet':>8} {'Sharpe':>7} {'HitRate':>8}")
    print("=" * 85)
    for name, w in configs.items():
        r = evaluate_config(daily_data, w, args.top_n)
        if r:
            print(f"{name:<22} {r['ic']:>7.4f} {r['icir']:>7.3f} {r['ic_pos']:>6.1%} "
                  f"{r['avg_ret']:>+7.2%} {r['ann_ret']:>+7.1%} {r['sharpe']:>7.3f} {r['hit_rate']:>7.1%}")
    sys.stdout.flush()

    # Cross-target IC (pred_X vs actual_10d)
    print(f"\n--- Cross-Target IC: pred_X → actual_10d ---")
    for target in ['3d', '5d', '10d', '15d']:
        ics = []
        for dd in daily_data:
            p = dd['predictions'][target]
            a = dd['actual_returns']
            v = dd['valid_mask']
            if v.sum() > 20:
                ic, _ = stats.spearmanr(p[v], a[v])
                ics.append(ic)
        ics = np.array(ics)
        icir = np.mean(ics) / np.std(ics) if np.std(ics) > 0 else 0
        print(f"  pred_{target}: IC={np.mean(ics):+.4f}, ICIR={icir:.3f}, IC>0={np.mean(ics>0)*100:.1f}%")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
