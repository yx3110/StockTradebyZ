#!/usr/bin/env python3
"""
Composite权重网格搜索: 找到3d/5d/10d/15d的最优组合权重

方法:
  1. 模型推理只做一次, 保存每天每只股票的4个raw predictions
  2. 网格搜索数百种权重组合, 每种只需向量运算
  3. 用ICIR/Sharpe/年化收益综合排序

用法:
    python3 backtest/grid_search_weights.py
    python3 backtest/grid_search_weights.py --start-date 2020-06-01 --granularity 0.05
"""

import sys
import os
import json
import time
import sqlite3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from typing import Dict, List, Tuple
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')


def get_trading_dates(start_date: str, end_date: str) -> List[str]:
    conn = sqlite3.connect(DB_PATH)
    dates = [r[0] for r in conn.execute("""
        SELECT DISTINCT trade_date FROM v39_feature_cache
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date
    """, (start_date, end_date)).fetchall()]
    conn.close()
    return dates


def preload_features(dates: List[str]) -> Dict[str, pd.DataFrame]:
    result = {}
    conn = sqlite3.connect(DB_PATH)
    CHUNK = 50
    for i in range(0, len(dates), CHUNK):
        chunk = dates[i:i+CHUNK]
        ph = ','.join(['?'] * len(chunk))
        df = pd.read_sql_query(f"""
            SELECT code, trade_date, features_json,
                   market_return_20d, market_return_10d, market_return_5d,
                   market_volatility_20d, market_volatility_10d,
                   market_up_ratio_20d, market_up_ratio_10d,
                   market_drawdown_20d, market_volume_ratio,
                   market_position_20d, market_momentum_20d, market_momentum_5d
            FROM v39_feature_cache
            WHERE trade_date IN ({ph})
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


def preload_forward_returns(dates: List[str], hold_days: int = 10) -> Dict[str, Dict[str, float]]:
    conn = sqlite3.connect(DB_PATH)
    all_dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT trade_date FROM daily_quotes ORDER BY trade_date"
    ).fetchall()]
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    result = {}
    for date in dates:
        idx = date_to_idx.get(date)
        if idx is None:
            continue
        buy_idx = idx + 1
        sell_idx = idx + 1 + hold_days
        if sell_idx >= len(all_dates):
            continue
        buy_date = all_dates[buy_idx]
        sell_date = all_dates[sell_idx]

        rows = conn.execute("""
            SELECT s.code, q_buy.close, q_sell.close
            FROM daily_quotes q_buy
            JOIN daily_quotes q_sell ON q_buy.security_id = q_sell.security_id
            JOIN securities s ON q_buy.security_id = s.id
            WHERE q_buy.trade_date = ? AND q_sell.trade_date = ?
              AND q_buy.close > 0 AND q_sell.close > 0
        """, (buy_date, sell_date)).fetchall()

        fwd = {}
        for code, buy_p, sell_p in rows:
            fwd[code] = (sell_p - buy_p) / buy_p
        result[date] = fwd
    conn.close()
    return result


def run_model_inference(scorer, features_df: pd.DataFrame, date: str):
    """模型推理, 返回 (codes, {target: predictions_array})"""
    df = features_df.copy()
    df = scorer._robust_zscore_normalize_features(df)
    df = scorer._load_daily_basic_features(df, date)
    df = scorer._load_technical_features(df, date)
    df = scorer._load_financial_features(df, date)
    df = scorer._load_daily_basic_extra(df, date)
    df = scorer._compute_microstructure_features(df, date)

    exclude_cols = {'code', 'trade_date'}
    if scorer.feature_cols:
        for col in scorer.feature_cols:
            if col not in df.columns:
                df[col] = 0
        available_cols = scorer.feature_cols
    else:
        available_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[available_cols].fillna(0).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    codes = df['code'].tolist()

    predictions = {}
    for target in ['3d', '5d', '10d', '15d']:
        if target not in scorer.models or not scorer.models[target]:
            predictions[target] = np.zeros(len(X))
            continue
        target_pred = np.zeros(len(X))
        total_weight = 0

        preds = {}
        for name, model in scorer.models[target].items():
            try:
                if name == 'xgb':
                    import xgboost as xgb_lib
                    preds[name] = model.predict(xgb_lib.DMatrix(X))
                else:
                    preds[name] = model.predict(X)
            except Exception:
                continue

        regression_names = [n for n in preds if n not in ('lgb_rank', 'lgb_listnet')]
        rank_names = [n for n in preds if n in ('lgb_rank', 'lgb_listnet')]
        if regression_names and rank_names:
            reg_means = [np.mean(preds[n]) for n in regression_names]
            reg_stds = [max(np.std(preds[n]), 1e-8) for n in regression_names]
            t_mean, t_std = np.mean(reg_means), np.mean(reg_stds)
            for rn in rank_names:
                rp = preds[rn]
                rp_std = max(np.std(rp), 1e-8)
                preds[rn] = (rp - np.mean(rp)) / rp_std * t_std + t_mean

        target_w = scorer.weights.get(f'label_{target}', {})
        for name, pred in preds.items():
            weight = target_w.get(name, 0.2)
            target_pred += weight * pred
            total_weight += weight

        if total_weight > 0:
            target_pred /= total_weight
        predictions[target] = target_pred

    return codes, predictions


def generate_weight_grid(granularity: float = 0.05) -> List[Tuple[float, float, float, float]]:
    """生成所有(w3d, w5d, w10d, w15d)组合, 权重和为1"""
    steps = int(1.0 / granularity) + 1
    values = [round(i * granularity, 2) for i in range(steps)]
    combos = []
    for w3 in values:
        for w5 in values:
            for w10 in values:
                w15 = round(1.0 - w3 - w5 - w10, 2)
                if w15 >= 0 and w15 <= 1.0:
                    combos.append((w3, w5, w10, w15))
    return combos


def evaluate_weights(daily_data: List[dict], weights: Tuple[float, float, float, float],
                     top_n: int = 10) -> dict:
    """对一组权重评估IC/ICIR/Sharpe/年化"""
    w3, w5, w10, w15 = weights
    ics = []
    top_rets = []

    for dd in daily_data:
        preds = dd['predictions']
        actual = dd['actual_returns']
        valid = dd['valid_mask']

        composite = w3 * preds['3d'] + w5 * preds['5d'] + w10 * preds['10d'] + w15 * preds['15d']

        # IC
        if valid.sum() > 20:
            ic, _ = stats.spearmanr(composite[valid], actual[valid])
            ics.append(ic)

        # Top-N
        ranked = np.argsort(-composite)
        top_idx = []
        for idx in ranked:
            if valid[idx]:
                top_idx.append(idx)
            if len(top_idx) >= top_n:
                break
        if top_idx:
            top_rets.append(np.mean(actual[top_idx]))

    if not ics or not top_rets:
        return None

    ics = np.array(ics)
    top_rets = np.array(top_rets)

    ic_mean = np.mean(ics)
    ic_std = np.std(ics)
    icir = ic_mean / ic_std if ic_std > 0 else 0
    ic_pos = np.mean(ics > 0)

    avg_ret = np.mean(top_rets)
    periods_per_year = 245 / 10  # hold_days=10
    ann_ret = (1 + avg_ret) ** periods_per_year - 1
    ret_std = np.std(top_rets)
    sharpe = avg_ret / ret_std * np.sqrt(periods_per_year) if ret_std > 0 else 0
    hit_rate = np.mean(top_rets > 0)

    return {
        'weights': weights,
        'ic': ic_mean,
        'icir': icir,
        'ic_pos': ic_pos,
        'avg_ret': avg_ret,
        'ann_ret': ann_ret,
        'sharpe': sharpe,
        'hit_rate': hit_rate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', default='2020-06-01')
    parser.add_argument('--end-date', default='2026-02-13')
    parser.add_argument('--top-n', type=int, default=10)
    parser.add_argument('--hold-days', type=int, default=10)
    parser.add_argument('--granularity', type=float, default=0.05,
                        help='Weight grid step size (0.05=231 combos, 0.10=286 combos)')
    args = parser.parse_args()

    weight_grid = generate_weight_grid(args.granularity)
    print(f"=== Composite Weight Grid Search ===")
    print(f"Period: {args.start_date} ~ {args.end_date}, Top-{args.top_n}, Hold {args.hold_days}d")
    print(f"Grid: step={args.granularity}, {len(weight_grid)} weight combinations\n")

    # Load scorer
    print("Loading V4.7.3 scorer...")
    from ml_models.v39.v473_production_scorer import V473ProductionScorer
    scorer = V473ProductionScorer()

    # Load data
    dates = get_trading_dates(args.start_date, args.end_date)
    print(f"\nLoading features for {len(dates)} dates...")
    t0 = time.time()
    features_cache = preload_features(dates)
    print(f"  Features: {time.time()-t0:.1f}s")

    t0 = time.time()
    fwd_returns = preload_forward_returns(dates, args.hold_days)
    print(f"  Forward returns: {time.time()-t0:.1f}s")

    # Phase 1: Model inference (one pass, save predictions)
    print(f"\nPhase 1: Model inference...")
    t0 = time.time()
    daily_data = []
    for di, date in enumerate(dates):
        fdf = features_cache.get(date)
        fwd = fwd_returns.get(date)
        if fdf is None or fwd is None or len(fdf) < 100:
            continue

        codes, preds = run_model_inference(scorer, fdf, date)
        if len(codes) < 50:
            continue

        actual = np.array([fwd.get(c, np.nan) for c in codes])
        valid = ~np.isnan(actual)
        if valid.sum() < 50:
            continue

        daily_data.append({
            'date': date,
            'codes': codes,
            'predictions': preds,
            'actual_returns': actual,
            'valid_mask': valid,
        })

        if (di + 1) % 100 == 0:
            print(f"  {di+1}/{len(dates)} dates ({time.time()-t0:.0f}s)")

    print(f"  Done: {len(daily_data)} valid dates in {time.time()-t0:.0f}s")

    # Phase 2: Grid search (fast, pure numpy)
    print(f"\nPhase 2: Evaluating {len(weight_grid)} weight combinations...")
    t0 = time.time()
    results = []
    for wi, weights in enumerate(weight_grid):
        r = evaluate_weights(daily_data, weights, args.top_n)
        if r:
            results.append(r)
        if (wi + 1) % 50 == 0:
            print(f"  {wi+1}/{len(weight_grid)} combos ({time.time()-t0:.1f}s)")

    print(f"  Done: {len(results)} valid results in {time.time()-t0:.1f}s")

    # Sort by different criteria
    print(f"\n{'='*100}")

    # Top 15 by Sharpe
    print(f"\n--- Top 15 by Sharpe ---")
    print(f"{'Rank':>4} {'w3d':>5} {'w5d':>5} {'w10d':>5} {'w15d':>5} | {'IC':>7} {'ICIR':>7} {'IC>0%':>6} {'AvgRet':>8} {'AnnRet':>8} {'Sharpe':>7} {'HitRate':>8}")
    print(f"{'-'*100}")
    by_sharpe = sorted(results, key=lambda x: x['sharpe'], reverse=True)
    for i, r in enumerate(by_sharpe[:15]):
        w = r['weights']
        print(f"{i+1:>4} {w[0]:>5.2f} {w[1]:>5.2f} {w[2]:>5.2f} {w[3]:>5.2f} | "
              f"{r['ic']:>7.4f} {r['icir']:>7.3f} {r['ic_pos']:>5.1%} {r['avg_ret']:>+7.2%} "
              f"{r['ann_ret']:>+7.1%} {r['sharpe']:>7.3f} {r['hit_rate']:>7.1%}")

    # Top 15 by ICIR
    print(f"\n--- Top 15 by ICIR ---")
    print(f"{'Rank':>4} {'w3d':>5} {'w5d':>5} {'w10d':>5} {'w15d':>5} | {'IC':>7} {'ICIR':>7} {'IC>0%':>6} {'AvgRet':>8} {'AnnRet':>8} {'Sharpe':>7} {'HitRate':>8}")
    print(f"{'-'*100}")
    by_icir = sorted(results, key=lambda x: x['icir'], reverse=True)
    for i, r in enumerate(by_icir[:15]):
        w = r['weights']
        print(f"{i+1:>4} {w[0]:>5.2f} {w[1]:>5.2f} {w[2]:>5.2f} {w[3]:>5.2f} | "
              f"{r['ic']:>7.4f} {r['icir']:>7.3f} {r['ic_pos']:>5.1%} {r['avg_ret']:>+7.2%} "
              f"{r['ann_ret']:>+7.1%} {r['sharpe']:>7.3f} {r['hit_rate']:>7.1%}")

    # Top 15 by Annual Return
    print(f"\n--- Top 15 by Annual Return ---")
    print(f"{'Rank':>4} {'w3d':>5} {'w5d':>5} {'w10d':>5} {'w15d':>5} | {'IC':>7} {'ICIR':>7} {'IC>0%':>6} {'AvgRet':>8} {'AnnRet':>8} {'Sharpe':>7} {'HitRate':>8}")
    print(f"{'-'*100}")
    by_ann = sorted(results, key=lambda x: x['ann_ret'], reverse=True)
    for i, r in enumerate(by_ann[:15]):
        w = r['weights']
        print(f"{i+1:>4} {w[0]:>5.2f} {w[1]:>5.2f} {w[2]:>5.2f} {w[3]:>5.2f} | "
              f"{r['ic']:>7.4f} {r['icir']:>7.3f} {r['ic_pos']:>5.1%} {r['avg_ret']:>+7.2%} "
              f"{r['ann_ret']:>+7.1%} {r['sharpe']:>7.3f} {r['hit_rate']:>7.1%}")

    # Reference points
    print(f"\n--- Reference Configs ---")
    ref_weights = {
        'Current (0.20/0.25/0.35/0.20)': (0.20, 0.25, 0.35, 0.20),
        'Long (0.10/0.20/0.40/0.30)': (0.10, 0.20, 0.40, 0.30),
        'Pure 10d': (0.00, 0.00, 1.00, 0.00),
    }
    print(f"{'Config':<35} | {'IC':>7} {'ICIR':>7} {'IC>0%':>6} {'AvgRet':>8} {'AnnRet':>8} {'Sharpe':>7}")
    print(f"{'-'*90}")
    for name, w in ref_weights.items():
        r = evaluate_weights(daily_data, w, args.top_n)
        if r:
            print(f"{name:<35} | {r['ic']:>7.4f} {r['icir']:>7.3f} {r['ic_pos']:>5.1%} "
                  f"{r['avg_ret']:>+7.2%} {r['ann_ret']:>+7.1%} {r['sharpe']:>7.3f}")

    # Heatmap: 10d weight vs other weights (fixing ratio of others)
    print(f"\n--- 10d Weight Sensitivity (others proportional) ---")
    print(f"{'w10d':>6} {'w3d':>5} {'w5d':>5} {'w15d':>5} | {'IC':>7} {'ICIR':>7} {'Sharpe':>7} {'AnnRet':>8}")
    print(f"{'-'*70}")
    for w10 in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # Distribute remaining weight proportionally (20:25:20 ratio for 3d:5d:15d)
        remain = 1.0 - w10
        w3 = round(remain * 20/65, 2)
        w5 = round(remain * 25/65, 2)
        w15 = round(remain - w3 - w5, 2)
        if w15 < 0:
            w15 = 0
            w5 = round(remain - w3, 2)
        r = evaluate_weights(daily_data, (w3, w5, w10, w15), args.top_n)
        if r:
            print(f"{w10:>6.2f} {w3:>5.2f} {w5:>5.2f} {w15:>5.2f} | "
                  f"{r['ic']:>7.4f} {r['icir']:>7.3f} {r['sharpe']:>7.3f} {r['ann_ret']:>+7.1%}")


if __name__ == '__main__':
    main()
