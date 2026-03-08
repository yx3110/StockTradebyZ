#!/usr/bin/env python3
"""
评分管线消融实验: 量化每个后处理环节对选股质量的影响

实验配置:
  A. 当前管线: Regime权重 + Bear blend + Isotonic
  B. 去Bear:   Regime权重 + 无Bear + 无Isotonic
  C. 固定权重: 固定权重(0.20/0.25/0.35/0.20) + 无Bear + 无Isotonic
  D. 纯10d:   只用10d预测排名 (最简基线)

方法:
  - 模型推理只做一次, 保存4个目标的raw predictions
  - 用不同后处理组合生成composite, 各自选top-10
  - 查找10天实际收益, 计算IC/ICIR/命中率/平均收益

用法:
    python3 backtest/ablation_scoring_pipeline.py
    python3 backtest/ablation_scoring_pipeline.py --start-date 2025-01-01 --end-date 2026-02-13
    python3 backtest/ablation_scoring_pipeline.py --top-n 5
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
from typing import Dict, List, Optional, Tuple
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
    """预加载每个日期每只股票的forward return (T+1买入, T+1+hold_days卖出)"""
    conn = sqlite3.connect(DB_PATH)
    # 获取所有交易日序列
    all_dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT trade_date FROM daily_quotes ORDER BY trade_date"
    ).fetchall()]
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    result = {}
    for date in dates:
        idx = date_to_idx.get(date)
        if idx is None:
            continue
        # T+1 = 买入日, T+1+hold_days = 卖出日
        buy_idx = idx + 1
        sell_idx = idx + 1 + hold_days
        if sell_idx >= len(all_dates):
            continue
        buy_date = all_dates[buy_idx]
        sell_date = all_dates[sell_idx]

        rows = conn.execute("""
            SELECT s.code,
                   q_buy.close AS buy_price,
                   q_sell.close AS sell_price
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


def run_model_inference(scorer, features_df: pd.DataFrame, date: str) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """只做模型推理(Step 1-2), 返回raw predictions, 不做任何后处理"""
    # 特征工程 (与scorer一致)
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

        # Rescale rank models
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

    return codes, predictions, X


def apply_bear_blend(scorer, predictions: Dict[str, np.ndarray],
                     date: str, X: np.ndarray) -> Dict[str, np.ndarray]:
    """应用bear specialist blending, 返回修改后的predictions副本"""
    blended = {k: v.copy() for k, v in predictions.items()}
    if not scorer.bear_models:
        return blended

    market_ret = scorer._get_market_return_20d(date)
    if market_ret is None or market_ret > -0.03:
        return blended

    bear_weight = min(0.60, max(0.15, (abs(market_ret) - 0.03) * 6.4 + 0.15))
    for target_key, bear_model in scorer.bear_models.items():
        try:
            bear_pred = bear_model.predict(X)
        except Exception:
            continue
        if target_key in blended:
            blended[target_key] = (1 - bear_weight) * blended[target_key] + bear_weight * bear_pred

    return blended


def apply_isotonic(scorer, predictions: Dict[str, np.ndarray],
                   codes: List[str]) -> Dict[str, np.ndarray]:
    """应用per-target isotonic校准"""
    calibrated = {k: v.copy() for k, v in predictions.items()}
    if not scorer.isotonic_calibration:
        return calibrated
    for target_key, iso_model in scorer.isotonic_calibration.items():
        if target_key in calibrated:
            try:
                calibrated[target_key] = iso_model.predict(calibrated[target_key])
            except Exception:
                pass
    return calibrated


def compute_composite(predictions: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    return (weights['3d'] * predictions['3d'] +
            weights['5d'] * predictions['5d'] +
            weights['10d'] * predictions['10d'] +
            weights['15d'] * predictions['15d'])


def get_regime_weights(scorer, date: str) -> Dict[str, float]:
    rw = scorer._get_regime_target_weights(date)
    return {
        '3d': rw.get('label_3d', 0.20),
        '5d': rw.get('label_5d', 0.25),
        '10d': rw.get('label_10d', 0.35),
        '15d': rw.get('label_15d', 0.20),
    }


FIXED_WEIGHTS = {'3d': 0.20, '5d': 0.25, '10d': 0.35, '15d': 0.20}
# V4.7.5 实验表明短周期噪声大, 试试偏长周期的权重
LONG_WEIGHTS = {'3d': 0.10, '5d': 0.20, '10d': 0.40, '15d': 0.30}


def main():
    parser = argparse.ArgumentParser(description='Scoring Pipeline Ablation Study')
    parser.add_argument('--start-date', default='2025-01-01')
    parser.add_argument('--end-date', default='2026-02-13')
    parser.add_argument('--top-n', type=int, default=10)
    parser.add_argument('--hold-days', type=int, default=10)
    args = parser.parse_args()

    print(f"=== Scoring Pipeline Ablation Study ===")
    print(f"Period: {args.start_date} ~ {args.end_date}, Top-{args.top_n}, Hold {args.hold_days}d\n")

    # 1. Load scorer
    print("Loading V4.7.3 scorer...")
    from ml_models.v39.v473_production_scorer import V473ProductionScorer
    scorer = V473ProductionScorer()

    # Re-load isotonic from model for config A
    import joblib
    model_dir = Path(PROJECT_ROOT) / 'ml_models' / 'trained_models' / 'v473'
    latest = sorted(model_dir.glob('v473_multi_target_*.pkl'))[-1]
    model_data = joblib.load(latest)
    original_isotonic = model_data.get('isotonic_calibration', {})
    print(f"  Isotonic calibration: {list(original_isotonic.keys()) if original_isotonic else 'none'}")

    # 2. Get dates & preload
    dates = get_trading_dates(args.start_date, args.end_date)
    print(f"\nLoading features for {len(dates)} trading dates...")
    t0 = time.time()
    features_cache = preload_features(dates)
    print(f"  Features loaded in {time.time()-t0:.1f}s")

    print(f"Loading forward returns ({args.hold_days}d)...")
    t0 = time.time()
    fwd_returns = preload_forward_returns(dates, args.hold_days)
    print(f"  Forward returns loaded in {time.time()-t0:.1f}s")

    # 3. Define configs
    configs = {
        'A_regime+bear+iso': {'regime': True, 'bear': True, 'isotonic': True, 'weights': None},
        'B_regime_only':     {'regime': True, 'bear': False, 'isotonic': False, 'weights': None},
        'C_fixed_weights':   {'regime': False, 'bear': False, 'isotonic': False, 'weights': FIXED_WEIGHTS},
        'D_long_weights':    {'regime': False, 'bear': False, 'isotonic': False, 'weights': LONG_WEIGHTS},
        'E_pure_10d':        {'regime': False, 'bear': False, 'isotonic': False, 'weights': {'3d': 0, '5d': 0, '10d': 1.0, '15d': 0}},
    }

    # 4. Run ablation
    # Per-config: {date: [top-n forward returns]}
    config_results = {name: {'ics': [], 'top_returns': [], 'hit_counts': [], 'dates': []}
                      for name in configs}

    valid_dates = 0
    print(f"\nRunning inference + ablation...")
    t0 = time.time()

    for di, date in enumerate(dates):
        fdf = features_cache.get(date)
        fwd = fwd_returns.get(date)
        if fdf is None or fwd is None or len(fdf) < 100:
            continue

        # Model inference (once per date)
        codes, raw_preds, X = run_model_inference(scorer, fdf, date)
        if len(codes) < 50:
            continue

        # Build code->index mapping
        code_to_idx = {c: i for i, c in enumerate(codes)}

        # Forward returns array (aligned to codes)
        actual_returns = np.array([fwd.get(c, np.nan) for c in codes])
        valid_mask = ~np.isnan(actual_returns)
        if valid_mask.sum() < 50:
            continue

        valid_dates += 1

        for config_name, cfg in configs.items():
            # Step 1: optionally apply bear blend
            preds = apply_bear_blend(scorer, raw_preds, date, X) if cfg['bear'] else raw_preds

            # Step 2: optionally apply isotonic
            if cfg['isotonic'] and original_isotonic:
                preds_cal = {k: v.copy() for k, v in preds.items()}
                for tk, iso_model in original_isotonic.items():
                    if tk in preds_cal:
                        try:
                            preds_cal[tk] = iso_model.predict(preds_cal[tk])
                        except Exception:
                            pass
                preds = preds_cal

            # Step 3: compute composite
            if cfg['weights'] is not None:
                w = cfg['weights']
            elif cfg['regime']:
                w = get_regime_weights(scorer, date)
            else:
                w = FIXED_WEIGHTS

            composite = compute_composite(preds, w)

            # IC: rank correlation between composite and actual returns (valid only)
            if valid_mask.sum() > 20:
                ic, _ = stats.spearmanr(composite[valid_mask], actual_returns[valid_mask])
                config_results[config_name]['ics'].append(ic)

            # Top-N selection
            ranked_indices = np.argsort(-composite)  # descending
            top_indices = []
            for idx in ranked_indices:
                if valid_mask[idx]:
                    top_indices.append(idx)
                if len(top_indices) >= args.top_n:
                    break

            if top_indices:
                top_rets = actual_returns[top_indices]
                avg_ret = np.mean(top_rets)
                hits = np.sum(top_rets > 0)
                config_results[config_name]['top_returns'].append(avg_ret)
                config_results[config_name]['hit_counts'].append(hits)
                config_results[config_name]['dates'].append(date)

        if (di + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {di+1}/{len(dates)} dates processed ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\nDone: {valid_dates} valid dates, {elapsed:.1f}s total ({elapsed/max(valid_dates,1):.2f}s/date)")

    # 5. Print results
    print(f"\n{'='*90}")
    print(f"{'Config':<25} {'IC':>7} {'ICIR':>7} {'IC>0%':>7} {'AvgRet':>8} {'AnnRet':>8} {'HitRate':>8} {'Sharpe':>7}")
    print(f"{'='*90}")

    for config_name in configs:
        r = config_results[config_name]
        ics = np.array(r['ics'])
        top_rets = np.array(r['top_returns'])
        hits = np.array(r['hit_counts'])

        ic_mean = np.mean(ics) if len(ics) > 0 else 0
        ic_std = np.std(ics) if len(ics) > 1 else 1
        icir = ic_mean / ic_std if ic_std > 0 else 0
        ic_pos = np.mean(ics > 0) * 100 if len(ics) > 0 else 0

        avg_ret = np.mean(top_rets) if len(top_rets) > 0 else 0
        # Annualized: assume ~245 trading days, rebalance every hold_days
        periods_per_year = 245 / args.hold_days
        ann_ret = (1 + avg_ret) ** periods_per_year - 1
        hit_rate = np.mean(hits / args.top_n) * 100 if len(hits) > 0 else 0

        # Sharpe of top-N returns
        ret_std = np.std(top_rets) if len(top_rets) > 1 else 1
        sharpe = avg_ret / ret_std * np.sqrt(periods_per_year) if ret_std > 0 else 0

        print(f"{config_name:<25} {ic_mean:>7.4f} {icir:>7.3f} {ic_pos:>6.1f}% {avg_ret:>+7.2%} {ann_ret:>+7.1%} {hit_rate:>7.1f}% {sharpe:>7.3f}")

    # 6. Bear blend impact analysis
    print(f"\n--- Bear Blend Impact Analysis ---")
    r_a = config_results['A_regime+bear+iso']
    r_b = config_results['B_regime_only']
    if len(r_a['ics']) == len(r_b['ics']) and len(r_a['ics']) > 0:
        ic_diff = np.array(r_a['ics']) - np.array(r_b['ics'])
        print(f"IC difference (A-B): mean={np.mean(ic_diff):+.4f}, "
              f"A better {np.sum(ic_diff > 0)}/{len(ic_diff)} days ({np.mean(ic_diff > 0)*100:.1f}%)")
        ret_diff = np.array(r_a['top_returns']) - np.array(r_b['top_returns'])
        print(f"Return difference (A-B): mean={np.mean(ret_diff):+.4%}/period, "
              f"A better {np.sum(ret_diff > 0)}/{len(ret_diff)} days ({np.mean(ret_diff > 0)*100:.1f}%)")

    # 7. Regime vs Fixed weights
    print(f"\n--- Regime vs Fixed Weights ---")
    r_c = config_results['C_fixed_weights']
    if len(r_b['ics']) == len(r_c['ics']) and len(r_b['ics']) > 0:
        ic_diff = np.array(r_b['ics']) - np.array(r_c['ics'])
        print(f"IC difference (B_regime - C_fixed): mean={np.mean(ic_diff):+.4f}, "
              f"Regime better {np.sum(ic_diff > 0)}/{len(ic_diff)} days ({np.mean(ic_diff > 0)*100:.1f}%)")
        ret_diff = np.array(r_b['top_returns']) - np.array(r_c['top_returns'])
        print(f"Return difference (B-C): mean={np.mean(ret_diff):+.4%}/period, "
              f"Regime better {np.sum(ret_diff > 0)}/{len(ret_diff)} days ({np.mean(ret_diff > 0)*100:.1f}%)")

    # 8. Monthly breakdown for best config
    print(f"\n--- Monthly Breakdown (best 2 configs) ---")
    # Find best by ICIR
    best_configs = sorted(configs.keys(),
                          key=lambda c: np.mean(config_results[c]['ics']) / max(np.std(config_results[c]['ics']), 0.01)
                          if len(config_results[c]['ics']) > 0 else -999,
                          reverse=True)[:2]

    for config_name in best_configs:
        r = config_results[config_name]
        print(f"\n  [{config_name}]")
        # Group by month
        monthly = {}
        for date, ret, ic in zip(r['dates'], r['top_returns'], r['ics'][:len(r['dates'])]):
            month = date[:7]
            if month not in monthly:
                monthly[month] = {'rets': [], 'ics': []}
            monthly[month]['rets'].append(ret)
            monthly[month]['ics'].append(ic)

        print(f"  {'Month':<10} {'AvgRet':>8} {'IC':>7} {'ICIR':>7} {'Days':>5}")
        for month in sorted(monthly.keys()):
            m = monthly[month]
            m_ic = np.mean(m['ics'])
            m_icir = m_ic / max(np.std(m['ics']), 0.001)
            m_ret = np.mean(m['rets'])
            print(f"  {month:<10} {m_ret:>+7.2%} {m_ic:>7.4f} {m_icir:>7.3f} {len(m['rets']):>5}")


if __name__ == '__main__':
    main()
