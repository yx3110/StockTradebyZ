#!/usr/bin/env python3
"""
诊断为什么10d预测在选股中远优于3d/5d/15d

分析维度:
  1. 各目标对自身horizon的IC vs 对10d实际收益的IC (cross-target IC)
  2. Top-10选股重叠度: 不同目标选出的股票有多大交集?
  3. 预测稳定性: 各目标的日间排名变化率
  4. 收益持续性: 3d高预测的股票在10d后还好吗?
  5. 噪声分析: 各horizon实际收益的截面方差和信噪比
"""

import sys, os, json, time, sqlite3
import numpy as np
import pandas as pd
from typing import Dict, List
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


def preload_multi_horizon_returns(dates, horizons=[3, 5, 10, 15]):
    """预加载多个horizon的forward returns"""
    conn = sqlite3.connect(DB_PATH)
    all_dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT trade_date FROM daily_quotes ORDER BY trade_date").fetchall()]
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    result = {h: {} for h in horizons}
    for date in dates:
        idx = date_to_idx.get(date)
        if idx is None:
            continue
        buy_idx = idx + 1
        for h in horizons:
            sell_idx = buy_idx + h
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
            for code, bp, sp in rows:
                fwd[code] = (sp - bp) / bp
            result[h][date] = fwd
    conn.close()
    return result


def run_model_inference(scorer, features_df, date):
    df = features_df.copy()
    df = scorer._robust_zscore_normalize_features(df)
    df = scorer._load_daily_basic_features(df, date)
    df = scorer._load_technical_features(df, date)
    df = scorer._load_financial_features(df, date)
    df = scorer._load_daily_basic_extra(df, date)
    df = scorer._compute_microstructure_features(df, date)

    if scorer.feature_cols:
        for col in scorer.feature_cols:
            if col not in df.columns:
                df[col] = 0
        available_cols = scorer.feature_cols
    else:
        available_cols = [c for c in df.columns if c not in {'code', 'trade_date'}]

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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', default='2024-01-01')
    parser.add_argument('--end-date', default='2026-02-13')
    parser.add_argument('--top-n', type=int, default=10)
    args = parser.parse_args()

    print(f"=== 10d Dominance Diagnosis ===")
    print(f"Period: {args.start_date} ~ {args.end_date}\n")

    from ml_models.v39.v473_production_scorer import V473ProductionScorer
    scorer = V473ProductionScorer()

    dates = get_trading_dates(args.start_date, args.end_date)
    print(f"Loading features ({len(dates)} dates)...")
    features_cache = preload_features(dates)
    print(f"Loading multi-horizon forward returns...")
    fwd_all = preload_multi_horizon_returns(dates, [3, 5, 10, 15])

    # Collect per-date data
    targets = ['3d', '5d', '10d', '15d']
    horizons = [3, 5, 10, 15]

    # Accumulators
    # 1. Cross-target IC matrix: pred_X vs actual_Y
    cross_ic = {t: {h: [] for h in horizons} for t in targets}
    # 2. Top-N overlap
    overlap_counts = {t1: {t2: [] for t2 in targets} for t1 in targets}
    # 3. Prediction stability (rank correlation with previous day)
    prev_preds = {t: None for t in targets}
    prev_codes = None
    rank_stability = {t: [] for t in targets}
    # 4. Top-N actual returns at each horizon
    topn_returns = {t: {h: [] for h in horizons} for t in targets}
    # 5. Cross-section stats
    pred_std = {t: [] for t in targets}
    actual_std = {h: [] for h in horizons}
    # 6. Prediction correlation between targets
    pred_corr = {(t1, t2): [] for t1 in targets for t2 in targets if t1 < t2}

    print(f"Running inference...")
    t0 = time.time()

    for di, date in enumerate(dates):
        fdf = features_cache.get(date)
        if fdf is None or len(fdf) < 100:
            continue

        # Check all horizons have data
        has_all = all(date in fwd_all[h] and len(fwd_all[h][date]) > 50 for h in horizons)
        if not has_all:
            continue

        codes, preds = run_model_inference(scorer, fdf, date)
        if len(codes) < 50:
            continue

        code_to_idx = {c: i for i, c in enumerate(codes)}
        n = len(codes)

        # Get actual returns for all horizons
        actuals = {}
        valid_masks = {}
        for h in horizons:
            fwd = fwd_all[h].get(date, {})
            arr = np.array([fwd.get(c, np.nan) for c in codes])
            actuals[h] = arr
            valid_masks[h] = ~np.isnan(arr)

        # Common valid mask (valid for all horizons)
        common_valid = np.ones(n, dtype=bool)
        for h in horizons:
            common_valid &= valid_masks[h]
        if common_valid.sum() < 50:
            continue

        # 1. Cross-target IC: pred_target vs actual_horizon
        for t in targets:
            t_h = int(t.replace('d', ''))
            p = preds[t][common_valid]
            for h in horizons:
                a = actuals[h][common_valid]
                ic, _ = stats.spearmanr(p, a)
                cross_ic[t][h].append(ic)

        # 2. Top-N overlap between targets
        top_sets = {}
        for t in targets:
            ranked = np.argsort(-preds[t])
            top_idx = []
            for idx in ranked:
                if common_valid[idx]:
                    top_idx.append(idx)
                if len(top_idx) >= args.top_n:
                    break
            top_sets[t] = set(top_idx)

        for t1 in targets:
            for t2 in targets:
                overlap = len(top_sets[t1] & top_sets[t2])
                overlap_counts[t1][t2].append(overlap)

        # 3. Prediction stability
        if prev_codes is not None and prev_codes == codes:
            for t in targets:
                rho, _ = stats.spearmanr(preds[t], prev_preds[t])
                rank_stability[t].append(rho)

        prev_preds = {t: preds[t].copy() for t in targets}
        prev_codes = codes

        # 4. Top-N returns at each horizon (selected by each target)
        for t in targets:
            top_idx = list(top_sets[t])
            for h in horizons:
                rets = actuals[h][top_idx]
                topn_returns[t][h].append(np.mean(rets))

        # 5. Cross-section variance
        for t in targets:
            pred_std[t].append(np.std(preds[t][common_valid]))
        for h in horizons:
            actual_std[h].append(np.std(actuals[h][common_valid]))

        # 6. Prediction correlation between targets
        for t1 in targets:
            for t2 in targets:
                if t1 < t2:
                    rho, _ = stats.spearmanr(preds[t1][common_valid], preds[t2][common_valid])
                    pred_corr[(t1, t2)].append(rho)

        if (di + 1) % 50 == 0:
            print(f"  {di+1}/{len(dates)} ({time.time()-t0:.0f}s)")

    elapsed = time.time() - t0
    n_days = len(cross_ic['3d'][3])
    print(f"  Done: {n_days} valid dates in {elapsed:.0f}s\n")

    # === RESULTS ===

    # 1. Cross-Target IC Matrix
    print("=" * 80)
    print("1. CROSS-TARGET IC MATRIX")
    print("   Rows = prediction target, Columns = actual return horizon")
    print("   Shows: which prediction best predicts which horizon\n")
    print(f"{'Pred \\ Actual':>15}", end='')
    for h in horizons:
        print(f"  actual_{h}d", end='')
    print(f"  {'(own ICIR)':>10}")
    print("-" * 75)
    for t in targets:
        t_h = int(t.replace('d', ''))
        print(f"  pred_{t:>3}      ", end='')
        for h in horizons:
            ic = np.mean(cross_ic[t][h])
            marker = " *" if h == t_h else "  "
            print(f"  {ic:>+.4f}{marker}", end='')
        # Own-target ICIR
        own_ics = np.array(cross_ic[t][t_h])
        own_icir = np.mean(own_ics) / np.std(own_ics) if np.std(own_ics) > 0 else 0
        print(f"  {own_icir:>8.3f}")

    # Cross-target ICIR for predicting 10d actual
    print(f"\n  ICIR for predicting 10d actual returns:")
    for t in targets:
        ics_10d = np.array(cross_ic[t][10])
        icir_10d = np.mean(ics_10d) / np.std(ics_10d) if np.std(ics_10d) > 0 else 0
        print(f"    pred_{t}: IC={np.mean(ics_10d):+.4f}, ICIR={icir_10d:.3f}, IC>0={np.mean(ics_10d > 0)*100:.1f}%")

    # 2. Top-N Overlap Matrix
    print(f"\n{'=' * 80}")
    print(f"2. TOP-{args.top_n} OVERLAP MATRIX (avg stocks in common)")
    print(f"   Shows: how much different targets agree on top picks\n")
    print(f"{'':>10}", end='')
    for t2 in targets:
        print(f"  {t2:>6}", end='')
    print()
    for t1 in targets:
        print(f"  {t1:>6}  ", end='')
        for t2 in targets:
            avg_overlap = np.mean(overlap_counts[t1][t2])
            print(f"  {avg_overlap:>5.1f}", end='')
        print()

    # 3. Prediction Stability
    print(f"\n{'=' * 80}")
    print(f"3. PREDICTION STABILITY (day-to-day rank correlation)")
    print(f"   Higher = more stable rankings = less turnover\n")
    for t in targets:
        stab = np.array(rank_stability[t])
        print(f"  pred_{t}: avg_rho={np.mean(stab):.4f}, std={np.std(stab):.4f}, "
              f"min={np.min(stab):.4f}, median={np.median(stab):.4f}")

    # 4. Top-N Actual Returns by Selection Target × Return Horizon
    print(f"\n{'=' * 80}")
    print(f"4. TOP-{args.top_n} ACTUAL RETURNS: select by pred_X, measure at horizon Y")
    print(f"   Key question: do 3d-selected stocks still perform at 10d?\n")
    print(f"{'Select by':>12}", end='')
    for h in horizons:
        print(f"  ret_{h}d", end='')
    print(f"  {'ret_10d/period ann.':>20}")
    print("-" * 80)
    for t in targets:
        print(f"  pred_{t:>3}    ", end='')
        for h in horizons:
            avg_ret = np.mean(topn_returns[t][h])
            print(f"  {avg_ret:>+.3%}", end='')
        # Annualized 10d return (non-overlapping cumulative NAV)
        rets_10d = np.array(topn_returns[t][10])
        non_overlap = rets_10d[::10]
        nav = np.cumprod(1 + non_overlap)
        total_years = len(rets_10d) / 245.0
        ann_10d = nav[-1] ** (1 / total_years) - 1 if total_years > 0 and nav[-1] > 0 else 0
        print(f"  {ann_10d:>+18.1%}")

    # 5. Signal-to-Noise Analysis
    print(f"\n{'=' * 80}")
    print(f"5. SIGNAL-TO-NOISE ANALYSIS")
    print(f"   Prediction spread vs actual return spread\n")
    print(f"  {'Target':>8} {'Pred StdDev':>12} {'Actual StdDev':>14} {'Ratio':>8}")
    print(f"  {'-'*45}")
    for t in targets:
        h = int(t.replace('d', ''))
        p_std = np.mean(pred_std[t])
        a_std = np.mean(actual_std[h])
        ratio = p_std / a_std if a_std > 0 else 0
        print(f"  {t:>8} {p_std:>12.6f} {a_std:>14.4%} {ratio:>8.4f}")

    # 6. Prediction Correlation Matrix
    print(f"\n{'=' * 80}")
    print(f"6. PREDICTION CORRELATION BETWEEN TARGETS")
    print(f"   High correlation = redundant information\n")
    print(f"{'':>10}", end='')
    for t2 in targets:
        print(f"  {t2:>7}", end='')
    print()
    for t1 in targets:
        print(f"  {t1:>6}  ", end='')
        for t2 in targets:
            if t1 == t2:
                print(f"  {'1.000':>7}", end='')
            elif t1 < t2:
                print(f"  {np.mean(pred_corr[(t1, t2)]):>7.3f}", end='')
            else:
                print(f"  {np.mean(pred_corr[(t2, t1)]):>7.3f}", end='')
        print()

    # 7. Summary hypothesis
    print(f"\n{'=' * 80}")
    print(f"7. DIAGNOSIS SUMMARY\n")

    # Check key hypotheses
    # H1: 3d predicts 3d well but 3d returns don't persist to 10d
    ic_3d_own = np.mean(cross_ic['3d'][3])
    ic_3d_10d = np.mean(cross_ic['3d'][10])
    ic_10d_10d = np.mean(cross_ic['10d'][10])

    ret_3d_at_3d = np.mean(topn_returns['3d'][3])
    ret_3d_at_10d = np.mean(topn_returns['3d'][10])
    ret_10d_at_10d = np.mean(topn_returns['10d'][10])

    stab_3d = np.mean(rank_stability['3d'])
    stab_10d = np.mean(rank_stability['10d'])

    overlap_3d_10d = np.mean(overlap_counts['3d']['10d'])

    print(f"  H1: 3d signal doesn't persist to 10d holding period")
    print(f"    pred_3d→actual_3d IC: {ic_3d_own:+.4f}")
    print(f"    pred_3d→actual_10d IC: {ic_3d_10d:+.4f}  (decay: {(1-ic_3d_10d/ic_3d_own)*100:.0f}%)")
    print(f"    pred_10d→actual_10d IC: {ic_10d_10d:+.4f}")
    print(f"    Top-10 by 3d: 3d ret={ret_3d_at_3d:+.3%}, 10d ret={ret_3d_at_10d:+.3%}")
    print(f"    Top-10 by 10d: 10d ret={ret_10d_at_10d:+.3%}")
    print()
    print(f"  H2: 3d rankings are less stable (higher turnover cost)")
    print(f"    pred_3d day-to-day rank corr: {stab_3d:.4f}")
    print(f"    pred_10d day-to-day rank corr: {stab_10d:.4f}")
    print()
    print(f"  H3: Different targets select different stocks")
    print(f"    3d vs 10d top-{args.top_n} overlap: {overlap_3d_10d:.1f}/{args.top_n} stocks")


if __name__ == '__main__':
    main()
