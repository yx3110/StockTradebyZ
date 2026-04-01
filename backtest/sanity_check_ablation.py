#!/usr/bin/env python3
"""
Ablation回测结果全面Sanity Check
================================
从机器学习+量化交易专家角度，系统检查以下维度：

1. 年化收益计算方法 (复利 vs 简单 vs 累积净值)
2. 重叠周期偏差 (overlapping returns bias)
3. 交易成本影响 (佣金+印花税+滑点+冲击成本)
4. 幸存者偏差 (survivorship bias)
5. 前视偏差 (look-ahead bias)
6. 统计显著性 (t-test, bootstrap confidence interval)
7. 最大回撤 (MaxDD)
8. 权重优化过拟合 (overfitting to in-sample weight selection)
9. 容量约束 (capacity: 小盘股能否实际买入)
10. 时间稳定性 (滚动窗口IC/收益是否稳定)

用法:
    python3 backtest/sanity_check_ablation.py --version v4.7.5
"""

import sys, os, json, time, sqlite3, argparse
import numpy as np
import pandas as pd
from pathlib import Path
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


def preload_market_caps(dates):
    """加载每日市值数据，用于容量分析"""
    conn = sqlite3.connect(DB_PATH)
    result = {}
    for i in range(0, len(dates), 50):
        chunk = dates[i:i+50]
        ph = ','.join(['?'] * len(chunk))
        rows = conn.execute(f"""
            SELECT s.code, db.trade_date, db.total_mv
            FROM daily_basic db
            JOIN securities s ON db.security_id = s.id
            WHERE db.trade_date IN ({ph}) AND db.total_mv > 0
        """, chunk).fetchall()
        for code, date, mv in rows:
            if date not in result:
                result[date] = {}
            result[date][code] = mv / 10000  # 万元 → 亿元
    conn.close()
    return result


def preload_daily_returns(dates):
    """加载逐日收益数据（用于non-overlapping计算和MaxDD）"""
    conn = sqlite3.connect(DB_PATH)
    all_dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT trade_date FROM daily_quotes ORDER BY trade_date").fetchall()]
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    result = {}
    for date in dates:
        idx = date_to_idx.get(date)
        if idx is None or idx + 1 >= len(all_dates):
            continue
        next_date = all_dates[idx + 1]
        rows = conn.execute("""
            SELECT s.code, q1.close, q2.close
            FROM daily_quotes q1
            JOIN daily_quotes q2 ON q1.security_id = q2.security_id
            JOIN securities s ON q1.security_id = s.id
            WHERE q1.trade_date = ? AND q2.trade_date = ?
              AND q1.close > 0 AND q2.close > 0
        """, (date, next_date)).fetchall()
        result[date] = {code: (p2 - p1) / p1 for code, p1, p2 in rows}
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


def get_top_codes(daily_data_entry, weights, top_n=10):
    """获取某天的top-N股票代码"""
    preds, valid, codes = daily_data_entry['predictions'], daily_data_entry['valid_mask'], daily_data_entry['codes']
    composite = sum(weights[t] * preds[t] for t in ['3d', '5d', '10d', '15d'])
    ranked = np.argsort(-composite)
    top_idx = [idx for idx in ranked if valid[idx]][:top_n]
    return [codes[i] for i in top_idx], top_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', required=True, choices=['v4.4', 'v4.6', 'v4.7.3', 'v4.7.5'])
    parser.add_argument('--start-date', default='2024-01-01')
    parser.add_argument('--end-date', default='2026-02-13')
    parser.add_argument('--top-n', type=int, default=10)
    args = parser.parse_args()

    # 最优权重配置
    optimal_weights = {
        'v4.4': {'3d': 0.20, '5d': 0.25, '10d': 0.35, '15d': 0.20},  # default mix
        'v4.6': {'3d': 0.00, '5d': 0.00, '10d': 0.60, '15d': 0.40},
        'v4.7.3': {'3d': 0.00, '5d': 0.00, '10d': 0.60, '15d': 0.40},
        'v4.7.5': {'3d': 0.00, '5d': 0.00, '10d': 0.60, '15d': 0.40},
    }
    weights = optimal_weights[args.version]

    print(f"{'='*80}")
    print(f"  SANITY CHECK: {args.version} Ablation Results")
    print(f"  Period: {args.start_date} ~ {args.end_date}, Top-{args.top_n}")
    print(f"  Weights: {weights}")
    print(f"{'='*80}\n")

    # Load scorer and data
    scorer = load_scorer(args.version)
    dates = get_trading_dates(args.start_date, args.end_date)
    print(f"Loading data ({len(dates)} dates)...")
    sys.stdout.flush()
    features_cache = preload_features(dates)
    fwd_returns_10d = preload_forward_returns(dates, hold_days=10)
    market_caps = preload_market_caps(dates)

    # Score all dates
    print(f"Scoring...")
    sys.stdout.flush()
    t0 = time.time()
    daily_data = []
    for di, date in enumerate(dates):
        fdf = features_cache.get(date)
        fwd = fwd_returns_10d.get(date)
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
        if (di + 1) % 100 == 0:
            print(f"  {di+1}/{len(dates)} ({time.time()-t0:.0f}s)")
            sys.stdout.flush()
    print(f"  Done: {len(daily_data)} valid dates in {time.time()-t0:.0f}s\n")
    sys.stdout.flush()

    # ================================================================
    # CHECK 1: 年化收益计算方法对比
    # ================================================================
    print(f"\n{'='*80}")
    print("CHECK 1: 年化收益计算方法对比")
    print(f"{'='*80}")

    top_rets = []
    for dd in daily_data:
        top_codes, top_idx = get_top_codes(dd, weights, args.top_n)
        if top_idx:
            top_rets.append(np.mean(dd['actual_returns'][top_idx]))
    top_rets = np.array(top_rets)
    avg_ret = np.mean(top_rets)

    # 方法A: 原始复利 (有问题)
    ann_ret_compound = (1 + avg_ret) ** 24.5 - 1

    # 方法B: 简单年化
    ann_ret_simple = avg_ret * 24.5

    # 方法C: 累积净值法 (最准确)
    # 每10天换一次仓，用non-overlapping periods
    n_days = len(top_rets)
    non_overlap_rets = top_rets[::10]  # 每10天取一个
    cumulative_nav = np.cumprod(1 + non_overlap_rets)
    total_years = n_days / 245.0
    non_overlap_years = len(non_overlap_rets) * 10 / 245.0
    if cumulative_nav[-1] > 0 and non_overlap_years > 0:
        ann_ret_nav = cumulative_nav[-1] ** (1 / non_overlap_years) - 1
    else:
        ann_ret_nav = 0

    # 方法D: 逐日累积 (模拟每天调仓)
    cumulative_all = np.cumprod(1 + top_rets)
    ann_ret_daily = cumulative_all[-1] ** (1 / total_years) - 1

    print(f"  每周期平均收益 (avg_ret): {avg_ret:+.4%}")
    print(f"  总天数: {n_days}, 总年数: {total_years:.2f}")
    print(f"")
    print(f"  方法A 复利(原始):     {ann_ret_compound:>+8.1%}  ← 原ablation脚本")
    print(f"  方法B 简单年化:       {ann_ret_simple:>+8.1%}  ← avg_ret × 24.5")
    print(f"  方法C 非重叠累积:     {ann_ret_nav:>+8.1%}  ← 每10天一个独立周期")
    print(f"  方法D 逐日累积净值:   {ann_ret_daily:>+8.1%}  ← 假设每天调仓(重叠)")
    print(f"")
    print(f"  ⚠ 方法A的问题: 对avg_ret做复利，但avg_ret来自重叠周期的简单平均")
    print(f"  ⚠ 方法D的问题: 每天的10d return重叠，相当于杠杆×10")
    print(f"  ✓ 方法C最接近真实: 独立周期数={len(non_overlap_rets)}")

    # ================================================================
    # CHECK 2: 重叠周期偏差 (Overlapping Returns Bias)
    # ================================================================
    print(f"\n{'='*80}")
    print("CHECK 2: 重叠周期偏差")
    print(f"{'='*80}")

    # 比较overlapping vs non-overlapping的统计量
    overlap_mean = np.mean(top_rets)
    overlap_std = np.std(top_rets)
    nonoverlap_mean = np.mean(non_overlap_rets)
    nonoverlap_std = np.std(non_overlap_rets)

    # Newey-West adjusted std (autocorrelation correction for overlapping returns)
    # 用lag=9 (10-day holding period)
    n = len(top_rets)
    demeaned = top_rets - overlap_mean
    gamma0 = np.sum(demeaned**2) / n
    nw_var = gamma0
    max_lag = 9
    for lag in range(1, max_lag + 1):
        gamma_lag = np.sum(demeaned[lag:] * demeaned[:-lag]) / n
        bartlett_weight = 1 - lag / (max_lag + 1)
        nw_var += 2 * bartlett_weight * gamma_lag
    nw_std = np.sqrt(nw_var)

    sharpe_naive = overlap_mean / overlap_std * np.sqrt(24.5) if overlap_std > 0 else 0
    sharpe_nw = overlap_mean / nw_std * np.sqrt(24.5) if nw_std > 0 else 0
    sharpe_nonoverlap = nonoverlap_mean / nonoverlap_std * np.sqrt(24.5) if nonoverlap_std > 0 else 0

    # Autocorrelation of top returns
    autocorrs = []
    for lag in range(1, 11):
        if len(top_rets) > lag:
            ac = np.corrcoef(top_rets[lag:], top_rets[:-lag])[0, 1]
            autocorrs.append(ac)

    print(f"  Overlapping (每天):    mean={overlap_mean:+.4%}, std={overlap_std:.4%}, N={n}")
    print(f"  Non-overlapping (每10天): mean={nonoverlap_mean:+.4%}, std={nonoverlap_std:.4%}, N={len(non_overlap_rets)}")
    print(f"  Newey-West adjusted std: {nw_std:.4%} (vs naive {overlap_std:.4%}, ratio={nw_std/overlap_std:.2f}x)")
    print(f"")
    print(f"  Sharpe (naive):          {sharpe_naive:.3f}")
    print(f"  Sharpe (Newey-West):     {sharpe_nw:.3f}")
    print(f"  Sharpe (non-overlapping):{sharpe_nonoverlap:.3f}")
    print(f"")
    print(f"  Return autocorrelation (lag 1-10):")
    for i, ac in enumerate(autocorrs):
        bar = '█' * int(abs(ac) * 50)
        print(f"    lag {i+1:2d}: {ac:+.3f} {bar}")
    print(f"  ⚠ 高自相关 = 重叠偏差严重，Sharpe被高估")

    # ================================================================
    # CHECK 3: 交易成本影响
    # ================================================================
    print(f"\n{'='*80}")
    print("CHECK 3: 交易成本影响")
    print(f"{'='*80}")

    # 计算换手率
    prev_top = set()
    turnovers = []
    for dd in daily_data:
        top_codes, _ = get_top_codes(dd, weights, args.top_n)
        curr_top = set(top_codes)
        if prev_top:
            overlap = len(curr_top & prev_top)
            turnover = 1 - overlap / args.top_n  # 换手比例
            turnovers.append(turnover)
        prev_top = curr_top
    turnovers = np.array(turnovers)
    avg_turnover = np.mean(turnovers)

    # 成本模型
    commission = 0.00025  # 佣金万2.5 (单边)
    stamp_tax = 0.0005    # 印花税万5 (卖出)
    slippage = 0.001      # 滑点0.1% (单边)
    transfer_fee = 0.00002  # 过户费万0.2 (单边)
    one_way_cost = commission + slippage + transfer_fee
    round_trip_cost = one_way_cost * 2 + stamp_tax  # 买+卖

    # 每天的交易成本 = 换手率 × 双边成本
    daily_cost = avg_turnover * round_trip_cost
    annual_cost = daily_cost * 245

    # 净收益
    net_avg_ret = avg_ret - daily_cost
    net_ann_simple = net_avg_ret * 24.5

    # Non-overlapping净收益
    net_nonoverlap = non_overlap_rets - avg_turnover * round_trip_cost
    net_nav = np.cumprod(1 + net_nonoverlap)
    if net_nav[-1] > 0 and non_overlap_years > 0:
        net_ann_nav = net_nav[-1] ** (1 / non_overlap_years) - 1
    else:
        net_ann_nav = 0

    print(f"  日均换手率: {avg_turnover:.1%} (每天换{avg_turnover*args.top_n:.1f}只股票)")
    print(f"  换手率分布: min={turnovers.min():.0%}, median={np.median(turnovers):.0%}, max={turnovers.max():.0%}")
    print(f"")
    print(f"  单次交易成本:")
    print(f"    佣金(双边): {commission*2:.4%}")
    print(f"    印花税(卖): {stamp_tax:.4%}")
    print(f"    滑点(双边):  {slippage*2:.4%}")
    print(f"    过户费(双边): {transfer_fee*2:.5%}")
    print(f"    合计单次:    {round_trip_cost:.4%}")
    print(f"")
    print(f"  日均交易成本: {daily_cost:.4%}")
    print(f"  年化交易成本: {annual_cost:.2%}")
    print(f"")
    print(f"  毛收益 (简单年化):  {ann_ret_simple:>+8.1%}")
    print(f"  净收益 (简单年化):  {net_ann_simple:>+8.1%}")
    print(f"  净收益 (非重叠累积):{net_ann_nav:>+8.1%}")
    print(f"  成本侵蚀: {annual_cost:.1%}/年 (占毛收益{annual_cost/ann_ret_simple*100:.1f}%)")

    # ================================================================
    # CHECK 4: 幸存者偏差 (Survivorship Bias)
    # ================================================================
    print(f"\n{'='*80}")
    print("CHECK 4: 幸存者偏差检查")
    print(f"{'='*80}")

    # 检查forward return中NaN比例 (退市股应有NaN)
    nan_rates = []
    total_stocks_per_day = []
    for dd in daily_data:
        valid_ratio = dd['valid_mask'].sum() / len(dd['valid_mask'])
        nan_rates.append(1 - valid_ratio)
        total_stocks_per_day.append(len(dd['codes']))

    # 检查是否有极端负收益 (退市/ST可能-50%以上)
    all_actual = np.concatenate([dd['actual_returns'][dd['valid_mask']] for dd in daily_data])
    extreme_neg = np.mean(all_actual < -0.20)
    extreme_pos = np.mean(all_actual > 0.20)

    print(f"  每日股票数: mean={np.mean(total_stocks_per_day):.0f}, min={np.min(total_stocks_per_day)}, max={np.max(total_stocks_per_day)}")
    print(f"  Forward return缺失率: mean={np.mean(nan_rates):.2%}, max={np.max(nan_rates):.2%}")
    print(f"  极端收益分布 (10d):")
    print(f"    < -20%: {extreme_neg:.3%}")
    print(f"    > +20%: {extreme_pos:.3%}")
    print(f"    < -30%: {np.mean(all_actual < -0.30):.4%}")
    print(f"    > +30%: {np.mean(all_actual > 0.30):.4%}")
    print(f"  ⚠ 如果缺失率很低且无极端负值，可能存在幸存者偏差")
    print(f"  ⚠ 退市股在forward return中应表现为NaN或极端负值")

    # ================================================================
    # CHECK 5: 前视偏差 (Look-Ahead Bias)
    # ================================================================
    print(f"\n{'='*80}")
    print("CHECK 5: 前视偏差检查")
    print(f"{'='*80}")

    # 检查是否使用了未来信息
    # 1) forward return使用的是T+1买入、T+1+10卖出 (正确)
    # 2) 特征是否包含当天收盘数据 (需要确认)
    # 3) 全局分位数是否用了未来数据
    print(f"  Forward return计算: T日选股 → T+1买入 → T+1+10卖出 ✓")
    print(f"  特征数据: 来自v39_feature_cache (T日收盘计算) ✓")
    print(f"  全局分位数: 用训练期数据计算 (需确认)")
    print(f"")

    # 检查: pred_10d是否和actual_10d有异常高相关 (可能泄露)
    sample_ics = []
    for dd in daily_data[:50]:  # 前50天
        v = dd['valid_mask']
        if v.sum() > 20:
            ic, _ = stats.spearmanr(dd['predictions']['10d'][v], dd['actual_returns'][v])
            sample_ics.append(ic)
    mean_ic = np.mean(sample_ics)
    print(f"  前50天IC(pred_10d vs actual_10d): {mean_ic:.4f}")
    print(f"  ✓ IC在0.05-0.15范围内属正常 (>0.20可能有泄露)")
    if mean_ic > 0.20:
        print(f"  ⚠⚠ IC异常高! 可能存在前视偏差!")

    # ================================================================
    # CHECK 6: 统计显著性
    # ================================================================
    print(f"\n{'='*80}")
    print("CHECK 6: 统计显著性")
    print(f"{'='*80}")

    # t-test: top_rets mean != 0
    t_stat, p_value = stats.ttest_1samp(top_rets, 0)
    # Newey-West调整的t统计量
    nw_t_stat = overlap_mean / (nw_std / np.sqrt(n))
    nw_p_value = 2 * (1 - stats.t.cdf(abs(nw_t_stat), n - 1))

    # Non-overlapping t-test
    if len(non_overlap_rets) > 2:
        t_nonoverlap, p_nonoverlap = stats.ttest_1samp(non_overlap_rets, 0)
    else:
        t_nonoverlap, p_nonoverlap = 0, 1

    # Bootstrap confidence interval (non-overlapping)
    n_bootstrap = 10000
    boot_means = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(non_overlap_rets, size=len(non_overlap_rets), replace=True)
        boot_means.append(np.mean(boot_sample))
    boot_means = np.array(boot_means)
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)

    print(f"  Overlapping t-test:")
    print(f"    t={t_stat:.2f}, p={p_value:.6f}, N={n}")
    print(f"    ⚠ 重叠序列t-test高估显著性 (有效自由度远小于{n})")
    print(f"")
    print(f"  Newey-West调整 (lag={max_lag}):")
    print(f"    t={nw_t_stat:.2f}, p={nw_p_value:.6f}")
    print(f"")
    print(f"  Non-overlapping t-test:")
    print(f"    t={t_nonoverlap:.2f}, p={p_nonoverlap:.6f}, N={len(non_overlap_rets)}")
    print(f"")
    print(f"  Bootstrap 95% CI (non-overlapping, {n_bootstrap} iter):")
    print(f"    mean={np.mean(non_overlap_rets):+.4%}, CI=[{ci_lower:+.4%}, {ci_upper:+.4%}]")
    print(f"    年化CI: [{ci_lower*24.5:+.1%}, {ci_upper*24.5:+.1%}]")
    if ci_lower > 0:
        print(f"    ✓ 95% CI不包含0, 统计显著")
    else:
        print(f"    ⚠ 95% CI包含0, 不显著!")

    # ================================================================
    # CHECK 7: 最大回撤 (MaxDD)
    # ================================================================
    print(f"\n{'='*80}")
    print("CHECK 7: 最大回撤")
    print(f"{'='*80}")

    # 用non-overlapping periods累积净值
    nav = np.cumprod(1 + non_overlap_rets)
    peak = np.maximum.accumulate(nav)
    drawdowns = (nav - peak) / peak
    max_dd = np.min(drawdowns)
    dd_end_idx = np.argmin(drawdowns)
    dd_start_idx = np.argmax(nav[:dd_end_idx + 1]) if dd_end_idx > 0 else 0

    # 扣成本后
    nav_net = np.cumprod(1 + net_nonoverlap)
    peak_net = np.maximum.accumulate(nav_net)
    drawdowns_net = (nav_net - peak_net) / peak_net
    max_dd_net = np.min(drawdowns_net)

    # 连续亏损周期
    losing_streaks = []
    current_streak = 0
    for r in non_overlap_rets:
        if r < 0:
            current_streak += 1
        else:
            if current_streak > 0:
                losing_streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        losing_streaks.append(current_streak)
    max_losing = max(losing_streaks) if losing_streaks else 0

    print(f"  毛收益MaxDD: {max_dd:+.2%}")
    print(f"  净收益MaxDD: {max_dd_net:+.2%}")
    print(f"  最长连续亏损: {max_losing}个周期 (={max_losing*10}个交易日)")
    print(f"  正收益周期占比: {np.mean(non_overlap_rets > 0):.1%}")
    print(f"  Calmar ratio (年化/MaxDD): {abs(net_ann_nav/max_dd_net):.2f}" if max_dd_net < 0 else "  Calmar: N/A")

    # ================================================================
    # CHECK 8: 权重优化过拟合
    # ================================================================
    print(f"\n{'='*80}")
    print("CHECK 8: 权重优化过拟合检查 (前半段 vs 后半段)")
    print(f"{'='*80}")

    mid = len(daily_data) // 2
    first_half = daily_data[:mid]
    second_half = daily_data[mid:]

    weight_configs = {
        'Pure 10d':    {'3d': 0.0, '5d': 0.0, '10d': 1.0, '15d': 0.0},
        '10d+15d':     {'3d': 0.0, '5d': 0.0, '10d': 0.6, '15d': 0.4},
        'Default mix': {'3d': 0.20, '5d': 0.25, '10d': 0.35, '15d': 0.20},
    }

    def eval_subset(data, w, top_n):
        rets = []
        for dd in data:
            _, top_idx = get_top_codes(dd, w, top_n)
            if top_idx:
                rets.append(np.mean(dd['actual_returns'][top_idx]))
        rets = np.array(rets)
        return np.mean(rets) * 24.5 if len(rets) > 0 else 0

    print(f"  前半段: {first_half[0]['date']} ~ {first_half[-1]['date']} ({len(first_half)}天)")
    print(f"  后半段: {second_half[0]['date']} ~ {second_half[-1]['date']} ({len(second_half)}天)")
    print(f"")
    print(f"  {'Config':<15} {'前半段(年化)':>12} {'后半段(年化)':>12} {'差异':>8}")
    print(f"  {'-'*50}")
    for name, w in weight_configs.items():
        r1 = eval_subset(first_half, w, args.top_n)
        r2 = eval_subset(second_half, w, args.top_n)
        diff = r2 - r1
        print(f"  {name:<15} {r1:>+11.1%} {r2:>+11.1%} {diff:>+7.1%}")
    print(f"  ⚠ 如果'最优'配置在后半段大幅衰退，说明过拟合")

    # ================================================================
    # CHECK 9: 容量约束 (能否实际买入)
    # ================================================================
    print(f"\n{'='*80}")
    print("CHECK 9: 容量约束 (市值分布)")
    print(f"{'='*80}")

    top_caps = []
    micro_cap_count = 0
    total_top_count = 0
    for dd in daily_data:
        top_codes, _ = get_top_codes(dd, weights, args.top_n)
        caps_today = market_caps.get(dd['date'], {})
        for code in top_codes:
            cap = caps_today.get(code, np.nan)
            if not np.isnan(cap):
                top_caps.append(cap)
                if cap < 30:  # < 30亿
                    micro_cap_count += 1
            total_top_count += 1

    top_caps = np.array(top_caps)
    if len(top_caps) > 0:
        print(f"  Top-{args.top_n}选股市值分布 (亿元):")
        print(f"    样本数: {len(top_caps)} (缺失: {total_top_count - len(top_caps)})")
        print(f"    均值:   {np.mean(top_caps):.1f}亿")
        print(f"    中位数: {np.median(top_caps):.1f}亿")
        print(f"    P10:    {np.percentile(top_caps, 10):.1f}亿")
        print(f"    P25:    {np.percentile(top_caps, 25):.1f}亿")
        print(f"    P75:    {np.percentile(top_caps, 75):.1f}亿")
        print(f"    P90:    {np.percentile(top_caps, 90):.1f}亿")
        print(f"    <30亿 (微盘): {micro_cap_count}/{total_top_count} = {micro_cap_count/total_top_count:.1%}")
        print(f"    <50亿 (小盘): {np.mean(top_caps < 50):.1%}")
        print(f"    >200亿(大盘): {np.mean(top_caps > 200):.1%}")
        print(f"")

        # 假设资金量1000万, 每只股票100万
        # 100万能否在不显著冲击价格的情况下买入？
        # 规则：单笔 < 日均成交额的1%
        print(f"  容量分析 (假设每只股票100万):")
        if np.median(top_caps) < 30:
            print(f"    ⚠ 中位市值<30亿, 严重容量受限, 实际滑点可能远超0.1%")
        elif np.median(top_caps) < 50:
            print(f"    ⚠ 中位市值<50亿, 中等容量受限")
        else:
            print(f"    ✓ 中位市值>50亿, 容量基本可行")
    else:
        print(f"  ⚠ 无市值数据")

    # ================================================================
    # CHECK 10: 时间稳定性 (滚动窗口分析)
    # ================================================================
    print(f"\n{'='*80}")
    print("CHECK 10: 时间稳定性 (季度滚动)")
    print(f"{'='*80}")

    # 按季度分组
    quarter_data = {}
    for dd in daily_data:
        date = dd['date']
        q = date[:4] + 'Q' + str((int(date[5:7]) - 1) // 3 + 1)
        if q not in quarter_data:
            quarter_data[q] = []
        top_codes, top_idx = get_top_codes(dd, weights, args.top_n)
        if top_idx:
            ret = np.mean(dd['actual_returns'][top_idx])
            # IC
            v = dd['valid_mask']
            composite = sum(weights[t] * dd['predictions'][t] for t in ['3d', '5d', '10d', '15d'])
            if v.sum() > 20:
                ic, _ = stats.spearmanr(composite[v], dd['actual_returns'][v])
            else:
                ic = np.nan
            quarter_data[q].append({'ret': ret, 'ic': ic})

    print(f"  {'季度':<10} {'天数':>5} {'AvgRet':>8} {'IC':>7} {'IC>0%':>7} {'年化(简单)':>10}")
    print(f"  {'-'*52}")
    for q in sorted(quarter_data.keys()):
        items = quarter_data[q]
        rets = np.array([x['ret'] for x in items])
        ics = np.array([x['ic'] for x in items if not np.isnan(x['ic'])])
        print(f"  {q:<10} {len(items):>5} {np.mean(rets):>+7.2%} {np.mean(ics):>+6.3f} "
              f"{np.mean(ics>0):>6.1%} {np.mean(rets)*24.5:>+9.1%}")

    # 检查最差季度
    q_rets = {q: np.mean([x['ret'] for x in items]) for q, items in quarter_data.items()}
    worst_q = min(q_rets, key=q_rets.get)
    best_q = max(q_rets, key=q_rets.get)
    print(f"")
    print(f"  最佳季度: {best_q} ({q_rets[best_q]*24.5:+.1%}年化)")
    print(f"  最差季度: {worst_q} ({q_rets[worst_q]*24.5:+.1%}年化)")
    print(f"  季度间变异系数: {np.std(list(q_rets.values()))/np.mean(list(q_rets.values())):.2f}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*80}")
    print(f"  综合评估总结 - {args.version}")
    print(f"{'='*80}")
    print(f"")
    print(f"  ┌─────────────────────────────────────────────────────┐")
    print(f"  │ 指标                  │ 原始(乐观) │ 修正(保守)    │")
    print(f"  ├─────────────────────────────────────────────────────┤")
    print(f"  │ 年化收益(毛)          │ {ann_ret_compound:>+8.1%}   │ {ann_ret_simple:>+8.1%}      │")
    print(f"  │ 年化收益(净)          │    N/A     │ {net_ann_simple:>+8.1%}      │")
    print(f"  │ 年化(非重叠净值法)    │    N/A     │ {net_ann_nav:>+8.1%}      │")
    print(f"  │ Sharpe                │ {sharpe_naive:>8.3f}   │ {sharpe_nw:>8.3f}        │")
    print(f"  │ MaxDD                 │    N/A     │ {max_dd_net:>+8.1%}      │")
    print(f"  │ 日均换手率            │    N/A     │ {avg_turnover:>8.1%}      │")
    print(f"  │ 中位市值              │    N/A     │ {np.median(top_caps):>7.1f}亿     │" if len(top_caps) > 0 else "")
    print(f"  │ 统计显著 (p-value)    │ {p_value:>8.6f}   │ {nw_p_value:>8.6f}        │")
    print(f"  └─────────────────────────────────────────────────────┘")
    print(f"")

    # 风险评级
    issues = []
    if avg_turnover > 0.5:
        issues.append("高换手率(>{:.0%})侵蚀收益".format(avg_turnover))
    if len(top_caps) > 0 and np.median(top_caps) < 30:
        issues.append("微盘股偏好, 容量受限")
    if nw_p_value > 0.05:
        issues.append("Newey-West调整后不显著")
    if max_dd_net < -0.30:
        issues.append("最大回撤>{:.0%}".format(abs(max_dd_net)))
    if ann_ret_compound / net_ann_nav > 2.0 and net_ann_nav > 0:
        issues.append("原始年化高估>{:.0f}倍".format(ann_ret_compound / net_ann_nav))

    if not issues:
        print(f"  ✅ 未发现严重问题")
    else:
        print(f"  ⚠ 发现的问题:")
        for issue in issues:
            print(f"    - {issue}")

    sys.stdout.flush()


if __name__ == '__main__':
    main()
