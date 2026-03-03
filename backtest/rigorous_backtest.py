#!/usr/bin/env python3
"""
V4.6 严谨长期回测系统

8个分析模块覆盖5年+多市场周期:
  A. 全期间北极星V2评分
  B. 逐年滚动分析
  C. 市况分解 (牛/熊/震荡)
  D. 基准对比 + 多空分析
  E. 统计显著性 (t检验, Bootstrap, Monte Carlo)
  F. 回撤分析 (Top 5事件)
  G. 交易成本敏感度
  H. Walk-Forward对比

用法:
    # 完整流程 (生成报告 + 回测)
    python3 backtest/rigorous_backtest.py \\
        --version v4.6 --start-date 2020-06-01 --end-date 2026-02-13

    # 跳过报告生成 (已有报告)
    python3 backtest/rigorous_backtest.py \\
        --version v4.6 --start-date 2020-06-01 --end-date 2026-02-13 --skip-generation

    # 自定义参数
    python3 backtest/rigorous_backtest.py \\
        --version v4.6 --top-n 10 --focus-days 10 --report-dir reports/daily_selection_v4.6_full_history
"""

import sys
import os
import subprocess
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from scipy.stats import spearmanr, ttest_1samp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from backtest.backtest_report_based import (
    load_reports, get_future_returns, run_single_backtest,
    _compute_period_risk_metrics, _aggregate_benchmark_to_periods,
    HOLDING_DAYS, get_next_trading_date,
)
from backtest.north_star_metrics import (
    compute_drawdown_series, load_benchmark_returns, compute_benchmark_comparison,
    NORTH_STAR_TARGETS_V2, V2_LAYER_NAMES, score_metric_v2, compute_v2_grade,
    classify_market_regime, compute_regime_conditional_metrics,
)


# ═══════════════════════════════════════════════════════════════
# Module A: 全期间北极星V2评分
# ═══════════════════════════════════════════════════════════════

def module_a_north_star_v2(backtest_result, focus_days):
    """计算并返回V2评分卡数据"""
    s = backtest_result['summary'].get(focus_days, {})
    if not s:
        return {'total_score': 0, 'max_score': 105, 'grade': 'D', 'layers': {}, 'metrics': {}}

    metric_value_map = {
        'daily_ic': s.get('ic_mean', 0),
        'icir': s.get('icir', 0),
        'ic_positive_pct': s.get('ic_positive_pct', 0),
        'ic_monotonicity': s.get('ic_monotonicity', 0),
        'ic_time_stability': s.get('ic_time_stability', 999),
        'signal_half_life': s.get('signal_half_life', 0),
        'annual_turnover': s.get('annual_turnover', 0),
        'annual_cost_drag': s.get('annual_cost_drag', 0),
        'net_gross_ratio': s.get('net_gross_ratio', 0),
        'limit_up_fail_rate': s.get('limit_up_fail_rate', 0),
        'liquidity_coverage': s.get('liquidity_coverage', 0),
        'max_drawdown': s.get('max_drawdown', 0),
        'sharpe_ratio': s.get('sharpe_ratio', 0),
        'sortino_ratio': s.get('sortino_ratio', 0),
        'calmar_ratio': s.get('calmar_ratio', 0),
        'worst_rolling_60d_icir': s.get('worst_rolling_60d_icir', None),
        'annual_return': s.get('annual_return', 0),
        'monthly_win_rate': s.get('monthly_win_rate', 0),
        'half_period_consistency': s.get('half_period_consistency', 0),
        'cap_balance_ratio': s.get('cap_balance_ratio', 0),
        'median_market_cap_bn': s.get('median_market_cap_bn', 0),
    }

    total_score = 0
    max_score = 0
    metrics = {}
    layers = {}

    for layer_id in sorted(V2_LAYER_NAMES.keys()):
        layer_metrics = [(k, v) for k, v in NORTH_STAR_TARGETS_V2.items() if v['layer'] == layer_id]
        layer_score = 0
        layer_scored_count = 0
        for metric_key, target_info in layer_metrics:
            current = metric_value_map.get(metric_key)
            if current is None:
                continue
            score, grade_str = score_metric_v2(current, target_info)
            total_score += score
            max_score += 5
            layer_score += score
            layer_scored_count += 1
            metrics[metric_key] = {
                'value': current, 'score': score, 'grade': grade_str,
                'display': target_info['display'],
            }
        layers[layer_id] = {
            'name': V2_LAYER_NAMES[layer_id],
            'score': layer_score,
            'max': layer_scored_count * 5,
        }

    grade = compute_v2_grade(total_score, max_score)

    return {
        'total_score': total_score, 'max_score': max_score,
        'grade': grade, 'layers': layers, 'metrics': metrics,
        'summary': s,
    }


# ═══════════════════════════════════════════════════════════════
# Module B: 逐年滚动分析
# ═══════════════════════════════════════════════════════════════

def module_b_yearly_rolling(backtest_result, focus_days):
    """逐年分析IC/ICIR/收益/回撤"""
    df = backtest_result['daily_results']
    picks_df = backtest_result['picks']

    # Define year windows
    windows = [
        ('2020H2', '2020-06-01', '2020-12-31'),
        ('2021', '2021-01-01', '2021-12-31'),
        ('2022', '2022-01-01', '2022-12-31'),
        ('2023', '2023-01-01', '2023-12-31'),
        ('2024', '2024-01-01', '2024-12-31'),
        ('2025', '2025-01-01', '2025-12-31'),
        ('2026-YTD', '2026-01-01', '2026-12-31'),
    ]

    results = []
    days = focus_days

    for name, start, end in windows:
        sub = df[(df['days'] == days) & (df['date'] >= start) & (df['date'] <= end)]
        sub_picks = picks_df[(picks_df['date'] >= start) & (picks_df['date'] <= end)]

        if len(sub) < 3:
            continue

        # Non-overlapping returns
        if days == 1:
            non_overlap = sub
        else:
            non_overlap = sub.iloc[::days]

        period_ret = non_overlap['avg_top_return']
        cumulative = (1 + period_ret).prod() - 1
        total_days_covered = len(non_overlap) * days
        annual_return = (1 + cumulative) ** (252 / max(total_days_covered, 1)) - 1

        # IC calculation
        ret_col = f'return_{days}d'
        sub_picks_valid = sub_picks[sub_picks[ret_col].notna()]
        ic_values = []
        for date in sorted(sub_picks_valid['date'].unique()):
            day_picks = sub_picks_valid[sub_picks_valid['date'] == date]
            if len(day_picks) >= 5:
                ic_val, _ = spearmanr(day_picks['score'], day_picks[ret_col])
                if not np.isnan(ic_val):
                    ic_values.append(ic_val)

        # Non-overlapping ICIR
        if days > 1 and len(ic_values) >= days * 2:
            ic_sub = ic_values[::days]
        else:
            ic_sub = ic_values

        ic_mean = np.mean(ic_sub) if ic_sub else 0
        ic_std = np.std(ic_sub) if len(ic_sub) > 1 else 0
        icir = ic_mean / ic_std if ic_std > 0 else 0

        # Max drawdown
        dd_series = compute_drawdown_series(period_ret.reset_index(drop=True))
        max_dd = dd_series.min() if not dd_series.empty else 0

        # Sharpe (simple)
        ann_vol = period_ret.std() * np.sqrt(252 / days) if len(period_ret) > 1 else 0
        sharpe = (annual_return - 0.02) / ann_vol if ann_vol > 1e-8 else 0

        # Spread
        avg_spread = sub['spread'].mean() * 100 if 'spread' in sub.columns else 0

        results.append({
            'window': name,
            'n_days': len(sub),
            'n_non_overlap': len(non_overlap),
            'ic_mean': ic_mean,
            'icir': icir,
            'cumulative_return': cumulative,
            'annual_return': annual_return,
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'spread_bps': avg_spread,
            'win_rate': (period_ret > 0).mean() * 100,
        })

    return results


# ═══════════════════════════════════════════════════════════════
# Module C: 市况分解
# ═══════════════════════════════════════════════════════════════

def module_c_regime_analysis(backtest_result, focus_days, benchmark_code='000905.SH'):
    """按市况分解IC和收益"""
    df = backtest_result['daily_results']
    picks_df = backtest_result['picks']
    days = focus_days

    # Load benchmark
    all_dates = sorted(df['date'].unique())
    bm_ret = load_benchmark_returns(benchmark_code, start_date=min(all_dates), end_date=max(all_dates))
    if bm_ret.empty:
        return {}

    # Classify monthly regime by benchmark monthly return
    bm_monthly = bm_ret.groupby(bm_ret.index.to_period('M')).apply(lambda x: (1 + x).prod() - 1)

    results = {}
    for regime_name, condition in [('牛市(>+3%)', lambda x: x > 0.03),
                                    ('熊市(<-3%)', lambda x: x < -0.03),
                                    ('震荡', lambda x: (-0.03 <= x) & (x <= 0.03))]:
        regime_months = bm_monthly[condition(bm_monthly)].index

        # Filter picks by regime months
        picks_copy = picks_df.copy()
        picks_copy['month'] = pd.to_datetime(picks_copy['date']).dt.to_period('M')
        regime_picks = picks_copy[picks_copy['month'].isin(regime_months)]

        # Filter daily results
        df_copy = df.copy()
        df_copy['month'] = pd.to_datetime(df_copy['date']).dt.to_period('M')
        regime_daily = df_copy[(df_copy['month'].isin(regime_months)) & (df_copy['days'] == days)]

        if len(regime_daily) < 3:
            results[regime_name] = {'n_months': len(regime_months), 'n_days': 0}
            continue

        # IC
        ret_col = f'return_{days}d'
        valid_picks = regime_picks[regime_picks[ret_col].notna()]
        ic_values = []
        for date in sorted(valid_picks['date'].unique()):
            day_picks = valid_picks[valid_picks['date'] == date]
            if len(day_picks) >= 5:
                ic_val, _ = spearmanr(day_picks['score'], day_picks[ret_col])
                if not np.isnan(ic_val):
                    ic_values.append(ic_val)

        ic_mean = np.mean(ic_values) if ic_values else 0
        ic_std = np.std(ic_values) if len(ic_values) > 1 else 0
        icir = ic_mean / ic_std if ic_std > 0 else 0
        ic_pos_pct = np.mean([v > 0 for v in ic_values]) * 100 if ic_values else 0

        avg_top = regime_daily['avg_top_return'].mean() * 100
        avg_spread = regime_daily['spread'].mean() * 100

        results[regime_name] = {
            'n_months': len(regime_months),
            'n_days': len(regime_daily),
            'ic_mean': ic_mean,
            'icir': icir,
            'ic_pos_pct': ic_pos_pct,
            'avg_top_return': avg_top,
            'spread_bps': avg_spread,
        }

    return results


# ═══════════════════════════════════════════════════════════════
# Module D: 基准对比 + 多空分析
# ═══════════════════════════════════════════════════════════════

def module_d_benchmark_longshort(backtest_result, focus_days):
    """双基准Alpha + 多空t检验"""
    df = backtest_result['daily_results']
    days = focus_days

    sub = df[df['days'] == days].sort_values('date')
    if len(sub) < 5:
        return {}

    # Non-overlapping
    non_overlap = sub if days == 1 else sub.iloc[::days]
    period_ret = non_overlap.set_index('date')['avg_top_return'].sort_index()
    period_ret.index = pd.to_datetime(period_ret.index)

    all_dates = sorted(df['date'].unique())
    start_d, end_d = min(all_dates), max(all_dates)

    result = {}
    for bm_name, bm_code in [('中证500', '000905.SH'), ('中证2000', '932000.CSI')]:
        bm_daily = load_benchmark_returns(bm_code, start_date=start_d, end_date=end_d)
        if bm_daily.empty:
            continue

        buy_dates = non_overlap['buy_date'].tolist()
        periods_per_year = 252 / days

        if days == 1:
            buy_ret = non_overlap.set_index('buy_date')['avg_top_return'].sort_index()
            buy_ret.index = pd.to_datetime(buy_ret.index)
            bm_info = compute_benchmark_comparison(buy_ret, bm_daily, periods_per_year=252)
        else:
            bm_aligned = _aggregate_benchmark_to_periods(bm_daily, buy_dates, days)
            if len(bm_aligned) >= 3:
                buy_ret = non_overlap.set_index('buy_date')['avg_top_return'].sort_index()
                buy_ret.index = pd.to_datetime(buy_ret.index)
                bm_info = compute_benchmark_comparison(buy_ret, bm_aligned, periods_per_year=periods_per_year)
            else:
                bm_info = {}

        result[bm_name] = bm_info

    # Long-short analysis: spread t-test
    spread_values = non_overlap['spread'].values
    if len(spread_values) >= 5:
        t_stat, p_value = ttest_1samp(spread_values, 0)
        result['long_short'] = {
            'mean_spread': np.mean(spread_values) * 100,
            'std_spread': np.std(spread_values) * 100,
            't_stat': t_stat,
            'p_value': p_value,
            'n_periods': len(spread_values),
            'pct_positive': np.mean(spread_values > 0) * 100,
        }

    return result


# ═══════════════════════════════════════════════════════════════
# Module E: 统计显著性
# ═══════════════════════════════════════════════════════════════

def module_e_statistical_significance(backtest_result, focus_days):
    """IC t检验 + Bootstrap CI + Monte Carlo随机基线"""
    df = backtest_result['daily_results']
    picks_df = backtest_result['picks']
    days = focus_days

    result = {}

    # --- E1: IC t-test ---
    ret_col = f'return_{days}d'
    valid_picks = picks_df[picks_df[ret_col].notna()]
    ic_values = []
    for date in sorted(valid_picks['date'].unique()):
        day_picks = valid_picks[valid_picks['date'] == date]
        if len(day_picks) >= 5:
            ic_val, _ = spearmanr(day_picks['score'], day_picks[ret_col])
            if not np.isnan(ic_val):
                ic_values.append(ic_val)

    # Non-overlapping IC for t-test
    if days > 1 and len(ic_values) >= days * 2:
        ic_nonoverlap = ic_values[::days]
    else:
        ic_nonoverlap = ic_values

    if len(ic_nonoverlap) >= 5:
        t_stat, p_value = ttest_1samp(ic_nonoverlap, 0)
        result['ic_ttest'] = {
            'n_samples': len(ic_nonoverlap),
            'ic_mean': np.mean(ic_nonoverlap),
            'ic_std': np.std(ic_nonoverlap),
            't_stat': t_stat,
            'p_value': p_value,
            'significant_5pct': p_value < 0.05,
            'significant_1pct': p_value < 0.01,
        }

    # --- E2: Bootstrap 95% CI ---
    sub = df[df['days'] == days].sort_values('date')
    non_overlap = sub if days == 1 else sub.iloc[::days]
    period_returns = non_overlap['avg_top_return'].values

    if len(period_returns) >= 10:
        rng = np.random.default_rng(42)
        n_boot = 1000
        n = len(period_returns)
        periods_per_year = 252 / days

        boot_annual_returns = []
        boot_sharpes = []
        for _ in range(n_boot):
            sample = rng.choice(period_returns, size=n, replace=True)
            cum = np.prod(1 + sample) - 1
            ann = (1 + cum) ** (periods_per_year / n) - 1
            vol = np.std(sample) * np.sqrt(periods_per_year)
            sharpe = (ann - 0.02) / vol if vol > 1e-8 else 0
            boot_annual_returns.append(ann)
            boot_sharpes.append(sharpe)

        result['bootstrap'] = {
            'n_boot': n_boot,
            'annual_return_ci_5': np.percentile(boot_annual_returns, 2.5),
            'annual_return_ci_95': np.percentile(boot_annual_returns, 97.5),
            'annual_return_median': np.median(boot_annual_returns),
            'sharpe_ci_5': np.percentile(boot_sharpes, 2.5),
            'sharpe_ci_95': np.percentile(boot_sharpes, 97.5),
            'sharpe_median': np.median(boot_sharpes),
        }

    # --- E3: Monte Carlo random baseline ---
    # For each date, randomly pick top_n stocks from universe, compute return
    if len(picks_df) > 0 and len(non_overlap) >= 10:
        rng = np.random.default_rng(123)
        n_sim = 500
        dates_list = sorted(picks_df['date'].unique())

        # Build per-date return pools
        date_returns = {}
        for date in dates_list:
            day_data = picks_df[(picks_df['date'] == date) & (picks_df[ret_col].notna())]
            if len(day_data) >= 10:
                date_returns[date] = day_data[ret_col].values

        if len(date_returns) >= 10:
            sim_annual = []
            # Get the top_n from actual results
            actual_top_n = int(non_overlap['n_top'].median()) if 'n_top' in non_overlap.columns else 10

            sim_dates = sorted(date_returns.keys())
            # Non-overlapping dates for simulation
            sim_dates_nonoverlap = sim_dates[::days] if days > 1 else sim_dates

            for _ in range(n_sim):
                sim_period_returns = []
                for date in sim_dates_nonoverlap:
                    pool = date_returns.get(date)
                    if pool is not None and len(pool) >= actual_top_n:
                        random_picks = rng.choice(pool, size=actual_top_n, replace=False)
                        sim_period_returns.append(np.mean(random_picks))

                if len(sim_period_returns) >= 5:
                    cum = np.prod(1 + np.array(sim_period_returns)) - 1
                    n_periods = len(sim_period_returns)
                    ann = (1 + cum) ** (252 / (n_periods * days)) - 1
                    sim_annual.append(ann)

            if sim_annual:
                # Actual annual return
                actual_cum = np.prod(1 + period_returns) - 1
                actual_ann = (1 + actual_cum) ** (252 / (len(period_returns) * days)) - 1
                percentile_rank = np.mean([a < actual_ann for a in sim_annual]) * 100

                result['monte_carlo'] = {
                    'n_simulations': len(sim_annual),
                    'random_mean_annual': np.mean(sim_annual),
                    'random_median_annual': np.median(sim_annual),
                    'random_std_annual': np.std(sim_annual),
                    'random_p5': np.percentile(sim_annual, 5),
                    'random_p95': np.percentile(sim_annual, 95),
                    'actual_annual': actual_ann,
                    'percentile_rank': percentile_rank,
                }

    return result


# ═══════════════════════════════════════════════════════════════
# Module F: 回撤分析
# ═══════════════════════════════════════════════════════════════

def module_f_drawdown_analysis(backtest_result, focus_days):
    """完整回撤分析 + Top 5事件"""
    df = backtest_result['daily_results']
    days = focus_days

    sub = df[df['days'] == days].sort_values('date')
    non_overlap = sub if days == 1 else sub.iloc[::days]

    period_ret = non_overlap['avg_top_return'].values
    dates = non_overlap['date'].values

    if len(period_ret) < 5:
        return {}

    # Build NAV series
    nav = np.cumprod(1 + period_ret)
    peak = np.maximum.accumulate(nav)
    drawdown = (nav - peak) / peak

    # Find top 5 drawdown events
    events = []
    in_dd = False
    dd_start = 0
    for i in range(len(drawdown)):
        if drawdown[i] < 0 and not in_dd:
            in_dd = True
            dd_start = i
        elif drawdown[i] >= 0 and in_dd:
            # Drawdown ended
            dd_min_idx = dd_start + np.argmin(drawdown[dd_start:i])
            events.append({
                'start_date': dates[dd_start],
                'trough_date': dates[dd_min_idx],
                'recovery_date': dates[i],
                'depth': float(drawdown[dd_min_idx]),
                'duration_periods': i - dd_start,
                'duration_days': (i - dd_start) * days,
            })
            in_dd = False

    # Handle ongoing drawdown
    if in_dd:
        dd_min_idx = dd_start + np.argmin(drawdown[dd_start:])
        events.append({
            'start_date': dates[dd_start],
            'trough_date': dates[dd_min_idx],
            'recovery_date': 'N/A (未恢复)',
            'depth': float(drawdown[dd_min_idx]),
            'duration_periods': len(drawdown) - dd_start,
            'duration_days': (len(drawdown) - dd_start) * days,
        })

    # Sort by depth (most negative first)
    events.sort(key=lambda x: x['depth'])
    top5 = events[:5]

    # Underwater time percentage
    underwater_pct = np.mean(drawdown < 0) * 100

    return {
        'top5_events': top5,
        'underwater_pct': underwater_pct,
        'max_drawdown': float(np.min(drawdown)),
        'n_drawdown_events': len(events),
        'avg_drawdown_depth': np.mean([e['depth'] for e in events]) if events else 0,
    }


# ═══════════════════════════════════════════════════════════════
# Module G: 交易成本敏感度
# ═══════════════════════════════════════════════════════════════

def module_g_cost_sensitivity(backtest_result, focus_days):
    """不同成本档位下的收益指标"""
    df = backtest_result['daily_results']
    days = focus_days

    sub = df[df['days'] == days].sort_values('date')
    non_overlap = sub if days == 1 else sub.iloc[::days]

    period_ret = non_overlap['avg_top_return'].values
    if len(period_ret) < 5:
        return {}

    cost_levels = [0.0, 0.0015, 0.00302, 0.005]
    results = []

    for cost in cost_levels:
        net_ret = period_ret - cost
        risk = _compute_period_risk_metrics(pd.Series(net_ret), days)
        results.append({
            'cost_pct': cost * 100,
            'annual_return': risk['annual_return'],
            'sharpe': risk['sharpe_ratio'],
            'max_drawdown': risk['max_drawdown'],
            'cumulative': (np.prod(1 + net_ret) - 1),
        })

    # Break-even cost: binary search for cost where annual_return = 0
    lo, hi = 0.0, 0.05
    for _ in range(50):
        mid = (lo + hi) / 2
        net = period_ret - mid
        cum = np.prod(1 + net) - 1
        total_days_covered = len(net) * days
        ann = (1 + cum) ** (252 / max(total_days_covered, 1)) - 1
        if ann > 0:
            lo = mid
        else:
            hi = mid

    breakeven_cost = (lo + hi) / 2

    return {
        'levels': results,
        'breakeven_cost_pct': breakeven_cost * 100,
    }


# ═══════════════════════════════════════════════════════════════
# Module H: Walk-Forward对比
# ═══════════════════════════════════════════════════════════════

def module_h_walk_forward_comparison(backtest_result, focus_days):
    """对比回测IC/ICIR与训练WF OOS指标"""
    # V4.6 WF OOS reference values from training
    wf_reference = {
        3: {'ic': 0.090, 'icir': None},
        5: {'ic': 0.067, 'icir': None},
        10: {'ic': 0.045, 'icir': None},
        15: {'ic': 0.041, 'icir': None},
    }

    summary = backtest_result['summary']
    results = {}

    for hd in [3, 5, 10, 15]:
        if hd not in summary:
            continue
        s = summary[hd]
        wf = wf_reference.get(hd, {})

        backtest_ic = s.get('ic_mean', 0)
        wf_ic = wf.get('ic', 0)
        ratio = backtest_ic / wf_ic if wf_ic > 0 else 0

        results[hd] = {
            'backtest_ic': backtest_ic,
            'backtest_icir': s.get('icir', 0),
            'wf_ic': wf_ic,
            'ratio': ratio,
            'inflated': ratio > 1.3,  # >30% higher = likely inflated
            'credible': 0.7 <= ratio <= 1.3,
        }

    return results


# ═══════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════

def generate_report(version, focus_days, top_n, start_date, end_date,
                    ns_result, yearly, regime, benchmark_ls, stats, drawdown,
                    cost_sens, wf_compare, backtest_result):
    """生成Markdown报告"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    s = ns_result.get('summary', {})

    lines = []
    lines.append(f"# {version.upper()} 严谨长期回测报告")
    lines.append(f"")
    lines.append(f"*生成时间: {now}*")
    lines.append(f"")

    # Section 0: 回测说明
    lines.append(f"## 0. 回测说明")
    lines.append(f"")
    lines.append(f"| 项目 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 数据范围 | {start_date} ~ {end_date} |")
    lines.append(f"| 核心持仓期 | {focus_days}日 |")
    lines.append(f"| 每日选股数 | Top {top_n} |")
    lines.append(f"| 交易成本 | 0.302% (佣金+印花税+滑点+过户费) |")

    # Count trading days
    daily_results = backtest_result.get('daily_results')
    n_trading_days = 0
    if daily_results is not None:
        n_trading_days = len(daily_results[daily_results['days'] == focus_days])
    lines.append(f"| 回测交易日 | {n_trading_days} |")
    n_non_overlap = n_trading_days // focus_days if focus_days > 1 else n_trading_days
    lines.append(f"| 非重叠调仓期 | {n_non_overlap} |")
    lines.append(f"")
    lines.append(f"> **重要声明**: {version.upper()}模型训练数据包含2020-2026，回测并非真正OOS。")
    lines.append(f"> 本报告通过多维度分析提供可信度评估，但结果应谨慎解读。")
    lines.append(f"")

    # Section 1: 北极星V2评分卡
    lines.append(f"## 1. 全期间北极星V2评分卡")
    lines.append(f"")
    lines.append(f"**总分: {ns_result['total_score']}/{ns_result['max_score']} "
                 f"({ns_result['total_score']/ns_result['max_score']*100:.0f}%) → {ns_result['grade']}级**")
    lines.append(f"")

    for layer_id in sorted(ns_result['layers'].keys()):
        layer = ns_result['layers'][layer_id]
        lines.append(f"### Layer {layer_id}: {layer['name']} ({layer['score']}/{layer['max']})")
        lines.append(f"")
        lines.append(f"| 指标 | 当前值 | 评分 | 评级 |")
        lines.append(f"|------|--------|------|------|")

        for metric_key, target_info in NORTH_STAR_TARGETS_V2.items():
            if target_info['layer'] != layer_id:
                continue
            m = ns_result['metrics'].get(metric_key)
            if m is None:
                continue  # 跳过None指标（数据不足）
            val = m.get('value', 0)
            # Format value
            pct_keys = {'max_drawdown', 'annual_return', 'annual_cost_drag',
                        'net_gross_ratio', 'limit_up_fail_rate', 'liquidity_coverage',
                        'half_period_consistency', 'cap_balance_ratio'}
            if metric_key in pct_keys:
                val_str = f"{val:.1%}"
            elif metric_key in {'ic_positive_pct', 'monthly_win_rate', 'annual_turnover',
                                'signal_half_life', 'median_market_cap_bn'}:
                val_str = f"{val:.1f}"
            else:
                val_str = f"{val:.3f}"
            lines.append(f"| {m.get('display', metric_key)} | {val_str} | {m.get('score', 0)}/5 | {m.get('grade', '')} |")
        lines.append(f"")

    # Section 2: 逐年滚动分析
    lines.append(f"## 2. 逐年滚动分析 ({focus_days}日持仓)")
    lines.append(f"")
    if yearly:
        lines.append(f"| 年份 | 天数 | IC | ICIR | 累计收益 | 年化收益 | MaxDD | Sharpe | 多空价差 | 胜率 |")
        lines.append(f"|------|------|-----|------|----------|----------|-------|--------|----------|------|")
        for y in yearly:
            lines.append(f"| {y['window']} | {y['n_days']} | {y['ic_mean']:.3f} | {y['icir']:.3f} "
                         f"| {y['cumulative_return']:.1%} | {y['annual_return']:.1%} "
                         f"| {y['max_drawdown']:.1%} | {y['sharpe']:.2f} "
                         f"| {y['spread_bps']:+.2f}% | {y['win_rate']:.0f}% |")
    else:
        lines.append("*数据不足*")
    lines.append(f"")

    # Section 3: 市况分解
    lines.append(f"## 3. 市况分解 ({focus_days}日持仓)")
    lines.append(f"")
    if regime:
        lines.append(f"| 市况 | 月份数 | 天数 | IC | ICIR | IC>0% | Top均收益 | 多空价差 |")
        lines.append(f"|------|--------|------|-----|------|-------|-----------|----------|")
        for name, data in regime.items():
            if data.get('n_days', 0) > 0:
                lines.append(f"| {name} | {data['n_months']} | {data['n_days']} "
                             f"| {data.get('ic_mean', 0):.3f} | {data.get('icir', 0):.3f} "
                             f"| {data.get('ic_pos_pct', 0):.0f}% | {data.get('avg_top_return', 0):+.3f}% "
                             f"| {data.get('spread_bps', 0):+.3f}% |")
            else:
                lines.append(f"| {name} | {data.get('n_months', 0)} | 0 | - | - | - | - | - |")
    else:
        lines.append("*数据不足*")
    lines.append(f"")

    # Section 4: 基准对比
    lines.append(f"## 4. 基准对比 + 多空分析")
    lines.append(f"")
    if benchmark_ls:
        for bm_name in ['中证500', '中证2000']:
            bm = benchmark_ls.get(bm_name, {})
            if bm:
                lines.append(f"### vs {bm_name}")
                lines.append(f"")
                lines.append(f"| 指标 | 值 |")
                lines.append(f"|------|-----|")
                lines.append(f"| Alpha (年化) | {bm.get('alpha', 0):.1%} |")
                lines.append(f"| Beta | {bm.get('beta', 0):.3f} |")
                lines.append(f"| Information Ratio | {bm.get('information_ratio', 0):.3f} |")
                lines.append(f"| 超额年化收益 | {bm.get('excess_annual_return', 0):.1%} |")
                lines.append(f"")

        ls = benchmark_ls.get('long_short', {})
        if ls:
            lines.append(f"### 多空分析 (Top{top_n} vs Bottom{top_n})")
            lines.append(f"")
            lines.append(f"| 指标 | 值 |")
            lines.append(f"|------|-----|")
            lines.append(f"| 平均多空价差 | {ls['mean_spread']:+.3f}% |")
            lines.append(f"| 价差标准差 | {ls['std_spread']:.3f}% |")
            lines.append(f"| t统计量 | {ls['t_stat']:.3f} |")
            lines.append(f"| p值 | {ls['p_value']:.4f} |")
            lines.append(f"| 正价差占比 | {ls['pct_positive']:.1f}% |")
            lines.append(f"| 非重叠期数 | {ls['n_periods']} |")
            sig = "**显著**" if ls['p_value'] < 0.05 else "不显著"
            lines.append(f"| 统计显著性(5%) | {sig} |")
            lines.append(f"")

    # Section 5: 统计显著性
    lines.append(f"## 5. 统计显著性检验")
    lines.append(f"")

    if 'ic_ttest' in stats:
        tt = stats['ic_ttest']
        lines.append(f"### 5.1 IC t检验 (H0: IC=0)")
        lines.append(f"")
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 非重叠IC样本数 | {tt['n_samples']} |")
        lines.append(f"| IC均值 | {tt['ic_mean']:.4f} |")
        lines.append(f"| IC标准差 | {tt['ic_std']:.4f} |")
        lines.append(f"| t统计量 | {tt['t_stat']:.3f} |")
        lines.append(f"| p值 | {tt['p_value']:.4f} |")
        sig5 = "是" if tt['significant_5pct'] else "否"
        sig1 = "是" if tt['significant_1pct'] else "否"
        lines.append(f"| 5%显著 | {sig5} |")
        lines.append(f"| 1%显著 | {sig1} |")
        lines.append(f"")

    if 'bootstrap' in stats:
        bs = stats['bootstrap']
        lines.append(f"### 5.2 Bootstrap 95% 置信区间 ({bs['n_boot']}次)")
        lines.append(f"")
        lines.append(f"| 指标 | 2.5% | 中位数 | 97.5% |")
        lines.append(f"|------|------|--------|-------|")
        lines.append(f"| 年化收益 | {bs['annual_return_ci_5']:.1%} | {bs['annual_return_median']:.1%} | {bs['annual_return_ci_95']:.1%} |")
        lines.append(f"| Sharpe | {bs['sharpe_ci_5']:.2f} | {bs['sharpe_median']:.2f} | {bs['sharpe_ci_95']:.2f} |")
        lines.append(f"")

    if 'monte_carlo' in stats:
        mc = stats['monte_carlo']
        lines.append(f"### 5.3 Monte Carlo 随机基线 ({mc['n_simulations']}次模拟)")
        lines.append(f"")
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 随机选股年化均值 | {mc['random_mean_annual']:.1%} |")
        lines.append(f"| 随机选股年化中位数 | {mc['random_median_annual']:.1%} |")
        lines.append(f"| 随机选股年化标准差 | {mc['random_std_annual']:.1%} |")
        lines.append(f"| 随机选股P5~P95 | {mc['random_p5']:.1%} ~ {mc['random_p95']:.1%} |")
        lines.append(f"| **{version.upper()}实际年化** | **{mc['actual_annual']:.1%}** |")
        lines.append(f"| **百分位排名** | **{mc['percentile_rank']:.1f}%** |")
        skill = "有选股技能" if mc['percentile_rank'] > 75 else ("边际技能" if mc['percentile_rank'] > 55 else "无明显技能")
        lines.append(f"| 结论 | {skill} |")
        lines.append(f"")

    # Section 6: 回撤分析
    lines.append(f"## 6. 回撤分析")
    lines.append(f"")
    if drawdown:
        lines.append(f"- 最大回撤: **{drawdown['max_drawdown']:.1%}**")
        lines.append(f"- 水下时间占比: {drawdown['underwater_pct']:.1f}%")
        lines.append(f"- 回撤事件总数: {drawdown['n_drawdown_events']}")
        lines.append(f"- 平均回撤深度: {drawdown['avg_drawdown_depth']:.1%}")
        lines.append(f"")

        if drawdown.get('top5_events'):
            lines.append(f"### Top 5 最大回撤事件")
            lines.append(f"")
            lines.append(f"| # | 起始日 | 最低点 | 恢复日 | 深度 | 持续天数 |")
            lines.append(f"|---|--------|--------|--------|------|----------|")
            for i, evt in enumerate(drawdown['top5_events']):
                lines.append(f"| {i+1} | {evt['start_date']} | {evt['trough_date']} "
                             f"| {evt['recovery_date']} | {evt['depth']:.1%} | {evt['duration_days']}天 |")
        lines.append(f"")

    # Section 7: 交易成本敏感度
    lines.append(f"## 7. 交易成本敏感度")
    lines.append(f"")
    if cost_sens:
        lines.append(f"| 单边成本 | 年化收益 | Sharpe | MaxDD | 累计收益 |")
        lines.append(f"|----------|----------|--------|-------|----------|")
        for lv in cost_sens.get('levels', []):
            lines.append(f"| {lv['cost_pct']:.2f}% | {lv['annual_return']:.1%} | {lv['sharpe']:.2f} "
                         f"| {lv['max_drawdown']:.1%} | {lv['cumulative']:.1%} |")
        lines.append(f"")
        lines.append(f"**盈亏平衡成本**: {cost_sens.get('breakeven_cost_pct', 0):.3f}% (高于此成本则亏损)")
        lines.append(f"")

    # Section 8: Walk-Forward对比
    lines.append(f"## 8. Walk-Forward 训练IC对比")
    lines.append(f"")
    if wf_compare:
        lines.append(f"| 持仓期 | 回测IC | WF OOS IC | 比值 | 判定 |")
        lines.append(f"|--------|--------|-----------|------|------|")
        for hd, data in sorted(wf_compare.items()):
            status = "可信" if data['credible'] else ("膨胀⚠️" if data['inflated'] else "偏低")
            lines.append(f"| {hd}日 | {data['backtest_ic']:.4f} | {data['wf_ic']:.4f} "
                         f"| {data['ratio']:.2f} | {status} |")
        lines.append(f"")
        lines.append(f"> 比值在0.7~1.3之间为可信；>1.3可能样本内膨胀；<0.7可能过拟合衰减")
        lines.append(f"")

    # Section 9: 综合结论
    lines.append(f"## 9. 综合结论与可信度评估")
    lines.append(f"")

    # Compute overall credibility
    credibility_flags = []

    # IC significance
    if 'ic_ttest' in stats:
        if stats['ic_ttest']['significant_5pct']:
            credibility_flags.append(("IC统计显著 (p<0.05)", True))
        else:
            credibility_flags.append(("IC不显著 (p≥0.05)", False))

    # Bootstrap
    if 'bootstrap' in stats:
        bs = stats['bootstrap']
        if bs['annual_return_ci_5'] > 0:
            credibility_flags.append(("年化收益95%CI下限>0", True))
        else:
            credibility_flags.append(("年化收益95%CI包含0", False))

    # Monte Carlo
    if 'monte_carlo' in stats:
        mc = stats['monte_carlo']
        if mc['percentile_rank'] > 75:
            credibility_flags.append((f"Monte Carlo P{mc['percentile_rank']:.0f} (>P75)", True))
        else:
            credibility_flags.append((f"Monte Carlo P{mc['percentile_rank']:.0f} (≤P75)", False))

    # Walk-Forward
    if wf_compare:
        wf_focus = wf_compare.get(focus_days, wf_compare.get(10, {}))
        if wf_focus.get('credible'):
            credibility_flags.append(("WF IC比值可信", True))
        elif wf_focus.get('inflated'):
            credibility_flags.append(("WF IC膨胀 (样本内偏高)", False))

    # Regime consistency
    if regime:
        bear_data = regime.get('熊市(<-3%)', {})
        if bear_data.get('ic_mean', 0) > 0:
            credibility_flags.append(("熊市IC>0 (全天候)", True))
        else:
            credibility_flags.append(("熊市IC≤0 (牛市依赖)", False))

    # Year consistency
    if yearly:
        positive_years = sum(1 for y in yearly if y['annual_return'] > 0)
        total_years = len(yearly)
        if positive_years >= total_years * 0.7:
            credibility_flags.append((f"{positive_years}/{total_years}年正收益", True))
        else:
            credibility_flags.append((f"{positive_years}/{total_years}年正收益 (不稳定)", False))

    lines.append(f"### 可信度检查清单")
    lines.append(f"")
    lines.append(f"| 检查项 | 结果 |")
    lines.append(f"|--------|------|")
    for flag_name, passed in credibility_flags:
        icon = "PASS" if passed else "FAIL"
        lines.append(f"| {flag_name} | {icon} |")

    n_pass = sum(1 for _, p in credibility_flags if p)
    n_total = len(credibility_flags)
    lines.append(f"")
    lines.append(f"**可信度: {n_pass}/{n_total} ({n_pass/n_total*100:.0f}%)**")
    lines.append(f"")

    # Overall assessment
    grade = ns_result['grade']
    total_score = ns_result['total_score']
    lines.append(f"### 总体评估")
    lines.append(f"")
    lines.append(f"- 北极星V2: **{total_score}/105 ({total_score/105*100:.0f}%) {grade}级**")
    if s:
        lines.append(f"- 年化收益(净): **{s.get('net_annual_return', s.get('annual_return', 0)):.1%}**")
        lines.append(f"- Sharpe: **{s.get('sharpe_ratio', 0):.2f}**")
        lines.append(f"- 最大回撤: **{s.get('max_drawdown', 0):.1%}**")
        lines.append(f"- ICIR: **{s.get('icir', 0):.3f}**")
    lines.append(f"")

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='严谨长期回测系统')
    parser.add_argument('--version', default='v4.6',
                        help='模型版本 (default: v4.6)')
    parser.add_argument('--start-date', default='2020-06-01',
                        help='开始日期 (default: 2020-06-01)')
    parser.add_argument('--end-date', default='2026-02-13',
                        help='结束日期 (default: 2026-02-13)')
    parser.add_argument('--top-n', type=int, default=10,
                        help='每日选股数 (default: 10)')
    parser.add_argument('--focus-days', type=int, default=10,
                        help='核心持仓天数 (default: 10)')
    parser.add_argument('--report-dir', default=None,
                        help='报告目录 (default: reports/daily_selection_{version}_full_history)')
    parser.add_argument('--skip-generation', action='store_true',
                        help='跳过报告生成')
    parser.add_argument('--benchmark', default='000905.SH',
                        help='基准指数 (default: 000905.SH)')
    args = parser.parse_args()

    report_dir = args.report_dir or f'reports/daily_selection_{args.version}_full_history'

    print(f"{'='*70}")
    print(f"  {args.version.upper()} 严谨长期回测系统")
    print(f"  {args.start_date} ~ {args.end_date}")
    print(f"  Top-N: {args.top_n}, 持仓: {args.focus_days}日")
    print(f"{'='*70}")

    # Step 1: Generate reports
    if not args.skip_generation:
        print(f"\n[Step 1/4] 生成历史报告...")
        cmd = [
            sys.executable, 'backtest/batch_generate_v395_reports.py',
            '--version', args.version,
            '--start-date', args.start_date,
            '--end-date', args.end_date,
            '--output-dir', report_dir,
        ]
        print(f"  命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"  报告生成失败!")
            return
    else:
        print(f"\n[Step 1/4] 跳过报告生成 (--skip-generation)")

    # Count reports
    report_path = Path(report_dir)
    n_reports = len(list(report_path.glob('analysis_data_*.json')))
    print(f"\n  报告目录: {report_dir}")
    print(f"  报告数量: {n_reports}")
    if n_reports < 10:
        print(f"  报告数量不足，无法进行有效回测")
        return

    # Step 2: Run core backtest
    print(f"\n[Step 2/4] 运行核心回测...")
    reports = load_reports(report_dir)
    print(f"  加载 {len(reports)} 天报告")

    # Optimize: limit HOLDING_DAYS to reduce computation
    # Full [1,3,5,7,10,15] takes 5h+ for 1390 days; [3,5,10,15] takes ~2h
    import backtest.backtest_report_based as _brb
    _original_holding_days = _brb.HOLDING_DAYS
    _brb.HOLDING_DAYS = [3, 5, 10, 15]
    print(f"  持仓期: {_brb.HOLDING_DAYS} (优化模式，跳过1d/7d)")

    backtest_result = run_single_backtest(
        reports, label=f'{args.version.upper()} Long-Term',
        top_n=args.top_n, benchmark_code=args.benchmark,
        focus_days=args.focus_days,
    )

    # Restore
    _brb.HOLDING_DAYS = _original_holding_days

    if not backtest_result:
        print("  回测失败!")
        return

    # Step 3: 8个分析模块
    print(f"\n[Step 3/4] 运行8个分析模块...")

    print(f"  [A] 北极星V2评分...")
    ns_result = module_a_north_star_v2(backtest_result, args.focus_days)
    print(f"      → {ns_result['total_score']}/{ns_result['max_score']} {ns_result['grade']}级")

    print(f"  [B] 逐年滚动分析...")
    yearly = module_b_yearly_rolling(backtest_result, args.focus_days)
    print(f"      → {len(yearly)}个窗口")

    print(f"  [C] 市况分解...")
    regime = module_c_regime_analysis(backtest_result, args.focus_days, args.benchmark)
    print(f"      → {len(regime)}种市况")

    print(f"  [D] 基准对比 + 多空分析...")
    benchmark_ls = module_d_benchmark_longshort(backtest_result, args.focus_days)
    for bm_name in ['中证500', '中证2000']:
        bm = benchmark_ls.get(bm_name, {})
        if bm:
            print(f"      → vs {bm_name}: Alpha={bm.get('alpha', 0):.1%}, IR={bm.get('information_ratio', 0):.3f}")

    print(f"  [E] 统计显著性检验...")
    stats = module_e_statistical_significance(backtest_result, args.focus_days)
    if 'ic_ttest' in stats:
        tt = stats['ic_ttest']
        print(f"      → IC t检验: t={tt['t_stat']:.2f}, p={tt['p_value']:.4f}")
    if 'monte_carlo' in stats:
        mc = stats['monte_carlo']
        print(f"      → Monte Carlo: P{mc['percentile_rank']:.0f}")

    print(f"  [F] 回撤分析...")
    drawdown = module_f_drawdown_analysis(backtest_result, args.focus_days)
    if drawdown:
        print(f"      → MaxDD={drawdown['max_drawdown']:.1%}, 水下{drawdown['underwater_pct']:.0f}%")

    print(f"  [G] 交易成本敏感度...")
    cost_sens = module_g_cost_sensitivity(backtest_result, args.focus_days)
    if cost_sens:
        print(f"      → 盈亏平衡成本: {cost_sens['breakeven_cost_pct']:.3f}%")

    print(f"  [H] Walk-Forward对比...")
    wf_compare = module_h_walk_forward_comparison(backtest_result, args.focus_days)
    for hd, data in sorted(wf_compare.items()):
        status = "可信" if data['credible'] else "膨胀" if data['inflated'] else "偏低"
        print(f"      → {hd}日: 回测IC={data['backtest_ic']:.4f} vs WF={data['wf_ic']:.3f} ({status})")

    # Step 4: Generate report
    print(f"\n[Step 4/4] 生成报告...")
    report_content = generate_report(
        args.version, args.focus_days, args.top_n,
        args.start_date, args.end_date,
        ns_result, yearly, regime, benchmark_ls, stats, drawdown,
        cost_sens, wf_compare, backtest_result,
    )

    # Save report
    report_out_dir = Path('reports/backtest')
    report_out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d')
    report_file = report_out_dir / f'{args.version.upper()}_rigorous_backtest_{date_str}.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"\n{'='*70}")
    print(f"  报告已保存: {report_file}")
    print(f"  总分: {ns_result['total_score']}/{ns_result['max_score']} → {ns_result['grade']}级")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
