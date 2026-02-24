#!/usr/bin/env python3
"""
基于报告的回测引擎 (v2 — 北极星指标增强版)

新增功能:
- Sharpe / Sortino / Calmar 风险调整收益
- 最大回撤 + 持续期 + 恢复时间
- 年化收益率计算
- 交易成本扣减 (佣金+印花税+滑点)
- 换手率统计
- 基准指数对比 (中证500/中证1000)
- 北极星评分卡自动对比

用法:
    # 回测新模型v3.9
    python3 backtest/backtest_report_based.py --report-dir reports/daily_selection_v3.9_model20260222 --label "v3.9新模型"

    # 对比新旧模型
    python3 backtest/backtest_report_based.py \
        --report-dir reports/daily_selection_v3.9_model20260222 \
        --compare-dir reports/daily_selection_v3.9 \
        --label "v3.9新模型" --compare-label "v3.9旧模型"

    # 四模型全面对比
    python3 backtest/backtest_report_based.py --all

    # 指定基准和持仓天数
    python3 backtest/backtest_report_based.py --all --benchmark 000852.SH --focus-days 10
"""
import sys
import os
import sqlite3
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from scipy.stats import spearmanr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from backtest.north_star_metrics import (
    NorthStarEvaluator, compute_risk_metrics, compute_transaction_costs,
    compute_turnover, load_benchmark_returns, compute_benchmark_comparison,
    compute_drawdown_series, NORTH_STAR_TARGETS, TRANSACTION_COST,
    # V2 imports
    NORTH_STAR_TARGETS_V2, V2_LAYER_NAMES, score_metric_v2, compute_v2_grade,
    compute_ic_monotonicity, compute_ic_time_stability, compute_signal_half_life,
    compute_half_period_consistency, compute_worst_rolling_icir, compute_net_gross_ratio,
    batch_load_market_cap_data, batch_load_limit_up_data,
    batch_load_universe_median_cap, compute_executability_metrics,
    classify_market_regime, compute_regime_conditional_metrics,
)

DB_PATH = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')

HOLDING_DAYS = [1, 3, 5, 7, 10, 15]


def load_reports(report_dir):
    """加载所有JSON报告，返回 {date: [{code, score, predicted_return_5d}, ...]}"""
    report_dir = Path(report_dir)
    reports = {}

    for json_file in sorted(report_dir.glob('analysis_data_*.json')):
        date_str = json_file.stem.replace('analysis_data_', '')
        # 转为 YYYY-MM-DD 格式
        date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  跳过 {json_file.name}: {e}")
            continue

        stocks = data.get('all_stocks_with_scores', [])
        if not stocks:
            continue

        stock_list = []
        for s in stocks:
            code = s.get('stock_code', '')
            score = s.get('score', 0)
            pred_ret = s.get('predicted_return_5d', None)
            if code and score > 0:
                stock_list.append({
                    'code': code,
                    'score': score,
                    'predicted_return_5d': pred_ret,
                    'strategies': s.get('strategies', []),
                    'n_strategies': s.get('selected_by_strategies', 1),
                })

        if stock_list:
            # 按分数排序
            stock_list.sort(key=lambda x: x['score'], reverse=True)
            reports[date] = stock_list

    return reports


def get_next_trading_date(trade_date):
    """获取下一个交易日（报告日期的次日开盘买入）"""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("""
        SELECT DISTINCT trade_date FROM daily_quotes
        WHERE trade_date > ?
        ORDER BY trade_date
        LIMIT 1
    """, (trade_date,)).fetchone()
    conn.close()
    return row[0] if row else None


def get_future_returns(codes, buy_date, holding_days_list=None):
    """获取买入日买入后N天的实际收益率（以买入日开盘价买入，持仓N天后收盘价卖出）"""
    if holding_days_list is None:
        holding_days_list = HOLDING_DAYS
    conn = sqlite3.connect(DB_PATH)

    # 获取buy_date当天及之后的交易日
    future_dates = [r[0] for r in conn.execute("""
        SELECT DISTINCT trade_date FROM daily_quotes
        WHERE trade_date >= ?
        ORDER BY trade_date
        LIMIT ?
    """, (buy_date, max(holding_days_list) + 2)).fetchall()]

    if not future_dates or future_dates[0] != buy_date:
        conn.close()
        return {}

    codes_str = ','.join([f"'{c}'" for c in codes])

    # 获取买入日开盘价（次日开盘买入）
    buy_prices = {}
    rows = conn.execute(f"""
        SELECT s.code, dq.open
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.code IN ({codes_str}) AND dq.trade_date = ?
    """, (buy_date,)).fetchall()
    for code, open_price in rows:
        if open_price and open_price > 0:
            buy_prices[code] = open_price

    # 获取未来各天收盘价
    results = {}
    for days in holding_days_list:
        idx = days  # future_dates[0] = buy_date, future_dates[days] = buy+days天
        if idx >= len(future_dates):
            continue
        sell_date = future_dates[idx]

        sell_rows = conn.execute(f"""
            SELECT s.code, dq.close
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.code IN ({codes_str}) AND dq.trade_date = ?
        """, (sell_date,)).fetchall()

        for code, close in sell_rows:
            if code in buy_prices and buy_prices[code] > 0:
                ret = (close - buy_prices[code]) / buy_prices[code]
                if code not in results:
                    results[code] = {}
                results[code][f'return_{days}d'] = ret

    conn.close()
    return results


def _compute_period_risk_metrics(period_returns: pd.Series, holding_days: int,
                                  risk_free_rate: float = 0.02) -> dict:
    """
    计算基于调仓期收益的风险指标（正确年化）

    Args:
        period_returns: 每个调仓期的收益率序列（每个值覆盖N天）
        holding_days: 持仓天数
        risk_free_rate: 年化无风险利率
    """
    if len(period_returns) < 5:
        return {k: 0 for k in [
            'annual_return', 'annual_volatility', 'downside_volatility',
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'max_drawdown', 'max_dd_duration_days', 'max_dd_recovery_days',
            'var_95', 'cvar_95', 'omega_ratio',
            'monthly_win_rate', 'worst_month', 'best_month',
            'max_consecutive_loss_months', 'n_trading_days', 'positive_day_pct',
        ]}

    returns = period_returns.dropna()
    n_periods = len(returns)
    periods_per_year = 252 / holding_days  # 年化因子

    # 累计收益 & 年化收益
    cumulative_return = (1 + returns).prod() - 1
    # 实际覆盖天数
    total_days = n_periods * holding_days
    annual_return = (1 + cumulative_return) ** (252 / total_days) - 1

    # 波动率 (period-level std × sqrt(periods_per_year))
    period_rf = (1 + risk_free_rate) ** (holding_days / 252) - 1
    annual_volatility = returns.std() * np.sqrt(periods_per_year)

    # 下行波动率
    negative_returns = returns[returns < 0]
    downside_vol = negative_returns.std() * np.sqrt(periods_per_year) if len(negative_returns) > 0 else 1e-8

    # Sharpe = (annual_return - Rf) / annual_vol
    sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 1e-8 else 0

    # Sortino
    sortino = (annual_return - risk_free_rate) / downside_vol if downside_vol > 1e-8 else 0

    # 回撤
    dd_series = compute_drawdown_series(returns)
    max_dd = dd_series.min()

    # 回撤持续期（用period数 × holding_days估算天数）
    max_dd_periods = 0
    current = 0
    cumul = (1 + returns).cumprod()
    running_max = cumul.cummax()
    is_underwater = cumul < running_max
    for uw in is_underwater:
        if uw:
            current += 1
            max_dd_periods = max(max_dd_periods, current)
        else:
            current = 0

    max_dd_duration_days = max_dd_periods * holding_days

    # 恢复时间（使用位置索引避免非唯一DatetimeIndex的KeyError）
    recovery_periods = 0
    if not dd_series.empty:
        min_pos = dd_series.values.argmin()
        after_dd = cumul.iloc[min_pos:]
        pre_dd_max_val = running_max.iloc[min_pos]
        recovered = after_dd >= pre_dd_max_val
        if recovered.any():
            recovery_periods = recovered.values.argmax() + 1
        else:
            recovery_periods = len(after_dd)
    recovery_days = recovery_periods * holding_days

    # Calmar
    calmar = annual_return / abs(max_dd) if abs(max_dd) > 1e-8 else 0

    # VaR / CVaR (period-level)
    var_95 = np.percentile(returns, 5)
    tail = returns[returns <= var_95]
    cvar_95 = tail.mean() if len(tail) > 0 else var_95

    # Omega
    gains = returns[returns > period_rf] - period_rf
    losses = period_rf - returns[returns <= period_rf]
    omega = gains.sum() / losses.sum() if losses.sum() > 1e-8 else float('inf')

    # 月度统计 (按实际日历月份分组，比固定chunk更准确)
    if isinstance(returns.index, pd.DatetimeIndex):
        monthly_groups = returns.groupby(returns.index.to_period('M'))
        monthly_returns = monthly_groups.apply(lambda x: (1 + x).prod() - 1)
    else:
        # 回退到固定chunk方式
        periods_per_month = max(1, int(periods_per_year / 12))
        monthly_returns_list = []
        for i in range(0, n_periods, periods_per_month):
            chunk = returns.iloc[i:i+periods_per_month]
            monthly_returns_list.append((1 + chunk).prod() - 1)
        monthly_returns = pd.Series(monthly_returns_list)

    monthly_win_rate = (monthly_returns > 0).mean() * 100 if len(monthly_returns) > 0 else 0
    worst_month = monthly_returns.min() if len(monthly_returns) > 0 else 0
    best_month = monthly_returns.max() if len(monthly_returns) > 0 else 0

    # 连续亏损月
    max_consec = 0
    curr = 0
    for r in monthly_returns:
        if r < 0:
            curr += 1
            max_consec = max(max_consec, curr)
        else:
            curr = 0

    # 日度胜率（period级别）
    positive_pct = (returns > 0).mean() * 100

    return {
        'annual_return': annual_return,
        'cumulative_return': cumulative_return,
        'annual_volatility': annual_volatility,
        'downside_volatility': downside_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_dd,
        'max_dd_duration_days': max_dd_duration_days,
        'max_dd_recovery_days': recovery_days,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'omega_ratio': omega,
        'monthly_win_rate': monthly_win_rate,
        'worst_month': worst_month,
        'best_month': best_month,
        'max_consecutive_loss_months': max_consec,
        'n_trading_days': total_days,
        'positive_day_pct': positive_pct,
    }


def _aggregate_benchmark_to_periods(benchmark_daily: pd.Series,
                                     rebal_dates: list, holding_days: int) -> pd.Series:
    """
    将日度基准收益聚合到与调仓周期匹配的N日收益

    Args:
        benchmark_daily: 日度基准收益率 (DatetimeIndex)
        rebal_dates: 调仓日期列表 (str YYYY-MM-DD)
        holding_days: 持仓天数
    """
    bm = benchmark_daily.sort_index()
    results = {}
    for date_str in rebal_dates:
        dt = pd.Timestamp(date_str)
        # 取该日起的N个交易日的基准收益
        mask = bm.index >= dt
        period_bm = bm[mask].iloc[:holding_days]
        if len(period_bm) >= max(1, holding_days // 2):
            # N日累计收益
            period_return = (1 + period_bm).prod() - 1
            results[dt] = period_return
    if not results:
        return pd.Series(dtype=float)
    s = pd.Series(results)
    s.index = pd.DatetimeIndex(s.index)
    return s


def run_single_backtest(reports, label, top_n=20, benchmark_code='000905.SH',
                        focus_days=10, retention_bonus=0.0):
    """运行单个报告目录的回测（含北极星指标）

    Args:
        retention_bonus: 0.0-1.0, 持仓保留加分比例。>0时已持有股票得到分数加成,
                        减少换手率。0.0=无加成(默认), 0.3=30%加成
    """
    print(f"\n{'='*80}")
    print(f"  报告回测: {label}")
    print(f"  报告天数: {len(reports)}, Top N: {top_n}")
    if retention_bonus > 0:
        print(f"  持仓保留加分: {retention_bonus:.0%}")
    print(f"{'='*80}\n")

    daily_results = []
    all_picks = []
    holdings_by_date = {}  # 用于换手率计算
    skipped = 0
    prev_top_codes = set()

    dates = sorted(reports.keys())
    for i, date in enumerate(dates):
        stocks = reports[date]

        # 持仓保留加分: 已持有股票的score乘以(1+bonus)
        if retention_bonus > 0 and prev_top_codes:
            adjusted_stocks = []
            for s in stocks:
                s_copy = dict(s)
                if s['code'] in prev_top_codes:
                    s_copy['score'] = s['score'] * (1 + retention_bonus)
                adjusted_stocks.append(s_copy)
            adjusted_stocks.sort(key=lambda x: x['score'], reverse=True)
            top_stocks = adjusted_stocks[:top_n]
        else:
            top_stocks = stocks[:top_n]

        bottom_stocks = stocks[-top_n:] if len(stocks) >= top_n * 2 else stocks[-(len(stocks)//2):]

        # 买入日 = 报告日的下一个交易日
        buy_date = get_next_trading_date(date)
        if not buy_date:
            skipped += 1
            continue

        top_codes = [s['code'] for s in top_stocks]
        bottom_codes = [s['code'] for s in bottom_stocks]
        # 记录持仓（用于换手率）
        holdings_by_date[date] = top_codes
        prev_top_codes = set(top_codes)

        # 查询所有候选股票的收益（用于逐日IC计算）
        all_codes = list(set([s['code'] for s in stocks]))

        future_returns = get_future_returns(all_codes, buy_date, HOLDING_DAYS)

        if not future_returns:
            skipped += 1
            continue

        for days in HOLDING_DAYS:
            key = f'return_{days}d'
            top_returns = [future_returns.get(c, {}).get(key, 0) for c in top_codes
                          if key in future_returns.get(c, {})]
            bottom_returns = [future_returns.get(c, {}).get(key, 0) for c in bottom_codes
                             if key in future_returns.get(c, {})]

            if top_returns:
                avg_top = np.mean(top_returns)
                avg_bottom = np.mean(bottom_returns) if bottom_returns else 0

                daily_results.append({
                    'date': date,
                    'buy_date': buy_date,
                    'days': days,
                    'avg_top_return': avg_top,
                    'avg_bottom_return': avg_bottom,
                    'spread': avg_top - avg_bottom,
                    'top_positive_pct': np.mean([r > 0 for r in top_returns]),
                    'n_top': len(top_returns),
                    'n_bottom': len(bottom_returns),
                    'n_total_stocks': len(stocks),
                })

        # 记录所有候选股票明细（用于逐日IC计算）
        top_code_set = set(top_codes)
        for s in stocks:
            pick = {
                'date': date,
                'buy_date': buy_date,
                'code': s['code'],
                'score': s['score'],
                'predicted_return_5d': s.get('predicted_return_5d'),
                'n_strategies': s.get('n_strategies', 1),
                'is_top': s['code'] in top_code_set,
            }
            for days in HOLDING_DAYS:
                key = f'return_{days}d'
                pick[key] = future_returns.get(s['code'], {}).get(key, None)
            all_picks.append(pick)

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(dates)}] {date} → 买入{buy_date}: "
                  f"{len(stocks)}只候选, top{min(top_n, len(stocks))}只")

    if skipped:
        print(f"  跳过 {skipped} 天（无交易数据）")

    if not daily_results:
        print("  无回测结果!")
        return None

    df = pd.DataFrame(daily_results)
    picks_df = pd.DataFrame(all_picks)

    # ═══════════════════════════════════════════════════
    # 基础IC/收益统计（保留原有逻辑）
    # ═══════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  {label} 回测结果")
    print(f"{'─'*70}")

    summary = {}
    daily_ic_series = {}

    for days in HOLDING_DAYS:
        sub = df[df['days'] == days]
        if len(sub) == 0:
            continue

        avg_top = sub['avg_top_return'].mean() * 100
        avg_bottom = sub['avg_bottom_return'].mean() * 100
        avg_spread = sub['spread'].mean() * 100
        win_rate = (sub['avg_top_return'] > 0).mean() * 100
        avg_positive_pct = sub['top_positive_pct'].mean() * 100

        # 全局IC
        sub_picks = picks_df[picks_df[f'return_{days}d'].notna()]
        if len(sub_picks) > 10:
            ic, p_val = spearmanr(sub_picks['score'], sub_picks[f'return_{days}d'])
        else:
            ic, p_val = 0, 1

        # 逐日IC序列
        ic_records = []
        for date in sorted(picks_df['date'].unique()):
            day_picks = picks_df[(picks_df['date'] == date) & (picks_df[f'return_{days}d'].notna())]
            if len(day_picks) >= 5:
                day_ic, day_p = spearmanr(day_picks['score'], day_picks[f'return_{days}d'])
                if not np.isnan(day_ic):
                    ic_records.append({'date': date, 'ic': day_ic, 'p_val': day_p, 'n_stocks': len(day_picks)})

        ic_df = pd.DataFrame(ic_records) if ic_records else pd.DataFrame()
        daily_ic_series[days] = ic_df

        # ICIR = mean(daily_IC) / std(daily_IC)
        if len(ic_df) > 5:
            ic_mean = ic_df['ic'].mean()
            ic_std = ic_df['ic'].std()
            icir = ic_mean / ic_std if ic_std > 0 else 0
            ic_positive_pct = (ic_df['ic'] > 0).mean() * 100
        else:
            ic_mean, ic_std, icir, ic_positive_pct = ic, 0, 0, 0

        # 累计收益
        cumulative = (1 + sub['avg_top_return']).prod() - 1

        summary[days] = {
            'avg_top': avg_top,
            'avg_bottom': avg_bottom,
            'spread': avg_spread,
            'win_rate': win_rate,
            'positive_pct': avg_positive_pct,
            'ic': ic,
            'ic_p': p_val,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'icir': icir,
            'ic_positive_pct': ic_positive_pct,
            'cumulative': cumulative * 100,
            'n_days': len(sub),
            'n_ic_days': len(ic_df),
        }

        print(f"\n  📊 {days}日持仓 ({len(sub)}天):")
        print(f"    Top{top_n} 日均收益:    {avg_top:+.3f}%")
        print(f"    Bottom 日均收益:      {avg_bottom:+.3f}%")
        print(f"    多空价差:             {avg_spread:+.3f}%")
        print(f"    Top{top_n} 盈利天数占比: {win_rate:.1f}%")
        print(f"    Top{top_n} 内盈利股占比: {avg_positive_pct:.1f}%")
        print(f"    全局IC (Spearman):    {ic:.4f} (p={p_val:.4f})")
        print(f"    逐日IC均值:           {ic_mean:.4f} ± {ic_std:.4f}")
        print(f"    ICIR:                 {icir:.4f}")
        print(f"    IC>0天数占比:         {ic_positive_pct:.1f}% ({len(ic_df)}天)")

    # ═══════════════════════════════════════════════════
    # 北极星增强: 风险指标 + 交易成本 + 基准对比
    # ═══════════════════════════════════════════════════

    north_star = {}  # 存储所有北极星指标

    # 加载基准数据（一次性）
    all_dates = df['date'].tolist()
    start_d = min(all_dates)
    end_d = max(all_dates)
    benchmark_daily_ret = load_benchmark_returns(
        benchmark_code, start_date=start_d, end_date=end_d
    )

    # V2: 批量加载市值/涨停数据 (3个SQL调用覆盖所有日期)
    all_buy_dates = sorted(set(df['buy_date'].tolist()))
    market_cap_data = batch_load_market_cap_data(all_buy_dates)
    limit_up_data = batch_load_limit_up_data(all_buy_dates)
    universe_median_cap = batch_load_universe_median_cap(all_buy_dates)

    for days in HOLDING_DAYS:
        sub = df[df['days'] == days].sort_values('date')
        if len(sub) == 0:
            continue

        # 构建非重叠调仓期收益序列
        # 对于N日持仓，每N天调仓一次（subsample every N rows）
        # 1日持仓无重叠，直接使用全部数据
        if days == 1:
            non_overlap = sub
        else:
            non_overlap = sub.iloc[::days]

        period_ret_series = non_overlap.set_index('date')['avg_top_return'].sort_index()
        period_ret_series.index = pd.to_datetime(period_ret_series.index)

        # --- 风险指标（非重叠period收益，正确年化）---
        risk = _compute_period_risk_metrics(period_ret_series, days)

        # --- 多偏移量鲁棒月度胜率（消除起始偏移artifact）---
        # 单一偏移量的月度胜率受起始点影响大，改为平均所有可能的偏移量
        if days > 1 and len(sub) > days * 3:
            all_monthly = []
            for offset in range(min(days, len(sub))):
                offset_sub = sub.iloc[offset::days]
                if len(offset_sub) < 3:
                    continue
                offset_rets = offset_sub.set_index('date')['avg_top_return']
                offset_rets.index = pd.to_datetime(offset_rets.index)
                monthly = offset_rets.groupby(offset_rets.index.to_period('M')).apply(
                    lambda x: (1 + x).prod() - 1)
                all_monthly.append(monthly)
            if all_monthly:
                combined = pd.concat(all_monthly, axis=1)
                avg_monthly = combined.mean(axis=1)
                robust_win_rate = (avg_monthly > 0).mean() * 100
                risk['monthly_win_rate'] = robust_win_rate
                risk['worst_month'] = avg_monthly.min()
                risk['best_month'] = avg_monthly.max()

        # --- 换手率（仅在调仓日之间计算）---
        rebal_dates = non_overlap['date'].tolist()
        rebal_holdings = {d: holdings_by_date.get(d, []) for d in rebal_dates
                          if d in holdings_by_date}
        turnover_info = compute_turnover(rebal_holdings)
        avg_turnover = turnover_info.get('avg_turnover', 0.5)
        # 年化换手 = 单次换手(双边) × 年调仓次数
        rebal_freq_annual = 252 / days
        annual_turnover_val = avg_turnover * 2 * rebal_freq_annual

        # --- 交易成本（基于已计算的毛年化收益）---
        cost_info = compute_transaction_costs(
            avg_turnover, days,
            gross_annual_return=risk['annual_return']
        )

        # --- 基准对比（用buy_date对齐，避免日期偏移）---
        benchmark_info = {}
        buy_dates = non_overlap['buy_date'].tolist()
        periods_per_year = 252 / days
        if not benchmark_daily_ret.empty and days == 1:
            # 1日持仓：用buy_date索引portfolio收益，与日度基准对齐
            buy_ret = non_overlap.set_index('buy_date')['avg_top_return'].sort_index()
            buy_ret.index = pd.to_datetime(buy_ret.index)
            benchmark_info = compute_benchmark_comparison(
                buy_ret, benchmark_daily_ret, periods_per_year=252
            )
        elif not benchmark_daily_ret.empty and days > 1:
            # N日持仓：聚合基准到N日收益，按buy_date匹配
            bm_aligned = _aggregate_benchmark_to_periods(
                benchmark_daily_ret, buy_dates, days
            )
            if len(bm_aligned) >= 3:
                buy_ret = non_overlap.set_index('buy_date')['avg_top_return'].sort_index()
                buy_ret.index = pd.to_datetime(buy_ret.index)
                benchmark_info = compute_benchmark_comparison(
                    buy_ret, bm_aligned, periods_per_year=periods_per_year
                )

        # --- V2新增指标 ---
        # IC单调性
        ic_mono = 0
        sub_picks_days = picks_df[picks_df[f'return_{days}d'].notna()]
        if len(sub_picks_days) > 50:
            ic_mono = compute_ic_monotonicity(
                sub_picks_days['score'], sub_picks_days[f'return_{days}d'],
                sub_picks_days['date']
            )

        # IC时间稳定性
        ic_df_days = daily_ic_series.get(days, pd.DataFrame())
        ic_stability = compute_ic_time_stability(ic_df_days)
        ic_time_stability_cv = ic_stability.get('cv', 999.0)

        # 最差滚动ICIR
        worst_rolling = compute_worst_rolling_icir(ic_df_days)

        # 前后半段一致性
        half_consistency = compute_half_period_consistency(period_ret_series, days)

        # 净/毛收益比
        ngr = compute_net_gross_ratio(risk['annual_return'], cost_info['net_annual_return'])

        # 可执行性指标 (使用非重叠调仓期的持仓)
        rebal_holdings_for_exec = {d: holdings_by_date.get(d, []) for d in rebal_dates
                                    if d in holdings_by_date}
        exec_metrics = compute_executability_metrics(
            rebal_holdings_for_exec, market_cap_data, limit_up_data, universe_median_cap
        )

        # 合并到summary
        summary[days].update({
            # 风险指标
            'annual_return': risk['annual_return'],
            'annual_volatility': risk['annual_volatility'],
            'sharpe_ratio': risk['sharpe_ratio'],
            'sortino_ratio': risk['sortino_ratio'],
            'calmar_ratio': risk['calmar_ratio'],
            'max_drawdown': risk['max_drawdown'],
            'max_dd_duration_days': risk['max_dd_duration_days'],
            'max_dd_recovery_days': risk['max_dd_recovery_days'],
            'var_95': risk['var_95'],
            'cvar_95': risk['cvar_95'],
            'omega_ratio': risk['omega_ratio'],
            'monthly_win_rate': risk['monthly_win_rate'],
            'worst_month': risk['worst_month'],
            'best_month': risk['best_month'],
            'max_consecutive_loss_months': risk['max_consecutive_loss_months'],
            # 交易成本
            'annual_cost_drag': cost_info['annual_cost_drag'],
            'net_annual_return': cost_info['net_annual_return'],
            'gross_annual_return': cost_info['gross_annual_return'],
            # 换手率
            'avg_turnover': avg_turnover,
            'annual_turnover': annual_turnover_val,
            # 基准对比
            'alpha': benchmark_info.get('alpha', 0),
            'beta': benchmark_info.get('beta', 0),
            'information_ratio': benchmark_info.get('information_ratio', 0),
            'excess_annual_return': benchmark_info.get('excess_annual_return', 0),
            'benchmark_annual': benchmark_info.get('benchmark_annual', 0),
            # V2新增
            'ic_monotonicity': ic_mono,
            'ic_time_stability': ic_time_stability_cv,
            'worst_rolling_60d_icir': worst_rolling.get('worst_icir', -999),
            'half_period_consistency': half_consistency.get('ratio', 0),
            'net_gross_ratio': ngr,
            'limit_up_fail_rate': exec_metrics.get('limit_up_fail_rate', 0),
            'liquidity_coverage': exec_metrics.get('liquidity_coverage', 0),
            'small_cap_bias_ratio': exec_metrics.get('small_cap_bias_ratio', 0),
            'median_market_cap_bn': exec_metrics.get('median_market_cap_bn', 0),
        })

        north_star[days] = summary[days]

        # 打印增强指标
        n_rebal = len(non_overlap)
        print(f"\n  🎯 {days}日持仓 北极星指标 ({n_rebal}个非重叠调仓期):")
        print(f"    年化收益(毛):  {risk['annual_return']:.1%}")
        print(f"    年化收益(净):  {cost_info['net_annual_return']:.1%}  (扣成本{cost_info['annual_cost_drag']:.1%}/年)")
        print(f"    年化波动率:    {risk['annual_volatility']:.1%}")
        print(f"    Sharpe:        {risk['sharpe_ratio']:.3f}")
        print(f"    Sortino:       {risk['sortino_ratio']:.3f}")
        print(f"    Calmar:        {risk['calmar_ratio']:.3f}")
        print(f"    最大回撤:      {risk['max_drawdown']:.1%} (持续{risk['max_dd_duration_days']}天, 恢复{risk['max_dd_recovery_days']}天)")
        print(f"    VaR(95%):      {risk['var_95']:.2%}")
        print(f"    CVaR(95%):     {risk['cvar_95']:.2%}")
        print(f"    Omega:         {risk['omega_ratio']:.3f}")
        print(f"    月度胜率:      {risk['monthly_win_rate']:.1f}%")
        print(f"    最差月:        {risk['worst_month']:.1%}")
        print(f"    最佳月:        {risk['best_month']:.1%}")
        print(f"    连续亏损月:    {risk['max_consecutive_loss_months']}月")
        print(f"    换手率(单次):  {avg_turnover:.1%}")
        print(f"    换手率(年化):  {annual_turnover_val:.1f}倍 (调仓{rebal_freq_annual:.0f}次/年)")
        print(f"    IC单调性:      {ic_mono:.2f}/5.0")
        print(f"    IC稳定性(CV):  {ic_time_stability_cv:.2f}")
        print(f"    最差60日ICIR:  {worst_rolling.get('worst_icir', 0):.3f}")
        print(f"    前后半段一致:  {half_consistency.get('ratio', 0):.2f}")
        print(f"    净/毛收益比:   {ngr:.2f}")
        print(f"    涨停失败率:    {exec_metrics.get('limit_up_fail_rate', 0):.1%}")
        print(f"    流动性覆盖:    {exec_metrics.get('liquidity_coverage', 0):.1%}")
        print(f"    中位市值:      {exec_metrics.get('median_market_cap_bn', 0):.1f}亿")
        if benchmark_info:
            print(f"    Alpha:         {benchmark_info.get('alpha', 0):.1%}")
            print(f"    Beta:          {benchmark_info.get('beta', 0):.3f}")
            print(f"    信息比率(IR):  {benchmark_info.get('information_ratio', 0):.3f}")
            print(f"    超额年化:      {benchmark_info.get('excess_annual_return', 0):.1%}")

    # ═══════════════════════════════════════════════════
    # V2: 信号半衰期 (需要所有持仓期的ICIR)
    # ═══════════════════════════════════════════════════
    icir_by_days = {}
    for days in HOLDING_DAYS:
        if days in summary and 'icir' in summary[days]:
            icir_by_days[days] = summary[days]['icir']

    signal_hl = compute_signal_half_life(icir_by_days)
    # 写入所有持仓期的summary
    for days in HOLDING_DAYS:
        if days in summary:
            summary[days]['signal_half_life'] = signal_hl

    if signal_hl > 0:
        print(f"\n  信号半衰期: {signal_hl:.1f}天")

    # ═══════════════════════════════════════════════════
    # 北极星评分卡（focus_days）
    # ═══════════════════════════════════════════════════

    if focus_days in summary:
        s = summary[focus_days]
        _print_scorecard(s, label, focus_days)
        _print_scorecard_v2(s, label, focus_days)

    # 月度分解 (5日持仓)
    sub5 = df[df['days'] == 5].copy()
    if len(sub5) > 0:
        print(f"\n  📅 月度收益 (5日持仓):")
        sub5['month'] = pd.to_datetime(sub5['date']).dt.to_period('M')
        for month, group in sub5.groupby('month'):
            monthly_ret = group['avg_top_return'].mean() * 100
            monthly_win = (group['avg_top_return'] > 0).mean() * 100
            n_days = len(group)
            print(f"    {month}: {monthly_ret:+.3f}% (盈利{monthly_win:.0f}%, {n_days}天)")

    # 月度IC分解
    for days in [5, 10]:
        ic_df = daily_ic_series.get(days)
        if ic_df is not None and len(ic_df) > 0:
            print(f"\n  📅 月度IC ({days}日持仓):")
            ic_df_copy = ic_df.copy()
            ic_df_copy['month'] = pd.to_datetime(ic_df_copy['date']).dt.to_period('M')
            for month, group in ic_df_copy.groupby('month'):
                m_ic = group['ic'].mean()
                m_std = group['ic'].std()
                m_icir = m_ic / m_std if m_std > 0 else 0
                m_pos = (group['ic'] > 0).mean() * 100
                print(f"    {month}: IC={m_ic:+.4f} ±{m_std:.4f}, ICIR={m_icir:+.3f}, IC>0={m_pos:.0f}% ({len(group)}天)")

    return {
        'label': label,
        'summary': summary,
        'daily_results': df,
        'picks': picks_df,
        'daily_ic_series': daily_ic_series,
        'holdings_by_date': holdings_by_date,
    }


def _print_scorecard(s, label, days):
    """打印北极星评分卡"""
    print(f"\n  {'═'*60}")
    print(f"  北极星评分卡: {label} ({days}日持仓)")
    print(f"  {'═'*60}")
    print(f"  {'指标':<18s} {'当前值':>10s} {'及格':>8s} {'目标':>8s} {'评级':>6s}")
    print(f"  {'─'*54}")

    scorecard_items = [
        ('Daily IC',       s.get('ic_mean', 0),           'daily_ic'),
        ('ICIR',           s.get('icir', 0),              'icir'),
        ('IC>0%',          s.get('ic_positive_pct', 0),   'ic_positive_pct'),
        ('年化收益(毛)',    s.get('annual_return', 0),     'annual_return'),
        ('Sharpe',         s.get('sharpe_ratio', 0),      'sharpe_ratio'),
        ('Sortino',        s.get('sortino_ratio', 0),     'sortino_ratio'),
        ('Calmar',         s.get('calmar_ratio', 0),      'calmar_ratio'),
        ('最大回撤',       s.get('max_drawdown', 0),      'max_drawdown'),
        ('月度胜率%',      s.get('monthly_win_rate', 0),  'monthly_win_rate'),
        ('年化成本',       s.get('annual_cost_drag', 0),  'annual_cost_drag'),
        ('年化换手',       s.get('annual_turnover', 0),   'annual_turnover'),
    ]

    total_score = 0
    max_score = 0

    for name, current, target_key in scorecard_items:
        tgt = NORTH_STAR_TARGETS.get(target_key)
        if not tgt:
            continue

        target = tgt['target']
        pass_val = tgt['pass']
        good_val = tgt['good']
        higher = tgt['direction'] == 'higher'

        # 评级
        if higher:
            if current >= target:
                grade, score = "★★★", 3
            elif current >= good_val:
                grade, score = "★★☆", 2
            elif current >= pass_val:
                grade, score = "★☆☆", 1
            else:
                grade, score = "☆☆☆", 0
        else:
            if current <= target:
                grade, score = "★★★", 3
            elif current <= good_val:
                grade, score = "★★☆", 2
            elif current <= pass_val:
                grade, score = "★☆☆", 1
            else:
                grade, score = "☆☆☆", 0

        total_score += score
        max_score += 3

        # 格式化
        if target_key in ('max_drawdown', 'annual_return', 'annual_cost_drag'):
            c_str = f"{current:.1%}"
            t_str = f"{target:.1%}"
            p_str = f"{pass_val:.1%}"
        elif target_key in ('ic_positive_pct', 'monthly_win_rate', 'annual_turnover'):
            c_str = f"{current:.1f}"
            t_str = f"{target:.1f}"
            p_str = f"{pass_val:.1f}"
        else:
            c_str = f"{current:.3f}"
            t_str = f"{target:.3f}"
            p_str = f"{pass_val:.3f}"

        print(f"  {name:<18s} {c_str:>10s} {p_str:>8s} {t_str:>8s} {grade:>6s}")

    if max_score > 0:
        pct = total_score / max_score * 100
        if pct >= 80:
            grade_str = "A+"
        elif pct >= 60:
            grade_str = "A"
        elif pct >= 40:
            grade_str = "B"
        else:
            grade_str = "C"
        print(f"\n  综合评分: {total_score}/{max_score} ({pct:.0f}%) → 等级 {grade_str}")

    print(f"  {'═'*60}")


def _print_scorecard_v2(s, label, days):
    """打印V2北极星评分卡 (21项, 6档, /105)"""
    print(f"\n  {'═'*70}")
    print(f"  北极星评分卡 V2: {label} ({days}日持仓)")
    print(f"  {'═'*70}")

    # 映射 summary key → V2 target key
    metric_value_map = {
        'daily_ic':              s.get('ic_mean', 0),
        'icir':                  s.get('icir', 0),
        'ic_positive_pct':       s.get('ic_positive_pct', 0),
        'ic_monotonicity':       s.get('ic_monotonicity', 0),
        'ic_time_stability':     s.get('ic_time_stability', 999),
        'signal_half_life':      s.get('signal_half_life', 0),
        'annual_turnover':       s.get('annual_turnover', 0),
        'annual_cost_drag':      s.get('annual_cost_drag', 0),
        'net_gross_ratio':       s.get('net_gross_ratio', 0),
        'limit_up_fail_rate':    s.get('limit_up_fail_rate', 0),
        'liquidity_coverage':    s.get('liquidity_coverage', 0),
        'max_drawdown':          s.get('max_drawdown', 0),
        'sharpe_ratio':          s.get('sharpe_ratio', 0),
        'sortino_ratio':         s.get('sortino_ratio', 0),
        'calmar_ratio':          s.get('calmar_ratio', 0),
        'worst_rolling_60d_icir': s.get('worst_rolling_60d_icir', -999),
        'annual_return':         s.get('annual_return', 0),
        'monthly_win_rate':      s.get('monthly_win_rate', 0),
        'half_period_consistency': s.get('half_period_consistency', 0),
        'small_cap_bias_ratio':  s.get('small_cap_bias_ratio', 0),
        'median_market_cap_bn':  s.get('median_market_cap_bn', 0),
    }

    # 百分比格式化的指标
    pct_fmt_keys = {'max_drawdown', 'annual_return', 'annual_cost_drag',
                    'net_gross_ratio', 'limit_up_fail_rate', 'liquidity_coverage',
                    'half_period_consistency', 'small_cap_bias_ratio'}
    plain_fmt_keys = {'ic_positive_pct', 'monthly_win_rate', 'annual_turnover',
                      'signal_half_life', 'median_market_cap_bn'}

    total_score = 0
    max_score = 0

    for layer_id in sorted(V2_LAYER_NAMES.keys()):
        layer_name = V2_LAYER_NAMES[layer_id]
        # 获取属于此layer的指标
        layer_metrics = [(k, v) for k, v in NORTH_STAR_TARGETS_V2.items() if v['layer'] == layer_id]
        if not layer_metrics:
            continue

        print(f"\n  ┌─ Layer {layer_id}: {layer_name}")
        print(f"  │ {'指标':<18s} {'当前值':>10s} {'及格':>8s} {'目标':>8s} {'分数':>4s} {'评级':>10s}")
        print(f"  │ {'─'*58}")

        for metric_key, target_info in layer_metrics:
            current = metric_value_map.get(metric_key)
            if current is None:
                continue

            score, grade_str = score_metric_v2(current, target_info)
            total_score += score
            max_score += 5

            display = target_info['display']
            target_val = target_info['target']
            pass_val = target_info['pass']

            # 格式化
            if metric_key in pct_fmt_keys:
                c_str = f"{current:.1%}" if abs(current) < 10 else f"{current:.0%}"
                t_str = f"{target_val:.1%}" if abs(target_val) < 10 else f"{target_val:.0%}"
                p_str = f"{pass_val:.1%}" if abs(pass_val) < 10 else f"{pass_val:.0%}"
            elif metric_key in plain_fmt_keys:
                c_str = f"{current:.1f}"
                t_str = f"{target_val:.1f}"
                p_str = f"{pass_val:.1f}"
            else:
                c_str = f"{current:.3f}"
                t_str = f"{target_val:.3f}"
                p_str = f"{pass_val:.3f}"

            print(f"  │ {display:<18s} {c_str:>10s} {p_str:>8s} {t_str:>8s} {score}/5  {grade_str}")

    # 总分
    print(f"\n  {'─'*70}")
    if max_score > 0:
        pct = total_score / max_score * 100
        grade = compute_v2_grade(total_score, max_score)
        print(f"  综合评分: {total_score}/{max_score} ({pct:.0f}%) → 等级 {grade}")

        # 分层小计
        for layer_id in sorted(V2_LAYER_NAMES.keys()):
            layer_metrics = [k for k, v in NORTH_STAR_TARGETS_V2.items() if v['layer'] == layer_id]
            layer_score = sum(
                score_metric_v2(metric_value_map.get(k, 0), NORTH_STAR_TARGETS_V2[k])[0]
                for k in layer_metrics if k in metric_value_map
            )
            layer_max = len(layer_metrics) * 5
            layer_pct = layer_score / layer_max * 100 if layer_max > 0 else 0
            print(f"    Layer {layer_id} {V2_LAYER_NAMES[layer_id]}: "
                  f"{layer_score}/{layer_max} ({layer_pct:.0f}%)")

    print(f"  {'═'*70}")


def compare_results(result_a, result_b, focus_days=10):
    """对比两个回测结果（含北极星指标）"""
    label_a = result_a['label']
    label_b = result_b['label']

    print(f"\n{'='*80}")
    print(f"  模型对比: {label_a} vs {label_b}")
    print(f"{'='*80}")

    header = f"| 指标 | {label_a} | {label_b} | 差异 | 优胜 |"
    sep = f"|:-----|:----------:|:----------:|:------:|:----:|"

    for days in HOLDING_DAYS:
        sa = result_a['summary'].get(days)
        sb = result_b['summary'].get(days)
        if not sa or not sb:
            continue

        print(f"\n  📊 {days}日持仓对比:")
        print(f"  {header}")
        print(f"  {sep}")

        metrics = [
            ('日均收益',     'avg_top',          '%',  True,  '+.3f'),
            ('多空价差',     'spread',           '%',  True,  '+.3f'),
            ('盈利天数比',   'win_rate',         '%',  True,  '.1f'),
            ('盈利股占比',   'positive_pct',     '%',  True,  '.1f'),
            ('逐日IC均值',   'ic_mean',          '',   True,  '.4f'),
            ('ICIR',         'icir',             '',   True,  '.4f'),
            ('IC>0占比',     'ic_positive_pct',  '%',  True,  '.1f'),
            ('IC单调性',     'ic_monotonicity',  '',   True,  '.2f'),
            ('累计收益',     'cumulative',       '%',  True,  '+.2f'),
            ('年化收益(毛)', 'annual_return',    '',   True,  '.1%'),
            ('年化收益(净)', 'net_annual_return', '',  True,  '.1%'),
            ('净/毛收益比',  'net_gross_ratio',  '',   True,  '.2f'),
            ('Sharpe',       'sharpe_ratio',     '',   True,  '.3f'),
            ('Sortino',      'sortino_ratio',    '',   True,  '.3f'),
            ('Calmar',       'calmar_ratio',     '',   True,  '.3f'),
            ('最大回撤',     'max_drawdown',     '',   True,  '.1%'),
            ('月度胜率',     'monthly_win_rate', '%',  True,  '.1f'),
            ('前后半段一致', 'half_period_consistency', '', True, '.2f'),
            ('最差60日ICIR', 'worst_rolling_60d_icir', '', True, '.3f'),
            ('涨停失败率',   'limit_up_fail_rate', '', False, '.1%'),
            ('流动性覆盖',   'liquidity_coverage', '', True, '.1%'),
            ('中位市值(亿)', 'median_market_cap_bn', '', True, '.1f'),
            ('Alpha',        'alpha',            '',   True,  '.1%'),
            ('信息比率',     'information_ratio', '',  True,  '.3f'),
        ]

        for name, key, unit, higher_better, fmt in metrics:
            va = sa.get(key)
            vb = sb.get(key)
            if va is None or vb is None:
                continue

            diff = va - vb
            if key == 'max_drawdown':
                # 回撤越小越好（数值越大越好，因为是负数）
                winner = label_a if va > vb else label_b if vb > va else "平"
            elif higher_better:
                winner = label_a if va > vb else label_b if vb > va else "平"
            else:
                winner = label_a if va < vb else label_b if vb < va else "平"

            if '%' in fmt:
                va_str = f"{va:{fmt}}"
                vb_str = f"{vb:{fmt}}"
                diff_str = f"{diff:+.2%}"
            else:
                va_str = f"{va:{fmt}}{unit}"
                vb_str = f"{vb:{fmt}}{unit}"
                diff_str = f"{diff:+.3f}"

            print(f"  | {name} | {va_str} | {vb_str} | {diff_str} | {winner} |")


def generate_report(results, output_dir='reports/backtest', benchmark_code='000905.SH',
                    focus_days=10):
    """生成Markdown回测报告（含北极星指标）"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    labels = [r['label'] for r in results]
    label_str = '_vs_'.join([l.replace(' ', '').replace('.', '') for l in labels])

    report_lines = [
        f"# 选股报告回测对比 (北极星指标增强版)",
        f"",
        f"**回测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**基准指数**: {benchmark_code}",
        f"**重点持仓期**: {focus_days}天",
        f"",
        f"## 回测参数",
        f"",
    ]

    for r in results:
        summary = r['summary']
        n_days = summary.get(5, {}).get('n_days', 0) if summary else 0
        report_lines.append(f"- **{r['label']}**: {n_days} 交易日")

    # ═══════════════════════════════════════
    # 北极星评分卡
    # ═══════════════════════════════════════
    report_lines.extend([
        f"",
        f"## 北极星评分卡 ({focus_days}日持仓)",
        f"",
        f"| 指标 |" + " | ".join([r['label'] for r in results]) + " | 北极星目标 |",
        f"|:-----|" + "|".join([":----------:" for _ in results]) + "|:----------:|",
    ])

    scorecard_rows = [
        ('Daily IC',       'ic_mean',           '.4f',  ''),
        ('ICIR',           'icir',              '.4f',  ''),
        ('IC>0%',          'ic_positive_pct',   '.1f',  '%'),
        ('IC单调性',       'ic_monotonicity',   '.2f',  ''),
        ('IC稳定性(CV)',   'ic_time_stability', '.2f',  ''),
        ('信号半衰期',     'signal_half_life',  '.1f',  '天'),
        ('年化收益(毛)',    'annual_return',     '.1%',  ''),
        ('年化收益(净)',    'net_annual_return', '.1%',  ''),
        ('净/毛收益比',    'net_gross_ratio',   '.2f',  ''),
        ('Sharpe',         'sharpe_ratio',      '.3f',  ''),
        ('Sortino',        'sortino_ratio',     '.3f',  ''),
        ('Calmar',         'calmar_ratio',      '.3f',  ''),
        ('最大回撤',       'max_drawdown',      '.1%',  ''),
        ('最差60日ICIR',   'worst_rolling_60d_icir', '.3f', ''),
        ('月度胜率',       'monthly_win_rate',  '.1f',  '%'),
        ('前后半段一致性', 'half_period_consistency', '.2f', ''),
        ('年化成本',       'annual_cost_drag',  '.1%',  ''),
        ('年化换手',       'annual_turnover',   '.1f',  ''),
        ('涨停失败率',     'limit_up_fail_rate', '.1%', ''),
        ('流动性覆盖',     'liquidity_coverage', '.1%', ''),
        ('中位市值(亿)',   'median_market_cap_bn', '.1f', ''),
        ('Alpha',          'alpha',             '.1%',  ''),
        ('信息比率(IR)',   'information_ratio', '.3f',  ''),
    ]

    for name, key, fmt, unit in scorecard_rows:
        row = f"| {name} |"
        for r in results:
            val = r['summary'].get(focus_days, {}).get(key)
            if val is not None:
                if '%' in fmt:
                    row += f" {val:{fmt}} |"
                else:
                    row += f" {val:{fmt}}{unit} |"
            else:
                row += " - |"
        # 北极星V2目标 (优先), fallback V1
        lookup_key = key.replace('net_annual_return', 'annual_return')
        tgt = NORTH_STAR_TARGETS_V2.get(lookup_key, NORTH_STAR_TARGETS.get(lookup_key, {}))
        if tgt:
            t = tgt.get('target', '-')
            if isinstance(t, (int, float)):
                pct_keys = {'annual_return', 'net_annual_return', 'max_drawdown',
                            'annual_cost_drag', 'alpha', 'net_gross_ratio',
                            'limit_up_fail_rate', 'liquidity_coverage',
                            'half_period_consistency', 'small_cap_bias_ratio'}
                plain_keys = {'ic_positive_pct', 'monthly_win_rate', 'annual_turnover',
                              'signal_half_life', 'median_market_cap_bn'}
                if key in pct_keys:
                    row += f" {t:.1%} |"
                elif key in plain_keys:
                    row += f" {t:.1f}{unit} |"
                else:
                    row += f" {t:.3f} |"
            else:
                row += f" {t} |"
        else:
            row += " - |"
        report_lines.append(row)

    report_lines.append("")

    # ═══════════════════════════════════════
    # 综合对比（保留原有表格）
    # ═══════════════════════════════════════
    report_lines.extend([
        f"## 综合对比",
        f"",
    ])

    for days in HOLDING_DAYS:
        report_lines.extend([
            f"### {days}日持仓",
            f"",
            f"| 模型 | 日均收益 | 多空价差 | 盈利天数比 | ICIR | 累计收益 | Sharpe | MaxDD | 年化(净) |",
            f"|:-----|:--------:|:--------:|:----------:|:----:|:--------:|:------:|:-----:|:--------:|",
        ])

        for r in results:
            s = r['summary'].get(days)
            if not s:
                continue
            report_lines.append(
                f"| {r['label']} "
                f"| {s['avg_top']:+.3f}% "
                f"| {s['spread']:+.3f}% "
                f"| {s['win_rate']:.1f}% "
                f"| {s.get('icir', 0):.4f} "
                f"| {s['cumulative']:+.2f}% "
                f"| {s.get('sharpe_ratio', 0):.3f} "
                f"| {s.get('max_drawdown', 0):.1%} "
                f"| {s.get('net_annual_return', 0):.1%} |"
            )
        report_lines.append("")

    # 月度对比 (focus_days持仓)
    report_lines.extend([
        f"## 月度收益对比 ({focus_days}日持仓)",
        f"",
    ])

    months_header = "| 月份 |"
    months_sep = "|:----:|"
    for r in results:
        months_header += f" {r['label']} |"
        months_sep += ":----------:|"
    report_lines.append(months_header)
    report_lines.append(months_sep)

    all_months = set()
    monthly_data = {}
    for r in results:
        df_r = r['daily_results']
        sub = df_r[df_r['days'] == focus_days].copy()
        if len(sub) > 0:
            sub['month'] = pd.to_datetime(sub['date']).dt.to_period('M')
            for month, group in sub.groupby('month'):
                all_months.add(month)
                if month not in monthly_data:
                    monthly_data[month] = {}
                monthly_data[month][r['label']] = group['avg_top_return'].mean() * 100

    for month in sorted(all_months):
        row = f"| {month} |"
        for r in results:
            val = monthly_data.get(month, {}).get(r['label'])
            if val is not None:
                row += f" {val:+.3f}% |"
            else:
                row += " - |"
        report_lines.append(row)

    report_lines.append("")

    # 保存
    report_file = output_dir / f'report_backtest_{label_str}_{timestamp}.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    # 保存选股明细CSV
    for r in results:
        csv_file = output_dir / f'report_backtest_{r["label"].replace(" ", "").replace(".", "")}_{timestamp}_picks.csv'
        r['picks'].to_csv(csv_file, index=False, encoding='utf-8-sig')

    print(f"\n📊 回测报告: {report_file}")
    return str(report_file)


def main():
    parser = argparse.ArgumentParser(description='基于报告的回测引擎 (北极星指标增强版)')
    parser.add_argument('--report-dir', help='主报告目录')
    parser.add_argument('--compare-dir', help='对比报告目录')
    parser.add_argument('--label', default='模型A', help='主报告标签')
    parser.add_argument('--compare-label', default='模型B', help='对比报告标签')
    parser.add_argument('--top-n', type=int, default=20, help='每日选取Top N只 (default: 20)')
    parser.add_argument('--benchmark', default='000905.SH',
                        help='基准指数代码 (default: 000905.SH 中证500)')
    parser.add_argument('--focus-days', type=int, default=10,
                        help='重点评估的持仓天数 (default: 10)')
    parser.add_argument('--all', action='store_true', help='四模型全面对比')
    args = parser.parse_args()

    if args.all:
        configs = [
            ('reports/daily_selection_v3.9', 'v3.9旧模型'),
            ('reports/daily_selection_v3.9_model20260222', 'v3.9新模型'),
            ('reports/daily_selection_v3.95', 'v3.95旧模型'),
            ('reports/daily_selection_v3.95_model20260221', 'v3.95新模型'),
        ]

        results = []
        for dir_path, label in configs:
            if not Path(dir_path).exists():
                print(f"  跳过 {label}: {dir_path} 不存在")
                continue
            reports = load_reports(dir_path)
            print(f"加载 {label}: {len(reports)} 天报告")
            result = run_single_backtest(reports, label, args.top_n,
                                        args.benchmark, args.focus_days)
            if result:
                results.append(result)

        if len(results) >= 2:
            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    compare_results(results[i], results[j], args.focus_days)

            generate_report(results, benchmark_code=args.benchmark,
                          focus_days=args.focus_days)

    elif args.report_dir:
        reports_a = load_reports(args.report_dir)
        print(f"加载 {args.label}: {len(reports_a)} 天报告")

        result_a = run_single_backtest(reports_a, args.label, args.top_n,
                                       args.benchmark, args.focus_days)

        results = [result_a] if result_a else []

        if args.compare_dir:
            reports_b = load_reports(args.compare_dir)
            print(f"加载 {args.compare_label}: {len(reports_b)} 天报告")
            result_b = run_single_backtest(reports_b, args.compare_label, args.top_n,
                                           args.benchmark, args.focus_days)

            if result_a and result_b:
                results.append(result_b)
                compare_results(result_a, result_b, args.focus_days)

        if results:
            generate_report(results, benchmark_code=args.benchmark,
                          focus_days=args.focus_days)
    else:
        parser.print_help()
        print("\n示例:")
        print("  python3 backtest/backtest_report_based.py --all")
        print("  python3 backtest/backtest_report_based.py --all --benchmark 000852.SH --focus-days 10")
        print("  python3 backtest/backtest_report_based.py --report-dir reports/daily_selection_v3.9_model20260222 --label 'v3.9新模型'")


if __name__ == '__main__':
    main()
