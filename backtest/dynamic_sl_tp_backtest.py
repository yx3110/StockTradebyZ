#!/usr/bin/env python3
"""
动态止损/止盈回测引擎 — 支持trailing stop + 参数化SL/TP + 组合仓位

基于 backtest_stop_target_direct.py 的 preload_all_quotes 基础设施,
新增: 移动止盈, portfolio-level仿真, A/B对比.
"""

import sys
import os
import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')


def simulate_trade_with_trailing(
    code: str,
    buy_date: str,
    buy_price: float,
    stop_loss: float,
    target_price: float,
    all_quotes: pd.DataFrame,
    params: dict,
) -> Optional[dict]:
    """
    模拟单笔交易: 限价买入 → trailing stop / 止盈 / 止损 / 到期退出

    与 backtest_stop_target_direct.simulate_trade 的区别:
    1. 支持移动止盈 (trailing stop)
    2. T+1: 买入当天不可卖
    3. 超参数全部从params读取
    4. 返回更详细的退出信息
    """
    max_hold_days = params['hold']['max_hold_days']
    trailing_trigger = params['trailing']['trailing_trigger_pct']
    trailing_fallback = params['trailing']['trailing_fallback_pct']

    try:
        code_quotes = all_quotes.loc[code]
    except KeyError:
        return None

    # 找到 buy_date 之后的交易日 (buy_date是分析日, 次日开盘买入)
    future = code_quotes[code_quotes.index > buy_date].head(max_hold_days + 1)
    if future.empty or buy_price <= 0 or stop_loss <= 0 or target_price <= 0:
        return None

    # Step 1: 限价买入检查 (第一个交易日)
    first_day = future.iloc[0]
    if first_day['low'] > buy_price * 1.005:
        # 无法成交
        return {'outcome': 'no_fill', 'trade_return': 0, 'hold_days': 0,
                'actual_entry': 0, 'exit_price': 0, 'exit_date': None}

    actual_entry = min(first_day['open'], buy_price)
    if actual_entry <= 0:
        return None

    # Step 2: 逐日模拟 (T+1, 从第二天开始可卖)
    current_stop = stop_loss
    highest_since_entry = max(actual_entry, first_day['high'])
    profit_range = target_price - actual_entry

    remaining = future.iloc[1:]  # 跳过买入日
    for day_idx, (date, row) in enumerate(remaining.iterrows()):
        # 止损检查 (优先)
        if row['low'] <= current_stop:
            exit_price = current_stop
            ret = (exit_price - actual_entry) / actual_entry
            return {'outcome': 'stop_loss', 'trade_return': ret,
                    'hold_days': day_idx + 1, 'actual_entry': actual_entry,
                    'exit_price': exit_price, 'exit_date': date}

        # 止盈检查
        if row['high'] >= target_price:
            exit_price = target_price
            ret = (exit_price - actual_entry) / actual_entry
            return {'outcome': 'take_profit', 'trade_return': ret,
                    'hold_days': day_idx + 1, 'actual_entry': actual_entry,
                    'exit_price': exit_price, 'exit_date': date}

        # 更新最高价
        highest_since_entry = max(highest_since_entry, row['high'])

        # 移动止盈: 浮盈超过目标的trigger% → 至少保本
        if profit_range > 0 and highest_since_entry >= actual_entry + profit_range * trailing_trigger:
            current_stop = max(current_stop, actual_entry)  # 保本线

        # Trailing: 最高价回撤 fallback% × profit_range
        if profit_range > 0:
            trail = highest_since_entry - profit_range * trailing_fallback
            current_stop = max(current_stop, trail)

    # 到期: 按最后一天收盘价卖出
    last_row = remaining.iloc[-1] if len(remaining) > 0 else first_day
    exit_price = last_row['close']
    ret = (exit_price - actual_entry) / actual_entry
    exit_date = remaining.index[-1] if len(remaining) > 0 else buy_date
    return {'outcome': 'max_hold', 'trade_return': ret,
            'hold_days': len(remaining), 'actual_entry': actual_entry,
            'exit_price': exit_price, 'exit_date': exit_date}


def preload_backtest_data(report_dir: str, start_date: str = None,
                          end_date: str = None) -> dict:
    """预加载回测数据, 供网格搜索时复用 (避免每次重新加载3M+行)"""
    from backtest.backtest_report_based import load_reports
    from backtest.backtest_stop_target_direct import preload_all_quotes

    reports = load_reports(report_dir, rank_field='composite')
    dates = sorted(reports.keys())
    if start_date:
        dates = [d for d in dates if d >= start_date]
    if end_date:
        dates = [d for d in dates if d <= end_date]

    if not dates:
        return {}

    print(f"  预加载回测数据: {len(dates)}天, {dates[0]} -> {dates[-1]}")
    all_quotes = preload_all_quotes(dates[0], '2026-12-31')

    conn = sqlite3.connect(DB_PATH, timeout=30)
    lookback_dates = [r[0] for r in conn.execute("""
        SELECT DISTINCT trade_date FROM daily_quotes
        WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT 80
    """, (dates[0],)).fetchall()]
    lookback_start = lookback_dates[-1] if lookback_dates else dates[0]
    conn.close()

    kline_quotes = preload_all_quotes(lookback_start, '2026-12-31')

    # 预加载env_scores
    env_scores = {}
    for date in dates:
        json_path = Path(report_dir) / f"analysis_data_{date.replace('-','')}.json"
        if json_path.exists():
            try:
                with open(json_path) as f:
                    data = json.load(f)
                te = data.get('trading_environment', {})
                env_scores[date] = te.get('total_score', 50.0)
            except Exception:
                env_scores[date] = 50.0
        else:
            env_scores[date] = 50.0

    return {
        'reports': reports,
        'dates': dates,
        'all_quotes': all_quotes,
        'kline_quotes': kline_quotes,
        'env_scores': env_scores,
    }


def run_portfolio_backtest(
    report_dir: str,
    params: dict,
    benchmark_codes: List[str] = None,
    start_date: str = None,
    end_date: str = None,
    use_optimizer: bool = True,
    label: str = 'Optimized',
    preloaded_data: dict = None,
    quiet: bool = False,
) -> dict:
    """
    组合级别回测: 读取报告 → 计算价格 → 模拟交易 → 聚合收益

    Args:
        report_dir: 含 analysis_data_*.json 的报告目录
        params: 超参数 (来自optimizer_params.json)
        benchmark_codes: 基准指数列表, 默认 ['000300.SH', '932000.CSI']
        use_optimizer: True=新逻辑, False=旧逻辑(对照组)
        label: 回测标签
        preloaded_data: 预加载的数据缓存 {reports, dates, all_quotes, kline_quotes, env_scores}
                       用于网格搜索时避免重复加载
        quiet: 静默模式 (减少打印)

    Returns:
        {
            'label': str,
            'trades': List[dict],  # 所有交易记录
            'metrics': dict,  # 年化收益/Sharpe/MaxDD/超额等
            'exit_stats': dict,  # 止损/止盈/到期退出比例
        }
    """
    if benchmark_codes is None:
        benchmark_codes = ['000300.SH', '932000.CSI']

    if preloaded_data:
        reports = preloaded_data['reports']
        dates = preloaded_data['dates']
        all_quotes = preloaded_data['all_quotes']
        kline_quotes = preloaded_data['kline_quotes']
    else:
        # 导入现有基础设施
        from backtest.backtest_report_based import load_reports
        from backtest.backtest_stop_target_direct import preload_all_quotes

        # 加载报告
        reports = load_reports(report_dir, rank_field='composite')
        dates = sorted(reports.keys())
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]

        if not dates:
            print(f"  无可用报告: {report_dir}")
            return {}

        # 预加载日线数据 — 需要报告期+持仓期的数据
        all_quotes = preload_all_quotes(dates[0], '2026-12-31')

        # 预加载60日K线用于ATR/支撑/阻力计算
        conn = sqlite3.connect(DB_PATH, timeout=30)
        lookback_dates = [r[0] for r in conn.execute("""
            SELECT DISTINCT trade_date FROM daily_quotes
            WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT 80
        """, (dates[0],)).fetchall()]
        lookback_start = lookback_dates[-1] if lookback_dates else dates[0]
        conn.close()

        kline_quotes = preload_all_quotes(lookback_start, '2026-12-31')

    if not dates:
        return {}

    if not quiet:
        print(f"\n{'='*60}")
        print(f"  回测: {label}")
        print(f"  报告: {len(dates)}天, {dates[0]} -> {dates[-1]}")
        print(f"  模式: {'新Optimizer' if use_optimizer else '旧逻辑'}")

    # 导入PortfolioOptimizer
    if use_optimizer:
        from portfolio_optimizer import PortfolioOptimizer
        optimizer = PortfolioOptimizer(params=params)

    # 模拟交易
    all_trades = []
    trade_cost = 0.00302  # 双边交易成本0.302%
    max_hold = params['hold']['max_hold_days']

    for date in dates:
        stocks = reports[date]
        if not stocks:
            continue

        if use_optimizer:
            # 新逻辑: 动态价格+仓位
            env_score = 50.0  # 默认值, 下方从JSON覆盖

            # 读取env_score (如果JSON中有)
            json_path = Path(report_dir) / f"analysis_data_{date.replace('-','')}.json"
            if json_path.exists():
                try:
                    with open(json_path) as f:
                        data = json.load(f)
                    te = data.get('trading_environment', {})
                    env_score = te.get('total_score', 50.0)
                except Exception:
                    pass

            # 为每只股票计算自适应价格
            for s in stocks:
                code = s.get('code', '')
                try:
                    kline = kline_quotes.loc[code]
                    mask = kline.index <= date
                    kline_before = kline[mask].tail(80)
                    if len(kline_before) < 20:
                        continue
                    highs = kline_before['high'].values
                    lows = kline_before['low'].values
                    closes = kline_before['close'].values

                    stock_info = {
                        'stock_code': code,
                        'close_price': closes[-1],
                        'pred_10d': s.get('pred_10d', 0),
                    }
                    stock_info = optimizer.compute_prices(stock_info, highs, lows, closes, env_score)
                    s['buy_price'] = stock_info['suggested_buy_price']
                    s['stop_loss'] = stock_info['stop_loss_price']
                    s['target'] = stock_info['take_profit_price']
                    s['atr_pct'] = stock_info.get('atr_pct', 0.03)
                    s['composite'] = s.get('rank_score', s.get('score', 0))
                except (KeyError, IndexError):
                    # 无法计算价格的股票跳过
                    continue

            # 信号筛选+仓位分配
            valid = [s for s in stocks if s.get('buy_price', 0) > 0]
            selected = optimizer.filter_and_allocate(valid, env_score)

        else:
            # 旧逻辑: 固定百分比 (Top 10, 等权)
            sorted_stocks = sorted(stocks, key=lambda s: s.get('rank_score', 0), reverse=True)
            selected = sorted_stocks[:10]
            for s in selected:
                code = s.get('code', '')
                close = 0
                try:
                    kline = kline_quotes.loc[code]
                    mask = kline.index <= date
                    if len(kline[mask]) > 0:
                        close = kline[mask].iloc[-1]['close']
                except (KeyError, IndexError):
                    close = 0
                pred_10d = s.get('pred_10d', 0)
                is_wide = code.startswith('30') or code.startswith('688')
                s['buy_price'] = round(close * (1 - (0 if (pred_10d or 0) >= 0 else 0.025)), 2)
                base_stop = 0.15 if is_wide else 0.10
                s['stop_loss'] = round(close * (1 - base_stop), 2)
                base_target = 0.12 if is_wide else 0.08
                s['target'] = round(close * (1 + base_target), 2)
                s['position_pct'] = 10.0  # 等权10%

        # 执行交易模拟
        for s in selected:
            code = s.get('code', '')
            result = simulate_trade_with_trailing(
                code, date,
                s.get('buy_price', 0),
                s.get('stop_loss', 0),
                s.get('target', 0),
                all_quotes, params,
            )
            if result and result['outcome'] != 'no_fill':
                result['code'] = code
                result['date'] = date
                result['position_pct'] = s.get('position_pct', 10.0)
                result['trade_return_net'] = result['trade_return'] - trade_cost
                all_trades.append(result)

    # 聚合结果
    if not all_trades:
        print("  无有效交易")
        return {}

    trades_df = pd.DataFrame(all_trades)
    metrics = _compute_metrics(trades_df, max_hold, dates)

    # 基准收益
    for bench_code in benchmark_codes:
        bench_ret = _get_benchmark_return(bench_code, dates[0], dates[-1], max_hold)
        metrics[f'benchmark_{bench_code}_annual'] = bench_ret
        metrics[f'excess_{bench_code}'] = metrics.get('annual_return', 0) - bench_ret

    # 退出统计
    exit_stats = trades_df['outcome'].value_counts(normalize=True).to_dict()

    if not quiet:
        print(f"\n  交易笔数: {len(all_trades)}")
        print(f"  年化收益: {metrics.get('annual_return', 0):.1%}")
        print(f"  Sharpe: {metrics.get('sharpe', 0):.3f}")
        print(f"  MaxDD: {metrics.get('max_drawdown', 0):.1%}")
        for bc in benchmark_codes:
            print(f"  超额({bc}): {metrics.get(f'excess_{bc}', 0):.1%}")
        print(f"  止损率: {exit_stats.get('stop_loss', 0):.1%}")
        print(f"  止盈率: {exit_stats.get('take_profit', 0):.1%}")
        print(f"  到期率: {exit_stats.get('max_hold', 0):.1%}")

    return {
        'label': label,
        'trades': all_trades,
        'metrics': metrics,
        'exit_stats': exit_stats,
    }


def _compute_metrics(trades_df: pd.DataFrame, hold_days: int,
                     report_dates: list) -> dict:
    """从交易记录计算组合级别指标"""
    # 加权收益 (按仓位权重)
    trades_df['weighted_return'] = trades_df['trade_return_net'] * trades_df['position_pct'] / 100

    # 按日期聚合
    daily_returns = trades_df.groupby('date')['weighted_return'].sum()

    # 年化
    n_periods = len(report_dates)
    periods_per_year = 252 / hold_days
    total_return = (1 + daily_returns).prod() - 1
    n_holding_periods = n_periods  # 每个报告日一个持仓期
    annual_return = (1 + total_return) ** (periods_per_year / max(n_holding_periods, 1)) - 1

    # Sharpe
    if len(daily_returns) > 1:
        mean_ret = daily_returns.mean()
        std_ret = daily_returns.std()
        sharpe = (mean_ret * periods_per_year - 0.02) / (std_ret * np.sqrt(periods_per_year)) if std_ret > 0 else 0
    else:
        sharpe = 0

    # MaxDD
    cumulative = (1 + daily_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()

    # 月度胜率
    if len(daily_returns) >= 2:
        monthly_groups = trades_df.groupby(trades_df['date'].str[:7])['weighted_return'].sum()
        monthly_win_rate = (monthly_groups > 0).mean()
    else:
        monthly_win_rate = 0

    return {
        'annual_return': annual_return,
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'monthly_win_rate': monthly_win_rate,
        'n_trades': len(trades_df),
        'avg_return': trades_df['trade_return_net'].mean(),
        'win_rate': (trades_df['trade_return_net'] > 0).mean(),
        'avg_hold_days': trades_df['hold_days'].mean(),
    }


def _get_benchmark_return(index_code: str, start_date: str, end_date: str,
                          hold_days: int) -> float:
    """获取基准年化收益"""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    df = pd.read_sql_query("""
        SELECT dq.trade_date, dq.close
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.code = ? AND dq.trade_date >= ? AND dq.trade_date <= ?
        ORDER BY dq.trade_date
    """, conn, params=[index_code, start_date, end_date])
    conn.close()

    if len(df) < 2:
        return 0

    total_ret = df['close'].iloc[-1] / df['close'].iloc[0] - 1
    n_days = len(df)
    annual_ret = (1 + total_ret) ** (252 / max(n_days, 1)) - 1
    return annual_ret
