#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
北极星指标计算模块 (North Star Metrics)

统一的量化交易系统评估指标计算，覆盖4个层级：
- Layer 1: 信号质量 (IC, ICIR, Rank IC)
- Layer 2: 组合构建 (换手率, 交易成本, 选股精准度)
- Layer 3: 风险控制 (MaxDD, Sharpe, Sortino, Calmar, VaR, CVaR)
- Layer 4: 最终盈利 (年化收益, 月度收益, 超额收益)

用法:
    from backtest.north_star_metrics import NorthStarEvaluator
    evaluator = NorthStarEvaluator()
    report = evaluator.full_evaluation(daily_returns, daily_ic_series, benchmark_returns)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Dict, List, Optional, Tuple
import sqlite3
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')

# ═══════════════════════════════════════════════════════
# 北极星目标定义
# ═══════════════════════════════════════════════════════

NORTH_STAR_TARGETS = {
    # Layer 1: 信号质量
    'daily_ic':         {'target': 0.06, 'pass': 0.03, 'good': 0.05, 'direction': 'higher'},
    'icir':             {'target': 0.50, 'pass': 0.30, 'good': 0.40, 'direction': 'higher'},
    'ic_positive_pct':  {'target': 60.0, 'pass': 55.0, 'good': 58.0, 'direction': 'higher'},
    'rank_ic':          {'target': 0.07, 'pass': 0.04, 'good': 0.06, 'direction': 'higher'},

    # Layer 3: 风险控制
    'max_drawdown':     {'target': -0.12, 'pass': -0.25, 'good': -0.15, 'direction': 'higher'},  # less negative = better
    'sharpe_ratio':     {'target': 2.0,  'pass': 1.0,  'good': 1.5,  'direction': 'higher'},
    'sortino_ratio':    {'target': 2.5,  'pass': 1.5,  'good': 2.0,  'direction': 'higher'},
    'calmar_ratio':     {'target': 2.5,  'pass': 1.0,  'good': 2.0,  'direction': 'higher'},

    # Layer 4: 最终盈利
    'annual_return':    {'target': 0.30, 'pass': 0.15, 'good': 0.20, 'direction': 'higher'},
    'monthly_win_rate': {'target': 75.0, 'pass': 58.0, 'good': 67.0, 'direction': 'higher'},
    'annual_turnover':  {'target': 30.0, 'pass': 50.0, 'good': 35.0, 'direction': 'lower'},
    'annual_cost_drag': {'target': 0.08, 'pass': 0.15, 'good': 0.10, 'direction': 'lower'},
}

# A股交易成本参数（散户级别）
TRANSACTION_COST = {
    'commission_rate': 0.00025,    # 佣金 0.025% 单边
    'stamp_tax_rate': 0.0005,      # 印花税 0.05% (仅卖出)
    'transfer_fee_rate': 0.00001,  # 过户费 0.001%
    'slippage_rate': 0.001,        # 滑点 0.1% 单边
}


# ═══════════════════════════════════════════════════════
# Layer 1: 信号质量指标
# ═══════════════════════════════════════════════════════

def compute_daily_ic(scores: pd.Series, returns: pd.Series, dates: pd.Series,
                     min_stocks: int = 20) -> pd.DataFrame:
    """
    计算每日截面IC（Spearman Rank Correlation）

    Args:
        scores: 预测分数
        returns: 实际收益
        dates: 交易日期
        min_stocks: 每天最少股票数

    Returns:
        DataFrame with columns: [date, ic, n_stocks]
    """
    records = []
    for date in sorted(dates.unique()):
        mask = (dates == date) & scores.notna() & returns.notna()
        day_scores = scores[mask]
        day_returns = returns[mask]

        if len(day_scores) < min_stocks:
            continue

        ic, p_val = spearmanr(day_scores, day_returns)
        if not np.isnan(ic):
            records.append({
                'date': date,
                'ic': ic,
                'p_val': p_val,
                'n_stocks': len(day_scores),
            })

    return pd.DataFrame(records)


def compute_ic_summary(ic_df: pd.DataFrame) -> Dict:
    """
    从daily IC序列计算ICIR等汇总指标

    Args:
        ic_df: compute_daily_ic的输出

    Returns:
        dict with ic_mean, ic_std, icir, ic_positive_pct, etc.
    """
    if ic_df.empty or len(ic_df) < 3:
        return {
            'ic_mean': 0, 'ic_std': 0, 'icir': 0,
            'ic_positive_pct': 0, 'ic_median': 0, 'n_days': 0,
        }

    ic_vals = ic_df['ic']
    ic_mean = ic_vals.mean()
    ic_std = ic_vals.std()
    icir = ic_mean / ic_std if ic_std > 1e-8 else 0
    ic_positive_pct = (ic_vals > 0).mean() * 100
    ic_median = ic_vals.median()

    return {
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        'ic_positive_pct': ic_positive_pct,
        'ic_median': ic_median,
        'n_days': len(ic_df),
    }


def compute_monthly_ic(ic_df: pd.DataFrame) -> pd.DataFrame:
    """按月分解IC"""
    if ic_df.empty:
        return pd.DataFrame()

    ic_copy = ic_df.copy()
    ic_copy['month'] = pd.to_datetime(ic_copy['date']).dt.to_period('M')

    monthly = []
    for month, group in ic_copy.groupby('month'):
        m_ic = group['ic'].mean()
        m_std = group['ic'].std()
        m_icir = m_ic / m_std if m_std > 1e-8 else 0
        m_pos = (group['ic'] > 0).mean() * 100
        monthly.append({
            'month': str(month),
            'ic_mean': m_ic,
            'ic_std': m_std,
            'icir': m_icir,
            'ic_positive_pct': m_pos,
            'n_days': len(group),
        })

    return pd.DataFrame(monthly)


def compute_top_bottom_spread(scores: pd.Series, returns: pd.Series,
                              dates: pd.Series, quantile: float = 0.1,
                              min_stocks: int = 20) -> Dict:
    """
    计算Top/Bottom分位组合超额收益

    Args:
        quantile: Top/Bottom分位数 (0.1 = 10%)
    """
    top_returns_list = []
    bottom_returns_list = []

    for date in sorted(dates.unique()):
        mask = (dates == date) & scores.notna() & returns.notna()
        day_scores = scores[mask]
        day_returns = returns[mask]

        if len(day_scores) < min_stocks:
            continue

        n = max(1, int(len(day_scores) * quantile))
        sorted_idx = day_scores.argsort()
        top_idx = sorted_idx.iloc[-n:]
        bottom_idx = sorted_idx.iloc[:n]

        top_ret = day_returns.iloc[top_idx.values].mean()
        bottom_ret = day_returns.iloc[bottom_idx.values].mean()

        top_returns_list.append(top_ret)
        bottom_returns_list.append(bottom_ret)

    if not top_returns_list:
        return {'top_return': 0, 'bottom_return': 0, 'spread': 0}

    return {
        'top_return': np.mean(top_returns_list),
        'bottom_return': np.mean(bottom_returns_list),
        'spread': np.mean(top_returns_list) - np.mean(bottom_returns_list),
        'n_days': len(top_returns_list),
    }


# ═══════════════════════════════════════════════════════
# Layer 2: 组合构建指标
# ═══════════════════════════════════════════════════════

def compute_turnover(holdings_by_date: Dict[str, List[str]]) -> Dict:
    """
    计算换手率

    Args:
        holdings_by_date: {date: [stock_codes]} 每天持仓列表

    Returns:
        dict with avg_turnover, total_rebalances, annual_turnover_estimate
    """
    dates = sorted(holdings_by_date.keys())
    if len(dates) < 2:
        return {'avg_turnover': 0, 'total_rebalances': 0, 'annual_turnover_estimate': 0}

    turnovers = []
    for i in range(1, len(dates)):
        prev = set(holdings_by_date[dates[i - 1]])
        curr = set(holdings_by_date[dates[i]])

        if not prev and not curr:
            continue

        union = prev | curr
        changed = len(prev.symmetric_difference(curr))
        turnover = changed / (2 * max(len(union), 1))  # 单边换手
        turnovers.append(turnover)

    avg_turnover = np.mean(turnovers) if turnovers else 0
    n_rebalances = len(turnovers)

    # 估算年化换手率：每次换手率 × 年调仓次数
    # 假设持仓N天，则年调仓 252/N 次
    trading_days_span = len(dates)
    if trading_days_span > 1:
        rebalance_freq = n_rebalances / trading_days_span * 252
        annual_turnover = avg_turnover * 2 * rebalance_freq  # 双边
    else:
        annual_turnover = 0

    return {
        'avg_turnover': avg_turnover,
        'total_rebalances': n_rebalances,
        'rebalance_freq_annual': rebalance_freq if trading_days_span > 1 else 0,
        'annual_turnover_estimate': annual_turnover,
    }


def compute_transaction_costs(turnover_per_rebal: float,
                              holding_days: int,
                              gross_annual_return: float = 0,
                              cost_params: Dict = None) -> Dict:
    """
    计算交易成本拖累

    Args:
        turnover_per_rebal: 每次调仓单边换手率
        holding_days: 持仓天数
        gross_annual_return: 已计算的毛年化收益率（由调用方传入）
        cost_params: 交易成本参数

    Returns:
        dict with annual_cost_drag, net_annual_return, etc.
    """
    if cost_params is None:
        cost_params = TRANSACTION_COST

    # 每次调仓的总交易成本（买入+卖出）
    buy_cost = cost_params['commission_rate'] + cost_params['transfer_fee_rate'] + cost_params['slippage_rate']
    sell_cost = buy_cost + cost_params['stamp_tax_rate']
    round_trip_cost = buy_cost + sell_cost

    # 每次调仓的成本 = 换手率 × 双边成本
    cost_per_rebalance = turnover_per_rebal * round_trip_cost

    # 年化调仓次数
    annual_rebalances = 252 / holding_days

    # 年化成本拖累
    annual_cost_drag = cost_per_rebalance * annual_rebalances

    net_annual = gross_annual_return - annual_cost_drag

    return {
        'round_trip_cost': round_trip_cost,
        'cost_per_rebalance': cost_per_rebalance,
        'annual_rebalances': annual_rebalances,
        'annual_cost_drag': annual_cost_drag,
        'gross_annual_return': gross_annual_return,
        'net_annual_return': net_annual,
    }


# ═══════════════════════════════════════════════════════
# Layer 3: 风险控制指标
# ═══════════════════════════════════════════════════════

def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    """计算回撤序列"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown


def compute_risk_metrics(daily_returns: pd.Series, risk_free_rate: float = 0.02) -> Dict:
    """
    计算全套风险指标

    Args:
        daily_returns: 日收益率序列
        risk_free_rate: 年化无风险利率 (默认2%, 中国国债)

    Returns:
        dict with sharpe, sortino, calmar, max_drawdown, etc.
    """
    if len(daily_returns) < 10:
        return {k: 0 for k in [
            'annual_return', 'annual_volatility', 'downside_volatility',
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'max_drawdown', 'max_dd_duration_days', 'max_dd_recovery_days',
            'var_95', 'cvar_95', 'omega_ratio',
            'monthly_win_rate', 'worst_month', 'best_month',
            'n_trading_days', 'positive_day_pct',
        ]}

    returns = daily_returns.dropna()
    n_days = len(returns)

    # 基础收益
    cumulative_return = (1 + returns).prod() - 1
    annual_return = (1 + cumulative_return) ** (252 / n_days) - 1

    # 波动率
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess_returns = returns - daily_rf
    annual_volatility = returns.std() * np.sqrt(252)

    # 下行波动率 (仅负收益)
    negative_returns = returns[returns < 0]
    downside_volatility = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 1e-8

    # Sharpe Ratio
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 1e-8 else 0

    # Sortino Ratio
    sortino_ratio = (annual_return - risk_free_rate) / downside_volatility if downside_volatility > 1e-8 else 0

    # 回撤分析
    drawdown = compute_drawdown_series(returns)
    max_drawdown = drawdown.min()  # 最大回撤（负值）

    # 回撤持续期和恢复时间
    dd_duration, dd_recovery = _compute_dd_duration(returns)

    # Calmar Ratio
    calmar_ratio = annual_return / abs(max_drawdown) if abs(max_drawdown) > 1e-8 else 0

    # VaR and CVaR (95%)
    var_95 = np.percentile(returns, 5)  # 5th percentile = 95% VaR
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

    # Omega Ratio
    threshold = daily_rf
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    omega_ratio = gains.sum() / losses.sum() if losses.sum() > 1e-8 else float('inf')

    # 月度统计
    returns_with_index = returns.copy()
    if not isinstance(returns_with_index.index, pd.DatetimeIndex):
        try:
            returns_with_index.index = pd.to_datetime(returns_with_index.index)
        except Exception:
            # 用简单序列索引
            pass

    monthly_returns = _compute_monthly_returns(returns_with_index)
    if len(monthly_returns) > 0:
        monthly_win_rate = (monthly_returns > 0).mean() * 100
        worst_month = monthly_returns.min()
        best_month = monthly_returns.max()
        max_consecutive_loss_months = _max_consecutive(monthly_returns < 0)
    else:
        monthly_win_rate = 0
        worst_month = 0
        best_month = 0
        max_consecutive_loss_months = 0

    # 日度胜率
    positive_day_pct = (returns > 0).mean() * 100

    return {
        'annual_return': annual_return,
        'cumulative_return': cumulative_return,
        'annual_volatility': annual_volatility,
        'downside_volatility': downside_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'max_dd_duration_days': dd_duration,
        'max_dd_recovery_days': dd_recovery,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'omega_ratio': omega_ratio,
        'monthly_win_rate': monthly_win_rate,
        'worst_month': worst_month,
        'best_month': best_month,
        'max_consecutive_loss_months': max_consecutive_loss_months,
        'n_trading_days': n_days,
        'positive_day_pct': positive_day_pct,
    }


def _compute_dd_duration(returns: pd.Series) -> Tuple[int, int]:
    """计算最大回撤持续天数和恢复天数"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    is_underwater = cumulative < running_max

    max_duration = 0
    current_duration = 0

    for underwater in is_underwater:
        if underwater:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    # 恢复时间：从最大回撤点到回到前高的天数
    drawdown = (cumulative - running_max) / running_max
    if drawdown.empty:
        return max_duration, 0

    # 使用位置索引避免非唯一DatetimeIndex的KeyError
    min_pos = drawdown.values.argmin()

    after_dd = cumulative.iloc[min_pos:]
    pre_dd_max = running_max.iloc[min_pos]
    recovered = after_dd >= pre_dd_max
    if recovered.any():
        # 找到恢复点的位置索引
        recovery_pos = recovered.values.argmax()
        recovery_days = recovery_pos + 1  # 包含起始点
    else:
        recovery_days = len(after_dd)  # 还未恢复

    return max_duration, recovery_days


def _compute_monthly_returns(returns: pd.Series) -> pd.Series:
    """计算月度收益率"""
    try:
        if isinstance(returns.index, pd.DatetimeIndex):
            monthly = (1 + returns).resample('ME').prod() - 1
            return monthly.dropna()
    except Exception:
        pass

    # fallback: 按20天分组
    n = len(returns)
    if n < 20:
        return pd.Series(dtype=float)

    monthly = []
    for i in range(0, n, 20):
        chunk = returns.iloc[i:i+20]
        monthly.append((1 + chunk).prod() - 1)

    return pd.Series(monthly)


def _max_consecutive(bool_series: pd.Series) -> int:
    """计算最长连续True的长度"""
    max_count = 0
    current = 0
    for val in bool_series:
        if val:
            current += 1
            max_count = max(max_count, current)
        else:
            current = 0
    return max_count


# ═══════════════════════════════════════════════════════
# Layer 4: 基准对比
# ═══════════════════════════════════════════════════════

def load_benchmark_returns(index_code: str = '000905.SH',
                           start_date: str = None, end_date: str = None,
                           db_path: str = None) -> pd.Series:
    """
    从数据库加载基准指数收益率

    Args:
        index_code: 指数代码 (默认中证500: 000905.SH)
            常用: 000300.SH(沪深300), 000905.SH(中证500),
                  000852.SH(中证1000), 932000.CSI(中证2000)
        start_date: 开始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD
    """
    if db_path is None:
        db_path = DB_PATH

    conn = sqlite3.connect(db_path)

    query = """
        SELECT dq.trade_date, dq.close, dq.price_change_pct
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.code = ?
    """
    params = [index_code]

    if start_date:
        query += " AND dq.trade_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND dq.trade_date <= ?"
        params.append(end_date)

    query += " ORDER BY dq.trade_date"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if df.empty:
        return pd.Series(dtype=float)

    # price_change_pct 已经是小数形式的日涨跌幅 (e.g., 0.0117 = 1.17%)
    returns = df.set_index('trade_date')['price_change_pct']
    returns.index = pd.to_datetime(returns.index)
    returns.name = index_code

    return returns


def compute_benchmark_comparison(portfolio_returns: pd.Series,
                                 benchmark_returns: pd.Series,
                                 periods_per_year: float = 252) -> Dict:
    """
    计算相对基准的表现

    Args:
        portfolio_returns: 策略收益率 (DatetimeIndex)
        benchmark_returns: 基准收益率 (DatetimeIndex), 频率需与portfolio一致
        periods_per_year: 年化因子 (日频=252, 10日频=25.2)
    """
    # 对齐日期
    common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
    min_required = min(10, max(3, int(periods_per_year * 0.1)))
    if len(common_dates) < min_required:
        return {
            'alpha': 0, 'beta': 0, 'tracking_error': 0,
            'information_ratio': 0, 'excess_annual_return': 0,
        }

    p_ret = portfolio_returns.loc[common_dates]
    b_ret = benchmark_returns.loc[common_dates]

    # Alpha / Beta (CAPM回归)
    cov = np.cov(p_ret, b_ret)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 1e-10 else 0

    excess = p_ret - b_ret
    alpha_per_period = excess.mean()
    alpha_annual = alpha_per_period * periods_per_year

    # Tracking Error
    tracking_error = excess.std() * np.sqrt(periods_per_year)

    # Information Ratio
    information_ratio = alpha_annual / tracking_error if tracking_error > 1e-8 else 0

    # 超额年化收益
    n_periods = len(common_dates)
    total_equivalent_days = n_periods * (252 / periods_per_year)
    p_annual = (1 + p_ret).prod() ** (252 / total_equivalent_days) - 1
    b_annual = (1 + b_ret).prod() ** (252 / total_equivalent_days) - 1
    excess_annual = p_annual - b_annual

    # 超额收益胜率
    excess_win_rate = (excess > 0).mean() * 100

    return {
        'alpha': alpha_annual,
        'beta': beta,
        'tracking_error': tracking_error,
        'information_ratio': information_ratio,
        'excess_annual_return': excess_annual,
        'portfolio_annual': p_annual,
        'benchmark_annual': b_annual,
        'excess_win_rate': excess_win_rate,
        'n_common_periods': n_periods,
    }


# ═══════════════════════════════════════════════════════
# 综合评估器
# ═══════════════════════════════════════════════════════

class NorthStarEvaluator:
    """
    北极星指标综合评估器

    用法:
        evaluator = NorthStarEvaluator()

        # 信号质量评估
        ic_df = evaluator.evaluate_signal_quality(scores, returns, dates)

        # 组合风险收益评估
        risk_report = evaluator.evaluate_portfolio(daily_returns)

        # 完整评估（含基准对比）
        full_report = evaluator.full_evaluation(
            daily_returns, scores, actual_returns, dates,
            benchmark_code='000905.SH'
        )

        # 打印评分卡
        evaluator.print_scorecard(full_report)
    """

    def __init__(self, db_path: str = None, risk_free_rate: float = 0.02):
        self.db_path = db_path or DB_PATH
        self.risk_free_rate = risk_free_rate

    def evaluate_signal_quality(self, scores: pd.Series, returns: pd.Series,
                                dates: pd.Series, holding_days: int = 5) -> Dict:
        """Layer 1: 评估信号质量"""
        ic_df = compute_daily_ic(scores, returns, dates)
        ic_summary = compute_ic_summary(ic_df)
        monthly_ic = compute_monthly_ic(ic_df)
        spread = compute_top_bottom_spread(scores, returns, dates, quantile=0.1)

        return {
            'holding_days': holding_days,
            'ic_df': ic_df,
            'ic_summary': ic_summary,
            'monthly_ic': monthly_ic,
            'top_bottom_spread': spread,
        }

    def evaluate_portfolio(self, daily_returns: pd.Series,
                           dates: pd.Series = None) -> Dict:
        """Layer 3+4: 评估组合风险收益"""
        risk = compute_risk_metrics(daily_returns, self.risk_free_rate)
        return risk

    def evaluate_with_benchmark(self, daily_returns: pd.Series,
                                benchmark_code: str = '000905.SH',
                                start_date: str = None,
                                end_date: str = None) -> Dict:
        """Layer 4: 基准对比"""
        benchmark = load_benchmark_returns(
            benchmark_code, start_date, end_date, self.db_path
        )
        if benchmark.empty:
            return {}

        return compute_benchmark_comparison(daily_returns, benchmark)

    def full_evaluation(self, daily_returns: pd.Series,
                        scores: pd.Series = None,
                        actual_returns: pd.Series = None,
                        dates: pd.Series = None,
                        holdings_by_date: Dict = None,
                        holding_days: int = 10,
                        benchmark_code: str = '000905.SH') -> Dict:
        """
        完整的北极星评估

        Returns:
            dict with all metrics organized by layer
        """
        report = {
            'layer1_signal': {},
            'layer2_portfolio': {},
            'layer3_risk': {},
            'layer4_pnl': {},
            'benchmark': {},
        }

        # Layer 1: 信号质量
        if scores is not None and actual_returns is not None and dates is not None:
            report['layer1_signal'] = self.evaluate_signal_quality(
                scores, actual_returns, dates, holding_days
            )

        # Layer 2: 换手率
        if holdings_by_date:
            report['layer2_portfolio'] = compute_turnover(holdings_by_date)

        # Layer 3: 风险
        if daily_returns is not None and len(daily_returns) > 0:
            report['layer3_risk'] = compute_risk_metrics(daily_returns, self.risk_free_rate)

        # Layer 4: 基准对比
        if daily_returns is not None and len(daily_returns) > 0:
            # 确保有DatetimeIndex
            if not isinstance(daily_returns.index, pd.DatetimeIndex):
                try:
                    daily_returns_dt = daily_returns.copy()
                    daily_returns_dt.index = pd.to_datetime(daily_returns_dt.index)
                except Exception:
                    daily_returns_dt = daily_returns
            else:
                daily_returns_dt = daily_returns

            benchmark = load_benchmark_returns(
                benchmark_code, db_path=self.db_path
            )
            if not benchmark.empty:
                report['benchmark'] = compute_benchmark_comparison(
                    daily_returns_dt, benchmark
                )

        # 交易成本
        if daily_returns is not None and len(daily_returns) > 0:
            turnover = report['layer2_portfolio'].get('avg_turnover', 0.5)
            report['layer4_pnl'] = compute_transaction_costs(
                daily_returns, turnover, holding_days
            )

        return report

    def print_scorecard(self, report: Dict, label: str = "模型评估"):
        """
        打印北极星评分卡

        对比当前指标与北极星目标，标记达标/未达标
        """
        print(f"\n{'═'*70}")
        print(f"  北极星评分卡: {label}")
        print(f"{'═'*70}")

        # 收集所有可用指标
        metrics_map = {}

        # Layer 1
        signal = report.get('layer1_signal', {})
        ic_summary = signal.get('ic_summary', {})
        if ic_summary:
            metrics_map['daily_ic'] = ic_summary.get('ic_mean', 0)
            metrics_map['icir'] = ic_summary.get('icir', 0)
            metrics_map['ic_positive_pct'] = ic_summary.get('ic_positive_pct', 0)

        # Layer 3
        risk = report.get('layer3_risk', {})
        if risk:
            metrics_map['max_drawdown'] = risk.get('max_drawdown', 0)
            metrics_map['sharpe_ratio'] = risk.get('sharpe_ratio', 0)
            metrics_map['sortino_ratio'] = risk.get('sortino_ratio', 0)
            metrics_map['calmar_ratio'] = risk.get('calmar_ratio', 0)
            metrics_map['annual_return'] = risk.get('annual_return', 0)
            metrics_map['monthly_win_rate'] = risk.get('monthly_win_rate', 0)

        # Layer 4
        pnl = report.get('layer4_pnl', {})
        if pnl:
            metrics_map['annual_cost_drag'] = pnl.get('annual_cost_drag', 0)

        portfolio = report.get('layer2_portfolio', {})
        if portfolio:
            metrics_map['annual_turnover'] = portfolio.get('annual_turnover_estimate', 0)

        # 打印评分表
        print(f"\n  {'指标':<20s} {'当前值':>10s} {'及格线':>10s} {'目标':>10s} {'评级':>8s}")
        print(f"  {'─'*62}")

        total_score = 0
        max_score = 0

        for metric_name, target_info in NORTH_STAR_TARGETS.items():
            if metric_name not in metrics_map:
                continue

            current = metrics_map[metric_name]
            target = target_info['target']
            pass_val = target_info['pass']
            higher = target_info['direction'] == 'higher'

            # 评级
            if higher:
                if current >= target:
                    grade = "★★★"
                    score = 3
                elif current >= target_info['good']:
                    grade = "★★☆"
                    score = 2
                elif current >= pass_val:
                    grade = "★☆☆"
                    score = 1
                else:
                    grade = "☆☆☆"
                    score = 0
            else:
                if current <= target:
                    grade = "★★★"
                    score = 3
                elif current <= target_info['good']:
                    grade = "★★☆"
                    score = 2
                elif current <= pass_val:
                    grade = "★☆☆"
                    score = 1
                else:
                    grade = "☆☆☆"
                    score = 0

            total_score += score
            max_score += 3

            # 格式化
            if metric_name in ('max_drawdown', 'annual_return', 'annual_cost_drag',
                               'gross_annual_return', 'net_annual_return'):
                current_str = f"{current:.1%}"
                target_str = f"{target:.1%}"
                pass_str = f"{pass_val:.1%}"
            elif metric_name in ('ic_positive_pct', 'monthly_win_rate', 'annual_turnover'):
                current_str = f"{current:.1f}"
                target_str = f"{target:.1f}"
                pass_str = f"{pass_val:.1f}"
            else:
                current_str = f"{current:.3f}"
                target_str = f"{target:.3f}"
                pass_str = f"{pass_val:.3f}"

            display_name = {
                'daily_ic': 'Daily IC',
                'icir': 'ICIR',
                'ic_positive_pct': 'IC>0%',
                'rank_ic': 'Rank IC',
                'max_drawdown': '最大回撤',
                'sharpe_ratio': 'Sharpe',
                'sortino_ratio': 'Sortino',
                'calmar_ratio': 'Calmar',
                'annual_return': '年化收益',
                'monthly_win_rate': '月度胜率%',
                'annual_turnover': '年化换手',
                'annual_cost_drag': '年化成本',
            }.get(metric_name, metric_name)

            print(f"  {display_name:<20s} {current_str:>10s} {pass_str:>10s} {target_str:>10s} {grade:>8s}")

        # 总分
        if max_score > 0:
            pct = total_score / max_score * 100
            print(f"\n  综合评分: {total_score}/{max_score} ({pct:.0f}%)")

            if pct >= 80:
                print(f"  等级: A+ (优秀，接近北极星目标)")
            elif pct >= 60:
                print(f"  等级: A  (良好，大部分指标达标)")
            elif pct >= 40:
                print(f"  等级: B  (及格，仍有较大提升空间)")
            else:
                print(f"  等级: C  (不及格，需重点改进)")

        # 基准对比
        bm = report.get('benchmark', {})
        if bm:
            print(f"\n  基准对比:")
            print(f"    策略年化:     {bm.get('portfolio_annual', 0):.1%}")
            print(f"    基准年化:     {bm.get('benchmark_annual', 0):.1%}")
            print(f"    超额收益:     {bm.get('excess_annual_return', 0):.1%}")
            print(f"    Alpha:        {bm.get('alpha', 0):.1%}")
            print(f"    Beta:         {bm.get('beta', 0):.3f}")
            print(f"    信息比率(IR):  {bm.get('information_ratio', 0):.3f}")
            print(f"    跟踪误差:     {bm.get('tracking_error', 0):.1%}")

        print(f"{'═'*70}\n")


def format_signal_report(signal_results: Dict, holding_days: int = None) -> str:
    """格式化信号质量报告（训练脚本可调用）"""
    lines = []
    ic_s = signal_results.get('ic_summary', {})
    spread = signal_results.get('top_bottom_spread', {})
    hd = holding_days or signal_results.get('holding_days', '?')

    lines.append(f"\n  信号质量评估 ({hd}d持仓):")
    lines.append(f"    Daily IC均值:   {ic_s.get('ic_mean', 0):.4f} ± {ic_s.get('ic_std', 0):.4f}")
    lines.append(f"    ICIR:           {ic_s.get('icir', 0):.4f}")
    lines.append(f"    IC>0占比:       {ic_s.get('ic_positive_pct', 0):.1f}% ({ic_s.get('n_days', 0)}天)")
    lines.append(f"    Top10%-Bottom10% Spread: {spread.get('spread', 0):.4f}")

    # 月度IC
    monthly = signal_results.get('monthly_ic')
    if monthly is not None and len(monthly) > 0:
        lines.append(f"    月度IC分解:")
        for _, row in monthly.iterrows():
            lines.append(
                f"      {row['month']}: IC={row['ic_mean']:+.4f}, "
                f"ICIR={row['icir']:+.3f}, IC>0={row['ic_positive_pct']:.0f}%"
            )

    return '\n'.join(lines)
