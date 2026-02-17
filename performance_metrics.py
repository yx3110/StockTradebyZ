from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    计算年化夏普比率 (Sharpe Ratio)。

    Parameters
    ----------
    returns : pd.Series
        收益率序列（日收益率）。
    risk_free_rate : float, default 0.0
        无风险利率（年化）。
    periods_per_year : int, default 252
        每年交易日数（默认 252 天）。

    Returns
    -------
    float
        年化夏普比率。若标准差为 0 或数据不足，返回 0.0。
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0

    mean_return = returns.mean()
    std_return = returns.std(ddof=1)

    if std_return == 0 or pd.isna(std_return):
        return 0.0

    # 计算夏普比率并年化
    daily_sharpe = (mean_return - risk_free_rate / periods_per_year) / std_return
    annualized_sharpe = daily_sharpe * np.sqrt(periods_per_year)

    return float(annualized_sharpe)


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    计算最大回撤 (Maximum Drawdown)。

    Parameters
    ----------
    equity_curve : pd.Series
        权益曲线（账户净值序列）。

    Returns
    -------
    float
        最大回撤百分比（负数，如 -0.15 表示 -15%）。
        若无回撤或数据不足，返回 0.0。
    """
    equity_curve = equity_curve.dropna()
    if len(equity_curve) < 2:
        return 0.0

    # 计算累计最大值
    running_max = equity_curve.expanding(min_periods=1).max()

    # 计算每个时刻的回撤
    drawdown = (equity_curve - running_max) / running_max

    # 返回最大回撤（最小值，因为是负数）
    max_dd = drawdown.min()

    return float(max_dd) if pd.notna(max_dd) else 0.0


def calculate_total_return(
    equity_curve: pd.Series,
) -> float:
    """
    计算总收益率。

    Parameters
    ----------
    equity_curve : pd.Series
        权益曲线（账户净值序列）。

    Returns
    -------
    float
        总收益率（如 0.25 表示 25%）。
    """
    equity_curve = equity_curve.dropna()
    if len(equity_curve) < 2:
        return 0.0

    initial_value = equity_curve.iloc[0]
    final_value = equity_curve.iloc[-1]

    if initial_value == 0:
        return 0.0

    total_return = (final_value - initial_value) / initial_value
    return float(total_return)


def calculate_win_rate(
    trade_returns: pd.Series,
) -> float:
    """
    计算胜率（盈利交易占比）。

    Parameters
    ----------
    trade_returns : pd.Series
        每笔交易的收益率序列。

    Returns
    -------
    float
        胜率（0-1 之间，如 0.6 表示 60%）。
    """
    trade_returns = trade_returns.dropna()
    if len(trade_returns) == 0:
        return 0.0

    winning_trades = (trade_returns > 0).sum()
    total_trades = len(trade_returns)

    return float(winning_trades / total_trades)


def calculate_average_gain(
    trade_returns: pd.Series,
) -> float:
    """
    计算平均盈利（仅计算盈利交易）。

    Parameters
    ----------
    trade_returns : pd.Series
        每笔交易的收益率序列。

    Returns
    -------
    float
        平均盈利（如 0.05 表示平均盈利 5%）。
        若无盈利交易，返回 0.0。
    """
    trade_returns = trade_returns.dropna()
    winning_trades = trade_returns[trade_returns > 0]

    if len(winning_trades) == 0:
        return 0.0

    return float(winning_trades.mean())


def calculate_average_loss(
    trade_returns: pd.Series,
) -> float:
    """
    计算平均亏损（仅计算亏损交易）。

    Parameters
    ----------
    trade_returns : pd.Series
        每笔交易的收益率序列。

    Returns
    -------
    float
        平均亏损（负数，如 -0.03 表示平均亏损 -3%）。
        若无亏损交易，返回 0.0。
    """
    trade_returns = trade_returns.dropna()
    losing_trades = trade_returns[trade_returns < 0]

    if len(losing_trades) == 0:
        return 0.0

    return float(losing_trades.mean())


def calculate_profit_factor(
    trade_returns: pd.Series,
) -> float:
    """
    计算盈亏比 (Profit Factor)。

    Parameters
    ----------
    trade_returns : pd.Series
        每笔交易的收益率序列。

    Returns
    -------
    float
        盈亏比（总盈利 / 总亏损的绝对值）。
        若无亏损交易，返回 inf；若无盈利交易，返回 0.0。
    """
    trade_returns = trade_returns.dropna()
    if len(trade_returns) == 0:
        return 0.0

    gross_profit = trade_returns[trade_returns > 0].sum()
    gross_loss = abs(trade_returns[trade_returns < 0].sum())

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0

    return float(gross_profit / gross_loss)


def calculate_all_metrics(
    equity_curve: pd.Series,
    trade_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """
    计算所有性能指标。

    Parameters
    ----------
    equity_curve : pd.Series
        权益曲线（账户净值序列）。
    trade_returns : pd.Series, optional
        每笔交易的收益率序列（用于计算胜率、平均盈亏等）。
    risk_free_rate : float, default 0.0
        无风险利率（年化）。

    Returns
    -------
    Dict[str, float]
        包含所有性能指标的字典：
        - total_return: 总收益率
        - sharpe_ratio: 年化夏普比率
        - max_drawdown: 最大回撤
        - win_rate: 胜率（需要 trade_returns）
        - avg_gain: 平均盈利（需要 trade_returns）
        - avg_loss: 平均亏损（需要 trade_returns）
        - profit_factor: 盈亏比（需要 trade_returns）
    """
    metrics = {}

    # 基于权益曲线的指标
    metrics['total_return'] = calculate_total_return(equity_curve)
    metrics['max_drawdown'] = calculate_max_drawdown(equity_curve)

    # 计算日收益率用于夏普比率
    equity_curve_clean = equity_curve.dropna()
    if len(equity_curve_clean) >= 2:
        daily_returns = equity_curve_clean.pct_change().dropna()
        metrics['sharpe_ratio'] = calculate_sharpe_ratio(daily_returns, risk_free_rate)
    else:
        metrics['sharpe_ratio'] = 0.0

    # 基于交易收益的指标
    if trade_returns is not None:
        metrics['win_rate'] = calculate_win_rate(trade_returns)
        metrics['avg_gain'] = calculate_average_gain(trade_returns)
        metrics['avg_loss'] = calculate_average_loss(trade_returns)
        metrics['profit_factor'] = calculate_profit_factor(trade_returns)
    else:
        metrics['win_rate'] = 0.0
        metrics['avg_gain'] = 0.0
        metrics['avg_loss'] = 0.0
        metrics['profit_factor'] = 0.0

    return metrics
