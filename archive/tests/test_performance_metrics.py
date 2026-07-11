"""
Unit tests for performance_metrics module.

Tests all performance calculation functions including Sharpe ratio,
max drawdown, win rate, profit factor, and edge cases.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from performance_metrics import (
    calculate_all_metrics,
    calculate_average_gain,
    calculate_average_loss,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_total_return,
    calculate_win_rate,
)


def test_sharpe_ratio_positive_returns():
    """Test Sharpe ratio with positive returns."""
    returns = pd.Series([0.01, 0.02, 0.015, 0.01, 0.02])

    sharpe = calculate_sharpe_ratio(returns)

    assert sharpe > 0
    # With positive mean and low volatility, Sharpe should be high
    assert sharpe > 10  # Annualized


def test_sharpe_ratio_negative_returns():
    """Test Sharpe ratio with negative returns."""
    returns = pd.Series([-0.01, -0.02, -0.015, -0.01, -0.02])

    sharpe = calculate_sharpe_ratio(returns)

    assert sharpe < 0


def test_sharpe_ratio_zero_returns():
    """Test Sharpe ratio with all zero returns."""
    returns = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])

    sharpe = calculate_sharpe_ratio(returns)

    assert sharpe == 0.0  # Zero std deviation


def test_sharpe_ratio_single_data_point():
    """Test Sharpe ratio with insufficient data."""
    returns = pd.Series([0.01])

    sharpe = calculate_sharpe_ratio(returns)

    assert sharpe == 0.0  # Not enough data


def test_sharpe_ratio_with_risk_free_rate():
    """Test Sharpe ratio with non-zero risk-free rate."""
    returns = pd.Series([0.01, 0.02, 0.015, 0.01, 0.02])
    risk_free_rate = 0.03  # 3% annual

    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=risk_free_rate)

    assert isinstance(sharpe, float)


def test_sharpe_ratio_mixed_returns():
    """Test Sharpe ratio with mixed positive and negative returns."""
    returns = pd.Series([0.02, -0.01, 0.015, -0.005, 0.01, -0.002])

    sharpe = calculate_sharpe_ratio(returns)

    # Should have positive Sharpe if mean > 0
    assert isinstance(sharpe, float)


def test_max_drawdown_declining_equity():
    """Test max drawdown with declining equity."""
    equity = pd.Series([100, 110, 105, 95, 90, 85])

    dd = calculate_max_drawdown(equity)

    # Max drawdown from peak 110 to trough 85 = (85-110)/110 ≈ -0.227
    assert dd < 0
    assert dd < -0.20
    assert dd > -0.25


def test_max_drawdown_rising_equity():
    """Test max drawdown with only rising equity."""
    equity = pd.Series([100, 105, 110, 115, 120])

    dd = calculate_max_drawdown(equity)

    assert dd == 0.0  # No drawdown


def test_max_drawdown_flat_equity():
    """Test max drawdown with flat equity."""
    equity = pd.Series([100, 100, 100, 100, 100])

    dd = calculate_max_drawdown(equity)

    assert dd == 0.0


def test_max_drawdown_v_shaped_recovery():
    """Test max drawdown with recovery."""
    equity = pd.Series([100, 90, 80, 90, 100, 110])

    dd = calculate_max_drawdown(equity)

    # Max drawdown from 100 to 80 = -0.20
    assert abs(dd - (-0.20)) < 0.01


def test_max_drawdown_single_data_point():
    """Test max drawdown with insufficient data."""
    equity = pd.Series([100])

    dd = calculate_max_drawdown(equity)

    assert dd == 0.0


def test_total_return_profit():
    """Test total return with profit."""
    equity = pd.Series([100, 110, 120, 125, 130])

    total_return = calculate_total_return(equity)

    # Return = (130 - 100) / 100 = 0.30
    assert abs(total_return - 0.30) < 0.01


def test_total_return_loss():
    """Test total return with loss."""
    equity = pd.Series([100, 95, 90, 85, 80])

    total_return = calculate_total_return(equity)

    # Return = (80 - 100) / 100 = -0.20
    assert abs(total_return - (-0.20)) < 0.01


def test_total_return_flat():
    """Test total return with no change."""
    equity = pd.Series([100, 102, 98, 101, 100])

    total_return = calculate_total_return(equity)

    # Return = (100 - 100) / 100 = 0.0
    assert abs(total_return - 0.0) < 0.01


def test_total_return_insufficient_data():
    """Test total return with insufficient data."""
    equity = pd.Series([100])

    total_return = calculate_total_return(equity)

    assert total_return == 0.0


def test_win_rate_all_wins():
    """Test win rate with all winning trades."""
    trade_returns = pd.Series([0.05, 0.10, 0.03, 0.07, 0.02])

    win_rate = calculate_win_rate(trade_returns)

    assert win_rate == 1.0  # 100%


def test_win_rate_all_losses():
    """Test win rate with all losing trades."""
    trade_returns = pd.Series([-0.05, -0.10, -0.03, -0.07, -0.02])

    win_rate = calculate_win_rate(trade_returns)

    assert win_rate == 0.0  # 0%


def test_win_rate_mixed_results():
    """Test win rate with mixed wins and losses."""
    trade_returns = pd.Series([0.05, -0.03, 0.07, -0.02, 0.04, -0.01])

    win_rate = calculate_win_rate(trade_returns)

    # 3 wins out of 6 = 0.5
    assert abs(win_rate - 0.5) < 0.01


def test_win_rate_no_trades():
    """Test win rate with no trades."""
    trade_returns = pd.Series([])

    win_rate = calculate_win_rate(trade_returns)

    assert win_rate == 0.0


def test_win_rate_with_zero_returns():
    """Test win rate with some zero returns."""
    trade_returns = pd.Series([0.05, 0.0, -0.03, 0.0, 0.02])

    win_rate = calculate_win_rate(trade_returns)

    # 2 wins out of 5 = 0.4
    assert abs(win_rate - 0.4) < 0.01


def test_average_gain_positive_trades():
    """Test average gain calculation."""
    trade_returns = pd.Series([0.05, -0.03, 0.10, -0.02, 0.07])

    avg_gain = calculate_average_gain(trade_returns)

    # Average of [0.05, 0.10, 0.07] = 0.0733
    expected = (0.05 + 0.10 + 0.07) / 3
    assert abs(avg_gain - expected) < 0.01


def test_average_gain_no_wins():
    """Test average gain with no winning trades."""
    trade_returns = pd.Series([-0.05, -0.03, -0.10, -0.02])

    avg_gain = calculate_average_gain(trade_returns)

    assert avg_gain == 0.0


def test_average_gain_all_wins():
    """Test average gain with all winning trades."""
    trade_returns = pd.Series([0.05, 0.10, 0.07, 0.03])

    avg_gain = calculate_average_gain(trade_returns)

    expected = (0.05 + 0.10 + 0.07 + 0.03) / 4
    assert abs(avg_gain - expected) < 0.01


def test_average_loss_negative_trades():
    """Test average loss calculation."""
    trade_returns = pd.Series([0.05, -0.03, 0.10, -0.05, 0.07, -0.02])

    avg_loss = calculate_average_loss(trade_returns)

    # Average of [-0.03, -0.05, -0.02] = -0.0333
    expected = (-0.03 + -0.05 + -0.02) / 3
    assert abs(avg_loss - expected) < 0.01


def test_average_loss_no_losses():
    """Test average loss with no losing trades."""
    trade_returns = pd.Series([0.05, 0.03, 0.10, 0.02])

    avg_loss = calculate_average_loss(trade_returns)

    assert avg_loss == 0.0


def test_average_loss_all_losses():
    """Test average loss with all losing trades."""
    trade_returns = pd.Series([-0.05, -0.03, -0.10, -0.02])

    avg_loss = calculate_average_loss(trade_returns)

    expected = (-0.05 + -0.03 + -0.10 + -0.02) / 4
    assert abs(avg_loss - expected) < 0.01


def test_profit_factor_positive():
    """Test profit factor with profitable trading."""
    trade_returns = pd.Series([0.10, -0.03, 0.08, -0.02, 0.06])

    pf = calculate_profit_factor(trade_returns)

    # Gross profit = 0.10 + 0.08 + 0.06 = 0.24
    # Gross loss = 0.03 + 0.02 = 0.05
    # PF = 0.24 / 0.05 = 4.8
    expected = (0.10 + 0.08 + 0.06) / (0.03 + 0.02)
    assert abs(pf - expected) < 0.1


def test_profit_factor_no_losses():
    """Test profit factor with no losing trades."""
    trade_returns = pd.Series([0.10, 0.05, 0.08, 0.03])

    pf = calculate_profit_factor(trade_returns)

    # Should return infinity
    assert pf == float('inf')


def test_profit_factor_no_wins():
    """Test profit factor with no winning trades."""
    trade_returns = pd.Series([-0.05, -0.03, -0.10, -0.02])

    pf = calculate_profit_factor(trade_returns)

    # Should return 0
    assert pf == 0.0


def test_profit_factor_breakeven():
    """Test profit factor near breakeven."""
    trade_returns = pd.Series([0.10, -0.10, 0.05, -0.05])

    pf = calculate_profit_factor(trade_returns)

    # Gross profit = 0.15, Gross loss = 0.15
    # PF = 1.0
    assert abs(pf - 1.0) < 0.01


def test_profit_factor_no_trades():
    """Test profit factor with no trades."""
    trade_returns = pd.Series([])

    pf = calculate_profit_factor(trade_returns)

    assert pf == 0.0


def test_calculate_all_metrics_comprehensive():
    """Test calculate_all_metrics with comprehensive data."""
    equity = pd.Series([100, 105, 103, 108, 110, 107, 112, 115])
    trade_returns = pd.Series([0.05, -0.02, 0.04, -0.01, 0.03])

    metrics = calculate_all_metrics(equity, trade_returns)

    assert 'total_return' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'win_rate' in metrics
    assert 'avg_gain' in metrics
    assert 'avg_loss' in metrics
    assert 'profit_factor' in metrics

    # Verify all metrics are numeric
    for key, value in metrics.items():
        assert isinstance(value, (int, float))


def test_calculate_all_metrics_no_trade_returns():
    """Test calculate_all_metrics without trade returns."""
    equity = pd.Series([100, 105, 110, 115, 120])

    metrics = calculate_all_metrics(equity, trade_returns=None)

    # Should have basic metrics
    assert metrics['total_return'] > 0
    assert metrics['sharpe_ratio'] >= 0
    assert metrics['max_drawdown'] == 0.0

    # Trade-based metrics should be zero
    assert metrics['win_rate'] == 0.0
    assert metrics['avg_gain'] == 0.0
    assert metrics['avg_loss'] == 0.0
    assert metrics['profit_factor'] == 0.0


def test_calculate_all_metrics_insufficient_equity_data():
    """Test calculate_all_metrics with insufficient equity data."""
    equity = pd.Series([100])
    trade_returns = pd.Series([0.05])

    metrics = calculate_all_metrics(equity, trade_returns)

    # Should return zero/default values for equity-based metrics
    assert metrics['total_return'] == 0.0
    assert metrics['sharpe_ratio'] == 0.0
    assert metrics['max_drawdown'] == 0.0


def test_calculate_all_metrics_with_risk_free_rate():
    """Test calculate_all_metrics with custom risk-free rate."""
    equity = pd.Series([100, 105, 110, 115, 120])
    trade_returns = pd.Series([0.05, 0.03, 0.04])

    metrics = calculate_all_metrics(equity, trade_returns, risk_free_rate=0.05)

    assert 'sharpe_ratio' in metrics
    assert isinstance(metrics['sharpe_ratio'], float)


def test_sharpe_ratio_annualization():
    """Test Sharpe ratio annualization factor."""
    # Daily returns with known mean and std
    returns = pd.Series([0.001] * 100)  # Constant small positive return

    sharpe = calculate_sharpe_ratio(returns, periods_per_year=252)

    # Should be annualized (multiplied by sqrt(252))
    assert sharpe > 0
    # Annualization factor ≈ 15.87 (sqrt(252))
    assert sharpe > 10


def test_max_drawdown_calculation_accuracy():
    """Test max drawdown calculation with known scenario."""
    # Scenario: peak at 120, trough at 80
    equity = pd.Series([100, 110, 120, 100, 90, 80, 90, 100])

    dd = calculate_max_drawdown(equity)

    # Expected: (80 - 120) / 120 = -0.3333
    expected = (80 - 120) / 120
    assert abs(dd - expected) < 0.01


def test_win_rate_edge_case_single_win():
    """Test win rate with single winning trade."""
    trade_returns = pd.Series([0.05])

    win_rate = calculate_win_rate(trade_returns)

    assert win_rate == 1.0


def test_win_rate_edge_case_single_loss():
    """Test win rate with single losing trade."""
    trade_returns = pd.Series([-0.05])

    win_rate = calculate_win_rate(trade_returns)

    assert win_rate == 0.0


def test_metrics_handle_nan_values():
    """Test metrics handling of NaN values."""
    equity = pd.Series([100, np.nan, 110, 115, np.nan, 120])
    trade_returns = pd.Series([0.05, np.nan, -0.02, np.nan, 0.03])

    metrics = calculate_all_metrics(equity, trade_returns)

    # Should handle NaN gracefully
    assert all(not np.isnan(v) or v == float('inf') for v in metrics.values())


def test_total_return_percentage_accuracy():
    """Test total return percentage calculation."""
    equity = pd.Series([100, 125])

    total_return = calculate_total_return(equity)

    # 25% return
    assert abs(total_return - 0.25) < 0.0001


def test_profit_factor_edge_case_large_wins():
    """Test profit factor with large wins vs small losses."""
    trade_returns = pd.Series([0.20, 0.15, -0.01, -0.02, 0.10])

    pf = calculate_profit_factor(trade_returns)

    # Gross profit = 0.45, Gross loss = 0.03
    # PF = 15
    expected = (0.20 + 0.15 + 0.10) / (0.01 + 0.02)
    assert abs(pf - expected) < 0.1


def test_sharpe_ratio_high_volatility():
    """Test Sharpe ratio with high volatility."""
    # High volatility returns
    returns = pd.Series([0.05, -0.04, 0.06, -0.05, 0.07, -0.06])

    sharpe = calculate_sharpe_ratio(returns)

    # High volatility should lower Sharpe
    assert sharpe < 5  # Lower than stable returns


def test_average_gain_loss_asymmetry():
    """Test that average gain and loss handle asymmetry correctly."""
    # More frequent small losses, fewer large wins
    trade_returns = pd.Series([0.20, -0.02, -0.01, -0.02, -0.01])

    avg_gain = calculate_average_gain(trade_returns)
    avg_loss = calculate_average_loss(trade_returns)

    assert avg_gain == 0.20  # Only one win
    # Average loss = (-0.02 - 0.01 - 0.02 - 0.01) / 4 = -0.015
    assert abs(avg_loss - (-0.015)) < 0.001
