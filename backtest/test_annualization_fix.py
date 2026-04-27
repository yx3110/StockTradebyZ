"""
P0.2 单测: _compute_period_risk_metrics calendar-time 年化修复.

验证:
1. 100% trade 天数 (无 cash gap): calendar == active 年化
2. 50% cash gap (稀疏): calendar 年化 ≈ active 年化 / 2
3. 无 DatetimeIndex (legacy fallback): calendar == active
4. cash_ratio 正确推算
5. Sharpe > 4 触发 warning
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import pytest

from backtest.backtest_report_based import _compute_period_risk_metrics


def _make_dense_returns(n_periods: int, holding_days: int,
                        per_period_ret: float, start: str = "2024-01-01"):
    """生成连续 trade 的 period 收益序列 (无 cash gap)."""
    start_dt = pd.Timestamp(start)
    dates = [start_dt + pd.Timedelta(days=i * holding_days * 7 / 5) for i in range(n_periods)]
    returns = pd.Series([per_period_ret] * n_periods,
                        index=pd.DatetimeIndex(dates))
    return returns


def _make_sparse_returns(n_periods: int, holding_days: int,
                         per_period_ret: float, gap_factor: float,
                         start: str = "2024-01-01"):
    """生成稀疏 trade (gap_factor=2 → 跳过 50% 天数, calendar 是 active 的 2 倍)."""
    start_dt = pd.Timestamp(start)
    dates = [start_dt + pd.Timedelta(days=i * holding_days * 7 / 5 * gap_factor)
             for i in range(n_periods)]
    returns = pd.Series([per_period_ret] * n_periods,
                        index=pd.DatetimeIndex(dates))
    return returns


def test_dense_calendar_equals_active():
    """100% trade 时, calendar 年化和 active 年化应一致."""
    returns = _make_dense_returns(n_periods=20, holding_days=10,
                                  per_period_ret=0.02)
    result = _compute_period_risk_metrics(returns, holding_days=10)

    assert abs(result['annual_return'] - result['annual_return_active']) < 0.01, \
        f"dense case: calendar {result['annual_return']:.4f} != active {result['annual_return_active']:.4f}"
    assert result['cash_ratio'] < 0.05, f"dense case cash_ratio={result['cash_ratio']}"


def test_sparse_calendar_approximately_half_active():
    """gap_factor=2 (cash 50%) 时, calendar 年化应约为 active 年化的 1/2 (复利意义下).

    具体: (1+r)^(252/2T) vs (1+r)^(252/T), 前者是后者的"开根号"减 1.
    """
    n = 20
    h = 10
    r = 0.02
    dense = _compute_period_risk_metrics(
        _make_dense_returns(n, h, r), holding_days=h)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sparse = _compute_period_risk_metrics(
            _make_sparse_returns(n, h, r, gap_factor=2.0), holding_days=h)

    # 累计收益完全相同
    assert abs(dense['cumulative_return'] - sparse['cumulative_return']) < 1e-10

    # active 年化两者也应该一致 (n_periods × holding_days 不变)
    assert abs(dense['annual_return_active'] - sparse['annual_return_active']) < 0.01

    # calendar 年化: sparse 应远小于 dense (≈ sqrt 关系)
    assert sparse['annual_return'] < dense['annual_return'] * 0.7, \
        f"sparse calendar {sparse['annual_return']:.4f} 没有显著低于 dense {dense['annual_return']:.4f}"

    # cash_ratio 应该 ≈ 0.5
    assert 0.4 < sparse['cash_ratio'] < 0.6, \
        f"sparse cash_ratio={sparse['cash_ratio']:.3f} 不在 (0.4, 0.6)"


def test_legacy_fallback_no_datetime_index():
    """无 DatetimeIndex 时回退到 active 口径 (legacy)."""
    n = 20
    returns = pd.Series([0.02] * n)  # RangeIndex
    result = _compute_period_risk_metrics(returns, holding_days=10)

    assert abs(result['annual_return'] - result['annual_return_active']) < 1e-10
    assert result['cash_ratio'] == 0.0


def test_sparse_trading_warning():
    """Sharpe > 4 或 cash_ratio > 20% 应触发 sparse-trading warning."""
    # 高 Sharpe 场景: 收益低但稳定, gap_factor 大
    returns = _make_sparse_returns(n_periods=30, holding_days=10,
                                   per_period_ret=0.005, gap_factor=3.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _compute_period_risk_metrics(returns, holding_days=10)

    sparse_warnings = [w for w in caught if "sparse-trading" in str(w.message)]
    assert len(sparse_warnings) >= 1, "高 cash_ratio 场景未触发 warning"


def test_too_few_periods_returns_zeros():
    """少于 5 期返回全零 dict (含新字段)."""
    returns = pd.Series([0.01, 0.02, 0.03])
    result = _compute_period_risk_metrics(returns, holding_days=10)
    assert result['annual_return'] == 0
    assert result['annual_return_active'] == 0
    assert result['cash_ratio'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
