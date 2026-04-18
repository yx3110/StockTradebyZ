"""Tests for ng1.3.0 Tier C amount-based candidate factors."""
import numpy as np
import pytest


def test_amihud_illiq_20d_basic():
    """Amihud = mean(|ret| / amount × 1e8), positive for normal data."""
    from ml_models.ng.ng130_amount_factors import compute_amihud_illiq_20d
    closes = np.linspace(10.0, 11.0, 21)
    amounts = np.full(21, 1e8)
    result = compute_amihud_illiq_20d(closes, amounts)
    assert not np.isnan(result)
    assert result > 0


def test_amihud_illiq_20d_insufficient_data():
    """< 21 closes → NaN."""
    from ml_models.ng.ng130_amount_factors import compute_amihud_illiq_20d
    closes = np.array([10.0] * 10)
    amounts = np.array([1e8] * 10)
    assert np.isnan(compute_amihud_illiq_20d(closes, amounts))


def test_amihud_illiq_20d_zero_amount():
    """Zero amount days gracefully handled (no inf)."""
    from ml_models.ng.ng130_amount_factors import compute_amihud_illiq_20d
    closes = np.linspace(10.0, 11.0, 21)
    amounts = np.full(21, 1e8)
    amounts[5] = 0.0
    result = compute_amihud_illiq_20d(closes, amounts)
    assert not np.isnan(result)
    assert not np.isinf(result)


def test_vwap_close_ratio_basic():
    """VWAP ratio 恒零 for close==vwap."""
    from ml_models.ng.ng130_amount_factors import compute_vwap_close_ratio_20d
    closes = np.array([10.0] * 20)
    amounts = np.array([1e8] * 20)
    volumes = amounts / 10.0
    result = compute_vwap_close_ratio_20d(closes, amounts, volumes)
    assert abs(result) < 1e-6


def test_vwap_close_ratio_close_above_vwap():
    """Close > vwap 时 ratio > 0 (高位收盘)."""
    from ml_models.ng.ng130_amount_factors import compute_vwap_close_ratio_20d
    closes = np.array([11.0] * 20)
    amounts = np.array([1e8] * 20)
    volumes = amounts / 10.0
    result = compute_vwap_close_ratio_20d(closes, amounts, volumes)
    assert result > 0.05


def test_amount_acceleration_5d():
    """量能加速 = ma5/ma20 - 1. 最近 5 天放量 → 正值."""
    from ml_models.ng.ng130_amount_factors import compute_amount_acceleration_5d
    amounts = np.array([1e8] * 15 + [3e8] * 5)
    result = compute_amount_acceleration_5d(amounts)
    assert abs(result - 1.0) < 0.01


def test_amount_acceleration_5d_insufficient():
    """< 20 days → NaN."""
    from ml_models.ng.ng130_amount_factors import compute_amount_acceleration_5d
    amounts = np.array([1e8] * 10)
    assert np.isnan(compute_amount_acceleration_5d(amounts))


def test_tail_beta_60d_basic():
    """Tail beta: beta on bottom-30% market days."""
    from ml_models.ng.ng130_amount_factors import compute_tail_beta_60d
    np.random.seed(42)
    market_rets = np.random.randn(60) * 0.01
    stock_rets = 1.5 * market_rets + np.random.randn(60) * 0.005
    result = compute_tail_beta_60d(stock_rets, market_rets)
    assert not np.isnan(result)
    assert 0.3 <= result <= 3.5


def test_tail_beta_60d_insufficient():
    """< 60 days → NaN."""
    from ml_models.ng.ng130_amount_factors import compute_tail_beta_60d
    market_rets = np.random.randn(30) * 0.01
    stock_rets = np.random.randn(30) * 0.01
    assert np.isnan(compute_tail_beta_60d(stock_rets, market_rets))
