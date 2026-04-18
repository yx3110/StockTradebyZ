"""Tests for ng1.3.0 downside label (min-cumret over N days)."""
import numpy as np
import pytest


def test_downside_basic_drop():
    """股价先涨后大跌 → downside 反映最深跌幅."""
    from ml_models.ng.ng130_downside_label import compute_downside_label
    future_closes = np.array([11, 12, 8, 9, 10], dtype=float)
    t_close = 10.0
    result = compute_downside_label(t_close, future_closes)
    assert abs(result - (-0.20)) < 1e-6


def test_downside_only_rally():
    """连续上涨 → downside = min(future) / t_close - 1, 仍是(小)负或小正."""
    from ml_models.ng.ng130_downside_label import compute_downside_label
    future_closes = np.array([10.5, 11, 11.5, 12, 12.5], dtype=float)
    t_close = 10.0
    result = compute_downside_label(t_close, future_closes)
    assert abs(result - 0.05) < 1e-6


def test_downside_insufficient_future():
    """未来窗口天数不够 → NaN."""
    from ml_models.ng.ng130_downside_label import compute_downside_label
    future_closes = np.array([], dtype=float)
    assert np.isnan(compute_downside_label(10.0, future_closes))


def test_downside_zero_tclose():
    """t_close = 0 保护除零."""
    from ml_models.ng.ng130_downside_label import compute_downside_label
    future = np.array([5.0, 4.0], dtype=float)
    assert np.isnan(compute_downside_label(0.0, future))


def test_compute_all_horizons():
    """compute_all_downside_horizons 返回 4 个 horizon."""
    from ml_models.ng.ng130_downside_label import compute_all_downside_horizons
    future_closes = np.array([9, 8, 7, 8, 9, 10, 11, 10, 9, 10, 11, 12, 11, 10, 9], dtype=float)
    t_close = 10.0
    result = compute_all_downside_horizons(t_close, future_closes)
    assert 'downside_3d' in result
    assert 'downside_5d' in result
    assert 'downside_10d' in result
    assert 'downside_15d' in result
    # 3d: min(9,8,7)=7, downside=7/10-1=-0.30
    assert abs(result['downside_3d'] - (-0.30)) < 1e-6
    # 15d: min of full = 7, same as 3d since 7 is in first 3
    assert abs(result['downside_15d'] - (-0.30)) < 1e-6


def test_compute_all_horizons_insufficient():
    """future_closes 不够 15 天 → 较短 horizon 仍计算, 超出部分 NaN."""
    from ml_models.ng.ng130_downside_label import compute_all_downside_horizons
    future_closes = np.array([8.0, 7.0, 6.5], dtype=float)  # 只有 3 天
    t_close = 10.0
    result = compute_all_downside_horizons(t_close, future_closes)
    assert abs(result['downside_3d'] - (-0.35)) < 1e-6  # min=6.5
    assert np.isnan(result['downside_5d'])
    assert np.isnan(result['downside_10d'])
    assert np.isnan(result['downside_15d'])
