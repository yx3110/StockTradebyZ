"""Unit tests for ng1.2.3 soft-downside label transform."""
import numpy as np
import pytest

from ml_models.ng.ng123_label_transform import (
    compute_path_min_kd,
    compute_downside_kd,
    apply_downside_penalty,
)


# --- compute_path_min_kd ----------------------------------------------------

def test_path_min_simple_decline():
    """Closes drop monotonically; path_min = lowest / today - 1."""
    today_close = 100.0
    future = np.array([99.0, 95.0, 92.0, 90.0, 88.0])
    pm = compute_path_min_kd(today_close, future)
    assert abs(pm - (88.0 / 100.0 - 1.0)) < 1e-9  # = -0.12


def test_path_min_all_above_today():
    """If all future closes > today, path_min is positive (rare but valid)."""
    today_close = 100.0
    future = np.array([101.0, 102.0, 105.0])
    pm = compute_path_min_kd(today_close, future)
    assert abs(pm - (101.0 / 100.0 - 1.0)) < 1e-9  # = +0.01


def test_path_min_with_recovery():
    """Drops then recovers; path_min captures the trough."""
    today_close = 100.0
    future = np.array([95.0, 88.0, 92.0, 105.0])  # bottoms at 88
    pm = compute_path_min_kd(today_close, future)
    assert abs(pm - (88.0 / 100.0 - 1.0)) < 1e-9


def test_path_min_empty_future():
    """No future data → NaN."""
    pm = compute_path_min_kd(100.0, np.array([]))
    assert np.isnan(pm)


def test_path_min_zero_today():
    """Today close = 0 → NaN (avoid div by zero)."""
    pm = compute_path_min_kd(0.0, np.array([1.0, 2.0]))
    assert np.isnan(pm)


# --- compute_downside_kd ----------------------------------------------------

def test_downside_from_negative_path_min():
    """Negative path_min → downside is its magnitude."""
    assert abs(compute_downside_kd(-0.12) - 0.12) < 1e-9


def test_downside_from_positive_path_min():
    """Positive path_min → downside = 0 (no drawdown)."""
    assert compute_downside_kd(0.05) == 0.0


def test_downside_from_zero():
    assert compute_downside_kd(0.0) == 0.0


def test_downside_nan_propagates():
    assert np.isnan(compute_downside_kd(np.nan))


# --- apply_downside_penalty -------------------------------------------------

def test_penalty_clean_winner():
    """excess=+0.05, downside=0.03, λ=0.3 → label = 0.05 - 0.009 = 0.041."""
    res = apply_downside_penalty(excess=0.05, downside=0.03, lam=0.3)
    assert abs(res - 0.041) < 1e-9


def test_penalty_volatile_winner():
    """excess=+0.05, downside=0.15, λ=0.3 → label = 0.05 - 0.045 = 0.005 (demoted)."""
    res = apply_downside_penalty(excess=0.05, downside=0.15, lam=0.3)
    assert abs(res - 0.005) < 1e-9


def test_penalty_lambda_zero_passthrough():
    """λ=0 → label = excess (sanity for ablation)."""
    res = apply_downside_penalty(excess=0.05, downside=0.15, lam=0.0)
    assert abs(res - 0.05) < 1e-9


def test_penalty_nan_excess():
    res = apply_downside_penalty(excess=np.nan, downside=0.05, lam=0.3)
    assert np.isnan(res)


def test_penalty_nan_downside():
    res = apply_downside_penalty(excess=0.05, downside=np.nan, lam=0.3)
    assert np.isnan(res)


def test_penalty_vectorized():
    """Function must accept arrays."""
    excess = np.array([0.05, 0.10, -0.05, 0.0])
    downside = np.array([0.03, 0.15, 0.20, 0.0])
    res = apply_downside_penalty(excess=excess, downside=downside, lam=0.3)
    expected = np.array([0.041, 0.055, -0.110, 0.0])
    assert np.allclose(res, expected, atol=1e-9)
