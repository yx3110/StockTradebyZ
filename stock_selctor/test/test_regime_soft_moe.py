"""P2.7: tests for soft-MOE bull probability + score blending."""
from __future__ import annotations

import numpy as np
import pytest

from indicators.regime_classifier import compute_bull_proba, blend_scores, smooth_proba_ema


def test_strong_bull_inputs_gives_high_prob():
    """Position above MA60 + macd water + bull streak → P >> 0.5.

    With pos=+10%, water=1, rising=0 (single elem), streak=+5 → z=2.5 → sigmoid≈0.92.
    """
    var1 = np.array([110.0])
    ma60 = np.array([100.0])
    macd = np.array([0.5])
    streak = np.array([5])
    p = compute_bull_proba(var1, ma60, macd, streak)
    assert p[0] > 0.90


def test_strong_bear_inputs_gives_low_prob():
    """Position below MA60 + falling macd + bear streak → P < 0.20."""
    var1 = np.array([90.0, 90.0])
    ma60 = np.array([100.0, 100.0])
    macd = np.array([-0.5, -0.6])  # macd[1] not rising
    streak = np.array([-5, -5])
    p = compute_bull_proba(var1, ma60, macd, streak)
    # pos=-10%, water=0, rising=0, streak=-1 → z=-1.5 → sigmoid≈0.18
    assert p[1] < 0.20


def test_flat_inputs_around_50():
    """var1 == ma60, macd == 0, no streak → ≈ 0.5 within reason."""
    var1 = np.array([100.0])
    ma60 = np.array([100.0])
    macd = np.array([0.0])
    streak = np.array([0])
    p = compute_bull_proba(var1, ma60, macd, streak)
    # macd > 0 is False, macd > prev rising is False (first row), streak 0 → z = 0
    assert 0.45 < p[0] < 0.55


def test_streak_optional_defaults_to_zero():
    var1 = np.array([110.0, 110.0])
    ma60 = np.array([100.0, 100.0])
    macd = np.array([0.5, 0.6])
    p_with_streak = compute_bull_proba(var1, ma60, macd, np.array([5, 5]))
    p_no_streak = compute_bull_proba(var1, ma60, macd, None)
    # both bullish, but streak adds extra weight
    assert (p_with_streak >= p_no_streak).all()


def test_proba_is_monotone_in_position():
    """Holding macd / streak fixed, P increases as var1 / ma60 grows."""
    macd = np.array([0.5, 0.5, 0.5])
    streak = np.array([0, 0, 0])
    p_below = compute_bull_proba(np.array([90.0]), np.array([100.0]), macd[:1], streak[:1])
    p_at = compute_bull_proba(np.array([100.0]), np.array([100.0]), macd[:1], streak[:1])
    p_above = compute_bull_proba(np.array([110.0]), np.array([100.0]), macd[:1], streak[:1])
    assert p_below[0] < p_at[0] < p_above[0]


def test_proba_clamped_to_unit_interval():
    """Extreme inputs (huge position, max streak) still ∈ (0, 1)."""
    var1 = np.array([1e6])
    ma60 = np.array([1.0])
    macd = np.array([100.0])
    streak = np.array([100])
    p = compute_bull_proba(var1, ma60, macd, streak)
    assert 0.0 < p[0] < 1.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_bull_proba(np.array([1.0, 2.0]), np.array([1.0]), np.array([0.0]))


# ─────────────── blend_scores ───────────────

def test_blend_pure_bull():
    p = np.array([1.0, 1.0])
    bull = np.array([10.0, 20.0])
    bear = np.array([0.0, 0.0])
    out = blend_scores(p, bull, bear)
    assert (out == bull).all()


def test_blend_pure_bear():
    p = np.array([0.0, 0.0])
    bull = np.array([10.0, 20.0])
    bear = np.array([5.0, 8.0])
    out = blend_scores(p, bull, bear)
    assert (out == bear).all()


def test_blend_50_50():
    p = np.array([0.5])
    bull = np.array([10.0])
    bear = np.array([4.0])
    out = blend_scores(p, bull, bear)
    assert abs(out[0] - 7.0) < 1e-9


def test_blend_clips_out_of_range_proba():
    """P > 1 or < 0 should be clamped before blending (defensive)."""
    p = np.array([1.5, -0.3])
    bull = np.array([10.0, 10.0])
    bear = np.array([0.0, 0.0])
    out = blend_scores(p, bull, bear)
    assert out[0] == 10.0  # clipped to 1.0
    assert out[1] == 0.0   # clipped to 0.0


# ─────────────── smooth_proba_ema ───────────────

def test_smooth_constant_input_unchanged():
    p = np.array([0.7] * 20)
    out = smooth_proba_ema(p, span=5)
    assert np.allclose(out, 0.7)


def test_smooth_reduces_step_response():
    """Step from 0 to 1: smoothed series approaches but stays below raw step."""
    raw = np.array([0.0] * 5 + [1.0] * 5)
    out = smooth_proba_ema(raw, span=5)
    # First post-step value should be much less than 1.0
    assert out[5] < 0.5
    # Eventually approaches 1.0
    assert out[-1] > 0.85


def test_smooth_first_value_passthrough():
    p = np.array([0.42, 0.99])
    out = smooth_proba_ema(p, span=5)
    assert out[0] == 0.42


def test_smooth_empty_safe():
    out = smooth_proba_ema(np.array([]))
    assert out.size == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
