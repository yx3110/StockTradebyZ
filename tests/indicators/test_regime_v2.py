"""ng2.0a multi-beta vote regime unit tests."""
import numpy as np
import pandas as pd
import pytest

from indicators.regime_classifier import compute_regime_v2


def test_regime_v2_unanimous_bull():
    n = 30
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    v11 = pd.Series([1] * n, index=idx)
    b1 = pd.Series([1] * n, index=idx)
    b2 = pd.Series([1] * n, index=idx)
    out = compute_regime_v2(v11, b1, b2, system_streak=3)
    # After streak_days=3, all should be bull (+1); flip confirmed at index 2
    assert (out['regime_v2'].iloc[2:] == 1).all()


def test_regime_v2_unanimous_bear():
    n = 30
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    v11 = pd.Series([0] * n, index=idx)
    b1 = pd.Series([0] * n, index=idx)
    b2 = pd.Series([0] * n, index=idx)
    out = compute_regime_v2(v11, b1, b2, system_streak=3)
    assert (out['regime_v2'].iloc[3:] == -1).all()


def test_regime_v2_majority_vote():
    n = 30
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    # 2-of-3 bull (V11+B1, B2 bear)
    v11 = pd.Series([1] * n, index=idx)
    b1 = pd.Series([1] * n, index=idx)
    b2 = pd.Series([0] * n, index=idx)
    out = compute_regime_v2(v11, b1, b2, system_streak=3)
    assert out['vote_count'].iloc[-1] == 2
    assert out['regime_v2_raw'].iloc[-1] == 1
    assert out['regime_v2'].iloc[-1] == 1


def test_regime_v2_minority_is_bear():
    n = 30
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    v11 = pd.Series([1] * n, index=idx)
    b1 = pd.Series([0] * n, index=idx)
    b2 = pd.Series([0] * n, index=idx)
    out = compute_regime_v2(v11, b1, b2, system_streak=3)
    assert out['vote_count'].iloc[-1] == 1
    assert out['regime_v2_raw'].iloc[-1] == -1


def test_regime_v2_streak_blocks_one_day_flip():
    """Single day vote-flip shouldn't propagate without streak."""
    n = 10
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    # Mostly bear, 1 day bull majority on day 5
    v11 = pd.Series([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], index=idx)
    b1 = pd.Series([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], index=idx)
    b2 = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], index=idx)
    out = compute_regime_v2(v11, b1, b2, system_streak=3)
    # Day 5 raw flips to +1 but streak<3 → confirmed regime_v2 should stay -1
    assert out['regime_v2'].iloc[4] == -1


def test_regime_v2_returns_required_columns():
    n = 30
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    v11 = pd.Series([1] * n, index=idx)
    b1 = pd.Series([1] * n, index=idx)
    b2 = pd.Series([1] * n, index=idx)
    out = compute_regime_v2(v11, b1, b2)
    assert {'vote_count', 'regime_v2_raw', 'regime_v2', 'regime_v2_streak'} <= set(out.columns)


def test_regime_v2_unanimous_vote_requires_all_3():
    """vote_threshold=3 means all 3 signals must be bull to flip bull."""
    n = 30
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    # 2-of-3 bull (V11+B1, B2 bear) — should NOT trigger bull when threshold=3
    v11 = pd.Series([1] * n, index=idx)
    b1 = pd.Series([1] * n, index=idx)
    b2 = pd.Series([0] * n, index=idx)
    out = compute_regime_v2(v11, b1, b2, system_streak=3, vote_threshold=3)
    assert out['vote_count'].iloc[-1] == 2
    # raw should be -1 (bear) since vote=2 < threshold=3
    assert out['regime_v2_raw'].iloc[-1] == -1
    assert out['regime_v2'].iloc[-1] == -1


def test_regime_v2_unanimous_vote_3_of_3_bull():
    """All 3 bull with threshold=3 should still produce bull."""
    n = 30
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    v11 = pd.Series([1] * n, index=idx)
    b1 = pd.Series([1] * n, index=idx)
    b2 = pd.Series([1] * n, index=idx)
    out = compute_regime_v2(v11, b1, b2, system_streak=3, vote_threshold=3)
    assert out['vote_count'].iloc[-1] == 3
    assert (out['regime_v2'].iloc[3:] == 1).all()


def test_regime_v2_default_threshold_is_majority_2():
    """Default vote_threshold=2 should preserve backward-compat (majority vote)."""
    n = 30
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    v11 = pd.Series([1] * n, index=idx)
    b1 = pd.Series([1] * n, index=idx)
    b2 = pd.Series([0] * n, index=idx)
    # Don't pass vote_threshold — should default to 2
    out = compute_regime_v2(v11, b1, b2, system_streak=3)
    assert out['regime_v2_raw'].iloc[-1] == 1  # 2 of 3 = bull at default threshold
