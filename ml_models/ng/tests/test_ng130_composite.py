"""Tests for ng1.3.0 β composite scoring."""
import numpy as np
import pandas as pd
import pytest


def test_rank_pct_basic():
    from ml_models.ng.ng130_composite import rank_pct
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    r = rank_pct(s)
    assert abs(r.iloc[0] - 0.25) < 1e-6
    assert abs(r.iloc[-1] - 1.0) < 1e-6


def test_composite_beta_zero_equals_excess():
    """β=0 → composite score equals Z_excess (ng1.0.1 baseline behavior)."""
    from ml_models.ng.ng130_composite import compute_composite, rank_pct
    pred_excess = pd.Series([0.1, 0.2, 0.3, 0.4])
    pred_downside = pd.Series([-0.05, -0.10, -0.02, -0.15])
    score = compute_composite(pred_excess, pred_downside, beta=0.0)
    z_ex = rank_pct(pred_excess)
    pd.testing.assert_series_equal(score, z_ex, check_names=False)


def test_composite_beta_positive_penalizes_downside():
    """β=0.3: high downside rank (best) gets most penalty."""
    from ml_models.ng.ng130_composite import compute_composite
    pred_excess = pd.Series([0.2, 0.2, 0.2, 0.2])  # all equal → ties
    pred_downside = pd.Series([-0.20, -0.05, -0.10, -0.15])
    score = compute_composite(pred_excess, pred_downside, beta=0.3)
    # Stock 0 has worst (most negative) downside → rank_pct lowest → least penalty
    # Stock 1 has best downside (-0.05) → rank_pct highest → most penalty
    # So stock 0 composite > stock 1 composite
    assert score.iloc[0] > score.iloc[1]


def test_composite_beta_validates_range():
    from ml_models.ng.ng130_composite import compute_composite
    ps = pd.Series([0.1, 0.2])
    pd_ = pd.Series([-0.1, -0.2])
    with pytest.raises(ValueError):
        compute_composite(ps, pd_, beta=-0.1)
    with pytest.raises(ValueError):
        compute_composite(ps, pd_, beta=1.5)
