"""Unit tests for ng1.2.3 mined factor scaffold (pre-Stage-2 baseline)."""
import numpy as np
import pandas as pd
import pytest

from ml_models.ng.ng123_mined_factors import (
    MINED_FACTOR_SPEC,
    compute_mined_factor_value,
    compute_all_mined_factors_for_stock,
    get_mined_factor_names,
)


def _mk_ohlcv(n_days: int = 120, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV — random walk."""
    rng = np.random.RandomState(seed)
    closes = 100 * np.cumprod(1 + rng.randn(n_days) * 0.02)
    return pd.DataFrame({
        'open': closes * (1 - rng.uniform(0, 0.01, n_days)),
        'high': closes * (1 + rng.uniform(0, 0.02, n_days)),
        'low': closes * (1 - rng.uniform(0, 0.02, n_days)),
        'close': closes,
        'volume': rng.randint(1e6, 1e7, n_days).astype(float),
        'price_change_pct': np.diff(closes, prepend=closes[0]) / closes,
        'turnover_rate': rng.uniform(0.5, 5.0, n_days),
    })


def test_mined_factor_spec_is_list():
    """MINED_FACTOR_SPEC must be a list (empty until Stage 2 populates)."""
    assert isinstance(MINED_FACTOR_SPEC, list)


def test_get_mined_factor_names_returns_list():
    """get_mined_factor_names returns list of names (empty when spec empty)."""
    names = get_mined_factor_names()
    assert isinstance(names, list)
    # Each entry in spec should have 'name'
    for spec in MINED_FACTOR_SPEC:
        assert 'name' in spec


def test_compute_all_empty_when_spec_empty():
    """If MINED_FACTOR_SPEC is empty, orchestrator returns empty dict."""
    df = _mk_ohlcv(120)
    if len(MINED_FACTOR_SPEC) == 0:
        assert compute_all_mined_factors_for_stock(df) == {}


def test_compute_single_factor_shape():
    """Compute for each factor (if any) returns array of correct length."""
    df = _mk_ohlcv(120)
    for spec in MINED_FACTOR_SPEC:
        vals = compute_mined_factor_value(spec, df)
        assert vals is not None, f"Factor {spec['name']} returned None"
        assert len(vals) == len(df), f"Factor {spec['name']} wrong length"


def test_sign_flip_applied_when_true():
    """If spec has sign_flip=True, result should be negated vs raw."""
    df = _mk_ohlcv(120)
    for spec in MINED_FACTOR_SPEC:
        if spec.get('sign_flip'):
            vals_flip = compute_mined_factor_value(spec, df)
            spec_no_flip = {**spec, 'sign_flip': False}
            vals_raw = compute_mined_factor_value(spec_no_flip, df)
            mask = np.isfinite(vals_flip) & np.isfinite(vals_raw)
            if mask.sum() > 0:
                assert np.allclose(vals_flip[mask], -vals_raw[mask], atol=1e-9), \
                    f"Sign flip not applied for {spec['name']}"


def test_synthetic_depth1_factor():
    """Test the factor computation engine directly with a synthetic spec."""
    df = _mk_ohlcv(120)
    # Use a known depth-1 spec: ts_mean of close over 20 days
    synthetic_spec = {
        'name': 'test_ts_mean_close_20',
        'type': 'unary_ts',
        'op': 'ts_mean',
        'operand': 'close',
        'window': 20,
        'sign_flip': False,
    }
    vals = compute_mined_factor_value(synthetic_spec, df)
    assert len(vals) == 120
    # After 20+ days, should have finite values
    assert np.sum(np.isfinite(vals[20:])) > 0
