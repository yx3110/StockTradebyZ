"""B2 realized vol signal unit tests."""
import numpy as np
import pandas as pd

from indicators.realized_vol import compute_realized_vol_signal


def _fake_index_close(n=400, vol=0.012, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2023-01-01', periods=n, freq='B')
    rets = rng.normal(0, vol, size=n)
    prices = 3000.0 * np.exp(np.cumsum(rets))
    return pd.Series(prices, index=dates, name='close')


def test_realized_vol_returns_required_cols():
    s = _fake_index_close()
    out = compute_realized_vol_signal(s, rv_window=60, percentile_window=252, streak_days=3)
    assert {'rv_60d', 'rv_percentile_252', 'b2_bull', 'b2_streak'} <= set(out.columns)
    assert len(out) == len(s)


def test_realized_vol_low_vol_is_bull():
    """Constant-low-vol series → b2_bull=1 after streak."""
    n = 400
    dates = pd.date_range('2023-01-01', periods=n, freq='B')
    rng = np.random.default_rng(1)
    rets_high = rng.normal(0, 0.020, size=252)
    rets_low = rng.normal(0, 0.005, size=n - 252)
    rets = np.concatenate([rets_high, rets_low])
    prices = 3000.0 * np.exp(np.cumsum(rets))
    s = pd.Series(prices, index=dates)
    out = compute_realized_vol_signal(s, rv_window=60, percentile_window=252, streak_days=3)
    assert out['b2_bull'].iloc[-1] == 1
    assert out['rv_percentile_252'].iloc[-1] < 0.30


def test_realized_vol_high_vol_is_bear():
    n = 400
    dates = pd.date_range('2023-01-01', periods=n, freq='B')
    rng = np.random.default_rng(2)
    rets_low = rng.normal(0, 0.005, size=252)
    rets_high = rng.normal(0, 0.030, size=n - 252)
    rets = np.concatenate([rets_low, rets_high])
    prices = 3000.0 * np.exp(np.cumsum(rets))
    s = pd.Series(prices, index=dates)
    out = compute_realized_vol_signal(s, rv_window=60, percentile_window=252, streak_days=3)
    assert out['b2_bull'].iloc[-1] == 0
    assert out['rv_percentile_252'].iloc[-1] > 0.70


def test_realized_vol_no_lookahead():
    """Output at t must only use data up to t (no future-looking)."""
    s = _fake_index_close(n=350)
    out_full = compute_realized_vol_signal(s, rv_window=60, percentile_window=252, streak_days=3)
    out_truncated = compute_realized_vol_signal(s.iloc[:300], rv_window=60, percentile_window=252, streak_days=3)
    common = out_full.index.intersection(out_truncated.index)
    # Use .equals() for NaN-safe comparison (NaN == NaN is False with ==).
    assert out_full.loc[common, 'rv_60d'].equals(out_truncated.loc[common, 'rv_60d'])
    assert (out_full.loc[common, 'rv_percentile_252'].round(8)
            .equals(out_truncated.loc[common, 'rv_percentile_252'].round(8)))
