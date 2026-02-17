"""Shared pytest fixtures for generating synthetic stock OHLCV DataFrames."""

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(
    n: int = 60,
    start_price: float = 10.0,
    trend: float = 0.001,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with deterministic random data.

    Parameters
    ----------
    n:
        Number of trading days to generate.
    start_price:
        Starting close price.
    trend:
        Daily drift added to the log-price walk (positive = uptrend).
    seed:
        NumPy random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [open, high, low, close, volume] and a
        DatetimeIndex at business-day frequency.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2024-01-01", periods=n)

    log_returns = rng.normal(loc=trend, scale=0.01, size=n)
    close = start_price * np.exp(np.cumsum(log_returns))

    # Spread: high ≥ close ≥ open ≥ low
    spread = rng.uniform(0.002, 0.015, size=n) * close
    high = close + spread * rng.uniform(0.5, 1.0, size=n)
    low = close - spread * rng.uniform(0.5, 1.0, size=n)
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, size=n)

    volume = rng.integers(100_000, 5_000_000, size=n).astype(float)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    """Standard 60-day synthetic OHLCV DataFrame (neutral trend)."""
    return _make_ohlcv(n=60, trend=0.001, seed=42)


@pytest.fixture
def ohlcv_uptrend_df() -> pd.DataFrame:
    """60-day OHLCV DataFrame with a clear upward price trend."""
    return _make_ohlcv(n=60, trend=0.005, seed=7)


@pytest.fixture
def ohlcv_downtrend_df() -> pd.DataFrame:
    """60-day OHLCV DataFrame with a clear downward price trend."""
    return _make_ohlcv(n=60, trend=-0.005, seed=13)


@pytest.fixture
def ohlcv_flat_df() -> pd.DataFrame:
    """60-day OHLCV DataFrame with a near-flat (sideways) price trend."""
    return _make_ohlcv(n=60, trend=0.0, seed=99)


@pytest.fixture
def ohlcv_long_df() -> pd.DataFrame:
    """200-day synthetic OHLCV DataFrame for tests requiring longer history."""
    return _make_ohlcv(n=200, trend=0.001, seed=77)


@pytest.fixture
def ohlcv_single_row_df() -> pd.DataFrame:
    """Single-row OHLCV DataFrame for edge-case tests."""
    return _make_ohlcv(n=1, seed=0)


@pytest.fixture
def ohlcv_empty_df() -> pd.DataFrame:
    """Empty OHLCV DataFrame for empty-input edge-case tests."""
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
