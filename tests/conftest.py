"""Shared pytest fixtures for generating synthetic stock OHLCV DataFrames."""

import os
import sys
import types

import numpy as np
import pandas as pd
import pytest


def _preload_selector_compat() -> None:
    """Pre-load Selector.py with `from __future__ import annotations` injected.

    Selector.py uses ``int | None`` union syntax (PEP 604) which requires
    Python 3.10+.  Running the test-suite under Python 3.9 (the system
    interpreter) would raise a TypeError at import time.  To work around
    this without modifying the source file we:
    1. Read Selector.py as text.
    2. Prepend ``from __future__ import annotations`` so that ALL annotations
       are treated as lazy strings (PEP 563 back-port behaviour).
    3. Compile + exec the modified source and register the resulting module
       in ``sys.modules["Selector"]`` before any test file tries to import it.
    """
    module_name = "Selector"
    if module_name in sys.modules:
        return

    # conftest.py lives in tests/; Selector.py is one level up
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    selector_path = os.path.join(base_dir, "Selector.py")

    with open(selector_path, "r", encoding="utf-8") as fh:
        source = fh.read()

    # Inject future-annotations import to enable lazy annotation evaluation
    patched_source = "from __future__ import annotations\n" + source

    code = compile(patched_source, selector_path, "exec")
    module = types.ModuleType(module_name)
    module.__file__ = selector_path
    exec(code, module.__dict__)  # noqa: S102
    sys.modules[module_name] = module


_preload_selector_compat()


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
