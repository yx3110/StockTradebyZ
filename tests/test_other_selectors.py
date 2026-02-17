"""Unit tests for SuperB1Selector, BBIShortLongSelector, PeakKDJSelector,
MA60CrossVolumeWaveSelector, and BigBullishVolumeSelector in Selector.py.

Covers:
  - Instantiation with default and custom parameters
  - Empty data handling (select() and _passes_filters())
  - select() returning a list type
"""

import numpy as np
import pandas as pd
import pytest

from Selector import (
    BBIShortLongSelector,
    BigBullishVolumeSelector,
    MA60CrossVolumeWaveSelector,
    PeakKDJSelector,
    SuperB1Selector,
)


# ---------------------------------------------------------------------------
# Shared helper factories
# ---------------------------------------------------------------------------


def _make_ohlcv(
    n: int = 200,
    start_price: float = 10.0,
    trend: float = 0.001,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with a DatetimeIndex (no 'date' col).

    Produces small daily moves (scale=0.005) so that ``passes_day_constraints_today``
    is not tripped, giving other filters a chance to be exercised.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2024-01-01", periods=n)

    log_returns = rng.normal(loc=trend, scale=0.005, size=n)
    close = start_price * np.exp(np.cumsum(log_returns))

    spread = rng.uniform(0.002, 0.008, size=n) * close
    high = close + spread * rng.uniform(0.4, 0.8, size=n)
    low = close - spread * rng.uniform(0.4, 0.8, size=n)
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


def _make_ohlcv_with_date(
    n: int = 200,
    start_price: float = 10.0,
    trend: float = 0.001,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with an explicit 'date' column.

    Required for selector.select() which filters rows via ``df[df["date"] <= date]``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2024-01-01", periods=n)

    log_returns = rng.normal(loc=trend, scale=0.005, size=n)
    close = start_price * np.exp(np.cumsum(log_returns))

    spread = rng.uniform(0.002, 0.008, size=n) * close
    high = close + spread * rng.uniform(0.4, 0.8, size=n)
    low = close - spread * rng.uniform(0.4, 0.8, size=n)
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, size=n)
    volume = rng.integers(100_000, 5_000_000, size=n).astype(float)

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _empty_ohlcv_with_date() -> pd.DataFrame:
    """Return an empty DataFrame with the expected OHLCV + date columns."""
    return pd.DataFrame(
        columns=["date", "open", "high", "low", "close", "volume"]
    )


# Default B1_params for SuperB1Selector (use BBIKDJSelector defaults)
_DEFAULT_B1_PARAMS: dict = {}


# ===========================================================================
# SuperB1Selector
# ===========================================================================


class TestSuperB1SelectorInit:
    """Instantiation tests for SuperB1Selector."""

    def test_default_instantiation_requires_b1_params(self):
        """SuperB1Selector raises ValueError when B1_params is not provided."""
        with pytest.raises(ValueError, match="bbi_params"):
            SuperB1Selector()

    def test_instantiation_with_b1_params(self):
        """SuperB1Selector can be instantiated when B1_params is provided."""
        sel = SuperB1Selector(B1_params=_DEFAULT_B1_PARAMS)
        assert sel is not None

    def test_default_lookback_n(self):
        """Default lookback_n is 60."""
        sel = SuperB1Selector(B1_params=_DEFAULT_B1_PARAMS)
        assert sel.lookback_n == 60

    def test_default_close_vol_pct(self):
        """Default close_vol_pct is 0.05."""
        sel = SuperB1Selector(B1_params=_DEFAULT_B1_PARAMS)
        assert sel.close_vol_pct == pytest.approx(0.05)

    def test_default_price_drop_pct(self):
        """Default price_drop_pct is 0.03."""
        sel = SuperB1Selector(B1_params=_DEFAULT_B1_PARAMS)
        assert sel.price_drop_pct == pytest.approx(0.03)

    def test_default_j_threshold(self):
        """Default j_threshold is -5."""
        sel = SuperB1Selector(B1_params=_DEFAULT_B1_PARAMS)
        assert sel.j_threshold == -5

    def test_custom_lookback_n(self):
        """Custom lookback_n is stored correctly."""
        sel = SuperB1Selector(lookback_n=30, B1_params=_DEFAULT_B1_PARAMS)
        assert sel.lookback_n == 30

    def test_invalid_lookback_n_raises(self):
        """lookback_n < 2 raises ValueError."""
        with pytest.raises(ValueError):
            SuperB1Selector(lookback_n=1, B1_params=_DEFAULT_B1_PARAMS)

    def test_invalid_close_vol_pct_raises(self):
        """close_vol_pct outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError):
            SuperB1Selector(close_vol_pct=1.5, B1_params=_DEFAULT_B1_PARAMS)

    def test_invalid_price_drop_pct_raises(self):
        """price_drop_pct outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError):
            SuperB1Selector(price_drop_pct=1.5, B1_params=_DEFAULT_B1_PARAMS)

    def test_bbi_selector_is_created(self):
        """SuperB1Selector creates an internal bbi_selector attribute."""
        sel = SuperB1Selector(B1_params=_DEFAULT_B1_PARAMS)
        assert hasattr(sel, "bbi_selector")


class TestSuperB1SelectorEmptyData:
    """Empty / insufficient data tests for SuperB1Selector."""

    def test_select_returns_list(self):
        """select() always returns a list."""
        sel = SuperB1Selector(B1_params=_DEFAULT_B1_PARAMS)
        result = sel.select(pd.Timestamp("2025-01-01"), {})
        assert isinstance(result, list)

    def test_select_empty_dict_returns_empty_list(self):
        """select() with an empty data dict returns []."""
        sel = SuperB1Selector(B1_params=_DEFAULT_B1_PARAMS)
        result = sel.select(pd.Timestamp("2025-01-01"), {})
        assert result == []

    def test_select_date_before_all_data_returns_empty_list(self):
        """select() returns [] when query date precedes all rows."""
        sel = SuperB1Selector(B1_params=_DEFAULT_B1_PARAMS)
        df = _make_ohlcv_with_date(n=200)
        result = sel.select(pd.Timestamp("2000-01-01"), {"000001": df})
        assert result == []

    def test_select_insufficient_history_returns_empty_list(self):
        """select() returns [] when there are too few rows for the selector."""
        sel = SuperB1Selector(B1_params=_DEFAULT_B1_PARAMS)
        df = pd.DataFrame(
            {
                "date": pd.bdate_range("2024-01-01", periods=5),
                "open": [10.0] * 5,
                "high": [10.5] * 5,
                "low": [9.5] * 5,
                "close": [10.0] * 5,
                "volume": [1_000_000.0] * 5,
            }
        )
        result = sel.select(pd.Timestamp("2024-01-10"), {"A": df})
        assert result == []

    def test_passes_filters_empty_df_returns_false(self):
        """_passes_filters() returns False for an empty DataFrame."""
        sel = SuperB1Selector(B1_params=_DEFAULT_B1_PARAMS)
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        assert sel._passes_filters(empty) is False

    def test_select_result_elements_are_strings(self):
        """All elements in the select() result are strings."""
        sel = SuperB1Selector(B1_params=_DEFAULT_B1_PARAMS)
        df = _make_ohlcv_with_date(n=200)
        date = df["date"].iloc[-1]
        result = sel.select(date, {"000001": df, "000002": df})
        for code in result:
            assert isinstance(code, str)


# ===========================================================================
# BBIShortLongSelector
# ===========================================================================


class TestBBIShortLongSelectorInit:
    """Instantiation tests for BBIShortLongSelector."""

    def test_default_instantiation(self):
        """BBIShortLongSelector can be instantiated with no arguments."""
        sel = BBIShortLongSelector()
        assert sel is not None

    def test_default_n_short(self):
        """Default n_short is 3."""
        sel = BBIShortLongSelector()
        assert sel.n_short == 3

    def test_default_n_long(self):
        """Default n_long is 21."""
        sel = BBIShortLongSelector()
        assert sel.n_long == 21

    def test_default_m(self):
        """Default m is 3."""
        sel = BBIShortLongSelector()
        assert sel.m == 3

    def test_default_bbi_min_window(self):
        """Default bbi_min_window is 90."""
        sel = BBIShortLongSelector()
        assert sel.bbi_min_window == 90

    def test_default_max_window(self):
        """Default max_window is 150."""
        sel = BBIShortLongSelector()
        assert sel.max_window == 150

    def test_custom_n_short(self):
        """Custom n_short is stored correctly."""
        sel = BBIShortLongSelector(n_short=5)
        assert sel.n_short == 5

    def test_custom_n_long(self):
        """Custom n_long is stored correctly."""
        sel = BBIShortLongSelector(n_long=30)
        assert sel.n_long == 30

    def test_custom_m(self):
        """Custom m is stored correctly."""
        sel = BBIShortLongSelector(m=5)
        assert sel.m == 5

    def test_invalid_m_raises(self):
        """m < 2 raises ValueError."""
        with pytest.raises(ValueError):
            BBIShortLongSelector(m=1)

    def test_all_custom_params(self):
        """BBIShortLongSelector stores all custom params correctly."""
        sel = BBIShortLongSelector(
            n_short=5,
            n_long=30,
            m=4,
            bbi_min_window=60,
            max_window=120,
            bbi_q_threshold=0.10,
            upper_rsv_threshold=80,
            lower_rsv_threshold=20,
        )
        assert sel.n_short == 5
        assert sel.n_long == 30
        assert sel.m == 4
        assert sel.bbi_min_window == 60
        assert sel.max_window == 120
        assert sel.bbi_q_threshold == pytest.approx(0.10)
        assert sel.upper_rsv_threshold == pytest.approx(80)
        assert sel.lower_rsv_threshold == pytest.approx(20)


class TestBBIShortLongSelectorEmptyData:
    """Empty / insufficient data tests for BBIShortLongSelector."""

    def test_select_returns_list(self):
        """select() always returns a list."""
        sel = BBIShortLongSelector()
        result = sel.select(pd.Timestamp("2025-01-01"), {})
        assert isinstance(result, list)

    def test_select_empty_dict_returns_empty_list(self):
        """select() with an empty data dict returns []."""
        sel = BBIShortLongSelector()
        result = sel.select(pd.Timestamp("2025-01-01"), {})
        assert result == []

    def test_select_date_before_all_data_returns_empty_list(self):
        """select() returns [] when query date precedes all rows."""
        sel = BBIShortLongSelector()
        df = _make_ohlcv_with_date(n=200)
        result = sel.select(pd.Timestamp("2000-01-01"), {"000001": df})
        assert result == []

    def test_select_insufficient_history_returns_empty_list(self):
        """select() returns [] when there are too few rows for the selector."""
        sel = BBIShortLongSelector()
        df = pd.DataFrame(
            {
                "date": pd.bdate_range("2024-01-01", periods=5),
                "open": [10.0] * 5,
                "high": [10.5] * 5,
                "low": [9.5] * 5,
                "close": [10.0] * 5,
                "volume": [1_000_000.0] * 5,
            }
        )
        result = sel.select(pd.Timestamp("2024-01-10"), {"A": df})
        assert result == []

    def test_passes_filters_empty_df_returns_false(self):
        """_passes_filters() returns False for an empty DataFrame."""
        sel = BBIShortLongSelector()
        # Need close column for compute_bbi
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        assert sel._passes_filters(empty) is False

    def test_select_downtrend_returns_empty_list(self):
        """select() returns [] for stocks with a persistent downtrend (BBI falling)."""
        sel = BBIShortLongSelector()
        df_down = _make_ohlcv_with_date(n=300, trend=-0.01, seed=99)
        date = df_down["date"].iloc[-1]
        result = sel.select(date, {"A": df_down, "B": df_down})
        assert result == []

    def test_select_result_elements_are_strings(self):
        """All elements in the select() result are strings."""
        sel = BBIShortLongSelector()
        df = _make_ohlcv_with_date(n=300)
        date = df["date"].iloc[-1]
        result = sel.select(date, {"000001": df, "000002": df})
        for code in result:
            assert isinstance(code, str)


# ===========================================================================
# PeakKDJSelector
# ===========================================================================


class TestPeakKDJSelectorInit:
    """Instantiation tests for PeakKDJSelector."""

    def test_default_instantiation(self):
        """PeakKDJSelector can be instantiated with no arguments."""
        sel = PeakKDJSelector()
        assert sel is not None

    def test_default_j_threshold(self):
        """Default j_threshold is -5."""
        sel = PeakKDJSelector()
        assert sel.j_threshold == -5

    def test_default_max_window(self):
        """Default max_window is 90."""
        sel = PeakKDJSelector()
        assert sel.max_window == 90

    def test_default_fluc_threshold(self):
        """Default fluc_threshold is 0.03."""
        sel = PeakKDJSelector()
        assert sel.fluc_threshold == pytest.approx(0.03)

    def test_default_gap_threshold(self):
        """Default gap_threshold is 0.02."""
        sel = PeakKDJSelector()
        assert sel.gap_threshold == pytest.approx(0.02)

    def test_default_j_q_threshold(self):
        """Default j_q_threshold is 0.10."""
        sel = PeakKDJSelector()
        assert sel.j_q_threshold == pytest.approx(0.10)

    def test_custom_j_threshold(self):
        """Custom j_threshold is stored correctly."""
        sel = PeakKDJSelector(j_threshold=-10)
        assert sel.j_threshold == -10

    def test_custom_max_window(self):
        """Custom max_window is stored correctly."""
        sel = PeakKDJSelector(max_window=120)
        assert sel.max_window == 120

    def test_custom_fluc_threshold(self):
        """Custom fluc_threshold is stored correctly."""
        sel = PeakKDJSelector(fluc_threshold=0.05)
        assert sel.fluc_threshold == pytest.approx(0.05)

    def test_all_custom_params(self):
        """PeakKDJSelector stores all custom params correctly."""
        sel = PeakKDJSelector(
            j_threshold=-8,
            max_window=120,
            fluc_threshold=0.05,
            gap_threshold=0.03,
            j_q_threshold=0.15,
        )
        assert sel.j_threshold == -8
        assert sel.max_window == 120
        assert sel.fluc_threshold == pytest.approx(0.05)
        assert sel.gap_threshold == pytest.approx(0.03)
        assert sel.j_q_threshold == pytest.approx(0.15)


class TestPeakKDJSelectorEmptyData:
    """Empty / insufficient data tests for PeakKDJSelector."""

    def test_select_returns_list(self):
        """select() always returns a list."""
        sel = PeakKDJSelector()
        result = sel.select(pd.Timestamp("2025-01-01"), {})
        assert isinstance(result, list)

    def test_select_empty_dict_returns_empty_list(self):
        """select() with an empty data dict returns []."""
        sel = PeakKDJSelector()
        result = sel.select(pd.Timestamp("2025-01-01"), {})
        assert result == []

    def test_select_date_before_all_data_returns_empty_list(self):
        """select() returns [] when query date precedes all rows."""
        sel = PeakKDJSelector()
        df = _make_ohlcv_with_date(n=200)
        result = sel.select(pd.Timestamp("2000-01-01"), {"000001": df})
        assert result == []

    def test_select_insufficient_history_returns_empty_list(self):
        """select() returns [] when there are too few rows."""
        sel = PeakKDJSelector()
        df = pd.DataFrame(
            {
                "date": pd.bdate_range("2024-01-01", periods=5),
                "open": [10.0] * 5,
                "high": [10.5] * 5,
                "low": [9.5] * 5,
                "close": [10.0] * 5,
                "volume": [1_000_000.0] * 5,
            }
        )
        result = sel.select(pd.Timestamp("2024-01-10"), {"A": df})
        assert result == []

    def test_passes_filters_empty_df_returns_false(self):
        """_passes_filters() returns False for an empty DataFrame."""
        sel = PeakKDJSelector()
        empty = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        assert sel._passes_filters(empty) is False

    def test_select_result_elements_are_strings(self):
        """All elements in the select() result are strings."""
        sel = PeakKDJSelector()
        df = _make_ohlcv_with_date(n=200)
        date = df["date"].iloc[-1]
        result = sel.select(date, {"000001": df, "000002": df})
        for code in result:
            assert isinstance(code, str)


# ===========================================================================
# MA60CrossVolumeWaveSelector
# ===========================================================================


class TestMA60CrossVolumeWaveSelectorInit:
    """Instantiation tests for MA60CrossVolumeWaveSelector."""

    def test_default_instantiation(self):
        """MA60CrossVolumeWaveSelector can be instantiated with default arguments."""
        sel = MA60CrossVolumeWaveSelector()
        assert sel is not None

    def test_default_lookback_n(self):
        """Default lookback_n is 60."""
        sel = MA60CrossVolumeWaveSelector()
        assert sel.lookback_n == 60

    def test_default_vol_multiple(self):
        """Default vol_multiple is 1.5."""
        sel = MA60CrossVolumeWaveSelector()
        assert sel.vol_multiple == pytest.approx(1.5)

    def test_default_j_threshold(self):
        """Default j_threshold is -5.0."""
        sel = MA60CrossVolumeWaveSelector()
        assert sel.j_threshold == pytest.approx(-5.0)

    def test_default_j_q_threshold(self):
        """Default j_q_threshold is 0.10."""
        sel = MA60CrossVolumeWaveSelector()
        assert sel.j_q_threshold == pytest.approx(0.10)

    def test_default_ma60_slope_days(self):
        """Default ma60_slope_days is 5."""
        sel = MA60CrossVolumeWaveSelector()
        assert sel.ma60_slope_days == 5

    def test_default_max_window(self):
        """Default max_window is 120."""
        sel = MA60CrossVolumeWaveSelector()
        assert sel.max_window == 120

    def test_custom_lookback_n(self):
        """Custom lookback_n is stored correctly."""
        sel = MA60CrossVolumeWaveSelector(lookback_n=30)
        assert sel.lookback_n == 30

    def test_invalid_lookback_n_raises(self):
        """lookback_n < 2 raises ValueError."""
        with pytest.raises(ValueError):
            MA60CrossVolumeWaveSelector(lookback_n=1)

    def test_invalid_j_q_threshold_raises(self):
        """j_q_threshold outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError):
            MA60CrossVolumeWaveSelector(j_q_threshold=1.5)

    def test_invalid_ma60_slope_days_raises(self):
        """ma60_slope_days < 2 raises ValueError."""
        with pytest.raises(ValueError):
            MA60CrossVolumeWaveSelector(ma60_slope_days=1)

    def test_all_custom_params(self):
        """MA60CrossVolumeWaveSelector stores all custom params correctly."""
        sel = MA60CrossVolumeWaveSelector(
            lookback_n=30,
            vol_multiple=2.0,
            j_threshold=-10.0,
            j_q_threshold=0.05,
            ma60_slope_days=10,
            max_window=90,
        )
        assert sel.lookback_n == 30
        assert sel.vol_multiple == pytest.approx(2.0)
        assert sel.j_threshold == pytest.approx(-10.0)
        assert sel.j_q_threshold == pytest.approx(0.05)
        assert sel.ma60_slope_days == 10
        assert sel.max_window == 90


class TestMA60CrossVolumeWaveSelectorEmptyData:
    """Empty / insufficient data tests for MA60CrossVolumeWaveSelector."""

    def test_select_returns_list(self):
        """select() always returns a list."""
        sel = MA60CrossVolumeWaveSelector()
        result = sel.select(pd.Timestamp("2025-01-01"), {})
        assert isinstance(result, list)

    def test_select_empty_dict_returns_empty_list(self):
        """select() with an empty data dict returns []."""
        sel = MA60CrossVolumeWaveSelector()
        result = sel.select(pd.Timestamp("2025-01-01"), {})
        assert result == []

    def test_select_date_before_all_data_returns_empty_list(self):
        """select() returns [] when query date precedes all rows."""
        sel = MA60CrossVolumeWaveSelector()
        df = _make_ohlcv_with_date(n=300)
        result = sel.select(pd.Timestamp("2000-01-01"), {"000001": df})
        assert result == []

    def test_select_insufficient_history_returns_empty_list(self):
        """select() returns [] when there are too few rows."""
        sel = MA60CrossVolumeWaveSelector()
        df = pd.DataFrame(
            {
                "date": pd.bdate_range("2024-01-01", periods=5),
                "open": [10.0] * 5,
                "high": [10.5] * 5,
                "low": [9.5] * 5,
                "close": [10.0] * 5,
                "volume": [1_000_000.0] * 5,
            }
        )
        result = sel.select(pd.Timestamp("2024-01-10"), {"A": df})
        assert result == []

    def test_passes_filters_empty_df_returns_false(self):
        """_passes_filters() returns False for an empty DataFrame."""
        sel = MA60CrossVolumeWaveSelector()
        empty = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        assert sel._passes_filters(empty) is False

    def test_select_result_elements_are_strings(self):
        """All elements in the select() result are strings."""
        sel = MA60CrossVolumeWaveSelector()
        df = _make_ohlcv_with_date(n=300)
        date = df["date"].iloc[-1]
        result = sel.select(date, {"000001": df, "000002": df})
        for code in result:
            assert isinstance(code, str)


# ===========================================================================
# BigBullishVolumeSelector
# ===========================================================================


class TestBigBullishVolumeSelectorInit:
    """Instantiation tests for BigBullishVolumeSelector."""

    def test_default_instantiation(self):
        """BigBullishVolumeSelector can be instantiated with default arguments."""
        sel = BigBullishVolumeSelector()
        assert sel is not None

    def test_default_up_pct_threshold(self):
        """Default up_pct_threshold is 0.04."""
        sel = BigBullishVolumeSelector()
        assert sel.up_pct_threshold == pytest.approx(0.04)

    def test_default_upper_wick_pct_max(self):
        """Default upper_wick_pct_max is 0.5."""
        sel = BigBullishVolumeSelector()
        assert sel.upper_wick_pct_max == pytest.approx(0.5)

    def test_default_vol_lookback_n(self):
        """Default vol_lookback_n is 20."""
        sel = BigBullishVolumeSelector()
        assert sel.vol_lookback_n == 20

    def test_default_vol_multiple(self):
        """Default vol_multiple is 1.5."""
        sel = BigBullishVolumeSelector()
        assert sel.vol_multiple == pytest.approx(1.5)

    def test_default_require_bullish_close(self):
        """Default require_bullish_close is True."""
        sel = BigBullishVolumeSelector()
        assert sel.require_bullish_close is True

    def test_default_ignore_zero_volume(self):
        """Default ignore_zero_volume is True."""
        sel = BigBullishVolumeSelector()
        assert sel.ignore_zero_volume is True

    def test_default_close_lt_zxdq_mult(self):
        """Default close_lt_zxdq_mult is 1.0."""
        sel = BigBullishVolumeSelector()
        assert sel.close_lt_zxdq_mult == pytest.approx(1.0)

    def test_custom_up_pct_threshold(self):
        """Custom up_pct_threshold is stored correctly."""
        sel = BigBullishVolumeSelector(up_pct_threshold=0.06)
        assert sel.up_pct_threshold == pytest.approx(0.06)

    def test_custom_vol_multiple(self):
        """Custom vol_multiple is stored correctly."""
        sel = BigBullishVolumeSelector(vol_multiple=2.0)
        assert sel.vol_multiple == pytest.approx(2.0)

    def test_invalid_up_pct_threshold_raises(self):
        """up_pct_threshold <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            BigBullishVolumeSelector(up_pct_threshold=0.0)

    def test_invalid_upper_wick_pct_max_raises(self):
        """upper_wick_pct_max < 0 raises ValueError."""
        with pytest.raises(ValueError):
            BigBullishVolumeSelector(upper_wick_pct_max=-0.1)

    def test_invalid_vol_lookback_n_raises(self):
        """vol_lookback_n < 1 raises ValueError."""
        with pytest.raises(ValueError):
            BigBullishVolumeSelector(vol_lookback_n=0)

    def test_invalid_vol_multiple_raises(self):
        """vol_multiple <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            BigBullishVolumeSelector(vol_multiple=0.0)

    def test_invalid_close_lt_zxdq_mult_raises(self):
        """close_lt_zxdq_mult <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            BigBullishVolumeSelector(close_lt_zxdq_mult=0.0)

    def test_all_custom_params(self):
        """BigBullishVolumeSelector stores all custom params correctly."""
        sel = BigBullishVolumeSelector(
            up_pct_threshold=0.05,
            upper_wick_pct_max=0.3,
            vol_lookback_n=10,
            vol_multiple=2.0,
            min_history=15,
            require_bullish_close=False,
            ignore_zero_volume=False,
            close_lt_zxdq_mult=1.02,
        )
        assert sel.up_pct_threshold == pytest.approx(0.05)
        assert sel.upper_wick_pct_max == pytest.approx(0.3)
        assert sel.vol_lookback_n == 10
        assert sel.vol_multiple == pytest.approx(2.0)
        assert sel.min_history == 15
        assert sel.require_bullish_close is False
        assert sel.ignore_zero_volume is False
        assert sel.close_lt_zxdq_mult == pytest.approx(1.02)


class TestBigBullishVolumeSelectorEmptyData:
    """Empty / insufficient data tests for BigBullishVolumeSelector."""

    def test_select_returns_list(self):
        """select() always returns a list."""
        sel = BigBullishVolumeSelector()
        result = sel.select(pd.Timestamp("2025-01-01"), {})
        assert isinstance(result, list)

    def test_select_empty_dict_returns_empty_list(self):
        """select() with an empty data dict returns []."""
        sel = BigBullishVolumeSelector()
        result = sel.select(pd.Timestamp("2025-01-01"), {})
        assert result == []

    def test_select_date_before_all_data_returns_empty_list(self):
        """select() returns [] when query date precedes all rows."""
        sel = BigBullishVolumeSelector()
        df = _make_ohlcv_with_date(n=100)
        result = sel.select(pd.Timestamp("2000-01-01"), {"000001": df})
        assert result == []

    def test_select_insufficient_history_returns_empty_list(self):
        """select() returns [] when there are too few rows."""
        sel = BigBullishVolumeSelector()
        df = pd.DataFrame(
            {
                "date": pd.bdate_range("2024-01-01", periods=3),
                "open": [10.0] * 3,
                "high": [10.5] * 3,
                "low": [9.5] * 3,
                "close": [10.0] * 3,
                "volume": [1_000_000.0] * 3,
            }
        )
        result = sel.select(pd.Timestamp("2024-01-10"), {"A": df})
        assert result == []

    def test_passes_filters_empty_df_returns_false(self):
        """_passes_filters() returns False for an empty DataFrame."""
        sel = BigBullishVolumeSelector()
        empty = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        assert sel._passes_filters(empty) is False

    def test_passes_filters_none_returns_false(self):
        """_passes_filters() returns False when passed None."""
        sel = BigBullishVolumeSelector()
        assert sel._passes_filters(None) is False

    def test_select_none_df_entry_skipped(self):
        """select() skips entries whose DataFrame is None."""
        sel = BigBullishVolumeSelector()
        date = pd.Timestamp("2025-01-01")
        result = sel.select(date, {"000001": None})
        assert "000001" not in result

    def test_select_empty_df_entry_skipped(self):
        """select() skips entries with an empty DataFrame."""
        sel = BigBullishVolumeSelector()
        date = pd.Timestamp("2025-01-01")
        result = sel.select(date, {"000001": _empty_ohlcv_with_date()})
        assert "000001" not in result

    def test_select_result_elements_are_strings(self):
        """All elements in the select() result are strings."""
        sel = BigBullishVolumeSelector()
        df = _make_ohlcv_with_date(n=100)
        date = df["date"].iloc[-1]
        result = sel.select(date, {"000001": df, "000002": df})
        for code in result:
            assert isinstance(code, str)
