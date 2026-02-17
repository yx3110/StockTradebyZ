"""Unit tests for peak and ZX-line utility functions in Selector.py.

Covers:
  - _find_peaks    : single peak, multiple peaks, no peaks, invalid column
  - compute_zx_lines : ZXDQ (double-EMA) and ZXDKX (4-MA average) accuracy
"""

import numpy as np
import pandas as pd
import pytest

from Selector import _find_peaks, compute_zx_lines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_from_close(close_values: list[float]) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of close prices.

    high = close + 0.5, low = close - 0.5, open = close, volume = 1000.
    """
    close = pd.Series(close_values, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000.0,
        }
    )


def _make_long_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate a longer OHLCV DataFrame for compute_zx_lines tests."""
    rng = np.random.default_rng(seed)
    close = 10.0 * np.exp(np.cumsum(rng.normal(0.001, 0.01, size=n)))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000.0,
        }
    )


# ---------------------------------------------------------------------------
# _find_peaks
# ---------------------------------------------------------------------------


class TestFindPeaks:
    """Tests for _find_peaks utility function."""

    # --- single peak ---

    def test_single_peak_returns_one_row(self):
        """A series with one clear peak returns a DataFrame with exactly one row."""
        # Close values form an arch: peak at index 4
        close_values = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        df = _make_ohlcv_from_close(close_values)
        # Use 'close' column explicitly so the peak at index 4 is detected
        result = _find_peaks(df, column="close")
        assert len(result) == 1

    def test_single_peak_has_is_peak_column(self):
        """Result DataFrame contains an 'is_peak' column set to True."""
        close_values = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        df = _make_ohlcv_from_close(close_values)
        result = _find_peaks(df, column="close")
        assert "is_peak" in result.columns
        assert result["is_peak"].all()

    def test_single_peak_correct_index(self):
        """The peak is detected at the correct positional index."""
        close_values = [1.0, 2.0, 3.0, 5.0, 3.0, 2.0, 1.0]
        df = _make_ohlcv_from_close(close_values)
        result = _find_peaks(df, column="close")
        # Peak is at position 3 (0-based)
        assert len(result) == 1
        assert result.index[0] == 3

    def test_single_peak_preserves_original_columns(self):
        """The result DataFrame keeps the original OHLCV columns."""
        close_values = [1.0, 3.0, 5.0, 3.0, 1.0]
        df = _make_ohlcv_from_close(close_values)
        result = _find_peaks(df, column="close")
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    # --- multiple peaks ---

    def test_multiple_peaks_correct_count(self):
        """A series with two clear peaks returns two rows."""
        # W-shape with two peaks at indices 2 and 6
        close_values = [1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0]
        df = _make_ohlcv_from_close(close_values)
        result = _find_peaks(df, column="close")
        assert len(result) == 2

    def test_multiple_peaks_indices(self):
        """Multiple peaks are detected at the expected positions."""
        close_values = [1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0]
        df = _make_ohlcv_from_close(close_values)
        result = _find_peaks(df, column="close")
        detected_indices = list(result.index)
        assert 2 in detected_indices
        assert 6 in detected_indices

    def test_multiple_peaks_all_marked_is_peak(self):
        """All rows in a multi-peak result have is_peak == True."""
        close_values = [1.0, 4.0, 1.0, 4.0, 1.0, 4.0, 1.0]
        df = _make_ohlcv_from_close(close_values)
        result = _find_peaks(df, column="close")
        assert len(result) > 1
        assert result["is_peak"].all()

    def test_distance_parameter_limits_peaks(self):
        """The distance parameter suppresses peaks that are too close together."""
        # Three peaks at indices 1, 3, 5
        close_values = [0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0]
        df = _make_ohlcv_from_close(close_values)
        result_no_dist = _find_peaks(df, column="close")
        result_with_dist = _find_peaks(df, column="close", distance=3)
        # With distance=3 fewer peaks should be returned
        assert len(result_with_dist) < len(result_no_dist)

    # --- no peaks ---

    def test_monotonic_increasing_has_no_peaks(self):
        """A strictly monotonically increasing series has no interior peaks."""
        close_values = [float(i) for i in range(1, 10)]
        df = _make_ohlcv_from_close(close_values)
        result = _find_peaks(df, column="close")
        assert len(result) == 0

    def test_monotonic_decreasing_has_no_peaks(self):
        """A strictly monotonically decreasing series has no peaks."""
        close_values = [float(i) for i in range(10, 0, -1)]
        df = _make_ohlcv_from_close(close_values)
        result = _find_peaks(df, column="close")
        assert len(result) == 0

    def test_flat_series_has_no_peaks(self):
        """A perfectly flat series has no peaks."""
        close_values = [5.0] * 10
        df = _make_ohlcv_from_close(close_values)
        result = _find_peaks(df, column="close")
        assert len(result) == 0

    def test_no_peaks_returns_empty_dataframe(self):
        """When no peaks are found, the returned DataFrame is empty."""
        close_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        df = _make_ohlcv_from_close(close_values)
        result = _find_peaks(df, column="close")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    # --- invalid column ---

    def test_invalid_column_raises_key_error(self):
        """Requesting a non-existent column raises KeyError."""
        close_values = [1.0, 3.0, 2.0, 4.0, 1.0]
        df = _make_ohlcv_from_close(close_values)
        with pytest.raises(KeyError):
            _find_peaks(df, column="nonexistent_column")

    def test_invalid_column_error_message_contains_column_name(self):
        """The KeyError message includes the requested column name."""
        close_values = [1.0, 3.0, 2.0, 4.0, 1.0]
        df = _make_ohlcv_from_close(close_values)
        with pytest.raises(KeyError, match="bad_col"):
            _find_peaks(df, column="bad_col")

    def test_default_column_is_high(self):
        """Default column for peak detection is 'high'."""
        # Build df where 'high' has a clear peak but 'close' does not
        df = pd.DataFrame(
            {
                "open": [1.0, 1.0, 1.0, 1.0, 1.0],
                "high": [1.0, 2.0, 5.0, 2.0, 1.0],
                "low": [0.5, 0.5, 0.5, 0.5, 0.5],
                "close": [1.0, 1.2, 1.3, 1.4, 1.5],  # monotone, no peaks
                "volume": [1000.0] * 5,
            }
        )
        result = _find_peaks(df)  # uses column="high" by default
        assert len(result) == 1

    # --- prominence / height filtering ---

    def test_prominence_filter_removes_small_peaks(self):
        """High prominence requirement eliminates small peaks."""
        # One tall peak (amplitude 10) and one small peak (amplitude 1)
        close_values = [0.0, 1.0, 0.0, 0.0, 10.0, 0.0, 0.0, 1.0, 0.0]
        df = _make_ohlcv_from_close(close_values)
        result_all = _find_peaks(df, column="close")
        result_filtered = _find_peaks(df, column="close", prominence=5.0)
        assert len(result_filtered) < len(result_all)
        assert len(result_filtered) == 1


# ---------------------------------------------------------------------------
# compute_zx_lines
# ---------------------------------------------------------------------------


class TestComputeZxLines:
    """Tests for compute_zx_lines indicator function."""

    # --- return types and structure ---

    def test_returns_tuple_of_two_series(self):
        """compute_zx_lines returns a tuple of two pd.Series."""
        df = _make_long_ohlcv(n=150)
        result = compute_zx_lines(df)
        assert isinstance(result, tuple)
        assert len(result) == 2
        zxdq, zxdkx = result
        assert isinstance(zxdq, pd.Series)
        assert isinstance(zxdkx, pd.Series)

    def test_output_length_matches_input(self):
        """Both output Series have the same length as the input DataFrame."""
        n = 150
        df = _make_long_ohlcv(n=n)
        zxdq, zxdkx = compute_zx_lines(df)
        assert len(zxdq) == n
        assert len(zxdkx) == n

    # --- ZXDQ (double EMA) ---

    def test_zxdq_no_nan_values(self):
        """ZXDQ (EMA of EMA) is defined from the first row — no NaN values."""
        df = _make_long_ohlcv(n=50)
        zxdq, _ = compute_zx_lines(df)
        assert zxdq.notna().all()

    def test_zxdq_formula_correctness(self):
        """ZXDQ must equal EMA(EMA(close, 10), 10) exactly."""
        df = _make_long_ohlcv(n=100)
        zxdq, _ = compute_zx_lines(df)

        close = df["close"].astype(float)
        expected_zxdq = (
            close.ewm(span=10, adjust=False).mean()
            .ewm(span=10, adjust=False).mean()
        )
        pd.testing.assert_series_equal(zxdq, expected_zxdq, check_names=False)

    def test_zxdq_smooths_prices(self):
        """ZXDQ (double EMA) should be smoother than the raw close prices."""
        df = _make_long_ohlcv(n=100)
        zxdq, _ = compute_zx_lines(df)
        close_std = df["close"].std()
        zxdq_std = zxdq.std()
        # Double EMA smooths out volatility, so its std < raw close std
        assert zxdq_std < close_std

    def test_zxdq_custom_params_still_double_ema(self):
        """With custom m1-m4 params, ZXDQ formula is unchanged (always span=10)."""
        df = _make_long_ohlcv(n=100)
        zxdq_default, _ = compute_zx_lines(df)
        zxdq_custom, _ = compute_zx_lines(df, m1=5, m2=10, m3=20, m4=40)
        # ZXDQ does not depend on m1-m4, so both should be identical
        pd.testing.assert_series_equal(zxdq_default, zxdq_custom, check_names=False)

    # --- ZXDKX (4-MA average) ---

    def test_zxdkx_nan_before_warmup(self):
        """ZXDKX is NaN for the first (m4 - 1) rows where MA(m4) is undefined."""
        n = 200
        m4 = 114
        df = _make_long_ohlcv(n=n)
        _, zxdkx = compute_zx_lines(df)
        # First (m4 - 1) rows must be NaN because MA(114) requires 114 points
        assert zxdkx.iloc[: m4 - 1].isna().all()

    def test_zxdkx_valid_after_warmup(self):
        """ZXDKX produces valid (non-NaN) values once all 4 MAs are available."""
        n = 200
        m4 = 114
        df = _make_long_ohlcv(n=n)
        _, zxdkx = compute_zx_lines(df)
        assert zxdkx.iloc[m4 - 1 :].notna().all()

    def test_zxdkx_formula_correctness(self):
        """ZXDKX must equal (MA14 + MA28 + MA57 + MA114) / 4 for last row."""
        n = 200
        df = _make_long_ohlcv(n=n)
        _, zxdkx = compute_zx_lines(df)

        close = df["close"].astype(float)
        ma14 = close.rolling(window=14, min_periods=14).mean()
        ma28 = close.rolling(window=28, min_periods=28).mean()
        ma57 = close.rolling(window=57, min_periods=57).mean()
        ma114 = close.rolling(window=114, min_periods=114).mean()
        expected = (ma14 + ma28 + ma57 + ma114) / 4.0

        pd.testing.assert_series_equal(
            zxdkx.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_zxdkx_custom_ma_periods(self):
        """Custom m1-m4 parameters change the ZXDKX calculation accordingly."""
        n = 200
        df = _make_long_ohlcv(n=n)
        _, zxdkx_default = compute_zx_lines(df)
        _, zxdkx_custom = compute_zx_lines(df, m1=5, m2=10, m3=20, m4=40)
        # Different MA windows produce different results
        assert not zxdkx_default.equals(zxdkx_custom)

    def test_zxdkx_last_value_is_average_of_four_mas(self):
        """Spot-check: ZXDKX last value equals the mean of four MA last values."""
        n = 200
        df = _make_long_ohlcv(n=n)
        _, zxdkx = compute_zx_lines(df)

        close = df["close"].astype(float)
        last_ma14 = close.rolling(14, min_periods=14).mean().iloc[-1]
        last_ma28 = close.rolling(28, min_periods=28).mean().iloc[-1]
        last_ma57 = close.rolling(57, min_periods=57).mean().iloc[-1]
        last_ma114 = close.rolling(114, min_periods=114).mean().iloc[-1]
        expected_last = (last_ma14 + last_ma28 + last_ma57 + last_ma114) / 4.0

        assert zxdkx.iloc[-1] == pytest.approx(expected_last, rel=1e-9)

    def test_zxdkx_nan_count_equals_m4_minus_one(self):
        """The number of leading NaN values in ZXDKX equals m4 - 1."""
        n = 200
        m4 = 114
        df = _make_long_ohlcv(n=n)
        _, zxdkx = compute_zx_lines(df)
        nan_count = zxdkx.isna().sum()
        assert nan_count == m4 - 1
