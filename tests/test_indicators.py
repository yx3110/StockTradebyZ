"""Unit tests for indicator functions in Selector.py.

Covers:
  - compute_kdj  : normal data, empty DataFrame, single-row edge case
  - compute_bbi  : basic calculation, rolling window consistency
  - compute_rsv  : calculation correctness, value range
  - compute_dif  : MACD DIF correctness
"""

import numpy as np
import pandas as pd
import pytest

from Selector import compute_bbi, compute_dif, compute_kdj, compute_rsv


# ---------------------------------------------------------------------------
# compute_kdj
# ---------------------------------------------------------------------------


class TestComputeKdj:
    """Tests for compute_kdj indicator function."""

    def test_normal_data_returns_columns(self, ohlcv_df):
        """compute_kdj adds K, D, J columns to the DataFrame."""
        result = compute_kdj(ohlcv_df)
        assert "K" in result.columns
        assert "D" in result.columns
        assert "J" in result.columns

    def test_normal_data_shape_preserved(self, ohlcv_df):
        """Output DataFrame has same number of rows as input."""
        result = compute_kdj(ohlcv_df)
        assert len(result) == len(ohlcv_df)

    def test_first_k_d_equal_50(self, ohlcv_df):
        """K[0] and D[0] are both initialised to 50 per the KDJ formula."""
        result = compute_kdj(ohlcv_df)
        assert result["K"].iloc[0] == pytest.approx(50.0)
        assert result["D"].iloc[0] == pytest.approx(50.0)

    def test_j_equals_3k_minus_2d(self, ohlcv_df):
        """J = 3*K - 2*D must hold for every row."""
        result = compute_kdj(ohlcv_df)
        expected_j = 3 * result["K"] - 2 * result["D"]
        pd.testing.assert_series_equal(
            result["J"].reset_index(drop=True),
            expected_j.reset_index(drop=True),
            check_names=False,
        )

    def test_no_nan_values_normal_data(self, ohlcv_df):
        """K, D, J columns should be fully populated for normal data."""
        result = compute_kdj(ohlcv_df)
        assert result["K"].notna().all()
        assert result["D"].notna().all()
        assert result["J"].notna().all()

    # --- empty DataFrame edge case ---

    def test_empty_dataframe_returns_kd_j_columns(self, ohlcv_empty_df):
        """compute_kdj on empty DataFrame returns a DataFrame with K, D, J columns."""
        result = compute_kdj(ohlcv_empty_df)
        assert "K" in result.columns
        assert "D" in result.columns
        assert "J" in result.columns

    def test_empty_dataframe_has_zero_rows(self, ohlcv_empty_df):
        """compute_kdj on empty DataFrame has zero rows."""
        result = compute_kdj(ohlcv_empty_df)
        assert len(result) == 0

    # --- single-row edge case ---

    def test_single_row_k_d_equal_50(self, ohlcv_single_row_df):
        """Single-row DataFrame: K and D are initialised to 50."""
        result = compute_kdj(ohlcv_single_row_df)
        assert result["K"].iloc[0] == pytest.approx(50.0)
        assert result["D"].iloc[0] == pytest.approx(50.0)

    def test_single_row_j_equals_50(self, ohlcv_single_row_df):
        """Single-row DataFrame: J = 3*50 - 2*50 = 50."""
        result = compute_kdj(ohlcv_single_row_df)
        assert result["J"].iloc[0] == pytest.approx(50.0)

    def test_single_row_has_one_row(self, ohlcv_single_row_df):
        """Single-row input produces single-row output."""
        result = compute_kdj(ohlcv_single_row_df)
        assert len(result) == 1

    # --- parameter sensitivity ---

    def test_custom_n_produces_different_results(self, ohlcv_df):
        """compute_kdj with n=9 and n=14 produce different K values."""
        result_9 = compute_kdj(ohlcv_df, n=9)
        result_14 = compute_kdj(ohlcv_df, n=14)
        assert not result_9["K"].equals(result_14["K"])


# ---------------------------------------------------------------------------
# compute_bbi
# ---------------------------------------------------------------------------


class TestComputeBbi:
    """Tests for compute_bbi (Bull and Bear Index) indicator function."""

    def test_returns_series(self, ohlcv_df):
        """compute_bbi returns a pd.Series."""
        result = compute_bbi(ohlcv_df)
        assert isinstance(result, pd.Series)

    def test_length_matches_input(self, ohlcv_df):
        """Output Series length equals input DataFrame length."""
        result = compute_bbi(ohlcv_df)
        assert len(result) == len(ohlcv_df)

    def test_early_rows_are_nan(self, ohlcv_df):
        """Rows before the 24-bar warm-up (indices 0-22) must be NaN.

        BBI = (MA3 + MA6 + MA12 + MA24) / 4.  MA24 dominates; its first
        valid value is at index 23 (0-based), so indices 0-22 are NaN.
        """
        result = compute_bbi(ohlcv_df)
        assert result.iloc[:23].isna().all()

    def test_values_after_warmup_not_nan(self, ohlcv_df):
        """All rows from index 23 onward should produce valid BBI values."""
        result = compute_bbi(ohlcv_df)
        assert result.iloc[23:].notna().all()

    def test_basic_calculation_known_data(self):
        """Verify BBI equals (MA3+MA6+MA12+MA24)/4 for simple sequential data."""
        close = pd.Series(float(i) for i in range(1, 31))
        df = pd.DataFrame({"close": close})

        result = compute_bbi(df)

        ma3 = close.rolling(3).mean().iloc[-1]
        ma6 = close.rolling(6).mean().iloc[-1]
        ma12 = close.rolling(12).mean().iloc[-1]
        ma24 = close.rolling(24).mean().iloc[-1]
        expected = (ma3 + ma6 + ma12 + ma24) / 4

        assert result.iloc[-1] == pytest.approx(expected)

    def test_rolling_window_consistency(self, ohlcv_long_df):
        """Manually verify BBI at an interior index against expected calculation."""
        result = compute_bbi(ohlcv_long_df)
        close = ohlcv_long_df["close"]

        # Index 50 is well past the 24-bar warm-up
        idx = 50
        ma3 = close.iloc[idx - 2 : idx + 1].mean()
        ma6 = close.iloc[idx - 5 : idx + 1].mean()
        ma12 = close.iloc[idx - 11 : idx + 1].mean()
        ma24 = close.iloc[idx - 23 : idx + 1].mean()
        expected = (ma3 + ma6 + ma12 + ma24) / 4

        assert result.iloc[idx] == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# compute_rsv
# ---------------------------------------------------------------------------


class TestComputeRsv:
    """Tests for compute_rsv indicator function."""

    def test_returns_series(self, ohlcv_df):
        """compute_rsv returns a pd.Series."""
        result = compute_rsv(ohlcv_df, n=9)
        assert isinstance(result, pd.Series)

    def test_length_matches_input(self, ohlcv_df):
        """Output Series length equals input DataFrame length."""
        result = compute_rsv(ohlcv_df, n=9)
        assert len(result) == len(ohlcv_df)

    def test_values_between_0_and_100(self, ohlcv_df):
        """RSV values must lie within [0, 100] for typical OHLCV data."""
        result = compute_rsv(ohlcv_df, n=9)
        assert (result >= 0).all()
        assert (result <= 100 + 1e-6).all()

    def test_calculation_correctness(self):
        """Validate RSV formula against manual computation for known data."""
        n = 3
        close = pd.Series([10.0, 12.0, 11.0, 13.0, 15.0])
        low = pd.Series([9.0, 11.0, 10.0, 12.0, 14.0])
        df = pd.DataFrame({"close": close, "low": low})

        result = compute_rsv(df, n=n)

        low_n = low.rolling(window=n, min_periods=1).min().iloc[-1]
        high_close_n = close.rolling(window=n, min_periods=1).max().iloc[-1]
        expected = (close.iloc[-1] - low_n) / (high_close_n - low_n + 1e-9) * 100.0

        assert result.iloc[-1] == pytest.approx(expected)

    def test_flat_data_rsv_near_zero(self):
        """When close equals low everywhere, RSV numerator is ~0 → RSV ≈ 0."""
        flat_close = pd.Series([10.0] * 20)
        flat_low = pd.Series([10.0] * 20)
        df = pd.DataFrame({"close": flat_close, "low": flat_low})

        result = compute_rsv(df, n=9)
        assert (result.abs() < 1e-3).all()

    def test_n_parameter_changes_result(self, ohlcv_df):
        """Different n values yield different RSV series."""
        rsv_9 = compute_rsv(ohlcv_df, n=9)
        rsv_14 = compute_rsv(ohlcv_df, n=14)
        assert not rsv_9.equals(rsv_14)

    def test_no_nan_values(self, ohlcv_df):
        """compute_rsv with min_periods=1 should produce no NaN values."""
        result = compute_rsv(ohlcv_df, n=9)
        assert result.notna().all()


# ---------------------------------------------------------------------------
# compute_dif
# ---------------------------------------------------------------------------


class TestComputeDif:
    """Tests for compute_dif (MACD DIF line) indicator function."""

    def test_returns_series(self, ohlcv_df):
        """compute_dif returns a pd.Series."""
        result = compute_dif(ohlcv_df)
        assert isinstance(result, pd.Series)

    def test_length_matches_input(self, ohlcv_df):
        """Output Series length equals input DataFrame length."""
        result = compute_dif(ohlcv_df)
        assert len(result) == len(ohlcv_df)

    def test_no_nan_values(self, ohlcv_df):
        """EWM starts from the first row so DIF should have no NaN values."""
        result = compute_dif(ohlcv_df)
        assert result.notna().all()

    def test_dif_formula_correctness(self, ohlcv_df):
        """DIF must equal EMA(close, fast) - EMA(close, slow) exactly."""
        fast, slow = 12, 26
        result = compute_dif(ohlcv_df, fast=fast, slow=slow)

        ema_fast = ohlcv_df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = ohlcv_df["close"].ewm(span=slow, adjust=False).mean()
        expected = ema_fast - ema_slow

        pd.testing.assert_series_equal(result, expected)

    def test_uptrend_dif_positive_late_rows(self, ohlcv_uptrend_df):
        """In a strong uptrend the fast EMA converges above the slow EMA."""
        result = compute_dif(ohlcv_uptrend_df)
        assert result.iloc[-1] > 0

    def test_custom_fast_slow_params_differ(self, ohlcv_df):
        """Different fast/slow parameters produce different DIF series."""
        result_default = compute_dif(ohlcv_df, fast=12, slow=26)
        result_custom = compute_dif(ohlcv_df, fast=5, slow=20)
        assert not result_default.equals(result_custom)

    def test_default_params_are_12_26(self, ohlcv_df):
        """compute_dif() with no args uses fast=12, slow=26."""
        result_default = compute_dif(ohlcv_df)
        result_explicit = compute_dif(ohlcv_df, fast=12, slow=26)
        pd.testing.assert_series_equal(result_default, result_explicit)
