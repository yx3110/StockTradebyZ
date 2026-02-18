"""Unit tests for trend utility functions in Selector.py.

Covers:
  - bbi_deriv_uptrend         : uptrend/downtrend/flat scenarios, q_threshold edge
                                 cases, min_window enforcement
  - last_valid_ma_cross_up    : cross found, no cross, lookback limit
  - passes_day_constraints_today : pass/fail scenarios
  - zx_condition_at_positions : condition combinations, NaN long line, boundary positions
"""

import numpy as np
import pandas as pd
import pytest

from Selector import (
    bbi_deriv_uptrend,
    compute_bbi,
    last_valid_ma_cross_up,
    passes_day_constraints_today,
    zx_condition_at_positions,
)

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_bbi(values) -> pd.Series:
    """Wrap a list of floats into a BBI pd.Series."""
    return pd.Series(values, dtype=float)


def _make_df_with_close(close_values) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame for zx_condition_at_positions tests.

    Only the ``close`` column is used by ``zx_condition_at_positions`` /
    ``compute_zx_lines``; the other columns are included so the DataFrame
    is structurally consistent with project conventions.
    """
    n = len(close_values)
    dates = pd.bdate_range(start="2020-01-01", periods=n)
    close = np.array(close_values, dtype=float)
    return pd.DataFrame(
        {
            "open": close - 0.05,
            "high": close + 0.10,
            "low": close - 0.10,
            "close": close,
            "volume": np.ones(n) * 1_000_000.0,
        },
        index=dates,
    )


def _make_two_day_df(
    prev_close: float,
    today_close: float,
    today_high: float,
    today_low: float,
) -> pd.DataFrame:
    """Build a two-row OHLCV DataFrame for passes_day_constraints_today tests."""
    return pd.DataFrame(
        {
            "open": [prev_close, today_close - 0.01],
            "high": [prev_close + 0.10, today_high],
            "low": [prev_close - 0.10, today_low],
            "close": [prev_close, today_close],
            "volume": [1_000_000.0, 1_000_000.0],
        }
    )


# ---------------------------------------------------------------------------
# bbi_deriv_uptrend
# ---------------------------------------------------------------------------


class TestBbiDerivUptrend:
    """Tests for bbi_deriv_uptrend."""

    # --- uptrend scenarios ---

    def test_strict_uptrend_q0_returns_true(self):
        """Strictly increasing BBI with q_threshold=0 should return True."""
        bbi = _make_bbi(list(range(1, 31)))  # 1, 2, ..., 30
        assert bbi_deriv_uptrend(bbi, min_window=5) is True

    def test_uptrend_default_q_threshold(self):
        """Default q_threshold=0 detects a monotone uptrend."""
        bbi = _make_bbi([float(i) for i in range(10, 40)])
        assert bbi_deriv_uptrend(bbi, min_window=10) is True

    def test_uptrend_fixture(self, ohlcv_uptrend_df):
        """BBI derived from a strong uptrend OHLCV fixture passes with relaxed q_threshold."""
        bbi = compute_bbi(ohlcv_uptrend_df)
        # q_threshold=0.2 tolerates minor BBI oscillations in noisy uptrend data
        assert bbi_deriv_uptrend(bbi, min_window=5, q_threshold=0.2) is True

    # --- downtrend scenarios ---

    def test_strict_downtrend_q0_returns_false(self):
        """Strictly decreasing BBI returns False when q_threshold=0."""
        bbi = _make_bbi(list(range(30, 0, -1)))  # 30, 29, ..., 1
        assert bbi_deriv_uptrend(bbi, min_window=5, q_threshold=0.0) is False

    def test_strict_downtrend_q1_returns_false(self):
        """Pure downtrend: max(diffs) < 0, so even q_threshold=1.0 returns False."""
        bbi = _make_bbi(list(range(30, 0, -1)))
        # quantile(diffs, 1.0) = max(diffs) = -1 < 0 → False for every sub-window
        assert bbi_deriv_uptrend(bbi, min_window=5, q_threshold=1.0) is False

    # --- flat scenario ---

    def test_flat_bbi_q0_returns_true(self):
        """Flat BBI (all diffs == 0): quantile(0.0) == 0 >= 0, returns True."""
        bbi = _make_bbi([5.0] * 30)
        assert bbi_deriv_uptrend(bbi, min_window=5, q_threshold=0.0) is True

    # --- min_window enforcement ---

    def test_insufficient_data_returns_false(self):
        """Returns False when BBI has fewer valid values than min_window."""
        bbi = _make_bbi([1.0, 2.0, 3.0, 4.0])  # 4 elements
        assert bbi_deriv_uptrend(bbi, min_window=10) is False

    def test_exactly_min_window_length_uptrend_returns_true(self):
        """Exactly min_window elements in a strict uptrend should return True."""
        bbi = _make_bbi([1.0, 2.0, 3.0, 4.0, 5.0])  # exactly 5 elements
        assert bbi_deriv_uptrend(bbi, min_window=5) is True

    def test_nan_values_reduce_effective_length(self):
        """Leading NaN values are dropped; if remaining data < min_window → False."""
        bbi = _make_bbi([np.nan, np.nan, np.nan, 1.0, 2.0])  # only 2 valid
        assert bbi_deriv_uptrend(bbi, min_window=5) is False

    # --- q_threshold edge cases ---

    def test_invalid_q_threshold_above_1_raises(self):
        """q_threshold > 1 raises ValueError."""
        bbi = _make_bbi(list(range(1, 20)))
        with pytest.raises(ValueError):
            bbi_deriv_uptrend(bbi, min_window=5, q_threshold=1.5)

    def test_invalid_q_threshold_below_0_raises(self):
        """q_threshold < 0 raises ValueError."""
        bbi = _make_bbi(list(range(1, 20)))
        with pytest.raises(ValueError):
            bbi_deriv_uptrend(bbi, min_window=5, q_threshold=-0.1)

    def test_q_threshold_0_rejects_tail_dip(self):
        """With q_threshold=0, a dip at the tail and only one window available → False."""
        # [1,2,3,4,3]: diffs=[1,1,1,-1]; min_window=5 forces the only window to be
        # the full series, whose q=0 quantile is -1 < 0 → False.
        bbi = _make_bbi([1.0, 2.0, 3.0, 4.0, 3.0])
        assert bbi_deriv_uptrend(bbi, min_window=5, q_threshold=0.0) is False

    def test_q_threshold_0_valid_subwindow_at_tail(self):
        """q_threshold=0 passes when a clean tail sub-window exists (early dip excluded)."""
        # Dip at the very start then clean uptrend for the rest.
        # w=10 (last 10 elements) has all-positive diffs → True.
        values = [10.0, 5.0] + [float(5 + i) for i in range(1, 10)]  # dip at index 1
        bbi = _make_bbi(values)  # 11 elements
        assert bbi_deriv_uptrend(bbi, min_window=5, q_threshold=0.0) is True

    def test_q_threshold_half_allows_partial_drops_in_full_window(self):
        """q_threshold=0.5 passes when the majority (median) of diffs is positive."""
        # 8 steps up (+1), 2 steps down (-0.5) → median(diffs) = 1.0 >= 0
        diffs = [1.0] * 8 + [-0.5, -0.5]
        values = [10.0]
        for d in diffs:
            values.append(values[-1] + d)
        bbi = _make_bbi(values)  # 11 elements
        # Force exactly one window (min_window=max_window=11)
        assert bbi_deriv_uptrend(bbi, min_window=11, max_window=11, q_threshold=0.5) is True

    # --- max_window parameter ---

    def test_max_window_limits_longest_search(self):
        """max_window restricts the search to at most max_window elements."""
        bbi = _make_bbi(list(range(1, 31)))  # 30 elements, all uptrend
        # Only the last 5 elements are considered; still an uptrend → True
        assert bbi_deriv_uptrend(bbi, min_window=5, max_window=5) is True

    def test_max_window_none_uses_full_length(self):
        """max_window=None causes the function to consider the full BBI length."""
        bbi = _make_bbi(list(range(1, 11)))  # 10 uptrend elements
        assert bbi_deriv_uptrend(bbi, min_window=5, max_window=None) is True


# ---------------------------------------------------------------------------
# last_valid_ma_cross_up
# ---------------------------------------------------------------------------


class TestLastValidMaCrossUp:
    """Tests for last_valid_ma_cross_up."""

    def test_cross_found_returns_correct_position(self):
        """Returns the iloc position of the most recent valid crossover."""
        close = pd.Series([5.0, 5.0, 5.0, 12.0, 12.0, 12.0])
        ma = pd.Series([10.0] * 6)
        # close[2]=5 < ma[2]=10 and close[3]=12 >= ma[3]=10 → cross at index 3
        result = last_valid_ma_cross_up(close, ma)
        assert result == 3

    def test_no_cross_always_above_returns_none(self):
        """Returns None when close is always above MA (no crossing from below)."""
        close = pd.Series([12.0, 13.0, 14.0, 15.0])
        ma = pd.Series([10.0] * 4)
        assert last_valid_ma_cross_up(close, ma) is None

    def test_no_cross_always_below_returns_none(self):
        """Returns None when close is always below MA."""
        close = pd.Series([5.0, 4.0, 3.0, 2.0])
        ma = pd.Series([10.0] * 4)
        assert last_valid_ma_cross_up(close, ma) is None

    def test_multiple_crosses_returns_most_recent(self):
        """When multiple crossovers exist, the most recent (highest index) is returned."""
        # Cross at index 1 and index 3; function returns 3 (most recent)
        close = pd.Series([5.0, 12.0, 5.0, 12.0])
        ma = pd.Series([10.0] * 4)
        result = last_valid_ma_cross_up(close, ma)
        assert result == 3

    def test_lookback_n_restricts_to_recent_window(self):
        """lookback_n limits the search to the most recent N bars; older cross excluded."""
        # Crosses at index 5 and index 15 in a 20-element series
        close_vals = [5.0] * 5 + [12.0] * 5 + [5.0] * 5 + [12.0] * 5
        close = pd.Series(close_vals)
        ma = pd.Series([10.0] * 20)
        # lookback_n=8 → start = max(1, 20-8) = 12 → search i from 19 down to 12
        # Cross at i=15 is within range; cross at i=5 is outside → returns 15
        result = last_valid_ma_cross_up(close, ma, lookback_n=8)
        assert result == 15

    def test_lookback_n_excludes_only_cross(self):
        """Returns None when the only crossover falls outside the lookback window."""
        close = pd.Series([5.0, 5.0, 5.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0])
        ma = pd.Series([10.0] * 10)
        # Cross at index 3; lookback_n=5 → start = max(1, 10-5) = 5 → search 9..5
        # No cross in that range → None
        result = last_valid_ma_cross_up(close, ma, lookback_n=5)
        assert result is None

    def test_lookback_none_searches_full_history(self):
        """lookback_n=None searches the entire series history."""
        close = pd.Series([5.0, 12.0] + [12.0] * 18)
        ma = pd.Series([10.0] * 20)
        # Only cross is at index 1; lookback_n=None → found
        result = last_valid_ma_cross_up(close, ma, lookback_n=None)
        assert result == 1

    def test_cross_at_exact_ma_equality(self):
        """Crossover is detected when close equals MA (condition is >=, not >)."""
        close = pd.Series([5.0, 10.0])  # close[1] == ma[1]
        ma = pd.Series([10.0, 10.0])
        # close[0]=5 < ma[0]=10 and close[1]=10 >= ma[1]=10 → cross at index 1
        result = last_valid_ma_cross_up(close, ma)
        assert result == 1

    def test_nan_values_are_skipped_gracefully(self):
        """Rows with NaN in close or MA are skipped without raising an error."""
        close = pd.Series([5.0, np.nan, 5.0, 12.0])
        ma = pd.Series([10.0, np.nan, 10.0, 10.0])
        # i=3: close[2]=5<10, close[3]=12>=10 → cross at index 3
        result = last_valid_ma_cross_up(close, ma)
        assert result == 3


# ---------------------------------------------------------------------------
# passes_day_constraints_today
# ---------------------------------------------------------------------------


class TestPassesDayConstraintsToday:
    """Tests for passes_day_constraints_today."""

    # --- pass scenarios ---

    def test_stable_day_passes(self):
        """Small price change and small amplitude satisfy both constraints → True."""
        df = _make_two_day_df(
            prev_close=100.0,
            today_close=100.5,  # pct_chg = 0.5% < 2%
            today_high=101.0,
            today_low=100.0,  # amplitude = 1/100 = 1% < 7%
        )
        assert passes_day_constraints_today(df) is True

    def test_custom_relaxed_limits_pass(self):
        """With wider pct_limit and amp_limit, a more volatile day still passes."""
        df = _make_two_day_df(
            prev_close=100.0,
            today_close=104.0,  # pct_chg = 4%
            today_high=112.0,
            today_low=100.0,  # amplitude = 12%
        )
        assert passes_day_constraints_today(df, pct_limit=0.05, amp_limit=0.15) is True

    # --- fail: large percentage change ---

    def test_large_positive_pct_change_fails(self):
        """Close up more than pct_limit returns False."""
        df = _make_two_day_df(
            prev_close=100.0,
            today_close=105.0,  # pct_chg = 5% > default 2%
            today_high=105.5,
            today_low=104.5,
        )
        assert passes_day_constraints_today(df) is False

    def test_large_negative_pct_change_fails(self):
        """A large drop also exceeds pct_limit (absolute-value check) → False."""
        df = _make_two_day_df(
            prev_close=100.0,
            today_close=95.0,  # pct_chg = 5% (abs) > default 2%
            today_high=96.0,
            today_low=94.5,
        )
        assert passes_day_constraints_today(df) is False

    # --- fail: large amplitude ---

    def test_large_amplitude_fails(self):
        """High-low spread exceeding amp_limit returns False."""
        df = _make_two_day_df(
            prev_close=100.0,
            today_close=100.5,  # small pct change
            today_high=115.0,
            today_low=100.0,  # amplitude = 15% > 7%
        )
        assert passes_day_constraints_today(df) is False

    # --- edge cases ---

    def test_single_row_returns_false(self):
        """DataFrame with only one row cannot compute prev close → False."""
        df = pd.DataFrame(
            {
                "open": [10.0],
                "high": [10.1],
                "low": [9.9],
                "close": [10.0],
                "volume": [1_000_000.0],
            }
        )
        assert passes_day_constraints_today(df) is False

    def test_empty_df_returns_false(self, ohlcv_empty_df):
        """Empty DataFrame returns False."""
        assert passes_day_constraints_today(ohlcv_empty_df) is False

    def test_zero_prev_close_returns_false(self):
        """Zero previous close triggers the guard condition → False."""
        df = _make_two_day_df(
            prev_close=0.0,
            today_close=1.0,
            today_high=1.1,
            today_low=0.9,
        )
        assert passes_day_constraints_today(df) is False

    def test_zero_low_today_returns_false(self):
        """Zero today's low triggers the guard condition → False."""
        df = _make_two_day_df(
            prev_close=10.0,
            today_close=10.5,
            today_high=11.0,
            today_low=0.0,
        )
        assert passes_day_constraints_today(df) is False


# ---------------------------------------------------------------------------
# zx_condition_at_positions
# ---------------------------------------------------------------------------


class TestZxConditionAtPositions:
    """Tests for zx_condition_at_positions."""

    def test_empty_df_returns_false(self, ohlcv_empty_df):
        """Empty DataFrame returns False immediately."""
        assert zx_condition_at_positions(ohlcv_empty_df) is False

    def test_short_data_nan_long_line_returns_false(self):
        """With < 114 rows, ZXDKX (which needs MA114) is NaN → always False."""
        close_vals = [float(i) for i in range(1, 61)]  # 60 rows < 114
        df = _make_df_with_close(close_vals)
        assert zx_condition_at_positions(df) is False

    def test_out_of_bounds_pos_returns_false(self):
        """Position beyond the DataFrame length returns False."""
        close_vals = [float(i) for i in range(1, 61)]
        df = _make_df_with_close(close_vals)
        assert zx_condition_at_positions(df, pos=1000) is False

    def test_negative_pos_returns_false(self):
        """Negative position index returns False."""
        close_vals = [float(i) for i in range(1, 61)]
        df = _make_df_with_close(close_vals)
        assert zx_condition_at_positions(df, pos=-1) is False

    def test_true_when_all_conditions_met_uptrend(self):
        """Strong linear uptrend with 250 bars: close > long line and short > long line."""
        # Linear uptrend: slope 0.2 per bar ensures fast MAs > slow MAs > recent close
        n = 250
        close = [10.0 + 0.2 * i for i in range(n)]
        df = _make_df_with_close(close)
        result = zx_condition_at_positions(
            df, require_close_gt_long=True, require_short_gt_long=True
        )
        assert result is True

    def test_false_when_close_lt_long_in_downtrend(self):
        """In a downtrend, recent close falls below the long MA average → False."""
        n = 250
        # Linear downtrend: recent close is below all historical MAs
        close = [100.0 - 0.2 * i for i in range(n)]
        df = _make_df_with_close(close)
        result = zx_condition_at_positions(
            df, require_close_gt_long=True, require_short_gt_long=False
        )
        assert result is False

    def test_require_close_gt_long_false_skips_close_check(self):
        """With require_close_gt_long=False, only the short>long constraint applies."""
        n = 250
        close = [10.0 + 0.2 * i for i in range(n)]
        df = _make_df_with_close(close)
        # Disable the close check; short > long should still hold in uptrend
        result = zx_condition_at_positions(
            df, require_close_gt_long=False, require_short_gt_long=True
        )
        assert result is True

    def test_none_pos_and_last_pos_are_equivalent(self):
        """pos=None is equivalent to pos=len(df)-1 (the final row)."""
        n = 250
        close = [10.0 + 0.2 * i for i in range(n)]
        df = _make_df_with_close(close)
        result_none = zx_condition_at_positions(df, pos=None)
        result_last = zx_condition_at_positions(df, pos=len(df) - 1)
        assert result_none == result_last

    def test_all_constraints_disabled_returns_true_with_sufficient_data(self):
        """When both requires are False and long line is finite, returns True."""
        n = 250
        close = [10.0 + 0.2 * i for i in range(n)]
        df = _make_df_with_close(close)
        result = zx_condition_at_positions(
            df, require_close_gt_long=False, require_short_gt_long=False
        )
        assert result is True
