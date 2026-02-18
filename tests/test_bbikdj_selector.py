"""Unit tests for BBIKDJSelector in Selector.py.

Covers:
  - Instantiation with default and custom parameters
  - Empty data handling
  - select() returning a list
  - _passes_filters() rejecting bad data
"""

import numpy as np
import pandas as pd
import pytest

from Selector import BBIKDJSelector


# ---------------------------------------------------------------------------
# Local helper factories (no "date" column = suitable for _passes_filters;
# with "date" column = suitable for select())
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
    """Generate a synthetic OHLCV DataFrame that includes an explicit 'date' column.

    Required for ``BBIKDJSelector.select()`` which filters rows via
    ``df[df["date"] <= date]``.
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


# ---------------------------------------------------------------------------
# TestBBIKDJSelectorInit
# ---------------------------------------------------------------------------


class TestBBIKDJSelectorInit:
    """Tests for BBIKDJSelector instantiation."""

    def test_default_instantiation(self):
        """BBIKDJSelector can be instantiated with no arguments."""
        sel = BBIKDJSelector()
        assert sel is not None

    def test_default_j_threshold(self):
        """Default j_threshold is -5."""
        sel = BBIKDJSelector()
        assert sel.j_threshold == -5

    def test_default_bbi_min_window(self):
        """Default bbi_min_window is 90."""
        sel = BBIKDJSelector()
        assert sel.bbi_min_window == 90

    def test_default_max_window(self):
        """Default max_window is 90."""
        sel = BBIKDJSelector()
        assert sel.max_window == 90

    def test_default_price_range_pct(self):
        """Default price_range_pct is 100.0."""
        sel = BBIKDJSelector()
        assert sel.price_range_pct == pytest.approx(100.0)

    def test_default_bbi_q_threshold(self):
        """Default bbi_q_threshold is 0.05."""
        sel = BBIKDJSelector()
        assert sel.bbi_q_threshold == pytest.approx(0.05)

    def test_default_j_q_threshold(self):
        """Default j_q_threshold is 0.10."""
        sel = BBIKDJSelector()
        assert sel.j_q_threshold == pytest.approx(0.10)

    def test_custom_j_threshold(self):
        """Custom j_threshold is stored correctly."""
        sel = BBIKDJSelector(j_threshold=-10)
        assert sel.j_threshold == -10

    def test_custom_bbi_min_window(self):
        """Custom bbi_min_window is stored correctly."""
        sel = BBIKDJSelector(bbi_min_window=60)
        assert sel.bbi_min_window == 60

    def test_custom_max_window(self):
        """Custom max_window is stored correctly."""
        sel = BBIKDJSelector(max_window=120)
        assert sel.max_window == 120

    def test_custom_price_range_pct(self):
        """Custom price_range_pct is stored correctly."""
        sel = BBIKDJSelector(price_range_pct=50.0)
        assert sel.price_range_pct == pytest.approx(50.0)

    def test_custom_bbi_q_threshold(self):
        """Custom bbi_q_threshold is stored correctly."""
        sel = BBIKDJSelector(bbi_q_threshold=0.10)
        assert sel.bbi_q_threshold == pytest.approx(0.10)

    def test_custom_j_q_threshold(self):
        """Custom j_q_threshold is stored correctly."""
        sel = BBIKDJSelector(j_q_threshold=0.20)
        assert sel.j_q_threshold == pytest.approx(0.20)

    def test_all_custom_params(self):
        """BBIKDJSelector stores all custom parameters correctly when set together."""
        sel = BBIKDJSelector(
            j_threshold=-8,
            bbi_min_window=60,
            max_window=120,
            price_range_pct=50.0,
            bbi_q_threshold=0.10,
            j_q_threshold=0.15,
        )
        assert sel.j_threshold == -8
        assert sel.bbi_min_window == 60
        assert sel.max_window == 120
        assert sel.price_range_pct == pytest.approx(50.0)
        assert sel.bbi_q_threshold == pytest.approx(0.10)
        assert sel.j_q_threshold == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# TestBBIKDJSelectorPassesFilters
# ---------------------------------------------------------------------------


class TestBBIKDJSelectorPassesFilters:
    """Tests for BBIKDJSelector._passes_filters()."""

    def test_empty_dataframe_returns_false(self):
        """_passes_filters() returns False for an empty DataFrame."""
        sel = BBIKDJSelector()
        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = sel._passes_filters(empty_df)
        assert result is False

    def test_single_row_returns_false(self):
        """_passes_filters() returns False for a single-row DataFrame.

        ``passes_day_constraints_today`` requires at least 2 rows; a single row
        causes it to return False immediately.
        """
        sel = BBIKDJSelector()
        df = pd.DataFrame(
            {
                "open": [10.0],
                "high": [10.5],
                "low": [9.5],
                "close": [10.0],
                "volume": [1_000_000.0],
            }
        )
        result = sel._passes_filters(df)
        assert result is False

    def test_too_few_rows_returns_false(self):
        """_passes_filters() returns False when data is shorter than bbi_min_window.

        With only 20 rows, ``bbi_deriv_uptrend`` cannot satisfy min_window=90.
        """
        sel = BBIKDJSelector(bbi_min_window=90)
        n = 20
        close = np.linspace(10.0, 10.2, n)
        spread = 0.005 * close
        df = pd.DataFrame(
            {
                "open": close - spread * 0.5,
                "high": close + spread,
                "low": close - spread,
                "close": close,
                "volume": np.full(n, 1_000_000.0),
            }
        )
        result = sel._passes_filters(df)
        assert result is False

    def test_extreme_daily_price_change_returns_false(self):
        """_passes_filters() returns False if today's close/prev-close ratio exceeds 2%.

        A 10% drop on the last bar triggers the ``passes_day_constraints_today``
        guard and the function short-circuits with False.
        """
        sel = BBIKDJSelector()
        rng = np.random.default_rng(1)
        n = 200
        close = 10.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, n)))
        # Inject a 10 % price drop on the last bar to breach pct_limit=0.02
        close[-1] = close[-2] * 0.90
        spread = 0.003 * close
        df = pd.DataFrame(
            {
                "open": close - spread * 0.5,
                "high": close + spread,
                "low": close - spread,
                "close": close,
                "volume": np.full(n, 1_000_000.0),
            }
        )
        result = sel._passes_filters(df)
        assert result is False

    def test_large_amplitude_returns_false(self):
        """_passes_filters() returns False when today's amplitude exceeds 7%.

        High-Low / Low > amp_limit (0.07) causes ``passes_day_constraints_today``
        to return False.
        """
        sel = BBIKDJSelector()
        n = 200
        close = np.linspace(10.0, 10.2, n)
        spread = 0.003 * close
        high = close.copy()
        low = close.copy()
        # Force a 10% amplitude on the last bar (High - Low) / Low ≈ 0.10 > 0.07
        high[-1] = close[-1] * 1.07
        low[-1] = close[-1] * 0.97
        df = pd.DataFrame(
            {
                "open": close - spread * 0.5,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 1_000_000.0),
            }
        )
        result = sel._passes_filters(df)
        assert result is False

    def test_tight_price_range_constraint_returns_false(self):
        """_passes_filters() returns False when price_range_pct is violated.

        Setting price_range_pct=0.001 (0.1 %) is far tighter than typical data,
        so the price-range guard trips immediately.
        """
        sel = BBIKDJSelector(price_range_pct=0.001, max_window=90)
        n = 200
        # Linearly rising prices guarantee a range of about 100 % over the window
        close = np.linspace(10.0, 20.0, n)
        spread = 0.001 * close
        # Make the last two bars' pct change tiny to avoid the day-constraint guard
        close[-1] = close[-2] * 1.001
        df = pd.DataFrame(
            {
                "open": close - spread * 0.5,
                "high": close + spread,
                "low": close - spread,
                "close": close,
                "volume": np.full(n, 1_000_000.0),
            }
        )
        result = sel._passes_filters(df)
        assert result is False

    def test_returns_bool(self):
        """_passes_filters() always returns a Python bool."""
        sel = BBIKDJSelector()
        df = _make_ohlcv(n=200, trend=0.001, seed=42)
        result = sel._passes_filters(df)
        assert isinstance(result, bool)

    def test_downtrend_bbi_returns_false(self):
        """_passes_filters() returns False for a persistent downtrend.

        A sharply falling BBI fails the ``bbi_deriv_uptrend`` check.
        """
        sel = BBIKDJSelector(bbi_min_window=30, max_window=90)
        n = 200
        # Strong monotone downtrend: close falls 0.5 % per day on average
        close = 10.0 * np.exp(np.cumsum(np.full(n, -0.005)))
        spread = 0.002 * close
        # Keep last two bars' pct change within 2 % to pass day constraint
        close[-1] = close[-2] * 0.999
        df = pd.DataFrame(
            {
                "open": close - spread * 0.5,
                "high": close + spread,
                "low": close - spread,
                "close": close,
                "volume": np.full(n, 1_000_000.0),
            }
        )
        result = sel._passes_filters(df)
        assert result is False


# ---------------------------------------------------------------------------
# TestBBIKDJSelectorSelect
# ---------------------------------------------------------------------------


class TestBBIKDJSelectorSelect:
    """Tests for BBIKDJSelector.select()."""

    def test_select_returns_list(self):
        """select() always returns a list."""
        sel = BBIKDJSelector()
        date = pd.Timestamp("2025-01-01")
        result = sel.select(date, {})
        assert isinstance(result, list)

    def test_select_empty_data_dict_returns_empty_list(self):
        """select() with an empty data dict returns an empty list."""
        sel = BBIKDJSelector()
        date = pd.Timestamp("2025-01-01")
        result = sel.select(date, {})
        assert result == []

    def test_select_date_before_all_data_returns_empty_list(self):
        """select() returns [] when the given date precedes all rows in the DataFrame.

        ``df[df["date"] <= date]`` produces an empty hist, which is skipped.
        """
        sel = BBIKDJSelector()
        df = _make_ohlcv_with_date(n=200, trend=0.001, seed=5)
        # Use a date earlier than the earliest row
        date = pd.Timestamp("2000-01-01")
        result = sel.select(date, {"000001": df})
        assert result == []

    def test_select_insufficient_history_returns_empty_list(self):
        """select() returns [] when too few rows precede the given date.

        With only 5 rows of history, all filter checks fail early.
        """
        sel = BBIKDJSelector()
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
        date = pd.Timestamp("2024-01-10")
        result = sel.select(date, {"000001": df, "000002": df})
        assert result == []

    def test_select_downtrend_stocks_return_empty_list(self):
        """select() returns [] when all stocks are in a persistent downtrend.

        Strong downtrend data fails the BBI uptrend requirement.
        """
        sel = BBIKDJSelector()
        df_down = _make_ohlcv_with_date(n=200, trend=-0.01, seed=99)
        date = df_down["date"].iloc[-1]
        result = sel.select(date, {"A": df_down, "B": df_down, "C": df_down})
        assert result == []

    def test_select_result_elements_are_strings(self):
        """All stock codes returned by select() are strings."""
        sel = BBIKDJSelector()
        df = _make_ohlcv_with_date(n=200, trend=0.001, seed=10)
        date = df["date"].iloc[-1]
        result = sel.select(date, {"000001": df, "000002": df})
        for code in result:
            assert isinstance(code, str)

    def test_select_none_dataframe_entry_not_in_result(self):
        """select() skips entries whose DataFrame is empty after date filtering."""
        sel = BBIKDJSelector()
        # Build a df whose only row is *after* the query date
        future_df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2030-01-01")],
                "open": [10.0],
                "high": [10.5],
                "low": [9.5],
                "close": [10.0],
                "volume": [1_000_000.0],
            }
        )
        date = pd.Timestamp("2025-01-01")
        result = sel.select(date, {"FUTURE": future_df})
        assert "FUTURE" not in result

    def test_select_multiple_stocks_all_fail_filters(self):
        """select() returns [] when multiple stocks all fail _passes_filters()."""
        sel = BBIKDJSelector()
        # Flat price series: BBI will be flat (not uptrending), KDJ / DIF will fail too
        n = 200
        flat_prices = np.full(n, 10.0)
        dates = pd.bdate_range("2024-01-01", periods=n)
        # Add tiny noise so day constraint passes; keep it below 1 %
        rng = np.random.default_rng(7)
        noise = rng.uniform(-0.001, 0.001, n) * flat_prices
        close = flat_prices + noise
        df_flat = pd.DataFrame(
            {
                "date": dates,
                "open": close - 0.01,
                "high": close + 0.05,
                "low": close - 0.05,
                "close": close,
                "volume": np.full(n, 1_000_000.0),
            }
        )
        date = dates[-1]
        result = sel.select(date, {"X": df_flat, "Y": df_flat})
        assert result == []
