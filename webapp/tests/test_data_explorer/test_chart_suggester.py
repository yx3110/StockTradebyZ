"""Tests for chart_suggester.suggest() — rule table per spec §5.4."""
import pandas as pd

from core.data_explorer.chart_suggester import suggest


def test_r1_time_series_single_code() -> None:
    df = pd.DataFrame({
        "trade_date": pd.to_datetime(["2026-04-17", "2026-04-18"]),
        "code": ["600519.SH", "600519.SH"],
        "close": [1700.0, 1720.0],
    })
    hint = suggest(df)
    assert hint["type"] == "line"
    assert hint["x"] == "trade_date"
    assert hint["y"] == "close"


def test_r2_time_series_multi_code_returns_none() -> None:
    df = pd.DataFrame({
        "trade_date": pd.to_datetime(["2026-04-18", "2026-04-18"]),
        "code": ["600519.SH", "000858.SZ"],
        "close": [1720.0, 150.0],
    })
    assert suggest(df) is None


def test_r3_two_numerics_scatter() -> None:
    df = pd.DataFrame({"pb": [1.0, 2.0, 3.0], "roe": [10, 20, 30]})
    hint = suggest(df)
    assert hint["type"] == "scatter"
    # r reported in annotation; simply assert presence
    assert "annotations" in hint and "pearson_r" in hint["annotations"]


def test_r4_bar_small_categorical() -> None:
    df = pd.DataFrame({"code": [f"S{i}" for i in range(30)], "pred": range(30)})
    hint = suggest(df)
    assert hint["type"] == "bar"
    assert hint["x"] == "code"
    assert hint["y"] == "pred"


def test_r5_histogram_large_categorical() -> None:
    df = pd.DataFrame({"code": [f"S{i}" for i in range(100)], "pred": range(100)})
    hint = suggest(df)
    assert hint["type"] == "histogram"
    assert hint["x"] == "pred"


def test_r6_single_numeric_histogram() -> None:
    df = pd.DataFrame({"pred_10d": [0.01, 0.05, -0.02]})
    hint = suggest(df)
    assert hint["type"] == "histogram"
    assert hint["x"] == "pred_10d"


def test_else_no_hint() -> None:
    df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})  # two categoricals, no numeric
    assert suggest(df) is None


def test_empty_dataframe_no_hint() -> None:
    assert suggest(pd.DataFrame()) is None
