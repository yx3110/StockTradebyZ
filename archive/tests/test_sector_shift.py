"""Unit tests for SectorShift.py utility functions.

Covers:
  - _list_codes_from_data_dir : CSV/parquet/feather/pkl files, 6-digit code
                                extraction, non-matching filenames, empty dir
  - _load_industry_from_stocklist : valid stocklist with various column names
                                    ('symbol', 'ts_code', 'code', fallback),
                                    industry column ('industry' or '行业'),
                                    missing file, empty file, no parseable
                                    code column, no industry column
  - compute_j_industry_distribution : empty data dir (no codes → early return),
                                       invalid trade_date, YYYYMMDD format,
                                       YYYY-MM-DD format, datetime object
"""

from __future__ import annotations

import sys
import os
import types
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure conftest compat shim has run before we try to import SectorShift.
# (conftest._preload_selector_compat() is called at module level so Selector
#  is already in sys.modules by the time this file is collected.)
# ---------------------------------------------------------------------------

from SectorShift import (
    _list_codes_from_data_dir,
    _load_industry_from_stocklist,
    compute_j_industry_distribution,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _touch(directory: Path, filename: str) -> Path:
    """Create an empty file inside *directory* and return its path."""
    p = directory / filename
    p.touch()
    return p


def _write_stocklist_csv(
    directory: Path,
    *,
    code_col: str = "symbol",
    codes: list[str] | None = None,
    industry_col: str = "industry",
    industries: list[str] | None = None,
) -> Path:
    """Write a minimal stocklist CSV and return its path.

    Parameters
    ----------
    directory:
        Destination directory.
    code_col:
        Column name that holds the stock codes (e.g. 'symbol', 'ts_code',
        'code', or any arbitrary name for fallback tests).
    codes:
        6-digit stock codes to store.  Defaults to two example codes.
    industry_col:
        Column name that holds the industry label ('industry' or '行业').
    industries:
        Matching industry labels.  Must be same length as *codes*.
    """
    if codes is None:
        codes = ["000001", "000002"]
    if industries is None:
        industries = ["银行", "证券"]

    df = pd.DataFrame({code_col: codes, industry_col: industries})
    path = directory / "stocklist.csv"
    df.to_csv(path, index=False)
    return path


def _make_ohlcv_df(n: int = 60, start: str = "2024-01-01") -> pd.DataFrame:
    """Generate a minimal OHLCV DataFrame with a 'date' column."""
    dates = pd.bdate_range(start=start, periods=n)
    rng = np.random.default_rng(42)
    close = 10.0 + np.cumsum(rng.normal(0, 0.1, n))
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": rng.integers(100_000, 5_000_000, n).astype(float),
        }
    )


# ===========================================================================
# _list_codes_from_data_dir
# ===========================================================================


class TestListCodesFromDataDir:
    """Tests for _list_codes_from_data_dir."""

    def test_empty_directory_returns_empty_list(self, tmp_path):
        """An empty directory yields no codes."""
        result = _list_codes_from_data_dir(tmp_path)
        assert result == []

    def test_single_csv_with_6digit_code(self, tmp_path):
        """A CSV file whose stem contains a 6-digit code is detected."""
        _touch(tmp_path, "000001.csv")
        result = _list_codes_from_data_dir(tmp_path)
        assert result == ["000001"]

    def test_multiple_csv_files(self, tmp_path):
        """Multiple CSV files yield all 6-digit codes, sorted."""
        for code in ["600036", "000001", "300001"]:
            _touch(tmp_path, f"{code}.csv")
        result = _list_codes_from_data_dir(tmp_path)
        assert result == sorted(["600036", "000001", "300001"])

    def test_parquet_file_detected(self, tmp_path):
        """A .parquet file whose stem contains a 6-digit code is detected."""
        _touch(tmp_path, "000002.parquet")
        result = _list_codes_from_data_dir(tmp_path)
        assert result == ["000002"]

    def test_feather_file_detected(self, tmp_path):
        """A .feather file whose stem contains a 6-digit code is detected."""
        _touch(tmp_path, "000003.feather")
        result = _list_codes_from_data_dir(tmp_path)
        assert result == ["000003"]

    def test_pkl_file_detected(self, tmp_path):
        """A .pkl file whose stem contains a 6-digit code is detected."""
        _touch(tmp_path, "000004.pkl")
        result = _list_codes_from_data_dir(tmp_path)
        assert result == ["000004"]

    def test_mixed_file_types_all_detected(self, tmp_path):
        """CSV, parquet, feather and pkl files are all scanned."""
        _touch(tmp_path, "000001.csv")
        _touch(tmp_path, "000002.parquet")
        _touch(tmp_path, "000003.feather")
        _touch(tmp_path, "000004.pkl")
        result = _list_codes_from_data_dir(tmp_path)
        assert set(result) == {"000001", "000002", "000003", "000004"}

    def test_file_without_6digit_code_ignored(self, tmp_path):
        """Files whose stems contain no 6-digit sequence are skipped."""
        _touch(tmp_path, "stocklist.csv")
        _touch(tmp_path, "README.csv")
        result = _list_codes_from_data_dir(tmp_path)
        assert result == []

    def test_code_embedded_in_longer_name(self, tmp_path):
        """6-digit codes embedded inside longer filenames are extracted."""
        _touch(tmp_path, "daily_000001_2024.csv")
        result = _list_codes_from_data_dir(tmp_path)
        assert result == ["000001"]

    def test_duplicate_codes_deduped(self, tmp_path):
        """If the same code appears in multiple file types, it's listed once."""
        _touch(tmp_path, "000001.csv")
        _touch(tmp_path, "000001.parquet")
        result = _list_codes_from_data_dir(tmp_path)
        assert result == ["000001"]

    def test_result_is_sorted(self, tmp_path):
        """Returned codes are in ascending lexicographic order."""
        for code in ["600519", "000858", "002415"]:
            _touch(tmp_path, f"{code}.csv")
        result = _list_codes_from_data_dir(tmp_path)
        assert result == sorted(result)

    def test_accepts_string_path(self, tmp_path):
        """A plain string path is accepted (not just Path objects)."""
        _touch(tmp_path, "688001.csv")
        result = _list_codes_from_data_dir(str(tmp_path))
        assert result == ["688001"]

    def test_recursive_scan(self, tmp_path):
        """Files inside sub-directories are also discovered."""
        sub = tmp_path / "sub"
        sub.mkdir()
        _touch(sub, "000005.csv")
        result = _list_codes_from_data_dir(tmp_path)
        assert result == ["000005"]


# ===========================================================================
# _load_industry_from_stocklist
# ===========================================================================


class TestLoadIndustryFromStocklist:
    """Tests for _load_industry_from_stocklist."""

    # --- happy-path: code column variants ---

    def test_symbol_column_parsed(self, tmp_path):
        """Reads codes from a 'symbol' column."""
        path = _write_stocklist_csv(tmp_path, code_col="symbol")
        result = _load_industry_from_stocklist(path, ["000001", "000002"])
        assert set(result["代码"].tolist()) == {"000001", "000002"}

    def test_ts_code_column_parsed(self, tmp_path):
        """Reads 6-digit codes from a 'ts_code' column (e.g. '000001.SZ')."""
        df = pd.DataFrame(
            {"ts_code": ["000001.SZ", "000002.SZ"], "industry": ["银行", "证券"]}
        )
        path = tmp_path / "stocklist.csv"
        df.to_csv(path, index=False)

        result = _load_industry_from_stocklist(path, ["000001", "000002"])
        assert set(result["代码"].tolist()) == {"000001", "000002"}

    def test_code_column_parsed(self, tmp_path):
        """Reads codes from a generic 'code' column."""
        path = _write_stocklist_csv(tmp_path, code_col="code")
        result = _load_industry_from_stocklist(path, ["000001", "000002"])
        assert set(result["代码"].tolist()) == {"000001", "000002"}

    def test_fallback_column_parsed(self, tmp_path):
        """Falls back to the first column that contains a 6-digit sequence."""
        df = pd.DataFrame(
            {"ticker": ["000001", "000002"], "industry": ["银行", "证券"]}
        )
        path = tmp_path / "stocklist.csv"
        df.to_csv(path, index=False)

        result = _load_industry_from_stocklist(path, ["000001", "000002"])
        assert set(result["代码"].tolist()) == {"000001", "000002"}

    def test_chinese_industry_column(self, tmp_path):
        """Reads the industry from a '行业' column."""
        path = _write_stocklist_csv(
            tmp_path, code_col="symbol", industry_col="行业"
        )
        result = _load_industry_from_stocklist(path, ["000001", "000002"])
        assert "行业" in result.columns

    def test_industry_column_renamed_to_hanzi(self, tmp_path):
        """Output DataFrame always uses '行业' as the industry column name."""
        path = _write_stocklist_csv(tmp_path, industry_col="industry")
        result = _load_industry_from_stocklist(path, ["000001", "000002"])
        assert "行业" in result.columns
        assert "industry" not in result.columns

    def test_filters_to_requested_codes(self, tmp_path):
        """Only rows for the requested codes are returned."""
        path = _write_stocklist_csv(
            tmp_path,
            codes=["000001", "000002", "000003"],
            industries=["银行", "证券", "科技"],
        )
        result = _load_industry_from_stocklist(path, ["000001"])
        assert list(result["代码"]) == ["000001"]

    def test_nan_industry_filled_with_unknown(self, tmp_path):
        """NaN industry values are replaced with '未知'."""
        df = pd.DataFrame(
            {"symbol": ["000001", "000002"], "industry": [None, "银行"]}
        )
        path = tmp_path / "stocklist.csv"
        df.to_csv(path, index=False)

        result = _load_industry_from_stocklist(path, ["000001", "000002"])
        assert "未知" in result["行业"].values

    def test_returns_dataframe_with_expected_columns(self, tmp_path):
        """Returned DataFrame has columns ['代码', '行业']."""
        path = _write_stocklist_csv(tmp_path)
        result = _load_industry_from_stocklist(path, ["000001", "000002"])
        assert list(result.columns) == ["代码", "行业"]

    def test_no_duplicates_on_code(self, tmp_path):
        """Duplicate rows for the same code are deduplicated."""
        df = pd.DataFrame(
            {
                "symbol": ["000001", "000001", "000002"],
                "industry": ["银行", "银行", "证券"],
            }
        )
        path = tmp_path / "stocklist.csv"
        df.to_csv(path, index=False)

        result = _load_industry_from_stocklist(path, ["000001", "000002"])
        assert result["代码"].duplicated().sum() == 0

    # --- error cases ---

    def test_missing_file_raises_file_not_found(self, tmp_path):
        """Raises FileNotFoundError when stocklist.csv does not exist."""
        missing = tmp_path / "stocklist.csv"
        with pytest.raises(FileNotFoundError, match="stocklist.csv 不存在"):
            _load_industry_from_stocklist(missing, ["000001"])

    def test_empty_file_raises_value_error(self, tmp_path):
        """Raises ValueError when stocklist.csv has a header but no data rows."""
        path = tmp_path / "stocklist.csv"
        # Write only the header line so pandas parses it as an empty DataFrame
        # (triggering the `sl.empty` check rather than EmptyDataError)
        path.write_text("symbol,industry\n", encoding="utf-8")
        with pytest.raises(ValueError, match="stocklist.csv 为空"):
            _load_industry_from_stocklist(path, ["000001"])

    def test_no_parseable_code_column_raises_value_error(self, tmp_path):
        """Raises ValueError when no column yields 6-digit codes."""
        df = pd.DataFrame({"name": ["APPLE", "GOOGLE"], "industry": ["科技", "科技"]})
        path = tmp_path / "stocklist.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="没有可解析为 6 位证券代码的列"):
            _load_industry_from_stocklist(path, ["000001"])

    def test_no_industry_column_raises_value_error(self, tmp_path):
        """Raises ValueError when neither 'industry' nor '行业' column exists."""
        df = pd.DataFrame({"symbol": ["000001", "000002"], "sector": ["A", "B"]})
        path = tmp_path / "stocklist.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="缺少行业列"):
            _load_industry_from_stocklist(path, ["000001"])

    def test_accepts_string_path(self, tmp_path):
        """Accepts a plain string path (not just Path objects)."""
        path = _write_stocklist_csv(tmp_path)
        result = _load_industry_from_stocklist(str(path), ["000001"])
        assert len(result) == 1


# ===========================================================================
# compute_j_industry_distribution
# ===========================================================================


class TestComputeJIndustryDistribution:
    """Tests for compute_j_industry_distribution."""

    def test_empty_data_dir_returns_early(self, tmp_path):
        """When data_dir has no matching files, returns meta with zero counts."""
        # Create a stocklist so we don't fail on that step
        _write_stocklist_csv(tmp_path)

        result = compute_j_industry_distribution(
            data_dir=tmp_path,
            stocklist_path=tmp_path / "stocklist.csv",
        )

        assert result["meta"]["total_codes"] == 0
        assert result["meta"]["selected_count"] == 0
        assert result["industry_counts"] == []

    def test_empty_data_dir_trade_date_in_meta(self, tmp_path):
        """trade_date appears in meta even on early return path."""
        _write_stocklist_csv(tmp_path)

        result = compute_j_industry_distribution(
            data_dir=tmp_path,
            stocklist_path=tmp_path / "stocklist.csv",
            trade_date="20240101",
        )

        assert result["meta"]["trade_date"] == "2024-01-01"

    def test_invalid_trade_date_raises_value_error(self, tmp_path):
        """An unparseable trade_date string raises ValueError."""
        with pytest.raises(ValueError, match="无法解析 trade_date"):
            compute_j_industry_distribution(
                data_dir=tmp_path,
                stocklist_path=tmp_path / "stocklist.csv",
                trade_date="not-a-date",
            )

    def test_trade_date_yyyymmdd_format(self, tmp_path):
        """YYYYMMDD string is parsed correctly."""
        _write_stocklist_csv(tmp_path)

        result = compute_j_industry_distribution(
            data_dir=tmp_path,
            stocklist_path=tmp_path / "stocklist.csv",
            trade_date="20240115",
        )

        assert result["meta"]["trade_date"] == "2024-01-15"

    def test_trade_date_iso_format(self, tmp_path):
        """YYYY-MM-DD string is parsed correctly."""
        _write_stocklist_csv(tmp_path)

        result = compute_j_industry_distribution(
            data_dir=tmp_path,
            stocklist_path=tmp_path / "stocklist.csv",
            trade_date="2024-01-15",
        )

        assert result["meta"]["trade_date"] == "2024-01-15"

    def test_trade_date_datetime_object(self, tmp_path):
        """A datetime object is accepted and reflected in meta."""
        _write_stocklist_csv(tmp_path)

        result = compute_j_industry_distribution(
            data_dir=tmp_path,
            stocklist_path=tmp_path / "stocklist.csv",
            trade_date=datetime(2024, 3, 1),
        )

        assert result["meta"]["trade_date"] == "2024-03-01"

    def test_none_trade_date_gives_none_in_meta(self, tmp_path):
        """When trade_date is None, meta.trade_date is None."""
        _write_stocklist_csv(tmp_path)

        result = compute_j_industry_distribution(
            data_dir=tmp_path,
            stocklist_path=tmp_path / "stocklist.csv",
            trade_date=None,
        )

        assert result["meta"]["trade_date"] is None

    def test_result_structure(self, tmp_path):
        """Return value always has 'meta' and 'industry_counts' keys."""
        _write_stocklist_csv(tmp_path)

        result = compute_j_industry_distribution(
            data_dir=tmp_path,
            stocklist_path=tmp_path / "stocklist.csv",
        )

        assert "meta" in result
        assert "industry_counts" in result

    def test_meta_contains_j_threshold(self, tmp_path):
        """meta.j_threshold reflects the supplied threshold value."""
        _write_stocklist_csv(tmp_path)

        result = compute_j_industry_distribution(
            data_dir=tmp_path,
            stocklist_path=tmp_path / "stocklist.csv",
            j_threshold=20.0,
        )

        assert result["meta"]["j_threshold"] == 20.0

    @patch("SectorShift.load_data")
    @patch("SectorShift.compute_kdj")
    def test_stocks_with_low_j_are_counted(
        self, mock_compute_kdj, mock_load_data, tmp_path
    ):
        """Stocks whose J(日) < threshold appear in the industry_counts."""
        # Arrange: write one CSV file so _list_codes_from_data_dir picks it up
        csv_path = tmp_path / "000001.csv"
        df_ohlcv = _make_ohlcv_df()
        df_ohlcv.to_csv(csv_path, index=False)

        # Stocklist with a matching entry
        stocklist_path = _write_stocklist_csv(
            tmp_path, codes=["000001"], industries=["银行"]
        )

        # load_data returns our mock frame
        mock_load_data.return_value = {"000001": df_ohlcv}

        # compute_kdj returns a DataFrame whose last J value is below threshold
        kdj_df = pd.DataFrame({"K": [30.0], "D": [25.0], "J": [5.0]})
        mock_compute_kdj.return_value = kdj_df

        result = compute_j_industry_distribution(
            data_dir=tmp_path,
            stocklist_path=stocklist_path,
            j_threshold=15.0,
        )

        assert result["meta"]["selected_count"] == 1
        assert len(result["industry_counts"]) == 1
        assert result["industry_counts"][0]["行业"] == "银行"
        assert result["industry_counts"][0]["股票数"] == 1

    @patch("SectorShift.load_data")
    @patch("SectorShift.compute_kdj")
    def test_stocks_with_high_j_excluded(
        self, mock_compute_kdj, mock_load_data, tmp_path
    ):
        """Stocks whose J(日) >= threshold are excluded from counts."""
        csv_path = tmp_path / "000001.csv"
        df_ohlcv = _make_ohlcv_df()
        df_ohlcv.to_csv(csv_path, index=False)

        stocklist_path = _write_stocklist_csv(
            tmp_path, codes=["000001"], industries=["银行"]
        )

        mock_load_data.return_value = {"000001": df_ohlcv}

        # J value above threshold
        kdj_df = pd.DataFrame({"K": [80.0], "D": [75.0], "J": [90.0]})
        mock_compute_kdj.return_value = kdj_df

        result = compute_j_industry_distribution(
            data_dir=tmp_path,
            stocklist_path=stocklist_path,
            j_threshold=15.0,
        )

        assert result["meta"]["selected_count"] == 0
        assert result["industry_counts"] == []

    @patch("SectorShift.load_data")
    @patch("SectorShift.compute_kdj")
    def test_no_date_column_treated_as_nan(
        self, mock_compute_kdj, mock_load_data, tmp_path
    ):
        """A DataFrame without a 'date' column results in NaN J, which is excluded."""
        csv_path = tmp_path / "000001.csv"
        df_no_date = pd.DataFrame(
            {"open": [10.0], "high": [11.0], "low": [9.0], "close": [10.5]}
        )
        df_no_date.to_csv(csv_path, index=False)

        stocklist_path = _write_stocklist_csv(
            tmp_path, codes=["000001"], industries=["银行"]
        )

        mock_load_data.return_value = {"000001": df_no_date}

        result = compute_j_industry_distribution(
            data_dir=tmp_path,
            stocklist_path=stocklist_path,
            j_threshold=100.0,
        )

        assert result["meta"]["selected_count"] == 0

    @patch("SectorShift.load_data")
    @patch("SectorShift.compute_kdj")
    def test_trade_date_filters_future_rows(
        self, mock_compute_kdj, mock_load_data, tmp_path
    ):
        """When trade_date is set, only rows on or before that date are used."""
        csv_path = tmp_path / "000001.csv"
        df_ohlcv = _make_ohlcv_df(n=60, start="2024-06-01")
        df_ohlcv.to_csv(csv_path, index=False)

        stocklist_path = _write_stocklist_csv(
            tmp_path, codes=["000001"], industries=["银行"]
        )

        # All data is after the trade_date (2024-01-01), so the filtered df is
        # empty → NaN J value → excluded from results
        mock_load_data.return_value = {"000001": df_ohlcv}
        mock_compute_kdj.return_value = pd.DataFrame(
            {"K": [10.0], "D": [5.0], "J": [1.0]}
        )

        result = compute_j_industry_distribution(
            data_dir=tmp_path,
            stocklist_path=stocklist_path,
            j_threshold=15.0,
            trade_date="20240101",
        )

        # No rows pass the date filter → selected_count == 0
        assert result["meta"]["selected_count"] == 0
