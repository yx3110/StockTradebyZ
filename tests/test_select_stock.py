"""Unit tests for select_stock.py utility functions.

Covers:
  - load_data        : valid CSVs, missing files, empty directory / code list
  - load_config      : list format, dict with selectors key, single object,
                       invalid (non-existent) file
  - instantiate_selector : valid class, missing class field, bad class name
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from select_stock import instantiate_selector, load_config, load_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(directory: Path, code: str, rows: int = 5) -> None:
    """Write a minimal OHLCV CSV file for *code* into *directory*."""
    dates = pd.bdate_range(start="2024-01-01", periods=rows)
    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": [10.0] * rows,
            "high": [11.0] * rows,
            "low": [9.0] * rows,
            "close": [10.5] * rows,
            "volume": [1_000_000] * rows,
        }
    )
    df.to_csv(directory / f"{code}.csv", index=False)


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------


class TestLoadData:
    """Tests for the load_data function."""

    def test_valid_csvs_returns_frames(self, tmp_path):
        """load_data returns a dict entry for each existing CSV."""
        _write_csv(tmp_path, "000001")
        _write_csv(tmp_path, "000002")

        result = load_data(tmp_path, ["000001", "000002"])

        assert set(result.keys()) == {"000001", "000002"}
        assert isinstance(result["000001"], pd.DataFrame)
        assert isinstance(result["000002"], pd.DataFrame)

    def test_valid_csv_has_date_column(self, tmp_path):
        """load_data parses the 'date' column as datetime."""
        _write_csv(tmp_path, "000001")

        result = load_data(tmp_path, ["000001"])

        assert pd.api.types.is_datetime64_any_dtype(result["000001"]["date"])

    def test_valid_csv_sorted_by_date(self, tmp_path):
        """load_data returns rows sorted ascending by date."""
        _write_csv(tmp_path, "000001", rows=10)

        result = load_data(tmp_path, ["000001"])
        dates = result["000001"]["date"]

        assert dates.is_monotonic_increasing

    def test_missing_file_skipped(self, tmp_path):
        """load_data skips codes whose CSV does not exist and logs a warning."""
        _write_csv(tmp_path, "000001")

        result = load_data(tmp_path, ["000001", "999999"])

        assert "000001" in result
        assert "999999" not in result

    def test_all_files_missing_returns_empty(self, tmp_path):
        """load_data returns an empty dict when no requested CSV exists."""
        result = load_data(tmp_path, ["999998", "999999"])

        assert result == {}

    def test_empty_code_list_returns_empty(self, tmp_path):
        """load_data returns an empty dict when the codes iterable is empty."""
        _write_csv(tmp_path, "000001")

        result = load_data(tmp_path, [])

        assert result == {}

    def test_empty_directory_returns_empty(self, tmp_path):
        """load_data returns empty dict when the data directory has no matching CSVs."""
        result = load_data(tmp_path, ["000001"])

        assert result == {}


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Tests for the load_config function."""

    # --- valid formats ---

    def test_list_format(self, tmp_path):
        """load_config accepts a JSON array at the top level."""
        cfg_path = tmp_path / "configs.json"
        cfg_path.write_text(
            json.dumps([{"class": "BBIKDJSelector"}, {"class": "SuperB1Selector"}]),
            encoding="utf-8",
        )

        result = load_config(cfg_path)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["class"] == "BBIKDJSelector"
        assert result[1]["class"] == "SuperB1Selector"

    def test_dict_with_selectors_key(self, tmp_path):
        """load_config unwraps a dict that contains a 'selectors' key."""
        cfg_path = tmp_path / "configs.json"
        cfg_path.write_text(
            json.dumps({"selectors": [{"class": "BBIKDJSelector"}]}),
            encoding="utf-8",
        )

        result = load_config(cfg_path)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["class"] == "BBIKDJSelector"

    def test_single_object_wrapped_in_list(self, tmp_path):
        """load_config wraps a lone dict (no 'selectors' key) into a list."""
        cfg_path = tmp_path / "configs.json"
        cfg_path.write_text(
            json.dumps({"class": "PeakKDJSelector", "params": {}}),
            encoding="utf-8",
        )

        result = load_config(cfg_path)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["class"] == "PeakKDJSelector"

    # --- invalid / edge-case formats ---

    def test_nonexistent_file_exits(self, tmp_path):
        """load_config calls sys.exit(1) when the config file does not exist."""
        missing = tmp_path / "no_such_file.json"

        with pytest.raises(SystemExit) as exc_info:
            load_config(missing)

        assert exc_info.value.code == 1

    def test_empty_list_exits(self, tmp_path):
        """load_config calls sys.exit(1) when the JSON is an empty list."""
        cfg_path = tmp_path / "configs.json"
        cfg_path.write_text(json.dumps([]), encoding="utf-8")

        with pytest.raises(SystemExit) as exc_info:
            load_config(cfg_path)

        assert exc_info.value.code == 1

    def test_dict_with_empty_selectors_exits(self, tmp_path):
        """load_config calls sys.exit(1) when 'selectors' list is empty."""
        cfg_path = tmp_path / "configs.json"
        cfg_path.write_text(json.dumps({"selectors": []}), encoding="utf-8")

        with pytest.raises(SystemExit) as exc_info:
            load_config(cfg_path)

        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# instantiate_selector
# ---------------------------------------------------------------------------


class TestInstantiateSelector:
    """Tests for the instantiate_selector function."""

    def test_valid_class_returns_tuple(self):
        """instantiate_selector returns (alias, instance) for a known Selector class."""
        cfg = {"class": "BBIKDJSelector", "params": {}}

        alias, obj = instantiate_selector(cfg)

        assert alias == "BBIKDJSelector"
        assert obj.__class__.__name__ == "BBIKDJSelector"

    def test_alias_overrides_class_name(self):
        """instantiate_selector uses 'alias' key when present."""
        cfg = {"class": "BBIKDJSelector", "alias": "my_selector", "params": {}}

        alias, _ = instantiate_selector(cfg)

        assert alias == "my_selector"

    def test_missing_class_field_raises_value_error(self):
        """instantiate_selector raises ValueError when 'class' is absent."""
        cfg = {"params": {}}

        with pytest.raises(ValueError, match="缺少 class 字段"):
            instantiate_selector(cfg)

    def test_empty_class_field_raises_value_error(self):
        """instantiate_selector raises ValueError when 'class' is an empty string."""
        cfg = {"class": "", "params": {}}

        with pytest.raises(ValueError, match="缺少 class 字段"):
            instantiate_selector(cfg)

    def test_nonexistent_class_raises_import_error(self):
        """instantiate_selector raises ImportError for an unknown class name."""
        cfg = {"class": "NoSuchSelectorClass", "params": {}}

        with pytest.raises(ImportError, match="无法加载 Selector.NoSuchSelectorClass"):
            instantiate_selector(cfg)

    def test_params_forwarded_to_constructor(self):
        """instantiate_selector passes 'params' dict as keyword args to the class."""
        cfg = {
            "class": "BBIKDJSelector",
            "params": {"j_threshold": -10, "max_window": 120},
        }

        _, obj = instantiate_selector(cfg)

        assert obj.j_threshold == -10
        assert obj.max_window == 120

    def test_default_params_when_none_provided(self):
        """instantiate_selector uses class defaults when 'params' key is absent."""
        cfg = {"class": "BBIKDJSelector"}

        _, obj = instantiate_selector(cfg)

        # Default value defined in BBIKDJSelector.__init__
        assert obj.j_threshold == -5
