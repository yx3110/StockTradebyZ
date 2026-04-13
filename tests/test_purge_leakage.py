"""Unit tests for Purge Leakage Audit scripts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make scripts importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_purge_leakage import classify_verdict
from analyze_purge_leakage import (
    extract_per_label_mean_oos_ic,
    extract_n_windows,
)
from analyze_purge_leakage import load_runs, compute_delta_rows


# --- classify_verdict ---

class TestClassifyVerdict:
    def test_baseline_too_low_returns_na(self):
        assert "N/A" in classify_verdict(0.004, 0.05)
        assert "N/A" in classify_verdict(0.0049, 0.50)
        assert "N/A" in classify_verdict(None, 0.05)

    def test_delta_none_returns_na(self):
        assert "N/A" in classify_verdict(0.06, None)

    def test_green(self):
        assert "GREEN" in classify_verdict(0.06, 0.05)
        assert "GREEN" in classify_verdict(0.06, 0.0)
        assert "GREEN" in classify_verdict(0.06, -0.05)  # control > baseline
        assert "GREEN" in classify_verdict(0.06, 0.099)

    def test_yellow(self):
        assert "YELLOW" in classify_verdict(0.06, 0.10)
        assert "YELLOW" in classify_verdict(0.06, 0.29)

    def test_red(self):
        assert "RED" in classify_verdict(0.06, 0.30)
        assert "RED" in classify_verdict(0.06, 1.0)
        assert "RED" in classify_verdict(0.06, 0.999)


class TestExtractPerLabelMeanOOSIC:
    def test_happy_path(self):
        wf_summary = {
            "aggregate": {
                "label_3d_mean_ic": 0.0567,
                "label_5d_mean_ic": 0.0693,
                "label_10d_mean_ic": 0.0852,
                "label_15d_mean_ic": 0.0868,
                "label_3d_std_ic": 0.01,
                "other_key": "ignored",
            }
        }
        result = extract_per_label_mean_oos_ic(wf_summary)
        assert result == {
            "label_3d": 0.0567,
            "label_5d": 0.0693,
            "label_10d": 0.0852,
            "label_15d": 0.0868,
        }

    def test_missing_label_returns_none(self):
        wf_summary = {"aggregate": {"label_3d_mean_ic": 0.05}}
        result = extract_per_label_mean_oos_ic(wf_summary)
        assert result == {"label_3d": 0.05, "label_5d": None,
                          "label_10d": None, "label_15d": None}

    def test_missing_aggregate(self):
        assert extract_per_label_mean_oos_ic({}) == {
            "label_3d": None, "label_5d": None,
            "label_10d": None, "label_15d": None,
        }


class TestExtractNWindows:
    def test_happy_path(self):
        assert extract_n_windows({"n_windows": 3}) == 3

    def test_missing_returns_zero(self):
        assert extract_n_windows({}) == 0

    def test_fallback_to_wf_windows_list(self):
        """If n_windows not present, fall back to len(wf_windows)."""
        assert extract_n_windows({"wf_windows": [{}, {}, {}]}) == 3


class TestLoadRuns:
    def test_loads_multiple_run_jsons(self, tmp_path):
        for version, purge in [("ng1.0.1", 15), ("ng1.0.1", 30), ("ng106", 15)]:
            run_dir = tmp_path / f"{version}_purge{purge}"
            run_dir.mkdir()
            (run_dir / "run.json").write_text(json.dumps({
                "run_id": f"{version}_purge{purge}",
                "version": version,
                "purge_days": purge,
                "per_label_mean_oos_ic": {"label_5d": 0.05},
                "n_windows": 3,
                "returncode": 0,
            }))

        runs = load_runs(tmp_path)
        assert len(runs) == 3
        assert runs[("ng1.0.1", 15)]["per_label_mean_oos_ic"]["label_5d"] == 0.05
        assert runs[("ng106", 15)]["version"] == "ng106"

    def test_empty_dir_returns_empty_dict(self, tmp_path):
        assert load_runs(tmp_path) == {}

    def test_skips_dirs_without_run_json(self, tmp_path):
        (tmp_path / "empty_subdir").mkdir()
        (tmp_path / "ng106_purge15").mkdir()
        (tmp_path / "ng106_purge15" / "run.json").write_text(json.dumps({
            "version": "ng106", "purge_days": 15,
            "per_label_mean_oos_ic": {}, "n_windows": 0, "returncode": 0,
        }))
        runs = load_runs(tmp_path)
        assert len(runs) == 1


class TestComputeDeltaRows:
    def test_happy_path_single_version(self):
        runs = {
            ("ng1.0.1", 15): {"per_label_mean_oos_ic": {
                "label_3d": 0.060, "label_5d": 0.070,
                "label_10d": 0.080, "label_15d": 0.050,
            }},
            ("ng1.0.1", 30): {"per_label_mean_oos_ic": {
                "label_3d": 0.058, "label_5d": 0.060,
                "label_10d": 0.060, "label_15d": 0.020,
            }},
        }
        rows = compute_delta_rows(runs)
        assert len(rows) == 4
        row_3d = next(r for r in rows if r["label"] == "3d")
        assert abs(row_3d["baseline_ic"] - 0.060) < 1e-9
        assert abs(row_3d["control_ic"] - 0.058) < 1e-9
        assert abs(row_3d["delta_pct"] - (0.002 / 0.060)) < 1e-9
        assert "GREEN" in row_3d["verdict"]
        row_15d = next(r for r in rows if r["label"] == "15d")
        assert abs(row_15d["delta_pct"] - 0.6) < 1e-9
        assert "RED" in row_15d["verdict"]

    def test_missing_control_yields_na(self):
        runs = {
            ("ng1.0.1", 15): {"per_label_mean_oos_ic": {"label_5d": 0.06}},
        }
        rows = compute_delta_rows(runs)
        row_5d = next(r for r in rows if r["version"] == "ng1.0.1" and r["label"] == "5d")
        assert row_5d["control_ic"] is None
        assert row_5d["delta_pct"] is None
        assert "N/A" in row_5d["verdict"]

    def test_multiple_versions(self):
        runs = {
            ("ng1.0.1", 15): {"per_label_mean_oos_ic": {"label_5d": 0.06}},
            ("ng1.0.1", 30): {"per_label_mean_oos_ic": {"label_5d": 0.05}},
            ("ng106",   15): {"per_label_mean_oos_ic": {"label_5d": 0.07}},
            ("ng106",   30): {"per_label_mean_oos_ic": {"label_5d": 0.065}},
        }
        rows = compute_delta_rows(runs)
        assert len(rows) == 8
        versions = {r["version"] for r in rows}
        assert versions == {"ng1.0.1", "ng106"}
