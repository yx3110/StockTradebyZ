"""Unit tests for Purge Leakage Audit scripts."""
from __future__ import annotations

import json
import subprocess
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
from analyze_purge_leakage import render_report


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

    def test_nan_inputs_return_na(self):
        """NaN in baseline or delta → N/A (not fall through to RED)."""
        import math
        assert "N/A" in classify_verdict(math.nan, 0.05)
        assert "N/A" in classify_verdict(0.06, math.nan)
        assert "N/A" in classify_verdict(math.nan, math.nan)


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

    def test_baseline_too_low_yields_na_verdict(self):
        """Low baseline IC (< LOW_IC_CUTOFF=0.005) → N/A verdict even if delta computable."""
        runs = {
            ("ng_weak", 15): {"per_label_mean_oos_ic": {
                "label_3d": 0.003,  # below 0.005 threshold
                "label_5d": 0.06,
                "label_10d": None,
                "label_15d": 0.0,
            }},
            ("ng_weak", 30): {"per_label_mean_oos_ic": {
                "label_3d": 0.002,
                "label_5d": 0.055,
                "label_10d": None,
                "label_15d": 0.0,
            }},
        }
        rows = compute_delta_rows(runs)
        row_3d = next(r for r in rows if r["label"] == "3d")
        # delta_pct computes fine (baseline > 0), but verdict should be N/A due to low baseline
        assert row_3d["baseline_ic"] == 0.003
        assert "N/A" in row_3d["verdict"]
        # label_5d has healthy baseline → normal verdict
        row_5d = next(r for r in rows if r["label"] == "5d")
        assert "N/A" not in row_5d["verdict"]


class TestRenderReport:
    def test_contains_expected_sections(self):
        rows = [
            {"version": "ng1.0.1", "label": "5d", "baseline_ic": 0.07,
             "control_ic": 0.065, "delta_abs": 0.005, "delta_pct": 0.0714,
             "verdict": "🟢 GREEN"},
            {"version": "ng1.0.1", "label": "15d", "baseline_ic": 0.06,
             "control_ic": 0.02, "delta_abs": 0.04, "delta_pct": 0.6667,
             "verdict": "🔴 RED"},
        ]
        runs = {
            ("ng1.0.1", 15): {"run_id": "ng1.0.1_purge15", "elapsed_seconds": 2400,
                              "returncode": 0, "n_windows": 3},
            ("ng1.0.1", 30): {"run_id": "ng1.0.1_purge30", "elapsed_seconds": 2500,
                              "returncode": 0, "n_windows": 3},
        }
        body = render_report(rows, runs, audit_date="20260413")
        assert "Purge Leakage Audit" in body
        assert "ng1.0.1" in body
        assert "🟢 GREEN" in body
        assert "🔴 RED" in body
        assert "7.1%" in body or "+7.1" in body
        assert "66.7%" in body or "+66.7" in body
        assert "GREEN: 1" in body
        assert "RED: 1" in body

    def test_handles_none_values_gracefully(self):
        rows = [{
            "version": "ng106", "label": "3d",
            "baseline_ic": None, "control_ic": None,
            "delta_abs": None, "delta_pct": None,
            "verdict": "⚪ N/A",
        }]
        runs = {("ng106", 15): {"run_id": "ng106_purge15", "elapsed_seconds": 0,
                                 "returncode": 1, "n_windows": 0}}
        body = render_report(rows, runs, audit_date="20260413")
        assert "⚪ N/A" in body
        assert "ng106" in body
        assert "—" in body


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from run_purge_experiment import _snapshot_wf_summaries, _find_new_wf_summary


class TestSnapshotWfSummaries:
    def test_empty_dir(self, tmp_path):
        assert _snapshot_wf_summaries(tmp_path) == {}

    def test_nonexistent_root(self, tmp_path):
        assert _snapshot_wf_summaries(tmp_path / "nope") == {}

    def test_finds_nested_summaries(self, tmp_path):
        (tmp_path / "sub1").mkdir()
        (tmp_path / "sub1" / "wf_summary.json").write_text("{}")
        (tmp_path / "sub2").mkdir()
        (tmp_path / "sub2" / "wf_summary.json").write_text("{}")
        snap = _snapshot_wf_summaries(tmp_path)
        assert len(snap) == 2


class TestFindNewWfSummary:
    def test_new_file(self, tmp_path):
        pre = {}
        new_path = (tmp_path / "sub" / "wf_summary.json").resolve()
        post = {new_path: 1234567890.0}
        assert _find_new_wf_summary(pre, post) == new_path

    def test_updated_mtime(self, tmp_path):
        p = (tmp_path / "sub" / "wf_summary.json").resolve()
        pre = {p: 1000.0}
        post = {p: 2000.0}
        assert _find_new_wf_summary(pre, post) == p

    def test_no_change(self, tmp_path):
        p = (tmp_path / "sub" / "wf_summary.json").resolve()
        pre = {p: 1000.0}
        post = {p: 1000.0}
        assert _find_new_wf_summary(pre, post) is None

    def test_multiple_changes_picks_latest_mtime(self, tmp_path):
        p1 = (tmp_path / "s1" / "wf_summary.json").resolve()
        p2 = (tmp_path / "s2" / "wf_summary.json").resolve()
        pre = {}
        post = {p1: 1000.0, p2: 2000.0}
        assert _find_new_wf_summary(pre, post) == p2


class TestRunSingleTimeout:
    def test_timeout_writes_trainer_log_and_run_json(self, tmp_path, monkeypatch):
        """TimeoutExpired should still produce trainer.log and run.json with returncode=-1."""
        import run_purge_experiment

        # Prepare: create a fake model dir so pre_snapshot works
        fake_models = tmp_path / "trained_models"
        fake_models.mkdir()

        # Patch TRAINED_MODELS_DIR to tmp_path
        monkeypatch.setattr(run_purge_experiment, "TRAINED_MODELS_DIR", fake_models)

        # Patch TRAIN_TIMEOUT_SECONDS to a tiny value so real Popen times out fast
        monkeypatch.setattr(run_purge_experiment, "TRAIN_TIMEOUT_SECONDS", 1)

        # Patch the subprocess call to run a command that sleeps longer than timeout
        # We invoke a trivial python subprocess that never exits
        real_popen = subprocess.Popen

        def fake_popen(cmd, **kwargs):
            # Replace command with a sleep that will be killed
            fake_cmd = [sys.executable, "-c", "import time; time.sleep(10)"]
            return real_popen(fake_cmd, **kwargs)

        monkeypatch.setattr(run_purge_experiment.subprocess, "Popen", fake_popen)

        audit_dir = tmp_path / "audit"
        result = run_purge_experiment.run_single(
            version="ng1.0.1",
            purge_days=15,
            audit_dir=audit_dir,
            force=False,
            extra_args=None,
        )

        # Verify run.json and trainer.log both exist
        run_dir = audit_dir / "ng1.0.1_purge15"
        assert (run_dir / "run.json").exists(), "run.json should be written even on timeout"
        assert (run_dir / "trainer.log").exists(), "trainer.log should be written even on timeout"

        # Verify run.json contents
        assert result["returncode"] == -1
        assert result["version"] == "ng1.0.1"
        assert result["purge_days"] == 15
        # oos_ics stays at None defaults since no wf_summary
        assert result["per_label_mean_oos_ic"]["label_5d"] is None

        # Verify trainer.log contains timeout marker
        log_body = (run_dir / "trainer.log").read_text()
        assert "TIMEOUT" in log_body
