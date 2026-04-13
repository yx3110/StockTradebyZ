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
