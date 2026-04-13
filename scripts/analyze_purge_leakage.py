"""Purge Leakage Analyzer — 读 run.json × 算 Δ% × 写 REPORT.md.

Design doc: docs/superpowers/specs/2026-04-12-purge-leakage-audit-design.md
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_ROOT = PROJECT_ROOT / "reports"

VERSIONS = ['ng1.0.1', 'ng106', 'ng1.1.0']
PURGE_VALUES = [15, 30]
LABELS = ['3d', '5d', '10d', '15d']

GREEN_THRESHOLD = 0.10
RED_THRESHOLD = 0.30
LOW_IC_CUTOFF = 0.005


def classify_verdict(baseline_ic: float | None, delta_pct: float | None) -> str:
    """Classify leakage severity.

    Rules:
      baseline_ic None or < LOW_IC_CUTOFF → ⚪ N/A (baseline IC 过低)
      delta_pct None → ⚪ N/A
      delta_pct < GREEN_THRESHOLD → 🟢 GREEN
      GREEN_THRESHOLD <= delta_pct < RED_THRESHOLD → 🟡 YELLOW
      delta_pct >= RED_THRESHOLD → 🔴 RED
    (delta_pct < 0 means control IC ≥ baseline, treat as GREEN)
    """
    if baseline_ic is None or baseline_ic < LOW_IC_CUTOFF:
        return "⚪ N/A (baseline IC 过低)"
    if delta_pct is None:
        return "⚪ N/A"
    if delta_pct < GREEN_THRESHOLD:
        return "🟢 GREEN"
    if delta_pct < RED_THRESHOLD:
        return "🟡 YELLOW"
    return "🔴 RED"
