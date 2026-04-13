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


def extract_per_label_mean_oos_ic(wf_summary: dict) -> dict[str, float | None]:
    """Pull {label_Xd: mean_oos_ic} from wf_summary['aggregate'].

    Missing labels return None (not 0 — distinguishes "training produced
    this label" from "label missing").
    """
    agg = wf_summary.get("aggregate", {})
    return {f"label_{d}": agg.get(f"label_{d}_mean_ic") for d in LABELS}


def extract_n_windows(wf_summary: dict) -> int:
    """Pull window count. Prefer top-level n_windows, fall back to len(wf_windows)."""
    if "n_windows" in wf_summary:
        return int(wf_summary["n_windows"])
    return len(wf_summary.get("wf_windows", []))


def load_runs(audit_dir: Path) -> dict[tuple[str, int], dict]:
    """Scan audit_dir/*/run.json and organize by (version, purge_days) key."""
    runs = {}
    for run_dir in sorted(audit_dir.iterdir()) if audit_dir.exists() else []:
        if not run_dir.is_dir():
            continue
        run_path = run_dir / "run.json"
        if not run_path.exists():
            continue
        data = json.loads(run_path.read_text())
        key = (data["version"], int(data["purge_days"]))
        runs[key] = data
    return runs


def compute_delta_rows(runs: dict[tuple[str, int], dict]) -> list[dict]:
    """For each (version, label), compare baseline (purge=15) vs control (purge=30).

    Returns list of dicts with keys:
      version, label, baseline_ic, control_ic, delta_abs, delta_pct, verdict
    """
    versions_found = sorted({v for (v, _) in runs.keys()})
    rows = []
    for version in versions_found:
        baseline = runs.get((version, 15), {}).get("per_label_mean_oos_ic", {})
        control = runs.get((version, 30), {}).get("per_label_mean_oos_ic", {})
        for label in LABELS:
            b = baseline.get(f"label_{label}")
            c = control.get(f"label_{label}")
            if b is not None and c is not None and b > 0:
                delta_abs = b - c
                delta_pct = delta_abs / b
            else:
                delta_abs = None
                delta_pct = None
            rows.append({
                "version": version,
                "label": label,
                "baseline_ic": b,
                "control_ic": c,
                "delta_abs": delta_abs,
                "delta_pct": delta_pct,
                "verdict": classify_verdict(b, delta_pct),
            })
    return rows
