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


def _fmt_ic(ic: float | None) -> str:
    return "—" if ic is None else f"{ic:.4f}"


def _fmt_pct(pct: float | None) -> str:
    return "—" if pct is None else f"{pct*100:+.1f}%"


def _fmt_minutes(seconds: float | int | None) -> str:
    if seconds is None:
        return "—"
    return f"{seconds/60:.1f} min"


def render_report(rows: list[dict], runs: dict, audit_date: str) -> str:
    """Render Markdown report from computed rows and run metadata."""
    from collections import Counter
    counts = Counter()
    for r in rows:
        v = r["verdict"]
        if "GREEN" in v:
            counts["GREEN"] += 1
        elif "YELLOW" in v:
            counts["YELLOW"] += 1
        elif "RED" in v:
            counts["RED"] += 1
        else:
            counts["NA"] += 1
    total = len(rows)

    lines = [
        f"# Purge Leakage Audit — {audit_date}",
        "",
        "实验配置: 3 versions × 2 purge × 4 labels = 12 比较点",
        "- Baseline: `purge_days=15` (当前 ng_trainer 默认)",
        "- Control:  `purge_days=30` (double gap)",
        "- WF: 3 windows (default), `--start-date 2020-01-01`",
        "",
        "## 全量判决表",
        "",
        "| 版本 | 标签 | baseline IC (p=15) | control IC (p=30) | Δ% | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['version']} | {r['label']} | "
            f"{_fmt_ic(r['baseline_ic'])} | {_fmt_ic(r['control_ic'])} | "
            f"{_fmt_pct(r['delta_pct'])} | {r['verdict']} |"
        )

    lines += [
        "",
        "## 汇总",
        "",
        f"- 🟢 GREEN: {counts['GREEN']} / {total}",
        f"- 🟡 YELLOW: {counts['YELLOW']} / {total}",
        f"- 🔴 RED: {counts['RED']} / {total}",
        f"- ⚪ N/A: {counts['NA']} / {total}",
        "",
        "## 发现 / 建议",
        "",
    ]

    red_rows = [r for r in rows if "RED" in r["verdict"]]
    yellow_rows = [r for r in rows if "YELLOW" in r["verdict"]]
    if red_rows:
        for r in red_rows:
            lines.append(
                f"- 🔴 `{r['version']}` 的 `label_{r['label']}` 在 p=30 时 IC "
                f"下降 {_fmt_pct(r['delta_pct'])} → **`purge=15` 不够**，"
                f"建议默认值改为 ≥30，或进一步跑 purge=60 验证饱和点"
            )
    if yellow_rows:
        for r in yellow_rows:
            lines.append(
                f"- 🟡 `{r['version']}` 的 `label_{r['label']}` 有轻度泄漏 "
                f"({_fmt_pct(r['delta_pct'])}) → 建议观察，下次升级训练 pipeline "
                f"时考虑 purge=30"
            )
    if not red_rows and not yellow_rows:
        lines.append("- 所有比较点均 🟢 GREEN：当前 `purge_days=15` 足够，无显著泄漏")

    lines += [
        "",
        "## 运行元数据",
        "",
        "| run_id | 耗时 | returncode | WF windows |",
        "|---|---|---|---|",
    ]
    for (version, purge), run in sorted(runs.items()):
        lines.append(
            f"| {run.get('run_id', '—')} | "
            f"{_fmt_minutes(run.get('elapsed_seconds'))} | "
            f"{run.get('returncode', '—')} | "
            f"{run.get('n_windows', '—')} |"
        )

    lines.append("")
    return "\n".join(lines)
