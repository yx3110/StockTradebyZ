# Purge Leakage Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 StockTradebyZ 仓库内实施一次性审计工具，量化当前默认 `purge_days=15` 是否足够，对 3 个 NG 版本 × 2 个 purge × 4 个标签做对照实验，产出 Markdown 判决表。

**Architecture:** Runner (`scripts/run_purge_experiment.py`) 用 subprocess 驱动 `ml_models/ng/ng_trainer.py --purge-days {P}` 两次 × 3 版本，把产出的 `wf_summary.json` 解析成 `run.json`；Analyzer (`scripts/analyze_purge_leakage.py`) 读 6 份 run.json 生成 `REPORT.md`。两个 CLI 互不 import，用 JSON 文件通信。

**Tech Stack:** Python 3, stdlib `subprocess` / `json` / `argparse` / `pathlib`，`pytest` 测试（StockTradebyZ 标准，不同于 EMT 的 plain-script 风格）。

**Spec:** `docs/superpowers/specs/2026-04-12-purge-leakage-audit-design.md`

---

## File Structure

| 文件 | 操作 | 职责 |
|---|---|---|
| `scripts/run_purge_experiment.py` | Create | Runner CLI：subprocess 驱动 ng_trainer，定位 wf_summary，写 run.json |
| `scripts/analyze_purge_leakage.py` | Create | Analyzer CLI：读 run.json × 6，算 Δ%，写 REPORT.md |
| `tests/test_purge_leakage.py` | Create | pytest 单测（analyzer 纯函数 + runner 辅助函数） |
| `reports/purge_audit_20260413/` | Runtime | 实验产出目录（gitignored？`.gitignore` 不拦 reports/，但产物不提交） |

### `wf_summary.json` 数据形状（外部输入，已确认）

```python
{
  "version": "ng1.0.1",
  "generated_at": "...",
  "n_windows": 3,
  "aggregate": {
    "label_3d_mean_ic":  0.0567,
    "label_3d_std_ic":   0.0107,
    "label_3d_icir":     0.593,
    "label_5d_mean_ic":  0.0693,
    "label_10d_mean_ic": 0.0852,
    "label_15d_mean_ic": 0.0868,
    ...
  },
  "wf_windows": [
    {"window_id": 0, "metrics": {"label_3d_ic": 0.040, "label_5d_ic": 0.058, ...}},
    {"window_id": 1, "metrics": {...}},
    {"window_id": 2, "metrics": {...}}
  ]
}
```

### `run.json` schema（runner 写，analyzer 读）

```python
{
  "run_id": "ng106_purge30",
  "version": "ng106",
  "purge_days": 30,
  "started_at": "2026-04-13T14:22:10",
  "elapsed_seconds": 2507.3,
  "returncode": 0,
  "wf_summary_path": "ml_models/trained_models/ng/wf_summary.json",
  "n_windows": 3,
  "per_label_mean_oos_ic": {
    "label_3d": 0.0612,
    "label_5d": 0.0583,
    "label_10d": 0.0551,
    "label_15d": 0.0498
  }
}
```

顶部常量（`scripts/run_purge_experiment.py` 和 `scripts/analyze_purge_leakage.py` 各自定义）：

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VERSIONS = ['ng1.0.1', 'ng106', 'ng1.1.0']
PURGE_VALUES = [15, 30]
LABELS = ['3d', '5d', '10d', '15d']
GREEN_THRESHOLD = 0.10
RED_THRESHOLD = 0.30
LOW_IC_CUTOFF = 0.005
TRAIN_TIMEOUT_SECONDS = 14400  # 4 hours per run
```

---

## Task 1: 项目骨架 + `classify_verdict` 纯函数（TDD）

**Files:**
- Create: `scripts/analyze_purge_leakage.py`（骨架 + `classify_verdict`）
- Create: `tests/test_purge_leakage.py`（骨架 + 5 个边界测试）

- [ ] **Step 1: 创建 `scripts/analyze_purge_leakage.py` 骨架**

```python
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
```

- [ ] **Step 2: 创建 `tests/test_purge_leakage.py` 骨架**

```python
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
```

- [ ] **Step 3: 运行测试验证通过**

Run:
```bash
cd /Users/yangxu/StockTradebyZ && python3 -m pytest tests/test_purge_leakage.py -v
```

Expected: 5 tests pass.

- [ ] **Step 4: 提交**

```bash
cd /Users/yangxu/StockTradebyZ && git add scripts/analyze_purge_leakage.py tests/test_purge_leakage.py && git commit -m "Purge Leakage Audit: 骨架 + classify_verdict 纯函数"
```

---

## Task 2: `_extract_per_label_mean_oos_ic` + `_extract_n_windows`（wf_summary 解析）

**Files:**
- Modify: `scripts/analyze_purge_leakage.py`（添加两个纯函数）
- Modify: `tests/test_purge_leakage.py`（添加测试）

注意：这两个函数用于从 wf_summary.json 提取需要的字段，但 runner 也会用到。为了单文件原则，放在 analyzer 里；runner 直接 import（见 Task 6）。

- [ ] **Step 1: 加入测试**

```python
# Append to tests/test_purge_leakage.py

from analyze_purge_leakage import (
    extract_per_label_mean_oos_ic,
    extract_n_windows,
)


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
```

- [ ] **Step 2: 运行验证失败**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest tests/test_purge_leakage.py -v`

Expected: `ImportError: cannot import name 'extract_per_label_mean_oos_ic'`

- [ ] **Step 3: 实现**

Append to `scripts/analyze_purge_leakage.py`:

```python
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
```

- [ ] **Step 4: 运行验证通过**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest tests/test_purge_leakage.py -v`

Expected: 11 tests pass (5 from Task 1 + 6 new).

- [ ] **Step 5: 提交**

```bash
cd /Users/yangxu/StockTradebyZ && git add scripts/analyze_purge_leakage.py tests/test_purge_leakage.py && git commit -m "Purge Leakage Audit: wf_summary 解析函数"
```

---

## Task 3: `load_runs` + `compute_delta_rows`（聚合逻辑）

**Files:**
- Modify: `scripts/analyze_purge_leakage.py`
- Modify: `tests/test_purge_leakage.py`

- [ ] **Step 1: 加入测试**

```python
# Append to tests/test_purge_leakage.py

from analyze_purge_leakage import load_runs, compute_delta_rows


class TestLoadRuns:
    def test_loads_multiple_run_jsons(self, tmp_path):
        # Simulate 3 run subdirs
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
        # 4 labels × 1 version = 4 rows
        assert len(rows) == 4
        # label_3d: baseline=0.06, control=0.058, delta=0.002/0.06 ≈ 3.3%
        row_3d = next(r for r in rows if r["label"] == "3d")
        assert abs(row_3d["baseline_ic"] - 0.060) < 1e-9
        assert abs(row_3d["control_ic"] - 0.058) < 1e-9
        assert abs(row_3d["delta_pct"] - (0.002 / 0.060)) < 1e-9
        assert "GREEN" in row_3d["verdict"]
        # label_15d: baseline=0.05, control=0.02, delta=0.03/0.05 = 60%
        row_15d = next(r for r in rows if r["label"] == "15d")
        assert abs(row_15d["delta_pct"] - 0.6) < 1e-9
        assert "RED" in row_15d["verdict"]

    def test_missing_control_yields_na(self):
        runs = {
            ("ng1.0.1", 15): {"per_label_mean_oos_ic": {"label_5d": 0.06}},
            # No purge=30 for ng1.0.1
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
        # 2 versions × 4 labels = 8 rows
        assert len(rows) == 8
        versions = {r["version"] for r in rows}
        assert versions == {"ng1.0.1", "ng106"}
```

- [ ] **Step 2: 运行验证失败**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest tests/test_purge_leakage.py -v`

Expected: ImportError.

- [ ] **Step 3: 实现**

Append to `scripts/analyze_purge_leakage.py`:

```python
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
```

- [ ] **Step 4: 验证通过**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest tests/test_purge_leakage.py -v`

Expected: 17 tests pass (11 prior + 6 new).

- [ ] **Step 5: 提交**

```bash
cd /Users/yangxu/StockTradebyZ && git add scripts/analyze_purge_leakage.py tests/test_purge_leakage.py && git commit -m "Purge Leakage Audit: load_runs + compute_delta_rows"
```

---

## Task 4: `render_report`（Markdown 输出）

**Files:**
- Modify: `scripts/analyze_purge_leakage.py`
- Modify: `tests/test_purge_leakage.py`

- [ ] **Step 1: 加入测试**

```python
# Append to tests/test_purge_leakage.py

from analyze_purge_leakage import render_report


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
        assert "7.1%" in body or "+7.1" in body  # delta_pct for label 5d
        assert "66.7%" in body or "+66.7" in body  # delta_pct for label 15d
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
        # None fields rendered as "—" not crash
        assert "—" in body
```

- [ ] **Step 2: 运行验证失败**

Expected: ImportError.

- [ ] **Step 3: 实现**

Append to `scripts/analyze_purge_leakage.py`:

```python
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
    # Count verdicts
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

    # Auto-generated narrative
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
```

- [ ] **Step 4: 验证通过**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest tests/test_purge_leakage.py -v`

Expected: 19 tests pass.

- [ ] **Step 5: 提交**

```bash
cd /Users/yangxu/StockTradebyZ && git add scripts/analyze_purge_leakage.py tests/test_purge_leakage.py && git commit -m "Purge Leakage Audit: render_report Markdown 生成"
```

---

## Task 5: Analyzer CLI 入口

**Files:**
- Modify: `scripts/analyze_purge_leakage.py`（添加 `main()`）

- [ ] **Step 1: 添加 CLI 主函数**

Append to `scripts/analyze_purge_leakage.py`:

```python
def _find_latest_audit_dir() -> Path | None:
    candidates = sorted(REPORTS_ROOT.glob("purge_audit_*"))
    return candidates[-1] if candidates else None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Purge Leakage Analyzer")
    parser.add_argument("--date", default=None,
                        help="YYYYMMDD; default = latest purge_audit_* dir")
    args = parser.parse_args()

    if args.date:
        audit_dir = REPORTS_ROOT / f"purge_audit_{args.date}"
    else:
        audit_dir = _find_latest_audit_dir()
    if audit_dir is None or not audit_dir.exists():
        print(f"ERROR: audit dir not found: {audit_dir}", flush=True)
        return 1

    runs = load_runs(audit_dir)
    if not runs:
        print(f"ERROR: no run.json files found in {audit_dir}", flush=True)
        return 1

    rows = compute_delta_rows(runs)
    date_str = audit_dir.name.replace("purge_audit_", "")
    body = render_report(rows, runs, audit_date=date_str)

    out_path = audit_dir / "REPORT.md"
    out_path.write_text(body, encoding="utf-8")
    print(f"Wrote report: {out_path}")
    print(f"Rows: {len(rows)} | Green: {sum(1 for r in rows if 'GREEN' in r['verdict'])} | "
          f"Yellow: {sum(1 for r in rows if 'YELLOW' in r['verdict'])} | "
          f"Red: {sum(1 for r in rows if 'RED' in r['verdict'])}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

- [ ] **Step 2: 手动 smoke test（用假数据）**

Run:
```bash
cd /Users/yangxu/StockTradebyZ && mkdir -p /tmp/purge_smoke/purge_audit_20260413/ng106_purge15 && \
echo '{"run_id":"ng106_purge15","version":"ng106","purge_days":15,"per_label_mean_oos_ic":{"label_3d":0.06,"label_5d":0.07,"label_10d":0.08,"label_15d":0.05},"n_windows":3,"returncode":0,"elapsed_seconds":2400}' > /tmp/purge_smoke/purge_audit_20260413/ng106_purge15/run.json && \
mkdir -p /tmp/purge_smoke/purge_audit_20260413/ng106_purge30 && \
echo '{"run_id":"ng106_purge30","version":"ng106","purge_days":30,"per_label_mean_oos_ic":{"label_3d":0.058,"label_5d":0.065,"label_10d":0.06,"label_15d":0.02},"n_windows":3,"returncode":0,"elapsed_seconds":2500}' > /tmp/purge_smoke/purge_audit_20260413/ng106_purge30/run.json && \
python3 -c "
import sys; sys.path.insert(0, 'scripts')
import analyze_purge_leakage
analyze_purge_leakage.REPORTS_ROOT = __import__('pathlib').Path('/tmp/purge_smoke')
sys.argv = ['analyze_purge_leakage.py', '--date', '20260413']
raise SystemExit(analyze_purge_leakage.main())
"
```

Expected:
- `Wrote report: /tmp/purge_smoke/purge_audit_20260413/REPORT.md`
- `Rows: 4 | Green: ... | Yellow: ... | Red: 1` (label_15d 从 0.05 → 0.02 是 60%，RED)

Run `cat /tmp/purge_smoke/purge_audit_20260413/REPORT.md` → 看格式对.

- [ ] **Step 3: 提交**

```bash
cd /Users/yangxu/StockTradebyZ && git add scripts/analyze_purge_leakage.py && git commit -m "Purge Leakage Audit: analyzer CLI 入口"
```

---

## Task 6: Runner 辅助函数 `_snapshot_wf_summaries` + `_find_new_wf_summary`

**Files:**
- Create: `scripts/run_purge_experiment.py`（骨架 + 2 个辅助函数）
- Modify: `tests/test_purge_leakage.py`（添加测试）

- [ ] **Step 1: 创建 `scripts/run_purge_experiment.py` 骨架**

```python
"""Purge Leakage Runner — 驱动 ng_trainer 跑 N 次并写 run.json.

Design doc: docs/superpowers/specs/2026-04-12-purge-leakage-audit-design.md
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Reuse analyzer's parsers to avoid duplication
from analyze_purge_leakage import (
    extract_per_label_mean_oos_ic,
    extract_n_windows,
    REPORTS_ROOT,
)

TRAINED_MODELS_DIR = PROJECT_ROOT / "ml_models" / "trained_models"
VERSIONS = ["ng1.0.1", "ng106", "ng1.1.0"]
PURGE_VALUES = [15, 30]
TRAIN_TIMEOUT_SECONDS = 14400  # 4h

logger = logging.getLogger(__name__)


def _snapshot_wf_summaries(model_root: Path) -> dict[Path, float]:
    """Return {path: mtime} for all wf_summary.json files under model_root."""
    snapshot = {}
    if not model_root.exists():
        return snapshot
    for p in model_root.rglob("wf_summary.json"):
        try:
            snapshot[p.resolve()] = p.stat().st_mtime
        except FileNotFoundError:
            continue
    return snapshot


def _find_new_wf_summary(pre: dict[Path, float], post: dict[Path, float]) -> Path | None:
    """Find a wf_summary.json that's either new or has updated mtime.

    Returns the candidate with the latest post mtime (in case of multiple).
    Returns None if no change detected.
    """
    changed = []
    for path, post_mtime in post.items():
        pre_mtime = pre.get(path)
        if pre_mtime is None or post_mtime > pre_mtime:
            changed.append((post_mtime, path))
    if not changed:
        return None
    changed.sort(reverse=True)  # latest mtime first
    return changed[0][1]
```

- [ ] **Step 2: 加入测试**

```python
# Append to tests/test_purge_leakage.py

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
```

- [ ] **Step 3: 运行验证通过**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest tests/test_purge_leakage.py -v`

Expected: 26 tests pass (19 prior + 7 new).

- [ ] **Step 4: 提交**

```bash
cd /Users/yangxu/StockTradebyZ && git add scripts/run_purge_experiment.py tests/test_purge_leakage.py && git commit -m "Purge Leakage Audit: runner 骨架 + wf_summary 定位辅助函数"
```

---

## Task 7: Runner 主循环 + CLI

**Files:**
- Modify: `scripts/run_purge_experiment.py`（添加 `run_single` + `main`）

这一步不做单测（subprocess 驱动的 ROI 太低）。实际验证在 Task 8（ng_trainer 兼容性）和 Task 9（fast-check smoke）。

- [ ] **Step 1: 添加 `run_single` + CLI**

Append to `scripts/run_purge_experiment.py`:

```python
def run_single(
    version: str,
    purge_days: int,
    audit_dir: Path,
    force: bool = False,
    extra_args: list[str] | None = None,
) -> dict:
    """Run one (version, purge) combination. Returns the run.json content.

    If output already exists and force=False, loads and returns it.
    """
    run_id = f"{version}_purge{purge_days}"
    run_dir = audit_dir / run_id
    run_json_path = run_dir / "run.json"

    if run_json_path.exists() and not force:
        logger.info("skip %s (already exists, use --force to rerun)", run_id)
        return json.loads(run_json_path.read_text())

    run_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot existing wf_summary files before training
    pre_snapshot = _snapshot_wf_summaries(TRAINED_MODELS_DIR)

    cmd = [
        sys.executable, str(PROJECT_ROOT / "ml_models" / "ng" / "ng_trainer.py"),
        "--version", version,
        "--purge-days", str(purge_days),
        "--start-date", "2020-01-01",
    ]
    if extra_args:
        cmd.extend(extra_args)

    logger.info("start %s: %s", run_id, " ".join(cmd))
    start_iso = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TRAIN_TIMEOUT_SECONDS,
            cwd=str(PROJECT_ROOT),
        )
        returncode = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        logger.error("TIMEOUT %s after %d seconds", run_id, TRAIN_TIMEOUT_SECONDS)
        returncode = -1
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\n---TIMEOUT after {TRAIN_TIMEOUT_SECONDS}s---"

    elapsed = time.time() - t0
    (run_dir / "trainer.log").write_text(
        stdout + "\n---STDERR---\n" + stderr, encoding="utf-8"
    )

    # Locate new wf_summary
    post_snapshot = _snapshot_wf_summaries(TRAINED_MODELS_DIR)
    new_wf_path = _find_new_wf_summary(pre_snapshot, post_snapshot)

    oos_ics = {"label_3d": None, "label_5d": None, "label_10d": None, "label_15d": None}
    n_windows = 0
    if new_wf_path and new_wf_path.exists():
        try:
            wf = json.loads(new_wf_path.read_text())
            oos_ics = extract_per_label_mean_oos_ic(wf)
            n_windows = extract_n_windows(wf)
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", new_wf_path, exc)

    run_json = {
        "run_id": run_id,
        "version": version,
        "purge_days": purge_days,
        "started_at": start_iso,
        "elapsed_seconds": elapsed,
        "returncode": returncode,
        "wf_summary_path": str(new_wf_path) if new_wf_path else None,
        "n_windows": n_windows,
        "per_label_mean_oos_ic": oos_ics,
    }
    run_json_path.write_text(json.dumps(run_json, indent=2, ensure_ascii=False))
    logger.info("done %s: rc=%d elapsed=%.1fmin oos_ics=%s",
                run_id, returncode, elapsed / 60, oos_ics)
    return run_json


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Purge Leakage Runner")
    parser.add_argument("--date", default=None,
                        help="YYYYMMDD; default = today")
    parser.add_argument("--force", action="store_true",
                        help="Rerun even if run.json exists")
    parser.add_argument("--fast-check", action="store_true",
                        help="Pass --fast-check to ng_trainer (2-window smoke)")
    parser.add_argument("--only-version", default=None,
                        help="Limit to one version (for debugging)")
    parser.add_argument("--only-purge", type=int, default=None,
                        help="Limit to one purge value (for debugging)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    date_str = args.date if args.date else datetime.now().strftime("%Y%m%d")
    audit_dir = REPORTS_ROOT / f"purge_audit_{date_str}"
    audit_dir.mkdir(parents=True, exist_ok=True)

    versions = [args.only_version] if args.only_version else VERSIONS
    purges = [args.only_purge] if args.only_purge is not None else PURGE_VALUES
    extra = ["--fast-check"] if args.fast_check else None

    total = len(versions) * len(purges)
    logger.info("Running %d combinations into %s", total, audit_dir)

    completed = []
    for i, version in enumerate(versions):
        for j, purge in enumerate(purges):
            idx = i * len(purges) + j + 1
            logger.info("[%d/%d] %s purge=%d", idx, total, version, purge)
            try:
                result = run_single(
                    version, purge, audit_dir,
                    force=args.force, extra_args=extra,
                )
                completed.append(result)
            except Exception as exc:
                logger.error("run_single %s/%d failed: %s", version, purge, exc, exc_info=True)

    print(f"\nDone. {len(completed)}/{total} runs completed.")
    print(f"Next: python3 scripts/analyze_purge_leakage.py --date {date_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: 语法检查**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -c "import sys; sys.path.insert(0,'scripts'); import run_purge_experiment"`

Expected: no errors.

Also: `cd /Users/yangxu/StockTradebyZ && python3 scripts/run_purge_experiment.py --help`

Expected: help text shows `--date`, `--force`, `--fast-check`, `--only-version`, `--only-purge`, `-v`.

- [ ] **Step 3: 提交**

```bash
cd /Users/yangxu/StockTradebyZ && git add scripts/run_purge_experiment.py && git commit -m "Purge Leakage Audit: runner 主循环 + CLI"
```

---

## Task 8: ng_trainer 兼容性验证

**Files:** 无代码改动；手动验证步骤。

这一步验证 `ng_trainer.py --version ng106` 之类的新格式版本是否被接受，以及 `--help` 中 `--purge-days` / `--version` 参数符合预期。如果不接受，需要调整 `VERSIONS` 常量或找兼容形式。

- [ ] **Step 1: 查看 ng_trainer 的 CLI 参数**

Run: `cd /Users/yangxu/StockTradebyZ && python3 ml_models/ng/ng_trainer.py --help 2>&1 | head -50`

Expected: 输出包含 `--purge-days`, `--version`, `--fast-check` 参数。

- [ ] **Step 2: 尝试各版本 fast-check（不实际训练完，只看能否启动）**

Run (后台、超短、能启动就立即 kill)：
```bash
cd /Users/yangxu/StockTradebyZ && for v in ng1.0.1 ng106 ng1.1.0; do
  echo "--- trying $v ---"
  timeout 20 python3 ml_models/ng/ng_trainer.py --version $v --purge-days 15 --fast-check --start-date 2020-01-01 2>&1 | head -8
  echo ""
done
```

Expected: 每个 `--- trying $v ---` 之后输出应显示 logger 启动（e.g. `"NG ng106 Walk-Forward Training"` 或 `"NG 1.0.1 Walk-Forward Training"`），**不应**看到 `argparse: error` 或 `invalid version`。20 秒超时会被 kill，这是期望的。

- [ ] **Step 3: 记录结果**

如果某个 version 被拒（e.g. `ng106` 不接受），需要调整 `scripts/run_purge_experiment.py` 的 `VERSIONS` 常量（比如改成 `ng1.0.6`）或查 `ml_models/ng/ng_schema.py` 找 version_map。

**如有调整**，改 `VERSIONS` 常量，git commit：
```bash
cd /Users/yangxu/StockTradebyZ && git add scripts/run_purge_experiment.py && git commit -m "Purge Leakage Audit: 调整 VERSIONS 常量以匹配 ng_trainer 接受的格式"
```

如果所有 version 都能启动，**不需要 commit**，直接进入 Task 9。

---

## Task 9: Fast-check 管道验证（smoke）

**Files:** 无代码改动；运行 runner 验证端到端。

- [ ] **Step 1: 跑一次 fast-check（单 version 单 purge）**

Run:
```bash
cd /Users/yangxu/StockTradebyZ && python3 scripts/run_purge_experiment.py \
  --only-version ng1.0.1 --only-purge 15 --fast-check -v --date smoke_20260413 2>&1 | tail -40
```

Expected:
- 日志显示 `[1/1] ng1.0.1 purge=15`
- 调用 ng_trainer 开始训练（可看到 `NG Walk-Forward Training` 日志）
- 训练完成（fast-check 2 个 WF 窗口，大概 10-30 分钟）
- 输出 `done ng1.0.1_purge15: rc=0 elapsed=X.Xmin oos_ics={...}`
- 生成 `reports/purge_audit_smoke_20260413/ng1.0.1_purge15/run.json` 和 `trainer.log`

- [ ] **Step 2: 检查 run.json**

Run: `cd /Users/yangxu/StockTradebyZ && cat reports/purge_audit_smoke_20260413/ng1.0.1_purge15/run.json`

Expected:
- `returncode: 0`
- `per_label_mean_oos_ic` 四个 label 都非 null
- `n_windows`: 2 (fast-check)
- `wf_summary_path` 不是 null

- [ ] **Step 3: 跑 analyzer（单 run，2 行输出，主要验证不 crash）**

Run:
```bash
cd /Users/yangxu/StockTradebyZ && python3 scripts/analyze_purge_leakage.py --date smoke_20260413 2>&1
```

Expected:
- `Wrote report: reports/purge_audit_smoke_20260413/REPORT.md`
- `Rows: 4 | Green: ? | Yellow: ? | Red: ?`（只有 baseline 没有 control，所以应该全部 N/A）

- [ ] **Step 4: 清理 smoke 数据**

Run: `cd /Users/yangxu/StockTradebyZ && rm -rf reports/purge_audit_smoke_20260413`

**如果 Step 1-3 有任何一步失败**：调试，定位问题，修复 runner 或 analyzer。smoke 数据的优点就是能反复跑。

**不 commit**（smoke 目录已删除，工作树应该干净）。

---

## Task 10: 正式实验 — 6 次训练（后台）

**Files:** 无代码改动；运行长耗时命令。

- [ ] **Step 1: 确认机器可用时间**

预估 6 次 × 默认 3 窗口 = 3-9 小时。检查机器是否会睡眠、是否有其他 heavy workload。如果是笔记本，建议插电源 + `caffeinate` 防睡眠：

```bash
# macOS: prevent sleep while running
caffeinate -i python3 scripts/run_purge_experiment.py -v 2>&1 | tee /tmp/purge_audit_run.log
```

**或者**后台运行：
```bash
cd /Users/yangxu/StockTradebyZ && nohup python3 scripts/run_purge_experiment.py -v > /tmp/purge_audit_run.log 2>&1 &
echo $! > /tmp/purge_audit.pid
```

监控：`tail -f /tmp/purge_audit_run.log`
终止（万一）：`kill $(cat /tmp/purge_audit.pid)`

- [ ] **Step 2: 启动实验（前台或后台均可）**

Run (**等待 3-9 小时**):
```bash
cd /Users/yangxu/StockTradebyZ && python3 scripts/run_purge_experiment.py -v 2>&1 | tee /tmp/purge_audit_run.log
```

Expected end-of-output:
```
[6/6] ng1.1.0 purge=30
...
done ng1.1.0_purge30: rc=0 ...
Done. 6/6 runs completed.
Next: python3 scripts/analyze_purge_leakage.py --date 20260413
```

**中断恢复**：如果中途 Ctrl+C 或超时，重新运行相同命令即可；已完成的 run.json 会被 skip，只继续跑剩下的。

**如有某次训练 rc != 0**：继续。run.json 会记录 returncode，analyzer 会把该行 verdict 标为 N/A；不阻塞整体。人工看 trainer.log 诊断。

- [ ] **Step 3: 验证 6 个 run.json 都存在**

Run:
```bash
cd /Users/yangxu/StockTradebyZ && ls reports/purge_audit_20260413/*/run.json | wc -l
```

Expected: `6`

**如少于 6**：查看 /tmp/purge_audit_run.log 找错。对失败的 run 手动补跑：
```bash
python3 scripts/run_purge_experiment.py --only-version {VERSION} --only-purge {PURGE} --force -v --date 20260413
```

**不 commit**（产出在 reports/ 目录，不入 git）。

---

## Task 11: 生成 REPORT.md + 人眼 review

**Files:** 产出 `reports/purge_audit_20260413/REPORT.md`（非 commit 目标）。

- [ ] **Step 1: 跑 analyzer**

Run:
```bash
cd /Users/yangxu/StockTradebyZ && python3 scripts/analyze_purge_leakage.py --date 20260413
```

Expected:
- `Wrote report: reports/purge_audit_20260413/REPORT.md`
- `Rows: 12 | Green: N | Yellow: N | Red: N`

- [ ] **Step 2: 人眼 review**

Run:
```bash
cat reports/purge_audit_20260413/REPORT.md
```

检查：
- 12 行判决表（3 版本 × 4 标签）
- 汇总行数合计 = 12
- 运行元数据 6 行（3 版本 × 2 purge）
- 顶部实验配置字段一致
- RED/YELLOW 若有，建议文本合理

**如果报告格式显眼错误**（对齐、字段缺失、数字异常），debug analyzer 或 runner。

- [ ] **Step 3: 记录结论**

如果全 GREEN：结论"当前 `purge_days=15` 足够"。
如果有 YELLOW/RED：记录下具体 (version, label) 对应的 Δ%，待后续决策是否升级 `purge_days` 默认值或二期跑 purge=60。

**无 commit**（REPORT.md 在 reports/，不入 git）。

---

## Task 12: 跨仓更新 EMT `ideas_todo.md` 标记 #2 完成

**Files:**
- Modify: `/Users/yangxu/EastMoneyTrader/docs/ideas_todo.md`（在 EMT 仓库）

- [ ] **Step 1: 修改 EMT 的 ideas_todo.md**

找到当前 #2 行（类似）：
```
| 2 | 数据泄漏验证 | ⭐⭐⭐ | 2-4 天 | ⏸ 未开始 — 独立可做 |
```

改为：
```
| 2 | 数据泄漏验证 | ⭐⭐⭐ | 2-4 天 | ✅ **已完成** (2026-04-13, StockTradebyZ `scripts/run_purge_experiment.py` + `analyze_purge_leakage.py`) |
```

找到 "## 2. ⭐⭐⭐ 模型数据泄漏验证（Leakage Check）" 章节开头，紧跟在标题之后插入：

```markdown
> **实现** [已完成 2026-04-13]:
> - 审计工具在 StockTradebyZ 仓库：`scripts/run_purge_experiment.py` + `scripts/analyze_purge_leakage.py`
> - 对 3 个 NG 版本 (ng1.0.1 / ng106 / ng1.1.0) × 2 个 purge 值 (15/30) × 4 个标签做对照实验
> - 报告：`StockTradebyZ/reports/purge_audit_20260413/REPORT.md`（本地，未入 git）
> - 结论：见报告判决表。Spec/Plan 在 `StockTradebyZ/docs/superpowers/{specs,plans}/2026-04-1[23]-purge-leakage-audit*`
>
> ---
```

- [ ] **Step 2: 在 EMT 仓库提交**

```bash
cd /Users/yangxu/EastMoneyTrader && git add docs/ideas_todo.md && git commit -m "标记 ideas_todo #2 完成：purge leakage audit (StockTradebyZ 侧)"
```

---

## Self-Review Checklist

- [ ] Spec 覆盖：10 项决策全部映射到 task？
  - 决策 1（StockTradebyZ 仓库）→ 所有 task ✓
  - 决策 2（2×3×4 实验）→ Task 7 runner loop ✓
  - 决策 3（purge 15 vs 30）→ `PURGE_VALUES = [15, 30]` ✓
  - 决策 4（3 个版本）→ `VERSIONS = [...]` + Task 8 验证 ✓
  - 决策 5（4 个 label）→ `LABELS = ['3d','5d','10d','15d']` ✓
  - 决策 6（3 WF 窗口）→ ng_trainer 默认，Task 7 不传 `--wf-windows` ✓
  - 决策 7（runner + analyzer 分离）→ Task 1-5 analyzer，Task 6-7 runner ✓
  - 决策 8（Markdown 判决表）→ Task 4 render_report ✓
  - 决策 9（阈值 10/30/0.005）→ Task 1 classify_verdict ✓
  - 决策 10（存储 + skip-if-exists）→ Task 7 run_single 开头检查 ✓

- [ ] 无 TBD/TODO/"add error handling"/"similar to Task N" 占位
- [ ] 每个 Task 都有完整代码
- [ ] 函数签名在 Task 间一致：
  - `classify_verdict(baseline_ic, delta_pct)` — Task 1, 3, 4 ✓
  - `extract_per_label_mean_oos_ic(wf_summary)` — Task 2, 7 ✓
  - `extract_n_windows(wf_summary)` — Task 2, 7 ✓
  - `load_runs(audit_dir)` — Task 3, 5 ✓
  - `compute_delta_rows(runs)` — Task 3, 5 ✓
  - `render_report(rows, runs, audit_date)` — Task 4, 5 ✓
  - `_snapshot_wf_summaries(model_root)` — Task 6, 7 ✓
  - `_find_new_wf_summary(pre, post)` — Task 6, 7 ✓
  - `run_single(version, purge_days, audit_dir, force, extra_args)` — Task 7 ✓

- [ ] Task 顺序：先 analyzer（独立可测），再 runner（依赖 analyzer 的解析函数），最后实验+报告 → 合理 ✓

## Known Risks

1. **Task 8 发现 ng_trainer 不接受 `ng106` 格式**：要调 `VERSIONS` 或查 version_map；Task 8 明确处理。
2. **Task 10 训练中途失败**：已设计 skip-if-exists 支持断点续跑；某 run rc != 0 不阻塞整体，analyzer 会标 N/A。
3. **Task 10 单次 >4h**：TimeoutExpired 会被 runner 捕获并记录，不 crash 整体；但实际该 version 可能要拆分调用 `--wf-windows 2` 手动跑。
4. **Task 6 `_find_new_wf_summary` 的 mtime 边缘情况**：如果 ng_trainer 写 wf_summary 到和 pre 完全相同的 mtime（文件系统精度不够），会漏检。实际 mac HFS+ / APFS 精度纳秒，不会出此问题。如真出现，runner 会记 `wf_summary_path: null`，人工排查。
5. **PyTest 依赖**：StockTradebyZ 已有 pytest.ini 和 requirements-test.txt。Task 1-6 运行 pytest 前先确认 `cd /Users/yangxu/StockTradebyZ && pip install -r requirements-test.txt` 已执行（或验证 `python3 -c "import pytest"` 不报错）。如果 pytest 未安装，`pip install pytest` 即可。

---

*Plan created: 2026-04-13*
