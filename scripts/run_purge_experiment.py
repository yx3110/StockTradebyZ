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
