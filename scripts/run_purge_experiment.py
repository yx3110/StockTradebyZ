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
    LABELS,
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
    trainer_log_path = run_dir / "trainer.log"
    t0 = time.time()
    with open(trainer_log_path, "w", encoding="utf-8", errors="replace") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,  # merge stderr into same log file
            cwd=str(PROJECT_ROOT),
            text=True,
        )
        try:
            returncode = proc.wait(timeout=TRAIN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            logger.error("TIMEOUT %s after %d seconds", run_id, TRAIN_TIMEOUT_SECONDS)
            proc.kill()
            proc.wait()
            log_f.write(f"\n---TIMEOUT after {TRAIN_TIMEOUT_SECONDS}s---\n")
            returncode = -1

    elapsed = time.time() - t0

    # Locate new wf_summary
    post_snapshot = _snapshot_wf_summaries(TRAINED_MODELS_DIR)
    new_wf_path = _find_new_wf_summary(pre_snapshot, post_snapshot)

    oos_ics = {f"label_{d}": None for d in LABELS}
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

    failed = total - len(completed)
    nonzero_rc = sum(1 for r in completed if r.get("returncode", 0) != 0)
    print(f"\nDone. {len(completed)}/{total} runs returned, {nonzero_rc} had non-zero exit code.")
    print(f"Next: python3 scripts/analyze_purge_leakage.py --date {date_str}")
    # Non-zero exit if any run raised OR all runs had non-zero returncode
    if failed > 0 or (completed and nonzero_rc == len(completed)):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
