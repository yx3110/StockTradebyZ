#!/usr/bin/env python3
"""
NG Full Training Pipeline — Unattended
=======================================
1. Wait for v1.0.1 backfill to finish (2020-2025)
2. Supplement 2026 data for v1.0.1 cache
3. Train ng1.0.1 (new features, industry excess labels)
4. Save v1.0.1 code, restore v1.0.0 code from git
5. Clear cache, rebuild v1.0.0 features (full 2020-2026)
6. Train ng1.0.0 (old features, absolute labels, full data)
7. Restore v1.0.1 code
8. Fair comparison of both versions

Usage:
    nohup python3 scripts/ng_full_pipeline.py > logs/ng_pipeline.log 2>&1 &
"""

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
NG_DIR = PROJECT_ROOT / 'ml_models' / 'ng'
MODEL_DIR = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'ng'
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# v1.0.1 code files (current working directory has these)
V101_FILES = [
    'ml_models/ng/ng_feature_calculator.py',
    'ml_models/ng/ng_cache_updater.py',
    'ml_models/ng/ng_trainer.py',
    'ml_models/ng/ng_production_scorer.py',
    'ml_models/ng/ng_schema.py',
    'ml_models/ng/__init__.py',
]
V101_BACKUP_DIR = PROJECT_ROOT / 'ml_models' / 'ng' / '_v101_backup'


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


def run_cmd(cmd, timeout=None):
    """Run a command and return (returncode, stdout)."""
    log(f"  CMD: {cmd}")
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        timeout=timeout, cwd=str(PROJECT_ROOT)
    )
    if result.returncode != 0:
        log(f"  STDERR: {result.stderr[-500:]}")
    return result.returncode, result.stdout


def get_cache_stats():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cur = conn.execute(
        'SELECT COUNT(*), COUNT(DISTINCT trade_date), MIN(trade_date), MAX(trade_date) FROM ng_feature_cache'
    )
    row = cur.fetchone()
    conn.close()
    return {'count': row[0], 'dates': row[1], 'min_date': row[2], 'max_date': row[3]}


def is_backfill_running():
    """Check if the initial backfill process is still running."""
    rc, out = run_cmd("ps aux | grep -v grep | grep 'ng_cache_updater.*start-date 2020' | wc -l")
    return int(out.strip()) > 0


# ======================================================================
# STEP 1: Wait for initial backfill (2020-2025) to finish
# ======================================================================

def step1_wait_for_backfill():
    log("=" * 60)
    log("STEP 1: Waiting for v1.0.1 backfill (2020-2025) to finish...")
    log("=" * 60)

    while is_backfill_running():
        stats = get_cache_stats()
        log(f"  Progress: {stats['dates']} dates, {stats['count']:,} rows, "
            f"latest: {stats['max_date']}")
        time.sleep(60)

    stats = get_cache_stats()
    log(f"  Backfill complete: {stats['dates']} dates, {stats['count']:,} rows, "
        f"{stats['min_date']} to {stats['max_date']}")


# ======================================================================
# STEP 2: Supplement 2026 data
# ======================================================================

def step2_supplement_2026():
    log("=" * 60)
    log("STEP 2: Supplementing 2026 data for v1.0.1 cache...")
    log("=" * 60)

    # Check what's the latest daily_quotes date
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cur = conn.execute('SELECT MAX(trade_date) FROM daily_quotes')
    max_quote_date = cur.fetchone()[0]
    conn.close()

    stats = get_cache_stats()
    if stats['max_date'] and stats['max_date'] >= '2026-01-01':
        log(f"  Already has 2026 data (up to {stats['max_date']}), checking gap...")

    log(f"  daily_quotes max: {max_quote_date}")
    log(f"  Backfilling 2026-01-01 to {max_quote_date}...")

    rc, out = run_cmd(
        f"python3 ml_models/ng/ng_cache_updater.py "
        f"--start-date 2026-01-01 --end-date {max_quote_date}",
        timeout=3600
    )
    if rc != 0:
        log(f"  WARNING: 2026 supplement had errors, continuing anyway")

    # Also check if there's a gap between 2025-12-31 and 2026-01-01
    # The initial backfill may have ended early if end-date query cut off
    stats = get_cache_stats()
    log(f"  After supplement: {stats['dates']} dates, {stats['count']:,} rows, "
        f"{stats['min_date']} to {stats['max_date']}")


# ======================================================================
# STEP 3: Train ng1.0.1
# ======================================================================

def step3_train_v101():
    log("=" * 60)
    log("STEP 3: Training ng1.0.1 (new features, industry excess labels)...")
    log("=" * 60)

    stats = get_cache_stats()
    log(f"  Training data: {stats['dates']} dates, {stats['count']:,} rows")

    # Full WF training
    rc, out = run_cmd(
        "python3 ml_models/ng/ng_trainer.py "
        "--start-date 2020-01-01 "
        "--purge-days 15 "
        "--min-train-days 900 "
        "--val-days 120 --test-days 120 --step-days 120",
        timeout=36000  # 10 hours max
    )

    if rc != 0:
        log(f"  ERROR: v1.0.1 training failed!")
        log(f"  Last output: {out[-1000:]}")
        return False

    # Check model was saved
    models = sorted(MODEL_DIR.glob('ng_multi_target_*.pkl'), key=lambda f: f.stat().st_mtime)
    if models:
        latest = models[-1]
        log(f"  v1.0.1 model saved: {latest.name} ({latest.stat().st_size/1024/1024:.1f} MB)")
        # Rename to include v101 tag
        v101_name = latest.name.replace('ng_multi_target_', 'ng101_multi_target_')
        v101_path = latest.parent / v101_name
        shutil.copy2(str(latest), str(v101_path))
        log(f"  Copied as: {v101_name}")
    else:
        log(f"  WARNING: No model file found after training!")
        return False

    # Save training history
    hist_latest = MODEL_DIR / 'training_history_latest.json'
    if hist_latest.exists():
        v101_hist = MODEL_DIR / 'training_history_v101.json'
        shutil.copy2(str(hist_latest), str(v101_hist))
        log(f"  Training history saved as: training_history_v101.json")

    return True


# ======================================================================
# STEP 4: Save v1.0.1 code, restore v1.0.0 from git
# ======================================================================

def step4_swap_to_v100():
    log("=" * 60)
    log("STEP 4: Swapping to v1.0.0 code for retraining...")
    log("=" * 60)

    # Backup v1.0.1 files
    V101_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for fpath in V101_FILES:
        src = PROJECT_ROOT / fpath
        dst = V101_BACKUP_DIR / os.path.basename(fpath)
        if src.exists():
            shutil.copy2(str(src), str(dst))
    log(f"  v1.0.1 code backed up to {V101_BACKUP_DIR}")

    # Restore v1.0.0 from git (HEAD has the old version)
    for fpath in V101_FILES:
        rc, _ = run_cmd(f"git checkout HEAD -- {fpath}")
        if rc != 0:
            log(f"  WARNING: Failed to restore {fpath} from git")
    log(f"  v1.0.0 code restored from git")


# ======================================================================
# STEP 5: Clear cache, rebuild v1.0.0 features (full 2020-2026)
# ======================================================================

def step5_rebuild_v100_cache():
    log("=" * 60)
    log("STEP 5: Rebuilding v1.0.0 feature cache (full 2020-2026)...")
    log("=" * 60)

    # Clear cache
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute('DELETE FROM ng_feature_cache')
    conn.commit()
    log(f"  Cache cleared")

    # Get max daily_quotes date
    cur = conn.execute('SELECT MAX(trade_date) FROM daily_quotes')
    max_date = cur.fetchone()[0]
    conn.close()
    log(f"  Backfilling 2020-01-01 to {max_date} with v1.0.0 features...")

    rc, out = run_cmd(
        f"python3 ml_models/ng/ng_cache_updater.py "
        f"--start-date 2020-01-01 --end-date {max_date}",
        timeout=7200  # 2 hours max
    )

    stats = get_cache_stats()
    log(f"  v1.0.0 cache: {stats['dates']} dates, {stats['count']:,} rows, "
        f"{stats['min_date']} to {stats['max_date']}")

    return stats['count'] > 0


# ======================================================================
# STEP 6: Train ng1.0.0 with full data
# ======================================================================

def step6_train_v100():
    log("=" * 60)
    log("STEP 6: Training ng1.0.0 (old features, full data 2020-2026)...")
    log("=" * 60)

    stats = get_cache_stats()
    log(f"  Training data: {stats['dates']} dates, {stats['count']:,} rows")

    rc, out = run_cmd(
        "python3 ml_models/ng/ng_trainer.py "
        "--start-date 2020-01-01 "
        "--purge-days 15 "
        "--min-train-days 900 "
        "--val-days 120 --test-days 120 --step-days 120",
        timeout=36000  # 10 hours max
    )

    if rc != 0:
        log(f"  ERROR: v1.0.0 training failed!")
        log(f"  Last output: {out[-1000:]}")
        return False

    models = sorted(MODEL_DIR.glob('ng_multi_target_*.pkl'), key=lambda f: f.stat().st_mtime)
    if models:
        latest = models[-1]
        log(f"  v1.0.0 model saved: {latest.name} ({latest.stat().st_size/1024/1024:.1f} MB)")
        # Rename to include v100 tag
        v100_name = latest.name.replace('ng_multi_target_', 'ng100_multi_target_')
        v100_path = latest.parent / v100_name
        shutil.copy2(str(latest), str(v100_path))
        log(f"  Copied as: {v100_name}")
    else:
        log(f"  WARNING: No model file found!")
        return False

    # Save training history
    hist_latest = MODEL_DIR / 'training_history_latest.json'
    if hist_latest.exists():
        v100_hist = MODEL_DIR / 'training_history_v100.json'
        shutil.copy2(str(hist_latest), str(v100_hist))
        log(f"  Training history saved as: training_history_v100.json")

    return True


# ======================================================================
# STEP 7: Restore v1.0.1 code
# ======================================================================

def step7_restore_v101():
    log("=" * 60)
    log("STEP 7: Restoring v1.0.1 code...")
    log("=" * 60)

    for fpath in V101_FILES:
        backup = V101_BACKUP_DIR / os.path.basename(fpath)
        dst = PROJECT_ROOT / fpath
        if backup.exists():
            shutil.copy2(str(backup), str(dst))
    log(f"  v1.0.1 code restored from backup")


# ======================================================================
# STEP 8: Fair comparison
# ======================================================================

def step8_compare():
    log("=" * 60)
    log("STEP 8: Fair comparison of ng1.0.0 vs ng1.0.1")
    log("=" * 60)

    results = {}

    for ver, hist_file in [('ng1.0.0', 'training_history_v100.json'),
                            ('ng1.0.1', 'training_history_v101.json')]:
        hist_path = MODEL_DIR / hist_file
        if not hist_path.exists():
            log(f"  {ver}: history file not found!")
            continue

        with open(hist_path, 'r') as f:
            history = json.load(f)

        wf_windows = history.get('wf_windows', [])
        n_windows = len(wf_windows)

        # Extract OOS IC/ICIR per target
        target_ics = {}
        for target in ['label_3d', 'label_5d', 'label_10d', 'label_15d']:
            ics = []
            for w in wf_windows:
                metrics = w.get('test_metrics', w.get('oos_metrics', {}))
                ic = metrics.get(f'{target}_ic', metrics.get(f'ic_{target}'))
                if ic is not None:
                    try:
                        ic = float(ic)
                        if not (ic != ic):  # not NaN
                            ics.append(ic)
                    except (ValueError, TypeError):
                        pass
            if ics:
                import numpy as np
                mean_ic = np.mean(ics)
                std_ic = np.std(ics) if len(ics) > 1 else 0.01
                icir = mean_ic / (std_ic + 1e-8)
                target_ics[target] = {
                    'mean_ic': float(mean_ic),
                    'icir': float(icir),
                    'ic_positive_ratio': float(np.mean(np.array(ics) > 0)),
                    'n_windows': len(ics),
                }

        results[ver] = {
            'n_wf_windows': n_windows,
            'targets': target_ics,
            'version_info': history.get('version', 'unknown'),
        }

    # Print comparison table
    log("\n" + "=" * 80)
    log("FAIR COMPARISON: ng1.0.0 vs ng1.0.1 (same data range)")
    log("=" * 80)

    if 'ng1.0.0' in results and 'ng1.0.1' in results:
        v100 = results['ng1.0.0']
        v101 = results['ng1.0.1']

        log(f"\n{'Metric':<30} {'ng1.0.0':>12} {'ng1.0.1':>12} {'Delta':>12}")
        log("-" * 70)

        for target in ['label_3d', 'label_5d', 'label_10d', 'label_15d']:
            t100 = v100['targets'].get(target, {})
            t101 = v101['targets'].get(target, {})

            ic100 = t100.get('mean_ic', 0)
            ic101 = t101.get('mean_ic', 0)
            icir100 = t100.get('icir', 0)
            icir101 = t101.get('icir', 0)
            pos100 = t100.get('ic_positive_ratio', 0)
            pos101 = t101.get('ic_positive_ratio', 0)

            delta_ic = ic101 - ic100
            delta_icir = icir101 - icir100

            log(f"{target} Mean IC       {ic100:>12.4f} {ic101:>12.4f} {delta_ic:>+12.4f}")
            log(f"{target} ICIR          {icir100:>12.3f} {icir101:>12.3f} {delta_icir:>+12.3f}")
            log(f"{target} IC>0 ratio    {pos100:>12.1%} {pos101:>12.1%}")
            log("")

        log(f"{'WF Windows':<30} {v100['n_wf_windows']:>12} {v101['n_wf_windows']:>12}")

        # Overall verdict
        icir_10d_100 = v100['targets'].get('label_10d', {}).get('icir', 0)
        icir_10d_101 = v101['targets'].get('label_10d', {}).get('icir', 0)
        icir_15d_100 = v100['targets'].get('label_15d', {}).get('icir', 0)
        icir_15d_101 = v101['targets'].get('label_15d', {}).get('icir', 0)

        avg_100 = (icir_10d_100 + icir_15d_100) / 2
        avg_101 = (icir_10d_101 + icir_15d_101) / 2

        log("\n" + "=" * 70)
        if avg_101 > avg_100:
            log(f"VERDICT: ng1.0.1 WINS (avg 10d+15d ICIR: {avg_101:.3f} vs {avg_100:.3f})")
        elif avg_100 > avg_101:
            log(f"VERDICT: ng1.0.0 WINS (avg 10d+15d ICIR: {avg_100:.3f} vs {avg_101:.3f})")
        else:
            log(f"VERDICT: TIE (avg 10d+15d ICIR: {avg_100:.3f} vs {avg_101:.3f})")
        log("=" * 70)
    else:
        log("  Missing training history for one or both versions!")
        for ver, data in results.items():
            log(f"\n  {ver}: {data}")

    # Save comparison to file
    comparison_path = MODEL_DIR / 'comparison_v100_vs_v101.json'
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nComparison saved to: {comparison_path}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    start_time = time.time()
    log("=" * 60)
    log("NG FULL TRAINING PIPELINE — UNATTENDED")
    log(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    # Step 1: Wait for initial backfill
    step1_wait_for_backfill()

    # Step 2: Supplement 2026 data
    step2_supplement_2026()

    # Step 3: Train ng1.0.1
    v101_ok = step3_train_v101()
    if not v101_ok:
        log("FATAL: v1.0.1 training failed, aborting pipeline")
        return

    # Step 4: Swap to v1.0.0 code
    step4_swap_to_v100()

    # Step 5: Rebuild v1.0.0 cache
    cache_ok = step5_rebuild_v100_cache()
    if not cache_ok:
        log("FATAL: v1.0.0 cache rebuild failed")
        step7_restore_v101()
        return

    # Step 6: Train ng1.0.0
    v100_ok = step6_train_v100()

    # Step 7: Restore v1.0.1 code (always restore regardless of training result)
    step7_restore_v101()

    if not v100_ok:
        log("WARNING: v1.0.0 training failed, comparison may be incomplete")

    # Step 8: Compare
    step8_compare()

    elapsed = time.time() - start_time
    hours = elapsed / 3600
    log(f"\nPipeline complete! Total time: {hours:.1f} hours")
    log(f"Models in: {MODEL_DIR}")
    log(f"  ng101_multi_target_*.pkl — v1.0.1 (new features)")
    log(f"  ng100_multi_target_*.pkl — v1.0.0 (old features, full data)")


if __name__ == '__main__':
    main()
