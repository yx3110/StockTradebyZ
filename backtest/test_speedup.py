#!/usr/bin/env python3
"""Timing harness for North Star evaluation pipeline speedup validation."""
import sys
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from backtest import backtest_report_based as brb
from backtest import north_star_metrics as nsm

DB_PATH = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')
nsm.DB_PATH = DB_PATH
brb.DB_PATH = DB_PATH


def find_report_dir():
    """Find the first available report directory with >= 50 JSON files."""
    candidates = [
        'reports/daily_selection_v4.7.5',
        'reports/daily_selection_v4.7.3',
        'reports/daily_selection_v4.6_merged_extended',
        'reports/daily_selection_v3.9',
    ]
    for d in candidates:
        path = os.path.join(PROJECT_ROOT, d)
        if os.path.isdir(path):
            jsons = [f for f in os.listdir(path)
                     if f.startswith('analysis_data_') and f.endswith('.json')]
            if len(jsons) >= 50:
                return path
    return None


def time_stage(name, func):
    """Time a function and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - t0
    print(f"  {name}: {elapsed:.2f}s")
    return result, elapsed


def main():
    report_dir = find_report_dir()
    if not report_dir:
        print("ERROR: No report directory found with >= 50 JSON files")
        sys.exit(1)

    print(f"Report dir: {report_dir}")
    n_files = len([f for f in os.listdir(report_dir)
                   if f.startswith('analysis_data_') and f.endswith('.json')])
    print(f"JSON files: {n_files}")
    print()

    timings = {}

    # Stage 1: Load reports
    reports, t = time_stage(
        "load_reports",
        lambda: brb.load_reports(report_dir, rank_field='composite'),
    )
    timings['load_reports'] = t
    print(f"    -> {len(reports)} dates loaded")

    # Stage 2: Run backtest (suppress stdout)
    import io
    from contextlib import redirect_stdout

    def run_bt():
        f = io.StringIO()
        with redirect_stdout(f):
            return brb.run_single_backtest(
                reports, "timing_test", top_n=10,
                benchmark_code='000905.SH', focus_days=10,
            )

    result, t = time_stage("run_single_backtest", run_bt)
    timings['backtest'] = t

    # Total
    total = sum(timings.values())
    print(f"\n  TOTAL: {total:.2f}s")
    print(f"\n  Breakdown:")
    for k, v in timings.items():
        print(f"    {k}: {v:.2f}s ({v / total * 100:.0f}%)")

    return timings


if __name__ == '__main__':
    main()
