#!/usr/bin/env python3
"""Timing harness for North Star evaluation pipeline speedup validation.

Tests both cold (no cache) and warm (cached) runs to measure improvement.
Also validates that cached results match uncached results.
"""
import sys
import os
import time
import io
from contextlib import redirect_stdout

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from backtest import backtest_report_based as brb
from backtest import north_star_metrics as nsm
from backtest.eval_cache import EvalCache

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


def run_timed_backtest(report_dir, cache=None):
    """Run the full pipeline and return (timings_dict, backtest_result)."""
    timings = {}

    # Clear module-level caches to get fair measurements
    brb._future_returns_cache.clear()
    brb._next_trading_dates_cache.clear()

    # Stage 1: Load reports
    t0 = time.perf_counter()
    reports = brb.load_reports(report_dir, rank_field='composite', cache=cache)
    timings['load_reports'] = time.perf_counter() - t0
    print(f"  load_reports: {timings['load_reports']:.2f}s ({len(reports)} dates)")

    # Stage 2: Run backtest (suppress stdout)
    def run_bt():
        f = io.StringIO()
        with redirect_stdout(f):
            return brb.run_single_backtest(
                reports, "timing_test", top_n=10,
                benchmark_code='000905.SH', focus_days=10,
                cache=cache,
            )

    t0 = time.perf_counter()
    result = run_bt()
    timings['backtest'] = time.perf_counter() - t0
    print(f"  backtest: {timings['backtest']:.2f}s")

    return timings, result


def main():
    report_dir = find_report_dir()
    if not report_dir:
        print("ERROR: No report directory found with >= 50 JSON files")
        sys.exit(1)

    print(f"Report dir: {report_dir}")
    n_files = len([f for f in os.listdir(report_dir)
                   if f.startswith('analysis_data_') and f.endswith('.json')])
    print(f"JSON files: {n_files}")

    cache = EvalCache()

    # ─── Cold Run (clear cache) ───
    cache.clear()
    print(f"\n{'='*55}")
    print("COLD RUN (no cache)")
    print(f"{'='*55}")
    cold_timings, cold_result = run_timed_backtest(report_dir, cache)

    # ─── Warm Run (cache populated) ───
    print(f"\n{'='*55}")
    print("WARM RUN (cached)")
    print(f"{'='*55}")
    warm_timings, warm_result = run_timed_backtest(report_dir, cache)

    # ─── Comparison ───
    print(f"\n{'='*55}")
    print("COMPARISON")
    print(f"{'='*55}")
    print(f"  {'Stage':<25} {'Cold':>8} {'Warm':>8} {'Speedup':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for key in cold_timings:
        cold = cold_timings[key]
        warm = warm_timings[key]
        speedup = cold / warm if warm > 0.001 else float('inf')
        print(f"  {key:<25} {cold:>7.2f}s {warm:>7.2f}s {speedup:>7.1f}x")

    total_cold = sum(cold_timings.values())
    total_warm = sum(warm_timings.values())
    total_speedup = total_cold / total_warm if total_warm > 0.001 else float('inf')
    print(f"  {'TOTAL':<25} {total_cold:>7.2f}s {total_warm:>7.2f}s {total_speedup:>7.1f}x")
    print(f"\n  Cache stats: {cache.stats()}")

    # ─── Score Validation ───
    print(f"\n{'='*55}")
    print("SCORE VALIDATION (cold vs warm)")
    print(f"{'='*55}")

    cold_summary = cold_result['summary']
    warm_summary = warm_result['summary']
    all_ok = True

    for days in [5, 10, 15]:
        if days not in cold_summary or days not in warm_summary:
            continue
        for key in ['ic_mean', 'icir', 'sharpe_ratio', 'annual_return', 'max_drawdown']:
            cold_val = cold_summary[days].get(key, 0) or 0
            warm_val = warm_summary[days].get(key, 0) or 0
            diff = abs(cold_val - warm_val)
            status = "OK" if diff < 0.01 else "MISMATCH"
            if status != "OK":
                all_ok = False
            print(f"  {days}d {key:<20}: cold={cold_val:>9.4f} warm={warm_val:>9.4f} diff={diff:.6f} {status}")

    if all_ok:
        print(f"\n  All metrics match within tolerance.")
    else:
        print(f"\n  WARNING: Some metrics differ significantly!")

    # ─── Original Baseline Reference ───
    print(f"\n{'='*55}")
    print("vs ORIGINAL BASELINE (85.83s)")
    print(f"{'='*55}")
    baseline = 85.83
    print(f"  Original:  {baseline:.2f}s")
    print(f"  Cold now:  {total_cold:.2f}s ({baseline/total_cold:.1f}x faster)")
    print(f"  Warm now:  {total_warm:.2f}s ({baseline/total_warm:.1f}x faster)")


if __name__ == '__main__':
    main()
