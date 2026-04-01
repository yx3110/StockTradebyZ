# North Star Evaluation Pipeline Speedup Design

**Date**: 2026-03-28
**Goal**: Accelerate the full end-to-end North Star evaluation pipeline (report generation + backtest + scoring)
**Constraint**: Final North Star score and grade must match; intermediate process can be refactored

## Current Performance Profile

| Stage | Current Time (300-day window) | Bottleneck |
|-------|-------------------------------|-----------|
| Batch report generation | ~10 min | Serial subprocess per date |
| Backtest engine | ~5-10 min | Spearman IC (7000+ calls), risk metrics |
| North Star metrics | ~2-5 min | Multiple DB queries, 21+ metrics |
| **Total (single model)** | **~20-30 min** | |
| **Multi-model (N models)** | **N x 20-30 min** | No shared computation |

## Phase A: Code-Level Optimizations (3-5x speedup)

### A1. Spearman IC Vectorization

**File**: `backtest/backtest_report_based.py` (lines 1275-1286)

**Current**: Per-date loop calling `spearmanr()` individually — 7000+ calls (N_dates x 7 holding_periods).

**New approach**: Use pandas `groupby().rank()` to batch-compute ranks across all dates in one vectorized operation. Then compute Pearson correlation on ranks (equivalent to Spearman) per date using grouped dot products.

```python
# Pseudocode
ranks_score = valid_picks.groupby('date')['score'].rank()
ranks_return = valid_picks.groupby('date')[return_col].rank()
# Standardize within group, then compute correlation via grouped mean of products
```

**Expected speedup**: IC calculation from ~10-15s to ~2-3s (5-8x on this stage)

### A2. Report Generation Parallelization

**File**: `backtest/run_north_star_eval.py` (line 113)

**Current**: Serial subprocess loop. Comment says "avoid DB lock conflicts".

**New approach**: Use `batch_generate_v395_reports.py`'s in-process batch mode directly instead of spawning subprocesses. This script already has `bulk_preload_scorer_caches()` which pre-loads all caches before the loop, avoiding DB contention. If subprocess mode is still needed, enable SQLite WAL mode and use `ProcessPoolExecutor(max_workers=4)`.

**Expected speedup**: Report generation from ~10 min to ~2-3 min (4-6x)

### A3. Risk Metrics Optimization

**File**: `backtest/backtest_report_based.py` (lines 1383-1398)

**Current**:
- Repeated `pd.to_datetime()` conversions inside per-offset loops
- Manual `groupby(to_period('M'))` per offset

**New approach**:
- Pre-compute `datetime_index` once before the loop
- Use `resample('M')` instead of manual period grouping

**Expected speedup**: Risk computation ~2x faster

### A4. Future Returns Triple-Loop Optimization

**File**: `backtest/backtest_report_based.py` (lines 382-394)

**Current**: Three nested Python loops building dict of dicts.

**New approach**: Use pandas DataFrame operations (pivot/merge) to construct the returns structure in vectorized form, reducing Python loop overhead.

**Expected speedup**: Moderate (reduces Python overhead, main benefit is cleaner code)

### Phase A Summary

| Optimization | File | Speedup (stage) | Effort |
|-------------|------|-----------------|--------|
| A1. Vectorized IC | backtest_report_based.py | 5-8x | Medium |
| A2. Parallel reports | run_north_star_eval.py | 4-6x | Low |
| A3. Risk metrics | backtest_report_based.py | 2x | Low |
| A4. Future returns loop | backtest_report_based.py | 1.3x | Low |
| **Total end-to-end** | | **3-5x** | **1-2 days** |

**Target**: ~6-10 min per model evaluation (down from 20-30 min)

## Phase B: Persistent Cache + Incremental Computation (additional 2-3x)

### B1. Cache Module

**New file**: `backtest/eval_cache.py`

Centralized cache manager with three invalidation strategies:

```python
class EvalCache:
    cache_dir: Path  # backtest/.eval_cache/

    # Static cache (invalidated when DB changes)
    #   Key: MAX(trade_date) + row_count from daily_quotes
    #   Files: future_returns_matrix.npz, trading_dates_index.pkl, industry_map.pkl

    # Report cache (invalidated when report directory changes)
    #   Key: hash of file names + mtimes in report directory
    #   Files: {report_dir_hash}_parsed.pkl

    # Result cache (invalidated when backtest params change)
    #   Key: hash of (top_n, focus_days, rank_field, holding_days, cppi_params, ...)
    #   Files: {param_hash}_backtest.pkl
```

**Invalidation details**:
- Static cache version: `f"{max_trade_date}_{row_count}"` queried from `daily_quotes`
- Report cache version: `hashlib.md5(sorted(f"{name}:{mtime}" for each file))`
- Result cache version: `hashlib.md5(json.dumps(sorted_params))`

Cache files are stored in `backtest/.eval_cache/` (gitignored).

### B2. Future Returns Matrix Persistence

**Current**: `batch_get_all_future_returns()` runs 3 bulk SQL queries + 700K dict assignments every time.

**New approach**:
- First run: SQL -> DataFrame -> pivot to 3D numpy array `(dates x codes x holding_days)` -> `np.savez_compressed()`
- Subsequent runs: `np.load()` (~0.1s vs current ~5-8s)
- Incremental update: detect new trading dates, query only new rows, append to existing matrix, re-save

**Storage**: ~50-100MB compressed for 300-day window with 4000+ stocks x 7 holding periods

### B3. Report Parsing Cache

**Current**: `load_reports()` parses 500+ JSON files with ThreadPool+orjson (~3s).

**New approach**:
- First run: parse all -> `pickle.dump(reports_dict)`
- Subsequent runs: `pickle.load()` (~0.2s)
- Invalidation: re-parse when report directory has new/modified files (detected by mtime hash)

### B4. Multi-Model Shared Cache

When comparing N models, the following data is model-independent and should be computed once:

| Data | Size | Compute Cost |
|------|------|-------------|
| Future returns matrix | ~100MB | ~5-8s |
| Trading dates index | ~1KB | ~0.5s |
| Industry map | ~500KB | ~1s |
| Market cap / limit-up data | ~50MB | ~2-3s |
| Benchmark returns | ~100KB | ~0.5s |

**Design**: `EvalCache` instance created at top of `run_north_star_eval.py`, passed into each model's backtest function. First model populates cache, subsequent models read from it.

### B5. Incremental Evaluation

**Scenario**: Already evaluated 300 days, 10 new report days added.

**Flow**:
1. Detect: compare cached date list vs current report dates
2. Incremental: load only new reports + query only new dates' returns
3. Merge: append new data to cached `picks_df` and returns matrix
4. Recompute: IC/ICIR/Sharpe etc. must be fully recomputed (statistical aggregates are not incrementally decomposable), but input data preparation drops from 5-8s to ~0.3s

**Note**: The speedup is in data preparation, not in metric computation itself. For a 310-day window with 10 new days, the metric computation time is similar to a 300-day window.

### Phase B Summary

| Optimization | New/Modified Files | Speedup (stage) | Effort |
|-------------|-------------------|-----------------|--------|
| B1. Cache module | +eval_cache.py | Foundation | Medium |
| B2. Returns persistence | backtest_report_based.py | 50-80x (data load) | Medium |
| B3. Report cache | backtest_report_based.py | 15x (report parse) | Low |
| B4. Shared cache | run_north_star_eval.py | Nx1 -> 1+Nx0.3 | Medium |
| B5. Incremental eval | eval_cache.py + backtest_report_based.py | 2-3x (data prep) | Medium |
| **Total additional** | | **2-3x** | **2-3 days** |

## Combined Performance Targets

| Scenario | Current | After Phase A | After A+B |
|----------|---------|--------------|-----------|
| Single model (first run) | 20-30 min | 6-10 min | 5-8 min |
| Single model (repeat run) | 20-30 min | 6-10 min | 1-2 min |
| 4-model comparison (first) | 80-120 min | 24-40 min | 12-17 min |
| 4-model comparison (repeat) | 80-120 min | 24-40 min | 5-8 min |
| Incremental (+10 days) | 20-30 min | 6-10 min | 2-3 min |

## Files Changed

### Phase A (modify only)
1. `backtest/backtest_report_based.py` — A1 (IC vectorization), A3 (risk metrics), A4 (returns loop)
2. `backtest/run_north_star_eval.py` — A2 (parallel report generation)

### Phase B (new + modify)
3. `backtest/eval_cache.py` — **NEW** B1 (cache module)
4. `backtest/backtest_report_based.py` — B2 (returns persistence), B3 (report cache)
5. `backtest/run_north_star_eval.py` — B4 (shared cache), B5 (incremental eval)
6. `.gitignore` — add `backtest/.eval_cache/`

## Validation

- Run existing North Star evaluation on V4.7.5 model before and after
- Compare final scores: must produce identical grade and within +/-0.5 points on 105-point scale
- Compare timing: measure wall-clock time for each stage
- Test cache invalidation: modify DB, verify cache is rebuilt
- Test multi-model: compare 3+ models, verify shared cache is used
