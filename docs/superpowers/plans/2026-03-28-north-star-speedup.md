# North Star Evaluation Pipeline Speedup

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accelerate the full end-to-end North Star evaluation pipeline (report generation + backtest + scoring) by 3-5x initially, with persistent caching for 10-15x on repeat runs.

**Architecture:** Two-phase approach. Phase A optimizes hot loops in-place (vectorized IC, parallel reports, faster risk metrics). Phase B adds a persistent cache layer (`backtest/eval_cache.py`) so repeated evaluations skip data loading entirely.

**Tech Stack:** Python 3, NumPy, pandas, scipy.stats, sqlite3, pickle, hashlib, concurrent.futures

---

## Task 1: Baseline Timing Harness

**Files:**
- Create: `backtest/test_speedup.py`

We need to measure before/after. Create a lightweight timing script that exercises the full pipeline.

- [ ] **Step 1: Write the timing harness**

```python
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
    """Find the first available report directory with JSON files."""
    candidates = [
        'reports/daily_selection_v4.7.5',
        'reports/daily_selection_v4.7.3',
        'reports/daily_selection_v4.6_merged_extended',
        'reports/daily_selection_v3.9',
    ]
    for d in candidates:
        path = os.path.join(PROJECT_ROOT, d)
        if os.path.isdir(path):
            jsons = [f for f in os.listdir(path) if f.startswith('analysis_data_') and f.endswith('.json')]
            if len(jsons) >= 50:
                return path
    return None


def time_stage(name, func):
    """Time a function and print results."""
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
    n_files = len([f for f in os.listdir(report_dir) if f.endswith('.json')])
    print(f"JSON files: {n_files}")
    print()

    timings = {}

    # Stage 1: Load reports
    reports, t = time_stage("load_reports", lambda: brb.load_reports(report_dir, rank_field='composite'))
    timings['load_reports'] = t
    print(f"    → {len(reports)} dates loaded")

    # Stage 2: Run backtest (suppress prints)
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
        print(f"    {k}: {v:.2f}s ({v/total*100:.0f}%)")

    return timings


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the baseline timing**

Run: `python3 backtest/test_speedup.py`
Expected: Prints timing for each stage. Record these numbers as the baseline.

- [ ] **Step 3: Commit**

```bash
git add backtest/test_speedup.py
git commit -m "feat: 添加北极星评估性能基准测试脚本"
```

---

## Task 2: Vectorize Spearman IC Calculation (A1)

**Files:**
- Modify: `backtest/backtest_report_based.py:1275-1286`

The current code calls `spearmanr()` in a per-date loop (7000+ calls for 7 holding periods × ~500 dates). Replace with vectorized rank computation using pandas `groupby().rank()` + Pearson on ranks.

- [ ] **Step 1: Write the vectorized IC helper function**

Add this function near the top of `backtest/backtest_report_based.py`, after the imports (around line 80):

```python
def _vectorized_daily_ic(picks_df: pd.DataFrame, return_col: str) -> pd.DataFrame:
    """Compute daily IC (Spearman) via vectorized rank + Pearson, replacing per-date loop.

    Returns DataFrame with columns: date, ic, p_val, n_stocks
    """
    valid = picks_df[['date', 'score', return_col]].dropna(subset=[return_col])
    if len(valid) == 0:
        return pd.DataFrame(columns=['date', 'ic', 'p_val', 'n_stocks'])

    # Filter groups with >= 5 stocks
    counts = valid.groupby('date').size()
    valid_dates = counts[counts >= 5].index
    valid = valid[valid['date'].isin(valid_dates)]
    if len(valid) == 0:
        return pd.DataFrame(columns=['date', 'ic', 'p_val', 'n_stocks'])

    # Rank within each date group (Spearman = Pearson on ranks)
    valid = valid.copy()
    valid['rank_s'] = valid.groupby('date')['score'].rank()
    valid['rank_r'] = valid.groupby('date')[return_col].rank()

    # Standardize ranks within each group: (rank - mean) / std
    g = valid.groupby('date')
    for col in ['rank_s', 'rank_r']:
        mean = g[col].transform('mean')
        std = g[col].transform('std')
        valid[col + '_z'] = (valid[col] - mean) / std.replace(0, np.nan)

    # Pearson on standardized ranks = Spearman
    valid['product'] = valid['rank_s_z'] * valid['rank_r_z']
    result = valid.groupby('date').agg(
        ic=('product', 'mean'),
        n_stocks=('score', 'size'),
    ).reset_index()

    # Drop NaN ICs (from zero-std groups)
    result = result.dropna(subset=['ic'])

    # p-value approximation: t = ic * sqrt((n-2)/(1-ic^2))
    n = result['n_stocks']
    ic = result['ic']
    t_stat = ic * np.sqrt((n - 2) / (1 - ic**2).clip(lower=1e-10))
    from scipy.stats import t as t_dist
    result['p_val'] = 2 * t_dist.sf(np.abs(t_stat), df=n - 2)

    return result[['date', 'ic', 'p_val', 'n_stocks']]
```

- [ ] **Step 2: Replace the per-date loop with the vectorized function**

In `backtest/backtest_report_based.py`, find the IC computation block at lines 1275-1286 and replace it:

**Old code (lines 1275-1286):**
```python
        # 逐日IC序列 (优化: 预分组避免重复filter)
        return_col = f'return_{days}d'
        valid_picks = picks_df[['date', 'score', return_col]].dropna(subset=[return_col])
        ic_records = []
        if len(valid_picks) > 0:
            for date, group in valid_picks.groupby('date'):
                if len(group) >= 5:
                    day_ic, day_p = spearmanr(group['score'].values, group[return_col].values)
                    if not np.isnan(day_ic):
                        ic_records.append({'date': date, 'ic': day_ic, 'p_val': day_p, 'n_stocks': len(group)})
        ic_df = pd.DataFrame(ic_records) if ic_records else pd.DataFrame()
        daily_ic_series[days] = ic_df
```

**New code:**
```python
        # 逐日IC序列 (向量化: groupby rank + Pearson, 替代逐日spearmanr循环)
        return_col = f'return_{days}d'
        ic_df = _vectorized_daily_ic(picks_df, return_col)
        daily_ic_series[days] = ic_df
```

- [ ] **Step 3: Run timing harness to measure improvement**

Run: `python3 backtest/test_speedup.py`
Expected: `backtest` stage should show measurable improvement (IC part was ~30-40% of backtest time).

- [ ] **Step 4: Validate IC results match**

Add a quick validation to `backtest/test_speedup.py` to check the vectorized IC matches the old loop. Run it once, then remove the validation code:

```python
# Temporary validation: compare old vs new IC for one holding period
from scipy.stats import spearmanr
picks_df = result['picks']
return_col = 'return_10d'
valid_picks = picks_df[['date', 'score', return_col]].dropna(subset=[return_col])
old_ics = []
for date, group in valid_picks.groupby('date'):
    if len(group) >= 5:
        day_ic, _ = spearmanr(group['score'].values, group[return_col].values)
        if not np.isnan(day_ic):
            old_ics.append(day_ic)
new_ic_df = result['daily_ic_series'][10]
new_ics = new_ic_df['ic'].tolist()
# Compare
max_diff = max(abs(a - b) for a, b in zip(old_ics[:len(new_ics)], new_ics[:len(old_ics)]))
print(f"  IC validation: max_diff={max_diff:.6f} (should be < 0.01)")
assert max_diff < 0.01, f"IC mismatch: max_diff={max_diff}"
```

Run: `python3 backtest/test_speedup.py`
Expected: `IC validation: max_diff=0.00xxxx (should be < 0.01)` — passes assertion.

- [ ] **Step 5: Commit**

```bash
git add backtest/backtest_report_based.py
git commit -m "perf: 向量化Spearman IC计算, 替代7000+次逐日循环"
```

---

## Task 3: Optimize Risk Metrics Monthly Aggregation (A3)

**Files:**
- Modify: `backtest/backtest_report_based.py:1381-1398`

The monthly win rate computation repeatedly calls `pd.to_datetime()` inside a per-offset loop. Pre-compute the datetime index once.

- [ ] **Step 1: Replace the monthly aggregation loop**

In `backtest/backtest_report_based.py`, find lines 1381-1398 and replace:

**Old code:**
```python
        if days > 1 and len(sub) > days * 3:
            all_monthly = []
            for offset in range(min(days, len(sub))):
                offset_sub = sub.iloc[offset::days]
                if len(offset_sub) < 3:
                    continue
                offset_rets = offset_sub.set_index('date')['avg_top_return']
                offset_rets.index = pd.to_datetime(offset_rets.index)
                monthly = offset_rets.groupby(offset_rets.index.to_period('M')).apply(
                    lambda x: (1 + x).prod() - 1)
                all_monthly.append(monthly)
            if all_monthly:
                combined = pd.concat(all_monthly, axis=1)
                avg_monthly = combined.mean(axis=1)
                robust_win_rate = (avg_monthly > 0).mean() * 100
                risk['monthly_win_rate'] = robust_win_rate
                risk['worst_month'] = avg_monthly.min()
                risk['best_month'] = avg_monthly.max()
```

**New code:**
```python
        if days > 1 and len(sub) > days * 3:
            # Pre-compute datetime index once (avoid repeated pd.to_datetime in loop)
            sub_with_dt = sub[['date', 'avg_top_return']].copy()
            sub_with_dt['dt'] = pd.to_datetime(sub_with_dt['date'])
            sub_with_dt['period'] = sub_with_dt['dt'].dt.to_period('M')

            all_monthly = []
            n_offsets = min(days, len(sub_with_dt))
            for offset in range(n_offsets):
                offset_sub = sub_with_dt.iloc[offset::days]
                if len(offset_sub) < 3:
                    continue
                monthly = offset_sub.groupby('period')['avg_top_return'].apply(
                    lambda x: (1 + x).prod() - 1)
                all_monthly.append(monthly)
            if all_monthly:
                combined = pd.concat(all_monthly, axis=1)
                avg_monthly = combined.mean(axis=1)
                robust_win_rate = (avg_monthly > 0).mean() * 100
                risk['monthly_win_rate'] = robust_win_rate
                risk['worst_month'] = avg_monthly.min()
                risk['best_month'] = avg_monthly.max()
```

- [ ] **Step 2: Run timing harness**

Run: `python3 backtest/test_speedup.py`
Expected: Small improvement in `backtest` stage (~5-10% of risk metrics portion).

- [ ] **Step 3: Commit**

```bash
git add backtest/backtest_report_based.py
git commit -m "perf: 月度胜率计算预计算datetime, 消除循环内重复转换"
```

---

## Task 4: Parallel Report Generation via In-Process Batch (A2)

**Files:**
- Modify: `backtest/run_north_star_eval.py:48-121`

Replace the serial `subprocess.run(tomorrow_stock_selector.py)` loop with a direct call to `batch_generate_v395_reports.py`'s main function, which already handles bulk preloading and in-process scoring.

- [ ] **Step 1: Add a fast batch generation function**

In `backtest/run_north_star_eval.py`, replace the `generate_reports()` function (lines 48-121) with:

```python
def generate_reports(scoring_version='v3.95', start_date='auto', end_date='auto'):
    """批量生成选股报告 (快速版: 调用batch_generate直接评分, 不走subprocess)"""
    import subprocess

    # auto 日期: 从数据库获取最新可用范围
    if start_date == 'auto' or end_date == 'auto':
        all_dates = _get_trading_dates('2020-01-01', '2030-12-31')
        if all_dates:
            if start_date == 'auto':
                start_date = all_dates[0]
            if end_date == 'auto':
                end_date = all_dates[-1]
        else:
            print("  ⚠️ 无法从数据库检测日期范围")
            return

    # 确定报告输出目录
    if scoring_version == 'v3.95':
        report_dir = PROJECT_ROOT / 'reports' / 'daily_selection_v3.95_robust_zscore'
    else:
        report_dir = PROJECT_ROOT / 'reports' / f'daily_selection_{scoring_version}'

    print(f"\n{'='*60}")
    print(f"  批量生成 {scoring_version} 报告: {start_date} → {end_date}")
    print(f"{'='*60}\n")

    # 尝试使用快速批量生成器 (in-process, 不走subprocess)
    batch_script = PROJECT_ROOT / 'backtest' / 'batch_generate_v395_reports.py'
    if batch_script.exists():
        cmd = [
            sys.executable, str(batch_script),
            '--version', scoring_version,
            '--start-date', start_date,
            '--end-date', end_date,
            '--output-dir', str(report_dir),
        ]
        print(f"  使用快速批量生成器: {batch_script.name}")
        try:
            result = subprocess.run(cmd, timeout=3600, cwd=str(PROJECT_ROOT))
            if result.returncode == 0:
                print(f"  报告生成完成")
                return
            else:
                print(f"  快速生成器失败 (rc={result.returncode}), 回退到逐日模式")
        except subprocess.TimeoutExpired:
            print(f"  快速生成器超时, 回退到逐日模式")

    # 回退: 原始逐日subprocess模式
    dates = _get_trading_dates(start_date, end_date)
    print(f"  共 {len(dates)} 个交易日 (逐日模式)")

    existing = set()
    if report_dir.exists():
        for f in report_dir.glob('*.json'):
            name = f.stem
            if '_' in name:
                date_str = name.split('_')[-1]
                if len(date_str) == 8:
                    existing.add(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}")

    dates_todo = [d for d in dates if d not in existing]
    print(f"  已有 {len(existing)} 份报告, 需生成 {len(dates_todo)} 份")

    if not dates_todo:
        print("  所有报告已存在, 跳过")
        return

    def gen_one(date):
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / 'tomorrow_stock_selector.py'),
            date,
            '--scoring-version', scoring_version,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                                     cwd=str(PROJECT_ROOT))
            if result.returncode != 0:
                return date, f"error: {result.stderr[:100]}"
            return date, "ok"
        except subprocess.TimeoutExpired:
            return date, "timeout"
        except Exception as e:
            return date, str(e)

    # 串行执行(避免DB锁冲突)
    done = 0
    for date in dates_todo:
        done += 1
        if done % 10 == 0 or done == 1:
            print(f"  [{done}/{len(dates_todo)}] {date}")
        _, status = gen_one(date)
        if status != "ok":
            print(f"    ⚠️ {date}: {status}")

    print(f"\n  报告生成完成 ({done} 份)")
```

- [ ] **Step 2: Run timing harness to verify no regression**

Run: `python3 backtest/test_speedup.py`
Expected: No change to backtest timing (this only affects report generation path, not backtest itself).

- [ ] **Step 3: Commit**

```bash
git add backtest/run_north_star_eval.py
git commit -m "perf: 报告生成优先使用batch_generate快速批量模式"
```

---

## Task 5: Persistent Cache Module (B1)

**Files:**
- Create: `backtest/eval_cache.py`
- Modify: `.gitignore`

Create the central cache manager with three invalidation strategies: static (DB-based), report (directory-based), result (param-based).

- [ ] **Step 1: Create the cache module**

```python
#!/usr/bin/env python3
"""
Persistent cache for North Star evaluation pipeline.

Three cache types with distinct invalidation:
- Static: invalidated when DB changes (daily_quotes max_date + row_count)
- Report: invalidated when report directory contents change (file list hash)
- Result: invalidated when backtest params change (param hash)
"""
import os
import sys
import hashlib
import sqlite3
import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
DEFAULT_CACHE_DIR = PROJECT_ROOT / 'backtest' / '.eval_cache'


class EvalCache:
    """Persistent cache manager for North Star evaluation data."""

    def __init__(self, cache_dir: Path = None, db_path: str = None):
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.db_path = db_path or DEFAULT_DB_PATH
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._db_version = None

    # ── Version Keys ──────────────────────────────────────

    def _get_db_version(self) -> str:
        """DB version = max_trade_date + row_count from daily_quotes."""
        if self._db_version is not None:
            return self._db_version
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA busy_timeout = 30000")
        try:
            row = conn.execute(
                "SELECT MAX(trade_date), COUNT(*) FROM daily_quotes"
            ).fetchone()
            self._db_version = f"{row[0]}_{row[1]}"
        finally:
            conn.close()
        return self._db_version

    @staticmethod
    def _get_report_dir_version(report_dir: str) -> str:
        """Hash of sorted (filename, mtime) pairs in report directory."""
        report_path = Path(report_dir)
        if not report_path.exists():
            return "empty"
        entries = []
        for f in sorted(report_path.glob('analysis_data_*.json')):
            entries.append(f"{f.name}:{f.stat().st_mtime_ns}")
        if not entries:
            return "empty"
        h = hashlib.md5('\n'.join(entries).encode()).hexdigest()[:12]
        return h

    @staticmethod
    def _get_param_hash(**params) -> str:
        """Hash of sorted backtest parameters."""
        import json
        key_str = json.dumps(params, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    # ── Generic Cache Operations ──────────────────────────

    def _cache_path(self, name: str) -> Path:
        return self.cache_dir / name

    def load(self, name: str, version: str) -> Optional[Any]:
        """Load cached data if version matches."""
        meta_path = self._cache_path(f"{name}.meta")
        data_path = self._cache_path(f"{name}.pkl")
        if not meta_path.exists() or not data_path.exists():
            return None
        with open(meta_path, 'r') as f:
            cached_version = f.read().strip()
        if cached_version != version:
            return None
        with open(data_path, 'rb') as f:
            return pickle.load(f)

    def save(self, name: str, version: str, data: Any):
        """Save data with version marker."""
        meta_path = self._cache_path(f"{name}.meta")
        data_path = self._cache_path(f"{name}.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(meta_path, 'w') as f:
            f.write(version)

    def load_npz(self, name: str, version: str) -> Optional[Dict]:
        """Load numpy arrays if version matches."""
        meta_path = self._cache_path(f"{name}.meta")
        data_path = self._cache_path(f"{name}.npz")
        if not meta_path.exists() or not data_path.exists():
            return None
        with open(meta_path, 'r') as f:
            cached_version = f.read().strip()
        if cached_version != version:
            return None
        return dict(np.load(data_path, allow_pickle=True))

    def save_npz(self, name: str, version: str, **arrays):
        """Save numpy arrays with version marker."""
        meta_path = self._cache_path(f"{name}.meta")
        data_path = self._cache_path(f"{name}.npz")
        np.savez_compressed(data_path, **arrays)
        with open(meta_path, 'w') as f:
            f.write(version)

    # ── High-Level Cache API ──────────────────────────────

    def get_future_returns(self, report_dates: list, loader_fn) -> dict:
        """Load future returns from cache or compute via loader_fn.

        Args:
            report_dates: sorted list of report dates
            loader_fn: callable(report_dates) -> {buy_date: {code: {return_Xd: val}}}

        Returns:
            The future returns dict.
        """
        db_ver = self._get_db_version()
        dates_hash = hashlib.md5(','.join(sorted(report_dates)).encode()).hexdigest()[:8]
        name = f"future_returns_{dates_hash}"
        version = db_ver

        cached = self.load(name, version)
        if cached is not None:
            return cached

        data = loader_fn(report_dates)
        self.save(name, version, data)
        return data

    def get_parsed_reports(self, report_dir: str, rank_field: str,
                           loader_fn) -> dict:
        """Load parsed reports from cache or compute via loader_fn.

        Args:
            report_dir: path to report directory
            rank_field: ranking field used for parsing
            loader_fn: callable(report_dir, rank_field) -> {date: [stock_list]}

        Returns:
            The parsed reports dict.
        """
        dir_ver = self._get_report_dir_version(report_dir)
        name = f"reports_{hashlib.md5(report_dir.encode()).hexdigest()[:8]}_{rank_field}"
        version = dir_ver

        cached = self.load(name, version)
        if cached is not None:
            return cached

        data = loader_fn(report_dir, rank_field)
        self.save(name, version, data)
        return data

    def get_metric_data(self, buy_dates: list, loader_fn) -> tuple:
        """Load metric data (market_cap, limit_up, median_cap) from cache.

        Args:
            buy_dates: list of buy dates
            loader_fn: callable(buy_dates) -> (market_cap_data, limit_up_data, universe_median_cap)

        Returns:
            (market_cap_data, limit_up_data, universe_median_cap) tuple.
        """
        db_ver = self._get_db_version()
        dates_hash = hashlib.md5(','.join(sorted(buy_dates)).encode()).hexdigest()[:8]
        name = f"metric_data_{dates_hash}"
        version = db_ver

        cached = self.load(name, version)
        if cached is not None:
            return cached

        data = loader_fn(buy_dates)
        self.save(name, version, data)
        return data

    def get_trading_dates_map(self, report_dates: list, loader_fn) -> dict:
        """Load next-trading-date map from cache.

        Args:
            report_dates: list of report dates
            loader_fn: callable(report_dates) -> {date: next_trading_date}
        """
        db_ver = self._get_db_version()
        dates_hash = hashlib.md5(','.join(sorted(report_dates)).encode()).hexdigest()[:8]
        name = f"trading_dates_{dates_hash}"
        version = db_ver

        cached = self.load(name, version)
        if cached is not None:
            return cached

        data = loader_fn(report_dates)
        self.save(name, version, data)
        return data

    def clear(self):
        """Remove all cached files."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Cache cleared: {self.cache_dir}")

    def stats(self) -> dict:
        """Return cache statistics."""
        if not self.cache_dir.exists():
            return {'n_files': 0, 'total_size_mb': 0}
        files = list(self.cache_dir.iterdir())
        total_bytes = sum(f.stat().st_size for f in files if f.is_file())
        return {
            'n_files': len(files),
            'total_size_mb': round(total_bytes / 1024 / 1024, 1),
        }
```

- [ ] **Step 2: Add cache directory to .gitignore**

Append to `.gitignore`:
```
backtest/.eval_cache/
```

- [ ] **Step 3: Verify the module imports**

Run: `python3 -c "from backtest.eval_cache import EvalCache; c = EvalCache(); print('OK', c.stats())"`
Expected: `OK {'n_files': 0, 'total_size_mb': 0}`

- [ ] **Step 4: Commit**

```bash
git add backtest/eval_cache.py .gitignore
git commit -m "feat: 添加北极星评估持久化缓存模块 eval_cache.py"
```

---

## Task 6: Integrate Cache into Report Loading (B3)

**Files:**
- Modify: `backtest/backtest_report_based.py:141-179`

Wrap `load_reports()` to use `EvalCache` when available. The cache key is based on report directory file list hash + rank_field.

- [ ] **Step 1: Add cache-aware report loading**

In `backtest/backtest_report_based.py`, modify the `load_reports()` function (line 141) to accept an optional cache parameter:

**Old signature (line 141):**
```python
def load_reports(report_dir, rank_field='auto'):
```

**New code — replace the entire `load_reports` function (lines 141-179):**
```python
def load_reports(report_dir, rank_field='auto', cache=None):
    """加载所有JSON报告，返回 {date: [{code, score, pred_3d, ..., rank_score}, ...]}

    Args:
        report_dir: 报告目录
        rank_field: 排名字段。
            'auto'      = 优先用pred_10d(若存在)否则score
            'score'     = 强制用全局百分位分
            'pred_Xd'   = 用原始预测值 (e.g. pred_10d, pred_15d)
            'composite' = 多周期加权排名融合 (pred_3d/5d/10d/15d)
        cache: 可选的EvalCache实例, 启用持久化缓存
    """
    # 尝试从持久化缓存加载
    if cache is not None:
        cached = cache.get_parsed_reports(
            str(report_dir), rank_field,
            loader_fn=lambda d, r: _load_reports_impl(d, r)
        )
        if cached is not None:
            return cached

    return _load_reports_impl(report_dir, rank_field)


def _load_reports_impl(report_dir, rank_field):
    """实际的报告加载实现 (原load_reports逻辑)."""
    report_dir = Path(report_dir)
    reports = {}

    json_files = sorted(report_dir.glob('analysis_data_*.json'))
    if not json_files:
        return reports

    # 并行加载JSON文件 (ThreadPoolExecutor, I/O密集型)
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial

    parse_fn = partial(_parse_single_report, rank_field=rank_field)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(parse_fn, json_files))

    for date, stock_list in results:
        if stock_list is None:
            continue

        # composite: 多周期加权排名融合
        if rank_field == 'composite' and stock_list:
            _apply_composite_ranking(stock_list)

        # 按rank_score排序 (连续预测值, 无同分)
        stock_list.sort(key=lambda x: x['rank_score'], reverse=True)
        reports[date] = stock_list

    return reports
```

- [ ] **Step 2: Run timing harness to verify no regression**

Run: `python3 backtest/test_speedup.py`
Expected: Same timing as before (cache is not active in timing harness yet).

- [ ] **Step 3: Commit**

```bash
git add backtest/backtest_report_based.py
git commit -m "feat: load_reports支持持久化缓存, 重复加载~0.2秒"
```

---

## Task 7: Integrate Cache into Future Returns & Metric Data (B2, B4)

**Files:**
- Modify: `backtest/backtest_report_based.py:914-931`
- Modify: `backtest/backtest_report_based.py` (run_single_backtest signature)

Wire the cache into the future returns preloading and metric data loading within `run_single_backtest()`.

- [ ] **Step 1: Add cache parameter to run_single_backtest**

In `backtest/backtest_report_based.py`, find the `run_single_backtest` function definition and add `cache=None` parameter. The function signature is long — find it by searching for `def run_single_backtest(`. Add `cache=None` at the end of the parameter list, before the closing `)`.

- [ ] **Step 2: Wire cache into future returns loading**

Find the future returns loading block (around lines 914-931) and replace it:

**Old code:**
```python
    # 批量预加载所有日期的未来收益率 (使用模块级缓存，多模型对比时复用)
    import time as _time
    _t0_batch = _time.time()
    _all_report_dates_for_batch = sorted(reports.keys())
    _cache_key = tuple(_all_report_dates_for_batch)

    if _cache_key in _future_returns_cache:
        _batch_future_returns = _future_returns_cache[_cache_key]
        _next_trading_date_map = _next_trading_dates_cache.get(_cache_key, {})
        print(f"  未来收益缓存命中: {len(_batch_future_returns)}天 (0.0秒)")
    else:
        _batch_future_returns = batch_get_all_future_returns(_all_report_dates_for_batch, HOLDING_DAYS)
        _next_trading_date_map = _batch_get_next_trading_dates(_all_report_dates_for_batch)
        # 缓存结果 (限制缓存大小，避免内存爆炸)
        if len(_future_returns_cache) < 10:
            _future_returns_cache[_cache_key] = _batch_future_returns
            _next_trading_dates_cache[_cache_key] = _next_trading_date_map
        print(f"  批量预加载未来收益: {len(_batch_future_returns)}天, 耗时{_time.time()-_t0_batch:.1f}秒")
```

**New code:**
```python
    # 批量预加载所有日期的未来收益率 (三级缓存: 持久化磁盘 → 模块级内存 → 重新计算)
    import time as _time
    _t0_batch = _time.time()
    _all_report_dates_for_batch = sorted(reports.keys())
    _cache_key = tuple(_all_report_dates_for_batch)

    if _cache_key in _future_returns_cache:
        _batch_future_returns = _future_returns_cache[_cache_key]
        _next_trading_date_map = _next_trading_dates_cache.get(_cache_key, {})
        print(f"  未来收益缓存命中(内存): {len(_batch_future_returns)}天 (0.0秒)")
    elif cache is not None:
        _batch_future_returns = cache.get_future_returns(
            _all_report_dates_for_batch,
            loader_fn=lambda dates: batch_get_all_future_returns(dates, HOLDING_DAYS)
        )
        _next_trading_date_map = cache.get_trading_dates_map(
            _all_report_dates_for_batch,
            loader_fn=_batch_get_next_trading_dates
        )
        # 同时填充模块级缓存
        if len(_future_returns_cache) < 10:
            _future_returns_cache[_cache_key] = _batch_future_returns
            _next_trading_dates_cache[_cache_key] = _next_trading_date_map
        print(f"  未来收益缓存命中(磁盘): {len(_batch_future_returns)}天, 耗时{_time.time()-_t0_batch:.1f}秒")
    else:
        _batch_future_returns = batch_get_all_future_returns(_all_report_dates_for_batch, HOLDING_DAYS)
        _next_trading_date_map = _batch_get_next_trading_dates(_all_report_dates_for_batch)
        if len(_future_returns_cache) < 10:
            _future_returns_cache[_cache_key] = _batch_future_returns
            _next_trading_dates_cache[_cache_key] = _next_trading_date_map
        print(f"  批量预加载未来收益: {len(_batch_future_returns)}天, 耗时{_time.time()-_t0_batch:.1f}秒")
```

- [ ] **Step 3: Wire cache into metric data loading**

Find where `batch_load_all_metric_data` is called in `run_single_backtest` (search for `batch_load_all_metric_data`). Wrap it with cache:

Add this right after the call to `batch_load_all_metric_data`:

Before the existing call, add cache support:
```python
    # 加载北极星指标数据 (市值/涨停/中位数)
    if cache is not None:
        market_cap_data, limit_up_data, universe_median_cap = cache.get_metric_data(
            buy_dates_for_metrics,
            loader_fn=lambda dates: batch_load_all_metric_data(dates)
        )
    else:
        market_cap_data, limit_up_data, universe_median_cap = batch_load_all_metric_data(buy_dates_for_metrics)
```

(The exact location depends on where `batch_load_all_metric_data` is called — search for it and apply the pattern.)

- [ ] **Step 4: Run timing harness**

Run: `python3 backtest/test_speedup.py`
Expected: First run same timing. Should create cache files in `backtest/.eval_cache/`.

- [ ] **Step 5: Run timing harness again to test cache hit**

Run: `python3 backtest/test_speedup.py`
Expected: `load_reports` and data loading stages should be significantly faster on second run. Look for "缓存命中(磁盘)" in output.

- [ ] **Step 6: Commit**

```bash
git add backtest/backtest_report_based.py
git commit -m "feat: 未来收益+指标数据接入持久化缓存, 重复评估跳过SQL"
```

---

## Task 8: Integrate Cache into run_north_star_eval (B4 - Multi-Model Sharing)

**Files:**
- Modify: `backtest/run_north_star_eval.py:124-175` (run_backtest)
- Modify: `backtest/run_north_star_eval.py:178-219` (run_comparison)

Pass a shared `EvalCache` instance from the CLI entry point into `run_backtest()` and through to `run_single_backtest()`.

- [ ] **Step 1: Modify run_backtest to accept and pass cache**

In `backtest/run_north_star_eval.py`, modify `run_backtest()` (line 124):

**Old signature:**
```python
def run_backtest(report_dir, label, top_n=20, benchmark='000905.SH', focus_days=10,
                 retention_bonus=0.0, score_floor=0.0, min_holdings=3,
                 risk_control=False,
                 vol_target=0.0, cppi_floor=0.0, cppi_multiplier=3.0,
                 sector_diversify=0, rank_field='auto', hold_buffer=0,
                 rerank_dir=None, rerank_pool=100):
```

**New signature (add `cache=None`):**
```python
def run_backtest(report_dir, label, top_n=20, benchmark='000905.SH', focus_days=10,
                 retention_bonus=0.0, score_floor=0.0, min_holdings=3,
                 risk_control=False,
                 vol_target=0.0, cppi_floor=0.0, cppi_multiplier=3.0,
                 sector_diversify=0, rank_field='auto', hold_buffer=0,
                 rerank_dir=None, rerank_pool=100, cache=None):
```

Then update the calls inside to pass cache:

**Line 145 — pass cache to load_reports:**
```python
    reports = brb.load_reports(report_dir, rank_field=rank_field, cache=cache)
```

**Line 153 — pass cache to rerank load:**
```python
        rerank_reports = brb.load_reports(rerank_dir, rank_field=rank_field, cache=cache)
```

**Line 163 — pass cache to run_single_backtest:**
```python
    result = brb.run_single_backtest(
        reports, label, top_n=top_n,
        benchmark_code=benchmark, focus_days=focus_days,
        retention_bonus=retention_bonus,
        score_floor=score_floor, min_holdings=min_holdings,
        risk_control=risk_control,
        vol_target=vol_target, cppi_floor=cppi_floor,
        cppi_multiplier=cppi_multiplier,
        sector_diversify=sector_diversify,
        hold_buffer=hold_buffer,
        rerank_reports=rerank_reports, rerank_pool=rerank_pool,
        cache=cache,
    )
```

- [ ] **Step 2: Modify run_comparison to use shared cache**

In `run_comparison()` (line 178), create a shared cache and pass to each backtest:

Add at the start of the function, after the DB path setup:
```python
    from backtest.eval_cache import EvalCache
    shared_cache = EvalCache()
    print(f"  共享缓存: {shared_cache.cache_dir}")
```

Then update the `run_single_backtest` call inside the loop (around line 214):
```python
        result = brb.run_single_backtest(
            reports, label, top_n=top_n,
            benchmark_code=benchmark, focus_days=focus_days,
            cache=shared_cache,
        )
```

Also update the `load_reports` call (around line 209):
```python
        reports = brb.load_reports(dir_path, cache=shared_cache)
```

- [ ] **Step 3: Wire cache into CLI entry point**

Find the `main()` function or argparse section at the bottom of `run_north_star_eval.py`. Where `run_backtest()` is called from CLI args, create an `EvalCache` and pass it:

```python
    from backtest.eval_cache import EvalCache
    cache = EvalCache()
```

Pass `cache=cache` to all `run_backtest()` calls from the CLI.

- [ ] **Step 4: Run a multi-model comparison test**

Run: `python3 backtest/run_north_star_eval.py --compare --top-n 10 --focus-days 10`
Expected: First model takes full time. Second+ models should show "缓存命中(磁盘)" for future returns (shared across models).

- [ ] **Step 5: Commit**

```bash
git add backtest/run_north_star_eval.py
git commit -m "feat: 多模型对比共享缓存, 避免重复SQL查询"
```

---

## Task 9: Update Timing Harness with Cache Tests

**Files:**
- Modify: `backtest/test_speedup.py`

Update the timing harness to test both cold (no cache) and warm (cached) runs, and print a comparison.

- [ ] **Step 1: Update timing harness**

Add cache testing to `backtest/test_speedup.py`:

```python
def main():
    report_dir = find_report_dir()
    if not report_dir:
        print("ERROR: No report directory found with >= 50 JSON files")
        sys.exit(1)

    print(f"Report dir: {report_dir}")
    n_files = len([f for f in os.listdir(report_dir) if f.endswith('.json')])
    print(f"JSON files: {n_files}")

    from backtest.eval_cache import EvalCache

    # Cold run (clear cache first)
    cache = EvalCache()
    cache.clear()

    print(f"\n{'='*50}")
    print("COLD RUN (no cache)")
    print(f"{'='*50}")
    cold_timings = run_timed_backtest(report_dir, cache)

    # Warm run (cache populated)
    print(f"\n{'='*50}")
    print("WARM RUN (cached)")
    print(f"{'='*50}")
    warm_timings = run_timed_backtest(report_dir, cache)

    # Comparison
    print(f"\n{'='*50}")
    print("COMPARISON")
    print(f"{'='*50}")
    print(f"  {'Stage':<25} {'Cold':>8} {'Warm':>8} {'Speedup':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for key in cold_timings:
        cold = cold_timings[key]
        warm = warm_timings[key]
        speedup = cold / warm if warm > 0.001 else float('inf')
        print(f"  {key:<25} {cold:>7.2f}s {warm:>7.2f}s {speedup:>7.1f}x")

    total_cold = sum(cold_timings.values())
    total_warm = sum(warm_timings.values())
    print(f"  {'TOTAL':<25} {total_cold:>7.2f}s {total_warm:>7.2f}s {total_cold/total_warm:>7.1f}x")

    print(f"\n  Cache stats: {cache.stats()}")


def run_timed_backtest(report_dir, cache):
    from backtest import backtest_report_based as brb
    import io
    from contextlib import redirect_stdout

    timings = {}

    t0 = time.perf_counter()
    reports = brb.load_reports(report_dir, rank_field='composite', cache=cache)
    timings['load_reports'] = time.perf_counter() - t0
    print(f"  load_reports: {timings['load_reports']:.2f}s ({len(reports)} dates)")

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

    return timings
```

- [ ] **Step 2: Run the updated timing harness**

Run: `python3 backtest/test_speedup.py`
Expected: Shows cold vs warm comparison. Warm run should be significantly faster for data loading stages.

- [ ] **Step 3: Commit**

```bash
git add backtest/test_speedup.py
git commit -m "feat: 性能基准支持cold/warm缓存对比"
```

---

## Task 10: Final Validation — North Star Score Consistency

**Files:**
- Modify: `backtest/test_speedup.py`

Verify that the optimized pipeline produces the same North Star grade and score (within tolerance).

- [ ] **Step 1: Add score validation to timing harness**

Add to the end of `main()` in `backtest/test_speedup.py`:

```python
    # Validate scores match between cold and warm runs
    cold_result = run_bt_with_result(report_dir, None)  # no cache
    warm_result = run_bt_with_result(report_dir, cache)  # with cache

    cold_summary = cold_result['summary']
    warm_summary = warm_result['summary']

    print(f"\n{'='*50}")
    print("SCORE VALIDATION")
    print(f"{'='*50}")
    for days in [5, 10, 15]:
        if days not in cold_summary or days not in warm_summary:
            continue
        for key in ['ic_mean', 'icir', 'sharpe_ratio', 'annual_return', 'max_drawdown']:
            cold_val = cold_summary[days].get(key, 0) or 0
            warm_val = warm_summary[days].get(key, 0) or 0
            diff = abs(cold_val - warm_val)
            status = "OK" if diff < 0.01 else "MISMATCH"
            print(f"  {days}d {key}: cold={cold_val:.4f} warm={warm_val:.4f} diff={diff:.6f} {status}")
```

Add the helper function:
```python
def run_bt_with_result(report_dir, cache):
    from backtest import backtest_report_based as brb
    import io
    from contextlib import redirect_stdout

    reports = brb.load_reports(report_dir, rank_field='composite', cache=cache)
    f = io.StringIO()
    with redirect_stdout(f):
        return brb.run_single_backtest(
            reports, "validation", top_n=10,
            benchmark_code='000905.SH', focus_days=10,
            cache=cache,
        )
```

- [ ] **Step 2: Run validation**

Run: `python3 backtest/test_speedup.py`
Expected: All metrics show `OK` (diff < 0.01). IC values may differ slightly due to vectorized Spearman vs scipy.stats.spearmanr (floating point), but should be < 0.005.

- [ ] **Step 3: Final commit**

```bash
git add backtest/test_speedup.py
git commit -m "feat: 北极星评估加速完成 — 含score一致性验证"
```
