# ng1.2.3 三轴重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 ng1.0.1 (V5.2=72.1% A+) 基础上，同时改三轴 — drop 12 个弱特征 + 加 12 个 moneyflow 因子 + 加 6 个 mined 因子 + 引入 `excess - 0.3·downside` 软风险标签。验收门槛: V5.2 ≥ 73% AND MaxDD ≤ -10% AND Pre-2020 ≥ 70%。

**Architecture:** 三阶段 fast-check gate (~1 day) 决定是否进入完整训练 (~3 day)。每阶段 pass/fail 独立，单一失败可"砍那一轴 + 推进剩余"。复用现有 `_load_moneyflow_data` (`ng_cache_updater.py:475`) + 3-seed ensemble + auto-WF + 双向评估流水线。`PRODUCTION_VERSION='ng1.0.1'` 全程不动直至 manual gate 通过。

**Tech Stack:** Python 3.13, LightGBM/XGBoost/CatBoost 3-seed ensemble, SQLite, NG trainer (`ml_models/ng/ng_trainer.py`), pytest, factor mining pipeline (`scripts/factor_mining_pipeline.py`).

**Spec:** `docs/superpowers/specs/2026-04-14-ng123-design.md`

---

## 前置：文件结构

**新增文件**
- `ml_models/ng/ng123_label_transform.py` — `compute_path_min_kd` + `apply_downside_penalty`
- `ml_models/ng/ng123_moneyflow_factors.py` — 12 个 moneyflow 因子函数 + cs_rank 包装
- `ml_models/ng/ng123_mined_factors.py` — `MINED_FACTOR_SPEC` + 6 个因子计算函数
- `ml_models/ng/tests/test_ng123_label_transform.py`
- `ml_models/ng/tests/test_ng123_moneyflow_factors.py`
- `ml_models/ng/tests/test_ng123_mined_factors.py`
- `scripts/ng123/__init__.py`
- `scripts/ng123/stage1_moneyflow_ic.py` — Stage 1 IC + corr 矩阵
- `scripts/ng123/stage2_mined_validate.py` — Stage 2 mined re-validate + 跨 regime 稳定性
- `scripts/ng123/stage3_lambda_ablation.py` — Stage 3 λ ablation (5×2 mini-training)

**修改文件**
- `ml_models/ng/ng_schema.py` — 注册 `ng1.2.3` → `ng123_feature_cache`，扩 `_schema_sql` 加 `downside_kd` 列
- `ml_models/ng/ng_cache_updater.py` — `process_date` 流程里调用 ng123 因子函数 + 写 `downside_kd`
- `ml_models/ng/ng_feature_calculator.py` — 加 `compute_ng123_features(...)` 入口（不动 ng1.0.x 函数）
- `ml_models/ng/ng_trainer.py` — `--version ng1.2.3` 分支 + label 变换 hook
- `ml_models/ng/ng_production_scorer.py` — 验证 `version='ng1.2.3'` 自动加载 `ng123_*.pkl`

**输出目录** (gitignore 外，commit)
- `reports/ng123/fastcheck/{stage1_moneyflow_ic.csv,stage2_mined_factors.csv,stage3_lambda_ablation.csv,decision.md}`
- `reports/ng123/training/{seed42,seed123,seed456}/wf_summary.json`
- `reports/ng123/evaluation/{wf_oos_v52.md,pre2020_v52.md}`
- `reports/ng123/decision.md`
- (失败时) `docs/wiki/architecture/ng123_postmortem.md`

---

## Phase 0: 基础设施搭建

### Task 1: Schema 注册 + 表创建

**Files:**
- Modify: `ml_models/ng/ng_schema.py`
- Test: 通过 CLI smoke-test 验证

- [ ] **Step 1: 修改 `VERSION_TABLE_MAP` 加入 ng1.2.3**

打开 `ml_models/ng/ng_schema.py`，找到 `VERSION_TABLE_MAP` (line 19-30)，在 `'ng1.2.2'` 行后追加：

```python
    'ng1.2.3': 'ng123_feature_cache',  # 三轴重构: -12 弱特征 + 12 moneyflow + 6 mined + downside label
```

- [ ] **Step 2: 修改 `SCHEMA_VERSION_MAP` 加入 ng1.2.3 (own schema)**

找到 `SCHEMA_VERSION_MAP` (line 40-45)，追加：

```python
    'ng1.2.3': 'ng1.2.3',  # own schema (adds downside_kd label cols)
```

- [ ] **Step 3: 修改 `_schema_sql` 加 downside_kd 列 + 防止列名冲突 + 防止 ng1.2.1 列泄漏**

找到 `_schema_sql` 函数 (line 74-146)。需要做 3 处修改：

(a) 给 `if version_ge(ver, 'ng1.0.2'):` 块加 `not is_12` 守卫（避免与 ng1.2.3 新 `downside_10d` 命名冲突）：
```python
    if version_ge(ver, 'ng1.0.2') and not is_12:
        extra_cols = '\n    downside_10d REAL,'
```

(b) 给 `if is_12 and version_ge(ver, 'ng1.2.1'):` 块加上界守卫（避免 vn_label_/path_ 列泄漏到 ng1.2.3）：
```python
    if is_12 and version_ge(ver, 'ng1.2.1') and not version_ge(ver, 'ng1.2.3'):
        extra_cols += '\n    vn_label_3d REAL,'
        ...（保持原 7 列不变）
```

(c) 在 ng1.2.1 block 后追加 ng1.2.3 块（含前瞻注释）：

```python
    # ng1.2.3 adds soft-downside label columns (per spec section 5).
    # IMPORTANT: When ng1.2.4 is added with its own schema, add an upper-bound
    # guard `and not version_ge(ver, 'ng1.2.4')` here AND on the ng1.2.1 block.
    if is_12 and version_ge(ver, 'ng1.2.3'):
        extra_cols += '\n    downside_3d REAL,'
        extra_cols += '\n    downside_5d REAL,'
        extra_cols += '\n    downside_10d REAL,'
        extra_cols += '\n    downside_15d REAL,'
```

⚠️ 当年原 plan 漏掉了 (a)(b)，导致首次实施时一是 SQL duplicate column error，二是 ng1.2.1 的 vn_label/path 列泄漏到 ng123_feature_cache。这两个 guard 现已写入 plan 防止再踩。

- [ ] **Step 4: 创建表 + smoke-test**

```bash
python3 ml_models/ng/ng_schema.py ng1.2.3
```

Expected output:
```
ng123_feature_cache table ready: /Users/yangxu/StockTradebyZ/data_adapter/stock_data.db
```

验证表存在：
```bash
python3 -c "import sqlite3; c=sqlite3.connect('data_adapter/stock_data.db'); cols=c.execute('PRAGMA table_info(ng123_feature_cache)').fetchall(); print('\n'.join(f'{r[1]} {r[2]}' for r in cols))"
```

Expected: 列表中应包含 `downside_3d`, `downside_5d`, `downside_10d`, `downside_15d` 共 4 个新列。

- [ ] **Step 5: Commit**

```bash
git add ml_models/ng/ng_schema.py
git commit -m "feat(ng123): 注册 ng1.2.3 schema + downside_kd 标签列"
```

---

### Task 2: 创建 scripts/ng123 目录

- [ ] **Step 1: 创建目录 + __init__.py**

```bash
mkdir -p scripts/ng123 reports/ng123/fastcheck reports/ng123/training reports/ng123/evaluation
touch scripts/ng123/__init__.py
```

- [ ] **Step 2: Commit (空目录占位)**

```bash
# .gitkeep 不必要 — 后续 task 会写文件进去
```

---

### Task 3: Label 变换模块 (TDD)

**Files:**
- Create: `ml_models/ng/ng123_label_transform.py`
- Create: `ml_models/ng/tests/test_ng123_label_transform.py`

- [ ] **Step 1: 写测试 (failing)**

创建 `ml_models/ng/tests/test_ng123_label_transform.py`：

```python
"""Unit tests for ng1.2.3 soft-downside label transform."""
import numpy as np
import pytest

from ml_models.ng.ng123_label_transform import (
    compute_path_min_kd,
    compute_downside_kd,
    apply_downside_penalty,
)


# --- compute_path_min_kd ----------------------------------------------------

def test_path_min_simple_decline():
    """Closes drop monotonically; path_min = lowest / today - 1."""
    today_close = 100.0
    future = np.array([99.0, 95.0, 92.0, 90.0, 88.0])
    pm = compute_path_min_kd(today_close, future)
    assert abs(pm - (88.0 / 100.0 - 1.0)) < 1e-9  # = -0.12


def test_path_min_all_above_today():
    """If all future closes > today, path_min is positive (rare but valid)."""
    today_close = 100.0
    future = np.array([101.0, 102.0, 105.0])
    pm = compute_path_min_kd(today_close, future)
    assert abs(pm - (101.0 / 100.0 - 1.0)) < 1e-9  # = +0.01


def test_path_min_with_recovery():
    """Drops then recovers; path_min captures the trough."""
    today_close = 100.0
    future = np.array([95.0, 88.0, 92.0, 105.0])  # bottoms at 88
    pm = compute_path_min_kd(today_close, future)
    assert abs(pm - (88.0 / 100.0 - 1.0)) < 1e-9


def test_path_min_empty_future():
    """No future data → NaN."""
    pm = compute_path_min_kd(100.0, np.array([]))
    assert np.isnan(pm)


def test_path_min_zero_today():
    """Today close = 0 → NaN (avoid div by zero)."""
    pm = compute_path_min_kd(0.0, np.array([1.0, 2.0]))
    assert np.isnan(pm)


# --- compute_downside_kd ----------------------------------------------------

def test_downside_from_negative_path_min():
    """Negative path_min → downside is its magnitude."""
    assert abs(compute_downside_kd(-0.12) - 0.12) < 1e-9


def test_downside_from_positive_path_min():
    """Positive path_min → downside = 0 (no drawdown)."""
    assert compute_downside_kd(0.05) == 0.0


def test_downside_from_zero():
    assert compute_downside_kd(0.0) == 0.0


def test_downside_nan_propagates():
    assert np.isnan(compute_downside_kd(np.nan))


# --- apply_downside_penalty -------------------------------------------------

def test_penalty_clean_winner():
    """excess=+0.05, downside=0.03, λ=0.3 → label = 0.05 - 0.009 = 0.041."""
    res = apply_downside_penalty(excess=0.05, downside=0.03, lam=0.3)
    assert abs(res - 0.041) < 1e-9


def test_penalty_volatile_winner():
    """excess=+0.05, downside=0.15, λ=0.3 → label = 0.05 - 0.045 = 0.005 (demoted)."""
    res = apply_downside_penalty(excess=0.05, downside=0.15, lam=0.3)
    assert abs(res - 0.005) < 1e-9


def test_penalty_lambda_zero_passthrough():
    """λ=0 → label = excess (sanity for ablation)."""
    res = apply_downside_penalty(excess=0.05, downside=0.15, lam=0.0)
    assert abs(res - 0.05) < 1e-9


def test_penalty_nan_excess():
    res = apply_downside_penalty(excess=np.nan, downside=0.05, lam=0.3)
    assert np.isnan(res)


def test_penalty_nan_downside():
    res = apply_downside_penalty(excess=0.05, downside=np.nan, lam=0.3)
    assert np.isnan(res)


def test_penalty_vectorized():
    """Function must accept arrays."""
    excess = np.array([0.05, 0.10, -0.05, 0.0])
    downside = np.array([0.03, 0.15, 0.20, 0.0])
    res = apply_downside_penalty(excess=excess, downside=downside, lam=0.3)
    expected = np.array([0.041, 0.055, -0.110, 0.0])
    assert np.allclose(res, expected, atol=1e-9)
```

- [ ] **Step 2: 运行测试，验证失败**

```bash
cd /Users/yangxu/StockTradebyZ
python3 -m pytest ml_models/ng/tests/test_ng123_label_transform.py -v
```

Expected: ALL FAIL — `ModuleNotFoundError: No module named 'ml_models.ng.ng123_label_transform'`

- [ ] **Step 3: 写最小实现**

创建 `ml_models/ng/ng123_label_transform.py`：

```python
"""ng1.2.3 soft-downside label transform.

Per spec §5 (docs/superpowers/specs/2026-04-14-ng123-design.md):
    label_kd = industry_excess_kd - lambda * max(0, -path_min_kd)

with default lambda=0.3 (1/5 of ng1.0.4's failed 1.5).
"""
from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]

DEFAULT_LAMBDA = 0.3


def compute_path_min_kd(today_close: float, future_closes: np.ndarray) -> float:
    """Return min(future_closes) / today_close - 1.

    Args:
        today_close: scalar close at date t (anchor).
        future_closes: 1-D array of closes from t+1 to t+k.

    Returns:
        path_min in [-1, +inf), typically negative. NaN if input invalid.
    """
    if today_close is None or today_close <= 0 or np.isnan(today_close):
        return np.nan
    if future_closes is None or len(future_closes) == 0:
        return np.nan
    arr = np.asarray(future_closes, dtype=np.float64)
    if not np.any(np.isfinite(arr)):
        return np.nan
    return float(np.nanmin(arr) / today_close - 1.0)


def compute_downside_kd(path_min: ArrayLike) -> ArrayLike:
    """downside = max(0, -path_min). NaN propagates."""
    if isinstance(path_min, np.ndarray):
        return np.where(np.isnan(path_min), np.nan, np.maximum(0.0, -path_min))
    if path_min is None or (isinstance(path_min, float) and np.isnan(path_min)):
        return np.nan
    return max(0.0, -float(path_min))


def apply_downside_penalty(
    excess: ArrayLike, downside: ArrayLike, lam: float = DEFAULT_LAMBDA
) -> ArrayLike:
    """Return excess - lam * downside (NaN-safe; broadcast on arrays)."""
    if lam < 0:
        raise ValueError(f"lambda must be non-negative, got {lam}")
    excess_arr = np.asarray(excess, dtype=np.float64)
    downside_arr = np.asarray(downside, dtype=np.float64)
    return excess_arr - lam * downside_arr
```

- [ ] **Step 4: 运行测试，验证通过**

```bash
python3 -m pytest ml_models/ng/tests/test_ng123_label_transform.py -v
```

Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add ml_models/ng/ng123_label_transform.py ml_models/ng/tests/test_ng123_label_transform.py
git commit -m "feat(ng123): label 变换模块 — path_min + downside + soft penalty (TDD)"
```

---

## Phase 1 - Stage 1: 12 个 Moneyflow 因子实现 (TDD)

### Task 4: Moneyflow 聚合 helper

**Files:**
- Create: `ml_models/ng/ng123_moneyflow_factors.py` (initial structure + helper)
- Create: `ml_models/ng/tests/test_ng123_moneyflow_factors.py` (helper tests only)

- [ ] **Step 1: 写 helper 测试**

创建 `ml_models/ng/tests/test_ng123_moneyflow_factors.py`：

```python
"""Unit tests for ng1.2.3 moneyflow factors."""
import numpy as np
import pytest

from ml_models.ng.ng123_moneyflow_factors import (
    aggregate_moneyflow_window,
    EMPTY_MF_RESULT,
)


def _mk_row(buy_sm=0, sell_sm=0, buy_md=0, sell_md=0,
            buy_lg=0, sell_lg=0, buy_elg=0, sell_elg=0):
    return {
        'buy_sm_amount': buy_sm, 'sell_sm_amount': sell_sm,
        'buy_md_amount': buy_md, 'sell_md_amount': sell_md,
        'buy_lg_amount': buy_lg, 'sell_lg_amount': sell_lg,
        'buy_elg_amount': buy_elg, 'sell_elg_amount': sell_elg,
    }


def test_aggregate_5d_simple():
    """5 days with consistent +100 net_elg each day → sum_net_elg=500."""
    rows = [_mk_row(buy_elg=200, sell_elg=100)] * 5  # net_elg = 100/day
    agg = aggregate_moneyflow_window(rows, n_days=5)
    assert agg['sum_net_elg'] == 500
    assert agg['sum_buy_elg'] == 1000
    assert agg['sum_sell_elg'] == 500


def test_aggregate_total_amount():
    """sum(buy_total + sell_total) over window."""
    rows = [_mk_row(buy_sm=10, sell_sm=10, buy_lg=20, sell_lg=20)] * 3
    agg = aggregate_moneyflow_window(rows, n_days=3)
    # Per row: total = (10+10) + (20+20) = 60. Over 3 days: 180.
    assert agg['sum_total_amount'] == 180


def test_aggregate_empty_returns_empty():
    """No rows → empty dict (all NaN downstream)."""
    agg = aggregate_moneyflow_window([], n_days=5)
    assert agg == EMPTY_MF_RESULT


def test_aggregate_fewer_rows_than_n_days():
    """Only 3 rows but n_days=5 → use what's available."""
    rows = [_mk_row(buy_elg=100, sell_elg=50)] * 3
    agg = aggregate_moneyflow_window(rows, n_days=5)
    assert agg['sum_net_elg'] == 150  # 50 * 3
    assert agg['n_days_actual'] == 3


def test_aggregate_takes_last_n():
    """Function uses LAST n_days rows (most recent)."""
    rows = [_mk_row(buy_elg=10, sell_elg=0)] * 3 + [_mk_row(buy_elg=100, sell_elg=0)] * 5
    agg = aggregate_moneyflow_window(rows, n_days=5)
    # Last 5 rows: each has buy_elg=100. sum_net_elg = 500.
    assert agg['sum_net_elg'] == 500


def test_daily_signs():
    """Daily sign array length matches n_days, in correct order (oldest → newest)."""
    rows = [
        _mk_row(buy_elg=100, sell_elg=50),   # net +50, sign +1
        _mk_row(buy_elg=50, sell_elg=100),   # net -50, sign -1
        _mk_row(buy_elg=100, sell_elg=100),  # net 0, sign 0
    ]
    agg = aggregate_moneyflow_window(rows, n_days=3)
    assert agg['daily_sign_net_elg'].tolist() == [1, -1, 0]
```

- [ ] **Step 2: 写最小 helper 实现**

创建 `ml_models/ng/ng123_moneyflow_factors.py`：

```python
"""ng1.2.3 moneyflow factor functions.

Per spec §4.1 (docs/superpowers/specs/2026-04-14-ng123-design.md):
  - 12 factors split into 4 groups (A: net flow, B: persistence, C: divergence, D: cs_rank)
  - All factors NaN-safe; division-by-zero guarded with +1e-8

Input shape: List[Dict] returned by ng_cache_updater._load_moneyflow_data
  Each dict has keys: buy_{sm,md,lg,elg}_amount, sell_{sm,md,lg,elg}_amount, net_mf_amount
  Sorted oldest → newest, length up to 20.
"""
from typing import Dict, List

import numpy as np


# Sentinel: returned when no moneyflow data available
EMPTY_MF_RESULT = {
    'sum_net_sm': np.nan, 'sum_net_md': np.nan,
    'sum_net_lg': np.nan, 'sum_net_elg': np.nan,
    'sum_buy_elg': np.nan, 'sum_sell_elg': np.nan,
    'sum_total_amount': np.nan,
    'daily_sign_net_elg': np.array([], dtype=np.int8),
    'daily_sign_net_lg': np.array([], dtype=np.int8),
    'daily_sign_net_sm': np.array([], dtype=np.int8),
    'n_days_actual': 0,
}


def aggregate_moneyflow_window(
    rows: List[Dict], n_days: int
) -> Dict:
    """Aggregate last n_days of moneyflow rows into summary stats.

    Returns dict with sums per order-size class + per-day sign arrays.
    """
    if not rows or n_days <= 0:
        return EMPTY_MF_RESULT.copy()

    window = rows[-n_days:]  # take last n_days (oldest → newest preserved)
    n_actual = len(window)

    # Per-day net flows (buy - sell) for each class
    net_sm = np.array(
        [(r.get('buy_sm_amount') or 0) - (r.get('sell_sm_amount') or 0) for r in window],
        dtype=np.float64)
    net_md = np.array(
        [(r.get('buy_md_amount') or 0) - (r.get('sell_md_amount') or 0) for r in window],
        dtype=np.float64)
    net_lg = np.array(
        [(r.get('buy_lg_amount') or 0) - (r.get('sell_lg_amount') or 0) for r in window],
        dtype=np.float64)
    net_elg = np.array(
        [(r.get('buy_elg_amount') or 0) - (r.get('sell_elg_amount') or 0) for r in window],
        dtype=np.float64)

    sum_buy_elg = float(sum((r.get('buy_elg_amount') or 0) for r in window))
    sum_sell_elg = float(sum((r.get('sell_elg_amount') or 0) for r in window))

    # Total amount = sum of all (buy + sell) across all 4 classes
    sum_total_amount = 0.0
    for r in window:
        for k in ('buy_sm_amount', 'sell_sm_amount', 'buy_md_amount', 'sell_md_amount',
                  'buy_lg_amount', 'sell_lg_amount', 'buy_elg_amount', 'sell_elg_amount'):
            sum_total_amount += (r.get(k) or 0)

    return {
        'sum_net_sm': float(net_sm.sum()),
        'sum_net_md': float(net_md.sum()),
        'sum_net_lg': float(net_lg.sum()),
        'sum_net_elg': float(net_elg.sum()),
        'sum_buy_elg': sum_buy_elg,
        'sum_sell_elg': sum_sell_elg,
        'sum_total_amount': sum_total_amount,
        'daily_sign_net_elg': np.sign(net_elg).astype(np.int8),
        'daily_sign_net_lg': np.sign(net_lg).astype(np.int8),
        'daily_sign_net_sm': np.sign(net_sm).astype(np.int8),
        'n_days_actual': n_actual,
    }
```

- [ ] **Step 3: 运行测试，验证通过**

```bash
python3 -m pytest ml_models/ng/tests/test_ng123_moneyflow_factors.py -v
```

Expected: 6 passed.

- [ ] **Step 4: Commit**

```bash
git add ml_models/ng/ng123_moneyflow_factors.py ml_models/ng/tests/test_ng123_moneyflow_factors.py
git commit -m "feat(ng123): moneyflow 聚合 helper + EMPTY 哨兵 (TDD)"
```

---

### Task 5: Group A - 4 个净流入幅度因子

**Files:**
- Modify: `ml_models/ng/ng123_moneyflow_factors.py`
- Modify: `ml_models/ng/tests/test_ng123_moneyflow_factors.py`

- [ ] **Step 1: 追加测试**

在测试文件末尾追加：

```python
# --- Group A: Net Flow Magnitude (4 factors) -------------------------------

def test_mf_net_elg_5d_ratio_basic():
    """5 days each net_elg=+100, total_amount=+1000 → ratio = 500/5000 = 0.10"""
    from ml_models.ng.ng123_moneyflow_factors import compute_group_a_factors
    rows = [_mk_row(buy_elg=200, sell_elg=100, buy_sm=400, sell_sm=400)] * 5
    # Per row: net_elg = 100, total = 200+100+400+400 = 1100
    # 5d: sum_net_elg = 500, sum_total = 5500 → ratio = 500/5500 ≈ 0.0909
    res = compute_group_a_factors(rows)
    assert abs(res['mf_net_elg_5d_ratio'] - 500/5500) < 1e-6


def test_mf_net_elg_5d_ratio_zero_total():
    """Edge case: all amounts zero → NaN (not div by zero)."""
    from ml_models.ng.ng123_moneyflow_factors import compute_group_a_factors
    rows = [_mk_row()] * 5  # all zeros
    res = compute_group_a_factors(rows)
    assert np.isnan(res['mf_net_elg_5d_ratio'])


def test_mf_net_elg_20d_ratio():
    """20d ratio aggregates over 20 days when available."""
    from ml_models.ng.ng123_moneyflow_factors import compute_group_a_factors
    rows = [_mk_row(buy_elg=100, sell_elg=50, buy_sm=50, sell_sm=50)] * 25
    # Last 20 used: net_elg=50/d * 20 = 1000; total=(100+50+50+50)/d * 20 = 5000
    # ratio = 1000/5000 = 0.2
    res = compute_group_a_factors(rows)
    assert abs(res['mf_net_elg_20d_ratio'] - 0.2) < 1e-6


def test_mf_net_lg_5d_ratio():
    """Large-order net flow ratio (parallel to elg)."""
    from ml_models.ng.ng123_moneyflow_factors import compute_group_a_factors
    rows = [_mk_row(buy_lg=300, sell_lg=200, buy_sm=100, sell_sm=100)] * 5
    # Per row: net_lg=100, total=300+200+100+100=700; 5d: 500/3500≈0.1429
    res = compute_group_a_factors(rows)
    assert abs(res['mf_net_lg_5d_ratio'] - 500/3500) < 1e-6


def test_mf_smart_net_share_20d():
    """Share of (net_elg+net_lg) over total absolute net flow."""
    from ml_models.ng.ng123_moneyflow_factors import compute_group_a_factors
    # net_elg=+100, net_lg=+50, net_md=-30, net_sm=-20 per day, 20 days
    rows = [_mk_row(buy_elg=200, sell_elg=100, buy_lg=150, sell_lg=100,
                     buy_md=70, sell_md=100, buy_sm=80, sell_sm=100)] * 20
    # Per day: net = +100, +50, -30, -20 → smart sum = +150, abs sum = 200
    # 20d: smart_sum = 3000, abs_sum = 4000 → share = 0.75
    res = compute_group_a_factors(rows)
    assert abs(res['mf_smart_net_share_20d'] - 0.75) < 1e-6


def test_group_a_empty_input():
    """No rows → all 4 NaN."""
    from ml_models.ng.ng123_moneyflow_factors import compute_group_a_factors
    res = compute_group_a_factors([])
    assert all(np.isnan(res[k]) for k in
               ['mf_net_elg_5d_ratio', 'mf_net_elg_20d_ratio',
                'mf_net_lg_5d_ratio', 'mf_smart_net_share_20d'])
```

- [ ] **Step 2: 运行测试 (failing)**

```bash
python3 -m pytest ml_models/ng/tests/test_ng123_moneyflow_factors.py -v -k group_a
```

Expected: ImportError on `compute_group_a_factors`.

- [ ] **Step 3: 实现 Group A**

在 `ml_models/ng/ng123_moneyflow_factors.py` 末尾追加：

```python
# ---------------------------------------------------------------------------
# Group A: Smart Money Net Flow Magnitude (4 factors)
# ---------------------------------------------------------------------------

def compute_group_a_factors(rows: List[Dict]) -> Dict[str, float]:
    """Compute factors 1-4 from spec §4.1 Group A."""
    result = {
        'mf_net_elg_5d_ratio': np.nan,
        'mf_net_elg_20d_ratio': np.nan,
        'mf_net_lg_5d_ratio': np.nan,
        'mf_smart_net_share_20d': np.nan,
    }
    if not rows:
        return result

    agg5 = aggregate_moneyflow_window(rows, n_days=5)
    agg20 = aggregate_moneyflow_window(rows, n_days=20)

    # Factor 1: mf_net_elg_5d_ratio = sum_net_elg_5d / sum_total_5d
    if agg5['n_days_actual'] > 0 and agg5['sum_total_amount'] > 1e-8:
        result['mf_net_elg_5d_ratio'] = agg5['sum_net_elg'] / agg5['sum_total_amount']

    # Factor 2: mf_net_elg_20d_ratio = sum_net_elg_20d / sum_total_20d
    if agg20['n_days_actual'] > 0 and agg20['sum_total_amount'] > 1e-8:
        result['mf_net_elg_20d_ratio'] = agg20['sum_net_elg'] / agg20['sum_total_amount']

    # Factor 3: mf_net_lg_5d_ratio = sum_net_lg_5d / sum_total_5d
    if agg5['n_days_actual'] > 0 and agg5['sum_total_amount'] > 1e-8:
        result['mf_net_lg_5d_ratio'] = agg5['sum_net_lg'] / agg5['sum_total_amount']

    # Factor 4: mf_smart_net_share_20d = sum(net_elg + net_lg, 20d) / sum(|net_*|, 20d)
    if agg20['n_days_actual'] > 0:
        smart_net = agg20['sum_net_elg'] + agg20['sum_net_lg']
        abs_net_total = (abs(agg20['sum_net_elg']) + abs(agg20['sum_net_lg'])
                         + abs(agg20['sum_net_md']) + abs(agg20['sum_net_sm']))
        if abs_net_total > 1e-8:
            result['mf_smart_net_share_20d'] = smart_net / abs_net_total

    return result
```

- [ ] **Step 4: 运行测试 (passing)**

```bash
python3 -m pytest ml_models/ng/tests/test_ng123_moneyflow_factors.py -v -k group_a
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add ml_models/ng/ng123_moneyflow_factors.py ml_models/ng/tests/test_ng123_moneyflow_factors.py
git commit -m "feat(ng123): Group A 4 个 moneyflow 净流入幅度因子 (TDD)"
```

---

### Task 6: Group B + C - 持续性 + 分歧/加速 (4 因子)

**Files:**
- Modify: `ml_models/ng/ng123_moneyflow_factors.py`
- Modify: `ml_models/ng/tests/test_ng123_moneyflow_factors.py`

- [ ] **Step 1: 追加测试**

```python
# --- Group B: Persistence (2 factors) --------------------------------------

def test_mf_elg_persistence_20d_all_positive():
    """20 days all net_elg > 0 → persistence = +1.0"""
    from ml_models.ng.ng123_moneyflow_factors import compute_group_b_factors
    rows = [_mk_row(buy_elg=100, sell_elg=50)] * 20
    res = compute_group_b_factors(rows)
    assert abs(res['mf_elg_persistence_20d'] - 1.0) < 1e-9


def test_mf_elg_persistence_20d_mixed():
    """10 positive + 10 negative → persistence = 0."""
    from ml_models.ng.ng123_moneyflow_factors import compute_group_b_factors
    pos = _mk_row(buy_elg=100, sell_elg=50)
    neg = _mk_row(buy_elg=50, sell_elg=100)
    rows = [pos] * 10 + [neg] * 10
    res = compute_group_b_factors(rows)
    assert abs(res['mf_elg_persistence_20d'] - 0.0) < 1e-9


def test_mf_smart_consistency_5d_all_aligned():
    """All 5 days net_elg sign = net_lg sign → consistency = 1.0"""
    from ml_models.ng.ng123_moneyflow_factors import compute_group_b_factors
    rows = [_mk_row(buy_elg=100, sell_elg=50, buy_lg=80, sell_lg=40)] * 5
    res = compute_group_b_factors(rows)
    assert abs(res['mf_smart_consistency_5d'] - 1.0) < 1e-9


def test_mf_smart_consistency_5d_misaligned():
    """3 aligned + 2 misaligned → consistency = 0.6"""
    from ml_models.ng.ng123_moneyflow_factors import compute_group_b_factors
    aligned = _mk_row(buy_elg=100, sell_elg=50, buy_lg=80, sell_lg=40)  # both +
    misaligned = _mk_row(buy_elg=100, sell_elg=50, buy_lg=40, sell_lg=80)  # elg+, lg-
    rows = [aligned] * 3 + [misaligned] * 2
    res = compute_group_b_factors(rows)
    assert abs(res['mf_smart_consistency_5d'] - 0.6) < 1e-9


# --- Group C: Divergence + Acceleration (2 factors) ------------------------

def test_mf_smart_retail_divergence_5d_smart_in_retail_out():
    """net_elg_5d > 0, net_sm_5d < 0 → divergence = sign(+) - sign(-) = +2"""
    from ml_models.ng.ng123_moneyflow_factors import compute_group_c_factors
    rows = [_mk_row(buy_elg=200, sell_elg=100,  # net_elg = +100
                     buy_sm=50, sell_sm=200)] * 5  # net_sm = -150
    res = compute_group_c_factors(rows)
    assert res['mf_smart_retail_divergence_5d'] == 2


def test_mf_smart_retail_divergence_5d_aligned():
    """Both positive → divergence = 0"""
    from ml_models.ng.ng123_moneyflow_factors import compute_group_c_factors
    rows = [_mk_row(buy_elg=200, sell_elg=100, buy_sm=200, sell_sm=100)] * 5
    res = compute_group_c_factors(rows)
    assert res['mf_smart_retail_divergence_5d'] == 0


def test_mf_elg_acceleration_5_20():
    """5d ratio - 20d ratio: if 5d more positive, acceleration positive."""
    from ml_models.ng.ng123_moneyflow_factors import compute_group_c_factors
    # 15 days neutral (net_elg=0), 5 days positive (net_elg=+100, total=200/day)
    neutral = _mk_row(buy_elg=100, sell_elg=100, buy_sm=0, sell_sm=0)  # total=200
    positive = _mk_row(buy_elg=200, sell_elg=100, buy_sm=0, sell_sm=0)  # net=+100, total=300
    rows = [neutral] * 15 + [positive] * 5
    res = compute_group_c_factors(rows)
    # 5d: net_elg_sum=500, total=1500 → ratio=0.333
    # 20d: net_elg_sum=500, total=15*200+5*300=4500 → ratio=0.111
    # acc = 0.333 - 0.111 = 0.222
    assert abs(res['mf_elg_acceleration_5_20'] - (500/1500 - 500/4500)) < 1e-6
```

- [ ] **Step 2: 运行测试 (failing)**

```bash
python3 -m pytest ml_models/ng/tests/test_ng123_moneyflow_factors.py -v -k "group_b or group_c"
```

Expected: ImportError.

- [ ] **Step 3: 实现 Group B + Group C**

在 `ml_models/ng/ng123_moneyflow_factors.py` 末尾追加：

```python
# ---------------------------------------------------------------------------
# Group B: Persistence (2 factors)
# ---------------------------------------------------------------------------

def compute_group_b_factors(rows: List[Dict]) -> Dict[str, float]:
    """Compute factors 5-6 from spec §4.1 Group B."""
    result = {
        'mf_elg_persistence_20d': np.nan,
        'mf_smart_consistency_5d': np.nan,
    }
    if not rows:
        return result

    agg20 = aggregate_moneyflow_window(rows, n_days=20)
    agg5 = aggregate_moneyflow_window(rows, n_days=5)

    # Factor 5: mf_elg_persistence_20d = sum(sign(net_elg_daily), 20d) / 20
    if agg20['n_days_actual'] > 0:
        signs = agg20['daily_sign_net_elg']
        # Use n_days_actual as denominator for short-history stocks
        result['mf_elg_persistence_20d'] = float(signs.sum()) / agg20['n_days_actual']

    # Factor 6: mf_smart_consistency_5d = fraction of 5d where sign(net_elg)=sign(net_lg)
    if agg5['n_days_actual'] > 0:
        s_elg = agg5['daily_sign_net_elg']
        s_lg = agg5['daily_sign_net_lg']
        aligned = (s_elg == s_lg).astype(np.float64)
        result['mf_smart_consistency_5d'] = float(aligned.sum()) / agg5['n_days_actual']

    return result


# ---------------------------------------------------------------------------
# Group C: Divergence + Acceleration (2 factors)
# ---------------------------------------------------------------------------

def compute_group_c_factors(rows: List[Dict]) -> Dict[str, float]:
    """Compute factors 7-8 from spec §4.1 Group C."""
    result = {
        'mf_smart_retail_divergence_5d': np.nan,
        'mf_elg_acceleration_5_20': np.nan,
    }
    if not rows:
        return result

    agg5 = aggregate_moneyflow_window(rows, n_days=5)
    agg20 = aggregate_moneyflow_window(rows, n_days=20)

    # Factor 7: mf_smart_retail_divergence_5d = sign(sum_net_elg_5d) - sign(sum_net_sm_5d)
    if agg5['n_days_actual'] > 0:
        sign_elg = float(np.sign(agg5['sum_net_elg']))
        sign_sm = float(np.sign(agg5['sum_net_sm']))
        result['mf_smart_retail_divergence_5d'] = sign_elg - sign_sm

    # Factor 8: mf_elg_acceleration_5_20 = ratio_5d - ratio_20d
    if (agg5['n_days_actual'] > 0 and agg20['n_days_actual'] > 0
            and agg5['sum_total_amount'] > 1e-8 and agg20['sum_total_amount'] > 1e-8):
        ratio_5 = agg5['sum_net_elg'] / agg5['sum_total_amount']
        ratio_20 = agg20['sum_net_elg'] / agg20['sum_total_amount']
        result['mf_elg_acceleration_5_20'] = ratio_5 - ratio_20

    return result
```

- [ ] **Step 4: 运行测试 (passing)**

```bash
python3 -m pytest ml_models/ng/tests/test_ng123_moneyflow_factors.py -v -k "group_b or group_c"
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add ml_models/ng/ng123_moneyflow_factors.py ml_models/ng/tests/test_ng123_moneyflow_factors.py
git commit -m "feat(ng123): Group B+C 4 个因子 — 持续性 + 分歧/加速 (TDD)"
```

---

### Task 7: Group D - 4 个 cs_rank 因子 (industry-relative)

**Files:**
- Modify: `ml_models/ng/ng123_moneyflow_factors.py`
- Modify: `ml_models/ng/tests/test_ng123_moneyflow_factors.py`

- [ ] **Step 1: 追加测试**

```python
# --- Group D: Cross-Sectional Industry Ranks (4 factors) --------------------

def test_compute_stock_mf_scalars_for_cs_rank():
    """Helper that returns scalar values needed for cs_rank wrapper."""
    from ml_models.ng.ng123_moneyflow_factors import compute_stock_mf_scalars
    rows = [_mk_row(buy_elg=200, sell_elg=100, buy_sm=400, sell_sm=400)] * 20
    s = compute_stock_mf_scalars(rows)
    # Should expose net_elg_5d_ratio, net_elg_20d_ratio, smart_net_share_20d, persistence_20d
    assert 'net_elg_5d_ratio' in s
    assert 'net_elg_20d_ratio' in s
    assert 'smart_net_share_20d' in s
    assert 'persistence_20d' in s


def test_compute_group_d_factors_basic():
    """cs_rank factors return percentile rank in [0, 1]."""
    from ml_models.ng.ng123_moneyflow_factors import compute_group_d_factors
    # Stock self values
    stock_scalars = {
        'net_elg_5d_ratio': 0.10,
        'net_elg_20d_ratio': 0.05,
        'smart_net_share_20d': 0.30,
        'persistence_20d': 0.40,
    }
    # Peer values: 5 peers including self
    peer_scalars = {
        'net_elg_5d_ratio': np.array([0.02, 0.05, 0.08, 0.10, 0.15]),  # self at rank 4 of 5
        'net_elg_20d_ratio': np.array([0.01, 0.03, 0.05, 0.05, 0.10]),  # tied
        'smart_net_share_20d': np.array([-0.20, 0.10, 0.30, 0.40, 0.50]),
        'persistence_20d': np.array([-0.50, 0.0, 0.40, 0.60, 0.80]),
    }
    res = compute_group_d_factors(stock_scalars, peer_scalars)
    assert 'cs_rank_mf_net_elg_5d' in res
    assert 'cs_rank_mf_net_elg_20d' in res
    assert 'cs_rank_mf_smart_net_share_20d' in res
    assert 'cs_rank_mf_elg_persistence_20d' in res
    # All should be in [0, 1]
    for v in res.values():
        assert 0.0 <= v <= 1.0
    # Stock at value 0.10 with peers [0.02, 0.05, 0.08, 0.10, 0.15] → rank ≈ 0.6
    # (3 of 5 strictly less than 0.10)
    assert abs(res['cs_rank_mf_net_elg_5d'] - 3/5) < 1e-9


def test_compute_group_d_factors_empty_peers():
    """No peers → return 0.5 (neutral)."""
    from ml_models.ng.ng123_moneyflow_factors import compute_group_d_factors
    stock_scalars = {
        'net_elg_5d_ratio': 0.10, 'net_elg_20d_ratio': 0.05,
        'smart_net_share_20d': 0.30, 'persistence_20d': 0.40,
    }
    peer_scalars = {
        'net_elg_5d_ratio': np.array([]),
        'net_elg_20d_ratio': np.array([]),
        'smart_net_share_20d': np.array([]),
        'persistence_20d': np.array([]),
    }
    res = compute_group_d_factors(stock_scalars, peer_scalars)
    for v in res.values():
        assert v == 0.5
```

- [ ] **Step 2: 运行测试 (failing)**

```bash
python3 -m pytest ml_models/ng/tests/test_ng123_moneyflow_factors.py -v -k group_d
```

Expected: ImportError.

- [ ] **Step 3: 实现 Group D + helper**

在 `ml_models/ng/ng123_moneyflow_factors.py` 末尾追加：

```python
# ---------------------------------------------------------------------------
# Helper: compute the 4 scalar values needed for cs_rank wrapper
# ---------------------------------------------------------------------------

def compute_stock_mf_scalars(rows: List[Dict]) -> Dict[str, float]:
    """Compute the 4 raw scalars that feed Group D cs_rank factors.

    Returns NaN-filled dict for empty input. Used by ng_cache_updater to
    pre-compute peer arrays per industry per date.
    """
    result = {
        'net_elg_5d_ratio': np.nan,
        'net_elg_20d_ratio': np.nan,
        'smart_net_share_20d': np.nan,
        'persistence_20d': np.nan,
    }
    if not rows:
        return result

    a = compute_group_a_factors(rows)
    b = compute_group_b_factors(rows)
    result['net_elg_5d_ratio'] = a['mf_net_elg_5d_ratio']
    result['net_elg_20d_ratio'] = a['mf_net_elg_20d_ratio']
    result['smart_net_share_20d'] = a['mf_smart_net_share_20d']
    result['persistence_20d'] = b['mf_elg_persistence_20d']
    return result


# ---------------------------------------------------------------------------
# Group D: Cross-Sectional Industry Ranks (4 factors)
# ---------------------------------------------------------------------------

def _industry_percentile_rank_safe(value: float, peer_values: np.ndarray) -> float:
    """Mirror of ng_feature_calculator._industry_percentile_rank.

    Returns 0.5 if peer array empty or all NaN; otherwise fraction strictly < value.
    """
    if peer_values is None or len(peer_values) == 0:
        return 0.5
    valid = peer_values[~np.isnan(peer_values)]
    if len(valid) < 2:
        return 0.5
    if np.isnan(value):
        return 0.5
    return float(np.mean(valid < value))


def compute_group_d_factors(
    stock_scalars: Dict[str, float],
    peer_scalars: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Compute factors 9-12 from spec §4.1 Group D.

    Args:
        stock_scalars: dict from compute_stock_mf_scalars(self_rows).
        peer_scalars: dict {factor_name → 1D array of peer values incl. self}.
    """
    return {
        'cs_rank_mf_net_elg_5d': _industry_percentile_rank_safe(
            stock_scalars.get('net_elg_5d_ratio', np.nan),
            peer_scalars.get('net_elg_5d_ratio', np.array([]))),
        'cs_rank_mf_net_elg_20d': _industry_percentile_rank_safe(
            stock_scalars.get('net_elg_20d_ratio', np.nan),
            peer_scalars.get('net_elg_20d_ratio', np.array([]))),
        'cs_rank_mf_smart_net_share_20d': _industry_percentile_rank_safe(
            stock_scalars.get('smart_net_share_20d', np.nan),
            peer_scalars.get('smart_net_share_20d', np.array([]))),
        'cs_rank_mf_elg_persistence_20d': _industry_percentile_rank_safe(
            stock_scalars.get('persistence_20d', np.nan),
            peer_scalars.get('persistence_20d', np.array([]))),
    }


# ---------------------------------------------------------------------------
# Top-level orchestrator: compute all 12 factors for a single stock
# ---------------------------------------------------------------------------

def compute_all_moneyflow_factors(
    rows: List[Dict],
    stock_scalars: Dict[str, float] = None,
    peer_scalars: Dict[str, np.ndarray] = None,
) -> Dict[str, float]:
    """Compute all 12 ng1.2.3 moneyflow factors for one stock on one date.

    Args:
        rows: List of moneyflow dicts (last 20 days), oldest → newest.
        stock_scalars: Pre-computed via compute_stock_mf_scalars(rows). If None,
            computed inline. Pass pre-computed when called in batch context.
        peer_scalars: Industry peer arrays (pre-computed once per industry-date).
            Pass empty dict {} if peer info unavailable (cs_rank → 0.5).
    """
    result = {}
    result.update(compute_group_a_factors(rows))
    result.update(compute_group_b_factors(rows))
    result.update(compute_group_c_factors(rows))
    if stock_scalars is None:
        stock_scalars = compute_stock_mf_scalars(rows)
    if peer_scalars is None:
        peer_scalars = {}
    result.update(compute_group_d_factors(stock_scalars, peer_scalars))
    return result
```

- [ ] **Step 4: 运行测试 (passing)**

```bash
python3 -m pytest ml_models/ng/tests/test_ng123_moneyflow_factors.py -v
```

Expected: All ~18 tests passed.

- [ ] **Step 5: Commit**

```bash
git add ml_models/ng/ng123_moneyflow_factors.py ml_models/ng/tests/test_ng123_moneyflow_factors.py
git commit -m "feat(ng123): Group D 4 cs_rank moneyflow 因子 + 顶层 orchestrator (TDD)"
```

---

## Phase 1 - Stage 1: Moneyflow IC 验证

### Task 8: Stage 1 IC + 正交性脚本

**Files:**
- Create: `scripts/ng123/stage1_moneyflow_ic.py`

- [ ] **Step 1: 写脚本**

创建 `scripts/ng123/stage1_moneyflow_ic.py`：

```python
#!/usr/bin/env python3
"""ng1.2.3 Stage 1: Moneyflow factor IC validation + orthogonality vs ng101.

Per spec §6.1 — Pass criteria (ALL):
  - >= 6 factors with |IC| > 0.02 AND |ICIR| > 0.3
  - >= 4 factors with |IC| > 0.04 AND |ICIR| > 0.5
  - >= 8 factors with max |corr| < 0.5 vs all 64 ng101 features
  - >= 2 of 4 cs_rank factors pass

Output: reports/ng123/fastcheck/stage1_moneyflow_ic.csv

Estimated runtime: ~4 hours on full universe × 2022-01..2026-04.
"""
import argparse
import json
import os
import sys
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml_models.ng.ng123_moneyflow_factors import compute_all_moneyflow_factors, compute_stock_mf_scalars

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
OUTPUT_DIR = PROJECT_ROOT / 'reports' / 'ng123' / 'fastcheck'

# Pass thresholds
MIN_IC = 0.02
MIN_ICIR = 0.3
STRONG_IC = 0.04
STRONG_ICIR = 0.5
MAX_CORR_VS_NG101 = 0.5


def load_moneyflow_per_stock(start_date: str, end_date: str, n_stocks: int = None
                              ) -> Dict[str, List[Dict]]:
    """Load moneyflow rows per stock for the date range, returning chronological lists."""
    print(f"  Loading moneyflow {start_date} → {end_date}...", flush=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Get list of A-stock codes with moneyflow coverage
    codes_q = """
        SELECT DISTINCT s.code
        FROM securities s
        JOIN moneyflow_daily mf ON mf.code = s.code
        WHERE s.type = 'A股'
          AND mf.trade_date BETWEEN ? AND ?
    """
    codes = [r[0] for r in conn.execute(codes_q, (start_date, end_date)).fetchall()]
    if n_stocks and len(codes) > n_stocks:
        codes = list(np.random.RandomState(42).choice(codes, n_stocks, replace=False))
    print(f"  Universe: {len(codes)} stocks", flush=True)

    # Bulk load
    chunk = 800
    per_stock: Dict[str, List[Dict]] = defaultdict(list)
    for i in range(0, len(codes), chunk):
        batch = codes[i:i+chunk]
        rows = conn.execute(
            f"""SELECT code, trade_date, buy_sm_amount, sell_sm_amount,
                       buy_md_amount, sell_md_amount, buy_lg_amount, sell_lg_amount,
                       buy_elg_amount, sell_elg_amount, net_mf_amount
                FROM moneyflow_daily
                WHERE trade_date BETWEEN ? AND ?
                  AND code IN ({','.join('?' * len(batch))})
                ORDER BY code, trade_date""",
            [start_date, end_date] + batch
        ).fetchall()
        for r in rows:
            per_stock[r['code']].append(dict(r))

    conn.close()
    return dict(per_stock)


def load_label_10d(start_date: str, end_date: str) -> pd.DataFrame:
    """Load 10d industry-excess label from ng101_feature_cache for IC computation."""
    print(f"  Loading label_10d (industry excess) from ng101_feature_cache...", flush=True)
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        """SELECT code, trade_date, label_10d
           FROM ng101_feature_cache
           WHERE trade_date BETWEEN ? AND ?
             AND label_10d IS NOT NULL""",
        conn, params=[start_date, end_date])
    conn.close()
    # Convert ng101 code (no suffix? verify) — moneyflow uses 000001.SZ format
    print(f"  Labels: {len(df):,} rows", flush=True)
    return df


def load_ng101_features(start_date: str, end_date: str) -> pd.DataFrame:
    """Load ng101 features_json for orthogonality check (sample)."""
    print(f"  Loading ng101 features (sample)...", flush=True)
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """SELECT code, trade_date, features_json
           FROM ng101_feature_cache
           WHERE trade_date BETWEEN ? AND ?
             AND features_json IS NOT NULL
           LIMIT 100000""",
        (start_date, end_date)).fetchall()
    conn.close()

    parsed = []
    for code, td, fjson in rows:
        try:
            d = json.loads(fjson)
            d['code'] = code
            d['trade_date'] = td
            parsed.append(d)
        except (json.JSONDecodeError, TypeError):
            continue
    return pd.DataFrame(parsed)


def compute_factors_for_universe(
    mf_per_stock: Dict[str, List[Dict]],
    industry_map: Dict[str, str],  # code → industry
    universe: List[str],
) -> pd.DataFrame:
    """Compute all 12 mf factors per (code, date). Returns long DataFrame."""
    print(f"  Computing 12 mf factors per (code, date)...", flush=True)

    records = []
    for i, code in enumerate(universe):
        stock_rows = mf_per_stock.get(code, [])
        if len(stock_rows) < 5:
            continue
        # For each date in this stock's series, compute factors using rolling window
        for j in range(5, len(stock_rows)):
            window = stock_rows[max(0, j-19):j+1]  # up to 20 days
            scalars = compute_stock_mf_scalars(window)
            # cs_rank requires peers — defer to second pass; here compute Group A/B/C only
            from ml_models.ng.ng123_moneyflow_factors import (
                compute_group_a_factors, compute_group_b_factors, compute_group_c_factors)
            factors = {}
            factors.update(compute_group_a_factors(window))
            factors.update(compute_group_b_factors(window))
            factors.update(compute_group_c_factors(window))
            factors.update({f'_scalar_{k}': v for k, v in scalars.items()})
            factors['code'] = code
            factors['trade_date'] = stock_rows[j]['trade_date']
            records.append(factors)
        if i % 200 == 0:
            print(f"    progress: {i}/{len(universe)} stocks", flush=True)

    df = pd.DataFrame(records)
    print(f"  Raw factor rows: {len(df):,}", flush=True)
    return df


def add_cs_rank_factors(df_factors: pd.DataFrame, industry_map: Dict[str, str]) -> pd.DataFrame:
    """Compute cs_rank Group D factors per (industry, date)."""
    print(f"  Computing cs_rank Group D factors...", flush=True)
    df_factors['industry'] = df_factors['code'].map(industry_map).fillna('UNKNOWN')

    # For each (industry, date), rank the 4 scalars
    scalar_cols = ['_scalar_net_elg_5d_ratio', '_scalar_net_elg_20d_ratio',
                   '_scalar_smart_net_share_20d', '_scalar_persistence_20d']
    cs_rank_cols = ['cs_rank_mf_net_elg_5d', 'cs_rank_mf_net_elg_20d',
                    'cs_rank_mf_smart_net_share_20d', 'cs_rank_mf_elg_persistence_20d']

    for sc, cr in zip(scalar_cols, cs_rank_cols):
        df_factors[cr] = df_factors.groupby(['industry', 'trade_date'])[sc].rank(pct=True)

    # Drop scalar helpers
    df_factors = df_factors.drop(columns=scalar_cols + ['industry'])
    return df_factors


def compute_ic_per_factor(df_factors: pd.DataFrame, df_labels: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman IC per factor, daily, then aggregate to mean+std+ICIR."""
    print(f"  Computing IC per factor (Spearman, daily cross-section)...", flush=True)
    merged = df_factors.merge(df_labels, on=['code', 'trade_date'], how='inner')
    print(f"    Merged: {len(merged):,} rows", flush=True)

    factor_cols = [c for c in df_factors.columns
                   if c not in ('code', 'trade_date') and not c.startswith('_')]

    results = []
    for fc in factor_cols:
        ics = []
        for date, grp in merged.groupby('trade_date'):
            sub = grp[[fc, 'label_10d']].dropna()
            if len(sub) < 50:
                continue
            try:
                ic, _ = spearmanr(sub[fc].values, sub['label_10d'].values)
                if np.isfinite(ic):
                    ics.append(ic)
            except Exception:
                pass
        if len(ics) < 30:
            results.append({'factor': fc, 'n_days': len(ics), 'ic_mean': np.nan,
                            'ic_std': np.nan, 'icir': np.nan})
            continue
        ic_mean = float(np.mean(ics))
        ic_std = float(np.std(ics))
        icir = ic_mean / max(ic_std, 1e-8)
        results.append({
            'factor': fc, 'n_days': len(ics),
            'ic_mean': ic_mean, 'ic_std': ic_std, 'icir': icir,
            'ic_positive_pct': float(np.mean(np.array(ics) > 0)),
        })
    return pd.DataFrame(results)


def compute_max_corr_vs_ng101(df_factors: pd.DataFrame, df_ng101: pd.DataFrame) -> pd.DataFrame:
    """For each new factor, compute max |corr| vs all ng101 features."""
    print(f"  Computing max |corr| vs ng101...", flush=True)
    merged = df_factors.merge(df_ng101, on=['code', 'trade_date'], how='inner')
    print(f"    Merged for corr: {len(merged):,} rows", flush=True)
    if len(merged) < 1000:
        print("    WARN: too few rows for corr; skipping", flush=True)
        return pd.DataFrame()

    factor_cols = [c for c in df_factors.columns
                   if c not in ('code', 'trade_date') and not c.startswith('_')]
    ng101_cols = [c for c in df_ng101.columns
                  if c not in ('code', 'trade_date') and c not in factor_cols]

    rows = []
    for fc in factor_cols:
        max_abs_corr = 0.0
        worst_pair = ''
        for nc in ng101_cols:
            sub = merged[[fc, nc]].dropna()
            if len(sub) < 100:
                continue
            try:
                c = sub[fc].corr(sub[nc])
                if pd.notna(c) and abs(c) > max_abs_corr:
                    max_abs_corr = abs(c)
                    worst_pair = nc
            except Exception:
                continue
        rows.append({
            'factor': fc, 'max_abs_corr': max_abs_corr, 'worst_ng101_feature': worst_pair,
        })
    return pd.DataFrame(rows)


def evaluate_pass_criteria(ic_df: pd.DataFrame, corr_df: pd.DataFrame) -> Dict[str, bool]:
    """Apply spec §6.1 pass criteria. Returns per-criterion bool dict."""
    n_pass_basic = ((ic_df['ic_mean'].abs() > MIN_IC)
                    & (ic_df['icir'].abs() > MIN_ICIR)).sum()
    n_pass_strong = ((ic_df['ic_mean'].abs() > STRONG_IC)
                     & (ic_df['icir'].abs() > STRONG_ICIR)).sum()
    n_pass_corr = (corr_df['max_abs_corr'] < MAX_CORR_VS_NG101).sum()

    cs_rank_factors = ic_df[ic_df['factor'].str.startswith('cs_rank_mf')]
    n_cs_pass = ((cs_rank_factors['ic_mean'].abs() > MIN_IC)
                 & (cs_rank_factors['icir'].abs() > MIN_ICIR)).sum()

    return {
        'criterion_1_basic_count': int(n_pass_basic),
        'criterion_1_basic_pass': n_pass_basic >= 6,
        'criterion_2_strong_count': int(n_pass_strong),
        'criterion_2_strong_pass': n_pass_strong >= 4,
        'criterion_3_corr_count': int(n_pass_corr),
        'criterion_3_corr_pass': n_pass_corr >= 8,
        'criterion_4_csrank_count': int(n_cs_pass),
        'criterion_4_csrank_pass': n_cs_pass >= 2,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start-date', default='2022-01-01')
    p.add_argument('--end-date', default='2026-04-14')
    p.add_argument('--n-stocks', type=int, default=None,
                   help='Sample n stocks (default: full universe)')
    args = p.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    mf_per_stock = load_moneyflow_per_stock(args.start_date, args.end_date, args.n_stocks)
    df_labels = load_label_10d(args.start_date, args.end_date)
    df_ng101 = load_ng101_features(args.start_date, args.end_date)

    # 2. Industry map (for cs_rank)
    conn = sqlite3.connect(DB_PATH)
    industry_rows = conn.execute(
        "SELECT code, industry FROM stock_basic_info").fetchall()
    industry_map = {r[0]: r[1] for r in industry_rows if r[1]}
    conn.close()

    # 3. Compute factors
    universe = list(mf_per_stock.keys())
    df_raw = compute_factors_for_universe(mf_per_stock, industry_map, universe)
    df_factors = add_cs_rank_factors(df_raw, industry_map)

    # 4. IC + ICIR
    ic_df = compute_ic_per_factor(df_factors, df_labels)
    print("\n=== Factor IC Summary ===")
    print(ic_df.to_string())

    # 5. Orthogonality vs ng101
    corr_df = compute_max_corr_vs_ng101(df_factors, df_ng101)
    print("\n=== Factor Orthogonality (max |corr| vs ng101) ===")
    print(corr_df.to_string())

    # 6. Pass criteria
    criteria = evaluate_pass_criteria(ic_df, corr_df)
    print("\n=== Pass Criteria (spec §6.1) ===")
    for k, v in criteria.items():
        print(f"  {k}: {v}")

    overall_pass = all(v for k, v in criteria.items() if k.endswith('_pass'))
    print(f"\nSTAGE 1 OVERALL: {'PASS ✅' if overall_pass else 'FAIL ❌'}")

    # 7. Save
    out_csv = OUTPUT_DIR / 'stage1_moneyflow_ic.csv'
    merged = ic_df.merge(corr_df, on='factor', how='outer')
    merged.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # 8. Status JSON
    status = {
        'stage': 1, 'overall_pass': overall_pass,
        'criteria': criteria, 'n_factors': len(ic_df),
        'config': {'start_date': args.start_date, 'end_date': args.end_date,
                   'n_stocks': args.n_stocks},
    }
    with open(OUTPUT_DIR / 'stage1_status.json', 'w') as f:
        json.dump(status, f, indent=2)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke-test 脚本结构 (用 100 股 sample)**

```bash
python3 scripts/ng123/stage1_moneyflow_ic.py --n-stocks 100 --start-date 2024-01-01 --end-date 2024-12-31
```

Expected output: 终端打印 IC summary + Pass criteria + 写出 `reports/ng123/fastcheck/stage1_moneyflow_ic.csv` 和 `stage1_status.json`。运行时间约 5 分钟。

如果 smoke-test 通过 → 进入 Step 3。如果失败 → 调试脚本逻辑（注意 code 格式可能不一致：moneyflow 用 `000001.SZ`，ng101 cache 可能用 `000001`；如有问题在脚本里加 normalize）。

- [ ] **Step 3: 全量运行 (~4 小时)**

```bash
python3 scripts/ng123/stage1_moneyflow_ic.py 2>&1 | tee logs/ng123_stage1_$(date +%Y%m%d_%H%M%S).log
```

- [ ] **Step 4: Commit 脚本和结果**

```bash
git add scripts/ng123/stage1_moneyflow_ic.py
git add reports/ng123/fastcheck/stage1_moneyflow_ic.csv
git add reports/ng123/fastcheck/stage1_status.json
git commit -m "feat(ng123): Stage 1 moneyflow IC 验证脚本 + 全量结果"
```

- [ ] **Step 5: 决定是否进入 Stage 2**

读取 `reports/ng123/fastcheck/stage1_status.json`：
- 若 `overall_pass=true` → 进入 Task 9
- 若 `overall_pass=false` 且 `criterion_1_basic_count >= 6` → 进入 Task 9，但只用 IC pass 的因子（更新 ng123_moneyflow_factors.py 移除 fail 的）
- 若 `criterion_1_basic_count < 6` → **STOP**：moneyflow 假设不成立，不进入 Stage 2/3，跳到 Postmortem

---

## Phase 1 - Stage 2: Mined Factor 重验

### Task 9: 重跑 factor mining + 二次筛选

**Files:**
- Create: `scripts/ng123/stage2_mined_validate.py`
- Modify: `ml_models/ng/ng123_mined_factors.py`

- [ ] **Step 1: 重跑 factor mining (1 hour)**

```bash
python3 scripts/factor_mining_pipeline.py \
  --start-date 2022-01-01 \
  --end-date 2026-04-14 \
  --depth 2 \
  --n-stocks 800 \
  --min-icir 0.5 \
  --min-ic 0.04 \
  --max-corr 0.5 \
  2>&1 | tee logs/ng123_mining_$(date +%Y%m%d_%H%M%S).log
```

输出: `scripts/mined_factors_results.json` (覆盖旧版本)

- [ ] **Step 2: 写 Stage 2 验证脚本**

创建 `scripts/ng123/stage2_mined_validate.py`：

```python
#!/usr/bin/env python3
"""ng1.2.3 Stage 2: Mined factor re-validation.

Apply secondary filters per spec §6.2:
  1. Orthogonality vs ng101 + already-passed moneyflow factors
  2. Sign flip if IC < 0
  3. Cross-regime stability: IC same sign on 2022 (bear) + 2024 (recovery), |IC|>0.02 in both

Output: reports/ng123/fastcheck/stage2_mined_factors.csv
        reports/ng123/fastcheck/stage2_status.json
        ml_models/ng/ng123_mined_factors.py — populated MINED_FACTOR_SPEC
"""
import argparse
import json
import os
import sys
import sqlite3
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
OUTPUT_DIR = PROJECT_ROOT / 'reports' / 'ng123' / 'fastcheck'
MINING_RESULTS = PROJECT_ROOT / 'scripts' / 'mined_factors_results.json'

REGIME_PERIODS = {
    'bear_2022': ('2022-01-01', '2022-12-31'),
    'recovery_2024': ('2024-01-01', '2024-12-31'),
}

MIN_ABS_IC_REGIME = 0.02


def load_mining_results() -> List[Dict]:
    if not MINING_RESULTS.exists():
        raise FileNotFoundError(f"{MINING_RESULTS} — run factor_mining_pipeline.py first")
    with open(MINING_RESULTS) as f:
        return json.load(f)['factors']


def load_ng101_factors_sample(n_rows: int = 100000) -> pd.DataFrame:
    """Load ng101 features for orthogonality check."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        f"""SELECT code, trade_date, features_json
            FROM ng101_feature_cache
            WHERE features_json IS NOT NULL
            ORDER BY RANDOM() LIMIT {n_rows}""").fetchall()
    conn.close()
    parsed = []
    for code, td, fjson in rows:
        try:
            d = json.loads(fjson)
            d['code'] = code
            d['trade_date'] = td
            parsed.append(d)
        except Exception:
            continue
    return pd.DataFrame(parsed)


def recompute_factor_for_validation(factor_spec: Dict, period: tuple,
                                     n_stocks: int = 500) -> pd.DataFrame:
    """Recompute a single mined factor on a given period for IC validation.

    Returns DataFrame with columns [code, trade_date, factor_value, label_10d].
    """
    from scripts.factor_mining_pipeline import (
        generate_operands, compute_factor)

    start, end = period
    conn = sqlite3.connect(DB_PATH)

    # Sample stocks
    codes = [r[0] for r in conn.execute(
        "SELECT DISTINCT code FROM securities WHERE type='A股'").fetchall()]
    if len(codes) > n_stocks:
        codes = list(np.random.RandomState(42).choice(codes, n_stocks, replace=False))

    records = []
    for code in codes:
        df_stk = pd.read_sql(
            """SELECT q.trade_date, q.open, q.high, q.low, q.close,
                      q.volume, q.price_change_pct
               FROM daily_quotes q JOIN securities s ON q.security_id=s.id
               WHERE s.code = ? AND q.trade_date BETWEEN ? AND ?
               ORDER BY q.trade_date""",
            conn, params=[code, start, end])
        if len(df_stk) < 60:
            continue
        operands = generate_operands(df_stk)
        try:
            factor_vals = compute_factor(factor_spec, operands)
            if factor_vals is None:
                continue
        except Exception:
            continue
        # Label = 10d forward return
        df_stk['label_10d'] = df_stk['close'].shift(-10) / df_stk['close'] - 1
        for i, td in enumerate(df_stk['trade_date'].values):
            v = factor_vals.iloc[i] if hasattr(factor_vals, 'iloc') else factor_vals[i]
            l = df_stk['label_10d'].values[i]
            if not (np.isnan(v) or np.isnan(l)):
                records.append({'code': code, 'trade_date': td,
                                'factor_value': v, 'label_10d': l})

    conn.close()
    return pd.DataFrame(records)


def compute_ic_for_period(df: pd.DataFrame) -> float:
    """Mean Spearman IC across daily cross-sections."""
    if len(df) < 1000:
        return np.nan
    ics = []
    for date, grp in df.groupby('trade_date'):
        sub = grp.dropna()
        if len(sub) < 50:
            continue
        try:
            ic, _ = spearmanr(sub['factor_value'], sub['label_10d'])
            if np.isfinite(ic):
                ics.append(ic)
        except Exception:
            continue
    return float(np.mean(ics)) if len(ics) >= 30 else np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--top-n-candidates', type=int, default=30,
                   help='Top N factors from mining to validate (default: 30)')
    p.add_argument('--moneyflow-csv', default=str(OUTPUT_DIR / 'stage1_moneyflow_ic.csv'),
                   help='Path to Stage 1 results (for orthogonality vs mf factors)')
    args = p.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load mining candidates
    candidates = load_mining_results()
    candidates.sort(key=lambda x: abs(x['icir']), reverse=True)
    candidates = candidates[:args.top_n_candidates]
    print(f"Validating top {len(candidates)} mined factors...")

    # 2. For each candidate: cross-regime IC check + sign flip detection
    results = []
    for i, c in enumerate(candidates):
        print(f"\n[{i+1}/{len(candidates)}] {c['name']}: original IC={c['ic_mean']:+.4f}, ICIR={c['icir']:+.3f}")
        sign_flip = c['ic_mean'] < 0

        ic_per_regime = {}
        regime_pass = True
        for regime_name, period in REGIME_PERIODS.items():
            df = recompute_factor_for_validation(c, period, n_stocks=500)
            ic = compute_ic_for_period(df)
            if sign_flip:
                ic = -ic if not np.isnan(ic) else ic
            ic_per_regime[regime_name] = ic
            print(f"  {regime_name}: IC={ic:+.4f}")
            if np.isnan(ic) or abs(ic) < MIN_ABS_IC_REGIME:
                regime_pass = False

        # Cross-regime same sign?
        signs = [np.sign(v) for v in ic_per_regime.values() if not np.isnan(v)]
        same_sign = len(signs) >= 2 and all(s == signs[0] for s in signs)

        results.append({
            'name': c['name'],
            'sign_flip': sign_flip,
            'original_ic': c['ic_mean'],
            'original_icir': c['icir'],
            'ic_bear_2022': ic_per_regime.get('bear_2022', np.nan),
            'ic_recovery_2024': ic_per_regime.get('recovery_2024', np.nan),
            'same_sign_across_regimes': same_sign,
            'regime_stable': regime_pass and same_sign,
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('regime_stable', ascending=False)

    print("\n\n=== Stage 2 Results ===")
    print(df_results.to_string())

    n_stable = df_results['regime_stable'].sum()
    overall_pass = n_stable >= 6
    print(f"\nRegime-stable factors: {n_stable}")
    print(f"STAGE 2 OVERALL: {'PASS ✅' if overall_pass else 'FAIL ❌'}")

    # 3. Save
    df_results.to_csv(OUTPUT_DIR / 'stage2_mined_factors.csv', index=False)
    status = {
        'stage': 2, 'overall_pass': bool(overall_pass), 'n_stable': int(n_stable),
        'top_6': df_results[df_results['regime_stable']].head(6).to_dict('records'),
    }
    with open(OUTPUT_DIR / 'stage2_status.json', 'w') as f:
        json.dump(status, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_DIR / 'stage2_mined_factors.csv'}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: 运行 Stage 2 (~1 hour)**

```bash
python3 scripts/ng123/stage2_mined_validate.py 2>&1 | tee logs/ng123_stage2_$(date +%Y%m%d_%H%M%S).log
```

- [ ] **Step 4: 把 top 6 写入 `ng123_mined_factors.py`**

```bash
python3 -c "
import json
with open('reports/ng123/fastcheck/stage2_status.json') as f:
    s = json.load(f)
print('TOP 6 FACTORS:')
for f in s.get('top_6', []):
    print(f)
"
```

复制 top 6 名字到 `ml_models/ng/ng123_mined_factors.py` 的 `MINED_FACTOR_SPEC` (待 Task 10 创建)。

- [ ] **Step 5: Commit**

```bash
git add scripts/ng123/stage2_mined_validate.py
git add scripts/mined_factors_results.json
git add reports/ng123/fastcheck/stage2_mined_factors.csv reports/ng123/fastcheck/stage2_status.json
git commit -m "feat(ng123): Stage 2 mined factor 跨 regime 重验"
```

- [ ] **Step 6: 决定是否进入 Stage 3**

- 若 `overall_pass=true` → 进入 Task 10（含 mined factors）
- 若 `overall_pass=false` → 进入 Task 10（**仅 moneyflow，跳过 mined**）

---

### Task 10: 实现选定的 Mined Factor 计算函数

**Files:**
- Create: `ml_models/ng/ng123_mined_factors.py`
- Create: `ml_models/ng/tests/test_ng123_mined_factors.py`

- [ ] **Step 1: 写测试 (用 Stage 2 选出的真实因子)**

⚠️ Top 6 因子名是 Stage 2 输出决定的。假设示例（实际以 stage2_status.json 为准）：
- `neg_ts_decay_ret_60` (sign_flip=True, op=ts_decay, operand=ret, window=60)
- `neg_ts_cov_volume_ret_60` (sign_flip=True, ts_cov binary, operand_a=volume, operand_b=ret, window=60)
- ...

创建 `ml_models/ng/tests/test_ng123_mined_factors.py`：

```python
"""Unit tests for ng1.2.3 mined factor compute_value() functions."""
import numpy as np
import pandas as pd
import pytest

from ml_models.ng.ng123_mined_factors import (
    MINED_FACTOR_SPEC,
    compute_mined_factor_value,
)


def _mk_ohlcv(n_days: int = 100, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV: random walk."""
    rng = np.random.RandomState(seed)
    closes = 100 * np.cumprod(1 + rng.randn(n_days) * 0.02)
    return pd.DataFrame({
        'open': closes * (1 - rng.uniform(0, 0.01, n_days)),
        'high': closes * (1 + rng.uniform(0, 0.02, n_days)),
        'low': closes * (1 - rng.uniform(0, 0.02, n_days)),
        'close': closes,
        'volume': rng.randint(1e6, 1e7, n_days).astype(float),
        'price_change_pct': np.diff(closes, prepend=closes[0]) / closes,
        'turnover_rate': rng.uniform(0.5, 5.0, n_days),
    })


def test_mined_factor_spec_loaded():
    """MINED_FACTOR_SPEC must contain at least 1 factor entry."""
    assert isinstance(MINED_FACTOR_SPEC, list)
    assert len(MINED_FACTOR_SPEC) >= 1
    for f in MINED_FACTOR_SPEC:
        assert 'name' in f
        assert 'sign_flip' in f


def test_compute_returns_numeric_or_nan():
    """For each factor in spec, compute_mined_factor_value returns a finite or nan series."""
    df = _mk_ohlcv(120)
    for spec in MINED_FACTOR_SPEC:
        vals = compute_mined_factor_value(spec, df)
        assert vals is not None, f"Factor {spec['name']} returned None"
        assert len(vals) == len(df), f"Factor {spec['name']} wrong length"


def test_sign_flip_applied():
    """If spec.sign_flip=True, output should be negated vs raw computation."""
    df = _mk_ohlcv(120)
    for spec in MINED_FACTOR_SPEC:
        if spec.get('sign_flip'):
            # compute with sign_flip=True (default)
            vals_flip = compute_mined_factor_value(spec, df)
            # compute with sign_flip=False
            spec_no_flip = {**spec, 'sign_flip': False}
            vals_raw = compute_mined_factor_value(spec_no_flip, df)
            # Where both finite, vals_flip = -vals_raw
            mask = np.isfinite(vals_flip) & np.isfinite(vals_raw)
            if mask.sum() > 0:
                assert np.allclose(vals_flip[mask], -vals_raw[mask], atol=1e-9), \
                    f"Sign flip not applied for {spec['name']}"
```

- [ ] **Step 2: 写实现**

创建 `ml_models/ng/ng123_mined_factors.py`：

```python
"""ng1.2.3 mined alpha factors.

Selected via factor mining pipeline + Stage 2 cross-regime validation.
Per spec §4.2 (docs/superpowers/specs/2026-04-14-ng123-design.md).

⚠️ MINED_FACTOR_SPEC populated by Task 10 from reports/ng123/fastcheck/stage2_status.json.
"""
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.factor_mining_pipeline import generate_operands, compute_factor


# ============================================================================
# MINED_FACTOR_SPEC — populated from Stage 2 results.
# ============================================================================

# Format per entry (matches scripts/factor_mining_pipeline.py spec format):
#   {
#       'name': 'neg_ts_decay_ret_60',     # canonical with sign-flip prefix
#       'sign_flip': True,                  # multiply output by -1
#       'type': 'unary_ts',                 # 'unary_ts' | 'binary_ts' | 'depth2' | 'depth2_ts'
#       'op': 'ts_decay',                   # for unary_ts
#       'operand': 'ret',
#       'window': 60,
#       'ic': 0.096, 'icir': 1.020,         # post-validation
#       'cross_regime_ic': {'2022': ..., '2024': ...},
#       'semantic': '60-day decay-weighted return reversal',
#   }

MINED_FACTOR_SPEC: List[Dict] = [
    # POPULATED BY Task 10 — copy top 6 entries from stage2_status.json.
    # If Stage 2 failed (< 6 stable), this list stays empty AND
    # ng_cache_updater must skip mined factor computation (Task 12).
]


def compute_mined_factor_value(spec: Dict, df_stock: pd.DataFrame) -> np.ndarray:
    """Compute a single mined factor for one stock's full OHLCV time series.

    Returns numpy array same length as df_stock. NaN where not computable.
    Applies sign flip if spec.sign_flip is True.
    """
    operands = generate_operands(df_stock)
    val_series = compute_factor(spec, operands)
    if val_series is None:
        return np.full(len(df_stock), np.nan)
    vals = np.asarray(val_series.values, dtype=np.float64)
    if spec.get('sign_flip', False):
        vals = -vals
    return vals


def compute_all_mined_factors_for_stock(df_stock: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Compute all mined factors for one stock's time series. Returns {name: array}."""
    if not MINED_FACTOR_SPEC:
        return {}
    return {spec['name']: compute_mined_factor_value(spec, df_stock)
            for spec in MINED_FACTOR_SPEC}


def get_mined_factor_names() -> List[str]:
    """Return the list of mined factor names (used by trainer + cache schema)."""
    return [s['name'] for s in MINED_FACTOR_SPEC]
```

- [ ] **Step 3: 填充 MINED_FACTOR_SPEC (基于 Stage 2 输出)**

```bash
python3 -c "
import json
with open('reports/ng123/fastcheck/stage2_status.json') as f:
    s = json.load(f)
print('# Top 6 to populate MINED_FACTOR_SPEC:')
for entry in s.get('top_6', []):
    print(entry)
"
```

按上面输出，**手动编辑** `ml_models/ng/ng123_mined_factors.py` 的 `MINED_FACTOR_SPEC = [...]` 块，填入 6 个 entries。每个 entry 至少包含: `name`, `sign_flip`, `type`, `op` (or `operand_a/operand_b` for binary), `operand`, `window`。

- [ ] **Step 4: 运行测试**

```bash
python3 -m pytest ml_models/ng/tests/test_ng123_mined_factors.py -v
```

Expected: 3 passed (假设至少 1 个 factor 进 spec)。

- [ ] **Step 5: Commit**

```bash
git add ml_models/ng/ng123_mined_factors.py ml_models/ng/tests/test_ng123_mined_factors.py
git commit -m "feat(ng123): Mined factor 计算模块 + 选定 6 个 (TDD)"
```

---

## Phase 1 - Stage 3: Label λ Ablation

### Task 11: 集成 ng123 因子 + label 到 ng_cache_updater

**Files:**
- Modify: `ml_models/ng/ng_cache_updater.py`
- Modify: `ml_models/ng/ng_feature_calculator.py` (新增 ng123 入口)

⚠️ **Task 1 simplify Round 1 必修项**: `ng_cache_updater.py:1574` 现有 `is_12 and version_ge(..., 'ng1.2.1')` 行缺少与 ng_schema.py 同步的上界 guard。本 task 必须将其改为：
```python
is_12 and version_ge(self.schema_version, 'ng1.2.1') and not version_ge(self.schema_version, 'ng1.2.3')
```
否则 ng1.2.3 流程会尝试写入 ng1.2.1 列（vn_label_/path_/downside_std_10d）但表里没这些列，触发 SQL 错误。

- [ ] **Step 1: 在 ng_feature_calculator.py 添加 ng123 入口**

打开 `ml_models/ng/ng_feature_calculator.py`，在文件**末尾**追加：

```python
# ---------------------------------------------------------------------------
# ng1.2.3 entry: orchestrate moneyflow + mined + ng1.0.1 base features
# ---------------------------------------------------------------------------

def get_ng123_drop_features() -> set:
    """Return the 12 stock features dropped from ng1.0.1 in ng1.2.3.

    Per spec §4.3. Final list matches ng111 audit output.
    """
    return {
        'lower_shadow_ratio',
        'volume_cv',
        'volume_contraction',
        'volume_price_corr',
        'industry_hhi',
        'industry_volume_change',
        'n_sectors_strong',
        'peg_proxy',
        'pb_roe_ratio',
        'dv_ratio',
        'up_volume_ratio',
        'ocf_quality',
    }


def filter_ng123_features(features_dict: Dict[str, float]) -> Dict[str, float]:
    """Remove the 12 dropped features from a ng1.0.1 feature dict."""
    drop = get_ng123_drop_features()
    return {k: v for k, v in features_dict.items() if k not in drop}
```

- [ ] **Step 2: 修改 ng_cache_updater.py 在 process_date 流程里调用 ng123 因子**

打开 `ml_models/ng/ng_cache_updater.py`。找到 `process_date` 方法（应在 line 600+ 附近）。在已经计算 ng1.0.1 features 的循环里，**对 version='ng1.2.3' 添加新逻辑**：

由于 `ng_cache_updater.py` 较大（1762 lines），最简洁的做法是创建一个新的 `_compute_ng123_features_for_stock` 方法。在 `class NGCacheUpdater` 内（找到合适位置插入，例如 `_compute_stock_returns` 之后），添加：

```python
    def _compute_ng123_extra_features(
        self, code: str, base_features: Dict[str, float],
        moneyflow_rows: List[Dict],
        peer_mf_scalars: Dict[str, np.ndarray],
        df_stock: 'pd.DataFrame',
    ) -> Dict[str, float]:
        """For ng1.2.3: filter base ng101 features (drop 12) + add 12 mf + 6 mined.

        Args:
            code: stock code
            base_features: ng1.0.1 features dict (64 keys)
            moneyflow_rows: list of last-20-days mf dicts for this stock
            peer_mf_scalars: dict of {scalar_name → industry peer values array}
            df_stock: full OHLCV DataFrame for this stock (for mined factors)
        """
        from ml_models.ng.ng_feature_calculator import filter_ng123_features
        from ml_models.ng.ng123_moneyflow_factors import (
            compute_all_moneyflow_factors, compute_stock_mf_scalars)
        from ml_models.ng.ng123_mined_factors import compute_all_mined_factors_for_stock

        # 1. Filter base (drop 12)
        result = filter_ng123_features(base_features)

        # 2. Add 12 moneyflow
        stock_scalars = compute_stock_mf_scalars(moneyflow_rows)
        mf_factors = compute_all_moneyflow_factors(
            moneyflow_rows,
            stock_scalars=stock_scalars,
            peer_scalars=peer_mf_scalars,
        )
        result.update(mf_factors)

        # 3. Add 6 mined (last value per factor's full time series)
        mined_arrays = compute_all_mined_factors_for_stock(df_stock)
        for name, arr in mined_arrays.items():
            result[name] = float(arr[-1]) if len(arr) > 0 and not np.isnan(arr[-1]) else np.nan

        return result
```

- [ ] **Step 3: 在 process_date 主流程里 dispatch ng123**

在 `process_date` 方法里，找到现有的"per-stock feature computation loop"。在调用现有 `compute_*_features` 之后、`INSERT INTO` 之前，添加版本分支：

```python
            # ng1.2.3 — apply drops + add moneyflow + mined factors
            if self.version == 'ng1.2.3':
                # peer_mf_scalars must be pre-computed once per (industry, date)
                # See peer_mf_scalars_per_industry computation block above
                code_industry = universe[sec_id].get('industry', 'UNKNOWN')
                peer_mf = peer_mf_scalars_per_industry.get(code_industry, {})
                stock_features = self._compute_ng123_extra_features(
                    code=universe[sec_id]['code'],
                    base_features=stock_features,
                    moneyflow_rows=mf_data.get(universe[sec_id]['code'], []),
                    peer_mf_scalars=peer_mf,
                    df_stock=stock_df_for_mining,  # OHLCV DataFrame
                )
```

⚠️ 这一段必须根据 `process_date` 现有结构精确插入。打开文件 search `INSERT INTO {table_name}` 找正确插入点；如果结构与本 plan 假设不符，调整变量名（`mf_data`, `stock_df_for_mining`, `peer_mf_scalars_per_industry`）。

**预备工作**：在 `process_date` 顶部追加 peer_mf_scalars 预计算（仅当 version='ng1.2.3'）：

```python
        # ng1.2.3: pre-compute peer moneyflow scalars per industry for this date
        peer_mf_scalars_per_industry: Dict[str, Dict[str, np.ndarray]] = {}
        if self.version == 'ng1.2.3':
            from ml_models.ng.ng123_moneyflow_factors import compute_stock_mf_scalars
            from collections import defaultdict
            industry_scalars = defaultdict(lambda: defaultdict(list))
            for sec_id, info in universe.items():
                code = info['code']
                ind = info.get('industry', 'UNKNOWN')
                rows = mf_data.get(code, [])
                if not rows:
                    continue
                scalars = compute_stock_mf_scalars(rows)
                for k, v in scalars.items():
                    if not np.isnan(v):
                        industry_scalars[ind][k].append(v)
            for ind, scalar_dict in industry_scalars.items():
                peer_mf_scalars_per_industry[ind] = {
                    k: np.array(vlist) for k, vlist in scalar_dict.items()
                }
```

- [ ] **Step 4: 添加 downside_kd 计算**

在 `process_date` 方法的"label computation"块附近（搜索 `label_3d` / `compute_label_*`），添加 ng1.2.3 downside 计算：

```python
            # ng1.2.3: compute downside_kd from future closes
            if self.version == 'ng1.2.3':
                from ml_models.ng.ng123_label_transform import (
                    compute_path_min_kd, compute_downside_kd)
                today_close = float(price_data[sec_id][-1].get('close', np.nan))
                downside_values = {}
                for k in [3, 5, 10, 15]:
                    future_dates_k = self._get_future_dates(conn, date, k)
                    future_closes_k = []
                    for fd in future_dates_k:
                        if sec_id in future_prices and fd in future_prices[sec_id]:
                            fc = future_prices[sec_id][fd].get('close')
                            if fc is not None and not np.isnan(fc):
                                future_closes_k.append(fc)
                    pm = compute_path_min_kd(today_close, np.array(future_closes_k))
                    downside_values[f'downside_{k}d'] = compute_downside_kd(pm)
                # Pass to INSERT statement (need to extend column list)
```

- [ ] **Step 5: 修改 INSERT SQL 加 downside_kd 列**

找到 `INSERT INTO ng123_feature_cache` 或通用的 INSERT 模板。加上 4 个新列 `downside_3d, downside_5d, downside_10d, downside_15d`。具体位置取决于现有代码结构 — 用 grep 找：

```bash
grep -n "INSERT.*features_json" ml_models/ng/ng_cache_updater.py
```

修改对应 SQL 加列名 + 在 `executemany` 的 row tuple 里加 4 个值。

- [ ] **Step 6: Smoke-test 单日**

```bash
python3 ml_models/ng/ng_cache_updater.py \
  --start-date 2024-06-03 --end-date 2024-06-03 --version ng1.2.3
```

Expected: 完成 1 个交易日的 ng123_feature_cache 写入。

验证：
```bash
python3 -c "
import sqlite3, json, sys
sys.path.insert(0, '.')
from ml_models.ng.ng123_mined_factors import get_mined_factor_names
from ml_models.ng.ng_feature_calculator import get_ng123_drop_features

c = sqlite3.connect('data_adapter/stock_data.db')
row = c.execute(\"SELECT features_json, downside_10d FROM ng123_feature_cache WHERE trade_date='2024-06-03' LIMIT 1\").fetchone()
if row:
    feats = json.loads(row[0])
    print(f'Feature count: {len(feats)}')
    mf_keys = [k for k in feats if k.startswith('mf_') or k.startswith('cs_rank_mf_')]
    print(f'Moneyflow keys ({len(mf_keys)}): {sorted(mf_keys)}')
    mined_names = get_mined_factor_names()
    mined_present = [n for n in mined_names if n in feats]
    print(f'Mined keys present ({len(mined_present)}/{len(mined_names)}): {mined_present}')
    drops = get_ng123_drop_features()
    drops_leaked = [d for d in drops if d in feats]
    print(f'Dropped feats leaked: {drops_leaked} (should be empty)')
    print(f'downside_10d: {row[1]}')
"
```

预期: feature count ≈ 70 (52 base + 12 mf + 6 mined)，moneyflow keys = 12，mined keys = 6，drops_leaked = []，downside_10d 是浮点数。

- [ ] **Step 7: Commit**

```bash
git add ml_models/ng/ng_cache_updater.py ml_models/ng/ng_feature_calculator.py
git commit -m "feat(ng123): cache_updater 集成 moneyflow + mined factors + downside label 列"
```

---

### Task 12: Trainer 集成 ng1.2.3

**Files:**
- Modify: `ml_models/ng/ng_trainer.py`

- [ ] **Step 1: 注册 ng1.2.3 版本分支**

打开 `ml_models/ng/ng_trainer.py`，找到处理 version 字符串的入口（例如 `__init__` 或 `_load_features`）。添加 ng1.2.3 分支。

具体位置：搜索现有的 `version_ge('ng1.0.4')` 或 `_is_1_2_branch`：

```bash
grep -n "_is_1_2_branch\|version_ge.*'ng1\." ml_models/ng/ng_trainer.py
```

在 SELECT 列扩展逻辑里，对 ng1.2.3 添加 downside 列：

```python
        if version_ge(self.version, 'ng1.2.3') and _is_1_2_branch(self.version):
            extra_select += ", downside_3d, downside_5d, downside_10d, downside_15d"
```

- [ ] **Step 2: 在 label processing 步骤添加 downside penalty**

在 trainer 中 label 数据加载之后、训练之前的 transform block，添加：

```python
        # ng1.2.3: apply soft downside penalty to multi-target labels
        if version_ge(self.version, 'ng1.2.3') and _is_1_2_branch(self.version):
            from ml_models.ng.ng123_label_transform import apply_downside_penalty, DEFAULT_LAMBDA
            lam = float(getattr(self, 'lambda_downside', DEFAULT_LAMBDA))
            for h in [3, 5, 10, 15]:
                excess_col = f'label_{h}d'
                downside_col = f'downside_{h}d'
                if downside_col in df_labels.columns:
                    df_labels[excess_col] = apply_downside_penalty(
                        excess=df_labels[excess_col].values,
                        downside=df_labels[downside_col].values,
                        lam=lam,
                    )
            logger.info(f"  ng1.2.3: applied downside penalty (lambda={lam}) to {[3,5,10,15]} labels")
```

- [ ] **Step 3: 添加 `--lambda-downside` CLI 参数**

在 trainer 的 `argparse` block 加：

```python
    parser.add_argument('--lambda-downside', type=float, default=0.3,
                        help='ng1.2.3: downside penalty multiplier for label transform (default 0.3)')
```

并在初始化时存为 `self.lambda_downside`。

- [ ] **Step 4: Smoke-test trainer 配置**

```bash
python3 ml_models/ng/ng_trainer.py --version ng1.2.3 --start-date 2024-01-01 --end-date 2024-06-30 \
  --fast-check --no-auto-wf --lambda-downside 0.3 --purge-days 15 \
  2>&1 | head -50
```

Expected: 训练启动，加载 ng123_feature_cache，应用 downside penalty (log 行 "applied downside penalty (lambda=0.3)")，跑 2 个 fast-check WF 窗口。

注意：smoke-test 需要先有 ng123_feature_cache 数据 — Task 11 已写入 2024-06-03 一天，足够 fast-check 路径触发但不足以训练。所以这步主要看 trainer 的初始化和 schema 加载是否报错。

如果初始化报错，调试到能正常启动；如果只是数据不足训练失败，可接受。

- [ ] **Step 5: Commit**

```bash
git add ml_models/ng/ng_trainer.py
git commit -m "feat(ng123): trainer 注册 ng1.2.3 + downside label penalty + --lambda-downside CLI"
```

---

### Task 13: Mini backfill (200 天 for ablation)

- [ ] **Step 1: 跑 mini backfill 用于 Stage 3 ablation**

```bash
python3 ml_models/ng/ng_cache_updater.py \
  --start-date 2022-01-01 --end-date 2022-12-31 --version ng1.2.3 \
  2>&1 | tee logs/ng123_mini_backfill_$(date +%Y%m%d_%H%M%S).log
```

Expected: ~1 hour. 写入 ng123_feature_cache 2022 全年 (~250 trading days × 5621 stocks = ~1.4M rows)。

- [ ] **Step 2: 验证 cache 完整性**

```bash
python3 -c "
import sqlite3
c = sqlite3.connect('data_adapter/stock_data.db')
total = c.execute('SELECT COUNT(*) FROM ng123_feature_cache').fetchone()[0]
dates = c.execute('SELECT MIN(trade_date), MAX(trade_date), COUNT(DISTINCT trade_date) FROM ng123_feature_cache').fetchone()
print(f'Total rows: {total:,}')
print(f'Date range: {dates[0]} → {dates[1]} ({dates[2]} dates)')
"
```

Expected: ~1.4M rows, 2022-01-04 → 2022-12-30, ~243 dates.

- [ ] **Step 3: Commit (cache content not in git, but log)**

```bash
# logs/ngs123_mini_backfill_*.log 已被 .gitignore 忽略
# 不需要 commit 数据库内容
# 但可以记录在 reports/ng123/fastcheck/ng123_cache_backfill.md
echo "Mini backfill complete: 2022-01-01 → 2022-12-31 (~1.4M rows)" \
  > reports/ng123/fastcheck/cache_backfill_status.md
git add reports/ng123/fastcheck/cache_backfill_status.md
git commit -m "chore(ng123): mini-backfill 2022 完成 (Stage 3 准备)"
```

---

### Task 14: Stage 3 λ Ablation

**Files:**
- Create: `scripts/ng123/stage3_lambda_ablation.py`

- [ ] **Step 1: 写 ablation 脚本**

创建 `scripts/ng123/stage3_lambda_ablation.py`：

```python
#!/usr/bin/env python3
"""ng1.2.3 Stage 3: Label lambda ablation.

Per spec §6.3 — 5 lambda × 2 WF windows = 10 mini-trainings, ~30 min total.

Output: reports/ng123/fastcheck/stage3_lambda_ablation.csv
        reports/ng123/fastcheck/stage3_status.json
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / 'reports' / 'ng123' / 'fastcheck'

LAMBDAS = [0.0, 0.15, 0.30, 0.45, 0.60]
NG101_BASELINE_10D_ICIR = 0.93  # from MEMORY.md ng_production_switch
PASS_THRESHOLD = 0.95  # require >= 95% of baseline


def run_one_lambda(lam: float, log_dir: Path) -> dict:
    """Run a fast-check training for one lambda. Returns metrics dict."""
    log_file = log_dir / f'lambda_{lam:.2f}.log'
    cmd = [
        'python3', 'ml_models/ng/ng_trainer.py',
        '--version', 'ng1.2.3',
        '--start-date', '2022-01-01', '--end-date', '2022-12-31',
        '--fast-check', '--no-auto-wf',
        '--lambda-downside', str(lam),
        '--purge-days', '15',
    ]
    print(f"\n  λ={lam:.2f}: running fast-check...", flush=True)
    with open(log_file, 'w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                cwd=str(PROJECT_ROOT))

    # Parse log for ICIR
    icir_10d = None
    with open(log_file) as f:
        for line in f:
            # Trainer prints ICIR per horizon; pattern depends on actual trainer output
            # Adjust regex below based on real log format
            if '10d ICIR' in line or '10d_icir' in line:
                try:
                    parts = line.split('=')
                    icir_10d = float(parts[-1].strip().split()[0])
                except Exception:
                    pass

    return {
        'lambda': lam,
        'icir_10d': icir_10d if icir_10d is not None else float('nan'),
        'log_file': str(log_file.relative_to(PROJECT_ROOT)),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = OUTPUT_DIR / 'stage3_logs'
    log_dir.mkdir(exist_ok=True)

    print("=== Stage 3: λ Ablation ===")
    print(f"Lambdas: {LAMBDAS}")
    print(f"Baseline 10d ICIR: {NG101_BASELINE_10D_ICIR}")
    print(f"Pass threshold: {PASS_THRESHOLD * 100}% × baseline = {NG101_BASELINE_10D_ICIR * PASS_THRESHOLD:.3f}")

    results = []
    for lam in LAMBDAS:
        r = run_one_lambda(lam, log_dir)
        results.append(r)
        print(f"  λ={lam:.2f} → 10d ICIR = {r['icir_10d']:.3f}")

    df = pd.DataFrame(results)
    df['ratio_to_baseline'] = df['icir_10d'] / NG101_BASELINE_10D_ICIR
    df['passes'] = df['ratio_to_baseline'] >= PASS_THRESHOLD
    df = df.sort_values('lambda')

    print("\n=== Results ===")
    print(df.to_string())

    # Selection rule: largest lambda > 0 with passes=True
    passing = df[df['passes'] & (df['lambda'] > 0)]
    if len(passing) > 0:
        best_lam = passing['lambda'].max()
        overall_pass = True
    elif df.iloc[0]['passes']:  # only λ=0 passes → label change has no value
        best_lam = 0.0
        overall_pass = True  # technically pass but with λ=0 (no label change)
    else:
        best_lam = None
        overall_pass = False

    print(f"\nSelected λ = {best_lam}")
    print(f"STAGE 3 OVERALL: {'PASS ✅' if overall_pass else 'FAIL ❌'}")

    df.to_csv(OUTPUT_DIR / 'stage3_lambda_ablation.csv', index=False)
    status = {
        'stage': 3, 'overall_pass': bool(overall_pass), 'selected_lambda': best_lam,
        'baseline_icir': NG101_BASELINE_10D_ICIR,
        'results': df.to_dict('records'),
    }
    with open(OUTPUT_DIR / 'stage3_status.json', 'w') as f:
        json.dump(status, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_DIR / 'stage3_lambda_ablation.csv'}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 运行 Stage 3 (~30 min)**

```bash
python3 scripts/ng123/stage3_lambda_ablation.py 2>&1 | tee logs/ng123_stage3_$(date +%Y%m%d_%H%M%S).log
```

如果 trainer 的 log format 与脚本预期不一致（步骤 1 中的 ICIR 解析正则可能需要调），手动调整 `run_one_lambda()` 里的 parse 逻辑。

- [ ] **Step 3: Commit**

```bash
git add scripts/ng123/stage3_lambda_ablation.py
git add reports/ng123/fastcheck/stage3_lambda_ablation.csv reports/ng123/fastcheck/stage3_status.json
git commit -m "feat(ng123): Stage 3 λ ablation 脚本 + 选定 lambda"
```

- [ ] **Step 4: 写综合 decision.md**

```bash
python3 -c "
import json, datetime
from pathlib import Path
out = Path('reports/ng123/fastcheck')
s1 = json.loads((out / 'stage1_status.json').read_text())
s2 = json.loads((out / 'stage2_status.json').read_text())
s3 = json.loads((out / 'stage3_status.json').read_text())

go_a = s1['overall_pass'] and s2['overall_pass']
go_b = s1['overall_pass'] and not s2['overall_pass']
go_c = not s1['overall_pass'] or not s3['overall_pass']

with open(out / 'decision.md', 'w') as f:
    f.write(f'''# ng1.2.3 Fast-Check Decision

**Date**: {datetime.datetime.now().strftime(\"%Y-%m-%d %H:%M\")}

## Stage 1 (Moneyflow IC): {\"✅ PASS\" if s1[\"overall_pass\"] else \"❌ FAIL\"}
- Basic count: {s1[\"criteria\"][\"criterion_1_basic_count\"]} (need >= 6)
- Strong count: {s1[\"criteria\"][\"criterion_2_strong_count\"]} (need >= 4)
- Orthogonal count: {s1[\"criteria\"][\"criterion_3_corr_count\"]} (need >= 8)
- cs_rank pass: {s1[\"criteria\"][\"criterion_4_csrank_count\"]} (need >= 2)

## Stage 2 (Mined Factors): {\"✅ PASS\" if s2[\"overall_pass\"] else \"❌ FAIL\"}
- Stable factors: {s2[\"n_stable\"]} (need >= 6)

## Stage 3 (λ Ablation): {\"✅ PASS\" if s3[\"overall_pass\"] else \"❌ FAIL\"}
- Selected λ: {s3[\"selected_lambda\"]}

## Decision
''')
    if go_a:
        f.write('**GO Condition A**: 12 mf + 6 mined + λ={} → full training (70 stock features)\\n'.format(s3['selected_lambda']))
    elif go_b:
        f.write('**GO Condition B**: 12 mf only + λ={} → full training (64 stock features after drop+add)\\n'.format(s3['selected_lambda']))
    else:
        f.write('**Condition C**: project terminated; write postmortem.\\n')

print('decision.md written')
"

git add reports/ng123/fastcheck/decision.md
git commit -m "docs(ng123): fast-check 综合 decision.md"
```

- [ ] **Step 5: Stage 1+2+3 综合 gate**

读 `reports/ng123/fastcheck/decision.md` 决定下一步：
- Condition A → Task 15
- Condition B → Task 15 (但移除 mined factors 配置)
- Condition C → 跳到 Task 22 (Postmortem)

---

## Phase 2: 完整 Backfill

### Task 15: 全量 ng123_feature_cache backfill (2018-2026)

- [ ] **Step 1: 删除 mini backfill 数据 (避免重叠 INSERT 冲突)**

```bash
python3 -c "
import sqlite3
c = sqlite3.connect('data_adapter/stock_data.db')
n = c.execute('DELETE FROM ng123_feature_cache').rowcount
c.commit()
print(f'Deleted {n} rows')
"
```

- [ ] **Step 2: 全量 backfill 2018-01-01 → 2026-04-14 (~6-8 hours)**

```bash
nohup python3 ml_models/ng/ng_cache_updater.py \
  --start-date 2018-01-01 --end-date 2026-04-14 --version ng1.2.3 \
  > logs/ng123_full_backfill_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "Backfill PID: $!"
```

定期检查进度：
```bash
tail -50 logs/ng123_full_backfill_*.log
```

- [ ] **Step 3: 验证完成**

```bash
python3 -c "
import sqlite3
c = sqlite3.connect('data_adapter/stock_data.db')
total = c.execute('SELECT COUNT(*) FROM ng123_feature_cache').fetchone()[0]
dates = c.execute('SELECT MIN(trade_date), MAX(trade_date), COUNT(DISTINCT trade_date) FROM ng123_feature_cache').fetchone()
nulls = c.execute('SELECT COUNT(*) FROM ng123_feature_cache WHERE features_json IS NULL').fetchone()[0]
ds = c.execute('SELECT COUNT(*) FROM ng123_feature_cache WHERE downside_10d IS NULL').fetchone()[0]
print(f'Total rows: {total:,}')
print(f'Date range: {dates[0]} → {dates[1]} ({dates[2]} dates)')
print(f'features_json NULL: {nulls:,}')
print(f'downside_10d NULL: {ds:,}')
"
```

Expected:
- Total rows: ~10-12M
- Date range: 2018-01-02 → 2026-04-14
- features_json NULL = 0
- downside_10d NULL: 仅最后 15 个交易日（forward-looking 数据不足）

- [ ] **Step 4: Commit 状态记录**

```bash
echo "Full backfill complete: $(date)" > reports/ng123/full_backfill_complete.md
git add reports/ng123/full_backfill_complete.md
git commit -m "chore(ng123): 全量 backfill 2018-2026 完成"
```

---

## Phase 3: 完整训练

### Task 16: 训练 3 seeds + 自动双向评估

- [ ] **Step 0: 读取选定 λ**

```bash
SELECTED_LAMBDA=$(python3 -c "import json; print(json.load(open('reports/ng123/fastcheck/stage3_status.json'))['selected_lambda'])")
echo "Using λ = $SELECTED_LAMBDA"
```

- [ ] **Step 1: 训练 seed 42 (~2 hours)**

```bash
nohup python3 ml_models/training/train_v395_multi_target.py \
  --version ng1.2.3 --seed 42 \
  --lambda-downside $SELECTED_LAMBDA \
  --purge-days 15 \
  > logs/ng123_train_seed42_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "Seed 42 PID: $!"
```

- [ ] **Step 2: 训练 seed 123 (~2 hours, 可与 seed 42 并行)**

```bash
nohup python3 ml_models/training/train_v395_multi_target.py \
  --version ng1.2.3 --seed 123 \
  --lambda-downside $SELECTED_LAMBDA \
  --purge-days 15 \
  > logs/ng123_train_seed123_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "Seed 123 PID: $!"
```

- [ ] **Step 3: 训练 seed 456 (~2 hours)**

```bash
nohup python3 ml_models/training/train_v395_multi_target.py \
  --version ng1.2.3 --seed 456 \
  --lambda-downside $SELECTED_LAMBDA \
  --purge-days 15 \
  > logs/ng123_train_seed456_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "Seed 456 PID: $!"
```

- [ ] **Step 4: 等待全部完成 (~2-6 hours 取决于并行度)**

```bash
# 监控进度
for log in logs/ng123_train_seed*_*.log; do
    echo "=== $log ==="
    tail -3 "$log"
done
```

- [ ] **Step 5: 验证模型文件 + WF/Pre-2020 报告**

```bash
ls -la ml_models/trained_models/ng/ng123_seed{42,123,456}_*.pkl
ls reports/daily_selection_ng123_wf_oos/ | head
ls reports/daily_selection_ng123_pre2020/ | head
```

Expected: 3 个 .pkl 文件 + 2 个报告目录各有 ~500+ daily reports。

- [ ] **Step 6: Commit (模型文件被 .gitignore，仅 commit 训练状态)**

```bash
echo "Training complete: 3 seeds × ng1.2.3 ($(date))" \
  > reports/ng123/training/training_complete.md
git add reports/ng123/training/training_complete.md
git commit -m "chore(ng123): 3-seed 完整训练完成 + WF + Pre-2020 报告生成"
```

---

## Phase 4: 评估 + 决策

### Task 17: 北极星评估 (双向)

- [ ] **Step 1: WF-OOS 评估**

```bash
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_ng123_wf_oos \
    --label NG123-WFOOS --top-n 10 --focus-days 10 --rank-field composite \
    > reports/ng123/evaluation/wf_oos_v52.txt 2>&1

cat reports/ng123/evaluation/wf_oos_v52.txt | tail -30
```

- [ ] **Step 2: Pre-2020 评估**

```bash
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_ng123_pre2020 \
    --label NG123-PRE2020 --top-n 10 --focus-days 10 --rank-field composite \
    > reports/ng123/evaluation/pre2020_v52.txt 2>&1

cat reports/ng123/evaluation/pre2020_v52.txt | tail -30
```

- [ ] **Step 3: 提取关键指标**

```bash
python3 -c "
import re
out = {}
for label, path in [('WF-OOS', 'reports/ng123/evaluation/wf_oos_v52.txt'),
                    ('Pre-2020', 'reports/ng123/evaluation/pre2020_v52.txt')]:
    with open(path) as f:
        text = f.read()
    # Extract V5.2 score, MaxDD, Sharpe, annual return — adjust regex per actual report format
    v52_match = re.search(r'V5\.2.*?(\d+\.\d+)%', text)
    maxdd_match = re.search(r'MaxDD[:\s]+(-?\d+\.\d+)%', text)
    sharpe_match = re.search(r'Sharpe[:\s]+(\d+\.\d+)', text)
    annual_match = re.search(r'年化[:\s]+(\d+\.\d+)%', text)
    out[label] = {
        'V5.2': float(v52_match.group(1)) if v52_match else None,
        'MaxDD': float(maxdd_match.group(1)) if maxdd_match else None,
        'Sharpe': float(sharpe_match.group(1)) if sharpe_match else None,
        'Annual': float(annual_match.group(1)) if annual_match else None,
    }
import json
print(json.dumps(out, indent=2))
with open('reports/ng123/evaluation/summary.json', 'w') as f:
    json.dump(out, f, indent=2)
"
```

- [ ] **Step 4: Commit**

```bash
git add reports/ng123/evaluation/
git commit -m "eval(ng123): WF-OOS + Pre-2020 北极星评估完成"
```

---

### Task 18: 应用验收门槛 + 决策

- [ ] **Step 1: 比对验收门槛**

```bash
python3 << 'EOF'
import json

with open('reports/ng123/evaluation/summary.json') as f:
    metrics = json.load(f)

# 验收门槛 (per spec §8)
THRESHOLDS = {
    'WF-OOS': {'V5.2': 73.0, 'MaxDD_max': -10.0, 'Sharpe_min': 2.5, 'Annual_min': 130.0},
    'Pre-2020': {'V5.2': 70.0, 'Sharpe_min': 1.5},
}

passed = True
for label, t in THRESHOLDS.items():
    m = metrics.get(label, {})
    print(f"\n=== {label} ===")
    if 'V5.2' in t and m.get('V5.2') is not None:
        ok = m['V5.2'] >= t['V5.2']
        print(f"  V5.2 = {m['V5.2']}% {'✅' if ok else '❌'} (need >= {t['V5.2']}%)")
        passed &= ok
    if 'MaxDD_max' in t and m.get('MaxDD') is not None:
        ok = m['MaxDD'] >= t['MaxDD_max']  # MaxDD is negative; -8% >= -10% means OK
        print(f"  MaxDD = {m['MaxDD']}% {'✅' if ok else '❌'} (need >= {t['MaxDD_max']}%)")
        passed &= ok
    if 'Sharpe_min' in t and m.get('Sharpe') is not None:
        ok = m['Sharpe'] >= t['Sharpe_min']
        print(f"  Sharpe = {m['Sharpe']} {'✅' if ok else '❌'} (need >= {t['Sharpe_min']})")
        passed &= ok
    if 'Annual_min' in t and m.get('Annual') is not None:
        ok = m['Annual'] >= t['Annual_min']
        print(f"  Annual = {m['Annual']}% {'✅' if ok else '❌'} (need >= {t['Annual_min']}%)")
        passed &= ok

print(f"\n{'='*50}")
print(f"OVERALL: {'✅ ACCEPT — promote to production' if passed else '❌ REJECT — do not promote'}")
print(f"{'='*50}")

with open('reports/ng123/decision.md', 'w') as f:
    f.write(f'# ng1.2.3 Decision\n\n')
    f.write(f'**Status**: {"ACCEPTED ✅" if passed else "REJECTED ❌"}\n\n')
    f.write(f'## Metrics vs Thresholds\n\n')
    f.write(json.dumps(metrics, indent=2))
EOF
```

- [ ] **Step 2: Commit decision**

```bash
git add reports/ng123/decision.md
git commit -m "decision(ng123): 应用验收门槛 — $(grep -o 'ACCEPTED\|REJECTED' reports/ng123/decision.md | head -1)"
```

- [ ] **Step 3: 跳转**

- 若 ACCEPTED → Task 19 (Production Switch)
- 若 REJECTED → Task 22 (Postmortem)

---

## Phase 5A: Production Switch (if ACCEPTED)

### Task 19: 切换 PRODUCTION_VERSION

- [ ] **Step 1: 修改 ng_schema.py**

```python
# ml_models/ng/ng_schema.py:34
PRODUCTION_VERSION = 'ng1.2.3'  # 切换前: 'ng1.0.1'
```

- [ ] **Step 2: 验证 daily 流程**

```bash
# Smoke-test 选股流程 (用今天日期)
python3 tomorrow_stock_selector.py 2026-04-14
# 验证默认 scoring-version 已变成 ng1.2.3
```

- [ ] **Step 3: 验证 production scorer**

```bash
python3 -c "
from ml_models.ng.ng_production_scorer import NGProductionScorer
s = NGProductionScorer(version='ng1.2.3')
print(f'Loaded: {s.model_path if hasattr(s, \"model_path\") else \"OK\"}')
print(f'Feature count: {len(s.feature_names) if hasattr(s, \"feature_names\") else \"check\"}')
"
```

- [ ] **Step 4: Commit + push**

```bash
git add ml_models/ng/ng_schema.py
git commit -m "feat(ng123): 切换 PRODUCTION_VERSION = 'ng1.2.3'"
git push  # ⚠️ 仅在用户明确确认后执行
```

---

### Task 20: 更新 CLAUDE.md + Wiki + MEMORY.md

- [ ] **Step 1: CLAUDE.md 更新生产推荐章节**

打开 `CLAUDE.md`，找到 "🏆 NG v1.0.1 Production" 章节（搜索 `ng1.0.1`），更新为：

```markdown
1. **🏆 NG v1.2.3 Production** (生产推荐, 2026-04-14, 三轴重构):
   - 70 stock features (52 ng1.0.1 retained + 12 moneyflow + 6 mined) + 10 market features
   - 行业超额标签 - 0.3 × downside_kd 软风险惩罚
   - ICIR 自适应权重, 3-seed ensemble (42/123/456)
   - 性能: V5.2=<from decision.md>, 年化<X>%, Sharpe=<Y>, MaxDD=<Z>%
   - Pre-2020 OOS: <P>% A+
   - Scorer: `ml_models/ng/ng_production_scorer.py` (version='ng1.2.3')
   - 模型: `ml_models/trained_models/ng/ng123_seed{42,123,456}_*.pkl`
   - 缓存表: `ng123_feature_cache`
```

将原 ng1.0.1 章节降级为 "Previous Production":

```markdown
2. **NG v1.0.1** (旧生产, 2026-04-12 to 2026-04-14):
   - 64 features, V5.2=72.1% A+, Sharpe=2.753
   - 仍可调用: `--scoring-version ng1.0.1`
```

- [ ] **Step 2: 更新 MEMORY.md**

```bash
cat >> /Users/yangxu/.claude/projects/-Users-yangxu-StockTradebyZ/memory/ng123_production_switch.md << 'EOF'
---
name: NG v1.2.3 生产切换 (2026-04-14)
description: 三轴重构成功 — moneyflow + mined + downside label，新生产基线
type: project
---
**当前生产**: ng1.2.3 (之前: ng1.0.1)

**关键改动**:
- Drop 12 弱 ng1.0.1 特征 + 加 12 moneyflow + 6 mined = 70 stock features
- Label: industry_excess - 0.3 × downside_kd (软风险惩罚)
- 缓存表: ng123_feature_cache (独立，与 ng101 共存)

**性能** (vs ng1.0.1 baseline V5.2=72.1%):
- V5.2: <X>%
- MaxDD: <Y>%
- Sharpe: <Z>
- Pre-2020: <P>%

**回滚**: 单行 git revert `ng_schema.py:PRODUCTION_VERSION`
EOF
```

并更新 `MEMORY.md` 头部加入新条目（保持 200 行限制）。

- [ ] **Step 3: Wiki 更新**

```bash
# 在 docs/wiki/models/ng-series.md 加 ng1.2.3 章节
# 在 docs/wiki/index.md 索引里加链接
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/wiki/
git commit -m "docs(ng123): 更新 CLAUDE.md + Wiki 反映 ng1.2.3 上线"
```

---

## Phase 5B: Postmortem (if REJECTED)

### Task 21: 写 Postmortem

- [ ] **Step 1: 创建 postmortem**

创建 `docs/wiki/architecture/ng123_postmortem.md`：

```markdown
# ng1.2.3 Postmortem

**Date**: <YYYY-MM-DD>
**Status**: REJECTED (did not meet acceptance thresholds)

## Final Metrics

(Paste reports/ng123/evaluation/summary.json contents)

## Threshold Comparison

| Metric | Achieved | Threshold | Pass? |
|---|---|---|---|
| WF-OOS V5.2 | X% | >= 73% | ❌/✅ |
| ... |

## Root Cause Analysis

(Per spec §8.1 failure handling table — identify which failure mode applies)

- [ ] If V5.2 < 73%: ablation results — which feature subset caused regression?
- [ ] If Pre-2020 < 70%: SHAP per-feature analysis — which factor was regime-fragile?
- [ ] If MaxDD > -10%: λ analysis — was 0.3 too small?

## What We Learned

- Moneyflow alpha: <quantify gain or no-gain>
- Mined factors: <quantify gain or no-gain>
- Downside label: <effect on MaxDD vs ICIR trade>

## Next Steps (ng1.2.4 ideas)

- [ ] ...

## Artifacts Preserved

- Models: ml_models/trained_models/ng/ng123_seed{42,123,456}_*.pkl
- Cache: ng123_feature_cache (preserved for diagnostics)
- Reports: reports/daily_selection_ng123_{wf_oos,pre2020}/
- Decision: reports/ng123/decision.md
- Diagnostic: `python3 tomorrow_stock_selector.py 2026-04-14 --scoring-version ng1.2.3`
```

- [ ] **Step 2: 更新 MEMORY.md**

```bash
cat >> /Users/yangxu/.claude/projects/-Users-yangxu-StockTradebyZ/memory/ng123_failed.md << 'EOF'
---
name: ng1.2.3 三轴重构失败
description: moneyflow + mined + downside label 未达验收门槛
type: project
---
**结果**: REJECTED (生产保持 ng1.0.1)

**失败模式**: <V5.2 不及格 / Pre-2020 退化 / MaxDD 未改善>

**保留的产物** (供研究/复盘):
- 模型: `ng123_seed*_*.pkl`
- Cache: `ng123_feature_cache`
- 报告: `reports/daily_selection_ng123_*`
- Postmortem: `docs/wiki/architecture/ng123_postmortem.md`

**教训**: <key insight>
EOF
```

- [ ] **Step 3: Commit**

```bash
git add docs/wiki/architecture/ng123_postmortem.md
git commit -m "docs(ng123): 失败 postmortem + MEMORY.md 落档"
```

---

## 总结：任务清单

**Phase 0 — Setup** (~½ day): Tasks 1-3
**Phase 1 Stage 1 — Moneyflow IC** (~5 hours): Tasks 4-8
**Phase 1 Stage 2 — Mined Validate** (~2 hours): Tasks 9-10
**Phase 1 Stage 3 — λ Ablation** (~2 hours): Tasks 11-14
**Phase 2 — Full Backfill** (~6-8 hours): Task 15
**Phase 3 — Training** (~2-6 hours): Task 16
**Phase 4 — Evaluation + Decision**: Tasks 17-18
**Phase 5A or 5B**: Task 19-20 (accept) OR Task 21 (reject)

**Total: 4-5 working days** (assuming no major surprises requiring re-iteration)

---

## 关键安全点

1. **PRODUCTION_VERSION 不动**：从 Phase 0 到 Phase 4 全程 `'ng1.0.1'`，仅 Task 19 在验收通过后切换
2. **ng1.0.1 模型 + cache 完全保留**：ng123_feature_cache 是独立表，ng101_feature_cache 不受影响
3. **每个 Stage 都有独立 fail action**：Stage 1 失败可终止；Stage 2 失败可降级到 moneyflow-only；Stage 3 失败可用 λ=0
4. **回滚成本 5 分钟**：Production 切换是单行 PRODUCTION_VERSION 修改，git revert 即可恢复

## 关键调试 tips

1. **Code 格式不一致**：moneyflow 用 `000001.SZ`，ng cache 可能用 `000001`。Stage 1 的 IC 计算前应 normalize（在脚本里加 `code = code.split('.')[0]` 或反向）
2. **NaN 容忍**：所有因子函数应返回 NaN 而非抛异常；LightGBM 原生支持 NaN
3. **Tushare 单位**：moneyflow 数值单位是 `万元`（10K 元），但所有因子都是比率所以单位自动消去
4. **Trainer log 解析**：Stage 3 的 λ ablation 脚本依赖 trainer log 输出 ICIR；如果 log 格式不匹配，调整 `run_one_lambda` 的 regex
