# Repo 深层重构计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 消除 simplify review 发现的跨文件重复和效率问题，降低维护成本

**Architecture:** 提取共享模块，用 import 替代 copy-paste，保持所有现有功能不变

**Tech Stack:** Python 3, SQLite, pandas, numpy

---

## 文件清单

| 文件 | 操作 | 职责 |
|------|------|------|
| `fetch_data/label_utils.py` | 新建 | 统一标签计算函数 |
| `fetch_data/shared_data_loader.py` | 新建 | 共享数据加载（stock_data + industry mapping） |
| `fetch_data/v39_feature_cache_updater.py` | 修改 | 使用共享模块 |
| `fetch_data/v40_feature_cache_updater.py` | 修改 | 使用共享模块 |
| `fetch_data/alpha158_feature_cache_updater.py` | 修改 | 使用共享模块 |
| `ml_models/ng/ng_cache_updater.py` | 修改 | 使用 label_utils |
| `tomorrow_stock_selector.py` | 修改 | registry dict 替代 elif 链 |

---

### Task 1: 提取 label_utils.py — 统一标签计算

**问题:** 标签公式 `close[T+1+N] / open[T+1] - 1` 在 7+ 文件中独立实现，包含相同的停牌检测、边界检查、log_return 变体。任何标签语义变更需改 7+ 处。

**Files:**
- Create: `fetch_data/label_utils.py`
- Modify: `fetch_data/v39_feature_cache_updater.py`
- Modify: `fetch_data/v40_feature_cache_updater.py`
- Modify: `fetch_data/alpha158_feature_cache_updater.py`
- Modify: `ml_models/ng/ng_cache_updater.py`

- [ ] **Step 1: 创建 label_utils.py**

```python
"""统一标签计算函数 — 所有 feature cache updater 共用。

标签定义: label_Nd = close[T+1+N] / open[T+1] - 1
  - T = 信号日 (trade_date)
  - T+1 = 买入日 (次日开盘买入)
  - T+1+N = 卖出日 (N天后收盘卖出)
  - 停牌检测: volume[T+1] == 0 → 跳过
"""
import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_aligned_labels(
    trade_dates: List[str],
    opens: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    current_idx: int,
    horizons: Tuple[int, ...] = (3, 5, 10, 15),
    log_return: bool = False,
) -> Dict[str, float]:
    """计算对齐标签（从 sorted price arrays 的 current_idx 位置开始）。

    Args:
        trade_dates: 日期数组 (sorted ascending)
        opens/closes/volumes: 对应价格数组
        current_idx: 信号日在数组中的位置
        horizons: 持仓天数列表
        log_return: True 时用 log(close/open) 替代 close/open - 1

    Returns:
        {'label_3d': float, 'label_5d': float, ...}  NaN if unavailable
    """
    labels = {}
    n = len(trade_dates)
    buy_idx = current_idx + 1  # T+1

    if buy_idx >= n:
        return {f'label_{h}d': np.nan for h in horizons}

    buy_open = opens[buy_idx]
    buy_volume = volumes[buy_idx]

    # 停牌检测
    if buy_volume == 0 or buy_open <= 0 or np.isnan(buy_open):
        return {f'label_{h}d': np.nan for h in horizons}

    for h in horizons:
        sell_idx = buy_idx + h
        if sell_idx >= n:
            labels[f'label_{h}d'] = np.nan
            continue

        sell_close = closes[sell_idx]
        if sell_close <= 0 or np.isnan(sell_close):
            labels[f'label_{h}d'] = np.nan
            continue

        if log_return:
            labels[f'label_{h}d'] = float(np.log(sell_close / buy_open))
        else:
            labels[f'label_{h}d'] = float(sell_close / buy_open - 1.0)

    return labels


def compute_labels_from_future_prices(
    base_open: float,
    future_closes: Dict[int, float],
    horizons: Tuple[int, ...] = (3, 5, 10, 15),
) -> Dict[str, float]:
    """从预加载的 future prices 计算标签（NG cache updater 风格）。

    Args:
        base_open: T+1 日开盘价
        future_closes: {horizon_days: close_price}
        horizons: 持仓天数

    Returns:
        {'label_3d': float, ...}
    """
    if base_open <= 0 or np.isnan(base_open):
        return {f'label_{h}d': np.nan for h in horizons}

    labels = {}
    for h in horizons:
        close = future_closes.get(h, np.nan)
        if close is None or np.isnan(close) or close <= 0:
            labels[f'label_{h}d'] = np.nan
        else:
            labels[f'label_{h}d'] = float(close / base_open - 1.0)
    return labels
```

- [ ] **Step 2: 在 v39_feature_cache_updater.py 中使用 label_utils**

替换 `calculate_labels()` 和 `_compute_labels_from_preloaded()` 中的内联公式，改为调用 `compute_aligned_labels()`。保留函数签名和返回格式不变。

- [ ] **Step 3: 在 v40_feature_cache_updater.py 中使用 label_utils**

替换 `_compute_excess_labels()` 中的内联公式。v40 的 excess 标签 = stock_label - benchmark_label，两个 raw label 都用 `compute_aligned_labels` 计算。

- [ ] **Step 4: 在 alpha158_feature_cache_updater.py 中使用 label_utils**

替换内联 label 计算。alpha158 使用 `log_return=True`。

- [ ] **Step 5: 在 ng_cache_updater.py 中使用 label_utils**

替换 `_compute_labels()` 中的公式为 `compute_labels_from_future_prices()`。

- [ ] **Step 6: 验证所有 updater 的标签输出一致**

```bash
# 对比 v39 单日输出
python3 -c "
from fetch_data.v39_feature_cache_updater import V39FeatureCacheUpdater
u = V39FeatureCacheUpdater()
# 验证标签计算结果未变
"
```

- [ ] **Step 7: Commit**

```bash
git add fetch_data/label_utils.py fetch_data/v39_feature_cache_updater.py fetch_data/v40_feature_cache_updater.py fetch_data/alpha158_feature_cache_updater.py ml_models/ng/ng_cache_updater.py
git commit -m "refactor: 提取label_utils.py — 7处标签计算统一为单一来源"
```

---

### Task 2: 提取 shared_data_loader.py — 共享数据加载

**问题:** `_batch_load_stock_data()` 和 `_load_sw_industry_mapping()` 在 v39/v40/v390_scorer 中各有一份近乎相同的实现。

**Files:**
- Create: `fetch_data/shared_data_loader.py`
- Modify: `fetch_data/v39_feature_cache_updater.py`
- Modify: `fetch_data/v40_feature_cache_updater.py`

- [ ] **Step 1: 创建 shared_data_loader.py**

提取两个函数:
- `batch_load_stock_data(db_path, start_date, end_date) -> Dict[str, pd.DataFrame]`
- `load_sw_industry_mapping(db_path) -> Tuple[Dict[str, str], Dict[str, int]]`

- [ ] **Step 2: v39_feature_cache_updater.py 替换为 import**

删除 `_batch_load_stock_data()` 和 `_load_sw_industry_mapping()` 方法，改为 `from fetch_data.shared_data_loader import ...`

- [ ] **Step 3: v40_feature_cache_updater.py 替换为 import**

同上。

- [ ] **Step 4: 验证**

```bash
python3 fetch_data/v39_feature_cache_updater.py --date 2026-04-03
python3 fetch_data/v40_feature_cache_updater.py --date 2026-04-03
```

- [ ] **Step 5: Commit**

```bash
git commit -m "refactor: 提取shared_data_loader.py — batch_load_stock_data + sw_industry统一"
```

---

### Task 3: tomorrow_stock_selector.py — registry dict 替代 elif 链

**问题:** 30+ 个近乎相同的 `elif scoring_version == "v4.X.Y"` 块，每个块做相同的事：import scorer class → 实例化。新增版本需 copy-paste 一个块。

**Files:**
- Modify: `tomorrow_stock_selector.py`

- [ ] **Step 1: 定义版本注册表**

在文件顶部定义:

```python
SCORER_REGISTRY = {
    'v3.9': ('ml_models.v39.v390_production_scorer', 'V390ProductionScorer'),
    'v3.95': ('ml_models.v39.v395_production_scorer', 'V395ProductionScorer'),
    'v4.3': ('ml_models.v39.v43_production_scorer', 'V43ProductionScorer'),
    # ... all versions ...
    'ng1.0.0': ('ml_models.ng.ng_production_scorer', 'NGProductionScorer'),
    'ng1.0.1': ('ml_models.ng.ng_production_scorer', 'NGProductionScorer'),
    'ng1.0.2': ('ml_models.ng.ng_production_scorer', 'NGProductionScorer'),
    'ng1.1.0': ('ml_models.ng.ng_production_scorer', 'NGProductionScorer'),
}
```

- [ ] **Step 2: 替换 elif 链为 registry lookup**

```python
if scoring_version in SCORER_REGISTRY:
    module_path, class_name = SCORER_REGISTRY[scoring_version]
    import importlib
    mod = importlib.import_module(module_path)
    ScorerClass = getattr(mod, class_name)
    self.scoring_engine_v44 = ScorerClass(model_type='small_data')
    self.v44_batch_cache = {}
    from ml_models.v39.strategy_based_return_predictor import StrategyBasedReturnPredictor
    self.strategy_return_predictor = StrategyBasedReturnPredictor()
elif scoring_version in DEPRECATED_VERSIONS:
    logger.warning(f"版本 {scoring_version} 已弃用")
    # ... fallback ...
else:
    raise ValueError(f"Unknown scoring version: {scoring_version}")
```

- [ ] **Step 3: 同时用 ACTIVE_VERSIONS 做校验**

```python
ACTIVE_VERSIONS = set(SCORER_REGISTRY.keys())
```

- [ ] **Step 4: 验证**

```bash
python3 tomorrow_stock_selector.py 2026-04-03 --scoring-version ng1.1.0 --dry-run
python3 tomorrow_stock_selector.py 2026-04-03 --scoring-version v4.9.0.1 --dry-run
```

- [ ] **Step 5: Commit**

```bash
git commit -m "refactor: tomorrow_stock_selector registry dict替代30+ elif链"
```

---

### Task 4: backfill_labels N+1 查询优化

**问题:** v39/v40/alpha158 的 `backfill_labels()` 对每个缺失标签行做独立 SQL 查询（N+1 模式）。全量回填时可达 50,000+ 查询。

**Files:**
- Modify: `fetch_data/v39_feature_cache_updater.py`
- Modify: `fetch_data/v40_feature_cache_updater.py`
- Modify: `fetch_data/alpha158_feature_cache_updater.py`

- [ ] **Step 1: v39 backfill_labels 改为 batch 预加载**

```python
def backfill_labels(self):
    # 1. 找出所有缺失标签的 (code, trade_date)
    missing = self._find_missing_labels()
    if not missing:
        return
    
    # 2. 批量预加载所需价格数据
    codes = list(set(m[0] for m in missing))
    min_date = min(m[1] for m in missing)
    price_data = self._batch_load_prices_for_labels(codes, min_date)
    
    # 3. 用 label_utils 计算标签
    from fetch_data.label_utils import compute_aligned_labels
    updates = []
    for code, trade_date in missing:
        prices = price_data.get(code)
        if prices is None:
            continue
        idx = prices['dates'].index(trade_date) if trade_date in prices['dates'] else -1
        if idx < 0:
            continue
        labels = compute_aligned_labels(
            prices['dates'], prices['opens'], prices['closes'], prices['volumes'],
            idx, horizons=(3, 5, 10, 15))
        updates.append((labels, code, trade_date))
    
    # 4. 批量 UPDATE
    ...
```

- [ ] **Step 2: v40 backfill_labels 同理**

额外预加载 HS300 数据（一次查询，缓存 by date）。

- [ ] **Step 3: alpha158 backfill_labels 同理**

使用 `log_return=True`。

- [ ] **Step 4: 验证**

```bash
python3 fetch_data/v39_feature_cache_updater.py --backfill-labels --dry-run
```

- [ ] **Step 5: Commit**

```bash
git commit -m "perf: backfill_labels batch预加载替代N+1查询 — 预计100x加速"
```
