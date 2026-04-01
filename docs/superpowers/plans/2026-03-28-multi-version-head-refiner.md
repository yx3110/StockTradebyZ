# Multi-Version Head Refiner 训练 + 对比回测

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为V4.7.5~V4.8.6共7个模型各训练一个Head Refiner，然后统一回测对比全期+近2月表现

**Architecture:**
- 改造 `train_head_refiner.py` 支持 `--version` 参数，自动加载对应scorer和模型目录
- 为V4.7.5添加 `_per_model_preds` 保存（当前只有V4.8.1+有此功能）
- 写统一回测脚本，对比8个版本(含v487)的 Composite/Consensus/Refiner 三种策略

**Tech Stack:** LightGBM, scipy, joblib, sqlite3

---

### Task 1: 改造 train_head_refiner.py 支持多版本

**Files:**
- Modify: `ml_models/training/train_head_refiner.py`

- [ ] **Step 1: 添加 --version 参数和scorer自动加载**

在 `main()` 的 argparse 中添加:
```python
parser.add_argument('--version', default='v4.8.7',
    choices=['v4.7.5','v4.8.1','v4.8.2','v4.8.3','v4.8.4','v4.8.5','v4.8.6','v4.8.7'])
```

添加版本→scorer映射函数:
```python
def load_scorer(version):
    mapping = {
        'v4.7.5': ('v475', 'V475ProductionScorer'),
        'v4.8.1': ('v481', 'V481ProductionScorer'),
        'v4.8.2': ('v482', 'V482ProductionScorer'),
        'v4.8.3': ('v483', 'V483ProductionScorer'),
        'v4.8.4': ('v484', 'V484ProductionScorer'),
        'v4.8.5': ('v485', 'V485ProductionScorer'),
        'v4.8.6': ('v486', 'V486ProductionScorer'),
        'v4.8.7': ('v487', 'V487ProductionScorer'),
    }
    vkey, cls_name = mapping[version]
    mod = __import__(f'ml_models.v39.{vkey}_production_scorer', fromlist=[cls_name])
    cls = getattr(mod, cls_name)
    return cls(model_type='small_data'), vkey
```

修改 `MODEL_DIR` 为动态: `MODEL_DIR = PROJECT_ROOT / 'ml_models' / 'trained_models' / vkey`

修改保存文件名: `head_refiner_{vkey}_{timestamp}.pkl`

- [ ] **Step 2: 验证可为v484训练**

```bash
python3 ml_models/training/train_head_refiner.py --version v4.8.4 --start-date 2022-01-01 --end-date 2026-03-20
```
预期: 在 `ml_models/trained_models/v484/` 下生成 `head_refiner_v484_*.pkl`

---

### Task 2: 为V4.7.5添加 _per_model_preds 保存

**Files:**
- Modify: `ml_models/v39/v475_production_scorer.py`

V475继承V473→V44→V395链, predict_scores不经过V481。需要在V475的predict_scores中添加保存逻辑。

- [ ] **Step 1: 查找V475 predict_scores中的子模型预测循环**

V475.predict_scores 调用 super().predict_scores() → V473 → 最终到基类。
需要找到子模型预测循环位置，添加 `self._per_model_preds[target] = dict(preds)` 和 `self._last_pred_codes = codes`。

如果V475调用链太深难改，替代方案: 在V475.predict_scores中，在super()调用后，用`self.models['10d']`重新跑一次预测，存到`_per_model_preds`。

- [ ] **Step 2: 验证V475保存了per_model_preds**

```python
from ml_models.v39.v475_production_scorer import V475ProductionScorer
scorer = V475ProductionScorer(model_type='small_data')
results = scorer.predict_scores(codes[:100], '2026-03-27')
assert '10d' in scorer._per_model_preds
assert len(scorer._last_pred_codes) == 100
```

---

### Task 3: 批量训练7个版本的Head Refiner

**Files:**
- Create: `scripts/train_all_head_refiners.sh`

- [ ] **Step 1: 写批量训练脚本**

```bash
#!/bin/bash
for v in v4.7.5 v4.8.1 v4.8.2 v4.8.3 v4.8.4 v4.8.5 v4.8.6; do
    echo "=== Training Head Refiner for $v ==="
    python3 ml_models/training/train_head_refiner.py \
        --version $v --start-date 2022-01-01 --end-date 2026-03-20 \
        --top-pct 0.02 --n-stock-features 20
    echo ""
done
```

- [ ] **Step 2: 运行批量训练**

```bash
bash scripts/train_all_head_refiners.sh 2>&1 | tee /tmp/train_all_refiners.log
```

预期耗时: ~5分钟/版本 × 7 = ~35分钟 (feature cache加载共享，主要时间在LightGBM训练)

- [ ] **Step 3: 验证所有refiner已生成**

```bash
for v in v475 v481 v482 v483 v484 v485 v486; do
    ls ml_models/trained_models/${v}/head_refiner_*.pkl
done
```

---

### Task 4: 统一回测对比脚本

**Files:**
- Create: `scripts/backtest_all_refiners.py`

- [ ] **Step 1: 写统一回测脚本**

脚本逻辑:
1. 加载8个版本的scorer + refiner
2. 采样~40个交易日 (2025-01-01 ~ 2026-03-27)
3. 对每个(版本, 日期): 全市场predict_scores → 提取composite/consensus/refiner → 匹配forward 10d returns
4. 输出对比表: 全期 + 近2月，3种策略(Composite Top-10 / Consensus>=2 / Refiner Top-10)

关键: 各版本用自己的scorer做预测，用自己的refiner做精筛，公平对比。

- [ ] **Step 2: 运行回测**

```bash
python3 scripts/backtest_all_refiners.py 2>&1 | tee /tmp/backtest_all_refiners.log
```

预期耗时: 8个版本 × 40日 × ~10s = ~53分钟

- [ ] **Step 3: 输出对比表**

格式:
```
=== 全期对比 (10d持仓) ===
版本     | Composite Top-10          | Consensus>=2              | Refiner Top-10
         | avg    wr    PF           | avg    wr    PF           | avg    wr    PF
V4.7.5   | +x.xx% xx.x% x.xxx      | +x.xx% xx.x% x.xxx      | +x.xx% xx.x% x.xxx
...

=== 近2月对比 ===
...
```

---

### Task 5: 更新 recommendation_thresholds.json

- [ ] **Step 1: 对表现最好的版本+策略组合，更新阈值文件**

根据回测结果，为每个版本选择最优策略(Refiner阈值 or Consensus阈值)，更新对应的 `recommendation_thresholds.json`。

---
