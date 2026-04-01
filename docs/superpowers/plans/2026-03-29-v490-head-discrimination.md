# V4.9.0 Head Discrimination 训练改进 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将Q95分位数模型和头部加权训练内化到V4.9.0训练pipeline中，从根本上提升头部区分度

**Architecture:** 基于V4.8.5训练器(V485Trainer)创建V490Trainer，三项改进：(1) Q95作为第7个base model纳入ensemble (2) 头尾20%样本权重×5 (3) LambdaRank truncation从50降到10。训练完成后创建V490ProductionScorer，内置Widen-then-Concentrate pipeline。

**Tech Stack:** LightGBM (quantile + lambdarank), XGBoost, CatBoost, sklearn

---

## File Structure

| File | Responsibility |
|------|---------------|
| `ml_models/training/train_v395_multi_target.py` | 新增V490Trainer类 (继承V485Trainer) |
| `ml_models/v39/v490_production_scorer.py` | V4.9.0 生产scorer (继承V485, 内置Q95 Widen-then-Concentrate) |
| `ml_models/trained_models/v490/` | 训练产物目录 |

---

### Task 1: 创建V490Trainer — Q95 + 头部加权 + truncation=10

**Files:**
- Modify: `ml_models/training/train_v395_multi_target.py` (在文件末尾class V488Trainer之前添加V490Trainer)

- [ ] **Step 1: 添加V490Trainer类**

在 `class V486Trainer` 之后、`class V488Trainer` 之前插入：

```python
class V490Trainer(V485Trainer):
    """V4.9.0 训练器 — V4.8.5底座 + Q95分位数模型 + 头部加权 + truncation=10

    三项头部区分度改进:
      1. lgb_q95: LightGBM quantile(alpha=0.95)作为第7个base model
      2. 头尾20%样本权重×5 (compute_sample_weights)
      3. LambdaRank truncation_level: 50→10, relevance: 5档→10档
    """

    HEAD_TAIL_PCT = 0.20    # 头尾各20%样本加权
    HEAD_TAIL_WEIGHT = 5.0  # 权重倍数
    RANK_TRUNCATION = 10    # LambdaRank truncation (was 50)
    RANK_RELEVANCE_GRADES = 10  # relevance档数 (was 5)
    Q95_ALPHA = 0.95        # quantile分位数

    def compute_sample_weights(self, df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """V4.9.0: V4.8.5权重 + 头尾20%加权×5"""
        weights = super().compute_sample_weights(df, y)

        if 'trade_date' not in df.columns:
            return weights

        dates_arr = df['trade_date'].values
        n_head_tail = 0
        for d in np.unique(dates_arr):
            mask = dates_arr == d
            idx = np.where(mask)[0]
            n = len(idx)
            if n < 50:
                continue
            y_d = y[idx]
            from scipy.stats import rankdata as _rankdata
            ranks = _rankdata(y_d)
            pct = (ranks - 1) / (n - 1)
            # Top-20% and Bottom-20%
            head_tail_mask = (pct >= (1 - self.HEAD_TAIL_PCT)) | (pct <= self.HEAD_TAIL_PCT)
            weights[idx[head_tail_mask]] *= self.HEAD_TAIL_WEIGHT
            n_head_tail += head_tail_mask.sum()

        logger.info(f"    V4.9.0 头尾{self.HEAD_TAIL_PCT*100:.0f}%加权: "
                     f"{n_head_tail:,} 样本 × {self.HEAD_TAIL_WEIGHT}")
        return weights

    def train_single_target_models(self, X_train, X_val, y_train, y_val, target_name: str,
                                    sample_weights_train=None):
        """V4.9.0: V4.8.5的6个模型 + Q95分位数模型 + 改进LambdaRank"""
        import gc

        # 1. 继承V4.7.1的6个模型 (lgb, xgb, cb, rf, hgb, lgb_rank)
        # 但先覆盖LambdaRank参数
        original_train = super().train_single_target_models
        models, pred_train, pred_val = original_train(
            X_train, X_val, y_train, y_val, target_name,
            sample_weights_train=sample_weights_train)

        # 2. 重新训练LambdaRank with truncation=10 (覆盖上面的truncation=50版本)
        train_dates = getattr(self, 'train_dates', None)
        val_dates = getattr(self, 'val_dates', None)

        if train_dates is not None and len(train_dates) == len(y_train):
            logger.info(f"  V4.9.0 重训LambdaRank (truncation={self.RANK_TRUNCATION}, "
                         f"{self.RANK_RELEVANCE_GRADES}档)...")
            try:
                from scipy.stats import rankdata

                n_grades = self.RANK_RELEVANCE_GRADES
                relevance_train = np.zeros(len(y_train), dtype=np.int32)
                group_train = []
                for d in np.unique(train_dates):
                    mask = train_dates == d
                    n = mask.sum()
                    group_train.append(n)
                    if n >= 10:
                        ranks = rankdata(y_train[mask])
                        pct = (ranks - 1) / (n - 1)
                        relevance_train[mask] = np.clip((pct * n_grades).astype(int), 0, n_grades - 1)
                    else:
                        relevance_train[mask] = n_grades // 2

                relevance_val = np.zeros(len(y_val), dtype=np.int32)
                group_val = []
                if val_dates is not None and len(val_dates) == len(y_val):
                    for d in np.unique(val_dates):
                        mask = val_dates == d
                        n = mask.sum()
                        group_val.append(n)
                        if n >= 10:
                            ranks = rankdata(y_val[mask])
                            pct = (ranks - 1) / (n - 1)
                            relevance_val[mask] = np.clip((pct * n_grades).astype(int), 0, n_grades - 1)
                        else:
                            relevance_val[mask] = n_grades // 2

                lgb_rank_params = {
                    'objective': 'lambdarank',
                    'metric': 'ndcg',
                    'eval_at': [5, 10],
                    'lambdarank_truncation_level': self.RANK_TRUNCATION,
                    'lambdarank_norm': True,
                    'num_leaves': 31,
                    'learning_rate': 0.03,
                    'feature_fraction': 0.7,
                    'bagging_fraction': 0.7,
                    'bagging_freq': 5,
                    'reg_alpha': 0.5,
                    'reg_lambda': 2.0,
                    'min_data_in_leaf': 200,
                    'verbose': -1,
                }

                lgb_rank_train = lgb.Dataset(
                    X_train, label=relevance_train, group=group_train,
                    weight=sample_weights_train, free_raw_data=True)
                lgb_rank_val = lgb.Dataset(
                    X_val, label=relevance_val, group=group_val,
                    reference=lgb_rank_train, free_raw_data=True)

                lgb_rank_model = lgb.train(
                    lgb_rank_params, lgb_rank_train,
                    num_boost_round=1000,
                    valid_sets=[lgb_rank_val],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

                models['lgb_rank'] = lgb_rank_model  # 覆盖旧版
                pred_train['lgb_rank'] = lgb_rank_model.predict(X_train)
                pred_val['lgb_rank'] = lgb_rank_model.predict(X_val)
                logger.info(f"    LambdaRank(trunc={self.RANK_TRUNCATION}): {lgb_rank_model.num_trees()} trees")

                del lgb_rank_train, lgb_rank_val
                gc.collect()
            except Exception as e:
                logger.warning(f"    V4.9.0 LambdaRank重训失败: {e}")

        # 3. Q95 分位数模型
        logger.info(f"  训练 LGB-Q95 ({target_name}, alpha={self.Q95_ALPHA})...")
        try:
            lgb_q95_params = {
                'objective': 'quantile',
                'alpha': self.Q95_ALPHA,
                'metric': 'quantile',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 100,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'verbose': -1,
            }

            q95_train = lgb.Dataset(X_train, y_train, weight=sample_weights_train)
            q95_val = lgb.Dataset(X_val, y_val, reference=q95_train)

            q95_model = lgb.train(
                lgb_q95_params, q95_train,
                num_boost_round=1000,
                valid_sets=[q95_val],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

            models['lgb_q95'] = q95_model
            pred_train['lgb_q95'] = q95_model.predict(X_train)
            pred_val['lgb_q95'] = q95_model.predict(X_val)
            logger.info(f"    LGB-Q95 ({target_name}): {q95_model.num_trees()} trees")

            del q95_train, q95_val
            gc.collect()
        except Exception as e:
            logger.warning(f"    LGB-Q95 ({target_name}) 训练失败: {e}")

        return models, pred_train, pred_val

    def walk_forward_train(self, start_date=None, end_date=None,
                            purge_days=15, min_train_days=900,
                            val_days=120, test_days=120, step_days=90):
        """V4.9.0 Walk-Forward — 调用V4.8.5的流程, 然后重命名为v490"""
        # V4.8.5的walk_forward会保存为v485格式
        super().walk_forward_train(
            start_date=start_date, end_date=end_date,
            purge_days=purge_days, min_train_days=min_train_days,
            val_days=val_days, test_days=test_days, step_days=step_days)

        # 重命名v485 → v490
        import shutil
        v485_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v485'
        v490_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v490'
        v490_dir.mkdir(parents=True, exist_ok=True)

        v485_files = sorted(v485_dir.glob('v485_*.pkl'), key=lambda f: f.stat().st_mtime)
        if v485_files:
            latest = v485_files[-1]
            timestamp = latest.stem.replace('v485_multi_target_', '')
            new_path = v490_dir / f'v490_multi_target_{timestamp}.pkl'
            model_data = joblib.load(str(latest))
            model_data['version'] = 'v4.9.0'
            model_data['v490_innovations'] = {
                'lgb_q95': f'quantile(alpha={self.Q95_ALPHA})',
                'head_tail_weight': f'{self.HEAD_TAIL_PCT*100:.0f}% × {self.HEAD_TAIL_WEIGHT}',
                'lambdarank_truncation': self.RANK_TRUNCATION,
                'relevance_grades': self.RANK_RELEVANCE_GRADES,
            }
            joblib.dump(model_data, str(new_path), compress=3)
            logger.info(f"  V4.9.0 模型已保存: {new_path}")

            # Copy auxiliary files
            for aux in ['global_quantiles.npy', 'winsorize_bounds.npy']:
                src = v485_dir / aux
                if src.exists():
                    shutil.copy2(str(src), str(v490_dir / aux))

            # Training history
            for hf in v485_dir.glob(f'training_history_{timestamp}*'):
                import json as _json
                history = _json.loads(hf.read_text())
                history['version'] = 'v4.9.0'
                dest = v490_dir / hf.name
                dest.write_text(_json.dumps(history, indent=2, ensure_ascii=False))
        else:
            logger.warning("V4.9.0: No v485 model found to rename")
```

- [ ] **Step 2: 添加CLI入口 `--v490`**

在 `main()` 函数的 argparse 部分添加:
```python
parser.add_argument('--v490', action='store_true',
    help='V4.9.0: V4.8.5+Q95分位数+头尾20%%加权+LambdaRank trunc=10')
```

在 `if args.v488:` 之前添加:
```python
if args.v490:
    trainer = V490Trainer()
    _apply_overrides(trainer)
    if args.skip_wf:
        trainer.train_production_only(
            start_date=args.start_date, end_date=args.end_date,
            purge_days=max(args.purge_days, 15))
    else:
        trainer.walk_forward_train(
            start_date=args.start_date, end_date=args.end_date,
            purge_days=max(args.purge_days, 15))
elif args.v488:
```

- [ ] **Step 3: 验证V490Trainer可实例化**

```bash
python3 -c "
from ml_models.training.train_v395_multi_target import V490Trainer
t = V490Trainer()
print(f'V490Trainer: HEAD_TAIL_PCT={t.HEAD_TAIL_PCT}, RANK_TRUNCATION={t.RANK_TRUNCATION}, Q95_ALPHA={t.Q95_ALPHA}')
print(f'Models for 10d: will include lgb, xgb, cb, rf, hgb, lgb_rank, lgb_q95')
"
```

---

### Task 2: 创建V490ProductionScorer

**Files:**
- Create: `ml_models/v39/v490_production_scorer.py`

- [ ] **Step 1: 创建scorer文件**

```python
#!/usr/bin/env python3
"""
V4.9.0 production scorer — V4.8.5底座 + Q95 Widen-then-Concentrate (内置)

训练改进: Q95第7模型 + 头尾20%加权 + LambdaRank truncation=10
推理改进: Widen-then-Concentrate (MSE Top-30 → Q95 Top-10)

Q95模型已包含在训练产物中(models['10d']['lgb_q95']), 无需额外加载。
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List
from scipy.stats import rankdata

from .v485_production_scorer import V485ProductionScorer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

WIDEN_TOP_K = 30
HEAD_SELECT = 10


class V490ProductionScorer(V485ProductionScorer):
    """V4.9.0 scorer — 内置Q95 Widen-then-Concentrate"""

    def __init__(self, model_type: str = 'small_data'):
        self._v490_model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v490'
        super().__init__(model_type=model_type)

    def _load_models(self):
        v490_files = list(self._v490_model_dir.glob('v490_*.pkl'))
        if v490_files:
            self.model_dir = self._v490_model_dir
            latest = max(v490_files, key=lambda f: f.stat().st_mtime)
            self._load_model_from_file(latest, label='V4.9.0')
            # Check Q95 model is present
            has_q95 = 'lgb_q95' in self.models.get('10d', {})
            logger.info(f"  Q95 in ensemble: {has_q95}")
            return
        super()._load_models()

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """V4.9.0: 基础预测 + Widen-then-Concentrate via 内置lgb_q95"""
        results = super().predict_scores(stock_codes, date)

        # Widen-then-Concentrate using built-in Q95
        per_model_preds = getattr(self, '_per_model_preds', {})
        pred_codes = getattr(self, '_last_pred_codes', [])

        if pred_codes and '10d' in per_model_preds:
            preds_10d = per_model_preds['10d']
            q95_pred = preds_10d.get('lgb_q95')
            n = len(pred_codes)

            if q95_pred is not None and len(q95_pred) == n:
                # MSE composite (exclude Q95 from composite for clean separation)
                mse_names = [nm for nm in preds_10d if nm != 'lgb_q95']
                tw = self.weights.get('label_10d', {})
                mse_composite = np.zeros(n)
                total_w = 0
                for nm in mse_names:
                    if nm in preds_10d and len(preds_10d[nm]) == n:
                        w = tw.get(nm, 0.2)
                        mse_composite += w * preds_10d[nm]
                        total_w += w
                if total_w > 0:
                    mse_composite /= total_w

                mse_rank = rankdata(-mse_composite, method='ordinal')

                # Stage 1: MSE Top-30
                pool_mask = mse_rank <= WIDEN_TOP_K
                pool_idx = np.where(pool_mask)[0]

                if len(pool_idx) >= 3:
                    # Stage 2: Q95 reranking within pool
                    q95_in_pool = q95_pred[pool_idx]
                    q95_pool_rank = rankdata(-q95_in_pool, method='ordinal')

                    for ii, idx in enumerate(pool_idx):
                        code = pred_codes[idx]
                        if code in results:
                            results[code]['head_rank'] = int(q95_pool_rank[ii])
                            results[code]['in_head_pool'] = True
                            results[code]['q95_pred_10d'] = float(q95_pred[idx])

                for i, code in enumerate(pred_codes):
                    if code in results and 'in_head_pool' not in results[code]:
                        results[code]['head_rank'] = WIDEN_TOP_K + int(mse_rank[i])
                        results[code]['in_head_pool'] = False

        return results
```

- [ ] **Step 2: 注册到tomorrow_stock_selector.py**

在 `ACTIVE_VERSIONS` set中添加 `'v4.9.0'`，在版本初始化分支中添加:
```python
elif scoring_version == "v4.9.0":
    from ml_models.v39.v490_production_scorer import V490ProductionScorer
    self.scoring_engine_v44 = V490ProductionScorer(model_type='small_data')
    self.v44_batch_cache = {}
    logger.info("🔬 已初始化V4.9.0评分系统 (Q95+头部加权+truncation=10)")
```

在报告目录映射中添加:
```python
elif scoring_version == "v4.9.0":
    report_dir = Path("reports/daily_selection_v4.9.0")
```

---

### Task 3: 训练V4.9.0模型

- [ ] **Step 1: 运行训练**

```bash
python3 ml_models/training/train_v395_multi_target.py \
    --v490 --start-date 2020-01-01 --purge-days 15 --sharpe-blend 0.3 \
    2>&1 | tee /tmp/train_v490.log
```

预期耗时: ~6-8小时 (Walk-Forward 4-5窗口 × 7个模型/target × 4 targets)

- [ ] **Step 2: 验证训练产物**

```bash
ls -la ml_models/trained_models/v490/v490_*.pkl
python3 -c "
import joblib
d = joblib.load('ml_models/trained_models/v490/v490_multi_target_*.pkl')
for t in ['3d','5d','10d','15d']:
    models = list(d['models'][t]['models'].keys()) if t in d['models'] else []
    print(f'{t}: {models}')
print(f'v490_innovations: {d.get(\"v490_innovations\")}')
"
```

预期输出: 每个target有7个模型 (lgb, xgb, cb, rf, hgb, lgb_rank, lgb_q95)

---

### Task 4: 回测V4.9.0 vs V4.8.5基线

- [ ] **Step 1: 生成V4.9.0批量报告**

```bash
python3 backtest/batch_generate_v395_reports.py \
    --version v4.9.0 --start-date 2024-01-01 --end-date 2026-03-27 \
    --output-dir reports/daily_selection_v4.9.0
```

- [ ] **Step 2: 运行北极星V4评估**

```bash
# V4.9.0 composite排名
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_v4.9.0 \
    --label "V4.9.0-Comp" --top-n 10 --focus-days 10

# V4.9.0 head_rank排名 (Q95 Widen-then-Concentrate)
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_v4.9.0 \
    --rank-field head_rank \
    --label "V4.9.0-Q95" --top-n 10 --focus-days 10
```

- [ ] **Step 3: 对比V4.8.5基线**

对比维度:
- V3/V4北极星总分
- 10d/月调仓 Sharpe
- 近2月弱市表现
- 换手率变化

---

### Task 5: 校准全局分位数 + 阈值

- [ ] **Step 1: 校准global_quantiles**

```bash
python3 ml_models/calibrate_global_quantiles.py --version v4.9.0
```

- [ ] **Step 2: 优化推荐阈值**

使用现有阈值优化脚本:
```bash
# 构建回测数据
python3 scripts/build_v487_backtest_pkl.py  # 改路径为v490

# 网格搜索
python3 scripts/autoresearch_v487_thresholds.py  # 改路径为v490
```

---
