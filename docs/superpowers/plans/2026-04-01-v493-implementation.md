# V4.9.3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Iterate V4901→V4.9.3 via feature engineering (delete 13 useless + add 3 BRAIN) and concentration risk mitigation (feature_fraction_bynode + ensemble weight capping), targeting V5.1 ≥82%.

**Architecture:** New V493Trainer inherits V4901Trainer, overrides PRUNE_FEATURES (adds 13), adds 3 BRAIN factor computation in prepare_features, overrides LGB/XGB params for bynode sampling, and clips ensemble weights post-optimization. New V493 scorer inherits V4901 scorer with model dir pointed to v493.

**Tech Stack:** Python 3, LightGBM, XGBoost, CatBoost, NumPy, Pandas

**Spec:** `docs/superpowers/specs/2026-04-01-v493-iteration-design.md`

---

### Task 1: Create V493Trainer — Feature Pruning + BRAIN Factors

**Files:**
- Modify: `ml_models/training/train_v395_multi_target.py` (insert V493Trainer after V4902Trainer, before V486Trainer)

- [ ] **Step 1: Insert V493Trainer class**

Find V4902Trainer class end (search for `class V486Trainer` which starts after V4902). Insert V493Trainer between V4902Trainer and V486Trainer.

```python
class V493Trainer(V4901Trainer):
    """V4.9.3 训练器 — 纯特征工程 + 浓度风险对策

    基于V4901 (不改Sharpe-blend/样本权重/ensemble方法):
    1. 删除13个确认无效特征 (61→48)
    2. 添加3个BRAIN因子 (48→51)
    3. feature_fraction_bynode=0.5 (防单特征浓度)
    4. ensemble权重上限0.35 + ICIR收缩0.3 (防单模型浓度)
    5. min_data_in_leaf=400 (减少细粒度过拟合)
    """

    # V4.9.3 额外裁剪特征 (在V4.7.5的PRUNE_FEATURES基础上)
    V493_EXTRA_PRUNE = [
        'dv_ttm', 'max_pct_change_5d', 'cci_14', 'macd_hist',
        'brain_roll_spread', 'vol_price_div', 'return_skewness_proxy',
        'return_10d', 'ma10_ratio', 'return_1d',
        'price_acceleration', 'max_ret_20d', 'avg_pct_change_5d',
    ]

    # V4.9.3 新增BRAIN因子
    V493_BRAIN_FACTORS = [
        'brain_vol_clustering',
        'brain_tail_risk',
        'brain_ret_autocorr',
    ]

    # 浓度对策: ensemble权重上限
    ENSEMBLE_CLIP_MAX = 0.35
    ENSEMBLE_CLIP_MIN = 0.10
    ICIR_SHRINKAGE = 0.3  # 向等权收缩30%

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """V4.9.3: V4901特征 + 额外裁剪13个 + 新增3个BRAIN因子"""
        # 先用V4901/V485的prepare_features (得到61特征)
        X, y_3d, y_5d, y_10d, y_15d, df_out = super().prepare_features(df)

        # 额外裁剪13个无效特征
        if self.feature_names:
            keep_indices = []
            pruned = []
            for i, name in enumerate(self.feature_names):
                if name in self.V493_EXTRA_PRUNE:
                    pruned.append(name)
                else:
                    keep_indices.append(i)

            if pruned:
                X = X[:, keep_indices]
                self.feature_names = [self.feature_names[i] for i in keep_indices]
                logger.info(f"  V4.9.3 额外裁剪: {len(pruned)} 特征 → {len(self.feature_names)} 剩余")
                logger.info(f"    裁剪: {pruned[:5]}...")

        # 新增3个BRAIN因子
        n_before = X.shape[1]
        brain_features = self._compute_brain_factors(df_out)
        if brain_features is not None and brain_features.shape[1] > 0:
            X = np.hstack([X, brain_features])
            for name in self.V493_BRAIN_FACTORS:
                if name not in self.feature_names:
                    self.feature_names.append(name)
            logger.info(f"  V4.9.3 BRAIN因子: +{X.shape[1] - n_before} → {len(self.feature_names)} 总特征")

        return X, y_3d, y_5d, y_10d, y_15d, df_out

    def _compute_brain_factors(self, df: pd.DataFrame) -> np.ndarray:
        """计算3个BRAIN因子, 返回numpy array (n_samples, 3)"""
        import warnings
        results = []

        # brain_vol_clustering: ewm_vol_5d / rolling_vol_20d
        # 需要按股票分组计算
        try:
            if 'price_change_pct' in df.columns and 'code' in df.columns:
                vol_clustering = np.full(len(df), np.nan)
                for code, group in df.groupby('code'):
                    idx = group.index
                    ret = group['price_change_pct']
                    ewm_vol = ret.ewm(span=5, min_periods=3).std()
                    roll_vol = ret.rolling(20, min_periods=10).std()
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        ratio = ewm_vol / roll_vol
                    vol_clustering[idx] = ratio.values
                results.append(vol_clustering)
            else:
                # fallback: 从已有特征估算
                if 'volatility_10d' in df.columns and 'volatility_20d' in df.columns:
                    ratio = df['volatility_10d'].values / np.maximum(df['volatility_20d'].values, 1e-8)
                    results.append(ratio)
                else:
                    results.append(np.zeros(len(df)))
        except Exception as e:
            logger.warning(f"  brain_vol_clustering failed: {e}")
            results.append(np.zeros(len(df)))

        # brain_tail_risk: 过去20日|收益|>2σ的天数占比
        try:
            if 'price_change_pct' in df.columns and 'code' in df.columns:
                tail_risk = np.full(len(df), np.nan)
                for code, group in df.groupby('code'):
                    idx = group.index
                    ret = group['price_change_pct']
                    sigma = ret.rolling(60, min_periods=20).std()
                    extreme = (ret.abs() > 2 * sigma).astype(float)
                    tail_risk[idx] = extreme.rolling(20, min_periods=10).mean().values
                results.append(tail_risk)
            else:
                results.append(np.zeros(len(df)))
        except Exception as e:
            logger.warning(f"  brain_tail_risk failed: {e}")
            results.append(np.zeros(len(df)))

        # brain_ret_autocorr: 过去20日收益的1阶自相关
        try:
            if 'price_change_pct' in df.columns and 'code' in df.columns:
                ret_autocorr = np.full(len(df), np.nan)
                for code, group in df.groupby('code'):
                    idx = group.index
                    ret = group['price_change_pct']
                    autocorr = ret.rolling(20, min_periods=10).apply(
                        lambda x: x.autocorr(lag=1) if len(x) >= 10 else 0, raw=False)
                    ret_autocorr[idx] = autocorr.values
                results.append(ret_autocorr)
            else:
                results.append(np.zeros(len(df)))
        except Exception as e:
            logger.warning(f"  brain_ret_autocorr failed: {e}")
            results.append(np.zeros(len(df)))

        brain_arr = np.column_stack(results)
        # NaN → 0 (cross-sectional normalization will handle later)
        brain_arr = np.nan_to_num(brain_arr, nan=0.0)
        return brain_arr

    def train_single_target_models(self, X_train, X_val, y_train, y_val,
                                    target_name: str, sample_weights_train=None):
        """V4.9.3: 覆盖LGB/XGB参数 + 浓度对策"""
        # 临时覆盖LGB参数
        import lightgbm as lgb

        # 保存原始参数
        original_lgb_ff = None
        original_lgb_ffbn = None

        # 注入bynode参数 (通过monkey-patch lgb_params)
        # V4.9.3的参数覆盖在_get_lgb_params中处理
        self._v493_params_active = True

        # 调用父类训练
        models, pred_train, pred_val = super().train_single_target_models(
            X_train, X_val, y_train, y_val, target_name,
            sample_weights_train=sample_weights_train)

        self._v493_params_active = False

        # 应用ensemble权重上限 + ICIR收缩
        if isinstance(models, dict):
            self._clip_ensemble_weights(models, target_name)

        return models, pred_train, pred_val

    def _clip_ensemble_weights(self, models: dict, target_name: str):
        """对模型权重做clip + shrinkage"""
        # models是 {model_name: model_obj} 或包含weights的结构
        # 实际权重在 _compute_val_ic_weights 中计算, 存在 result['weights']
        # 这里在训练后处理, 通过覆盖 _compute_val_ic_weights 实现
        pass  # 实际通过覆盖 _compute_val_ic_weights 实现

    def _compute_val_ic_weights(self, predictions_val, y_val_target, val_dates=None):
        """V4.9.3: 父类IC+单调性权重 + clip[0.10, 0.35] + shrinkage 0.3"""
        weights, mean_ics = super()._compute_val_ic_weights(
            predictions_val, y_val_target, val_dates)

        # 应用clip和shrinkage
        n = len(weights)
        if n < 2:
            return weights, mean_ics

        equal_w = 1.0 / n
        new_weights = {}
        for k, v in weights.items():
            # shrinkage toward equal weight
            shrunk = self.ICIR_SHRINKAGE * equal_w + (1 - self.ICIR_SHRINKAGE) * v
            # clip
            clipped = max(self.ENSEMBLE_CLIP_MIN, min(self.ENSEMBLE_CLIP_MAX, shrunk))
            new_weights[k] = clipped

        # renormalize
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v / total for k, v in new_weights.items()}

        logger.info(f"  V4.9.3 权重clip+shrinkage: "
                    f"{', '.join(f'{k}={v:.3f}' for k, v in new_weights.items())}")
        return new_weights, mean_ics

    def _get_lgb_params(self, target_name: str = '') -> dict:
        """V4.9.3: 覆盖LGB参数 — bynode + feature_fraction + min_data"""
        params = super()._get_lgb_params(target_name) if hasattr(super(), '_get_lgb_params') else {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1,
            'path_smooth': 5,
            'min_data_in_leaf': 200,
        }
        # V4.9.3 覆盖
        params['feature_fraction'] = 0.7          # was 0.8
        params['feature_fraction_bynode'] = 0.5   # 新增
        params['min_data_in_leaf'] = 400          # was 200
        return params

    def walk_forward_train(self, start_date=None, end_date=None,
                            purge_days=15, min_train_days=900,
                            val_days=120, test_days=120, step_days=90):
        """V4.9.3 Walk-Forward — 保存为v493格式"""
        import shutil
        import json as _json

        version_tag = 'v493'
        version_str = 'v4.9.3'

        logger.info("=" * 60)
        logger.info(f"{version_str} Walk-Forward (特征工程 + 浓度风险对策)")
        logger.info("=" * 60)
        logger.info(f"  底座: V4.9.0.1 (61特征, Q95+trunc=10)")
        logger.info(f"  裁剪: -{len(self.V493_EXTRA_PRUNE)} 无效特征")
        logger.info(f"  新增: +{len(self.V493_BRAIN_FACTORS)} BRAIN因子")
        logger.info(f"  浓度对策: bynode=0.5, clip=[{self.ENSEMBLE_CLIP_MIN},{self.ENSEMBLE_CLIP_MAX}], "
                    f"shrinkage={self.ICIR_SHRINKAGE}")

        model_data, history = V485Trainer.walk_forward_train(
            self, start_date=start_date, end_date=end_date,
            purge_days=purge_days, min_train_days=min_train_days,
            val_days=val_days, test_days=test_days, step_days=step_days)

        if model_data.get('fast_check'):
            return model_data, history

        # 重命名 v485 → v493
        v485_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v485'
        out_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / version_tag
        out_dir.mkdir(parents=True, exist_ok=True)

        v485_files = sorted(v485_dir.glob('v485_*.pkl'), key=lambda f: f.stat().st_mtime)
        if v485_files:
            latest = v485_files[-1]
            timestamp = latest.stem.replace('v485_multi_target_', '')
            new_path = out_dir / f'{version_tag}_multi_target_{timestamp}.pkl'

            import joblib
            model_data['version'] = version_str
            model_data['v493_innovations'] = {
                'base': 'V4.9.0.1 (unchanged training config)',
                'feature_pruning': f'-{len(self.V493_EXTRA_PRUNE)} useless features',
                'brain_factors': self.V493_BRAIN_FACTORS,
                'concentration_fix': {
                    'feature_fraction_bynode': 0.5,
                    'ensemble_clip': [self.ENSEMBLE_CLIP_MIN, self.ENSEMBLE_CLIP_MAX],
                    'icir_shrinkage': self.ICIR_SHRINKAGE,
                    'min_data_in_leaf': 400,
                },
            }
            joblib.dump(model_data, new_path)
            logger.info(f"\n{version_str} model saved: {new_path}")

            for aux in ['global_quantiles.npy', 'recommendation_thresholds.json']:
                src = v485_dir / aux
                if src.exists():
                    shutil.copy2(str(src), str(out_dir / aux))

            latest.unlink()
            for hf in v485_dir.glob(f'training_history_{timestamp}*'):
                hf.unlink()

            history['version'] = version_str
            history_path = out_dir / f'training_history_{timestamp}.json'
            with open(history_path, 'w', encoding='utf-8') as f:
                _json.dump(history, f, indent=2, ensure_ascii=False)
            with open(out_dir / 'training_history_latest.json', 'w', encoding='utf-8') as f:
                _json.dump(history, f, indent=2, ensure_ascii=False)

        return model_data, history
```

- [ ] **Step 2: Add --v493 CLI flag**

Find the argparse section (search for `--v4902`). Add after it:

```python
parser.add_argument('--v493', action='store_true',
                    help='V4.9.3训练 (V4.9.0.1 + 特征裁剪 + 3 BRAIN因子 + 浓度对策)')
```

In the dispatch section (search for `args.v4902`), add after V4902's block:

```python
if args.v493:
    trainer = V493Trainer()
    _apply_overrides(trainer)
    trainer.walk_forward_train(
        start_date=args.start_date, end_date=args.end_date,
        purge_days=args.purge_days)
    return
```

- [ ] **Step 3: Handle LGB params override**

The V493Trainer needs to inject its params into the LGB training. Search for where LGB params are constructed in the training pipeline. The base class (V395MultiTargetTrainer) constructs params inline. V493 needs to override them.

Find the `lgb.train(` calls in the base trainer and check how params are passed. The `_get_lgb_params` method may not exist in the base class — if so, V493 needs to inject params differently.

The simplest approach: override `train_single_target_models` to temporarily patch the params dict before calling super().

Read the actual LGB training section in V485Trainer or the base trainer to find where params are defined, then patch accordingly.

- [ ] **Step 4: Verify class instantiation**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -c "
from ml_models.training.train_v395_multi_target import V493Trainer
t = V493Trainer()
print(f'V493 loaded, extra prune: {len(t.V493_EXTRA_PRUNE)}, brain: {len(t.V493_BRAIN_FACTORS)}')
print(f'Clip: [{t.ENSEMBLE_CLIP_MIN}, {t.ENSEMBLE_CLIP_MAX}], shrinkage: {t.ICIR_SHRINKAGE}')
"`

- [ ] **Step 5: Commit**

```bash
git add ml_models/training/train_v395_multi_target.py
git commit -m "feat: V4.9.3 Trainer — 删13特征+加3BRAIN+浓度对策(bynode/clip/shrinkage)"
```

---

### Task 2: Create V493 Production Scorer

**Files:**
- Create: `ml_models/v39/v493_production_scorer.py`
- Modify: `ml_models/v39/__init__.py` (add import)

- [ ] **Step 1: Create scorer**

```python
#!/usr/bin/env python3
"""
V4.9.3 production scorer — V4901底座 + V493自有模型

与V4901的区别:
  - 使用V493模型 (51特征: 61-13+3)
  - 继承V4901的全部scoring逻辑 (composite排序等)
"""

import logging
from pathlib import Path
from typing import Dict, List

from .v4901_production_scorer import V4901ProductionScorer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class V493ProductionScorer(V4901ProductionScorer):
    """V4.9.3 scorer — V493模型, V4901 scoring逻辑"""

    def __init__(self, model_type: str = 'small_data'):
        self._v493_model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v493'
        super().__init__(model_type=model_type)

    def _load_models(self):
        """加载v493模型, fallback到v4901"""
        v493_files = list(self._v493_model_dir.glob('v493_*.pkl'))
        if v493_files:
            self.model_dir = self._v493_model_dir
            latest = max(v493_files, key=lambda f: f.stat().st_mtime)
            self._load_model_from_file(latest, label='V4.9.3')
            has_q95 = 'lgb_q95' in self.models.get('10d', {})
            logger.info(f"  Q95 in ensemble: {has_q95}")
            return
        logger.warning("  V493模型未找到, fallback到V4901")
        super()._load_models()
```

- [ ] **Step 2: Add to __init__.py**

```python
from .v493_production_scorer import V493ProductionScorer
```

- [ ] **Step 3: Register in batch_generate**

In `backtest/batch_generate_v395_reports.py`, add `'v4.9.3'` to the version choices list and add the scorer dispatch:

```python
elif args.version == 'v4.9.3':
    from ml_models.v39.v493_production_scorer import V493ProductionScorer
    scorer = V493ProductionScorer(model_type='small_data')
```

Also add `'v4.9.3'` to the `if version in (...)` check for V485-derived scorers.

- [ ] **Step 4: Smoke test**

Run: `python3 -c "from ml_models.v39.v493_production_scorer import V493ProductionScorer; print('OK')"`

- [ ] **Step 5: Commit**

```bash
git add ml_models/v39/v493_production_scorer.py ml_models/v39/__init__.py backtest/batch_generate_v395_reports.py
git commit -m "feat: V4.9.3 scorer + batch_generate注册"
```

---

### Task 3: Fast-Check Validation

- [ ] **Step 1: Run fast-check**

```bash
cd /Users/yangxu/StockTradebyZ && python3 ml_models/training/train_v395_multi_target.py \
    --v493 --fast-check --start-date 2019-01-01 2>&1 | \
    grep -E 'IC=.*ICIR|裁剪|BRAIN|浓度|权重clip|Walk-Forward 汇总|FAST-CHECK' | tail -30
```

Expected:
- "V4.9.3 额外裁剪: 13 特征 → 48 剩余"
- "V4.9.3 BRAIN因子: +3 → 51 总特征"
- 10d IC > 0.04, ICIR > 0.5
- 权重clip: 每个模型 ≤ 0.35

If 10d IC < 0.03 or ICIR < 0.3 → STOP, investigate which change caused degradation.

- [ ] **Step 2: Compare with V4901 baseline**

V4901 WF: 10d IC=0.045, ICIR=0.494 (from training history)
V4.9.3 should be within ±20% of these values.

---

### Task 4: Full Training

- [ ] **Step 1: Launch training**

```bash
mkdir -p logs && nohup python3 ml_models/training/train_v395_multi_target.py \
    --v493 --start-date 2020-01-01 --purge-days 15 \
    > logs/train_v493_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"
```

- [ ] **Step 2: Verify model saved**

After completion (~6h):
```bash
ls -la ml_models/trained_models/v493/
# Expected: v493_multi_target_*.pkl, global_quantiles.npy
```

---

### Task 5: Generate Reports + V5.1 Evaluation

- [ ] **Step 1: Generate reports**

```bash
python3 backtest/batch_generate_v395_reports.py \
    --version v4.9.3 --start-date 2024-01-01 --end-date 2026-03-31 \
    --output-dir reports/daily_selection_v493
```

- [ ] **Step 2: V5.1 evaluation (production config)**

```bash
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_v493 \
    --score-version v51 --top-n 10 --focus-days 15 --n-trials 50 \
    --cppi-floor 0.08 --cppi-multiplier 20 \
    --ema-alpha 0.7 --score-floor 30 --retention-bonus 0.2
```

- [ ] **Step 3: Compare V4901 vs V4.9.3**

```bash
echo "=== V4901 ===" && python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_v4901 --score-version v51 \
    --top-n 10 --focus-days 15 --cppi-floor 0.08 --cppi-multiplier 20 \
    --ema-alpha 0.7 --score-floor 30 --retention-bonus 0.2 \
    2>&1 | grep '加权评分' | tail -1 && \
echo "=== V4.9.3 ===" && python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_v493 --score-version v51 \
    --top-n 10 --focus-days 15 --cppi-floor 0.08 --cppi-multiplier 20 \
    --ema-alpha 0.7 --score-floor 30 --retention-bonus 0.2 \
    2>&1 | grep '加权评分' | tail -1
```

Expected: V4.9.3 > V4901 (82%+ vs 77-81%).

- [ ] **Step 4: Commit results**

```bash
git add -A && git commit -m "feat: V4.9.3完整迭代 — 51特征+浓度对策, V5.1目标82%+"
```
