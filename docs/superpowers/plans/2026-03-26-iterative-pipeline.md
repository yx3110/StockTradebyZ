# Iterative Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 4-level gated validation pipeline (L1→L2→L3→L4) that gives North Star score estimates in 5-7 minutes instead of 5-6 hours.

**Architecture:** Single orchestrator script (`scripts/iterative_pipeline.py`) delegates to a lightweight L1 trainer (`scripts/l1_fast_trainer.py`) for fast screening, then reuses existing batch report generation and North Star evaluation code for L2-L4. Gate conditions at each level auto-promote or stop.

**Tech Stack:** Python 3, LightGBM, pandas, scipy, joblib. Reuses existing `backtest/batch_generate_v395_reports.py`, `backtest/backtest_report_based.py`, `backtest/north_star_metrics.py`, and `ml_models/training/train_v395_multi_target.py`.

**Spec:** `docs/superpowers/specs/2026-03-26-iterative-pipeline-design.md`

---

## File Structure

```
scripts/
├── l1_fast_trainer.py              # NEW: L1 lightweight trainer (~300 lines)
│   └── L1FastTrainer class: load data, single-fold split, LGB train, IC eval
│
├── iterative_pipeline.py           # NEW: Main orchestrator + CLI (~450 lines)
│   ├── run_l1(params) → l1_result
│   ├── run_l2(params, l1_result) → l2_result
│   ├── run_l3(params) → l3_result
│   ├── run_l4(params) → l4_result
│   ├── run_auto_gate(params) → final_result
│   ├── run_batch(param_files, promote_top) → comparison
│   ├── append_comparison(result) → writes TSV
│   └── main() with argparse CLI
│
├── params/                         # NEW: directory for param variants
│   └── example_params.json         # NEW: example param file
│
├── iteration_comparison.tsv        # AUTO-GENERATED: comparison table
└── l2_calibration.json             # AUTO-GENERATED: calibration data

backtest/
├── backtest_report_based.py        # MODIFY: add compute_ns_scores() helper
└── (other files unchanged)

tests/
└── test_iterative_pipeline.py      # NEW: pipeline tests (~150 lines)
```

**Key dependency:** `backtest/backtest_report_based.py` currently prints NS scores to console but doesn't return them. We need a small helper function that calls `north_star_metrics.compute_v3_score()` on the backtest summary and returns the structured result.

---

### Task 1: Extract NS scoring into a callable function

Currently `run_single_backtest()` in `backtest/backtest_report_based.py` prints V2/V3 scorecards but doesn't return them. We need a function that computes and returns scores programmatically.

**Files:**
- Modify: `backtest/backtest_report_based.py` (add helper function near the scorecard printing code, ~lines 1620-1660)
- Reference: `backtest/north_star_metrics.py` (for `compute_v3_score` signature)

- [ ] **Step 1: Read the scorecard printing code to understand the metric keys**

Read `backtest/backtest_report_based.py` lines 1600-1660 to see how `_print_scorecard_v2()` and `_print_scorecard_v3()` are called and what metrics they use.

- [ ] **Step 2: Add `compute_ns_scores()` function**

Add this function right before the `run_single_backtest` return statement area:

```python
def compute_ns_scores(summary: dict, focus_days: int, n_trading_days: int = 0,
                      n_trials: int = 1, benchmark_code: str = '000905.SH') -> dict:
    """Compute North Star V2/V3 scores from backtest summary.

    Args:
        summary: {days: {metric: value}} from run_single_backtest
        focus_days: which holding period to score (e.g. 10)
        n_trading_days: total trading days in backtest (for V3 length discount)
        n_trials: number of strategy variants tested (for V3 DSR)
        benchmark_code: benchmark index code

    Returns:
        {
            'v2_score': int,        # 0-105
            'v2_pct': float,        # 0-100%
            'v2_grade': str,        # S/A+/A/B/C/D
            'v3_score': int,        # 0-125
            'v3_pct': float,        # 0-100% (after length discount)
            'v3_grade': str,        # S/A+/A/B/C/D
            'v3_details': dict,     # layer breakdown
        }
    """
    from backtest import north_star_metrics as nsm

    metrics = summary.get(focus_days, {})
    if not metrics:
        return {'v2_score': 0, 'v2_pct': 0, 'v2_grade': 'D',
                'v3_score': 0, 'v3_pct': 0, 'v3_grade': 'D', 'v3_details': {}}

    # V2: simple sum scoring
    v2_total = 0
    v2_max = 0
    for metric_name, target_info in nsm.V2_METRICS.items():
        val = metrics.get(metric_name, 0)
        score, _ = nsm.score_metric_v2(val, target_info)
        v2_total += score
        v2_max += 5
    v2_pct = v2_total / v2_max * 100 if v2_max > 0 else 0
    v2_grade = nsm.compute_v2_grade(v2_total, v2_max)

    # V3: weighted layer scoring with length discount
    if n_trading_days == 0:
        n_trading_days = len(set(metrics.get('_dates', []))) or 100
    v3_result = nsm.compute_v3_score(metrics, n_trading_days=n_trading_days,
                                      n_trials=n_trials)

    return {
        'v2_score': v2_total,
        'v2_pct': round(v2_pct, 1),
        'v2_grade': v2_grade,
        'v3_score': v3_result.get('total_score', 0),
        'v3_pct': round(v3_result.get('final_pct', 0), 1),
        'v3_grade': v3_result.get('grade', 'D'),
        'v3_details': v3_result.get('layer_details', {}),
    }
```

Note: The exact metric dict keys (`V2_METRICS`, `score_metric_v2`, `compute_v2_grade`, `compute_v3_score`) may have slightly different names in the actual `north_star_metrics.py`. Adapt to match the real names found in step 1.

- [ ] **Step 3: Verify the function works**

Run a quick smoke test using an existing backtest result:

```bash
cd /Users/yangxu/StockTradebyZ
python3 -c "
from backtest.run_north_star_eval import run_backtest
result = run_backtest(
    report_dir='reports/daily_selection_v3.9',
    label='test', top_n=10, focus_days=10
)
from backtest.backtest_report_based import compute_ns_scores
ns = compute_ns_scores(result['summary'], focus_days=10, n_trading_days=100)
print(f'V2: {ns[\"v2_score\"]}/{ns[\"v2_pct\"]}% {ns[\"v2_grade\"]}')
print(f'V3: {ns[\"v3_score\"]}/{ns[\"v3_pct\"]}% {ns[\"v3_grade\"]}')
"
```

Expected: V2 and V3 scores printed without error. Scores should roughly match what the console scorecard would show.

- [ ] **Step 4: Commit**

```bash
git add backtest/backtest_report_based.py
git commit -m "refactor: 提取compute_ns_scores()函数供管线调用"
```

---

### Task 2: L1 Fast Trainer - Data Loading & Split

**Files:**
- Create: `scripts/l1_fast_trainer.py`

- [ ] **Step 1: Write test for data loading and split**

Create `tests/test_iterative_pipeline.py`:

```python
"""Tests for iterative pipeline components."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np


class TestL1FastTrainer:
    """Test L1 fast trainer functionality."""

    def test_load_data_returns_dataframe(self):
        """L1 trainer loads data from v39_feature_cache with expected columns."""
        from scripts.l1_fast_trainer import L1FastTrainer

        params = {
            'variant_name': 'test_load',
            'training': {'l1_start_date': '20250101'},
        }
        trainer = L1FastTrainer(params)
        df = trainer._load_data()

        assert len(df) > 0, "Should load some data"
        assert 'code' in df.columns
        assert 'trade_date' in df.columns
        assert 'label_5d' in df.columns
        assert 'label_10d' in df.columns
        assert len(trainer.feature_cols) > 10, "Should have 10+ features"

    def test_split_data_no_leakage(self):
        """Train/val/test splits have no overlapping dates, with purge gap."""
        from scripts.l1_fast_trainer import L1FastTrainer

        params = {
            'variant_name': 'test_split',
            'training': {'l1_start_date': '20250101', 'purge_days': 10},
        }
        trainer = L1FastTrainer(params)
        df = trainer._load_data()
        train_df, val_df, test_df = trainer._split_data(df)

        train_dates = set(train_df['trade_date'].unique())
        val_dates = set(val_df['trade_date'].unique())
        test_dates = set(test_df['trade_date'].unique())

        assert len(train_dates & val_dates) == 0, "Train/val dates must not overlap"
        assert len(val_dates & test_dates) == 0, "Val/test dates must not overlap"
        assert len(train_dates & test_dates) == 0, "Train/test dates must not overlap"
        assert max(train_dates) < min(val_dates), "Train must come before val"
        assert max(val_dates) < min(test_dates), "Val must come before test"
        assert len(train_df) > len(val_df), "Train should be larger than val"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/yangxu/StockTradebyZ && python3 -m pytest tests/test_iterative_pipeline.py::TestL1FastTrainer::test_load_data_returns_dataframe -v
```

Expected: `ModuleNotFoundError: No module named 'scripts.l1_fast_trainer'`

- [ ] **Step 3: Implement L1FastTrainer data loading and split**

Create `scripts/l1_fast_trainer.py`:

```python
"""L1 Fast Trainer: 单折 LGB 快速筛选训练器.

用途: 3-5分钟内评估一组参数/特征是否值得深入训练。
配置: 2年数据, LGB only, 150 rounds, 单折 70/15/15 split.
"""

import json
import time
import sqlite3
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / 'data_adapter' / 'stock_data.db')


class L1FastTrainer:
    """L1 快筛训练器."""

    # 默认 L1 门控阈值
    L1_GATE = {
        'test_ic_10d': 0.04,
        'test_icir_10d': 0.40,
        'train_val_gap': 0.05,
    }

    def __init__(self, params: dict):
        self.params = params
        self.variant_name = params.get('variant_name', 'unnamed')
        self.targets = ['label_5d', 'label_10d']
        self.models = {}
        self.feature_cols = []
        self._train_ic = {}

    def _load_data(self) -> pd.DataFrame:
        """Load data from v39_feature_cache, parse features, return flat DataFrame."""
        cfg = self.params.get('training', {})
        start_date = cfg.get('l1_start_date', None)
        if start_date is None:
            start_date = (pd.Timestamp.now() - pd.DateOffset(years=2)).strftime('%Y%m%d')

        conn = sqlite3.connect(DB_PATH, timeout=30)
        df = pd.read_sql_query(
            """SELECT code, trade_date, features_json,
                      label_5d, label_10d,
                      market_return_20d, market_return_10d, market_return_5d,
                      market_volatility_20d, market_volatility_10d,
                      market_up_ratio_20d, market_up_ratio_10d,
                      market_drawdown_20d, market_volume_ratio,
                      market_position_20d, market_momentum_20d, market_momentum_5d
               FROM v39_feature_cache
               WHERE trade_date >= ?
               ORDER BY trade_date, code""",
            conn, params=[start_date])
        conn.close()

        if len(df) == 0:
            raise ValueError(f"No data found in v39_feature_cache after {start_date}")

        # Parse features JSON (vectorized)
        parsed = df['features_json'].apply(json.loads)
        features = pd.DataFrame(parsed.tolist())

        # Determine feature columns
        feature_remove = set(self.params.get('features', {}).get('remove', []))
        base_cols = [c for c in features.columns if c not in feature_remove]
        market_cols = [c for c in df.columns if c.startswith('market_')]
        self.feature_cols = base_cols + market_cols

        # Build result DataFrame
        result = pd.DataFrame({
            'code': df['code'].values,
            'trade_date': df['trade_date'].values,
            'label_5d': df['label_5d'].values,
            'label_10d': df['label_10d'].values,
        })
        for col in base_cols:
            result[col] = features[col].values
        for col in market_cols:
            result[col] = df[col].values

        return result

    def _split_data(self, df: pd.DataFrame):
        """Single fold 70/15/15 split with purge gap."""
        purge_days = self.params.get('training', {}).get('purge_days', 10)

        dates = sorted(df['trade_date'].unique())
        n = len(dates)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        train_dates = set(dates[:train_end])
        val_start = min(train_end + purge_days, n)
        val_dates = set(dates[val_start:val_end])
        test_start = min(val_end + purge_days, n)
        test_dates = set(dates[test_start:])

        train_df = df[df['trade_date'].isin(train_dates)].copy()
        val_df = df[df['trade_date'].isin(val_dates)].copy()
        test_df = df[df['trade_date'].isin(test_dates)].copy()

        return train_df, val_df, test_df
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/yangxu/StockTradebyZ && python3 -m pytest tests/test_iterative_pipeline.py::TestL1FastTrainer -v
```

Expected: Both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/l1_fast_trainer.py tests/test_iterative_pipeline.py
git commit -m "feat: L1 fast trainer - 数据加载和分折"
```

---

### Task 3: L1 Fast Trainer - Training, Evaluation & Gate

**Files:**
- Modify: `scripts/l1_fast_trainer.py` (add training, normalize, evaluate, gate methods)
- Modify: `tests/test_iterative_pipeline.py` (add training tests)

- [ ] **Step 1: Add test for full L1 training**

Append to `tests/test_iterative_pipeline.py`:

```python
    def test_train_returns_valid_result(self):
        """Full L1 training returns structured result with metrics and gate decision."""
        from scripts.l1_fast_trainer import L1FastTrainer

        params = {
            'variant_name': 'test_full',
            'training': {
                'l1_start_date': '20250101',
                'l1_num_boost_round': 50,  # minimal for speed
                'purge_days': 10,
            },
        }
        trainer = L1FastTrainer(params)
        result = trainer.train()

        assert result['level'] == 'L1'
        assert result['variant_name'] == 'test_full'
        assert 'duration_sec' in result
        assert 'metrics' in result
        assert 'gate_pass' in result
        assert 'model_path' in result

        m = result['metrics']
        assert 'test_ic_5d' in m
        assert 'test_ic_10d' in m
        assert 'test_icir_10d' in m
        assert 'n_features' in m
        assert isinstance(m['top10_feature_importance'], list)
        assert len(m['top10_feature_importance']) <= 10
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_iterative_pipeline.py::TestL1FastTrainer::test_train_returns_valid_result -v
```

Expected: `AttributeError: 'L1FastTrainer' object has no attribute 'train'`

- [ ] **Step 3: Add normalization, training, evaluation, and gate methods**

Append to `scripts/l1_fast_trainer.py` inside the `L1FastTrainer` class:

```python
    def train(self) -> dict:
        """Run full L1 fast training pipeline. Returns structured result."""
        t0 = time.time()
        print(f"[L1] 开始快筛训练: {self.variant_name}")

        # 1. Load & split
        df = self._load_data()
        train_df, val_df, test_df = self._split_data(df)
        print(f"[L1] 数据: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, "
              f"features={len(self.feature_cols)}")

        # 2. Normalize (cross-sectional robust z-score)
        train_df = self._normalize(train_df)
        val_df = self._normalize(val_df)
        test_df = self._normalize(test_df)

        # 3. Train LGB for each target
        for target in self.targets:
            print(f"[L1] 训练 {target}...")
            self.models[target] = self._train_lgb(train_df, val_df, target)

        # 4. Evaluate
        metrics = self._evaluate(train_df, val_df, test_df)

        # 5. Save model
        model_path = self._save_model()

        duration = time.time() - t0
        gate_pass = self._check_gate(metrics)
        status = "PASS ✓" if gate_pass else "FAIL ✗"
        print(f"[L1] 完成 ({duration:.0f}s) | IC_10d={metrics['test_ic_10d']:.4f} "
              f"ICIR_10d={metrics['test_icir_10d']:.3f} | {status}")

        return {
            'variant_name': self.variant_name,
            'level': 'L1',
            'duration_sec': round(duration, 1),
            'metrics': metrics,
            'gate_pass': gate_pass,
            'model_path': model_path,
            'feature_cols': self.feature_cols,
        }

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional robust z-score normalization (vectorized)."""
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
                continue
            grouped = df.groupby('trade_date')[col]
            medians = grouped.transform('median')
            mads = grouped.transform(lambda x: (x - x.median()).abs().median()) * 1.4826
            mads = mads.replace(0, np.nan).fillna(1)
            df[col] = ((df[col] - medians) / mads).clip(-5, 5).fillna(0)
        return df

    def _train_lgb(self, train_df, val_df, target):
        """Train a single LightGBM model."""
        cfg = self.params.get('training', {})

        X_train = train_df[self.feature_cols].fillna(0).values
        y_train = train_df[target].values
        X_val = val_df[self.feature_cols].fillna(0).values
        y_val = val_df[target].values

        # Remove NaN labels
        train_mask = ~np.isnan(y_train)
        val_mask = ~np.isnan(y_val)

        train_data = lgb.Dataset(X_train[train_mask], y_train[train_mask],
                                 feature_name=self.feature_cols, free_raw_data=False)
        val_data = lgb.Dataset(X_val[val_mask], y_val[val_mask],
                               feature_name=self.feature_cols, reference=train_data,
                               free_raw_data=False)

        lgb_params = {
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': cfg.get('num_leaves', 31),
            'min_data_in_leaf': cfg.get('min_data_in_leaf', 200),
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42,
        }

        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=cfg.get('l1_num_boost_round', 150),
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
        )
        return model

    def _compute_daily_ic(self, df, target, pred_col):
        """Compute daily Spearman IC series."""
        daily_ics = []
        for date in df['trade_date'].unique():
            day = df[df['trade_date'] == date]
            y_true = day[target].values
            y_pred = day[pred_col].values
            valid = ~(np.isnan(y_true) | np.isnan(y_pred))
            if valid.sum() >= 30:
                ic, _ = spearmanr(y_true[valid], y_pred[valid])
                if not np.isnan(ic):
                    daily_ics.append(ic)
        return daily_ics

    def _evaluate(self, train_df, val_df, test_df):
        """Compute IC/ICIR metrics on train, val, and test sets."""
        metrics = {}

        for target in self.targets:
            horizon = target.replace('label_', '')  # '5d' or '10d'
            pred_col = f'pred_{horizon}'

            # Predict on all sets
            for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
                X = split_df[self.feature_cols].fillna(0).values
                split_df[pred_col] = self.models[target].predict(X)
                ics = self._compute_daily_ic(split_df, target, pred_col)

                if ics:
                    ic_mean = np.mean(ics)
                    ic_std = np.std(ics)
                    icir = ic_mean / ic_std if ic_std > 1e-8 else 0
                else:
                    ic_mean = icir = 0

                metrics[f'{name}_ic_{horizon}'] = round(ic_mean, 4)
                metrics[f'{name}_icir_{horizon}'] = round(icir, 3)

        # Overfitting check: train IC - test IC
        metrics['train_val_gap'] = round(
            metrics.get('train_ic_10d', 0) - metrics.get('test_ic_10d', 0), 4)

        metrics['n_features'] = len(self.feature_cols)

        # Feature importance (10d model)
        imp = self.models['label_10d'].feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[-10:][::-1]
        metrics['top10_feature_importance'] = [self.feature_cols[i] for i in top_idx]

        return metrics

    def _save_model(self) -> str:
        """Save L1 model artifacts to temp file."""
        import joblib
        model_data = {
            'models': {t: self.models[t] for t in self.targets},
            'feature_cols': self.feature_cols,
            'variant_name': self.variant_name,
            'params': self.params,
            'timestamp': time.strftime('%Y%m%d_%H%M%S'),
        }
        path = f'/tmp/l1_{self.variant_name}_{int(time.time())}.pkl'
        joblib.dump(model_data, path)
        return path

    def _check_gate(self, metrics: dict) -> bool:
        """Check L1 gate conditions. All must pass."""
        for key, threshold in self.L1_GATE.items():
            val = metrics.get(key, 0)
            if key == 'train_val_gap':
                if val > threshold:
                    return False
            else:
                if val < threshold:
                    return False
        return True
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/test_iterative_pipeline.py::TestL1FastTrainer -v
```

Expected: All 3 tests PASS. `test_train_returns_valid_result` may take 1-3 minutes (actual L1 training on real data).

- [ ] **Step 5: Commit**

```bash
git add scripts/l1_fast_trainer.py tests/test_iterative_pipeline.py
git commit -m "feat: L1 fast trainer - 训练/评估/门控完整实现"
```

---

### Task 4: L2 Mini Report Generation + NS Evaluation

**Files:**
- Modify: `scripts/iterative_pipeline.py` (create with `run_l2()`)
- Modify: `tests/test_iterative_pipeline.py` (add L2 test)

- [ ] **Step 1: Add L2 test**

Append to `tests/test_iterative_pipeline.py`:

```python
class TestL2QuickEval:
    """Test L2 mini report generation and NS evaluation."""

    def test_run_l2_returns_mini_ns(self):
        """L2 generates mini reports and returns NS score."""
        from scripts.l1_fast_trainer import L1FastTrainer
        from scripts.iterative_pipeline import run_l2

        # First run L1 to get a model
        params = {
            'variant_name': 'test_l2',
            'training': {
                'l1_start_date': '20250601',
                'l1_num_boost_round': 30,
                'purge_days': 10,
            },
            'scoring': {
                'top_n': 10,
                'focus_days': 10,
                'rank_field': 'pred_10d',
            },
        }
        trainer = L1FastTrainer(params)
        l1_result = trainer.train()

        # Run L2
        l2_result = run_l2(params, l1_result, n_days=20)  # 20 days for fast test

        assert l2_result['level'] == 'L2'
        assert 'mini_ns_raw' in l2_result
        assert 'metrics' in l2_result
        assert 'gate_pass' in l2_result
        assert isinstance(l2_result['mini_ns_raw'], (int, float))
```

- [ ] **Step 2: Create `scripts/iterative_pipeline.py` with `run_l2()`**

```python
"""分级验证迭代管线.

用法:
    python3 scripts/iterative_pipeline.py --level L1 --params params.json
    python3 scripts/iterative_pipeline.py --auto-gate --params params.json
    python3 scripts/iterative_pipeline.py --batch --params a.json b.json --promote-top 2
"""

import os
import sys
import json
import time
import shutil
import tempfile
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.l1_fast_trainer import L1FastTrainer


# ========== L2 Gate Thresholds ==========
L2_GATE = {
    'mini_ns_raw': 40,    # min NS score (V2 raw)
    'ic_10d': 0.05,       # min IC
    'max_drawdown': -0.30, # max acceptable drawdown
}


def run_l1(params: dict) -> dict:
    """Run L1 fast screening."""
    trainer = L1FastTrainer(params)
    return trainer.train()


def run_l2(params: dict, l1_result: dict, n_days: int = 60) -> dict:
    """L2: Generate mini reports from L1 model, run mini North Star eval.

    Args:
        params: variant params dict
        l1_result: result from run_l1()
        n_days: number of recent trading days for mini reports (default 60)

    Returns:
        l2_result dict with mini_ns_raw, metrics, gate_pass
    """
    import joblib
    from backtest.batch_generate_v395_reports import (
        fast_preload_feature_cache, load_securities_info, get_trading_dates,
    )
    from backtest.run_north_star_eval import run_backtest
    from backtest.backtest_report_based import compute_ns_scores

    t0 = time.time()
    print(f"[L2] 开始快评: {params.get('variant_name', '?')}")

    # 1. Load L1 model
    model_data = joblib.load(l1_result['model_path'])
    models = model_data['models']
    feature_cols = model_data['feature_cols']

    # 2. Get recent trading dates
    all_dates = get_trading_dates('auto', 'auto')
    dates = all_dates[-n_days:]
    print(f"[L2] 生成 {len(dates)} 天迷你报告...")

    # 3. Preload features
    cache = fast_preload_feature_cache(dates)
    sec_info = load_securities_info()

    # 4. Generate reports to temp dir
    report_dir = tempfile.mkdtemp(prefix=f'l2_{params.get("variant_name", "x")}_')

    for date in dates:
        features_df = cache.get(date)
        if features_df is None or len(features_df) == 0:
            continue

        # Ensure all feature columns exist
        for col in feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0

        X = features_df[feature_cols].fillna(0).values
        pred_5d = models['label_5d'].predict(X)
        pred_10d = models['label_10d'].predict(X)

        # Cross-sectional ranking
        from scipy.stats import rankdata
        ranks_10d = rankdata(pred_10d)
        n = len(ranks_10d)
        scores = 30 + (ranks_10d - 1) / max(n - 1, 1) * 60

        stocks = []
        codes = features_df['code'].values
        for i in range(len(codes)):
            code = str(codes[i])
            info = sec_info.get(code, {})
            stocks.append({
                'stock_code': code,
                'stock_name': info.get('name', ''),
                'industry': info.get('industry', ''),
                'score': round(float(scores[i]), 2),
                'predicted_return_5d': round(float(pred_5d[i]), 6),
                'pred_5d': round(float(pred_5d[i]), 6),
                'pred_10d': round(float(pred_10d[i]), 6),
                'rank_score': round(float(ranks_10d[i] / n), 4),
                'strategies': ['ML_Score'],
                'analysis_date': date,
            })

        report = {
            'analysis_date': date,
            'scoring_version': 'l1_fast',
            'total_scored_stocks': len(stocks),
            'all_stocks_with_scores': stocks,
        }
        with open(os.path.join(report_dir, f'analysis_data_{date}.json'), 'w') as f:
            json.dump(report, f, ensure_ascii=False)

    # 5. Run backtest + NS evaluation
    scoring_cfg = params.get('scoring', {})
    top_n = scoring_cfg.get('top_n', 10)
    focus_days = scoring_cfg.get('focus_days', 10)
    rank_field = scoring_cfg.get('rank_field', 'pred_10d')

    bt_result = run_backtest(
        report_dir=report_dir,
        label=params.get('variant_name', 'L2'),
        top_n=top_n,
        focus_days=focus_days,
        rank_field=rank_field,
    )

    # 6. Compute NS scores
    ns = compute_ns_scores(bt_result['summary'], focus_days=focus_days,
                           n_trading_days=len(dates))

    # 7. Calibrate (if calibration data exists)
    mini_ns_raw = ns['v2_score']
    calibrated = _calibrate_ns(mini_ns_raw)

    # 8. Extract key metrics
    summary = bt_result['summary'].get(focus_days, {})
    metrics = {
        'ic_10d': round(summary.get('daily_ic', summary.get('ic', 0)), 4),
        'icir_10d': round(summary.get('icir', 0), 3),
        'annual_return_gross': round(summary.get('annual_return', 0), 4),
        'max_drawdown': round(summary.get('max_drawdown', 0), 4),
        'sharpe': round(summary.get('sharpe_ratio', 0), 3),
    }

    gate_pass = _check_l2_gate(mini_ns_raw, metrics)

    duration = time.time() - t0
    status = "PASS ✓" if gate_pass else "FAIL ✗"
    print(f"[L2] 完成 ({duration:.0f}s) | NS={mini_ns_raw}/{ns['v2_grade']} "
          f"IC={metrics['ic_10d']:.4f} | {status}")

    # Cleanup temp dir
    # (keep if gate_pass for potential debugging; clean otherwise)
    if not gate_pass:
        shutil.rmtree(report_dir, ignore_errors=True)
        report_dir = None

    return {
        'variant_name': params.get('variant_name'),
        'level': 'L2',
        'duration_sec': round(duration, 1),
        'mini_ns_raw': mini_ns_raw,
        'mini_ns_calibrated': calibrated,
        'mini_ns_grade': ns['v2_grade'],
        'metrics': metrics,
        'gate_pass': gate_pass,
        'reports_dir': report_dir,
    }


def _check_l2_gate(mini_ns: float, metrics: dict) -> bool:
    """Check L2 gate conditions."""
    if mini_ns < L2_GATE['mini_ns_raw']:
        return False
    if metrics.get('ic_10d', 0) < L2_GATE['ic_10d']:
        return False
    if metrics.get('max_drawdown', -1) < L2_GATE['max_drawdown']:
        return False
    return True


def _calibrate_ns(mini_ns_raw: float) -> float:
    """Apply L2 calibration if calibration data exists."""
    cal_path = Path(__file__).parent / 'l2_calibration.json'
    if not cal_path.exists():
        return float(mini_ns_raw)
    try:
        with open(cal_path) as f:
            cal = json.load(f)
        if len(cal.get('pairs', [])) < 5:
            return float(mini_ns_raw)
        slope = cal.get('slope', 1.0)
        intercept = cal.get('intercept', 0.0)
        return round(slope * mini_ns_raw + intercept, 1)
    except Exception:
        return float(mini_ns_raw)
```

- [ ] **Step 3: Run test**

```bash
python3 -m pytest tests/test_iterative_pipeline.py::TestL2QuickEval -v --timeout=300
```

Expected: PASS (may take 2-4 minutes due to L1 training + L2 report generation on real data).

- [ ] **Step 4: Commit**

```bash
git add scripts/iterative_pipeline.py tests/test_iterative_pipeline.py
git commit -m "feat: L2 快评 - 迷你报告生成+北极星评分"
```

---

### Task 5: L3/L4 Integration via Subprocess

L3 and L4 call existing training and evaluation scripts with different parameters. Use subprocess for isolation (avoids global state conflicts in the training code).

**Files:**
- Modify: `scripts/iterative_pipeline.py` (add `run_l3()`, `run_l4()`)

- [ ] **Step 1: Add `run_l3()` and `run_l4()`**

Append to `scripts/iterative_pipeline.py`:

```python
import subprocess


def _get_version_flag(params: dict) -> str:
    """Map base_version in params to training script CLI flag."""
    version_map = {
        'v395': '--v395', 'v3.95': '--v395',
        'v43': '--v43', 'v4.3': '--v43',
        'v44': '--v44', 'v4.4': '--v44',
        'v46': '--v46', 'v4.6': '--v46',
        'v473': '--v473', 'v4.7.3': '--v473',
        'v475': '--v475', 'v4.7.5': '--v475',
        'v486': '--v486', 'v4.8.6': '--v486',
    }
    base = params.get('base_version', 'v395')
    return version_map.get(base, '--v395')


def _get_report_version(params: dict) -> str:
    """Map base_version to batch_generate --version flag."""
    version_map = {
        'v395': 'v3.95', 'v3.95': 'v3.95',
        'v43': 'v4.3', 'v4.3': 'v4.3',
        'v44': 'v4.4', 'v4.4': 'v4.4',
        'v46': 'v4.6', 'v4.6': 'v4.6',
        'v473': 'v4.7.3', 'v4.7.3': 'v4.7.3',
        'v475': 'v4.7.5', 'v4.7.5': 'v4.7.5',
        'v486': 'v4.8.6', 'v4.8.6': 'v4.8.6',
    }
    base = params.get('base_version', 'v395')
    return version_map.get(base, 'v3.95')


def run_l3(params: dict) -> dict:
    """L3 确认: 3-fold WF, 3 models (LGB+XGB+CB), 300 rounds, 3 years, 200-day reports.

    Calls existing training and evaluation scripts via subprocess.
    """
    t0 = time.time()
    variant = params.get('variant_name', 'unnamed')
    print(f"[L3] 开始确认训练: {variant}")

    cfg = params.get('training', {})
    scoring_cfg = params.get('scoring', {})
    version_flag = _get_version_flag(params)
    report_version = _get_report_version(params)

    # Step 1: Train (3-fold WF, 3 years, 300 rounds)
    train_cmd = [
        sys.executable, 'ml_models/training/train_v395_multi_target.py',
        version_flag,
        '--start-date', cfg.get('l3_start_date', '2023-01-01'),
        '--purge-days', str(cfg.get('purge_days', 15)),
        '--sharpe-blend', str(cfg.get('sharpe_blend', 0.3)),
    ]
    print(f"[L3] 训练命令: {' '.join(train_cmd)}")
    result = subprocess.run(train_cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"[L3] 训练失败:\n{result.stderr[-500:]}")
        return {'variant_name': variant, 'level': 'L3', 'gate_pass': False,
                'error': 'training failed', 'duration_sec': time.time() - t0}

    # Step 2: Generate 200-day reports
    report_dir = f'reports/daily_selection_{report_version}_l3_{variant}'
    report_cmd = [
        sys.executable, 'backtest/batch_generate_v395_reports.py',
        '--version', report_version,
        '--output-dir', report_dir,
        '--start-date', cfg.get('l3_report_start', 'auto'),
        '--end-date', 'auto',
    ]
    print(f"[L3] 报告生成...")
    result = subprocess.run(report_cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"[L3] 报告生成失败:\n{result.stderr[-500:]}")
        return {'variant_name': variant, 'level': 'L3', 'gate_pass': False,
                'error': 'report generation failed', 'duration_sec': time.time() - t0}

    # Step 3: North Star evaluation
    from backtest.run_north_star_eval import run_backtest
    from backtest.backtest_report_based import compute_ns_scores

    top_n = scoring_cfg.get('top_n', 10)
    focus_days = scoring_cfg.get('focus_days', 10)
    rank_field = scoring_cfg.get('rank_field', 'pred_10d')

    bt_result = run_backtest(
        report_dir=os.path.join(PROJECT_ROOT, report_dir),
        label=variant,
        top_n=top_n, focus_days=focus_days,
        rank_field=rank_field,
    )
    ns = compute_ns_scores(bt_result['summary'], focus_days=focus_days,
                           n_trading_days=200)

    summary = bt_result['summary'].get(focus_days, {})
    duration = time.time() - t0
    gate_pass = ns['v2_score'] >= 60  # A级门槛

    status = "PASS ✓" if gate_pass else "FAIL ✗"
    print(f"[L3] 完成 ({duration/60:.0f}min) | NS={ns['v2_score']}/{ns['v2_grade']} | {status}")

    return {
        'variant_name': variant,
        'level': 'L3',
        'duration_sec': round(duration, 1),
        'ns_score': ns['v2_score'],
        'ns_grade': ns['v2_grade'],
        'ns_v3_pct': ns['v3_pct'],
        'metrics': {
            'ic_10d': round(summary.get('daily_ic', summary.get('ic', 0)), 4),
            'icir_10d': round(summary.get('icir', 0), 3),
            'annual_return_gross': round(summary.get('annual_return', 0), 4),
            'max_drawdown': round(summary.get('max_drawdown', 0), 4),
            'sharpe': round(summary.get('sharpe_ratio', 0), 3),
        },
        'gate_pass': gate_pass,
        'report_dir': report_dir,
    }


def run_l4(params: dict) -> dict:
    """L4 生产: Full training + full reports + full NS eval.

    Uses existing production pipeline, no modifications.
    """
    t0 = time.time()
    variant = params.get('variant_name', 'unnamed')
    print(f"[L4] 开始生产训练: {variant}")

    cfg = params.get('training', {})
    scoring_cfg = params.get('scoring', {})
    version_flag = _get_version_flag(params)
    report_version = _get_report_version(params)

    # Step 1: Full training
    train_cmd = [
        sys.executable, 'ml_models/training/train_v395_multi_target.py',
        version_flag,
        '--start-date', cfg.get('start_date', '2020-01-01'),
        '--purge-days', str(cfg.get('purge_days', 15)),
        '--sharpe-blend', str(cfg.get('sharpe_blend', 0.3)),
    ]
    print(f"[L4] 全量训练开始 (预计5-6小时)...")
    result = subprocess.run(train_cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"[L4] 训练失败:\n{result.stderr[-500:]}")
        return {'variant_name': variant, 'level': 'L4', 'gate_pass': False,
                'error': 'training failed', 'duration_sec': time.time() - t0}

    # Step 2: Full reports (500+ days)
    report_dir = f'reports/daily_selection_{report_version}_l4_{variant}'
    report_cmd = [
        sys.executable, 'backtest/batch_generate_v395_reports.py',
        '--version', report_version,
        '--output-dir', report_dir,
    ]
    print(f"[L4] 全量报告生成...")
    result = subprocess.run(report_cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)

    # Step 3: Full NS evaluation
    from backtest.run_north_star_eval import run_backtest
    from backtest.backtest_report_based import compute_ns_scores

    top_n = scoring_cfg.get('top_n', 10)
    focus_days = scoring_cfg.get('focus_days', 10)
    rank_field = scoring_cfg.get('rank_field', 'pred_10d')

    bt_result = run_backtest(
        report_dir=os.path.join(PROJECT_ROOT, report_dir),
        label=variant,
        top_n=top_n, focus_days=focus_days,
        rank_field=rank_field,
    )
    ns = compute_ns_scores(bt_result['summary'], focus_days=focus_days)

    summary = bt_result['summary'].get(focus_days, {})
    duration = time.time() - t0

    print(f"[L4] 完成 ({duration/3600:.1f}h) | NS={ns['v2_score']}/{ns['v2_grade']}")

    return {
        'variant_name': variant,
        'level': 'L4',
        'duration_sec': round(duration, 1),
        'ns_score': ns['v2_score'],
        'ns_grade': ns['v2_grade'],
        'ns_v3_pct': ns['v3_pct'],
        'metrics': {
            'ic_10d': round(summary.get('daily_ic', summary.get('ic', 0)), 4),
            'icir_10d': round(summary.get('icir', 0), 3),
            'annual_return_gross': round(summary.get('annual_return', 0), 4),
            'max_drawdown': round(summary.get('max_drawdown', 0), 4),
            'sharpe': round(summary.get('sharpe_ratio', 0), 3),
        },
        'gate_pass': True,  # L4 always "passes" - it's the final level
        'report_dir': report_dir,
    }
```

- [ ] **Step 2: Commit**

```bash
git add scripts/iterative_pipeline.py
git commit -m "feat: L3/L4 集成 - 调用现有训练和评估脚本"
```

---

### Task 6: Auto-Gate, Batch Mode & Comparison Table

**Files:**
- Modify: `scripts/iterative_pipeline.py` (add auto-gate, batch, comparison table)
- Create: `scripts/params/example_params.json`

- [ ] **Step 1: Add auto-gate and batch functions**

Append to `scripts/iterative_pipeline.py`:

```python
def run_auto_gate(params: dict, max_level: str = 'L4') -> dict:
    """Run L1→L2→L3→L4 with automatic gate promotion.

    Stops at first failed gate or max_level.
    Returns the result from the highest completed level.
    """
    levels = ['L1', 'L2', 'L3', 'L4']
    max_idx = levels.index(max_level)
    results = []

    # L1
    l1 = run_l1(params)
    results.append(l1)
    append_comparison(l1)
    if not l1['gate_pass'] or max_idx < 1:
        return {'final': l1, 'all_results': results}

    # L2
    l2 = run_l2(params, l1)
    results.append(l2)
    append_comparison(l2)
    if not l2['gate_pass'] or max_idx < 2:
        return {'final': l2, 'all_results': results}

    # L3
    l3 = run_l3(params)
    results.append(l3)
    append_comparison(l3)
    if not l3['gate_pass'] or max_idx < 3:
        return {'final': l3, 'all_results': results}

    # L4
    l4 = run_l4(params)
    results.append(l4)
    append_comparison(l4)

    # Update L2 calibration with (mini_ns, full_ns) pair
    _update_calibration(l2['mini_ns_raw'], l4['ns_score'], params.get('variant_name'))

    return {'final': l4, 'all_results': results}


def run_batch(param_files: list, promote_top: int = 2, max_level: str = 'L3') -> list:
    """Run multiple variants through L1+L2, promote top N to L3/L4.

    Args:
        param_files: list of param JSON file paths
        promote_top: how many top variants to promote beyond L2
        max_level: max level for promoted variants

    Returns:
        list of all results
    """
    all_results = []

    # Phase 1: L1+L2 for all variants
    l2_results = []
    for pf in param_files:
        with open(pf) as f:
            params = json.load(f)
        print(f"\n{'='*60}")
        print(f"[BATCH] 变体: {params.get('variant_name', pf)}")
        print(f"{'='*60}")

        l1 = run_l1(params)
        append_comparison(l1)
        all_results.append(l1)

        if l1['gate_pass']:
            l2 = run_l2(params, l1)
            append_comparison(l2)
            all_results.append(l2)
            if l2['gate_pass']:
                l2_results.append((params, l2))

    # Phase 2: Rank by mini_ns and promote top N
    l2_results.sort(key=lambda x: x[1].get('mini_ns_raw', 0), reverse=True)
    promoted = l2_results[:promote_top]

    for params, l2 in promoted:
        print(f"\n{'='*60}")
        print(f"[BATCH] 升级到 {max_level}: {params.get('variant_name')}"
              f" (mini_ns={l2['mini_ns_raw']})")
        print(f"{'='*60}")

        if max_level in ('L3', 'L4'):
            l3 = run_l3(params)
            append_comparison(l3)
            all_results.append(l3)

            if max_level == 'L4' and l3['gate_pass']:
                l4 = run_l4(params)
                append_comparison(l4)
                all_results.append(l4)

    return all_results


# ========== Comparison Table ==========

COMPARISON_FILE = Path(__file__).parent / 'iteration_comparison.tsv'

def append_comparison(result: dict):
    """Append a result to the comparison TSV file."""
    metrics = result.get('metrics', {})

    def fmt_duration(sec):
        if sec < 60:
            return f"{sec:.0f}s"
        elif sec < 3600:
            return f"{sec/60:.0f}m{sec%60:.0f}s"
        else:
            return f"{sec/3600:.1f}h"

    row = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M'),
        'variant': result.get('variant_name', '?'),
        'level': result.get('level', '?'),
        'duration': fmt_duration(result.get('duration_sec', 0)),
        'gate': 'PASS' if result.get('gate_pass') else 'FAIL',
        'ic_10d': f"{metrics.get('test_ic_10d', metrics.get('ic_10d', '')):.4f}"
                  if isinstance(metrics.get('test_ic_10d', metrics.get('ic_10d', '')), (int, float)) else '-',
        'icir_10d': f"{metrics.get('test_icir_10d', metrics.get('icir_10d', '')):.3f}"
                    if isinstance(metrics.get('test_icir_10d', metrics.get('icir_10d', '')), (int, float)) else '-',
        'mini_ns': str(result.get('mini_ns_raw', '-')),
        'ns_200d': str(result.get('ns_score', '-')) if result.get('level') == 'L3' else '-',
        'ns_full': str(result.get('ns_score', '-')) if result.get('level') == 'L4' else '-',
        'grade': result.get('mini_ns_grade', result.get('ns_grade', '-')),
    }

    header = '\t'.join(row.keys())
    line = '\t'.join(str(v) for v in row.values())

    write_header = not COMPARISON_FILE.exists()
    with open(COMPARISON_FILE, 'a') as f:
        if write_header:
            f.write(header + '\n')
        f.write(line + '\n')


# ========== L2 Calibration ==========

CALIBRATION_FILE = Path(__file__).parent / 'l2_calibration.json'

def _update_calibration(mini_ns: float, full_ns: float, variant: str):
    """Add a (mini_ns, full_ns) pair and refit linear calibration."""
    cal = {'pairs': [], 'slope': 1.0, 'intercept': 0.0, 'r_squared': 0.0}
    if CALIBRATION_FILE.exists():
        with open(CALIBRATION_FILE) as f:
            cal = json.load(f)

    cal['pairs'].append({
        'variant': variant,
        'mini_ns': mini_ns,
        'full_ns': full_ns,
        'timestamp': time.strftime('%Y-%m-%d'),
    })

    # Refit if enough pairs
    if len(cal['pairs']) >= 5:
        x = np.array([p['mini_ns'] for p in cal['pairs']])
        y = np.array([p['full_ns'] for p in cal['pairs']])
        if np.std(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            ss_res = np.sum((y - (slope * x + intercept)) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            cal['slope'] = round(float(slope), 4)
            cal['intercept'] = round(float(intercept), 2)
            cal['r_squared'] = round(float(r_sq), 3)

    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(cal, f, indent=2, ensure_ascii=False)
```

- [ ] **Step 2: Create example params file**

Create `scripts/params/example_params.json`:

```json
{
    "variant_name": "v488_example",
    "base_version": "v475",
    "features": {
        "add": [],
        "remove": []
    },
    "training": {
        "start_date": "2020-01-01",
        "l1_start_date": "20240101",
        "l1_num_boost_round": 150,
        "l3_start_date": "2023-01-01",
        "num_boost_round": 500,
        "num_leaves": 31,
        "min_data_in_leaf": 200,
        "sharpe_blend": 0.3,
        "purge_days": 15
    },
    "scoring": {
        "rank_field": "pred_10d",
        "top_n": 10,
        "focus_days": 10
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add scripts/iterative_pipeline.py scripts/params/example_params.json
git commit -m "feat: auto-gate + batch mode + 对比表 + L2校准"
```

---

### Task 7: CLI Entry Point

**Files:**
- Modify: `scripts/iterative_pipeline.py` (add `main()` with argparse)

- [ ] **Step 1: Add CLI main function**

Append to `scripts/iterative_pipeline.py`:

```python
def main():
    parser = argparse.ArgumentParser(
        description='分级验证迭代管线 (L1快筛→L2快评→L3确认→L4生产)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单级别快筛
  python3 scripts/iterative_pipeline.py --level L1 --params scripts/params/example.json

  # 自动升级 (推荐)
  python3 scripts/iterative_pipeline.py --auto-gate --params scripts/params/example.json

  # 批量对比，取最好2个进L3
  python3 scripts/iterative_pipeline.py --batch \\
      --params scripts/params/a.json scripts/params/b.json scripts/params/c.json \\
      --promote-top 2

  # 只跑到L2 (快速对比)
  python3 scripts/iterative_pipeline.py --auto-gate --params scripts/params/example.json --max-level L2
        """)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--level', choices=['L1', 'L2', 'L3', 'L4'],
                      help='运行单个级别')
    mode.add_argument('--auto-gate', action='store_true',
                      help='自动升级模式: L1→L2→L3→L4，不过门控自动停')
    mode.add_argument('--batch', action='store_true',
                      help='批量模式: 多个变体L1+L2快筛，选最优升级')

    parser.add_argument('--params', nargs='+', required=True,
                        help='参数文件路径 (JSON)')
    parser.add_argument('--max-level', choices=['L1', 'L2', 'L3', 'L4'], default='L4',
                        help='最高运行级别 (默认: L4)')
    parser.add_argument('--promote-top', type=int, default=2,
                        help='批量模式: L2后升级最好的N个变体 (默认: 2)')
    parser.add_argument('--l2-days', type=int, default=60,
                        help='L2迷你报告天数 (默认: 60)')

    args = parser.parse_args()

    # Load params
    if args.batch:
        results = run_batch(args.params, promote_top=args.promote_top,
                            max_level=args.max_level)
    elif args.auto_gate:
        with open(args.params[0]) as f:
            params = json.load(f)
        result = run_auto_gate(params, max_level=args.max_level)
        results = result['all_results']
    else:
        # Single level
        with open(args.params[0]) as f:
            params = json.load(f)

        if args.level == 'L1':
            r = run_l1(params)
            append_comparison(r)
        elif args.level == 'L2':
            l1 = run_l1(params)
            append_comparison(l1)
            if l1['gate_pass']:
                r = run_l2(params, l1, n_days=args.l2_days)
                append_comparison(r)
            else:
                print("[L2] 跳过: L1未通过门控")
        elif args.level == 'L3':
            r = run_l3(params)
            append_comparison(r)
        elif args.level == 'L4':
            r = run_l4(params)
            append_comparison(r)

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"对比表: {COMPARISON_FILE}")
    if COMPARISON_FILE.exists():
        print(COMPARISON_FILE.read_text())


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Test CLI help**

```bash
python3 scripts/iterative_pipeline.py --help
```

Expected: Help text with all options shown correctly.

- [ ] **Step 3: Test L1 via CLI**

```bash
python3 scripts/iterative_pipeline.py --level L1 --params scripts/params/example_params.json
```

Expected: L1 runs in 3-5 minutes, prints IC/ICIR metrics, writes to comparison table.

- [ ] **Step 4: Commit**

```bash
git add scripts/iterative_pipeline.py
git commit -m "feat: CLI入口 - 支持 --level/--auto-gate/--batch 三种模式"
```

---

### Task 8: End-to-End Integration Test

**Files:**
- Modify: `tests/test_iterative_pipeline.py`

- [ ] **Step 1: Add integration test for L1+L2 auto-gate**

Append to `tests/test_iterative_pipeline.py`:

```python
class TestAutoGate:
    """Integration test: auto-gate L1→L2 on real data."""

    def test_auto_gate_l1_l2(self):
        """Auto-gate runs L1+L2 and returns structured results."""
        from scripts.iterative_pipeline import run_auto_gate

        params = {
            'variant_name': 'integration_test',
            'training': {
                'l1_start_date': '20250601',
                'l1_num_boost_round': 30,
                'purge_days': 10,
            },
            'scoring': {
                'top_n': 10,
                'focus_days': 10,
                'rank_field': 'pred_10d',
            },
        }
        result = run_auto_gate(params, max_level='L2')

        assert 'final' in result
        assert 'all_results' in result
        assert len(result['all_results']) >= 1  # At least L1

        final = result['final']
        assert final['level'] in ('L1', 'L2')
        assert 'gate_pass' in final

        # Check comparison file was updated
        from scripts.iterative_pipeline import COMPARISON_FILE
        assert COMPARISON_FILE.exists()
        content = COMPARISON_FILE.read_text()
        assert 'integration_test' in content
```

- [ ] **Step 2: Run integration test**

```bash
python3 -m pytest tests/test_iterative_pipeline.py::TestAutoGate -v --timeout=600
```

Expected: PASS. Runs L1 (3-5 min with reduced boost) → L2 (2 min) → result.

- [ ] **Step 3: Clean up comparison table from test runs**

```bash
# Remove test entries
python3 -c "
from pathlib import Path
p = Path('scripts/iteration_comparison.tsv')
if p.exists():
    lines = p.read_text().splitlines()
    clean = [l for l in lines if 'integration_test' not in l and 'test_' not in l]
    p.write_text('\n'.join(clean) + '\n' if clean else '')
    print(f'Cleaned {len(lines) - len(clean)} test entries')
"
```

- [ ] **Step 4: Final commit**

```bash
git add tests/test_iterative_pipeline.py
git commit -m "test: L1+L2 auto-gate 端到端集成测试"
```

---

### Task 9: Documentation & Final Polish

**Files:**
- Modify: `scripts/iterative_pipeline.py` (module docstring update)
- Verify: all tests pass, CLI works

- [ ] **Step 1: Run full test suite**

```bash
python3 -m pytest tests/test_iterative_pipeline.py -v --timeout=600
```

Expected: All tests pass.

- [ ] **Step 2: Verify CLI end-to-end with real params**

```bash
# Quick L1+L2 run
python3 scripts/iterative_pipeline.py --auto-gate \
    --params scripts/params/example_params.json \
    --max-level L2

# Verify comparison table
cat scripts/iteration_comparison.tsv
```

Expected: Full L1+L2 pipeline completes in ~7 minutes, comparison table shows results.

- [ ] **Step 3: Commit final state**

```bash
git add -A
git commit -m "feat: 分级验证迭代管线完成 (L1快筛+L2快评+L3确认+L4生产)"
```

---

## Self-Review Checklist

- [x] **Spec coverage**: All spec sections mapped to tasks:
  - L1 快筛 → Task 2-3
  - L2 快评 → Task 4
  - L3/L4 → Task 5
  - Auto-gate + batch → Task 6
  - CLI → Task 7
  - Comparison table + calibration → Task 6
  - NS scoring extraction → Task 1 (prerequisite)

- [x] **Placeholder scan**: No TBD/TODO. All code blocks contain actual implementation.

- [x] **Type consistency**: Checked:
  - `l1_result['model_path']` used consistently in L1 and L2
  - `params` dict structure consistent across all functions
  - `compute_ns_scores()` signature matches usage in L2, L3, L4
  - `append_comparison()` called with same result structure everywhere

- [x] **Missing from spec**: `params.json` format defined in spec → covered by Task 6 `example_params.json`. `--batch --promote-top` → covered in Task 6.
