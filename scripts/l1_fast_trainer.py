#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L1 Fast Trainer — 轻量 LightGBM 快速筛选训练器

用途: 3-5分钟快速验证管线的第一道门 (L1筛选)
特点:
- 单折 70/15/15 分割
- 仅使用 LightGBM (mae损失)
- 截面 Robust Z-Score 归一化
- 返回 IC / ICIR 门控指标

作者: Claude Code
创建时间: 2026-03-26
"""

import sys
import os
import json
import time
import sqlite3
import logging
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# Project root setup
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Optional fast JSON
try:
    import orjson
    _json_loads = orjson.loads
except ImportError:
    _json_loads = json.loads

# ------------------------------------------------------------------
# Default params
# ------------------------------------------------------------------
DEFAULT_PARAMS = {
    'variant_name': 'l1_default',
    'training': {
        'l1_start_date': None,          # None → 2 years ago
        'purge_days': 10,
        'l1_num_boost_round': 150,
        'num_leaves': 31,
        'min_data_in_leaf': 200,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
    },
    'features': {
        'remove': [],
    },
}


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------
class L1FastTrainer:
    """轻量 LightGBM L1 快速筛选训练器."""

    def __init__(self, params: dict):
        """初始化，存储参数，初始化空模型和特征列."""
        # Deep-merge with defaults
        self.params = self._merge_params(DEFAULT_PARAMS, params)
        self.models: Dict[str, lgb.Booster] = {}
        self.feature_cols: List[str] = []
        self.variant_name: str = self.params.get('variant_name', 'l1_default')

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_params(base: dict, override: dict) -> dict:
        """递归合并参数字典."""
        result = base.copy()
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = L1FastTrainer._merge_params(result[k], v)
            else:
                result[k] = v
        return result

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------

    def _load_data(self) -> pd.DataFrame:
        """从 v39_feature_cache 加载训练数据."""
        train_cfg = self.params.get('training', {})
        l1_start = train_cfg.get('l1_start_date', None)

        if l1_start is None:
            two_years_ago = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            l1_start = two_years_ago

        print(f"[L1] 加载数据 from {l1_start} ...")

        conn = sqlite3.connect(DB_PATH, timeout=30)
        try:
            query = """
                SELECT code, trade_date,
                       features_json,
                       label_5d, label_10d,
                       market_return_20d, market_return_10d, market_return_5d,
                       market_volatility_20d, market_volatility_10d,
                       market_up_ratio_20d, market_up_ratio_10d,
                       market_drawdown_20d, market_volume_ratio,
                       market_position_20d, market_momentum_20d, market_momentum_5d
                FROM v39_feature_cache
                WHERE trade_date >= ?
                  AND label_5d IS NOT NULL
                  AND label_10d IS NOT NULL
                ORDER BY trade_date, code
            """
            df = pd.read_sql_query(query, conn, params=(l1_start,))
        finally:
            conn.close()

        if df.empty:
            raise ValueError(f"No data found from {l1_start}")

        print(f"[L1] 原始记录: {len(df):,}  日期: {df['trade_date'].nunique()}")

        # Parse features_json vectorized
        feature_dicts = df['features_json'].apply(_json_loads)
        feat_df = pd.DataFrame(list(feature_dicts), index=df.index)

        # Apply feature removal
        remove_list = self.params.get('features', {}).get('remove', [])
        if remove_list:
            feat_df.drop(columns=[c for c in remove_list if c in feat_df.columns],
                         inplace=True)

        # Market feature columns
        market_cols = [c for c in df.columns if c.startswith('market_')]

        # Build final dataframe
        meta_cols = ['code', 'trade_date', 'label_5d', 'label_10d']
        result = pd.concat([
            df[meta_cols].reset_index(drop=True),
            feat_df.reset_index(drop=True),
            df[market_cols].reset_index(drop=True),
        ], axis=1)

        # Set feature_cols
        base_features = list(feat_df.columns)
        self.feature_cols = base_features + market_cols

        print(f"[L1] 特征数: {len(self.feature_cols)}  (base={len(base_features)}, market={len(market_cols)})")

        return result

    # ------------------------------------------------------------------
    # Step 2: Split data
    # ------------------------------------------------------------------

    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """单折 70/15/15 按日期分割，含 purge gap."""
        purge_days = self.params.get('training', {}).get('purge_days', 10)

        sorted_dates = sorted(df['trade_date'].unique())
        n = len(sorted_dates)
        i70 = int(n * 0.70)
        i85 = int(n * 0.85)

        train_end_date = sorted_dates[i70 - 1]
        val_start_date = sorted_dates[i70]
        val_end_date = sorted_dates[i85 - 1]
        test_start_date = sorted_dates[i85]

        # Apply purge gap: remove the first purge_days of val and test
        val_dates_with_purge = sorted_dates[i70 + purge_days:]
        if len(val_dates_with_purge) == 0:
            raise ValueError("Not enough dates for val after purge gap")
        val_end_idx = sorted_dates.index(val_end_date)
        val_dates_no_overlap = [d for d in val_dates_with_purge if d <= val_end_date]

        test_dates_with_purge = sorted_dates[i85 + purge_days:]
        test_dates_valid = [d for d in test_dates_with_purge]

        train_df = df[df['trade_date'] <= train_end_date].copy()
        val_df = df[df['trade_date'].isin(val_dates_no_overlap)].copy()
        test_df = df[df['trade_date'].isin(test_dates_valid)].copy()

        print(f"[L1] Train: {train_df['trade_date'].min()} ~ {train_df['trade_date'].max()}  ({train_df['trade_date'].nunique()} days)")
        print(f"[L1] Val:   {val_df['trade_date'].min()} ~ {val_df['trade_date'].max()}  ({val_df['trade_date'].nunique()} days)")
        print(f"[L1] Test:  {test_df['trade_date'].min()} ~ {test_df['trade_date'].max()}  ({test_df['trade_date'].nunique()} days)")

        return train_df, val_df, test_df

    # ------------------------------------------------------------------
    # Step 3: Normalize
    # ------------------------------------------------------------------

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """截面 Robust Z-Score 归一化: (x - median) / (MAD * 1.4826)."""
        df = df.copy()
        feat_cols = [c for c in self.feature_cols if c in df.columns]

        # Vectorized cross-sectional robust z-score normalization
        # Avoid groupby.apply deprecation by using transform on each feature
        for col in feat_cols:
            med = df.groupby('trade_date')[col].transform('median')
            mad = df.groupby('trade_date')[col].transform(
                lambda x: (x - x.median()).abs().median()
            )
            valid_mad = mad > 1e-10
            normalized = (df[col] - med) / (mad * 1.4826)
            df[col] = np.where(valid_mad, normalized, 0.0)

        # Clip and fill
        df[feat_cols] = df[feat_cols].clip(-5, 5).fillna(0)

        return df

    # ------------------------------------------------------------------
    # Step 4: Train LGB
    # ------------------------------------------------------------------

    def _train_lgb(self, train_df: pd.DataFrame, val_df: pd.DataFrame, target: str) -> lgb.Booster:
        """训练单个 LightGBM 回归模型 (mae)."""
        train_cfg = self.params.get('training', {})

        feat_cols = [c for c in self.feature_cols if c in train_df.columns]

        # Remove NaN labels
        train_data = train_df[train_df[target].notna()].copy()
        val_data = val_df[val_df[target].notna()].copy()

        X_train = train_data[feat_cols].values
        y_train = train_data[target].values
        X_val = val_data[feat_cols].values
        y_val = val_data[target].values

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feat_cols)
        dval = lgb.Dataset(X_val, label=y_val, feature_name=feat_cols, reference=dtrain)

        lgb_params = {
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': train_cfg.get('num_leaves', 31),
            'min_data_in_leaf': train_cfg.get('min_data_in_leaf', 200),
            'learning_rate': train_cfg.get('learning_rate', 0.05),
            'feature_fraction': train_cfg.get('feature_fraction', 0.8),
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42,
        }

        num_boost_round = train_cfg.get('l1_num_boost_round', 150)

        callbacks = [
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=-1),
        ]

        model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        print(f"[L1] {target}: best_iter={model.best_iteration}, "
              f"best_val_mae={model.best_score.get('valid_0', {}).get('l1', 'N/A'):.4f}")

        return model

    # ------------------------------------------------------------------
    # Step 5: Compute daily IC
    # ------------------------------------------------------------------

    def _compute_daily_ic(self, df: pd.DataFrame, target: str, pred_col: str) -> list:
        """计算每日 Spearman IC，过滤少于 30 只股票的日期."""
        ic_list = []
        for date, grp in df.groupby('trade_date'):
            valid = grp[[target, pred_col]].dropna()
            if len(valid) < 30:
                continue
            ic, _ = spearmanr(valid[target].values, valid[pred_col].values)
            if not np.isnan(ic):
                ic_list.append(ic)
        return ic_list

    # ------------------------------------------------------------------
    # Step 6: Evaluate
    # ------------------------------------------------------------------

    def _evaluate(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                  test_df: pd.DataFrame) -> dict:
        """对所有目标进行预测并计算 IC / ICIR 指标."""
        metrics = {}
        feat_cols = [c for c in self.feature_cols if c in train_df.columns]

        all_ic_10d = {}  # for train_val_gap

        for target_suffix, target in [('5d', 'label_5d'), ('10d', 'label_10d')]:
            model = self.models.get(target)
            if model is None:
                continue

            pred_col = f'pred_{target_suffix}'

            for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
                data = split_df[split_df[target].notna()].copy()
                if data.empty:
                    continue
                X = data[feat_cols].fillna(0).values
                data[pred_col] = model.predict(X)
                split_df.loc[data.index, pred_col] = data[pred_col]

                ic_list = self._compute_daily_ic(data, target, pred_col)
                ic_mean = float(np.mean(ic_list)) if ic_list else 0.0
                ic_std = float(np.std(ic_list)) if ic_list else 1.0
                icir = ic_mean / ic_std if ic_std > 1e-10 else 0.0

                metrics[f'{split_name}_ic_{target_suffix}'] = ic_mean
                metrics[f'{split_name}_icir_{target_suffix}'] = icir
                all_ic_10d[split_name] = ic_mean

            # Feature importance (top 10)
            if target_suffix == '10d':
                importance = model.feature_importance(importance_type='gain')
                feat_imp = sorted(zip(feat_cols, importance), key=lambda x: x[1], reverse=True)
                metrics['top10_feature_importance'] = feat_imp[:10]

        # train_val_gap = train_ic_10d - val_ic_10d (overfitting proxy)
        metrics['train_val_gap'] = metrics.get('train_ic_10d', 0.0) - metrics.get('val_ic_10d', 0.0)
        metrics['n_features'] = len(feat_cols)

        return metrics

    # ------------------------------------------------------------------
    # Step 7: Save model
    # ------------------------------------------------------------------

    def _save_model(self) -> str:
        """保存模型到 /tmp/l1_xxx.pkl."""
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f'/tmp/l1_{self.variant_name}_{ts}.pkl'

        payload = {
            'models': self.models,
            'feature_cols': self.feature_cols,
            'variant_name': self.variant_name,
            'params': self.params,
            'timestamp': ts,
        }
        joblib.dump(payload, model_path)
        print(f"[L1] 模型已保存: {model_path}")
        return model_path

    # ------------------------------------------------------------------
    # Step 8: Gate check
    # ------------------------------------------------------------------

    def _check_gate(self, metrics: dict) -> bool:
        """L1 门控检查.

        通过条件:
        - test_ic_10d >= 0.04
        - test_icir_10d >= 0.40
        - train_val_gap <= 0.05
        """
        test_ic = metrics.get('test_ic_10d', 0.0)
        test_icir = metrics.get('test_icir_10d', 0.0)
        gap = metrics.get('train_val_gap', 999.0)

        checks = {
            'test_ic_10d >= 0.04': test_ic >= 0.04,
            'test_icir_10d >= 0.40': test_icir >= 0.40,
            'train_val_gap <= 0.05': gap <= 0.05,
        }

        all_pass = all(checks.values())

        print("\n[L1] 门控检查:")
        for name, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}")

        return all_pass

    # ------------------------------------------------------------------
    # Step 9: Main orchestrator
    # ------------------------------------------------------------------

    def train(self) -> dict:
        """主训练流程: 加载 → 分割 → 归一化 → 训练 → 评估 → 保存 → 门控."""
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"[L1] 开始 L1 快速训练: {self.variant_name}")
        print(f"{'='*60}")

        # 1. Load
        df = self._load_data()

        # 2. Split
        train_df, val_df, test_df = self._split_data(df)

        # 3. Normalize (each set independently to avoid leakage)
        print("[L1] 归一化 ...")
        train_df = self._normalize(train_df)
        val_df = self._normalize(val_df)
        test_df = self._normalize(test_df)

        # 4. Train each target
        for target in ['label_5d', 'label_10d']:
            print(f"\n[L1] 训练 {target} ...")
            model = self._train_lgb(train_df, val_df, target)
            self.models[target] = model

        # 5. Evaluate
        print("\n[L1] 评估 ...")
        metrics = self._evaluate(train_df, val_df, test_df)

        # 6. Save
        model_path = self._save_model()

        # 7. Gate check
        gate_pass = self._check_gate(metrics)

        duration = time.time() - t0

        # Summary
        print(f"\n[L1] 结果摘要:")
        print(f"  test_ic_5d   = {metrics.get('test_ic_5d', 0):.4f}")
        print(f"  test_ic_10d  = {metrics.get('test_ic_10d', 0):.4f}")
        print(f"  test_icir_5d = {metrics.get('test_icir_5d', 0):.4f}")
        print(f"  test_icir_10d= {metrics.get('test_icir_10d', 0):.4f}")
        print(f"  val_ic_10d   = {metrics.get('val_ic_10d', 0):.4f}")
        print(f"  train_val_gap= {metrics.get('train_val_gap', 0):.4f}")
        print(f"  n_features   = {metrics.get('n_features', 0)}")
        print(f"  gate_pass    = {gate_pass}")
        print(f"  duration     = {duration:.1f}s")

        return {
            'variant_name': self.variant_name,
            'level': 'L1',
            'duration_sec': duration,
            'metrics': metrics,
            'gate_pass': gate_pass,
            'model_path': model_path,
            'feature_cols': self.feature_cols,
        }


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='L1 Fast LightGBM Trainer')
    parser.add_argument('--variant', default='l1_default', help='变体名称')
    parser.add_argument('--start-date', default=None, help='训练起始日期 (YYYYMMDD 或 YYYY-MM-DD)')
    parser.add_argument('--num-boost-round', type=int, default=150, help='LGB 迭代次数')
    parser.add_argument('--purge-days', type=int, default=10, help='purge gap 天数')
    args = parser.parse_args()

    params = {
        'variant_name': args.variant,
        'training': {
            'l1_num_boost_round': args.num_boost_round,
            'purge_days': args.purge_days,
        },
    }
    if args.start_date:
        start = args.start_date.replace('-', '')
        if len(start) == 8:
            start = f"{start[:4]}-{start[4:6]}-{start[6:8]}"
        params['training']['l1_start_date'] = start

    trainer = L1FastTrainer(params)
    result = trainer.train()

    print("\n[L1] 完成.")
    if result['gate_pass']:
        print("[L1] ✅ 门控通过 → 可进入 L2 完整训练")
    else:
        print("[L1] ❌ 门控未通过 → 终止管线")

    sys.exit(0 if result['gate_pass'] else 1)
