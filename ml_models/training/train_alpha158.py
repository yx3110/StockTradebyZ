#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpha158 Baseline 训练脚本

两种模式:
1. qlib_standard: 单个LightGBM, Qlib原版超参
2. ensemble: 5模型Ensemble (LGB+XGB+CB+RF+HGB), 与V3.95对齐

标签: 3d/5d/10d log return
预处理: RobustZScore + 10天purge gap

用法:
    # Qlib标准模式 (单LightGBM)
    python3 ml_models/training/train_alpha158.py --mode qlib_standard

    # Ensemble模式 (5模型)
    python3 ml_models/training/train_alpha158.py --mode ensemble

    # 自定义日期范围
    python3 ml_models/training/train_alpha158.py --mode ensemble \
        --train-start 2020-01-02 --train-end 2025-06-30
"""

import sys
import os
import json
import gc
import numpy as np
import pandas as pd
import sqlite3
import joblib
import logging
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore')

import lightgbm as lgb
import xgboost as xgb
try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("Warning: CatBoost not installed, skipping")

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import RobustScaler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')


class Alpha158Trainer:
    """Alpha158 Baseline 训练器"""

    def __init__(self, mode: str = 'ensemble',
                 train_start: str = '2020-01-02',
                 train_end: str = '2025-06-30',
                 val_months: int = 3,
                 test_months: int = 3):
        self.mode = mode
        self.train_start = train_start
        self.train_end = train_end
        self.val_months = val_months
        self.test_months = test_months

        self.feature_names = None
        self.target_weights = {'label_3d': 0.35, 'label_5d': 0.35, 'label_10d': 0.30}

    def load_data(self):
        """从 alpha158_feature_cache 加载训练数据 (分块解析避免OOM)"""
        logger.info(f"加载 Alpha158 特征缓存: {self.train_start} ~ {self.train_end}")

        conn = sqlite3.connect(DB_PATH)

        # 检查表是否存在
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alpha158_feature_cache'")
        if not cursor.fetchone():
            logger.error("alpha158_feature_cache 表不存在! 请先运行 fetch_data/alpha158_feature_cache_updater.py")
            conn.close()
            return None

        # 先获取记录数和特征名
        count_q = """SELECT COUNT(*) FROM alpha158_feature_cache
                     WHERE trade_date >= ? AND trade_date <= ? AND label_10d IS NOT NULL"""
        n_total = cursor.execute(count_q, (self.train_start, self.train_end)).fetchone()[0]
        if n_total == 0:
            logger.error("未找到有效数据")
            conn.close()
            return None

        # 获取特征列名 (从第一条记录)
        sample_q = """SELECT features_json FROM alpha158_feature_cache
                      WHERE trade_date >= ? AND trade_date <= ? AND label_10d IS NOT NULL LIMIT 1"""
        sample_json = cursor.execute(sample_q, (self.train_start, self.train_end)).fetchone()[0]
        sample_dict = json.loads(sample_json)
        self.feature_names = sorted(sample_dict.keys())
        n_features = len(self.feature_names)
        logger.info(f"记录数: {n_total:,}, 特征数: {n_features}")

        # 分块加载: 直接写入预分配的numpy数组, 避免创建6.8M个Python dict
        features_arr = np.empty((n_total, n_features), dtype=np.float32)
        meta_code = np.empty(n_total, dtype=object)
        meta_date = np.empty(n_total, dtype=object)
        labels_3d = np.empty(n_total, dtype=np.float32)
        labels_5d = np.empty(n_total, dtype=np.float32)
        labels_10d = np.empty(n_total, dtype=np.float32)

        query = """
        SELECT code, trade_date, features_json, label_3d, label_5d, label_10d
        FROM alpha158_feature_cache
        WHERE trade_date >= ? AND trade_date <= ?
        AND label_10d IS NOT NULL
        ORDER BY trade_date, code
        """

        CHUNK_SIZE = 500_000
        row_idx = 0
        logger.info("分块解析 JSON 特征...")

        for chunk_df in pd.read_sql_query(query, conn, params=(self.train_start, self.train_end),
                                           chunksize=CHUNK_SIZE):
            chunk_len = len(chunk_df)
            end_idx = row_idx + chunk_len

            meta_code[row_idx:end_idx] = chunk_df['code'].values
            meta_date[row_idx:end_idx] = chunk_df['trade_date'].values
            labels_3d[row_idx:end_idx] = chunk_df['label_3d'].values.astype(np.float32)
            labels_5d[row_idx:end_idx] = chunk_df['label_5d'].values.astype(np.float32)
            labels_10d[row_idx:end_idx] = chunk_df['label_10d'].values.astype(np.float32)

            # 分块解析: 500K dicts → DataFrame → numpy → 释放 (峰值~1.2GB/chunk)
            parsed = chunk_df['features_json'].apply(json.loads)
            chunk_feat = pd.DataFrame(parsed.tolist())[self.feature_names]
            features_arr[row_idx:end_idx] = chunk_feat.values.astype(np.float32)
            del parsed, chunk_feat

            row_idx = end_idx
            logger.info(f"  已解析: {row_idx:,}/{n_total:,} ({100*row_idx/n_total:.0f}%)")
            del chunk_df
            gc.collect()

        conn.close()

        # 构建最终DataFrame (特征已在numpy中, 无需pd.DataFrame(list_of_dicts))
        features_df = pd.DataFrame(features_arr, columns=self.feature_names)
        del features_arr
        gc.collect()

        features_df['code'] = meta_code
        features_df['trade_date'] = meta_date
        features_df['label_3d'] = labels_3d
        features_df['label_5d'] = labels_5d
        features_df['label_10d'] = labels_10d
        del meta_code, meta_date, labels_3d, labels_5d, labels_10d
        gc.collect()

        n_days = features_df['trade_date'].nunique()
        logger.info(f"加载完成: {len(features_df):,} 条记录, {n_days} 天, 特征维度: {n_features}")
        return features_df

    def split_data(self, df: pd.DataFrame):
        """时间序列分割: train / val / test (含10天purge gap)"""
        dates = sorted(df['trade_date'].unique())
        n_dates = len(dates)

        # test: 最后 test_months 月 (~60天)
        test_days = self.test_months * 20
        # val: test之前 val_months 月
        val_days = self.val_months * 20

        test_start_idx = max(n_dates - test_days, 0)
        val_start_idx = max(test_start_idx - val_days, 0)
        # purge gap: val开始前10天不用于训练
        train_end_idx = max(val_start_idx - 10, 0)

        train_dates = set(dates[:train_end_idx])
        val_dates = set(dates[val_start_idx:test_start_idx])
        test_dates = set(dates[test_start_idx:])

        train_df = df[df['trade_date'].isin(train_dates)]
        val_df = df[df['trade_date'].isin(val_dates)]
        test_df = df[df['trade_date'].isin(test_dates)]

        logger.info(f"数据分割:")
        logger.info(f"  Train: {len(train_df):,} ({len(train_dates)}天, "
                     f"{sorted(train_dates)[0] if train_dates else 'N/A'} ~ "
                     f"{sorted(train_dates)[-1] if train_dates else 'N/A'})")
        logger.info(f"  Val:   {len(val_df):,} ({len(val_dates)}天)")
        logger.info(f"  Test:  {len(test_df):,} ({len(test_dates)}天)")

        return train_df, val_df, test_df

    def prepare_features(self, train_df, val_df, test_df):
        """准备特征矩阵, RobustZScore归一化"""
        X_train = train_df[self.feature_names].values.astype(np.float32)
        X_val = val_df[self.feature_names].values.astype(np.float32)
        X_test = test_df[self.feature_names].values.astype(np.float32)

        # RobustZScore归一化 (基于训练集的median/IQR)
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Clip outliers
        X_train = np.clip(X_train, -5, 5)
        X_val = np.clip(X_val, -5, 5)
        X_test = np.clip(X_test, -5, 5)

        # Replace NaN
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        self.scaler = scaler
        self.val_dates = val_df['trade_date'].values
        self.test_dates = test_df['trade_date'].values

        return X_train, X_val, X_test

    def train_qlib_standard(self, X_train, X_val, y_train, y_val, target_name: str):
        """Qlib 原版超参: 单个 LightGBM"""
        logger.info(f"  训练 Qlib Standard LightGBM ({target_name})...")
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'boosting_type': 'gbdt',
            'num_leaves': 210,
            'learning_rate': 0.2,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'reg_alpha': 205.6999,
            'reg_lambda': 580.9768,
            'verbose': -1,
        }

        lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train, free_raw_data=True)

        model = lgb.train(
            params, lgb_train,
            num_boost_round=800,
            valid_sets=[lgb_train, lgb_val],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        models = {'lgb': model}
        weights = {'lgb': 1.0}
        del lgb_train, lgb_val
        gc.collect()

        return models, weights

    def train_ensemble(self, X_train, X_val, y_train, y_val, target_name: str):
        """5模型Ensemble (与V3.95对齐)"""
        models = {}
        predictions_val = {}

        # 1. LightGBM
        logger.info(f"  训练 LightGBM ({target_name})...")
        lgb_params = {
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
            'verbose': -1
        }
        lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train, free_raw_data=True)
        lgb_model = lgb.train(
            lgb_params, lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_train, lgb_val],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        models['lgb'] = lgb_model
        predictions_val['lgb'] = lgb_model.predict(X_val)
        del lgb_train, lgb_val
        gc.collect()

        # 2. XGBoost
        logger.info(f"  训练 XGBoost ({target_name})...")
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbosity': 0
        }
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        xgb_model = xgb.train(
            xgb_params, dtrain,
            num_boost_round=500,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        models['xgb'] = xgb_model
        predictions_val['xgb'] = xgb_model.predict(dval)
        del dtrain, dval
        gc.collect()

        # 3. CatBoost
        if HAS_CATBOOST:
            logger.info(f"  训练 CatBoost ({target_name})...")
            cb_model = cb.CatBoostRegressor(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3,
                random_seed=42,
                verbose=False,
                early_stopping_rounds=50
            )
            cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            models['cb'] = cb_model
            predictions_val['cb'] = cb_model.predict(X_val)

        # 4. RandomForest
        logger.info(f"  训练 RandomForest ({target_name})...")
        rf_max_samples = min(200_000, X_train.shape[0])
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            max_samples=rf_max_samples,
            max_features=0.8,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        rf_model.fit(X_train, y_train)
        models['rf'] = rf_model
        predictions_val['rf'] = rf_model.predict(X_val)

        # 5. HistGradientBoosting
        logger.info(f"  训练 HistGradientBoosting ({target_name})...")
        hgb_model = HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.05,
            max_depth=6,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            early_stopping=False,
            random_state=42,
            verbose=0
        )
        hgb_model.fit(X_train, y_train)
        models['hgb'] = hgb_model
        predictions_val['hgb'] = hgb_model.predict(X_val)

        # IC-based ensemble weights
        weights = self._calculate_ensemble_weights(predictions_val, y_val)
        return models, weights

    def _calculate_ensemble_weights(self, predictions_val: dict, y_val) -> dict:
        """基于验证集IC计算Ensemble权重"""
        val_dates = self.val_dates
        unique_dates = np.unique(val_dates)

        mean_ics = {}
        for name, pred in predictions_val.items():
            daily_ics = []
            for date in unique_dates:
                mask = val_dates == date
                n = mask.sum()
                if n < 10:
                    continue
                ic, _ = spearmanr(pred[mask], y_val[mask])
                if not np.isnan(ic):
                    daily_ics.append(ic)
            mean_ics[name] = float(np.mean(daily_ics)) if daily_ics else 0.0

        logger.info(f"  模型IC: {', '.join(f'{k}={v:.4f}' for k, v in mean_ics.items())}")

        # Softmax权重
        ics = np.array(list(mean_ics.values()))
        ics_clipped = np.maximum(ics, 0.0)  # 负IC的模型权重为0
        if ics_clipped.sum() > 0:
            weights_arr = ics_clipped / ics_clipped.sum()
        else:
            weights_arr = np.ones(len(ics)) / len(ics)

        weights = {name: float(w) for name, w in zip(mean_ics.keys(), weights_arr)}
        logger.info(f"  权重: {', '.join(f'{k}={v:.3f}' for k, v in weights.items())}")
        return weights

    def evaluate(self, models_dict: dict, X_test, test_df: pd.DataFrame):
        """评估测试集性能"""
        logger.info("\n" + "=" * 60)
        logger.info("测试集评估")
        logger.info("=" * 60)

        test_dates = test_df['trade_date'].values

        for target_key in ['3d', '5d', '10d']:
            target_data = models_dict[target_key]
            models = target_data['models']
            weights = target_data['weights']
            y_test = test_df[f'label_{target_key}'].values

            # Ensemble prediction
            ensemble_pred = np.zeros(len(X_test))
            total_w = 0
            for name, model in models.items():
                w = weights.get(name, 0.2)
                if name == 'xgb':
                    pred = model.predict(xgb.DMatrix(X_test))
                else:
                    pred = model.predict(X_test)
                ensemble_pred += w * pred
                total_w += w
            if total_w > 0:
                ensemble_pred /= total_w

            # Global IC
            valid_mask = ~np.isnan(y_test) & ~np.isnan(ensemble_pred)
            ic_global, _ = spearmanr(ensemble_pred[valid_mask], y_test[valid_mask])

            # Daily IC / ICIR
            unique_dates = np.unique(test_dates)
            daily_ics = []
            for d in unique_dates:
                mask = (test_dates == d) & valid_mask
                if mask.sum() < 10:
                    continue
                ic, _ = spearmanr(ensemble_pred[mask], y_test[mask])
                if not np.isnan(ic):
                    daily_ics.append(ic)

            mean_ic = np.mean(daily_ics) if daily_ics else 0
            std_ic = np.std(daily_ics) if daily_ics else 1
            icir = mean_ic / std_ic if std_ic > 1e-6 else 0
            ic_positive_pct = np.mean(np.array(daily_ics) > 0) * 100 if daily_ics else 0

            # Top-10% return
            top10_idx = np.argsort(ensemble_pred)[-int(len(ensemble_pred) * 0.1):]
            top10_return = np.mean(y_test[top10_idx]) * 100
            bottom10_idx = np.argsort(ensemble_pred)[:int(len(ensemble_pred) * 0.1)]
            bottom10_return = np.mean(y_test[bottom10_idx]) * 100

            logger.info(f"\n  {target_key}:")
            logger.info(f"    Global IC:    {ic_global:.4f}")
            logger.info(f"    Daily IC:     {mean_ic:.4f} (std={std_ic:.4f})")
            logger.info(f"    ICIR:         {icir:.4f}")
            logger.info(f"    IC>0:         {ic_positive_pct:.1f}%")
            logger.info(f"    Top-10% ret:  {top10_return:+.3f}%")
            logger.info(f"    Bot-10% ret:  {bottom10_return:+.3f}%")
            logger.info(f"    多空价差:     {top10_return - bottom10_return:+.3f}%")

    def train(self):
        """主训练流程"""
        start_time = datetime.now()
        logger.info(f"Alpha158 训练开始 (mode={self.mode})")
        logger.info(f"日期范围: {self.train_start} ~ {self.train_end}")

        # 1. 加载数据
        df = self.load_data()
        if df is None:
            return

        # 2. 分割数据
        train_df, val_df, test_df = self.split_data(df)
        del df
        gc.collect()

        # 3. 准备特征
        X_train, X_val, X_test = self.prepare_features(train_df, val_df, test_df)

        # 4. 训练各目标
        all_results = {}
        ensemble_weights = {}
        targets = [
            ('3d', train_df['label_3d'].values, val_df['label_3d'].values, 'label_3d'),
            ('5d', train_df['label_5d'].values, val_df['label_5d'].values, 'label_5d'),
            ('10d', train_df['label_10d'].values, val_df['label_10d'].values, 'label_10d'),
        ]

        for target_key, y_train, y_val, target_name in targets:
            if self.mode == 'qlib_standard':
                models, weights = self.train_qlib_standard(
                    X_train, X_val, y_train, y_val, target_name)
            else:
                models, weights = self.train_ensemble(
                    X_train, X_val, y_train, y_val, target_name)

            all_results[target_key] = {'models': models, 'weights': weights}
            ensemble_weights[f'label_{target_key}'] = weights

        # 5. 保存模型 (先保存再评估, 避免评估出错丢失2.5小时训练结果)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        output_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'alpha158'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        model_data = {
            'version': 'alpha158',
            'mode': self.mode,
            'models': all_results,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'ensemble_weights': ensemble_weights,
            'target_weights': self.target_weights,
            'train_start': self.train_start,
            'train_end': self.train_end,
            'n_features': len(self.feature_names),
            'training_duration_seconds': duration,
        }

        model_path = output_dir / f'alpha158_{self.mode}_{timestamp}.pkl'
        joblib.dump(model_data, model_path)
        logger.info(f"\n模型已保存: {model_path}")
        logger.info(f"模型大小: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
        logger.info(f"训练耗时: {duration:.0f}秒 ({duration/60:.1f}分钟)")

        # 同时保存一个 latest link
        latest_path = output_dir / f'alpha158_{self.mode}_latest.pkl'
        if latest_path.exists():
            latest_path.unlink()
        import shutil
        shutil.copy2(model_path, latest_path)
        logger.info(f"Latest link: {latest_path}")

        # 6. 评估 (模型已保存, 评估失败不影响模型)
        self.evaluate(all_results, X_test, test_df)


def main():
    parser = argparse.ArgumentParser(description='Alpha158 Baseline 训练')
    parser.add_argument('--mode', default='ensemble',
                        choices=['qlib_standard', 'ensemble'],
                        help='训练模式 (default: ensemble)')
    parser.add_argument('--train-start', default='2020-01-02',
                        help='训练开始日期 (default: 2020-01-02)')
    parser.add_argument('--train-end', default='2025-12-31',
                        help='训练结束日期 (default: 2025-12-31)')
    parser.add_argument('--val-months', type=int, default=3,
                        help='验证集月数 (default: 3)')
    parser.add_argument('--test-months', type=int, default=3,
                        help='测试集月数 (default: 3)')
    args = parser.parse_args()

    trainer = Alpha158Trainer(
        mode=args.mode,
        train_start=args.train_start,
        train_end=args.train_end,
        val_months=args.val_months,
        test_months=args.test_months,
    )
    trainer.train()


if __name__ == '__main__':
    main()
