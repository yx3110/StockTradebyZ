#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.91 多周期模型训练器
训练3个独立模型分别预测5天、10天、15天收益
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class V391MultiPeriodTrainer:
    """V3.91 多周期模型训练器"""

    # 多周期权重配置
    PERIOD_WEIGHTS = {
        '5d': 0.40,   # 短线权重40%
        '10d': 0.35,  # 中短线权重35%
        '15d': 0.25,  # 中线权重25%
    }

    def __init__(self, db_path=None):
        self.db_path = db_path or str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
        self.models = {
            '5d': {},
            '10d': {},
            '15d': {}
        }
        self.meta_models = {
            '5d': None,
            '10d': None,
            '15d': None
        }
        self.feature_columns = None

    def load_cached_features(self, min_samples=1000):
        """
        从数据库加载预计算的特征和多周期标签
        """
        logger.info("=" * 80)
        logger.info("📥 加载多周期训练数据...")
        logger.info("=" * 80)

        conn = sqlite3.connect(self.db_path)

        # 查询所有有完整标签的样本
        query = """
            SELECT code, trade_date, features_json, label_5d, label_10d, label_15d
            FROM v39_feature_cache
            WHERE label_5d IS NOT NULL
            AND label_10d IS NOT NULL
            AND label_15d IS NOT NULL
            ORDER BY trade_date, code
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        logger.info(f"✅ 加载 {len(df):,} 个完整样本")

        if len(df) < min_samples:
            raise ValueError(f"样本数不足！需要至少{min_samples}个，实际{len(df)}个")

        # 解析JSON特征
        logger.info("📊 解析JSON特征...")
        features_list = []
        labels_5d = []
        labels_10d = []
        labels_15d = []

        for idx, row in df.iterrows():
            try:
                features_dict = json.loads(row['features_json'])
                features_list.append(features_dict)
                labels_5d.append(row['label_5d'])
                labels_10d.append(row['label_10d'])
                labels_15d.append(row['label_15d'])

            except Exception as e:
                logger.warning(f"跳过无效样本: {e}")
                continue

            if (idx + 1) % 50000 == 0:
                logger.info(f"  已处理: {idx+1:,}/{len(df):,}")

        # 转换为DataFrame
        X = pd.DataFrame(features_list)
        y = {
            '5d': np.array(labels_5d),
            '10d': np.array(labels_10d),
            '15d': np.array(labels_15d)
        }

        self.feature_columns = X.columns.tolist()

        logger.info(f"✅ 特征矩阵: {X.shape}")
        logger.info(f"✅ 特征数量: {len(self.feature_columns)}")

        # 处理缺失值
        if X.isnull().any().any():
            logger.warning("⚠️  检测到缺失值，使用0填充")
            X = X.fillna(0)

        # 标签分布统计
        logger.info("\n📊 标签分布统计:")
        for period, labels in y.items():
            logger.info(f"  {period}: min={labels.min():.4f}, max={labels.max():.4f}, "
                       f"mean={labels.mean():.4f}, std={labels.std():.4f}")

        return X, y

    def _create_base_models(self):
        """创建基础模型配置"""
        return {
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            ),
            'catboost': CatBoostRegressor(
                iterations=200,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3,
                random_state=42,
                verbose=False
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        }

    def train_period_models(self, period, X_train, y_train, X_val, y_val):
        """
        训练单个周期的基础模型

        Args:
            period: '5d', '10d', 或 '15d'
            X_train, y_train: 训练数据
            X_val, y_val: 验证数据

        Returns:
            dict: 训练好的模型字典
        """
        logger.info(f"\n🔹 训练 {period} 周期模型...")

        models_config = self._create_base_models()
        trained_models = {}

        for name, model in models_config.items():
            start_time = datetime.now()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"    {name}: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.4f}, {elapsed:.1f}秒")

            trained_models[name] = model

        self.models[period] = trained_models
        return trained_models

    def train_period_meta_model(self, period, X_train, y_train, X_val, y_val):
        """
        训练单个周期的元模型
        """
        logger.info(f"  训练 {period} 元模型...")

        # 生成元特征
        meta_features_train = np.column_stack([
            model.predict(X_train) for model in self.models[period].values()
        ])
        meta_features_val = np.column_stack([
            model.predict(X_val) for model in self.models[period].values()
        ])

        # 元模型
        meta_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )

        meta_model.fit(meta_features_train, y_train)

        y_pred = meta_model.predict(meta_features_val)
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        logger.info(f"    元模型: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.4f}")

        self.meta_models[period] = meta_model
        return meta_model

    def save_model(self, output_path='ml_models/trained_models/v391'):
        """保存多周期模型"""
        logger.info("\n" + "=" * 80)
        logger.info("💾 保存V3.91多周期模型...")
        logger.info("=" * 80)

        Path(output_path).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存完整系统
        full_model = {
            'version': 'v3.91',
            'base_models': self.models,
            'meta_models': self.meta_models,
            'feature_columns': self.feature_columns,
            'period_weights': self.PERIOD_WEIGHTS,
            'timestamp': timestamp
        }

        full_path = f"{output_path}/v391_multiperiod_{timestamp}.pkl"
        with open(full_path, 'wb') as f:
            pickle.dump(full_model, f)
        logger.info(f"  ✅ 完整模型: {full_path}")

        # 创建/更新生产用链接
        latest_path = f"{output_path}/v391_multiperiod_latest.pkl"
        with open(latest_path, 'wb') as f:
            pickle.dump(full_model, f)
        logger.info(f"  ✅ 生产链接: {latest_path}")

        return full_path

    def train(self, test_size=0.2, random_state=42):
        """
        完整多周期训练流程
        """
        # 1. 加载数据
        X, y = self.load_cached_features()

        # 2. 划分数据集（使用相同的划分确保一致性）
        logger.info("\n" + "=" * 80)
        logger.info("📊 划分训练/验证集...")
        logger.info("=" * 80)

        X_train, X_val, indices_train, indices_val = train_test_split(
            X, np.arange(len(X)),
            test_size=test_size,
            random_state=random_state,
            shuffle=False  # 时序数据禁止shuffle
        )

        logger.info(f"  训练集: {X_train.shape[0]:,} 样本")
        logger.info(f"  验证集: {X_val.shape[0]:,} 样本")

        # 3. 训练每个周期的模型
        logger.info("\n" + "=" * 80)
        logger.info("🔧 训练多周期基础模型...")
        logger.info("=" * 80)

        for period in ['5d', '10d', '15d']:
            y_train_period = y[period][indices_train]
            y_val_period = y[period][indices_val]

            self.train_period_models(period, X_train, y_train_period, X_val, y_val_period)

        # 4. 训练元模型
        logger.info("\n" + "=" * 80)
        logger.info("🔧 训练多周期元模型...")
        logger.info("=" * 80)

        for period in ['5d', '10d', '15d']:
            y_train_period = y[period][indices_train]
            y_val_period = y[period][indices_val]

            self.train_period_meta_model(period, X_train, y_train_period, X_val, y_val_period)

        # 5. 评估综合性能
        logger.info("\n" + "=" * 80)
        logger.info("📊 综合评估...")
        logger.info("=" * 80)

        # 计算加权综合预测
        composite_pred_val = np.zeros(len(X_val))
        composite_true_val = np.zeros(len(X_val))

        for period, weight in self.PERIOD_WEIGHTS.items():
            # 元特征
            meta_features = np.column_stack([
                model.predict(X_val) for model in self.models[period].values()
            ])
            # 元模型预测
            pred = self.meta_models[period].predict(meta_features)
            composite_pred_val += weight * pred
            composite_true_val += weight * y[period][indices_val]

        mse = mean_squared_error(composite_true_val, composite_pred_val)
        mae = mean_absolute_error(composite_true_val, composite_pred_val)
        r2 = r2_score(composite_true_val, composite_pred_val)

        logger.info(f"  综合评分 (加权组合):")
        logger.info(f"    MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.4f}")
        logger.info(f"    权重: 5d={self.PERIOD_WEIGHTS['5d']}, "
                   f"10d={self.PERIOD_WEIGHTS['10d']}, 15d={self.PERIOD_WEIGHTS['15d']}")

        # 6. 保存模型
        model_path = self.save_model()

        logger.info("\n" + "=" * 80)
        logger.info("🎉 V3.91 多周期模型训练完成!")
        logger.info("=" * 80)

        return model_path


def main():
    parser = argparse.ArgumentParser(description='V3.91 多周期模型训练')
    parser.add_argument('--db-path', type=str, default=str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db'))
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default=str(PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v391'))

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("🚀 V3.91 多周期模型训练")
    logger.info("=" * 80)
    logger.info(f"数据库: {args.db_path}")
    logger.info(f"验证集比例: {args.test_size}")

    trainer = V391MultiPeriodTrainer(db_path=args.db_path)
    model_path = trainer.train(test_size=args.test_size, random_state=args.random_state)

    logger.info(f"\n✅ 模型已保存至: {model_path}")


if __name__ == "__main__":
    main()
