#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9模型训练脚本 - 基于预计算特征缓存
优势：直接从数据库读取预计算特征，无需重复计算，训练速度快10-100倍
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

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


class V390CachedTrainer:
    """基于预计算特征的V3.9训练器"""

    def __init__(self, db_path='data_adapter/stock_data.db'):
        self.db_path = db_path
        self.models = {}
        self.meta_model = None

    def load_cached_features(self, min_samples=1000):
        """
        从数据库加载预计算的特征

        Args:
            min_samples: 最小样本数阈值

        Returns:
            (X, y): 特征矩阵和标签向量
        """
        logger.info("="*80)
        logger.info("📥 从数据库加载预计算特征...")
        logger.info("="*80)

        conn = sqlite3.connect(self.db_path)

        # 查询所有有效样本
        query = """
            SELECT code, trade_date, features_json, label_5d
            FROM v39_feature_cache
            WHERE label_5d IS NOT NULL
            ORDER BY trade_date, code
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        logger.info(f"✅ 加载 {len(df):,} 个样本")

        if len(df) < min_samples:
            raise ValueError(f"样本数不足！需要至少{min_samples}个，实际{len(df)}个")

        # 解析JSON特征
        logger.info("📊 解析JSON特征...")
        features_list = []
        labels = []

        for idx, row in df.iterrows():
            try:
                # 解析JSON
                features_dict = json.loads(row['features_json'])
                features_list.append(features_dict)
                labels.append(row['label_5d'])

            except Exception as e:
                logger.warning(f"跳过无效样本 {row['code']} {row['trade_date']}: {e}")
                continue

            # 进度报告
            if (idx + 1) % 10000 == 0:
                logger.info(f"  已处理: {idx+1:,}/{len(df):,} ({(idx+1)/len(df)*100:.1f}%)")

        # 转换为DataFrame
        X = pd.DataFrame(features_list)
        y = np.array(labels)

        logger.info(f"✅ 特征矩阵: {X.shape}")
        logger.info(f"✅ 标签向量: {y.shape}")
        logger.info(f"✅ 特征列数: {X.shape[1]}")

        # 处理缺失值
        if X.isnull().any().any():
            logger.warning("⚠️  检测到缺失值，使用0填充")
            X = X.fillna(0)

        return X, y

    def train_base_models(self, X_train, y_train, X_val, y_val):
        """
        训练基础模型

        Returns:
            dict: 基础模型字典
        """
        logger.info("\n" + "="*80)
        logger.info("🔧 训练基础模型...")
        logger.info("="*80)

        # 模型配置
        models_config = {
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

        # 训练每个模型
        for name, model in models_config.items():
            logger.info(f"\n🔹 训练 {name}...")
            start_time = datetime.now()

            model.fit(X_train, y_train)

            # 验证集预测
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)

            elapsed = (datetime.now() - start_time).total_seconds()

            logger.info(f"  ✅ {name}: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.4f}, 耗时={elapsed:.1f}秒")

            self.models[name] = model

        return self.models

    def train_meta_model(self, X_train, y_train, X_val, y_val):
        """
        训练元模型（Stacking）
        """
        logger.info("\n" + "="*80)
        logger.info("🔧 训练元模型 (Stacking)...")
        logger.info("="*80)

        # 生成元特征（基础模型的预测）
        meta_features_train = np.column_stack([
            model.predict(X_train) for model in self.models.values()
        ])

        meta_features_val = np.column_stack([
            model.predict(X_val) for model in self.models.values()
        ])

        # 元模型配置
        self.meta_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )

        # 训练元模型
        logger.info("🔹 训练Gradient Boosting元模型...")
        self.meta_model.fit(meta_features_train, y_train)

        # 评估
        y_pred = self.meta_model.predict(meta_features_val)
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        logger.info(f"✅ 元模型: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.4f}")

        return self.meta_model

    def save_model(self, output_path='models/v39'):
        """保存模型"""
        logger.info("\n" + "="*80)
        logger.info("💾 保存模型...")
        logger.info("="*80)

        Path(output_path).mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存基础模型
        for name, model in self.models.items():
            model_path = f"{output_path}/v390_{name}_{timestamp}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"  ✅ {name}: {model_path}")

        # 保存元模型
        meta_path = f"{output_path}/v390_meta_{timestamp}.pkl"
        with open(meta_path, 'wb') as f:
            pickle.dump(self.meta_model, f)
        logger.info(f"  ✅ meta_model: {meta_path}")

        # 保存完整系统
        full_model = {
            'base_models': self.models,
            'meta_model': self.meta_model,
            'timestamp': timestamp
        }
        full_path = f"{output_path}/v390_full_system_{timestamp}.pkl"
        with open(full_path, 'wb') as f:
            pickle.dump(full_model, f)
        logger.info(f"  ✅ full_system: {full_path}")

        return full_path

    def train(self, test_size=0.2, random_state=42):
        """
        完整训练流程
        """
        # 1. 加载数据
        X, y = self.load_cached_features()

        # 2. 划分训练/验证集
        logger.info("\n" + "="*80)
        logger.info("📊 划分训练/验证集...")
        logger.info("="*80)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )

        logger.info(f"  训练集: {X_train.shape[0]:,} 样本")
        logger.info(f"  验证集: {X_val.shape[0]:,} 样本")

        # 3. 训练基础模型
        self.train_base_models(X_train, y_train, X_val, y_val)

        # 4. 训练元模型
        self.train_meta_model(X_train, y_train, X_val, y_val)

        # 5. 保存模型
        model_path = self.save_model()

        logger.info("\n" + "="*80)
        logger.info("🎉 训练完成!")
        logger.info("="*80)

        return model_path


def main():
    parser = argparse.ArgumentParser(description='V3.9模型训练（基于预计算特征）')
    parser.add_argument('--db-path', type=str, default='data_adapter/stock_data.db', help='数据库路径')
    parser.add_argument('--test-size', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--random-state', type=int, default=42, help='随机种子')
    parser.add_argument('--output-dir', type=str, default='models/v39', help='模型输出目录')

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("🚀 V3.9模型训练 - 基于预计算特征缓存")
    logger.info("="*80)
    logger.info(f"数据库: {args.db_path}")
    logger.info(f"验证集比例: {args.test_size}")
    logger.info(f"输出目录: {args.output_dir}")

    # 训练
    trainer = V390CachedTrainer(db_path=args.db_path)
    model_path = trainer.train(test_size=args.test_size, random_state=args.random_state)

    logger.info(f"\n✅ 模型已保存至: {model_path}")


if __name__ == "__main__":
    main()
