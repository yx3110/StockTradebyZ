#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9模型完整训练和评估脚本
集成ModelEvaluator进行全面验证
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# 导入评估器
from model_evaluator import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class V390ComprehensiveTrainer:
    """V3.9完整训练和评估器"""

    def __init__(self, db_path=None):
        self.db_path = db_path or str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
        self.models = {}
        self.meta_model = None

    def load_cached_features(self, date_filter=None):
        """
        从数据库加载预计算的特征

        Args:
            date_filter: 日期过滤器 {'start': 'YYYY-MM-DD', 'end': 'YYYY-MM-DD'}

        Returns:
            (X, y, dates): 特征矩阵、标签向量、日期列表
        """
        logger.info("="*80)
        logger.info("📥 从数据库加载预计算特征...")
        logger.info("="*80)

        conn = sqlite3.connect(self.db_path)

        # 构建查询
        if date_filter:
            query = """
                SELECT code, trade_date, features_json, label_5d
                FROM v39_feature_cache
                WHERE label_5d IS NOT NULL
                  AND trade_date >= ?
                  AND trade_date <= ?
                ORDER BY trade_date, code
            """
            df = pd.read_sql_query(query, conn, params=(date_filter['start'], date_filter['end']))
        else:
            query = """
                SELECT code, trade_date, features_json, label_5d
                FROM v39_feature_cache
                WHERE label_5d IS NOT NULL
                ORDER BY trade_date, code
            """
            df = pd.read_sql_query(query, conn)

        conn.close()

        logger.info(f"✅ 加载 {len(df):,} 个样本")
        if date_filter:
            logger.info(f"   日期范围: {date_filter['start']} 至 {date_filter['end']}")

        # 解析JSON特征
        logger.info("📊 解析JSON特征...")
        features_list = []
        labels = []
        dates = []

        for idx, row in df.iterrows():
            try:
                features_dict = json.loads(row['features_json'])
                features_list.append(features_dict)
                labels.append(row['label_5d'])
                dates.append(row['trade_date'])
            except Exception as e:
                continue

            if (idx + 1) % 50000 == 0:
                logger.info(f"  已处理: {idx+1:,}/{len(df):,}")

        X = pd.DataFrame(features_list).fillna(0)
        y = np.array(labels)
        dates = pd.Series(dates)

        logger.info(f"✅ 特征矩阵: {X.shape}")
        logger.info(f"✅ 特征数量: {X.shape[1]}")
        logger.info(f"✅ 日期范围: {dates.min()} ~ {dates.max()}")

        return X, y, dates

    def train_models(self, X_train, y_train):
        """训练基础模型和元学习器"""
        logger.info("\n" + "="*80)
        logger.info("🎯 训练Ensemble模型...")
        logger.info("="*80)

        # Layer 1: 基础模型
        base_models = {
            'lgb': lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                random_state=42,
                verbose=-1
            ),
            'xgb': xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                verbosity=0
            ),
            'catboost': CatBoostRegressor(
                iterations=100,
                learning_rate=0.05,
                depth=6,
                random_state=42,
                verbose=False
            ),
            'rf': RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
        }

        # 训练基础模型
        logger.info("\n[Layer 1] 训练4个基础模型...")
        for name, model in base_models.items():
            logger.info(f"  训练 {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model

        # Layer 2: 元学习器
        logger.info("\n[Layer 2] 训练元学习器...")
        meta_features_train = np.column_stack([
            model.predict(X_train) for model in base_models.values()
        ])

        self.meta_model = lgb.LGBMRegressor(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
            verbose=-1
        )
        self.meta_model.fit(meta_features_train, y_train)

        logger.info("✅ 模型训练完成")

    def predict(self, X):
        """使用Ensemble预测"""
        # Layer 1预测
        base_predictions = np.column_stack([
            model.predict(X) for model in self.models.values()
        ])

        # Layer 2元学习
        final_predictions = self.meta_model.predict(base_predictions)

        return final_predictions

    def comprehensive_evaluation(self, X_test, y_test):
        """使用ModelEvaluator进行完整评估"""
        logger.info("\n" + "="*80)
        logger.info("📊 开始完整模型评估...")
        logger.info("="*80)

        # 预测
        y_pred = self.predict(X_test)

        # 使用ModelEvaluator评估
        evaluator = ModelEvaluator(y_test, y_pred)
        results = evaluator.print_report()

        return results

    def save_model(self, filepath='models/v390_comprehensive.pkl'):
        """保存模型"""
        model_data = {
            'models': self.models,
            'meta_model': self.meta_model,
            'timestamp': datetime.now().isoformat()
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"✅ 模型已保存: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='V3.9完整训练和评估')
    parser.add_argument('--start-date', type=str, default=None,
                       help='开始日期 (YYYY-MM-DD), 默认使用所有数据')
    parser.add_argument('--end-date', type=str, default=None,
                       help='结束日期 (YYYY-MM-DD), 默认使用所有数据')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='测试集比例 (默认0.2)')
    parser.add_argument('--output', type=str, default='models/v390_comprehensive.pkl',
                       help='模型保存路径')

    args = parser.parse_args()

    # 初始化
    trainer = V390ComprehensiveTrainer()

    # 加载数据
    date_filter = None
    if args.start_date and args.end_date:
        date_filter = {'start': args.start_date, 'end': args.end_date}

    X, y, dates = trainer.load_cached_features(date_filter)

    # 按时间划分训练集和测试集
    logger.info(f"\n📊 划分数据集 (测试集比例: {args.test_size*100:.0f}%)")

    # 按日期排序，最新的数据作为测试集
    sorted_indices = dates.argsort()
    split_idx = int(len(sorted_indices) * (1 - args.test_size))

    train_indices = sorted_indices[:split_idx]
    test_indices = sorted_indices[split_idx:]

    X_train, y_train = X.iloc[train_indices], y[train_indices]
    X_test, y_test = X.iloc[test_indices], y[test_indices]
    test_dates = dates.iloc[test_indices]

    logger.info(f"  训练集: {len(X_train):,} 样本 ({dates.iloc[train_indices].min()} ~ {dates.iloc[train_indices].max()})")
    logger.info(f"  测试集: {len(X_test):,} 样本 ({test_dates.min()} ~ {test_dates.max()})")

    # 训练模型
    trainer.train_models(X_train, y_train)

    # 完整评估
    results = trainer.comprehensive_evaluation(X_test, y_test)

    # 保存模型
    grade = results['comprehensive']['grade']
    score = results['comprehensive']['score']

    logger.info("\n" + "="*80)
    logger.info(f"📈 最终评分: {score:.1f}/100  等级: {grade}")
    logger.info("="*80)

    if grade in ['A', 'B']:
        logger.info("✅ 模型通过验收，保存模型文件")
        trainer.save_model(args.output)
        logger.info(f"\n💡 建议: 模型达到{grade}级，可以考虑实盘测试（建议从小仓位开始）")
    elif grade == 'C':
        logger.info("🟡 模型勉强及格，保存模型文件")
        trainer.save_model(args.output)
        logger.info(f"\n💡 建议: 模型达到{grade}级，建议先模拟盘测试1-2个月")
    else:
        logger.info("❌ 模型未通过验收，不保存模型")
        logger.info(f"\n💡 建议: 模型仅达到{grade}级，需要继续优化")
        logger.info("   可能的改进方向:")
        logger.info("   1. 增加近期数据的采样权重")
        logger.info("   2. 调整特征工程")
        logger.info("   3. 优化超参数")
        logger.info("   4. 检查数据质量")

    return results


if __name__ == '__main__':
    results = main()
