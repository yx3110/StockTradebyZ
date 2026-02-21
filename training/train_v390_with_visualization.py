#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9模型训练脚本 - 增强版（包含可视化）
特性：训练曲线、特征重要性、模型对比、残差分析等
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
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class V390VisualizedTrainer:
    """增强版V3.9训练器 - 包含完整可视化"""

    def __init__(self, db_path=None, output_dir=str(PROJECT_ROOT / 'reports' / 'v39_training'):
        self.db_path = db_path or str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.models = {}
        self.meta_model = None
        self.training_history = []
        self.feature_names = []

    def load_cached_features(self, min_samples=1000):
        """从数据库加载预计算的特征"""
        logger.info("="*80)
        logger.info("📥 从数据库加载预计算特征...")
        logger.info("="*80)

        conn = sqlite3.connect(self.db_path)
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
                features_dict = json.loads(row['features_json'])
                features_list.append(features_dict)
                labels.append(row['label_5d'])
            except Exception as e:
                continue

            if (idx + 1) % 10000 == 0:
                logger.info(f"  已处理: {idx+1:,}/{len(df):,} ({(idx+1)/len(df)*100:.1f}%)")

        X = pd.DataFrame(features_list)
        y = np.array(labels)

        self.feature_names = X.columns.tolist()

        logger.info(f"✅ 特征矩阵: {X.shape}")
        logger.info(f"✅ 特征列数: {X.shape[1]}")

        if X.isnull().any().any():
            logger.warning("⚠️  检测到缺失值，使用0填充")
            X = X.fillna(0)

        return X, y

    def plot_learning_curves(self, model, model_name, X_train, y_train):
        """绘制学习曲线"""
        logger.info(f"  📈 绘制{model_name}学习曲线...")

        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            cv=3,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='neg_mean_squared_error',
            random_state=42
        )

        train_scores_mean = -train_scores.mean(axis=1)
        val_scores_mean = -val_scores.mean(axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores_mean, 'o-', label='训练集MSE', linewidth=2)
        plt.plot(train_sizes, val_scores_mean, 'o-', label='验证集MSE', linewidth=2)
        plt.xlabel('训练样本数', fontsize=12)
        plt.ylabel('MSE', fontsize=12)
        plt.title(f'{model_name} - 学习曲线', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = self.output_dir / f'learning_curve_{model_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  ✅ 保存至: {save_path}")

    def plot_feature_importance(self, model, model_name):
        """绘制特征重要性"""
        logger.info(f"  📊 绘制{model_name}特征重要性...")

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            logger.warning(f"  ⚠️  {model_name}没有feature_importances_属性")
            return

        # 获取Top 20特征
        indices = np.argsort(importances)[::-1][:20]
        top_features = [self.feature_names[i] for i in indices]
        top_importances = importances[indices]

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_importances, align='center')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('重要性', fontsize=12)
        plt.title(f'{model_name} - Top 20 特征重要性', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        save_path = self.output_dir / f'feature_importance_{model_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  ✅ 保存至: {save_path}")

    def plot_predictions_vs_actual(self, y_true, y_pred, model_name):
        """绘制预测vs实际散点图"""
        logger.info(f"  📉 绘制{model_name}预测vs实际...")

        plt.figure(figsize=(10, 10))
        plt.scatter(y_true, y_pred, alpha=0.3, s=10)

        # 完美预测线
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')

        plt.xlabel('实际收益率', fontsize=12)
        plt.ylabel('预测收益率', fontsize=12)
        plt.title(f'{model_name} - 预测 vs 实际', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = self.output_dir / f'predictions_{model_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  ✅ 保存至: {save_path}")

    def plot_residuals(self, y_true, y_pred, model_name):
        """绘制残差分布"""
        logger.info(f"  📊 绘制{model_name}残差分布...")

        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 残差直方图
        axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('残差', fontsize=12)
        axes[0].set_ylabel('频数', fontsize=12)
        axes[0].set_title(f'{model_name} - 残差分布', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # 残差vs预测值
        axes[1].scatter(y_pred, residuals, alpha=0.3, s=10)
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('预测值', fontsize=12)
        axes[1].set_ylabel('残差', fontsize=12)
        axes[1].set_title(f'{model_name} - 残差 vs 预测值', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / f'residuals_{model_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  ✅ 保存至: {save_path}")

    def train_base_models(self, X_train, y_train, X_val, y_val, enable_visualization=True):
        """训练基础模型并可视化"""
        logger.info("\n" + "="*80)
        logger.info("🔧 训练基础模型...")
        logger.info("="*80)

        models_config = {
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=6,
                num_leaves=31, min_child_samples=20, subsample=0.8,
                colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=-1
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=6,
                min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbosity=0
            ),
            'catboost': CatBoostRegressor(
                iterations=200, learning_rate=0.05, depth=6,
                l2_leaf_reg=3, random_state=42, verbose=False
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_split=10,
                min_samples_leaf=5, random_state=42, n_jobs=-1
            )
        }

        performance_summary = []

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

            performance_summary.append({
                'model': name,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'time': elapsed
            })

            self.models[name] = model

            # 可视化
            if enable_visualization:
                self.plot_learning_curves(model, name, X_train, y_train)
                self.plot_feature_importance(model, name)
                self.plot_predictions_vs_actual(y_val, y_pred, name)
                self.plot_residuals(y_val, y_pred, name)

        # 模型对比图
        if enable_visualization:
            self.plot_models_comparison(performance_summary)

        return self.models

    def plot_models_comparison(self, performance_summary):
        """绘制模型性能对比图"""
        logger.info("\n📊 绘制模型性能对比...")

        df = pd.DataFrame(performance_summary)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # MSE对比
        axes[0, 0].bar(df['model'], df['mse'], color='skyblue', edgecolor='black')
        axes[0, 0].set_ylabel('MSE', fontsize=12)
        axes[0, 0].set_title('MSE 对比 (越低越好)', fontsize=14, fontweight='bold')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        # MAE对比
        axes[0, 1].bar(df['model'], df['mae'], color='lightcoral', edgecolor='black')
        axes[0, 1].set_ylabel('MAE', fontsize=12)
        axes[0, 1].set_title('MAE 对比 (越低越好)', fontsize=14, fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # R²对比
        axes[1, 0].bar(df['model'], df['r2'], color='lightgreen', edgecolor='black')
        axes[1, 0].set_ylabel('R²', fontsize=12)
        axes[1, 0].set_title('R² 对比 (越高越好)', fontsize=14, fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # 训练时间对比
        axes[1, 1].bar(df['model'], df['time'], color='plum', edgecolor='black')
        axes[1, 1].set_ylabel('时间 (秒)', fontsize=12)
        axes[1, 1].set_title('训练时间对比', fontsize=14, fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        save_path = self.output_dir / 'models_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"✅ 保存至: {save_path}")

    def train_meta_model(self, X_train, y_train, X_val, y_val, enable_visualization=True):
        """训练元模型"""
        logger.info("\n" + "="*80)
        logger.info("🔧 训练元模型 (Stacking)...")
        logger.info("="*80)

        meta_features_train = np.column_stack([
            model.predict(X_train) for model in self.models.values()
        ])

        meta_features_val = np.column_stack([
            model.predict(X_val) for model in self.models.values()
        ])

        self.meta_model = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        )

        logger.info("🔹 训练Gradient Boosting元模型...")
        self.meta_model.fit(meta_features_train, y_train)

        y_pred = self.meta_model.predict(meta_features_val)
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        logger.info(f"✅ 元模型: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.4f}")

        if enable_visualization:
            self.plot_predictions_vs_actual(y_val, y_pred, 'meta_model')
            self.plot_residuals(y_val, y_pred, 'meta_model')

        return self.meta_model

    def save_model(self, output_path='ml_models/trained_models/v39'):
        """保存模型"""
        logger.info("\n" + "="*80)
        logger.info("💾 保存模型...")
        logger.info("="*80)

        Path(output_path).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        for name, model in self.models.items():
            model_path = f"{output_path}/v390_{name}_{timestamp}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"  ✅ {name}: {model_path}")

        meta_path = f"{output_path}/v390_meta_{timestamp}.pkl"
        with open(meta_path, 'wb') as f:
            pickle.dump(self.meta_model, f)
        logger.info(f"  ✅ meta_model: {meta_path}")

        full_model = {
            'base_models': self.models,
            'meta_model': self.meta_model,
            'timestamp': timestamp,
            'feature_names': self.feature_names
        }
        full_path = f"{output_path}/v390_full_system_{timestamp}.pkl"
        with open(full_path, 'wb') as f:
            pickle.dump(full_model, f)
        logger.info(f"  ✅ full_system: {full_path}")

        return full_path

    def train(self, test_size=0.2, random_state=42, enable_visualization=True):
        """完整训练流程"""
        X, y = self.load_cached_features()

        logger.info("\n" + "="*80)
        logger.info("📊 划分训练/验证集...")
        logger.info("="*80)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )

        logger.info(f"  训练集: {X_train.shape[0]:,} 样本")
        logger.info(f"  验证集: {X_val.shape[0]:,} 样本")

        self.train_base_models(X_train, y_train, X_val, y_val, enable_visualization)
        self.train_meta_model(X_train, y_train, X_val, y_val, enable_visualization)
        model_path = self.save_model()

        logger.info("\n" + "="*80)
        logger.info("🎉 训练完成!")
        logger.info(f"📊 可视化报告保存至: {self.output_dir}")
        logger.info("="*80)

        return model_path


def main():
    parser = argparse.ArgumentParser(description='V3.9模型训练（可视化版）')
    parser.add_argument('--db-path', type=str, default=str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db'))
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default=str(PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v39'))
    parser.add_argument('--viz-dir', type=str, default=str(PROJECT_ROOT / 'reports' / 'v39_training')
    parser.add_argument('--no-viz', action='store_true', help='禁用可视化')

    args = parser.parse_args()

    trainer = V390VisualizedTrainer(db_path=args.db_path, output_dir=args.viz_dir)
    model_path = trainer.train(
        test_size=args.test_size,
        random_state=args.random_state,
        enable_visualization=not args.no_viz
    )

    logger.info(f"\n✅ 模型已保存至: {model_path}")
    logger.info(f"📊 可视化报告: {args.viz_dir}")


if __name__ == "__main__":
    main()
