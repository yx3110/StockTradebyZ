#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TrainingHistoryRecorder 集成示例

这个脚本演示如何在训练脚本中使用TrainingHistoryRecorder来记录训练过程，
使得webapp可以展示loss曲线等训练指标。
"""
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import logging

# 导入训练历史记录器
from ml_models.training_history_recorder import TrainingHistoryRecorder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def example_training_with_recorder():
    """
    示例：如何在训练过程中使用TrainingHistoryRecorder
    """
    # 生成示例数据
    X, y = make_regression(n_samples=10000, n_features=20, noise=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. 初始化记录器
    recorder = TrainingHistoryRecorder(
        model_version='v3.9',
        output_dir='models'
    )

    logger.info("开始训练...")

    # 2. 训练 LightGBM
    logger.info("训练 LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbosity=-1
    )

    # 使用eval_set记录训练过程
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=['training', 'valid_0'],
        callbacks=[lgb.log_evaluation(0)]  # 静默模式
    )

    # 从模型中提取训练历史
    recorder.record_lgb_training('lightgbm', lgb_model, metric_name='l2')

    # 3. 训练 XGBoost
    logger.info("训练 XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbosity=0
    )

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # 从模型中提取训练历史
    recorder.record_xgb_training('xgboost', xgb_model, metric_name='rmse')

    # 4. 训练 CatBoost
    logger.info("训练 CatBoost...")
    cat_model = CatBoostRegressor(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        random_state=42,
        verbose=False
    )

    cat_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False
    )

    # 从模型中提取训练历史
    recorder.record_catboost_training('catboost', cat_model, metric_name='RMSE')

    # 5. 设置训练汇总信息
    recorder.set_summary(
        training_samples=len(X_train),
        validation_samples=len(X_val),
        feature_count=X_train.shape[1],
        final_metrics={
            'train_rmse': 0.05,
            'val_rmse': 0.07,
            'ic': 0.08,
            'r2': 0.12
        },
        training_params={
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6
        }
    )

    # 6. 完成记录并保存
    recorder.finish(status='completed')

    logger.info("训练完成！训练历史已保存到 models/v39/training_history_latest.json")


def example_manual_recording():
    """
    示例：手动记录训练曲线（适用于自定义训练循环）
    """
    recorder = TrainingHistoryRecorder(
        model_version='v3.9',
        output_dir='models'
    )

    # 模拟训练过程
    train_losses = []
    val_losses = []

    for epoch in range(50):
        # 模拟loss下降
        train_loss = 0.1 * np.exp(-epoch / 20) + 0.05 + np.random.normal(0, 0.001)
        val_loss = 0.12 * np.exp(-epoch / 20) + 0.06 + np.random.normal(0, 0.002)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # 记录自定义模型的训练历史
    recorder.record_model_training(
        model_name='custom_model',
        train_losses=train_losses,
        val_losses=val_losses,
        metric_name='mse',
        best_iteration=35,
        additional_metrics={'learning_rate': 0.01}
    )

    recorder.set_summary(
        training_samples=8000,
        validation_samples=2000,
        feature_count=50,
        final_metrics={'val_mse': val_losses[-1]}
    )

    recorder.finish()


if __name__ == '__main__':
    # 运行示例
    example_training_with_recorder()
