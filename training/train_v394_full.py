#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9.4 完整模型训练脚本

合并 v3.9.0 的 42 个特征 + 6 个活跃市值特征 = 48 个特征

数据来源：
- v39_feature_cache: v3.9.0 的 42 个技术/基本面/市场特征
- active_mv_feature_cache: 6 个活跃市值特征

训练目标：
1. 评估活跃市值特征对模型性能的提升
2. 与 v3.9.0 进行 A/B 对比

作者: Claude Code
创建时间: 2025-11-28
"""

import numpy as np
import pandas as pd
import sqlite3
import json
from datetime import datetime
import logging
from pathlib import Path
import joblib
from tqdm import tqdm

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')


def load_merged_features():
    """
    加载并合并 v3.9.0 特征和活跃市值特征
    """
    conn = sqlite3.connect(DB_PATH)

    logger.info("加载并合并特征数据...")

    # 加载 v3.9.0 特征
    logger.info("  加载 v39_feature_cache...")
    v39_df = pd.read_sql("""
        SELECT code, trade_date, features_json, label_5d
        FROM v39_feature_cache
        WHERE trade_date >= '2025-01-01'
    """, conn)

    logger.info(f"  v39 数据: {len(v39_df)} 条")

    # 解析 JSON 特征
    logger.info("  解析 JSON 特征...")
    features_list = []
    for _, row in tqdm(v39_df.iterrows(), total=len(v39_df), desc="解析v39特征"):
        try:
            features = json.loads(row['features_json'])
            features['code'] = row['code']
            features['trade_date'] = row['trade_date']
            features['label'] = row['label_5d']
            features_list.append(features)
        except:
            continue

    v39_features = pd.DataFrame(features_list)
    logger.info(f"  解析完成: {len(v39_features)} 条, {len(v39_features.columns) - 3} 个特征")

    # 加载活跃市值特征
    logger.info("  加载 active_mv_feature_cache...")
    active_mv_df = pd.read_sql("""
        SELECT
            code, trade_date,
            market_active_mv_ratio,
            market_active_mv_zscore,
            market_active_mv_trend,
            stock_active_mv_rank,
            stock_relative_liquidity,
            market_cap_quality_score
        FROM active_mv_feature_cache
    """, conn)

    logger.info(f"  活跃市值数据: {len(active_mv_df)} 条")

    # 合并特征
    logger.info("  合并特征...")
    merged_df = v39_features.merge(
        active_mv_df,
        on=['code', 'trade_date'],
        how='inner'
    )

    logger.info(f"  合并完成: {len(merged_df)} 条, {len(merged_df.columns) - 3} 个特征")

    conn.close()

    return merged_df


def prepare_train_test_split(df, test_start='2025-10-01'):
    """
    按时间划分训练集和测试集
    """
    df['trade_date'] = pd.to_datetime(df['trade_date'])

    train_df = df[df['trade_date'] < test_start]
    test_df = df[df['trade_date'] >= test_start]

    logger.info(f"训练集: {len(train_df)} ({train_df['trade_date'].min()} ~ {train_df['trade_date'].max()})")
    logger.info(f"测试集: {len(test_df)} ({test_df['trade_date'].min()} ~ {test_df['trade_date'].max()})")

    return train_df, test_df


def train_model(X_train, y_train, X_val=None, y_val=None):
    """
    训练 LightGBM 模型
    """
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 50,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }

    train_data = lgb.Dataset(X_train, label=y_train)

    callbacks = [
        lgb.log_evaluation(period=100)
    ]

    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val)
        callbacks.append(lgb.early_stopping(stopping_rounds=50))
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )
    else:
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data],
            callbacks=callbacks
        )

    return model


def evaluate_model(model, X_test, y_test, test_info):
    """
    评估模型性能
    """
    y_pred = model.predict(X_test)

    # 基础指标
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    ic = np.corrcoef(y_test, y_pred)[0, 1]
    direction_acc = ((y_test > 0) == (y_pred > 0)).mean()

    # Top 20 分析
    test_result = test_info.copy()
    test_result['pred'] = y_pred
    test_result['actual'] = y_test.values

    top_returns = []
    for date, group in test_result.groupby('trade_date'):
        if len(group) >= 20:
            top_20 = group.nlargest(20, 'pred')
            top_returns.extend(top_20['actual'].tolist())

    top_20_mean = np.mean(top_returns) * 100 if top_returns else 0
    top_20_win_rate = (np.array(top_returns) > 0).mean() * 100 if top_returns else 0

    # 特征重要性
    importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    results = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'ic': ic,
        'direction_accuracy': direction_acc,
        'top_20_return': top_20_mean,
        'top_20_win_rate': top_20_win_rate,
        'feature_importance': importance
    }

    return results


def main():
    logger.info("=" * 70)
    logger.info("V3.9.4 完整模型训练 (42 + 6 = 48 特征)")
    logger.info("=" * 70)

    # 1. 加载数据
    df = load_merged_features()

    # 2. 划分训练测试集
    train_df, test_df = prepare_train_test_split(df, test_start='2025-10-01')

    # 3. 准备特征
    exclude_cols = ['code', 'trade_date', 'label']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # 定义活跃市值特征
    active_mv_features = [
        'market_active_mv_ratio', 'market_active_mv_zscore', 'market_active_mv_trend',
        'stock_active_mv_rank', 'stock_relative_liquidity', 'market_cap_quality_score'
    ]

    # v3.9.0 原有特征
    v39_features = [f for f in feature_cols if f not in active_mv_features]

    logger.info(f"\nv3.9.0 特征: {len(v39_features)} 个")
    logger.info(f"活跃市值特征: {len([f for f in active_mv_features if f in feature_cols])} 个")
    logger.info(f"总特征: {len(feature_cols)} 个")

    # 4. 准备训练数据
    X_train = train_df[feature_cols].fillna(0).astype(float)
    y_train = train_df['label'].astype(float)

    X_test = test_df[feature_cols].fillna(0).astype(float)
    y_test = test_df['label'].astype(float)
    test_info = test_df[['code', 'trade_date']].copy()

    # 验证集
    val_split = int(len(X_train) * 0.8)
    X_train_final = X_train.iloc[:val_split]
    y_train_final = y_train.iloc[:val_split]
    X_val = X_train.iloc[val_split:]
    y_val = y_train.iloc[val_split:]

    logger.info(f"\n训练集: {len(X_train_final)} 样本")
    logger.info(f"验证集: {len(X_val)} 样本")
    logger.info(f"测试集: {len(X_test)} 样本")

    # ============================================================
    # 5. 训练 v3.9.4 完整模型 (48 特征)
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("训练 V3.9.4 完整模型 (48 特征)")
    logger.info("=" * 70)

    model_v394 = train_model(X_train_final, y_train_final, X_val, y_val)
    results_v394 = evaluate_model(model_v394, X_test, y_test, test_info)

    logger.info(f"\nV3.9.4 评估结果:")
    logger.info(f"  RMSE: {results_v394['rmse']:.4f}")
    logger.info(f"  MAE: {results_v394['mae']:.4f}")
    logger.info(f"  R²: {results_v394['r2']:.4f}")
    logger.info(f"  IC: {results_v394['ic']:.4f}")
    logger.info(f"  方向准确率: {results_v394['direction_accuracy']*100:.2f}%")
    logger.info(f"  Top 20 平均收益: {results_v394['top_20_return']:.2f}%")
    logger.info(f"  Top 20 胜率: {results_v394['top_20_win_rate']:.2f}%")

    # ============================================================
    # 6. 训练 v3.9.0 基准模型 (42 特征) - 用于对比
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("训练 V3.9.0 基准模型 (42 特征) - 用于 A/B 对比")
    logger.info("=" * 70)

    X_train_v39 = X_train_final[v39_features]
    X_val_v39 = X_val[v39_features]
    X_test_v39 = X_test[v39_features]

    model_v390 = train_model(X_train_v39, y_train_final, X_val_v39, y_val)
    results_v390 = evaluate_model(model_v390, X_test_v39, y_test, test_info)

    logger.info(f"\nV3.9.0 评估结果:")
    logger.info(f"  RMSE: {results_v390['rmse']:.4f}")
    logger.info(f"  MAE: {results_v390['mae']:.4f}")
    logger.info(f"  R²: {results_v390['r2']:.4f}")
    logger.info(f"  IC: {results_v390['ic']:.4f}")
    logger.info(f"  方向准确率: {results_v390['direction_accuracy']*100:.2f}%")
    logger.info(f"  Top 20 平均收益: {results_v390['top_20_return']:.2f}%")
    logger.info(f"  Top 20 胜率: {results_v390['top_20_win_rate']:.2f}%")

    # ============================================================
    # 7. A/B 对比
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("A/B 对比: V3.9.4 vs V3.9.0")
    logger.info("=" * 70)

    comparison = pd.DataFrame({
        '指标': ['RMSE', 'MAE', 'R²', 'IC', '方向准确率', 'Top20收益', 'Top20胜率'],
        'V3.9.0': [
            f"{results_v390['rmse']:.4f}",
            f"{results_v390['mae']:.4f}",
            f"{results_v390['r2']:.4f}",
            f"{results_v390['ic']:.4f}",
            f"{results_v390['direction_accuracy']*100:.2f}%",
            f"{results_v390['top_20_return']:.2f}%",
            f"{results_v390['top_20_win_rate']:.2f}%"
        ],
        'V3.9.4': [
            f"{results_v394['rmse']:.4f}",
            f"{results_v394['mae']:.4f}",
            f"{results_v394['r2']:.4f}",
            f"{results_v394['ic']:.4f}",
            f"{results_v394['direction_accuracy']*100:.2f}%",
            f"{results_v394['top_20_return']:.2f}%",
            f"{results_v394['top_20_win_rate']:.2f}%"
        ],
        '变化': [
            f"{(results_v394['rmse'] - results_v390['rmse']):.4f}",
            f"{(results_v394['mae'] - results_v390['mae']):.4f}",
            f"{(results_v394['r2'] - results_v390['r2']):.4f}",
            f"{(results_v394['ic'] - results_v390['ic']):.4f}",
            f"{(results_v394['direction_accuracy'] - results_v390['direction_accuracy'])*100:.2f}%",
            f"{(results_v394['top_20_return'] - results_v390['top_20_return']):.2f}%",
            f"{(results_v394['top_20_win_rate'] - results_v390['top_20_win_rate']):.2f}%"
        ]
    })

    print("\n" + comparison.to_string(index=False))

    # 活跃市值特征重要性
    logger.info("\n活跃市值特征在 V3.9.4 中的重要性排名:")
    importance = results_v394['feature_importance']
    for feat in active_mv_features:
        if feat in importance['feature'].values:
            rank = importance[importance['feature'] == feat].index[0] + 1
            imp = importance[importance['feature'] == feat]['importance'].values[0]
            logger.info(f"  #{rank:2d} {feat}: {imp:.1f}")

    # Top 20 特征
    logger.info("\nV3.9.4 Top 20 特征:")
    for i, (_, row) in enumerate(importance.head(20).iterrows()):
        marker = "⭐" if row['feature'] in active_mv_features else "  "
        logger.info(f"  {marker} {i+1:2d}. {row['feature']}: {row['importance']:.1f}")

    # ============================================================
    # 8. 保存模型
    # ============================================================
    model_dir = PROJECT_ROOT / 'models' / 'v394'
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / 'v394_full_model.pkl'
    joblib.dump({
        'model': model_v394,
        'feature_cols': feature_cols,
        'v39_features': v39_features,
        'active_mv_features': active_mv_features,
        'results': results_v394,
        'comparison': comparison.to_dict(),
        'config': {
            'n_features': len(feature_cols),
            'n_train': len(X_train_final),
            'n_test': len(X_test)
        }
    }, model_path)

    logger.info(f"\n✅ V3.9.4 模型已保存: {model_path}")

    # 也保存 v3.9.0 基准模型
    baseline_path = model_dir / 'v390_baseline_model.pkl'
    joblib.dump({
        'model': model_v390,
        'feature_cols': v39_features,
        'results': results_v390
    }, baseline_path)

    logger.info(f"✅ V3.9.0 基准模型已保存: {baseline_path}")

    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
