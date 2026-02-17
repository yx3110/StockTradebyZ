#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9.2快速训练验证脚本
用5000样本快速验证Phase 2的15个方向预测特征

🎯 Phase 2优化目标:
- 综合评分: 67.1 → 70+ (+3分)
- 方向准确率: 57.26% → 60%+ (+2.74%)
- 特征数: 52 → 60 (52 - 7无用 + 15新增)

🚀 Phase 2新特征 (15个):
1. 动量增强 (5个): price_acceleration_5d, momentum_alignment, momentum_decay, normalized_momentum, momentum_persistence
2. 趋势转折 (4个): macd_price_divergence, rsi_reversal_strength, adx_change_rate, channel_position
3. 量价关系 (3个): volume_strength, volume_price_divergence, large_order_intensity
4. 波动率模式 (3个): volatility_trend, volatility_spike, volatility_reversion
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

from ml_models.v39.v390_enhanced_feature_ml_system import V390EnhancedFeatureMLSystem
from model_evaluator import ModelEvaluator
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

print("="*80)
print("V3.9.2快速训练验证 (Phase 2方向预测特征)")
print("="*80)
print("目标: 验证15个Phase 2特征是否提升方向预测能力")
print("方法: 5000样本快速训练 + 完整评估")
print("="*80)

# 1. 获取训练样本
print("\n[1/5] 获取训练样本...")
conn = sqlite3.connect('data_adapter/stock_data.db')

# 随机采样5000个(code, date)对，覆盖2025年数据
query = """
SELECT code, trade_date
FROM (
    SELECT DISTINCT s.code, dq.trade_date
    FROM securities s
    JOIN daily_quotes dq ON s.id = dq.security_id
    WHERE s.type = 'A股'
    AND dq.trade_date >= '2025-06-01'
    AND dq.trade_date <= '2025-10-28'
    ORDER BY RANDOM()
    LIMIT 5000
)
ORDER BY trade_date
"""

df_samples = pd.read_sql_query(query, conn)
conn.close()

print(f"  采样数: {len(df_samples)}")
print(f"  日期范围: {df_samples['trade_date'].min()} ~ {df_samples['trade_date'].max()}")

# 2. 实时提取v3.9.2特征(60个)
print("\n[2/5] 实时提取v3.9.2特征(60个: 52 Phase1 - 7无用 + 15 Phase2)...")
system = V390EnhancedFeatureMLSystem(config={
    'use_enhanced_features': True,   # 启用Phase 1增强特征 (10个)
    'use_phase2_features': True      # 启用Phase 2方向预测特征 (15个)
})

features_list = []
labels_list = []
failed = 0

from tqdm import tqdm
for idx, row in tqdm(df_samples.iterrows(), total=len(df_samples), desc="提取特征"):
    code = row['code']
    date = row['trade_date']

    try:
        # 提取特征
        features = system.extract_features(code, date)
        if features is None:
            failed += 1
            continue

        # 计算标签
        label = system.calculate_label(code, date)
        if label is None:
            failed += 1
            continue

        features_list.append(features.iloc[0])
        labels_list.append(label)

    except Exception as e:
        failed += 1
        continue

print(f"\n  成功: {len(features_list)}")
print(f"  失败: {failed}")

if len(features_list) < 1000:
    print("❌ 有效样本不足，退出")
    sys.exit(1)

# 转换为DataFrame
X = pd.DataFrame(features_list).reset_index(drop=True)
y = pd.Series(labels_list).reset_index(drop=True)

print(f"\n  特征矩阵: {X.shape}")
print(f"  标签向量: {y.shape}")
print(f"  特征数: {X.shape[1]} (v3.9.0=42, v3.9.1=52, v3.9.2=60)")

# 检查Phase 1增强特征
phase1_cols = [c for c in X.columns if any(kw in c for kw in [
    'momentum_5d', 'momentum_20d', 'momentum_strength',
    'relative_strength_to_market',
    'volatility_asymmetry', 'price_ma_ratio_squared', 'roe_momentum_interaction'
])]
print(f"\n  ✅ Phase 1特征: {len(phase1_cols)}个")
for col in phase1_cols:
    print(f"    - {col}")

# 检查Phase 2方向预测特征
phase2_cols = [c for c in X.columns if any(kw in c for kw in [
    'price_acceleration', 'momentum_alignment', 'momentum_decay', 'momentum_persistence',
    'normalized_momentum', 'macd_price_divergence', 'rsi_reversal_strength',
    'adx_change_rate', 'channel_position', 'volume_strength', 'volume_price_divergence',
    'large_order_intensity', 'volatility_trend', 'volatility_spike', 'volatility_reversion'
])]
print(f"\n  🚀 Phase 2特征: {len(phase2_cols)}个")
for col in phase2_cols:
    print(f"    - {col}")

# 验证被剔除的特征
excluded_features = [
    'limit_up_count', 'northbound_net_inflow', 'concept_heat_index', 'supertrend_signal',
    'relative_strength_to_industry', 'ma_alignment_score', 'volume_confirmation'
]
remaining_excluded = [f for f in excluded_features if f in X.columns]
if len(remaining_excluded) > 0:
    print(f"\n  ⚠️  警告: 以下特征应该被剔除但仍存在:")
    for feat in remaining_excluded:
        print(f"    - {feat}")
else:
    print(f"\n  ✅ 成功剔除7个无用特征")

# 3. 划分训练/测试集
print("\n[3/5] 划分训练/测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  训练集: {X_train.shape[0]}")
print(f"  测试集: {X_test.shape[0]}")

# 4. 训练v3.9.2模型
print("\n[4/5] 训练v3.9.2模型...")
model = LGBMRegressor(
    objective='regression',
    metric='rmse',
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=500,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    verbose=-1,
    random_state=42
)

model.fit(X_train, y_train)
print("  ✅ 训练完成")

# 5. 完整评估
print("\n[5/5] 使用ModelEvaluator进行完整评估...")
print("="*80)

y_pred = model.predict(X_test)
evaluator = ModelEvaluator(y_test.values, y_pred)
results = evaluator.print_report()

# 保存模型
model_path = 'models/v392_quick_test.pkl'
with open(model_path, 'wb') as f:
    pickle.dump({
        'model': model,
        'feature_names': X.columns.tolist(),
        'n_features': X.shape[1],
        'timestamp': datetime.now().isoformat(),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'evaluation': results,
        'phase1_features': phase1_cols,
        'phase2_features': phase2_cols
    }, f)

print(f"\n✅ 模型已保存: {model_path}")

# 特征重要性
print("\n" + "="*80)
print("Top 30 重要特征 (标注Phase 1/2特征)")
print("="*80)
feature_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_imp.head(30).iterrows():
    marker = ''
    if row['feature'] in phase1_cols:
        marker = '🅿️1'  # Phase 1
    elif row['feature'] in phase2_cols:
        marker = '🅿️2'  # Phase 2
    else:
        marker = '   '

    print(f"{marker} {row['feature']:45s} {row['importance']:.4f}")

# 统计各类特征的重要性
phase1_importance = feature_imp[feature_imp['feature'].isin(phase1_cols)]['importance'].sum()
phase2_importance = feature_imp[feature_imp['feature'].isin(phase2_cols)]['importance'].sum()
total_importance = feature_imp['importance'].sum()

print(f"\n特征重要性统计:")
print(f"  Phase 1特征重要性: {phase1_importance:.2f} ({phase1_importance/total_importance*100:.1f}%)")
print(f"  Phase 2特征重要性: {phase2_importance:.2f} ({phase2_importance/total_importance*100:.1f}%)")
print(f"  Phase 1+2数量占比: {(len(phase1_cols)+len(phase2_cols))/len(X.columns)*100:.1f}%")

print("\n" + "="*80)
print("快速验证完成！")
print("="*80)
print(f"对比 v3.9.1 (52特征, 67.1分, 57.26%方向准确率):")
print(f"  v3.9.2特征数: {X.shape[1]}")
print(f"  v3.9.2评分: {results['综合评分']:.1f}")
print(f"  方向准确率: {results['方向准确率']*100:.2f}%")
print(f"  IC: {results['IC']:.4f}")
print(f"  Top20平均收益: {results['Top 20平均收益']*100:.2f}%")
print(f"  Top20胜率: {results['Top 20胜率']*100:.1f}%")

# 判断是否达到目标
if results['综合评分'] >= 70 and results['方向准确率'] >= 0.60:
    print("\n🎉 恭喜! v3.9.2达到目标，可以启动完整训练！")
elif results['综合评分'] >= 70:
    print("\n✅ 综合评分达标70+，但方向准确率未达60%")
    print("   建议: 考虑微调Phase 2特征权重")
elif results['方向准确率'] >= 0.60:
    print("\n✅ 方向准确率达标60%+，但综合评分未达70")
    print("   建议: 继续优化或启动完整训练")
else:
    print("\n⚠️  v3.9.2未完全达到目标，但可能有改进")
    print("   建议: 分析特征重要性，考虑调整或重新设计部分特征")

print("="*80)
