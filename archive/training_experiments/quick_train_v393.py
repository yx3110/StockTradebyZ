#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9.3快速训练验证脚本 (Phase 3精简版)
用5000样本验证精简策略：只保留5个优质Phase 2特征

🎯 v3.9.3优化目标:
- 特征数: 60 → 50 (-10个噪音特征)
- 综合评分: 55.2 → 63-66/100 (目标)
- 方向准确率: 保持58.78%或更高

✅ 保留的5个Phase 2特征:
1. price_acceleration_5d (#2, importance=439.0)
2. volume_strength (#6, importance=404.0)
3. volatility_trend (#33, importance=276.0)
4. momentum_decay (#35, importance=252.0)
5. normalized_momentum (#45, importance=160.0)

❌ 剔除的10个Phase 2特征:
- macd_price_divergence (0.0)
- adx_change_rate (0.0)
- channel_position (0.0)
- large_order_intensity (0.0)
- rsi_reversal_strength (0.0)
- momentum_persistence (80.0)
- volume_price_divergence (26.0)
- momentum_alignment (15.0)
- volatility_spike (8.0)
- volatility_reversion (107.0)
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
print("V3.9.3快速训练验证 (Phase 3精简版)")
print("="*80)
print("策略: 保留5个优质Phase 2特征，剔除10个噪音特征")
print("目标: 综合评分 55.2 → 63-66/100")
print("="*80)

# 1. 获取训练样本
print("\n[1/5] 获取训练样本...")
conn = sqlite3.connect('data_adapter/stock_data.db')

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

# 2. 实时提取v3.9.3特征(50个)
print("\n[2/5] 实时提取v3.9.3特征(50个: Phase 1 + 5个优质Phase 2)...")
system = V390EnhancedFeatureMLSystem(config={
    'use_enhanced_features': True,   # 启用Phase 1增强特征
    'use_phase2_features': True,     # 启用Phase 2方向预测特征
    'use_phase3_refined': True       # 🆕 启用Phase 3精简 (剔除10个失败特征)
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
print(f"  特征数: {X.shape[1]} (v3.9.1=52, v3.9.2=60, v3.9.3=50)")

# 验证特征数量
if X.shape[1] != 50:
    print(f"⚠️  警告: 特征数 {X.shape[1]} != 预期50个")

# 检查保留的Phase 2特征
kept_phase2 = [c for c in X.columns if any(kw in c for kw in [
    'price_acceleration_5d', 'volume_strength', 'volatility_trend',
    'momentum_decay', 'normalized_momentum'
])]
print(f"\n  ✅ 保留的Phase 2特征 ({len(kept_phase2)}个):")
for col in kept_phase2:
    print(f"    - {col}")

# 验证剔除的特征
excluded_phase2 = [
    'macd_price_divergence', 'adx_change_rate', 'channel_position',
    'large_order_intensity', 'rsi_reversal_strength', 'momentum_persistence',
    'volume_price_divergence', 'momentum_alignment', 'volatility_spike',
    'volatility_reversion'
]
remaining_excluded = [f for f in excluded_phase2 if f in X.columns]
if len(remaining_excluded) > 0:
    print(f"\n  ⚠️  警告: 以下特征应该被剔除但仍存在:")
    for feat in remaining_excluded:
        print(f"    - {feat}")
else:
    print(f"\n  ✅ 成功剔除10个失败的Phase 2特征")

# 3. 划分训练/测试集
print("\n[3/5] 划分训练/测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  训练集: {X_train.shape[0]}")
print(f"  测试集: {X_test.shape[0]}")

# 4. 训练v3.9.3模型
print("\n[4/5] 训练v3.9.3模型...")
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
model_path = 'ml_models/trained_models/v393_quick_test.pkl'
with open(model_path, 'wb') as f:
    pickle.dump({
        'model': model,
        'feature_names': X.columns.tolist(),
        'n_features': X.shape[1],
        'timestamp': datetime.now().isoformat(),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'evaluation': results,
        'kept_phase2_features': kept_phase2,
        'version': 'v3.9.3'
    }, f)

print(f"\n✅ 模型已保存: {model_path}")

# 特征重要性
print("\n" + "="*80)
print("Top 30 重要特征")
print("="*80)
feature_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_imp.head(30).iterrows():
    marker = ''
    if row['feature'] in kept_phase2:
        marker = '🅿️2✅'  # 保留的Phase 2特征
    else:
        marker = '     '

    print(f"{marker} {row['feature']:45s} {row['importance']:.4f}")

# 统计Phase 2特征重要性
phase2_importance = feature_imp[feature_imp['feature'].isin(kept_phase2)]['importance'].sum()
total_importance = feature_imp['importance'].sum()

print(f"\n保留Phase 2特征统计:")
print(f"  特征数: {len(kept_phase2)}个 (占 {len(kept_phase2)/len(X.columns)*100:.1f}%)")
print(f"  重要性: {phase2_importance:.2f} (占 {phase2_importance/total_importance*100:.1f}%)")

# 对比报告
print("\n" + "="*80)
print("版本对比")
print("="*80)

# 读取v3.9.1和v3.9.2结果
try:
    with open('ml_models/trained_models/v391_quick_test.pkl', 'rb') as f:
        v391 = pickle.load(f)
    with open('ml_models/trained_models/v392_quick_test.pkl', 'rb') as f:
        v392 = pickle.load(f)

    v391_score = v391['evaluation']['综合评分']
    v391_dir_acc = v391['evaluation']['方向准确率']
    v392_score = v392['evaluation']['综合评分']
    v392_dir_acc = v392['evaluation']['方向准确率']
    v393_score = results['综合评分']
    v393_dir_acc = results['方向准确率']

    print(f"{'版本':<10} {'特征数':<8} {'综合评分':<12} {'方向准确率':<12} {'备注'}")
    print(f"{'-'*70}")
    print(f"v3.9.1     52个     {v391_score:.1f}/100     {v391_dir_acc*100:.2f}%      Phase 1基准")
    print(f"v3.9.2     60个     {v392_score:.1f}/100     {v392_dir_acc*100:.2f}%      Phase 2失败 ❌")
    print(f"v3.9.3     {X.shape[1]}个     {v393_score:.1f}/100     {v393_dir_acc*100:.2f}%      Phase 3精简")

    print(f"\nv3.9.3 vs v3.9.1:")
    print(f"  综合评分: {v391_score:.1f} → {v393_score:.1f} ({v393_score-v391_score:+.1f})")
    print(f"  方向准确率: {v391_dir_acc*100:.2f}% → {v393_dir_acc*100:.2f}% ({(v393_dir_acc-v391_dir_acc)*100:+.2f}%)")

    print(f"\nv3.9.3 vs v3.9.2:")
    print(f"  综合评分: {v392_score:.1f} → {v393_score:.1f} ({v393_score-v392_score:+.1f})")
    print(f"  方向准确率: {v392_dir_acc*100:.2f}% → {v393_dir_acc*100:.2f}% ({(v393_dir_acc-v392_dir_acc)*100:+.2f}%)")

    # 目标达成判断
    print(f"\n" + "="*80)
    if v393_score >= 63 and v393_dir_acc >= 0.57:
        print("🎉 v3.9.3达到预期目标！")
        print(f"  ✅ 综合评分: {v393_score:.1f} >= 63.0")
        print(f"  ✅ 方向准确率: {v393_dir_acc*100:.2f}% >= 57%")
        print("\n建议: 可以考虑启动完整训练 (291K数据)")
    elif v393_score > v391_score:
        print("✅ v3.9.3优于v3.9.1基准")
        print(f"  提升: +{v393_score-v391_score:.1f}分")
        print("\n建议: Phase 3精简策略有效，可以继续优化")
    else:
        print("⚠️  v3.9.3未超越v3.9.1基准")
        print("\n建议: 考虑其他优化策略或直接使用v3.9.1")

except Exception as e:
    print(f"无法加载历史模型进行对比: {e}")

print("="*80)
