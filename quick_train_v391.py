#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9.1快速训练验证脚本
用5000样本快速验证新增10个特征的效果
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
print("V3.9.1快速训练验证")
print("="*80)
print("目标: 验证新增10个特征是否提升性能")
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

# 2. 实时提取v3.9.1特征(52个)
print("\n[2/5] 实时提取v3.9.1特征(52个)...")
system = V390EnhancedFeatureMLSystem(config={'use_enhanced_features': True})

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
print(f"  特征数: {X.shape[1]} (v3.9.0=42, v3.9.1=52)")

# 检查增强特征
enhanced_cols = [c for c in X.columns if any(kw in c for kw in [
    'momentum', 'relative', 'alignment', 'confirmation', 'asymmetry', 'interaction'
])]
print(f"\n  ✅ 增强特征: {len(enhanced_cols)}个")
for col in enhanced_cols:
    print(f"    - {col}")

# 3. 划分训练/测试集
print("\n[3/5] 划分训练/测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  训练集: {X_train.shape[0]}")
print(f"  测试集: {X_test.shape[0]}")

# 4. 训练v3.9.1模型
print("\n[4/5] 训练v3.9.1模型...")
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
model_path = 'models/v391_quick_test.pkl'
with open(model_path, 'wb') as f:
    pickle.dump({
        'model': model,
        'feature_names': X.columns.tolist(),
        'n_features': X.shape[1],
        'timestamp': datetime.now().isoformat(),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'evaluation': results
    }, f)

print(f"\n✅ 模型已保存: {model_path}")

# 特征重要性
print("\n" + "="*80)
print("Top 20 重要特征")
print("="*80)
feature_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_imp.head(20).iterrows():
    is_enhanced = '🆕' if any(kw in row['feature'] for kw in [
        'momentum', 'relative', 'alignment', 'confirmation', 'asymmetry', 'interaction'
    ]) else '  '
    print(f"{is_enhanced} {row['feature']:40s} {row['importance']:.4f}")

# 统计增强特征的重要性
enhanced_importance = feature_imp[feature_imp['feature'].isin(enhanced_cols)]['importance'].sum()
total_importance = feature_imp['importance'].sum()
enhanced_pct = (enhanced_importance / total_importance) * 100

print(f"\n增强特征重要性占比: {enhanced_pct:.2f}%")
print(f"增强特征数量占比: {len(enhanced_cols)/len(X.columns)*100:.2f}%")

print("\n" + "="*80)
print("快速验证完成！")
print("="*80)
print(f"对比v3.9.0 (42特征, 61.7分):")
print(f"  v3.9.1特征数: {X.shape[1]}")
print(f"  v3.9.1评分: {results['综合评分']:.1f}")
print(f"  方向准确率: {results['方向准确率']*100:.2f}%")
print(f"  IC: {results['IC']:.4f}")
print("="*80)
