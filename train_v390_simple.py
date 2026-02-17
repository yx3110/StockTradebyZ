#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用现有缓存训练V3.9.0完整版本 (简化版)
- 缓存: 291K样本, 2022-2025, 42个基础特征
- 目标: 验证基础版本性能，对比v3.9.1快速测试
- 方法: 与quick_train脚本保持一致，使用ModelEvaluator
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from model_evaluator import ModelEvaluator
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

print("="*80)
print("V3.9.0完整版本训练 (使用291K缓存数据)")
print("="*80)
print("数据源: v39_feature_cache表 (42个基础特征)")
print("样本量: 291,574 (2022-2025)")
print("="*80)

# 1. 加载缓存数据
print("\n[1/5] 从缓存加载数据...")
conn = sqlite3.connect('data_adapter/stock_data.db')

df_cache = pd.read_sql_query("""
SELECT code, trade_date, features_json, label_5d
FROM v39_feature_cache
WHERE label_5d IS NOT NULL
ORDER BY trade_date
""", conn)

print(f"  加载记录: {len(df_cache):,}")
print(f"  日期范围: {df_cache['trade_date'].min()} ~ {df_cache['trade_date'].max()}")
print(f"  涉及股票: {df_cache['code'].nunique()}")

conn.close()

# 2. 解析特征
print("\n[2/5] 解析特征...")
features_list = []
labels_list = []

from tqdm import tqdm
for idx, row in tqdm(df_cache.iterrows(), total=len(df_cache), desc="解析进度"):
    try:
        features = json.loads(row['features_json'])
        features_list.append(features)
        labels_list.append(row['label_5d'])
    except:
        continue

# 转换为DataFrame
X = pd.DataFrame(features_list).reset_index(drop=True)
y = pd.Series(labels_list).reset_index(drop=True)

print(f"\n  特征矩阵: {X.shape}")
print(f"  标签向量: {y.shape}")
print(f"  特征数: {X.shape[1]} (v3.9.0基础版)")

# 处理缺失值
print("\n  处理缺失值...")
X = X.fillna(X.median())

# 3. 划分训练/测试集
print("\n[3/5] 划分训练/测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
print(f"  训练集: {X_train.shape[0]:,}")
print(f"  测试集: {X_test.shape[0]:,}")

# 4. 训练v3.9.0模型
print("\n[4/5] 训练v3.9.0模型...")
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
model_path = 'models/v390_full_from_cache.pkl'
with open(model_path, 'wb') as f:
    pickle.dump({
        'model': model,
        'feature_names': X.columns.tolist(),
        'n_features': X.shape[1],
        'timestamp': datetime.now().isoformat(),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'evaluation': results,
        'version': 'v3.9.0',
        'data_source': 'v39_feature_cache (291K samples, 2022-2025)'
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
    print(f"{row['feature']:45s} {row['importance']:.4f}")

# 对比v3.9.1快速测试
print("\n" + "="*80)
print("版本对比")
print("="*80)

try:
    with open('models/v391_quick_test.pkl', 'rb') as f:
        v391 = pickle.load(f)

    v391_score = v391['evaluation']['综合评分']
    v391_dir_acc = v391['evaluation']['方向准确率']
    v391_samples = v391['test_samples']

    v390_score = results['综合评分']
    v390_dir_acc = results['方向准确率']

    print(f"{'版本':<15} {'样本量':<15} {'特征数':<10} {'综合评分':<15} {'方向准确率':<15}")
    print(f"{'-'*80}")
    print(f"v3.9.0 (完整)   {len(X_test):,}         42个       {v390_score:.1f}/100        {v390_dir_acc*100:.2f}%")
    print(f"v3.9.1 (快速)   {v391_samples:,}           52个       {v391_score:.1f}/100        {v391_dir_acc*100:.2f}%")

    print(f"\n差异分析:")
    print(f"  综合评分: {v390_score:.1f} vs {v391_score:.1f} ({v391_score-v390_score:+.1f}分)")
    print(f"  方向准确率: {v390_dir_acc*100:.2f}% vs {v391_dir_acc*100:.2f}% ({(v391_dir_acc-v390_dir_acc)*100:+.2f}%)")
    print(f"  样本量: {len(X_test):,} vs {v391_samples:,} (多{len(X_test)-v391_samples:,}个)")

    print(f"\n结论:")
    if v390_score >= v391_score:
        print(f"  ⚠️  v3.9.0 (42特征) 不弱于 v3.9.1 (52特征)")
        print(f"  说明: Phase 1增强特征可能收益有限，需要更多分析")
    else:
        print(f"  ✅ v3.9.1 (52特征) 优于 v3.9.0 (42特征)")
        print(f"  提升: +{v391_score-v390_score:.1f}分, +{(v391_dir_acc-v390_dir_acc)*100:.2f}%方向准确率")

except Exception as e:
    print(f"  无法加载v3.9.1模型进行对比: {e}")

print("="*80)
print("V3.9.0完整版本训练完成!")
print("="*80)
