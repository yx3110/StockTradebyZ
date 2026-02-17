#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速验证v3.9模型
测试模型加载和评分功能
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from ml_models.v39 import V390EnhancedFeatureMLSystem

print("\n" + "="*80)
print("🔍 V3.9模型快速验证")
print("="*80)

# 1. 初始化系统
print("\n📦 Step 1: 初始化V3.9系统...")
system = V390EnhancedFeatureMLSystem()

# 2. 加载模型
print("\n💾 Step 2: 加载模型...")
model_path = "models/v39/v390_model_20251104.pkl"
try:
    system.load_model(model_path)
    print(f"✅ 模型加载成功: {model_path}")
    print(f"   特征数量: {len(system.feature_names)}")
    print(f"   模型架构: Layer1 (4个基础模型) + Layer2 (Meta) + Layer3 (Ensemble)")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    sys.exit(1)

# 3. 测试评分
print("\n🔮 Step 3: 测试评分功能...")
test_stocks = [
    ('000001.SZ', '平安银行'),
    ('600000.SH', '浦发银行'),
    ('000002.SZ', '万科A'),
    ('600036.SH', '招商银行'),
    ('000858.SZ', '五粮液')
]

test_date = '2025-10-31'
success_count = 0
scores = []

print(f"测试日期: {test_date}")
print(f"测试股票数: {len(test_stocks)}")
print()

for code, name in test_stocks:
    try:
        result = system.score_stock(code, test_date)
        if result and 'score' in result:
            score = result['score']
            scores.append(score)
            print(f"✅ {code} ({name}): {score:.2f}/100")
            success_count += 1
        else:
            print(f"❌ {code} ({name}): 评分失败 (无结果)")
    except Exception as e:
        print(f"❌ {code} ({name}): 评分失败 - {e}")

# 4. 分析结果
print("\n" + "="*80)
print("📊 验证结果汇总")
print("="*80)
print(f"成功率: {success_count}/{len(test_stocks)} ({success_count/len(test_stocks)*100:.1f}%)")

if scores:
    import numpy as np
    print(f"\n评分统计:")
    print(f"   均值: {np.mean(scores):.2f}")
    print(f"   标准差: {np.std(scores):.2f}")
    print(f"   最小值: {np.min(scores):.2f}")
    print(f"   最大值: {np.max(scores):.2f}")
    print(f"   中位数: {np.median(scores):.2f}")

    # 检查范围
    if np.min(scores) >= 0 and np.max(scores) <= 100:
        print(f"\n✅ 评分范围正常 [0, 100]")
    else:
        print(f"\n⚠️  评分超出范围!")

    # 检查分布
    if np.std(scores) > 5:
        print(f"✅ 评分分布合理 (标准差={np.std(scores):.2f})")
    else:
        print(f"⚠️  评分过于集中 (标准差={np.std(scores):.2f})")

# 5. 特征重要性
print("\n📊 Top 10 特征重要性 (LightGBM):")
if system.feature_importance and 'layer1_lgb' in system.feature_importance:
    importance_dict = system.feature_importance['layer1_lgb']
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    for rank, (feat, imp) in enumerate(sorted_features, 1):
        print(f"   {rank:2d}. {feat:30s}: {imp:.0f}")

print("\n" + "="*80)
if success_count >= len(test_stocks) * 0.8:
    print("🎉 验证通过！模型功能正常")
    print("="*80)
    sys.exit(0)
else:
    print("⚠️  验证部分失败，需要进一步检查")
    print("="*80)
    sys.exit(1)
