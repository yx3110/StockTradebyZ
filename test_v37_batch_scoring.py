#!/usr/bin/env python3
"""
测试V3.7批量评分修复
"""
import sys
sys.path.insert(0, '/Users/yangxu/StockTradebyZ')

from ml_models.v37.v370_advanced_ml_system import V370AdvancedMLSystem
import logging

logging.basicConfig(level=logging.INFO)

# 初始化V3.7系统
v37 = V370AdvancedMLSystem(auto_load_model=False)

# 加载模型
model_path = 'models/v370/v370_quality_optimized_20250923_224054.pkl'
v37.load_models(model_path)

print("\n" + "="*60)
print("🧪 测试V3.7批量评分修复")
print("="*60)

# 测试单只股票
print("\n📌 测试1: 单只股票")
print("-"*60)
test_stocks_single = ['000001']
features_single = v37.extract_advanced_features(
    test_stocks_single,
    start_date='2025-09-23',
    end_date='2025-09-23'
)

if features_single is not None and not features_single.empty:
    result_single = v37.predict_three_layer_ensemble(
        features_single,
        target_col='target_1d'
    )
    print(f"✅ 单只股票结果类型: {type(result_single)}")
    if isinstance(result_single, dict):
        print(f"✅ 返回字典，包含score: {result_single.get('score', 'N/A'):.2f}")
    else:
        print(f"❌ 预期返回dict，实际返回: {type(result_single)}")
else:
    print("❌ 特征提取失败")

# 测试多只股票
print("\n📌 测试2: 三只股票")
print("-"*60)
test_stocks_multi = ['000001', '000002', '600036']
features_multi = v37.extract_advanced_features(
    test_stocks_multi,
    start_date='2025-09-23',
    end_date='2025-09-23'
)

if features_multi is not None and not features_multi.empty:
    print(f"📊 提取了 {len(features_multi)} 条特征记录")
    result_multi = v37.predict_three_layer_ensemble(
        features_multi,
        target_col='target_1d'
    )
    print(f"✅ 批量结果类型: {type(result_multi)}")

    import numpy as np
    if isinstance(result_multi, np.ndarray):
        print(f"✅ 返回numpy数组，长度: {len(result_multi)}")
        print(f"📊 评分:")
        for i, stock in enumerate(test_stocks_multi):
            if i < len(result_multi):
                print(f"  {stock}: {result_multi[i]:.2f}")
            else:
                print(f"  {stock}: ❌ 缺失")
    elif isinstance(result_multi, dict):
        print(f"❌ 预期返回numpy array，实际返回dict: {result_multi}")
    else:
        print(f"❌ 预期返回numpy array，实际返回: {type(result_multi)}")
else:
    print("❌ 特征提取失败")

print("\n" + "="*60)
print("🎉 测试完成")
print("="*60)