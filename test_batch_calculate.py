#!/usr/bin/env python3
"""
测试batch_calculate_v37_scores函数
"""
import sys
sys.path.insert(0, '/Users/yangxu/StockTradebyZ')

from ml_models.v37.backtest_v37_engine_optimized import batch_calculate_v37_scores
import logging

logging.basicConfig(level=logging.INFO)

print("\n" + "="*60)
print("🧪 测试batch_calculate_v37_scores修复")
print("="*60)

# 测试参数
test_stocks = ['000001', '000002', '600036']
date = '2025-09-23'
model_path = 'models/v370/v370_quality_optimized_20250923_224054.pkl'

print(f"\n📌 测试股票: {test_stocks}")
print(f"📌 日期: {date}")
print(f"📌 模型: {model_path}")
print("-"*60)

# 调用函数
results = batch_calculate_v37_scores(test_stocks, date, model_path)

print(f"\n✅ 返回结果类型: {type(results)}")
print(f"✅ 返回结果数量: {len(results)}")
print(f"\n📊 评分详情:")
for stock_code, score in results:
    print(f"  {stock_code}: {score:.2f}")

print(f"\n平均分: {sum(s for _, s in results) / len(results):.2f}" if results else "N/A")
print(f"最高分: {max(s for _, s in results):.2f}" if results else "N/A")
print(f"最低分: {min(s for _, s in results):.2f}" if results else "N/A")

# 验证是否每个股票都有且只有一个评分
stock_codes = [code for code, _ in results]
unique_codes = set(stock_codes)
print(f"\n🔍 验证:")
print(f"  请求股票数: {len(test_stocks)}")
print(f"  返回评分数: {len(results)}")
print(f"  唯一股票数: {len(unique_codes)}")

if len(results) == len(test_stocks) == len(unique_codes):
    print("  ✅ 每个股票都有且只有一个评分")
else:
    print("  ❌ 股票评分数量不匹配")
    print(f"  缺失: {set(test_stocks) - unique_codes}")
    print(f"  重复: {[c for c in stock_codes if stock_codes.count(c) > 1]}")

print("\n" + "="*60)
print("🎉 测试完成")
print("="*60)