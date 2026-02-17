#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试V3.81修复效果
"""

import sys
sys.path.append('/Users/yangxu/StockTradebyZ')

from tomorrow_stock_selector import TomorrowStockSelector
import logging

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_v381_fix():
    """测试V3.81缓存修复效果"""
    print("🔧 测试V3.81缓存修复效果")

    # 创建选择器
    selector = TomorrowStockSelector(scoring_version="v3.81", stocks_only=True)

    # 测试单只股票的综合评分计算（这会使用缓存）
    test_codes = ['002211', '000004', '835640']
    test_date = '2025-09-23'

    print(f"\n🧪 测试股票: {test_codes}")
    print(f"📅 测试日期: {test_date}")

    # 首先填充缓存（模拟批处理）
    from v380_level4_integrated_system import V380Level4IntegratedSystem
    v381_system = V380Level4IntegratedSystem()

    print("\n1️⃣ 批处理阶段 - 填充缓存...")
    batch_predictions = v381_system.predict_scores_with_quality(test_codes, test_date)

    print("📊 批处理结果:")
    for code, result in batch_predictions.items():
        if isinstance(result, dict):
            final_score = result.get('overall_score', 'N/A')
            recommendation = result.get('recommendation', 'N/A')
            quality_score = result.get('quality_score', 'N/A')
            print(f"  {code}: {final_score}分 → {recommendation} (质量:{quality_score})")

    # 填充缓存到选择器
    selector.v381_batch_cache = batch_predictions.copy()
    print(f"\n✅ 缓存已填充，包含 {len(selector.v381_batch_cache)} 只股票")

    print("\n2️⃣ 个别股票处理阶段 - 使用缓存...")

    # 测试个别股票计算（应该使用缓存）
    for code in test_codes:
        stock_info = {'stock_code': code}
        final_score, detailed_info = selector.calculate_comprehensive_score(stock_info, test_date)
        recommendation = detailed_info.get('recommendation', 'N/A')
        quality_score = detailed_info.get('quality_score', 'N/A')

        print(f"  {code}: {final_score}分 → {recommendation} (质量:{quality_score})")

        # 检查是否使用了缓存
        if code in selector.v381_batch_cache:
            print(f"    ✅ 使用了缓存")
        else:
            print(f"    ❌ 未使用缓存")

    print("\n🎯 测试完成！")

if __name__ == "__main__":
    test_v381_fix()