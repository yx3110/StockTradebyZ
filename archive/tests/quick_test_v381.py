#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试V3.81投资建议修复效果
"""

import sys
sys.path.append('/Users/yangxu/StockTradebyZ')

from v380_level4_integrated_system import V380Level4IntegratedSystem

def test_v381_recommendations():
    print("🚀 快速测试V3.81投资建议修复效果")

    # 初始化系统
    system = V380Level4IntegratedSystem()
    print("✅ V3.81系统初始化完成")

    # 测试几只股票
    test_codes = ['000001.SZ', '000002.SZ', '600000.SH']
    test_date = '2025-09-23'

    print(f"\n🧪 测试股票: {test_codes}")
    print(f"📅 测试日期: {test_date}")

    # 获取预测结果
    predictions = system.predict_scores_with_quality(test_codes, test_date)

    print("\n📊 预测结果:")
    for code, result in predictions.items():
        if isinstance(result, dict):
            final_score = result.get('overall_score', 'N/A')
            quality_score = result.get('quality_score', 'N/A')
            recommendation = result.get('recommendation', 'N/A')
            print(f"  {code}:")
            print(f"    综合评分: {final_score}")
            print(f"    质量评分: {quality_score}")
            print(f"    投资建议: {recommendation}")
            print(f"    ----------------------------------------")
        else:
            print(f"  {code}: 评分={result}")

    print("\n🎯 投资建议阈值说明:")
    print("  强烈买入: >=85分")
    print("  买入: >=75分")
    print("  谨慎买入: >=65分")
    print("  观望: >=50分")
    print("  谨慎卖出: >=35分")
    print("  卖出: <35分")

if __name__ == "__main__":
    test_v381_recommendations()