#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug V3.81投资建议问题
"""

import sys
sys.path.append('/Users/yangxu/StockTradebyZ')

from v380_level4_integrated_system import V380Level4IntegratedSystem

def debug_v381_recommendations():
    print("🔧 Debug V3.81投资建议问题")

    # 初始化系统
    system = V380Level4IntegratedSystem()

    # 测试几只综合评分很高的股票
    test_codes = ['002211', '835640', '000004']  # 这些应该是强烈买入
    test_date = '2025-09-23'

    print(f"🧪 测试股票: {test_codes}")
    print(f"📅 测试日期: {test_date}")

    # 获取预测结果
    predictions = system.predict_scores_with_quality(test_codes, test_date)

    print("\n🔍 详细预测结果:")
    for code, result in predictions.items():
        print(f"\n📊 {code}:")
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"  {key}: {value}")
        else:
            print(f"  简单评分: {result}")

    print("\n🎯 投资建议分析:")
    for code, result in predictions.items():
        if isinstance(result, dict):
            final_score = result.get('overall_score', 'N/A')
            quality_score = result.get('quality_score', 'N/A')
            recommendation = result.get('recommendation', 'N/A')

            print(f"\n{code}:")
            print(f"  综合评分: {final_score}")
            print(f"  质量评分: {quality_score}")
            print(f"  投资建议: {recommendation}")

            # 根据V3.81逻辑分析
            if final_score >= 85:
                expected = "强烈买入"
            elif final_score >= 80:
                expected = "买入"
            elif final_score >= 70:
                expected = "轻仓买入"
            else:
                expected = "观望"

            print(f"  期望建议: {expected}")
            print(f"  是否匹配: {'✅' if recommendation == expected else '❌'}")

if __name__ == "__main__":
    debug_v381_recommendations()