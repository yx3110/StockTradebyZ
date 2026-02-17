#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug V3.81缓存内容
"""

import sys
sys.path.append('/Users/yangxu/StockTradebyZ')

from v380_level4_integrated_system import V380Level4IntegratedSystem

def debug_v381_cache():
    print("🔧 Debug V3.81缓存内容")

    # 初始化系统
    system = V380Level4IntegratedSystem()

    # 测试几只高分股票
    test_codes = ['002211', '835640', '000004']
    test_date = '2025-09-23'

    print(f"🧪 测试股票: {test_codes}")
    print(f"📅 测试日期: {test_date}")

    # 获取批处理结果
    predictions = system.predict_scores_with_quality(test_codes, test_date)

    print("\n🔍 批处理缓存详细内容:")
    for code, result in predictions.items():
        print(f"\n📊 {code}:")
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"  {key}: {value}")

            # 特别检查投资建议
            recommendation = result.get('recommendation', 'MISSING')
            overall_score = result.get('overall_score', 'MISSING')
            quality_score = result.get('quality_score', 'MISSING')

            print(f"\n🎯 关键字段检查:")
            print(f"  overall_score: {overall_score}")
            print(f"  quality_score: {quality_score}")
            print(f"  recommendation: {recommendation}")

            # 验证推荐逻辑
            if overall_score != 'MISSING':
                if overall_score >= 85:
                    expected = "强烈买入"
                elif overall_score >= 80:
                    expected = "买入"
                elif overall_score >= 70:
                    expected = "轻仓买入"
                else:
                    expected = "观望"

                print(f"  期望推荐: {expected}")
                print(f"  匹配: {'✅' if recommendation == expected else '❌'}")
        else:
            print(f"  简单评分: {result}")

if __name__ == "__main__":
    debug_v381_cache()