#!/usr/bin/env python3
"""
快速调试V3.81投资建议问题
"""

import sys
sys.path.append('/Users/yangxu/StockTradebyZ')

from v380_level4_integrated_system import V380Level4IntegratedSystem

def debug_v381_recommendations():
    """调试V3.81投资建议生成"""
    print("🔍 调试V3.81投资建议生成问题")
    print("=" * 50)

    # 初始化系统
    system = V380Level4IntegratedSystem()

    # 测试几只股票
    test_codes = ['002211', '835640', '000004', '839946', '300197']
    date_str = '2025-09-23'

    print(f"📊 测试日期: {date_str}")
    print(f"📋 测试股票: {test_codes}")
    print()

    # 获取预测结果
    predictions = system.predict_scores(test_codes, date_str)

    for code in test_codes:
        if code in predictions:
            result = predictions[code]

            print(f"🎯 股票代码: {code}")
            print(f"  overall_score: {result.get('overall_score', 'N/A')}")
            print(f"  final_score: {result.get('final_score', 'N/A')}")  # 可能不存在
            print(f"  quality_score: {result.get('quality_score', 'N/A')}")
            print(f"  confidence_score: {result.get('confidence_score', 'N/A')}")
            print(f"  recommendation: {result.get('recommendation', 'N/A')}")

            # 手动验证推荐逻辑
            overall_score = result.get('overall_score', 50.0)
            quality_score = result.get('quality_score', 0.5)

            print(f"\n  🧠 手动验证推荐逻辑:")
            if overall_score >= 85:
                expected = "强烈买入"
            elif overall_score >= 80 or (overall_score >= 75 and quality_score >= 0.7):
                expected = "买入"
            elif overall_score >= 70 or (overall_score >= 65 and quality_score >= 0.6):
                expected = "轻仓买入"
            elif overall_score <= 40 or (overall_score <= 50 and quality_score <= 0.3):
                expected = "卖出"
            elif overall_score <= 50 or (overall_score <= 60 and quality_score <= 0.4):
                expected = "减仓"
            else:
                expected = "观望"

            actual = result.get('recommendation', 'N/A')
            match = "✅ 匹配" if expected == actual else "❌ 不匹配"

            print(f"    期望推荐: {expected}")
            print(f"    实际推荐: {actual}")
            print(f"    验证结果: {match}")
            print("-" * 40)
        else:
            print(f"❌ 股票 {code} 没有预测结果")
            print("-" * 40)

if __name__ == "__main__":
    debug_v381_recommendations()