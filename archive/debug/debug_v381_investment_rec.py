#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug V3.81投资建议生成过程
"""

import sys
sys.path.append('/Users/yangxu/StockTradebyZ')

from tomorrow_stock_selector import TomorrowStockSelector
import logging

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_v381_investment_recommendation():
    print("🔧 Debug V3.81投资建议生成过程")

    # 创建选择器
    selector = TomorrowStockSelector(scoring_version="v3.81", stocks_only=True)

    # 先填充缓存
    from v380_level4_integrated_system import V380Level4IntegratedSystem
    v381_system = V380Level4IntegratedSystem()

    test_codes = ['002211', '835640', '000004']
    test_date = '2025-09-23'

    print(f"🧪 测试股票: {test_codes}")
    print(f"📅 测试日期: {test_date}")

    # 填充缓存
    predictions = v381_system.predict_scores_with_quality(test_codes, test_date)
    selector.v381_batch_cache = predictions.copy()
    print(f"✅ 缓存已填充，包含 {len(selector.v381_batch_cache)} 只股票")

    # 测试生成投资建议过程
    for code in test_codes:
        print(f"\n🔍 测试 {code}:")

        # 构建stock_info
        stock_info = {
            'stock_code': code,
            'selected_by_strategies': 1,  # 单策略
            'strategies': ['少妇战法']
        }

        # Step 1: 测试calculate_comprehensive_score
        score, detailed_info = selector.calculate_comprehensive_score(stock_info, test_date)
        print(f"  📊 calculate_comprehensive_score结果:")
        print(f"    score: {score}")
        print(f"    detailed_info包含recommendation: {'recommendation' in detailed_info}")
        if 'recommendation' in detailed_info:
            print(f"    recommendation: {detailed_info['recommendation']}")

        # Step 2: 测试generate_investment_recommendation
        investment_rec = selector.generate_investment_recommendation(stock_info)
        print(f"  🎯 generate_investment_recommendation结果:")
        print(f"    recommendation: {investment_rec.get('recommendation', 'MISSING')}")
        print(f"    score: {investment_rec.get('score', 'MISSING')}")

if __name__ == "__main__":
    debug_v381_investment_recommendation()