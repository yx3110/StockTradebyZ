#!/usr/bin/env python3
"""
Debug V3.81 cache mechanism
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tomorrow_stock_selector import TomorrowStockSelector
from datetime import datetime, timedelta
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_cache_mechanism():
    """测试V3.81缓存机制"""
    print("=== V3.81缓存机制调试 ===")

    # 初始化选股器
    selector = TomorrowStockSelector(scoring_version="v3.81", stocks_only=True)

    # 检查初始状态
    print(f"初始缓存状态: {len(selector.v381_batch_cache)} 条记录")

    # 手动设置一些测试数据到缓存
    test_cache = {
        '002211': {
            'overall_score': 90.0,
            'short_term_score': 90.0,
            'medium_term_score': 86.2,
            'long_term_score': 66.5,
            'confidence_score': 0.606,
            'quality_score': 0.67,
            'recommendation': '强烈买入',
            'confidence_level': 'medium'
        },
        '835640': {
            'overall_score': 87.5,
            'short_term_score': 71.2,
            'medium_term_score': 67.3,
            'long_term_score': 90.0,
            'confidence_score': 0.601,
            'quality_score': 0.42,
            'recommendation': '买入',
            'confidence_level': 'medium'
        }
    }

    selector.v381_batch_cache = test_cache.copy()
    print(f"设置测试缓存后: {len(selector.v381_batch_cache)} 条记录")
    print(f"缓存键: {list(selector.v381_batch_cache.keys())}")

    # 测试缓存访问
    test_date = "2025-09-23"

    for stock_code in ['002211', '835640']:
        print(f"\n--- 测试股票 {stock_code} ---")

        # 检查缓存中是否有数据
        if stock_code in selector.v381_batch_cache:
            print(f"✅ 缓存中找到 {stock_code}")
            cached_data = selector.v381_batch_cache[stock_code]
            print(f"缓存数据: {cached_data}")
        else:
            print(f"❌ 缓存中未找到 {stock_code}")

        # 调用calculate_comprehensive_score看是否使用缓存
        try:
            score, details = selector.calculate_comprehensive_score(stock_code, test_date)
            print(f"综合评分: {score}")
            print(f"投资建议: {details.get('recommendation', 'N/A')}")
            print(f"质量评分: {details.get('quality_score', 'N/A')}")
            print(f"评分方法: {details.get('scoring_method', 'N/A')}")
        except Exception as e:
            print(f"计算评分失败: {e}")

    print("\n=== 调试完成 ===")

if __name__ == "__main__":
    debug_cache_mechanism()