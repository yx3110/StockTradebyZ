#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试V3.80与量化系统集成
"""

import sys
import os
sys.path.append('/Users/yangxu/StockTradebyZ')

def test_v380_integration():
    """测试V3.80与量化系统集成"""
    print("🚀 测试V3.80与量化系统集成")
    print("="*50)

    try:
        # 初始化V3.8选股器
        from tomorrow_stock_selector import TomorrowStockSelector

        selector = TomorrowStockSelector(scoring_version="v3.8")
        print("✅ V3.8选股器初始化成功")

        # 测试单只股票评分
        test_code = "000001.SZ"
        test_date = "2025-09-19"

        print(f"\n🎯 测试单只股票评分:")
        print(f"  股票代码: {test_code}")
        print(f"  评分日期: {test_date}")

        score, details = selector.calculate_comprehensive_score(test_code, test_date)

        print(f"\n📊 评分结果:")
        print(f"  综合评分: {score:.2f}")
        print(f"  详细信息: {details}")

        if score > 0:
            print(f"\n✅ V3.80集成测试成功!")
            return True
        else:
            print(f"\n⚠️ 评分为0，可能需要检查")
            return False

    except Exception as e:
        print(f"\n💥 集成测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_v380_integration()

    if success:
        print(f"\n🎉 V3.80集成成功!")
    else:
        print(f"\n❌ 集成需要修复")