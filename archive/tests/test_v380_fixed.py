#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的V3.80评分系统
"""

import sys
sys.path.append('/Users/yangxu/StockTradebyZ')

from v380_advanced_incremental_ml_system import V380AdvancedIncrementalMLSystem
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_v380_fixed():
    """测试修复后的V3.80评分系统"""
    print("🔍 测试修复后的V3.80评分系统")

    try:
        # 初始化系统
        system = V380AdvancedIncrementalMLSystem()

        # 测试股票
        test_codes = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '002415.SZ']
        test_date = '2025-09-20'

        print(f"📊 测试股票: {test_codes}")
        print(f"📅 测试日期: {test_date}")

        # 直接调用修复后的predict_scores函数
        predictions = system.predict_scores(test_codes, test_date)

        print(f"\n🎯 预测结果:")
        for code, score in predictions.items():
            print(f"  {code}: {score}")

        # 检查是否所有得分相同 - 提取overall_score进行比较
        overall_scores = [pred['overall_score'] for pred in predictions.values()]
        unique_scores = set(overall_scores)
        if len(unique_scores) == 1:
            print(f"❌ 所有股票得分相同: {list(unique_scores)[0]}")
        else:
            print(f"✅ 得分有差异: {sorted(unique_scores)}")
            print(f"🎯 得分范围: {min(overall_scores):.2f} - {max(overall_scores):.2f}")

            # 显示各时期得分的变化范围
            short_scores = [pred['short_term_score'] for pred in predictions.values()]
            medium_scores = [pred['medium_term_score'] for pred in predictions.values()]
            long_scores = [pred['long_term_score'] for pred in predictions.values()]

            print(f"📊 短期得分范围: {min(short_scores):.2f} - {max(short_scores):.2f}")
            print(f"📊 中期得分范围: {min(medium_scores):.2f} - {max(medium_scores):.2f}")
            print(f"📊 长期得分范围: {min(long_scores):.2f} - {max(long_scores):.2f}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_v380_fixed()