#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试V3.8模型预测功能
"""

import sys
import os
sys.path.append('/Users/yangxu/StockTradebyZ')

def test_v380_prediction():
    """测试V3.8模型预测功能"""
    print("🧪 测试V3.8模型预测功能")
    print("="*50)

    try:
        from v380_advanced_incremental_ml_system import V380AdvancedIncrementalMLSystem

        # 初始化系统
        system = V380AdvancedIncrementalMLSystem()
        print(f"✅ {system.version} 系统初始化成功")

        # 测试股票列表
        test_codes = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '300750']
        test_date = '2025-09-16'

        print(f"\n🎯 预测测试:")
        print(f"  股票代码: {test_codes}")
        print(f"  预测日期: {test_date}")

        # 执行预测
        predictions = system.predict_scores(
            codes=test_codes,
            date_str=test_date
        )

        if predictions and len(predictions) > 0:
            print(f"\n📈 预测结果:")
            for code, score in predictions.items():
                print(f"  {code}: {score:.2f}")

            # 分析预测质量
            scores = list(predictions.values())
            score_range = max(scores) - min(scores)
            avg_score = sum(scores) / len(scores)

            print(f"\n📊 预测质量分析:")
            print(f"  平均评分: {avg_score:.2f}")
            print(f"  评分范围: {min(scores):.2f} - {max(scores):.2f}")
            print(f"  差异化度: {score_range:.2f}")
            print(f"  模型状态: {'正常' if score_range > 5 else '需要检查'}")

            print(f"\n✅ V3.8预测功能测试成功!")
            return True
        else:
            print(f"\n❌ 预测失败，未获得有效结果")
            return False

    except Exception as e:
        print(f"\n💥 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_v380_prediction()

    if success:
        print(f"\n🎉 V3.8预测功能正常!")
    else:
        print(f"\n⚠️ 预测功能需要修复")