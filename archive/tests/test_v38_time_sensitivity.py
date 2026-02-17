#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.8时间敏感性修复验证

验证修复后的V3.8是否能在不同日期产生不同评分
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# 添加项目路径
sys.path.append('/Users/yangxu/StockTradebyZ')

def test_time_sensitivity():
    """测试V3.8时间敏感性"""
    print("🔍 V3.8时间敏感性修复验证")
    print("="*50)

    # 配置日志
    logging.basicConfig(level=logging.INFO)

    try:
        from adaptive_scoring.v38_selector_adapter import V38SelectorAdapter

        v38_adapter = V38SelectorAdapter()

        # 测试股票和日期
        test_stock = "000001"
        test_dates = ["2025-09-10", "2025-09-11", "2025-09-12", "2025-09-13", "2025-09-16"]

        print(f"📊 测试配置:")
        print(f"  测试股票: {test_stock}")
        print(f"  测试日期: {test_dates}")

        # 收集不同日期的评分
        scores_by_date = {}

        for date in test_dates:
            print(f"\n📅 评估日期: {date}")

            results = v38_adapter.evaluate_stocks([test_stock], date)

            if 'stocks' in results and results['stocks']:
                stock_data = results['stocks'][0]
                score = stock_data.get('final_score', 0)
                confidence = stock_data.get('confidence', 0)

                scores_by_date[date] = {
                    'score': score,
                    'confidence': confidence
                }

                print(f"    评分: {score:.4f}, 置信度: {confidence:.3f}")
            else:
                print(f"    ⚠️ 未获取到评分数据")
                scores_by_date[date] = {'score': 0, 'confidence': 0}

        # 分析时间敏感性
        print(f"\n📊 时间敏感性分析:")
        print(f"  {'日期':<12} {'评分':<8} {'置信度':<8}")
        print(f"  {'-'*12} {'-'*8} {'-'*8}")

        all_scores = []
        for date, data in scores_by_date.items():
            print(f"  {date:<12} {data['score']:<8.4f} {data['confidence']:<8.3f}")
            all_scores.append(data['score'])

        # 计算敏感性指标
        if len(all_scores) >= 2:
            score_std = np.std(all_scores)
            score_range = max(all_scores) - min(all_scores)
            score_mean = np.mean(all_scores)
            score_cv = (score_std / score_mean * 100) if score_mean > 0 else 0

            print(f"\n🎯 敏感性指标:")
            print(f"  评分标准差: {score_std:.6f}")
            print(f"  评分范围: {score_range:.6f}")
            print(f"  变异系数: {score_cv:.2f}%")

            # 判断是否修复成功
            target_sensitivity = 5.0  # 目标≥5%变异系数

            if score_cv >= target_sensitivity:
                print(f"\n✅ 时间敏感性修复成功！")
                print(f"   变异系数: {score_cv:.2f}% ≥ 目标{target_sensitivity}%")
                return True
            elif score_cv >= 1.0:
                print(f"\n⚡ 时间敏感性有改善但未达标")
                print(f"   变异系数: {score_cv:.2f}% < 目标{target_sensitivity}%")
                print(f"   建议进一步调优参数")
                return False
            else:
                print(f"\n❌ 时间敏感性修复失败")
                print(f"   变异系数: {score_cv:.2f}% << 目标{target_sensitivity}%")
                return False
        else:
            print(f"\n❌ 评分数据不足，无法计算敏感性")
            return False

    except Exception as e:
        print(f"\n💥 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_time_sensitivity()
    if success:
        print(f"\n🚀 V3.8时间敏感性修复验证成功！")
    else:
        print(f"\n🔧 需要进一步调优时间敏感性")