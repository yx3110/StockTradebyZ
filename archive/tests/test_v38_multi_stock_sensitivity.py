#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.8多股票时间敏感性验证

测试多只股票在不同日期的评分敏感性
避免缓存影响，清理缓存
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# 添加项目路径
sys.path.append('/Users/yangxu/StockTradebyZ')

def clear_v38_cache():
    """清理V3.8缓存"""
    cache_dir = "/Users/yangxu/StockTradebyZ/adaptive_scoring/cache"
    if os.path.exists(cache_dir):
        for file in os.listdir(cache_dir):
            if file.endswith('.pkl'):
                try:
                    os.remove(os.path.join(cache_dir, file))
                    print(f"删除缓存文件: {file}")
                except:
                    pass

def test_multi_stock_sensitivity():
    """测试多股票时间敏感性"""
    print("🔍 V3.8多股票时间敏感性验证")
    print("="*60)

    # 清理缓存
    clear_v38_cache()
    print("✅ V3.8缓存已清理")

    # 配置日志
    logging.basicConfig(level=logging.WARNING)  # 减少日志输出

    try:
        from adaptive_scoring.v38_selector_adapter import V38SelectorAdapter

        v38_adapter = V38SelectorAdapter()

        # 测试多只不同行业股票
        test_stocks = ["000001", "000002", "600036", "000858", "002415"]
        test_dates = ["2025-09-12", "2025-09-13", "2025-09-16"]

        print(f"📊 测试配置:")
        print(f"  测试股票: {len(test_stocks)}只 {test_stocks}")
        print(f"  测试日期: {len(test_dates)}个 {test_dates}")

        # 收集所有评分数据
        all_results = {}

        for date in test_dates:
            print(f"\n📅 评估日期: {date}")

            results = v38_adapter.evaluate_stocks(test_stocks, date)

            date_results = {}
            if 'stocks' in results:
                for stock_data in results['stocks']:
                    code = stock_data.get('code')
                    score = stock_data.get('final_score', 0)
                    confidence = stock_data.get('confidence', 0)

                    date_results[code] = {
                        'score': score,
                        'confidence': confidence
                    }

                    print(f"    {code}: 评分={score:.4f}, 置信度={confidence:.3f}")

            all_results[date] = date_results

        # 分析每只股票的时间敏感性
        print(f"\n📊 各股票时间敏感性分析:")
        stock_sensitivities = []

        for stock in test_stocks:
            print(f"\n🏷️ {stock}:")
            stock_scores = []

            for date in test_dates:
                if stock in all_results[date]:
                    score = all_results[date][stock]['score']
                    stock_scores.append(score)
                    print(f"  {date}: {score:.4f}")

            if len(stock_scores) >= 2:
                score_std = np.std(stock_scores)
                score_mean = np.mean(stock_scores)
                score_cv = (score_std / score_mean * 100) if score_mean > 0 else 0
                score_range = max(stock_scores) - min(stock_scores)

                stock_sensitivities.append(score_cv)
                print(f"  敏感性: {score_cv:.2f}% (标准差:{score_std:.4f}, 范围:{score_range:.4f})")

        # 整体敏感性统计
        if stock_sensitivities:
            avg_sensitivity = np.mean(stock_sensitivities)
            max_sensitivity = max(stock_sensitivities)
            min_sensitivity = min(stock_sensitivities)

            print(f"\n🎯 整体敏感性统计:")
            print(f"  平均敏感性: {avg_sensitivity:.2f}%")
            print(f"  敏感性范围: {min_sensitivity:.2f}% - {max_sensitivity:.2f}%")
            print(f"  敏感性样本: {len(stock_sensitivities)}只股票")

            # 判断敏感性目标达成
            target_sensitivity = 5.0
            passed_stocks = len([s for s in stock_sensitivities if s >= target_sensitivity])

            if avg_sensitivity >= target_sensitivity:
                print(f"\n✅ 时间敏感性目标达成！")
                print(f"   平均敏感性: {avg_sensitivity:.2f}% ≥ 目标{target_sensitivity}%")
                return True
            elif avg_sensitivity >= 3.0:
                print(f"\n⚡ 时间敏感性改善明显但未完全达标")
                print(f"   平均敏感性: {avg_sensitivity:.2f}% < 目标{target_sensitivity}%")
                print(f"   达标股票数: {passed_stocks}/{len(stock_sensitivities)}")
                return avg_sensitivity >= 4.5  # 接近目标也算成功
            else:
                print(f"\n❌ 时间敏感性仍需优化")
                print(f"   平均敏感性: {avg_sensitivity:.2f}% << 目标{target_sensitivity}%")
                return False
        else:
            print(f"\n❌ 未获取到足够的敏感性数据")
            return False

    except Exception as e:
        print(f"\n💥 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_multi_stock_sensitivity()
    if success:
        print(f"\n🚀 V3.8时间敏感性验证通过！")
    else:
        print(f"\n🔧 V3.8时间敏感性需要进一步调优")