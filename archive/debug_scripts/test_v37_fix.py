#!/usr/bin/env python3
"""
测试V3.7修复后的Meta学习器特征维度问题
"""

import sys
import os
sys.path.append('/Users/yangxu/StockTradebyZ')

from v370_advanced_ml_system import V370AdvancedMLSystem
import pandas as pd
from datetime import datetime

def test_v37_scoring():
    """测试V3.7评分系统"""
    print("🧪 测试V3.7修复后的评分系统...")

    try:
        # 初始化V3.7系统
        v37_system = V370AdvancedMLSystem()

        # 测试股票列表
        test_stocks = ['000001', '000002', '002594', '600036', '000858']
        test_date = '2025-09-26'

        print(f"📊 测试股票: {test_stocks}")
        print(f"📅 测试日期: {test_date}")

        # 获取特征数据（提取当天特征）
        from datetime import datetime, timedelta
        end_date = datetime.strptime(test_date, '%Y-%m-%d')
        start_date = end_date - timedelta(days=60)  # 60天历史数据用于特征计算

        print(f"📈 提取特征数据: {start_date.strftime('%Y-%m-%d')} 到 {test_date}")

        # 批量提取特征
        features_df = v37_system.extract_advanced_features(
            test_stocks,
            start_date.strftime('%Y-%m-%d'),
            test_date
        )

        if features_df.empty:
            print("❌ 无法获取股票特征数据")
            return False, {}

        print(f"📊 特征数据形状: {features_df.shape}")

        # 进行评分
        prediction_result = v37_system.predict_three_layer_ensemble(features_df)

        # V3.7返回字典格式 {'score': float, 'factor_scores': dict}
        if isinstance(prediction_result, dict) and 'score' in prediction_result:
            main_score = prediction_result['score']
            factor_scores = prediction_result.get('factor_scores', {})

            print(f"✅ V3.7主评分: {main_score:.2f}")
            print(f"📊 因子评分: {factor_scores}")

            # 由于是批量数据，主评分可能是多个股票的平均值
            # 简化测试，给所有股票分配相同分数
            scores = {stock: main_score for stock in test_stocks}
        else:
            print(f"⚠️ 预测结果格式异常: {type(prediction_result)}, {prediction_result}")
            scores = {}

        print(f"\n✅ V3.7评分结果:")
        for stock, score in scores.items():
            print(f"  {stock}: {score:.2f}")

        # 统计信息
        score_values = list(scores.values())
        avg_score = sum(score_values) / len(score_values)
        print(f"\n📈 平均分: {avg_score:.2f}")
        print(f"📊 最高分: {max(score_values):.2f}")
        print(f"📉 最低分: {min(score_values):.2f}")

        # 检查是否有高分股票
        high_score_stocks = [stock for stock, score in scores.items() if score >= 70]
        print(f"\n🎯 高分股票(>=70): {len(high_score_stocks)}只")
        for stock in high_score_stocks:
            print(f"  {stock}: {scores[stock]:.2f}")

        return True, scores

    except Exception as e:
        print(f"❌ V3.7测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

if __name__ == "__main__":
    success, scores = test_v37_scoring()
    if success:
        print("\n🎉 V3.7修复测试成功！")
    else:
        print("\n💥 V3.7修复测试失败！")