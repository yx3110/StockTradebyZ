#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
置信度系统Bug诊断脚本

检查置信度评估器是否正常工作
定位0.000置信度问题的根源

Created: 2025-09-16
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append('/Users/yangxu/StockTradebyZ')

def test_confidence_system():
    """测试置信度系统"""
    print("🔍 置信度系统Bug诊断")
    print("="*50)

    try:
        from adaptive_scoring.confidence.confidence_estimator import ConfidenceEstimator

        # 初始化置信度评估器
        estimator = ConfidenceEstimator(
            history_window=50,
            confidence_levels=[0.68, 0.95],
            min_samples_for_calibration=10
        )

        print("✅ 置信度评估器初始化成功")

        # 1. 创建测试数据
        test_prediction_scores = {
            'short_term': 0.65,
            'medium_term': 0.72,
            'long_term': 0.58
        }

        # 创建模拟股票数据
        test_data = pd.DataFrame({
            'trade_date': pd.date_range('2025-09-01', periods=20),
            'close': np.random.normal(100, 5, 20),
            'volume': np.random.normal(1000000, 100000, 20),
            'turnover_rate': np.random.normal(2.0, 0.5, 20)
        })

        test_model_metadata = {
            'model_complexity': 0.6,
            'output_stability': 0.8,
            'feature_importances': [0.3, 0.4, 0.3]
        }

        test_market_context = {
            'volatility_regime': 1.2,
            'trend_strength': 0.5,
            'market_sentiment': 0.3,
            'market_regime': 'normal'
        }

        print("✅ 测试数据准备完成")

        # 2. 调用置信度评估
        print("\n🎯 执行置信度评估...")

        confidence_result = estimator.estimate_confidence(
            prediction_scores=test_prediction_scores,
            input_data=test_data,
            model_metadata=test_model_metadata,
            market_context=test_market_context
        )

        print("✅ 置信度评估完成")

        # 3. 分析结果
        print(f"\n📊 置信度评估结果:")
        print(f"  总体置信度: {confidence_result['confidence_score']:.3f}")
        print(f"  置信度等级: {confidence_result['confidence_level']}")

        # 检查各组件置信度
        components = confidence_result.get('component_confidences', {})
        print(f"\n🔧 组件置信度分析:")

        if 'model_uncertainty' in components:
            model_conf = components['model_uncertainty'].get('model_confidence', 0)
            print(f"  模型置信度: {model_conf:.3f}")

        if 'historical_reliability' in components:
            hist_conf = components['historical_reliability'].get('overall_reliability', 0)
            print(f"  历史可靠性: {hist_conf:.3f}")

        if 'data_quality' in components:
            data_conf = components['data_quality'].get('overall_quality', 0)
            print(f"  数据质量: {data_conf:.3f}")

        if 'market_impact' in components:
            market_conf = components['market_impact'].get('market_reliability', 0)
            print(f"  市场影响: {market_conf:.3f}")

        # 4. 诊断问题
        print(f"\n🔍 问题诊断:")

        if confidence_result['confidence_score'] < 0.01:
            print("❌ 发现问题: 置信度过低 (<0.01)")

            # 检查各个组件是否有问题
            if components.get('model_uncertainty', {}).get('model_confidence', 0) < 0.1:
                print("  - 模型不确定性评估有问题")

            if components.get('historical_reliability', {}).get('overall_reliability', 0) < 0.1:
                print("  - 历史可靠性评估有问题 (可能缺少历史数据)")

            if components.get('data_quality', {}).get('overall_quality', 0) < 0.1:
                print("  - 数据质量评估有问题")

            if components.get('market_impact', {}).get('market_reliability', 0) < 0.1:
                print("  - 市场影响评估有问题")

        elif confidence_result['confidence_score'] > 0.3:
            print("✅ 置信度正常: 评估器工作正常")
        else:
            print("⚠️ 置信度偏低但在合理范围内")

        # 5. 测试历史记录影响
        print(f"\n📈 测试历史记录影响...")

        # 添加一些历史预测记录
        for i in range(15):
            fake_prediction = {
                'short_term': 0.6 + np.random.normal(0, 0.1),
                'medium_term': 0.65 + np.random.normal(0, 0.1),
                'long_term': 0.55 + np.random.normal(0, 0.1)
            }

            fake_result = estimator.estimate_confidence(
                prediction_scores=fake_prediction,
                input_data=test_data,
                model_metadata=test_model_metadata,
                market_context=test_market_context
            )

            # 模拟实际结果更新
            if len(estimator.prediction_history) > 0:
                estimator.update_prediction_outcome(
                    prediction_id=len(estimator.prediction_history) - 1,
                    actual_outcome=0.6 + np.random.normal(0, 0.15),
                    temporal_outcomes={'short_term': 0.65, 'medium_term': 0.62, 'long_term': 0.58}
                )

        # 重新评估
        print("重新评估置信度 (有历史记录后)...")

        final_result = estimator.estimate_confidence(
            prediction_scores=test_prediction_scores,
            input_data=test_data,
            model_metadata=test_model_metadata,
            market_context=test_market_context
        )

        print(f"  更新后置信度: {final_result['confidence_score']:.3f}")
        print(f"  历史记录数: {len(estimator.prediction_history)}")

        # 获取系统摘要
        summary = estimator.get_confidence_summary()
        print(f"\n📋 置信度系统摘要:")
        print(f"  总预测数: {summary['total_predictions']}")
        print(f"  有结果预测数: {summary['predictions_with_outcomes']}")
        print(f"  平均置信度: {summary['average_confidence']:.3f}")

        return True

    except Exception as e:
        print(f"❌ 置信度系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_confidence_system()

    if success:
        print(f"\n🎊 置信度系统诊断完成!")
    else:
        print(f"\n💥 置信度系统存在问题，需要修复")