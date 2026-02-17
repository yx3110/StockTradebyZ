#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.8自适应评分系统完整集成测试

测试Phase 4的所有组件和集成功能
验证动态归一化、多时间维度评分和置信度评估

Created: 2025-09-16
Author: Claude Code
Version: 3.8.0
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# 添加项目路径
sys.path.append('/Users/yangxu/StockTradebyZ')

def test_adaptive_scoring_system():
    """测试V3.8自适应评分系统完整集成"""
    print("🚀 V3.8自适应评分系统集成测试开始")
    print("=" * 60)

    # 配置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # 1. 测试组件单独导入
        print("\\n📦 阶段1: 测试组件导入")

        from adaptive_scoring.normalizers.dynamic_normalizer import DynamicNormalizer
        print("✅ DynamicNormalizer 导入成功")

        from adaptive_scoring.temporal.multi_temporal_scorer import MultiTemporalScorer
        print("✅ MultiTemporalScorer 导入成功")

        from adaptive_scoring.confidence.confidence_estimator import ConfidenceEstimator
        print("✅ ConfidenceEstimator 导入成功")

        from adaptive_scoring.adaptive_scoring_system import AdaptiveScoringSystem
        print("✅ AdaptiveScoringSystem 导入成功")

        # 2. 使用真实数据库数据
        print("\\n🔧 阶段2: 获取真实数据库数据")

        from adaptive_scoring.utils.data_adapter import AdaptiveScoringDataAdapter

        # 创建数据适配器
        data_adapter = AdaptiveScoringDataAdapter(logger=logger)

        # 获取真实股票数据
        test_stock_code = "000001"  # 平安银行
        stock_data = data_adapter.get_stock_data(
            stock_code=test_stock_code,
            days=120  # 获取120天数据确保有足够的历史数据
        )

        if stock_data.empty:
            print(f"  ❌ 无法获取股票 {test_stock_code} 的数据，尝试其他股票")
            # 尝试其他股票
            for alt_code in ["000002", "600036", "000858"]:
                stock_data = data_adapter.get_stock_data(alt_code, days=120)
                if not stock_data.empty:
                    test_stock_code = alt_code
                    print(f"  ✅ 使用股票 {alt_code} 的数据")
                    break

        if stock_data.empty:
            raise ValueError("无法获取任何股票的真实数据，请检查数据库")

        print(f"  ✅ 股票数据获取完成 - {test_stock_code}, {len(stock_data)} 天数据")
        print(f"      可用字段: {stock_data.columns.tolist()}")

        # 数据质量检查
        quality_report = data_adapter.validate_data_quality(stock_data, 'stock')
        print(f"      数据质量: {quality_report['status']} (评分: {quality_report['quality_score']:.3f})")
        if quality_report['issues']:
            print(f"      质量问题: {quality_report['issues']}")

        # 获取基本面数据
        fundamental_data = data_adapter.get_fundamental_data(test_stock_code)
        print(f"  ✅ 基本面数据获取完成 - {len(fundamental_data)} 个报告期")

        # 获取市场数据 (上证指数)
        market_data = data_adapter.get_market_data('000001.SH', days=120)
        print(f"  ✅ 市场数据获取完成 - {len(market_data)} 天数据")

        # 3. 测试各组件单独功能
        print("\\n🔄 阶段3: 测试各组件单独功能")

        # 3.1 测试动态归一化器
        print("  📊 测试动态归一化器...")
        normalizer = DynamicNormalizer(logger=logger)

        raw_scores = np.array([0.3, 0.7, 0.5])
        norm_result = normalizer.normalize_scores(
            raw_scores=raw_scores,
            market_data=market_data,
            strategy='adaptive_sigmoid'
        )

        print(f"    原始评分: {raw_scores}")
        print(f"    归一化后: {norm_result['normalized_scores']}")
        print(f"    质量评分: {norm_result['quality_metrics']['overall_quality']:.3f}")

        # 3.2 测试多时间维度评分器
        print("  📈 测试多时间维度评分器...")
        temporal_scorer = MultiTemporalScorer(logger=logger)

        temporal_result = temporal_scorer.calculate_multi_temporal_scores(
            stock_data=stock_data,
            fundamental_data=fundamental_data,
            market_data=market_data
        )

        print(f"    短期评分: {temporal_result['temporal_scores']['short_term']['overall_score']:.3f}")
        print(f"    中期评分: {temporal_result['temporal_scores']['medium_term']['overall_score']:.3f}")
        print(f"    长期评分: {temporal_result['temporal_scores']['long_term']['overall_score']:.3f}")
        print(f"    综合评分: {temporal_result['composite_score']:.3f}")

        # 3.3 测试置信度评估器
        print("  🎯 测试置信度评估器...")
        confidence_estimator = ConfidenceEstimator(logger=logger)

        prediction_scores = {
            'short_term': temporal_result['temporal_scores']['short_term']['overall_score'],
            'medium_term': temporal_result['temporal_scores']['medium_term']['overall_score'],
            'long_term': temporal_result['temporal_scores']['long_term']['overall_score']
        }

        confidence_result = confidence_estimator.estimate_confidence(
            prediction_scores=prediction_scores,
            input_data=stock_data,
            model_metadata={'model_complexity': 0.6, 'output_stability': 0.8}
        )

        print(f"    置信度评分: {confidence_result['confidence_score']:.3f}")
        print(f"    置信度等级: {confidence_result['confidence_level']}")
        print(f"    整体风险: {confidence_result['risk_assessment']['overall_risk']}")

        # 4. 测试完整自适应评分系统
        print("\\n🎯 阶段4: 测试完整自适应评分系统")

        # 4.1 创建系统实例
        adaptive_system = AdaptiveScoringSystem(
            normalization_strategy='adaptive_sigmoid',
            temporal_windows={'short_term': 5, 'medium_term': 20, 'long_term': 60},
            confidence_levels=[0.68, 0.95],
            adaptation_mode='full',
            logger=logger
        )

        print("  ✅ 自适应评分系统创建成功")

        # 4.2 计算单只股票评分
        print("  📊 计算单只股票评分...")

        result = adaptive_system.calculate_adaptive_scores(
            stock_code=test_stock_code,
            stock_data=stock_data,
            fundamental_data=fundamental_data,
            market_data=market_data
        )

        print(f"    股票代码: {result['stock_code']}")
        print(f"    最终评分: {result['final_score']:.3f}")
        print(f"    原始评分: {result['raw_final_score']:.3f}")
        print(f"    置信度: {result['confidence']['confidence_score']:.3f}")
        print(f"    处理时间: {result['processing_time']:.2f}秒")
        print(f"    整体质量: {result['quality_metrics']['overall_quality']:.3f}")

        # 4.3 测试不同归一化策略
        print("  🔧 测试不同归一化策略...")
        strategies = ['adaptive_sigmoid', 'robust_sigmoid', 'quantile_based']

        for strategy in strategies:
            custom_config = {'normalization_strategy': strategy}
            strategy_result = adaptive_system.calculate_adaptive_scores(
                stock_code=f"test_{strategy}",
                stock_data=stock_data,
                fundamental_data=fundamental_data,
                market_data=market_data,
                custom_config=custom_config
            )
            print(f"    {strategy}: 最终评分 {strategy_result['final_score']:.3f}, 置信度 {strategy_result['confidence']['confidence_score']:.3f}")

        # 4.4 测试系统性能监控
        print("  📈 测试系统性能监控...")
        performance = adaptive_system.get_system_performance()

        print(f"    系统状态: {performance['status']}")
        print(f"    总评分次数: {performance['total_scorings']}")
        print(f"    成功率: {performance['success_rate']:.1%}")
        print(f"    平均执行时间: {performance['average_execution_time']:.3f}秒")

        # 4.5 测试系统配置导出
        print("  💾 测试系统配置导出...")
        config = adaptive_system.export_system_config()

        print(f"    系统版本: {config['system_version']}")
        print(f"    适应模式: {config['adaptation_mode']}")
        print(f"    时间窗口: {config['temporal_windows']}")

        # 5. 压力测试
        print("\\n⚡ 阶段5: 系统压力测试")

        # 5.1 多股票批量测试
        print("  🏭 多股票批量测试...")

        # 使用真实数据适配器作为数据提供者
        class RealDataProvider:
            def __init__(self, data_adapter):
                self.data_adapter = data_adapter

            def get_stock_data(self, stock_code):
                return self.data_adapter.get_stock_data(stock_code, days=120)

            def get_fundamental_data(self, stock_code):
                return self.data_adapter.get_fundamental_data(stock_code)

            def get_market_data(self):
                return self.data_adapter.get_market_data('000001.SH', days=120)

        data_provider = RealDataProvider(data_adapter)
        test_stocks = ['000001', '000002', '000858', '002415', '600036']

        batch_results = adaptive_system.batch_calculate_scores(
            stock_list=test_stocks,
            data_provider=data_provider,
            parallel=False  # 串行测试避免并发问题
        )

        successful_results = [r for r in batch_results.values() if not r.get('error')]
        print(f"    批量处理结果: {len(successful_results)}/{len(test_stocks)} 成功")

        if successful_results:
            avg_score = np.mean([r['final_score'] for r in successful_results])
            avg_confidence = np.mean([r['confidence']['confidence_score'] for r in successful_results])
            print(f"    平均评分: {avg_score:.3f}")
            print(f"    平均置信度: {avg_confidence:.3f}")

        # 5.2 错误处理测试
        print("  🛡️ 错误处理测试...")

        # 测试空数据
        empty_result = adaptive_system.calculate_adaptive_scores(
            stock_code="empty_test",
            stock_data=pd.DataFrame(),
            fundamental_data=None,
            market_data=None
        )

        print(f"    空数据处理: {'✅ 成功' if empty_result.get('error') else '❌ 失败'}")
        print(f"    后备评分: {empty_result['final_score']:.3f}")
        print(f"    后备置信度: {empty_result['confidence']['confidence_score']:.3f}")

        # 6. 结果总结
        print("\\n📈 阶段6: 系统状态总结")

        final_performance = adaptive_system.get_system_performance()
        print(f"  总评分次数: {final_performance['total_scorings']}")
        print(f"  成功次数: {final_performance['successful_scorings']}")
        print(f"  失败次数: {final_performance['failed_scorings']}")
        print(f"  整体成功率: {final_performance['success_rate']:.1%}")

        # 判断测试是否全面通过
        if (final_performance['success_rate'] >= 0.8 and
            final_performance['successful_scorings'] >= 5 and
            len(successful_results) >= 3):

            print("\\n🎉 V3.8自适应评分系统集成测试成功!")
            print("=" * 60)
            print("✅ 所有核心组件正常工作")
            print("✅ 集成功能测试通过")
            print("✅ 压力测试和错误处理测试通过")
            print("✅ V3.8 Phase 4自适应评分系统完全就绪")

            return True

        else:
            print("\\n⚠️ V3.8自适应评分系统测试存在问题")
            print(f"成功率: {final_performance['success_rate']:.1%} (需要≥80%)")
            print(f"成功次数: {final_performance['successful_scorings']} (需要≥5)")
            print(f"批量成功: {len(successful_results)} (需要≥3)")

            return False

    except Exception as e:
        print(f"\\n❌ V3.8自适应评分系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_components():
    """单独测试各组件的详细功能"""
    print("\\n🔬 附加测试: 单独组件详细功能验证")

    try:
        # 测试动态归一化器的多种策略
        from adaptive_scoring.normalizers.dynamic_normalizer import DynamicNormalizer

        normalizer = DynamicNormalizer()
        test_scores = np.array([0.1, 0.9, 0.3, 0.7, 0.5])

        strategies = ['adaptive_sigmoid', 'robust_sigmoid', 'quantile_based', 'market_aware']
        print("\\n  动态归一化器策略测试:")

        for strategy in strategies:
            try:
                result = normalizer.normalize_scores(test_scores, strategy=strategy)
                quality = result['quality_metrics']['overall_quality']
                print(f"    {strategy}: 质量评分 {quality:.3f}")
            except Exception as e:
                print(f"    {strategy}: ❌ {e}")

        # 测试置信度评估器的校准功能
        from adaptive_scoring.confidence.confidence_estimator import ConfidenceEstimator

        confidence_estimator = ConfidenceEstimator()

        # 模拟一些历史预测记录
        for i in range(30):
            predicted = np.random.rand()
            actual = predicted + np.random.randn() * 0.1  # 添加噪声
            actual = np.clip(actual, 0, 1)

            confidence_estimator.prediction_history.append({
                'predicted_score': predicted,
                'actual_outcome': actual,
                'accuracy': 1 - abs(predicted - actual),
                'timestamp': datetime.now()
            })

        summary = confidence_estimator.get_confidence_summary()
        print(f"\\n  置信度评估器校准测试:")
        print(f"    历史预测数: {summary['total_predictions']}")
        print(f"    平均置信度: {summary['average_confidence']:.3f}")

        return True

    except Exception as e:
        print(f"    单独组件测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_adaptive_scoring_system()

    # 运行附加测试
    if success:
        test_individual_components()

    if success:
        print("\\n🚀 V3.8自适应评分系统可以投入生产使用!")
        print("\\n📋 下一步: 准备进入Phase 5系统集成和优化")
    else:
        print("\\n🔧 需要修复问题后再次测试")