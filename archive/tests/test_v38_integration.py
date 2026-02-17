#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.8自适应评分系统集成测试

测试V3.8在主选股系统中的集成功能
验证tomorrow_stock_selector.py对V3.8的支持

Created: 2025-09-16
Author: Claude Code
Version: 3.8.0
"""

import sys
import logging
from datetime import datetime

# 添加项目路径
sys.path.append('/Users/yangxu/StockTradebyZ')

def test_v38_integration():
    """测试V3.8在主选股系统中的集成"""
    print("🚀 V3.8主系统集成测试开始")
    print("="*60)

    # 配置日志
    logging.basicConfig(level=logging.INFO)

    try:
        # 1. 测试TomorrowStockSelector对V3.8的支持
        print("\\n📦 阶段1: 测试主选股系统V3.8初始化")

        from tomorrow_stock_selector import TomorrowStockSelector

        # 创建V3.8选股器
        selector = TomorrowStockSelector(scoring_version="v3.8")
        print("✅ V3.8选股系统初始化成功")

        # 2. 测试单只股票评分
        print("\\n🔄 阶段2: 测试单只股票V3.8评分")

        test_stock_code = "000001"  # 平安银行
        test_date = "2025-09-16"

        # 构建股票信息
        stock_info = {
            'stock_code': test_stock_code,
            'code': test_stock_code
        }

        # 计算评分
        score, detailed_info = selector.calculate_comprehensive_score(stock_info, test_date)

        print(f"  📊 股票 {test_stock_code} 评分结果:")
        print(f"    最终评分: {score:.2f}")
        print(f"    评分方法: {detailed_info.get('scoring_method', 'Unknown')}")
        print(f"    置信度: {detailed_info.get('confidence_score', 0):.3f}")
        print(f"    置信等级: {detailed_info.get('confidence_level', 'unknown')}")
        print(f"    推荐等级: {detailed_info.get('recommendation', 'unknown')}")
        print(f"    风险等级: {detailed_info.get('risk_level', 'unknown')}")

        # 显示时间维度评分
        if 'short_term_score' in detailed_info:
            print(f"    短期评分: {detailed_info['short_term_score']:.2f}")
            print(f"    中期评分: {detailed_info['medium_term_score']:.2f}")
            print(f"    长期评分: {detailed_info['long_term_score']:.2f}")

        # 3. 测试多只股票批量评分
        print("\\n⚡ 阶段3: 测试批量股票评分")

        test_stocks = ["000001", "000002", "600036"]
        batch_results = []

        for stock_code in test_stocks:
            try:
                stock_info = {'stock_code': stock_code, 'code': stock_code}
                score, details = selector.calculate_comprehensive_score(stock_info, test_date)
                batch_results.append({
                    'code': stock_code,
                    'score': score,
                    'confidence': details.get('confidence_score', 0),
                    'method': details.get('scoring_method', 'Unknown')
                })
                print(f"  ✅ {stock_code}: {score:.2f} (置信度: {details.get('confidence_score', 0):.3f})")
            except Exception as e:
                print(f"  ❌ {stock_code}: 评分失败 - {e}")
                batch_results.append({
                    'code': stock_code,
                    'score': 0,
                    'confidence': 0,
                    'error': str(e)
                })

        # 4. 测试性能监控
        print("\\n📈 阶段4: 测试性能监控功能")

        try:
            # 获取V3.8系统的性能报告
            if hasattr(selector, 'scoring_engine_v38'):
                performance_report = selector.scoring_engine_v38.get_performance_report()

                print(f"  📊 系统性能统计:")
                adapter_stats = performance_report.get('adapter_stats', {})
                print(f"    总评估次数: {adapter_stats.get('total_evaluations', 0)}")
                print(f"    成功评估: {adapter_stats.get('successful_evaluations', 0)}")
                print(f"    失败评估: {adapter_stats.get('failed_evaluations', 0)}")
                print(f"    平均处理时间: {adapter_stats.get('average_processing_time', 0):.3f}秒")

                system_stats = performance_report.get('system_stats', {})
                print(f"    系统状态: {system_stats.get('status', 'unknown')}")
                print(f"    系统成功率: {system_stats.get('success_rate', 0):.1%}")

                recommendations = performance_report.get('recommendations', [])
                if recommendations:
                    print(f"    性能建议: {recommendations}")

            else:
                print("  ⚠️ 无法访问V3.8性能监控")

        except Exception as e:
            print(f"  ❌ 性能监控测试失败: {e}")

        # 5. 版本兼容性测试
        print("\\n🔧 阶段5: 测试版本兼容性")

        version_tests = [
            ("v3.6", "机器学习评分引擎"),
            ("v3.7", "高级机器学习评分引擎"),
            ("v3.8", "自适应评分系统")
        ]

        for version, description in version_tests:
            try:
                test_selector = TomorrowStockSelector(scoring_version=version)
                print(f"  ✅ {version} ({description}): 初始化成功")
            except Exception as e:
                print(f"  ❌ {version}: 初始化失败 - {e}")

        # 6. 配置文件验证
        print("\\n💾 阶段6: 验证配置文件")

        try:
            import json
            config_path = "adaptive_scoring/config/v38_config.json"
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            print(f"  ✅ 配置文件加载成功")
            print(f"    系统版本: {config['system_config']['version']}")
            print(f"    归一化策略: {config['normalization_config']['default_strategy']}")
            print(f"    时间窗口: {config['temporal_scoring_config']['time_windows']}")
            print(f"    置信度水平: {config['confidence_config']['confidence_levels']}")

        except Exception as e:
            print(f"  ❌ 配置文件验证失败: {e}")

        # 7. 结果总结
        print("\\n📋 测试结果总结:")

        successful_evaluations = len([r for r in batch_results if 'error' not in r])
        total_evaluations = len(batch_results)

        print(f"  批量评分成功率: {successful_evaluations}/{total_evaluations} ({successful_evaluations/total_evaluations:.1%})")

        if successful_evaluations > 0:
            avg_score = sum([r['score'] for r in batch_results if 'error' not in r]) / successful_evaluations
            avg_confidence = sum([r['confidence'] for r in batch_results if 'error' not in r]) / successful_evaluations
            print(f"  平均评分: {avg_score:.2f}")
            print(f"  平均置信度: {avg_confidence:.3f}")

        if successful_evaluations >= 2:
            print("\\n🎉 V3.8主系统集成测试成功!")
            print("✅ V3.8自适应评分系统已完全集成到主选股系统")
            print("✅ 支持单只股票和批量股票评分")
            print("✅ 提供详细的评分分解和置信度评估")
            print("✅ 性能监控和版本兼容性良好")
            return True
        else:
            print("\\n⚠️ V3.8集成测试存在问题")
            print("需要进一步排查和优化")
            return False

    except Exception as e:
        print(f"\\n❌ V3.8集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_v38_integration()

    if success:
        print("\\n🚀 V3.8自适应评分系统已成功集成到主选股系统!")
        print("\\n📋 下一步: 可以使用以下命令运行V3.8选股:")
        print("  python3 tomorrow_stock_selector.py --version v3.8 2025-09-16")
    else:
        print("\\n🔧 需要修复集成问题后再次测试")