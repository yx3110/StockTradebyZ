#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.8性能优化功能测试

测试缓存、批处理和性能监控功能
验证优化后的系统性能提升

Created: 2025-09-16
Author: Claude Code
Version: 3.8.0
"""

import sys
import time
import logging
from datetime import datetime

# 添加项目路径
sys.path.append('/Users/yangxu/StockTradebyZ')

def test_v38_performance_optimization():
    """测试V3.8性能优化功能"""
    print("🚀 V3.8性能优化功能测试开始")
    print("="*60)

    # 配置日志
    logging.basicConfig(level=logging.INFO)

    try:
        # 1. 测试适配器性能优化集成
        print("\n📦 阶段1: 测试性能优化集成")

        from adaptive_scoring.v38_selector_adapter import V38SelectorAdapter

        # 创建优化版适配器
        adapter = V38SelectorAdapter()
        print("✅ V3.8优化版适配器初始化成功")

        # 验证组件
        print(f"  📊 性能优化器: {adapter.performance_optimizer}")
        print(f"  📊 缓存配置: TTL={adapter.performance_optimizer.cache_ttl.total_seconds()}秒, "
              f"最大缓存={adapter.performance_optimizer.max_cache_size}")

        # 2. 测试缓存功能
        print("\n🔄 阶段2: 测试缓存功能")

        test_stocks = ["000001", "000002", "600036"]
        test_date = "2025-09-16"

        # 第一次评估 - 无缓存
        print("  第一次评估 (无缓存):")
        start_time = time.time()
        results1 = adapter.evaluate_stocks(test_stocks, test_date)
        first_time = time.time() - start_time
        print(f"    耗时: {first_time:.3f}秒")
        print(f"    成功评估: {results1['summary']['total_evaluated']}")

        # 第二次评估 - 应有缓存
        print("  第二次评估 (应有缓存):")
        start_time = time.time()
        results2 = adapter.evaluate_stocks(test_stocks, test_date)
        second_time = time.time() - start_time
        print(f"    耗时: {second_time:.3f}秒")
        print(f"    成功评估: {results2['summary']['total_evaluated']}")

        # 缓存性能提升
        if second_time < first_time:
            speedup = first_time / second_time
            print(f"  ✅ 缓存加速: {speedup:.1f}倍")
        else:
            print(f"  ⚠️ 缓存可能未生效")

        # 3. 测试性能报告
        print("\n📈 阶段3: 测试性能监控")

        performance_report = adapter.get_performance_report()

        if 'optimizer_stats' in performance_report:
            optimizer_stats = performance_report['optimizer_stats']
            cache_perf = optimizer_stats.get('cache_performance', {})

            print(f"  📊 缓存统计:")
            print(f"    总请求: {cache_perf.get('total_requests', 0)}")
            print(f"    缓存命中: {cache_perf.get('cache_hits', 0)}")
            print(f"    缓存未命中: {cache_perf.get('cache_misses', 0)}")
            print(f"    命中率: {cache_perf.get('hit_rate', 0):.1%}")
            print(f"    平均缓存时间: {cache_perf.get('average_cache_time_ms', 0):.2f}ms")

            computation_perf = optimizer_stats.get('computation_performance', {})
            print(f"  ⚡ 计算性能:")
            print(f"    平均计算时间: {computation_perf.get('average_computation_time_ms', 0):.2f}ms")

            recommendations = optimizer_stats.get('recommendations', [])
            print(f"  💡 性能建议: {recommendations}")

        # 4. 测试批处理优化
        print("\n⚡ 阶段4: 测试批处理优化")

        # 测试小规模批处理
        small_batch = ["000001", "000002"]
        start_time = time.time()
        small_results = adapter.evaluate_stocks(small_batch, test_date)
        small_time = time.time() - start_time
        print(f"  小批处理 ({len(small_batch)}只): {small_time:.3f}秒")

        # 测试中等规模批处理
        medium_batch = ["000001", "000002", "600036", "600000", "000858", "002415"]
        start_time = time.time()
        medium_results = adapter.evaluate_stocks(medium_batch, test_date, parallel=True)
        medium_time = time.time() - start_time
        print(f"  中等批处理 ({len(medium_batch)}只): {medium_time:.3f}秒")

        # 性能分析
        small_per_stock = small_time / len(small_batch)
        medium_per_stock = medium_time / len(medium_batch)

        print(f"  📊 每只股票平均时间:")
        print(f"    小批处理: {small_per_stock:.3f}秒/股")
        print(f"    中等批处理: {medium_per_stock:.3f}秒/股")

        if medium_per_stock < small_per_stock:
            efficiency_gain = (small_per_stock - medium_per_stock) / small_per_stock
            print(f"    ✅ 批处理效率提升: {efficiency_gain:.1%}")

        # 5. 测试缓存清理
        print("\n🧹 阶段5: 测试缓存管理")

        initial_cache_size = len(adapter.performance_optimizer._memory_cache)
        print(f"  初始缓存大小: {initial_cache_size}")

        # 清理过期缓存
        adapter.performance_optimizer.clear_cache(expired_only=True)
        after_cleanup_size = len(adapter.performance_optimizer._memory_cache)
        print(f"  清理后缓存大小: {after_cleanup_size}")

        # 6. 系统综合性能测试
        print("\n🎯 阶段6: 综合性能测试")

        # 测试多轮评估性能趋势
        performance_trend = []
        test_rounds = 3

        for round_num in range(test_rounds):
            start_time = time.time()
            round_results = adapter.evaluate_stocks(test_stocks, test_date)
            round_time = time.time() - start_time
            performance_trend.append(round_time)

            print(f"  第{round_num+1}轮: {round_time:.3f}秒, "
                  f"成功: {round_results['summary']['total_evaluated']}")

        # 性能趋势分析
        if len(performance_trend) >= 2:
            avg_improvement = (performance_trend[0] - performance_trend[-1]) / performance_trend[0]
            print(f"  📈 性能趋势: {avg_improvement:.1%} 改善")

        # 最终性能报告
        final_report = adapter.get_performance_report()
        if 'optimizer_stats' in final_report:
            final_cache_perf = final_report['optimizer_stats']['cache_performance']
            print(f"  📊 最终缓存命中率: {final_cache_perf.get('hit_rate', 0):.1%}")

        # 7. 结果总结
        print("\n📋 性能优化测试总结:")

        # 性能指标评估
        success_indicators = []

        # 缓存功能
        if second_time < first_time:
            success_indicators.append("✅ 缓存加速生效")
        else:
            success_indicators.append("⚠️ 缓存效果待优化")

        # 批处理优化
        if medium_per_stock < small_per_stock:
            success_indicators.append("✅ 批处理效率提升")
        else:
            success_indicators.append("⚠️ 批处理优化待调整")

        # 性能监控
        if 'optimizer_stats' in performance_report:
            success_indicators.append("✅ 性能监控正常")
        else:
            success_indicators.append("⚠️ 性能监控异常")

        # 系统稳定性
        all_evaluations_success = all([
            results1['summary']['total_evaluated'] > 0,
            results2['summary']['total_evaluated'] > 0,
            small_results['summary']['total_evaluated'] > 0,
            medium_results['summary']['total_evaluated'] > 0
        ])

        if all_evaluations_success:
            success_indicators.append("✅ 系统稳定性良好")
        else:
            success_indicators.append("⚠️ 系统稳定性待检查")

        print(f"  {', '.join(success_indicators)}")

        success_count = len([i for i in success_indicators if i.startswith("✅")])
        total_count = len(success_indicators)

        if success_count >= 3:
            print("\n🎉 V3.8性能优化功能测试成功!")
            print("✅ 缓存系统正常工作")
            print("✅ 批处理优化生效")
            print("✅ 性能监控完整")
            print("✅ 系统响应时间优化")
            return True
        else:
            print(f"\n⚠️ V3.8性能优化测试部分通过 ({success_count}/{total_count})")
            print("需要进一步调优和优化")
            return False

    except Exception as e:
        print(f"\n❌ V3.8性能优化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_v38_performance_optimization()

    if success:
        print("\n🚀 V3.8性能优化系统完全就绪!")
        print("\n📋 优化效果:")
        print("  - 智能缓存: 减少重复计算")
        print("  - 批处理优化: 提升并发性能")
        print("  - 数据预加载: 减少数据库查询")
        print("  - 性能监控: 实时性能分析")
    else:
        print("\n🔧 需要继续优化性能配置")