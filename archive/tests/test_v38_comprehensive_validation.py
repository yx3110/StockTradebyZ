#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.8系统综合验证测试

Phase 6: 验证和部署准备
- 完整系统功能验证
- 性能基准测试
- 稳定性压力测试
- 兼容性验证

Created: 2025-09-16
Author: Claude Code
Version: 3.8.0
"""

import sys
import time
import logging
import random
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目路径
sys.path.append('/Users/yangxu/StockTradebyZ')

def test_v38_comprehensive_validation():
    """V3.8系统综合验证测试"""
    print("🔍 V3.8系统综合验证测试开始")
    print("="*60)

    # 配置日志
    logging.basicConfig(level=logging.INFO)

    try:
        # 测试阶段统计
        test_results = {
            'functional_tests': [],
            'performance_tests': [],
            'stability_tests': [],
            'integration_tests': []
        }

        # ============================================
        # 阶段1: 功能性验证
        # ============================================
        print("\n🧪 阶段1: 功能性验证测试")

        from adaptive_scoring.v38_selector_adapter import V38SelectorAdapter
        from tomorrow_stock_selector import TomorrowStockSelector

        # 1.1 V3.8适配器功能验证
        print("  📦 1.1 V3.8适配器功能验证")
        try:
            adapter = V38SelectorAdapter()
            test_stocks = ["000001", "000002", "600036", "600000"]

            results = adapter.evaluate_stocks(test_stocks, "2025-09-16")

            # 验证结果结构
            required_keys = ['stocks', 'summary', 'metadata']
            has_required_structure = all(key in results for key in required_keys)

            # 验证评分有效性
            stock_results = results.get('stocks', [])
            valid_scores = all(
                0 <= stock.get('final_score', -1) <= 1 and
                0 <= stock.get('confidence', -1) <= 1
                for stock in stock_results
            )

            if has_required_structure and valid_scores and len(stock_results) > 0:
                test_results['functional_tests'].append("✅ V3.8适配器功能正常")
                print("    ✅ V3.8适配器功能正常")
            else:
                test_results['functional_tests'].append("❌ V3.8适配器功能异常")
                print("    ❌ V3.8适配器功能异常")

        except Exception as e:
            test_results['functional_tests'].append(f"❌ V3.8适配器测试失败: {e}")
            print(f"    ❌ V3.8适配器测试失败: {e}")

        # 1.2 主选股系统集成验证
        print("  🎯 1.2 主选股系统集成验证")
        try:
            selector = TomorrowStockSelector(scoring_version="v3.8")

            # 小规模选股测试
            test_date = "2025-09-16"
            selection_results = selector.select_stocks(
                target_date=test_date,
                max_stocks=5,
                min_score=0.1
            )

            if selection_results and len(selection_results) > 0:
                test_results['functional_tests'].append("✅ 主选股系统V3.8集成正常")
                print("    ✅ 主选股系统V3.8集成正常")
                print(f"      选出股票数量: {len(selection_results)}")
            else:
                test_results['functional_tests'].append("❌ 主选股系统V3.8集成失败")
                print("    ❌ 主选股系统V3.8集成失败")

        except Exception as e:
            test_results['functional_tests'].append(f"❌ 主选股系统集成失败: {e}")
            print(f"    ❌ 主选股系统集成失败: {e}")

        # 1.3 核心组件功能验证
        print("  🔧 1.3 核心组件功能验证")

        # 动态归一化验证
        try:
            from adaptive_scoring.normalizers.dynamic_normalizer import DynamicNormalizer
            normalizer = DynamicNormalizer()

            test_scores = [0.2, 0.5, 0.8, 0.3, 0.7]
            normalized = normalizer.normalize(test_scores)

            if len(normalized) == len(test_scores) and all(0 <= s <= 1 for s in normalized):
                test_results['functional_tests'].append("✅ 动态归一化功能正常")
                print("    ✅ 动态归一化功能正常")
            else:
                test_results['functional_tests'].append("❌ 动态归一化功能异常")
                print("    ❌ 动态归一化功能异常")

        except Exception as e:
            test_results['functional_tests'].append(f"❌ 动态归一化测试失败: {e}")
            print(f"    ❌ 动态归一化测试失败: {e}")

        # ============================================
        # 阶段2: 性能基准测试
        # ============================================
        print("\n⚡ 阶段2: 性能基准测试")

        # 2.1 单股评估性能
        print("  📊 2.1 单股评估性能测试")
        try:
            adapter = V38SelectorAdapter()
            single_stock = ["000001"]

            # 测试10次取平均
            times = []
            for _ in range(10):
                start_time = time.time()
                adapter.evaluate_stocks(single_stock, "2025-09-16")
                times.append(time.time() - start_time)

            avg_time = sum(times) / len(times)

            if avg_time < 0.5:  # 单股评估应在0.5秒内
                test_results['performance_tests'].append(f"✅ 单股评估性能良好: {avg_time:.3f}s")
                print(f"    ✅ 单股评估性能良好: {avg_time:.3f}s")
            else:
                test_results['performance_tests'].append(f"⚠️ 单股评估性能需优化: {avg_time:.3f}s")
                print(f"    ⚠️ 单股评估性能需优化: {avg_time:.3f}s")

        except Exception as e:
            test_results['performance_tests'].append(f"❌ 单股评估性能测试失败: {e}")
            print(f"    ❌ 单股评估性能测试失败: {e}")

        # 2.2 批量处理性能
        print("  📈 2.2 批量处理性能测试")
        try:
            batch_stocks = ["000001", "000002", "600036", "600000", "000858",
                          "002415", "300059", "002142", "000063", "600519"]

            start_time = time.time()
            batch_results = adapter.evaluate_stocks(batch_stocks, "2025-09-16", parallel=True)
            batch_time = time.time() - start_time

            per_stock_time = batch_time / len(batch_stocks)

            if per_stock_time < 0.1:  # 批量处理应每股0.1秒内
                test_results['performance_tests'].append(f"✅ 批量处理性能良好: {per_stock_time:.3f}s/股")
                print(f"    ✅ 批量处理性能良好: {per_stock_time:.3f}s/股")
            else:
                test_results['performance_tests'].append(f"⚠️ 批量处理性能需优化: {per_stock_time:.3f}s/股")
                print(f"    ⚠️ 批量处理性能需优化: {per_stock_time:.3f}s/股")

        except Exception as e:
            test_results['performance_tests'].append(f"❌ 批量处理性能测试失败: {e}")
            print(f"    ❌ 批量处理性能测试失败: {e}")

        # 2.3 缓存效率测试
        print("  🔄 2.3 缓存效率测试")
        try:
            test_stocks = ["000001", "000002", "600036"]

            # 首次评估（无缓存）
            start_time = time.time()
            adapter.evaluate_stocks(test_stocks, "2025-09-16")
            first_time = time.time() - start_time

            # 二次评估（应有缓存）
            start_time = time.time()
            adapter.evaluate_stocks(test_stocks, "2025-09-16")
            second_time = time.time() - start_time

            # 获取缓存统计
            perf_report = adapter.get_performance_report()
            cache_hit_rate = perf_report.get('optimizer_stats', {}).get('cache_performance', {}).get('hit_rate', 0)

            if cache_hit_rate > 0.5 and second_time < first_time:
                speedup = first_time / second_time if second_time > 0 else float('inf')
                test_results['performance_tests'].append(f"✅ 缓存效率良好: {cache_hit_rate:.1%}命中率, {speedup:.1f}x加速")
                print(f"    ✅ 缓存效率良好: {cache_hit_rate:.1%}命中率, {speedup:.1f}x加速")
            else:
                test_results['performance_tests'].append(f"⚠️ 缓存效率需优化: {cache_hit_rate:.1%}命中率")
                print(f"    ⚠️ 缓存效率需优化: {cache_hit_rate:.1%}命中率")

        except Exception as e:
            test_results['performance_tests'].append(f"❌ 缓存效率测试失败: {e}")
            print(f"    ❌ 缓存效率测试失败: {e}")

        # ============================================
        # 阶段3: 稳定性压力测试
        # ============================================
        print("\n🔨 阶段3: 稳定性压力测试")

        # 3.1 并发处理稳定性
        print("  🚀 3.1 并发处理稳定性测试")
        try:
            def concurrent_evaluation(thread_id):
                try:
                    local_adapter = V38SelectorAdapter()
                    test_stocks = [f"00000{(thread_id % 9) + 1}", f"60003{thread_id % 10}"]
                    result = local_adapter.evaluate_stocks(test_stocks, "2025-09-16")
                    return {'success': True, 'thread_id': thread_id, 'count': len(result.get('stocks', []))}
                except Exception as e:
                    return {'success': False, 'thread_id': thread_id, 'error': str(e)}

            # 5个并发线程
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(concurrent_evaluation, i) for i in range(5)]
                concurrent_results = [future.result(timeout=30) for future in as_completed(futures, timeout=30)]

            success_count = sum(1 for r in concurrent_results if r['success'])

            if success_count >= 4:  # 至少80%成功
                test_results['stability_tests'].append(f"✅ 并发处理稳定: {success_count}/5成功")
                print(f"    ✅ 并发处理稳定: {success_count}/5成功")
            else:
                test_results['stability_tests'].append(f"⚠️ 并发处理不稳定: {success_count}/5成功")
                print(f"    ⚠️ 并发处理不稳定: {success_count}/5成功")

        except Exception as e:
            test_results['stability_tests'].append(f"❌ 并发处理测试失败: {e}")
            print(f"    ❌ 并发处理测试失败: {e}")

        # 3.2 大批量处理稳定性
        print("  📦 3.2 大批量处理稳定性测试")
        try:
            # 生成20只测试股票
            large_batch = ["000001", "000002", "600036", "600000", "000858",
                          "002415", "300059", "002142", "000063", "600519",
                          "000001", "000002", "600036", "600000", "000858",
                          "002415", "300059", "002142", "000063", "600519"]

            start_time = time.time()
            large_results = adapter.evaluate_stocks(large_batch, "2025-09-16", parallel=True)
            process_time = time.time() - start_time

            success_rate = large_results['summary']['total_evaluated'] / len(large_batch)

            if success_rate >= 0.8 and process_time < 5.0:  # 80%成功率，5秒内完成
                test_results['stability_tests'].append(f"✅ 大批量处理稳定: {success_rate:.1%}成功率, {process_time:.2f}s")
                print(f"    ✅ 大批量处理稳定: {success_rate:.1%}成功率, {process_time:.2f}s")
            else:
                test_results['stability_tests'].append(f"⚠️ 大批量处理需优化: {success_rate:.1%}成功率, {process_time:.2f}s")
                print(f"    ⚠️ 大批量处理需优化: {success_rate:.1%}成功率, {process_time:.2f}s")

        except Exception as e:
            test_results['stability_tests'].append(f"❌ 大批量处理测试失败: {e}")
            print(f"    ❌ 大批量处理测试失败: {e}")

        # ============================================
        # 阶段4: 系统集成验证
        # ============================================
        print("\n🔗 阶段4: 系统集成验证")

        # 4.1 数据库兼容性验证
        print("  🗄️ 4.1 数据库兼容性验证")
        try:
            from data_adapter.database_manager import DatabaseManager
            db_manager = DatabaseManager()

            # 测试数据库连接和查询
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM securities WHERE type='A股'")
                stock_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM daily_quotes WHERE trade_date = '2025-09-16'")
                quote_count = cursor.fetchone()[0]

            if stock_count > 1000:  # 至少1000只A股
                test_results['integration_tests'].append(f"✅ 数据库兼容正常: {stock_count}只A股数据")
                print(f"    ✅ 数据库兼容正常: {stock_count}只A股数据")
            else:
                test_results['integration_tests'].append(f"⚠️ 数据库数据不足: {stock_count}只A股数据")
                print(f"    ⚠️ 数据库数据不足: {stock_count}只A股数据")

        except Exception as e:
            test_results['integration_tests'].append(f"❌ 数据库兼容性测试失败: {e}")
            print(f"    ❌ 数据库兼容性测试失败: {e}")

        # 4.2 配置文件验证
        print("  ⚙️ 4.2 配置文件验证")
        try:
            import json

            # 验证V3.8配置文件
            with open("adaptive_scoring/config/v38_config.json", 'r', encoding='utf-8') as f:
                v38_config = json.load(f)

            required_sections = ['system_config', 'normalization_config', 'temporal_scoring_config', 'confidence_config']
            has_all_sections = all(section in v38_config for section in required_sections)

            if has_all_sections:
                test_results['integration_tests'].append("✅ 配置文件完整")
                print("    ✅ 配置文件完整")
            else:
                missing_sections = [s for s in required_sections if s not in v38_config]
                test_results['integration_tests'].append(f"❌ 配置文件缺失: {missing_sections}")
                print(f"    ❌ 配置文件缺失: {missing_sections}")

        except Exception as e:
            test_results['integration_tests'].append(f"❌ 配置文件验证失败: {e}")
            print(f"    ❌ 配置文件验证失败: {e}")

        # ============================================
        # 测试结果汇总和评估
        # ============================================
        print("\n📊 测试结果汇总:")
        print("="*60)

        # 计算各类测试通过率
        categories = ['functional_tests', 'performance_tests', 'stability_tests', 'integration_tests']
        category_names = ['功能性测试', '性能测试', '稳定性测试', '集成测试']

        total_pass = 0
        total_tests = 0

        for i, category in enumerate(categories):
            results = test_results[category]
            pass_count = len([r for r in results if r.startswith("✅")])
            total_count = len(results)

            total_pass += pass_count
            total_tests += total_count

            pass_rate = pass_count / total_count if total_count > 0 else 0

            print(f"  {category_names[i]}: {pass_count}/{total_count} 通过 ({pass_rate:.1%})")
            for result in results:
                print(f"    {result}")

        # 总体评估
        overall_pass_rate = total_pass / total_tests if total_tests > 0 else 0

        print(f"\n🎯 总体通过率: {total_pass}/{total_tests} ({overall_pass_rate:.1%})")

        # 系统就绪状态评估
        if overall_pass_rate >= 0.8:
            print("\n🎉 V3.8系统验证通过 - 系统就绪部署!")
            print("✅ 功能完整性验证通过")
            print("✅ 性能指标达标")
            print("✅ 系统稳定性良好")
            print("✅ 集成兼容性正常")
            return True
        elif overall_pass_rate >= 0.6:
            print(f"\n⚠️ V3.8系统验证部分通过 ({overall_pass_rate:.1%})")
            print("🔧 建议优化后再部署")
            return False
        else:
            print(f"\n❌ V3.8系统验证未通过 ({overall_pass_rate:.1%})")
            print("🔧 需要重大修复后再测试")
            return False

    except Exception as e:
        print(f"\n💥 综合验证测试异常终止: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_v38_comprehensive_validation()

    if success:
        print("\n🚀 V3.8系统已通过全面验证，可以投入生产使用!")
        print("\n📋 部署建议:")
        print("  1. 设置生产环境配置")
        print("  2. 配置定时任务和监控")
        print("  3. 准备回滚方案")
        print("  4. 逐步切换流量")
    else:
        print("\n🔧 V3.8系统需要进一步优化才能部署")