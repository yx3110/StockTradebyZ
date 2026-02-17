#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存性能基准测试

测试SmartCacheManager vs 简单字典缓存的性能差异
"""

import time
import sys
import os

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from hikyuu_integration.data_adapter import HikyuuStyleDataAdapter
from hikyuu_integration.query import Query
from data_adapter.database_manager import DatabaseManager

def benchmark_cache_performance():
    """
    测试缓存性能

    场景1: 预加载后重复查询（测试LRU缓存命中率）
    场景2: 子范围查询（测试智能缓存匹配）
    场景3: 大量股票回测（测试LRU淘汰机制）
    """
    print("=" * 80)
    print("🚀 Hikyuu集成框架 - 缓存性能基准测试")
    print("=" * 80)

    # 使用真实数据库
    db = DatabaseManager(db_path='stock_data.db')
    adapter = HikyuuStyleDataAdapter(db_manager=db, cache_capacity=100)

    # 获取可用的股票列表
    stocks = adapter.get_all_stocks(stock_type='A股')
    if not stocks or len(stocks) < 10:
        print("❌ 数据库中没有足够的股票数据")
        return

    test_stocks = stocks[:10]  # 使用前10只股票
    print(f"\n📋 测试股票: {test_stocks}")

    # ==================== 场景1: 预加载后重复查询 ====================
    print("\n" + "=" * 80)
    print("📊 场景1: 预加载后重复查询 (测试LRU缓存命中率)")
    print("=" * 80)

    start_date = '2025-07-01'
    end_date = '2025-09-30'

    # 预加载数据
    print(f"\n🔄 预加载 {len(test_stocks)} 只股票数据 ({start_date} → {end_date})...")
    preload_start = time.time()
    adapter.preload_data(test_stocks, start_date, end_date)
    preload_time = time.time() - preload_start
    print(f"✅ 预加载完成: {preload_time:.3f}秒")

    # 重复查询10次
    print(f"\n🔍 重复查询 {len(test_stocks)} 只股票 × 10次...")
    query_start = time.time()
    for _ in range(10):
        for code in test_stocks:
            kdata = adapter.get_kdata(code, Query(start=start_date, end=end_date))
    query_time = time.time() - query_start

    # 打印缓存统计
    print(f"✅ 查询完成: {query_time:.3f}秒")
    print(f"\n📊 缓存统计:")
    adapter.print_cache_stats()

    # ==================== 场景2: 子范围查询 ====================
    print("\n" + "=" * 80)
    print("📊 场景2: 子范围查询 (测试智能缓存匹配)")
    print("=" * 80)

    # 查询子范围（应该从缓存的大范围中提取）
    sub_start = '2025-08-01'
    sub_end = '2025-08-31'

    print(f"\n🔍 查询子范围 ({sub_start} → {sub_end})...")
    print(f"   （预加载范围: {start_date} → {end_date}）")

    sub_query_start = time.time()
    for code in test_stocks:
        kdata = adapter.get_kdata(code, Query(start=sub_start, end=sub_end))
        # print(f"   {code}: {len(kdata)}条数据")
    sub_query_time = time.time() - sub_query_start

    print(f"✅ 子范围查询完成: {sub_query_time:.3f}秒")
    print(f"\n📊 缓存统计 (after sub-range queries):")
    adapter.print_cache_stats()

    # ==================== 场景3: 大量股票回测 (测试LRU淘汰) ====================
    print("\n" + "=" * 80)
    print("📊 场景3: 大量股票回测 (测试LRU淘汰机制)")
    print("=" * 80)

    # 使用更多股票（超过缓存容量）
    large_stock_list = stocks[:50] if len(stocks) >= 50 else stocks
    print(f"\n🔄 回测 {len(large_stock_list)} 只股票 (缓存容量: 100)...")

    large_backtest_start = time.time()
    for code in large_stock_list:
        kdata = adapter.get_kdata(code, Query(start=start_date, end=end_date))
    large_backtest_time = time.time() - large_backtest_start

    print(f"✅ 回测完成: {large_backtest_time:.3f}秒")
    print(f"\n📊 最终缓存统计:")
    adapter.print_cache_stats()

    # ==================== 总结 ====================
    print("\n" + "=" * 80)
    print("📈 性能总结")
    print("=" * 80)

    cache_stats = adapter.get_cache_stats()

    print(f"\n✅ 场景1 - 预加载+重复查询:")
    print(f"   预加载时间: {preload_time:.3f}秒")
    print(f"   10轮查询时间: {query_time:.3f}秒")
    print(f"   平均每轮: {query_time/10:.3f}秒")

    print(f"\n✅ 场景2 - 智能缓存匹配:")
    print(f"   子范围查询: {sub_query_time:.3f}秒")

    print(f"\n✅ 场景3 - LRU淘汰测试:")
    print(f"   {len(large_stock_list)}只股票查询: {large_backtest_time:.3f}秒")

    print(f"\n✅ 缓存效率:")
    print(f"   命中率: {cache_stats['hit_rate']:.1f}%")
    print(f"   命中次数: {cache_stats['hits']}")
    print(f"   未命中次数: {cache_stats['misses']}")
    print(f"   淘汰次数: {cache_stats['evictions']}")
    print(f"   当前大小: {cache_stats['size']}/{cache_stats['capacity']}")

    print("\n" + "=" * 80)
    print("🎉 基准测试完成！")
    print("=" * 80)

if __name__ == '__main__':
    benchmark_cache_performance()
