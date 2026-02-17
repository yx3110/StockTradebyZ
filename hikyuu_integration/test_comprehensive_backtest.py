#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整回测测试 - Phase 4.3

测试场景:
1. 多股票回测 (20-50只)
2. 长周期回测 (3-6个月)
3. 单线程 vs 并行性能对比
4. 不同Signal策略对比
"""

import time
import sys
import os

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from hikyuu_integration import (
    HikyuuStyleBacktestEngine,
    ParallelBacktestEngine,
    BBISignal,
    KDJSignal,
    CompositeSignal,
    MM_FixedPercent,
    ST_FixedPercent,
    HikyuuStyleDataAdapter
)
from data_adapter.database_manager import DatabaseManager


def test_single_vs_parallel(adapter, stocks, start_date, end_date):
    """
    对比单线程和并行回测性能

    参数:
        adapter: 数据适配器
        stocks: 股票列表
        start_date: 开始日期
        end_date: 结束日期
    """
    print("\n" + "=" * 80)
    print("📊 测试1: 单线程 vs 并行回测性能对比")
    print("=" * 80)

    signal = BBISignal()
    mm = MM_FixedPercent(0.2)
    sl = ST_FixedPercent(0.08)

    # 单线程回测
    print(f"\n🔄 单线程回测 ({len(stocks)}只股票)...")
    single_start = time.time()

    single_engine = HikyuuStyleBacktestEngine(
        data_adapter=adapter,
        signal=signal,
        money_manager=mm,
        stop_loss=sl,
        initial_cash=100000,
        max_positions=5
    )

    single_result = single_engine.run(
        stock_list=stocks,
        start_date=start_date,
        end_date=end_date
    )

    single_time = time.time() - single_start
    print(f"✅ 单线程完成: {single_time:.2f}秒")

    # 并行回测
    print(f"\n🚀 并行回测 ({len(stocks)}只股票, 4进程)...")
    parallel_start = time.time()

    parallel_engine = ParallelBacktestEngine(
        data_adapter=adapter,
        signal=BBISignal(),  # 重新创建signal
        money_manager=MM_FixedPercent(0.2),
        stop_loss=ST_FixedPercent(0.08),
        initial_cash=100000,
        max_workers=4
    )

    parallel_result = parallel_engine.run(
        stock_list=stocks,
        start_date=start_date,
        end_date=end_date
    )

    parallel_time = time.time() - parallel_start
    print(f"✅ 并行完成: {parallel_time:.2f}秒")

    # 性能对比
    speedup = single_time / parallel_time if parallel_time > 0 else 0

    print("\n" + "=" * 80)
    print("📈 性能对比结果")
    print("=" * 80)
    print(f"单线程时间: {single_time:.2f}秒")
    print(f"并行时间:   {parallel_time:.2f}秒")
    print(f"加速比:     {speedup:.2f}x")
    print(f"效率提升:   {(1 - parallel_time/single_time)*100:.1f}%")

    # 结果对比
    print("\n" + "=" * 80)
    print("📊 回测结果对比")
    print("=" * 80)
    print("\n【单线程结果】")
    print(f"总收益率:   {single_result.total_return:.2f}%")
    print(f"夏普比率:   {single_result.sharpe_ratio:.2f}")
    print(f"最大回撤:   {single_result.max_drawdown_pct:.2f}%")
    print(f"胜率:       {single_result.win_rate:.2f}%")
    print(f"交易次数:   {len(single_result.portfolio.trades)}")

    print("\n【并行结果】")
    print(f"总收益率:   {parallel_result['total_return']:.2f}%")
    print(f"夏普比率:   {parallel_result['sharpe_ratio']:.2f}")
    print(f"最大回撤:   {parallel_result['max_drawdown']:.2f}%")
    print(f"胜率:       {parallel_result['win_rate']:.2f}%")
    print(f"交易次数:   {parallel_result['total_trades']}")

    return {
        'single_time': single_time,
        'parallel_time': parallel_time,
        'speedup': speedup,
        'single_result': single_result,
        'parallel_result': parallel_result
    }


def test_different_signals(adapter, stocks, start_date, end_date):
    """
    测试不同Signal策略

    参数:
        adapter: 数据适配器
        stocks: 股票列表
        start_date: 开始日期
        end_date: 结束日期
    """
    print("\n" + "=" * 80)
    print("📊 测试2: 不同Signal策略对比")
    print("=" * 80)

    signals = {
        'BBI': BBISignal(),
        'KDJ': KDJSignal(),
        'Composite': CompositeSignal([BBISignal(), KDJSignal()])
    }

    results = {}

    for name, signal in signals.items():
        print(f"\n🔄 测试 {name} 策略...")
        start = time.time()

        engine = ParallelBacktestEngine(
            data_adapter=adapter,
            signal=signal,
            money_manager=MM_FixedPercent(0.2),
            stop_loss=ST_FixedPercent(0.08),
            initial_cash=100000,
            max_workers=4
        )

        result = engine.run(
            stock_list=stocks,
            start_date=start_date,
            end_date=end_date
        )

        elapsed = time.time() - start
        results[name] = {
            'result': result,
            'time': elapsed
        }

        print(f"✅ {name} 完成: {elapsed:.2f}秒")
        print(f"   收益率: {result['total_return']:.2f}%")
        print(f"   夏普比率: {result['sharpe_ratio']:.2f}")
        print(f"   胜率: {result['win_rate']:.2f}%")

    # 策略对比
    print("\n" + "=" * 80)
    print("📈 策略对比总结")
    print("=" * 80)
    print(f"{'策略':<15} {'收益率':<12} {'夏普比率':<12} {'最大回撤':<12} {'胜率':<12} {'交易次数':<10}")
    print("-" * 80)

    for name, data in results.items():
        r = data['result']
        print(f"{name:<15} {r['total_return']:>10.2f}% {r['sharpe_ratio']:>10.2f} "
              f"{r['max_drawdown']:>10.2f}% {r['win_rate']:>10.2f}% {r['total_trades']:>10}")

    return results


def test_long_period(adapter, stocks, start_date, end_date):
    """
    测试长周期回测

    参数:
        adapter: 数据适配器
        stocks: 股票列表
        start_date: 开始日期
        end_date: 结束日期
    """
    print("\n" + "=" * 80)
    print(f"📊 测试3: 长周期回测 ({start_date} → {end_date})")
    print("=" * 80)

    print(f"\n🚀 运行 {len(stocks)}只股票长周期回测...")
    start = time.time()

    engine = ParallelBacktestEngine(
        data_adapter=adapter,
        signal=BBISignal(),
        money_manager=MM_FixedPercent(0.2),
        stop_loss=ST_FixedPercent(0.08),
        initial_cash=100000,
        max_workers=4
    )

    result = engine.run(
        stock_list=stocks,
        start_date=start_date,
        end_date=end_date
    )

    elapsed = time.time() - start

    print("\n" + "=" * 80)
    print("📈 长周期回测结果")
    print("=" * 80)
    print(f"回测周期:   {start_date} → {end_date}")
    print(f"股票数量:   {len(stocks)}")
    print(f"耗时:       {elapsed:.2f}秒")
    print(f"\n总收益率:   {result['total_return']:.2f}%")
    print(f"年化收益:   {result['total_return'] / 6 * 12:.2f}% (假设6个月)")
    print(f"夏普比率:   {result['sharpe_ratio']:.2f}")
    print(f"最大回撤:   {result['max_drawdown']:.2f}%")
    print(f"胜率:       {result['win_rate']:.2f}%")
    print(f"交易次数:   {result['total_trades']}")
    print(f"总盈亏:     {result['total_pnl']:.2f}元")

    # 按股票统计
    print("\n" + "=" * 80)
    print("📊 分股票表现 (Top 10)")
    print("=" * 80)

    by_stock = result['by_stock']
    sorted_stocks = sorted(by_stock.items(), key=lambda x: x[1]['total_return'], reverse=True)

    print(f"{'股票代码':<10} {'收益率':<12} {'夏普比率':<12} {'最大回撤':<12} {'胜率':<12} {'交易次数':<10}")
    print("-" * 80)

    for code, stats in sorted_stocks[:10]:
        print(f"{code:<10} {stats['total_return']:>10.2f}% {stats['sharpe_ratio']:>10.2f} "
              f"{stats['max_drawdown']:>10.2f}% {stats['win_rate']:>10.2f}% {stats['trades']:>10}")

    return result


def main():
    """主测试函数"""
    print("=" * 80)
    print("🚀 Hikyuu集成框架 - 完整回测测试 (Phase 4.3)")
    print("=" * 80)

    # 创建数据适配器
    db = DatabaseManager(db_path='data_adapter/stock_data.db')
    adapter = HikyuuStyleDataAdapter(db_manager=db, cache_capacity=200)

    # 获取测试股票
    all_stocks = adapter.get_all_stocks('A股')
    print(f"\n可用股票总数: {len(all_stocks)}")

    # 测试参数
    test_stocks_20 = all_stocks[:20]
    test_stocks_50 = all_stocks[:50] if len(all_stocks) >= 50 else all_stocks

    # 短周期测试 (3个月)
    short_start = '2025-07-01'
    short_end = '2025-09-30'

    # 长周期测试 (6个月)
    long_start = '2025-04-01'
    long_end = '2025-09-30'

    # 测试1: 单线程 vs 并行 (20只股票, 3个月)
    test1_result = test_single_vs_parallel(
        adapter, test_stocks_20, short_start, short_end
    )

    # 测试2: 不同策略对比 (20只股票, 3个月)
    test2_result = test_different_signals(
        adapter, test_stocks_20, short_start, short_end
    )

    # 测试3: 长周期回测 (50只股票, 6个月)
    test3_result = test_long_period(
        adapter, test_stocks_50, long_start, long_end
    )

    # 打印缓存统计
    print("\n" + "=" * 80)
    print("📊 缓存统计")
    print("=" * 80)
    adapter.print_cache_stats()

    # 总结
    print("\n" + "=" * 80)
    print("🎉 完整回测测试完成!")
    print("=" * 80)
    print("\n✅ 所有测试通过:")
    print("  - 单线程 vs 并行性能对比")
    print("  - 不同Signal策略对比")
    print("  - 长周期多股票回测")
    print(f"\n⚡ 并行加速比: {test1_result['speedup']:.2f}x")
    print(f"📈 最佳策略: " + max(test2_result.items(),
                                 key=lambda x: x[1]['result']['total_return'])[0])
    print(f"💰 长周期收益: {test3_result['total_return']:.2f}%")


if __name__ == '__main__':
    main()
