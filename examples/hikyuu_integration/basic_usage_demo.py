#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hikyuu整合基础使用演示

展示如何使用Hikyuu风格API访问我们的SQLite数据
"""

import sys
sys.path.append('../..')

from hikyuu_integration import HikyuuStyleDataAdapter, Query


def demo_basic_usage():
    """基础使用演示"""
    print("=" * 80)
    print("🎯 Hikyuu风格API - 基础使用演示")
    print("=" * 80)

    # 1. 创建数据适配器
    print("\n📋 步骤1: 创建数据适配器")
    adapter = HikyuuStyleDataAdapter()
    print("   ✅ 数据适配器初始化成功")

    # 2. 获取股票对象 (类似Hikyuu的sm['sh000001'])
    print("\n📋 步骤2: 获取股票对象")
    stock = adapter.get_stock('000001')
    print(f"   ✅ {stock}")
    print(f"   名称: {stock.name}")
    print(f"   行业: {stock.industry}")

    # 3. 获取K线数据 (类似Hikyuu的stock.get_kdata(Query(-150)))
    print("\n📋 步骤3: 获取K线数据")

    # 方法1: 最近N天
    kdata = stock.get_kdata(Query(-50))
    print(f"   ✅ 最近50天: {kdata}")
    print(f"   日期范围: {kdata.datetime[0]} 至 {kdata.datetime[-1]}")

    # 方法2: 指定日期区间
    kdata2 = adapter.get_kdata('000001', Query(start='2025-01-01', end='2025-09-30'))
    print(f"   ✅ 日期区间: {kdata2}")

    # 4. 访问K线数据
    print("\n📋 步骤4: 访问K线数据")

    # 数组方式
    print(f"   收盘价数组长度: {len(kdata.close)}")
    print(f"   最新收盘价: {kdata.close[-1]:.2f}")
    print(f"   成交量: {kdata.volume[-1]:,.0f}")

    # 索引方式
    last_bar = kdata[-1]
    print(f"   最后一根K线: {last_bar['datetime']}")
    print(f"     开盘: {last_bar['open']:.2f}")
    print(f"     最高: {last_bar['high']:.2f}")
    print(f"     最低: {last_bar['low']:.2f}")
    print(f"     收盘: {last_bar['close']:.2f}")

    # 5. 获取技术指标
    print("\n📋 步骤5: 获取技术指标")

    bbi = kdata.get_indicator('BBI')
    if bbi is not None:
        print(f"   ✅ BBI指标: 最新值={bbi[-1]:.2f}")

    kdj_k = kdata.get_indicator('KDJ_K')
    if kdj_k is not None:
        print(f"   ✅ KDJ_K: 最新值={kdj_k[-1]:.2f}")

    # 6. 简单策略判断
    print("\n📋 步骤6: 简单策略判断")
    close_price = kdata.close[-1]
    bbi_value = bbi[-1] if bbi is not None else 0
    kdj_k_value = kdj_k[-1] if kdj_k is not None else 0

    print(f"   当前收盘价: {close_price:.2f}")
    print(f"   BBI: {bbi_value:.2f}")
    print(f"   KDJ_K: {kdj_k_value:.2f}")

    if close_price > bbi_value and kdj_k_value < 20:
        print("   🟢 买入信号: 价格 > BBI 且 KDJ超卖")
    elif close_price < bbi_value:
        print("   🔴 卖出信号: 价格 < BBI")
    else:
        print("   ⚪ 观望: 无明确信号")


def demo_batch_query():
    """批量查询演示"""
    print("\n" + "=" * 80)
    print("🎯 批量查询演示")
    print("=" * 80)

    adapter = HikyuuStyleDataAdapter()

    # 获取多只股票
    print("\n📋 获取多只股票数据...")
    stock_list = ['000001', '000002', '600000', '601318']

    for code in stock_list:
        stock = adapter.get_stock(code)
        kdata = stock.get_kdata(Query(-1))  # 最新一天

        if len(kdata) > 0:
            close = kdata.close[-1]
            print(f"   {code} ({stock.name}): 最新收盘价={close:.2f}")


def demo_performance():
    """性能优化演示"""
    print("\n" + "=" * 80)
    print("🎯 性能优化演示")
    print("=" * 80)

    import time

    adapter = HikyuuStyleDataAdapter()
    stock_list = ['000001', '000002', '600000', '600036', '601318']

    # 方法1: 逐个查询
    print("\n方法1: 逐个查询")
    start = time.time()
    for code in stock_list:
        kdata = adapter.get_kdata(code, Query(start='2025-01-01', end='2025-09-30'))
    time1 = time.time() - start
    print(f"   耗时: {time1:.3f}秒")

    # 方法2: 预加载
    print("\n方法2: 预加载到缓存")
    adapter2 = HikyuuStyleDataAdapter()
    start = time.time()
    adapter2.preload_data(stock_list, '2025-01-01', '2025-09-30')
    time2 = time.time() - start
    print(f"   耗时: {time2:.3f}秒")
    print(f"   🚀 速度提升: {time1/time2:.1f}倍")


if __name__ == '__main__':
    print("""
╔═══════════════════════════════════════════════════════════════╗
║  Hikyuu风格API基础使用演示                                     ║
║                                                               ║
║  Phase 1 完成: 数据适配层                                     ║
║  ✅ Query查询对象                                            ║
║  ✅ KData K线数据对象                                        ║
║  ✅ Stock股票对象                                            ║
║  ✅ HikyuuStyleDataAdapter数据适配器                         ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    # 运行演示
    demo_basic_usage()
    demo_batch_query()
    demo_performance()

    print("\n" + "=" * 80)
    print("✅ Phase 1 完成！数据适配层工作正常")
    print("=" * 80)
    print("\n📖 下一步:")
    print("   Phase 2: 实现Signal基类和ML评分Signal适配器")
    print("   Phase 3: 实现回测引擎")
    print("   Phase 4: 性能优化和完整测试")
    print("\n详细计划见: HIKYUU_INTEGRATION_PLAN.md")
    print("=" * 80)
