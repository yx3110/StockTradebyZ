#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
框架对比测试 - Phase 4.4

对比Hikyuu风格回测框架 vs extensible_backtest_engine:
1. 使用相同的测试参数
2. 对比回测结果（收益率、夏普比率、交易次数等）
3. 分析性能差异
4. 识别各自优势
"""

import time
import sys
import os

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from hikyuu_integration import (
    HikyuuStyleBacktestEngine,
    BBISignal,
    MM_FixedPercent,
    ST_FixedPercent,
    HikyuuStyleDataAdapter
)
from data_adapter.database_manager import DatabaseManager


def run_hikyuu_backtest(stocks, start_date, end_date, initial_cash):
    """
    运行Hikyuu风格回测

    参数:
        stocks: 股票列表
        start_date: 开始日期
        end_date: 结束日期
        initial_cash: 初始资金

    返回:
        回测结果字典
    """
    print("\n" + "=" * 80)
    print("📊 运行Hikyuu风格回测框架")
    print("=" * 80)

    # 创建数据适配器
    db = DatabaseManager(db_path='data_adapter/stock_data.db')
    adapter = HikyuuStyleDataAdapter(db_manager=db, cache_capacity=200)

    # 创建回测引擎
    engine = HikyuuStyleBacktestEngine(
        data_adapter=adapter,
        signal=BBISignal(),
        money_manager=MM_FixedPercent(0.2),  # 20%资金比例
        stop_loss=ST_FixedPercent(0.08),     # 8%止损
        initial_cash=initial_cash,
        max_positions=10
    )

    # 运行回测
    print(f"测试股票: {len(stocks)}只")
    print(f"测试周期: {start_date} → {end_date}")
    print(f"初始资金: {initial_cash:,.0f}元\n")

    start_time = time.time()
    result = engine.run(
        stock_list=stocks,
        start_date=start_date,
        end_date=end_date
    )
    elapsed = time.time() - start_time

    print(f"\n✅ Hikyuu框架回测完成，用时: {elapsed:.2f}秒")
    print(f"收益率: {result.total_return:.2f}%")
    print(f"夏普比率: {result.sharpe_ratio:.2f}")
    print(f"最大回撤: {result.max_drawdown_pct:.2f}%")
    print(f"胜率: {result.win_rate:.2f}%")
    print(f"交易次数: {len(result.portfolio.trades)}")

    # 返回标准化结果
    return {
        'framework': 'Hikyuu-Style',
        'total_return': result.total_return,
        'annual_return': result.annualized_return,
        'sharpe_ratio': result.sharpe_ratio,
        'max_drawdown': result.max_drawdown_pct,
        'win_rate': result.win_rate,
        'total_trades': len(result.portfolio.trades),
        'final_capital': result.portfolio.get_total_value(),
        'backtest_time': elapsed,
        'result_object': result
    }


def run_extensible_backtest(version, stocks, start_date, end_date, initial_cash):
    """
    运行Extensible Backtest Engine

    参数:
        version: ML模型版本 ('V3.7', 'V3.8', 'V3.81')
        stocks: 股票列表 (不使用，由引擎自动获取股票池)
        start_date: 开始日期
        end_date: 结束日期
        initial_cash: 初始资金

    返回:
        回测结果字典
    """
    print("\n" + "=" * 80)
    print(f"📊 运行Extensible Backtest Engine ({version})")
    print("=" * 80)

    try:
        from extensible_backtest_engine import ExtensibleBacktestEngine

        # 创建回测引擎
        engine = ExtensibleBacktestEngine(
            initial_capital=initial_cash,
            max_workers=4,
            commission_rate=0.0003,  # 万三手续费
            stamp_tax=0.001,         # 千一印花税
            min_score_threshold=80.0  # 80分阈值
        )

        # 运行回测
        print(f"ML版本: {version}")
        print(f"测试周期: {start_date} → {end_date}")
        print(f"初始资金: {initial_cash:,.0f}元")
        print(f"评分阈值: 80.0\n")

        start_time = time.time()
        results = engine.run_backtest(
            versions=[version],
            start_date=start_date,
            end_date=end_date
        )
        elapsed = time.time() - start_time

        # 提取结果
        result = results['individual_results'].get(version, {})

        if 'error' in result:
            print(f"\n❌ Extensible框架回测失败: {result['error']}")
            return None

        print(f"\n✅ Extensible框架回测完成，用时: {elapsed:.2f}秒")
        print(f"收益率: {result.get('total_return', 0) * 100:.2f}%")
        print(f"夏普比率: {result.get('sharpe_ratio', 0):.2f}")
        print(f"最大回撤: {result.get('max_drawdown', 0) * 100:.2f}%")
        print(f"胜率: {result.get('win_rate', 0) * 100:.2f}%")
        print(f"交易次数: {result.get('total_trades', 0)}")

        # 返回标准化结果
        return {
            'framework': f'Extensible-{version}',
            'total_return': result.get('total_return', 0) * 100,  # 转换为百分比
            'annual_return': result.get('annual_return', 0) * 100,
            'sharpe_ratio': result.get('sharpe_ratio', 0),
            'max_drawdown': result.get('max_drawdown', 0) * 100,
            'win_rate': result.get('win_rate', 0) * 100,
            'total_trades': result.get('total_trades', 0),
            'final_capital': result.get('final_capital', initial_cash),
            'backtest_time': elapsed,
            'result_object': result
        }

    except Exception as e:
        print(f"\n❌ Extensible框架运行失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_results(results):
    """
    对比多个框架的结果

    参数:
        results: 结果列表 [{framework, total_return, ...}, ...]
    """
    # 过滤None结果
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        print("\n❌ 没有有效的回测结果可供对比")
        return

    print("\n" + "=" * 80)
    print("📊 框架对比分析")
    print("=" * 80)

    # 对比表格
    print(f"\n{'框架':<20} {'收益率':<12} {'年化收益':<12} {'夏普比率':<12} {'最大回撤':<12} {'胜率':<12} {'交易次数':<10} {'用时(秒)':<10}")
    print("-" * 120)

    for r in valid_results:
        print(f"{r['framework']:<20} "
              f"{r['total_return']:>10.2f}% "
              f"{r['annual_return']:>10.2f}% "
              f"{r['sharpe_ratio']:>10.2f} "
              f"{r['max_drawdown']:>10.2f}% "
              f"{r['win_rate']:>10.2f}% "
              f"{r['total_trades']:>10} "
              f"{r['backtest_time']:>10.2f}")

    # 最佳表现
    print("\n" + "=" * 80)
    print("🏆 最佳表现")
    print("=" * 80)

    best_return = max(valid_results, key=lambda x: x['total_return'])
    print(f"🥇 最高收益率: {best_return['framework']} ({best_return['total_return']:.2f}%)")

    best_sharpe = max(valid_results, key=lambda x: x['sharpe_ratio'])
    print(f"📈 最佳夏普比率: {best_sharpe['framework']} ({best_sharpe['sharpe_ratio']:.2f})")

    best_win_rate = max(valid_results, key=lambda x: x['win_rate'])
    print(f"🎯 最高胜率: {best_win_rate['framework']} ({best_win_rate['win_rate']:.2f}%)")

    fastest = min(valid_results, key=lambda x: x['backtest_time'])
    print(f"⚡ 最快速度: {fastest['framework']} ({fastest['backtest_time']:.2f}秒)")

    # 框架特点分析
    print("\n" + "=" * 80)
    print("📋 框架特点对比")
    print("=" * 80)

    print("\n✅ Hikyuu-Style框架:")
    print("  - 基于技术指标Signal (BBI)")
    print("  - 信号驱动的买卖判断")
    print("  - 每日检查交易机会")
    print("  - 灵活的MM和SL策略")
    print("  - 轻量级，速度快")

    print("\n✅ Extensible框架:")
    print("  - 基于ML模型评分 (V3.7/V3.8/V3.81)")
    print("  - 评分阈值筛选 (>=80分)")
    print("  - 定期调仓 (5天)")
    print("  - 完整的资金管理")
    print("  - 可扩展性强")

    # 差异分析
    print("\n" + "=" * 80)
    print("🔍 差异分析")
    print("=" * 80)

    hikyuu_result = next((r for r in valid_results if r['framework'] == 'Hikyuu-Style'), None)
    extensible_results = [r for r in valid_results if r['framework'].startswith('Extensible')]

    if hikyuu_result and extensible_results:
        for ext_result in extensible_results:
            return_diff = ext_result['total_return'] - hikyuu_result['total_return']
            sharpe_diff = ext_result['sharpe_ratio'] - hikyuu_result['sharpe_ratio']
            trade_diff = ext_result['total_trades'] - hikyuu_result['total_trades']

            print(f"\n{ext_result['framework']} vs Hikyuu-Style:")
            print(f"  收益率差异: {return_diff:+.2f}% ({'+优' if return_diff > 0 else '-劣'})")
            print(f"  夏普比率差异: {sharpe_diff:+.2f} ({'+优' if sharpe_diff > 0 else '-劣'})")
            print(f"  交易次数差异: {trade_diff:+d} ({'+多' if trade_diff > 0 else '-少'})")

            # 原因分析
            print(f"\n  差异原因:")
            if abs(return_diff) < 5:
                print("  - 两个框架表现接近")
            elif return_diff > 0:
                print(f"  - {ext_result['framework']}的ML评分系统更有效")
                print("  - 可能在选股精度上更优")
            else:
                print("  - Hikyuu的技术指标Signal更稳定")
                print("  - 可能在市场时机把握上更优")


def main():
    """主测试函数"""
    print("=" * 80)
    print("🚀 Hikyuu风格框架 vs Extensible框架对比测试 (Phase 4.4)")
    print("=" * 80)

    # 测试参数
    start_date = '2025-08-01'
    end_date = '2025-09-30'
    initial_cash = 100000  # 10万元

    # 创建数据适配器获取股票池
    db = DatabaseManager(db_path='data_adapter/stock_data.db')
    adapter = HikyuuStyleDataAdapter(db_manager=db)

    all_stocks = adapter.get_all_stocks('A股')
    test_stocks = all_stocks[:50]  # 使用前50只股票

    print(f"\n测试参数:")
    print(f"  股票池: {len(test_stocks)}只股票")
    print(f"  测试周期: {start_date} → {end_date}")
    print(f"  初始资金: {initial_cash:,.0f}元")

    # 收集所有结果
    all_results = []

    # 1. 运行Hikyuu框架
    hikyuu_result = run_hikyuu_backtest(test_stocks, start_date, end_date, initial_cash)
    if hikyuu_result:
        all_results.append(hikyuu_result)

    # 2. 运行Extensible框架 (V3.7)
    ext_v37_result = run_extensible_backtest('V3.7', test_stocks, start_date, end_date, initial_cash)
    if ext_v37_result:
        all_results.append(ext_v37_result)

    # 3. 运行Extensible框架 (V3.81)
    ext_v381_result = run_extensible_backtest('V3.81', test_stocks, start_date, end_date, initial_cash)
    if ext_v381_result:
        all_results.append(ext_v381_result)

    # 4. 对比分析
    compare_results(all_results)

    print("\n" + "=" * 80)
    print("🎉 框架对比测试完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
