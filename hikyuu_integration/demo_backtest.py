#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hikyuu风格回测框架演示

演示如何使用HikyuuStyleBacktestEngine进行快速回测
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hikyuu_integration import (
    HikyuuStyleDataAdapter, Query,
    HikyuuStyleBacktestEngine,
    BBISignal, KDJSignal, CompositeSignal,
    MLScoringSignal, MLCombinedSignal,
    MM_FixedPercent, MM_FixedRisk,
    ST_FixedPercent, ST_ProfitGoal, ST_Trailing, ST_Composite
)


def demo_1_basic_backtest():
    """
    演示1: 基础回测

    使用BBI信号，固定比例资金管理，无止损
    """
    print("\n" + "=" * 80)
    print("📊 演示1: 基础回测 (BBI信号)")
    print("=" * 80)

    # 1. 创建数据适配器
    adapter = HikyuuStyleDataAdapter()

    # 2. 创建回测引擎
    engine = HikyuuStyleBacktestEngine(
        data_adapter=adapter,
        signal=BBISignal(),                    # BBI信号
        money_manager=MM_FixedPercent(0.2),   # 每次20%资金
        initial_cash=100000,                   # 10万初始资金
        max_positions=5,                       # 最多5个持仓
        enable_t1=True,                        # 启用T+1
        enable_limit_check=True                # 启用涨跌停检查
    )

    # 3. 运行回测
    result = engine.run(
        stock_list=['000001', '000002', '000651', '002594', '600519'],
        start_date='2025-07-01',
        end_date='2025-09-30'
    )

    # 4. 查看结果
    result.print_summary()


def demo_2_stop_loss_backtest():
    """
    演示2: 带止损回测

    使用KDJ信号，固定风险管理，组合止损策略
    """
    print("\n" + "=" * 80)
    print("📊 演示2: 带止损回测 (KDJ信号 + 组合止损)")
    print("=" * 80)

    adapter = HikyuuStyleDataAdapter()

    # 创建组合止损策略
    stop_loss = ST_Composite([
        ST_FixedPercent(0.08),    # 固定止损8%
        ST_ProfitGoal(0.20),      # 止盈20%
        ST_Trailing(0.05)         # 追踪止损5%
    ])

    engine = HikyuuStyleBacktestEngine(
        data_adapter=adapter,
        signal=KDJSignal(oversold=20, overbought=80),
        money_manager=MM_FixedRisk(risk_pct=0.02, stop_loss_pct=0.08),
        stop_loss=stop_loss,
        initial_cash=100000,
        max_positions=10,
        enable_t1=True,
        enable_limit_check=True
    )

    result = engine.run(
        stock_list=['000001', '000002', '000651', '002594', '600519', '600036'],
        start_date='2025-06-01',
        end_date='2025-09-30'
    )

    result.print_summary()

    # 导出交易记录
    trades_df = result.get_trades_df()
    if len(trades_df) > 0:
        print("\n交易记录预览:")
        print(trades_df.head(10))


def demo_3_ml_backtest():
    """
    演示3: ML评分回测

    使用v3.81 ML信号，固定比例资金管理，止盈策略
    """
    print("\n" + "=" * 80)
    print("📊 演示3: ML评分回测 (v3.81 ML信号)")
    print("=" * 80)

    adapter = HikyuuStyleDataAdapter()

    engine = HikyuuStyleBacktestEngine(
        data_adapter=adapter,
        signal=MLScoringSignal(ml_version='v3.81', min_score=80),
        money_manager=MM_FixedPercent(0.1),
        stop_loss=ST_ProfitGoal(0.25),  # 25%止盈
        initial_cash=100000,
        max_positions=8,
        enable_t1=True,
        enable_limit_check=True
    )

    # 使用更多股票测试
    stock_list = [
        '000001', '000002', '000651', '002594', '600519', '600036',
        '000858', '002415', '300059', '601318'
    ]

    result = engine.run(
        stock_list=stock_list,
        start_date='2025-05-01',
        end_date='2025-09-30'
    )

    result.print_summary()


def demo_4_combined_signal_backtest():
    """
    演示4: 组合信号回测

    使用ML+技术指标组合信号
    """
    print("\n" + "=" * 80)
    print("📊 演示4: 组合信号回测 (ML + BBI + KDJ)")
    print("=" * 80)

    adapter = HikyuuStyleDataAdapter()

    # 创建组合信号
    signal = MLCombinedSignal(
        ml_version='v3.81',
        min_score=75,
        use_bbi=True,
        use_kdj=True
    )

    engine = HikyuuStyleBacktestEngine(
        data_adapter=adapter,
        signal=signal,
        money_manager=MM_FixedPercent(0.15),
        stop_loss=ST_FixedPercent(0.08),
        initial_cash=100000,
        max_positions=6,
        enable_t1=True,
        enable_limit_check=True
    )

    result = engine.run(
        stock_list=['000001', '000002', '000651', '002594', '600519'],
        start_date='2025-07-01',
        end_date='2025-09-30'
    )

    result.print_summary()


def demo_5_multi_signal_comparison():
    """
    演示5: 多信号对比

    对比BBI、KDJ、组合信号的回测效果
    """
    print("\n" + "=" * 80)
    print("📊 演示5: 多信号对比")
    print("=" * 80)

    adapter = HikyuuStyleDataAdapter()

    # 测试配置
    stock_list = ['000001', '000002', '000651', '002594']
    start_date = '2025-07-01'
    end_date = '2025-09-30'
    initial_cash = 100000

    # 信号列表
    signals = [
        ('BBI信号', BBISignal()),
        ('KDJ信号', KDJSignal()),
        ('BBI+KDJ组合(AND)', CompositeSignal([BBISignal(), KDJSignal()], mode='AND')),
        ('BBI+KDJ组合(OR)', CompositeSignal([BBISignal(), KDJSignal()], mode='OR')),
    ]

    results = []

    for name, signal in signals:
        print(f"\n测试 {name}...")

        engine = HikyuuStyleBacktestEngine(
            data_adapter=adapter,
            signal=signal,
            money_manager=MM_FixedPercent(0.2),
            initial_cash=initial_cash,
            max_positions=4,
            enable_t1=False,  # 关闭T+1加快测试
            enable_limit_check=False
        )

        result = engine.run(stock_list, start_date, end_date)

        results.append({
            'signal': name,
            'return': result.total_return_pct,
            'annualized': result.annualized_return,
            'max_dd': result.max_drawdown_pct,
            'sharpe': result.sharpe_ratio,
            'trades': result.trade_count
        })

    # 打印对比表
    print("\n" + "=" * 80)
    print("对比结果:")
    print("=" * 80)
    print(f"{'信号':<20} {'收益率':<10} {'年化':<10} {'最大回撤':<10} {'夏普':<10} {'交易次数':<10}")
    print("-" * 80)

    for r in results:
        print(f"{r['signal']:<20} {r['return']:>8.2f}% {r['annualized']:>8.2f}% "
              f"{r['max_dd']:>8.2f}% {r['sharpe']:>8.2f} {r['trades']:>8}")

    print("=" * 80)


def demo_6_custom_callback():
    """
    演示6: 自定义回调函数

    演示如何使用on_bar回调函数实现自定义逻辑
    """
    print("\n" + "=" * 80)
    print("📊 演示6: 自定义回调函数")
    print("=" * 80)

    adapter = HikyuuStyleDataAdapter()

    # 自定义回调函数
    def on_bar_callback(date, portfolio, broker):
        """
        每日回调函数

        可以在这里实现自定义逻辑，比如：
        - 动态调整持仓
        - 打印持仓信息
        - 自定义风险控制
        """
        # 每周打印一次持仓
        if portfolio.get_position_count() > 0:
            total_value = portfolio.get_total_value()
            pnl_pct = portfolio.get_total_pnl_pct()
            print(f"{date}: 持仓{portfolio.get_position_count()}个, "
                  f"总资产={total_value:,.0f}, 盈亏={pnl_pct:.2f}%")

    engine = HikyuuStyleBacktestEngine(
        data_adapter=adapter,
        signal=BBISignal(),
        money_manager=MM_FixedPercent(0.2),
        initial_cash=100000,
        max_positions=3,
        enable_t1=False,
        enable_limit_check=False
    )

    result = engine.run(
        stock_list=['000001', '000002', '000651'],
        start_date='2025-08-01',
        end_date='2025-09-30',
        on_bar=on_bar_callback  # 传入回调函数
    )

    result.print_summary()


def main():
    """运行所有演示"""
    print("\n🚀 Hikyuu风格回测框架演示")
    print("=" * 80)

    # 演示1: 基础回测
    demo_1_basic_backtest()

    # 演示2: 带止损回测
    demo_2_stop_loss_backtest()

    # 演示3: ML评分回测
    demo_3_ml_backtest()

    # 演示4: 组合信号回测
    demo_4_combined_signal_backtest()

    # 演示5: 多信号对比
    demo_5_multi_signal_comparison()

    # 演示6: 自定义回调
    demo_6_custom_callback()

    print("\n" + "=" * 80)
    print("✅ 所有演示完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
