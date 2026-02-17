#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试优化后的调仓策略"""

import sys
sys.path.insert(0, '/Users/yangxu/StockTradebyZ')

import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 100)
print("🧪 调仓策略优化测试")
print("=" * 100)

# 导入回测引擎
from extensible_backtest_engine import ExtensibleBacktestEngine

# 测试参数 - 使用较短周期快速验证
TEST_PARAMS = {
    'start_date': '2025-09-01',  # 缩短到1个月
    'end_date': '2025-09-30',
    'initial_capital': 1000000,
    'min_score_threshold': 80.0,
    'max_workers': 6
}

print(f"\n📋 测试配置:")
print(f"  回测周期: {TEST_PARAMS['start_date']} → {TEST_PARAMS['end_date']} (1个月)")
print(f"  初始资金: {TEST_PARAMS['initial_capital']:,.0f}元")
print(f"  评分阈值: {TEST_PARAMS['min_score_threshold']}")

# 创建回测引擎
engine = ExtensibleBacktestEngine(
    initial_capital=TEST_PARAMS['initial_capital'],
    max_workers=TEST_PARAMS['max_workers'],
    min_score_threshold=TEST_PARAMS['min_score_threshold']
)

# 显示新增参数
print(f"\n🔧 新增风控参数:")
print(f"  止盈线: {engine.take_profit_pct * 100}%")
print(f"  止损线: {engine.stop_loss_pct * 100}%")
print(f"  最长持仓: {engine.max_holding_days}天")
print(f"  持仓最低评分: {engine.min_score_for_hold}")
print(f"  启用调仓卖出: {engine.enable_rebalance_sell}")

# 测试V3.7
print(f"\n" + "=" * 100)
print(f"🎯 开始测试 V3.7 (新策略)")
print(f"=" * 100)

import time
start_time = time.time()

try:
    results = engine.run_backtest(
        versions=['V3.7'],
        start_date=TEST_PARAMS['start_date'],
        end_date=TEST_PARAMS['end_date']
    )

    elapsed = time.time() - start_time

    print(f"\n" + "=" * 100)
    print(f"📈 测试结果")
    print(f"=" * 100)

    v37_result = results['individual_results'].get('V3.7', {})

    if 'error' in v37_result:
        print(f"❌ 测试失败: {v37_result['error']}")
    else:
        print(f"\n💰 资金情况:")
        print(f"   初始资金:   {TEST_PARAMS['initial_capital']:>15,.2f}元")
        print(f"   最终资产:   {v37_result.get('final_capital', 0):>15,.2f}元")
        print(f"   总收益:     {(v37_result.get('final_capital', 0) - TEST_PARAMS['initial_capital']):>15,.2f}元")

        print(f"\n📈 收益指标:")
        print(f"   总收益率:   {v37_result.get('total_return', 0)*100:>15.2f}%")
        print(f"   年化收益:   {v37_result.get('annual_return', 0)*100:>15.2f}%")

        print(f"\n📊 风险指标:")
        print(f"   夏普比率:   {v37_result.get('sharpe_ratio', 0):>15.2f}")
        print(f"   最大回撤:   {v37_result.get('max_drawdown', 0)*100:>15.2f}%")
        print(f"   胜率:       {v37_result.get('win_rate', 0)*100:>15.2f}% 🆕")

        print(f"\n🔄 交易统计:")
        print(f"   总交易次数: {v37_result.get('total_trades', 0):>15} (买入)")
        print(f"   成功交易:   {v37_result.get('successful_trades', 0):>15} (盈利卖出)")
        print(f"   失败交易:   {v37_result.get('failed_trades', 0):>15} (亏损卖出)")
        print(f"   平均评分:   {v37_result.get('avg_score', 0):>15.1f}")

        print(f"\n⚡ 性能:")
        print(f"   执行时间:   {elapsed:>15.1f}秒")

        # 关键对比
        print(f"\n" + "=" * 100)
        print(f"🎉 策略优化效果")
        print(f"=" * 100)

        win_rate = v37_result.get('win_rate', 0) * 100
        total_trades = v37_result.get('total_trades', 0)
        sell_trades = v37_result.get('successful_trades', 0) + v37_result.get('failed_trades', 0)

        print(f"\n📊 卖出交易分析:")
        print(f"   买入次数: {total_trades}")
        print(f"   卖出次数: {sell_trades}")
        print(f"   盈利卖出: {v37_result.get('successful_trades', 0)}")
        print(f"   亏损卖出: {v37_result.get('failed_trades', 0)}")
        print(f"   胜率: {win_rate:.2f}%")

        if win_rate > 0:
            print(f"\n✅ 成功! 新策略产生了盈利卖出交易")
            print(f"   原策略胜率: 0.00% (100%止损)")
            print(f"   新策略胜率: {win_rate:.2f}%")
            print(f"   改进: ↑ {win_rate:.2f} 个百分点")
        else:
            print(f"\n⚠️  注意: 胜率仍为0%")
            print(f"   可能原因:")
            print(f"     1. 测试周期太短 (仅1个月)")
            print(f"     2. 市场环境不利")
            print(f"     3. 需要调整参数 (止盈/止损线)")

except Exception as e:
    logger.error(f"测试失败: {e}")
    import traceback
    traceback.print_exc()

print(f"\n" + "=" * 100)
print(f"📝 下一步:")
print(f"=" * 100)
print(f"1. 如果测试成功，运行完整回测 (3个月)")
print(f"2. 对比 V3.7 vs V3.81")
print(f"3. 分析卖出原因分布 (止盈/止损/调仓/超期)")
print(f"4. 根据结果调优参数")
print(f"\n" + "=" * 100)
