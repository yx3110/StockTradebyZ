#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版3策略对比回测 - 使用市场环境自适应策略

改进点:
1. ✅ 市场环境自适应参数调整
2. ✅ 延长持仓周期
3. ✅ 评分加权仓位管理
4. ✅ 减少频繁调仓

Author: Claude Code
Date: 2025-10-16
"""

import sys
sys.path.insert(0, '/Users/yangxu/StockTradebyZ')

import time
import json
from pathlib import Path
from datetime import datetime

from trading_strategy_improved import (
    AdaptiveConservativeStrategy,
    AdaptiveBalancedStrategy,
    AdaptiveAggressiveStrategy
)
from extensible_backtest_engine import ExtensibleBacktestEngine

print("=" * 100)
print("🚀 改进版3策略对比回测 (V3.81 + 市场环境自适应)")
print("=" * 100)

# 配置参数
INITIAL_CAPITAL = 1000000
START_DATE = '2025-07-01'
END_DATE = '2025-09-30'
ML_VERSION = 'V3.81'
MIN_SCORE = 80.0
MAX_WORKERS = 4

print(f"\n📋 回测配置:")
print(f"  ML模型: {ML_VERSION}")
print(f"  回测周期: {START_DATE} → {END_DATE}")
print(f"  初始资金: {INITIAL_CAPITAL:,.0f}元")
print(f"  最低评分: {MIN_SCORE}")
print(f"  并行进程: {MAX_WORKERS}")
print(f"\n✨ 改进特性:")
print(f"  ✅ 市场环境自适应 (牛市/熊市/震荡市)")
print(f"  ✅ 延长持仓周期 (40-60天 vs 原10-30天)")
print(f"  ✅ 评分加权仓位 (高分股票获得更大仓位)")
print(f"  ✅ 减少频繁调仓 (评分下降>12-15分才卖出)")

# 策略配置
strategies = {
    'conservative': {
        'instance': AdaptiveConservativeStrategy(),
        'name': '自适应保守策略',
        'emoji': '🛡️',
        'original_name': '保守策略'
    },
    'balanced': {
        'instance': AdaptiveBalancedStrategy(),
        'name': '自适应平衡策略',
        'emoji': '⚖️',
        'original_name': '平衡策略'
    },
    'aggressive': {
        'instance': AdaptiveAggressiveStrategy(),
        'name': '自适应激进策略',
        'emoji': '🚀',
        'original_name': '激进策略'
    }
}

print(f"\n🎯 测试策略 (初始参数，运行时将根据市场环境自适应调整):")
for key, config in strategies.items():
    strategy = config['instance']
    print(f"  {config['emoji']} {config['name']}:")
    print(f"     初始止盈/止损: {strategy.config.take_profit_pct*100}% / {strategy.config.stop_loss_pct*100}%")
    print(f"     最大持仓数: {strategy.config.max_positions}只")
    print(f"     初始持仓天数: {strategy.config.max_holding_days}天")
    print(f"     初始调仓频率: {strategy.config.rebalance_frequency}天")
    print(f"     评分加权: ✅ 启用")

# 运行回测
results = {}
all_results = []

for strategy_key, config in strategies.items():
    print(f"\n{'=' * 100}")
    print(f"{config['emoji']} 运行 {config['name']} 回测...")
    print("=" * 100)

    start_time = time.time()

    try:
        # 创建回测引擎
        engine = ExtensibleBacktestEngine(
            strategy=config['instance'],
            initial_capital=INITIAL_CAPITAL,
            max_workers=MAX_WORKERS,
            commission_rate=0.0003,
            stamp_tax=0.001,
            min_score_threshold=MIN_SCORE
        )

        # 运行回测
        result = engine.run_backtest(
            versions=[ML_VERSION],
            start_date=START_DATE,
            end_date=END_DATE
        )

        elapsed_time = time.time() - start_time

        # 提取结果
        individual_results = result.get('individual_results', {})
        if ML_VERSION in individual_results:
            ml_result = individual_results[ML_VERSION]

            # 添加策略信息
            strategy_result = {
                'strategy_key': strategy_key,
                'strategy_name': config['name'],
                'original_name': config['original_name'],
                'emoji': config['emoji'],
                'ml_version': ML_VERSION,
                'backtest_time': elapsed_time,
                'is_improved': True,
                **ml_result  # 包含所有回测结果
            }

            results[strategy_key] = strategy_result
            all_results.append(strategy_result)

            # 打印摘要
            print(f"\n✅ {config['name']} 回测完成 (耗时: {elapsed_time:.1f}秒)")
            print(f"   总收益率: {ml_result.get('total_return', 0)*100:.2f}%")
            print(f"   年化收益: {ml_result.get('annual_return', 0)*100:.2f}%")
            print(f"   夏普比率: {ml_result.get('sharpe_ratio', 0):.2f}")
            print(f"   最大回撤: {ml_result.get('max_drawdown', 0)*100:.2f}%")
            print(f"   胜率: {ml_result.get('win_rate', 0)*100:.2f}%")
            print(f"   交易次数: {ml_result.get('total_trades', 0)}")
        else:
            print(f"❌ {config['name']} 回测失败: 未找到{ML_VERSION}结果")

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ {config['name']} 回测失败 (耗时: {elapsed_time:.1f}秒)")
        print(f"   错误: {e}")
        import traceback
        traceback.print_exc()

        results[strategy_key] = {
            'strategy_key': strategy_key,
            'strategy_name': config['name'],
            'original_name': config['original_name'],
            'emoji': config['emoji'],
            'error': str(e),
            'backtest_time': elapsed_time,
            'is_improved': True
        }

# 生成对比报告
print(f"\n{'=' * 100}")
print("📊 改进版策略对比结果")
print("=" * 100)

# 创建对比表格
if all_results:
    print(f"\n| {'策略':<18} | {'总收益率':<10} | {'年化收益':<10} | {'夏普比率':<10} | {'最大回撤':<10} | {'胜率':<10} | {'交易次数':<10} |")
    print(f"|{'-'*20}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|")

    for result in all_results:
        if 'error' not in result:
            print(f"| {result['emoji']} {result['strategy_name']:<14} | "
                  f"{result.get('total_return', 0)*100:>9.2f}% | "
                  f"{result.get('annual_return', 0)*100:>9.2f}% | "
                  f"{result.get('sharpe_ratio', 0):>10.2f} | "
                  f"{result.get('max_drawdown', 0)*100:>9.2f}% | "
                  f"{result.get('win_rate', 0)*100:>9.2f}% | "
                  f"{result.get('total_trades', 0):>10} |")

    # 找出最佳策略
    print(f"\n🏆 最佳策略 (改进版):")
    best_return = max(all_results, key=lambda x: x.get('total_return', -999))
    best_sharpe = max(all_results, key=lambda x: x.get('sharpe_ratio', -999))
    best_winrate = max(all_results, key=lambda x: x.get('win_rate', 0))

    print(f"  🥇 最高收益: {best_return['emoji']} {best_return['strategy_name']} ({best_return.get('total_return', 0)*100:.2f}%)")
    print(f"  🥇 最佳夏普: {best_sharpe['emoji']} {best_sharpe['strategy_name']} (夏普{best_sharpe.get('sharpe_ratio', 0):.2f})")
    print(f"  🥇 最高胜率: {best_winrate['emoji']} {best_winrate['strategy_name']} ({best_winrate.get('win_rate', 0)*100:.2f}%)")

# 保存结果到文件
report_dir = Path('reports/strategy_comparison')
report_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
json_file = report_dir / f"strategy_comparison_{ML_VERSION.replace('.', '')}_improved_{timestamp}.json"
md_file = report_dir / f"strategy_comparison_{ML_VERSION.replace('.', '')}_improved_{timestamp}.md"

# 保存JSON
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump({
        'config': {
            'ml_version': ML_VERSION,
            'start_date': START_DATE,
            'end_date': END_DATE,
            'initial_capital': INITIAL_CAPITAL,
            'min_score': MIN_SCORE,
            'improvements': [
                '市场环境自适应',
                '延长持仓周期',
                '评分加权仓位',
                '减少频繁调仓'
            ]
        },
        'results': results
    }, f, ensure_ascii=False, indent=2)

print(f"\n💾 结果已保存:")
print(f"   JSON: {json_file}")

# 生成Markdown报告
md_content = []
md_content.append(f"# 改进版3策略对比回测报告\n\n")
md_content.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

md_content.append(f"## ✨ 改进特性\n\n")
md_content.append(f"1. **市场环境自适应**: 根据牛市/熊市/震荡市自动调整止盈止损参数\n")
md_content.append(f"2. **延长持仓周期**: 40-60天 (vs 原版10-30天)\n")
md_content.append(f"3. **评分加权仓位**: 高分股票(90+)获得1.2-1.5倍仓位\n")
md_content.append(f"4. **减少频繁调仓**: 评分下降>12-15分才卖出 (vs 原版低于阈值就卖出)\n\n")

md_content.append(f"## 测试配置\n\n")
md_content.append(f"- ML模型: {ML_VERSION}\n")
md_content.append(f"- 回测周期: {START_DATE} → {END_DATE}\n")
md_content.append(f"- 初始资金: {INITIAL_CAPITAL:,.0f}元\n")
md_content.append(f"- 最低评分: {MIN_SCORE}\n\n")

md_content.append(f"## 策略对比\n\n")

if all_results:
    md_content.append(f"| 策略 | 总收益率 | 年化收益 | 夏普比率 | 最大回撤 | 胜率 | 交易次数 |\n")
    md_content.append(f"|------|----------|----------|----------|----------|------|----------|\n")

    for result in all_results:
        if 'error' not in result:
            md_content.append(f"| {result['emoji']} {result['strategy_name']} | "
                            f"{result.get('total_return', 0)*100:.2f}% | "
                            f"{result.get('annual_return', 0)*100:.2f}% | "
                            f"{result.get('sharpe_ratio', 0):.2f} | "
                            f"{result.get('max_drawdown', 0)*100:.2f}% | "
                            f"{result.get('win_rate', 0)*100:.2f}% | "
                            f"{result.get('total_trades', 0)} |\n")

    md_content.append(f"\n## 最佳策略\n\n")
    md_content.append(f"- 🥇 **最高收益**: {best_return['emoji']} {best_return['strategy_name']} ({best_return.get('total_return', 0)*100:.2f}%)\n")
    md_content.append(f"- 🥇 **最佳夏普**: {best_sharpe['emoji']} {best_sharpe['strategy_name']} (夏普{best_sharpe.get('sharpe_ratio', 0):.2f})\n")
    md_content.append(f"- 🥇 **最高胜率**: {best_winrate['emoji']} {best_winrate['strategy_name']} ({best_winrate.get('win_rate', 0)*100:.2f}%)\n")

    # 与原版对比
    md_content.append(f"\n## 与原版对比\n\n")
    md_content.append(f"**原版V3.81结果** (2025-10-15回测):\n")
    md_content.append(f"- 保守策略: -0.78% (44.4%胜率, 46笔交易)\n")
    md_content.append(f"- 平衡策略: -3.80% (44.4%胜率, 46笔交易)\n")
    md_content.append(f"- 激进策略: -11.11% (52.8%胜率, 107笔交易)\n\n")

    md_content.append(f"**改进版结果**:\n")
    for result in all_results:
        if 'error' not in result:
            md_content.append(f"- {result['original_name']}: "
                            f"{result.get('total_return', 0)*100:.2f}% "
                            f"({result.get('win_rate', 0)*100:.1f}%胜率, "
                            f"{result.get('total_trades', 0)}笔交易)\n")

with open(md_file, 'w', encoding='utf-8') as f:
    f.writelines(md_content)

print(f"   Markdown: {md_file}")

print(f"\n{'=' * 100}")
print("🎉 改进版3策略对比回测完成!")
print(f"{'=' * 100}\n")

# 打印与原版的对比
print("📊 与原版V3.81对比:")
print("\n原版最佳: 保守策略 -0.78% (夏普0.05, 44.4%胜率, 46笔交易)")
if all_results:
    print(f"改进版最佳: {best_return['strategy_name']} {best_return.get('total_return', 0)*100:.2f}% "
          f"(夏普{best_return.get('sharpe_ratio', 0):.2f}, "
          f"{best_return.get('win_rate', 0)*100:.1f}%胜率, "
          f"{best_return.get('total_trades', 0)}笔交易)")

    improvement = best_return.get('total_return', 0) - (-0.0078)
    print(f"\n{'🎯 改进效果: ' if improvement > 0 else '⚠️ 改进效果: '}"
          f"{improvement*100:+.2f}个百分点")
