#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略对比工具 - 对比不同交易策略的回测表现

使用方法:
    python3 compare_strategies.py --model V3.7 --start 2025-07-01 --end 2025-09-30

Author: Claude Code
Date: 2025-10-12
"""

import sys
sys.path.insert(0, '/Users/yangxu/StockTradebyZ')

import argparse
import time
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 100)
print("🏆 交易策略对比工具")
print("=" * 100)


def run_strategy_comparison(model_version: str, start_date: str, end_date: str,
                           initial_capital: float = 1000000,
                           min_score: float = 80.0):
    """
    对比多个策略在同一模型下的表现

    Args:
        model_version: ML模型版本 (V3.7, V3.81等)
        start_date: 开始日期
        end_date: 结束日期
        initial_capital: 初始资金
        min_score: 最低评分阈值
    """

    # 注意: 这里先打印计划，实际集成需要等回测引擎改造完成
    print(f"\n📋 对比计划:")
    print(f"  ML模型: {model_version}")
    print(f"  回测周期: {start_date} → {end_date}")
    print(f"  初始资金: {initial_capital:,.0f}元")
    print(f"  评分阈值: {min_score}")

    # 策略列表
    from trading_strategy import StrategyFactory

    strategies = {
        'conservative': '保守策略',
        'balanced': '平衡策略',
        'aggressive': '激进策略'
    }

    print(f"\n🎯 对比策略:")
    for key, name in strategies.items():
        info = StrategyFactory.get_strategy_info(key)
        print(f"  {name}:")
        print(f"    止盈: {info['config']['take_profit_pct']*100}%")
        print(f"    止损: {info['config']['stop_loss_pct']*100}%")
        print(f"    持仓天数: {info['config']['max_holding_days']}天")
        print(f"    调仓频率: {info['config']['rebalance_frequency']}天")

    print(f"\n⚠️  注意:")
    print(f"  当前版本仅展示对比计划")
    print(f"  实际运行需要等待回测引擎改造完成 (支持策略注入)")

    print(f"\n📝 下一步:")
    print(f"  1. 改造 ExtensibleBacktestEngine 支持策略注入")
    print(f"  2. 为每个策略运行回测")
    print(f"  3. 生成对比分析报告")

    # 模拟结果结构 (实际结果需要真正运行回测)
    print(f"\n📊 预期对比结果格式:")
    print(f"\n| 策略 | 总收益率 | 年化收益 | 夏普比率 | 最大回撤 | 胜率 | 交易次数 |")
    print(f"|------|----------|----------|----------|----------|------|----------|")
    print(f"| 保守 | TBD | TBD | TBD | TBD | TBD | TBD |")
    print(f"| 平衡 | TBD | TBD | TBD | TBD | TBD | TBD |")
    print(f"| 激进 | TBD | TBD | TBD | TBD | TBD | TBD |")

    return None


def analyze_strategy_results(results: dict) -> pd.DataFrame:
    """
    分析策略对比结果

    Args:
        results: 策略回测结果字典

    Returns:
        对比分析表格
    """
    comparison = []

    for strategy_name, result in results.items():
        if 'error' in result:
            continue

        comparison.append({
            '策略名称': strategy_name,
            '风险等级': result.get('strategy_info', {}).get('risk_level', 'N/A'),
            '总收益率': f"{result.get('total_return', 0)*100:.2f}%",
            '年化收益': f"{result.get('annual_return', 0)*100:.2f}%",
            '夏普比率': f"{result.get('sharpe_ratio', 0):.2f}",
            '最大回撤': f"{result.get('max_drawdown', 0)*100:.2f}%",
            '胜率': f"{result.get('win_rate', 0)*100:.2f}%",
            '交易次数': result.get('total_trades', 0),
            '平均评分': f"{result.get('avg_score', 0):.1f}",
            '执行时间': f"{result.get('backtest_time', 0):.1f}秒"
        })

    df = pd.DataFrame(comparison)

    # 按夏普比率排序
    if not df.empty:
        df['夏普_数值'] = df['夏普比率'].str.replace('%', '').astype(float)
        df = df.sort_values('夏普_数值', ascending=False)
        df = df.drop('夏普_数值', axis=1)

    return df


def generate_strategy_comparison_report(results: dict, output_file: str):
    """
    生成策略对比报告

    Args:
        results: 策略回测结果
        output_file: 输出文件路径
    """
    report = []

    report.append("# 交易策略对比报告\n")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    report.append("## 测试配置\n\n")
    report.append(f"- ML模型: {results.get('model_version', 'N/A')}\n")
    report.append(f"- 回测周期: {results.get('test_period', 'N/A')}\n")
    report.append(f"- 初始资金: {results.get('initial_capital', 0):,.0f}元\n\n")

    report.append("## 策略对比\n\n")

    # 生成对比表格
    df = analyze_strategy_results(results.get('strategy_results', {}))

    if not df.empty:
        report.append(df.to_markdown(index=False))
        report.append("\n\n")

    report.append("## 分析结论\n\n")

    # 找出最佳策略
    strategy_results = results.get('strategy_results', {})
    if strategy_results:
        best_return = max(strategy_results.items(), key=lambda x: x[1].get('total_return', 0))
        best_sharpe = max(strategy_results.items(), key=lambda x: x[1].get('sharpe_ratio', 0))

        report.append(f"- **最佳收益策略**: {best_return[0]} ({best_return[1].get('total_return', 0)*100:.2f}%)\n")
        report.append(f"- **最佳风险调整收益**: {best_sharpe[0]} (夏普{best_sharpe[1].get('sharpe_ratio', 0):.2f})\n")

    # 保存报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(report)

    print(f"📄 报告已保存: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='交易策略对比工具')

    parser.add_argument('--model', default='V3.7',
                       help='ML模型版本 (默认: V3.7)')
    parser.add_argument('--start', default='2025-07-01',
                       help='开始日期 (默认: 2025-07-01)')
    parser.add_argument('--end', default='2025-09-30',
                       help='结束日期 (默认: 2025-09-30)')
    parser.add_argument('--capital', type=float, default=1000000,
                       help='初始资金 (默认: 1000000)')
    parser.add_argument('--min-score', type=float, default=80.0,
                       help='最低评分阈值 (默认: 80.0)')

    args = parser.parse_args()

    # 运行对比
    results = run_strategy_comparison(
        model_version=args.model,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        min_score=args.min_score
    )

    # 如果有结果，生成报告
    if results:
        report_dir = Path('reports/strategy_comparison')
        report_dir.mkdir(parents=True, exist_ok=True)

        report_file = report_dir / f"strategy_comparison_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        generate_strategy_comparison_report(results, str(report_file))

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
