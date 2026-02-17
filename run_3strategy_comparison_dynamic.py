#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态策略3策略对比回测脚本

对比动态保守、动态平衡、动态激进三个策略

Author: Claude Code
Date: 2025-10-28
"""

import sys
import os
from datetime import datetime
from typing import Dict, Any
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extensible_backtest_engine import ExtensibleBacktestEngine
from strategy.trading_strategy_dynamic import (
    DynamicConservativeStrategy,
    DynamicBalancedStrategy,
    DynamicAggressiveStrategy
)
from data_adapter.database_manager import DatabaseManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_dynamic_strategy_comparison(
    start_date: str = '2025-07-01',
    end_date: str = '2025-09-30',
    initial_capital: float = 1000000
) -> Dict[str, Any]:
    """
    运行动态策略对比回测

    Args:
        start_date: 开始日期
        end_date: 结束日期
        initial_capital: 初始资金

    Returns:
        3个策略的回测结果
    """
    logger.info("=" * 80)
    logger.info("🚀 动态策略3策略对比回测")
    logger.info("=" * 80)

    # 创建数据库管理器
    db_manager = DatabaseManager()

    # 定义3个动态策略
    strategies = {
        'conservative': {
            'name': '动态保守策略',
            'emoji': '🛡️',
            'strategy': DynamicConservativeStrategy()
        },
        'balanced': {
            'name': '动态平衡策略',
            'emoji': '⚖️',
            'strategy': DynamicBalancedStrategy()
        },
        'aggressive': {
            'name': '动态激进策略',
            'emoji': '🚀',
            'strategy': DynamicAggressiveStrategy()
        }
    }

    results = {}

    # 对每个策略运行回测
    for key, config in strategies.items():
        logger.info(f"\n📊 运行 {config['emoji']} {config['name']}")
        logger.info(f"  止盈目标: {config['strategy'].config.take_profit_pct*100:.0f}%")
        logger.info(f"  止损阈值: {config['strategy'].config.stop_loss_pct*100:.0f}%")
        logger.info(f"  Trailing Stop: 基准{config['strategy'].trailing_stop.base_distance_pct*100:.0f}%, "
                   f"激活{config['strategy'].trailing_stop.activation_ratio*100:.0f}%目标")

        try:
            # 创建回测引擎
            engine = ExtensibleBacktestEngine(
                strategy=config['strategy'],
                initial_capital=initial_capital,
                max_workers=4,
                commission_rate=0.0003,
                stamp_tax=0.001,
                min_score_threshold=80.0
            )

            # 设置数据库管理器
            engine.db_manager = db_manager

            # 运行回测（使用V3.81模型）
            result = engine.run_backtest(
                versions=['V3.81'],
                start_date=start_date,
                end_date=end_date
            )

            # 提取结果
            v381_result = result['individual_results'].get('V3.81', {})

            results[key] = {
                'strategy_name': config['name'],
                'emoji': config['emoji'],
                'result': v381_result,
                'config': config['strategy'].get_info()
            }

            logger.info(f"  ✅ 完成: 收益率 {v381_result.get('total_return', 0):.2%}, "
                       f"交易 {v381_result.get('total_trades', 0)}次")

        except Exception as e:
            logger.error(f"  ❌ 失败: {e}")
            results[key] = {
                'strategy_name': config['name'],
                'emoji': config['emoji'],
                'result': {'error': str(e)},
                'config': config['strategy'].get_info()
            }

    return results


def generate_comparison_report(results: Dict[str, Any]) -> str:
    """
    生成对比报告

    Args:
        results: 回测结果

    Returns:
        Markdown格式报告
    """
    report_lines = []

    report_lines.append("# 动态策略3策略对比回测报告")
    report_lines.append("")
    report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # 特性说明
    report_lines.append("## ✨ 动态系统特性")
    report_lines.append("")
    report_lines.append("1. **Trailing Stop移动止损**: 股票创新高后，回撤超过动态距离（3%-12%）时卖出")
    report_lines.append("2. **动态保护阈值**: 根据盈利进度调整保护强度（30%→50%→70%→85%）")
    report_lines.append("3. **波动率自适应**: 高波动股票给予更大空间，低波动股票更早止盈")
    report_lines.append("4. **持仓参数固定**: 保留Bug修复机制，买入时参数不受市场环境变化影响")
    report_lines.append("")

    # 测试配置
    report_lines.append("## 测试配置")
    report_lines.append("")
    report_lines.append("- ML模型: V3.81")
    report_lines.append("- 回测周期: 2025-07-01 → 2025-09-30")
    report_lines.append("- 初始资金: 1,000,000元")
    report_lines.append("- 最低评分: 80.0")
    report_lines.append("")

    # 策略对比
    report_lines.append("## 策略对比")
    report_lines.append("")
    report_lines.append("| 策略 | 总收益率 | 年化收益 | 夏普比率 | 最大回撤 | 胜率 | 交易次数 |")
    report_lines.append("|------|----------|----------|----------|----------|------|----------|")

    for key in ['conservative', 'balanced', 'aggressive']:
        if key not in results:
            continue

        data = results[key]
        result = data['result']

        if 'error' in result:
            continue

        emoji = data['emoji']
        name = data['strategy_name']
        total_return = result.get('total_return', 0) * 100
        annual_return = result.get('annualized_return', 0) * 100
        sharpe = result.get('sharpe_ratio', 0)
        max_dd = result.get('max_drawdown', 0) * 100
        win_rate = result.get('win_rate', 0) * 100
        trades = result.get('total_trades', 0)

        report_lines.append(
            f"| {emoji} {name} | {total_return:.2f}% | {annual_return:.2f}% | "
            f"{sharpe:.2f} | {max_dd:.2f}% | {win_rate:.2f}% | {trades} |"
        )

    report_lines.append("")

    # 最佳策略
    report_lines.append("## 最佳策略")
    report_lines.append("")

    # 找出最高收益
    best_return_key = max(
        [k for k in results.keys() if 'error' not in results[k]['result']],
        key=lambda k: results[k]['result'].get('total_return', -999),
        default=None
    )

    # 找出最佳夏普
    best_sharpe_key = max(
        [k for k in results.keys() if 'error' not in results[k]['result']],
        key=lambda k: results[k]['result'].get('sharpe_ratio', -999),
        default=None
    )

    # 找出最高胜率
    best_wr_key = max(
        [k for k in results.keys() if 'error' not in results[k]['result']],
        key=lambda k: results[k]['result'].get('win_rate', -999),
        default=None
    )

    if best_return_key:
        data = results[best_return_key]
        report_lines.append(f"- 🥇 **最高收益**: {data['emoji']} {data['strategy_name']} "
                           f"({data['result'].get('total_return', 0)*100:.2f}%)")

    if best_sharpe_key:
        data = results[best_sharpe_key]
        report_lines.append(f"- 🥇 **最佳夏普**: {data['emoji']} {data['strategy_name']} "
                           f"(夏普{data['result'].get('sharpe_ratio', 0):.2f})")

    if best_wr_key:
        data = results[best_wr_key]
        report_lines.append(f"- 🥇 **最高胜率**: {data['emoji']} {data['strategy_name']} "
                           f"({data['result'].get('win_rate', 0)*100:.2f}%)")

    report_lines.append("")

    # 与静态系统对比
    report_lines.append("## 与静态系统对比")
    report_lines.append("")
    report_lines.append("**静态系统结果** (Bug修复后):")
    report_lines.append("- 保守策略: -0.90% (40.0%胜率, 31笔交易)")
    report_lines.append("- 平衡策略: -3.41% (47.4%胜率, 40笔交易)")
    report_lines.append("- 激进策略: -4.82% (46.4%胜率, 59笔交易)")
    report_lines.append("")
    report_lines.append("**动态系统结果**:")

    for key in ['conservative', 'balanced', 'aggressive']:
        if key not in results or 'error' in results[key]['result']:
            continue

        data = results[key]
        result = data['result']
        name = data['strategy_name']
        total_return = result.get('total_return', 0) * 100
        win_rate = result.get('win_rate', 0) * 100
        trades = result.get('total_trades', 0)

        report_lines.append(f"- {name}: {total_return:.2f}% ({win_rate:.1f}%胜率, {trades}笔交易)")

    report_lines.append("")

    return "\n".join(report_lines)


def save_report(report: str, filename: str):
    """保存报告到文件"""
    filepath = f"reports/strategy_comparison/{filename}"

    os.makedirs("reports/strategy_comparison", exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)

    logger.info(f"📝 报告已保存: {filepath}")


def main():
    """主函数"""
    logger.info("🚀 开始动态策略对比回测")

    # 运行回测
    results = run_dynamic_strategy_comparison(
        start_date='2025-07-01',
        end_date='2025-09-30',
        initial_capital=1000000
    )

    # 生成报告
    report = generate_comparison_report(results)

    # 保存报告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_report(report, f"dynamic_strategy_comparison_{timestamp}.md")

    # 输出到控制台
    print("\n" + "=" * 80)
    print(report)
    print("=" * 80)

    logger.info("✅ 动态策略对比回测完成")


if __name__ == "__main__":
    main()
