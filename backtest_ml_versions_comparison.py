#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML模型版本回测对比 - V3.0 vs V3.7 vs V3.81

对比三个版本的ML评分系统在相同市场环境下的表现
"""

import sys
sys.path.insert(0, '/Users/yangxu/StockTradebyZ')

import time
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 100)
print("🚀 ML模型版本回测对比测试")
print("=" * 100)

# 测试参数
TEST_PARAMS = {
    'start_date': '2025-07-01',
    'end_date': '2025-09-30',
    'initial_capital': 1000000,  # 100万初始资金
    'min_score_threshold': 80.0,
    'max_workers': 6
}

print(f"\n📋 测试配置:")
print(f"  回测周期: {TEST_PARAMS['start_date']} → {TEST_PARAMS['end_date']}")
print(f"  初始资金: {TEST_PARAMS['initial_capital']:,.0f}元")
print(f"  评分阈值: {TEST_PARAMS['min_score_threshold']}")
print(f"  并行进程: {TEST_PARAMS['max_workers']}")

# 导入回测引擎
from extensible_backtest_engine import ExtensibleBacktestEngine

# 创建回测引擎
engine = ExtensibleBacktestEngine(
    initial_capital=TEST_PARAMS['initial_capital'],
    max_workers=TEST_PARAMS['max_workers'],
    min_score_threshold=TEST_PARAMS['min_score_threshold']
)

# 列出可用模型
print(f"\n📊 检查可用模型:")
models = engine.list_all_models()
for model in models:
    print(f"  {model['version']}: {model['name']}")
    print(f"    状态: {model['status']}")
    print(f"    特征数: {model['features_count']}")

# 确定要测试的版本
test_versions = ['V3.7', 'V3.81']

# 检查V3.0是否可用
available_models = engine.get_available_models()
print(f"\n🔍 模型可用性检查:")
for version in ['V3.7', 'V3.80', 'V3.81']:
    status = "✅ 可用" if available_models.get(version, False) else "❌ 不可用"
    print(f"  {version}: {status}")

# V3.0需要特殊处理（使用tomorrow_stock_selector.py的V3.0评分系统）
print(f"\n⚠️  注意: V3.0是量化评分系统，不是ML系统")
print(f"    V3.0评分基于tomorrow_stock_selector.py的量化规则")
print(f"    本次对比将测试: {test_versions}")

print(f"\n" + "=" * 100)
print(f"🎯 开始回测对比")
print(f"=" * 100)

# 运行回测
start_time = time.time()

try:
    results = engine.run_backtest(
        versions=test_versions,
        start_date=TEST_PARAMS['start_date'],
        end_date=TEST_PARAMS['end_date']
    )

    elapsed = time.time() - start_time

    # 打印详细结果
    print(f"\n" + "=" * 100)
    print(f"📈 回测结果详情")
    print(f"=" * 100)

    for version, result in results['individual_results'].items():
        print(f"\n{'='*50}")
        print(f"📊 {version} 详细结果")
        print(f"{'='*50}")

        if 'error' in result:
            print(f"❌ 回测失败: {result['error']}")
            continue

        print(f"💰 资金情况:")
        print(f"   初始资金:   {TEST_PARAMS['initial_capital']:>15,.2f}元")
        print(f"   最终资产:   {result.get('final_capital', 0):>15,.2f}元")
        print(f"   总收益:     {(result.get('final_capital', 0) - TEST_PARAMS['initial_capital']):>15,.2f}元")

        print(f"\n📈 收益指标:")
        print(f"   总收益率:   {result.get('total_return', 0)*100:>15.2f}%")
        print(f"   年化收益:   {result.get('annual_return', 0)*100:>15.2f}%")

        print(f"\n📊 风险指标:")
        print(f"   夏普比率:   {result.get('sharpe_ratio', 0):>15.2f}")
        print(f"   最大回撤:   {result.get('max_drawdown', 0)*100:>15.2f}%")
        print(f"   胜率:       {result.get('win_rate', 0)*100:>15.2f}%")

        print(f"\n🔄 交易统计:")
        print(f"   总交易次数: {result.get('total_trades', 0):>15}")
        print(f"   成功交易:   {result.get('successful_trades', 0):>15}")
        print(f"   失败交易:   {result.get('failed_trades', 0):>15}")
        print(f"   平均评分:   {result.get('avg_score', 0):>15.1f}")

        print(f"\n⚡ 性能:")
        print(f"   执行时间:   {result.get('backtest_time', 0):>15.1f}秒")

    # 对比分析
    analysis = results.get('comparison_analysis', {})
    if 'best_performance' in analysis and analysis['best_performance']['version']:
        print(f"\n" + "=" * 100)
        print(f"🏆 版本对比分析")
        print(f"=" * 100)

        best = analysis['best_performance']
        print(f"\n🥇 最佳收益版本: {best['version']}")
        print(f"   收益率: {best['return']*100:.2f}%")

        fastest = analysis.get('fastest_execution', {})
        if fastest.get('version'):
            print(f"\n⚡ 最快执行版本: {fastest['version']}")
            print(f"   耗时: {fastest['time']:.1f}秒")

        print(f"\n📊 收益率排名:")
        for i, rank in enumerate(analysis.get('performance_ranking', []), 1):
            print(f"   {i}. {rank['version']}: {rank['return']*100:>8.2f}%")

        # 统计数据
        stats = analysis.get('statistics', {})
        if stats:
            print(f"\n📉 统计数据:")
            print(f"   平均收益率: {stats.get('avg_return', 0)*100:>8.2f}%")
            print(f"   平均交易数: {stats.get('avg_trades', 0):>8.0f}")
            print(f"   收益标准差: {stats.get('return_std', 0)*100:>8.2f}%")

    print(f"\n" + "=" * 100)
    print(f"🎉 回测完成!")
    print(f"=" * 100)
    print(f"总用时: {elapsed:.1f}秒")
    print(f"测试版本数: {len(test_versions)}")
    print(f"成功版本数: {len([v for v in results['individual_results'].values() if 'error' not in v])}")

    # 保存结果
    import json
    import os

    report_dir = 'reports/backtest'
    os.makedirs(report_dir, exist_ok=True)

    report_file = f"{report_dir}/ml_versions_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n📄 详细报告已保存: {report_file}")

    # 创建Markdown报告
    md_file = f"{report_dir}/ml_versions_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(f"# ML模型版本回测对比报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 测试配置\n\n")
        f.write(f"- 回测周期: {TEST_PARAMS['start_date']} → {TEST_PARAMS['end_date']}\n")
        f.write(f"- 初始资金: {TEST_PARAMS['initial_capital']:,.0f}元\n")
        f.write(f"- 评分阈值: {TEST_PARAMS['min_score_threshold']}\n\n")

        f.write(f"## 回测结果\n\n")
        f.write(f"| 版本 | 总收益率 | 年化收益 | 夏普比率 | 最大回撤 | 胜率 | 交易次数 | 平均评分 |\n")
        f.write(f"|------|----------|----------|----------|----------|------|----------|----------|\n")

        for version, result in results['individual_results'].items():
            if 'error' not in result:
                f.write(f"| {version} ")
                f.write(f"| {result.get('total_return', 0)*100:.2f}% ")
                f.write(f"| {result.get('annual_return', 0)*100:.2f}% ")
                f.write(f"| {result.get('sharpe_ratio', 0):.2f} ")
                f.write(f"| {result.get('max_drawdown', 0)*100:.2f}% ")
                f.write(f"| {result.get('win_rate', 0)*100:.2f}% ")
                f.write(f"| {result.get('total_trades', 0)} ")
                f.write(f"| {result.get('avg_score', 0):.1f} |\n")

        f.write(f"\n## 对比分析\n\n")
        if 'best_performance' in analysis:
            best = analysis['best_performance']
            f.write(f"🥇 **最佳收益版本**: {best['version']} ({best['return']*100:.2f}%)\n\n")

        f.write(f"## 结论\n\n")
        f.write(f"本次回测对比了{len(test_versions)}个ML模型版本，")
        f.write(f"共执行{elapsed:.1f}秒。\n")

    print(f"📄 Markdown报告已保存: {md_file}")

except Exception as e:
    logger.error(f"回测失败: {e}")
    import traceback
    traceback.print_exc()

print(f"\n" + "=" * 100)
