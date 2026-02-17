#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速检查回测结果"""

import json
import os
from pathlib import Path
from datetime import datetime

print("=" * 100)
print("📊 ML模型回测结果查看器")
print("=" * 100)

report_dir = Path('reports/backtest')

if not report_dir.exists():
    print("\n❌ 报告目录不存在")
    exit(1)

# 查找最新的回测报告
json_files = list(report_dir.glob('ml_versions_comparison_*.json'))

if not json_files:
    print("\n⏳ 回测仍在进行中，暂无完成的报告...")
    print("\n💡 提示: 回测需要15-20分钟，请稍后再检查")
    print(f"\n📁 监控目录: {report_dir}")
    exit(0)

# 获取最新文件
latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
file_time = datetime.fromtimestamp(latest_file.stat().st_mtime)

print(f"\n📄 最新报告: {latest_file.name}")
print(f"⏰ 生成时间: {file_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"⏱️  距今: {(datetime.now() - file_time).total_seconds() / 60:.1f}分钟")

# 读取并显示结果
try:
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("\n" + "=" * 100)
    print("📈 回测结果摘要")
    print("=" * 100)

    print(f"\n🎯 测试配置:")
    print(f"  测试期间: {data.get('test_period', 'N/A')}")
    print(f"  测试版本: {', '.join(data.get('versions_tested', []))}")

    results = data.get('individual_results', {})

    if not results:
        print("\n❌ 无可用结果")
        exit(0)

    print(f"\n" + "-" * 100)
    print(f"{'版本':<10} {'总收益率':<12} {'年化收益':<12} {'夏普比率':<12} {'最大回撤':<12} {'胜率':<10} {'交易次数':<10}")
    print("-" * 100)

    for version, result in sorted(results.items()):
        if 'error' not in result:
            print(f"{version:<10} "
                  f"{result.get('total_return', 0)*100:>10.2f}% "
                  f"{result.get('annual_return', 0)*100:>10.2f}% "
                  f"{result.get('sharpe_ratio', 0):>10.2f} "
                  f"{result.get('max_drawdown', 0)*100:>10.2f}% "
                  f"{result.get('win_rate', 0)*100:>8.2f}% "
                  f"{result.get('total_trades', 0):>10}")
        else:
            print(f"{version:<10} ❌ 失败: {result.get('error', 'Unknown')}")

    # 对比分析
    analysis = data.get('comparison_analysis', {})
    if 'best_performance' in analysis and analysis['best_performance'].get('version'):
        print(f"\n" + "=" * 100)
        print("🏆 对比分析")
        print("=" * 100)

        best = analysis['best_performance']
        print(f"\n🥇 最佳收益版本: {best['version']}")
        print(f"   收益率: {best['return']*100:.2f}%")

        if 'fastest_execution' in analysis and analysis['fastest_execution'].get('version'):
            fastest = analysis['fastest_execution']
            print(f"\n⚡ 最快执行版本: {fastest['version']}")
            print(f"   耗时: {fastest['time']:.1f}秒")

        print(f"\n📊 收益率排名:")
        for i, rank in enumerate(analysis.get('performance_ranking', []), 1):
            print(f"   {i}. {rank['version']}: {rank['return']*100:>8.2f}%")

        stats = analysis.get('statistics', {})
        if stats:
            print(f"\n📉 统计数据:")
            print(f"   平均收益率: {stats.get('avg_return', 0)*100:>8.2f}%")
            print(f"   平均交易数: {stats.get('avg_trades', 0):>8.0f}")
            print(f"   收益标准差: {stats.get('return_std', 0)*100:>8.2f}%")

    # 检查Markdown报告
    md_file = latest_file.with_suffix('.md')
    if md_file.exists():
        print(f"\n📄 详细Markdown报告: {md_file}")

    print("\n" + "=" * 100)
    print("✅ 结果查看完成")
    print("=" * 100)

except Exception as e:
    print(f"\n❌ 读取报告失败: {e}")
    import traceback
    traceback.print_exc()
