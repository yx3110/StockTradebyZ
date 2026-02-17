#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析回测交易记录"""

import json
from pathlib import Path

report_dir = Path('reports/backtest')
json_files = list(report_dir.glob('ml_versions_comparison_*.json'))

if not json_files:
    print("❌ 没有找到回测报告")
    exit(1)

latest_file = max(json_files, key=lambda f: f.stat().st_mtime)

with open(latest_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 100)
print("📊 回测交易结构分析")
print("=" * 100)

for version, result in data['individual_results'].items():
    if 'error' in result:
        continue

    print(f"\n{version}:")
    print(f"  总交易次数: {result.get('total_trades', 0)}")
    print(f"  成功交易: {result.get('successful_trades', 0)}")
    print(f"  失败交易: {result.get('failed_trades', 0)}")
    print(f"  当前胜率: {result.get('win_rate', 0)*100:.2f}%")

    # 分析：如果successful_trades + failed_trades远小于total_trades，
    # 说明大部分是买入交易（没有profit字段），只有少量卖出交易
    sell_trades_count = result.get('successful_trades', 0) + result.get('failed_trades', 0)
    buy_trades_count = result.get('total_trades', 0)  # 这个实际是买入次数

    print(f"\n  推断:")
    print(f"    买入交易数: {buy_trades_count}")
    print(f"    卖出交易数: {sell_trades_count} (有profit字段)")

    if sell_trades_count > 0:
        corrected_win_rate = result.get('successful_trades', 0) / sell_trades_count * 100
        print(f"    修正后胜率: {corrected_win_rate:.2f}%")
    else:
        print(f"    修正后胜率: 0.00% (无卖出交易)")

    # 分析问题
    if sell_trades_count < buy_trades_count * 0.1:
        print(f"\n  ⚠️  问题发现:")
        print(f"     卖出交易({sell_trades_count})远少于买入交易({buy_trades_count})")
        print(f"     可能原因：回测策略只在调仓时买入新股票，")
        print(f"             没有卖出旧持仓，导致大量买入但极少卖出。")
        print(f"             这是回测逻辑的设计问题，不是胜率计算问题。")

print("\n" + "=" * 100)
