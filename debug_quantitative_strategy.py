#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细调试量化策略执行
"""

import sys
sys.path.insert(0, '/Users/yangxu/StockTradebyZ')

import pandas as pd
import logging

# 设置详细日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from extensible_backtest_engine import ExtensibleBacktestEngine
from trading_strategy import BalancedStrategy

print("=" * 80)
print("🔍 详细调试量化策略执行")
print("=" * 80)

engine = ExtensibleBacktestEngine(
    strategy=BalancedStrategy(),
    initial_capital=1000000,
    max_workers=4,
    min_score_threshold=80.0
)

print("\n1. 加载数据缓存...")
engine._batch_load_stock_data('2025-07-01', '2025-07-10')

print("\n2. 测试量化策略过滤（5只股票）...")
candidates = ['000001', '000002', '000004', '000006', '000007']
print(f"   输入候选: {candidates}")

result = engine._run_quantitative_strategies(candidates, '2025-07-01')

print(f"\n3. 结果:")
print(f"   输出结果: {result}")
print(f"   过滤效果: {len(candidates)} -> {len(result)} 只")
print(f"   是否工作: {'✅ 是' if len(result) < len(candidates) else '❌ 否（fallback）'}")
