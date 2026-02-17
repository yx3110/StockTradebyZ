#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试量化策略预过滤功能
"""

import sys
sys.path.insert(0, '/Users/yangxu/StockTradebyZ')

import pandas as pd
from extensible_backtest_engine import ExtensibleBacktestEngine
from trading_strategy import BalancedStrategy

print("=" * 80)
print("🧪 测试量化策略预过滤功能")
print("=" * 80)

# 创建测试引擎
print("\n1️⃣ 创建回测引擎...")
try:
    engine = ExtensibleBacktestEngine(
        strategy=BalancedStrategy(),
        initial_capital=1000000,
        max_workers=4,
        min_score_threshold=80.0
    )
    print("✅ 引擎创建成功")
except Exception as e:
    print(f"❌ 引擎创建失败: {e}")
    sys.exit(1)

# 加载数据缓存
print("\n2️⃣ 加载数据缓存...")
try:
    engine._batch_load_stock_data('2025-07-01', '2025-07-10')
    print(f"✅ 数据缓存加载成功，共 {len(engine.data_cache)} 个缓存项")
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试基础筛选
print("\n3️⃣ 测试基础筛选...")
try:
    stock_universe = engine._get_stock_universe('2025-07-01', '2025-07-10')
    print(f"   股票池: {len(stock_universe)} 只")

    candidates = engine._basic_stock_screening(stock_universe, '2025-07-01')
    print(f"✅ 基础筛选成功: {len(candidates)} 只候选股票")
except Exception as e:
    print(f"❌ 基础筛选失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试量化策略过滤
print("\n4️⃣ 测试量化策略过滤...")
try:
    # 取前100只测试（避免太慢）
    test_candidates = candidates[:100]
    print(f"   使用 {len(test_candidates)} 只股票进行测试")

    quantitative_candidates = engine._run_quantitative_strategies(test_candidates, '2025-07-01')
    print(f"✅ 量化策略过滤成功: {len(test_candidates)} -> {len(quantitative_candidates)} 只")

    if quantitative_candidates:
        print(f"   示例股票: {quantitative_candidates[:5]}")
except Exception as e:
    print(f"❌ 量化策略过滤失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试完整选股流程
print("\n5️⃣ 测试完整选股流程...")
try:
    from extensible_backtest_engine import V37ModelAdapter, ModelConfig

    config = ModelConfig(
        version="V3.7",
        name="V3.70 Advanced ML System",
        module_name="v370_advanced_ml_system",
        class_name="V370AdvancedMLSystem",
        model_path_pattern="models/v370/v370_*.pkl",
        features_count=53,
        description="Test",
        requires_modules=[]
    )

    adapter = V37ModelAdapter(config, max_workers=4)

    # 使用前50只测试（避免太慢）
    test_universe = stock_universe[:50]
    selected = engine._universal_stock_selection(adapter, '2025-07-01', test_universe)

    print(f"✅ 完整选股流程成功: 选出 {len(selected)} 只股票")
    if selected:
        print(f"   最高分: {selected[0]['score']:.1f}")
        print(f"   示例: {selected[0]['stock_code']}")
except Exception as e:
    print(f"❌ 完整选股流程失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("🎉 所有测试通过！量化策略预过滤功能正常")
print("=" * 80)
