#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug缓存数据格式
"""

import sys
sys.path.insert(0, '/Users/yangxu/StockTradebyZ')

from ml_models.v37.backtest_v37_engine_optimized import V37BacktestEngineOptimized

print("=" * 80)
print("🔍 检查缓存数据格式")
print("=" * 80)

# 加载数据
engine = V37BacktestEngineOptimized()
data_cache = engine.batch_load_stock_data('2025-07-01', '2025-07-10')

print(f"\n缓存项数量: {len(data_cache)}")

# 检查第一个股票的数据格式
for stock_code, df in list(data_cache.items())[:1]:
    if not df.empty:

        print(f"\n示例股票: {stock_code}")
        print(f"数据类型: {type(df)}")
        print(f"形状: {df.shape}")
        print(f"索引类型: {type(df.index)}")
        print(f"索引名: {df.index.name}")
        print(f"列名: {list(df.columns)}")
        print(f"\n前5行:")
        print(df.head())

        # 检查是否有技术指标
        tech_cols = ['kdj_j', 'kdj_k', 'kdj_d', 'bbi', 'macd_dif']
        for col in tech_cols:
            if col in df.columns:
                print(f"✅ {col}: 有")
            else:
                print(f"❌ {col}: 无")

        break
