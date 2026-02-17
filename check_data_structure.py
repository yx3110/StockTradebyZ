#!/usr/bin/env python3
"""
检查股票数据的字段结构
"""
from data_adapter.stock_data_loader import StockDataLoader

loader = StockDataLoader()
stock_data = loader.load_all_stock_data(
    days=50,
    security_types=['A股'],
    target_date='2025-10-10'
)

# 获取第一只股票的数据
first_code = list(stock_data.keys())[0]
first_df = stock_data[first_code]

print(f"股票代码: {first_code}")
print(f"数据形状: {first_df.shape}")
print(f"\n列名:")
print(first_df.columns.tolist())
print(f"\n最新一行数据:")
print(first_df.iloc[-1])
print(f"\n数据类型:")
print(first_df.dtypes)
