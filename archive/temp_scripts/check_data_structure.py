#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_adapter.stock_data_loader import StockDataLoader

data_loader = StockDataLoader()
stock_data = data_loader.load_all_stock_data(days=10)

# 找一只股票检查数据结构
sample_code = list(stock_data.keys())[0]
sample_data = stock_data[sample_code]

print(f"样本股票: {sample_code}")
print(f"数据列: {sample_data.columns.tolist()}")
print("样本数据:")
print(sample_data.head())