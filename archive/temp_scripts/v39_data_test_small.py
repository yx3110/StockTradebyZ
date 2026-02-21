#!/usr/bin/env python3
"""
v3.9数据抓取小范围测试

测试前100只股票，1个月数据
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v39_data_initializer import V39DataInitializer
import sqlite3

def main():
    # 获取前100只股票
    conn = sqlite3.connect('data_adapter/stock_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT code FROM securities WHERE type = 'A股' LIMIT 100")
    stock_list = [row[0] for row in cursor.fetchall()]
    conn.close()

    print(f"✅ 测试股票数: {len(stock_list)}")
    print(f"   示例: {stock_list[:5]}")

    # 创建初始化器
    initializer = V39DataInitializer()

    # 测试Step 2: daily_basic (最近1个月)
    print("\n测试Step 2: daily_basic")
    initializer.step2_fetch_daily_basic(stock_list, '20251001', '20251031')

    # 测试Step 3: financial_indicator (最近1个季度)
    print("\n测试Step 3: financial_indicator")
    initializer.step3_fetch_financial_indicator(stock_list, '20240701', '20251031')

    # 打印统计
    initializer.print_summary()

if __name__ == "__main__":
    main()
