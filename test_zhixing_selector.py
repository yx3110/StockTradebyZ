#!/usr/bin/env python3
"""
使用完整的ZhiXingSelector测试选股
"""
import pandas as pd
from data_adapter.stock_data_loader import StockDataLoader
from stock_selctor.Selector import ZhiXingSelector

def test_zhixing_selector():
    print("加载股票数据...")
    loader = StockDataLoader()
    stock_data = loader.load_all_stock_data(
        days=250,
        security_types=['A股'],
        target_date='2025-10-10'
    )
    print(f"成功加载 {len(stock_data)} 只股票数据\n")

    # 创建知行选股器（使用当前参数）
    selector = ZhiXingSelector(
        j_threshold=13.0,
        min_change_pct=-2.0,
        max_change_pct=1.8,
        max_amplitude_pct=7.0,
        close_threshold_pct=97.0,
        max_window=120
    )

    # 运行选股
    target_date = pd.Timestamp('2025-10-10')
    print(f"运行知行战法选股（目标日期: {target_date}）...")

    selected_stocks = selector.select(target_date, stock_data)

    print(f"\n选股结果:")
    print(f"  共选出 {len(selected_stocks)} 只股票")

    if len(selected_stocks) > 0:
        print(f"\n选中的股票代码:")
        for i, code in enumerate(selected_stocks[:20], 1):  # 显示前20只
            print(f"  {i}. {code}")
        if len(selected_stocks) > 20:
            print(f"  ... 还有 {len(selected_stocks) - 20} 只股票")
    else:
        print("\n❌ 没有选中任何股票！")
        print("\n可能原因:")
        print("1. 当前市场条件不符合知行战法的所有条件")
        print("2. 参数过于严格")
        print("3. 数据加载方式与选股器预期不同")

if __name__ == '__main__':
    test_zhixing_selector()
