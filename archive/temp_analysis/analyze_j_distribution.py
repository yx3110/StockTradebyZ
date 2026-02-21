#!/usr/bin/env python3
"""
分析当前市场J值分布，为知行战法和SuperB1战法提供合理参数建议
"""
import pandas as pd
import numpy as np
from data_adapter.stock_data_loader import StockDataLoader
from stock_selctor.Selector import compute_kdj

def analyze_j_distribution(stock_data, target_date='2025-10-10'):
    """分析J值的分布"""
    print("="*60)
    print("分析当前市场J值分布")
    print("="*60)

    target_date = pd.Timestamp(target_date)

    j_values = []
    for code, df in stock_data.items():
        if df.empty:
            continue

        # 过滤到目标日期
        df = df[df['date'] <= target_date]
        if len(df) < 10:
            continue

        # 使用compute_kdj计算KDJ指标
        df_with_kdj = compute_kdj(df)
        latest = df_with_kdj.iloc[-1]
        j_value = latest.get('J', None)

        if j_value is not None and not pd.isna(j_value):
            j_values.append(j_value)

    if len(j_values) == 0:
        print("没有找到有效的J值数据")
        return

    j_array = np.array(j_values)

    print(f"\nJ值统计（基于 {len(j_values)} 只股票）:")
    print(f"  最小值: {j_array.min():.2f}")
    print(f"  最大值: {j_array.max():.2f}")
    print(f"  平均值: {j_array.mean():.2f}")
    print(f"  中位数: {np.median(j_array):.2f}")
    print(f"  标准差: {j_array.std():.2f}")

    print(f"\nJ值分位数:")
    percentiles = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 95]
    for p in percentiles:
        value = np.percentile(j_array, p)
        count = (j_array <= value).sum()
        print(f"  {p:2d}%分位: {value:6.2f}  ({count}只股票, {count/len(j_values)*100:.1f}%)")

    print(f"\n按阈值统计:")
    thresholds = [5, 10, 13, 15, 20, 25, 30, 40, 50]
    for threshold in thresholds:
        count = (j_array < threshold).sum()
        print(f"  J < {threshold:2d}: {count:4d}只股票 ({count/len(j_values)*100:5.1f}%)")

    print("\n建议:")
    print(f"  知行战法当前参数: J < 13.0 (选中 {(j_array < 13).sum()}只, {(j_array < 13).sum()/len(j_values)*100:.1f}%)")
    print(f"  建议调整为: J < 25 (选中 {(j_array < 25).sum()}只, {(j_array < 25).sum()/len(j_values)*100:.1f}%)")
    print(f"              或: J < 30 (选中 {(j_array < 30).sum()}只, {(j_array < 30).sum()/len(j_values)*100:.1f}%)")

    print(f"\n  SuperB1战法当前参数: J < 10 (选中 {(j_array < 10).sum()}只, {(j_array < 10).sum()/len(j_values)*100:.1f}%)")
    print(f"  建议调整为: J < 20 (选中 {(j_array < 20).sum()}只, {(j_array < 20).sum()/len(j_values)*100:.1f}%)")
    print(f"              或: J < 25 (选中 {(j_array < 25).sum()}只, {(j_array < 25).sum()/len(j_values)*100:.1f}%)")


def main():
    print("加载股票数据...")
    loader = StockDataLoader()
    stock_data = loader.load_all_stock_data(
        days=250,
        security_types=['A股'],
        target_date='2025-10-10'
    )
    print(f"成功加载 {len(stock_data)} 只股票数据\n")

    analyze_j_distribution(stock_data, '2025-10-10')


if __name__ == '__main__':
    main()
