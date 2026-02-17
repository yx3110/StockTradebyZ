#!/usr/bin/env python3
"""
测试Tushare API返回的原始数据格式
"""

import pandas as pd
import tushare as ts
import json
import os
import sys

# 添加项目根目录到path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 读取配置
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)
    ts_token = config['tushare']['token']

ts.set_token(ts_token)
pro = ts.pro_api()

def test_tushare_original_data(date_str='20250822'):
    """测试Tushare API返回的原始数据格式"""
    print(f"测试日期: {date_str}")
    print("="*60)
    
    try:
        # 获取小样本A股数据
        df = pro.daily(
            trade_date=date_str,
            fields='ts_code,trade_date,open,close,high,low,pct_chg,limit'
        )
        
        if df.empty:
            print(f"没有找到 {date_str} 的数据")
            return
        
        print(f"获取到 {len(df)} 只股票的数据")
        
        # 分析价格变动分布
        pct_changes = df['pct_chg'].dropna()
        
        print(f"\n价格变动统计:")
        print(f"  总数量: {len(pct_changes)}")
        print(f"  最大涨幅: {pct_changes.max():.2f}%")
        print(f"  最大跌幅: {pct_changes.min():.2f}%")
        print(f"  平均变动: {pct_changes.mean():.2f}%")
        
        # 统计不同涨跌幅区间的股票数量
        ranges = [
            (10, float('inf'), '涨停及以上'),
            (5, 10, '5%-10%'),
            (3, 5, '3%-5%'),
            (1, 3, '1%-3%'),
            (0, 1, '0%-1%'),
            (-1, 0, '-1%-0%'),
            (-3, -1, '-3%--1%'),
            (-5, -3, '-5%--3%'),
            (-10, -5, '-10%--5%'),
            (float('-inf'), -10, '跌停及以下')
        ]
        
        print(f"\n涨跌幅分布:")
        for min_pct, max_pct, label in ranges:
            if max_pct == float('inf'):
                count = len(pct_changes[pct_changes >= min_pct])
            elif min_pct == float('-inf'):
                count = len(pct_changes[pct_changes <= max_pct])
            else:
                count = len(pct_changes[(pct_changes >= min_pct) & (pct_changes < max_pct)])
            print(f"  {label}: {count} 只")
        
        # 显示涨跌停股票
        limit_ups = df[df['limit'] == 'U']
        limit_downs = df[df['limit'] == 'D']
        
        print(f"\n涨跌停统计:")
        print(f"  涨停股票: {len(limit_ups)} 只")
        print(f"  跌停股票: {len(limit_downs)} 只")
        
        # 显示前10只涨幅最大的股票
        top_gainers = df.nlargest(10, 'pct_chg')[['ts_code', 'pct_chg', 'limit']]
        print(f"\n涨幅前10:")
        for _, row in top_gainers.iterrows():
            limit_flag = f" ({row['limit']})" if pd.notna(row['limit']) else ""
            print(f"  {row['ts_code']}: {row['pct_chg']:+.2f}%{limit_flag}")
        
        # 显示前10只跌幅最大的股票  
        top_losers = df.nsmallest(10, 'pct_chg')[['ts_code', 'pct_chg', 'limit']]
        print(f"\n跌幅前10:")
        for _, row in top_losers.iterrows():
            limit_flag = f" ({row['limit']})" if pd.notna(row['limit']) else ""
            print(f"  {row['ts_code']}: {row['pct_chg']:+.2f}%{limit_flag}")
            
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == "__main__":
    test_tushare_original_data('20250822')