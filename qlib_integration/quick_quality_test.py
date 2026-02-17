#!/usr/bin/env python3
"""快速质量测试"""

import sqlite3
import pandas as pd
from pathlib import Path

def quick_test():
    db_path = 'data_adapter/stock_data.db'
    conn = sqlite3.connect(db_path)
    
    # 使用已知有数据的股票
    test_code = '000002.SZ'
    
    # 获取stock info
    stock_info_query = "SELECT security_id, fullname FROM stock_basic_info WHERE ts_code = ?"
    stock_info = pd.read_sql_query(stock_info_query, conn, params=[test_code])
    print(f"Stock info: {stock_info}")
    
    if not stock_info.empty:
        security_id = stock_info.iloc[0]['security_id']
        
        # 测试各种日期查询方式
        queries = [
            ("直接比较", f"SELECT COUNT(*) as count FROM daily_quotes WHERE security_id = {security_id} AND trade_date >= '2024-01-01' AND trade_date <= '2024-12-31'"),
            ("LIKE查询", f"SELECT COUNT(*) as count FROM daily_quotes WHERE security_id = {security_id} AND trade_date LIKE '2024%'"),
            ("BETWEEN", f"SELECT COUNT(*) as count FROM daily_quotes WHERE security_id = {security_id} AND trade_date BETWEEN '2024-01-01' AND '2024-12-31'"),
        ]
        
        for name, query in queries:
            result = pd.read_sql_query(query, conn)
            print(f"{name}: {result.iloc[0]['count']} 条记录")
        
        # 获取实际数据样本
        sample_query = f"""
        SELECT trade_date, close, volume, price_change_pct, is_suspend
        FROM daily_quotes
        WHERE security_id = {security_id} AND trade_date LIKE '2024%'
        ORDER BY trade_date
        LIMIT 10
        """
        
        df_sample = pd.read_sql_query(sample_query, conn)
        print(f"\n2024年数据样本 ({len(df_sample)} 条记录):")
        print(df_sample)
        
        # 测试质量评估
        if len(df_sample) > 0:
            trading_days = len(df_sample)
            missing_ratio = df_sample.isna().sum().sum() / (len(df_sample) * len(df_sample.columns))
            zero_volume_ratio = (df_sample['volume'] == 0).sum() / len(df_sample) if 'volume' in df_sample.columns else 0
            avg_volume = df_sample['volume'].mean() if 'volume' in df_sample.columns else 0
            
            print(f"\n质量指标:")
            print(f"交易天数: {trading_days}")
            print(f"缺失比例: {missing_ratio:.4f}")
            print(f"零成交量比例: {zero_volume_ratio:.4f}")
            print(f"平均成交量: {avg_volume:,.0f}")
            
            # 简单质量评分
            quality_score = 100.0
            issues = []
            
            if trading_days < 200:
                issues.append('insufficient_trading_days')
                quality_score -= 20
                
            if missing_ratio > 0.05:
                issues.append('excessive_missing_data')
                quality_score -= 15
                
            if zero_volume_ratio > 0.01:
                issues.append('frequent_zero_volume')
                quality_score -= 10
                
            if avg_volume < 10000:
                issues.append('low_liquidity')
                quality_score -= 15
            
            quality_score = max(0.0, quality_score)
            
            print(f"\n质量评分: {quality_score}")
            print(f"问题: {issues if issues else '无'}")
            
            if quality_score >= 70:
                print("✅ 推荐使用")
            else:
                print("❌ 不推荐使用")
    
    conn.close()

if __name__ == "__main__":
    quick_test()