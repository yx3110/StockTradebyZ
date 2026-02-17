#!/usr/bin/env python3
"""
V3.60数据调试脚本
"""
import os
import sys
import pandas as pd
sys.path.append('/Users/yangxu/StockTradebyZ')

from data_adapter.database_manager import DatabaseManager

def debug_data():
    db_manager = DatabaseManager("data_adapter/stock_data.db")
    
    print("🔍 数据库调试信息")
    print("="*50)
    
    # 1. 检查股票基本信息
    with db_manager.get_connection() as conn:
        securities = pd.read_sql_query("SELECT * FROM securities WHERE code IN ('000001', '000002') LIMIT 10", conn)
        print(f"📊 股票信息:")
        print(securities[['id', 'code', 'name']].head())
        print()
        
        # 2. 检查最新的日线数据
        latest_quotes = pd.read_sql_query("""
            SELECT s.code, dq.trade_date, dq.close, dq.volume 
            FROM daily_quotes dq 
            JOIN securities s ON dq.security_id = s.id 
            WHERE s.code IN ('000001', '000002') 
            ORDER BY dq.trade_date DESC 
            LIMIT 10
        """, conn)
        print(f"📈 最新日线数据:")
        print(latest_quotes)
        print()
        
        # 3. 检查2025年数据
        data_2025 = pd.read_sql_query("""
            SELECT s.code, COUNT(*) as count, MIN(dq.trade_date) as min_date, MAX(dq.trade_date) as max_date
            FROM daily_quotes dq 
            JOIN securities s ON dq.security_id = s.id 
            WHERE s.code IN ('000001', '000002') AND dq.trade_date >= '2025-01-01'
            GROUP BY s.code
        """, conn)
        print(f"📅 2025年数据统计:")
        print(data_2025)
        print()
        
        # 4. 检查技术指标数据
        tech_data = pd.read_sql_query("""
            SELECT s.code, ti.trade_date, ti.bbi, ti.rsi12, ti.kdj_k
            FROM technical_indicators ti
            JOIN securities s ON ti.security_id = s.id 
            WHERE s.code IN ('000001', '000002') AND ti.trade_date >= '2025-08-01'
            ORDER BY ti.trade_date DESC
            LIMIT 5
        """, conn)
        print(f"🔧 技术指标数据:")
        print(tech_data)
        print()
        
        # 5. 检查基本面数据  
        basic_data = pd.read_sql_query("""
            SELECT s.code, db.trade_date, db.pe_ttm, db.pb, db.total_mv
            FROM daily_basic db
            JOIN securities s ON db.security_id = s.id 
            WHERE s.code IN ('000001', '000002') AND db.trade_date >= '2025-08-01'
            ORDER BY db.trade_date DESC
            LIMIT 5
        """, conn)
        print(f"💰 基本面数据:")
        print(basic_data)

if __name__ == "__main__":
    debug_data()