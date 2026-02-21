#!/usr/bin/env python3
"""
更新股票名称到数据库
"""

import pandas as pd
import tushare as ts
import logging
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_adapter.database_manager import DatabaseManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# 读取配置
with open('config.json', 'r') as f:
    config = json.load(f)
    ts_token = config['tushare']['token']

ts.set_token(ts_token)
pro = ts.pro_api()

# 初始化数据库管理器
db_manager = DatabaseManager("data_adapter/stock_data.db")

def update_stock_names():
    """更新所有股票名称"""
    logger.info("开始更新股票名称...")
    
    try:
        # 获取股票基本信息
        logger.info("获取A股列表...")
        stock_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,market,list_date')
        
        updated_count = 0
        for _, row in stock_basic.iterrows():
            code = row['symbol']
            name = row['name']
            exchange = row['ts_code'].split('.')[1]
            list_date = row['list_date']
            
            # 更新数据库
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE securities 
                    SET name = ?, list_date = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE code = ? AND type = 'A股'
                """, (name, list_date, code))
                
                if cursor.rowcount > 0:
                    updated_count += 1
                    
                conn.commit()
        
        logger.info(f"A股名称更新完成: {updated_count} 只")
        
        # 获取ETF基本信息
        logger.info("获取ETF列表...")
        fund_basic = pro.fund_basic(market='E', fields='ts_code,name,list_date')
        
        etf_count = 0
        for _, row in fund_basic.iterrows():
            code = row['ts_code'].split('.')[0]
            name = row['name']
            list_date = row['list_date']
            
            # 更新数据库
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE securities 
                    SET name = ?, list_date = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE code = ? AND type = 'ETF_基金'
                """, (name, list_date, code))
                
                if cursor.rowcount > 0:
                    etf_count += 1
                    
                conn.commit()
        
        logger.info(f"ETF名称更新完成: {etf_count} 只")
        logger.info(f"总计更新: {updated_count + etf_count} 只证券名称")
        
    except Exception as e:
        logger.error(f"更新股票名称失败: {e}")

if __name__ == "__main__":
    update_stock_names()