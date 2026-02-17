#!/usr/bin/env python3
"""
补全缺失的历史数据
"""

import pandas as pd
import tushare as ts
import logging
import json
import sys
import os
import time
from datetime import datetime, timedelta
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

def identify_missing_data():
    """识别缺失数据的证券"""
    logger.info("识别缺失数据的证券...")
    
    with db_manager.get_connection() as conn:
        # 查找最近30天没有数据的证券
        query = """
        SELECT 
            s.id as security_id,
            s.code,
            s.name,
            s.type,
            s.exchange,
            COUNT(q.trade_date) as recent_data_count,
            MAX(q.trade_date) as last_date
        FROM securities s
        LEFT JOIN daily_quotes q ON s.id = q.security_id 
            AND q.trade_date >= date('now', '-30 days')
        WHERE s.is_active = 1
        GROUP BY s.id, s.code, s.name, s.type, s.exchange
        HAVING recent_data_count < 20
        ORDER BY s.type, recent_data_count
        """
        
        df = pd.read_sql_query(query, conn)
        
    logger.info(f"发现 {len(df)} 只证券需要补全数据")
    return df

def fill_stock_data(code: str, exchange: str, start_date: str = None):
    """补全单只股票的数据"""
    try:
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y%m%d')
        
        end_date = datetime.now().strftime('%Y%m%d')
        ts_code = f"{code}.{exchange}"
        
        logger.info(f"获取 {ts_code} 从 {start_date} 到 {end_date} 的数据")
        
        # 获取日线数据
        df = pro.daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields='ts_code,trade_date,open,close,high,low,vol,pct_chg'
        )
        
        if df.empty:
            logger.warning(f"{ts_code} 无数据")
            return 0
        
        # 获取证券ID
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM securities WHERE code = ?", (code,))
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"数据库中未找到股票代码: {code}")
                return 0
            
            security_id = result['id']
        
        # 准备插入数据
        data_to_insert = []
        for _, row in df.iterrows():
            trade_date = pd.to_datetime(row['trade_date'], format='%Y%m%d').strftime('%Y-%m-%d')
            
            data_to_insert.append({
                'security_id': security_id,
                'trade_date': trade_date,
                'open': row['open'],
                'close': row['close'],
                'high': row['high'],
                'low': row['low'],
                'volume': row['vol'],
                'price_change_pct': row.get('pct_chg', 0) / 100 if row.get('pct_chg') else 0,
                'is_limit_up': False,
                'is_limit_down': False
            })
        
        # 插入数据库
        if data_to_insert:
            rows_inserted = db_manager.insert_daily_quotes(data_to_insert)
            logger.info(f"{ts_code} 成功插入 {rows_inserted} 条记录")
            return rows_inserted
        
        return 0
        
    except Exception as e:
        logger.error(f"获取 {code} 数据失败: {e}")
        return 0

def fill_etf_data(code: str, exchange: str, start_date: str = None):
    """补全ETF/基金数据"""
    try:
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y%m%d')
        
        end_date = datetime.now().strftime('%Y%m%d')
        ts_code = f"{code}.{exchange}"
        
        logger.info(f"获取ETF {ts_code} 从 {start_date} 到 {end_date} 的数据")
        
        # 获取基金日线数据
        df = pro.fund_daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields='ts_code,trade_date,open,close,high,low,vol,pct_chg'
        )
        
        if df.empty:
            logger.warning(f"ETF {ts_code} 无数据")
            return 0
        
        # 获取证券ID
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM securities WHERE code = ?", (code,))
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"数据库中未找到ETF代码: {code}")
                return 0
            
            security_id = result['id']
        
        # 准备插入数据
        data_to_insert = []
        for _, row in df.iterrows():
            trade_date = pd.to_datetime(row['trade_date'], format='%Y%m%d').strftime('%Y-%m-%d')
            
            data_to_insert.append({
                'security_id': security_id,
                'trade_date': trade_date,
                'open': row['open'],
                'close': row['close'],
                'high': row['high'],
                'low': row['low'],
                'volume': row['vol'],
                'price_change_pct': row.get('pct_chg', 0) / 100 if row.get('pct_chg') else 0,
                'is_limit_up': False,
                'is_limit_down': False
            })
        
        # 插入数据库
        if data_to_insert:
            rows_inserted = db_manager.insert_daily_quotes(data_to_insert)
            logger.info(f"ETF {ts_code} 成功插入 {rows_inserted} 条记录")
            return rows_inserted
        
        return 0
        
    except Exception as e:
        logger.error(f"获取ETF {code} 数据失败: {e}")
        return 0

def fill_missing_data_batch(limit: int = 50):
    """批量补全缺失数据"""
    logger.info("开始批量补全缺失数据...")
    
    # 识别缺失数据
    missing_securities = identify_missing_data()
    
    if missing_securities.empty:
        logger.info("没有需要补全的数据")
        return
    
    # 限制处理数量
    if limit:
        missing_securities = missing_securities.head(limit)
        logger.info(f"限制处理 {len(missing_securities)} 只证券")
    
    total_inserted = 0
    success_count = 0
    
    for idx, row in missing_securities.iterrows():
        try:
            code = row['code']
            security_type = row['type']
            exchange = row['exchange'] or 'SZ'  # 默认深交所
            
            logger.info(f"处理 {idx+1}/{len(missing_securities)}: {code} ({security_type})")
            
            # 根据类型选择不同的API
            if security_type == 'A股':
                rows = fill_stock_data(code, exchange)
            elif security_type == 'ETF_基金':
                rows = fill_etf_data(code, exchange)
            else:
                logger.warning(f"未知证券类型: {security_type}")
                continue
            
            if rows > 0:
                total_inserted += rows
                success_count += 1
            
            # API限制：每秒最多调用1次
            time.sleep(1.2)
            
        except Exception as e:
            logger.error(f"处理 {row['code']} 失败: {e}")
            continue
    
    logger.info("="*60)
    logger.info(f"数据补全完成!")
    logger.info(f"处理证券数: {len(missing_securities)}")
    logger.info(f"成功补全: {success_count} 只")
    logger.info(f"总计插入: {total_inserted} 条记录")
    logger.info("="*60)

def fill_specific_gaps():
    """补全特定的数据间隙"""
    logger.info("补全特定的数据间隙...")
    
    # 从质量检查结果中找到的有间隙的股票
    gap_stocks = [
        '000545', '000633', '001267', '002176', '002762', 
        '002778', '002955', '160526', '160617', '160618'
    ]
    
    total_inserted = 0
    
    for code in gap_stocks:
        try:
            logger.info(f"补全 {code} 的数据间隙...")
            
            # 获取交易所信息
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT exchange, type FROM securities WHERE code = ?", (code,))
                result = cursor.fetchone()
                
                if not result:
                    logger.warning(f"未找到股票 {code}")
                    continue
                
                exchange = result['exchange'] or 'SZ'
                security_type = result['type']
            
            # 补全最近60天的数据
            if security_type == 'A股':
                rows = fill_stock_data(code, exchange)
            else:
                rows = fill_etf_data(code, exchange)
            
            total_inserted += rows
            time.sleep(1.2)
            
        except Exception as e:
            logger.error(f"补全 {code} 间隙失败: {e}")
            continue
    
    logger.info(f"间隙补全完成，插入 {total_inserted} 条记录")

def main():
    """主函数"""
    logger.info("开始数据补全任务...")
    
    # 1. 补全缺失数据较多的证券（前50只）
    fill_missing_data_batch(limit=50)
    
    # 2. 补全特定间隙
    fill_specific_gaps()
    
    logger.info("数据补全任务完成!")

if __name__ == "__main__":
    main()