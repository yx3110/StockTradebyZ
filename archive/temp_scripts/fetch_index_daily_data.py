#!/usr/bin/env python3
"""
获取市场指数日线数据填充index_daily表
"""

import tushare as ts
import pandas as pd
import sqlite3
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/fetch_index_daily.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IndexDataFetcher:
    def __init__(self):
        """初始化指数数据获取器"""
        # 读取配置
        with open('config.json', 'r') as f:
            config = json.load(f)
            self.ts_token = config['tushare']['token']
        
        # 初始化Tushare
        ts.set_token(self.ts_token)
        self.pro = ts.pro_api()
        
        # 数据库路径
        self.db_path = 'data_adapter/stock_data.db'
        
        # 主要指数列表（Tushare代码）
        self.major_indices = {
            '000001.SH': '上证指数',
            '399001.SZ': '深证成指', 
            '399006.SZ': '创业板指',
            '000300.SH': '沪深300',
            '000016.SH': '上证50',
            '000905.SH': '中证500',
            '000852.SH': '中证1000',
            '399003.SZ': '成份B指',
            '399005.SZ': '中小板指',
            '000688.SH': '科创50',
            '930050.SH': 'ESG基准'
        }
        
        # API限制配置
        self.api_delay = 0.5  # 每次调用间隔0.5秒
        
    def get_index_ids(self):
        """获取或创建指数在market_indices表中的ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            index_ids = {}
            
            for ts_code, name in self.major_indices.items():
                # 查看是否已存在
                cursor.execute("""
                    SELECT id FROM market_indices 
                    WHERE ts_code = ?
                """, (ts_code,))
                
                result = cursor.fetchone()
                if result:
                    index_ids[ts_code] = result[0]
                else:
                    # 插入新指数
                    cursor.execute("""
                        INSERT INTO market_indices (ts_code, name, category)
                        VALUES (?, ?, ?)
                    """, (ts_code, name, '主要指数'))
                    index_ids[ts_code] = cursor.lastrowid
                    logger.info(f"创建指数记录: {name} ({ts_code})")
            
            conn.commit()
            conn.close()
            
            return index_ids
            
        except Exception as e:
            logger.error(f"获取指数ID失败: {e}")
            return {}
    
    def fetch_index_daily_data(self, ts_code, start_date='20200101', end_date=None):
        """获取单个指数的日线数据"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        try:
            logger.info(f"获取 {self.major_indices[ts_code]} ({ts_code}) 从 {start_date} 到 {end_date}")
            
            # API限流
            time.sleep(self.api_delay)
            
            # 获取指数日线数据
            df = self.pro.index_daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if df.empty:
                logger.warning(f"未获取到 {ts_code} 的数据")
                return pd.DataFrame()
            
            logger.info(f"获取到 {len(df)} 条记录: {ts_code}")
            return df
            
        except Exception as e:
            logger.error(f"获取 {ts_code} 数据失败: {e}")
            return pd.DataFrame()
    
    def save_index_data(self, index_id, df, ts_code):
        """保存指数数据到数据库"""
        if df.empty:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            saved_count = 0
            
            for _, row in df.iterrows():
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO index_daily 
                        (index_id, trade_date, open, high, low, close, vol, amount)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        index_id,
                        row['trade_date'],
                        row.get('open'),
                        row.get('high'), 
                        row.get('low'),
                        row.get('close'),
                        row.get('vol'),
                        row.get('amount')
                    ))
                    saved_count += 1
                except Exception as e:
                    logger.debug(f"插入记录失败: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            logger.info(f"保存 {self.major_indices[ts_code]} 数据: {saved_count} 条记录")
            
        except Exception as e:
            logger.error(f"保存 {ts_code} 数据失败: {e}")
    
    def fetch_all_index_data(self, start_date='20200101'):
        """获取所有主要指数的数据"""
        logger.info("开始获取市场指数日线数据...")
        
        # 获取指数ID映射
        index_ids = self.get_index_ids()
        if not index_ids:
            logger.error("无法获取指数ID")
            return
        
        total_records = 0
        
        for ts_code, name in self.major_indices.items():
            if ts_code not in index_ids:
                logger.warning(f"跳过 {name}: 未找到ID")
                continue
            
            # 获取数据
            df = self.fetch_index_daily_data(ts_code, start_date)
            
            if not df.empty:
                # 保存数据
                self.save_index_data(index_ids[ts_code], df, ts_code)
                total_records += len(df)
        
        logger.info(f"指数数据获取完成！总共 {total_records} 条记录")
        
        # 验证数据
        self.verify_index_data()
    
    def verify_index_data(self):
        """验证指数数据完整性"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 统计信息
            cursor.execute("""
                SELECT mi.name, COUNT(id.id) as record_count,
                       MIN(id.trade_date) as earliest_date,
                       MAX(id.trade_date) as latest_date
                FROM market_indices mi
                LEFT JOIN index_daily id ON mi.id = id.index_id
                GROUP BY mi.id, mi.name
                ORDER BY mi.name
            """)
            
            results = cursor.fetchall()
            conn.close()
            
            logger.info("\n指数数据验证结果:")
            logger.info("=" * 60)
            for name, count, earliest, latest in results:
                logger.info(f"{name:12} | {count:6} 条 | {earliest} ~ {latest}")
            
        except Exception as e:
            logger.error(f"数据验证失败: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='获取市场指数日线数据')
    parser.add_argument('--start-date', default='20200101', help='开始日期 (YYYYMMDD)')
    parser.add_argument('--verify', action='store_true', help='仅验证数据')
    
    args = parser.parse_args()
    
    # 确保日志目录存在
    Path('logs').mkdir(exist_ok=True)
    
    fetcher = IndexDataFetcher()
    
    if args.verify:
        fetcher.verify_index_data()
    else:
        fetcher.fetch_all_index_data(args.start_date)

if __name__ == "__main__":
    main()