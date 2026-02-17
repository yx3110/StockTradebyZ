#!/usr/bin/env python3
"""
完整的数据库补全脚本
从Tushare获取所有缺失的数据并填充到数据库
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
        logging.FileHandler('logs/complete_database_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CompleteDatabaseUpdater:
    def __init__(self):
        """初始化完整数据更新器"""
        # 读取配置
        with open('config.json', 'r') as f:
            config = json.load(f)
            self.ts_token = config['tushare']['token']
        
        # 初始化Tushare
        ts.set_token(self.ts_token)
        self.pro = ts.pro_api()
        
        # 数据库路径
        self.db_path = 'data_adapter/stock_data.db'
        
        # API限制配置
        self.api_delay = 0.5
    
    def update_securities_info(self):
        """更新securities表的industry和area字段"""
        logger.info("开始更新securities表的行业和地区信息...")
        
        try:
            time.sleep(self.api_delay)
            # 获取股票基础信息
            df = self.pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,symbol,name,area,industry,market,list_date'
            )
            
            logger.info(f"获取到 {len(df)} 只股票的基础信息")
            
            # 更新数据库
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            updated_count = 0
            
            for _, row in df.iterrows():
                try:
                    ts_code = row['ts_code']
                    code = ts_code.split('.')[0]
                    
                    # 更新securities表
                    cursor.execute("""
                        UPDATE securities 
                        SET name = ?, industry = ?, area = ?
                        WHERE code = ?
                    """, (
                        row.get('name'),
                        row.get('industry'),
                        row.get('area'),
                        code
                    ))
                    
                    if cursor.rowcount > 0:
                        updated_count += 1
                        
                except Exception as e:
                    logger.debug(f"更新{code}失败: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            logger.info(f"成功更新 {updated_count} 条securities记录")
            
        except Exception as e:
            logger.error(f"更新securities信息失败: {e}")
    
    def update_historical_daily_basic(self, days=30):
        """更新最近N天的daily_basic数据"""
        logger.info(f"开始更新最近{days}天的daily_basic数据...")
        
        try:
            # 获取日期范围
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            for i in range(days):
                trade_date = (start_date + timedelta(days=i)).strftime('%Y%m%d')
                
                try:
                    time.sleep(self.api_delay)
                    
                    # 获取当日数据
                    df = self.pro.daily_basic(
                        ts_code='',
                        trade_date=trade_date,
                        fields='ts_code,trade_date,close,turnover_rate,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,total_share,float_share,free_share,total_mv,circ_mv'
                    )
                    
                    if df.empty:
                        logger.debug(f"跳过{trade_date}: 无数据")
                        continue
                    
                    # 保存数据
                    self._save_daily_basic(df, trade_date)
                    
                except Exception as e:
                    logger.debug(f"获取{trade_date}数据失败: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"更新历史daily_basic失败: {e}")
    
    def _save_daily_basic(self, df, trade_date):
        """保存daily_basic数据"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            saved_count = 0
            
            for _, row in df.iterrows():
                try:
                    ts_code = row['ts_code']
                    code = ts_code.split('.')[0]
                    
                    # 查找security_id
                    cursor.execute("SELECT id FROM securities WHERE code = ?", (code,))
                    result = cursor.fetchone()
                    
                    if result:
                        security_id = result[0]
                        
                        cursor.execute("""
                            INSERT OR REPLACE INTO daily_basic 
                            (security_id, trade_date, close, turnover_rate, volume_ratio,
                             pe, pe_ttm, pb, ps, ps_ttm, total_share, float_share, 
                             free_share, total_mv, circ_mv)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            security_id,
                            row['trade_date'],
                            row.get('close'),
                            row.get('turnover_rate'),
                            row.get('volume_ratio'),
                            row.get('pe'),
                            row.get('pe_ttm'),
                            row.get('pb'),
                            row.get('ps'),
                            row.get('ps_ttm'),
                            row.get('total_share'),
                            row.get('float_share'),
                            row.get('free_share'),
                            row.get('total_mv'),
                            row.get('circ_mv')
                        ))
                        saved_count += 1
                        
                except Exception as e:
                    continue
                    
            conn.commit()
            conn.close()
            
            if saved_count > 0:
                logger.info(f"保存{trade_date}: {saved_count}条记录")
                
        except Exception as e:
            logger.error(f"保存{trade_date}数据失败: {e}")
    
    def verify_data_completeness(self):
        """验证数据完整性"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 各表记录数统计
            tables = [
                'securities', 'stock_basic_info', 'daily_basic', 
                'financial_indicator', 'technical_indicators', 
                'market_indices', 'index_daily'
            ]
            
            logger.info("=" * 60)
            logger.info("数据库完整性验证:")
            logger.info("=" * 60)
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                logger.info(f"{table:20} | {count:8} 条记录")
            
            # 检查关键字段完整性
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(industry) as with_industry,
                    COUNT(area) as with_area
                FROM securities 
                WHERE type IN ('A股', '科创板', '创业板')
            """)
            
            total, with_industry, with_area = cursor.fetchone()
            
            logger.info("=" * 60)
            logger.info("关键字段完整性:")
            logger.info(f"A股总数: {total}")
            logger.info(f"有行业信息: {with_industry} ({with_industry/total*100:.1f}%)")
            logger.info(f"有地区信息: {with_area} ({with_area/total*100:.1f}%)")
            
            # 最新数据日期
            cursor.execute("SELECT MAX(trade_date) FROM daily_basic")
            latest_date = cursor.fetchone()[0]
            logger.info(f"最新daily_basic日期: {latest_date}")
            
            conn.close()
            
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"数据验证失败: {e}")
    
    def run_complete_update(self):
        """运行完整更新"""
        logger.info("开始完整数据库更新...")
        
        # 1. 更新securities基础信息
        self.update_securities_info()
        
        # 2. 更新历史daily_basic数据
        self.update_historical_daily_basic(days=10)  # 最近10天
        
        # 3. 验证数据完整性
        self.verify_data_completeness()
        
        logger.info("完整数据库更新完成!")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='完整数据库更新')
    parser.add_argument('--securities', action='store_true', help='只更新securities信息')
    parser.add_argument('--daily-basic', action='store_true', help='只更新daily_basic')
    parser.add_argument('--verify', action='store_true', help='只验证数据')
    parser.add_argument('--all', action='store_true', help='完整更新')
    
    args = parser.parse_args()
    
    # 确保日志目录存在
    Path('logs').mkdir(exist_ok=True)
    
    updater = CompleteDatabaseUpdater()
    
    if args.verify:
        updater.verify_data_completeness()
    elif args.securities:
        updater.update_securities_info()
        updater.verify_data_completeness()
    elif args.daily_basic:
        updater.update_historical_daily_basic()
        updater.verify_data_completeness()
    elif args.all:
        updater.run_complete_update()
    else:
        logger.info("请指定操作选项，使用 --help 查看帮助")

if __name__ == "__main__":
    main()