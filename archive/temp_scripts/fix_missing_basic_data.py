#!/usr/bin/env python3
"""
补充缺失的股票基础信息数据
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
        logging.FileHandler('logs/fix_missing_basic_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BasicDataFixer:
    def __init__(self):
        """初始化数据修复器"""
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
        self.api_delay = 1  # 每次调用间隔1秒
    
    def fix_stock_basic_info(self):
        """填充股票基础信息表"""
        logger.info("开始填充stock_basic_info表...")
        
        try:
            # 获取股票基础信息
            logger.info("从Tushare获取股票基础信息...")
            time.sleep(self.api_delay)
            
            df = self.pro.stock_basic(
                exchange='',
                list_status='L',  # 只要上市的股票
                fields='ts_code,name,area,industry,market,list_date,main_business,employees'
            )
            
            logger.info(f"获取到 {len(df)} 只股票的基础信息")
            
            # 保存到数据库
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            saved_count = 0
            
            for _, row in df.iterrows():
                try:
                    ts_code = row['ts_code']
                    code = ts_code.split('.')[0]
                    exchange = ts_code.split('.')[1]
                    
                    # 查找对应的security_id
                    cursor.execute("""
                        SELECT id FROM securities 
                        WHERE code = ?
                    """, (code,))
                    
                    result = cursor.fetchone()
                    if result:
                        security_id = result[0]
                        
                        # 插入基础信息
                        cursor.execute("""
                            INSERT OR REPLACE INTO stock_basic_info 
                            (security_id, market, list_date, main_business, employees)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            security_id,
                            row.get('market'),
                            row.get('list_date'),
                            row.get('main_business'),
                            row.get('employees')
                        ))
                        saved_count += 1
                        
                except Exception as e:
                    logger.debug(f"处理{ts_code}失败: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            logger.info(f"成功保存 {saved_count} 条股票基础信息")
            
        except Exception as e:
            logger.error(f"填充stock_basic_info失败: {e}")
    
    def update_daily_basic_latest(self):
        """更新最新的daily_basic数据"""
        logger.info("开始更新最新daily_basic数据...")
        
        try:
            # 获取最新交易日
            today = datetime.now()
            trade_date = today.strftime('%Y%m%d')
            
            logger.info(f"获取{trade_date}的基础数据...")
            time.sleep(self.api_delay)
            
            # 获取最新基础数据
            df = self.pro.daily_basic(
                ts_code='',
                trade_date=trade_date,
                fields='ts_code,trade_date,close,turnover_rate,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,total_share,float_share,free_share,total_mv,circ_mv'
            )
            
            if df.empty:
                # 尝试前一个交易日
                prev_date = (today - timedelta(days=1)).strftime('%Y%m%d')
                logger.info(f"尝试获取{prev_date}的基础数据...")
                time.sleep(self.api_delay)
                
                df = self.pro.daily_basic(
                    ts_code='',
                    trade_date=prev_date,
                    fields='ts_code,trade_date,close,turnover_rate,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,total_share,float_share,free_share,total_mv,circ_mv'
                )
            
            if df.empty:
                logger.warning("无法获取最新基础数据")
                return
            
            logger.info(f"获取到 {len(df)} 只股票的最新基础数据")
            
            # 保存到数据库
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            saved_count = 0
            
            for _, row in df.iterrows():
                try:
                    ts_code = row['ts_code']
                    code = ts_code.split('.')[0]
                    exchange = ts_code.split('.')[1]
                    
                    # 查找对应的security_id
                    cursor.execute("""
                        SELECT id FROM securities 
                        WHERE code = ?
                    """, (code,))
                    
                    result = cursor.fetchone()
                    if result:
                        security_id = result[0]
                        
                        # 插入最新基础数据
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
                    logger.debug(f"处理{ts_code}失败: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            logger.info(f"成功保存 {saved_count} 条最新基础数据")
            
        except Exception as e:
            logger.error(f"更新daily_basic失败: {e}")
    
    def verify_data(self):
        """验证数据完整性"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 验证stock_basic_info
            cursor.execute("SELECT COUNT(*) FROM stock_basic_info")
            basic_info_count = cursor.fetchone()[0]
            
            # 验证daily_basic最新数据
            cursor.execute("""
                SELECT trade_date, COUNT(*) 
                FROM daily_basic 
                GROUP BY trade_date 
                ORDER BY trade_date DESC 
                LIMIT 1
            """)
            latest_basic = cursor.fetchone()
            
            conn.close()
            
            logger.info("=" * 50)
            logger.info("数据验证结果:")
            logger.info(f"stock_basic_info记录数: {basic_info_count}")
            if latest_basic:
                logger.info(f"最新daily_basic: {latest_basic[0]} ({latest_basic[1]}条)")
            else:
                logger.info("daily_basic: 无数据")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"数据验证失败: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='修复缺失的基础数据')
    parser.add_argument('--stock-basic', action='store_true', help='填充股票基础信息')
    parser.add_argument('--daily-basic', action='store_true', help='更新最新daily_basic')
    parser.add_argument('--all', action='store_true', help='执行所有修复')
    parser.add_argument('--verify', action='store_true', help='仅验证数据')
    
    args = parser.parse_args()
    
    # 确保日志目录存在
    Path('logs').mkdir(exist_ok=True)
    
    fixer = BasicDataFixer()
    
    if args.verify:
        fixer.verify_data()
    elif args.all:
        logger.info("开始执行完整数据修复...")
        fixer.fix_stock_basic_info()
        fixer.update_daily_basic_latest()
        fixer.verify_data()
    elif args.stock_basic:
        fixer.fix_stock_basic_info()
        fixer.verify_data()
    elif args.daily_basic:
        fixer.update_daily_basic_latest()
        fixer.verify_data()
    else:
        logger.info("请指定操作选项，使用 --help 查看帮助")

if __name__ == "__main__":
    main()