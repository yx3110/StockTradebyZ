#!/usr/bin/env python3
"""
一次性更新8月份发布财报的所有公司财务数据
Update financial data for all companies that published reports in August 2025
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append("data_adapter")

import tushare as ts
import pandas as pd
import json
import logging
import time
import sqlite3
import concurrent.futures
import threading
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """加载配置"""
    with open('config.json') as f:
        return json.load(f)

def get_all_a_stocks():
    """获取所有A股代码"""
    conn = sqlite3.connect('data_adapter/stock_data.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT s.code || '.' || 
               CASE WHEN s.exchange = 'SH' THEN 'SH' 
                    WHEN s.exchange = 'SZ' THEN 'SZ'
                    ELSE s.exchange END as ts_code
        FROM securities s 
        WHERE s.type = 'A股'
        ORDER BY s.id
    """)
    stock_codes = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return stock_codes

def save_financial_data_to_db(df_list):
    """保存财务数据到数据库"""
    if not df_list:
        return 0
        
    try:
        # 合并所有数据
        df = pd.concat(df_list, ignore_index=True)
        
        conn = sqlite3.connect('data_adapter/stock_data.db')
        cursor = conn.cursor()
        
        # 获取证券ID映射
        cursor.execute('SELECT code, id FROM securities')
        security_map = {row[0]: row[1] for row in cursor.fetchall()}
        
        count = 0
        for _, row in df.iterrows():
            code = row['ts_code'][:6]  # 获取6位代码
            if code in security_map:
                try:
                    # 插入或更新财务数据
                    cursor.execute('''
                        INSERT OR REPLACE INTO financial_indicator 
                        (security_id, ann_date, end_date, eps, roe, roa, netprofit_margin, debt_to_assets, netprofit_yoy, or_yoy)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        security_map[code], 
                        row.get('ann_date'), 
                        row.get('end_date'),
                        row.get('eps'), 
                        row.get('roe'), 
                        row.get('roa'),
                        row.get('netprofit_margin'),
                        row.get('debt_to_assets'),
                        row.get('netprofit_yoy'),
                        row.get('or_yoy')
                    ))
                    count += 1
                except Exception as e:
                    logger.debug(f"插入财务数据失败 {code}: {e}")
        
        conn.commit()
        conn.close()
        return count
        
    except Exception as e:
        logger.error(f"保存财务数据失败: {e}")
        return 0

def main():
    """主函数"""
    logger.info("开始更新8月份发布财报的所有公司财务数据")
    
    # 加载配置
    config = load_config()
    ts.set_token(config['tushare']['token'])
    pro = ts.pro_api()
    
    # 获取所有A股
    stock_codes = get_all_a_stocks()
    logger.info(f"获取到 {len(stock_codes)} 只A股代码")
    
    # 定义8月份的日期范围
    august_dates = [str(20250800 + day) for day in range(1, 27)]  # 20250801 到 20250826
    logger.info(f"检查日期范围：20250801 到 20250826 ({len(august_dates)} 天)")
    
    # 并行检查
    companies_with_august_reports = []
    checked_count = 0
    lock = threading.Lock()
    
    def check_single_stock(ts_code):
        """检查单只股票8月份的财务数据发布情况"""
        nonlocal checked_count
        try:
            # 获取该股票最新的财务指标
            fina_df = pro.fina_indicator(
                ts_code=ts_code,
                fields='ts_code,ann_date,end_date,eps,roe,roa,netprofit_margin,debt_to_assets,netprofit_yoy,or_yoy'
            )
            
            if not fina_df.empty:
                latest = fina_df.iloc[0]
                ann_date = str(latest['ann_date']) if latest['ann_date'] else ""
                
                # 检查是否在8月份发布
                if ann_date in august_dates:
                    with lock:
                        companies_with_august_reports.append(latest)
                    logger.info(f"✅ 发现 {ts_code} 在 {ann_date} 发布财务数据")
            
            # 更新进度
            with lock:
                checked_count += 1
                if checked_count % 200 == 0:
                    logger.info(f"进度: {checked_count}/{len(stock_codes)} ({checked_count/len(stock_codes)*100:.1f}%)")
                    
            return True
                
        except Exception as e:
            logger.debug(f"检查 {ts_code} 失败: {e}")
            with lock:
                checked_count += 1
            return False
    
    # 使用线程池并行处理
    max_workers = 8  # 增加并发数，因为是一次性批量操作
    logger.info(f"使用 {max_workers} 线程并行处理")
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(check_single_stock, ts_code) for ts_code in stock_codes]
        
        # 等待所有任务完成
        concurrent.futures.wait(futures)
    
    end_time = time.time()
    
    logger.info(f"检查完成！耗时: {end_time - start_time:.1f} 秒")
    logger.info(f"发现 {len(companies_with_august_reports)} 家公司在8月份发布了财务数据")
    
    if companies_with_august_reports:
        # 保存到数据库
        df_list = [pd.DataFrame([data]) for data in companies_with_august_reports]
        saved_count = save_financial_data_to_db(df_list)
        
        logger.info(f"✅ 成功更新 {saved_count} 条财务数据记录")
        
        # 显示一些统计信息
        august_companies = {}
        for data in companies_with_august_reports:
            ann_date = str(data['ann_date'])
            if ann_date not in august_companies:
                august_companies[ann_date] = []
            august_companies[ann_date].append(data['ts_code'])
        
        logger.info("8月份各日期财报发布统计:")
        for date in sorted(august_companies.keys()):
            logger.info(f"  {date}: {len(august_companies[date])} 家公司")
            
    else:
        logger.info("未发现8月份发布财报的公司")
    
    logger.info("8月份财务数据更新完成！")

if __name__ == "__main__":
    main()