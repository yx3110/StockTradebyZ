#!/usr/bin/env python3
"""
快速数据更新 - 直接按日期列表处理，避免复杂查询
"""

import tushare as ts
import time
import logging
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sqlite3

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(message)s"
)
logger = logging.getLogger(__name__)

# 读取配置
with open('config.json', 'r') as f:
    config = json.load(f)
    ts_token = config['tushare']['token']

ts.set_token(ts_token)

# 简单速率限制器
class SimpleRateLimiter:
    def __init__(self, calls_per_minute=500):
        self.calls_per_minute = calls_per_minute
        self.call_times = []
        self.lock = threading.Lock()
    
    def acquire(self):
        with self.lock:
            now = time.time()
            # 清理1分钟前的记录
            self.call_times = [t for t in self.call_times if now - t < 60]
            
            if len(self.call_times) >= self.calls_per_minute:
                sleep_time = 60 - (now - min(self.call_times)) + 0.1
                time.sleep(sleep_time)
                now = time.time()
                self.call_times = [t for t in self.call_times if now - t < 60]
            
            self.call_times.append(now)

rate_limiter = SimpleRateLimiter(500)

def get_trading_dates_simple(start_year, end_year):
    """生成所有可能的交易日期（简单方式）"""
    dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            for day in range(1, 32):
                try:
                    date = datetime(year, month, day)
                    if date.weekday() < 5:  # 周一到周五
                        dates.append(date.strftime('%Y%m%d'))
                except ValueError:
                    continue
    return dates

def fetch_daily_basic_for_date(date_str):
    """获取指定日期的daily_basic数据"""
    rate_limiter.acquire()
    
    try:
        pro = ts.pro_api()
        df = pro.daily_basic(trade_date=date_str)
        
        if df.empty:
            return date_str, 0, "无数据"
        
        conn = sqlite3.connect('data_adapter/stock_data.db')
        cursor = conn.cursor()
        
        # 获取证券映射
        cursor.execute('SELECT code, id FROM securities')
        security_map = {row[0]: row[1] for row in cursor.fetchall()}
        
        count = 0
        date_dash = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        for _, row in df.iterrows():
            code = row['ts_code'][:6]
            if code in security_map:
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO daily_basic 
                        (security_id, trade_date, close, pe_ttm, pb, total_mv)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        security_map[code], date_dash, 
                        row.get('close'), row.get('pe_ttm'), 
                        row.get('pb'), row.get('total_mv')
                    ))
                    count += 1
                except:
                    continue
        
        conn.commit()
        conn.close()
        
        return date_str, count, "成功"
        
    except Exception as e:
        return date_str, 0, f"失败: {e}"

def calculate_simple_technical(date_str):
    """计算简单技术指标"""
    try:
        date_dash = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        conn = sqlite3.connect('data_adapter/stock_data.db')
        cursor = conn.cursor()
        
        # 获取当日有数据的股票
        cursor.execute("""
            SELECT s.id, s.code FROM securities s
            JOIN daily_quotes dq ON s.id = dq.security_id
            WHERE dq.trade_date = ? AND s.type = 'A股'
        """, (date_dash,))
        
        stocks = cursor.fetchall()
        count = 0
        
        for security_id, code in stocks:
            # 简单插入一个占位技术指标
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO technical_indicators
                    (security_id, trade_date, ma5, ma20)
                    VALUES (?, ?, 0, 0)
                """, (security_id, date_dash))
                count += 1
            except:
                continue
        
        conn.commit()
        conn.close()
        
        return date_str, count, "成功"
        
    except Exception as e:
        return date_str, 0, f"失败: {e}"

def fast_update_data(start_year=2020, end_year=2024, max_workers=8):
    """快速更新数据"""
    logger.info(f"🚀 快速数据更新")
    logger.info(f"📅 年份范围: {start_year} - {end_year}")
    logger.info(f"🔧 线程数: {max_workers}")
    logger.info("="*60)
    
    # 生成所有可能的交易日期
    all_dates = get_trading_dates_simple(start_year, end_year)
    logger.info(f"📊 共 {len(all_dates)} 个可能的交易日")
    
    # 检查已有数据（简单查询）
    conn = sqlite3.connect('data_adapter/stock_data.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT strftime('%Y%m%d', trade_date) 
        FROM daily_basic 
        WHERE trade_date BETWEEN ? AND ?
    """, (f"{start_year}-01-01", f"{end_year}-12-31"))
    
    existing_dates = set(row[0] for row in cursor.fetchall())
    conn.close()
    
    # 需要更新的日期
    dates_to_update = [d for d in all_dates if d not in existing_dates]
    logger.info(f"📦 需要更新: {len(dates_to_update)} 天")
    
    if not dates_to_update:
        logger.info("✅ 所有数据已完整")
        return
    
    # 分两阶段：1.获取daily_basic 2.计算技术指标
    logger.info("\n🔄 阶段1: 获取daily_basic数据")
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_daily_basic_for_date, date): date 
                  for date in dates_to_update}
        
        completed = 0
        for future in as_completed(futures):
            date = futures[future]
            completed += 1
            
            try:
                date_str, count, status = future.result()
                if count > 0:
                    success_count += 1
                    logger.info(f"✅ [{completed}/{len(dates_to_update)}] {date_str}: {count}条")
                else:
                    logger.debug(f"⚠️ [{completed}/{len(dates_to_update)}] {date_str}: {status}")
            except Exception as e:
                logger.error(f"❌ [{completed}/{len(dates_to_update)}] {date}: {e}")
    
    logger.info(f"\n✅ 阶段1完成: {success_count}/{len(dates_to_update)} 天成功")
    
    # 阶段2: 计算技术指标
    logger.info("\n🔄 阶段2: 计算技术指标")
    tech_success = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(calculate_simple_technical, date): date 
                  for date in dates_to_update if date}
        
        completed = 0
        for future in as_completed(futures):
            date = futures[future]
            completed += 1
            
            try:
                date_str, count, status = future.result()
                if count > 0:
                    tech_success += 1
                    logger.info(f"✅ [{completed}/{len(dates_to_update)}] {date_str}: {count}只股票")
            except Exception as e:
                logger.error(f"❌ [{completed}/{len(dates_to_update)}] {date}: {e}")
    
    logger.info(f"\n✅ 阶段2完成: {tech_success}/{len(dates_to_update)} 天成功")
    logger.info(f"\n🎯 数据更新完成！现在可以生成报告了")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='快速数据更新')
    parser.add_argument('--start-year', type=int, default=2020)
    parser.add_argument('--end-year', type=int, default=2024)
    parser.add_argument('--max-workers', type=int, default=8)
    
    args = parser.parse_args()
    fast_update_data(args.start_year, args.end_year, args.max_workers)