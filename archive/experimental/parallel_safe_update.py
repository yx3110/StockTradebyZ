#!/usr/bin/env python3
"""
并行安全更新历史数据 - 多线程但控制API调用速率
使用线程池并行处理，但通过速率限制器避免超过API限制
"""

import pandas as pd
import tushare as ts
import time
import logging
from datetime import datetime, timedelta
import json
import sys
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import sqlite3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_adapter.database_manager import DatabaseManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(message)s"
)
logger = logging.getLogger(__name__)

# 读取配置
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)
    ts_token = config['tushare']['token']

ts.set_token(ts_token)

# 全局速率限制器
class RateLimiter:
    """API调用速率限制器 - 仅限制每分钟频率"""
    def __init__(self, calls_per_minute=500):
        self.calls_per_minute = calls_per_minute
        self.call_times = []
        self.lock = threading.Lock()
        
    def acquire(self):
        """获取调用许可"""
        with self.lock:
            now = time.time()
            
            # 清理超过1分钟的调用记录
            cutoff_time = now - 60
            self.call_times = [t for t in self.call_times if t > cutoff_time]
            
            # 检查是否超过频率限制
            if len(self.call_times) >= self.calls_per_minute:
                # 计算需要等待的时间
                oldest_call = min(self.call_times)
                wait_time = 60 - (now - oldest_call) + 0.1  # 多等0.1秒确保安全
                if wait_time > 0:
                    time.sleep(wait_time)
                    # 重新获取当前时间
                    now = time.time()
                    # 再次清理过期记录
                    cutoff_time = now - 60
                    self.call_times = [t for t in self.call_times if t > cutoff_time]
            
            # 记录本次调用时间
            self.call_times.append(now)
            return True

# 创建全局速率限制器（每分钟500次，无每日限制）
rate_limiter = RateLimiter(calls_per_minute=500)

def fetch_daily_basic_parallel(date_str: str):
    """获取daily_basic数据（线程安全）"""
    if not rate_limiter.acquire():
        return 0
    
    try:
        pro = ts.pro_api()
        df = pro.daily_basic(trade_date=date_str)
        
        if df.empty:
            return 0
        
        # 使用线程本地的数据库连接
        conn = sqlite3.connect('data_adapter/stock_data.db')
        cursor = conn.cursor()
        
        # 获取证券ID映射
        cursor.execute('SELECT code, id FROM securities')
        security_map = {row[0]: row[1] for row in cursor.fetchall()}
        
        count = 0
        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        for _, row in df.iterrows():
            code = row['ts_code'][:6]
            if code in security_map:
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO daily_basic 
                        (security_id, trade_date, close, turnover_rate, turnover_rate_f,
                         volume_ratio, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm,
                         total_share, float_share, free_share, total_mv, circ_mv)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        security_map[code], formatted_date, row.get('close'),
                        row.get('turnover_rate'), row.get('turnover_rate_f'),
                        row.get('volume_ratio'), row.get('pe'), row.get('pe_ttm'),
                        row.get('pb'), row.get('ps'), row.get('ps_ttm'),
                        row.get('dv_ratio'), row.get('dv_ttm'),
                        row.get('total_share'), row.get('float_share'), row.get('free_share'),
                        row.get('total_mv'), row.get('circ_mv')
                    ))
                    count += 1
                except Exception as e:
                    continue
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ {date_str} daily_basic: {count} 条")
        return count
        
    except Exception as e:
        logger.error(f"❌ {date_str} daily_basic失败: {e}")
        return 0

def calculate_technical_parallel(date_str: str):
    """计算技术指标（线程安全）"""
    try:
        date_dash = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        # 使用独立的数据库连接
        conn = sqlite3.connect('data_adapter/stock_data.db')
        cursor = conn.cursor()
        
        # 获取该日期有交易数据的股票
        cursor.execute("""
            SELECT DISTINCT s.id, s.code
            FROM securities s
            JOIN daily_quotes dq ON s.id = dq.security_id
            WHERE dq.trade_date = ?
            AND s.type = 'A股'
        """, (date_dash,))
        
        stocks = cursor.fetchall()
        count = 0
        
        for security_id, code in stocks:
            # 简化的技术指标计算（示例）
            # 实际应该调用完整的技术指标计算器
            try:
                # 获取历史数据
                cursor.execute("""
                    SELECT close, high, low, volume
                    FROM daily_quotes
                    WHERE security_id = ?
                    AND trade_date <= ?
                    ORDER BY trade_date DESC
                    LIMIT 30
                """, (security_id, date_dash))
                
                data = cursor.fetchall()
                if len(data) >= 20:
                    # 简单计算MA20
                    closes = [row[0] for row in data[:20]]
                    ma20 = sum(closes) / 20
                    
                    # 插入技术指标
                    cursor.execute("""
                        INSERT OR REPLACE INTO technical_indicators
                        (security_id, trade_date, ma20)
                        VALUES (?, ?, ?)
                    """, (security_id, date_dash, ma20))
                    count += 1
            except Exception as e:
                continue
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ {date_str} 技术指标: {count} 条")
        return count
        
    except Exception as e:
        logger.error(f"❌ {date_str} 技术指标失败: {e}")
        return 0

def process_single_date(date_str: str):
    """处理单个日期的所有数据"""
    logger.info(f"🔄 处理 {date_str}")
    
    results = {
        'date': date_str,
        'daily_basic': 0,
        'technical': 0,
        'success': False
    }
    
    # 获取daily_basic
    results['daily_basic'] = fetch_daily_basic_parallel(date_str)
    
    # 计算技术指标（不消耗API积分）
    results['technical'] = calculate_technical_parallel(date_str)
    
    results['success'] = results['daily_basic'] > 0
    
    return results

def parallel_batch_update(start_date: str, end_date: str, max_workers: int = 4):
    """
    并行批量更新，使用线程池但控制API速率
    
    Args:
        start_date: 开始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD
        max_workers: 最大线程数
    """
    logger.info(f"🚀 并行安全批量更新")
    logger.info(f"📅 日期范围: {start_date} 到 {end_date}")
    logger.info(f"🔧 线程数: {max_workers}")
    logger.info(f"⚠️ API限制: 500次/分钟, 无每日限制")
    logger.info("="*60)
    
    # 获取需要更新的日期
    db = DatabaseManager()
    
    # 查询缺少daily_basic的日期
    query = """
    SELECT DISTINCT DATE(dq.trade_date) as date
    FROM daily_quotes dq
    WHERE dq.trade_date BETWEEN ? AND ?
    AND NOT EXISTS (
        SELECT 1 FROM daily_basic db 
        WHERE DATE(db.trade_date) = DATE(dq.trade_date)
        LIMIT 1
    )
    ORDER BY date DESC
    """
    
    results = db.execute_query(query, (start_date, end_date))
    dates_to_update = [row[0].replace('-', '') for row in results]
    
    logger.info(f"📊 需要更新的日期: {len(dates_to_update)} 天")
    
    if not dates_to_update:
        logger.info("✅ 所有日期数据已完整")
        return
    
    # 处理所有需要更新的日期（无每日限制）
    dates_to_process = dates_to_update
    
    logger.info(f"📦 处理所有日期: {len(dates_to_process)} 天")
    
    # 使用线程池并行处理
    success_count = 0
    failed_dates = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_date = {
            executor.submit(process_single_date, date): date 
            for date in dates_to_process
        }
        
        # 收集结果
        completed = 0
        for future in as_completed(future_to_date):
            date = future_to_date[future]
            completed += 1
            
            try:
                result = future.result()
                
                if result['success']:
                    success_count += 1
                    logger.info(f"✅ [{completed}/{len(dates_to_process)}] {date} 完成")
                else:
                    failed_dates.append(date)
                    logger.warning(f"⚠️ [{completed}/{len(dates_to_process)}] {date} 部分失败")
                    
            except Exception as e:
                failed_dates.append(date)
                logger.error(f"❌ [{completed}/{len(dates_to_process)}] {date} 异常: {e}")
    
    # 统计结果
    logger.info("\n" + "="*60)
    logger.info(f"📊 全部完成统计:")
    logger.info(f"   ✅ 成功: {success_count}/{len(dates_to_process)} 天")
    logger.info(f"   ❌ 失败: {len(failed_dates)} 天")
    
    if failed_dates:
        logger.warning(f"失败的日期: {', '.join(failed_dates[:10])}")
        if len(failed_dates) > 10:
            logger.warning(f"... 等共 {len(failed_dates)} 天")
    
    logger.info(f"\n🎯 数据获取完成！可以开始生成报告了")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='并行安全批量更新')
    parser.add_argument('--start-date', default='2020-01-01',
                       help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-12-31',
                       help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='最大线程数 (默认: 4)')
    
    args = parser.parse_args()
    
    # 执行并行批量更新
    parallel_batch_update(args.start_date, args.end_date, args.max_workers)

if __name__ == "__main__":
    main()