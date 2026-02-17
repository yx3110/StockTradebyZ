#!/usr/bin/env python3
"""
高速获取所有A股的财务指标数据
使用批量API调用和并行处理
"""

import tushare as ts
import pandas as pd
import sqlite3
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/fetch_financial_data_fast.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FastFinancialDataFetcher:
    def __init__(self):
        """初始化高速财务数据获取器"""
        # 读取配置
        with open('config.json', 'r') as f:
            config = json.load(f)
            self.ts_token = config['tushare']['token']
        
        # 初始化Tushare
        ts.set_token(self.ts_token)
        self.pro = ts.pro_api()
        
        # 数据库路径
        self.db_path = 'data_adapter/stock_data.db'
        
        # 并行配置
        self.max_workers = 8  # 并行线程数
        self.batch_size = 100  # 批量获取大小
        
        # API限制配置
        self.api_delay = 0.05  # 每次调用间隔0.05秒
        self.api_lock = threading.Lock()  # API调用锁
        self.last_api_call = 0
        
        # 数据队列
        self.data_queue = queue.Queue()
        
        # 统计信息
        self.stats = {
            'total_stocks': 0,
            'processed_stocks': 0,
            'success_count': 0,
            'failed_count': 0,
            'total_records': 0,
            'start_time': None,
            'end_time': None
        }
        self.stats_lock = threading.Lock()
    
    def get_all_stocks(self):
        """获取所有A股股票列表"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 获取所有A股
            cursor.execute("""
                SELECT DISTINCT s.id, s.code, s.exchange, s.name
                FROM securities s
                WHERE s.type IN ('A股', '科创板', '创业板') 
                AND s.is_active = 1
                ORDER BY s.code
            """)
            
            stocks = cursor.fetchall()
            conn.close()
            
            logger.info(f"找到 {len(stocks)} 只股票需要获取财务数据")
            return stocks
            
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []
    
    def batch_fetch_financial_data(self, stock_batch):
        """批量获取财务数据（使用通用接口）"""
        try:
            # 构建股票代码列表
            ts_codes = ','.join([f"{code}.{exchange}" for _, code, exchange, _ in stock_batch])
            
            # API限流
            with self.api_lock:
                current_time = time.time()
                time_since_last = current_time - self.last_api_call
                if time_since_last < self.api_delay:
                    time.sleep(self.api_delay - time_since_last)
                
                # 获取最新一期财务数据（批量）
                df = self.pro.query(
                    'fina_indicator',
                    ts_code=ts_codes,
                    period='20240930',  # 获取最新季度
                    fields='ts_code,ann_date,end_date,eps,roe,roa,gross_margin,netprofit_margin,current_ratio,debt_to_assets,netprofit_yoy,or_yoy,basic_eps_yoy'
                )
                
                self.last_api_call = time.time()
            
            if df.empty:
                return []
            
            # 整理数据
            results = []
            for _, row in df.iterrows():
                ts_code = row['ts_code']
                code = ts_code.split('.')[0]
                
                # 找到对应的股票信息
                stock_info = None
                for s in stock_batch:
                    if s[1] == code:
                        stock_info = s
                        break
                
                if stock_info:
                    results.append((stock_info[0], row))  # (security_id, data)
            
            return results
            
        except Exception as e:
            logger.error(f"批量获取财务数据失败: {e}")
            return []
    
    def fetch_historical_data(self, stock_info):
        """获取单只股票的历史财务数据"""
        security_id, code, exchange, name = stock_info
        
        try:
            ts_code = f'{code}.{exchange}'
            
            # API限流
            with self.api_lock:
                current_time = time.time()
                time_since_last = current_time - self.last_api_call
                if time_since_last < self.api_delay:
                    time.sleep(self.api_delay - time_since_last)
                
                # 获取历史财务数据
                df = self.pro.fina_indicator(ts_code=ts_code, limit=8)
                self.last_api_call = time.time()
            
            if df.empty:
                return []
            
            # 返回数据
            results = []
            for _, row in df.iterrows():
                results.append((security_id, row))
            
            with self.stats_lock:
                self.stats['processed_stocks'] += 1
                self.stats['success_count'] += 1
                self.stats['total_records'] += len(results)
            
            return results
            
        except Exception as e:
            logger.debug(f"获取 {code} 历史数据失败: {e}")
            with self.stats_lock:
                self.stats['processed_stocks'] += 1
                self.stats['failed_count'] += 1
            return []
    
    def save_financial_data(self, data_batch):
        """批量保存财务数据到数据库"""
        if not data_batch:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for security_id, row in data_batch:
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO financial_indicator 
                        (security_id, ann_date, end_date, eps, roe, roa, gross_margin, 
                         netprofit_margin, current_ratio, debt_to_assets, 
                         netprofit_yoy, or_yoy, basic_eps_yoy)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        security_id,
                        row.get('ann_date'),
                        row.get('end_date'),
                        row.get('eps'),
                        row.get('roe'),
                        row.get('roa'),
                        row.get('gross_margin'),
                        row.get('netprofit_margin'),
                        row.get('current_ratio'),
                        row.get('debt_to_assets'),
                        row.get('netprofit_yoy'),
                        row.get('or_yoy'),
                        row.get('basic_eps_yoy')
                    ))
                except Exception as e:
                    logger.debug(f"插入记录失败: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            logger.info(f"批量保存 {len(data_batch)} 条财务数据")
            
        except Exception as e:
            logger.error(f"批量保存数据失败: {e}")
    
    def data_saver_worker(self):
        """数据保存工作线程"""
        batch = []
        batch_size = 500  # 每500条记录保存一次
        
        while True:
            try:
                # 从队列获取数据（超时1秒）
                data = self.data_queue.get(timeout=1)
                
                if data is None:  # 结束信号
                    if batch:
                        self.save_financial_data(batch)
                    break
                
                batch.extend(data)
                
                # 批量保存
                if len(batch) >= batch_size:
                    self.save_financial_data(batch)
                    batch = []
                    
            except queue.Empty:
                # 队列空了，保存剩余数据
                if batch:
                    self.save_financial_data(batch)
                    batch = []
            except Exception as e:
                logger.error(f"数据保存线程错误: {e}")
    
    def fetch_all_financial_data_fast(self):
        """高速获取所有股票的财务数据"""
        self.stats['start_time'] = datetime.now()
        
        # 获取股票列表
        stocks = self.get_all_stocks()
        if not stocks:
            logger.error("无法获取股票列表")
            return
        
        self.stats['total_stocks'] = len(stocks)
        
        # 启动数据保存线程
        saver_thread = threading.Thread(target=self.data_saver_worker)
        saver_thread.start()
        
        # 方案1：先批量获取最新数据（速度最快）
        logger.info("第1阶段：批量获取最新财务数据...")
        
        for i in range(0, len(stocks), self.batch_size):
            batch = stocks[i:i+self.batch_size]
            logger.info(f"处理批次 {i//self.batch_size + 1}/{(len(stocks)-1)//self.batch_size + 1}")
            
            # 批量获取
            data = self.batch_fetch_financial_data(batch)
            if data:
                self.data_queue.put(data)
                with self.stats_lock:
                    self.stats['processed_stocks'] += len(batch)
                    self.stats['total_records'] += len(data)
            
            # 显示进度
            if (i // self.batch_size) % 10 == 0:
                self._print_progress()
        
        # 方案2：并行获取历史数据
        logger.info("\n第2阶段：并行获取历史财务数据...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            futures = []
            for stock_info in stocks[:1000]:  # 先处理前1000只
                future = executor.submit(self.fetch_historical_data, stock_info)
                futures.append(future)
            
            # 处理完成的任务
            for future in as_completed(futures):
                try:
                    data = future.result()
                    if data:
                        self.data_queue.put(data)
                    
                    # 定期显示进度
                    with self.stats_lock:
                        if self.stats['processed_stocks'] % 100 == 0:
                            self._print_progress()
                            
                except Exception as e:
                    logger.error(f"处理任务失败: {e}")
        
        # 发送结束信号给保存线程
        self.data_queue.put(None)
        saver_thread.join()
        
        self.stats['end_time'] = datetime.now()
        self._print_final_stats()
    
    def _print_progress(self):
        """打印进度信息"""
        with self.stats_lock:
            progress = self.stats['processed_stocks'] / self.stats['total_stocks'] * 100 if self.stats['total_stocks'] > 0 else 0
            elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
            speed = self.stats['processed_stocks'] / elapsed if elapsed > 0 else 0
            
            logger.info(f"""
进度: {self.stats['processed_stocks']}/{self.stats['total_stocks']} ({progress:.1f}%)
成功: {self.stats['success_count']} | 失败: {self.stats['failed_count']}
已获取记录: {self.stats['total_records']}
速度: {speed:.1f} 只/秒
            """)
    
    def _print_final_stats(self):
        """打印最终统计信息"""
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        logger.info(f"""
========================================
财务数据高速获取完成！
========================================
总股票数: {self.stats['total_stocks']}
处理股票: {self.stats['processed_stocks']}
成功获取: {self.stats['success_count']}
失败数量: {self.stats['failed_count']}
总记录数: {self.stats['total_records']}
总耗时: {duration:.1f} 秒 ({duration/60:.1f} 分钟)
平均速度: {self.stats['total_stocks']/duration:.1f} 只/秒
========================================
        """)
    
    def verify_data(self):
        """验证数据完整性"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 统计信息
            cursor.execute("SELECT COUNT(DISTINCT security_id) FROM financial_indicator")
            stock_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM financial_indicator")
            record_count = cursor.fetchone()[0]
            
            conn.close()
            
            logger.info(f"""
数据验证结果:
- 覆盖股票数: {stock_count}
- 总记录数: {record_count}
            """)
            
        except Exception as e:
            logger.error(f"数据验证失败: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='高速获取所有A股财务数据')
    parser.add_argument('--verify', action='store_true', help='仅验证数据')
    
    args = parser.parse_args()
    
    # 确保日志目录存在
    Path('logs').mkdir(exist_ok=True)
    
    fetcher = FastFinancialDataFetcher()
    
    if args.verify:
        fetcher.verify_data()
    else:
        logger.info("开始高速获取所有A股财务数据...")
        fetcher.fetch_all_financial_data_fast()
        fetcher.verify_data()

if __name__ == "__main__":
    main()