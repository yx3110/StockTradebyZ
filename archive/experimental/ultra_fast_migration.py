#!/usr/bin/env python3
"""
超高速CSV迁移工具
使用批量读取、预处理和单连接大批量插入
"""

import pandas as pd
import sqlite3
from pathlib import Path
import logging
import time
import os
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class UltraFastMigration:
    def __init__(self, csv_dir="full_securities_data", db_path="data_adapter/stock_data.db", batch_size=10000):
        self.csv_dir = Path(csv_dir)
        self.db_path = db_path
        self.batch_size = batch_size
        self.data_queue = queue.Queue(maxsize=100)
        self.security_cache = {}
        
    def load_security_cache(self):
        """预加载证券信息到缓存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT code, id FROM securities")
        for code, security_id in cursor.fetchall():
            self.security_cache[code] = security_id
        conn.close()
        logger.info(f"加载了 {len(self.security_cache)} 个证券到缓存")
    
    def parse_csv_batch(self, csv_files_batch):
        """批量解析CSV文件"""
        all_data = []
        new_securities = []
        
        for csv_file in csv_files_batch:
            try:
                filename = csv_file.name
                if '_' not in filename or not filename.endswith('.csv'):
                    continue
                    
                code, type_ext = filename.rsplit('_', 1)
                security_type = type_ext.replace('.csv', '')
                
                # 检查是否需要创建新证券
                if code not in self.security_cache:
                    new_securities.append((code, code, security_type, None, 1))
                    # 临时分配一个ID（稍后更新）
                    self.security_cache[code] = None
                
                # 读取CSV数据 - 只读取必要的列
                try:
                    df = pd.read_csv(csv_file, usecols=['date', 'open', 'high', 'low', 'close', 'volume'])
                    if df.empty:
                        continue
                        
                    # 数据预处理
                    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                    df['code'] = code
                    all_data.append(df)
                    
                except Exception as e:
                    logger.warning(f"读取 {filename} 失败: {e}")
                    continue
                    
            except Exception as e:
                logger.warning(f"处理 {csv_file} 失败: {e}")
                continue
        
        return all_data, new_securities
    
    def insert_new_securities(self, new_securities):
        """批量插入新证券"""
        if not new_securities:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.executemany("""
                INSERT OR IGNORE INTO securities (code, name, type, exchange, is_active)
                VALUES (?, ?, ?, ?, ?)
            """, new_securities)
            conn.commit()
            
            # 更新缓存
            for code, _, _, _, _ in new_securities:
                cursor.execute("SELECT id FROM securities WHERE code = ?", (code,))
                result = cursor.fetchone()
                if result:
                    self.security_cache[code] = result[0]
                    
            logger.info(f"插入了 {len(new_securities)} 个新证券")
            
        finally:
            conn.close()
    
    def insert_quotes_batch(self, data_frames):
        """批量插入行情数据"""
        if not data_frames:
            return 0
            
        # 合并所有数据框
        combined_df = pd.concat(data_frames, ignore_index=True)
        
        # 添加security_id
        combined_df['security_id'] = combined_df['code'].map(self.security_cache)
        
        # 过滤掉没有security_id的行
        combined_df = combined_df.dropna(subset=['security_id'])
        combined_df['security_id'] = combined_df['security_id'].astype(int)
        
        if combined_df.empty:
            return 0
        
        # 准备插入数据
        rows_data = []
        for _, row in combined_df.iterrows():
            rows_data.append((
                row['security_id'],
                row['date'],
                float(row['open']) if pd.notna(row['open']) else None,
                float(row['high']) if pd.notna(row['high']) else None,
                float(row['low']) if pd.notna(row['low']) else None,
                float(row['close']) if pd.notna(row['close']) else None,
                int(row['volume']) if pd.notna(row['volume']) else None,
            ))
        
        # 批量插入
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.executemany("""
                INSERT OR REPLACE INTO daily_quotes (
                    security_id, trade_date, open, high, low, close, volume,
                    amount, adj_open, adj_high, adj_low, adj_close, adj_factor,
                    price_change, price_change_pct, is_limit_up, is_limit_down,
                    is_st, is_suspend, ma5, ma10, ma20, ma60, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL,
                         NULL, NULL, 0, 0, 0, 0, NULL, NULL, NULL, NULL, datetime('now'))
            """, rows_data)
            conn.commit()
            return len(rows_data)
            
        finally:
            conn.close()
    
    def run_migration(self):
        """运行迁移"""
        # 获取所有CSV文件
        csv_files = list(self.csv_dir.glob("*.csv"))
        logger.info(f"找到 {len(csv_files)} 个CSV文件")
        
        if not csv_files:
            logger.warning("没有找到CSV文件")
            return
        
        # 加载证券缓存
        self.load_security_cache()
        
        start_time = time.time()
        processed_files = 0
        total_records = 0
        
        # 批量处理文件
        batch_size = 50  # 每批处理50个文件
        
        for i in range(0, len(csv_files), batch_size):
            batch_files = csv_files[i:i+batch_size]
            
            try:
                # 解析CSV批次
                data_frames, new_securities = self.parse_csv_batch(batch_files)
                
                # 插入新证券
                self.insert_new_securities(new_securities)
                
                # 插入行情数据
                records_inserted = self.insert_quotes_batch(data_frames)
                
                processed_files += len(batch_files)
                total_records += records_inserted
                
                # 进度报告
                elapsed = time.time() - start_time
                rate = processed_files / elapsed
                remaining_files = len(csv_files) - processed_files
                eta = remaining_files / rate if rate > 0 else 0
                
                logger.info(f"进度: {processed_files}/{len(csv_files)} ({processed_files/len(csv_files)*100:.1f}%) "
                          f"记录: {total_records:,} 速度: {rate:.1f}文件/秒 预计剩余: {eta/60:.1f}分钟")
                
            except Exception as e:
                logger.error(f"处理批次 {i//batch_size + 1} 时出错: {e}")
                continue
        
        elapsed_time = time.time() - start_time
        logger.info(f"迁移完成！")
        logger.info(f"总处理文件: {processed_files}/{len(csv_files)}")
        logger.info(f"总插入记录: {total_records:,}")
        logger.info(f"总耗时: {elapsed_time/60:.2f} 分钟")
        logger.info(f"平均速度: {processed_files/elapsed_time:.2f} 文件/秒")

def main():
    migrator = UltraFastMigration()
    migrator.run_migration()

if __name__ == "__main__":
    main()