#!/usr/bin/env python3
"""
并行挤压动量计算器 - 为权重优化提供历史数据
专门为weight_optimization_cache.db设计的高速批量计算工具
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
import sqlite3
import multiprocessing as mp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'scoring_improvements'))

from database_manager import DatabaseManager
from squeeze_momentum_calculator import SqueezeMomentumCalculator

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ParallelSqueezeMomentumCalculator:
    """并行挤压动量计算器"""
    
    def __init__(self, cache_db_path: str = None, max_workers: int = None):
        self.cache_db_path = cache_db_path or os.path.join(project_root, 'weight_optimization_cache.db')
        self.max_workers = max_workers or max(4, mp.cpu_count() - 1)
        self.db_manager = DatabaseManager()
        
        # 确保缓存数据库存在
        self._init_cache_database()
        
    def _init_cache_database(self):
        """初始化缓存数据库"""
        os.makedirs(os.path.dirname(self.cache_db_path), exist_ok=True)
        
        with sqlite3.connect(self.cache_db_path) as conn:
            cursor = conn.cursor()
            
            # 创建挤压动量缓存表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS squeeze_momentum_cache (
                    security_id INTEGER,
                    trade_date DATE,
                    stock_code TEXT,
                    
                    -- 肯特纳通道
                    kc_upper DECIMAL(10,3),
                    kc_middle DECIMAL(10,3),
                    kc_lower DECIMAL(10,3),
                    kc_width DECIMAL(10,4),
                    
                    -- 挤压状态
                    squeeze_state BOOLEAN DEFAULT 0,
                    squeeze_release BOOLEAN DEFAULT 0,
                    squeeze_intensity DECIMAL(5,3),
                    squeeze_days INTEGER DEFAULT 0,
                    recent_releases INTEGER DEFAULT 0,
                    
                    -- 动量指标
                    squeeze_momentum DECIMAL(10,4),
                    momentum_direction INTEGER DEFAULT 0,
                    momentum_strength DECIMAL(10,4),
                    momentum_acceleration DECIMAL(10,6),
                    momentum_consistency DECIMAL(5,4),
                    
                    -- 元数据
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    
                    PRIMARY KEY (security_id, trade_date)
                )
            """)
            
            # 创建索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_squeeze_cache_date 
                ON squeeze_momentum_cache(trade_date)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_squeeze_cache_code 
                ON squeeze_momentum_cache(stock_code)
            """)
            
            conn.commit()
            logger.info(f"✅ 缓存数据库初始化完成: {self.cache_db_path}")
    
    def get_securities_batch(self, start_date: str, end_date: str, batch_size: int = 500) -> List[Tuple]:
        """获取需要计算的证券批次"""
        query = """
        SELECT DISTINCT s.id, s.code, s.name
        FROM securities s
        JOIN daily_quotes dq ON s.id = dq.security_id
        WHERE s.is_active = 1 
        AND s.type = 'A股'
        AND dq.trade_date >= ?
        AND dq.trade_date <= ?
        ORDER BY s.code
        """
        
        results = self.db_manager.execute_query(query, [start_date, end_date])
        securities = [(row[0], row[1], row[2]) for row in results]
        
        # 分批返回
        batches = []
        for i in range(0, len(securities), batch_size):
            batches.append(securities[i:i + batch_size])
            
        logger.info(f"📊 找到 {len(securities)} 只股票，分为 {len(batches)} 个批次")
        return batches
    
    def calculate_batch_squeeze_momentum(self, args: Tuple) -> Dict:
        """计算一个批次的挤压动量数据"""
        batch_securities, start_date, end_date, batch_id = args
        
        process_id = os.getpid()
        logger.info(f"🚀 进程 {process_id}: 开始处理批次 {batch_id} ({len(batch_securities)} 只股票)")
        
        # 每个进程需要独立的数据库连接
        db_manager = DatabaseManager()
        calculator = SqueezeMomentumCalculator()
        
        batch_results = []
        processed_count = 0
        
        for security_id, stock_code, stock_name in batch_securities:
            try:
                # 获取OHLC数据
                ohlc_data = self._get_stock_ohlc_data(db_manager, security_id, start_date, end_date)
                if ohlc_data is None or len(ohlc_data) < 60:
                    continue
                
                # 计算挤压动量指标
                indicators = calculator.calculate_squeeze_momentum_indicators(
                    high=ohlc_data['high'],
                    low=ohlc_data['low'],
                    close=ohlc_data['close']
                )
                
                if not indicators:
                    continue
                
                # 转换为缓存记录
                cache_records = self._convert_to_cache_records(
                    security_id, stock_code, ohlc_data, indicators
                )
                batch_results.extend(cache_records)
                processed_count += 1
                
                # 每处理100只股票报告一次进度
                if processed_count % 100 == 0:
                    logger.info(f"📈 进程 {process_id}: 已处理 {processed_count}/{len(batch_securities)} 只股票")
                    
            except Exception as e:
                logger.error(f"❌ 进程 {process_id}: 处理 {stock_code} 失败: {e}")
                continue
        
        logger.info(f"✅ 进程 {process_id}: 批次 {batch_id} 完成，处理了 {processed_count} 只股票，生成 {len(batch_results)} 条记录")
        
        return {
            'batch_id': batch_id,
            'process_id': process_id,
            'processed_count': processed_count,
            'records_count': len(batch_results),
            'records': batch_results
        }
    
    def _get_stock_ohlc_data(self, db_manager: DatabaseManager, security_id: int, 
                           start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取股票OHLC数据"""
        query = """
        SELECT trade_date, open, high, low, close, volume
        FROM daily_quotes
        WHERE security_id = ? AND trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date
        """
        
        with db_manager.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=[security_id, start_date, end_date])
            
        if len(df) < 60:  # 需要至少60天数据
            return None
            
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        
        return df
    
    def _convert_to_cache_records(self, security_id: int, stock_code: str, 
                                ohlc_data: pd.DataFrame, indicators: Dict) -> List[Dict]:
        """转换指标数据为缓存记录格式"""
        records = []
        
        for i, date in enumerate(ohlc_data.index):
            date_str = date.strftime('%Y-%m-%d')
            
            def safe_get(series_or_array, index, default=None):
                try:
                    if hasattr(series_or_array, 'iloc'):
                        value = series_or_array.iloc[index]
                    else:
                        value = series_or_array[index]
                    
                    if pd.isna(value) or np.isnan(float(value)):
                        return default
                    return float(value)
                except (IndexError, KeyError, TypeError, ValueError):
                    return default
            
            record = {
                'security_id': security_id,
                'trade_date': date_str,
                'stock_code': stock_code,
                'kc_upper': safe_get(indicators['kc_upper'], i),
                'kc_middle': safe_get(indicators['kc_middle'], i),
                'kc_lower': safe_get(indicators['kc_lower'], i),
                'kc_width': safe_get(indicators['kc_width'], i),
                'squeeze_state': bool(safe_get(indicators['squeeze_state'], i, False)),
                'squeeze_release': bool(safe_get(indicators['squeeze_release'], i, False)),
                'squeeze_intensity': safe_get(indicators['squeeze_intensity'], i, 1.0),
                'squeeze_days': int(safe_get(indicators['squeeze_days'], i, 0)),
                'recent_releases': int(safe_get(indicators['recent_releases'], i, 0)),
                'squeeze_momentum': safe_get(indicators['momentum'], i, 0.0),
                'momentum_direction': int(safe_get(indicators['momentum_direction'], i, 0)),
                'momentum_strength': safe_get(indicators['momentum_strength'], i, 0.0),
                'momentum_acceleration': safe_get(indicators['momentum_acceleration'], i, 0.0),
                'momentum_consistency': safe_get(indicators['momentum_consistency'], i, 0.0)
            }
            
            records.append(record)
        
        return records
    
    def save_batch_to_cache(self, batch_result: Dict):
        """保存批次结果到缓存数据库"""
        if not batch_result['records']:
            return
        
        records = batch_result['records']
        
        # 准备插入数据
        insert_sql = """
        INSERT OR REPLACE INTO squeeze_momentum_cache (
            security_id, trade_date, stock_code,
            kc_upper, kc_middle, kc_lower, kc_width,
            squeeze_state, squeeze_release, squeeze_intensity, squeeze_days, recent_releases,
            squeeze_momentum, momentum_direction, momentum_strength, 
            momentum_acceleration, momentum_consistency
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        batch_data = []
        for record in records:
            batch_data.append((
                record['security_id'], record['trade_date'], record['stock_code'],
                record['kc_upper'], record['kc_middle'], record['kc_lower'], record['kc_width'],
                record['squeeze_state'], record['squeeze_release'], record['squeeze_intensity'],
                record['squeeze_days'], record['recent_releases'],
                record['squeeze_momentum'], record['momentum_direction'], record['momentum_strength'],
                record['momentum_acceleration'], record['momentum_consistency']
            ))
        
        # 批量插入
        with sqlite3.connect(self.cache_db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(insert_sql, batch_data)
            conn.commit()
            
        logger.info(f"💾 批次 {batch_result['batch_id']} 保存完成: {len(batch_data)} 条记录")
    
    def calculate_historical_squeeze_momentum(self, start_date: str, end_date: str, 
                                           batch_size: int = 200) -> Dict:
        """并行计算历史挤压动量数据"""
        logger.info(f"🚀 开始并行计算挤压动量数据 ({start_date} 到 {end_date})")
        logger.info(f"📊 配置: {self.max_workers} 个进程，批次大小 {batch_size}")
        
        # 获取证券批次
        security_batches = self.get_securities_batch(start_date, end_date, batch_size)
        
        # 准备参数
        task_args = [
            (batch, start_date, end_date, i+1) 
            for i, batch in enumerate(security_batches)
        ]
        
        # 统计信息
        stats = {
            'total_batches': len(task_args),
            'completed_batches': 0,
            'total_stocks_processed': 0,
            'total_records_generated': 0,
            'start_time': datetime.now(),
            'failed_batches': 0
        }
        
        logger.info(f"📈 开始处理 {stats['total_batches']} 个批次...")
        
        # 并行处理
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_batch = {
                executor.submit(self.calculate_batch_squeeze_momentum, args): args[3]
                for args in task_args
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    batch_result = future.result()
                    
                    # 保存结果到缓存
                    self.save_batch_to_cache(batch_result)
                    
                    # 更新统计
                    stats['completed_batches'] += 1
                    stats['total_stocks_processed'] += batch_result['processed_count']
                    stats['total_records_generated'] += batch_result['records_count']
                    
                    # 报告进度
                    progress = stats['completed_batches'] / stats['total_batches'] * 100
                    elapsed = datetime.now() - stats['start_time']
                    rate = stats['completed_batches'] / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
                    
                    logger.info(f"📊 进度: {stats['completed_batches']}/{stats['total_batches']} "
                              f"({progress:.1f}%)，速度: {rate:.2f} 批次/秒")
                    
                except Exception as e:
                    logger.error(f"❌ 批次 {batch_id} 处理失败: {e}")
                    stats['failed_batches'] += 1
        
        # 完成统计
        stats['end_time'] = datetime.now()
        stats['duration'] = stats['end_time'] - stats['start_time']
        
        logger.info("🎉 挤压动量历史数据计算完成！")
        logger.info(f"📊 最终统计:")
        logger.info(f"  - 总批次: {stats['total_batches']}")
        logger.info(f"  - 成功批次: {stats['completed_batches']}")
        logger.info(f"  - 失败批次: {stats['failed_batches']}")
        logger.info(f"  - 处理股票: {stats['total_stocks_processed']} 只")
        logger.info(f"  - 生成记录: {stats['total_records_generated']} 条")
        logger.info(f"  - 总耗时: {stats['duration']}")
        logger.info(f"  - 平均速度: {stats['total_stocks_processed']/stats['duration'].total_seconds():.1f} 股票/秒")
        
        return stats
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        with sqlite3.connect(self.cache_db_path) as conn:
            cursor = conn.cursor()
            
            # 总记录数
            cursor.execute("SELECT COUNT(*) FROM squeeze_momentum_cache")
            total_records = cursor.fetchone()[0]
            
            # 股票数量
            cursor.execute("SELECT COUNT(DISTINCT stock_code) FROM squeeze_momentum_cache")
            total_stocks = cursor.fetchone()[0]
            
            # 日期范围
            cursor.execute("SELECT MIN(trade_date), MAX(trade_date) FROM squeeze_momentum_cache")
            date_range = cursor.fetchone()
            
            # 数据库大小
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            db_size = cursor.fetchone()[0]
            
        return {
            'total_records': total_records,
            'total_stocks': total_stocks,
            'date_range': date_range,
            'db_size_mb': round(db_size / 1024 / 1024, 2),
            'cache_db_path': self.cache_db_path
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='并行挤压动量历史数据计算器')
    parser.add_argument('--start-date', default='2024-01-01', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2025-08-25', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--batch-size', type=int, default=200, help='批次大小')
    parser.add_argument('--max-workers', type=int, help='最大进程数')
    parser.add_argument('--cache-db-path', help='缓存数据库路径')
    parser.add_argument('--stats-only', action='store_true', help='只显示缓存统计信息')
    
    args = parser.parse_args()
    
    calculator = ParallelSqueezeMomentumCalculator(
        cache_db_path=args.cache_db_path,
        max_workers=args.max_workers
    )
    
    if args.stats_only:
        stats = calculator.get_cache_stats()
        logger.info("📊 缓存数据库统计:")
        for key, value in stats.items():
            logger.info(f"  - {key}: {value}")
    else:
        stats = calculator.calculate_historical_squeeze_momentum(
            args.start_date, 
            args.end_date, 
            args.batch_size
        )
        
        # 显示最终缓存统计
        logger.info("\n📊 缓存数据库最终统计:")
        cache_stats = calculator.get_cache_stats()
        for key, value in cache_stats.items():
            logger.info(f"  - {key}: {value}")


if __name__ == "__main__":
    main()