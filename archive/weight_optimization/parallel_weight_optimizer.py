#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行化系统权重优化器
用于科学地找到v3.1评分系统的最优权重组合

基于用户需求的三步方法论：
1. 计算全市场所有股票的v3.1指标
2. 测试不同权重向量的预测效果  
3. 迭代优化直到收敛
"""

import sqlite3
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import time

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from scoring.v3.quantitative_scorer_v3_1 import QuantitativeScorerV31
from data_adapter.database_manager import DatabaseManager

class ParallelWeightOptimizer:
    """并行化权重优化器"""
    
    def __init__(self, max_workers=None):
        """初始化优化器"""
        self.max_workers = max_workers or max(4, cpu_count() - 1)
        self.db_path = "weight_optimization_cache.db"
        self.main_db_path = "data_adapter/stock_data.db"
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.setup_database()
    
    def setup_database(self):
        """创建缓存数据库"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
        CREATE TABLE IF NOT EXISTS stock_indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL,
            date TEXT NOT NULL,
            technical REAL,
            fundamental REAL,
            performance REAL,
            sentiment REAL,
            risk_control REAL,
            market_regime REAL,
            return_1d REAL,
            return_3d REAL,
            return_5d REAL,
            return_10d REAL,
            return_20d REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(code, date)
        )
        ''')
        conn.commit()
        conn.close()
        
    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日列表"""
        conn = sqlite3.connect(self.main_db_path)
        query = """
        SELECT DISTINCT trade_date 
        FROM daily_quotes 
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date
        """
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        
        return df['trade_date'].tolist()
    
    def get_active_stocks_for_date(self, date_str: str) -> List[str]:
        """获取指定日期的活跃股票列表"""
        conn = sqlite3.connect(self.main_db_path)
        query = """
        SELECT DISTINCT dq.security_id, s.code
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id  
        WHERE dq.trade_date = ? AND s.type = 'A股'
        AND dq.volume > 0 AND dq.close > 0
        ORDER BY s.code
        """
        df = pd.read_sql_query(query, conn, params=(date_str,))
        conn.close()
        
        return df['code'].tolist()
    
    def calculate_future_returns(self, code: str, date_str: str) -> Dict[str, float]:
        """计算未来收益率"""
        conn = sqlite3.connect(self.main_db_path)
        
        # 获取当前价格
        current_query = """
        SELECT dq.close as current_close
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.code = ? AND dq.trade_date = ?
        """
        current_df = pd.read_sql_query(conn, current_query, params=(code, date_str))
        if current_df.empty:
            conn.close()
            return {}
            
        current_close = current_df.iloc[0]['current_close']
        
        # 获取未来价格数据
        future_query = """
        SELECT dq.trade_date, dq.close
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.code = ? AND dq.trade_date > ?
        ORDER BY dq.trade_date
        LIMIT 30
        """
        future_df = pd.read_sql_query(conn, future_query, params=(code, date_str))
        conn.close()
        
        if future_df.empty:
            return {}
        
        returns = {}
        for period in [1, 3, 5, 10, 20]:
            if len(future_df) >= period:
                future_close = future_df.iloc[period-1]['close']
                returns[f'return_{period}d'] = (future_close / current_close - 1) * 100
        
        return returns

def calculate_stock_features_batch(args):
    """批量计算股票特征的工作函数"""
    codes_batch, date_str, batch_id = args
    
    # 在每个进程中创建独立的评分器实例
    scorer = QuantitativeScorerV31()
    results = []
    
    print(f"🔄 进程 {batch_id}: 处理 {len(codes_batch)} 只股票 (日期: {date_str})")
    
    for i, code in enumerate(codes_batch):
        try:
            # 计算v3.1评分
            scoring_result = scorer.calculate_stock_score(code, date_str)
            if not scoring_result or not scoring_result.get('scores'):
                continue
            
            scores = scoring_result['scores']
            
            # 计算未来收益率
            optimizer = ParallelWeightOptimizer()
            future_returns = optimizer.calculate_future_returns(code, date_str)
            
            # 组装结果
            feature_dict = {
                'code': code,
                'date': date_str,
                'technical': float(scores.get('technical', 0)),
                'fundamental': float(scores.get('fundamental', 0)),
                'performance': float(scores.get('performance', 0)),
                'sentiment': float(scores.get('sentiment', 0)),
                'risk_control': float(scores.get('risk_control', 0)),
                'market_regime': float(scores.get('market_regime', 0)),
            }
            
            # 添加未来收益率
            for key, value in future_returns.items():
                feature_dict[key] = float(value)
            
            results.append(feature_dict)
            
            # 打印进度
            if (i + 1) % 100 == 0:
                print(f"  进程 {batch_id}: 已处理 {i+1}/{len(codes_batch)}")
                
        except Exception as e:
            print(f"❌ 进程 {batch_id} 处理股票 {code} 失败: {e}")
            continue
    
    print(f"✅ 进程 {batch_id}: 完成，返回 {len(results)} 条有效记录")
    return results

class ParallelWeightOptimizer:
    """并行化权重优化器"""
    
    def __init__(self, max_workers=None):
        """初始化优化器"""
        self.max_workers = max_workers or max(4, cpu_count() - 1)
        self.db_path = "weight_optimization_cache.db"
        self.main_db_path = "data_adapter/stock_data.db"
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.setup_database()
        
    def setup_database(self):
        """创建缓存数据库"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
        CREATE TABLE IF NOT EXISTS stock_indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL,
            date TEXT NOT NULL,
            technical REAL,
            fundamental REAL,
            performance REAL,
            sentiment REAL,
            risk_control REAL,
            market_regime REAL,
            return_1d REAL,
            return_3d REAL,
            return_5d REAL,
            return_10d REAL,
            return_20d REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(code, date)
        )
        ''')
        conn.commit()
        conn.close()
        
    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日列表"""
        conn = sqlite3.connect(self.main_db_path)
        query = """
        SELECT DISTINCT trade_date 
        FROM daily_quotes 
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date
        """
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        
        return df['trade_date'].tolist()
    
    def get_active_stocks_for_date(self, date_str: str) -> List[str]:
        """获取指定日期的活跃股票列表"""
        conn = sqlite3.connect(self.main_db_path)
        query = """
        SELECT DISTINCT dq.security_id, s.code
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id  
        WHERE dq.trade_date = ? AND s.type = 'A股'
        AND dq.volume > 0 AND dq.close > 0
        ORDER BY s.code
        """
        df = pd.read_sql_query(query, conn, params=(date_str,))
        conn.close()
        
        return df['code'].tolist()
        
    def calculate_future_returns(self, code: str, date_str: str) -> Dict[str, float]:
        """计算未来收益率"""
        conn = sqlite3.connect(self.main_db_path)
        
        try:
            # 获取当前价格
            current_query = """
            SELECT dq.close as current_close
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.code = ? AND dq.trade_date = ?
            """
            current_df = pd.read_sql_query(current_query, conn, params=(code, date_str))
            if current_df.empty:
                return {}
                
            current_close = current_df.iloc[0]['current_close']
            
            # 获取未来价格数据
            future_query = """
            SELECT dq.trade_date, dq.close
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.code = ? AND dq.trade_date > ?
            ORDER BY dq.trade_date
            LIMIT 30
            """
            future_df = pd.read_sql_query(future_query, conn, params=(code, date_str))
            
            if future_df.empty:
                return {}
            
            returns = {}
            for period in [1, 3, 5, 10, 20]:
                if len(future_df) >= period:
                    future_close = future_df.iloc[period-1]['close']
                    returns[f'return_{period}d'] = (future_close / current_close - 1) * 100
            
            return returns
            
        finally:
            conn.close()
            
    def process_date_parallel(self, date_str: str):
        """并行处理单个日期的数据"""
        print(f"\n📅 并行处理日期: {date_str}")
        
        # 获取活跃股票
        stock_codes = self.get_active_stocks_for_date(date_str)
        if not stock_codes:
            print(f"⚠️ 日期 {date_str} 没有找到活跃股票")
            return
        
        print(f"📊 找到 {len(stock_codes)} 只活跃股票")
        
        # 检查是否已经处理过
        conn = sqlite3.connect(self.db_path)
        existing_count = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM stock_indicators WHERE date = ?", 
            conn, params=(date_str,)
        ).iloc[0]['count']
        conn.close()
        
        if existing_count > 0:
            print(f"✅ 日期 {date_str} 已存在 {existing_count} 条记录，跳过")
            return
            
        # 分批处理 - 每个进程处理约200只股票
        batch_size = max(100, len(stock_codes) // self.max_workers)
        batches = []
        
        for i in range(0, len(stock_codes), batch_size):
            batch = stock_codes[i:i+batch_size]
            batches.append((batch, date_str, i//batch_size + 1))
        
        print(f"🚀 启动 {len(batches)} 个并行批次，每批约 {batch_size} 只股票")
        
        # 并行处理
        start_time = time.time()
        with Pool(processes=self.max_workers) as pool:
            batch_results = pool.map(calculate_stock_features_batch, batches)
        
        processing_time = time.time() - start_time
        print(f"⏱️ 并行计算耗时: {processing_time:.1f} 秒")
        
        # 合并结果
        all_results = []
        for batch_result in batch_results:
            all_results.extend(batch_result)
        
        if not all_results:
            print(f"⚠️ 日期 {date_str} 没有生成有效数据")
            return
            
        # 批量插入数据库
        print(f"💾 保存 {len(all_results)} 条记录到数据库...")
        self.batch_insert_indicators(all_results)
        
        print(f"✅ 日期 {date_str} 处理完成: {len(all_results)} 条有效记录")
        
    def batch_insert_indicators(self, data_list: List[Dict]):
        """批量插入指标数据"""
        if not data_list:
            return
            
        conn = sqlite3.connect(self.db_path)
        
        # 准备插入语句
        columns = ['code', 'date', 'technical', 'fundamental', 'performance', 
                  'sentiment', 'risk_control', 'market_regime', 
                  'return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d']
        
        placeholders = ', '.join(['?' for _ in columns])
        insert_sql = f"""
        INSERT OR REPLACE INTO stock_indicators 
        ({', '.join(columns)}) 
        VALUES ({placeholders})
        """
        
        # 批量插入
        rows = []
        for item in data_list:
            row = [item.get(col, None) for col in columns]
            rows.append(row)
        
        try:
            conn.executemany(insert_sql, rows)
            conn.commit()
            print(f"✅ 成功插入 {len(rows)} 条记录")
        except Exception as e:
            print(f"❌ 批量插入失败: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def build_comprehensive_dataset(self, start_date: str, end_date: str):
        """构建全面的数据集"""
        print(f"\n🚀 开始构建全面数据集: {start_date} 到 {end_date}")
        print(f"🔧 使用 {self.max_workers} 个并行进程")
        
        # 获取所有交易日
        trading_dates = self.get_trading_dates(start_date, end_date)
        print(f"📅 找到 {len(trading_dates)} 个交易日")
        
        # 逐日处理
        total_start = time.time()
        for i, date_str in enumerate(trading_dates):
            date_start = time.time()
            print(f"\n{'='*60}")
            print(f"📈 进度: {i+1}/{len(trading_dates)} - {date_str}")
            
            self.process_date_parallel(date_str)
            
            date_time = time.time() - date_start
            remaining_dates = len(trading_dates) - i - 1
            estimated_remaining = date_time * remaining_dates
            
            print(f"⏱️ 当前日期耗时: {date_time:.1f} 秒")
            if remaining_dates > 0:
                print(f"📊 预计剩余时间: {estimated_remaining/60:.1f} 分钟")
        
        total_time = time.time() - total_start
        print(f"\n🎉 全面数据集构建完成!")
        print(f"⏱️ 总耗时: {total_time/60:.1f} 分钟")
        
        # 显示统计信息
        self.show_dataset_stats()
    
    def show_dataset_stats(self):
        """显示数据集统计信息"""
        conn = sqlite3.connect(self.db_path)
        
        stats_query = """
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT code) as unique_stocks,
            COUNT(DISTINCT date) as unique_dates,
            MIN(date) as start_date,
            MAX(date) as end_date,
            AVG(technical) as avg_technical,
            AVG(fundamental) as avg_fundamental,
            AVG(performance) as avg_performance,
            AVG(sentiment) as avg_sentiment,
            AVG(risk_control) as avg_risk_control,
            AVG(market_regime) as avg_market_regime
        FROM stock_indicators
        """
        
        stats = pd.read_sql_query(stats_query, conn)
        conn.close()
        
        print("\n" + "="*60)
        print("📊 数据集统计信息")
        print("="*60)
        
        if not stats.empty:
            row = stats.iloc[0]
            print(f"📈 总记录数: {row['total_records']:,}")
            print(f"🏢 股票数量: {row['unique_stocks']:,}")
            print(f"📅 交易日数: {row['unique_dates']:,}")
            print(f"📅 时间范围: {row['start_date']} 至 {row['end_date']}")
            print(f"📊 平均得分:")
            print(f"  技术面: {row['avg_technical']:.4f}")
            print(f"  基本面: {row['avg_fundamental']:.4f}")
            print(f"  市场表现: {row['avg_performance']:.4f}")
            print(f"  情绪面: {row['avg_sentiment']:.4f}")
            print(f"  风险控制: {row['avg_risk_control']:.4f}")
            print(f"  市场环境: {row['avg_market_regime']:.4f}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="并行化权重优化器")
    parser.add_argument("--start-date", required=True, help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="结束日期 (YYYY-MM-DD)")  
    parser.add_argument("--workers", type=int, help="并行进程数")
    
    args = parser.parse_args()
    
    # 创建优化器
    optimizer = ParallelWeightOptimizer(max_workers=args.workers)
    
    # 构建数据集
    optimizer.build_comprehensive_dataset(args.start_date, args.end_date)

if __name__ == "__main__":
    main()