#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新权重优化缓存数据库中的市场环境得分

使用新的增强版市场环境评分系统重新计算所有历史数据的市场环境得分
"""

import sqlite3
import pandas as pd
import numpy as np
from enhanced_market_regime_scorer import EnhancedMarketRegimeScorer
import logging
from typing import List
import time
from datetime import datetime

class MarketRegimeCacheUpdater:
    """市场环境缓存数据更新器"""
    
    def __init__(self, cache_db_path: str = "weight_optimization_cache.db"):
        self.cache_db_path = cache_db_path
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # 初始化增强版市场环境评分器
        self.scorer = EnhancedMarketRegimeScorer()
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def get_all_dates_in_cache(self) -> List[str]:
        """获取缓存数据库中的所有日期"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                query = "SELECT DISTINCT date FROM stock_indicators ORDER BY date"
                dates_df = pd.read_sql_query(query, conn)
                dates = dates_df['date'].tolist()
                
            self.logger.info(f"📅 找到 {len(dates)} 个交易日需要更新市场环境得分")
            self.logger.info(f"📅 日期范围: {min(dates)} 到 {max(dates)}")
            
            return dates
            
        except Exception as e:
            self.logger.error(f"获取日期列表失败: {e}")
            return []
    
    def calculate_enhanced_market_regime_scores(self, dates: List[str]) -> dict:
        """批量计算增强版市场环境得分"""
        self.logger.info(f"🔄 开始批量计算增强版市场环境得分...")
        
        # 使用增强版评分器的批量计算方法
        scores = self.scorer.batch_calculate_market_scores(dates)
        
        self.logger.info(f"✅ 计算完成！得分统计:")
        self.logger.info(f"  最小值: {min(scores.values()):.4f}")
        self.logger.info(f"  最大值: {max(scores.values()):.4f}")  
        self.logger.info(f"  平均值: {np.mean(list(scores.values())):.4f}")
        self.logger.info(f"  标准差: {np.std(list(scores.values())):.4f}")
        self.logger.info(f"  变化范围: {max(scores.values()) - min(scores.values()):.4f}")
        
        return scores
    
    def backup_original_scores(self):
        """备份原始市场环境得分"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                # 添加备份列（如果不存在）
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) as count
                    FROM pragma_table_info('stock_indicators') 
                    WHERE name = 'market_regime_original'
                """)
                
                if cursor.fetchone()[0] == 0:
                    self.logger.info("📦 创建原始市场环境得分备份列...")
                    cursor.execute("""
                        ALTER TABLE stock_indicators 
                        ADD COLUMN market_regime_original REAL
                    """)
                    
                    # 备份原始数据
                    cursor.execute("""
                        UPDATE stock_indicators 
                        SET market_regime_original = market_regime
                        WHERE market_regime_original IS NULL
                    """)
                    
                    conn.commit()
                    self.logger.info("✅ 原始数据备份完成")
                else:
                    self.logger.info("📦 原始数据备份已存在，跳过")
                    
        except Exception as e:
            self.logger.error(f"备份原始数据失败: {e}")
            raise
    
    def update_market_regime_scores(self, new_scores: dict):
        """更新数据库中的市场环境得分"""
        try:
            self.logger.info(f"💾 开始更新数据库中的市场环境得分...")
            
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.cursor()
                
                # 批量更新
                update_count = 0
                for date, new_score in new_scores.items():
                    cursor.execute("""
                        UPDATE stock_indicators 
                        SET market_regime = ?
                        WHERE date = ?
                    """, (new_score, date))
                    
                    update_count += cursor.rowcount
                
                conn.commit()
                
            self.logger.info(f"✅ 更新完成！共更新 {update_count:,} 条记录")
            
        except Exception as e:
            self.logger.error(f"更新数据库失败: {e}")
            raise
    
    def verify_updates(self, original_sample_size: int = 10):
        """验证更新结果"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                # 随机采样验证
                query = f"""
                SELECT date, market_regime_original, market_regime 
                FROM stock_indicators 
                WHERE market_regime_original IS NOT NULL
                ORDER BY RANDOM()
                LIMIT {original_sample_size}
                """
                
                sample_df = pd.read_sql_query(query, conn)
                
            self.logger.info("🔍 验证更新结果 (随机样本):")
            for _, row in sample_df.iterrows():
                old_score = row['market_regime_original']
                new_score = row['market_regime'] 
                improvement = (new_score - old_score) / old_score * 100 if old_score > 0 else 0
                
                self.logger.info(f"  日期 {row['date']}: {old_score:.4f} → {new_score:.4f} ({improvement:+.0f}%)")
            
            # 整体统计
            with sqlite3.connect(self.cache_db_path) as conn:
                stats_query = """
                SELECT 
                    COUNT(*) as total_records,
                    MIN(market_regime_original) as old_min,
                    MAX(market_regime_original) as old_max, 
                    AVG(market_regime_original) as old_avg,
                    MIN(market_regime) as new_min,
                    MAX(market_regime) as new_max,
                    AVG(market_regime) as new_avg
                FROM stock_indicators
                WHERE market_regime_original IS NOT NULL
                """
                
                stats = pd.read_sql_query(stats_query, conn).iloc[0]
                
            self.logger.info("📊 整体更新统计:")
            self.logger.info(f"  总记录数: {stats['total_records']:,}")
            self.logger.info(f"  旧评分范围: {stats['old_min']:.4f} - {stats['old_max']:.4f}")
            self.logger.info(f"  新评分范围: {stats['new_min']:.4f} - {stats['new_max']:.4f}")  
            self.logger.info(f"  评分范围提升: {(stats['new_max'] - stats['new_min']) / (stats['old_max'] - stats['old_min']):.1f}倍")
            
        except Exception as e:
            self.logger.error(f"验证更新结果失败: {e}")
    
    def run_full_update(self):
        """执行完整的市场环境得分更新流程"""
        start_time = time.time()
        
        self.logger.info("🚀 开始市场环境得分更新流程...")
        
        try:
            # 1. 获取所有需要更新的日期
            dates = self.get_all_dates_in_cache()
            if not dates:
                self.logger.error("❌ 没有找到需要更新的日期")
                return
            
            # 2. 备份原始数据
            self.backup_original_scores()
            
            # 3. 计算新的市场环境得分
            new_scores = self.calculate_enhanced_market_regime_scores(dates)
            
            # 4. 更新数据库
            self.update_market_regime_scores(new_scores)
            
            # 5. 验证更新结果
            self.verify_updates()
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"🎉 市场环境得分更新完成！耗时: {elapsed_time:.1f} 秒")
            
        except Exception as e:
            self.logger.error(f"更新流程失败: {e}")
            raise

def main():
    """主函数"""
    updater = MarketRegimeCacheUpdater()
    updater.run_full_update()

if __name__ == "__main__":
    main()