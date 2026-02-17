#!/usr/bin/env python3
"""
数据库索引优化脚本
基于实际查询模式分析，优化数据库性能
"""

import sqlite3
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseOptimizer:
    """数据库性能优化器"""
    
    def __init__(self, db_path="data_adapter/stock_data.db"):
        self.db_path = db_path
    
    def analyze_current_performance(self):
        """分析当前性能基线"""
        logger.info("📊 分析当前性能基线...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取测试数据
        cursor.execute("SELECT code FROM securities WHERE type='A股' LIMIT 5")
        test_codes = [row[0] for row in cursor.fetchall()]
        
        baseline = {}
        
        # 测试关键查询
        test_queries = {
            'single_stock_history': (
                "SELECT * FROM daily_quotes dq JOIN securities s ON dq.security_id = s.id WHERE s.code = ? AND trade_date >= '2024-01-01'",
                (test_codes[0],)
            ),
            'market_scan_by_date': (
                "SELECT s.code, dq.close, dq.volume FROM daily_quotes dq JOIN securities s ON dq.security_id = s.id WHERE dq.trade_date = '2024-08-01' AND s.type = 'A股'",
                ()
            ),
            'limit_up_filter': (
                "SELECT s.code, dq.close FROM daily_quotes dq JOIN securities s ON dq.security_id = s.id WHERE dq.trade_date = '2024-08-01' AND dq.is_limit_up = 1",
                ()
            ),
            'volume_ranking': (
                "SELECT s.code, dq.volume FROM daily_quotes dq JOIN securities s ON dq.security_id = s.id WHERE dq.trade_date = '2024-08-01' ORDER BY dq.volume DESC LIMIT 100",
                ()
            )
        }
        
        for query_name, (query, params) in test_queries.items():
            times = []
            for _ in range(3):  # 运行3次取平均
                start_time = time.time()
                cursor.execute(query, params)
                cursor.fetchall()
                times.append(time.time() - start_time)
            
            baseline[query_name] = sum(times) / len(times)
            logger.info(f"  {query_name}: {baseline[query_name]:.4f}秒")
        
        conn.close()
        return baseline
    
    def create_optimized_indexes(self):
        """创建优化的索引"""
        logger.info("🔧 创建优化索引...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 优化索引策略
        optimized_indexes = [
            # 1. 针对市场扫描的复合索引（最重要的优化）
            ("idx_daily_quotes_date_type_optimized", 
             """CREATE INDEX IF NOT EXISTS idx_daily_quotes_date_type_optimized 
                ON daily_quotes(trade_date, security_id) 
                WHERE trade_date >= '2024-01-01'"""),
            
            # 2. 针对涨跌停筛选的覆盖索引
            ("idx_daily_quotes_limit_covering",
             """CREATE INDEX IF NOT EXISTS idx_daily_quotes_limit_covering 
                ON daily_quotes(trade_date, is_limit_up, is_limit_down, security_id, close, price_change_pct)"""),
            
            # 3. 针对成交量排序的专用索引
            ("idx_daily_quotes_volume_ranking",
             """CREATE INDEX IF NOT EXISTS idx_daily_quotes_volume_ranking 
                ON daily_quotes(trade_date, volume DESC, security_id)"""),
            
            # 4. 针对技术分析的时间序列索引
            ("idx_daily_quotes_timeseries",
             """CREATE INDEX IF NOT EXISTS idx_daily_quotes_timeseries 
                ON daily_quotes(security_id, trade_date, close, volume, high, low)"""),
            
            # 5. 证券类型优化索引
            ("idx_securities_type_active",
             """CREATE INDEX IF NOT EXISTS idx_securities_type_active 
                ON securities(type, is_active, id) WHERE is_active = 1"""),
            
            # 6. 针对最新数据查询的部分索引
            ("idx_daily_quotes_recent",
             """CREATE INDEX IF NOT EXISTS idx_daily_quotes_recent 
                ON daily_quotes(security_id, trade_date DESC, close, volume) 
                WHERE trade_date >= '2024-01-01'"""),
            
            # 7. 针对价格变化分析的索引
            ("idx_daily_quotes_price_analysis",
             """CREATE INDEX IF NOT EXISTS idx_daily_quotes_price_analysis 
                ON daily_quotes(security_id, trade_date, price_change_pct, close)""")
        ]
        
        for index_name, index_sql in optimized_indexes:
            try:
                logger.info(f"  创建索引: {index_name}")
                cursor.execute(index_sql)
                conn.commit()
            except Exception as e:
                logger.warning(f"  索引创建失败 {index_name}: {e}")
        
        conn.close()
        logger.info("✅ 优化索引创建完成")
    
    def create_materialized_views(self):
        """创建物化视图（模拟）用于常用查询"""
        logger.info("📋 创建辅助表用于常用查询...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 1. 每日市场概览表（加速市场扫描）
        logger.info("  创建每日市场概览表...")
        cursor.executescript("""
            DROP TABLE IF EXISTS daily_market_overview;
            
            CREATE TABLE daily_market_overview AS
            SELECT 
                dq.trade_date,
                s.code,
                s.name,
                s.type,
                dq.open,
                dq.high,
                dq.low,
                dq.close,
                dq.volume,
                dq.price_change_pct,
                dq.is_limit_up,
                dq.is_limit_down,
                RANK() OVER (PARTITION BY dq.trade_date ORDER BY dq.volume DESC) as volume_rank
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.is_active = 1 AND dq.trade_date >= '2024-01-01';
            
            CREATE INDEX idx_market_overview_date ON daily_market_overview(trade_date);
            CREATE INDEX idx_market_overview_date_type ON daily_market_overview(trade_date, type);
            CREATE INDEX idx_market_overview_limits ON daily_market_overview(trade_date, is_limit_up, is_limit_down);
            CREATE INDEX idx_market_overview_volume ON daily_market_overview(trade_date, volume_rank);
        """)
        
        # 2. 股票最新状态表
        logger.info("  创建股票最新状态表...")
        cursor.executescript("""
            DROP TABLE IF EXISTS latest_stock_status;
            
            CREATE TABLE latest_stock_status AS
            SELECT 
                s.code,
                s.name,
                s.type,
                dq.trade_date as last_trade_date,
                dq.close as last_close,
                dq.volume as last_volume,
                dq.price_change_pct as last_change_pct
            FROM securities s
            JOIN daily_quotes dq ON s.id = dq.security_id
            WHERE s.is_active = 1 
                AND dq.trade_date = (
                    SELECT MAX(trade_date) 
                    FROM daily_quotes 
                    WHERE security_id = s.id
                );
            
            CREATE INDEX idx_latest_status_code ON latest_stock_status(code);
            CREATE INDEX idx_latest_status_type ON latest_stock_status(type);
        """)
        
        conn.commit()
        conn.close()
        logger.info("✅ 辅助表创建完成")
    
    def optimize_database_settings(self):
        """优化数据库设置"""
        logger.info("⚙️ 优化数据库设置...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # SQLite性能优化设置
        optimizations = [
            "PRAGMA journal_mode = WAL",        # Write-Ahead Logging
            "PRAGMA synchronous = NORMAL",       # 平衡性能和安全性
            "PRAGMA cache_size = 10000",         # 增加缓存大小
            "PRAGMA temp_store = MEMORY",        # 临时表存储在内存
            "PRAGMA mmap_size = 268435456",      # 256MB内存映射
            "PRAGMA optimize"                    # 优化查询计划
        ]
        
        for optimization in optimizations:
            try:
                cursor.execute(optimization)
                logger.info(f"  应用: {optimization}")
            except Exception as e:
                logger.warning(f"  优化失败: {optimization}, {e}")
        
        conn.close()
        logger.info("✅ 数据库设置优化完成")
    
    def benchmark_optimized_performance(self):
        """测试优化后的性能"""
        logger.info("🚀 测试优化后性能...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取测试数据
        cursor.execute("SELECT code FROM securities WHERE type='A股' LIMIT 5")
        test_codes = [row[0] for row in cursor.fetchall()]
        
        # 优化后的查询（使用新索引和辅助表）
        optimized_queries = {
            'single_stock_history': (
                "SELECT * FROM daily_quotes dq JOIN securities s ON dq.security_id = s.id WHERE s.code = ? AND trade_date >= '2024-01-01' ORDER BY trade_date",
                (test_codes[0],)
            ),
            'market_scan_optimized': (
                "SELECT code, close, volume FROM daily_market_overview WHERE trade_date = '2024-08-01' AND type = 'A股'",
                ()
            ),
            'limit_up_optimized': (
                "SELECT code, close FROM daily_market_overview WHERE trade_date = '2024-08-01' AND is_limit_up = 1",
                ()
            ),
            'volume_ranking_optimized': (
                "SELECT code, volume FROM daily_market_overview WHERE trade_date = '2024-08-01' AND volume_rank <= 100 ORDER BY volume_rank",
                ()
            ),
            'latest_stock_status': (
                "SELECT code, last_close, last_volume FROM latest_stock_status WHERE type = 'A股' ORDER BY last_volume DESC LIMIT 100",
                ()
            )
        }
        
        optimized_performance = {}
        for query_name, (query, params) in optimized_queries.items():
            times = []
            for _ in range(3):
                start_time = time.time()
                cursor.execute(query, params)
                cursor.fetchall()
                times.append(time.time() - start_time)
            
            optimized_performance[query_name] = sum(times) / len(times)
            logger.info(f"  {query_name}: {optimized_performance[query_name]:.4f}秒")
        
        conn.close()
        return optimized_performance
    
    def full_optimization(self):
        """执行完整的数据库优化"""
        logger.info("🎯 开始数据库全面优化")
        
        # 1. 基线性能测试
        baseline = self.analyze_current_performance()
        
        # 2. 创建优化索引
        self.create_optimized_indexes()
        
        # 3. 创建辅助表
        self.create_materialized_views()
        
        # 4. 优化数据库设置
        self.optimize_database_settings()
        
        # 5. 测试优化后性能
        optimized = self.benchmark_optimized_performance()
        
        # 6. 性能对比报告
        logger.info("\n📊 性能优化报告")
        logger.info("=" * 50)
        
        common_queries = set(baseline.keys()) & set(optimized.keys())
        
        for query in common_queries:
            baseline_time = baseline[query]
            optimized_time = optimized.get(f"{query}_optimized", optimized.get(query, baseline_time))
            
            if optimized_time < baseline_time:
                improvement = (baseline_time - optimized_time) / baseline_time * 100
                logger.info(f"✅ {query}: {baseline_time:.4f}s → {optimized_time:.4f}s (提升 {improvement:.1f}%)")
            else:
                degradation = (optimized_time - baseline_time) / baseline_time * 100
                logger.info(f"⚠️ {query}: {baseline_time:.4f}s → {optimized_time:.4f}s (降低 {degradation:.1f}%)")
        
        # 新增查询性能
        new_queries = set(optimized.keys()) - set(baseline.keys())
        if new_queries:
            logger.info("\n🆕 新增优化查询:")
            for query in new_queries:
                logger.info(f"  {query}: {optimized[query]:.4f}秒")
        
        logger.info("\n🎉 数据库优化完成！")


def main():
    """主函数"""
    optimizer = DatabaseOptimizer()
    optimizer.full_optimization()

if __name__ == "__main__":
    main()