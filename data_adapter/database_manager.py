#!/usr/bin/env python3
"""
数据库管理器
负责数据库的初始化、连接管理和基础操作
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import pandas as pd
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQLite数据库管理器"""
    
    def __init__(self, db_path: str = "data_adapter/stock_data.db"):
        """
        初始化数据库管理器
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """初始化数据库，创建表结构（如果不存在）"""
        schema_file = Path(__file__).parent / "database_schema.sql"
        
        if not schema_file.exists():
            raise FileNotFoundError(f"数据库schema文件不存在: {schema_file}")
        
        # 检查数据库是否已经存在表结构
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='securities'")
            table_exists = cursor.fetchone() is not None
            
            # 启用外键约束
            cursor.execute("PRAGMA foreign_keys = ON")
            
            # 只有当表不存在时才创建
            if not table_exists:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    schema_sql = f.read()
                cursor.executescript(schema_sql)
                conn.commit()
                logger.info(f"数据库表结构创建完成: {self.db_path}")
            else:
                logger.info(f"数据库已存在，跳过表结构创建: {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = sqlite3.connect(
            str(self.db_path),
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            timeout=30  # busy_timeout 30秒
        )
        conn.row_factory = sqlite3.Row  # 支持通过列名访问
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[sqlite3.Row]:
        """执行查询并返回结果"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params is not None:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchall()
    
    def execute_many(self, query: str, data: List[tuple]) -> int:
        """批量执行SQL语句"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, data)
            conn.commit()
            return cursor.rowcount
    
    def get_security_map(self) -> Dict[str, int]:
        """一次查询获取所有证券的 {code: id} 映射"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT code, id FROM securities")
            return {row[0]: row[1] for row in cursor.fetchall()}

    def batch_insert_securities(self, securities: List[Dict[str, str]]) -> int:
        """批量插入新证券

        Args:
            securities: [{'code': '000001', 'name': '平安银行', 'type': 'A股', 'exchange': 'SZ'}, ...]
        Returns:
            插入的记录数
        """
        query = """
        INSERT OR IGNORE INTO securities (code, name, type, exchange)
        VALUES (?, ?, ?, ?)
        """
        rows = [(s['code'], s['name'], s['type'], s.get('exchange')) for s in securities]
        return self.execute_many(query, rows)

    def insert_security(self, code: str, name: str, security_type: str,
                       exchange: Optional[str] = None) -> int:
        """插入证券基本信息"""
        query = """
        INSERT OR IGNORE INTO securities (code, name, type, exchange)
        VALUES (?, ?, ?, ?)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (code, name, security_type, exchange))
            conn.commit()

            # 获取security_id
            cursor.execute("SELECT id FROM securities WHERE code = ?", (code,))
            result = cursor.fetchone()
            return result[0] if result else None
    
    def insert_daily_quotes(self, data: List[Dict[str, Any]]) -> int:
        """批量插入日线行情数据"""
        query = """
        INSERT OR REPLACE INTO daily_quotes (
            security_id, trade_date, open, high, low, close, volume,
            price_change_pct, is_limit_up, is_limit_down
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        rows_data = []
        for item in data:
            rows_data.append((
                item['security_id'],
                item['trade_date'],
                item['open'],
                item['high'],
                item['low'],
                item['close'],
                item['volume'],
                item.get('price_change_pct', 0),
                item.get('is_limit_up', False),
                item.get('is_limit_down', False)
            ))
        
        return self.execute_many(query, rows_data)
    
    def get_latest_date(self, security_id: int) -> Optional[str]:
        """获取某个证券的最新数据日期"""
        query = """
        SELECT MAX(trade_date) as latest_date
        FROM daily_quotes
        WHERE security_id = ?
        """
        result = self.execute_query(query, (security_id,))
        if result and result[0]['latest_date']:
            return result[0]['latest_date']
        return None
    
    def get_security_data(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取指定证券在日期范围内的数据"""
        query = """
        SELECT 
            s.code,
            s.name,
            q.trade_date,
            q.open,
            q.high,
            q.low,
            q.close,
            q.volume,
            q.price_change_pct,
            q.is_limit_up,
            q.is_limit_down
        FROM securities s
        JOIN daily_quotes q ON s.id = q.security_id
        WHERE s.code = ?
            AND q.trade_date >= ?
            AND q.trade_date <= ?
        ORDER BY q.trade_date
        """
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(code, start_date, end_date))
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            return df
    
    def get_all_securities(self, security_type: Optional[str] = None) -> pd.DataFrame:
        """获取所有证券列表"""
        query = "SELECT * FROM securities WHERE is_active = 1"
        params = []
        
        if security_type:
            query += " AND type = ?"
            params.append(security_type)
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params if params else None)
    
    # 技术指标表允许的列名白名单
    ALLOWED_INDICATOR_COLUMNS = frozenset({
        'kdj_k', 'kdj_d', 'kdj_j',
        'macd_dif', 'macd_dea', 'macd_macd',
        'rsi6', 'rsi12', 'rsi24',
        'boll_upper', 'boll_middle', 'boll_lower',
        'bbi',
        'volume_ma5', 'volume_ma10', 'volume_ratio',
        'kc_upper', 'kc_middle', 'kc_lower', 'kc_width',
        'squeeze_state', 'squeeze_release', 'squeeze_intensity',
        'squeeze_days', 'recent_releases',
        'squeeze_momentum', 'momentum_direction', 'momentum_strength',
        'momentum_acceleration', 'momentum_consistency',
        # 知行线指标
        'zhixing_short_trend', 'zhixing_multi_kong',
        # 额外MA/技术指标
        'ma14', 'ma28', 'ma57', 'ma114',
        'cci_14', 'atr_14',
    })

    def update_technical_indicators(self, security_id: int, date: str, indicators: Dict[str, float]):
        """更新技术指标"""
        # 列名白名单校验，防止SQL注入
        invalid_cols = set(indicators.keys()) - self.ALLOWED_INDICATOR_COLUMNS
        if invalid_cols:
            raise ValueError(f"非法的技术指标列名: {invalid_cols}")

        columns = ['security_id', 'trade_date'] + list(indicators.keys())
        placeholders = ['?'] * len(columns)
        values = [security_id, date] + list(indicators.values())

        query = f"""
        INSERT OR REPLACE INTO technical_indicators ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        """

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, values)
            conn.commit()
    
    def log_data_update(self, update_type: str, securities_updated: int, 
                       records_added: int, status: str, duration: float):
        """记录数据更新日志"""
        query = """
        INSERT INTO data_update_log (
            update_date, update_type, securities_updated, 
            records_added, status, duration_seconds
        ) VALUES (?, ?, ?, ?, ?, ?)
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (
                datetime.now().date(),
                update_type,
                securities_updated,
                records_added,
                status,
                int(duration)
            ))
            conn.commit()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        stats = {}
        
        with self.get_connection() as conn:
            # 证券总数
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM securities WHERE is_active = 1")
            stats['total_securities'] = cursor.fetchone()['count']
            
            # 按类型统计
            cursor.execute("""
                SELECT type, COUNT(*) as count 
                FROM securities 
                WHERE is_active = 1 
                GROUP BY type
            """)
            stats['securities_by_type'] = {row['type']: row['count'] for row in cursor.fetchall()}
            
            # 数据总量
            cursor.execute("SELECT COUNT(*) as count FROM daily_quotes")
            stats['total_quotes'] = cursor.fetchone()['count']
            
            # 日期范围
            cursor.execute("SELECT MIN(trade_date) as min_date, MAX(trade_date) as max_date FROM daily_quotes")
            result = cursor.fetchone()
            stats['date_range'] = {
                'start': result['min_date'],
                'end': result['max_date']
            }
            
            # 数据库文件大小
            stats['db_size_mb'] = self.db_path.stat().st_size / 1024 / 1024
            
        return stats
    
    def optimize_database(self):
        """优化数据库性能"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # 分析表以更新统计信息
            cursor.execute("ANALYZE")
            # 清理空间
            cursor.execute("VACUUM")
            conn.commit()
        logger.info("数据库优化完成")


if __name__ == "__main__":
    # 测试数据库管理器
    db = DatabaseManager()
    
    # 获取数据库统计信息
    stats = db.get_database_stats()
    print("\n数据库统计信息:")
    print(f"证券总数: {stats['total_securities']}")
    print(f"行情记录数: {stats['total_quotes']}")
    print(f"数据库大小: {stats['db_size_mb']:.2f} MB")
    
    # 测试插入证券
    security_id = db.insert_security("000001", "平安银行", "A股", "SZ")
    print(f"\n插入证券ID: {security_id}")
    
    # 获取所有A股
    securities = db.get_all_securities("A股")
    print(f"\nA股数量: {len(securities)}")