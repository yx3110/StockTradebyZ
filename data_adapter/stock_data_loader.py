#!/usr/bin/env python3
"""
股票数据加载器 - 从数据库加载数据供选股系统使用
"""

import pandas as pd
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from .database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class StockDataLoader:
    """股票数据加载器"""
    
    def __init__(self, db_path: str = "data_adapter/stock_data.db"):
        """初始化数据加载器"""
        self.db_manager = DatabaseManager(db_path)
    
    def load_all_stock_data(self, days: int = 120, security_types: Optional[List[str]] = None, target_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        加载所有股票数据
        
        Args:
            days: 加载最近多少天的数据
            security_types: 证券类型列表，默认为['A股', 'ETF_基金']
            target_date: 目标分析日期，格式'YYYY-MM-DD'，默认为当前日期
            
        Returns:
            {stock_code: DataFrame} 的字典
        """
        if security_types is None:
            security_types = ['A股', 'ETF_基金']
        
        if target_date:
            end_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        else:
            end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"从数据库加载 {start_date} 到 {end_date} 的数据...")
        
        data = {}
        
        with self.db_manager.get_connection() as conn:
            # 获取所有活跃的证券
            type_placeholders = ','.join(['?' for _ in security_types])
            query = f"""
            SELECT id, code, name, type 
            FROM securities 
            WHERE is_active = 1 AND type IN ({type_placeholders})
            """
            cursor = conn.cursor()
            cursor.execute(query, security_types)
            securities = cursor.fetchall()
            
            logger.info(f"找到 {len(securities)} 只证券")
            
            # 批量查询所有证券的数据
            for i, security in enumerate(securities):
                security_id = security['id']
                code = security['code']
                
                # 查询该证券的日线数据
                query = """
                SELECT 
                    trade_date as date,
                    open,
                    high,
                    low,
                    close,
                    volume
                FROM daily_quotes
                WHERE security_id = ?
                    AND trade_date >= ?
                    AND trade_date <= ?
                ORDER BY trade_date
                """
                
                df = pd.read_sql_query(
                    query, 
                    conn, 
                    params=(security_id, start_date, end_date),
                    parse_dates=['date']
                )
                
                if len(df) >= 20:  # 至少需要20天数据进行技术指标计算
                    data[code] = df
                
                if (i + 1) % 1000 == 0:
                    logger.info(f"已加载 {i + 1} 只证券数据...")
        
        logger.info(f"成功加载 {len(data)} 只证券的数据")
        return data
    
    def load_securities_info(self) -> Dict[str, Dict]:
        """加载证券基本信息"""
        securities_info = {}
        
        with self.db_manager.get_connection() as conn:
            query = """
            SELECT code, name, type, exchange, industry, area, CAST(list_date AS TEXT) as list_date
            FROM securities
            WHERE is_active = 1
            """
            cursor = conn.cursor()
            cursor.execute(query)
            
            for row in cursor.fetchall():
                code = row['code']
                securities_info[code] = {
                    'name': row['name'],
                    'type': row['type'],
                    'market': row['exchange'] or '未知',
                    'list_date': row['list_date'] or '未知',
                    'industry': row['industry'] or '未知',
                    'area': row['area'] or '未知'
                }
        
        return securities_info
    
    def load_stock_data_by_code(self, code: str, days: int = 120, target_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        加载单只股票的数据
        
        Args:
            code: 股票代码
            days: 加载最近多少天的数据
            target_date: 目标分析日期，格式'YYYY-MM-DD'，默认为当前日期
            
        Returns:
            DataFrame或None
        """
        if target_date:
            end_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        else:
            end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        with self.db_manager.get_connection() as conn:
            # 获取证券ID
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM securities WHERE code = ?", (code,))
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"未找到股票代码: {code}")
                return None
            
            security_id = result['id']
            
            # 查询日线数据
            query = """
            SELECT 
                trade_date as date,
                open,
                high,
                low,
                close,
                volume
            FROM daily_quotes
            WHERE security_id = ?
                AND trade_date >= ?
                AND trade_date <= ?
            ORDER BY trade_date
            """
            
            df = pd.read_sql_query(
                query, 
                conn, 
                params=(security_id, start_date, end_date),
                parse_dates=['date']
            )
            
            return df if not df.empty else None
    
    def get_latest_trading_date(self) -> Optional[datetime]:
        """获取数据库中最新的交易日期"""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(trade_date) as latest_date FROM daily_quotes")
            result = cursor.fetchone()
            
            if result and result['latest_date']:
                return pd.to_datetime(result['latest_date'])
            
            return None
    
    def load_technical_indicators(self, security_id: int, start_date: str, end_date: str) -> pd.DataFrame:
        """加载技术指标数据"""
        with self.db_manager.get_connection() as conn:
            query = """
            SELECT * FROM technical_indicators
            WHERE security_id = ?
                AND trade_date >= ?
                AND trade_date <= ?
            ORDER BY trade_date
            """
            
            df = pd.read_sql_query(
                query,
                conn,
                params=(security_id, start_date, end_date),
                parse_dates=['trade_date']
            )
            
            return df