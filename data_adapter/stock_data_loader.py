#!/usr/bin/env python3
"""
股票数据加载器 - 从数据库加载数据供选股系统使用
"""

import time
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
        加载所有股票数据（优先使用批量模式，失败回退到逐只模式）

        Args:
            days: 加载最近多少天的数据
            security_types: 证券类型列表，默认为['A股', 'ETF_基金']
            target_date: 目标分析日期，格式'YYYY-MM-DD'，默认为当前日期

        Returns:
            {stock_code: DataFrame} 的字典
        """
        try:
            return self._load_all_stock_data_batch(days, security_types, target_date)
        except Exception as e:
            logger.warning(f"批量加载失败，回退到逐只加载: {e}")
            return self._load_all_stock_data_sequential(days, security_types, target_date)

    def _load_all_stock_data_batch(self, days: int = 120, security_types: Optional[List[str]] = None, target_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        批量加载所有股票数据 - 使用单条JOIN查询 + groupby拆分

        比逐只查询快 ~10x（单条SQL vs ~5600条SQL）
        """
        if security_types is None:
            security_types = ['A股', 'ETF_基金']

        if target_date:
            end_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        else:
            end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)

        t0 = time.time()
        logger.info(f"[批量模式] 从数据库加载 {start_date} 到 {end_date} 的数据...")

        with self.db_manager.get_connection() as conn:
            type_placeholders = ','.join(['?' for _ in security_types])

            query = f"""
            SELECT
                s.code,
                dq.trade_date as date,
                dq.open,
                dq.high,
                dq.low,
                dq.close,
                dq.volume
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.is_active = 1
                AND s.type IN ({type_placeholders})
                AND dq.trade_date >= ?
                AND dq.trade_date <= ?
            ORDER BY s.code, dq.trade_date
            """

            params = list(security_types) + [str(start_date), str(end_date)]

            t1 = time.time()
            all_df = pd.read_sql_query(query, conn, params=params, parse_dates=['date'])
            t2 = time.time()
            logger.info(f"[批量模式] SQL查询完成: {len(all_df)} 行, 耗时 {t2-t1:.2f}秒")

        if all_df.empty:
            logger.warning("[批量模式] 未查询到任何数据")
            return {}

        # 按股票代码分组
        data = {}
        for code, group_df in all_df.groupby('code'):
            if len(group_df) >= 20:
                data[code] = group_df.drop(columns=['code']).reset_index(drop=True)

        t3 = time.time()
        logger.info(f"[批量模式] 成功加载 {len(data)} 只证券的数据, 总耗时 {t3-t0:.2f}秒")
        return data

    def _load_all_stock_data_sequential(self, days: int = 120, security_types: Optional[List[str]] = None, target_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        逐只加载所有股票数据（原始方式，作为回退）
        """
        if security_types is None:
            security_types = ['A股', 'ETF_基金']

        if target_date:
            end_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        else:
            end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)

        t0 = time.time()
        logger.info(f"[逐只模式] 从数据库加载 {start_date} 到 {end_date} 的数据...")

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

            # 逐只查询所有证券的数据
            for i, security in enumerate(securities):
                security_id = security['id']
                code = security['code']

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

                if len(df) >= 20:
                    data[code] = df

                if (i + 1) % 1000 == 0:
                    logger.info(f"已加载 {i + 1} 只证券数据...")

        t1 = time.time()
        logger.info(f"[逐只模式] 成功加载 {len(data)} 只证券的数据, 总耗时 {t1-t0:.2f}秒")
        return data
    
    def load_all_stock_data_wide(self, start_date: str, end_date: str,
                                 lookback_days: int = 200,
                                 security_types: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        加载宽日期范围的股票数据（用于批量报告生成）

        一次SQL查询加载 [start_date - lookback_days, end_date] 全部数据，
        避免每个日期重复查询。

        Args:
            start_date: 报告起始日期 YYYY-MM-DD
            end_date: 报告结束日期 YYYY-MM-DD
            lookback_days: 每个日期需要的历史回看天数
            security_types: 证券类型列表

        Returns:
            {stock_code: DataFrame} 字典
        """
        if security_types is None:
            security_types = ['A股']

        data_start = datetime.strptime(start_date, '%Y-%m-%d').date() - timedelta(days=lookback_days)
        data_end = datetime.strptime(end_date, '%Y-%m-%d').date()

        t0 = time.time()
        logger.info(f"[宽范围模式] 加载 {data_start} 到 {data_end} 的数据...")

        with self.db_manager.get_connection() as conn:
            type_placeholders = ','.join(['?' for _ in security_types])

            query = f"""
            SELECT
                s.code,
                dq.trade_date as date,
                dq.open,
                dq.high,
                dq.low,
                dq.close,
                dq.volume
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.is_active = 1
                AND s.type IN ({type_placeholders})
                AND dq.trade_date >= ?
                AND dq.trade_date <= ?
            ORDER BY s.code, dq.trade_date
            """

            params = list(security_types) + [str(data_start), str(data_end)]

            t1 = time.time()
            all_df = pd.read_sql_query(query, conn, params=params, parse_dates=['date'])
            t2 = time.time()
            logger.info(f"[宽范围模式] SQL查询完成: {len(all_df)} 行, 耗时 {t2-t1:.2f}秒")

        if all_df.empty:
            logger.warning("[宽范围模式] 未查询到任何数据")
            return {}

        data = {}
        for code, group_df in all_df.groupby('code'):
            if len(group_df) >= 20:
                data[code] = group_df.drop(columns=['code']).reset_index(drop=True)

        t3 = time.time()
        logger.info(f"[宽范围模式] 成功加载 {len(data)} 只证券的数据, 总耗时 {t3-t0:.2f}秒")
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