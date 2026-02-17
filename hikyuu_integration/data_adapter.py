#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HikyuuStyleDataAdapter - 数据适配器

将StockTradebyZ的SQLite数据库适配为Hikyuu风格的数据接口
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

from data_adapter.database_manager import DatabaseManager
from .query import Query
from .kdata import KData
from .stock import Stock
from .cache_manager import SmartCacheManager

logger = logging.getLogger(__name__)


class HikyuuStyleDataAdapter:
    """
    Hikyuu风格数据适配器

    将SQLite数据库适配为Hikyuu风格的API

    核心功能:
    - 提供get_stock()方法获取股票对象
    - 提供get_kdata()方法获取K线数据
    - 支持数据预加载和缓存优化
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None, cache_capacity: int = 1000):
        """
        初始化数据适配器

        参数:
            db_manager: 数据库管理器，如果为None则自动创建
            cache_capacity: 缓存容量（默认1000条记录）
        """
        self.db = db_manager or DatabaseManager()
        self.cache = SmartCacheManager(capacity=cache_capacity)
        self._stock_info_cache = {}
        self._trading_dates_cache = None

        logger.info(f"✅ HikyuuStyleDataAdapter initialized (cache capacity: {cache_capacity})")

    def get_stock(self, code: str) -> Stock:
        """
        获取股票对象 (类似Hikyuu的sm['sh000001'])

        参数:
            code: 股票代码 (如 '000001', '600000')

        返回:
            Stock对象

        用法:
            stock = adapter.get_stock('000001')
            kdata = stock.get_kdata(Query(-150))
        """
        # 从缓存获取股票信息
        if code not in self._stock_info_cache:
            info = self._fetch_stock_info(code)
            self._stock_info_cache[code] = info
        else:
            info = self._stock_info_cache[code]

        return Stock(code, self, info)

    def get_kdata(self, code: str, query: Query) -> KData:
        """
        获取K线数据 (类似Hikyuu的stock.get_kdata(Query(-150)))

        参数:
            code: 股票代码
            query: 查询对象

        返回:
            KData对象，包含K线和技术指标数据

        用法:
            kdata = adapter.get_kdata('000001', Query(-150))
            kdata = adapter.get_kdata('000001', Query(start='2024-01-01'))
        """
        # 尝试从智能缓存获取数据
        if query.start_date and query.end_date:
            cached_data = self.cache.find_matching_cache(code, query.start_date, query.end_date)
            if cached_data is not None and not cached_data.empty:
                logger.debug(f"Cache hit for {code} [{query.start_date}→{query.end_date}]")
                return KData(code, cached_data)

        # 构造SQL查询
        sql, params = self._build_kdata_query(code, query)

        # 执行查询
        try:
            result = self.db.execute_query(sql, params)

            if not result:
                logger.warning(f"No data found for {code} with query {query}")
                # 返回空KData
                return KData(code, pd.DataFrame())

            # 转换为DataFrame
            columns = self._get_kdata_columns()
            df = pd.DataFrame(result, columns=columns)

            # 创建KData对象
            return KData(code, df)

        except Exception as e:
            logger.error(f"Error fetching kdata for {code}: {e}")
            return KData(code, pd.DataFrame())

    def preload_data(self,
                     stock_list: List[str],
                     start_date: str,
                     end_date: str):
        """
        批量预加载数据到缓存 (性能优化)

        参数:
            stock_list: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        用法:
            adapter.preload_data(['000001', '000002'], '2024-01-01', '2025-09-30')
            # 后续get_kdata会直接从缓存读取，速度快10-20倍
        """
        logger.info(f"Preloading data for {len(stock_list)} stocks from {start_date} to {end_date}")

        # 批量查询所有数据
        placeholders = ','.join('?' * len(stock_list))

        sql = f"""
            SELECT s.code, dq.trade_date,
                   dq.open, dq.high, dq.low, dq.close, dq.volume, dq.amount,
                   dq.ma5, dq.ma10, dq.ma20, dq.ma60,
                   ti.rsi6, ti.rsi12, ti.rsi24,
                   ti.macd_dif, ti.macd_dea, ti.macd_macd,
                   ti.kdj_k, ti.kdj_d, ti.kdj_j,
                   ti.bbi
            FROM securities s
            JOIN daily_quotes dq ON s.id = dq.security_id
            LEFT JOIN technical_indicators ti ON s.id = ti.security_id AND dq.trade_date = ti.trade_date
            WHERE s.code IN ({placeholders})
            AND dq.trade_date BETWEEN ? AND ?
            ORDER BY s.code, dq.trade_date
        """

        params = stock_list + [start_date, end_date]

        try:
            result = self.db.execute_query(sql, params)

            if result:
                # 转换为DataFrame并按股票代码分组缓存
                columns = ['code', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'amount',
                          'ma5', 'ma10', 'ma20', 'ma60',
                          'rsi6', 'rsi12', 'rsi24',
                          'macd_dif', 'macd_dea', 'macd_macd',
                          'kdj_k', 'kdj_d', 'kdj_j', 'bbi']

                df = pd.DataFrame(result, columns=columns)

                # 按股票代码分组缓存（使用SmartCacheManager）
                for code in stock_list:
                    stock_data = df[df['code'] == code].copy()
                    if not stock_data.empty:
                        # 删除code列（KData不需要）
                        stock_data = stock_data.drop(columns=['code'])

                        cache_key = f"{code}_{start_date}_{end_date}"
                        self.cache.put(cache_key, stock_data, preload_info={
                            'stock_code': code,
                            'start_date': start_date,
                            'end_date': end_date
                        })

                logger.info(f"✅ Preloaded {len(df)} records into cache")
            else:
                logger.warning("No data to preload")

        except Exception as e:
            logger.error(f"Error preloading data: {e}")

    def stock_exists(self, code: str) -> bool:
        """检查股票是否存在"""
        sql = "SELECT 1 FROM securities WHERE code = ? LIMIT 1"
        result = self.db.execute_query(sql, [code])
        return bool(result)

    def get_market_value(self, code: str, date: str) -> Optional[float]:
        """获取市值"""
        sql = """
            SELECT db.market_cap
            FROM securities s
            JOIN daily_quotes dq ON s.id = dq.security_id
            JOIN daily_basic db ON dq.id = db.quote_id
            WHERE s.code = ? AND dq.trade_date = ?
        """
        result = self.db.execute_query(sql, [code, date])
        return float(result[0][0]) / 100000000 if result else None  # 转换为亿元

    def get_pe_ratio(self, code: str, date: str) -> Optional[float]:
        """获取市盈率"""
        sql = """
            SELECT db.pe_ttm
            FROM securities s
            JOIN daily_quotes dq ON s.id = dq.security_id
            JOIN daily_basic db ON dq.id = db.quote_id
            WHERE s.code = ? AND dq.trade_date = ?
        """
        result = self.db.execute_query(sql, [code, date])
        return float(result[0][0]) if result else None

    def get_pb_ratio(self, code: str, date: str) -> Optional[float]:
        """获取市净率"""
        sql = """
            SELECT db.pb
            FROM securities s
            JOIN daily_quotes dq ON s.id = dq.security_id
            JOIN daily_basic db ON dq.id = db.quote_id
            WHERE s.code = ? AND dq.trade_date = ?
        """
        result = self.db.execute_query(sql, [code, date])
        return float(result[0][0]) if result else None

    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """
        获取交易日期列表

        参数:
            start_date: 开始日期
            end_date: 结束日期

        返回:
            交易日期列表
        """
        sql = """
            SELECT DISTINCT trade_date
            FROM daily_quotes
            WHERE trade_date BETWEEN ? AND ?
            ORDER BY trade_date
        """
        result = self.db.execute_query(sql, [start_date, end_date])
        return [row[0] for row in result] if result else []

    def get_all_stocks(self, stock_type: str = 'A股') -> List[str]:
        """
        获取所有股票代码

        参数:
            stock_type: 股票类型 ('A股', 'ETF_基金'等)

        返回:
            股票代码列表
        """
        sql = """
            SELECT code
            FROM securities
            WHERE type = ? AND is_active = 1
            ORDER BY code
        """
        result = self.db.execute_query(sql, [stock_type])
        return [row[0] for row in result] if result else []

    # ==================== 私有方法 ====================

    def _fetch_stock_info(self, code: str) -> Dict:
        """获取股票基本信息"""
        sql = """
            SELECT s.name, s.type, s.exchange, s.industry, s.area, s.list_date
            FROM securities s
            WHERE s.code = ?
        """
        result = self.db.execute_query(sql, [code])

        if result:
            return {
                'name': result[0][0],
                'type': result[0][1],
                'exchange': result[0][2],
                'industry': result[0][3],
                'area': result[0][4],
                'list_date': result[0][5]
            }
        return {}

    def _build_kdata_query(self, code: str, query: Query) -> tuple:
        """
        构造K线数据查询SQL

        返回:
            (sql, params) 元组
        """
        base_sql = """
            SELECT dq.trade_date,
                   dq.open, dq.high, dq.low, dq.close, dq.volume, dq.amount,
                   dq.ma5, dq.ma10, dq.ma20, dq.ma60,
                   ti.rsi6, ti.rsi12, ti.rsi24,
                   ti.macd_dif, ti.macd_dea, ti.macd_macd,
                   ti.kdj_k, ti.kdj_d, ti.kdj_j,
                   ti.bbi,
                   ti.boll_upper, ti.boll_middle, ti.boll_lower
            FROM securities s
            JOIN daily_quotes dq ON s.id = dq.security_id
            LEFT JOIN technical_indicators ti ON s.id = ti.security_id AND dq.trade_date = ti.trade_date
            WHERE s.code = ?
        """

        params = [code]

        # 根据Query构造条件
        if query.is_recent_days():
            # 最近N天
            n_days = query.get_days_count()
            base_sql += " ORDER BY dq.trade_date DESC LIMIT ?"
            params.append(n_days)
        elif query.start_date and query.end_date:
            # 日期区间
            base_sql += " AND dq.trade_date BETWEEN ? AND ? ORDER BY dq.trade_date"
            params.extend([query.start_date, query.end_date])
        elif query.start_date:
            # 从某日期开始
            base_sql += " AND dq.trade_date >= ? ORDER BY dq.trade_date"
            params.append(query.start_date)
        else:
            # 全部数据
            base_sql += " ORDER BY dq.trade_date"

        return base_sql, params

    def _get_kdata_columns(self) -> List[str]:
        """获取K线数据列名"""
        return [
            'trade_date',
            'open', 'high', 'low', 'close', 'volume', 'amount',
            'ma5', 'ma10', 'ma20', 'ma60',
            'rsi6', 'rsi12', 'rsi24',
            'macd_dif', 'macd_dea', 'macd_macd',
            'kdj_k', 'kdj_d', 'kdj_j',
            'bbi',
            'boll_upper', 'boll_middle', 'boll_lower'
        ]

    def get_cache_stats(self) -> Dict:
        """
        获取缓存统计信息

        返回:
            缓存统计字典，包含命中率、大小等信息
        """
        return self.cache.get_stats()

    def print_cache_stats(self):
        """打印缓存统计信息"""
        self.cache.print_stats()

    def clear_cache(self):
        """清空所有缓存"""
        self.cache.clear()
        self._stock_info_cache.clear()
        self._trading_dates_cache = None
        logger.info("✅ All caches cleared")

    def __repr__(self):
        cache_stats = self.cache.get_stats()
        return (f"HikyuuStyleDataAdapter("
                f"cached_stocks={len(self._stock_info_cache)}, "
                f"kdata_cache={cache_stats['size']}/{cache_stats['capacity']}, "
                f"hit_rate={cache_stats['hit_rate']:.1f}%)")
