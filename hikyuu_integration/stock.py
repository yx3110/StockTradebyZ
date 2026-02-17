#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock股票对象 - Hikyuu风格API

模仿Hikyuu的Stock对象
"""

from typing import TYPE_CHECKING, Optional, Dict

if TYPE_CHECKING:
    from .data_adapter import HikyuuStyleDataAdapter
    from .query import Query
    from .kdata import KData


class Stock:
    """
    股票对象

    提供类似Hikyuu的API访问股票数据

    属性:
        code: 股票代码
        name: 股票名称
        type: 股票类型 (A股, ETF等)
        exchange: 交易所 (SH, SZ)
    """

    def __init__(self,
                 code: str,
                 adapter: 'HikyuuStyleDataAdapter',
                 info: Optional[Dict] = None):
        """
        初始化股票对象

        参数:
            code: 股票代码
            adapter: 数据适配器
            info: 股票基本信息字典 (可选)
        """
        self.code = code
        self._adapter = adapter
        self._info = info or {}

        # 从info提取基本属性
        self.name = self._info.get('name', '')
        self.type = self._info.get('type', 'A股')
        self.exchange = self._info.get('exchange', '')
        self.industry = self._info.get('industry', '')
        self.list_date = self._info.get('list_date', '')

    def get_kdata(self, query: 'Query') -> 'KData':
        """
        获取K线数据

        参数:
            query: 查询对象

        返回:
            KData对象

        用法:
            stock = Stock('000001', adapter)
            kdata = stock.get_kdata(Query(-150))  # 最近150天
        """
        return self._adapter.get_kdata(self.code, query)

    def get_market_value(self, date: str) -> Optional[float]:
        """
        获取指定日期的市值

        参数:
            date: 日期字符串 (YYYY-MM-DD)

        返回:
            市值（亿元），如果不存在返回None
        """
        return self._adapter.get_market_value(self.code, date)

    def get_pe_ratio(self, date: str) -> Optional[float]:
        """
        获取指定日期的市盈率(PE)

        参数:
            date: 日期字符串

        返回:
            市盈率，如果不存在返回None
        """
        return self._adapter.get_pe_ratio(self.code, date)

    def get_pb_ratio(self, date: str) -> Optional[float]:
        """
        获取指定日期的市净率(PB)

        参数:
            date: 日期字符串

        返回:
            市净率，如果不存在返回None
        """
        return self._adapter.get_pb_ratio(self.code, date)

    def is_valid(self) -> bool:
        """检查股票是否有效（是否存在于数据库）"""
        return self._adapter.stock_exists(self.code)

    def __repr__(self):
        return f"Stock(code={self.code}, name={self.name}, exchange={self.exchange})"

    def __str__(self):
        return f"{self.code}({self.name})"
