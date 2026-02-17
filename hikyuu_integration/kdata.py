#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KData K线数据对象 - Hikyuu风格API

模仿Hikyuu的KData对象，封装K线数据
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from datetime import datetime


class KData:
    """
    K线数据对象

    提供类似Hikyuu的API访问K线数据

    属性:
        datetime: 日期时间列表
        open: 开盘价数组
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        volume: 成交量数组
        amount: 成交额数组 (可选)
    """

    def __init__(self, stock_code: str, data: pd.DataFrame):
        """
        初始化K线数据

        参数:
            stock_code: 股票代码
            data: pandas DataFrame，包含K线数据
                  必须列: trade_date, open, high, low, close, volume
                  可选列: amount, 技术指标等
        """
        self.stock_code = stock_code
        self._data = data.copy()

        # 确保按日期排序
        if 'trade_date' in self._data.columns:
            self._data = self._data.sort_values('trade_date').reset_index(drop=True)

        # 提取核心数据为数组（性能优化）
        self._datetime_list = self._data['trade_date'].tolist() if 'trade_date' in self._data.columns else []
        self._open = self._data['open'].values if 'open' in self._data.columns else np.array([])
        self._high = self._data['high'].values if 'high' in self._data.columns else np.array([])
        self._low = self._data['low'].values if 'low' in self._data.columns else np.array([])
        self._close = self._data['close'].values if 'close' in self._data.columns else np.array([])
        self._volume = self._data['volume'].values if 'volume' in self._data.columns else np.array([])
        self._amount = self._data['amount'].values if 'amount' in self._data.columns else np.array([])

        # 缓存技术指标
        self._indicators_cache = {}

    def __len__(self) -> int:
        """K线数据长度"""
        return len(self._data)

    def __getitem__(self, index: int) -> Dict:
        """
        通过索引访问K线数据

        用法:
            kdata[0]  # 第一根K线
            kdata[-1] # 最后一根K线
        """
        if index < 0:
            index = len(self) + index

        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")

        row = self._data.iloc[index]
        return {
            'datetime': row.get('trade_date'),
            'open': row.get('open'),
            'high': row.get('high'),
            'low': row.get('low'),
            'close': row.get('close'),
            'volume': row.get('volume'),
            'amount': row.get('amount', 0)
        }

    @property
    def datetime(self) -> List:
        """日期时间列表"""
        return self._datetime_list

    @property
    def open(self) -> np.ndarray:
        """开盘价数组"""
        return self._open

    @property
    def high(self) -> np.ndarray:
        """最高价数组"""
        return self._high

    @property
    def low(self) -> np.ndarray:
        """最低价数组"""
        return self._low

    @property
    def close(self) -> np.ndarray:
        """收盘价数组"""
        return self._close

    @property
    def volume(self) -> np.ndarray:
        """成交量数组"""
        return self._volume

    @property
    def amount(self) -> np.ndarray:
        """成交额数组"""
        return self._amount

    def get_datetime(self, index: int) -> str:
        """
        获取指定索引的日期

        参数:
            index: 索引位置

        返回:
            日期字符串 (YYYY-MM-DD)
        """
        if index < 0:
            index = len(self) + index
        return self._datetime_list[index]

    def get_close(self, date: str) -> Optional[float]:
        """
        获取指定日期的收盘价

        参数:
            date: 日期字符串 (YYYY-MM-DD)

        返回:
            收盘价，如果日期不存在返回None
        """
        try:
            idx = self._datetime_list.index(date)
            return float(self._close[idx])
        except (ValueError, IndexError):
            return None

    def get_datetime_list(self) -> List[str]:
        """获取所有日期列表"""
        return self._datetime_list.copy()

    def get_indicator(self, name: str, **params) -> Optional[np.ndarray]:
        """
        获取技术指标数据

        参数:
            name: 指标名称 (MA, EMA, RSI, MACD, KDJ, BBI等)
            **params: 指标参数 (如 n=20 表示20日均线)

        返回:
            指标数据数组，如果不存在返回None

        用法:
            ma20 = kdata.get_indicator('MA', n=20)
            bbi = kdata.get_indicator('BBI')
            kdj_k = kdata.get_indicator('KDJ_K', n=9)
        """
        # 构造缓存键
        cache_key = f"{name}_{params}" if params else name

        if cache_key in self._indicators_cache:
            return self._indicators_cache[cache_key]

        # 从DataFrame查找指标列
        indicator_column = self._find_indicator_column(name, params)

        if indicator_column and indicator_column in self._data.columns:
            indicator_data = self._data[indicator_column].values
            self._indicators_cache[cache_key] = indicator_data
            return indicator_data

        return None

    def _find_indicator_column(self, name: str, params: Dict) -> Optional[str]:
        """
        查找技术指标对应的数据列名

        参数:
            name: 指标名称
            params: 参数字典

        返回:
            列名，如果不存在返回None
        """
        name_upper = name.upper()

        # 直接匹配列名
        if name_upper.lower() in self._data.columns:
            return name_upper.lower()

        # 带参数的指标 (如 MA20, EMA5等)
        if params and 'n' in params:
            n = params['n']
            possible_names = [
                f"{name_upper}{n}",           # MA20
                f"{name_upper}_{n}",          # MA_20
                f"{name_upper.lower()}{n}",   # ma20
                f"{name_upper.lower()}_{n}",  # ma_20
            ]

            for col in possible_names:
                if col in self._data.columns:
                    return col

        # 特殊处理KDJ
        if 'KDJ' in name_upper:
            kdj_mapping = {
                'KDJ_K': ['kdj_k', 'k', 'K'],
                'KDJ_D': ['kdj_d', 'd', 'D'],
                'KDJ_J': ['kdj_j', 'j', 'J']
            }

            if name_upper in kdj_mapping:
                for col in kdj_mapping[name_upper]:
                    if col in self._data.columns:
                        return col

        # 其他常见指标
        common_indicators = {
            'BBI': ['bbi', 'BBI'],
            'RSI': ['rsi', 'RSI', 'rsi_6', 'RSI_6'],
            'MACD': ['macd', 'MACD', 'dif'],
            'MACD_SIGNAL': ['dea', 'DEA', 'signal'],
            'MACD_HIST': ['macd_hist', 'MACD_HIST', 'histogram']
        }

        if name_upper in common_indicators:
            for col in common_indicators[name_upper]:
                if col in self._data.columns:
                    return col

        return None

    def to_dataframe(self) -> pd.DataFrame:
        """转换为pandas DataFrame"""
        return self._data.copy()

    def __repr__(self):
        return f"KData(stock={self.stock_code}, len={len(self)}, " \
               f"start={self._datetime_list[0] if self._datetime_list else 'N/A'}, " \
               f"end={self._datetime_list[-1] if self._datetime_list else 'N/A'})"
