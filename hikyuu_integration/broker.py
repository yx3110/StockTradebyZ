#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Broker - 交易执行类

借鉴Hikyuu的Broker设计，模拟真实交易环境
处理涨跌停、T+1、交易成本等约束
"""

from typing import Optional, Dict
from datetime import datetime, timedelta
import logging

from .portfolio import Portfolio
from .data_adapter import HikyuuStyleDataAdapter

logger = logging.getLogger(__name__)


class Broker:
    """
    交易执行类

    模拟真实交易环境，处理：
    - T+1交易规则（当天买入的股票次日才能卖出）
    - 涨跌停限制（涨停无法买入，跌停无法卖出）
    - 滑点模拟（可选）
    - 交易成本（通过Portfolio计算）

    示例:
        broker = Broker(portfolio, data_adapter)

        # 买入
        success = broker.buy('000001', 1000, '2025-09-30', price=10.0)

        # 卖出
        success = broker.sell('000001', 1000, '2025-10-01', price=11.0)
    """

    def __init__(self,
                 portfolio: Portfolio,
                 data_adapter: HikyuuStyleDataAdapter,
                 enable_t1: bool = True,
                 enable_limit_check: bool = True,
                 slippage: float = 0.0):
        """
        初始化Broker

        参数:
            portfolio: 组合对象
            data_adapter: 数据适配器
            enable_t1: 是否启用T+1规则（默认True）
            enable_limit_check: 是否检查涨跌停（默认True）
            slippage: 滑点（如0.001表示千分之一滑点）
        """
        self.portfolio = portfolio
        self.data_adapter = data_adapter
        self.enable_t1 = enable_t1
        self.enable_limit_check = enable_limit_check
        self.slippage = slippage

        # T+1持仓锁定: {stock_code: buy_date}
        self._t1_locks: Dict[str, str] = {}

        logger.info(f"初始化Broker: T+1={enable_t1}, "
                   f"涨跌停检查={enable_limit_check}, "
                   f"滑点={slippage}")

    def buy(self,
            stock_code: str,
            shares: int,
            date: str,
            price: Optional[float] = None,
            reason: str = '') -> bool:
        """
        买入股票

        参数:
            stock_code: 股票代码
            shares: 买入股数
            date: 交易日期
            price: 买入价格（如果为None，使用当日收盘价）
            reason: 买入原因

        返回:
            是否成功买入
        """
        # 获取价格
        if price is None:
            price = self._get_price(stock_code, date, 'close')

        if price is None:
            logger.warning(f"{date}: {stock_code} 无价格数据")
            return False

        # 检查涨跌停
        if self.enable_limit_check:
            if self._is_limit_up(stock_code, date):
                logger.debug(f"{date}: {stock_code} 涨停，无法买入")
                return False

        # 应用滑点（买入时价格更高）
        if self.slippage > 0:
            price = price * (1 + self.slippage)

        # 执行买入
        success = self.portfolio.buy(stock_code, shares, price, date, reason)

        if success and self.enable_t1:
            # 记录T+1锁定
            self._t1_locks[stock_code] = date
            logger.debug(f"{date}: {stock_code} T+1锁定至次日")

        return success

    def sell(self,
             stock_code: str,
             shares: int,
             date: str,
             price: Optional[float] = None,
             reason: str = '') -> bool:
        """
        卖出股票

        参数:
            stock_code: 股票代码
            shares: 卖出股数
            date: 交易日期
            price: 卖出价格（如果为None，使用当日收盘价）
            reason: 卖出原因

        返回:
            是否成功卖出
        """
        # 检查T+1规则
        if self.enable_t1:
            if not self._can_sell_t1(stock_code, date):
                logger.debug(f"{date}: {stock_code} T+1限制，无法卖出")
                return False

        # 获取价格
        if price is None:
            price = self._get_price(stock_code, date, 'close')

        if price is None:
            logger.warning(f"{date}: {stock_code} 无价格数据")
            return False

        # 检查涨跌停
        if self.enable_limit_check:
            if self._is_limit_down(stock_code, date):
                logger.debug(f"{date}: {stock_code} 跌停，无法卖出")
                return False

        # 应用滑点（卖出时价格更低）
        if self.slippage > 0:
            price = price * (1 - self.slippage)

        # 执行卖出
        success = self.portfolio.sell(stock_code, shares, price, date, reason)

        if success and stock_code in self._t1_locks:
            # 清除T+1锁定
            del self._t1_locks[stock_code]
            logger.debug(f"{date}: {stock_code} 清除T+1锁定")

        return success

    def _get_price(self, stock_code: str, date: str, price_type: str = 'close') -> Optional[float]:
        """
        获取价格

        参数:
            stock_code: 股票代码
            date: 日期
            price_type: 价格类型 ('open', 'high', 'low', 'close')

        返回:
            价格，如果不存在返回None
        """
        try:
            # 查询单日数据
            from .query import Query
            kdata = self.data_adapter.get_kdata(stock_code, Query(start=date, end=date))

            if len(kdata) == 0:
                return None

            # 获取指定类型的价格
            price_map = {
                'open': kdata.open[0],
                'high': kdata.high[0],
                'low': kdata.low[0],
                'close': kdata.close[0]
            }

            return price_map.get(price_type)

        except Exception as e:
            logger.warning(f"获取{stock_code} {date}价格失败: {e}")
            return None

    def _is_limit_up(self, stock_code: str, date: str) -> bool:
        """
        检查是否涨停

        参数:
            stock_code: 股票代码
            date: 日期

        返回:
            是否涨停
        """
        try:
            from .query import Query
            kdata = self.data_adapter.get_kdata(stock_code, Query(start=date, end=date))

            if len(kdata) == 0:
                return False

            # 从price_change_pct或is_limit_up判断
            # 简化实现：涨幅 >= 9.9% 视为涨停
            if hasattr(kdata._data.iloc[0], 'is_limit_up'):
                return kdata._data.iloc[0]['is_limit_up']

            # 否则通过涨幅判断
            if hasattr(kdata._data.iloc[0], 'price_change_pct'):
                return kdata._data.iloc[0]['price_change_pct'] >= 9.9

            return False

        except Exception as e:
            logger.debug(f"检查涨停失败: {e}")
            return False

    def _is_limit_down(self, stock_code: str, date: str) -> bool:
        """
        检查是否跌停

        参数:
            stock_code: 股票代码
            date: 日期

        返回:
            是否跌停
        """
        try:
            from .query import Query
            kdata = self.data_adapter.get_kdata(stock_code, Query(start=date, end=date))

            if len(kdata) == 0:
                return False

            # 从price_change_pct或is_limit_down判断
            if hasattr(kdata._data.iloc[0], 'is_limit_down'):
                return kdata._data.iloc[0]['is_limit_down']

            # 否则通过跌幅判断
            if hasattr(kdata._data.iloc[0], 'price_change_pct'):
                return kdata._data.iloc[0]['price_change_pct'] <= -9.9

            return False

        except Exception as e:
            logger.debug(f"检查跌停失败: {e}")
            return False

    def _can_sell_t1(self, stock_code: str, date: str) -> bool:
        """
        检查是否可以卖出（T+1规则）

        参数:
            stock_code: 股票代码
            date: 当前日期

        返回:
            是否可以卖出
        """
        if stock_code not in self._t1_locks:
            # 没有锁定，可以卖出
            return True

        buy_date = self._t1_locks[stock_code]

        # 检查日期是否在买入日之后
        # 简化实现：字符串比较（假设日期格式为YYYY-MM-DD或YYYYMMDD）
        if date > buy_date:
            return True

        return False

    def update_date(self, date: str):
        """
        更新日期（用于回测时推进时间）

        清理过期的T+1锁定

        参数:
            date: 当前日期
        """
        # 清理过期的T+1锁定
        expired_locks = [
            code for code, buy_date in self._t1_locks.items()
            if date > buy_date
        ]

        for code in expired_locks:
            logger.debug(f"{date}: 清理过期T+1锁定 {code}")
            # 保留锁定，直到实际卖出
            # del self._t1_locks[code]

    def get_t1_locks(self) -> Dict[str, str]:
        """获取T+1锁定列表"""
        return self._t1_locks.copy()

    def has_t1_lock(self, stock_code: str) -> bool:
        """检查是否有T+1锁定"""
        return stock_code in self._t1_locks

    def __repr__(self):
        return (f"Broker(T+1={self.enable_t1}, "
                f"涨跌停检查={self.enable_limit_check}, "
                f"滑点={self.slippage})")
