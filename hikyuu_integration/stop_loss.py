#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StopLossBase - 止损/止盈基类

借鉴Hikyuu的StopLoss设计
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class StopLossBase(ABC):
    """
    止损/止盈基类

    判断是否应该止损或止盈

    子类需要实现:
    - should_stop(): 判断是否触发止损/止盈
    """

    def __init__(self, name: str = 'StopLoss'):
        self.name = name

    @abstractmethod
    def should_stop(self,
                   date: str,
                   entry_price: float,
                   current_price: float,
                   entry_date: str) -> bool:
        """
        判断是否应该止损/止盈

        参数:
            date: 当前日期
            entry_price: 买入价格
            current_price: 当前价格
            entry_date: 买入日期

        返回:
            True表示应该止损/止盈，False表示不需要
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"


class ST_FixedPercent(StopLossBase):
    """
    固定百分比止损

    当亏损达到固定百分比时止损

    示例:
        st = ST_FixedPercent(0.08)  # 亏损8%止损
    """

    def __init__(self, stop_percent: float = 0.08):
        """
        参数:
            stop_percent: 止损百分比（如0.08表示亏损8%止损）
        """
        super().__init__(name=f'FixedPercent({stop_percent:.1%})')
        self.stop_percent = stop_percent

    def should_stop(self, date: str, entry_price: float, current_price: float, entry_date: str) -> bool:
        """判断是否触发止损"""
        loss_pct = (current_price - entry_price) / entry_price

        if loss_pct < -self.stop_percent:
            logger.debug(f"{date}: 触发止损，亏损{loss_pct:.2%}")
            return True

        return False


class ST_ProfitGoal(StopLossBase):
    """
    盈利目标止盈

    当盈利达到目标时止盈

    示例:
        st = ST_ProfitGoal(0.20)  # 盈利20%止盈
    """

    def __init__(self, profit_percent: float = 0.20):
        """
        参数:
            profit_percent: 止盈百分比（如0.20表示盈利20%止盈）
        """
        super().__init__(name=f'ProfitGoal({profit_percent:.1%})')
        self.profit_percent = profit_percent

    def should_stop(self, date: str, entry_price: float, current_price: float, entry_date: str) -> bool:
        """判断是否触发止盈"""
        profit_pct = (current_price - entry_price) / entry_price

        if profit_pct > self.profit_percent:
            logger.debug(f"{date}: 触发止盈，盈利{profit_pct:.2%}")
            return True

        return False


class ST_Trailing(StopLossBase):
    """
    追踪止损

    价格从最高点回撤达到指定百分比时止损

    示例:
        st = ST_Trailing(0.05)  # 从最高点回撤5%止损
    """

    def __init__(self, trailing_percent: float = 0.05):
        """
        参数:
            trailing_percent: 追踪止损百分比（如0.05表示回撤5%止损）
        """
        super().__init__(name=f'Trailing({trailing_percent:.1%})')
        self.trailing_percent = trailing_percent
        self._highest_prices = {}  # {stock_code: highest_price}

    def should_stop(self, date: str, entry_price: float, current_price: float, entry_date: str) -> bool:
        """判断是否触发追踪止损"""
        # 更新最高价（简化实现，实际需要传入stock_code）
        if entry_price not in self._highest_prices:
            self._highest_prices[entry_price] = entry_price

        highest = self._highest_prices[entry_price]

        # 更新最高价
        if current_price > highest:
            self._highest_prices[entry_price] = current_price
            highest = current_price

        # 计算从最高点的回撤
        drawdown = (current_price - highest) / highest

        if drawdown < -self.trailing_percent:
            logger.debug(f"{date}: 触发追踪止损，回撤{drawdown:.2%}")
            return True

        return False


class ST_Composite(StopLossBase):
    """
    组合止损

    组合多个止损策略，任一触发即止损

    示例:
        st = ST_Composite([
            ST_FixedPercent(0.08),    # 固定止损8%
            ST_ProfitGoal(0.20),      # 止盈20%
            ST_Trailing(0.05)         # 追踪止损5%
        ])
    """

    def __init__(self, stop_losses: list):
        """
        参数:
            stop_losses: 止损策略列表
        """
        super().__init__(name='CompositeStopLoss')
        self.stop_losses = stop_losses

    def should_stop(self, date: str, entry_price: float, current_price: float, entry_date: str) -> bool:
        """任一策略触发即止损"""
        for sl in self.stop_losses:
            if sl.should_stop(date, entry_price, current_price, entry_date):
                logger.debug(f"{date}: {sl.name} 触发")
                return True

        return False
