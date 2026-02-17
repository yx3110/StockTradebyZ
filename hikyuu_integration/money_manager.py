#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MoneyManagerBase - 资金管理基类

借鉴Hikyuu的MoneyManager设计
"""

from abc import ABC, abstractmethod
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MoneyManagerBase(ABC):
    """
    资金管理基类

    决定每次交易买入多少股票

    子类需要实现:
    - get_buy_num(): 计算买入数量
    """

    def __init__(self, name: str = 'MoneyManager'):
        self.name = name

    @abstractmethod
    def get_buy_num(self,
                    date: str,
                    stock_code: str,
                    price: float,
                    available_cash: float) -> int:
        """
        计算买入数量

        参数:
            date: 日期
            stock_code: 股票代码
            price: 买入价格
            available_cash: 可用资金

        返回:
            买入股数（必须是100的整数倍）
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"


class MM_FixedCount(MoneyManagerBase):
    """
    固定股数资金管理

    每次买入固定股数

    示例:
        mm = MM_FixedCount(1000)  # 每次买入1000股
    """

    def __init__(self, count: int = 1000):
        """
        参数:
            count: 固定买入股数（必须是100的整数倍）
        """
        super().__init__(name=f'FixedCount({count})')
        if count % 100 != 0:
            raise ValueError("count must be multiple of 100")
        self.count = count

    def get_buy_num(self, date: str, stock_code: str, price: float, available_cash: float) -> int:
        """返回固定股数"""
        # 检查资金是否足够
        required_cash = self.count * price
        if required_cash > available_cash:
            # 资金不足，返回0
            logger.debug(f"资金不足: 需要{required_cash:.2f}，可用{available_cash:.2f}")
            return 0

        return self.count


class MM_FixedRisk(MoneyManagerBase):
    """
    固定风险资金管理

    根据风险百分比计算买入数量

    示例:
        mm = MM_FixedRisk(risk_pct=0.02, stop_loss_pct=0.08)
        # 每次风险2%，止损8%
    """

    def __init__(self, risk_pct: float = 0.02, stop_loss_pct: float = 0.08):
        """
        参数:
            risk_pct: 风险百分比（如0.02表示每次风险2%）
            stop_loss_pct: 止损百分比（如0.08表示止损8%）
        """
        super().__init__(name=f'FixedRisk({risk_pct:.1%})')
        self.risk_pct = risk_pct
        self.stop_loss_pct = stop_loss_pct

    def get_buy_num(self, date: str, stock_code: str, price: float, available_cash: float) -> int:
        """
        根据风险计算买入数量

        计算公式:
        风险金额 = 总资金 × 风险百分比
        买入数量 = 风险金额 / (价格 × 止损百分比)
        """
        risk_amount = available_cash * self.risk_pct
        shares = int(risk_amount / (price * self.stop_loss_pct))

        # 转换为100的整数倍
        shares = (shares // 100) * 100

        # 检查资金是否足够
        required_cash = shares * price
        if required_cash > available_cash:
            return 0

        return shares


class MM_FixedPercent(MoneyManagerBase):
    """
    固定比例资金管理

    每次使用固定比例的可用资金

    示例:
        mm = MM_FixedPercent(0.1)  # 每次使用10%的可用资金
    """

    def __init__(self, percent: float = 0.1):
        """
        参数:
            percent: 使用资金的百分比（如0.1表示10%）
        """
        super().__init__(name=f'FixedPercent({percent:.1%})')
        if not 0 < percent <= 1:
            raise ValueError("percent must be between 0 and 1")
        self.percent = percent

    def get_buy_num(self, date: str, stock_code: str, price: float, available_cash: float) -> int:
        """根据固定比例计算买入数量"""
        buy_cash = available_cash * self.percent
        shares = int(buy_cash / price)

        # 转换为100的整数倍
        shares = (shares // 100) * 100

        return shares
