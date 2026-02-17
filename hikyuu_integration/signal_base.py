#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SignalBase - 信号基类

借鉴Hikyuu的Signal设计，提供买入/卖出信号生成框架
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from datetime import datetime
import logging

from .kdata import KData

logger = logging.getLogger(__name__)


class SignalBase(ABC):
    """
    信号基类

    借鉴Hikyuu的Signal设计，所有信号策略都继承此类

    核心方法:
    - _calculate(): 子类实现，计算买入/卖出信号
    - should_buy(): 判断是否应该买入
    - should_sell(): 判断是否应该卖出

    用法示例:
        class MySignal(SignalBase):
            def _calculate(self, kdata):
                for i in range(len(kdata)):
                    if kdata.close[i] > kdata.get_indicator('BBI')[i]:
                        self._add_buy_signal(kdata.datetime[i])
                    elif kdata.close[i] < kdata.get_indicator('BBI')[i]:
                        self._add_sell_signal(kdata.datetime[i])

        signal = MySignal(name='BBI_Signal')
        signal.calculate(kdata)
        if signal.should_buy('2025-09-30'):
            print("买入信号!")
    """

    def __init__(self, name: str = 'Signal', params: Optional[Dict] = None):
        """
        初始化信号基类

        参数:
            name: 信号名称
            params: 参数字典
        """
        self.name = name
        self.params = params or {}

        # 信号存储
        self._buy_signals = {}   # {日期: 信号强度}
        self._sell_signals = {}  # {日期: 信号强度}

        # 当前绑定的K线数据
        self._kdata = None

        # 计算状态
        self._is_calculated = False

    @abstractmethod
    def _calculate(self, kdata: KData):
        """
        计算信号 - 子类必须实现

        参数:
            kdata: K线数据对象

        子类在此方法中调用:
        - self._add_buy_signal(date, strength) 添加买入信号
        - self._add_sell_signal(date, strength) 添加卖出信号
        """
        pass

    def calculate(self, kdata: KData):
        """
        执行信号计算

        参数:
            kdata: K线数据对象
        """
        # 重置状态
        self._buy_signals.clear()
        self._sell_signals.clear()
        self._is_calculated = False

        # 保存K线数据
        self._kdata = kdata

        try:
            # 调用子类实现
            self._calculate(kdata)
            self._is_calculated = True

            logger.debug(f"{self.name} 计算完成: "
                        f"买入信号{len(self._buy_signals)}个, "
                        f"卖出信号{len(self._sell_signals)}个")

        except Exception as e:
            logger.error(f"{self.name} 计算失败: {e}")
            raise

    def should_buy(self, date: str) -> bool:
        """
        判断指定日期是否应该买入

        参数:
            date: 日期字符串 (YYYY-MM-DD)

        返回:
            True表示买入，False表示不买入
        """
        if not self._is_calculated:
            logger.warning(f"{self.name} 尚未计算，请先调用calculate()")
            return False

        return date in self._buy_signals

    def should_sell(self, date: str) -> bool:
        """
        判断指定日期是否应该卖出

        参数:
            date: 日期字符串 (YYYY-MM-DD)

        返回:
            True表示卖出，False表示不卖出
        """
        if not self._is_calculated:
            logger.warning(f"{self.name} 尚未计算，请先调用calculate()")
            return False

        return date in self._sell_signals

    def get_buy_signal_strength(self, date: str) -> float:
        """
        获取买入信号强度

        参数:
            date: 日期字符串

        返回:
            信号强度 (0-1)，没有信号返回0
        """
        return self._buy_signals.get(date, 0.0)

    def get_sell_signal_strength(self, date: str) -> float:
        """
        获取卖出信号强度

        参数:
            date: 日期字符串

        返回:
            信号强度 (0-1)，没有信号返回0
        """
        return self._sell_signals.get(date, 0.0)

    def get_all_buy_dates(self) -> List[str]:
        """获取所有买入信号日期"""
        return sorted(self._buy_signals.keys())

    def get_all_sell_dates(self) -> List[str]:
        """获取所有卖出信号日期"""
        return sorted(self._sell_signals.keys())

    # ==================== 子类使用的辅助方法 ====================

    def _add_buy_signal(self, date: str, strength: float = 1.0):
        """
        添加买入信号 - 供子类调用

        参数:
            date: 日期字符串
            strength: 信号强度 (0-1)，默认1.0
        """
        self._buy_signals[date] = max(0.0, min(1.0, strength))

    def _add_sell_signal(self, date: str, strength: float = 1.0):
        """
        添加卖出信号 - 供子类调用

        参数:
            date: 日期字符串
            strength: 信号强度 (0-1)，默认1.0
        """
        self._sell_signals[date] = max(0.0, min(1.0, strength))

    def set_param(self, key: str, value):
        """设置参数"""
        self.params[key] = value

    def get_param(self, key: str, default=None):
        """获取参数"""
        return self.params.get(key, default)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, params={self.params})"


class CompositeSignal(SignalBase):
    """
    组合信号 - 组合多个信号

    支持AND/OR逻辑组合

    示例:
        signal1 = BBISignal()
        signal2 = KDJSignal()

        # AND组合：两个信号都触发才买入
        composite_and = CompositeSignal([signal1, signal2], mode='AND')

        # OR组合：任一信号触发就买入
        composite_or = CompositeSignal([signal1, signal2], mode='OR')
    """

    def __init__(self, signals: List[SignalBase], mode: str = 'AND', name: str = 'CompositeSignal'):
        """
        初始化组合信号

        参数:
            signals: 信号列表
            mode: 组合模式 ('AND' 或 'OR')
            name: 信号名称
        """
        super().__init__(name=name, params={'mode': mode})
        self.signals = signals
        self.mode = mode.upper()

        if self.mode not in ['AND', 'OR']:
            raise ValueError("mode must be 'AND' or 'OR'")

    def _calculate(self, kdata: KData):
        """计算组合信号"""
        # 先计算所有子信号
        for signal in self.signals:
            signal.calculate(kdata)

        # 获取所有日期
        all_dates = set()
        for signal in self.signals:
            all_dates.update(signal.get_all_buy_dates())
            all_dates.update(signal.get_all_sell_dates())

        # 根据组合模式生成信号
        for date in all_dates:
            # 买入信号
            buy_signals = [s.should_buy(date) for s in self.signals]
            if self.mode == 'AND':
                if all(buy_signals):
                    # 取最小强度
                    strength = min([s.get_buy_signal_strength(date) for s in self.signals])
                    self._add_buy_signal(date, strength)
            else:  # OR
                if any(buy_signals):
                    # 取最大强度
                    strengths = [s.get_buy_signal_strength(date) for s in self.signals if s.should_buy(date)]
                    self._add_buy_signal(date, max(strengths))

            # 卖出信号
            sell_signals = [s.should_sell(date) for s in self.signals]
            if self.mode == 'AND':
                if all(sell_signals):
                    strength = min([s.get_sell_signal_strength(date) for s in self.signals])
                    self._add_sell_signal(date, strength)
            else:  # OR
                if any(sell_signals):
                    strengths = [s.get_sell_signal_strength(date) for s in self.signals if s.should_sell(date)]
                    self._add_sell_signal(date, max(strengths))


# ==================== 示例信号实现 ====================

class BBISignal(SignalBase):
    """
    BBI信号示例

    买入: 收盘价上穿BBI
    卖出: 收盘价下穿BBI
    """

    def __init__(self, name: str = 'BBI_Signal'):
        super().__init__(name=name)

    def _calculate(self, kdata: KData):
        """计算BBI信号"""
        bbi = kdata.get_indicator('BBI')
        if bbi is None:
            logger.warning(f"{self.name}: BBI指标不存在")
            return

        close = kdata.close

        # 从第2根K线开始判断（需要前一根进行对比）
        for i in range(1, len(kdata)):
            # 上穿：前一根收盘价 <= BBI，当前收盘价 > BBI
            if close[i-1] <= bbi[i-1] and close[i] > bbi[i]:
                self._add_buy_signal(kdata.datetime[i])

            # 下穿：前一根收盘价 >= BBI，当前收盘价 < BBI
            elif close[i-1] >= bbi[i-1] and close[i] < bbi[i]:
                self._add_sell_signal(kdata.datetime[i])


class KDJSignal(SignalBase):
    """
    KDJ信号示例

    买入: KDJ_K < 20 (超卖)
    卖出: KDJ_K > 80 (超买)
    """

    def __init__(self, oversold: float = 20, overbought: float = 80, name: str = 'KDJ_Signal'):
        super().__init__(name=name, params={'oversold': oversold, 'overbought': overbought})
        self.oversold = oversold
        self.overbought = overbought

    def _calculate(self, kdata: KData):
        """计算KDJ信号"""
        kdj_k = kdata.get_indicator('KDJ_K')
        if kdj_k is None:
            logger.warning(f"{self.name}: KDJ_K指标不存在")
            return

        for i in range(len(kdata)):
            k_value = kdj_k[i]

            # 超卖区间买入
            if k_value < self.oversold:
                # 信号强度与超卖程度成正比
                strength = (self.oversold - k_value) / self.oversold
                self._add_buy_signal(kdata.datetime[i], strength)

            # 超买区间卖出
            elif k_value > self.overbought:
                # 信号强度与超买程度成正比
                strength = (k_value - self.overbought) / (100 - self.overbought)
                self._add_sell_signal(kdata.datetime[i], strength)
