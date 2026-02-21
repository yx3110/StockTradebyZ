#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易策略系统 - 策略与回测引擎解耦

架构:
- TradingStrategy: 抽象基类，定义策略接口
- StrategyConfig: 策略配置数据类
- Position: 持仓信息数据类
- 具体策略: ConservativeStrategy, BalancedStrategy, AggressiveStrategy

Author: Claude Code
Date: 2025-10-12
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Position:
    """持仓信息"""
    stock_code: str
    shares: int
    avg_cost: float
    entry_date: str
    entry_score: float = 0.0
    # 🆕 Bug修复：保存买入时的止盈止损参数，避免持仓期间参数变化
    entry_take_profit_pct: Optional[float] = None  # 买入时止盈目标
    entry_stop_loss_pct: Optional[float] = None    # 买入时止损阈值
    entry_market_regime: Optional[str] = None      # 买入时市场环境


@dataclass
class StrategyConfig:
    """策略配置"""
    # 止盈止损
    take_profit_pct: float = 0.15
    stop_loss_pct: float = 0.08

    # 持仓管理
    max_holding_days: int = 20
    min_score_for_hold: float = 75.0

    # 调仓配置
    enable_rebalance_sell: bool = True
    rebalance_frequency: int = 5

    # 仓位管理
    max_positions: int = 10
    max_position_pct: float = 0.15

    # 策略元数据
    name: str = "未命名策略"
    description: str = ""
    risk_level: str = "中等"  # 低/中等/高


class TradingStrategy(ABC):
    """交易策略抽象基类"""

    def __init__(self, config: StrategyConfig):
        self.config = config

    # ========== 核心决策接口 ==========

    @abstractmethod
    def should_sell_on_rebalance(
        self,
        position: Position,
        current_price: float,
        current_date: str,
        selected_stocks: List[Dict]
    ) -> Tuple[bool, str]:
        """
        调仓时是否卖出某个持仓

        Args:
            position: 持仓信息
            current_price: 当前价格
            current_date: 当前日期
            selected_stocks: 新选出的股票列表

        Returns:
            (should_sell, reason)
        """
        pass

    @abstractmethod
    def should_take_profit(
        self,
        position: Position,
        current_price: float
    ) -> bool:
        """是否止盈"""
        pass

    @abstractmethod
    def should_stop_loss(
        self,
        position: Position,
        current_price: float
    ) -> bool:
        """是否止损"""
        pass

    @abstractmethod
    def should_check_holding_period(
        self,
        position: Position,
        current_date: str
    ) -> bool:
        """是否超过持仓期限"""
        pass

    @abstractmethod
    def calculate_position_size(
        self,
        stock_code: str,
        stock_price: float,
        available_capital: float,
        current_positions: int
    ) -> int:
        """
        计算买入股数

        Args:
            stock_code: 股票代码
            stock_price: 股票价格
            available_capital: 可用资金
            current_positions: 当前持仓数量

        Returns:
            买入股数 (手，100股/手)
        """
        pass

    # ========== 辅助方法 ==========

    def calculate_profit_pct(self, position: Position, current_price: float) -> float:
        """计算盈亏百分比"""
        if position.avg_cost <= 0:
            return 0.0
        return (current_price - position.avg_cost) / position.avg_cost

    def calculate_holding_days(self, entry_date: str, current_date: str) -> int:
        """计算持仓天数"""
        try:
            entry = datetime.strptime(entry_date, '%Y-%m-%d')
            current = datetime.strptime(current_date, '%Y-%m-%d')
            return (current - entry).days
        except Exception:
            return 0

    def get_config(self) -> StrategyConfig:
        """获取策略配置"""
        return self.config

    def get_info(self) -> Dict:
        """获取策略信息"""
        return {
            'name': self.config.name,
            'description': self.config.description,
            'risk_level': self.config.risk_level,
            'config': self.config.__dict__
        }


# ========== 具体策略实现 ==========

class BalancedStrategy(TradingStrategy):
    """平衡策略 - 中等风险收益 (当前默认策略)"""

    def __init__(self):
        config = StrategyConfig(
            take_profit_pct=0.15,
            stop_loss_pct=0.08,
            max_holding_days=20,
            min_score_for_hold=75.0,
            enable_rebalance_sell=True,
            rebalance_frequency=5,
            max_positions=10,
            max_position_pct=0.15,
            name="平衡策略",
            description="15%止盈，8%止损，最长持仓20天，评分75+继续持有",
            risk_level="中等"
        )
        super().__init__(config)

    def should_sell_on_rebalance(self, position, current_price, current_date, selected_stocks):
        """调仓时是否卖出"""
        selected_codes = {s['stock_code'] for s in selected_stocks}
        selected_scores = {s['stock_code']: s['score'] for s in selected_stocks}

        # 检查1: 不在新选股列表
        if position.stock_code not in selected_codes:
            return True, "rebalance_not_selected"

        # 检查2: 评分低于持有阈值
        if position.stock_code in selected_scores:
            if selected_scores[position.stock_code] < self.config.min_score_for_hold:
                return True, f"rebalance_low_score_{selected_scores[position.stock_code]:.1f}"

        # 检查3: 持仓时间过长
        holding_days = self.calculate_holding_days(position.entry_date, current_date)
        if holding_days > self.config.max_holding_days:
            return True, f"rebalance_max_holding_{holding_days}d"

        return False, ""

    def should_take_profit(self, position, current_price):
        """是否止盈"""
        profit_pct = self.calculate_profit_pct(position, current_price)
        return profit_pct > self.config.take_profit_pct

    def should_stop_loss(self, position, current_price):
        """是否止损"""
        profit_pct = self.calculate_profit_pct(position, current_price)
        return profit_pct < -self.config.stop_loss_pct

    def should_check_holding_period(self, position, current_date):
        """是否超过持仓期限"""
        holding_days = self.calculate_holding_days(position.entry_date, current_date)
        return holding_days > self.config.max_holding_days

    def calculate_position_size(self, stock_code, stock_price, available_capital, current_positions):
        """计算买入股数 - 平均分配"""
        max_new_positions = self.config.max_positions - current_positions
        if max_new_positions <= 0:
            return 0

        # 平均分配资金
        target_allocation = available_capital / max_new_positions

        # 计算股数 (按手，100股/手)
        shares = int(target_allocation / (stock_price * 100)) * 100
        return shares


class ConservativeStrategy(TradingStrategy):
    """保守策略 - 低风险，注重资本保护"""

    def __init__(self):
        config = StrategyConfig(
            take_profit_pct=0.10,      # 更低的止盈目标，及时锁定利润
            stop_loss_pct=0.05,        # 更严格的止损，快速止损
            max_holding_days=30,       # 更长的持仓周期，避免频繁交易
            min_score_for_hold=80.0,   # 更高的评分要求，只持有优质股
            enable_rebalance_sell=True,
            rebalance_frequency=5,
            max_positions=8,           # 更少的持仓数，更集中
            max_position_pct=0.12,     # 单只股票最大仓位12%
            name="保守策略",
            description="10%止盈，5%止损，高评分要求，长周期持有",
            risk_level="低"
        )
        super().__init__(config)

    def should_sell_on_rebalance(self, position, current_price, current_date, selected_stocks):
        """保守策略：更倾向于持有高质量股票"""
        selected_codes = {s['stock_code'] for s in selected_stocks}
        selected_scores = {s['stock_code']: s['score'] for s in selected_stocks}

        # 不在新列表 → 卖出
        if position.stock_code not in selected_codes:
            return True, "rebalance_not_selected"

        # 评分低于80 → 卖出 (更严格)
        if position.stock_code in selected_scores:
            if selected_scores[position.stock_code] < self.config.min_score_for_hold:
                return True, f"rebalance_low_score_{selected_scores[position.stock_code]:.1f}"

        # 持仓超过30天 → 卖出
        holding_days = self.calculate_holding_days(position.entry_date, current_date)
        if holding_days > self.config.max_holding_days:
            return True, f"rebalance_max_holding_{holding_days}d"

        return False, ""

    def should_take_profit(self, position, current_price):
        """保守策略：10%止盈"""
        profit_pct = self.calculate_profit_pct(position, current_price)
        return profit_pct > self.config.take_profit_pct

    def should_stop_loss(self, position, current_price):
        """保守策略：5%止损"""
        profit_pct = self.calculate_profit_pct(position, current_price)
        return profit_pct < -self.config.stop_loss_pct

    def should_check_holding_period(self, position, current_date):
        """保守策略：30天超期"""
        holding_days = self.calculate_holding_days(position.entry_date, current_date)
        return holding_days > self.config.max_holding_days

    def calculate_position_size(self, stock_code, stock_price, available_capital, current_positions):
        """保守策略：平均分配，但单只最多12%"""
        max_new_positions = self.config.max_positions - current_positions
        if max_new_positions <= 0:
            return 0

        target_allocation = available_capital / max_new_positions
        shares = int(target_allocation / (stock_price * 100)) * 100
        return shares


class AggressiveStrategy(TradingStrategy):
    """激进策略 - 高风险高收益，追求最大化收益"""

    def __init__(self):
        config = StrategyConfig(
            take_profit_pct=0.20,      # 更高的止盈目标
            stop_loss_pct=0.10,        # 更宽松的止损，容忍更大波动
            max_holding_days=10,       # 更短的持仓周期，快速轮换
            min_score_for_hold=70.0,   # 更低的评分要求
            enable_rebalance_sell=True,
            rebalance_frequency=3,     # 更频繁的调仓
            max_positions=15,          # 更多的持仓数，分散风险
            max_position_pct=0.10,     # 单只股票最多10%
            name="激进策略",
            description="20%止盈，10%止损，高频交易，快速轮换",
            risk_level="高"
        )
        super().__init__(config)

    def should_sell_on_rebalance(self, position, current_price, current_date, selected_stocks):
        """激进策略：更频繁换股"""
        selected_codes = {s['stock_code'] for s in selected_stocks}
        selected_scores = {s['stock_code']: s['score'] for s in selected_stocks}

        # 不在新列表 → 卖出
        if position.stock_code not in selected_codes:
            return True, "rebalance_not_selected"

        # 评分低于70 → 卖出
        if position.stock_code in selected_scores:
            if selected_scores[position.stock_code] < self.config.min_score_for_hold:
                return True, f"rebalance_low_score_{selected_scores[position.stock_code]:.1f}"

        # 持仓超过10天 → 卖出 (快速轮换)
        holding_days = self.calculate_holding_days(position.entry_date, current_date)
        if holding_days > self.config.max_holding_days:
            return True, f"rebalance_max_holding_{holding_days}d"

        return False, ""

    def should_take_profit(self, position, current_price):
        """激进策略：20%止盈"""
        profit_pct = self.calculate_profit_pct(position, current_price)
        return profit_pct > self.config.take_profit_pct

    def should_stop_loss(self, position, current_price):
        """激进策略：10%止损 (容忍更大波动)"""
        profit_pct = self.calculate_profit_pct(position, current_price)
        return profit_pct < -self.config.stop_loss_pct

    def should_check_holding_period(self, position, current_date):
        """激进策略：10天超期"""
        holding_days = self.calculate_holding_days(position.entry_date, current_date)
        return holding_days > self.config.max_holding_days

    def calculate_position_size(self, stock_code, stock_price, available_capital, current_positions):
        """激进策略：平均分配到更多股票"""
        max_new_positions = self.config.max_positions - current_positions
        if max_new_positions <= 0:
            return 0

        target_allocation = available_capital / max_new_positions
        shares = int(target_allocation / (stock_price * 100)) * 100
        return shares


# ========== 策略工厂 ==========

class StrategyFactory:
    """策略工厂 - 便捷创建策略实例"""

    _strategies = {
        'conservative': ConservativeStrategy,
        'balanced': BalancedStrategy,
        'aggressive': AggressiveStrategy
    }

    @classmethod
    def create(cls, strategy_name: str) -> TradingStrategy:
        """
        创建策略实例

        Args:
            strategy_name: 策略名称 ('conservative', 'balanced', 'aggressive')

        Returns:
            策略实例
        """
        if strategy_name not in cls._strategies:
            raise ValueError(f"未知策略: {strategy_name}. 可用策略: {list(cls._strategies.keys())}")

        return cls._strategies[strategy_name]()

    @classmethod
    def list_strategies(cls) -> List[str]:
        """列出所有可用策略"""
        return list(cls._strategies.keys())

    @classmethod
    def get_strategy_info(cls, strategy_name: str) -> Dict:
        """获取策略信息"""
        strategy = cls.create(strategy_name)
        return strategy.get_info()


if __name__ == "__main__":
    # 测试代码
    print("=" * 80)
    print("交易策略系统测试")
    print("=" * 80)

    print("\n可用策略:")
    for name in StrategyFactory.list_strategies():
        info = StrategyFactory.get_strategy_info(name)
        print(f"\n{name}:")
        print(f"  名称: {info['name']}")
        print(f"  描述: {info['description']}")
        print(f"  风险等级: {info['risk_level']}")
        print(f"  配置: {info['config']}")

    # 测试策略决策
    print("\n" + "=" * 80)
    print("策略决策测试")
    print("=" * 80)

    position = Position(
        stock_code='000001',
        shares=1000,
        avg_cost=10.0,
        entry_date='2025-09-01'
    )

    strategies = {
        '保守': ConservativeStrategy(),
        '平衡': BalancedStrategy(),
        '激进': AggressiveStrategy()
    }

    test_prices = [9.0, 10.0, 11.0, 11.5, 12.0]

    for price in test_prices:
        print(f"\n当前价格: {price}, 盈亏: {(price - position.avg_cost) / position.avg_cost * 100:.1f}%")

        for name, strategy in strategies.items():
            profit = strategy.should_take_profit(position, price)
            loss = strategy.should_stop_loss(position, price)

            action = "持有"
            if profit:
                action = f"止盈 ({strategy.config.take_profit_pct*100}%)"
            elif loss:
                action = f"止损 ({strategy.config.stop_loss_pct*100}%)"

            print(f"  {name}策略: {action}")

    print("\n" + "=" * 80)
