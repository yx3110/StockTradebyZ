#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版交易策略 - 市场环境自适应 + 评分加权仓位

改进点:
1. ✅ 根据市场环境自适应调整止盈止损参数
2. ✅ 延长持仓周期，减少频繁调仓
3. ✅ 评分加权仓位分配
4. ✅ 趋势确认机制

Author: Claude Code
Date: 2025-10-16
"""

from trading_strategy import (
    TradingStrategy, StrategyConfig, Position,
    StrategyFactory as BaseStrategyFactory
)
from strategy.market_regime_detector import MarketRegimeDetector
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AdaptiveConservativeStrategy(TradingStrategy):
    """自适应保守策略 - 根据市场环境自动调整参数"""

    def __init__(self):
        # 默认配置（震荡市）
        config = StrategyConfig(
            take_profit_pct=0.15,      # 震荡市默认15%止盈
            stop_loss_pct=0.08,        # 震荡市默认8%止损
            max_holding_days=40,       # 延长至40天
            min_score_for_hold=75.0,
            enable_rebalance_sell=True,
            rebalance_frequency=10,    # 减少调仓频率
            max_positions=8,
            max_position_pct=0.12,
            name="自适应保守策略",
            description="根据市场环境自动调整参数，延长持仓周期，评分加权",
            risk_level="低"
        )
        super().__init__(config)

        # 市场环境识别器
        self.regime_detector = MarketRegimeDetector()
        self.current_regime = 'SIDEWAYS'
        self.regime_info = {}

        # 评分加权配置
        self.enable_score_weighted = True
        self.min_score_for_full_position = 95.0
        self.base_score_for_position = 80.0

    def update_market_regime(self, date: str, lookback_days: int = 17):  # 🎯 v3.1: 15→17天 (折中，避免过度敏感)
        """更新市场环境并调整策略参数"""
        regime, info = self.regime_detector.detect_regime(date, lookback_days)

        if regime != self.current_regime or not self.regime_info:
            logger.info(f"市场环境变化: {self.current_regime} → {regime} (置信度: {info.get('confidence', 0):.1f}%)")
            self.current_regime = regime
            self.regime_info = info

            # 获取自适应参数
            adaptive_params = self.regime_detector.get_adaptive_parameters(regime, 'conservative')

            # 更新策略配置
            self.config.take_profit_pct = adaptive_params['take_profit_pct']
            self.config.stop_loss_pct = adaptive_params['stop_loss_pct']
            self.config.max_holding_days = adaptive_params['max_holding_days']
            self.config.rebalance_frequency = adaptive_params['rebalance_frequency']
            self.config.min_score_for_hold = adaptive_params['min_score_for_hold']

            logger.info(f"策略参数已更新: 止盈{self.config.take_profit_pct*100:.0f}% "
                       f"止损{self.config.stop_loss_pct*100:.0f}% "
                       f"持仓{self.config.max_holding_days}天 "
                       f"调仓频率{self.config.rebalance_frequency}天")

    def should_sell_on_rebalance(self, position, current_price, current_date, selected_stocks):
        """调仓时是否卖出 - 🆕 让利润奔跑，减少频繁换手"""
        selected_codes = {s['stock_code'] for s in selected_stocks}
        selected_scores = {s['stock_code']: s['score'] for s in selected_stocks}

        # 🆕 优先级0: 让利润奔跑 - 如果盈利接近止盈目标，不要卖出
        profit_pct = self.calculate_profit_pct(position, current_price)
        profit_threshold = self.config.take_profit_pct * 0.55  # 🎯 v3.1: 60%→55% (大幅降低，保护20-30%区间盈利)
        if profit_pct >= profit_threshold:
            return False, "profit_running"  # 让利润继续奔跑

        # 🎯 v3.1: 移除5天最小持仓限制（恢复灵活性，避免延误止损）
        # 原v3.0代码已删除

        # 检查1: 不在新选股列表
        if position.stock_code not in selected_codes:
            return True, "rebalance_not_selected"

        # 检查2: 评分下降超过阈值（🆕 从15分提升至20分）
        if position.stock_code in selected_scores:
            current_score = selected_scores[position.stock_code]
            score_drop = position.entry_score - current_score if position.entry_score > 0 else 0

            if score_drop > 20:  # 🆕 从15分提升至20分
                return True, f"rebalance_score_drop_{score_drop:.1f}"

            if current_score < self.config.min_score_for_hold:
                return True, f"rebalance_low_score_{current_score:.1f}"

        # 检查3: 持仓时间超过最大持仓期
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

    def calculate_position_size(self, stock_code, stock_price, available_capital, current_positions, stock_score=None):
        """计算买入股数 - 评分加权"""
        max_new_positions = self.config.max_positions - current_positions
        if max_new_positions <= 0:
            return 0

        # 基础分配
        base_allocation = available_capital / max_new_positions

        # 如果提供了评分，使用评分加权
        if self.enable_score_weighted and stock_score is not None:
            # 评分权重系数 (80-100分 → 0.5-1.5倍)
            if stock_score >= self.min_score_for_full_position:
                score_multiplier = 1.5
            elif stock_score >= 90:
                score_multiplier = 1.2
            elif stock_score >= 85:
                score_multiplier = 1.0
            elif stock_score >= 80:
                score_multiplier = 0.8
            else:
                score_multiplier = 0.5

            base_allocation *= score_multiplier

        # 计算股数 (按手，100股/手)
        shares = int(base_allocation / (stock_price * 100)) * 100

        # 限制单只股票最大仓位
        max_shares = int((available_capital * self.config.max_position_pct) / (stock_price * 100)) * 100
        shares = min(shares, max_shares)

        return shares


class AdaptiveBalancedStrategy(TradingStrategy):
    """自适应平衡策略 - 根据市场环境自动调整参数"""

    def __init__(self):
        config = StrategyConfig(
            take_profit_pct=0.20,
            stop_loss_pct=0.10,
            max_holding_days=30,
            min_score_for_hold=70.0,
            enable_rebalance_sell=True,
            rebalance_frequency=8,
            max_positions=10,
            max_position_pct=0.15,
            name="自适应平衡策略",
            description="根据市场环境自动调整参数，中等持仓周期",
            risk_level="中等"
        )
        super().__init__(config)

        self.regime_detector = MarketRegimeDetector()
        self.current_regime = 'SIDEWAYS'
        self.regime_info = {}
        self.enable_score_weighted = True
        self.min_score_for_full_position = 92.0
        self.base_score_for_position = 75.0

    def update_market_regime(self, date: str, lookback_days: int = 17):  # 🎯 v3.1: 15→17天 (折中，避免过度敏感)
        """更新市场环境并调整策略参数"""
        regime, info = self.regime_detector.detect_regime(date, lookback_days)

        if regime != self.current_regime or not self.regime_info:
            logger.info(f"市场环境变化: {self.current_regime} → {regime}")
            self.current_regime = regime
            self.regime_info = info

            adaptive_params = self.regime_detector.get_adaptive_parameters(regime, 'balanced')

            self.config.take_profit_pct = adaptive_params['take_profit_pct']
            self.config.stop_loss_pct = adaptive_params['stop_loss_pct']
            self.config.max_holding_days = adaptive_params['max_holding_days']
            self.config.rebalance_frequency = adaptive_params['rebalance_frequency']
            self.config.min_score_for_hold = adaptive_params['min_score_for_hold']

    def should_sell_on_rebalance(self, position, current_price, current_date, selected_stocks):
        """调仓时是否卖出 - 🆕 让利润奔跑"""
        selected_codes = {s['stock_code'] for s in selected_stocks}
        selected_scores = {s['stock_code']: s['score'] for s in selected_stocks}

        # 🆕 优先级0: 让利润奔跑
        profit_pct = self.calculate_profit_pct(position, current_price)
        profit_threshold = self.config.take_profit_pct * 0.55  # 🎯 v3.1: 60%→55% (大幅降低，保护20-30%区间盈利)
        if profit_pct >= profit_threshold:
            return False, "profit_running"

        # 🎯 v3.1: 移除5天最小持仓限制（恢复灵活性，避免延误止损）
        # 原v3.0代码已删除

        if position.stock_code not in selected_codes:
            return True, "rebalance_not_selected"

        if position.stock_code in selected_scores:
            current_score = selected_scores[position.stock_code]
            score_drop = position.entry_score - current_score if position.entry_score > 0 else 0

            if score_drop > 22:  # 🆕 从12分提升至22分
                return True, f"rebalance_score_drop_{score_drop:.1f}"

            if current_score < self.config.min_score_for_hold:
                return True, f"rebalance_low_score_{current_score:.1f}"

        holding_days = self.calculate_holding_days(position.entry_date, current_date)
        if holding_days > self.config.max_holding_days:
            return True, f"rebalance_max_holding_{holding_days}d"

        return False, ""

    def should_take_profit(self, position, current_price):
        profit_pct = self.calculate_profit_pct(position, current_price)
        return profit_pct > self.config.take_profit_pct

    def should_stop_loss(self, position, current_price):
        profit_pct = self.calculate_profit_pct(position, current_price)
        return profit_pct < -self.config.stop_loss_pct

    def should_check_holding_period(self, position, current_date):
        holding_days = self.calculate_holding_days(position.entry_date, current_date)
        return holding_days > self.config.max_holding_days

    def calculate_position_size(self, stock_code, stock_price, available_capital, current_positions, stock_score=None):
        """计算买入股数 - 评分加权"""
        max_new_positions = self.config.max_positions - current_positions
        if max_new_positions <= 0:
            return 0

        base_allocation = available_capital / max_new_positions

        if self.enable_score_weighted and stock_score is not None:
            if stock_score >= self.min_score_for_full_position:
                score_multiplier = 1.4
            elif stock_score >= 88:
                score_multiplier = 1.2
            elif stock_score >= 82:
                score_multiplier = 1.0
            elif stock_score >= 75:
                score_multiplier = 0.8
            else:
                score_multiplier = 0.6

            base_allocation *= score_multiplier

        shares = int(base_allocation / (stock_price * 100)) * 100
        max_shares = int((available_capital * self.config.max_position_pct) / (stock_price * 100)) * 100
        shares = min(shares, max_shares)

        return shares


class AdaptiveAggressiveStrategy(TradingStrategy):
    """自适应激进策略 - 根据市场环境自动调整参数"""

    def __init__(self):
        config = StrategyConfig(
            take_profit_pct=0.25,
            stop_loss_pct=0.12,
            max_holding_days=20,
            min_score_for_hold=65.0,
            enable_rebalance_sell=True,
            rebalance_frequency=5,
            max_positions=15,
            max_position_pct=0.10,
            name="自适应激进策略",
            description="根据市场环境自动调整参数，快速轮换",
            risk_level="高"
        )
        super().__init__(config)

        self.regime_detector = MarketRegimeDetector()
        self.current_regime = 'SIDEWAYS'
        self.regime_info = {}
        self.enable_score_weighted = True
        self.min_score_for_full_position = 90.0
        self.base_score_for_position = 70.0

    def update_market_regime(self, date: str, lookback_days: int = 17):  # 🎯 v3.1: 15→17天 (折中，避免过度敏感)
        """更新市场环境并调整策略参数"""
        regime, info = self.regime_detector.detect_regime(date, lookback_days)

        if regime != self.current_regime or not self.regime_info:
            logger.info(f"市场环境变化: {self.current_regime} → {regime}")
            self.current_regime = regime
            self.regime_info = info

            adaptive_params = self.regime_detector.get_adaptive_parameters(regime, 'aggressive')

            self.config.take_profit_pct = adaptive_params['take_profit_pct']
            self.config.stop_loss_pct = adaptive_params['stop_loss_pct']
            self.config.max_holding_days = adaptive_params['max_holding_days']
            self.config.rebalance_frequency = adaptive_params['rebalance_frequency']
            self.config.min_score_for_hold = adaptive_params['min_score_for_hold']

    def should_sell_on_rebalance(self, position, current_price, current_date, selected_stocks):
        """调仓时是否卖出 - 🆕 让利润奔跑"""
        selected_codes = {s['stock_code'] for s in selected_stocks}
        selected_scores = {s['stock_code']: s['score'] for s in selected_stocks}

        # 🆕 优先级0: 让利润奔跑
        profit_pct = self.calculate_profit_pct(position, current_price)
        profit_threshold = self.config.take_profit_pct * 0.55  # 🎯 v3.1: 保持55% (激进策略更早保护)
        if profit_pct >= profit_threshold:
            return False, "profit_running"

        # 🎯 v3.1: 移除5天最小持仓限制（恢复灵活性，避免延误止损）
        # 原v3.0代码已删除

        if position.stock_code not in selected_codes:
            return True, "rebalance_not_selected"

        if position.stock_code in selected_scores:
            current_score = selected_scores[position.stock_code]
            score_drop = position.entry_score - current_score if position.entry_score > 0 else 0

            if score_drop > 18:  # 🆕 从10分提升至18分
                return True, f"rebalance_score_drop_{score_drop:.1f}"

            if current_score < self.config.min_score_for_hold:
                return True, f"rebalance_low_score_{current_score:.1f}"

        holding_days = self.calculate_holding_days(position.entry_date, current_date)
        if holding_days > self.config.max_holding_days:
            return True, f"rebalance_max_holding_{holding_days}d"

        return False, ""

    def should_take_profit(self, position, current_price):
        profit_pct = self.calculate_profit_pct(position, current_price)
        return profit_pct > self.config.take_profit_pct

    def should_stop_loss(self, position, current_price):
        profit_pct = self.calculate_profit_pct(position, current_price)
        return profit_pct < -self.config.stop_loss_pct

    def should_check_holding_period(self, position, current_date):
        holding_days = self.calculate_holding_days(position.entry_date, current_date)
        return holding_days > self.config.max_holding_days

    def calculate_position_size(self, stock_code, stock_price, available_capital, current_positions, stock_score=None):
        """计算买入股数 - 评分加权"""
        max_new_positions = self.config.max_positions - current_positions
        if max_new_positions <= 0:
            return 0

        base_allocation = available_capital / max_new_positions

        if self.enable_score_weighted and stock_score is not None:
            if stock_score >= self.min_score_for_full_position:
                score_multiplier = 1.3
            elif stock_score >= 85:
                score_multiplier = 1.1
            elif stock_score >= 78:
                score_multiplier = 1.0
            elif stock_score >= 70:
                score_multiplier = 0.9
            else:
                score_multiplier = 0.7

            base_allocation *= score_multiplier

        shares = int(base_allocation / (stock_price * 100)) * 100
        max_shares = int((available_capital * self.config.max_position_pct) / (stock_price * 100)) * 100
        shares = min(shares, max_shares)

        return shares


# ========== 改进版策略工厂 ==========

class ImprovedStrategyFactory:
    """改进版策略工厂"""

    _strategies = {
        'conservative': AdaptiveConservativeStrategy,
        'balanced': AdaptiveBalancedStrategy,
        'aggressive': AdaptiveAggressiveStrategy
    }

    @classmethod
    def create(cls, strategy_name: str) -> TradingStrategy:
        """创建改进版策略实例"""
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
    # 测试改进版策略
    print("=" * 80)
    print("改进版交易策略测试")
    print("=" * 80)

    print("\n可用策略:")
    for name in ImprovedStrategyFactory.list_strategies():
        info = ImprovedStrategyFactory.get_strategy_info(name)
        print(f"\n{name}:")
        print(f"  名称: {info['name']}")
        print(f"  描述: {info['description']}")
        print(f"  风险等级: {info['risk_level']}")
        print(f"  止盈: {info['config']['take_profit_pct']*100:.0f}%")
        print(f"  止损: {info['config']['stop_loss_pct']*100:.0f}%")
        print(f"  最长持仓: {info['config']['max_holding_days']}天")

    # 测试市场环境自适应
    print("\n" + "=" * 80)
    print("市场环境自适应测试")
    print("=" * 80)

    strategy = AdaptiveConservativeStrategy()
    test_dates = ['2025-07-01', '2025-08-01', '2025-09-01']

    for date in test_dates:
        strategy.update_market_regime(date)
        print(f"\n{date}: {strategy.current_regime}")
        print(f"  止盈: {strategy.config.take_profit_pct*100:.0f}%")
        print(f"  止损: {strategy.config.stop_loss_pct*100:.0f}%")
        print(f"  最长持仓: {strategy.config.max_holding_days}天")
        print(f"  调仓频率: {strategy.config.rebalance_frequency}天")

    print("\n" + "=" * 80)
