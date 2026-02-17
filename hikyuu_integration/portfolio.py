#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio - 组合管理类

借鉴Hikyuu的Portfolio设计，管理持仓、现金、交易记录
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """
    持仓信息

    属性:
        stock_code: 股票代码
        shares: 持股数量
        entry_price: 买入均价
        entry_date: 买入日期
        current_price: 当前价格
        current_value: 当前市值
        cost: 成本
        pnl: 盈亏
        pnl_pct: 盈亏百分比
    """
    stock_code: str
    shares: int
    entry_price: float
    entry_date: str
    current_price: float = 0.0

    @property
    def cost(self) -> float:
        """成本"""
        return self.shares * self.entry_price

    @property
    def current_value(self) -> float:
        """当前市值"""
        return self.shares * self.current_price

    @property
    def pnl(self) -> float:
        """盈亏金额"""
        return self.current_value - self.cost

    @property
    def pnl_pct(self) -> float:
        """盈亏百分比"""
        if self.cost == 0:
            return 0.0
        return (self.pnl / self.cost) * 100

    def update_price(self, price: float):
        """更新当前价格"""
        self.current_price = price

    def __repr__(self):
        return (f"Position({self.stock_code}, {self.shares}股, "
                f"成本={self.entry_price:.2f}, "
                f"现价={self.current_price:.2f}, "
                f"盈亏={self.pnl_pct:.2f}%)")


@dataclass
class Trade:
    """
    交易记录

    属性:
        trade_id: 交易ID
        stock_code: 股票代码
        action: 买入/卖出 ('BUY'/'SELL')
        date: 交易日期
        price: 交易价格
        shares: 交易股数
        amount: 交易金额
        commission: 手续费
        reason: 交易原因（信号触发、止损、止盈等）
        pnl: 盈亏金额（仅SELL交易有效，BUY为0）
        entry_price: 买入均价（仅SELL交易记录，用于计算盈亏）
    """
    trade_id: int
    stock_code: str
    action: str  # 'BUY' or 'SELL'
    date: str
    price: float
    shares: int
    amount: float
    commission: float = 0.0
    reason: str = ''
    pnl: float = 0.0
    entry_price: float = 0.0

    def __repr__(self):
        return (f"Trade({self.action} {self.stock_code} {self.shares}股 "
                f"@{self.price:.2f} on {self.date})")


class Portfolio:
    """
    组合管理类

    管理现金、持仓、交易记录，计算组合绩效

    示例:
        portfolio = Portfolio(initial_cash=100000)

        # 买入
        portfolio.buy('000001', 1000, 10.0, '2025-09-30', reason='BBI信号')

        # 更新价格
        portfolio.update_prices({'000001': 11.0})

        # 卖出
        portfolio.sell('000001', 1000, 11.0, '2025-10-01', reason='止盈')

        # 查询绩效
        stats = portfolio.get_stats()
    """

    def __init__(self,
                 initial_cash: float = 100000,
                 commission_rate: float = 0.0003,
                 min_commission: float = 5.0,
                 stamp_tax_rate: float = 0.001):
        """
        初始化组合

        参数:
            initial_cash: 初始资金
            commission_rate: 手续费率（默认万三）
            min_commission: 最低手续费（默认5元）
            stamp_tax_rate: 印花税率（默认千一，仅卖出时收取）
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_tax_rate = stamp_tax_rate

        # 持仓: {stock_code: Position}
        self.positions: Dict[str, Position] = {}

        # 交易历史
        self.trades: List[Trade] = []
        self._trade_counter = 0

        # 历史价值记录: {date: total_value}
        self.value_history: Dict[str, float] = {}

        logger.info(f"初始化组合: 现金={initial_cash:.2f}")

    def _calculate_commission(self, amount: float, is_sell: bool = False) -> float:
        """
        计算手续费和税费

        参数:
            amount: 交易金额
            is_sell: 是否卖出（卖出需要收取印花税）

        返回:
            总手续费（包括佣金和印花税）
        """
        # 佣金
        commission = max(amount * self.commission_rate, self.min_commission)

        # 印花税（仅卖出时收取）
        if is_sell:
            stamp_tax = amount * self.stamp_tax_rate
            commission += stamp_tax

        return commission

    def buy(self,
            stock_code: str,
            shares: int,
            price: float,
            date: str,
            reason: str = '') -> bool:
        """
        买入股票

        参数:
            stock_code: 股票代码
            shares: 买入股数（必须是100的整数倍）
            price: 买入价格
            date: 交易日期
            reason: 买入原因

        返回:
            是否成功买入
        """
        if shares % 100 != 0:
            logger.warning(f"买入股数必须是100的整数倍: {shares}")
            return False

        amount = shares * price
        commission = self._calculate_commission(amount, is_sell=False)
        total_cost = amount + commission

        # 检查现金是否足够
        if total_cost > self.cash:
            logger.warning(f"现金不足: 需要{total_cost:.2f}, 可用{self.cash:.2f}")
            return False

        # 扣除现金
        self.cash -= total_cost

        # 更新持仓
        if stock_code in self.positions:
            # 加仓：重新计算平均成本
            old_pos = self.positions[stock_code]
            total_shares = old_pos.shares + shares
            total_cost_basis = old_pos.cost + amount
            avg_price = total_cost_basis / total_shares

            old_pos.shares = total_shares
            old_pos.entry_price = avg_price

            logger.debug(f"加仓 {stock_code}: {shares}股 @{price:.2f}, "
                        f"新均价={avg_price:.2f}")
        else:
            # 新建仓位
            self.positions[stock_code] = Position(
                stock_code=stock_code,
                shares=shares,
                entry_price=price,
                entry_date=date,
                current_price=price
            )
            logger.debug(f"建仓 {stock_code}: {shares}股 @{price:.2f}")

        # 记录交易
        self._trade_counter += 1
        trade = Trade(
            trade_id=self._trade_counter,
            stock_code=stock_code,
            action='BUY',
            date=date,
            price=price,
            shares=shares,
            amount=amount,
            commission=commission,
            reason=reason
        )
        self.trades.append(trade)

        logger.info(f"买入: {trade}, 剩余现金={self.cash:.2f}")
        return True

    def sell(self,
             stock_code: str,
             shares: int,
             price: float,
             date: str,
             reason: str = '') -> bool:
        """
        卖出股票

        参数:
            stock_code: 股票代码
            shares: 卖出股数（必须是100的整数倍）
            price: 卖出价格
            date: 交易日期
            reason: 卖出原因

        返回:
            是否成功卖出
        """
        if shares % 100 != 0:
            logger.warning(f"卖出股数必须是100的整数倍: {shares}")
            return False

        # 检查是否持仓
        if stock_code not in self.positions:
            logger.warning(f"没有持仓: {stock_code}")
            return False

        position = self.positions[stock_code]

        # 检查股数是否足够
        if shares > position.shares:
            logger.warning(f"股数不足: 持有{position.shares}, 要卖{shares}")
            return False

        amount = shares * price
        commission = self._calculate_commission(amount, is_sell=True)
        net_proceeds = amount - commission

        # 增加现金
        self.cash += net_proceeds

        # 更新持仓
        position.shares -= shares

        if position.shares == 0:
            # 清仓
            del self.positions[stock_code]
            logger.debug(f"清仓 {stock_code}: {shares}股 @{price:.2f}")
        else:
            logger.debug(f"减仓 {stock_code}: {shares}股 @{price:.2f}, "
                        f"剩余{position.shares}股")

        # 计算盈亏
        pnl = (price - position.entry_price) * shares - commission
        pnl_pct = (pnl / (position.entry_price * shares)) * 100

        # 记录交易
        self._trade_counter += 1
        trade = Trade(
            trade_id=self._trade_counter,
            stock_code=stock_code,
            action='SELL',
            date=date,
            price=price,
            shares=shares,
            amount=amount,
            commission=commission,
            reason=reason,
            pnl=pnl,
            entry_price=position.entry_price
        )
        self.trades.append(trade)

        logger.info(f"卖出: {trade}, 盈亏={pnl:.2f} ({pnl_pct:.2f}%), "
                   f"现金={self.cash:.2f}")

        return True

    def update_prices(self, prices: Dict[str, float]):
        """
        更新持仓价格

        参数:
            prices: {stock_code: current_price}
        """
        for stock_code, price in prices.items():
            if stock_code in self.positions:
                self.positions[stock_code].update_price(price)

    def record_value(self, date: str):
        """
        记录当日组合总价值

        参数:
            date: 日期
        """
        total_value = self.get_total_value()
        self.value_history[date] = total_value

        logger.debug(f"{date}: 组合总价值={total_value:.2f}")

    def get_position(self, stock_code: str) -> Optional[Position]:
        """获取持仓"""
        return self.positions.get(stock_code)

    def has_position(self, stock_code: str) -> bool:
        """是否持有某股票"""
        return stock_code in self.positions

    def get_all_positions(self) -> List[Position]:
        """获取所有持仓"""
        return list(self.positions.values())

    def get_position_count(self) -> int:
        """获取持仓数量"""
        return len(self.positions)

    def get_market_value(self) -> float:
        """获取市值（所有持仓的当前价值）"""
        return sum(pos.current_value for pos in self.positions.values())

    def get_total_value(self) -> float:
        """获取总资产（现金 + 市值）"""
        return self.cash + self.get_market_value()

    def get_total_pnl(self) -> float:
        """获取总盈亏"""
        return self.get_total_value() - self.initial_cash

    def get_total_pnl_pct(self) -> float:
        """获取总盈亏百分比"""
        return (self.get_total_pnl() / self.initial_cash) * 100

    def get_stats(self) -> Dict:
        """
        获取组合统计信息

        返回:
            统计字典
        """
        total_value = self.get_total_value()
        market_value = self.get_market_value()

        # 计算总手续费
        total_commission = sum(t.commission for t in self.trades)

        # 计算已实现盈亏（从已平仓交易计算）
        realized_pnl = self._calculate_realized_pnl()

        # 计算未实现盈亏（持仓盈亏）
        unrealized_pnl = sum(pos.pnl for pos in self.positions.values())

        return {
            'initial_cash': self.initial_cash,
            'cash': self.cash,
            'market_value': market_value,
            'total_value': total_value,
            'total_pnl': self.get_total_pnl(),
            'total_pnl_pct': self.get_total_pnl_pct(),
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_commission': total_commission,
            'position_count': self.get_position_count(),
            'trade_count': len(self.trades)
        }

    def _calculate_realized_pnl(self) -> float:
        """
        计算已实现盈亏

        遍历交易记录，匹配买入卖出计算盈亏
        """
        # 简化实现：仅计算卖出金额 - 买入金额 - 手续费
        buy_amount = sum(t.amount + t.commission for t in self.trades if t.action == 'BUY')
        sell_amount = sum(t.amount - t.commission for t in self.trades if t.action == 'SELL')

        return sell_amount - buy_amount

    def get_trades_df(self):
        """
        获取交易记录DataFrame（如果需要）

        返回:
            pd.DataFrame
        """
        import pandas as pd

        if not self.trades:
            return pd.DataFrame()

        data = [
            {
                'trade_id': t.trade_id,
                'stock_code': t.stock_code,
                'action': t.action,
                'date': t.date,
                'price': t.price,
                'shares': t.shares,
                'amount': t.amount,
                'commission': t.commission,
                'reason': t.reason
            }
            for t in self.trades
        ]

        return pd.DataFrame(data)

    def print_summary(self):
        """打印组合摘要"""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("📊 组合摘要")
        print("=" * 60)
        print(f"初始资金: {stats['initial_cash']:,.2f}")
        print(f"现金余额: {stats['cash']:,.2f}")
        print(f"持仓市值: {stats['market_value']:,.2f}")
        print(f"总资产:   {stats['total_value']:,.2f}")
        print(f"总盈亏:   {stats['total_pnl']:,.2f} ({stats['total_pnl_pct']:.2f}%)")
        print(f"已实现盈亏: {stats['realized_pnl']:,.2f}")
        print(f"未实现盈亏: {stats['unrealized_pnl']:,.2f}")
        print(f"总手续费: {stats['total_commission']:,.2f}")
        print(f"持仓数:   {stats['position_count']}")
        print(f"交易次数: {stats['trade_count']}")

        if self.positions:
            print("\n持仓明细:")
            for pos in self.positions.values():
                print(f"  {pos}")

        print("=" * 60 + "\n")

    def __repr__(self):
        return (f"Portfolio(现金={self.cash:.2f}, "
                f"市值={self.get_market_value():.2f}, "
                f"总资产={self.get_total_value():.2f}, "
                f"盈亏={self.get_total_pnl_pct():.2f}%)")
