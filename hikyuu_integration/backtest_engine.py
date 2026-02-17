#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HikyuuStyleBacktestEngine - Hikyuu风格回测引擎

借鉴Hikyuu的回测框架设计，提供快速、灵活的回测能力
"""

from typing import List, Dict, Optional, Callable
from datetime import datetime
import pandas as pd
import logging

from .data_adapter import HikyuuStyleDataAdapter
from .query import Query
from .signal_base import SignalBase
from .money_manager import MoneyManagerBase, MM_FixedPercent
from .stop_loss import StopLossBase
from .portfolio import Portfolio
from .broker import Broker

logger = logging.getLogger(__name__)


class BacktestResult:
    """
    回测结果

    包含回测统计指标和交易记录
    """

    def __init__(self, portfolio: Portfolio, start_date: str, end_date: str):
        """
        参数:
            portfolio: 组合对象
            start_date: 回测开始日期
            end_date: 回测结束日期
        """
        self.portfolio = portfolio
        self.start_date = start_date
        self.end_date = end_date

        # 计算统计指标
        self._calculate_metrics()

    def _calculate_metrics(self):
        """计算回测指标"""
        stats = self.portfolio.get_stats()

        self.initial_cash = stats['initial_cash']
        self.final_value = stats['total_value']
        # 修复：total_return应该是百分比，total_pnl是金额
        self.total_return = stats['total_pnl_pct']  # 百分比
        self.total_pnl = stats['total_pnl']  # 盈亏金额
        self.total_commission = stats['total_commission']
        self.trade_count = stats['trade_count']

        # 计算年化收益率
        self.annualized_return = self._calculate_annualized_return()

        # 计算夏普比率（需要历史价值数据）
        self.sharpe_ratio = self._calculate_sharpe_ratio()

        # 计算最大回撤
        self.max_drawdown, self.max_drawdown_pct = self._calculate_max_drawdown()

        # 胜率
        self.win_rate = self._calculate_win_rate()

    def _calculate_annualized_return(self) -> float:
        """
        计算年化收益率

        年化收益率 = (期末价值 / 期初价值) ^ (365 / 天数) - 1
        """
        try:
            # 计算交易天数（简化：按日期字符串差值估算）
            from datetime import datetime
            start = datetime.strptime(self.start_date.replace('-', ''), '%Y%m%d')
            end = datetime.strptime(self.end_date.replace('-', ''), '%Y%m%d')
            days = (end - start).days

            if days <= 0:
                return 0.0

            # 年化收益率
            total_return_ratio = self.final_value / self.initial_cash
            annualized = (total_return_ratio ** (365.0 / days) - 1) * 100

            return annualized

        except Exception as e:
            logger.warning(f"计算年化收益率失败: {e}")
            return 0.0

    def _calculate_sharpe_ratio(self) -> float:
        """
        计算夏普比率

        Sharpe Ratio = (年化收益率 - 无风险利率) / 年化波动率

        简化实现：使用组合价值历史计算
        """
        try:
            if not self.portfolio.value_history:
                return 0.0

            # 计算日收益率
            values = list(self.portfolio.value_history.values())
            if len(values) < 2:
                return 0.0

            returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]

            # 年化收益率
            mean_return = sum(returns) / len(returns) * 252  # 252个交易日

            # 年化波动率
            variance = sum((r - mean_return/252) ** 2 for r in returns) / len(returns)
            std_return = (variance ** 0.5) * (252 ** 0.5)

            if std_return == 0:
                return 0.0

            # 夏普比率（假设无风险利率为3%）
            risk_free_rate = 0.03
            sharpe = (mean_return - risk_free_rate) / std_return

            return sharpe

        except Exception as e:
            logger.warning(f"计算夏普比率失败: {e}")
            return 0.0

    def _calculate_max_drawdown(self) -> tuple:
        """
        计算最大回撤

        返回:
            (最大回撤金额, 最大回撤百分比)
        """
        try:
            if not self.portfolio.value_history:
                return 0.0, 0.0

            values = list(self.portfolio.value_history.values())

            max_value = values[0]
            max_dd = 0.0
            max_dd_pct = 0.0

            for value in values:
                if value > max_value:
                    max_value = value

                dd = max_value - value
                dd_pct = (dd / max_value) * 100

                if dd > max_dd:
                    max_dd = dd
                    max_dd_pct = dd_pct

            return max_dd, max_dd_pct

        except Exception as e:
            logger.warning(f"计算最大回撤失败: {e}")
            return 0.0, 0.0

    def _calculate_win_rate(self) -> float:
        """
        计算胜率

        胜率 = 盈利交易数 / 总交易数
        """
        try:
            trades = self.portfolio.trades

            # 匹配买入卖出计算盈亏
            sell_trades = [t for t in trades if t.action == 'SELL']

            if not sell_trades:
                return 0.0

            win_count = 0

            for sell_trade in sell_trades:
                # 找到对应的买入交易
                buy_trades = [t for t in trades
                             if t.action == 'BUY'
                             and t.stock_code == sell_trade.stock_code
                             and t.date <= sell_trade.date]

                if not buy_trades:
                    continue

                # 取最近的买入价格
                buy_price = buy_trades[-1].price
                sell_price = sell_trade.price

                if sell_price > buy_price:
                    win_count += 1

            win_rate = (win_count / len(sell_trades)) * 100

            return win_rate

        except Exception as e:
            logger.warning(f"计算胜率失败: {e}")
            return 0.0

    def print_summary(self):
        """打印回测摘要"""
        print("\n" + "=" * 80)
        print("📈 回测结果摘要")
        print("=" * 80)
        print(f"回测区间: {self.start_date} → {self.end_date}")
        print(f"\n资金情况:")
        print(f"  初始资金: {self.initial_cash:,.2f}")
        print(f"  期末资产: {self.final_value:,.2f}")
        print(f"  总收益:   {self.total_pnl:,.2f} ({self.total_return:.2f}%)")
        print(f"  年化收益: {self.annualized_return:.2f}%")
        print(f"\n风险指标:")
        print(f"  最大回撤: {self.max_drawdown:,.2f} ({self.max_drawdown_pct:.2f}%)")
        print(f"  夏普比率: {self.sharpe_ratio:.2f}")
        print(f"\n交易统计:")
        print(f"  交易次数: {self.trade_count}")
        print(f"  总手续费: {self.total_commission:,.2f}")
        print(f"  胜率:     {self.win_rate:.2f}%")
        print("=" * 80 + "\n")

        # 打印组合摘要
        self.portfolio.print_summary()

    def get_trades_df(self):
        """获取交易记录DataFrame"""
        return self.portfolio.get_trades_df()

    def __repr__(self):
        return (f"BacktestResult({self.start_date}→{self.end_date}, "
                f"收益={self.total_return:.2f}%, "
                f"年化={self.annualized_return:.2f}%, "
                f"夏普={self.sharpe_ratio:.2f})")


class HikyuuStyleBacktestEngine:
    """
    Hikyuu风格回测引擎

    核心组件:
    - Signal: 买卖信号生成
    - MoneyManager: 资金管理
    - StopLoss: 止损止盈
    - Broker: 交易执行
    - Portfolio: 组合管理

    示例:
        # 创建回测引擎
        engine = HikyuuStyleBacktestEngine(
            data_adapter=adapter,
            signal=MLScoringSignal(ml_version='v3.81'),
            money_manager=MM_FixedPercent(0.1),
            stop_loss=ST_FixedPercent(0.08)
        )

        # 运行回测
        result = engine.run(
            stock_list=['000001', '000002'],
            start_date='2024-01-01',
            end_date='2025-09-30'
        )

        # 查看结果
        result.print_summary()
    """

    def __init__(self,
                 data_adapter: HikyuuStyleDataAdapter,
                 signal: SignalBase,
                 money_manager: Optional[MoneyManagerBase] = None,
                 stop_loss: Optional[StopLossBase] = None,
                 initial_cash: float = 100000,
                 max_positions: int = 10,
                 enable_t1: bool = True,
                 enable_limit_check: bool = True):
        """
        初始化回测引擎

        参数:
            data_adapter: 数据适配器
            signal: 信号生成器
            money_manager: 资金管理器（默认10%固定比例）
            stop_loss: 止损策略（默认无止损）
            initial_cash: 初始资金
            max_positions: 最大持仓数
            enable_t1: 是否启用T+1规则
            enable_limit_check: 是否检查涨跌停
        """
        self.data_adapter = data_adapter
        self.signal = signal
        self.money_manager = money_manager or MM_FixedPercent(0.1)
        self.stop_loss = stop_loss
        self.initial_cash = initial_cash
        self.max_positions = max_positions
        self.enable_t1 = enable_t1
        self.enable_limit_check = enable_limit_check

        # 初始化组合和Broker
        self.portfolio = Portfolio(initial_cash=initial_cash)
        self.broker = Broker(
            portfolio=self.portfolio,
            data_adapter=data_adapter,
            enable_t1=enable_t1,
            enable_limit_check=enable_limit_check
        )

        logger.info(f"初始化回测引擎: {self}")

    def run(self,
            stock_list: List[str],
            start_date: str,
            end_date: str,
            on_bar: Optional[Callable] = None) -> BacktestResult:
        """
        运行回测

        参数:
            stock_list: 股票列表
            start_date: 开始日期
            end_date: 结束日期
            on_bar: 每日回调函数（可选，用于自定义逻辑）

        返回:
            BacktestResult对象
        """
        logger.info(f"开始回测: {len(stock_list)}只股票, {start_date} → {end_date}")

        # 预加载数据
        logger.info("预加载数据...")
        self.data_adapter.preload_data(stock_list, start_date, end_date)

        # 计算所有股票的信号
        logger.info("计算信号...")
        self._calculate_signals(stock_list, start_date, end_date)

        # 获取交易日历
        trading_dates = self._get_trading_dates(start_date, end_date)
        logger.info(f"交易日: {len(trading_dates)}天")

        # 逐日回测
        for i, date in enumerate(trading_dates):
            logger.debug(f"\n[{i+1}/{len(trading_dates)}] {date}")

            # 更新日期
            self.broker.update_date(date)

            # 更新持仓价格
            self._update_positions_prices(date)

            # 检查止损止盈
            if self.stop_loss:
                self._check_stop_loss(date)

            # 检查卖出信号
            self._check_sell_signals(date)

            # 检查买入信号
            self._check_buy_signals(date, stock_list)

            # 记录组合价值
            self.portfolio.record_value(date)

            # 用户自定义回调
            if on_bar:
                on_bar(date, self.portfolio, self.broker)

            # 定期输出进度
            if (i + 1) % 50 == 0:
                logger.info(f"进度: {i+1}/{len(trading_dates)}, "
                           f"总资产={self.portfolio.get_total_value():.2f}, "
                           f"盈亏={self.portfolio.get_total_pnl_pct():.2f}%")

        # 清空所有持仓（回测结束）
        self._close_all_positions(trading_dates[-1])

        # 生成回测结果
        result = BacktestResult(self.portfolio, start_date, end_date)

        logger.info(f"回测完成: {result}")

        return result

    def _calculate_signals(self, stock_list: List[str], start_date: str, end_date: str):
        """
        预先计算所有股票的信号

        参数:
            stock_list: 股票列表
            start_date: 开始日期
            end_date: 结束日期
        """
        # 为每只股票计算信号
        # 注意：Signal是单例，需要为每只股票分别计算
        for stock_code in stock_list:
            try:
                kdata = self.data_adapter.get_kdata(stock_code, Query(start=start_date, end=end_date))

                if len(kdata) == 0:
                    logger.warning(f"{stock_code}: 无数据")
                    continue

                # 计算信号
                self.signal.calculate(kdata)

                logger.debug(f"{stock_code}: 信号计算完成")

            except Exception as e:
                logger.warning(f"{stock_code}: 信号计算失败 - {e}")

    def _get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """
        获取交易日历

        参数:
            start_date: 开始日期
            end_date: 结束日期

        返回:
            交易日列表
        """
        # 从数据库查询交易日（使用任意股票的交易日期）
        try:
            # 查询000001的交易日期作为基准
            kdata = self.data_adapter.get_kdata('000001', Query(start=start_date, end=end_date))

            if len(kdata) == 0:
                logger.warning("无法获取交易日历，使用空列表")
                return []

            # 返回日期列表
            dates = [dt.strftime('%Y-%m-%d') if isinstance(dt, datetime) else dt
                    for dt in kdata.datetime]

            return dates

        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            return []

    def _update_positions_prices(self, date: str):
        """
        更新持仓价格

        参数:
            date: 日期
        """
        positions = self.portfolio.get_all_positions()

        if not positions:
            return

        prices = {}

        for pos in positions:
            try:
                kdata = self.data_adapter.get_kdata(pos.stock_code, Query(start=date, end=date))

                if len(kdata) > 0:
                    prices[pos.stock_code] = kdata.close[0]

            except Exception as e:
                logger.warning(f"获取{pos.stock_code}价格失败: {e}")

        self.portfolio.update_prices(prices)

    def _check_stop_loss(self, date: str):
        """
        检查止损止盈

        参数:
            date: 日期
        """
        positions = self.portfolio.get_all_positions()

        for pos in positions:
            # 检查是否触发止损
            if self.stop_loss.should_stop(date, pos.entry_price, pos.current_price, pos.entry_date):
                logger.info(f"{date}: {pos.stock_code} 触发止损")
                self.broker.sell(pos.stock_code, pos.shares, date, reason='止损')

    def _check_sell_signals(self, date: str):
        """
        检查卖出信号

        参数:
            date: 日期
        """
        positions = self.portfolio.get_all_positions()

        for pos in positions:
            # 检查卖出信号
            if self.signal.should_sell(date):
                logger.info(f"{date}: {pos.stock_code} 卖出信号")
                self.broker.sell(pos.stock_code, pos.shares, date, reason='卖出信号')

    def _check_buy_signals(self, date: str, stock_list: List[str]):
        """
        检查买入信号

        参数:
            date: 日期
            stock_list: 候选股票列表
        """
        # 检查持仓数量
        if self.portfolio.get_position_count() >= self.max_positions:
            logger.debug(f"{date}: 持仓已满({self.max_positions})")
            return

        # 遍历候选股票
        for stock_code in stock_list:
            # 已持有，跳过
            if self.portfolio.has_position(stock_code):
                continue

            # 检查买入信号
            if self.signal.should_buy(date):
                # 获取当前价格
                try:
                    kdata = self.data_adapter.get_kdata(stock_code, Query(start=date, end=date))

                    if len(kdata) == 0:
                        continue

                    price = kdata.close[0]

                    # 计算买入数量
                    shares = self.money_manager.get_buy_num(
                        date, stock_code, price, self.portfolio.cash
                    )

                    if shares > 0:
                        logger.info(f"{date}: {stock_code} 买入信号, {shares}股@{price:.2f}")
                        success = self.broker.buy(stock_code, shares, date, reason='买入信号')

                        if success:
                            # 持仓已满，退出
                            if self.portfolio.get_position_count() >= self.max_positions:
                                break

                except Exception as e:
                    logger.warning(f"{date}: {stock_code} 买入失败 - {e}")

    def _close_all_positions(self, date: str):
        """
        清空所有持仓（回测结束）

        参数:
            date: 日期
        """
        positions = self.portfolio.get_all_positions()

        logger.info(f"{date}: 回测结束，清空{len(positions)}个持仓")

        for pos in positions:
            self.broker.sell(pos.stock_code, pos.shares, date, reason='回测结束')

    def __repr__(self):
        return (f"HikyuuStyleBacktestEngine("
                f"signal={self.signal.name}, "
                f"mm={self.money_manager.name}, "
                f"sl={self.stop_loss.name if self.stop_loss else 'None'}, "
                f"cash={self.initial_cash}, "
                f"max_pos={self.max_positions})")
