#!/usr/bin/env python3
"""
Backtrader集成模块
提供与backtrader框架的完整集成接口
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from data_access import StockDataDAO, BacktraderDataAdapter
try:
    from .database_manager import DatabaseManager
except ImportError:
    from database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class DatabaseDataFeed(bt.feeds.PandasData):
    """
    从数据库读取数据的Backtrader数据源
    """
    
    # 定义数据列映射
    lines = ('is_limit_up', 'is_limit_down', 'is_st')  # 扩展数据线
    
    # 参数配置
    params = (
        ('datetime', None),  # 日期列索引
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', -1),  # 不使用持仓量
        ('is_limit_up', 'is_limit_up'),  # A股特有：涨停标记
        ('is_limit_down', 'is_limit_down'),  # A股特有：跌停标记
        ('is_st', 'is_st'),  # A股特有：ST标记
    )


class ChinaStockDataFeed(DatabaseDataFeed):
    """
    中国股票市场专用数据源
    包含A股特有的涨跌停、ST等标记
    """
    
    def __init__(self, dao: StockDataDAO, stock_code: str, 
                 start_date: str, end_date: str, **kwargs):
        """
        初始化中国股票数据源
        
        Args:
            dao: 数据访问对象
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
        """
        self.dao = dao
        self.stock_code = stock_code
        self.start_date = start_date
        self.end_date = end_date
        
        # 从数据库获取数据
        data = self._load_data()
        
        if data.empty:
            raise ValueError(f"股票 {stock_code} 在 {start_date} 至 {end_date} 期间无数据")
        
        # 调用父类初始化
        super().__init__(dataname=data, **kwargs)
    
    def _load_data(self) -> pd.DataFrame:
        """从数据库加载数据"""
        fields = ['open', 'high', 'low', 'close', 'volume', 
                 'is_limit_up', 'is_limit_down', 'is_st']
        
        data = self.dao.get_stock_data(
            self.stock_code, 
            self.start_date, 
            self.end_date, 
            fields
        )
        
        if not data.empty:
            # 添加openinterest列（backtrader需要）
            data['openinterest'] = 0
            
            # 确保布尔类型列
            data['is_limit_up'] = data['is_limit_up'].astype(bool)
            data['is_limit_down'] = data['is_limit_down'].astype(bool)
            data['is_st'] = data.get('is_st', False).astype(bool)
            
            # 重置索引，确保datetime作为列
            data = data.reset_index()
            data.rename(columns={'trade_date': 'datetime'}, inplace=True)
            data.set_index('datetime', inplace=True)
        
        return data


class ChinaCommissionInfo(bt.CommInfoBase):
    """
    中国A股交易佣金模型
    包含佣金、印花税、过户费等
    """
    
    params = (
        ('commission', 0.0003),  # 佣金率：万分之三
        ('stamp_tax', 0.001),    # 印花税：千分之一（仅卖出）
        ('transfer_fee', 0.00002),  # 过户费：万分之二（双向）
        ('min_commission', 5.0),    # 最低佣金：5元
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
    )
    
    def _getcommission(self, size, price, pseudoexec):
        """
        计算交易佣金
        
        Args:
            size: 交易数量（正数买入，负数卖出）
            price: 交易价格
            pseudoexec: 是否为模拟执行
            
        Returns:
            总交易成本
        """
        trade_value = abs(size) * price
        
        # 佣金（双向）
        commission = max(trade_value * self.p.commission, self.p.min_commission)
        
        # 过户费（双向）
        transfer_fee = trade_value * self.p.transfer_fee
        
        # 印花税（仅卖出）
        stamp_tax = 0.0
        if size < 0:  # 卖出
            stamp_tax = trade_value * self.p.stamp_tax
        
        total_cost = commission + transfer_fee + stamp_tax
        
        return total_cost


class ChinaSizer(bt.Sizer):
    """
    中国A股专用资金管理器
    支持整手交易（100股的倍数）
    """
    
    params = (
        ('stake', 100),  # 基础手数
        ('max_position_pct', 0.10),  # 单股最大仓位比例
    )
    
    def _getsizing(self, comminfo, cash, data, isbuy):
        """
        计算交易数量
        
        Args:
            comminfo: 佣金信息
            cash: 可用现金
            data: 股票数据
            isbuy: 是否为买入
            
        Returns:
            交易股数
        """
        if not isbuy:
            # 卖出时返回当前持仓
            position = self.broker.getposition(data)
            return position.size
        
        # 买入时计算可买数量
        price = data.close[0]
        max_value = cash * self.p.max_position_pct
        
        # 计算可买股数（整手）
        max_shares = int(max_value / price / 100) * 100
        
        return max_shares if max_shares >= 100 else 0


class ChinaTradeFilter:
    """
    中国A股交易规则过滤器
    处理涨跌停、T+1等特殊规则
    """
    
    @staticmethod
    def can_buy(data, today_idx: int) -> bool:
        """
        检查是否可以买入
        
        Args:
            data: 股票数据
            today_idx: 当前日期索引
            
        Returns:
            是否可以买入
        """
        # 涨停不能买入
        if hasattr(data, 'is_limit_up') and data.is_limit_up[today_idx]:
            return False
        
        # ST股票限制（可选）
        if hasattr(data, 'is_st') and data.is_st[today_idx]:
            return False
        
        return True
    
    @staticmethod
    def can_sell(data, today_idx: int) -> bool:
        """
        检查是否可以卖出
        
        Args:
            data: 股票数据
            today_idx: 当前日期索引
            
        Returns:
            是否可以卖出
        """
        # 跌停不能卖出
        if hasattr(data, 'is_limit_down') and data.is_limit_down[today_idx]:
            return False
        
        return True


class ChinaStockStrategy(bt.Strategy):
    """
    中国A股策略基类
    集成A股特有的交易规则和风险控制
    """
    
    params = (
        ('t1_trading', True),  # T+1交易规则
        ('respect_limits', True),  # 遵守涨跌停限制
    )
    
    def __init__(self):
        """初始化策略"""
        super().__init__()
        self.trade_filter = ChinaTradeFilter()
        self.t1_positions = {}  # T+1持仓记录
    
    def next(self):
        """策略逻辑（需要在子类中实现）"""
        pass
    
    def buy_with_filter(self, data=None, size=None, **kwargs):
        """
        带过滤的买入操作
        
        Args:
            data: 数据源
            size: 交易数量
            **kwargs: 其他参数
        """
        if data is None:
            data = self.data
        
        # 检查是否可以买入
        if self.p.respect_limits and not self.trade_filter.can_buy(data, 0):
            self.log(f'买入被拒绝：涨停或其他限制')
            return None
        
        return self.buy(data=data, size=size, **kwargs)
    
    def sell_with_filter(self, data=None, size=None, **kwargs):
        """
        带过滤的卖出操作
        
        Args:
            data: 数据源
            size: 交易数量
            **kwargs: 其他参数
        """
        if data is None:
            data = self.data
        
        # 检查T+1规则
        if self.p.t1_trading:
            position = self.getposition(data)
            if position.size > 0:
                # 检查是否为当日买入（简化实现）
                today = self.datetime.date()
                if data._name in self.t1_positions and self.t1_positions[data._name] == today:
                    self.log(f'卖出被拒绝：T+1规则限制')
                    return None
        
        # 检查是否可以卖出
        if self.p.respect_limits and not self.trade_filter.can_sell(data, 0):
            self.log(f'卖出被拒绝：跌停或其他限制')
            return None
        
        return self.sell(data=data, size=size, **kwargs)
    
    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Completed]:
            if order.isbuy():
                # 记录T+1买入
                if self.p.t1_trading:
                    self.t1_positions[order.data._name] = self.datetime.date()
                
                self.log(f'买入完成：{order.executed.size}股@{order.executed.price:.2f}')
            else:
                self.log(f'卖出完成：{order.executed.size}股@{order.executed.price:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'订单失败：{order.status}')
    
    def log(self, txt, dt=None):
        """日志记录"""
        dt = dt or self.datetime.date()
        print(f'{dt.isoformat()}: {txt}')


class DatabaseBacktraderBridge:
    """
    数据库-Backtrader桥接器
    简化回测设置和数据加载
    """
    
    def __init__(self, db_path: str = "data_adapter/stock_data.db"):
        """
        初始化桥接器
        
        Args:
            db_path: 数据库路径
        """
        self.db = DatabaseManager(db_path)
        self.dao = StockDataDAO(self.db)
    
    def create_cerebro(self, initial_cash: float = 1000000) -> bt.Cerebro:
        """
        创建配置好的Cerebro实例
        
        Args:
            initial_cash: 初始资金
            
        Returns:
            配置好的Cerebro
        """
        cerebro = bt.Cerebro()
        
        # 设置初始资金
        cerebro.broker.setcash(initial_cash)
        
        # 设置中国A股佣金模型
        cerebro.broker.addcommissioninfo(ChinaCommissionInfo())
        
        # 设置资金管理器
        cerebro.addsizer(ChinaSizer)
        
        # 添加分析器
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        return cerebro
    
    def add_data(self, cerebro: bt.Cerebro, stock_code: str, 
                start_date: str, end_date: str, name: Optional[str] = None):
        """
        向Cerebro添加股票数据
        
        Args:
            cerebro: Cerebro实例
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            name: 数据名称
        """
        try:
            data_feed = ChinaStockDataFeed(
                self.dao, stock_code, start_date, end_date
            )
            
            cerebro.adddata(data_feed, name=name or stock_code)
            logger.info(f"添加数据源：{stock_code} ({start_date} 至 {end_date})")
            
        except Exception as e:
            logger.error(f"添加数据源失败 {stock_code}: {e}")
    
    def run_backtest(self, strategy_class, stock_codes: List[str], 
                    start_date: str, end_date: str, 
                    strategy_params: Optional[Dict] = None) -> Dict:
        """
        运行回测
        
        Args:
            strategy_class: 策略类
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            strategy_params: 策略参数
            
        Returns:
            回测结果
        """
        cerebro = self.create_cerebro()
        
        # 添加策略
        if strategy_params:
            cerebro.addstrategy(strategy_class, **strategy_params)
        else:
            cerebro.addstrategy(strategy_class)
        
        # 添加数据
        for code in stock_codes:
            self.add_data(cerebro, code, start_date, end_date)
        
        # 运行回测
        logger.info("开始回测...")
        results = cerebro.run()
        
        # 提取分析结果
        strat = results[0]
        
        analysis = {
            'final_value': cerebro.broker.getvalue(),
            'sharpe_ratio': strat.analyzers.sharpe.get_analysis().get('sharperatio', 0),
            'max_drawdown': strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0),
            'total_trades': strat.analyzers.trades.get_analysis().get('total', {}).get('total', 0),
            'returns': strat.analyzers.returns.get_analysis()
        }
        
        logger.info(f"回测完成！最终价值：{analysis['final_value']:,.2f}")
        
        return analysis


# 示例策略：简单移动平均线交叉
class SMAStrategy(ChinaStockStrategy):
    """简单移动平均线交叉策略"""
    
    params = (
        ('fast_period', 5),
        ('slow_period', 20),
    )
    
    def __init__(self):
        super().__init__()
        
        # 计算移动平均线
        self.sma_fast = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.p.fast_period
        )
        self.sma_slow = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.p.slow_period
        )
        
        # 交叉信号
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)
    
    def next(self):
        if not self.position:
            if self.crossover > 0:  # 金叉
                self.buy_with_filter()
        else:
            if self.crossover < 0:  # 死叉
                self.sell_with_filter()


if __name__ == "__main__":
    # 测试Backtrader集成
    bridge = DatabaseBacktraderBridge()
    
    # 获取可用股票
    stocks = bridge.dao.get_stock_list("A股")
    if not stocks.empty:
        test_codes = stocks['code'].head(3).tolist()
        
        # 运行简单回测
        results = bridge.run_backtest(
            SMAStrategy,
            test_codes,
            "2024-01-01",
            "2024-06-30",
            {'fast_period': 5, 'slow_period': 20}
        )
        
        print("回测结果：")
        for key, value in results.items():
            print(f"  {key}: {value}")
    else:
        print("数据库中无股票数据，请先运行数据迁移")