"""
策略适配器：将StockTradebyZ的选股策略适配到Qlib回测框架

主要功能：
1. 包装现有的4个选股策略为Qlib BaseStrategy
2. 处理信号生成和交易决策转换
3. 支持动态权重分配和风险管理
4. 集成V3.0量化评分系统
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import logging

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))  
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# 导入Qlib组件
try:
    from qlib.strategy.base import BaseStrategy
    from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
    from qlib.backtest.position import BasePosition
    from qlib.utils import get_or_create_path
except ImportError as e:
    raise ImportError(f"请先安装qlib: pip install qlib\n{e}")

# 导入项目模块
from stock_selctor.Selector import (
    BBIKDJSelector,
    BBIShortLongSelector, 
    BreakoutVolumeKDJSelector,
    PeakKDJSelector
)

logger = logging.getLogger(__name__)


class StrategyAdapter:
    """
    策略适配器工厂类
    
    负责将StockTradebyZ的选股策略转换为Qlib兼容的策略
    """
    
    STRATEGY_MAP = {
        'bbikdj': BBIKDJSelector,         # 少负战法
        'bbilongshort': BBIShortLongSelector,  # 补票战法  
        'breakout': BreakoutVolumeKDJSelector,  # TePu战法
        'peak': PeakKDJSelector           # 填坑战法
    }
    
    @classmethod
    def create_qlib_strategy(cls, 
                           strategy_name: str,
                           strategy_params: Optional[Dict] = None,
                           max_positions: int = 10,
                           position_size: float = 0.1) -> 'StockTraderStrategy':
        """
        创建Qlib兼容的策略实例
        
        Args:
            strategy_name: 策略名称 ('bbikdj', 'bbilongshort', 'breakout', 'peak')
            strategy_params: 策略参数
            max_positions: 最大持仓数量
            position_size: 单个持仓占比
            
        Returns:
            StockTraderStrategy实例
        """
        if strategy_name not in cls.STRATEGY_MAP:
            raise ValueError(f"未知策略: {strategy_name}, 支持的策略: {list(cls.STRATEGY_MAP.keys())}")
        
        selector_class = cls.STRATEGY_MAP[strategy_name]
        
        return StockTraderStrategy(
            selector_class=selector_class,
            selector_params=strategy_params or {},
            max_positions=max_positions,
            position_size=position_size
        )


class StockTraderStrategy(BaseStrategy):
    """
    StockTradebyZ策略的Qlib适配器
    
    将选股信号转换为Qlib交易决策，支持仓位管理和风险控制
    """
    
    def __init__(self,
                 selector_class,
                 selector_params: Dict = None,
                 max_positions: int = 10,
                 position_size: float = 0.1,
                 stop_loss: float = 0.08,
                 take_profit: float = 0.15,
                 **kwargs):
        """
        初始化策略适配器
        
        Args:
            selector_class: 选股器类
            selector_params: 选股器参数
            max_positions: 最大持仓数量
            position_size: 单个持仓占比
            stop_loss: 止损比例
            take_profit: 止盈比例
        """
        super().__init__(**kwargs)
        
        # 初始化选股器
        self.selector = selector_class(**(selector_params or {}))
        self.max_positions = max_positions
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # 策略状态
        self.current_positions = {}
        self.entry_prices = {}
        self.last_signals = {}
        
        logger.info(f"策略初始化: {selector_class.__name__}, "
                   f"最大持仓: {max_positions}, 单仓占比: {position_size}")
    
    def generate_trade_decision(self, execute_result: list = None):
        """
        生成交易决策
        
        Args:
            execute_result: 上一步执行结果
            
        Returns:
            TradeDecision包含买卖订单
        """
        # 获取当前交易日期
        current_time = self.trade_calendar.get_current_datetime()
        
        try:
            # 生成选股信号
            signals = self._generate_stock_signals(current_time)
            
            # 风险检查和仓位管理
            buy_orders, sell_orders = self._process_signals(signals, current_time)
            
            # 创建交易决策
            trade_decision = TradeDecisionWO(
                order_list=buy_orders + sell_orders,
                strategy=self
            )
            
            logger.debug(f"{current_time}: 生成 {len(buy_orders)} 买单, {len(sell_orders)} 卖单")
            
            return trade_decision
            
        except Exception as e:
            logger.error(f"生成交易决策失败: {e}")
            # 返回空交易决策
            return TradeDecisionWO(order_list=[], strategy=self)
    
    def _generate_stock_signals(self, current_time) -> Dict[str, float]:
        """
        生成股票选择信号
        
        Args:
            current_time: 当前时间
            
        Returns:
            股票代码 -> 信号强度的字典
        """
        signals = {}
        
        try:
            # 从交易所获取可交易股票列表
            instruments = list(self.trade_exchange.get_all_instruments())
            
            # 为每只股票生成信号
            for instrument in instruments:
                try:
                    # 获取历史数据
                    hist_data = self._get_instrument_data(instrument, current_time, lookback=30)
                    
                    if hist_data.empty:
                        continue
                    
                    # 使用选股器生成信号
                    signal = self._evaluate_stock_signal(instrument, hist_data)
                    
                    if signal is not None and signal > 0:
                        signals[instrument] = signal
                        
                except Exception as e:
                    logger.debug(f"为 {instrument} 生成信号失败: {e}")
                    continue
            
            # 选择最强信号的股票
            if signals:
                sorted_signals = dict(sorted(signals.items(), key=lambda x: x[1], reverse=True))
                # 限制信号数量以控制持仓
                signals = dict(list(sorted_signals.items())[:self.max_positions * 2])
            
        except Exception as e:
            logger.error(f"生成股票信号失败: {e}")
            signals = {}
        
        return signals
    
    def _get_instrument_data(self, instrument: str, current_time, lookback: int = 30) -> pd.DataFrame:
        """
        获取股票历史数据
        
        Args:
            instrument: 股票代码
            current_time: 当前时间
            lookback: 回看天数
            
        Returns:
            历史数据DataFrame
        """
        try:
            # 计算开始时间
            start_time = current_time - timedelta(days=lookback)
            
            # 从交易所获取数据
            data = self.trade_exchange.get_history(
                instrument=instrument,
                start_time=start_time,
                end_time=current_time,
                fields=['open', 'high', 'low', 'close', 'volume']
            )
            
            return data
            
        except Exception as e:
            logger.debug(f"获取 {instrument} 历史数据失败: {e}")
            return pd.DataFrame()
    
    def _evaluate_stock_signal(self, instrument: str, hist_data: pd.DataFrame) -> Optional[float]:
        """
        评估股票信号强度
        
        Args:
            instrument: 股票代码
            hist_data: 历史数据
            
        Returns:
            信号强度 (0-1之间，越高越好)
        """
        try:
            # 检查数据完整性
            if len(hist_data) < 10:
                return None
            
            # 基于选股器类型评估信号
            if isinstance(self.selector, BBIKDJSelector):
                return self._evaluate_bbikdj_signal(hist_data)
            elif isinstance(self.selector, BBIShortLongSelector):  
                return self._evaluate_bbilongshort_signal(hist_data)
            elif isinstance(self.selector, BreakoutVolumeKDJSelector):
                return self._evaluate_breakout_signal(hist_data)
            elif isinstance(self.selector, PeakKDJSelector):
                return self._evaluate_peak_signal(hist_data)
            else:
                return self._evaluate_generic_signal(hist_data)
                
        except Exception as e:
            logger.debug(f"评估 {instrument} 信号失败: {e}")
            return None
    
    def _evaluate_bbikdj_signal(self, data: pd.DataFrame) -> Optional[float]:
        """评估BBI+KDJ信号"""
        try:
            # 计算BBI
            ma_3 = data['close'].rolling(3).mean()
            ma_6 = data['close'].rolling(6).mean() 
            ma_12 = data['close'].rolling(12).mean()
            ma_24 = data['close'].rolling(24).mean()
            bbi = (ma_3 + ma_6 + ma_12 + ma_24) / 4
            
            # 计算KDJ
            low_min = data['low'].rolling(9).min()
            high_max = data['high'].rolling(9).max()
            rsv = (data['close'] - low_min) / (high_max - low_min) * 100
            k = rsv.ewm(alpha=1/3).mean()
            d = k.ewm(alpha=1/3).mean()
            j = 3 * k - 2 * d
            
            # 信号判断
            current_close = data['close'].iloc[-1]
            current_bbi = bbi.iloc[-1]
            current_j = j.iloc[-1]
            
            if pd.isna(current_bbi) or pd.isna(current_j):
                return None
            
            # BBI金叉且J值从低位反弹
            bbi_signal = 1.0 if current_close > current_bbi else 0.0
            j_signal = max(0, (current_j - 20) / 30) if current_j < 80 else 0.0
            
            return (bbi_signal + j_signal) / 2
            
        except Exception:
            return None
    
    def _evaluate_bbilongshort_signal(self, data: pd.DataFrame) -> Optional[float]:
        """评估BBI长短期信号"""
        try:
            # 短期BBI (3,6,12,24)
            short_bbi = (
                data['close'].rolling(3).mean() + 
                data['close'].rolling(6).mean() +
                data['close'].rolling(12).mean() +
                data['close'].rolling(24).mean()
            ) / 4
            
            # 长期BBI (调整参数)
            long_bbi = (
                data['close'].rolling(6).mean() +
                data['close'].rolling(12).mean() + 
                data['close'].rolling(24).mean() +
                data['close'].rolling(48).mean()
            ) / 4
            
            current_short = short_bbi.iloc[-1]
            current_long = long_bbi.iloc[-1]
            
            if pd.isna(current_short) or pd.isna(current_long):
                return None
            
            # 短期BBI上穿长期BBI
            signal_strength = max(0, (current_short - current_long) / current_long * 10)
            return min(1.0, signal_strength)
            
        except Exception:
            return None
    
    def _evaluate_breakout_signal(self, data: pd.DataFrame) -> Optional[float]:
        """评估突破+成交量信号"""
        try:
            # 价格突破20日高点
            high_20 = data['high'].rolling(20).max()
            current_close = data['close'].iloc[-1] 
            breakout_signal = 1.0 if current_close > high_20.iloc[-2] else 0.0
            
            # 成交量放大
            vol_ma_5 = data['volume'].rolling(5).mean()
            current_vol = data['volume'].iloc[-1]
            vol_signal = min(1.0, max(0, (current_vol / vol_ma_5.iloc[-1] - 1.5) / 1.0))
            
            return (breakout_signal + vol_signal) / 2
            
        except Exception:
            return None
    
    def _evaluate_peak_signal(self, data: pd.DataFrame) -> Optional[float]:
        """评估填坑信号""" 
        try:
            # 寻找近期低点后的反弹
            low_10 = data['low'].rolling(10).min()
            current_close = data['close'].iloc[-1]
            recent_low = low_10.iloc[-1]
            
            # 从低点反弹的强度
            rebound_pct = (current_close - recent_low) / recent_low
            signal = max(0, min(1.0, rebound_pct * 10))
            
            return signal
            
        except Exception:
            return None
    
    def _evaluate_generic_signal(self, data: pd.DataFrame) -> Optional[float]:
        """通用信号评估"""
        try:
            # 简单的动量信号
            returns = data['close'].pct_change(5).iloc[-1]
            if pd.isna(returns):
                return None
            
            # 转换为0-1之间的信号强度
            signal = max(0, min(1.0, (returns + 0.05) * 5))
            return signal
            
        except Exception:
            return None
    
    def _process_signals(self, signals: Dict[str, float], current_time) -> tuple:
        """
        处理信号并生成买卖订单
        
        Args:
            signals: 股票信号字典
            current_time: 当前时间
            
        Returns:
            (买单列表, 卖单列表)
        """
        buy_orders = []
        sell_orders = []
        
        try:
            # 获取当前持仓
            current_position = self.trade_position
            held_instruments = set(current_position.get_stock_list())
            
            # 风险检查：检查止损止盈
            for instrument in held_instruments:
                if self._should_close_position(instrument, current_time):
                    # 生成卖单
                    sell_order = Order(
                        stock_id=instrument,
                        amount=current_position.get_stock_amount(instrument),
                        direction=OrderDir.SELL,
                        factor=1.0
                    )
                    sell_orders.append(sell_order)
            
            # 选择新的买入目标
            available_cash = current_position.get_cash()
            max_new_positions = self.max_positions - len(held_instruments) + len(sell_orders)
            
            # 按信号强度排序
            sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
            
            for instrument, signal_strength in sorted_signals[:max_new_positions]:
                if instrument not in held_instruments:
                    # 计算买入金额
                    buy_amount = available_cash * self.position_size
                    
                    if buy_amount > 100:  # 最小交易金额
                        buy_order = Order(
                            stock_id=instrument,
                            amount=buy_amount,
                            direction=OrderDir.BUY,
                            factor=signal_strength
                        )
                        buy_orders.append(buy_order)
                        available_cash -= buy_amount
            
        except Exception as e:
            logger.error(f"处理交易信号失败: {e}")
        
        return buy_orders, sell_orders
    
    def _should_close_position(self, instrument: str, current_time) -> bool:
        """
        检查是否应该平仓
        
        Args:
            instrument: 股票代码
            current_time: 当前时间
            
        Returns:
            是否应该平仓
        """
        try:
            # 获取入场价格
            entry_price = self.entry_prices.get(instrument)
            if entry_price is None:
                return False
            
            # 获取当前价格
            current_price = self.trade_exchange.get_last_price(instrument)
            if current_price is None:
                return False
            
            # 计算收益率
            return_pct = (current_price - entry_price) / entry_price
            
            # 止损或止盈
            if return_pct <= -self.stop_loss:
                logger.info(f"{instrument} 触发止损: {return_pct:.2%}")
                return True
            
            if return_pct >= self.take_profit:
                logger.info(f"{instrument} 触发止盈: {return_pct:.2%}")
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"检查 {instrument} 平仓条件失败: {e}")
            return False
    
    def post_exe_step(self, execute_result: Optional[list]) -> None:
        """
        执行后处理
        
        Args:
            execute_result: 执行结果
        """
        super().post_exe_step(execute_result)
        
        try:
            # 更新持仓记录
            current_position = self.trade_position
            for instrument in current_position.get_stock_list():
                if instrument not in self.entry_prices:
                    # 记录新买入股票的价格
                    self.entry_prices[instrument] = self.trade_exchange.get_last_price(instrument)
            
            # 清理已卖出股票的记录
            held_instruments = set(current_position.get_stock_list())
            self.entry_prices = {k: v for k, v in self.entry_prices.items() if k in held_instruments}
            
        except Exception as e:
            logger.debug(f"执行后处理失败: {e}")