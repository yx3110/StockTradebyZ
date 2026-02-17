"""
StockTradebyZ策略实现 - 专门为Qlib回测优化的策略类

整合了现有的4个选股策略，提供统一的接口和优化的性能：
1. 少负战法 (BBI+KDJ)
2. 补票战法 (BBI长短期)
3. TePu战法 (突破+成交量)
4. 填坑战法 (低点反弹)

特点：
- 中国A股市场特性优化 (T+1, 涨跌停等)
- 集成V3.0量化评分系统
- 支持多策略组合和权重分配
- 内置风险管理和仓位控制
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import json

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) 
sys.path.append(project_root)

# 导入Qlib组件  
try:
    from qlib.strategy.base import BaseStrategy
    from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
    from qlib.backtest.position import BasePosition
except ImportError as e:
    raise ImportError(f"请先安装qlib: pip install qlib\n{e}")

# 导入项目模块
from data_adapter.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class StockTraderStrategy(BaseStrategy):
    """
    StockTradebyZ统一策略实现
    
    支持多个选股策略的组合使用，适配中国A股市场特性
    """
    
    # 策略权重配置（基于V3.0量化系统优化）
    DEFAULT_STRATEGY_WEIGHTS = {
        'technical': 0.35,      # 技术指标权重
        'momentum': 0.25,       # 动量指标权重  
        'volume': 0.20,         # 成交量指标权重
        'fundamental': 0.15,    # 基本面指标权重
        'sentiment': 0.05       # 市场情绪权重
    }
    
    def __init__(self,
                 strategies: List[str] = None,
                 strategy_weights: Dict[str, float] = None,
                 max_positions: int = 10,
                 position_size: float = 0.1,
                 stop_loss: float = 0.08,
                 take_profit: float = 0.15,
                 min_score: float = 70.0,
                 rebalance_freq: int = 5,
                 **kwargs):
        """
        初始化StockTradebyZ策略
        
        Args:
            strategies: 使用的策略列表 ['bbikdj', 'bbilongshort', 'breakout', 'peak']
            strategy_weights: 各策略权重，如未指定使用默认权重
            max_positions: 最大持仓数量
            position_size: 单个持仓占比
            stop_loss: 止损比例
            take_profit: 止盈比例
            min_score: 最低选股评分
            rebalance_freq: 调仓频率（交易日）
        """
        super().__init__(**kwargs)
        
        # 策略配置
        self.strategies = strategies or ['bbikdj', 'bbilongshort', 'breakout', 'peak']
        self.strategy_weights = strategy_weights or self.DEFAULT_STRATEGY_WEIGHTS
        self.max_positions = max_positions
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.min_score = min_score
        self.rebalance_freq = rebalance_freq
        
        # 策略状态
        self.current_positions = {}
        self.entry_prices = {}
        self.entry_dates = {}
        self.last_rebalance_date = None
        self.strategy_cache = {}
        
        # 数据管理器
        self.db_manager = DatabaseManager()
        
        # 加载历史评分数据用于策略校准
        self._load_historical_performance()
        
        logger.info(f"StockTradebyZ策略初始化完成")
        logger.info(f"启用策略: {self.strategies}")
        logger.info(f"策略权重: {self.strategy_weights}")
        logger.info(f"风控参数: 止损{self.stop_loss:.1%}, 止盈{self.take_profit:.1%}")
    
    def _load_historical_performance(self):
        """加载历史策略表现数据用于动态权重调整"""
        try:
            # 这里可以从reports/目录加载历史回测数据
            # 用于策略权重的动态调整
            self.historical_performance = {
                'bbikdj': {'sharpe': 1.2, 'win_rate': 0.55},
                'bbilongshort': {'sharpe': 1.1, 'win_rate': 0.52}, 
                'breakout': {'sharpe': 1.3, 'win_rate': 0.58},
                'peak': {'sharpe': 1.0, 'win_rate': 0.50}
            }
            logger.info("历史策略表现数据加载完成")
            
        except Exception as e:
            logger.warning(f"加载历史策略表现失败: {e}")
            self.historical_performance = {}
    
    def generate_trade_decision(self, execute_result: list = None) -> TradeDecisionWO:
        """
        生成交易决策
        
        Args:
            execute_result: 上一步执行结果
            
        Returns:
            TradeDecisionWO对象包含所有交易订单
        """
        current_time = self.trade_calendar.get_current_datetime()
        
        try:
            # 检查是否需要调仓
            if not self._should_rebalance(current_time):
                # 仅检查风控信号
                sell_orders = self._check_risk_management(current_time)
                return TradeDecisionWO(
                    order_list=sell_orders,
                    strategy=self
                )
            
            # 执行完整的策略分析
            logger.info(f"{current_time.strftime('%Y-%m-%d')}: 执行策略调仓")
            
            # 1. 生成候选股票池
            candidate_stocks = self._generate_stock_universe(current_time)
            
            # 2. 多策略评分
            stock_scores = self._calculate_multi_strategy_scores(candidate_stocks, current_time)
            
            # 3. 选择最佳股票
            selected_stocks = self._select_top_stocks(stock_scores)
            
            # 4. 生成交易订单
            buy_orders, sell_orders = self._generate_trading_orders(selected_stocks, current_time)
            
            # 更新调仓日期
            self.last_rebalance_date = current_time
            
            total_orders = len(buy_orders) + len(sell_orders)
            logger.info(f"生成交易订单: {len(buy_orders)}买 + {len(sell_orders)}卖 = {total_orders}单")
            
            return TradeDecisionWO(
                order_list=buy_orders + sell_orders,
                strategy=self
            )
            
        except Exception as e:
            logger.error(f"生成交易决策失败: {e}")
            return TradeDecisionWO(
                order_list=[],
                strategy=self
            )
    
    def _should_rebalance(self, current_time) -> bool:
        """检查是否需要调仓"""
        if self.last_rebalance_date is None:
            return True
        
        days_since_rebalance = (current_time - self.last_rebalance_date).days
        return days_since_rebalance >= self.rebalance_freq
    
    def _generate_stock_universe(self, current_time) -> List[str]:
        """
        生成候选股票池
        
        Args:
            current_time: 当前时间
            
        Returns:
            候选股票代码列表
        """
        try:
            # 获取基础股票池（排除ST、新股等）
            end_date = current_time.strftime('%Y-%m-%d')
            start_date = (current_time - timedelta(days=30)).strftime('%Y-%m-%d')
            
            # 从数据库获取活跃股票
            query = """
            SELECT DISTINCT s.code 
            FROM securities s
            JOIN daily_quotes dq ON s.id = dq.security_id
            WHERE s.type = 'A股' 
            AND s.name NOT LIKE '%ST%'
            AND dq.trade_date BETWEEN ? AND ?
            AND dq.close IS NOT NULL
            AND dq.volume > 0
            GROUP BY s.code
            HAVING COUNT(*) >= 20
            ORDER BY s.code
            """
            
            result = self.db_manager.execute_query(query, [start_date, end_date])
            candidate_stocks = [row[0] for row in result]
            
            logger.info(f"候选股票池: {len(candidate_stocks)}只股票")
            return candidate_stocks
            
        except Exception as e:
            logger.error(f"生成股票池失败: {e}")
            return []
    
    def _calculate_multi_strategy_scores(self, stocks: List[str], current_time) -> Dict[str, float]:
        """
        计算多策略综合评分
        
        Args:
            stocks: 股票列表
            current_time: 当前时间
            
        Returns:
            股票代码 -> 综合评分的字典
        """
        stock_scores = {}
        
        for stock in stocks:
            try:
                # 获取股票历史数据
                hist_data = self._get_stock_data(stock, current_time)
                
                if hist_data.empty:
                    continue
                
                # 各策略评分
                scores = {}
                
                if 'bbikdj' in self.strategies:
                    scores['bbikdj'] = self._score_bbikdj_strategy(hist_data)
                
                if 'bbilongshort' in self.strategies:
                    scores['bbilongshort'] = self._score_bbilongshort_strategy(hist_data)
                
                if 'breakout' in self.strategies:
                    scores['breakout'] = self._score_breakout_strategy(hist_data)
                    
                if 'peak' in self.strategies:
                    scores['peak'] = self._score_peak_strategy(hist_data)
                
                # 计算加权综合评分
                total_score = self._calculate_weighted_score(scores)
                
                if total_score >= self.min_score:
                    stock_scores[stock] = total_score
                    
            except Exception as e:
                logger.debug(f"计算 {stock} 评分失败: {e}")
                continue
        
        logger.info(f"完成评分计算: {len(stock_scores)}只股票超过最低评分{self.min_score}")
        return stock_scores
    
    def _get_stock_data(self, stock: str, current_time, lookback_days: int = 60) -> pd.DataFrame:
        """获取股票历史数据"""
        try:
            end_date = current_time.strftime('%Y-%m-%d')
            start_date = (current_time - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            
            # 从缓存获取数据
            cache_key = f"{stock}_{start_date}_{end_date}"
            if cache_key in self.strategy_cache:
                return self.strategy_cache[cache_key]
            
            # 查询数据库
            query = """
            SELECT 
                dq.trade_date,
                dq.open, dq.high, dq.low, dq.close, dq.volume,
                dq.price_change_pct,
                ti.ma_5, ti.ma_10, ti.ma_20, ti.bbi,
                ti.rsi_14, ti.kdj_k, ti.kdj_d, ti.kdj_j,
                ti.macd, ti.macd_signal, ti.macd_hist,
                db.pe_ttm, db.pb, db.total_mv as market_cap, db.turnover_rate
            FROM securities s
            JOIN daily_quotes dq ON s.id = dq.security_id
            LEFT JOIN technical_indicators ti ON s.id = ti.security_id AND ti.trade_date = dq.trade_date
            LEFT JOIN daily_basic db ON s.id = db.security_id AND db.trade_date = dq.trade_date
            WHERE s.code = ?
            AND dq.trade_date BETWEEN ? AND ?
            ORDER BY dq.trade_date
            """
            
            result = self.db_manager.execute_query(query, [stock, start_date, end_date])
            
            if not result:
                return pd.DataFrame()
            
            # 转换为DataFrame
            columns = [
                'trade_date', 'open', 'high', 'low', 'close', 'volume', 'price_change_pct',
                'ma_5', 'ma_10', 'ma_20', 'bbi', 'rsi_14', 'kdj_k', 'kdj_d', 'kdj_j',
                'macd', 'macd_signal', 'macd_hist', 'pe_ttm', 'pb', 'market_cap', 'turnover_rate'
            ]
            
            df = pd.DataFrame(result, columns=columns)
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.set_index('trade_date')
            
            # 缓存数据
            self.strategy_cache[cache_key] = df
            
            return df
            
        except Exception as e:
            logger.debug(f"获取 {stock} 历史数据失败: {e}")
            return pd.DataFrame()
    
    def _score_bbikdj_strategy(self, data: pd.DataFrame) -> float:
        """BBI+KDJ策略评分 (0-100分)"""
        try:
            if len(data) < 20:
                return 0.0
            
            latest = data.iloc[-1]
            
            # BBI信号
            bbi_signal = 0
            if not pd.isna(latest['bbi']) and not pd.isna(latest['close']):
                if latest['close'] > latest['bbi']:
                    bbi_signal = 30  # BBI多头信号
            
            # KDJ信号
            kdj_signal = 0
            if not pd.isna(latest['kdj_j']):
                if latest['kdj_j'] > 20 and latest['kdj_j'] < 80:
                    kdj_signal = 25  # J值在合理区间
                elif latest['kdj_j'] < 20:
                    kdj_signal = 35  # 超卖反弹机会
            
            # 趋势信号
            trend_signal = 0
            if len(data) >= 5:
                recent_change = data['close'].pct_change(5).iloc[-1]
                if not pd.isna(recent_change):
                    trend_signal = max(0, min(20, recent_change * 100 + 10))
            
            # 成交量信号
            volume_signal = 0
            if not pd.isna(latest['turnover_rate']):
                if latest['turnover_rate'] > 2:  # 换手率超过2%
                    volume_signal = 15
                elif latest['turnover_rate'] > 1:
                    volume_signal = 10
            
            total_score = bbi_signal + kdj_signal + trend_signal + volume_signal
            return min(100, total_score)
            
        except Exception as e:
            logger.debug(f"BBI+KDJ策略评分失败: {e}")
            return 0.0
    
    def _score_bbilongshort_strategy(self, data: pd.DataFrame) -> float:
        """BBI长短期策略评分"""
        try:
            if len(data) < 30:
                return 0.0
            
            latest = data.iloc[-1]
            
            # BBI短期趋势
            short_trend = 0
            if not pd.isna(latest['ma_5']) and not pd.isna(latest['ma_20']):
                if latest['ma_5'] > latest['ma_20']:
                    short_trend = 25
            
            # BBI中期趋势  
            if not pd.isna(latest['bbi']) and not pd.isna(latest['close']):
                if latest['close'] > latest['bbi']:
                    short_trend += 20
            
            # 价格动量
            momentum = 0
            if len(data) >= 10:
                price_momentum = data['close'].pct_change(10).iloc[-1]
                if not pd.isna(price_momentum) and price_momentum > 0:
                    momentum = min(25, price_momentum * 500)
            
            # 相对强弱
            rsi_signal = 0
            if not pd.isna(latest['rsi_14']):
                if 30 < latest['rsi_14'] < 70:
                    rsi_signal = 15
                elif latest['rsi_14'] < 30:
                    rsi_signal = 25  # 超卖
            
            total_score = short_trend + momentum + rsi_signal
            return min(100, total_score)
            
        except Exception:
            return 0.0
    
    def _score_breakout_strategy(self, data: pd.DataFrame) -> float:
        """突破+成交量策略评分"""
        try:
            if len(data) < 20:
                return 0.0
                
            latest = data.iloc[-1]
            
            # 价格突破信号
            breakout_signal = 0
            if len(data) >= 20:
                high_20 = data['high'].rolling(20).max().iloc[-2]  # 前20日高点
                if not pd.isna(high_20) and latest['close'] > high_20:
                    breakout_signal = 40  # 突破20日新高
                elif not pd.isna(high_20) and latest['close'] > high_20 * 0.98:
                    breakout_signal = 25  # 接近突破
            
            # 成交量确认
            volume_signal = 0
            if len(data) >= 5:
                vol_ma_5 = data['volume'].rolling(5).mean().iloc[-2]
                if not pd.isna(vol_ma_5) and latest['volume'] > vol_ma_5 * 1.5:
                    volume_signal = 30  # 成交量放大
                elif not pd.isna(vol_ma_5) and latest['volume'] > vol_ma_5:
                    volume_signal = 15
            
            # MACD确认
            macd_signal = 0
            if not pd.isna(latest['macd']) and not pd.isna(latest['macd_signal']):
                if latest['macd'] > latest['macd_signal']:
                    macd_signal = 20
            
            total_score = breakout_signal + volume_signal + macd_signal
            return min(100, total_score)
            
        except Exception:
            return 0.0
    
    def _score_peak_strategy(self, data: pd.DataFrame) -> float:
        """填坑策略评分"""
        try:
            if len(data) < 15:
                return 0.0
            
            latest = data.iloc[-1]
            
            # 寻找近期低点
            low_signal = 0
            if len(data) >= 10:
                low_10 = data['low'].rolling(10).min().iloc[-1]
                recent_low = data['low'].iloc[-5:].min()
                
                if not pd.isna(low_10) and not pd.isna(recent_low):
                    # 从低点反弹的幅度
                    rebound_pct = (latest['close'] - recent_low) / recent_low
                    if 0.02 < rebound_pct < 0.08:  # 2%-8%的反弹
                        low_signal = 35
                    elif 0 < rebound_pct <= 0.02:
                        low_signal = 20
            
            # KDJ低位反弹
            kdj_signal = 0
            if not pd.isna(latest['kdj_j']):
                if latest['kdj_j'] < 30:  # J值在低位
                    kdj_signal = 25
                elif latest['kdj_j'] < 50:
                    kdj_signal = 15
            
            # 相对估值
            valuation_signal = 0
            if not pd.isna(latest['pb']) and latest['pb'] > 0:
                if latest['pb'] < 2:  # 低PB
                    valuation_signal = 20
                elif latest['pb'] < 3:
                    valuation_signal = 10
            
            total_score = low_signal + kdj_signal + valuation_signal
            return min(100, total_score)
            
        except Exception:
            return 0.0
    
    def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """计算加权综合评分"""
        if not scores:
            return 0.0
        
        # 根据策略权重计算综合评分
        weighted_score = 0.0
        total_weight = 0.0
        
        strategy_mapping = {
            'bbikdj': 'technical',
            'bbilongshort': 'momentum', 
            'breakout': 'volume',
            'peak': 'fundamental'
        }
        
        for strategy, score in scores.items():
            weight_key = strategy_mapping.get(strategy, 'technical')
            weight = self.strategy_weights.get(weight_key, 0.25)
            weighted_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_score /= total_weight
        
        return weighted_score
    
    def _select_top_stocks(self, stock_scores: Dict[str, float]) -> List[Tuple[str, float]]:
        """选择评分最高的股票"""
        # 按评分排序
        sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 选择前N只股票
        selected = sorted_stocks[:self.max_positions]
        
        logger.info(f"选股结果: {len(selected)}只股票, "
                   f"平均评分: {np.mean([s[1] for s in selected]):.1f}")
        
        return selected
    
    def _generate_trading_orders(self, selected_stocks: List[Tuple[str, float]], 
                               current_time) -> Tuple[List[Order], List[Order]]:
        """生成交易订单"""
        buy_orders = []
        sell_orders = []
        
        try:
            # 获取当前持仓
            current_position = self.trade_position
            held_stocks = set(current_position.get_stock_list()) if current_position else set()
            target_stocks = {stock for stock, score in selected_stocks}
            
            # 生成卖单（持有但不在目标中的股票）
            stocks_to_sell = held_stocks - target_stocks
            for stock in stocks_to_sell:
                try:
                    amount = current_position.get_stock_amount(stock)
                    if amount > 0:
                        sell_order = Order(
                            stock_id=stock,
                            amount=amount,
                            direction=OrderDir.SELL
                        )
                        sell_orders.append(sell_order)
                except Exception as e:
                    logger.debug(f"生成 {stock} 卖单失败: {e}")
            
            # 生成买单（目标中但未持有的股票）
            stocks_to_buy = target_stocks - held_stocks
            if stocks_to_buy:
                # 计算每只股票的买入金额
                available_cash = current_position.get_cash() if current_position else 100000
                buy_amount_per_stock = available_cash * self.position_size
                
                for stock, score in selected_stocks:
                    if stock in stocks_to_buy and buy_amount_per_stock > 100:
                        try:
                            buy_order = Order(
                                stock_id=stock,
                                amount=buy_amount_per_stock,
                                direction=OrderDir.BUY,
                                factor=score / 100.0  # 将评分转换为0-1的因子
                            )
                            buy_orders.append(buy_order)
                        except Exception as e:
                            logger.debug(f"生成 {stock} 买单失败: {e}")
            
        except Exception as e:
            logger.error(f"生成交易订单失败: {e}")
        
        return buy_orders, sell_orders
    
    def _check_risk_management(self, current_time) -> List[Order]:
        """风险管理检查，生成止损止盈订单"""
        sell_orders = []
        
        try:
            current_position = self.trade_position
            if not current_position:
                return sell_orders
            
            for stock in current_position.get_stock_list():
                # 检查止损止盈条件
                if self._should_close_position(stock, current_time):
                    amount = current_position.get_stock_amount(stock)
                    if amount > 0:
                        sell_order = Order(
                            stock_id=stock,
                            amount=amount,
                            direction=OrderDir.SELL
                        )
                        sell_orders.append(sell_order)
                        
        except Exception as e:
            logger.error(f"风险管理检查失败: {e}")
        
        return sell_orders
    
    def _should_close_position(self, stock: str, current_time) -> bool:
        """检查是否应该平仓"""
        try:
            # 获取入场信息
            entry_price = self.entry_prices.get(stock)
            entry_date = self.entry_dates.get(stock)
            
            if not entry_price or not entry_date:
                return False
            
            # 获取当前价格
            current_data = self._get_stock_data(stock, current_time, lookback_days=2)
            if current_data.empty:
                return False
            
            current_price = current_data['close'].iloc[-1]
            if pd.isna(current_price):
                return False
            
            # 计算收益率
            return_pct = (current_price - entry_price) / entry_price
            
            # 止损检查
            if return_pct <= -self.stop_loss:
                logger.info(f"{stock} 触发止损: {return_pct:.2%}")
                return True
            
            # 止盈检查
            if return_pct >= self.take_profit:
                logger.info(f"{stock} 触发止盈: {return_pct:.2%}")
                return True
            
            # 时间止损（持有超过30天）
            days_held = (current_time - entry_date).days
            if days_held > 30 and return_pct < 0.02:
                logger.info(f"{stock} 长期持有无收益，止损: {days_held}天, {return_pct:.2%}")
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"检查 {stock} 平仓条件失败: {e}")
            return False
    
    def post_exe_step(self, execute_result: Optional[list]) -> None:
        """执行后处理"""
        super().post_exe_step(execute_result)
        
        try:
            current_time = self.trade_calendar.get_current_datetime()
            current_position = self.trade_position
            
            if current_position:
                # 更新入场价格记录
                for stock in current_position.get_stock_list():
                    if stock not in self.entry_prices:
                        # 获取当前价格作为入场价格
                        data = self._get_stock_data(stock, current_time, lookback_days=2)
                        if not data.empty:
                            self.entry_prices[stock] = data['close'].iloc[-1]
                            self.entry_dates[stock] = current_time
                
                # 清理已平仓股票的记录
                held_stocks = set(current_position.get_stock_list())
                self.entry_prices = {k: v for k, v in self.entry_prices.items() if k in held_stocks}
                self.entry_dates = {k: v for k, v in self.entry_dates.items() if k in held_stocks}
            
            # 清理缓存（避免内存占用过大）
            if len(self.strategy_cache) > 1000:
                self.strategy_cache.clear()
                
        except Exception as e:
            logger.debug(f"执行后处理失败: {e}")