"""
中国A股交易所配置 - 专门适配中国股市特性的交易所实现

主要特性：
1. T+1交易制度：当日买入股票次日方可卖出
2. 涨跌停限制：一般股票±10%，ST股票±5%，科创板/创业板注册制±20%
3. 交易时间：9:30-11:30, 13:00-15:00
4. 交易成本：印花税、手续费、过户费等
5. 最小交易单位：100股（1手）
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
    from qlib.backtest.exchange import Exchange
    from qlib.backtest.decision import Order, OrderDir
    from qlib.data import D
except ImportError as e:
    raise ImportError(f"请先安装qlib: pip install qlib\n{e}")

logger = logging.getLogger(__name__)


class ChineseAShareExchange(Exchange):
    """
    中国A股交易所实现
    
    集成了中国股市的各种特殊规则和交易成本
    """
    
    # 交易成本配置
    TRADING_COSTS = {
        'stamp_tax': 0.001,        # 印花税 0.1% (仅卖出时收取)
        'commission_rate': 0.0003, # 佣金费率 0.03% (最低5元)
        'min_commission': 5.0,     # 最低佣金
        'transfer_fee': 0.00001,   # 过户费 0.001% (沪市)
        'min_transfer_fee': 1.0    # 最低过户费
    }
    
    # 涨跌停限制
    LIMIT_CONFIG = {
        'normal': 0.10,     # 普通股票 ±10%
        'st': 0.05,         # ST股票 ±5%  
        'kcb': 0.20,        # 科创板 ±20%
        'cyb_reg': 0.20,    # 创业板注册制 ±20%
        'new_stock': 0.44   # 新股首日 ±44%
    }
    
    def __init__(self, 
                 start_time: Union[str, pd.Timestamp],
                 end_time: Union[str, pd.Timestamp],
                 freq: str = "day",
                 codes: Optional[List[str]] = None,
                 deal_price: str = "close",
                 subscribe_fields: List[str] = None,
                 limit_threshold: Optional[float] = None,
                 volume_threshold: Union[tuple, dict, None] = None,
                 open_cost: float = None,
                 close_cost: float = None,
                 min_cost: float = None,
                 impact_cost: float = 0.0,
                 extra_quote: pd.DataFrame = None,
                 quote_cls = None,
                 **kwargs):
        """
        初始化中国A股交易所
        
        Args:
            start_time: 回测开始时间
            end_time: 回测结束时间
            freq: 交易频率
            codes: 股票代码列表
            deal_price: 成交价格类型
            subscribe_fields: 订阅字段
            limit_threshold: 涨跌停阈值
            volume_threshold: 成交量阈值
            open_cost: 开仓成本
            close_cost: 平仓成本
            min_cost: 最小成本
            impact_cost: 冲击成本
            extra_quote: 额外报价数据
            quote_cls: 报价类
        """
        
        # 提取trade_unit参数（如果提供）
        trade_unit = kwargs.pop('trade_unit', 100)
        
        # 设置中国市场默认参数
        if limit_threshold is None:
            limit_threshold = self.LIMIT_CONFIG['normal']
        
        if open_cost is None:
            open_cost = self.TRADING_COSTS['commission_rate']
        
        if close_cost is None:
            close_cost = self.TRADING_COSTS['commission_rate'] + self.TRADING_COSTS['stamp_tax']
        
        if min_cost is None:
            min_cost = self.TRADING_COSTS['min_commission']
        
        if subscribe_fields is None:
            subscribe_fields = ["open", "high", "low", "close", "volume", "factor"]
        
        # 准备传递给父类的参数，只包含父类支持的参数
        parent_kwargs = {
            'freq': freq,
            'start_time': start_time,
            'end_time': end_time,
            'codes': codes if codes is not None else 'all',
            'deal_price': deal_price,
            'subscribe_fields': subscribe_fields,
            'limit_threshold': limit_threshold,
            'volume_threshold': volume_threshold,
            'open_cost': open_cost,
            'close_cost': close_cost,
            'min_cost': min_cost,
            'impact_cost': impact_cost,
            'extra_quote': extra_quote,
            'trade_unit': trade_unit  # 明确传递trade_unit给父类
        }
        
        # 只添加非None的quote_cls
        if quote_cls is not None:
            parent_kwargs['quote_cls'] = quote_cls
        
        # 过滤掉自定义参数，只传递父类支持的参数
        filtered_kwargs = {}
        for key, value in kwargs.items():
            if key not in ['t_plus_1', 'min_trade_value', 'trade_unit']:
                filtered_kwargs[key] = value
        
        super().__init__(**parent_kwargs, **filtered_kwargs)
        
        # 中国市场特有属性
        self.t_plus_1 = True  # T+1交易制度
        self.trade_unit = trade_unit  # 交易单位
        self.position_limits = {}  # 持仓限制记录
        self.buy_records = {}  # 买入记录（用于T+1限制）
        
        logger.info(f"中国A股交易所初始化完成")
        logger.info(f"交易时间: {start_time} - {end_time}")
        logger.info(f"交易成本: 开仓{open_cost:.4f}, 平仓{close_cost:.4f}")
        logger.info(f"涨跌停阈值: ±{limit_threshold:.1%}")
    
    def check_stock_limit(self, stock_id: str, current_time: pd.Timestamp) -> Dict[str, float]:
        """
        检查股票涨跌停限制
        
        Args:
            stock_id: 股票代码
            current_time: 当前时间
            
        Returns:
            包含涨停价和跌停价的字典
        """
        try:
            # 获取前一交易日收盘价
            prev_close = self.get_deal_price(stock_id, current_time - pd.Timedelta(days=1))
            if prev_close is None or pd.isna(prev_close):
                return {'limit_up': None, 'limit_down': None}
            
            # 确定涨跌停幅度
            limit_pct = self._get_limit_percentage(stock_id, current_time)
            
            # 计算涨跌停价格
            limit_up = prev_close * (1 + limit_pct)
            limit_down = prev_close * (1 - limit_pct)
            
            return {
                'limit_up': round(limit_up, 2),
                'limit_down': round(limit_down, 2),
                'prev_close': prev_close,
                'limit_pct': limit_pct
            }
            
        except Exception as e:
            logger.debug(f"检查 {stock_id} 涨跌停失败: {e}")
            return {'limit_up': None, 'limit_down': None}
    
    def _get_limit_percentage(self, stock_id: str, current_time: pd.Timestamp) -> float:
        """
        获取股票涨跌停百分比
        
        Args:
            stock_id: 股票代码
            current_time: 当前时间
            
        Returns:
            涨跌停百分比
        """
        try:
            # ST股票判断
            if 'ST' in stock_id or stock_id.startswith('*ST'):
                return self.LIMIT_CONFIG['st']
            
            # 科创板判断 (688xxx)
            if stock_id.startswith('688'):
                return self.LIMIT_CONFIG['kcb']
            
            # 创业板注册制判断 (300xxx, 2020年8月24日后)
            if stock_id.startswith('300') and current_time >= pd.Timestamp('2020-08-24'):
                return self.LIMIT_CONFIG['cyb_reg']
            
            # 新股首日判断（简化处理）
            # 实际应该检查上市日期
            
            # 默认普通股票
            return self.LIMIT_CONFIG['normal']
            
        except Exception:
            return self.LIMIT_CONFIG['normal']
    
    def check_t_plus_1_restriction(self, stock_id: str, current_time: pd.Timestamp) -> bool:
        """
        检查T+1限制，判断股票是否可以卖出
        
        Args:
            stock_id: 股票代码
            current_time: 当前时间
            
        Returns:
            是否可以卖出
        """
        if not self.t_plus_1:
            return True
        
        # 检查是否有当日买入记录
        today_str = current_time.strftime('%Y-%m-%d')
        buy_key = f"{stock_id}_{today_str}"
        
        if buy_key in self.buy_records:
            logger.debug(f"{stock_id} 当日买入，T+1限制无法卖出")
            return False
        
        return True
    
    def record_buy_transaction(self, stock_id: str, current_time: pd.Timestamp, amount: float):
        """
        记录买入交易（用于T+1限制检查）
        
        Args:
            stock_id: 股票代码
            current_time: 交易时间
            amount: 买入数量
        """
        today_str = current_time.strftime('%Y-%m-%d')
        buy_key = f"{stock_id}_{today_str}"
        
        if buy_key not in self.buy_records:
            self.buy_records[buy_key] = 0
        
        self.buy_records[buy_key] += amount
        logger.debug(f"记录买入: {stock_id}, 数量: {amount}")
    
    def calculate_trading_cost(self, 
                             stock_id: str,
                             amount: float, 
                             price: float,
                             direction: str) -> float:
        """
        计算交易成本（佣金、印花税、过户费等）
        
        Args:
            stock_id: 股票代码
            amount: 交易数量
            price: 交易价格
            direction: 交易方向 ('buy' 或 'sell')
            
        Returns:
            总交易成本
        """
        try:
            trade_value = abs(amount) * price
            total_cost = 0.0
            
            # 佣金（买卖都收取）
            commission = max(
                trade_value * self.TRADING_COSTS['commission_rate'],
                self.TRADING_COSTS['min_commission']
            )
            total_cost += commission
            
            # 印花税（仅卖出时收取）
            if direction.lower() == 'sell':
                stamp_tax = trade_value * self.TRADING_COSTS['stamp_tax']
                total_cost += stamp_tax
            
            # 过户费（沪市股票，买卖都收取）
            if stock_id.startswith('6'):  # 沪市
                transfer_fee = max(
                    trade_value * self.TRADING_COSTS['transfer_fee'],
                    self.TRADING_COSTS['min_transfer_fee']
                )
                total_cost += transfer_fee
            
            return round(total_cost, 2)
            
        except Exception as e:
            logger.error(f"计算交易成本失败: {e}")
            return 0.0
    
    def is_trading_time(self, current_time: pd.Timestamp) -> bool:
        """
        检查是否在交易时间内
        
        Args:
            current_time: 当前时间
            
        Returns:
            是否在交易时间内
        """
        try:
            # 检查是否为工作日
            if current_time.weekday() >= 5:  # 周六日
                return False
            
            # 获取时间
            current_hour = current_time.hour
            current_minute = current_time.minute
            current_total_minutes = current_hour * 60 + current_minute
            
            # 上午交易时间: 9:30-11:30
            morning_start = 9 * 60 + 30    # 9:30
            morning_end = 11 * 60 + 30     # 11:30
            
            # 下午交易时间: 13:00-15:00  
            afternoon_start = 13 * 60      # 13:00
            afternoon_end = 15 * 60        # 15:00
            
            return (morning_start <= current_total_minutes <= morning_end or 
                   afternoon_start <= current_total_minutes <= afternoon_end)
            
        except Exception:
            return True  # 默认允许交易（日频回测）
    
    def validate_order(self, order: Order, current_time: pd.Timestamp) -> Dict[str, Any]:
        """
        验证订单是否有效
        
        Args:
            order: 订单对象
            current_time: 当前时间
            
        Returns:
            验证结果字典
        """
        result = {
            'valid': True,
            'reason': '',
            'adjusted_amount': order.amount,
            'limit_price': None
        }
        
        try:
            stock_id = order.stock_id
            direction = order.direction
            amount = order.amount
            
            # 1. 检查交易时间
            if not self.is_trading_time(current_time):
                result.update({
                    'valid': False, 
                    'reason': '非交易时间'
                })
                return result
            
            # 2. T+1限制检查（卖出）
            if direction == OrderDir.SELL:
                if not self.check_t_plus_1_restriction(stock_id, current_time):
                    result.update({
                        'valid': False,
                        'reason': 'T+1限制，当日买入股票次日方可卖出'
                    })
                    return result
            
            # 3. 涨跌停检查
            limit_info = self.check_stock_limit(stock_id, current_time)
            if limit_info['limit_up'] is not None:
                current_price = self.get_deal_price(stock_id, current_time)
                
                if current_price is not None:
                    # 买入时检查涨停
                    if direction == OrderDir.BUY and current_price >= limit_info['limit_up']:
                        result.update({
                            'valid': False,
                            'reason': f'股票涨停，无法买入 (涨停价:{limit_info["limit_up"]})'
                        })
                        return result
                    
                    # 卖出时检查跌停  
                    if direction == OrderDir.SELL and current_price <= limit_info['limit_down']:
                        result.update({
                            'valid': False,
                            'reason': f'股票跌停，无法卖出 (跌停价:{limit_info["limit_down"]})'
                        })
                        return result
            
            # 4. 最小交易单位检查
            if amount % self.trade_unit != 0 and direction == OrderDir.BUY:
                adjusted_amount = int(amount // self.trade_unit) * self.trade_unit
                if adjusted_amount <= 0:
                    result.update({
                        'valid': False,
                        'reason': f'买入数量不足最小交易单位{self.trade_unit}股'
                    })
                    return result
                
                result['adjusted_amount'] = adjusted_amount
                logger.debug(f"调整买入数量: {amount} -> {adjusted_amount}")
            
            # 5. 停牌检查（简化处理）
            current_volume = self.get_volume(stock_id, current_time)
            if current_volume is not None and current_volume == 0:
                result.update({
                    'valid': False,
                    'reason': '股票停牌'
                })
                return result
            
        except Exception as e:
            logger.error(f"验证订单失败: {e}")
            result.update({
                'valid': False,
                'reason': f'订单验证异常: {e}'
            })
        
        return result
    
    def process_order(self, order: Order, current_time: pd.Timestamp):
        """
        处理订单，包含中国市场特殊逻辑
        
        Args:
            order: 订单对象
            current_time: 当前时间
            
        Returns:
            处理后的订单结果
        """
        try:
            # 验证订单
            validation = self.validate_order(order, current_time)
            
            if not validation['valid']:
                logger.info(f"订单被拒绝: {order.stock_id} {order.direction} "
                           f"{order.amount}, 原因: {validation['reason']}")
                return None
            
            # 调整订单数量
            if validation['adjusted_amount'] != order.amount:
                order.amount = validation['adjusted_amount']
            
            # 记录买入交易（用于T+1检查）
            if order.direction == OrderDir.BUY:
                self.record_buy_transaction(order.stock_id, current_time, order.amount)
            
            # 调用父类处理逻辑
            result = super().process_order(order, current_time)
            
            # 计算实际交易成本
            if result and hasattr(result, 'deal_amount') and result.deal_amount > 0:
                actual_cost = self.calculate_trading_cost(
                    order.stock_id,
                    result.deal_amount, 
                    result.deal_price,
                    'buy' if order.direction == OrderDir.BUY else 'sell'
                )
                
                # 将成本信息添加到结果中
                if hasattr(result, 'cost'):
                    result.cost = actual_cost
            
            return result
            
        except Exception as e:
            logger.error(f"处理订单失败: {e}")
            return None
    
    def get_volume(self, stock_id: str, current_time: pd.Timestamp) -> Optional[float]:
        """获取股票成交量"""
        try:
            data = D.get_data(
                instruments=stock_id,
                start_time=current_time,
                end_time=current_time,
                fields=['volume']
            )
            
            if data.empty:
                return None
            
            return data.iloc[0]['volume']
            
        except Exception as e:
            logger.debug(f"获取 {stock_id} 成交量失败: {e}")
            return None
    
    def clean_expired_records(self, current_time: pd.Timestamp):
        """
        清理过期的买入记录（T+1限制用）
        
        Args:
            current_time: 当前时间
        """
        try:
            current_date_str = current_time.strftime('%Y-%m-%d')
            expired_keys = []
            
            for key in self.buy_records:
                record_date = key.split('_')[-1]
                if record_date < current_date_str:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.buy_records[key]
            
            if expired_keys:
                logger.debug(f"清理过期买入记录: {len(expired_keys)}条")
                
        except Exception as e:
            logger.debug(f"清理过期记录失败: {e}")
    
    def get_market_status(self, current_time: pd.Timestamp) -> Dict[str, Any]:
        """
        获取市场状态信息
        
        Args:
            current_time: 当前时间
            
        Returns:
            市场状态字典
        """
        return {
            'trading_time': self.is_trading_time(current_time),
            'market_type': '中国A股',
            'exchange_rules': {
                'T+1': self.t_plus_1,
                'trade_unit': self.trade_unit,
                'limit_threshold': self.limit_threshold
            },
            'trading_costs': self.TRADING_COSTS,
            'current_session': self._get_trading_session(current_time)
        }
    
    def _get_trading_session(self, current_time: pd.Timestamp) -> str:
        """获取当前交易时段"""
        if not self.is_trading_time(current_time):
            return '休市'
        
        hour = current_time.hour
        minute = current_time.minute
        current_minutes = hour * 60 + minute
        
        morning_start = 9 * 60 + 30
        morning_end = 11 * 60 + 30
        afternoon_start = 13 * 60
        
        if morning_start <= current_minutes <= morning_end:
            return '上午交易'
        elif current_minutes >= afternoon_start:
            return '下午交易'
        else:
            return '中午休市'