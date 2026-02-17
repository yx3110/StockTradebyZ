"""
Hikyuu风格回测框架整合模块

借鉴Hikyuu的优秀设计思想，创建轻量级但功能完整的回测框架，
无需编译C++代码即可获得高效回测能力。

主要组件:
- HikyuuStyleDataAdapter: 数据适配层，连接SQLite数据库
- Query: 查询对象
- KData: K线数据对象
- Stock: 股票对象
- MLScoringSignal: ML评分系统Signal适配器
- HikyuuStyleBacktestEngine: 快速回测引擎

Author: StockTradebyZ Team
Date: 2025-10-10
"""

__version__ = '0.2.1'

# 导入核心组件
from .query import Query
from .kdata import KData
from .stock import Stock
from .data_adapter import HikyuuStyleDataAdapter

# Signal相关
from .signal_base import SignalBase, BBISignal, KDJSignal, CompositeSignal
from .ml_signal_adapter import MLScoringSignal, MLCombinedSignal

# 资金管理
from .money_manager import MoneyManagerBase, MM_FixedCount, MM_FixedPercent, MM_FixedRisk

# 止损策略
from .stop_loss import StopLossBase, ST_FixedPercent, ST_ProfitGoal, ST_Trailing, ST_Composite

# 回测引擎
from .portfolio import Portfolio, Position, Trade
from .broker import Broker
from .backtest_engine import HikyuuStyleBacktestEngine, BacktestResult
from .parallel_backtest_engine import ParallelBacktestEngine

# 缓存管理
from .cache_manager import SmartCacheManager, LRUCache

__all__ = [
    '__version__',
    # 数据层
    'Query',
    'KData',
    'Stock',
    'HikyuuStyleDataAdapter',
    # Signal
    'SignalBase',
    'BBISignal',
    'KDJSignal',
    'CompositeSignal',
    'MLScoringSignal',
    'MLCombinedSignal',
    # 资金管理
    'MoneyManagerBase',
    'MM_FixedCount',
    'MM_FixedPercent',
    'MM_FixedRisk',
    # 止损
    'StopLossBase',
    'ST_FixedPercent',
    'ST_ProfitGoal',
    'ST_Trailing',
    'ST_Composite',
    # 回测引擎
    'Portfolio',
    'Position',
    'Trade',
    'Broker',
    'HikyuuStyleBacktestEngine',
    'ParallelBacktestEngine',
    'BacktestResult',
    # 缓存管理
    'SmartCacheManager',
    'LRUCache',
]
