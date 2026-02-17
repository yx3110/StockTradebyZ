"""
Qlib回测集成模块

将StockTradebyZ的选股策略和数据库集成到Qlib回测框架中，
提供专业级的回测能力和性能分析。

主要组件：
- data_adapter: SQLite数据库 -> Qlib数据格式转换
- strategy_adapter: 现有选股器 -> Qlib策略适配
- chinese_exchange: 中国A股市场特性配置  
- backtest_runner: 回测执行和结果分析

支持的策略：
- 少负战法 (BBIKDJSelector)
- 补票战法 (BBIShortLongSelector)  
- TePu战法 (BreakoutVolumeKDJSelector)
- 填坑战法 (PeakKDJSelector)
"""

__version__ = "1.0.0"
__author__ = "StockTradebyZ Team"

from .data_adapter import StockTradebyzDataAdapter
from .strategy_adapter import StrategyAdapter  
from .stocktrader_strategy import StockTraderStrategy
from .backtest_runner import BacktestRunner

__all__ = [
    "StockTradebyzDataAdapter",
    "StrategyAdapter", 
    "StockTraderStrategy",
    "BacktestRunner"
]