#!/usr/bin/env python3
"""
TradingAgents Integration System
集成TradingAgents和量化选股系统
"""

__version__ = "1.0.0"
__author__ = "Stock Trading System"
__description__ = "Integration system between quantitative stock selection and TradingAgents AI analysis"

# 导入主要类
from .adapters.china_stock_adapter import ChinaStockAdapter, ChinaMarketDataProvider
from .adapters.china_trading_agents import ChinaTradingAgents, ChinaStockAnalyzer
from .core.report_parser import ReportParser, StockInfo

__all__ = [
    'ChinaStockAdapter',
    'ChinaMarketDataProvider', 
    'ChinaTradingAgents',
    'ChinaStockAnalyzer',
    'ReportParser',
    'StockInfo'
]