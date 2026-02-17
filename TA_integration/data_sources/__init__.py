#!/usr/bin/env python3
"""
中国股票数据源模块
包含东方财富股吧等平台的数据获取接口
"""

from .eastmoney_api import EastMoneyAPI
from .sentiment_integrator import ChineseSentimentIntegrator, get_china_stock_sentiment

__all__ = [
    'EastMoneyAPI', 
    'ChineseSentimentIntegrator',
    'get_china_stock_sentiment'
]