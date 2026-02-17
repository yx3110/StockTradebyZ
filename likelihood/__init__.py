"""
股票相似度回测系统
Stock Similarity Backtest System

基于时间序列相似度算法的股票走势预测系统
"""

__version__ = '1.0.0'
__author__ = 'StockTradebyZ Team'

# 导入核心组件
from .algorithms.search_engine import SimilaritySearchEngine
from .backtest.backtest_engine import BacktestEngine
from .reports.report_generator import ReportGenerator

# 导入常用工具
from .data_preprocessing.data_loader import DataLoader
from .data_preprocessing.feature_engineering import FeatureEngineer

# 导出的类和函数
__all__ = [
    'SimilaritySearchEngine',
    'BacktestEngine',
    'ReportGenerator',
    'DataLoader',
    'FeatureEngineer'
]

# 默认配置路径
DEFAULT_CONFIG = 'configs/default_config.yaml'