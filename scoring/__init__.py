"""
Scoring System Package
优化后的股票评分系统

基于3949只股票实际选股表现数据优化的多因子评分框架
"""

from .core_scorer import StockScorer, ScoringConfig
from .factor_calculator import FactorCalculator
from .scoring_engine import ScoringEngine

__version__ = "2.0.0"
__author__ = "Claude Code Enhancement System"

# 默认配置
DEFAULT_CONFIG = {
    "momentum_factor_weight": 0.40,
    "mean_reversion_weight": 0.25,
    "volume_breakout_weight": 0.20,
    "relative_performance_weight": 0.10,
    "stability_factor_weight": 0.05,
    "buy_threshold": 75.0,
    "cautious_buy_threshold": 70.0,
    "watch_threshold": 60.0
}

__all__ = [
    'StockScorer',
    'ScoringConfig', 
    'FactorCalculator',
    'ScoringEngine',
    'DEFAULT_CONFIG'
]