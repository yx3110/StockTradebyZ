"""
v3.9 特征提取模块

包含:
- 24个技术特征 (TechnicalFeaturesV39)
- 10个基本面特征 (FundamentalFeaturesV39)
- 8个市场特征 (MarketFeaturesV39)
- 6个活跃市值特征 (ActiveMarketCapFeaturesV39) - V3.9.4新增
  * 市场层面: market_active_mv_ratio, market_active_mv_zscore, market_active_mv_trend
  * 个股层面: stock_active_mv_rank, stock_relative_liquidity, market_cap_quality_score
"""

from .technical_features import TechnicalFeaturesV39
from .fundamental_features import FundamentalFeaturesV39
from .market_features import MarketFeaturesV39
from .active_market_cap_features import ActiveMarketCapFeaturesV39

__all__ = [
    'TechnicalFeaturesV39',
    'FundamentalFeaturesV39',
    'MarketFeaturesV39',
    'ActiveMarketCapFeaturesV39'
]
