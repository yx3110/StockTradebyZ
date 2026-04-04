"""Daily Selection NG — Next Generation trend-following factor model."""

from .ng_trainer import NGTrainer, ALL_FEATURE_NAMES, STOCK_FEATURE_NAMES, MARKET_FEATURE_NAMES
from .ng_production_scorer import NGProductionScorer

__all__ = [
    'NGTrainer', 'NGProductionScorer',
    'ALL_FEATURE_NAMES', 'STOCK_FEATURE_NAMES', 'MARKET_FEATURE_NAMES',
]
