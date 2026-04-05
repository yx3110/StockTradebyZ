"""Daily Selection NG — Next Generation trend-following factor model.

v1.1.0: Industry excess labels, CS rank + residual factors, ICIR adaptive weights.
"""

from .ng_trainer import (
    NGTrainer, ALL_FEATURE_NAMES, STOCK_FEATURE_NAMES, MARKET_FEATURE_NAMES,
    NG_VERSION, NG_V1_VERSION,
)
from .ng_production_scorer import NGProductionScorer

__all__ = [
    'NGTrainer', 'NGProductionScorer',
    'ALL_FEATURE_NAMES', 'STOCK_FEATURE_NAMES', 'MARKET_FEATURE_NAMES',
    'NG_VERSION', 'NG_V1_VERSION',
]
