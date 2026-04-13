"""Daily Selection NG — Next Generation trend-following factor model.

v1.0.4: Risk-adjusted labels, multi-seed ensemble, IC stability screening, signal smoothing.
"""

from .ng_trainer import (
    NGTrainer, ALL_FEATURE_NAMES, STOCK_FEATURE_NAMES, MARKET_FEATURE_NAMES,
    MONEYFLOW_FEATURE_NAMES, INTERACTION_FEATURE_NAMES,
    SMOOTHING_FEATURE_NAMES, NG104_STOCK_FEATURES, NG104_ALL_FEATURES,
    NG_VERSION, NG_V1_VERSION, NG104_VERSION,
)
from .ng_production_scorer import NGProductionScorer
from .ng_schema import PRODUCTION_VERSION

__all__ = [
    'NGTrainer', 'NGProductionScorer',
    'ALL_FEATURE_NAMES', 'STOCK_FEATURE_NAMES', 'MARKET_FEATURE_NAMES',
    'MONEYFLOW_FEATURE_NAMES', 'INTERACTION_FEATURE_NAMES',
    'SMOOTHING_FEATURE_NAMES', 'NG104_STOCK_FEATURES', 'NG104_ALL_FEATURES',
    'NG_VERSION', 'NG_V1_VERSION', 'NG104_VERSION',
    'PRODUCTION_VERSION',
]
