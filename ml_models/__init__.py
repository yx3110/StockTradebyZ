"""
机器学习模型模块

活跃版本:
  - v3.9  (ml_models.v39) - V390ProductionScorer, V395ProductionScorer
已弃用 (保留向后兼容，将在未来移除):
  - v3.7  (ml_models.v37) - V370AdvancedMLSystem
  - v3.8  (ml_models.v38) - V380AdvancedIncrementalMLSystem
  - v3.81 (ml_models.v381) - V380Level4IntegratedSystem
"""

# Active versions
from .v39.v390_production_scorer import V390ProductionScorer
from .v39.v395_production_scorer import V395ProductionScorer

# Deprecated versions (kept for backward compatibility)
from .v37 import V370AdvancedMLSystem  # DEPRECATED: use V390ProductionScorer
from .v38 import V380AdvancedIncrementalMLSystem  # DEPRECATED: use V390ProductionScorer
from .v381 import V380Level4IntegratedSystem  # DEPRECATED: use V390ProductionScorer

__all__ = [
    # Active
    'V390ProductionScorer',
    'V395ProductionScorer',
    # Deprecated
    'V370AdvancedMLSystem',
    'V380AdvancedIncrementalMLSystem',
    'V380Level4IntegratedSystem',
]
