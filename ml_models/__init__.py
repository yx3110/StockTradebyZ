"""
机器学习模型模块

活跃版本:
  - v3.9  (ml_models.v39) - V390ProductionScorer, V394ProductionScorer, V395ProductionScorer
  - v3.96 (ml_models.v39) - V396ProductionScorer (Robust Z-Score + Industry-Excess)
  - v5.0  (ml_models.v39) - V500ProductionScorer (Unified Feature Fusion: v39+v40+neural)
已弃用 (保留模型文件，代码仍可用):
  - v3.8  (ml_models.v38) - V380AdvancedIncrementalMLSystem
已删除:
  - v3.7, v3.81 - 代码和模型已移除

模型文件存储: ml_models/trained_models/
训练脚本: ml_models/training/
"""

# Active versions
from .v39.v390_production_scorer import V390ProductionScorer
from .v39.v394_production_scorer import V394ProductionScorer
from .v39.v395_production_scorer import V395ProductionScorer
from .v39.v396_production_scorer import V396ProductionScorer
from .v39.v500_production_scorer import V500ProductionScorer
from .v39.alpha158_production_scorer import Alpha158ProductionScorer

# Deprecated versions (kept for backward compatibility)
try:
    from .v38 import V380AdvancedIncrementalMLSystem  # DEPRECATED: use V390ProductionScorer
except ImportError:
    V380AdvancedIncrementalMLSystem = None

__all__ = [
    # Active
    'V390ProductionScorer',
    'V394ProductionScorer',
    'V395ProductionScorer',
    'V396ProductionScorer',
    'V500ProductionScorer',
    'Alpha158ProductionScorer',
    # Deprecated
    'V380AdvancedIncrementalMLSystem',
]
