#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9 增强特征机器学习系统

核心组件:
- V3.9.0: 42个增强特征 (24技术 + 10基本面 + 8市场)
- V3.9.4: 48个特征 = 42基础 + 6活跃市值特征 (IC+166%, Top20胜率+8.93%)
- 三层Ensemble ML模型 (LightGBM + XGBoost + RandomForest + CatBoost)

使用示例:
    from ml_models.v39 import V390ProductionScorer, V394ProductionScorer

    # V3.9.0 (42特征)
    scorer_v390 = V390ProductionScorer()
    score = scorer_v390.predict_score('000001', '2025-11-03')

    # V3.9.4 (48特征, 推荐)
    scorer_v394 = V394ProductionScorer()
    score = scorer_v394.predict_score('000001', '2025-11-03')
"""

__version__ = '5.0.0'
__author__ = 'Claude Code'

# 导入核心系统
from .v390_enhanced_feature_ml_system import V390EnhancedFeatureMLSystem
from .v390_production_scorer import V390ProductionScorer
from .v394_production_scorer import V394ProductionScorer
from .v500_production_scorer import V500ProductionScorer

# 导出
__all__ = [
    'V390EnhancedFeatureMLSystem',
    'V390ProductionScorer',
    'V394ProductionScorer',
    'V500ProductionScorer',
]
