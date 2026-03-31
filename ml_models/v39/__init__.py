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
from .v44_production_scorer import V44ProductionScorer, V442ProductionScorer
from .v46_production_scorer import V46ProductionScorer
from .v47_production_scorer import V47ProductionScorer
from .v471_production_scorer import V471ProductionScorer
from .v472_production_scorer import V472ProductionScorer
from .v473_production_scorer import V473ProductionScorer
from .v474_production_scorer import V474ProductionScorer
from .v475_production_scorer import V475ProductionScorer
from .v476_production_scorer import V476ProductionScorer
from .v477_production_scorer import V477ProductionScorer
from .v478_production_scorer import V478ProductionScorer
from .v479_production_scorer import V479ProductionScorer
from .v480_production_scorer import V480ProductionScorer
from .v481_production_scorer import V481ProductionScorer
from .v482_production_scorer import V482ProductionScorer
from .v483_production_scorer import V483ProductionScorer
from .v484_production_scorer import V484ProductionScorer
from .v485_production_scorer import V485ProductionScorer
from .v486_production_scorer import V486ProductionScorer
from .v487_production_scorer import V487ProductionScorer
from .alpha158_production_scorer import Alpha158ProductionScorer
from .v4902_production_scorer import V4902ProductionScorer

# 补全缺失的 scorer 导入
from .v395_production_scorer import V395ProductionScorer
from .v396_production_scorer import V396ProductionScorer
from .v43_production_scorer import V43ProductionScorer

# 导出 (所有已注册的 scorer)
__all__ = [
    'V390EnhancedFeatureMLSystem',
    'V390ProductionScorer',
    'V394ProductionScorer',
    'V395ProductionScorer',
    'V396ProductionScorer',
    'V43ProductionScorer',
    'V500ProductionScorer',
    'V44ProductionScorer',
    'V442ProductionScorer',
    'V46ProductionScorer',
    'V47ProductionScorer',
    'V471ProductionScorer',
    'V472ProductionScorer',
    'V473ProductionScorer',
    'V474ProductionScorer',
    'V475ProductionScorer',
    'V476ProductionScorer',
    'V477ProductionScorer',
    'V478ProductionScorer',
    'V479ProductionScorer',
    'V480ProductionScorer',
    'V481ProductionScorer',
    'V482ProductionScorer',
    'V483ProductionScorer',
    'V484ProductionScorer',
    'V485ProductionScorer',
    'V486ProductionScorer',
    'V487ProductionScorer',
    'Alpha158ProductionScorer',
    'V4902ProductionScorer',
]
