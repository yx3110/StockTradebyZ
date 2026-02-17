"""
机器学习模型模块
包含v3.7、v3.8、v3.81版本的高级ML系统
"""
from .v37 import V370AdvancedMLSystem
from .v38 import V380AdvancedIncrementalMLSystem
from .v381 import V380Level4IntegratedSystem

__all__ = [
    'V370AdvancedMLSystem',
    'V380AdvancedIncrementalMLSystem', 
    'V380Level4IntegratedSystem'
]
