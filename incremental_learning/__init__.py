#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.8增量学习模块
包含增量学习引擎、实时特征计算、自适应评分等核心组件
"""

__version__ = "3.8.0"
__author__ = "Claude Code"

# 核心组件导入
from .engines import IncrementalLearningEngine
from .features import RealtimeFeatureCalculator
from .scoring import AdaptiveScoringSystem

__all__ = [
    'IncrementalLearningEngine',
    'RealtimeFeatureCalculator',
    'AdaptiveScoringSystem'
]