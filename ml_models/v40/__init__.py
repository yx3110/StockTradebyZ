#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V4.0 Cross-Sectional Alpha Model

核心改进:
- ~55个 cross-sectional 排名特征 (行业内百分位)
- 超额收益标签 (个股 - 沪深300)
- 学习"哪只股票能跑赢"，而非"大盘涨不涨"

使用示例:
    from ml_models.v40 import V400ProductionScorer

    scorer = V400ProductionScorer()
    result = scorer.predict_score('000001', '2026-02-13')
"""

__version__ = '4.0.0'
__author__ = 'Claude Code'

from .v400_production_scorer import V400ProductionScorer

__all__ = ['V400ProductionScorer']
