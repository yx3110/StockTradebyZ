#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生产评分器基类 — 提取v390/v394共享的通用方法

提供:
- 预测值→评分转换
- 置信度计算
- 推荐建议生成 (可配置阈值)
- DB路径解析
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

try:
    from core.config import get_db_path as _get_db_path
    DEFAULT_DB_PATH = str(_get_db_path())
except ImportError:
    DEFAULT_DB_PATH = str(Path(__file__).parent.parent.parent / 'data_adapter' / 'stock_data.db')


class BaseScorerMixin:
    """共享评分工具方法 — 作为Mixin供v390/v394继承"""

    # 子类可覆盖阈值
    _recommendation_thresholds: Dict[str, float] = {
        '强烈买入': 65,
        '买入': 62,
        '谨慎买入': 60,
        '持有观望': 57,
        '谨慎卖出': 54,
    }

    def _convert_prediction_to_score(self, prediction: float) -> float:
        """
        将5日收益率预测转换为0-100评分

        预测值分布: -10% ~ +10%
        映射到: 0 ~ 100分
        """
        prediction = np.clip(prediction, -0.15, 0.15)
        score = (prediction + 0.15) / 0.30 * 100
        return np.clip(score, 0, 100)

    def _calculate_confidence(self, features: pd.DataFrame, prediction: float) -> float:
        """
        计算预测置信度

        基于特征质量和预测强度
        """
        missing_rate = features.isna().sum().sum() / (features.shape[0] * features.shape[1])
        feature_quality = 1.0 - missing_rate
        prediction_strength = min(abs(prediction) / 0.10, 1.0)
        confidence = (feature_quality * 0.4 + prediction_strength * 0.6)
        return np.clip(confidence, 0.3, 0.95)

    def _get_recommendation(self, score: float) -> str:
        """根据评分给出投资建议 (使用_recommendation_thresholds)"""
        t = self._recommendation_thresholds
        if score >= t.get('强烈买入', 65):
            return "强烈买入"
        elif score >= t.get('买入', 62):
            return "买入"
        elif score >= t.get('谨慎买入', 60):
            return "谨慎买入"
        elif score >= t.get('持有观望', 57):
            return "持有观望"
        elif score >= t.get('谨慎卖出', 54):
            return "谨慎卖出"
        else:
            return "卖出"
