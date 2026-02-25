"""
Alpha158 特征集 - Microsoft Qlib 标准 baseline

158个纯OHLCV特征，用于学术标准对比评估。
"""

from .alpha158_features import Alpha158FeatureCalculator

__all__ = ['Alpha158FeatureCalculator']
