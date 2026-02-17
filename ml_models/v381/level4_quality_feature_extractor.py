#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 Level 4 质量特征提取器
从V380的Level 1-3预测结果中提取25维质量特征，用于训练质量元学习器

特征设计：
1. L1预测统计 (5维): mean, std, min, max, range
2. L2预测统计 (4维): technical, fundamental, macro, sentiment
3. L3最终评分 (1维): final_score
4. 预测一致性 (3维): L1_consistency, L2_consistency, L1L2_consistency
5. 模型置信度 (3维): L1_confidence, L2_confidence, meta_confidence
6. 特征质量 (3维): feature_completeness, feature_variance, outlier_ratio
7. 时间稳定性 (2维): temporal_variance, trend_consistency
8. 市场匹配 (2维): market_regime_match, volatility_match
9. 交叉验证 (2维): cv_score, model_consistency
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from scipy import stats
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class Level4QualityFeatureExtractor:
    """Level 4 质量特征提取器"""

    def __init__(self):
        self.feature_names = self._initialize_feature_names()

    def _initialize_feature_names(self) -> List[str]:
        """初始化25维特征名称"""
        feature_names = [
            # 1. L1预测统计 (5维)
            'l1_mean', 'l1_std', 'l1_min', 'l1_max', 'l1_range',

            # 2. L2预测统计 (4维)
            'l2_technical', 'l2_fundamental', 'l2_macro', 'l2_sentiment',

            # 3. L3最终评分 (1维)
            'l3_final_score',

            # 4. 预测一致性 (3维)
            'l1_consistency', 'l2_consistency', 'l1l2_consistency',

            # 5. 模型置信度 (3维)
            'l1_confidence', 'l2_confidence', 'meta_confidence',

            # 6. 特征质量 (3维)
            'feature_completeness', 'feature_variance', 'outlier_ratio',

            # 7. 时间稳定性 (2维)
            'temporal_variance', 'trend_consistency',

            # 8. 市场匹配 (2维)
            'market_regime_match', 'volatility_match',

            # 9. 交叉验证 (2维)
            'cv_score', 'model_consistency'
        ]
        return feature_names

    def extract_quality_features(self, prediction_data: Dict[str, Any],
                                market_regime: str = "normal",
                                stock_volatility: float = 0.02) -> np.ndarray:
        """
        从V380预测结果中提取25维质量特征

        Args:
            prediction_data: V380预测结果字典
            market_regime: 市场环境 ("bull", "bear", "normal", "volatile")
            stock_volatility: 股票波动率

        Returns:
            np.ndarray: 25维质量特征向量
        """
        try:
            features = np.zeros(25)

            # 提取基础数据
            level1_preds = prediction_data.get('level1_predictions', {})
            level2_preds = prediction_data.get('level2_predictions', {})
            final_score = prediction_data.get('overall_score', 50.0)
            confidence_score = prediction_data.get('confidence_score', 0.5)
            raw_predictions = prediction_data.get('raw_predictions', {})

            # 1. L1预测统计 (5维)
            l1_stats = self._calculate_l1_statistics(level1_preds)
            features[0:5] = l1_stats

            # 2. L2预测统计 (4维)
            l2_stats = self._calculate_l2_statistics(level2_preds)
            features[5:9] = l2_stats

            # 3. L3最终评分 (1维)
            features[9] = self._normalize_score(final_score)

            # 4. 预测一致性 (3维)
            consistency_stats = self._calculate_consistency(level1_preds, level2_preds)
            features[10:13] = consistency_stats

            # 5. 模型置信度 (3维)
            confidence_stats = self._calculate_confidence_stats(
                level1_preds, level2_preds, confidence_score
            )
            features[13:16] = confidence_stats

            # 6. 特征质量 (3维)
            quality_stats = self._calculate_feature_quality(prediction_data)
            features[16:19] = quality_stats

            # 7. 时间稳定性 (2维)
            stability_stats = self._calculate_temporal_stability(raw_predictions)
            features[19:21] = stability_stats

            # 8. 市场匹配 (2维)
            market_stats = self._calculate_market_match(final_score, market_regime, stock_volatility)
            features[21:23] = market_stats

            # 9. 交叉验证 (2维)
            cv_stats = self._calculate_cv_stats(level1_preds, level2_preds)
            features[23:25] = cv_stats

            # 确保所有特征在合理范围内
            features = np.clip(features, 0.0, 1.0)

            return features

        except Exception as e:
            logger.error(f"质量特征提取失败: {e}")
            # 返回默认特征向量
            return np.full(25, 0.5)

    def _calculate_l1_statistics(self, level1_preds: Dict[str, float]) -> np.ndarray:
        """计算Level 1预测统计特征 (5维)"""
        try:
            if not level1_preds:
                return np.array([0.5, 0.1, 0.0, 1.0, 1.0])

            values = list(level1_preds.values())
            if len(values) == 0:
                return np.array([0.5, 0.1, 0.0, 1.0, 1.0])

            # 标准化到0-1范围
            normalized_values = [(v + 10) / 20 for v in values if v is not None]  # 假设预测范围[-10, 10]

            if len(normalized_values) == 0:
                return np.array([0.5, 0.1, 0.0, 1.0, 1.0])

            mean_val = np.mean(normalized_values)
            std_val = np.std(normalized_values) if len(normalized_values) > 1 else 0.1
            min_val = np.min(normalized_values)
            max_val = np.max(normalized_values)
            range_val = max_val - min_val

            return np.array([mean_val, std_val, min_val, max_val, range_val])

        except Exception as e:
            logger.debug(f"L1统计计算失败: {e}")
            return np.array([0.5, 0.1, 0.0, 1.0, 1.0])

    def _calculate_l2_statistics(self, level2_preds: Dict[str, float]) -> np.ndarray:
        """计算Level 2专家预测统计 (4维)"""
        try:
            # 预期的专家类型
            expert_types = ['technical', 'fundamental', 'macro', 'sentiment']
            expert_values = []

            for expert_type in expert_types:
                # 寻找包含该专家类型的预测
                expert_pred = 0.5  # 默认值
                for key, value in level2_preds.items():
                    if expert_type in key.lower():
                        expert_pred = (value + 10) / 20  # 标准化到0-1
                        break
                expert_values.append(expert_pred)

            return np.array(expert_values)

        except Exception as e:
            logger.debug(f"L2统计计算失败: {e}")
            return np.array([0.5, 0.5, 0.5, 0.5])

    def _normalize_score(self, score: float) -> float:
        """标准化评分到0-1范围"""
        return np.clip(score / 100.0, 0.0, 1.0)

    def _calculate_consistency(self, level1_preds: Dict[str, float],
                             level2_preds: Dict[str, float]) -> np.ndarray:
        """计算预测一致性 (3维)"""
        try:
            # L1一致性：基于方差
            l1_consistency = 0.5
            if level1_preds:
                l1_values = [v for v in level1_preds.values() if v is not None]
                if len(l1_values) > 1:
                    cv = np.std(l1_values) / (abs(np.mean(l1_values)) + 1e-6)
                    l1_consistency = 1.0 / (1.0 + cv)  # 低变异系数=高一致性

            # L2一致性：专家间一致性
            l2_consistency = 0.5
            if level2_preds:
                l2_values = [v for v in level2_preds.values() if v is not None]
                if len(l2_values) > 1:
                    cv = np.std(l2_values) / (abs(np.mean(l2_values)) + 1e-6)
                    l2_consistency = 1.0 / (1.0 + cv)

            # L1-L2跨层一致性：比较平均预测方向
            l1l2_consistency = 0.5
            if level1_preds and level2_preds:
                l1_mean = np.mean([v for v in level1_preds.values() if v is not None])
                l2_mean = np.mean([v for v in level2_preds.values() if v is not None])

                # 计算方向一致性
                if (l1_mean > 0 and l2_mean > 0) or (l1_mean <= 0 and l2_mean <= 0):
                    l1l2_consistency = 0.8  # 方向一致
                else:
                    l1l2_consistency = 0.3  # 方向不一致

            return np.array([l1_consistency, l2_consistency, l1l2_consistency])

        except Exception as e:
            logger.debug(f"一致性计算失败: {e}")
            return np.array([0.5, 0.5, 0.5])

    def _calculate_confidence_stats(self, level1_preds: Dict[str, float],
                                  level2_preds: Dict[str, float],
                                  meta_confidence: float) -> np.ndarray:
        """计算模型置信度统计 (3维)"""
        try:
            # L1置信度：基于预测分布
            l1_conf = 0.5
            if level1_preds:
                l1_values = [v for v in level1_preds.values() if v is not None]
                if len(l1_values) > 0:
                    # 高绝对值=高置信度
                    l1_conf = min(np.mean([abs(v) for v in l1_values]) / 10.0, 1.0)

            # L2置信度：专家一致性
            l2_conf = 0.5
            if level2_preds:
                l2_values = [v for v in level2_preds.values() if v is not None]
                if len(l2_values) > 1:
                    # 低方差=高置信度
                    var = np.var(l2_values)
                    l2_conf = 1.0 / (1.0 + var)

            # Meta置信度：直接使用
            meta_conf = np.clip(meta_confidence, 0.0, 1.0)

            return np.array([l1_conf, l2_conf, meta_conf])

        except Exception as e:
            logger.debug(f"置信度计算失败: {e}")
            return np.array([0.5, 0.5, 0.5])

    def _calculate_feature_quality(self, prediction_data: Dict[str, Any]) -> np.ndarray:
        """计算特征质量指标 (3维)"""
        try:
            # 特征完整性：有多少个有效预测
            level1_preds = prediction_data.get('level1_predictions', {})
            level2_preds = prediction_data.get('level2_predictions', {})

            total_expected = 8  # 期望的预测数量 (5个L1 + 4个L2模型的4个周期)
            total_actual = len([v for v in level1_preds.values() if v is not None]) + \
                          len([v for v in level2_preds.values() if v is not None])

            completeness = min(total_actual / total_expected, 1.0) if total_expected > 0 else 0.5

            # 特征方差：预测的变异程度
            all_preds = list(level1_preds.values()) + list(level2_preds.values())
            all_preds = [v for v in all_preds if v is not None]

            variance = 0.2  # 默认中等方差
            if len(all_preds) > 1:
                variance = min(np.var(all_preds) / 100.0, 1.0)  # 标准化

            # 异常值比例：超出合理范围的预测比例
            outlier_ratio = 0.1  # 默认低异常值比例
            if len(all_preds) > 0:
                # 定义异常值为超出[-10, 10]范围的预测
                outliers = [v for v in all_preds if abs(v) > 10]
                outlier_ratio = len(outliers) / len(all_preds)

            return np.array([completeness, variance, outlier_ratio])

        except Exception as e:
            logger.debug(f"特征质量计算失败: {e}")
            return np.array([0.5, 0.2, 0.1])

    def _calculate_temporal_stability(self, raw_predictions: Dict[str, float]) -> np.ndarray:
        """计算时间稳定性 (2维)"""
        try:
            # 从多期间预测计算时间方差
            periods = ['target_1d', 'target_3d', 'target_5d', 'target_10d']
            period_values = []

            for period in periods:
                if period in raw_predictions and raw_predictions[period] is not None:
                    period_values.append(raw_predictions[period])

            temporal_variance = 0.2  # 默认中等方差
            if len(period_values) > 1:
                variance = np.var(period_values)
                temporal_variance = min(variance / 100.0, 1.0)  # 标准化

            # 趋势一致性：短期vs长期预测方向
            trend_consistency = 0.5
            if len(period_values) >= 2:
                short_term = period_values[0]  # 1日
                long_term = period_values[-1]  # 10日或最后一个

                if (short_term > 0 and long_term > 0) or (short_term <= 0 and long_term <= 0):
                    trend_consistency = 0.8  # 趋势一致
                else:
                    trend_consistency = 0.3  # 趋势不一致

            return np.array([temporal_variance, trend_consistency])

        except Exception as e:
            logger.debug(f"时间稳定性计算失败: {e}")
            return np.array([0.2, 0.5])

    def _calculate_market_match(self, final_score: float, market_regime: str,
                              stock_volatility: float) -> np.ndarray:
        """计算市场匹配度 (2维)"""
        try:
            # 市场环境匹配度
            regime_scores = {
                'bull': 0.8 if final_score > 60 else 0.4,    # 牛市偏好高分股票
                'bear': 0.8 if final_score < 40 else 0.4,    # 熊市偏好低风险
                'normal': 0.6,                               # 正常市场中性
                'volatile': 0.7 if 40 < final_score < 70 else 0.5  # 波动市场偏好中等分数
            }

            regime_match = regime_scores.get(market_regime, 0.5)

            # 波动性匹配度
            # 高波动股票在高分时可能风险较大
            volatility_match = 0.5
            if stock_volatility < 0.015:  # 低波动
                volatility_match = 0.7
            elif stock_volatility > 0.035:  # 高波动
                volatility_match = 0.8 if final_score > 70 else 0.3
            else:  # 中等波动
                volatility_match = 0.6

            return np.array([regime_match, volatility_match])

        except Exception as e:
            logger.debug(f"市场匹配计算失败: {e}")
            return np.array([0.5, 0.5])

    def _calculate_cv_stats(self, level1_preds: Dict[str, float],
                           level2_preds: Dict[str, float]) -> np.ndarray:
        """计算交叉验证统计 (2维)"""
        try:
            # CV分数：基于预测稳定性的估计
            all_preds = list(level1_preds.values()) + list(level2_preds.values())
            all_preds = [v for v in all_preds if v is not None]

            cv_score = 0.5  # 默认中等CV分数
            if len(all_preds) > 2:
                # 简单的稳定性评估：低方差=高CV分数
                variance = np.var(all_preds)
                cv_score = 1.0 / (1.0 + variance / 10.0)  # 标准化

            # 模型一致性：不同模型预测的相关性
            model_consistency = 0.5
            if len(level1_preds) > 1 and len(level2_preds) > 1:
                l1_values = [v for v in level1_preds.values() if v is not None]
                l2_values = [v for v in level2_preds.values() if v is not None]

                if len(l1_values) > 0 and len(l2_values) > 0:
                    # 简单的相关性估计
                    l1_mean = np.mean(l1_values)
                    l2_mean = np.mean(l2_values)

                    # 基于符号一致性
                    if (l1_mean > 0 and l2_mean > 0) or (l1_mean <= 0 and l2_mean <= 0):
                        model_consistency = 0.7
                    else:
                        model_consistency = 0.3

            return np.array([cv_score, model_consistency])

        except Exception as e:
            logger.debug(f"CV统计计算失败: {e}")
            return np.array([0.5, 0.5])

    def extract_batch_features(self, prediction_batch: List[Dict[str, Any]],
                             market_regime: str = "normal") -> np.ndarray:
        """批量提取质量特征"""
        try:
            feature_matrix = []

            for prediction_data in prediction_batch:
                features = self.extract_quality_features(prediction_data, market_regime)
                feature_matrix.append(features)

            return np.array(feature_matrix)

        except Exception as e:
            logger.error(f"批量特征提取失败: {e}")
            # 返回默认特征矩阵
            return np.full((len(prediction_batch), 25), 0.5)

    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        return self.feature_names.copy()

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """获取特征重要性分组"""
        return {
            'Level1_Statistics': self.feature_names[0:5],
            'Level2_Expert': self.feature_names[5:9],
            'Level3_Final': self.feature_names[9:10],
            'Consistency': self.feature_names[10:13],
            'Confidence': self.feature_names[13:16],
            'Feature_Quality': self.feature_names[16:19],
            'Temporal_Stability': self.feature_names[19:21],
            'Market_Match': self.feature_names[21:23],
            'Cross_Validation': self.feature_names[23:25]
        }

# 使用示例
if __name__ == "__main__":
    extractor = Level4QualityFeatureExtractor()

    # 模拟V380预测数据
    sample_prediction = {
        'overall_score': 75.0,
        'confidence_score': 0.65,
        'level1_predictions': {
            'lgb_target_1d': 2.3,
            'xgb_target_1d': 1.8,
            'catboost_target_1d': 2.1,
            'rf_target_1d': 1.9,
            'nn_target_1d': 2.0
        },
        'level2_predictions': {
            'technical_expert_target_1d': 1.5,
            'fundamental_expert_target_1d': 2.2,
            'macro_expert_target_1d': 1.8,
            'sentiment_expert_target_1d': 2.0
        },
        'raw_predictions': {
            'target_1d': 2.0,
            'target_3d': 1.8,
            'target_5d': 1.9,
            'target_10d': 2.1
        }
    }

    # 提取质量特征
    features = extractor.extract_quality_features(sample_prediction)
    feature_names = extractor.get_feature_names()

    print("🎯 25维质量特征提取结果:")
    for i, (name, value) in enumerate(zip(feature_names, features)):
        print(f"{i+1:2d}. {name:20s}: {value:.3f}")

    print(f"\n✅ 特征向量维度: {len(features)}")
    print(f"特征范围: [{features.min():.3f}, {features.max():.3f}]")