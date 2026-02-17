#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
置信度评估系统 - V3.8自适应评分系统核心组件

提供预测可靠性的量化评估
基于模型不确定性、历史准确性和数据质量的综合评估

Created: 2025-09-16
Author: Claude Code
Version: 3.8.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ConfidenceEstimator:
    """
    置信度评估系统

    核心功能:
    1. 模型不确定性量化
    2. 历史准确性分析
    3. 数据质量评估
    4. 综合置信度计算
    """

    def __init__(self,
                 history_window: int = 100,
                 confidence_levels: List[float] = [0.68, 0.95],
                 min_samples_for_calibration: int = 30,
                 logger: Optional[logging.Logger] = None):
        """
        初始化置信度评估系统

        Args:
            history_window: 历史准确性分析窗口
            confidence_levels: 置信度水平列表
            min_samples_for_calibration: 校准最小样本数
            logger: 日志记录器
        """
        self.history_window = history_window
        self.confidence_levels = confidence_levels
        self.min_samples = min_samples_for_calibration
        self.logger = logger or logging.getLogger(__name__)

        # 历史预测记录
        self.prediction_history = []
        self.accuracy_history = {
            'temporal_accuracies': {},  # 各时间维度准确性
            'strategy_accuracies': {},  # 各策略准确性
            'market_regime_accuracies': {}  # 不同市场状态准确性
        }

        # 校准参数
        self.calibration_params = {
            'score_to_probability': {},
            'uncertainty_scaling': 1.0,
            'confidence_adjustment': 0.0
        }

        # 数据质量指标权重
        self.quality_weights = {
            'completeness': 0.3,
            'consistency': 0.25,
            'timeliness': 0.2,
            'accuracy': 0.15,
            'stability': 0.1
        }

        self.logger.info("置信度评估系统初始化完成")

    def estimate_confidence(self,
                           prediction_scores: Dict[str, float],
                           input_data: pd.DataFrame,
                           model_metadata: Optional[Dict] = None,
                           market_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        估计预测置信度

        Args:
            prediction_scores: 各维度预测评分
            input_data: 输入数据
            model_metadata: 模型元数据
            market_context: 市场环境上下文

        Returns:
            包含置信度估计的详细结果
        """
        try:
            self.logger.info("开始置信度评估")

            # 1. 模型不确定性评估
            model_uncertainty = self._assess_model_uncertainty(
                prediction_scores,
                model_metadata or {}
            )

            # 2. 历史准确性评估
            historical_reliability = self._assess_historical_reliability(
                prediction_scores,
                market_context or {}
            )

            # 3. 数据质量评估
            data_quality = self._assess_data_quality(input_data)

            # 4. 市场环境影响评估
            market_impact = self._assess_market_impact(
                market_context or {},
                input_data
            )

            # 5. 综合置信度计算
            confidence_metrics = self._calculate_comprehensive_confidence(
                model_uncertainty,
                historical_reliability,
                data_quality,
                market_impact
            )

            # 6. 置信区间估计
            confidence_intervals = self._estimate_confidence_intervals(
                prediction_scores,
                confidence_metrics
            )

            # 7. 风险评估
            risk_assessment = self._assess_prediction_risks(
                prediction_scores,
                confidence_metrics,
                market_context or {}
            )

            result = {
                'confidence_score': confidence_metrics['overall_confidence'],
                'confidence_level': self._categorize_confidence(confidence_metrics['overall_confidence']),
                'confidence_intervals': confidence_intervals,
                'component_confidences': {
                    'model_uncertainty': model_uncertainty,
                    'historical_reliability': historical_reliability,
                    'data_quality': data_quality,
                    'market_impact': market_impact
                },
                'risk_assessment': risk_assessment,
                'reliability_factors': self._identify_reliability_factors(
                    model_uncertainty, historical_reliability, data_quality, market_impact
                ),
                'timestamp': datetime.now()
            }

            # 记录预测用于后续校准
            self._record_prediction(prediction_scores, result)

            self.logger.info(f"置信度评估完成 - 综合置信度: {confidence_metrics['overall_confidence']:.3f}")
            return result

        except Exception as e:
            self.logger.error(f"置信度评估失败: {e}")
            raise

    def _assess_model_uncertainty(self,
                                prediction_scores: Dict[str, float],
                                model_metadata: Dict) -> Dict[str, float]:
        """评估模型不确定性"""

        try:
            uncertainties = {}

            # 1. 评分一致性不确定性
            scores_list = list(prediction_scores.values())
            if len(scores_list) > 1:
                score_std = np.std(scores_list)
                score_range = max(scores_list) - min(scores_list)
                consistency_uncertainty = min(score_std * 2 + score_range * 0.5, 1.0)
            else:
                consistency_uncertainty = 0.5

            uncertainties['score_consistency'] = consistency_uncertainty

            # 2. 模型复杂度不确定性
            model_complexity = model_metadata.get('model_complexity', 0.5)
            complexity_uncertainty = min(model_complexity * 0.3, 0.3)
            uncertainties['model_complexity'] = complexity_uncertainty

            # 3. 特征重要性分散度
            feature_importances = model_metadata.get('feature_importances', [])
            if feature_importances:
                importance_entropy = -np.sum([p * np.log(p + 1e-8) for p in feature_importances])
                max_entropy = np.log(len(feature_importances))
                importance_uncertainty = importance_entropy / (max_entropy + 1e-8) * 0.2
            else:
                importance_uncertainty = 0.1

            uncertainties['feature_importance'] = importance_uncertainty

            # 4. 输出稳定性不确定性
            output_stability = model_metadata.get('output_stability', 0.8)
            stability_uncertainty = (1 - output_stability) * 0.3
            uncertainties['output_stability'] = stability_uncertainty

            # 综合模型不确定性
            total_uncertainty = sum(uncertainties.values())
            model_confidence = max(0, 1 - total_uncertainty)

            return {
                'model_confidence': model_confidence,
                'uncertainty_components': uncertainties,
                'total_uncertainty': total_uncertainty
            }

        except Exception as e:
            self.logger.warning(f"模型不确定性评估失败: {e}")
            return {
                'model_confidence': 0.5,
                'uncertainty_components': {},
                'total_uncertainty': 0.5
            }

    def _assess_historical_reliability(self,
                                     prediction_scores: Dict[str, float],
                                     market_context: Dict) -> Dict[str, float]:
        """评估历史可靠性"""

        try:
            if len(self.prediction_history) < self.min_samples:
                return {
                    'overall_reliability': 0.5,
                    'sample_size': len(self.prediction_history),
                    'temporal_reliability': {},
                    'market_regime_reliability': {}
                }

            # 获取最近的历史记录
            recent_history = self.prediction_history[-self.history_window:]

            # 1. 整体历史准确性
            accuracies = [record.get('accuracy', 0.5) for record in recent_history if record.get('accuracy') is not None]
            overall_accuracy = np.mean(accuracies) if accuracies else 0.5

            # 2. 各时间维度可靠性
            temporal_reliability = {}
            for timeframe in ['short_term', 'medium_term', 'long_term']:
                timeframe_accuracies = [
                    record.get('temporal_accuracies', {}).get(timeframe, 0.5)
                    for record in recent_history
                    if 'temporal_accuracies' in record
                ]
                if timeframe_accuracies:
                    temporal_reliability[timeframe] = np.mean(timeframe_accuracies)
                else:
                    temporal_reliability[timeframe] = 0.5

            # 3. 市场状态可靠性
            current_market_regime = market_context.get('market_regime', 'normal')
            regime_accuracies = [
                record.get('accuracy', 0.5)
                for record in recent_history
                if record.get('market_context', {}).get('market_regime') == current_market_regime
            ]

            regime_reliability = np.mean(regime_accuracies) if regime_accuracies else 0.5

            # 4. 趋势准确性
            trend_accuracies = []
            for record in recent_history:
                predicted_score = record.get('predicted_score', 0.5)
                actual_outcome = record.get('actual_outcome')

                # 跳过没有实际结果的记录
                if actual_outcome is None:
                    continue

                predicted_direction = 1 if predicted_score > 0.5 else -1
                actual_direction = 1 if actual_outcome > 0.5 else -1
                trend_accuracy = 1.0 if predicted_direction == actual_direction else 0.0
                trend_accuracies.append(trend_accuracy)

            trend_reliability = np.mean(trend_accuracies) if trend_accuracies else 0.5

            # 5. 校准质量 (预测概率与实际结果的一致性)
            calibration_score = self._assess_calibration_quality(recent_history)

            # 综合可靠性评分
            reliability_components = {
                'overall_accuracy': overall_accuracy,
                'trend_accuracy': trend_reliability,
                'regime_accuracy': regime_reliability,
                'calibration_quality': calibration_score
            }

            overall_reliability = (
                overall_accuracy * 0.4 +
                trend_reliability * 0.3 +
                regime_reliability * 0.2 +
                calibration_score * 0.1
            )

            return {
                'overall_reliability': overall_reliability,
                'sample_size': len(recent_history),
                'temporal_reliability': temporal_reliability,
                'market_regime_reliability': {current_market_regime: regime_reliability},
                'reliability_components': reliability_components
            }

        except Exception as e:
            self.logger.warning(f"历史可靠性评估失败: {e}")
            return {
                'overall_reliability': 0.5,
                'sample_size': 0,
                'temporal_reliability': {},
                'market_regime_reliability': {}
            }

    def _assess_calibration_quality(self, history: List[Dict]) -> float:
        """评估预测校准质量"""

        try:
            if len(history) < 20:
                return 0.5

            # 将预测分为若干区间，过滤None值
            valid_records = [(record.get('predicted_score', 0.5), record.get('actual_outcome'))
                           for record in history
                           if record.get('actual_outcome') is not None]

            if not valid_records:
                return 0.5

            predicted_probs = [r[0] for r in valid_records]
            actual_outcomes = [r[1] for r in valid_records]

            # 计算每个区间的校准误差
            bins = np.linspace(0, 1, 11)  # 10个区间
            calibration_errors = []

            for i in range(len(bins) - 1):
                bin_mask = (np.array(predicted_probs) >= bins[i]) & (np.array(predicted_probs) < bins[i + 1])
                if bin_mask.sum() == 0:
                    continue

                bin_predicted = np.mean(np.array(predicted_probs)[bin_mask])
                bin_actual = np.mean(np.array(actual_outcomes)[bin_mask])
                calibration_errors.append(abs(bin_predicted - bin_actual))

            if calibration_errors:
                mean_calibration_error = np.mean(calibration_errors)
                calibration_quality = max(0, 1 - mean_calibration_error * 5)  # 缩放到[0,1]
            else:
                calibration_quality = 0.5

            return calibration_quality

        except Exception:
            return 0.5

    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, float]:
        """评估数据质量"""

        try:
            quality_scores = {}

            # 1. 完整性评分
            if not data.empty:
                missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
                completeness_score = max(0, 1 - missing_ratio * 2)
            else:
                completeness_score = 0
            quality_scores['completeness'] = completeness_score

            # 2. 一致性评分 (数据变化的平滑性)
            consistency_scores = []
            for col in data.select_dtypes(include=[np.number]).columns:
                if len(data[col].dropna()) > 2:
                    values = data[col].dropna().values
                    changes = np.diff(values)
                    if len(changes) > 0:
                        change_std = np.std(changes)
                        value_std = np.std(values)
                        consistency = 1 - min(change_std / (value_std + 1e-8), 1.0)
                        consistency_scores.append(consistency)

            consistency_score = np.mean(consistency_scores) if consistency_scores else 0.5
            quality_scores['consistency'] = consistency_score

            # 3. 时效性评分 (基于数据更新时间)
            if 'trade_date' in data.columns:
                latest_date = pd.to_datetime(data['trade_date'].max())
                current_date = datetime.now()
                days_old = (current_date - latest_date).days
                timeliness_score = max(0, 1 - days_old / 7)  # 7天内为最佳
            else:
                timeliness_score = 0.7  # 假设较新
            quality_scores['timeliness'] = timeliness_score

            # 4. 数值准确性评分 (异常值检测)
            accuracy_scores = []
            for col in data.select_dtypes(include=[np.number]).columns:
                if len(data[col].dropna()) > 10:
                    values = data[col].dropna()
                    z_scores = np.abs(stats.zscore(values))
                    outlier_ratio = np.mean(z_scores > 3)
                    accuracy_score = max(0, 1 - outlier_ratio * 3)
                    accuracy_scores.append(accuracy_score)

            accuracy_score = np.mean(accuracy_scores) if accuracy_scores else 0.7
            quality_scores['accuracy'] = accuracy_score

            # 5. 稳定性评分 (时间序列稳定性)
            stability_scores = []
            for col in data.select_dtypes(include=[np.number]).columns:
                if len(data[col].dropna()) >= 20:
                    values = data[col].dropna().values
                    # 计算滚动标准差的稳定性
                    window_size = min(10, len(values) // 2)
                    rolling_std = pd.Series(values).rolling(window_size).std().dropna()
                    if len(rolling_std) > 1:
                        std_stability = 1 - np.std(rolling_std) / (np.mean(rolling_std) + 1e-8)
                        std_stability = max(0, min(std_stability, 1))
                        stability_scores.append(std_stability)

            stability_score = np.mean(stability_scores) if stability_scores else 0.6
            quality_scores['stability'] = stability_score

            # 综合数据质量评分
            overall_quality = sum(
                quality_scores[component] * self.quality_weights[component]
                for component in quality_scores
            )

            return {
                'overall_quality': overall_quality,
                'quality_components': quality_scores,
                'data_size': len(data)
            }

        except Exception as e:
            self.logger.warning(f"数据质量评估失败: {e}")
            return {
                'overall_quality': 0.5,
                'quality_components': {},
                'data_size': 0
            }

    def _assess_market_impact(self,
                            market_context: Dict,
                            input_data: pd.DataFrame) -> Dict[str, float]:
        """评估市场环境对预测可靠性的影响"""

        try:
            impact_factors = {}

            # 1. 波动率影响
            volatility_regime = market_context.get('volatility_regime', 1.0)
            if volatility_regime > 2.0:  # 极高波动
                volatility_impact = 0.3
            elif volatility_regime > 1.5:  # 高波动
                volatility_impact = 0.6
            elif volatility_regime < 0.5:  # 低波动
                volatility_impact = 0.9
            else:  # 正常波动
                volatility_impact = 0.8

            impact_factors['volatility_impact'] = volatility_impact

            # 2. 市场趋势影响
            trend_strength = abs(market_context.get('trend_strength', 0.0))
            if trend_strength > 1.5:  # 强趋势
                trend_impact = 0.8
            elif trend_strength > 0.7:  # 中等趋势
                trend_impact = 0.7
            else:  # 弱趋势/震荡
                trend_impact = 0.5

            impact_factors['trend_impact'] = trend_impact

            # 3. 市场情绪影响
            market_sentiment = abs(market_context.get('market_sentiment', 0.0))
            if market_sentiment > 0.8:  # 极端情绪
                sentiment_impact = 0.4
            elif market_sentiment > 0.5:  # 强情绪
                sentiment_impact = 0.6
            else:  # 中性情绪
                sentiment_impact = 0.8

            impact_factors['sentiment_impact'] = sentiment_impact

            # 4. 流动性影响
            if 'volume' in input_data.columns and len(input_data) >= 5:
                recent_volume = input_data['volume'].iloc[-5:].mean()
                historical_volume = input_data['volume'].mean()
                volume_ratio = recent_volume / (historical_volume + 1e-8)

                if volume_ratio > 2.0:  # 高流动性
                    liquidity_impact = 0.8
                elif volume_ratio < 0.5:  # 低流动性
                    liquidity_impact = 0.5
                else:  # 正常流动性
                    liquidity_impact = 0.7
            else:
                liquidity_impact = 0.6

            impact_factors['liquidity_impact'] = liquidity_impact

            # 综合市场影响评分
            market_reliability = (
                volatility_impact * 0.3 +
                trend_impact * 0.3 +
                sentiment_impact * 0.2 +
                liquidity_impact * 0.2
            )

            return {
                'market_reliability': market_reliability,
                'impact_factors': impact_factors
            }

        except Exception as e:
            self.logger.warning(f"市场影响评估失败: {e}")
            return {
                'market_reliability': 0.6,
                'impact_factors': {}
            }

    def _calculate_comprehensive_confidence(self,
                                          model_uncertainty: Dict,
                                          historical_reliability: Dict,
                                          data_quality: Dict,
                                          market_impact: Dict) -> Dict[str, float]:
        """计算综合置信度"""

        try:
            # 提取各组件的主要分数
            model_conf = model_uncertainty.get('model_confidence', 0.5)
            historical_conf = historical_reliability.get('overall_reliability', 0.5)
            data_conf = data_quality.get('overall_quality', 0.5)
            market_conf = market_impact.get('market_reliability', 0.6)

            # 样本量调整
            sample_size = historical_reliability.get('sample_size', 0)
            sample_adjustment = min(sample_size / self.min_samples, 1.0)

            # 计算基础综合置信度
            base_confidence = (
                model_conf * 0.25 +
                historical_conf * 0.35 +
                data_conf * 0.25 +
                market_conf * 0.15
            )

            # 样本量调整
            adjusted_confidence = base_confidence * 0.7 + (base_confidence * sample_adjustment * 0.3)

            # 额外的保守性调整 (新模型或数据不足时更保守)
            if sample_size < self.min_samples:
                conservative_factor = 0.8
            elif data_conf < 0.6:
                conservative_factor = 0.9
            else:
                conservative_factor = 1.0

            final_confidence = adjusted_confidence * conservative_factor

            # 确保在合理范围内
            final_confidence = np.clip(final_confidence, 0.1, 0.95)

            return {
                'overall_confidence': final_confidence,
                'base_confidence': base_confidence,
                'sample_adjustment': sample_adjustment,
                'conservative_factor': conservative_factor,
                'component_contributions': {
                    'model': model_conf * 0.25,
                    'historical': historical_conf * 0.35,
                    'data': data_conf * 0.25,
                    'market': market_conf * 0.15
                }
            }

        except Exception as e:
            self.logger.warning(f"综合置信度计算失败: {e}")
            return {
                'overall_confidence': 0.5,
                'base_confidence': 0.5,
                'sample_adjustment': 0.0,
                'conservative_factor': 1.0,
                'component_contributions': {}
            }

    def _estimate_confidence_intervals(self,
                                     prediction_scores: Dict[str, float],
                                     confidence_metrics: Dict) -> Dict[str, Dict]:
        """估计置信区间"""

        try:
            intervals = {}
            overall_confidence = confidence_metrics['overall_confidence']

            # 基于历史误差估计区间宽度
            if len(self.prediction_history) >= 20:
                historical_errors = [
                    abs(record.get('predicted_score', 0.5) - record.get('actual_outcome', 0.5))
                    for record in self.prediction_history[-50:]
                    if 'actual_outcome' in record
                ]

                if historical_errors:
                    error_std = np.std(historical_errors)
                    error_mean = np.mean(historical_errors)
                else:
                    error_std = 0.1
                    error_mean = 0.1
            else:
                error_std = 0.15
                error_mean = 0.1

            # 调整区间宽度基于置信度
            confidence_value = overall_confidence if overall_confidence is not None else 0.5
            width_factor = 1.5 - confidence_value  # 置信度低时区间更宽

            for score_name, score_value in prediction_scores.items():
                # 68%置信区间 (1σ)
                width_68 = error_std * width_factor
                intervals[score_name] = {
                    '68%': {
                        'lower': max(0, score_value - width_68),
                        'upper': min(1, score_value + width_68),
                        'width': width_68 * 2
                    }
                }

                # 95%置信区间 (2σ)
                width_95 = error_std * 2 * width_factor
                intervals[score_name]['95%'] = {
                    'lower': max(0, score_value - width_95),
                    'upper': min(1, score_value + width_95),
                    'width': width_95 * 2
                }

            return intervals

        except Exception as e:
            self.logger.warning(f"置信区间估计失败: {e}")
            return {}

    def _assess_prediction_risks(self,
                               prediction_scores: Dict[str, float],
                               confidence_metrics: Dict,
                               market_context: Dict) -> Dict[str, Any]:
        """评估预测风险"""

        try:
            risks = {}

            # 1. 模型风险
            model_confidence = confidence_metrics.get('component_contributions', {}).get('model', 0.5)
            if model_confidence < 0.3:
                risks['model_risk'] = 'high'
            elif model_confidence < 0.6:
                risks['model_risk'] = 'medium'
            else:
                risks['model_risk'] = 'low'

            # 2. 数据风险
            data_quality = confidence_metrics.get('component_contributions', {}).get('data', 0.5)
            if data_quality < 0.4:
                risks['data_risk'] = 'high'
            elif data_quality < 0.7:
                risks['data_risk'] = 'medium'
            else:
                risks['data_risk'] = 'low'

            # 3. 市场风险
            market_volatility = market_context.get('volatility_regime', 1.0)
            if market_volatility > 2.0:
                risks['market_risk'] = 'high'
            elif market_volatility > 1.3:
                risks['market_risk'] = 'medium'
            else:
                risks['market_risk'] = 'low'

            # 4. 预测极端性风险
            extreme_predictions = sum(1 for score in prediction_scores.values() if score < 0.2 or score > 0.8)
            total_predictions = len(prediction_scores)
            extreme_ratio = extreme_predictions / total_predictions if total_predictions > 0 else 0

            if extreme_ratio > 0.5:
                risks['extremity_risk'] = 'high'
            elif extreme_ratio > 0.3:
                risks['extremity_risk'] = 'medium'
            else:
                risks['extremity_risk'] = 'low'

            # 综合风险等级
            risk_scores = {'low': 1, 'medium': 2, 'high': 3}
            avg_risk_score = np.mean([risk_scores[risk] for risk in risks.values()])

            if avg_risk_score > 2.5:
                overall_risk = 'high'
            elif avg_risk_score > 1.5:
                overall_risk = 'medium'
            else:
                overall_risk = 'low'

            risks['overall_risk'] = overall_risk

            return risks

        except Exception as e:
            self.logger.warning(f"风险评估失败: {e}")
            return {'overall_risk': 'medium'}

    def _identify_reliability_factors(self,
                                    model_uncertainty: Dict,
                                    historical_reliability: Dict,
                                    data_quality: Dict,
                                    market_impact: Dict) -> List[str]:
        """识别影响可靠性的关键因素"""

        factors = []

        # 模型相关因素
        if model_uncertainty.get('model_confidence', 0.5) < 0.6:
            factors.append("模型预测不确定性较高")

        # 历史表现因素
        if historical_reliability.get('overall_reliability', 0.5) < 0.6:
            factors.append("历史预测准确率偏低")

        sample_size = historical_reliability.get('sample_size', 0)
        if sample_size < self.min_samples:
            factors.append(f"历史样本不足 ({sample_size}/{self.min_samples})")

        # 数据质量因素
        data_components = data_quality.get('quality_components', {})
        if data_components.get('completeness', 1.0) < 0.8:
            factors.append("数据完整性不足")
        if data_components.get('timeliness', 1.0) < 0.7:
            factors.append("数据时效性较差")

        # 市场环境因素
        impact_factors = market_impact.get('impact_factors', {})
        if impact_factors.get('volatility_impact', 0.8) < 0.5:
            factors.append("市场波动率过高")
        if impact_factors.get('sentiment_impact', 0.8) < 0.5:
            factors.append("市场情绪极端")

        return factors if factors else ["预测可靠性整体良好"]

    def _categorize_confidence(self, confidence_score: float) -> str:
        """将置信度分数转换为类别"""

        if confidence_score >= 0.8:
            return 'very_high'
        elif confidence_score >= 0.65:
            return 'high'
        elif confidence_score >= 0.5:
            return 'medium'
        elif confidence_score >= 0.35:
            return 'low'
        else:
            return 'very_low'

    def _record_prediction(self,
                         prediction_scores: Dict[str, float],
                         confidence_result: Dict):
        """记录预测用于后续校准"""

        record = {
            'timestamp': confidence_result['timestamp'],
            'predicted_score': np.mean(list(prediction_scores.values())),
            'prediction_scores': prediction_scores,
            'confidence_score': confidence_result['confidence_score'],
            'confidence_level': confidence_result['confidence_level'],
            'market_context': confidence_result.get('market_context', {}),
            'actual_outcome': None  # 待后续更新
        }

        self.prediction_history.append(record)

        # 保持历史记录在合理范围
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]

    def update_prediction_outcome(self,
                                prediction_id: int,
                                actual_outcome: float,
                                temporal_outcomes: Optional[Dict[str, float]] = None):
        """更新预测结果用于校准"""

        try:
            if 0 <= prediction_id < len(self.prediction_history):
                self.prediction_history[prediction_id]['actual_outcome'] = actual_outcome

                if temporal_outcomes:
                    self.prediction_history[prediction_id]['temporal_outcomes'] = temporal_outcomes

                # 更新准确性统计
                self._update_accuracy_statistics(prediction_id)

        except Exception as e:
            self.logger.warning(f"更新预测结果失败: {e}")

    def _update_accuracy_statistics(self, prediction_id: int):
        """更新准确性统计"""

        try:
            record = self.prediction_history[prediction_id]
            predicted = record.get('predicted_score', 0.5)
            actual = record.get('actual_outcome')

            if actual is not None:
                # 计算准确性
                accuracy = 1 - abs(predicted - actual)
                record['accuracy'] = accuracy

                # 更新各类准确性历史
                market_regime = record.get('market_context', {}).get('market_regime', 'normal')

                if market_regime not in self.accuracy_history['market_regime_accuracies']:
                    self.accuracy_history['market_regime_accuracies'][market_regime] = []

                self.accuracy_history['market_regime_accuracies'][market_regime].append(accuracy)

                # 限制历史长度
                if len(self.accuracy_history['market_regime_accuracies'][market_regime]) > 100:
                    self.accuracy_history['market_regime_accuracies'][market_regime] = \
                        self.accuracy_history['market_regime_accuracies'][market_regime][-100:]

        except Exception as e:
            self.logger.warning(f"更新准确性统计失败: {e}")

    def get_confidence_summary(self) -> Dict[str, Any]:
        """获取置信度系统摘要"""

        return {
            'total_predictions': len(self.prediction_history),
            'predictions_with_outcomes': len([r for r in self.prediction_history if r.get('actual_outcome') is not None]),
            'average_confidence': np.mean([r['confidence_score'] for r in self.prediction_history]) if self.prediction_history else 0.5,
            'calibration_samples': max(0, len(self.prediction_history) - self.min_samples),
            'confidence_levels': self.confidence_levels,
            'last_prediction': self.prediction_history[-1]['timestamp'] if self.prediction_history else None
        }