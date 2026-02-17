#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.8模型漂移检测器
检测模型性能漂移、数据分布漂移和概念漂移

Phase 3: 增量学习机制 - ModelDriftDetector组件
- 统计漂移检测
- 性能退化检测
- 数据分布变化检测
- 概念漂移识别
- 自动触发重训练建议

Created: 2025-09-16
Author: Claude Code
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency, mannwhitneyu
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import joblib

# 添加项目路径
sys.path.append('/Users/yangxu/StockTradebyZ')

class ModelDriftDetector:
    """
    模型漂移检测器

    核心功能：
    1. 性能漂移检测 - 监控模型准确率下降
    2. 数据漂移检测 - 检测输入特征分布变化
    3. 概念漂移检测 - 检测输入输出关系变化
    4. 统计显著性测试
    5. 自动重训练触发
    6. 漂移严重性评估
    """

    def __init__(self,
                 drift_config: Dict,
                 logger: logging.Logger,
                 reference_window: int = 100,
                 detection_window: int = 50):

        self.drift_config = drift_config
        self.logger = logger
        self.reference_window = reference_window
        self.detection_window = detection_window

        # 漂移检测历史
        self.drift_history = []
        self.reference_data = {}
        self.reference_performance = {}

        # 漂移阈值配置
        self.drift_thresholds = {
            'performance_decline': 0.1,      # 性能下降阈值
            'statistical_significance': 0.05, # 统计显著性阈值
            'feature_drift_threshold': 0.1,   # 特征漂移阈值
            'concept_drift_threshold': 0.15,  # 概念漂移阈值
            'severe_drift_threshold': 0.2     # 严重漂移阈值
        }

        # 漂移检测器组件
        self.drift_detectors = {
            'performance': PerformanceDriftDetector(logger),
            'statistical': StatisticalDriftDetector(logger),
            'distribution': DistributionDriftDetector(logger),
            'concept': ConceptDriftDetector(logger)
        }

        self.logger.info("🔍 模型漂移检测器初始化完成")

    def detect_drift(self,
                    model: Any,
                    model_name: str,
                    current_features: pd.DataFrame,
                    current_targets: pd.Series,
                    predictions: np.ndarray = None) -> Dict[str, Any]:
        """
        综合漂移检测

        Args:
            model: 当前模型
            model_name: 模型名称
            current_features: 当前特征数据
            current_targets: 当前目标值
            predictions: 模型预测值（可选）

        Returns:
            漂移检测结果
        """
        self.logger.info(f"🔍 开始漂移检测 - 模型: {model_name}, 样本数: {len(current_features)}")

        detection_result = {
            'model_name': model_name,
            'timestamp': datetime.now(),
            'sample_size': len(current_features),
            'drift_detected': False,
            'drift_types': [],
            'drift_severity': 'none'
        }

        try:
            # 生成预测值（如果没有提供）
            if predictions is None:
                predictions = self._make_predictions(model, model_name, current_features)

            # 1. 性能漂移检测
            performance_drift = self.drift_detectors['performance'].detect(
                current_targets, predictions, self.reference_performance.get(model_name)
            )
            detection_result['performance_drift'] = performance_drift

            # 2. 统计漂移检测
            statistical_drift = self.drift_detectors['statistical'].detect(
                current_features, self.reference_data.get(model_name, {}).get('features')
            )
            detection_result['statistical_drift'] = statistical_drift

            # 3. 分布漂移检测
            distribution_drift = self.drift_detectors['distribution'].detect(
                current_features, self.reference_data.get(model_name, {}).get('features')
            )
            detection_result['distribution_drift'] = distribution_drift

            # 4. 概念漂移检测
            concept_drift = self.drift_detectors['concept'].detect(
                current_features, current_targets, predictions,
                self.reference_data.get(model_name, {})
            )
            detection_result['concept_drift'] = concept_drift

            # 5. 综合漂移评估
            comprehensive_assessment = self._assess_comprehensive_drift(detection_result)
            detection_result.update(comprehensive_assessment)

            # 6. 更新参考数据
            self._update_reference_data(model_name, current_features, current_targets, predictions)

            # 7. 记录漂移历史
            self._record_drift_history(detection_result)

            # 8. 生成建议
            recommendations = self._generate_drift_recommendations(detection_result)
            detection_result['recommendations'] = recommendations

            drift_level = detection_result['drift_severity']
            self.logger.info(f"✅ {model_name} 漂移检测完成 - 漂移等级: {drift_level}")

            return detection_result

        except Exception as e:
            self.logger.error(f"❌ {model_name} 漂移检测失败: {e}")
            detection_result['error'] = str(e)
            return detection_result

    def _assess_comprehensive_drift(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        综合漂移评估
        """
        assessment = {
            'drift_detected': False,
            'drift_types': [],
            'drift_severity': 'none',
            'drift_score': 0.0,
            'requires_retraining': False
        }

        try:
            drift_scores = []
            detected_drifts = []

            # 评估各类型漂移
            drift_checks = [
                ('performance', detection_result.get('performance_drift', {})),
                ('statistical', detection_result.get('statistical_drift', {})),
                ('distribution', detection_result.get('distribution_drift', {})),
                ('concept', detection_result.get('concept_drift', {}))
            ]

            for drift_type, drift_data in drift_checks:
                if drift_data.get('drift_detected', False):
                    detected_drifts.append(drift_type)
                    drift_scores.append(drift_data.get('drift_magnitude', 0))

            # 计算综合漂移分数
            if drift_scores:
                overall_drift_score = np.mean(drift_scores)
                max_drift_score = np.max(drift_scores)
            else:
                overall_drift_score = 0
                max_drift_score = 0

            # 确定漂移严重性
            if max_drift_score >= self.drift_thresholds['severe_drift_threshold']:
                severity = 'severe'
                requires_retraining = True
            elif max_drift_score >= self.drift_thresholds['concept_drift_threshold']:
                severity = 'moderate'
                requires_retraining = True
            elif max_drift_score >= self.drift_thresholds['feature_drift_threshold']:
                severity = 'mild'
                requires_retraining = False
            else:
                severity = 'none'
                requires_retraining = False

            assessment.update({
                'drift_detected': len(detected_drifts) > 0,
                'drift_types': detected_drifts,
                'drift_severity': severity,
                'drift_score': overall_drift_score,
                'requires_retraining': requires_retraining,
                'drift_details': {
                    'num_drift_types': len(detected_drifts),
                    'max_drift_score': max_drift_score,
                    'drift_type_scores': dict(zip(
                        [check[0] for check in drift_checks],
                        [check[1].get('drift_magnitude', 0) for check in drift_checks]
                    ))
                }
            })

        except Exception as e:
            self.logger.error(f"❌ 综合漂移评估失败: {e}")
            assessment['error'] = str(e)

        return assessment

    def _generate_drift_recommendations(self, detection_result: Dict[str, Any]) -> List[str]:
        """
        生成漂移应对建议
        """
        recommendations = []

        try:
            severity = detection_result.get('drift_severity', 'none')
            drift_types = detection_result.get('drift_types', [])

            if severity == 'severe':
                recommendations.extend([
                    "🚨 检测到严重模型漂移，建议立即重新训练模型",
                    "暂停当前模型的预测服务，启用备用模型",
                    "收集更多最新数据进行模型更新"
                ])

            elif severity == 'moderate':
                recommendations.extend([
                    "⚠️ 检测到中等程度漂移，建议在1-2天内重新训练模型",
                    "增加模型监控频率，密切关注性能变化",
                    "准备重训练数据和计算资源"
                ])

            elif severity == 'mild':
                recommendations.extend([
                    "📊 检测到轻微漂移，建议一周内进行模型更新",
                    "继续监控模型性能，观察漂移趋势"
                ])

            # 针对特定类型的漂移给出建议
            if 'performance' in drift_types:
                recommendations.append("• 性能漂移：检查是否有新的市场条件或数据质量问题")

            if 'statistical' in drift_types:
                recommendations.append("• 统计漂移：特征分布发生变化，考虑特征工程优化")

            if 'distribution' in drift_types:
                recommendations.append("• 分布漂移：数据分布偏移，可能需要重新采样或数据预处理")

            if 'concept' in drift_types:
                recommendations.append("• 概念漂移：输入输出关系变化，需要重新学习新的模式")

            if not recommendations:
                recommendations.append("✅ 未检测到显著漂移，模型状态良好")

        except Exception as e:
            recommendations.append(f"❌ 建议生成失败: {str(e)}")

        return recommendations

    def _make_predictions(self, model: Any, model_name: str, features: pd.DataFrame) -> np.ndarray:
        """
        统一的模型预测接口
        """
        try:
            if model_name == 'lightgbm':
                return model.predict(features)
            elif model_name == 'xgboost':
                import xgboost as xgb
                dtest = xgb.DMatrix(features)
                return model.predict(dtest)
            elif model_name in ['catboost', 'random_forest', 'neural_network']:
                return model.predict(features)
            else:
                if hasattr(model, 'predict'):
                    return model.predict(features)
                else:
                    raise ValueError(f"Model {model_name} doesn't have predict method")

        except Exception as e:
            raise Exception(f"Prediction failed for {model_name}: {str(e)}")

    def _update_reference_data(self,
                              model_name: str,
                              features: pd.DataFrame,
                              targets: pd.Series,
                              predictions: np.ndarray):
        """
        更新参考数据
        """
        try:
            # 初始化模型的参考数据
            if model_name not in self.reference_data:
                self.reference_data[model_name] = {}

            # 保持滑动窗口的参考数据
            current_ref = self.reference_data[model_name]

            # 合并新数据到参考数据
            if 'features' in current_ref:
                combined_features = pd.concat([current_ref['features'], features])
                combined_targets = pd.concat([current_ref['targets'], targets])
                combined_predictions = np.concatenate([current_ref['predictions'], predictions])

                # 保持参考窗口大小
                if len(combined_features) > self.reference_window:
                    start_idx = len(combined_features) - self.reference_window
                    combined_features = combined_features.iloc[start_idx:]
                    combined_targets = combined_targets.iloc[start_idx:]
                    combined_predictions = combined_predictions[start_idx:]

            else:
                combined_features = features
                combined_targets = targets
                combined_predictions = predictions

            # 更新参考数据
            self.reference_data[model_name] = {
                'features': combined_features,
                'targets': combined_targets,
                'predictions': combined_predictions,
                'last_updated': datetime.now()
            }

            # 更新参考性能
            self.reference_performance[model_name] = {
                'r2': r2_score(combined_targets, combined_predictions),
                'mse': mean_squared_error(combined_targets, combined_predictions),
                'mae': mean_absolute_error(combined_targets, combined_predictions)
            }

        except Exception as e:
            self.logger.error(f"❌ 参考数据更新失败: {e}")

    def _record_drift_history(self, detection_result: Dict[str, Any]):
        """
        记录漂移检测历史
        """
        history_record = {
            'timestamp': detection_result['timestamp'],
            'model_name': detection_result['model_name'],
            'drift_detected': detection_result.get('drift_detected', False),
            'drift_severity': detection_result.get('drift_severity', 'none'),
            'drift_score': detection_result.get('drift_score', 0),
            'drift_types': detection_result.get('drift_types', [])
        }

        self.drift_history.append(history_record)

        # 保持历史记录长度
        max_history = 200
        if len(self.drift_history) > max_history:
            self.drift_history = self.drift_history[-max_history:]

    def get_drift_summary(self) -> Dict[str, Any]:
        """
        获取漂移检测摘要
        """
        if not self.drift_history:
            return {'status': 'no_history'}

        recent_detections = self.drift_history[-20:]

        summary = {
            'total_detections': len(self.drift_history),
            'recent_drift_rate': np.mean([d['drift_detected'] for d in recent_detections]),
            'severe_drift_count': len([d for d in recent_detections if d['drift_severity'] == 'severe']),
            'moderate_drift_count': len([d for d in recent_detections if d['drift_severity'] == 'moderate']),
            'mild_drift_count': len([d for d in recent_detections if d['drift_severity'] == 'mild']),
            'average_drift_score': np.mean([d['drift_score'] for d in recent_detections]),
            'drift_trend': self._calculate_drift_trend(),
            'most_common_drift_types': self._get_common_drift_types(recent_detections),
            'last_detection': self.drift_history[-1] if self.drift_history else None
        }

        return summary

    def _calculate_drift_trend(self) -> str:
        """
        计算漂移趋势
        """
        if len(self.drift_history) < 10:
            return 'insufficient_data'

        recent_scores = [d['drift_score'] for d in self.drift_history[-10:]]
        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]

        if trend > 0.01:
            return 'increasing'
        elif trend < -0.01:
            return 'decreasing'
        else:
            return 'stable'

    def _get_common_drift_types(self, detections: List[Dict]) -> Dict[str, int]:
        """
        获取常见的漂移类型
        """
        drift_type_counts = {}
        for detection in detections:
            for drift_type in detection.get('drift_types', []):
                drift_type_counts[drift_type] = drift_type_counts.get(drift_type, 0) + 1

        return dict(sorted(drift_type_counts.items(), key=lambda x: x[1], reverse=True))


# 具体漂移检测器实现类

class PerformanceDriftDetector:
    """性能漂移检测器"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def detect(self, current_targets: pd.Series, predictions: np.ndarray, reference_performance: Dict = None) -> Dict[str, Any]:
        """
        检测性能漂移
        """
        result = {
            'drift_detected': False,
            'drift_magnitude': 0.0,
            'detection_method': 'performance_comparison'
        }

        try:
            # 计算当前性能
            current_r2 = r2_score(current_targets, predictions)
            current_mse = mean_squared_error(current_targets, predictions)

            result['current_performance'] = {
                'r2': current_r2,
                'mse': current_mse
            }

            if reference_performance is None:
                result['status'] = 'no_reference'
                return result

            # 与参考性能比较
            reference_r2 = reference_performance.get('r2', current_r2)
            reference_mse = reference_performance.get('mse', current_mse)

            r2_decline = reference_r2 - current_r2
            mse_increase = (current_mse - reference_mse) / max(reference_mse, 0.001)

            # 判断是否存在显著的性能下降
            performance_decline_threshold = 0.1

            if r2_decline > performance_decline_threshold or mse_increase > 0.5:
                result['drift_detected'] = True
                result['drift_magnitude'] = max(r2_decline, mse_increase * 0.2)

            result['performance_comparison'] = {
                'r2_decline': r2_decline,
                'mse_increase': mse_increase,
                'reference_performance': reference_performance
            }

        except Exception as e:
            result['error'] = str(e)

        return result


class StatisticalDriftDetector:
    """统计漂移检测器"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def detect(self, current_features: pd.DataFrame, reference_features: pd.DataFrame = None) -> Dict[str, Any]:
        """
        统计检验漂移检测
        """
        result = {
            'drift_detected': False,
            'drift_magnitude': 0.0,
            'detection_method': 'statistical_tests'
        }

        if reference_features is None:
            result['status'] = 'no_reference'
            return result

        try:
            drift_p_values = []
            drift_statistics = []

            # 对每个特征进行统计检验
            for col in current_features.columns:
                if col in reference_features.columns:
                    current_values = current_features[col].dropna()
                    reference_values = reference_features[col].dropna()

                    if len(current_values) > 10 and len(reference_values) > 10:
                        # KS检验
                        ks_stat, ks_p = ks_2samp(reference_values, current_values)

                        # Mann-Whitney U检验
                        try:
                            mw_stat, mw_p = mannwhitneyu(reference_values, current_values, alternative='two-sided')
                        except:
                            mw_p = 1.0

                        # 取更严格的p值
                        min_p = min(ks_p, mw_p)
                        drift_p_values.append(min_p)
                        drift_statistics.append(ks_stat)

            if drift_p_values:
                # Bonferroni校正
                min_p_value = min(drift_p_values)
                bonferroni_corrected_p = min_p_value * len(drift_p_values)

                # 判断是否存在显著漂移
                if bonferroni_corrected_p < 0.05:
                    result['drift_detected'] = True
                    result['drift_magnitude'] = 1 - min_p_value  # 转换为漂移强度

                result['statistical_details'] = {
                    'min_p_value': min_p_value,
                    'bonferroni_corrected_p': bonferroni_corrected_p,
                    'num_features_tested': len(drift_p_values),
                    'significant_features': sum(1 for p in drift_p_values if p < 0.05)
                }

        except Exception as e:
            result['error'] = str(e)

        return result


class DistributionDriftDetector:
    """分布漂移检测器"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def detect(self, current_features: pd.DataFrame, reference_features: pd.DataFrame = None) -> Dict[str, Any]:
        """
        分布变化检测
        """
        result = {
            'drift_detected': False,
            'drift_magnitude': 0.0,
            'detection_method': 'distribution_comparison'
        }

        if reference_features is None:
            result['status'] = 'no_reference'
            return result

        try:
            distribution_changes = []

            for col in current_features.columns:
                if col in reference_features.columns:
                    current_values = current_features[col].dropna()
                    reference_values = reference_features[col].dropna()

                    if len(current_values) > 20 and len(reference_values) > 20:
                        # 计算分布统计量的变化
                        current_mean = current_values.mean()
                        reference_mean = reference_values.mean()
                        current_std = current_values.std()
                        reference_std = reference_values.std()

                        # 均值变化
                        mean_change = abs(current_mean - reference_mean) / (abs(reference_mean) + 0.001)

                        # 标准差变化
                        std_change = abs(current_std - reference_std) / (reference_std + 0.001)

                        # 分布形状变化（偏度和峰度）
                        current_skew = stats.skew(current_values)
                        reference_skew = stats.skew(reference_values)
                        skew_change = abs(current_skew - reference_skew)

                        # 综合分布变化指标
                        distribution_change = (mean_change * 0.4 + std_change * 0.4 + skew_change * 0.2)
                        distribution_changes.append(distribution_change)

            if distribution_changes:
                max_change = max(distribution_changes)
                avg_change = np.mean(distribution_changes)

                # 判断是否存在显著的分布漂移
                if max_change > 0.2 or avg_change > 0.1:
                    result['drift_detected'] = True
                    result['drift_magnitude'] = max_change

                result['distribution_details'] = {
                    'max_distribution_change': max_change,
                    'average_distribution_change': avg_change,
                    'num_features_analyzed': len(distribution_changes),
                    'significant_changes': sum(1 for c in distribution_changes if c > 0.15)
                }

        except Exception as e:
            result['error'] = str(e)

        return result


class ConceptDriftDetector:
    """概念漂移检测器"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def detect(self, current_features: pd.DataFrame, current_targets: pd.Series,
              predictions: np.ndarray, reference_data: Dict = None) -> Dict[str, Any]:
        """
        概念漂移检测（输入输出关系变化）
        """
        result = {
            'drift_detected': False,
            'drift_magnitude': 0.0,
            'detection_method': 'concept_analysis'
        }

        if not reference_data or 'features' not in reference_data:
            result['status'] = 'no_reference'
            return result

        try:
            reference_features = reference_data['features']
            reference_targets = reference_data['targets']

            # 计算特征与目标的相关性变化
            correlation_changes = []

            for col in current_features.columns:
                if col in reference_features.columns:
                    # 当前数据的相关性
                    current_corr = current_features[col].corr(current_targets)
                    # 参考数据的相关性
                    reference_corr = reference_features[col].corr(reference_targets)

                    if not (np.isnan(current_corr) or np.isnan(reference_corr)):
                        corr_change = abs(current_corr - reference_corr)
                        correlation_changes.append(corr_change)

            # 预测残差分析
            residual_analysis = self._analyze_residual_patterns(
                current_features, current_targets, predictions,
                reference_data.get('predictions')
            )

            # 综合概念漂移评估
            if correlation_changes:
                max_corr_change = max(correlation_changes)
                avg_corr_change = np.mean(correlation_changes)

                # 结合残差分析
                residual_drift = residual_analysis.get('drift_magnitude', 0)

                # 综合概念漂移分数
                concept_drift_score = (max_corr_change * 0.6 + residual_drift * 0.4)

                if concept_drift_score > 0.15:
                    result['drift_detected'] = True
                    result['drift_magnitude'] = concept_drift_score

                result['concept_details'] = {
                    'max_correlation_change': max_corr_change,
                    'average_correlation_change': avg_corr_change,
                    'residual_analysis': residual_analysis,
                    'concept_drift_score': concept_drift_score
                }

        except Exception as e:
            result['error'] = str(e)

        return result

    def _analyze_residual_patterns(self, current_features, current_targets, predictions,
                                  reference_predictions=None):
        """
        分析残差模式变化
        """
        analysis = {'drift_magnitude': 0.0}

        try:
            current_residuals = current_targets - predictions

            if reference_predictions is not None:
                reference_targets = current_targets.iloc[:len(reference_predictions)]
                reference_residuals = reference_targets - reference_predictions[:len(reference_targets)]

                # 残差分布比较
                current_residual_std = current_residuals.std()
                reference_residual_std = reference_residuals.std()

                residual_std_change = abs(current_residual_std - reference_residual_std) / (reference_residual_std + 0.001)

                # 残差自相关性变化
                current_residual_autocorr = self._calculate_autocorrelation(current_residuals)
                reference_residual_autocorr = self._calculate_autocorrelation(reference_residuals)

                autocorr_change = abs(current_residual_autocorr - reference_residual_autocorr)

                # 综合残差漂移
                residual_drift = (residual_std_change * 0.7 + autocorr_change * 0.3)

                analysis.update({
                    'drift_magnitude': residual_drift,
                    'residual_std_change': residual_std_change,
                    'autocorr_change': autocorr_change
                })

        except Exception as e:
            analysis['error'] = str(e)

        return analysis

    def _calculate_autocorrelation(self, series, lag=1):
        """
        计算序列的自相关性
        """
        try:
            if len(series) > lag:
                return series.iloc[lag:].corr(series.iloc[:-lag])
            else:
                return 0.0
        except:
            return 0.0


def main():
    """测试模型漂移检测器"""
    print("🔍 测试模型漂移检测器...")

    # 配置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 创建漂移检测器
    drift_config = {
        'performance_threshold': 0.1,
        'statistical_threshold': 0.05,
        'distribution_threshold': 0.1
    }

    drift_detector = ModelDriftDetector(
        drift_config=drift_config,
        logger=logger
    )

    # 生成测试数据
    np.random.seed(42)
    n_samples = 200
    n_features = 8

    # 创建基础数据
    base_features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    base_targets = pd.Series(
        base_features.sum(axis=1) + np.random.randn(n_samples) * 0.2,
        name='target'
    )

    # 创建简单模型
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(base_features, base_targets)

    # 第一次检测（建立基准）
    print("\n🔍 第一次检测（建立基准）...")
    first_detection = drift_detector.detect_drift(
        model=model,
        model_name='test_model',
        current_features=base_features,
        current_targets=base_targets
    )
    print(f"基准建立 - 漂移严重程度: {first_detection['drift_severity']}")

    # 创建漂移数据（特征分布变化）
    print("\n🔍 第二次检测（引入特征漂移）...")
    drift_features = base_features.copy()
    drift_features['feature_0'] += 0.5  # 特征0发生漂移
    drift_features['feature_1'] *= 1.3  # 特征1方差增大

    drift_targets = pd.Series(
        drift_features.sum(axis=1) + np.random.randn(n_samples) * 0.3,  # 噪声增大
        name='target'
    )

    second_detection = drift_detector.detect_drift(
        model=model,
        model_name='test_model',
        current_features=drift_features,
        current_targets=drift_targets
    )

    print(f"漂移检测结果:")
    print(f"  漂移检测: {second_detection['drift_detected']}")
    print(f"  漂移严重程度: {second_detection['drift_severity']}")
    print(f"  漂移分数: {second_detection.get('drift_score', 0):.3f}")
    print(f"  漂移类型: {second_detection.get('drift_types', [])}")

    # 显示建议
    if 'recommendations' in second_detection:
        print(f"\n📋 建议:")
        for rec in second_detection['recommendations']:
            print(f"  • {rec}")

    # 创建严重概念漂移数据
    print("\n🔍 第三次检测（引入概念漂移）...")
    concept_drift_features = base_features.copy()
    # 改变特征与目标的关系
    concept_drift_targets = pd.Series(
        -concept_drift_features.sum(axis=1) + np.random.randn(n_samples) * 0.4,  # 关系反转
        name='target'
    )

    third_detection = drift_detector.detect_drift(
        model=model,
        model_name='test_model',
        current_features=concept_drift_features,
        current_targets=concept_drift_targets
    )

    print(f"概念漂移检测结果:")
    print(f"  漂移检测: {third_detection['drift_detected']}")
    print(f"  漂移严重程度: {third_detection['drift_severity']}")
    print(f"  漂移分数: {third_detection.get('drift_score', 0):.3f}")
    print(f"  需要重训练: {third_detection.get('requires_retraining', False)}")

    # 获取漂移摘要
    print(f"\n📊 漂移检测摘要:")
    summary = drift_detector.get_drift_summary()
    for key, value in summary.items():
        if key not in ['last_detection', 'most_common_drift_types']:
            print(f"  {key}: {value}")

    if 'most_common_drift_types' in summary:
        print(f"  最常见漂移类型: {summary['most_common_drift_types']}")

    print("\n✅ 模型漂移检测器测试完成！")

if __name__ == "__main__":
    main()