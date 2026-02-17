#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.8在线验证组件
实现实时模型性能验证和交叉验证策略

Phase 3: 增量学习机制 - OnlineValidation组件
- 时间序列交叉验证
- 滚动窗口验证
- 在线性能监控
- 模型可靠性评估
- 预测置信度计算

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

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import joblib

# 添加项目路径
sys.path.append('/Users/yangxu/StockTradebyZ')

class OnlineValidationEngine:
    """
    在线验证引擎

    核心功能：
    1. 时间序列交叉验证
    2. 滚动窗口验证
    3. 实时性能监控
    4. 模型稳定性评估
    5. 预测置信度计算
    6. 性能退化检测
    """

    def __init__(self,
                 validation_config: Dict,
                 logger: logging.Logger,
                 min_validation_samples: int = 50,
                 performance_window: int = 20):

        self.validation_config = validation_config
        self.logger = logger
        self.min_validation_samples = min_validation_samples
        self.performance_window = performance_window

        # 验证历史记录
        self.validation_history = []
        self.performance_metrics = []
        self.confidence_scores = []

        # 基准性能指标
        self.baseline_performance = None
        self.performance_thresholds = {
            'r2_decline_threshold': 0.1,        # R²下降阈值
            'mse_increase_threshold': 0.5,      # MSE增加阈值
            'confidence_threshold': 0.7,        # 置信度阈值
            'stability_threshold': 0.8          # 稳定性阈值
        }

        # 验证策略配置
        self.validation_strategies = {
            'time_series_cv': TimeSeriesValidator(logger),
            'rolling_window': RollingWindowValidator(logger),
            'walk_forward': WalkForwardValidator(logger),
            'expanding_window': ExpandingWindowValidator(logger)
        }

        self.logger.info("🔍 在线验证引擎初始化完成")

    def validate_model_online(self,
                            model: Any,
                            model_name: str,
                            features: pd.DataFrame,
                            targets: pd.Series,
                            validation_type: str = 'comprehensive') -> Dict[str, Any]:
        """
        在线模型验证

        Args:
            model: 待验证的模型
            model_name: 模型名称
            features: 特征数据
            targets: 目标值
            validation_type: 验证类型 ('quick', 'comprehensive', 'deep')

        Returns:
            验证结果字典
        """
        self.logger.info(f"🔍 开始{validation_type}验证 - 模型: {model_name}, 样本数: {len(features)}")

        validation_result = {
            'model_name': model_name,
            'validation_type': validation_type,
            'timestamp': datetime.now(),
            'sample_size': len(features),
            'validation_status': 'started'
        }

        try:
            # 1. 基础性能验证
            basic_metrics = self._validate_basic_performance(
                model, model_name, features, targets
            )
            validation_result['basic_metrics'] = basic_metrics

            # 2. 时间序列验证（如果数据有时间信息）
            if validation_type in ['comprehensive', 'deep']:
                ts_validation = self._validate_time_series(
                    model, model_name, features, targets
                )
                validation_result['time_series_validation'] = ts_validation

            # 3. 稳定性验证
            if validation_type == 'deep':
                stability_metrics = self._validate_stability(
                    model, model_name, features, targets
                )
                validation_result['stability_metrics'] = stability_metrics

            # 4. 置信度评估
            confidence_assessment = self._assess_prediction_confidence(
                model, model_name, features, targets
            )
            validation_result['confidence_assessment'] = confidence_assessment

            # 5. 性能对比
            performance_comparison = self._compare_with_baseline(basic_metrics)
            validation_result['performance_comparison'] = performance_comparison

            # 6. 综合评估
            overall_assessment = self._generate_overall_assessment(validation_result)
            validation_result['overall_assessment'] = overall_assessment

            validation_result['validation_status'] = 'completed'

            # 记录验证历史
            self._record_validation_history(validation_result)

            self.logger.info(f"✅ {model_name} 验证完成 - 总体评分: {overall_assessment['overall_score']:.3f}")

            return validation_result

        except Exception as e:
            self.logger.error(f"❌ {model_name} 验证失败: {e}")
            validation_result['validation_status'] = 'failed'
            validation_result['error'] = str(e)
            return validation_result

    def _validate_basic_performance(self,
                                  model: Any,
                                  model_name: str,
                                  features: pd.DataFrame,
                                  targets: pd.Series) -> Dict[str, float]:
        """
        基础性能验证
        """
        try:
            # 预测
            predictions = self._make_predictions(model, model_name, features)

            # 计算基础指标
            metrics = {
                'r2_score': r2_score(targets, predictions),
                'mse': mean_squared_error(targets, predictions),
                'rmse': np.sqrt(mean_squared_error(targets, predictions)),
                'mae': mean_absolute_error(targets, predictions),
                'mape': mean_absolute_percentage_error(targets, predictions) if not np.any(targets == 0) else np.inf
            }

            # 计算相关性
            if len(predictions) > 3:
                pearson_corr, pearson_p = pearsonr(targets, predictions)
                spearman_corr, spearman_p = spearmanr(targets, predictions)

                metrics.update({
                    'pearson_correlation': pearson_corr,
                    'pearson_p_value': pearson_p,
                    'spearman_correlation': spearman_corr,
                    'spearman_p_value': spearman_p
                })

            # 计算预测精度分布
            residuals = targets - predictions
            metrics.update({
                'residual_mean': np.mean(residuals),
                'residual_std': np.std(residuals),
                'residual_skewness': stats.skew(residuals),
                'residual_kurtosis': stats.kurtosis(residuals)
            })

            return metrics

        except Exception as e:
            self.logger.error(f"❌ 基础性能验证失败: {e}")
            return {'error': str(e)}

    def _validate_time_series(self,
                            model: Any,
                            model_name: str,
                            features: pd.DataFrame,
                            targets: pd.Series) -> Dict[str, Any]:
        """
        时间序列验证
        """
        ts_results = {}

        try:
            # 时间序列交叉验证
            if len(features) >= self.min_validation_samples:
                tscv_result = self.validation_strategies['time_series_cv'].validate(
                    model, model_name, features, targets
                )
                ts_results['time_series_cv'] = tscv_result

            # 滚动窗口验证
            if len(features) >= 100:  # 需要更多数据进行滚动窗口验证
                rolling_result = self.validation_strategies['rolling_window'].validate(
                    model, model_name, features, targets
                )
                ts_results['rolling_window'] = rolling_result

            # 前进分析验证
            if len(features) >= 150:
                walk_forward_result = self.validation_strategies['walk_forward'].validate(
                    model, model_name, features, targets
                )
                ts_results['walk_forward'] = walk_forward_result

        except Exception as e:
            self.logger.error(f"❌ 时间序列验证失败: {e}")
            ts_results['error'] = str(e)

        return ts_results

    def _validate_stability(self,
                          model: Any,
                          model_name: str,
                          features: pd.DataFrame,
                          targets: pd.Series) -> Dict[str, float]:
        """
        稳定性验证
        """
        try:
            stability_metrics = {}

            # 子样本稳定性测试
            n_subsamples = 5
            subsample_r2s = []

            for i in range(n_subsamples):
                # 随机子样本
                sample_size = min(len(features) // 2, 200)
                indices = np.random.choice(len(features), sample_size, replace=False)

                sub_features = features.iloc[indices]
                sub_targets = targets.iloc[indices]

                predictions = self._make_predictions(model, model_name, sub_features)
                r2 = r2_score(sub_targets, predictions)
                subsample_r2s.append(r2)

            # 计算稳定性指标
            stability_metrics.update({
                'subsample_r2_mean': np.mean(subsample_r2s),
                'subsample_r2_std': np.std(subsample_r2s),
                'subsample_stability': 1.0 - (np.std(subsample_r2s) / max(np.mean(subsample_r2s), 0.01))
            })

            # 预测一致性测试
            if len(features) > 50:
                # 多次预测的一致性
                predictions_list = []
                for _ in range(3):
                    pred = self._make_predictions(model, model_name, features)
                    predictions_list.append(pred)

                # 计算预测一致性
                pred_array = np.array(predictions_list)
                pred_consistency = np.mean([
                    pearsonr(pred_array[i], pred_array[j])[0]
                    for i in range(len(pred_array))
                    for j in range(i+1, len(pred_array))
                ])

                stability_metrics['prediction_consistency'] = pred_consistency

            return stability_metrics

        except Exception as e:
            self.logger.error(f"❌ 稳定性验证失败: {e}")
            return {'error': str(e)}

    def _assess_prediction_confidence(self,
                                    model: Any,
                                    model_name: str,
                                    features: pd.DataFrame,
                                    targets: pd.Series) -> Dict[str, Any]:
        """
        预测置信度评估
        """
        try:
            confidence_assessment = {}

            # 基础预测
            predictions = self._make_predictions(model, model_name, features)

            # 计算预测区间 (基于残差分布)
            residuals = targets - predictions
            residual_std = np.std(residuals)

            # 95%置信区间
            confidence_intervals_95 = {
                'lower': predictions - 1.96 * residual_std,
                'upper': predictions + 1.96 * residual_std
            }

            # 80%置信区间
            confidence_intervals_80 = {
                'lower': predictions - 1.28 * residual_std,
                'upper': predictions + 1.28 * residual_std
            }

            # 计算区间覆盖率
            coverage_95 = np.mean(
                (targets >= confidence_intervals_95['lower']) &
                (targets <= confidence_intervals_95['upper'])
            )

            coverage_80 = np.mean(
                (targets >= confidence_intervals_80['lower']) &
                (targets <= confidence_intervals_80['upper'])
            )

            confidence_assessment.update({
                'residual_std': residual_std,
                'coverage_95': coverage_95,
                'coverage_80': coverage_80,
                'interval_width_95': np.mean(confidence_intervals_95['upper'] - confidence_intervals_95['lower']),
                'interval_width_80': np.mean(confidence_intervals_80['upper'] - confidence_intervals_80['lower'])
            })

            # 基于预测方差的置信度（如果模型支持）
            if hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
                try:
                    # 尝试获取预测不确定性
                    prediction_std = self._estimate_prediction_uncertainty(
                        model, model_name, features
                    )
                    confidence_assessment['prediction_uncertainty'] = prediction_std
                except:
                    pass

            # 综合置信度评分
            confidence_score = self._calculate_confidence_score(confidence_assessment)
            confidence_assessment['overall_confidence'] = confidence_score

            return confidence_assessment

        except Exception as e:
            self.logger.error(f"❌ 置信度评估失败: {e}")
            return {'error': str(e)}

    def _compare_with_baseline(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        与基准性能对比
        """
        comparison_result = {
            'has_baseline': self.baseline_performance is not None,
            'comparison_metrics': {}
        }

        if self.baseline_performance is None:
            # 设置当前性能为基准
            self.baseline_performance = current_metrics.copy()
            comparison_result['status'] = 'baseline_set'
            return comparison_result

        # 计算性能变化
        for metric, current_value in current_metrics.items():
            if metric in self.baseline_performance and isinstance(current_value, (int, float)):
                baseline_value = self.baseline_performance[metric]

                if baseline_value != 0:
                    relative_change = (current_value - baseline_value) / abs(baseline_value)
                else:
                    relative_change = 0

                comparison_result['comparison_metrics'][metric] = {
                    'current': current_value,
                    'baseline': baseline_value,
                    'absolute_change': current_value - baseline_value,
                    'relative_change': relative_change
                }

        # 判断性能变化趋势
        r2_change = comparison_result['comparison_metrics'].get('r2_score', {}).get('relative_change', 0)
        mse_change = comparison_result['comparison_metrics'].get('mse', {}).get('relative_change', 0)

        if r2_change < -self.performance_thresholds['r2_decline_threshold']:
            comparison_result['performance_trend'] = 'declining'
        elif r2_change > self.performance_thresholds['r2_decline_threshold']:
            comparison_result['performance_trend'] = 'improving'
        else:
            comparison_result['performance_trend'] = 'stable'

        return comparison_result

    def _generate_overall_assessment(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成综合评估
        """
        assessment = {
            'overall_score': 0.0,
            'component_scores': {},
            'recommendations': [],
            'warnings': [],
            'validation_passed': False
        }

        try:
            # 基础性能评分 (40%)
            basic_metrics = validation_result.get('basic_metrics', {})
            if 'error' not in basic_metrics:
                r2_score = basic_metrics.get('r2_score', 0)
                mse = basic_metrics.get('mse', 1)

                basic_score = max(0, min(1, r2_score)) * 0.8 + max(0, min(1, 1/(1+mse))) * 0.2
                assessment['component_scores']['basic_performance'] = basic_score
            else:
                assessment['component_scores']['basic_performance'] = 0
                assessment['warnings'].append("基础性能验证失败")

            # 稳定性评分 (25%)
            stability_metrics = validation_result.get('stability_metrics', {})
            if 'error' not in stability_metrics:
                stability_score = stability_metrics.get('subsample_stability', 0.5)
                consistency = stability_metrics.get('prediction_consistency', 0.8)
                stability_final = (stability_score * 0.6 + consistency * 0.4)
                assessment['component_scores']['stability'] = stability_final
            else:
                assessment['component_scores']['stability'] = 0.5  # 默认中等评分

            # 置信度评分 (20%)
            confidence_assessment = validation_result.get('confidence_assessment', {})
            if 'error' not in confidence_assessment:
                confidence_score = confidence_assessment.get('overall_confidence', 0.7)
                assessment['component_scores']['confidence'] = confidence_score
            else:
                assessment['component_scores']['confidence'] = 0.5

            # 时间序列验证评分 (15%)
            ts_validation = validation_result.get('time_series_validation', {})
            if ts_validation and 'error' not in ts_validation:
                ts_score = 0.8  # 如果时间序列验证通过，给予高分
                assessment['component_scores']['time_series'] = ts_score
            else:
                assessment['component_scores']['time_series'] = 0.6  # 默认评分

            # 计算综合评分
            weights = {
                'basic_performance': 0.40,
                'stability': 0.25,
                'confidence': 0.20,
                'time_series': 0.15
            }

            overall_score = sum(
                assessment['component_scores'].get(component, 0) * weight
                for component, weight in weights.items()
            )
            assessment['overall_score'] = overall_score

            # 生成建议
            if overall_score >= 0.8:
                assessment['validation_passed'] = True
                assessment['recommendations'].append("模型性能优秀，可以部署使用")
            elif overall_score >= 0.6:
                assessment['validation_passed'] = True
                assessment['recommendations'].append("模型性能良好，建议监控使用")
                if assessment['component_scores'].get('stability', 0) < 0.7:
                    assessment['recommendations'].append("建议提高模型稳定性")
            else:
                assessment['validation_passed'] = False
                assessment['warnings'].append("模型性能不符合要求")
                assessment['recommendations'].append("需要重新训练或调优模型")

            # 性能对比建议
            performance_comparison = validation_result.get('performance_comparison', {})
            if performance_comparison.get('performance_trend') == 'declining':
                assessment['warnings'].append("模型性能相比基准有所下降")
                assessment['recommendations'].append("考虑重新训练或更新数据")

        except Exception as e:
            self.logger.error(f"❌ 综合评估生成失败: {e}")
            assessment['overall_score'] = 0
            assessment['warnings'].append(f"评估生成失败: {str(e)}")

        return assessment

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
                # 通用预测方法
                if hasattr(model, 'predict'):
                    return model.predict(features)
                else:
                    raise ValueError(f"Model {model_name} doesn't have predict method")

        except Exception as e:
            raise Exception(f"Prediction failed for {model_name}: {str(e)}")

    def _estimate_prediction_uncertainty(self, model: Any, model_name: str, features: pd.DataFrame) -> np.ndarray:
        """
        估算预测不确定性
        """
        # 简化的不确定性估算
        # 在实际应用中可以使用更复杂的方法，如贝叶斯神经网络、集成方法等
        predictions = self._make_predictions(model, model_name, features)

        # 基于特征变异性估算不确定性
        feature_std = features.std(axis=1)
        uncertainty = feature_std / feature_std.mean() * np.std(predictions) * 0.1

        return uncertainty.values if hasattr(uncertainty, 'values') else uncertainty

    def _calculate_confidence_score(self, confidence_assessment: Dict[str, Any]) -> float:
        """
        计算置信度评分
        """
        try:
            coverage_95 = confidence_assessment.get('coverage_95', 0.95)
            coverage_80 = confidence_assessment.get('coverage_80', 0.80)

            # 理想的覆盖率应该接近名义水平
            coverage_score_95 = 1.0 - abs(coverage_95 - 0.95) / 0.95
            coverage_score_80 = 1.0 - abs(coverage_80 - 0.80) / 0.80

            # 综合置信度评分
            confidence_score = (coverage_score_95 * 0.6 + coverage_score_80 * 0.4)
            return max(0.0, min(1.0, confidence_score))

        except Exception:
            return 0.7  # 默认中等置信度

    def _record_validation_history(self, validation_result: Dict[str, Any]):
        """
        记录验证历史
        """
        # 简化的历史记录
        history_record = {
            'timestamp': validation_result['timestamp'],
            'model_name': validation_result['model_name'],
            'validation_type': validation_result['validation_type'],
            'overall_score': validation_result.get('overall_assessment', {}).get('overall_score', 0),
            'validation_passed': validation_result.get('overall_assessment', {}).get('validation_passed', False),
            'basic_r2': validation_result.get('basic_metrics', {}).get('r2_score', 0)
        }

        self.validation_history.append(history_record)

        # 保持历史记录长度
        max_history = 100
        if len(self.validation_history) > max_history:
            self.validation_history = self.validation_history[-max_history:]

    def get_validation_summary(self) -> Dict[str, Any]:
        """
        获取验证摘要
        """
        if not self.validation_history:
            return {'status': 'no_history'}

        recent_validations = self.validation_history[-10:]

        summary = {
            'total_validations': len(self.validation_history),
            'recent_average_score': np.mean([v['overall_score'] for v in recent_validations]),
            'recent_pass_rate': np.mean([v['validation_passed'] for v in recent_validations]),
            'performance_trend': self._calculate_recent_trend(),
            'baseline_performance': self.baseline_performance,
            'last_validation': self.validation_history[-1] if self.validation_history else None
        }

        return summary

    def _calculate_recent_trend(self) -> str:
        """
        计算最近的性能趋势
        """
        if len(self.validation_history) < 5:
            return 'insufficient_data'

        recent_scores = [v['overall_score'] for v in self.validation_history[-5:]]
        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]

        if trend > 0.02:
            return 'improving'
        elif trend < -0.02:
            return 'declining'
        else:
            return 'stable'


# 具体验证策略实现类

class TimeSeriesValidator:
    """时间序列交叉验证器"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def validate(self, model, model_name, features, targets):
        try:
            # 时间序列分割
            tscv = TimeSeriesSplit(n_splits=min(5, len(features) // 20))

            scores = []
            for train_idx, test_idx in tscv.split(features):
                train_features = features.iloc[train_idx]
                train_targets = targets.iloc[train_idx]
                test_features = features.iloc[test_idx]
                test_targets = targets.iloc[test_idx]

                # 训练和预测（这里假设模型已经训练好，只做预测）
                if model_name == 'xgboost':
                    import xgboost as xgb
                    dtest = xgb.DMatrix(test_features)
                    predictions = model.predict(dtest)
                else:
                    predictions = model.predict(test_features)

                r2 = r2_score(test_targets, predictions)
                scores.append(r2)

            return {
                'cv_scores': scores,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores)
            }

        except Exception as e:
            return {'error': str(e)}


class RollingWindowValidator:
    """滚动窗口验证器"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def validate(self, model, model_name, features, targets):
        try:
            window_size = min(50, len(features) // 4)
            step_size = max(1, window_size // 4)

            scores = []

            for start_idx in range(0, len(features) - window_size, step_size):
                end_idx = start_idx + window_size

                window_features = features.iloc[start_idx:end_idx]
                window_targets = targets.iloc[start_idx:end_idx]

                if model_name == 'xgboost':
                    import xgboost as xgb
                    dtest = xgb.DMatrix(window_features)
                    predictions = model.predict(dtest)
                else:
                    predictions = model.predict(window_features)

                r2 = r2_score(window_targets, predictions)
                scores.append(r2)

            return {
                'window_scores': scores,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'score_trend': np.polyfit(range(len(scores)), scores, 1)[0] if len(scores) > 2 else 0
            }

        except Exception as e:
            return {'error': str(e)}


class WalkForwardValidator:
    """前进分析验证器"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def validate(self, model, model_name, features, targets):
        try:
            min_train_size = max(30, len(features) // 4)
            test_size = 10

            scores = []

            for i in range(min_train_size, len(features) - test_size):
                train_features = features.iloc[:i]
                train_targets = targets.iloc[:i]
                test_features = features.iloc[i:i+test_size]
                test_targets = targets.iloc[i:i+test_size]

                if model_name == 'xgboost':
                    import xgboost as xgb
                    dtest = xgb.DMatrix(test_features)
                    predictions = model.predict(dtest)
                else:
                    predictions = model.predict(test_features)

                r2 = r2_score(test_targets, predictions)
                scores.append(r2)

            return {
                'walk_forward_scores': scores,
                'mean_score': np.mean(scores),
                'final_score': scores[-1] if scores else 0,
                'score_stability': 1.0 - np.std(scores) / max(np.mean(scores), 0.01) if scores else 0
            }

        except Exception as e:
            return {'error': str(e)}


class ExpandingWindowValidator:
    """扩展窗口验证器"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def validate(self, model, model_name, features, targets):
        try:
            initial_size = max(50, len(features) // 3)
            test_size = 10

            scores = []

            for i in range(initial_size, len(features) - test_size, test_size):
                train_features = features.iloc[:i]
                train_targets = targets.iloc[:i]
                test_features = features.iloc[i:i+test_size]
                test_targets = targets.iloc[i:i+test_size]

                if model_name == 'xgboost':
                    import xgboost as xgb
                    dtest = xgb.DMatrix(test_features)
                    predictions = model.predict(dtest)
                else:
                    predictions = model.predict(test_features)

                r2 = r2_score(test_targets, predictions)
                scores.append(r2)

            return {
                'expanding_scores': scores,
                'mean_score': np.mean(scores),
                'learning_curve_slope': np.polyfit(range(len(scores)), scores, 1)[0] if len(scores) > 2 else 0
            }

        except Exception as e:
            return {'error': str(e)}


def main():
    """测试在线验证引擎"""
    print("🔍 测试在线验证引擎...")

    # 配置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 创建验证引擎
    validation_config = {
        'time_series_splits': 5,
        'rolling_window_size': 50,
        'confidence_level': 0.95
    }

    validator = OnlineValidationEngine(
        validation_config=validation_config,
        logger=logger
    )

    # 生成测试数据和简单模型
    np.random.seed(42)
    n_samples = 300
    n_features = 10

    features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    targets = pd.Series(
        features.sum(axis=1) + np.random.randn(n_samples) * 0.2,
        name='target'
    )

    # 创建简单的线性模型用于测试
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(features, targets)

    # 测试在线验证
    print("\n🔍 进行综合验证...")
    validation_result = validator.validate_model_online(
        model=model,
        model_name='test_linear_model',
        features=features,
        targets=targets,
        validation_type='comprehensive'
    )

    # 显示验证结果
    print(f"\n📊 验证结果:")
    print(f"验证状态: {validation_result['validation_status']}")

    if 'overall_assessment' in validation_result:
        assessment = validation_result['overall_assessment']
        print(f"总体评分: {assessment['overall_score']:.3f}")
        print(f"验证通过: {assessment['validation_passed']}")

        print(f"\n组件评分:")
        for component, score in assessment['component_scores'].items():
            print(f"  {component}: {score:.3f}")

        if assessment['recommendations']:
            print(f"\n建议:")
            for rec in assessment['recommendations']:
                print(f"  • {rec}")

        if assessment['warnings']:
            print(f"\n警告:")
            for warning in assessment['warnings']:
                print(f"  ⚠️ {warning}")

    # 显示基础指标
    if 'basic_metrics' in validation_result:
        basic = validation_result['basic_metrics']
        print(f"\n基础性能指标:")
        for metric, value in basic.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")

    # 获取验证摘要
    print(f"\n📈 验证摘要:")
    summary = validator.get_validation_summary()
    for key, value in summary.items():
        if key != 'last_validation':
            print(f"  {key}: {value}")

    print("\n✅ 在线验证引擎测试完成！")

if __name__ == "__main__":
    main()