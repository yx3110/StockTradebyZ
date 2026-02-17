#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.8性能追踪器
全面监控和记录模型性能，提供详细的性能分析和可视化

Phase 3: 增量学习机制 - PerformanceTracker组件
- 实时性能监控
- 多维度性能指标
- 性能趋势分析
- 性能基准对比
- 性能报告生成
- 性能预警系统

Created: 2025-09-16
Author: Claude Code
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import sqlite3

# 添加项目路径
sys.path.append('/Users/yangxu/StockTradebyZ')

class PerformanceTracker:
    """
    性能追踪器

    核心功能：
    1. 实时性能指标计算和监控
    2. 多维度性能分析（准确性、稳定性、效率）
    3. 性能趋势分析和预测
    4. 基准性能对比
    5. 性能退化检测和预警
    6. 详细性能报告生成
    7. 性能数据持久化存储
    """

    def __init__(self,
                 tracker_config: Dict,
                 logger: logging.Logger,
                 db_path: str = None,
                 performance_window: int = 100):

        self.tracker_config = tracker_config
        self.logger = logger
        self.performance_window = performance_window

        # 性能历史存储
        self.performance_history = {}
        self.performance_baselines = {}

        # 性能指标配置
        self.metrics_config = {
            'core_metrics': ['r2_score', 'mse', 'mae', 'rmse'],
            'extended_metrics': ['mape', 'explained_variance', 'pearson_corr', 'spearman_corr'],
            'distribution_metrics': ['residual_mean', 'residual_std', 'residual_skewness', 'residual_kurtosis'],
            'stability_metrics': ['prediction_consistency', 'temporal_stability'],
            'efficiency_metrics': ['prediction_time', 'memory_usage']
        }

        # 性能阈值和预警配置
        self.alert_thresholds = {
            'r2_decline_warning': 0.05,      # R²下降5%预警
            'r2_decline_critical': 0.15,     # R²下降15%严重警告
            'mse_increase_warning': 0.2,     # MSE增加20%预警
            'mse_increase_critical': 0.5,    # MSE增加50%严重警告
            'stability_warning': 0.8,        # 稳定性低于0.8预警
            'efficiency_warning': 2.0        # 预测时间超过基准2倍预警
        }

        # 数据库连接（用于持久化）
        self.db_path = db_path
        if self.db_path:
            self._init_performance_database()

        # 性能分析器
        self.analyzers = {
            'trend': TrendAnalyzer(logger),
            'benchmark': BenchmarkAnalyzer(logger),
            'stability': StabilityAnalyzer(logger),
            'efficiency': EfficiencyAnalyzer(logger)
        }

        self.logger.info("📊 性能追踪器初始化完成")

    def track_performance(self,
                         model_name: str,
                         predictions: np.ndarray,
                         actual_values: pd.Series,
                         features: pd.DataFrame = None,
                         metadata: Dict = None) -> Dict[str, Any]:
        """
        追踪模型性能

        Args:
            model_name: 模型名称
            predictions: 模型预测值
            actual_values: 真实值
            features: 输入特征（可选）
            metadata: 额外元数据（如预测时间、模型版本等）

        Returns:
            性能追踪结果
        """
        self.logger.info(f"📊 追踪性能 - 模型: {model_name}, 样本数: {len(predictions)}")

        tracking_result = {
            'model_name': model_name,
            'timestamp': datetime.now(),
            'sample_size': len(predictions),
            'tracking_status': 'started'
        }

        try:
            # 1. 计算核心性能指标
            core_metrics = self._calculate_core_metrics(actual_values, predictions)
            tracking_result['core_metrics'] = core_metrics

            # 2. 计算扩展性能指标
            extended_metrics = self._calculate_extended_metrics(actual_values, predictions)
            tracking_result['extended_metrics'] = extended_metrics

            # 3. 计算分布统计指标
            distribution_metrics = self._calculate_distribution_metrics(actual_values, predictions)
            tracking_result['distribution_metrics'] = distribution_metrics

            # 4. 稳定性分析
            if len(predictions) > 20:
                stability_metrics = self._calculate_stability_metrics(
                    actual_values, predictions, features
                )
                tracking_result['stability_metrics'] = stability_metrics

            # 5. 效率分析
            if metadata:
                efficiency_metrics = self._calculate_efficiency_metrics(metadata)
                tracking_result['efficiency_metrics'] = efficiency_metrics

            # 6. 趋势分析
            trend_analysis = self._analyze_performance_trends(model_name, core_metrics)
            tracking_result['trend_analysis'] = trend_analysis

            # 7. 基准对比
            benchmark_comparison = self._compare_with_benchmark(model_name, core_metrics)
            tracking_result['benchmark_comparison'] = benchmark_comparison

            # 8. 性能评级
            performance_rating = self._calculate_performance_rating(tracking_result)
            tracking_result['performance_rating'] = performance_rating

            # 9. 预警检查
            alerts = self._check_performance_alerts(model_name, tracking_result)
            tracking_result['alerts'] = alerts

            # 10. 更新性能历史
            self._update_performance_history(model_name, tracking_result)

            # 11. 持久化存储
            if self.db_path:
                self._save_performance_record(tracking_result)

            tracking_result['tracking_status'] = 'completed'

            # 记录性能等级
            rating = performance_rating.get('overall_rating', 'unknown')
            self.logger.info(f"✅ {model_name} 性能追踪完成 - 评级: {rating}")

            return tracking_result

        except Exception as e:
            self.logger.error(f"❌ {model_name} 性能追踪失败: {e}")
            tracking_result['tracking_status'] = 'failed'
            tracking_result['error'] = str(e)
            return tracking_result

    def _calculate_core_metrics(self, actual_values: pd.Series, predictions: np.ndarray) -> Dict[str, float]:
        """
        计算核心性能指标
        """
        try:
            return {
                'r2_score': r2_score(actual_values, predictions),
                'mse': mean_squared_error(actual_values, predictions),
                'rmse': np.sqrt(mean_squared_error(actual_values, predictions)),
                'mae': mean_absolute_error(actual_values, predictions)
            }
        except Exception as e:
            self.logger.error(f"❌ 核心指标计算失败: {e}")
            return {'error': str(e)}

    def _calculate_extended_metrics(self, actual_values: pd.Series, predictions: np.ndarray) -> Dict[str, float]:
        """
        计算扩展性能指标
        """
        try:
            metrics = {}

            # MAPE (注意零值处理)
            if not np.any(actual_values == 0):
                metrics['mape'] = mean_absolute_percentage_error(actual_values, predictions)
            else:
                metrics['mape'] = np.inf

            # 解释方差分数
            metrics['explained_variance'] = explained_variance_score(actual_values, predictions)

            # 相关性指标
            if len(actual_values) > 3:
                pearson_corr, pearson_p = pearsonr(actual_values, predictions)
                spearman_corr, spearman_p = spearmanr(actual_values, predictions)

                metrics.update({
                    'pearson_correlation': pearson_corr,
                    'pearson_p_value': pearson_p,
                    'spearman_correlation': spearman_corr,
                    'spearman_p_value': spearman_p
                })

            # 相对误差指标
            relative_errors = np.abs((actual_values - predictions) / (np.abs(actual_values) + 1e-8))
            metrics.update({
                'mean_relative_error': np.mean(relative_errors),
                'median_relative_error': np.median(relative_errors),
                'max_relative_error': np.max(relative_errors)
            })

            return metrics

        except Exception as e:
            self.logger.error(f"❌ 扩展指标计算失败: {e}")
            return {'error': str(e)}

    def _calculate_distribution_metrics(self, actual_values: pd.Series, predictions: np.ndarray) -> Dict[str, float]:
        """
        计算分布统计指标
        """
        try:
            residuals = actual_values - predictions

            return {
                'residual_mean': np.mean(residuals),
                'residual_std': np.std(residuals),
                'residual_skewness': stats.skew(residuals),
                'residual_kurtosis': stats.kurtosis(residuals),
                'residual_median': np.median(residuals),
                'residual_iqr': np.percentile(residuals, 75) - np.percentile(residuals, 25),
                'residual_range': np.max(residuals) - np.min(residuals)
            }

        except Exception as e:
            self.logger.error(f"❌ 分布指标计算失败: {e}")
            return {'error': str(e)}

    def _calculate_stability_metrics(self, actual_values: pd.Series, predictions: np.ndarray, features: pd.DataFrame = None) -> Dict[str, float]:
        """
        计算稳定性指标
        """
        try:
            metrics = {}

            # 预测一致性（子集预测的一致性）
            if len(predictions) > 50:
                # 将数据分成多个子集，计算预测的一致性
                n_subsets = 5
                subset_size = len(predictions) // n_subsets
                subset_correlations = []

                for i in range(n_subsets - 1):
                    start_idx = i * subset_size
                    end_idx = (i + 1) * subset_size

                    subset1_pred = predictions[start_idx:end_idx]
                    subset1_actual = actual_values.iloc[start_idx:end_idx]

                    start_idx_next = (i + 1) * subset_size
                    end_idx_next = (i + 2) * subset_size

                    subset2_pred = predictions[start_idx_next:end_idx_next]
                    subset2_actual = actual_values.iloc[start_idx_next:end_idx_next]

                    # 计算预测质量的相关性
                    subset1_errors = np.abs(subset1_actual - subset1_pred)
                    subset2_errors = np.abs(subset2_actual - subset2_pred)

                    if len(subset1_errors) > 3 and len(subset2_errors) > 3:
                        corr, _ = pearsonr(subset1_errors, subset2_errors[:len(subset1_errors)])
                        if not np.isnan(corr):
                            subset_correlations.append(abs(corr))

                if subset_correlations:
                    metrics['prediction_consistency'] = np.mean(subset_correlations)

            # 时间稳定性（如果有时间序列特性）
            if len(predictions) > 20:
                # 计算滑动窗口内的性能稳定性
                window_size = min(20, len(predictions) // 4)
                window_r2s = []

                for i in range(len(predictions) - window_size + 1):
                    window_actual = actual_values.iloc[i:i+window_size]
                    window_pred = predictions[i:i+window_size]

                    window_r2 = r2_score(window_actual, window_pred)
                    window_r2s.append(window_r2)

                if len(window_r2s) > 1:
                    metrics['temporal_stability'] = 1.0 - (np.std(window_r2s) / max(np.mean(window_r2s), 0.01))

            # 方差稳定性
            if len(predictions) > 30:
                residuals = actual_values - predictions
                first_half_residuals = residuals[:len(residuals)//2]
                second_half_residuals = residuals[len(residuals)//2:]

                first_half_var = np.var(first_half_residuals)
                second_half_var = np.var(second_half_residuals)

                # 方差比率（接近1表示稳定）
                var_ratio = min(first_half_var, second_half_var) / max(first_half_var, second_half_var)
                metrics['variance_stability'] = var_ratio

            return metrics

        except Exception as e:
            self.logger.error(f"❌ 稳定性指标计算失败: {e}")
            return {'error': str(e)}

    def _calculate_efficiency_metrics(self, metadata: Dict) -> Dict[str, float]:
        """
        计算效率指标
        """
        try:
            metrics = {}

            # 预测时间
            if 'prediction_time' in metadata:
                metrics['prediction_time'] = metadata['prediction_time']
                metrics['prediction_time_per_sample'] = metadata['prediction_time'] / metadata.get('sample_size', 1)

            # 内存使用
            if 'memory_usage' in metadata:
                metrics['memory_usage'] = metadata['memory_usage']

            # 训练时间（如果有）
            if 'training_time' in metadata:
                metrics['training_time'] = metadata['training_time']

            # CPU使用率（如果有）
            if 'cpu_usage' in metadata:
                metrics['cpu_usage'] = metadata['cpu_usage']

            return metrics

        except Exception as e:
            self.logger.error(f"❌ 效率指标计算失败: {e}")
            return {'error': str(e)}

    def _analyze_performance_trends(self, model_name: str, current_metrics: Dict) -> Dict[str, Any]:
        """
        分析性能趋势
        """
        if model_name not in self.performance_history:
            return {'trend_status': 'insufficient_history'}

        try:
            history = self.performance_history[model_name]

            if len(history) < 3:
                return {'trend_status': 'insufficient_data'}

            # 分析R²趋势
            recent_r2_scores = [h['core_metrics']['r2_score'] for h in history[-10:] if 'r2_score' in h.get('core_metrics', {})]

            if len(recent_r2_scores) >= 3:
                # 线性回归拟合趋势
                x = np.arange(len(recent_r2_scores))
                r2_trend_slope, r2_trend_intercept, r2_trend_r, _, _ = stats.linregress(x, recent_r2_scores)

                # MSE趋势
                recent_mse_scores = [h['core_metrics']['mse'] for h in history[-10:] if 'mse' in h.get('core_metrics', {})]
                mse_trend_slope, _, _, _, _ = stats.linregress(x, recent_mse_scores) if len(recent_mse_scores) == len(recent_r2_scores) else (0, 0, 0, 0, 0)

                # 趋势判断
                if r2_trend_slope > 0.01:
                    trend_direction = 'improving'
                elif r2_trend_slope < -0.01:
                    trend_direction = 'declining'
                else:
                    trend_direction = 'stable'

                return {
                    'trend_status': 'analyzed',
                    'trend_direction': trend_direction,
                    'r2_trend_slope': r2_trend_slope,
                    'mse_trend_slope': mse_trend_slope,
                    'trend_confidence': abs(r2_trend_r),
                    'data_points': len(recent_r2_scores)
                }

        except Exception as e:
            return {'trend_status': 'analysis_failed', 'error': str(e)}

        return {'trend_status': 'insufficient_data'}

    def _compare_with_benchmark(self, model_name: str, current_metrics: Dict) -> Dict[str, Any]:
        """
        与基准性能对比
        """
        if model_name not in self.performance_baselines:
            # 设置当前性能为基准
            self.performance_baselines[model_name] = current_metrics.copy()
            return {'benchmark_status': 'baseline_set', 'baseline_metrics': current_metrics}

        try:
            baseline = self.performance_baselines[model_name]
            comparison = {}

            for metric, current_value in current_metrics.items():
                if metric in baseline and isinstance(current_value, (int, float)) and not np.isnan(current_value):
                    baseline_value = baseline[metric]

                    if baseline_value != 0:
                        relative_change = (current_value - baseline_value) / abs(baseline_value)
                        absolute_change = current_value - baseline_value
                    else:
                        relative_change = 0
                        absolute_change = current_value

                    comparison[metric] = {
                        'current': current_value,
                        'baseline': baseline_value,
                        'absolute_change': absolute_change,
                        'relative_change': relative_change,
                        'improvement': self._is_improvement(metric, relative_change)
                    }

            # 综合性能变化评估
            improvement_count = sum(1 for comp in comparison.values() if comp.get('improvement', False))
            total_metrics = len(comparison)

            overall_improvement = improvement_count / total_metrics if total_metrics > 0 else 0

            return {
                'benchmark_status': 'compared',
                'comparison_details': comparison,
                'overall_improvement_rate': overall_improvement,
                'performance_change': 'improved' if overall_improvement > 0.6 else 'declined' if overall_improvement < 0.4 else 'stable'
            }

        except Exception as e:
            return {'benchmark_status': 'comparison_failed', 'error': str(e)}

    def _is_improvement(self, metric_name: str, relative_change: float) -> bool:
        """
        判断指标变化是否为改善
        """
        # 对于这些指标，值越小越好
        lower_is_better = ['mse', 'rmse', 'mae', 'mape', 'mean_relative_error', 'residual_std']

        if metric_name in lower_is_better:
            return relative_change < 0  # 负变化是改善
        else:
            return relative_change > 0  # 正变化是改善

    def _calculate_performance_rating(self, tracking_result: Dict) -> Dict[str, Any]:
        """
        计算性能评级
        """
        try:
            rating = {
                'overall_rating': 'unknown',
                'component_ratings': {},
                'rating_score': 0.0
            }

            # 核心性能评级
            core_metrics = tracking_result.get('core_metrics', {})
            if 'r2_score' in core_metrics:
                r2_score_val = core_metrics['r2_score']

                if r2_score_val >= 0.9:
                    core_rating = 'excellent'
                    core_score = 1.0
                elif r2_score_val >= 0.8:
                    core_rating = 'good'
                    core_score = 0.8
                elif r2_score_val >= 0.6:
                    core_rating = 'fair'
                    core_score = 0.6
                elif r2_score_val >= 0.4:
                    core_rating = 'poor'
                    core_score = 0.4
                else:
                    core_rating = 'very_poor'
                    core_score = 0.2

                rating['component_ratings']['accuracy'] = core_rating

            # 稳定性评级
            stability_metrics = tracking_result.get('stability_metrics', {})
            if stability_metrics:
                stability_scores = []
                for key, value in stability_metrics.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        stability_scores.append(value)

                if stability_scores:
                    avg_stability = np.mean(stability_scores)
                    if avg_stability >= 0.9:
                        stability_rating = 'excellent'
                        stability_score = 1.0
                    elif avg_stability >= 0.8:
                        stability_rating = 'good'
                        stability_score = 0.8
                    elif avg_stability >= 0.6:
                        stability_rating = 'fair'
                        stability_score = 0.6
                    else:
                        stability_rating = 'poor'
                        stability_score = 0.4

                    rating['component_ratings']['stability'] = stability_rating

            # 趋势评级
            trend_analysis = tracking_result.get('trend_analysis', {})
            if trend_analysis.get('trend_status') == 'analyzed':
                trend_direction = trend_analysis.get('trend_direction', 'stable')
                if trend_direction == 'improving':
                    trend_rating = 'positive'
                    trend_score = 1.0
                elif trend_direction == 'stable':
                    trend_rating = 'neutral'
                    trend_score = 0.7
                else:
                    trend_rating = 'negative'
                    trend_score = 0.3

                rating['component_ratings']['trend'] = trend_rating

            # 综合评级
            component_scores = []
            if 'accuracy' in rating['component_ratings']:
                component_scores.append(core_score * 0.5)  # 准确性权重50%

            if 'stability' in rating['component_ratings']:
                component_scores.append(stability_score * 0.3)  # 稳定性权重30%

            if 'trend' in rating['component_ratings']:
                component_scores.append(trend_score * 0.2)  # 趋势权重20%

            if component_scores:
                overall_score = sum(component_scores)
                rating['rating_score'] = overall_score

                if overall_score >= 0.9:
                    rating['overall_rating'] = 'A+'
                elif overall_score >= 0.8:
                    rating['overall_rating'] = 'A'
                elif overall_score >= 0.7:
                    rating['overall_rating'] = 'B+'
                elif overall_score >= 0.6:
                    rating['overall_rating'] = 'B'
                elif overall_score >= 0.5:
                    rating['overall_rating'] = 'C+'
                elif overall_score >= 0.4:
                    rating['overall_rating'] = 'C'
                else:
                    rating['overall_rating'] = 'D'

            return rating

        except Exception as e:
            return {'overall_rating': 'error', 'error': str(e)}

    def _check_performance_alerts(self, model_name: str, tracking_result: Dict) -> List[Dict[str, Any]]:
        """
        检查性能预警
        """
        alerts = []

        try:
            # 基准对比预警
            benchmark_comparison = tracking_result.get('benchmark_comparison', {})
            if benchmark_comparison.get('benchmark_status') == 'compared':
                comparison_details = benchmark_comparison.get('comparison_details', {})

                # R²下降预警
                if 'r2_score' in comparison_details:
                    r2_change = comparison_details['r2_score']['relative_change']
                    if r2_change < -self.alert_thresholds['r2_decline_critical']:
                        alerts.append({
                            'type': 'critical',
                            'metric': 'r2_score',
                            'message': f"R²严重下降 {r2_change:.1%}，需要立即检查模型",
                            'severity': 'high'
                        })
                    elif r2_change < -self.alert_thresholds['r2_decline_warning']:
                        alerts.append({
                            'type': 'warning',
                            'metric': 'r2_score',
                            'message': f"R²下降 {r2_change:.1%}，建议关注模型性能",
                            'severity': 'medium'
                        })

                # MSE增加预警
                if 'mse' in comparison_details:
                    mse_change = comparison_details['mse']['relative_change']
                    if mse_change > self.alert_thresholds['mse_increase_critical']:
                        alerts.append({
                            'type': 'critical',
                            'metric': 'mse',
                            'message': f"均方误差显著增加 {mse_change:.1%}，模型精度严重下降",
                            'severity': 'high'
                        })
                    elif mse_change > self.alert_thresholds['mse_increase_warning']:
                        alerts.append({
                            'type': 'warning',
                            'metric': 'mse',
                            'message': f"均方误差增加 {mse_change:.1%}，建议检查数据质量",
                            'severity': 'medium'
                        })

            # 稳定性预警
            stability_metrics = tracking_result.get('stability_metrics', {})
            for metric_name, value in stability_metrics.items():
                if isinstance(value, (int, float)) and value < self.alert_thresholds['stability_warning']:
                    alerts.append({
                        'type': 'warning',
                        'metric': metric_name,
                        'message': f"模型稳定性指标 {metric_name} 过低: {value:.3f}",
                        'severity': 'medium'
                    })

            # 趋势预警
            trend_analysis = tracking_result.get('trend_analysis', {})
            if trend_analysis.get('trend_direction') == 'declining':
                trend_confidence = trend_analysis.get('trend_confidence', 0)
                if trend_confidence > 0.7:
                    alerts.append({
                        'type': 'warning',
                        'metric': 'performance_trend',
                        'message': f"检测到性能下降趋势，置信度: {trend_confidence:.2f}",
                        'severity': 'medium'
                    })

        except Exception as e:
            alerts.append({
                'type': 'error',
                'metric': 'alert_system',
                'message': f"预警系统检查失败: {str(e)}",
                'severity': 'low'
            })

        return alerts

    def _update_performance_history(self, model_name: str, tracking_result: Dict):
        """
        更新性能历史记录
        """
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []

        # 简化的历史记录
        history_record = {
            'timestamp': tracking_result['timestamp'],
            'core_metrics': tracking_result.get('core_metrics', {}),
            'performance_rating': tracking_result.get('performance_rating', {}),
            'alerts_count': len(tracking_result.get('alerts', []))
        }

        self.performance_history[model_name].append(history_record)

        # 保持历史记录长度
        if len(self.performance_history[model_name]) > self.performance_window:
            self.performance_history[model_name] = self.performance_history[model_name][-self.performance_window:]

    def _init_performance_database(self):
        """
        初始化性能数据库
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS performance_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        r2_score REAL,
                        mse REAL,
                        mae REAL,
                        performance_rating TEXT,
                        alerts_count INTEGER,
                        tracking_data TEXT
                    )
                ''')
                conn.commit()

            self.logger.info("📊 性能数据库初始化完成")

        except Exception as e:
            self.logger.error(f"❌ 性能数据库初始化失败: {e}")

    def _save_performance_record(self, tracking_result: Dict):
        """
        保存性能记录到数据库
        """
        if not self.db_path:
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                core_metrics = tracking_result.get('core_metrics', {})
                performance_rating = tracking_result.get('performance_rating', {})

                conn.execute('''
                    INSERT INTO performance_records
                    (model_name, timestamp, r2_score, mse, mae, performance_rating, alerts_count, tracking_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    tracking_result['model_name'],
                    tracking_result['timestamp'],
                    core_metrics.get('r2_score'),
                    core_metrics.get('mse'),
                    core_metrics.get('mae'),
                    performance_rating.get('overall_rating'),
                    len(tracking_result.get('alerts', [])),
                    json.dumps(tracking_result, default=str)
                ))
                conn.commit()

        except Exception as e:
            self.logger.error(f"❌ 性能记录保存失败: {e}")

    def get_performance_summary(self, model_name: str = None) -> Dict[str, Any]:
        """
        获取性能摘要
        """
        summary = {}

        try:
            if model_name:
                # 单个模型摘要
                if model_name in self.performance_history:
                    history = self.performance_history[model_name]
                    recent_records = history[-10:]

                    # 计算平均性能
                    avg_r2 = np.mean([r.get('core_metrics', {}).get('r2_score', 0) for r in recent_records])
                    avg_mse = np.mean([r.get('core_metrics', {}).get('mse', 0) for r in recent_records])

                    # 最新评级
                    latest_rating = recent_records[-1].get('performance_rating', {}).get('overall_rating', 'unknown')

                    # 预警统计
                    total_alerts = sum(r.get('alerts_count', 0) for r in recent_records)

                    summary[model_name] = {
                        'total_records': len(history),
                        'recent_avg_r2': avg_r2,
                        'recent_avg_mse': avg_mse,
                        'latest_rating': latest_rating,
                        'recent_alerts': total_alerts,
                        'last_tracked': history[-1]['timestamp'] if history else None
                    }
                else:
                    summary[model_name] = {'status': 'no_history'}
            else:
                # 所有模型摘要
                for name in self.performance_history:
                    model_summary = self.get_performance_summary(name)
                    summary.update(model_summary)

        except Exception as e:
            summary['error'] = str(e)

        return summary

    def generate_performance_report(self, model_name: str, period_days: int = 30) -> str:
        """
        生成性能报告
        """
        try:
            if model_name not in self.performance_history:
                return f"# 性能报告\n\n模型 {model_name} 无历史记录"

            history = self.performance_history[model_name]
            cutoff_date = datetime.now() - timedelta(days=period_days)
            recent_history = [h for h in history if h['timestamp'] > cutoff_date]

            if not recent_history:
                return f"# 性能报告\n\n模型 {model_name} 在过去 {period_days} 天内无记录"

            # 生成报告内容
            report_lines = []
            report_lines.append(f"# 性能报告 - {model_name}")
            report_lines.append(f"**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"**分析周期**: 过去 {period_days} 天")
            report_lines.append(f"**记录数量**: {len(recent_history)} 条")
            report_lines.append("")

            # 性能概览
            r2_scores = [h.get('core_metrics', {}).get('r2_score') for h in recent_history]
            r2_scores = [s for s in r2_scores if s is not None]

            if r2_scores:
                report_lines.append("## 性能概览")
                report_lines.append(f"- **R²分数**: 平均 {np.mean(r2_scores):.4f}, 最高 {max(r2_scores):.4f}, 最低 {min(r2_scores):.4f}")
                report_lines.append(f"- **性能稳定性**: 标准差 {np.std(r2_scores):.4f}")

            # 最新评级
            latest_rating = recent_history[-1].get('performance_rating', {})
            if latest_rating:
                report_lines.append(f"- **最新评级**: {latest_rating.get('overall_rating', 'unknown')}")
                if 'rating_score' in latest_rating:
                    report_lines.append(f"- **评级分数**: {latest_rating['rating_score']:.3f}")

            report_lines.append("")

            # 预警统计
            total_alerts = sum(h.get('alerts_count', 0) for h in recent_history)
            report_lines.append("## 预警统计")
            report_lines.append(f"- **总预警数**: {total_alerts}")
            report_lines.append(f"- **平均每次追踪预警数**: {total_alerts / len(recent_history):.1f}")
            report_lines.append("")

            # 性能趋势
            if len(r2_scores) >= 5:
                x = np.arange(len(r2_scores))
                slope, _, r_value, _, _ = stats.linregress(x, r2_scores)

                report_lines.append("## 性能趋势分析")
                if slope > 0.001:
                    trend = "📈 上升"
                elif slope < -0.001:
                    trend = "📉 下降"
                else:
                    trend = "➡️ 稳定"

                report_lines.append(f"- **趋势方向**: {trend}")
                report_lines.append(f"- **趋势强度**: {slope:.6f}")
                report_lines.append(f"- **趋势置信度**: {abs(r_value):.3f}")
                report_lines.append("")

            # 建议
            report_lines.append("## 建议")
            if np.mean(r2_scores) > 0.8:
                report_lines.append("✅ 模型性能表现良好，继续保持现有配置")
            elif np.mean(r2_scores) > 0.6:
                report_lines.append("⚠️ 模型性能中等，建议监控并考虑优化")
            else:
                report_lines.append("🚨 模型性能较差，建议重新训练或调整参数")

            if total_alerts > len(recent_history) * 0.5:
                report_lines.append("⚠️ 预警频繁触发，建议检查模型稳定性")

            return "\n".join(report_lines)

        except Exception as e:
            return f"# 性能报告\n\n报告生成失败: {str(e)}"


# 辅助分析器类（简化实现）

class TrendAnalyzer:
    def __init__(self, logger):
        self.logger = logger

class BenchmarkAnalyzer:
    def __init__(self, logger):
        self.logger = logger

class StabilityAnalyzer:
    def __init__(self, logger):
        self.logger = logger

class EfficiencyAnalyzer:
    def __init__(self, logger):
        self.logger = logger


def main():
    """测试性能追踪器"""
    print("📊 测试性能追踪器...")

    # 配置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 创建性能追踪器
    tracker_config = {
        'metrics_enabled': ['core', 'extended', 'stability'],
        'alert_enabled': True
    }

    tracker = PerformanceTracker(
        tracker_config=tracker_config,
        logger=logger
    )

    # 生成测试数据
    np.random.seed(42)
    n_samples = 200

    # 模拟真实值和预测值
    actual_values = pd.Series(np.random.randn(n_samples), name='actual')
    predictions_good = actual_values + np.random.randn(n_samples) * 0.1  # 好的预测
    predictions_poor = actual_values + np.random.randn(n_samples) * 0.5  # 差的预测

    # 测试1：好的预测性能追踪
    print("\n📊 测试1：好的模型性能追踪")
    result1 = tracker.track_performance(
        model_name='good_model',
        predictions=predictions_good,
        actual_values=actual_values,
        metadata={'prediction_time': 0.05, 'model_version': '1.0'}
    )

    print(f"追踪状态: {result1['tracking_status']}")
    if 'performance_rating' in result1:
        rating = result1['performance_rating']
        print(f"性能评级: {rating['overall_rating']} (分数: {rating['rating_score']:.3f})")

    if 'core_metrics' in result1:
        core = result1['core_metrics']
        print(f"核心指标: R² = {core['r2_score']:.4f}, MSE = {core['mse']:.4f}")

    # 测试2：差的预测性能追踪
    print("\n📊 测试2：差的模型性能追踪")
    result2 = tracker.track_performance(
        model_name='poor_model',
        predictions=predictions_poor,
        actual_values=actual_values,
        metadata={'prediction_time': 0.15, 'model_version': '1.0'}
    )

    print(f"追踪状态: {result2['tracking_status']}")
    if 'performance_rating' in result2:
        rating = result2['performance_rating']
        print(f"性能评级: {rating['overall_rating']} (分数: {rating['rating_score']:.3f})")

    # 显示预警
    if 'alerts' in result2 and result2['alerts']:
        print("预警信息:")
        for alert in result2['alerts']:
            print(f"  {alert['type']}: {alert['message']}")

    # 测试3：性能下降追踪
    print("\n📊 测试3：模拟性能下降")
    predictions_decline = actual_values + np.random.randn(n_samples) * 0.3  # 性能下降

    result3 = tracker.track_performance(
        model_name='good_model',  # 同一个模型
        predictions=predictions_decline,
        actual_values=actual_values
    )

    if 'benchmark_comparison' in result3:
        comparison = result3['benchmark_comparison']
        print(f"基准对比: {comparison.get('performance_change', 'unknown')}")

    if 'alerts' in result3 and result3['alerts']:
        print("性能下降预警:")
        for alert in result3['alerts']:
            print(f"  {alert['severity']}: {alert['message']}")

    # 获取性能摘要
    print("\n📈 性能摘要:")
    summary = tracker.get_performance_summary()
    for model_name, model_summary in summary.items():
        if isinstance(model_summary, dict) and 'status' not in model_summary:
            print(f"{model_name}:")
            for key, value in model_summary.items():
                if key != 'last_tracked':
                    print(f"  {key}: {value}")

    # 生成性能报告
    print("\n📄 生成性能报告:")
    report = tracker.generate_performance_report('good_model', period_days=7)
    print(report)

    print("\n✅ 性能追踪器测试完成！")

if __name__ == "__main__":
    main()