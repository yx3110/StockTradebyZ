#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态归一化引擎 - V3.8自适应评分系统核心组件

解决V3.7固定Sigmoid评分敏感性不足的问题
提供基于市场波动和数据特征的自适应归一化

Created: 2025-09-16
Author: Claude Code
Version: 3.8.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class DynamicNormalizer:
    """
    动态归一化引擎

    核心功能:
    1. 自适应Sigmoid参数调整
    2. 市场波动感知归一化
    3. 分布特征自适应调整
    4. 多策略归一化组合
    """

    def __init__(self,
                 market_volatility_window: int = 20,
                 adaptation_sensitivity: float = 0.3,
                 min_sigmoid_slope: float = 0.5,
                 max_sigmoid_slope: float = 5.0,
                 logger: Optional[logging.Logger] = None):
        """
        初始化动态归一化器

        Args:
            market_volatility_window: 市场波动计算窗口
            adaptation_sensitivity: 适应敏感度
            min_sigmoid_slope: Sigmoid斜率最小值
            max_sigmoid_slope: Sigmoid斜率最大值
            logger: 日志记录器
        """
        self.volatility_window = market_volatility_window
        self.adaptation_sensitivity = adaptation_sensitivity
        self.min_slope = min_sigmoid_slope
        self.max_slope = max_sigmoid_slope
        self.logger = logger or logging.getLogger(__name__)

        # 归一化策略库
        self.strategies = {
            'adaptive_sigmoid': self._adaptive_sigmoid,
            'robust_sigmoid': self._robust_sigmoid,
            'quantile_based': self._quantile_based_normalization,
            'market_aware': self._market_aware_normalization,
            'distribution_adaptive': self._distribution_adaptive_normalization
        }

        # 参数历史记录
        self.parameter_history = []
        self.performance_history = []

        # 市场状态缓存
        self.market_state_cache = {}
        self.last_market_update = None

        self.logger.info("动态归一化引擎初始化完成")

    def normalize_scores(self,
                        raw_scores: np.ndarray,
                        market_data: Optional[pd.DataFrame] = None,
                        strategy: str = 'adaptive_sigmoid',
                        custom_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        动态归一化评分

        Args:
            raw_scores: 原始评分数组
            market_data: 市场数据用于上下文感知
            strategy: 归一化策略
            custom_params: 自定义参数

        Returns:
            包含归一化结果和元数据的字典
        """
        try:
            self.logger.info(f"开始动态归一化 - 策略: {strategy}, 样本数: {len(raw_scores)}")

            # 数据验证
            if len(raw_scores) == 0:
                raise ValueError("评分数组不能为空")

            raw_scores = np.array(raw_scores)

            # 分析市场状态
            market_context = self._analyze_market_context(market_data)

            # 分析分布特征
            distribution_stats = self._analyze_distribution(raw_scores)

            # 选择和执行归一化策略
            if strategy not in self.strategies:
                self.logger.warning(f"未知策略 {strategy}, 使用默认adaptive_sigmoid")
                strategy = 'adaptive_sigmoid'

            # 执行归一化
            normalizer_func = self.strategies[strategy]
            normalized_result = normalizer_func(
                raw_scores,
                market_context,
                distribution_stats,
                custom_params or {}
            )

            # 质量评估
            quality_metrics = self._assess_normalization_quality(
                raw_scores,
                normalized_result['normalized_scores']
            )

            # 构建完整结果
            result = {
                'normalized_scores': normalized_result['normalized_scores'],
                'normalization_params': normalized_result['params'],
                'market_context': market_context,
                'distribution_stats': distribution_stats,
                'quality_metrics': quality_metrics,
                'strategy_used': strategy,
                'timestamp': datetime.now()
            }

            # 记录参数历史
            self._update_parameter_history(result)

            self.logger.info(f"归一化完成 - 质量评分: {quality_metrics['overall_quality']:.3f}")
            return result

        except Exception as e:
            self.logger.error(f"动态归一化失败: {e}")
            raise

    def _adaptive_sigmoid(self,
                         scores: np.ndarray,
                         market_context: Dict,
                         distribution_stats: Dict,
                         custom_params: Dict) -> Dict[str, Any]:
        """自适应Sigmoid归一化"""

        # 基础参数
        base_slope = custom_params.get('base_slope', 2.0)
        base_shift = custom_params.get('base_shift', 0.0)

        # 根据市场波动调整斜率
        volatility_factor = market_context.get('volatility_regime', 1.0)

        # 低波动时增加敏感度，高波动时降低敏感度
        if volatility_factor < 0.5:  # 低波动
            slope_adjustment = 1.5
        elif volatility_factor > 1.5:  # 高波动
            slope_adjustment = 0.7
        else:  # 正常波动
            slope_adjustment = 1.0

        # 根据分布特征调整
        skewness = distribution_stats['skewness']
        kurtosis = distribution_stats['kurtosis']

        # 偏度大时调整中心点
        if abs(skewness) > 1.0:
            shift_adjustment = -np.sign(skewness) * min(abs(skewness) * 0.1, 0.3)
        else:
            shift_adjustment = 0.0

        # 峰度高时增加斜率以增强区分度
        if kurtosis > 3.0:
            slope_adjustment *= min(1.0 + (kurtosis - 3.0) * 0.1, 1.5)

        # 应用约束
        final_slope = np.clip(
            base_slope * slope_adjustment * volatility_factor,
            self.min_slope,
            self.max_slope
        )
        final_shift = base_shift + shift_adjustment

        # 修复Sigmoid变换 - 避免过度收敛到0.5
        # 使用原始分数而不是减去均值，以保持差异化
        score_range = np.max(scores) - np.min(scores)

        if score_range < 0.01:  # 如果原始评分差异很小，增强差异
            # 使用标准化后再放大差异
            standardized = (scores - np.mean(scores)) / (np.std(scores) + 1e-8)
            normalized = 1 / (1 + np.exp(-final_slope * standardized * 2.0 + final_shift))
        else:
            # 使用调整后的公式保持更多原始差异
            centered_scores = scores - 0.5  # 相对于中性点0.5而不是均值
            normalized = 1 / (1 + np.exp(-final_slope * centered_scores + final_shift))

        return {
            'normalized_scores': normalized,
            'params': {
                'slope': final_slope,
                'shift': final_shift,
                'volatility_factor': volatility_factor,
                'slope_adjustment': slope_adjustment,
                'shift_adjustment': shift_adjustment
            }
        }

    def _robust_sigmoid(self,
                       scores: np.ndarray,
                       market_context: Dict,
                       distribution_stats: Dict,
                       custom_params: Dict) -> Dict[str, Any]:
        """抗异常值的鲁棒Sigmoid归一化"""

        # 使用分位数代替均值以提高鲁棒性
        q25, q50, q75 = np.percentile(scores, [25, 50, 75])
        iqr = q75 - q25

        # 检测并处理异常值
        outlier_threshold = 3.0
        outlier_mask = np.abs((scores - q50) / (iqr + 1e-8)) > outlier_threshold

        # 对异常值进行压缩
        robust_scores = scores.copy()
        if outlier_mask.any():
            # 将异常值压缩到合理范围
            robust_scores[outlier_mask] = q50 + np.sign(scores[outlier_mask] - q50) * outlier_threshold * iqr

        # 基于鲁棒统计量的Sigmoid参数
        robust_std = iqr / 1.349  # IQR到标准差的转换
        slope = custom_params.get('base_slope', 2.0) / (robust_std + 1e-8)
        slope = np.clip(slope, self.min_slope, self.max_slope)

        # 使用中位数作为中心
        normalized = 1 / (1 + np.exp(-slope * (robust_scores - q50)))

        return {
            'normalized_scores': normalized,
            'params': {
                'slope': slope,
                'center': q50,
                'robust_std': robust_std,
                'outlier_count': int(outlier_mask.sum()),
                'iqr': iqr
            }
        }

    def _quantile_based_normalization(self,
                                    scores: np.ndarray,
                                    market_context: Dict,
                                    distribution_stats: Dict,
                                    custom_params: Dict) -> Dict[str, Any]:
        """基于分位数的归一化"""

        # 计算分位数映射
        n_quantiles = custom_params.get('n_quantiles', 100)
        quantiles = np.linspace(0, 1, n_quantiles)
        quantile_values = np.percentile(scores, quantiles * 100)

        # 将评分映射到[0,1]区间
        normalized = np.interp(scores, quantile_values, quantiles)

        # 根据市场状态进行微调
        market_regime = market_context.get('market_regime', 'normal')
        if market_regime == 'extreme_volatility':
            # 极端波动时压缩极值
            normalized = 0.1 + 0.8 * normalized
        elif market_regime == 'low_volatility':
            # 低波动时增强区分度
            normalized = np.power(normalized, 0.8)

        return {
            'normalized_scores': normalized,
            'params': {
                'n_quantiles': n_quantiles,
                'quantile_range': [quantile_values[0], quantile_values[-1]],
                'market_regime': market_regime
            }
        }

    def _market_aware_normalization(self,
                                   scores: np.ndarray,
                                   market_context: Dict,
                                   distribution_stats: Dict,
                                   custom_params: Dict) -> Dict[str, Any]:
        """市场感知归一化"""

        # 获取市场状态
        volatility_regime = market_context.get('volatility_regime', 1.0)
        trend_strength = market_context.get('trend_strength', 0.0)
        market_sentiment = market_context.get('market_sentiment', 0.0)

        # 基础归一化
        base_normalized = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)

        # 市场状态调整
        # 1. 波动率调整
        if volatility_regime > 1.5:  # 高波动
            # 压缩评分区间，避免过度激进
            adjusted = 0.2 + 0.6 * base_normalized
        elif volatility_regime < 0.5:  # 低波动
            # 拉伸评分区间，增加区分度
            adjusted = np.power(base_normalized, 0.7)
        else:
            adjusted = base_normalized

        # 2. 趋势强度调整
        if abs(trend_strength) > 0.7:
            # 强趋势时偏向趋势方向
            trend_bias = np.sign(trend_strength) * 0.1
            adjusted = np.clip(adjusted + trend_bias * adjusted * (1 - adjusted), 0, 1)

        # 3. 市场情绪调整
        if abs(market_sentiment) > 0.5:
            # 极端情绪时进行均值回归
            sentiment_factor = 1 - abs(market_sentiment) * 0.3
            adjusted = adjusted * sentiment_factor + 0.5 * (1 - sentiment_factor)

        return {
            'normalized_scores': adjusted,
            'params': {
                'volatility_regime': volatility_regime,
                'trend_strength': trend_strength,
                'market_sentiment': market_sentiment,
                'adjustment_applied': True
            }
        }

    def _distribution_adaptive_normalization(self,
                                           scores: np.ndarray,
                                           market_context: Dict,
                                           distribution_stats: Dict,
                                           custom_params: Dict) -> Dict[str, Any]:
        """分布自适应归一化"""

        # 检测分布类型
        skewness = distribution_stats['skewness']
        kurtosis = distribution_stats['kurtosis']

        # 根据分布特征选择变换
        if abs(skewness) > 1.0:  # 高偏度
            if skewness > 0:  # 右偏
                # 对数变换减少右偏
                transformed = np.log1p(scores - np.min(scores) + 1)
            else:  # 左偏
                # 平方根变换
                transformed = np.sqrt(scores - np.min(scores) + 1)
        elif kurtosis > 5.0:  # 高峰度
            # Box-Cox变换
            from scipy.stats import boxcox
            if np.all(scores > 0):
                transformed, lambda_param = boxcox(scores)
            else:
                transformed = np.arctanh(np.clip((scores - np.min(scores)) / (np.max(scores) - np.min(scores)), -0.99, 0.99))
        else:
            # 标准化
            transformed = (scores - np.mean(scores)) / (np.std(scores) + 1e-8)

        # 映射到[0,1]
        normalized = (transformed - np.min(transformed)) / (np.max(transformed) - np.min(transformed) + 1e-8)

        return {
            'normalized_scores': normalized,
            'params': {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'transformation_applied': 'log' if skewness > 1.0 else 'sqrt' if skewness < -1.0 else 'boxcox' if kurtosis > 5.0 else 'standardize'
            }
        }

    def _analyze_market_context(self, market_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """分析市场环境上下文"""

        if market_data is None:
            return {
                'volatility_regime': 1.0,
                'trend_strength': 0.0,
                'market_sentiment': 0.0,
                'market_regime': 'normal'
            }

        try:
            # 计算市场波动率
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                current_vol = returns.rolling(self.volatility_window).std().iloc[-1]
                historical_vol = returns.std()
                volatility_regime = current_vol / (historical_vol + 1e-8)
            else:
                volatility_regime = 1.0

            # 计算趋势强度
            if 'close' in market_data.columns and len(market_data) >= 20:
                prices = market_data['close'].values[-20:]
                trend_slope = np.polyfit(range(len(prices)), prices, 1)[0]
                price_std = np.std(prices)
                trend_strength = trend_slope / (price_std + 1e-8)
                trend_strength = np.clip(trend_strength, -2.0, 2.0)
            else:
                trend_strength = 0.0

            # 估计市场情绪(基于价格动量)
            if 'volume' in market_data.columns and len(market_data) >= 5:
                recent_volume = market_data['volume'].iloc[-5:].mean()
                historical_volume = market_data['volume'].mean()
                volume_ratio = recent_volume / (historical_volume + 1e-8)

                price_momentum = market_data['close'].iloc[-1] / market_data['close'].iloc[-5] - 1
                market_sentiment = price_momentum * np.log1p(volume_ratio)
                market_sentiment = np.clip(market_sentiment, -1.0, 1.0)
            else:
                market_sentiment = 0.0

            # 确定市场状态
            if volatility_regime > 2.0:
                market_regime = 'extreme_volatility'
            elif volatility_regime < 0.3:
                market_regime = 'low_volatility'
            elif abs(trend_strength) > 1.0:
                market_regime = 'strong_trend'
            else:
                market_regime = 'normal'

            return {
                'volatility_regime': volatility_regime,
                'trend_strength': trend_strength,
                'market_sentiment': market_sentiment,
                'market_regime': market_regime
            }

        except Exception as e:
            self.logger.warning(f"市场分析失败，使用默认值: {e}")
            return {
                'volatility_regime': 1.0,
                'trend_strength': 0.0,
                'market_sentiment': 0.0,
                'market_regime': 'normal'
            }

    def _analyze_distribution(self, scores: np.ndarray) -> Dict[str, float]:
        """分析评分分布特征"""

        try:
            return {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'skewness': stats.skew(scores),
                'kurtosis': stats.kurtosis(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'q25': np.percentile(scores, 25),
                'median': np.median(scores),
                'q75': np.percentile(scores, 75),
                'iqr': np.percentile(scores, 75) - np.percentile(scores, 25),
                'outlier_ratio': np.mean(np.abs(stats.zscore(scores)) > 3)
            }
        except Exception as e:
            self.logger.warning(f"分布分析失败: {e}")
            return {
                'mean': 0.0, 'std': 1.0, 'skewness': 0.0, 'kurtosis': 0.0,
                'min': 0.0, 'max': 1.0, 'q25': 0.25, 'median': 0.5, 'q75': 0.75,
                'iqr': 0.5, 'outlier_ratio': 0.0
            }

    def _assess_normalization_quality(self,
                                    original_scores: np.ndarray,
                                    normalized_scores: np.ndarray) -> Dict[str, float]:
        """评估归一化质量"""

        try:
            # 1. 保序性检查
            original_ranks = stats.rankdata(original_scores)
            normalized_ranks = stats.rankdata(normalized_scores)
            rank_correlation = stats.spearmanr(original_ranks, normalized_ranks)[0]

            # 2. 分布特征保持
            original_spread = np.std(original_scores)
            normalized_spread = np.std(normalized_scores)
            spread_preservation = 1 - abs(normalized_spread - 0.25) / 0.25  # 期望std=0.25

            # 3. 区间利用度
            range_utilization = (np.max(normalized_scores) - np.min(normalized_scores))

            # 4. 异常值处理效果
            original_outliers = np.mean(np.abs(stats.zscore(original_scores)) > 3)
            normalized_outliers = np.mean(np.abs(stats.zscore(normalized_scores)) > 3)
            outlier_handling = max(0, (original_outliers - normalized_outliers) / (original_outliers + 1e-8))

            # 综合质量评分
            overall_quality = (
                rank_correlation * 0.4 +
                spread_preservation * 0.3 +
                range_utilization * 0.2 +
                outlier_handling * 0.1
            )

            return {
                'rank_correlation': rank_correlation,
                'spread_preservation': spread_preservation,
                'range_utilization': range_utilization,
                'outlier_handling': outlier_handling,
                'overall_quality': overall_quality
            }

        except Exception as e:
            self.logger.warning(f"质量评估失败: {e}")
            return {
                'rank_correlation': 0.0,
                'spread_preservation': 0.0,
                'range_utilization': 0.0,
                'outlier_handling': 0.0,
                'overall_quality': 0.0
            }

    def _update_parameter_history(self, result: Dict):
        """更新参数历史记录"""

        history_entry = {
            'timestamp': result['timestamp'],
            'strategy': result['strategy_used'],
            'params': result['normalization_params'],
            'quality': result['quality_metrics']['overall_quality'],
            'market_regime': result['market_context']['market_regime']
        }

        self.parameter_history.append(history_entry)

        # 保持历史记录在合理范围
        if len(self.parameter_history) > 1000:
            self.parameter_history = self.parameter_history[-1000:]

    def get_optimization_suggestions(self) -> Dict[str, Any]:
        """基于历史表现提供优化建议"""

        if len(self.parameter_history) < 10:
            return {'suggestion': '历史数据不足，需要更多样本进行分析'}

        try:
            # 分析不同策略的表现
            history_df = pd.DataFrame(self.parameter_history)

            # 按策略分组分析
            strategy_performance = history_df.groupby('strategy')['quality'].agg(['mean', 'std', 'count'])

            # 按市场状态分析
            market_performance = history_df.groupby('market_regime')['quality'].agg(['mean', 'std'])

            # 参数稳定性分析
            param_stability = {}
            for strategy in history_df['strategy'].unique():
                strategy_data = history_df[history_df['strategy'] == strategy]
                if len(strategy_data) >= 5:
                    # 分析参数变化幅度
                    param_changes = []
                    for i in range(1, len(strategy_data)):
                        prev_params = strategy_data.iloc[i-1]['params']
                        curr_params = strategy_data.iloc[i]['params']

                        if isinstance(prev_params, dict) and isinstance(curr_params, dict):
                            for key in prev_params:
                                if key in curr_params and isinstance(prev_params[key], (int, float)):
                                    change = abs(curr_params[key] - prev_params[key]) / (abs(prev_params[key]) + 1e-8)
                                    param_changes.append(change)

                    if param_changes:
                        param_stability[strategy] = np.mean(param_changes)

            return {
                'strategy_performance': strategy_performance.to_dict(),
                'market_performance': market_performance.to_dict(),
                'parameter_stability': param_stability,
                'best_strategy': strategy_performance['mean'].idxmax(),
                'recommendations': self._generate_recommendations(strategy_performance, market_performance)
            }

        except Exception as e:
            self.logger.warning(f"优化建议生成失败: {e}")
            return {'suggestion': f'分析失败: {e}'}

    def _generate_recommendations(self,
                                strategy_perf: pd.DataFrame,
                                market_perf: pd.DataFrame) -> List[str]:
        """生成具体优化建议"""

        recommendations = []

        # 策略选择建议
        best_strategy = strategy_perf['mean'].idxmax()
        recommendations.append(f"推荐使用 {best_strategy} 策略，平均质量评分: {strategy_perf.loc[best_strategy, 'mean']:.3f}")

        # 稳定性建议
        most_stable = strategy_perf['std'].idxmin()
        if most_stable != best_strategy:
            recommendations.append(f"如需稳定性，考虑 {most_stable} 策略 (标准差: {strategy_perf.loc[most_stable, 'std']:.3f})")

        # 市场状态建议
        for regime, performance in market_perf['mean'].items():
            if performance < 0.7:
                recommendations.append(f"在{regime}市场状态下表现较差，建议优化相关参数")

        return recommendations

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统运行状态"""

        return {
            'initialization_time': getattr(self, '_init_time', datetime.now()),
            'total_normalizations': len(self.parameter_history),
            'available_strategies': list(self.strategies.keys()),
            'parameter_history_size': len(self.parameter_history),
            'last_normalization': self.parameter_history[-1]['timestamp'] if self.parameter_history else None,
            'cache_status': {
                'market_state_cached': bool(self.market_state_cache),
                'last_market_update': self.last_market_update
            }
        }