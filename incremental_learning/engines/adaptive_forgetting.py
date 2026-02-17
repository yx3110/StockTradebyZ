#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.8自适应遗忘机制
实现智能的数据权重调整和历史信息遗忘策略

Phase 3: 增量学习机制 - AdaptiveForgetting组件
- 时间衰减权重
- 基于性能的权重调整
- 市场状态感知的遗忘策略
- 异常数据处理

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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats

# 添加项目路径
sys.path.append('/Users/yangxu/StockTradebyZ')

class AdaptiveForgettingEngine:
    """
    自适应遗忘机制引擎

    核心功能：
    1. 时间衰减：根据数据新旧程度调整权重
    2. 性能驱动：基于预测准确性调整权重
    3. 市场状态感知：根据市场环境调整遗忘策略
    4. 异常检测：识别并降低异常数据权重
    5. 自适应参数调整：动态优化遗忘参数
    """

    def __init__(self,
                 forgetting_factors: Dict[str, float],
                 market_regimes: List[str],
                 logger: logging.Logger,
                 min_weight: float = 0.01,
                 max_weight: float = 1.0):

        self.forgetting_factors = forgetting_factors
        self.market_regimes = market_regimes
        self.logger = logger
        self.min_weight = min_weight
        self.max_weight = max_weight

        # 市场状态检测器
        self.market_detector = MarketRegimeDetector(logger)

        # 异常检测器
        self.anomaly_detector = AnomalyDetector(logger)

        # 权重历史记录
        self.weight_history = []
        self.performance_history = []

        # 自适应参数
        self.adaptive_params = {
            'time_decay_rate': 0.95,
            'performance_weight': 0.3,
            'market_adjustment': 0.2,
            'anomaly_penalty': 0.5
        }

        self.logger.info("🧠 自适应遗忘机制初始化完成")

    def calculate_adaptive_weights(self,
                                 historical_data: pd.DataFrame,
                                 recent_performance: List[Dict],
                                 current_date: datetime = None) -> np.ndarray:
        """
        计算自适应权重

        Args:
            historical_data: 历史数据 (包含时间戳)
            recent_performance: 最近的性能记录
            current_date: 当前日期

        Returns:
            每个样本的权重数组
        """
        if current_date is None:
            current_date = datetime.now()

        self.logger.info(f"🔍 计算 {len(historical_data)} 条数据的自适应权重")

        # 1. 计算基础时间衰减权重
        time_weights = self._calculate_time_decay_weights(
            historical_data, current_date
        )

        # 2. 计算性能驱动权重
        performance_weights = self._calculate_performance_weights(
            historical_data, recent_performance
        )

        # 3. 计算市场状态调整权重
        market_weights = self._calculate_market_regime_weights(
            historical_data, current_date
        )

        # 4. 计算异常检测权重
        anomaly_weights = self._calculate_anomaly_weights(
            historical_data
        )

        # 5. 综合权重计算
        combined_weights = self._combine_weights(
            time_weights, performance_weights,
            market_weights, anomaly_weights
        )

        # 6. 权重归一化和约束
        final_weights = self._normalize_and_constrain_weights(combined_weights)

        # 7. 记录权重统计
        self._log_weight_statistics(final_weights)

        # 8. 更新权重历史
        self._update_weight_history(final_weights, current_date)

        return final_weights

    def _calculate_time_decay_weights(self,
                                    historical_data: pd.DataFrame,
                                    current_date: datetime) -> np.ndarray:
        """
        计算时间衰减权重
        """
        if 'trade_date' not in historical_data.columns:
            # 如果没有时间列，假设数据按时间排序
            n = len(historical_data)
            time_indices = np.arange(n)
            time_weights = np.power(self.forgetting_factors.get('short', 0.95),
                                   n - 1 - time_indices)
        else:
            # 基于实际时间差计算权重
            dates = pd.to_datetime(historical_data['trade_date'])
            days_ago = (current_date - dates).dt.days

            # 不同时间范围使用不同的衰减率
            time_weights = np.ones(len(historical_data))

            # 短期 (1-7天): 慢衰减
            short_mask = days_ago <= 7
            time_weights[short_mask] = np.power(
                self.forgetting_factors.get('short', 0.98),
                days_ago[short_mask]
            )

            # 中期 (8-30天): 中等衰减
            medium_mask = (days_ago > 7) & (days_ago <= 30)
            if medium_mask.any():
                short_min = time_weights[short_mask].min() if short_mask.any() else 1.0
                time_weights[medium_mask] = np.power(
                    self.forgetting_factors.get('medium', 0.95),
                    days_ago[medium_mask] - 7
                ) * short_min

            # 长期 (30天+): 快速衰减
            long_mask = days_ago > 30
            if long_mask.any():
                medium_min = time_weights[medium_mask].min() if medium_mask.any() else (
                    time_weights[short_mask].min() if short_mask.any() else 1.0
                )
                time_weights[long_mask] = np.power(
                    self.forgetting_factors.get('long', 0.90),
                    days_ago[long_mask] - 30
                ) * medium_min

        return time_weights

    def _calculate_performance_weights(self,
                                     historical_data: pd.DataFrame,
                                     recent_performance: List[Dict]) -> np.ndarray:
        """
        基于性能计算权重
        """
        n_samples = len(historical_data)
        performance_weights = np.ones(n_samples)

        if not recent_performance:
            return performance_weights

        # 计算滑动窗口内的预测准确性
        window_size = min(50, n_samples // 4)  # 动态窗口大小

        if len(recent_performance) >= window_size:
            # 计算每个时间窗口的平均性能
            performance_scores = [p.get('r2', 0.5) for p in recent_performance[-window_size:]]

            # 为每个样本分配性能权重
            for i in range(n_samples):
                # 为较新的、表现好的时期赋予更高权重
                window_idx = min(i // (n_samples // len(performance_scores)),
                               len(performance_scores) - 1)

                perf_score = performance_scores[window_idx]

                # 性能越好，权重越高
                if perf_score > 0.7:  # 高性能
                    performance_weights[i] = 1.2
                elif perf_score > 0.5:  # 中等性能
                    performance_weights[i] = 1.0
                elif perf_score > 0.3:  # 低性能
                    performance_weights[i] = 0.8
                else:  # 很差性能
                    performance_weights[i] = 0.5

        return performance_weights

    def _calculate_market_regime_weights(self,
                                       historical_data: pd.DataFrame,
                                       current_date: datetime) -> np.ndarray:
        """
        基于市场状态计算权重
        """
        n_samples = len(historical_data)
        market_weights = np.ones(n_samples)

        try:
            # 检测当前市场状态
            current_regime = self.market_detector.detect_current_regime(
                historical_data, current_date
            )

            # 为历史数据中的每个时期检测市场状态
            historical_regimes = self.market_detector.detect_historical_regimes(
                historical_data
            )

            # 根据市场状态相似性调整权重
            for i, historical_regime in enumerate(historical_regimes):
                if historical_regime == current_regime:
                    # 相同市场状态的数据权重增加
                    market_weights[i] *= 1.3
                elif self._are_similar_regimes(historical_regime, current_regime):
                    # 相似市场状态的数据权重略增
                    market_weights[i] *= 1.1
                else:
                    # 不同市场状态的数据权重降低
                    market_weights[i] *= 0.8

        except Exception as e:
            self.logger.warning(f"⚠️ 市场状态权重计算失败: {e}")

        return market_weights

    def _calculate_anomaly_weights(self, historical_data: pd.DataFrame) -> np.ndarray:
        """
        基于异常检测计算权重
        """
        n_samples = len(historical_data)
        anomaly_weights = np.ones(n_samples)

        try:
            # 检测异常样本
            anomaly_scores = self.anomaly_detector.detect_anomalies(historical_data)

            # 根据异常分数调整权重
            for i, score in enumerate(anomaly_scores):
                if score > 0.8:  # 高异常
                    anomaly_weights[i] = self.adaptive_params['anomaly_penalty']
                elif score > 0.6:  # 中等异常
                    anomaly_weights[i] = 0.7
                elif score > 0.4:  # 轻微异常
                    anomaly_weights[i] = 0.9
                # 正常数据保持权重 1.0

        except Exception as e:
            self.logger.warning(f"⚠️ 异常权重计算失败: {e}")

        return anomaly_weights

    def _combine_weights(self,
                        time_weights: np.ndarray,
                        performance_weights: np.ndarray,
                        market_weights: np.ndarray,
                        anomaly_weights: np.ndarray) -> np.ndarray:
        """
        综合各种权重
        """
        # 加权平均组合
        combined_weights = (
            time_weights * 0.4 +                           # 时间权重占40%
            performance_weights * self.adaptive_params['performance_weight'] +  # 性能权重占30%
            market_weights * self.adaptive_params['market_adjustment'] +        # 市场权重占20%
            anomaly_weights * 0.1                          # 异常权重占10%
        )

        return combined_weights

    def _normalize_and_constrain_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        权重归一化和约束
        """
        # 约束到最小最大范围
        constrained_weights = np.clip(weights, self.min_weight, self.max_weight)

        # L1归一化（保证权重总和等于样本数量）
        normalized_weights = constrained_weights / constrained_weights.mean()

        return normalized_weights

    def _log_weight_statistics(self, weights: np.ndarray):
        """
        记录权重统计信息
        """
        stats_info = {
            'mean': float(np.mean(weights)),
            'std': float(np.std(weights)),
            'min': float(np.min(weights)),
            'max': float(np.max(weights)),
            'median': float(np.median(weights))
        }

        self.logger.info(f"📊 权重统计 - 均值: {stats_info['mean']:.3f}, "
                        f"标准差: {stats_info['std']:.3f}, "
                        f"范围: [{stats_info['min']:.3f}, {stats_info['max']:.3f}]")

    def _update_weight_history(self, weights: np.ndarray, current_date: datetime):
        """
        更新权重历史记录
        """
        weight_record = {
            'date': current_date,
            'weights': weights.copy(),
            'stats': {
                'mean': np.mean(weights),
                'std': np.std(weights),
                'effective_sample_size': np.sum(weights) ** 2 / np.sum(weights ** 2)
            }
        }

        self.weight_history.append(weight_record)

        # 保持历史记录长度
        max_history = 100
        if len(self.weight_history) > max_history:
            self.weight_history = self.weight_history[-max_history:]

    def _are_similar_regimes(self, regime1: str, regime2: str) -> bool:
        """
        判断两个市场状态是否相似
        """
        similar_pairs = [
            ('bull_market', 'growth_phase'),
            ('bear_market', 'recession'),
            ('volatile_market', 'correction'),
            ('sideways_market', 'consolidation')
        ]

        for pair in similar_pairs:
            if (regime1 in pair and regime2 in pair):
                return True

        return False

    def adapt_parameters(self, recent_performance: List[Dict]):
        """
        根据最近性能自适应调整参数
        """
        if len(recent_performance) < 5:
            return

        try:
            # 计算性能趋势
            recent_scores = [p.get('r2', 0.5) for p in recent_performance[-10:]]
            performance_trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]

            # 根据趋势调整参数
            if performance_trend > 0.01:  # 性能提升
                # 减少遗忘，保留更多历史信息
                for key in self.forgetting_factors:
                    self.forgetting_factors[key] = min(0.99, self.forgetting_factors[key] * 1.01)
                self.adaptive_params['performance_weight'] *= 0.95

            elif performance_trend < -0.01:  # 性能下降
                # 增加遗忘，更关注新数据
                for key in self.forgetting_factors:
                    self.forgetting_factors[key] = max(0.85, self.forgetting_factors[key] * 0.99)
                self.adaptive_params['performance_weight'] *= 1.05

            # 约束参数范围
            self.adaptive_params['performance_weight'] = np.clip(
                self.adaptive_params['performance_weight'], 0.1, 0.5
            )

            self.logger.info(f"🔧 参数自适应调整完成 - 趋势: {performance_trend:.4f}")

        except Exception as e:
            self.logger.error(f"❌ 参数自适应调整失败: {e}")

    def get_forgetting_summary(self) -> Dict[str, Any]:
        """
        获取遗忘机制摘要
        """
        if not self.weight_history:
            return {'status': 'no_history'}

        recent_weights = self.weight_history[-1]['weights']

        return {
            'current_forgetting_factors': self.forgetting_factors.copy(),
            'adaptive_params': self.adaptive_params.copy(),
            'recent_weight_stats': {
                'effective_sample_size': self.weight_history[-1]['stats']['effective_sample_size'],
                'weight_concentration': 1.0 - np.std(recent_weights),
                'forgetting_intensity': 1.0 - np.mean(recent_weights)
            },
            'parameter_stability': self._calculate_parameter_stability(),
            'weight_history_length': len(self.weight_history)
        }

    def _calculate_parameter_stability(self) -> float:
        """
        计算参数稳定性得分
        """
        if len(self.weight_history) < 5:
            return 1.0

        # 计算最近权重分布的变化
        recent_means = [h['stats']['mean'] for h in self.weight_history[-5:]]
        stability_score = 1.0 - (np.std(recent_means) / np.mean(recent_means)) if np.mean(recent_means) > 0 else 1.0

        return max(0.0, min(1.0, stability_score))


class MarketRegimeDetector:
    """
    市场状态检测器
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def detect_current_regime(self, data: pd.DataFrame, current_date: datetime) -> str:
        """
        检测当前市场状态
        """
        try:
            # 简化的市场状态检测逻辑
            if 'price_change_pct' in data.columns:
                recent_returns = data['price_change_pct'].tail(20)

                volatility = recent_returns.std()
                trend = recent_returns.mean()

                if trend > 0.02 and volatility < 0.03:
                    return 'bull_market'
                elif trend < -0.02 and volatility < 0.03:
                    return 'bear_market'
                elif volatility > 0.05:
                    return 'volatile_market'
                else:
                    return 'sideways_market'
            else:
                return 'unknown'

        except Exception as e:
            self.logger.warning(f"⚠️ 市场状态检测失败: {e}")
            return 'unknown'

    def detect_historical_regimes(self, data: pd.DataFrame) -> List[str]:
        """
        检测历史市场状态
        """
        try:
            regimes = []
            window_size = 20

            if 'price_change_pct' not in data.columns:
                return ['unknown'] * len(data)

            for i in range(len(data)):
                start_idx = max(0, i - window_size + 1)
                window_data = data['price_change_pct'].iloc[start_idx:i+1]

                if len(window_data) < 5:
                    regimes.append('unknown')
                    continue

                volatility = window_data.std()
                trend = window_data.mean()

                if trend > 0.02 and volatility < 0.03:
                    regime = 'bull_market'
                elif trend < -0.02 and volatility < 0.03:
                    regime = 'bear_market'
                elif volatility > 0.05:
                    regime = 'volatile_market'
                else:
                    regime = 'sideways_market'

                regimes.append(regime)

            return regimes

        except Exception as e:
            self.logger.warning(f"⚠️ 历史市场状态检测失败: {e}")
            return ['unknown'] * len(data)


class AnomalyDetector:
    """
    异常检测器
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def detect_anomalies(self, data: pd.DataFrame) -> np.ndarray:
        """
        检测异常样本，返回异常分数 (0-1)
        """
        try:
            # 选择数值型特征进行异常检测
            numeric_columns = data.select_dtypes(include=[np.number]).columns

            if len(numeric_columns) == 0:
                return np.zeros(len(data))

            numeric_data = data[numeric_columns].fillna(0)

            # 使用多种方法检测异常
            anomaly_scores = np.zeros(len(data))

            # 1. 基于Z-score的异常检测
            z_scores = np.abs(stats.zscore(numeric_data, axis=0, nan_policy='omit'))
            z_anomalies = np.max(z_scores, axis=1)  # 取每行的最大Z-score
            z_anomalies = np.clip(z_anomalies / 4, 0, 1)  # 归一化到0-1

            # 2. 基于四分位距(IQR)的异常检测
            iqr_anomalies = np.zeros(len(data))
            for col in numeric_columns:
                col_data = numeric_data[col]
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
                iqr_anomalies[outlier_mask] = 1

            # 综合异常分数
            anomaly_scores = 0.6 * z_anomalies + 0.4 * iqr_anomalies
            anomaly_scores = np.clip(anomaly_scores, 0, 1)

            return anomaly_scores

        except Exception as e:
            self.logger.warning(f"⚠️ 异常检测失败: {e}")
            return np.zeros(len(data))


def main():
    """测试自适应遗忘机制"""
    print("🧠 测试自适应遗忘机制...")

    # 配置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 创建遗忘引擎
    forgetting_factors = {
        'short': 0.98,
        'medium': 0.95,
        'long': 0.90
    }

    market_regimes = ['bull_market', 'bear_market', 'volatile_market', 'sideways_market']

    forgetting_engine = AdaptiveForgettingEngine(
        forgetting_factors=forgetting_factors,
        market_regimes=market_regimes,
        logger=logger
    )

    # 生成测试数据
    np.random.seed(42)
    n_samples = 200

    # 模拟历史数据
    dates = pd.date_range(start='2025-01-01', periods=n_samples, freq='D')
    historical_data = pd.DataFrame({
        'trade_date': dates,
        'price_change_pct': np.random.normal(0.001, 0.02, n_samples),
        'volume': np.random.lognormal(15, 1, n_samples),
        'volatility': np.random.uniform(0.01, 0.05, n_samples)
    })

    # 添加一些异常值
    historical_data.loc[50:55, 'price_change_pct'] = np.random.normal(0, 0.1, 6)  # 异常波动
    historical_data.loc[100:102, 'volume'] = historical_data['volume'].median() * 10  # 异常成交量

    # 模拟性能历史
    recent_performance = [
        {'r2': 0.8, 'mse': 0.02},
        {'r2': 0.75, 'mse': 0.025},
        {'r2': 0.82, 'mse': 0.018},
        {'r2': 0.78, 'mse': 0.022},
        {'r2': 0.85, 'mse': 0.015}
    ]

    # 计算自适应权重
    print("\n🔍 计算自适应权重...")
    weights = forgetting_engine.calculate_adaptive_weights(
        historical_data,
        recent_performance,
        datetime.now()
    )

    print(f"✅ 权重计算完成")
    print(f"权重范围: [{weights.min():.3f}, {weights.max():.3f}]")
    print(f"权重均值: {weights.mean():.3f}")
    print(f"有效样本量: {len(weights[weights > 0.1])}/{len(weights)}")

    # 测试参数自适应
    print("\n🔧 测试参数自适应...")
    forgetting_engine.adapt_parameters(recent_performance)

    # 获取遗忘机制摘要
    print("\n📊 遗忘机制摘要:")
    summary = forgetting_engine.get_forgetting_summary()
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

    print("\n✅ 自适应遗忘机制测试完成！")

if __name__ == "__main__":
    main()