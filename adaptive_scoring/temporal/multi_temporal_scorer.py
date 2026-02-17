#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多时间维度评分系统 - V3.8自适应评分系统核心组件

解决单一时间尺度评分的局限性问题
提供短期、中期、长期多维度综合评分

Created: 2025-09-16
Author: Claude Code
Version: 3.8.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class MultiTemporalScorer:
    """
    多时间维度评分系统

    核心功能:
    1. 短期评分 (1-5天): 技术指标 + 动量
    2. 中期评分 (5-20天): 趋势 + 基本面
    3. 长期评分 (20-60天): 基本面 + 行业比较
    4. 时间维度权重自适应调整
    """

    def __init__(self,
                 short_term_window: int = 5,
                 medium_term_window: int = 20,
                 long_term_window: int = 60,
                 weight_adaptation_method: str = 'performance_based',
                 logger: Optional[logging.Logger] = None):
        """
        初始化多时间维度评分系统

        Args:
            short_term_window: 短期评分窗口
            medium_term_window: 中期评分窗口
            long_term_window: 长期评分窗口
            weight_adaptation_method: 权重自适应方法
            logger: 日志记录器
        """
        self.short_window = short_term_window
        self.medium_window = medium_term_window
        self.long_window = long_term_window
        self.weight_method = weight_adaptation_method
        self.logger = logger or logging.getLogger(__name__)

        # 时间维度配置
        self.temporal_configs = {
            'short_term': {
                'window': self.short_window,
                'weight_factors': ['volatility', 'volume', 'momentum'],
                'base_weight': 0.3,
                'indicators': ['rsi', 'kdj', 'macd', 'bb_position', 'volume_ratio']
            },
            'medium_term': {
                'window': self.medium_window,
                'weight_factors': ['trend_strength', 'sector_rotation'],
                'base_weight': 0.4,
                'indicators': ['ma_trend', 'ema_cross', 'trend_strength', 'relative_strength', 'pe_ratio']
            },
            'long_term': {
                'window': self.long_window,
                'weight_factors': ['fundamental_strength', 'macro_environment'],
                'base_weight': 0.3,
                'indicators': ['roe', 'growth_rate', 'debt_ratio', 'industry_ranking', 'valuation_score']
            }
        }

        # 历史权重记录
        self.weight_history = []
        self.performance_history = {}

        # 评分器组件
        self.scalers = {
            'short_term': StandardScaler(),
            'medium_term': StandardScaler(),
            'long_term': StandardScaler()
        }

        # PCA降维器(可选)
        self.pca_reducers = {
            'short_term': None,
            'medium_term': None,
            'long_term': None
        }

        self.logger.info("多时间维度评分系统初始化完成")

    def calculate_multi_temporal_scores(self,
                                      stock_data: pd.DataFrame,
                                      fundamental_data: Optional[pd.DataFrame] = None,
                                      market_data: Optional[pd.DataFrame] = None,
                                      custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        计算多时间维度综合评分

        Args:
            stock_data: 股票价格和技术指标数据
            fundamental_data: 基本面数据
            market_data: 市场环境数据
            custom_weights: 自定义时间维度权重

        Returns:
            包含各时间维度评分和综合评分的结果
        """
        try:
            self.logger.info("开始计算多时间维度评分")

            # 数据验证
            if stock_data.empty:
                raise ValueError("股票数据不能为空")

            # 计算各时间维度评分
            temporal_scores = {}

            # 1. 短期评分
            temporal_scores['short_term'] = self._calculate_short_term_score(
                stock_data.tail(max(self.short_window * 2, 20))
            )

            # 2. 中期评分
            temporal_scores['medium_term'] = self._calculate_medium_term_score(
                stock_data.tail(max(self.medium_window * 2, 50)),
                fundamental_data
            )

            # 3. 长期评分
            temporal_scores['long_term'] = self._calculate_long_term_score(
                stock_data.tail(max(self.long_window * 2, 120)),
                fundamental_data
            )

            # 计算自适应权重
            adaptive_weights = self._calculate_adaptive_weights(
                stock_data,
                market_data,
                custom_weights
            )

            # 计算综合评分
            composite_score = self._calculate_composite_score(
                temporal_scores,
                adaptive_weights
            )

            # 生成评分质量指标
            quality_metrics = self._assess_scoring_quality(
                temporal_scores,
                stock_data
            )

            result = {
                'temporal_scores': temporal_scores,
                'adaptive_weights': adaptive_weights,
                'composite_score': composite_score,
                'quality_metrics': quality_metrics,
                'timestamp': datetime.now()
            }

            # 记录历史
            self._update_scoring_history(result)

            self.logger.info(f"多时间维度评分完成 - 综合评分: {composite_score:.3f}")
            return result

        except Exception as e:
            self.logger.error(f"多时间维度评分失败: {e}")
            raise

    def _calculate_short_term_score(self, data: pd.DataFrame) -> Dict[str, Any]:
        """计算短期评分（1-5天）"""

        try:
            scores = {}
            weights = {}

            # 1. RSI评分 (超买超卖)
            if 'rsi' in data.columns:
                rsi = data['rsi'].iloc[-1]
                if rsi < 30:
                    rsi_score = 0.8  # 超卖，看涨
                elif rsi > 70:
                    rsi_score = 0.2  # 超买，看跌
                else:
                    rsi_score = 0.5 + (50 - rsi) / 100  # 中性区间线性映射
                scores['rsi'] = rsi_score
                weights['rsi'] = 0.25

            # 2. KDJ评分 (金叉死叉)
            if all(col in data.columns for col in ['kdj_k', 'kdj_d']):
                k_current = data['kdj_k'].iloc[-1]
                d_current = data['kdj_d'].iloc[-1]
                k_prev = data['kdj_k'].iloc[-2] if len(data) > 1 else k_current
                d_prev = data['kdj_d'].iloc[-2] if len(data) > 1 else d_current

                # 金叉死叉检测
                if k_prev <= d_prev and k_current > d_current:  # 金叉
                    kdj_score = 0.8
                elif k_prev >= d_prev and k_current < d_current:  # 死叉
                    kdj_score = 0.2
                else:
                    # 基于当前位置
                    avg_kd = (k_current + d_current) / 2
                    if avg_kd < 20:
                        kdj_score = 0.7
                    elif avg_kd > 80:
                        kdj_score = 0.3
                    else:
                        kdj_score = 0.5
                scores['kdj'] = kdj_score
                weights['kdj'] = 0.2

            # 3. MACD评分 (趋势动量)
            if all(col in data.columns for col in ['macd', 'macd_signal']):
                macd = data['macd'].iloc[-1]
                macd_signal = data['macd_signal'].iloc[-1]
                macd_hist = macd - macd_signal

                # MACD柱状图变化
                if len(data) > 1:
                    prev_hist = data['macd'].iloc[-2] - data['macd_signal'].iloc[-2]
                    if macd_hist > 0 and prev_hist <= 0:  # 金叉上穿
                        macd_score = 0.8
                    elif macd_hist < 0 and prev_hist >= 0:  # 死叉下穿
                        macd_score = 0.2
                    else:
                        # 基于柱状图强度
                        macd_score = 0.5 + np.tanh(macd_hist * 10) * 0.3
                else:
                    macd_score = 0.5 + np.tanh(macd_hist * 10) * 0.3

                scores['macd'] = macd_score
                weights['macd'] = 0.2

            # 4. 布林带位置评分
            if all(col in data.columns for col in ['close', 'bb_upper', 'bb_lower']):
                close = data['close'].iloc[-1]
                bb_upper = data['bb_upper'].iloc[-1]
                bb_lower = data['bb_lower'].iloc[-1]

                bb_position = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)
                if bb_position < 0.2:
                    bb_score = 0.8  # 接近下轨，超卖
                elif bb_position > 0.8:
                    bb_score = 0.2  # 接近上轨，超买
                else:
                    bb_score = 0.5

                scores['bb_position'] = bb_score
                weights['bb_position'] = 0.15

            # 5. 成交量比率评分
            if 'volume' in data.columns and len(data) >= 5:
                recent_vol = data['volume'].iloc[-1]
                avg_vol = data['volume'].iloc[-5:].mean()
                vol_ratio = recent_vol / (avg_vol + 1e-8)

                if vol_ratio > 2.0:  # 放量
                    vol_score = 0.7
                elif vol_ratio < 0.5:  # 缩量
                    vol_score = 0.3
                else:
                    vol_score = 0.5

                scores['volume_ratio'] = vol_score
                weights['volume_ratio'] = 0.1

            # 6. 价格动量评分
            if 'close' in data.columns and len(data) >= self.short_window:
                returns = data['close'].pct_change().dropna()
                recent_momentum = returns.iloc[-self.short_window:].mean()

                momentum_score = 0.5 + np.tanh(recent_momentum * 50) * 0.4
                scores['momentum'] = momentum_score
                weights['momentum'] = 0.1

            # 计算加权平均
            if scores:
                total_weight = sum(weights.values())
                weighted_score = sum(score * weights.get(indicator, 0) for indicator, score in scores.items()) / total_weight
            else:
                weighted_score = 0.5

            return {
                'overall_score': weighted_score,
                'component_scores': scores,
                'component_weights': weights,
                'indicators_used': len(scores)
            }

        except Exception as e:
            self.logger.warning(f"短期评分计算失败: {e}")
            return {
                'overall_score': 0.5,
                'component_scores': {},
                'component_weights': {},
                'indicators_used': 0
            }

    def _calculate_medium_term_score(self,
                                   data: pd.DataFrame,
                                   fundamental_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """计算中期评分（5-20天）"""

        try:
            scores = {}
            weights = {}

            # 1. 移动均线趋势评分
            if 'close' in data.columns and len(data) >= 20:
                close = data['close']
                ma5 = close.rolling(5).mean()
                ma10 = close.rolling(10).mean()
                ma20 = close.rolling(20).mean()

                current_close = close.iloc[-1]
                current_ma5 = ma5.iloc[-1]
                current_ma10 = ma10.iloc[-1]
                current_ma20 = ma20.iloc[-1]

                # 均线排列评分
                if current_close > current_ma5 > current_ma10 > current_ma20:
                    ma_trend_score = 0.9  # 多头排列
                elif current_close < current_ma5 < current_ma10 < current_ma20:
                    ma_trend_score = 0.1  # 空头排列
                else:
                    # 基于价格相对位置
                    ma_avg = (current_ma5 + current_ma10 + current_ma20) / 3
                    price_position = current_close / ma_avg
                    ma_trend_score = 0.5 + (price_position - 1) * 2
                    ma_trend_score = np.clip(ma_trend_score, 0, 1)

                scores['ma_trend'] = ma_trend_score
                weights['ma_trend'] = 0.3

            # 2. EMA交叉评分
            if 'close' in data.columns and len(data) >= 12:
                close = data['close']
                ema12 = close.ewm(span=12).mean()
                ema26 = close.ewm(span=26).mean()

                current_ema12 = ema12.iloc[-1]
                current_ema26 = ema26.iloc[-1]
                prev_ema12 = ema12.iloc[-2] if len(ema12) > 1 else current_ema12
                prev_ema26 = ema26.iloc[-2] if len(ema26) > 1 else current_ema26

                # EMA交叉检测
                if prev_ema12 <= prev_ema26 and current_ema12 > current_ema26:
                    ema_score = 0.8  # 金叉
                elif prev_ema12 >= prev_ema26 and current_ema12 < current_ema26:
                    ema_score = 0.2  # 死叉
                else:
                    # 基于EMA差距
                    ema_diff = (current_ema12 - current_ema26) / current_ema26
                    ema_score = 0.5 + np.tanh(ema_diff * 20) * 0.3

                scores['ema_cross'] = ema_score
                weights['ema_cross'] = 0.2

            # 3. 趋势强度评分
            if 'close' in data.columns and len(data) >= self.medium_window:
                prices = data['close'].iloc[-self.medium_window:]

                # 线性回归计算趋势
                x = np.arange(len(prices))
                slope, intercept = np.polyfit(x, prices, 1)

                # 趋势强度 = 斜率 / 价格标准差
                price_std = prices.std()
                trend_strength = slope / (price_std + 1e-8)

                # R²计算趋势一致性
                predicted = slope * x + intercept
                r_squared = 1 - np.sum((prices - predicted) ** 2) / np.sum((prices - prices.mean()) ** 2)

                # 综合趋势评分
                if r_squared > 0.7 and trend_strength > 0.1:
                    trend_score = 0.8
                elif r_squared > 0.7 and trend_strength < -0.1:
                    trend_score = 0.2
                else:
                    trend_score = 0.5 + trend_strength * 2
                    trend_score = np.clip(trend_score, 0, 1)

                scores['trend_strength'] = trend_score
                weights['trend_strength'] = 0.25

            # 4. 相对强度评分 (vs 大盘)
            if 'close' in data.columns and len(data) >= self.medium_window:
                # 简化版相对强度 (实际应用中需要大盘数据)
                stock_returns = data['close'].pct_change().dropna()
                recent_performance = (1 + stock_returns.iloc[-self.medium_window:]).prod() - 1

                # 假设市场平均收益率为0(实际应用中应使用真实市场数据)
                if recent_performance > 0.05:
                    relative_score = 0.7
                elif recent_performance < -0.05:
                    relative_score = 0.3
                else:
                    relative_score = 0.5 + recent_performance * 5
                    relative_score = np.clip(relative_score, 0, 1)

                scores['relative_strength'] = relative_score
                weights['relative_strength'] = 0.15

            # 5. 基本面评分 (如果有数据)
            if fundamental_data is not None and not fundamental_data.empty:
                fundamental_score = self._calculate_fundamental_score(fundamental_data)
                scores['fundamental'] = fundamental_score
                weights['fundamental'] = 0.1

            # 计算加权平均
            if scores:
                total_weight = sum(weights.values())
                weighted_score = sum(score * weights.get(indicator, 0) for indicator, score in scores.items()) / total_weight
            else:
                weighted_score = 0.5

            return {
                'overall_score': weighted_score,
                'component_scores': scores,
                'component_weights': weights,
                'indicators_used': len(scores)
            }

        except Exception as e:
            self.logger.warning(f"中期评分计算失败: {e}")
            return {
                'overall_score': 0.5,
                'component_scores': {},
                'component_weights': {},
                'indicators_used': 0
            }

    def _calculate_long_term_score(self,
                                 data: pd.DataFrame,
                                 fundamental_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """计算长期评分（20-60天）"""

        try:
            scores = {}
            weights = {}

            # 1. 长期趋势评分
            if 'close' in data.columns and len(data) >= 20:  # 降低最小数据要求
                close_prices = data['close'].dropna()  # 移除NaN值

                if len(close_prices) >= 20:  # 确保有足够的数据点
                    # 使用可用数据计算移动均线，如果有ma60字段优先使用
                    if 'ma60' in data.columns:
                        ma60_values = data['ma60'].dropna()
                        if not ma60_values.empty:
                            current_ma60 = ma60_values.iloc[-1]
                            price_ma_ratio = close_prices.iloc[-1] / current_ma60

                            # MA60趋势 (如果有足够的MA60数据)
                            if len(ma60_values) >= 10:
                                ma60_trend = (ma60_values.iloc[-1] - ma60_values.iloc[-10]) / ma60_values.iloc[-10]
                            else:
                                ma60_trend = 0
                        else:
                            # 如果ma60字段为空，计算简单长期移动平均
                            window_size = min(40, len(close_prices))  # 使用可用数据长度
                            ma_long = close_prices.rolling(window_size).mean().iloc[-1]
                            price_ma_ratio = close_prices.iloc[-1] / ma_long if not pd.isna(ma_long) else 1.0
                            ma60_trend = 0
                    else:
                        # 没有ma60字段，计算长期移动平均
                        window_size = min(40, len(close_prices))
                        ma_long = close_prices.rolling(window_size).mean()
                        if not ma_long.iloc[-1] or pd.isna(ma_long.iloc[-1]):
                            price_ma_ratio = 1.0
                            ma60_trend = 0
                        else:
                            price_ma_ratio = close_prices.iloc[-1] / ma_long.iloc[-1]
                            # 计算长期均线趋势
                            if len(ma_long) >= 10:
                                ma60_trend = (ma_long.iloc[-1] - ma_long.iloc[-10]) / ma_long.iloc[-10]
                            else:
                                ma60_trend = 0

                    # 综合长期趋势评分
                    trend_score = 0.5 + (price_ma_ratio - 1) * 1.5 + ma60_trend * 8  # 调整敏感度
                    trend_score = np.clip(trend_score, 0, 1)

                    scores['long_term_trend'] = trend_score
                    weights['long_term_trend'] = 0.4

            # 2. 基本面综合评分
            if fundamental_data is not None and not fundamental_data.empty:
                fundamental_score = self._calculate_comprehensive_fundamental_score(fundamental_data)
                scores['fundamental_comprehensive'] = fundamental_score
                weights['fundamental_comprehensive'] = 0.3

            # 3. 估值评分
            if fundamental_data is not None and 'pe_ratio' in fundamental_data.columns:
                pe_ratio = fundamental_data['pe_ratio'].iloc[-1] if not fundamental_data.empty else None

                if pe_ratio and pe_ratio > 0:
                    # PE估值评分 (行业中位数为基准)
                    if pe_ratio < 15:
                        valuation_score = 0.8  # 低估
                    elif pe_ratio > 40:
                        valuation_score = 0.2  # 高估
                    else:
                        # 15-40区间线性映射
                        valuation_score = 0.8 - (pe_ratio - 15) / 25 * 0.6
                else:
                    valuation_score = 0.5

                scores['valuation'] = valuation_score
                weights['valuation'] = 0.2

            # 4. 长期动量评分
            if 'close' in data.columns and len(data) >= self.long_window:
                # 计算不同周期的收益率
                returns_1m = (data['close'].iloc[-1] / data['close'].iloc[-20] - 1) if len(data) >= 20 else 0
                returns_3m = (data['close'].iloc[-1] / data['close'].iloc[-60] - 1) if len(data) >= 60 else 0

                # 动量一致性检查
                momentum_consistency = 1 if returns_1m * returns_3m > 0 else 0.5

                # 基于收益率计算动量评分
                avg_momentum = (returns_1m + returns_3m) / 2
                momentum_score = 0.5 + np.tanh(avg_momentum * 5) * 0.4 * momentum_consistency
                momentum_score = np.clip(momentum_score, 0, 1)

                scores['long_term_momentum'] = momentum_score
                weights['long_term_momentum'] = 0.1

            # 计算加权平均
            if scores:
                total_weight = sum(weights.values())
                weighted_score = sum(score * weights.get(indicator, 0) for indicator, score in scores.items()) / total_weight
            else:
                weighted_score = 0.5

            return {
                'overall_score': weighted_score,
                'component_scores': scores,
                'component_weights': weights,
                'indicators_used': len(scores)
            }

        except Exception as e:
            self.logger.warning(f"长期评分计算失败: {e}")
            return {
                'overall_score': 0.5,
                'component_scores': {},
                'component_weights': {},
                'indicators_used': 0
            }

    def _calculate_fundamental_score(self, fundamental_data: pd.DataFrame) -> float:
        """计算基本面评分"""

        try:
            score_components = []

            # ROE评分
            if 'roe' in fundamental_data.columns:
                roe = fundamental_data['roe'].iloc[-1]
                if roe > 15:
                    roe_score = 0.8
                elif roe > 10:
                    roe_score = 0.6
                elif roe > 5:
                    roe_score = 0.4
                else:
                    roe_score = 0.2
                score_components.append(roe_score)

            # 营收增长率评分
            if 'revenue_growth' in fundamental_data.columns:
                growth = fundamental_data['revenue_growth'].iloc[-1]
                growth_score = 0.5 + np.tanh(growth / 20) * 0.4
                score_components.append(growth_score)

            # 负债率评分
            if 'debt_ratio' in fundamental_data.columns:
                debt_ratio = fundamental_data['debt_ratio'].iloc[-1]
                if debt_ratio < 0.3:
                    debt_score = 0.8
                elif debt_ratio < 0.6:
                    debt_score = 0.6
                else:
                    debt_score = 0.3
                score_components.append(debt_score)

            return np.mean(score_components) if score_components else 0.5

        except Exception:
            return 0.5

    def _calculate_comprehensive_fundamental_score(self, fundamental_data: pd.DataFrame) -> float:
        """计算综合基本面评分"""

        try:
            scores = []

            # 盈利能力
            profitability_indicators = ['roe', 'roa', 'net_profit_margin']
            profitability_scores = []
            for indicator in profitability_indicators:
                if indicator in fundamental_data.columns:
                    value = fundamental_data[indicator].iloc[-1]
                    if indicator == 'roe':
                        score = min(value / 20, 1.0) if value > 0 else 0
                    elif indicator == 'roa':
                        score = min(value / 10, 1.0) if value > 0 else 0
                    elif indicator == 'net_profit_margin':
                        score = min(value / 15, 1.0) if value > 0 else 0
                    else:
                        score = 0.5
                    profitability_scores.append(score)

            if profitability_scores:
                scores.append(np.mean(profitability_scores))

            # 成长性
            growth_indicators = ['revenue_growth', 'profit_growth']
            growth_scores = []
            for indicator in growth_indicators:
                if indicator in fundamental_data.columns:
                    value = fundamental_data[indicator].iloc[-1]
                    score = 0.5 + np.tanh(value / 30) * 0.4
                    growth_scores.append(score)

            if growth_scores:
                scores.append(np.mean(growth_scores))

            # 财务健康
            if 'debt_ratio' in fundamental_data.columns:
                debt_ratio = fundamental_data['debt_ratio'].iloc[-1]
                debt_score = max(0, 1 - debt_ratio / 0.8)  # 负债率80%以下较好
                scores.append(debt_score)

            return np.mean(scores) if scores else 0.5

        except Exception:
            return 0.5

    def _calculate_adaptive_weights(self,
                                  stock_data: pd.DataFrame,
                                  market_data: Optional[pd.DataFrame],
                                  custom_weights: Optional[Dict[str, float]]) -> Dict[str, float]:
        """计算自适应权重"""

        try:
            # 如果提供了自定义权重，优先使用
            if custom_weights:
                total = sum(custom_weights.values())
                return {k: v/total for k, v in custom_weights.items()}

            # 基础权重
            base_weights = {
                'short_term': 0.3,
                'medium_term': 0.4,
                'long_term': 0.3
            }

            # 根据市场环境调整权重
            if market_data is not None and not market_data.empty:
                market_adjustments = self._analyze_market_for_weights(market_data)

                # 高波动市场：增加短期权重
                if market_adjustments.get('volatility_regime', 1.0) > 1.5:
                    base_weights['short_term'] += 0.1
                    base_weights['medium_term'] -= 0.05
                    base_weights['long_term'] -= 0.05

                # 低波动稳定市场：增加长期权重
                elif market_adjustments.get('volatility_regime', 1.0) < 0.5:
                    base_weights['short_term'] -= 0.1
                    base_weights['medium_term'] += 0.05
                    base_weights['long_term'] += 0.05

            # 根据历史表现调整 (如果有历史数据)
            if hasattr(self, 'performance_history') and self.performance_history:
                performance_adjustments = self._calculate_performance_based_weights()
                for key in base_weights:
                    if key in performance_adjustments:
                        base_weights[key] = base_weights[key] * 0.7 + performance_adjustments[key] * 0.3

            # 确保权重和为1
            total_weight = sum(base_weights.values())
            normalized_weights = {k: v/total_weight for k, v in base_weights.items()}

            return normalized_weights

        except Exception as e:
            self.logger.warning(f"权重计算失败，使用默认权重: {e}")
            return {'short_term': 0.3, 'medium_term': 0.4, 'long_term': 0.3}

    def _analyze_market_for_weights(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """分析市场环境用于权重调整"""

        try:
            # 市场波动率
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                current_vol = returns.rolling(20).std().iloc[-1] if len(returns) >= 20 else returns.std()
                historical_vol = returns.std()
                volatility_regime = current_vol / (historical_vol + 1e-8)
            else:
                volatility_regime = 1.0

            # 市场趋势强度
            if 'close' in market_data.columns and len(market_data) >= 20:
                prices = market_data['close'].iloc[-20:]
                trend_slope = np.polyfit(range(len(prices)), prices, 1)[0]
                price_std = prices.std()
                trend_strength = abs(trend_slope) / (price_std + 1e-8)
            else:
                trend_strength = 0.0

            return {
                'volatility_regime': volatility_regime,
                'trend_strength': trend_strength
            }

        except Exception:
            return {'volatility_regime': 1.0, 'trend_strength': 0.0}

    def _calculate_performance_based_weights(self) -> Dict[str, float]:
        """基于历史表现计算权重"""

        try:
            # 简化实现：基于各时间维度的历史准确性
            # 实际应用中需要维护详细的预测准确性记录

            if not hasattr(self, 'accuracy_history'):
                return {'short_term': 0.3, 'medium_term': 0.4, 'long_term': 0.3}

            # 计算各时间维度的平均准确性
            avg_accuracies = {}
            for timeframe in ['short_term', 'medium_term', 'long_term']:
                if timeframe in self.accuracy_history:
                    avg_accuracies[timeframe] = np.mean(self.accuracy_history[timeframe])
                else:
                    avg_accuracies[timeframe] = 0.5

            # 将准确性转换为权重
            total_accuracy = sum(avg_accuracies.values())
            performance_weights = {k: v/total_accuracy for k, v in avg_accuracies.items()}

            return performance_weights

        except Exception:
            return {'short_term': 0.3, 'medium_term': 0.4, 'long_term': 0.3}

    def _calculate_composite_score(self,
                                 temporal_scores: Dict[str, Dict],
                                 adaptive_weights: Dict[str, float]) -> float:
        """计算综合评分"""

        try:
            weighted_score = 0.0
            total_weight = 0.0

            for timeframe, weight in adaptive_weights.items():
                if timeframe in temporal_scores:
                    score = temporal_scores[timeframe].get('overall_score', 0.5)
                    # 处理NaN值
                    if pd.isna(score):
                        score = 0.5
                    weighted_score += score * weight
                    total_weight += weight

            return weighted_score / total_weight if total_weight > 0 else 0.5

        except Exception:
            return 0.5

    def _assess_scoring_quality(self,
                              temporal_scores: Dict[str, Dict],
                              stock_data: pd.DataFrame) -> Dict[str, Any]:
        """评估评分质量"""

        try:
            quality_metrics = {}

            # 1. 指标覆盖度
            total_indicators = 0
            for timeframe_data in temporal_scores.values():
                total_indicators += timeframe_data.get('indicators_used', 0)

            quality_metrics['indicator_coverage'] = total_indicators / 15  # 假设最多15个指标

            # 2. 评分分散度
            all_scores = []
            for timeframe_data in temporal_scores.values():
                all_scores.append(timeframe_data.get('overall_score', 0.5))

            score_std = np.std(all_scores) if len(all_scores) > 1 else 0
            quality_metrics['score_diversity'] = min(score_std * 4, 1.0)  # 标准化到[0,1]

            # 3. 数据充分度
            data_sufficiency = min(len(stock_data) / 60, 1.0)  # 60天为充分
            quality_metrics['data_sufficiency'] = data_sufficiency

            # 综合质量评分
            overall_quality = (
                quality_metrics['indicator_coverage'] * 0.4 +
                quality_metrics['score_diversity'] * 0.3 +
                quality_metrics['data_sufficiency'] * 0.3
            )
            quality_metrics['overall_quality'] = overall_quality

            return quality_metrics

        except Exception as e:
            self.logger.warning(f"质量评估失败: {e}")
            return {
                'indicator_coverage': 0.5,
                'score_diversity': 0.5,
                'data_sufficiency': 0.5,
                'overall_quality': 0.5
            }

    def _update_scoring_history(self, result: Dict):
        """更新评分历史"""

        history_entry = {
            'timestamp': result['timestamp'],
            'temporal_scores': result['temporal_scores'],
            'adaptive_weights': result['adaptive_weights'],
            'composite_score': result['composite_score'],
            'quality': result['quality_metrics']['overall_quality']
        }

        self.weight_history.append(history_entry)

        # 保持历史记录在合理范围
        if len(self.weight_history) > 1000:
            self.weight_history = self.weight_history[-1000:]

    def get_scoring_summary(self) -> Dict[str, Any]:
        """获取评分系统摘要"""

        return {
            'temporal_windows': {
                'short_term': self.short_window,
                'medium_term': self.medium_window,
                'long_term': self.long_window
            },
            'weight_adaptation_method': self.weight_method,
            'total_scorings': len(self.weight_history),
            'last_scoring': self.weight_history[-1]['timestamp'] if self.weight_history else None,
            'available_strategies': list(self.temporal_configs.keys())
        }