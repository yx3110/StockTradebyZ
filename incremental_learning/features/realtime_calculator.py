#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.8实时特征计算器
负责计算盘中实时特征和动态指标
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any

# 添加项目路径
sys.path.append('/Users/yangxu/StockTradebyZ')
from data_adapter.database_manager import DatabaseManager
from .realtime_data_fetcher import RealtimeDataFetcher
from .sentiment_indicators import SentimentIndicatorCalculator

class RealtimeFeatureCalculator:
    """
    实时特征计算器

    核心功能：
    1. 计算盘中动量特征
    2. 计算开盘相关特征
    3. 计算相对强度特征
    4. 计算波动率特征
    """

    def __init__(self, cache_ttl: int, db_manager: DatabaseManager, logger: logging.Logger):
        self.cache_ttl = cache_ttl
        self.db_manager = db_manager
        self.logger = logger

        # 初始化数据获取器
        self.data_fetcher = RealtimeDataFetcher(db_manager, cache_ttl)

        # 初始化情绪指标计算器 (Phase 2.2新增)
        self.sentiment_calculator = SentimentIndicatorCalculator(db_manager, cache_ttl)

        # 特征缓存
        self.feature_cache = {}
        self.cache_timestamps = {}

    def compute_intraday_features(self, code: str, current_time: datetime = None) -> Dict[str, float]:
        """
        计算盘中实时特征

        Args:
            code: 股票代码
            current_time: 当前时间

        Returns:
            Dict: 实时特征字典
        """
        if current_time is None:
            current_time = datetime.now()

        cache_key = f"{code}_{current_time.strftime('%Y%m%d_%H%M')}"

        # 检查缓存
        if self._is_cache_valid(cache_key):
            self.logger.info(f"📊 使用缓存的实时特征: {code}")
            return self.feature_cache[cache_key]

        try:
            # 获取今日开盘时间
            market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)

            # 获取分钟级数据
            minute_data = self.data_fetcher.get_minute_data(code, market_open, current_time)

            if minute_data.empty:
                self.logger.warning(f"⚠️ {code}无分钟数据，返回默认特征")
                return self._get_default_features()

            # 计算各类特征
            features = {}

            # 1. 动量特征
            features.update(self._compute_momentum_features(minute_data))

            # 2. 开盘相关特征
            features.update(self._compute_opening_features(minute_data, code))

            # 3. 成交量特征
            features.update(self._compute_volume_features(minute_data))

            # 4. 波动率特征
            features.update(self._compute_volatility_features(minute_data))

            # 5. 相对强度特征
            features.update(self._compute_relative_strength_features(code, current_time))

            # 6. 市场相关性特征
            features.update(self._compute_market_correlation_features(code, current_time))

            # 7. 情绪指标增强 (Phase 2.2新增)
            sentiment_features = self.sentiment_calculator.compute_sentiment_indicators(code, current_time)
            features.update(sentiment_features)

            # 更新缓存
            self.feature_cache[cache_key] = features
            self.cache_timestamps[cache_key] = current_time

            self.logger.info(f"📊 计算{code}实时特征完成: {len(features)}个特征")
            return features

        except Exception as e:
            self.logger.error(f"❌ 计算{code}实时特征失败: {e}")
            return self._get_default_features()

    def _compute_momentum_features(self, minute_data: pd.DataFrame) -> Dict[str, float]:
        """计算动量特征"""
        features = {}

        if len(minute_data) < 30:  # 至少需要30分钟数据
            return {
                'intraday_momentum_5m': 0.0,
                'intraday_momentum_15m': 0.0,
                'intraday_momentum_30m': 0.0
            }

        prices = minute_data['price'].values

        try:
            # 5分钟动量
            if len(prices) >= 5:
                momentum_5m = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] != 0 else 0
                features['intraday_momentum_5m'] = momentum_5m
            else:
                features['intraday_momentum_5m'] = 0.0

            # 15分钟动量
            if len(prices) >= 15:
                momentum_15m = (prices[-1] - prices[-15]) / prices[-15] if prices[-15] != 0 else 0
                features['intraday_momentum_15m'] = momentum_15m
            else:
                features['intraday_momentum_15m'] = 0.0

            # 30分钟动量
            if len(prices) >= 30:
                momentum_30m = (prices[-1] - prices[-30]) / prices[-30] if prices[-30] != 0 else 0
                features['intraday_momentum_30m'] = momentum_30m
            else:
                features['intraday_momentum_30m'] = 0.0

        except Exception as e:
            self.logger.error(f"计算动量特征失败: {e}")
            features = {
                'intraday_momentum_5m': 0.0,
                'intraday_momentum_15m': 0.0,
                'intraday_momentum_30m': 0.0
            }

        return features

    def _compute_opening_features(self, minute_data: pd.DataFrame, code: str = None) -> Dict[str, float]:
        """计算开盘相关特征"""
        features = {}

        if minute_data.empty:
            return {
                'opening_gap': 0.0,
                'opening_volume_surge': 0.0,
                'early_session_perf': 0.0
            }

        try:
            # 获取昨日收盘价 - 需要传入code参数
            trade_date = minute_data.iloc[0].get('datetime', datetime.now()).date() if not minute_data.empty else datetime.now().date()
            yesterday_close = self._get_previous_close(code.split('.')[0] if '.' in code else code)
            if yesterday_close is None:
                # 如果获取不到昨日收盘价，使用开盘价作为估算
                yesterday_close = minute_data['price'].iloc[0] * 0.98

            # 开盘缺口
            opening_price = minute_data['price'].iloc[0]
            opening_gap = (opening_price - yesterday_close) / yesterday_close if yesterday_close != 0 else 0
            features['opening_gap'] = opening_gap

            # 开盘成交量激增
            if len(minute_data) >= 10:
                opening_volume_avg = minute_data['volume'].iloc[:10].mean()
                normal_volume_avg = minute_data['volume'].iloc[10:].mean() if len(minute_data) > 10 else opening_volume_avg
                volume_surge = (opening_volume_avg / normal_volume_avg - 1) if normal_volume_avg != 0 else 0
                features['opening_volume_surge'] = volume_surge
            else:
                features['opening_volume_surge'] = 0.0

            # 早盘表现 (9:30-10:30)
            if len(minute_data) >= 60:
                early_session_data = minute_data.iloc[:60]
                early_perf = (early_session_data['price'].iloc[-1] - early_session_data['price'].iloc[0]) / early_session_data['price'].iloc[0]
                features['early_session_perf'] = early_perf
            else:
                features['early_session_perf'] = 0.0

        except Exception as e:
            self.logger.error(f"计算开盘特征失败: {e}")
            features = {
                'opening_gap': 0.0,
                'opening_volume_surge': 0.0,
                'early_session_perf': 0.0
            }

        return features

    def _compute_volume_features(self, minute_data: pd.DataFrame) -> Dict[str, float]:
        """计算成交量特征"""
        features = {}

        if minute_data.empty:
            return {'volume_intensity': 0.0, 'volume_consistency': 0.0}

        try:
            volumes = minute_data['volume'].values

            # 成交量强度 (相对于平均值)
            volume_mean = np.mean(volumes)
            current_volume = volumes[-1] if len(volumes) > 0 else 0
            volume_intensity = (current_volume / volume_mean - 1) if volume_mean != 0 else 0
            features['volume_intensity'] = volume_intensity

            # 成交量一致性 (变异系数的倒数)
            if len(volumes) > 1:
                volume_cv = np.std(volumes) / volume_mean if volume_mean != 0 else 1
                volume_consistency = 1 / (1 + volume_cv)  # 标准化到[0,1]
                features['volume_consistency'] = volume_consistency
            else:
                features['volume_consistency'] = 0.5

        except Exception as e:
            self.logger.error(f"计算成交量特征失败: {e}")
            features = {'volume_intensity': 0.0, 'volume_consistency': 0.0}

        return features

    def _compute_volatility_features(self, minute_data: pd.DataFrame) -> Dict[str, float]:
        """计算波动率特征"""
        features = {}

        if len(minute_data) < 2:
            return {'volatility_intraday': 0.0, 'price_efficiency': 0.0}

        try:
            prices = minute_data['price'].values

            # 盘中波动率
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) if len(returns) > 0 else 0
            features['volatility_intraday'] = volatility

            # 价格发现效率 (实际价格变化 vs 随机游走期望)
            total_return = abs((prices[-1] - prices[0]) / prices[0]) if prices[0] != 0 else 0
            expected_random_return = volatility * np.sqrt(len(returns))
            price_efficiency = total_return / expected_random_return if expected_random_return != 0 else 0
            features['price_efficiency'] = price_efficiency

        except Exception as e:
            self.logger.error(f"计算波动率特征失败: {e}")
            features = {'volatility_intraday': 0.0, 'price_efficiency': 0.0}

        return features

    def _compute_relative_strength_features(self, code: str, current_time: datetime) -> Dict[str, float]:
        """计算相对强度特征"""
        features = {}

        try:
            # 获取行业数据
            # TODO: Phase 2中实现真实的行业识别和数据获取
            sector = f"sector_{hash(code) % 10}"  # 模拟行业
            sector_data = self.data_fetcher.get_sector_data(sector)

            if sector_data:
                features['relative_sector_strength'] = sector_data.get('relative_strength', 1.0)
            else:
                features['relative_sector_strength'] = 1.0

        except Exception as e:
            self.logger.error(f"计算相对强度特征失败: {e}")
            features['relative_sector_strength'] = 1.0

        return features

    def _compute_market_correlation_features(self, code: str, current_time: datetime) -> Dict[str, float]:
        """计算市场相关性特征"""
        features = {}

        try:
            # 获取市场数据
            market_data = self.data_fetcher.get_market_snapshot()

            if market_data:
                # 模拟与大盘的相关性
                market_sentiment = market_data.get('market_sentiment', 0.5)
                # 基于股票代码计算模拟相关性
                base_correlation = 0.3 + (hash(code) % 100) / 200  # [0.3, 0.8]
                market_correlation = base_correlation * market_sentiment + (1 - base_correlation) * (1 - market_sentiment)
                features['market_correlation'] = market_correlation
            else:
                features['market_correlation'] = 0.5

        except Exception as e:
            self.logger.error(f"计算市场相关性特征失败: {e}")
            features['market_correlation'] = 0.5

        return features

    def _get_default_features(self) -> Dict[str, float]:
        """获取默认特征值"""
        return {
            # Phase 2.1 实时特征 (12个)
            'intraday_momentum_5m': 0.0,
            'intraday_momentum_15m': 0.0,
            'intraday_momentum_30m': 0.0,
            'opening_gap': 0.0,
            'opening_volume_surge': 0.0,
            'early_session_perf': 0.0,
            'volume_intensity': 0.0,
            'volume_consistency': 0.5,
            'volatility_intraday': 0.0,
            'price_efficiency': 0.0,
            'relative_sector_strength': 1.0,
            'market_correlation': 0.5,

            # Phase 2.2 情绪指标增强 (4个)
            'capital_flow_indicator': 0.0,
            'market_sentiment_index': 0.5,
            'sector_rotation_strength': 0.5,
            'northbound_capital_impact': 0.5
        }

    def _is_cache_valid(self, cache_key: str) -> bool:
        """检查缓存是否有效"""
        if cache_key not in self.feature_cache:
            return False

        if cache_key not in self.cache_timestamps:
            return False

        elapsed = (datetime.now() - self.cache_timestamps[cache_key]).total_seconds()
        return elapsed < self.cache_ttl

    def _get_previous_close(self, code: str) -> Optional[float]:
        """获取昨日收盘价"""
        try:
            # 标准化股票代码
            clean_code = code.split('.')[0] if '.' in code else code

            # 获取当前日期的前一个交易日
            current_date = datetime.now().date()

            # 查询最近的收盘价（排除今日）
            query = """
            SELECT close
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.code = ? AND dq.trade_date < ?
            ORDER BY dq.trade_date DESC
            LIMIT 1
            """

            with self.db_manager.get_connection() as conn:
                result = conn.execute(query, (clean_code, current_date)).fetchone()
                return float(result[0]) if result else None

        except Exception as e:
            self.logger.warning(f"获取{code}昨日收盘价失败: {e}")
            return None

    def clear_cache(self):
        """清理缓存"""
        self.feature_cache.clear()
        self.cache_timestamps.clear()
        self.data_fetcher.clear_cache()
        self.sentiment_calculator.clear_cache()  # Phase 2.2新增
        self.logger.info("🗑️ 清理实时特征缓存")