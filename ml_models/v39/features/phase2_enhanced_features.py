#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9 Phase 2增强特征提取器
专注于方向预测能力提升

目标: 提升方向准确率从57.26% → 60%+

新增特征组:
1. 价格动量增强 (5个)
2. 趋势转折检测 (4个)
3. 成交量-价格关系 (3个)
4. 波动率模式 (3个)

总计: 15个新特征
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class DirectionalFeatures:
    """
    V3.9 Phase 2方向预测特征提取器

    基于v3.9.1的特征重要性分析，设计针对方向预测的高价值特征
    """

    def __init__(self):
        self.feature_count = 0
        logger.info("✅ Phase 2方向预测特征提取器初始化")

    def extract_momentum_features(
        self,
        df: pd.DataFrame,
        close_col: str = 'close'
    ) -> pd.DataFrame:
        """
        提取价格动量增强特征 (5个)

        基于momentum_strength的成功（Top 20 #12），扩展动量特征组

        特征:
        1. price_acceleration_5d: 5日价格加速度
        2. momentum_alignment: 多时间框架动量一致性
        3. momentum_decay: 动量衰减率
        4. normalized_momentum: 归一化动量强度
        5. momentum_persistence: 动量持续性
        """
        features = pd.DataFrame(index=df.index)

        try:
            close = df[close_col]
            returns = close.pct_change()

            # 计算多周期动量
            momentum_5d = close / close.shift(5) - 1
            momentum_10d = close / close.shift(10) - 1
            momentum_20d = close / close.shift(20) - 1

            # 1. 价格加速度 (动量变化率)
            features['price_acceleration_5d'] = momentum_5d - momentum_10d
            features['price_acceleration_5d'] = features['price_acceleration_5d'].fillna(0.0).clip(-1, 1)

            # 2. 动量方向一致性 (0-1，越接近1表示多时间框架一致向上)
            features['momentum_alignment'] = (
                (momentum_5d > 0).astype(int) +
                (momentum_10d > 0).astype(int) +
                (momentum_20d > 0).astype(int)
            ) / 3.0

            # 3. 动量衰减率
            features['momentum_decay'] = (
                (momentum_5d - momentum_20d) /
                (abs(momentum_20d) + 1e-8)
            )
            features['momentum_decay'] = features['momentum_decay'].fillna(0.0).clip(-5, 5)

            # 4. 归一化动量强度 (动量/波动率)
            volatility_60d = returns.rolling(window=60, min_periods=30).std()
            features['normalized_momentum'] = (
                momentum_20d / (volatility_60d + 1e-8)
            )
            features['normalized_momentum'] = features['normalized_momentum'].fillna(0.0).clip(-10, 10)

            # 5. 动量持续性 (过去5天中向上动量的占比)
            momentum_direction = (momentum_5d > 0).rolling(window=5, min_periods=3).sum()
            features['momentum_persistence'] = momentum_direction / 5.0
            features['momentum_persistence'] = features['momentum_persistence'].fillna(0.5)

            self.feature_count += 5

        except Exception as e:
            logger.warning(f"动量增强特征提取失败: {e}")
            features['price_acceleration_5d'] = 0.0
            features['momentum_alignment'] = 0.5
            features['momentum_decay'] = 0.0
            features['normalized_momentum'] = 0.0
            features['momentum_persistence'] = 0.5

        return features

    def extract_trend_reversal_features(
        self,
        df: pd.DataFrame,
        technical_dict: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        提取趋势转折检测特征 (4个)

        特征:
        1. macd_price_divergence: MACD-价格背离信号
        2. rsi_reversal_strength: RSI转折强度
        3. adx_change_rate: 趋势强度变化率
        4. channel_position: 价格通道位置
        """
        features = pd.DataFrame(index=df.index)

        try:
            close = df['close']
            high = df.get('high', close)
            low = df.get('low', close)

            # 1. MACD-价格背离
            if technical_dict and 'macd_histogram' in technical_dict:
                macd_hist = technical_dict['macd_histogram']
                features['macd_price_divergence'] = self._detect_divergence(
                    macd_hist, close, window=10
                )
            else:
                features['macd_price_divergence'] = 0.0

            # 2. RSI转折强度
            if technical_dict and 'rsi_14' in technical_dict:
                rsi = technical_dict['rsi_14']
                features['rsi_reversal_strength'] = self._calculate_rsi_reversal(rsi)
            else:
                features['rsi_reversal_strength'] = 0.0

            # 3. ADX变化率
            if technical_dict and 'adx_14' in technical_dict:
                adx = technical_dict['adx_14']
                features['adx_change_rate'] = (
                    (adx - adx.shift(5)) / (adx.shift(5) + 1e-8)
                )
                features['adx_change_rate'] = features['adx_change_rate'].fillna(0.0).clip(-2, 2)
            else:
                features['adx_change_rate'] = 0.0

            # 4. 价格通道位置 (-1到1，-1在下轨，1在上轨)
            upper_channel = high.rolling(window=20, min_periods=10).max()
            lower_channel = low.rolling(window=20, min_periods=10).min()
            channel_width = upper_channel - lower_channel

            features['channel_position'] = (
                (close - lower_channel) / (channel_width + 1e-8) * 2 - 1
            )
            features['channel_position'] = features['channel_position'].fillna(0.0).clip(-1, 1)

            self.feature_count += 4

        except Exception as e:
            logger.warning(f"趋势转折特征提取失败: {e}")
            features['macd_price_divergence'] = 0.0
            features['rsi_reversal_strength'] = 0.0
            features['adx_change_rate'] = 0.0
            features['channel_position'] = 0.0

        return features

    def _detect_divergence(
        self,
        indicator: pd.Series,
        price: pd.Series,
        window: int = 10
    ) -> pd.Series:
        """
        检测指标与价格的背离
        返回: 1 (看涨背离), 0 (无背离), -1 (看跌背离)
        """
        divergence = pd.Series(0.0, index=price.index)

        for i in range(window, len(price)):
            # 看涨背离: 价格创新低，但指标不创新低
            price_window = price.iloc[i-window:i]
            indicator_window = indicator.iloc[i-window:i]

            price_lower_low = price.iloc[i] <= price_window.min()
            indicator_higher_low = indicator.iloc[i] > indicator_window.min()

            if price_lower_low and indicator_higher_low:
                divergence.iloc[i] = 1.0
                continue

            # 看跌背离: 价格创新高，但指标不创新高
            price_higher_high = price.iloc[i] >= price_window.max()
            indicator_lower_high = indicator.iloc[i] < indicator_window.max()

            if price_higher_high and indicator_lower_high:
                divergence.iloc[i] = -1.0

        return divergence

    def _calculate_rsi_reversal(self, rsi: pd.Series) -> pd.Series:
        """
        计算RSI转折强度
        超卖区域返回正值 (0-1)
        超买区域返回负值 (-1-0)
        """
        reversal = pd.Series(0.0, index=rsi.index)

        # 超卖 (RSI < 30)
        oversold = rsi < 30
        reversal[oversold] = (30 - rsi[oversold]) / 30

        # 超买 (RSI > 70)
        overbought = rsi > 70
        reversal[overbought] = -(rsi[overbought] - 70) / 30

        return reversal.clip(-1, 1)

    def extract_volume_price_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        提取成交量-价格关系特征 (3个)

        替换失败的volume_confirmation (importance=16)

        特征:
        1. volume_strength: 成交量强度 (量×涨跌幅)
        2. volume_price_divergence: 量价背离信号
        3. large_order_intensity: 大单资金流强度
        """
        features = pd.DataFrame(index=df.index)

        try:
            close = df['close']
            volume = df.get('volume', pd.Series(0, index=df.index))

            # 1. 成交量强度
            volume_ratio = volume / volume.rolling(window=20, min_periods=10).mean()
            price_change = abs(close.pct_change())
            features['volume_strength'] = (volume_ratio * price_change).fillna(0.0).clip(0, 10)

            # 2. 量价背离 (1: 量价齐升, 0: 正常, -1: 量价背离)
            price_trend_up = (close > close.shift(5))
            volume_trend_up = (volume > volume.shift(5))

            features['volume_price_divergence'] = 0.0
            # 量价背离 (价涨量跌)
            features.loc[price_trend_up & ~volume_trend_up, 'volume_price_divergence'] = -1.0
            # 量价齐升
            features.loc[price_trend_up & volume_trend_up, 'volume_price_divergence'] = 1.0

            # 3. 大单资金流强度 (如果有大单数据)
            if 'large_order_net_inflow' in df.columns:
                features['large_order_intensity'] = (
                    df['large_order_net_inflow'] / (volume + 1e-8)
                )
                features['large_order_intensity'] = features['large_order_intensity'].fillna(0.0).clip(-1, 1)
            else:
                features['large_order_intensity'] = 0.0

            self.feature_count += 3

        except Exception as e:
            logger.warning(f"量价关系特征提取失败: {e}")
            features['volume_strength'] = 0.0
            features['volume_price_divergence'] = 0.0
            features['large_order_intensity'] = 0.0

        return features

    def extract_volatility_pattern_features(
        self,
        df: pd.DataFrame,
        close_col: str = 'close'
    ) -> pd.DataFrame:
        """
        提取波动率模式特征 (3个)

        基于volatility_asymmetry的成功（Top 20 #13），扩展波动率特征组

        特征:
        1. volatility_trend: 波动率趋势 (短期/长期)
        2. volatility_spike: 波动率突变检测
        3. volatility_reversion: 波动率均值回归
        """
        features = pd.DataFrame(index=df.index)

        try:
            close = df[close_col]
            returns = close.pct_change()

            # 计算多周期波动率
            vol_5d = returns.rolling(window=5, min_periods=3).std()
            vol_20d = returns.rolling(window=20, min_periods=10).std()
            vol_60d = returns.rolling(window=60, min_periods=30).std()

            # 1. 波动率趋势 (短期/长期，>1表示波动率上升)
            features['volatility_trend'] = (vol_20d / (vol_60d + 1e-8)).fillna(1.0).clip(0.1, 5)

            # 2. 波动率突变 (短期波动率>长期波动率*1.5)
            features['volatility_spike'] = (vol_5d > vol_20d * 1.5).astype(float)

            # 3. 波动率均值回归 (当前偏离长期均值的程度)
            features['volatility_reversion'] = (
                (vol_20d - vol_60d) / (vol_60d + 1e-8)
            )
            features['volatility_reversion'] = features['volatility_reversion'].fillna(0.0).clip(-2, 2)

            self.feature_count += 3

        except Exception as e:
            logger.warning(f"波动率模式特征提取失败: {e}")
            features['volatility_trend'] = 1.0
            features['volatility_spike'] = 0.0
            features['volatility_reversion'] = 0.0

        return features

    def extract_all_features(
        self,
        stock_df: pd.DataFrame,
        market_df: Optional[pd.DataFrame] = None,
        technical_dict: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        提取所有Phase 2增强特征（一站式接口）

        Args:
            stock_df: 股票数据 (必须包含close, 最好有high/low/volume)
            market_df: 市场数据 (可选)
            technical_dict: 技术指标字典 (可选，包含macd_histogram, rsi_14, adx_14等)

        Returns:
            包含所有15个Phase 2特征的DataFrame
        """
        all_features = pd.DataFrame(index=stock_df.index)

        # 1. 价格动量增强 (5个)
        momentum_feat = self.extract_momentum_features(stock_df)
        all_features = pd.concat([all_features, momentum_feat], axis=1)

        # 2. 趋势转折检测 (4个)
        reversal_feat = self.extract_trend_reversal_features(stock_df, technical_dict)
        all_features = pd.concat([all_features, reversal_feat], axis=1)

        # 3. 成交量-价格关系 (3个)
        volume_feat = self.extract_volume_price_features(stock_df)
        all_features = pd.concat([all_features, volume_feat], axis=1)

        # 4. 波动率模式 (3个)
        volatility_feat = self.extract_volatility_pattern_features(stock_df)
        all_features = pd.concat([all_features, volatility_feat], axis=1)

        logger.info(f"✅ 提取了{self.feature_count}个Phase 2特征")

        return all_features


# 快速测试
if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100)
    test_df = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(100) * 2),
        'high': 100 + np.cumsum(np.random.randn(100) * 2) + np.random.rand(100),
        'low': 100 + np.cumsum(np.random.randn(100) * 2) - np.random.rand(100),
        'volume': 1000000 + np.random.randint(-100000, 100000, 100)
    }, index=dates)

    # 测试技术指标字典
    test_df['macd'] = test_df['close'].ewm(span=12).mean() - test_df['close'].ewm(span=26).mean()
    test_df['macd_signal'] = test_df['macd'].ewm(span=9).mean()

    technical_dict = {
        'macd_histogram': test_df['macd'] - test_df['macd_signal'],
        'rsi_14': pd.Series(50 + np.random.randn(100) * 20, index=dates).clip(0, 100),
        'adx_14': pd.Series(25 + np.random.randn(100) * 10, index=dates).clip(0, 100)
    }

    # 测试特征提取
    extractor = DirectionalFeatures()

    print("=" * 80)
    print("测试Phase 2方向预测特征提取器")
    print("=" * 80)

    # 测试各组特征
    momentum_feat = extractor.extract_momentum_features(test_df)
    print(f"\n1. 动量增强特征 ({len(momentum_feat.columns)}个):")
    print(momentum_feat.columns.tolist())
    print(momentum_feat.tail())

    reversal_feat = extractor.extract_trend_reversal_features(test_df, technical_dict)
    print(f"\n2. 趋势转折特征 ({len(reversal_feat.columns)}个):")
    print(reversal_feat.columns.tolist())
    print(reversal_feat.tail())

    volume_feat = extractor.extract_volume_price_features(test_df)
    print(f"\n3. 量价关系特征 ({len(volume_feat.columns)}个):")
    print(volume_feat.columns.tolist())
    print(volume_feat.tail())

    volatility_feat = extractor.extract_volatility_pattern_features(test_df)
    print(f"\n4. 波动率模式特征 ({len(volatility_feat.columns)}个):")
    print(volatility_feat.columns.tolist())
    print(volatility_feat.tail())

    # 测试一站式接口
    extractor2 = DirectionalFeatures()
    all_feat = extractor2.extract_all_features(test_df, technical_dict=technical_dict)
    print(f"\n所有Phase 2特征 ({len(all_feat.columns)}个):")
    print(all_feat.columns.tolist())
    print(all_feat.tail())

    print("\n✅ 测试完成！")
