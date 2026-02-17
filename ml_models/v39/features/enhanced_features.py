#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9.1增强特征提取器
新增10个高价值特征以提升方向预测能力
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class EnhancedFeatures:
    """
    V3.9.1增强特征提取器

    目标: 提升方向准确率从55.68% → 60%+

    新增特征组:
    1. 多时间框架动量 (3个)
    2. 相对强度 (2个)
    3. 趋势一致性 (2个)
    4. 非线性和交互 (3个)
    """

    def __init__(self):
        self.feature_count = 0
        logger.info("✅ v3.9.1增强特征提取器初始化")

    def extract_momentum_features(
        self,
        df: pd.DataFrame,
        close_col: str = 'close'
    ) -> pd.DataFrame:
        """
        提取多时间框架动量特征

        特征:
        1. momentum_5d: 5日动量
        2. momentum_20d: 20日动量
        3. momentum_strength: 动量强度 (动量/波动率)

        Args:
            df: 包含价格数据的DataFrame
            close_col: 收盘价列名

        Returns:
            包含3个动量特征的DataFrame
        """
        features = pd.DataFrame(index=df.index)

        try:
            close = df[close_col]

            # 1. 基础动量
            features['momentum_5d'] = (close / close.shift(5) - 1)
            features['momentum_20d'] = (close / close.shift(20) - 1)

            # 2. 动量强度 (标准化)
            # 计算5日波动率
            returns = close.pct_change()
            volatility_5d = returns.rolling(window=5, min_periods=3).std()

            # 动量强度 = |动量| / 波动率
            features['momentum_strength'] = (
                features['momentum_5d'].abs() /
                (volatility_5d + 1e-8)  # 避免除零
            )

            # 限制极端值
            features['momentum_strength'] = features['momentum_strength'].clip(0, 10)

            self.feature_count += 3

        except Exception as e:
            logger.warning(f"动量特征提取失败: {e}")
            features['momentum_5d'] = 0.0
            features['momentum_20d'] = 0.0
            features['momentum_strength'] = 1.0

        return features

    def extract_relative_strength_features(
        self,
        stock_df: pd.DataFrame,
        market_df: pd.DataFrame,
        industry_df: Optional[pd.DataFrame] = None,
        period: int = 20
    ) -> pd.DataFrame:
        """
        提取相对强度特征

        特征:
        1. relative_strength_to_market: 相对市场强度
        2. relative_strength_to_industry: 相对行业强度

        Args:
            stock_df: 股票数据
            market_df: 市场指数数据
            industry_df: 行业指数数据（可选）
            period: 计算周期

        Returns:
            包含2个相对强度特征的DataFrame
        """
        features = pd.DataFrame(index=stock_df.index)

        try:
            # 计算股票收益率
            stock_ret = stock_df['close'] / stock_df['close'].shift(period) - 1

            # 1. 相对市场强度
            if market_df is not None and len(market_df) > 0:
                market_ret = market_df['close'] / market_df['close'].shift(period) - 1
                # 对齐索引
                market_ret = market_ret.reindex(stock_ret.index, method='ffill')
                features['relative_strength_to_market'] = (
                    stock_ret / (market_ret + 1e-8)
                )
                # 限制极端值
                features['relative_strength_to_market'] = (
                    features['relative_strength_to_market'].clip(-5, 5)
                )
            else:
                features['relative_strength_to_market'] = 1.0

            # 2. 相对行业强度
            if industry_df is not None and len(industry_df) > 0:
                industry_ret = industry_df['close'] / industry_df['close'].shift(period) - 1
                industry_ret = industry_ret.reindex(stock_ret.index, method='ffill')
                features['relative_strength_to_industry'] = (
                    stock_ret / (industry_ret + 1e-8)
                )
                features['relative_strength_to_industry'] = (
                    features['relative_strength_to_industry'].clip(-5, 5)
                )
            else:
                # 如果没有行业数据，使用市场数据作为替代
                features['relative_strength_to_industry'] = (
                    features['relative_strength_to_market']
                )

            self.feature_count += 2

        except Exception as e:
            logger.warning(f"相对强度特征提取失败: {e}")
            features['relative_strength_to_market'] = 1.0
            features['relative_strength_to_industry'] = 1.0

        return features

    def extract_trend_consistency_features(
        self,
        df: pd.DataFrame,
        ma_5: Optional[pd.Series] = None,
        ma_10: Optional[pd.Series] = None,
        ma_20: Optional[pd.Series] = None,
        ma_60: Optional[pd.Series] = None,
        volume: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        提取趋势一致性特征

        特征:
        1. ma_alignment_score: 均线排列一致性评分
        2. volume_confirmation: 成交量确认信号

        Args:
            df: 包含价格数据的DataFrame
            ma_5, ma_10, ma_20, ma_60: 各周期均线（可选）
            volume: 成交量数据（可选）

        Returns:
            包含2个趋势一致性特征的DataFrame
        """
        features = pd.DataFrame(index=df.index)

        try:
            close = df['close']

            # 如果没有提供均线，计算它们
            if ma_5 is None:
                ma_5 = close.rolling(window=5, min_periods=3).mean()
            if ma_10 is None:
                ma_10 = close.rolling(window=10, min_periods=5).mean()
            if ma_20 is None:
                ma_20 = close.rolling(window=20, min_periods=10).mean()
            if ma_60 is None:
                ma_60 = close.rolling(window=60, min_periods=30).mean()

            # 1. 均线排列一致性评分 (0-1)
            score_components = []

            # 多头排列: ma5 > ma10 > ma20 > ma60
            score_components.append((ma_5 > ma_10).astype(int))
            score_components.append((ma_10 > ma_20).astype(int))
            score_components.append((ma_20 > ma_60).astype(int))
            # 价格在均线之上
            score_components.append((close > ma_5).astype(int))
            # 短期均线向上
            ma_5_slope = (ma_5 - ma_5.shift(5)) / (ma_5.shift(5) + 1e-8)
            score_components.append((ma_5_slope > 0).astype(int))

            # 计算平均分
            features['ma_alignment_score'] = (
                sum(score_components) / len(score_components)
            )

            # 2. 成交量确认
            if volume is not None:
                # 价格趋势
                price_trend_up = (close > close.shift(5))

                # 成交量趋势
                volume_ma5 = volume.rolling(window=5, min_periods=3).mean()
                volume_ma20 = volume.rolling(window=20, min_periods=10).mean()
                volume_trend_up = (volume_ma5 > volume_ma20)

                # 确认信号: 价格上涨 + 成交量放大
                features['volume_confirmation'] = (
                    (price_trend_up & volume_trend_up).astype(float)
                )
            else:
                features['volume_confirmation'] = 0.5

            self.feature_count += 2

        except Exception as e:
            logger.warning(f"趋势一致性特征提取失败: {e}")
            features['ma_alignment_score'] = 0.5
            features['volume_confirmation'] = 0.5

        return features

    def extract_nonlinear_and_interaction_features(
        self,
        df: pd.DataFrame,
        technical_features: Optional[Dict] = None,
        fundamental_features: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        提取非线性和特征交互项

        特征:
        1. volatility_asymmetry: 波动率非对称性
        2. price_ma_ratio_squared: 价格均线比率平方
        3. roe_momentum_interaction: ROE×动量交互

        Args:
            df: 价格数据
            technical_features: 技术特征字典
            fundamental_features: 基本面特征字典

        Returns:
            包含3个非线性/交互特征的DataFrame
        """
        features = pd.DataFrame(index=df.index)

        try:
            close = df['close']
            returns = close.pct_change()

            # 1. 波动率非对称性
            # 上行波动率
            upside_returns = returns[returns > 0]
            upside_vol = returns.rolling(window=60).apply(
                lambda x: x[x > 0].std() if len(x[x > 0]) > 0 else np.nan
            )

            # 下行波动率
            downside_returns = returns[returns < 0]
            downside_vol = returns.rolling(window=60).apply(
                lambda x: x[x < 0].std() if len(x[x < 0]) > 0 else np.nan
            )

            # 非对称性 (下行波动 / 上行波动)
            # 值>1表示下跌时波动更大(风险更高)
            features['volatility_asymmetry'] = (
                downside_vol / (upside_vol + 1e-8)
            )
            features['volatility_asymmetry'] = (
                features['volatility_asymmetry'].fillna(1.0).clip(0.1, 10)
            )

            # 2. 价格均线比率平方 (非线性关系)
            ma_20 = close.rolling(window=20, min_periods=10).mean()
            price_ma_ratio = (close - ma_20) / (ma_20 + 1e-8)
            features['price_ma_ratio_squared'] = price_ma_ratio ** 2
            features['price_ma_ratio_squared'] = (
                features['price_ma_ratio_squared'].clip(0, 1)
            )

            # 3. ROE×动量交互 (基本面×技术面)
            if (fundamental_features is not None and
                'roe_change_rate' in fundamental_features):
                roe_change = fundamental_features['roe_change_rate']
                momentum_20d = close / close.shift(20) - 1

                features['roe_momentum_interaction'] = (
                    roe_change * momentum_20d
                )
                features['roe_momentum_interaction'] = (
                    features['roe_momentum_interaction'].fillna(0.0).clip(-5, 5)
                )
            else:
                features['roe_momentum_interaction'] = 0.0

            self.feature_count += 3

        except Exception as e:
            logger.warning(f"非线性/交互特征提取失败: {e}")
            features['volatility_asymmetry'] = 1.0
            features['price_ma_ratio_squared'] = 0.0
            features['roe_momentum_interaction'] = 0.0

        return features

    def extract_all_enhanced_features(
        self,
        stock_df: pd.DataFrame,
        market_df: Optional[pd.DataFrame] = None,
        industry_df: Optional[pd.DataFrame] = None,
        technical_features: Optional[Dict] = None,
        fundamental_features: Optional[Dict] = None,
        ma_dict: Optional[Dict[str, pd.Series]] = None
    ) -> pd.DataFrame:
        """
        提取所有增强特征（一站式接口）

        Args:
            stock_df: 股票数据
            market_df: 市场数据
            industry_df: 行业数据
            technical_features: 已有技术特征
            fundamental_features: 已有基本面特征
            ma_dict: 均线字典 {'ma_5': Series, 'ma_10': Series, ...}

        Returns:
            包含所有10个增强特征的DataFrame
        """
        all_features = pd.DataFrame(index=stock_df.index)

        # 1. 动量特征 (3个)
        momentum_feat = self.extract_momentum_features(stock_df)
        all_features = pd.concat([all_features, momentum_feat], axis=1)

        # 2. 相对强度特征 (2个)
        rs_feat = self.extract_relative_strength_features(
            stock_df, market_df, industry_df
        )
        all_features = pd.concat([all_features, rs_feat], axis=1)

        # 3. 趋势一致性特征 (2个)
        ma_5 = ma_dict.get('ma_5') if ma_dict else None
        ma_10 = ma_dict.get('ma_10') if ma_dict else None
        ma_20 = ma_dict.get('ma_20') if ma_dict else None
        ma_60 = ma_dict.get('ma_60') if ma_dict else None
        volume = stock_df.get('volume')

        trend_feat = self.extract_trend_consistency_features(
            stock_df, ma_5, ma_10, ma_20, ma_60, volume
        )
        all_features = pd.concat([all_features, trend_feat], axis=1)

        # 4. 非线性和交互特征 (3个)
        nonlinear_feat = self.extract_nonlinear_and_interaction_features(
            stock_df, technical_features, fundamental_features
        )
        all_features = pd.concat([all_features, nonlinear_feat], axis=1)

        logger.info(f"✅ 提取了{self.feature_count}个增强特征")

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
        'volume': 1000000 + np.random.randint(-100000, 100000, 100)
    }, index=dates)

    market_df = pd.DataFrame({
        'close': 3000 + np.cumsum(np.random.randn(100) * 10)
    }, index=dates)

    # 测试特征提取
    extractor = EnhancedFeatures()

    print("="*80)
    print("测试增强特征提取器")
    print("="*80)

    # 测试各组特征
    momentum_feat = extractor.extract_momentum_features(test_df)
    print(f"\n1. 动量特征 ({len(momentum_feat.columns)}个):")
    print(momentum_feat.tail())

    rs_feat = extractor.extract_relative_strength_features(test_df, market_df)
    print(f"\n2. 相对强度特征 ({len(rs_feat.columns)}个):")
    print(rs_feat.tail())

    trend_feat = extractor.extract_trend_consistency_features(
        test_df, volume=test_df['volume']
    )
    print(f"\n3. 趋势一致性特征 ({len(trend_feat.columns)}个):")
    print(trend_feat.tail())

    nonlinear_feat = extractor.extract_nonlinear_and_interaction_features(test_df)
    print(f"\n4. 非线性/交互特征 ({len(nonlinear_feat.columns)}个):")
    print(nonlinear_feat.tail())

    # 测试一站式接口
    extractor2 = EnhancedFeatures()
    all_feat = extractor2.extract_all_enhanced_features(test_df, market_df)
    print(f"\n所有增强特征 ({len(all_feat.columns)}个):")
    print(all_feat.tail())
    print(f"\n特征名: {all_feat.columns.tolist()}")

    print("\n✅ 测试完成！")
