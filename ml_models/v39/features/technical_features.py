"""
v3.9 技术特征提取器

新增15个技术特征：
- 趋势强度类: ADX, Aroon, Ichimoku, SuperTrend
- 动量类: Williams %R, SMI, TSI
- 量价关系: A/D Line, CMF, VWAP deviation, 大单净流入
- 波动率: BB宽度, KC宽度, ATR%, 历史波动率分位数
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging
import sqlite3
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TechnicalFeaturesV39:
    """v3.9技术特征提取器"""

    def __init__(self, db_path: str = "data_adapter/stock_data.db"):
        self.db_path = db_path
        self.feature_names = [
            # 趋势强度类
            'adx_14', 'aroon_up_25', 'aroon_down_25', 'aroon_oscillator_25',
            'ichimoku_conversion', 'ichimoku_base', 'ichimoku_span_a', 'ichimoku_span_b',
            'supertrend_10_3', 'supertrend_signal',

            # 动量类
            'williams_r_14', 'smi_14', 'smi_signal_3', 'tsi_25_13', 'tsi_signal_7',

            # 量价关系
            'ad_line', 'ad_line_change_5', 'cmf_20', 'vwap_deviation', 'large_order_net_inflow',

            # 波动率
            'bb_width_20', 'kc_width_20', 'atr_percent_14', 'historical_volatility_percentile_60'
        ]
        logger.info(f"✅ v3.9技术特征提取器初始化，共{len(self.feature_names)}个特征")

    def _get_stock_data(self, code: str, end_date: str, days: int = 100) -> Optional[pd.DataFrame]:
        """
        从数据库获取股票OHLCV数据

        Args:
            code: 股票代码（如 000001）
            end_date: 结束日期 (YYYY-MM-DD)
            days: 回望天数

        Returns:
            包含OHLCV的DataFrame或None
        """
        try:
            # 规范化股票代码（去掉.SH/.SZ/.BJ后缀）
            normalized_code = code.split('.')[0] if '.' in code else code

            conn = sqlite3.connect(self.db_path)

            # 计算开始日期（往前推更多天数以确保有足够数据）
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            start_dt = end_dt - timedelta(days=days+50)  # 多取50天buffer
            start_date_str = start_dt.strftime('%Y-%m-%d')

            # 查询OHLCV数据
            query = """
                SELECT dq.trade_date, dq.open, dq.high, dq.low, dq.close, dq.volume
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.code = ? AND s.type = 'A股'
                AND dq.trade_date <= ?
                AND dq.trade_date >= ?
                ORDER BY dq.trade_date DESC
                LIMIT ?
            """

            df = pd.read_sql(query, conn, params=(normalized_code, end_date, start_date_str, days))
            conn.close()

            if df.empty:
                return None

            # 按日期升序排列
            df = df.sort_values('trade_date')
            df = df.reset_index(drop=True)

            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            return df

        except Exception as e:
            logger.error(f"获取{code}数据失败: {e}")
            return None

    def extract_features_from_code(self, code: str, date: str) -> Dict[str, float]:
        """
        从股票代码和日期提取技术特征（新接口）

        Args:
            code: 股票代码
            date: 日期 (YYYY-MM-DD)

        Returns:
            特征字典
        """
        # 获取数据
        df = self._get_stock_data(code, date, days=100)

        if df is None or df.empty:
            logger.warning(f"{code} {date}: 无法获取数据")
            return {name: np.nan for name in self.feature_names}

        # 调用原有的extract_features方法
        return self.extract_features(df, code)

    def extract_features(self, df: pd.DataFrame, stock_code: str) -> Dict[str, float]:
        """
        提取技术特征

        Args:
            df: 包含OHLCV数据的DataFrame (必须包含: open, high, low, close, volume)
            stock_code: 股票代码

        Returns:
            特征字典
        """
        try:
            if len(df) < 60:
                logger.warning(f"{stock_code}: 数据不足60条，无法计算技术特征")
                return {name: np.nan for name in self.feature_names}

            features = {}

            # 趋势强度类特征
            features.update(self._calculate_trend_features(df))

            # 动量类特征
            features.update(self._calculate_momentum_features(df))

            # 量价关系特征
            features.update(self._calculate_volume_price_features(df))

            # 波动率特征
            features.update(self._calculate_volatility_features(df))

            return features

        except Exception as e:
            logger.error(f"{stock_code}: 技术特征提取失败 - {e}")
            return {name: np.nan for name in self.feature_names}

    def _calculate_trend_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算趋势强度类特征"""
        features = {}

        try:
            # 1. ADX (Average Directional Index)
            adx_result = self._calculate_adx(df, period=14)
            features['adx_14'] = adx_result['adx']

            # 2. Aroon指标
            aroon_result = self._calculate_aroon(df, period=25)
            features['aroon_up_25'] = aroon_result['aroon_up']
            features['aroon_down_25'] = aroon_result['aroon_down']
            features['aroon_oscillator_25'] = aroon_result['aroon_oscillator']

            # 3. Ichimoku云图
            ichimoku_result = self._calculate_ichimoku(df)
            features['ichimoku_conversion'] = ichimoku_result['conversion']
            features['ichimoku_base'] = ichimoku_result['base']
            features['ichimoku_span_a'] = ichimoku_result['span_a']
            features['ichimoku_span_b'] = ichimoku_result['span_b']

            # 4. SuperTrend
            supertrend_result = self._calculate_supertrend(df, period=10, multiplier=3)
            features['supertrend_10_3'] = supertrend_result['supertrend']
            features['supertrend_signal'] = supertrend_result['signal']

        except Exception as e:
            logger.error(f"趋势特征计算失败: {e}")
            for key in ['adx_14', 'aroon_up_25', 'aroon_down_25', 'aroon_oscillator_25',
                       'ichimoku_conversion', 'ichimoku_base', 'ichimoku_span_a', 'ichimoku_span_b',
                       'supertrend_10_3', 'supertrend_signal']:
                features[key] = np.nan

        return features

    def _calculate_momentum_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算动量类特征"""
        features = {}

        try:
            # 5. Williams %R
            features['williams_r_14'] = self._calculate_williams_r(df, period=14)

            # 6. Stochastic Momentum Index (SMI)
            smi_result = self._calculate_smi(df, period=14, smooth=3)
            features['smi_14'] = smi_result['smi']
            features['smi_signal_3'] = smi_result['signal']

            # 7. True Strength Index (TSI)
            tsi_result = self._calculate_tsi(df, long_period=25, short_period=13, signal_period=7)
            features['tsi_25_13'] = tsi_result['tsi']
            features['tsi_signal_7'] = tsi_result['signal']

        except Exception as e:
            logger.error(f"动量特征计算失败: {e}")
            for key in ['williams_r_14', 'smi_14', 'smi_signal_3', 'tsi_25_13', 'tsi_signal_7']:
                features[key] = np.nan

        return features

    def _calculate_volume_price_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算量价关系特征"""
        features = {}

        try:
            # 8-9. Accumulation/Distribution Line
            ad_line = self._calculate_ad_line(df)
            features['ad_line'] = ad_line.iloc[-1]
            features['ad_line_change_5'] = (ad_line.iloc[-1] - ad_line.iloc[-6]) / abs(ad_line.iloc[-6]) if ad_line.iloc[-6] != 0 else 0

            # 10. Chaikin Money Flow
            features['cmf_20'] = self._calculate_cmf(df, period=20)

            # 11. VWAP Deviation
            features['vwap_deviation'] = self._calculate_vwap_deviation(df)

            # 12. 大单净流入比例 (简化版，使用成交量和价格变动估算)
            features['large_order_net_inflow'] = self._estimate_large_order_flow(df)

        except Exception as e:
            logger.error(f"量价特征计算失败: {e}")
            for key in ['ad_line', 'ad_line_change_5', 'cmf_20', 'vwap_deviation', 'large_order_net_inflow']:
                features[key] = np.nan

        return features

    def _calculate_volatility_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算波动率特征"""
        features = {}

        try:
            # 13. Bollinger Bands宽度
            features['bb_width_20'] = self._calculate_bb_width(df, period=20)

            # 14. Keltner Channel宽度
            features['kc_width_20'] = self._calculate_kc_width(df, period=20)

            # 15. ATR百分比
            features['atr_percent_14'] = self._calculate_atr_percent(df, period=14)

            # 16. 历史波动率分位数
            features['historical_volatility_percentile_60'] = self._calculate_volatility_percentile(df, period=60)

        except Exception as e:
            logger.error(f"波动率特征计算失败: {e}")
            for key in ['bb_width_20', 'kc_width_20', 'atr_percent_14', 'historical_volatility_percentile_60']:
                features[key] = np.nan

        return features

    # ========== 趋势指标计算方法 ==========

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> Dict[str, float]:
        """计算ADX (Average Directional Index)"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # 计算True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        # 计算+DM和-DM
        plus_dm = np.maximum(high[1:] - high[:-1], 0)
        minus_dm = np.maximum(low[:-1] - low[1:], 0)

        # 当+DM > -DM时，-DM = 0；反之亦然
        plus_dm = np.where(plus_dm > minus_dm, plus_dm, 0)
        minus_dm = np.where(minus_dm > plus_dm, minus_dm, 0)

        # 计算平滑的TR, +DM, -DM
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values / atr

        # 计算DX和ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(span=period, adjust=False).mean().iloc[-1]

        return {'adx': adx}

    def _calculate_aroon(self, df: pd.DataFrame, period: int = 25) -> Dict[str, float]:
        """计算Aroon指标"""
        high = df['high'].values
        low = df['low'].values

        # Aroon Up = ((period - periods since period high) / period) * 100
        # Aroon Down = ((period - periods since period low) / period) * 100

        if len(high) < period:
            return {'aroon_up': np.nan, 'aroon_down': np.nan, 'aroon_oscillator': np.nan}

        periods_since_high = period - np.argmax(high[-period:])
        periods_since_low = period - np.argmin(low[-period:])

        aroon_up = ((period - periods_since_high) / period) * 100
        aroon_down = ((period - periods_since_low) / period) * 100
        aroon_oscillator = aroon_up - aroon_down

        return {
            'aroon_up': aroon_up,
            'aroon_down': aroon_down,
            'aroon_oscillator': aroon_oscillator
        }

    def _calculate_ichimoku(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算Ichimoku云图"""
        high = df['high']
        low = df['low']
        close = df['close']

        # 转换线 (Tenkan-sen): (9日最高 + 9日最低) / 2
        conversion_line = (high.rolling(9).max() + low.rolling(9).min()) / 2

        # 基准线 (Kijun-sen): (26日最高 + 26日最低) / 2
        base_line = (high.rolling(26).max() + low.rolling(26).min()) / 2

        # 先行线A (Senkou Span A): (转换线 + 基准线) / 2，向前移26日
        span_a = ((conversion_line + base_line) / 2).shift(26)

        # 先行线B (Senkou Span B): (52日最高 + 52日最低) / 2，向前移26日
        span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

        # 返回当前值相对于各线的位置
        current_price = close.iloc[-1]

        return {
            'conversion': (current_price - conversion_line.iloc[-1]) / current_price if not pd.isna(conversion_line.iloc[-1]) else 0,
            'base': (current_price - base_line.iloc[-1]) / current_price if not pd.isna(base_line.iloc[-1]) else 0,
            'span_a': (current_price - span_a.iloc[-1]) / current_price if not pd.isna(span_a.iloc[-1]) else 0,
            'span_b': (current_price - span_b.iloc[-1]) / current_price if not pd.isna(span_b.iloc[-1]) else 0
        }

    def _calculate_supertrend(self, df: pd.DataFrame, period: int = 10, multiplier: float = 3) -> Dict[str, float]:
        """计算SuperTrend指标"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # 计算ATR
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        atr = pd.Series(tr).rolling(period).mean().values

        # 计算基本上下轨
        hl_avg = (high[1:] + low[1:]) / 2
        upper_band = hl_avg + multiplier * atr
        lower_band = hl_avg - multiplier * atr

        # SuperTrend计算
        supertrend = np.zeros(len(close) - 1)
        trend = np.ones(len(close) - 1)  # 1: 上涨, -1: 下跌

        for i in range(1, len(supertrend)):
            if close[i] > upper_band[i-1]:
                trend[i] = 1
                supertrend[i] = lower_band[i]
            elif close[i] < lower_band[i-1]:
                trend[i] = -1
                supertrend[i] = upper_band[i]
            else:
                trend[i] = trend[i-1]
                if trend[i] == 1:
                    supertrend[i] = max(lower_band[i], supertrend[i-1])
                else:
                    supertrend[i] = min(upper_band[i], supertrend[i-1])

        # 返回当前SuperTrend值相对于价格的位置和信号
        current_price = close[-1]
        st_value = supertrend[-1]

        return {
            'supertrend': (current_price - st_value) / current_price,
            'signal': float(trend[-1])
        }

    # ========== 动量指标计算方法 ==========

    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> float:
        """计算Williams %R"""
        high = df['high']
        low = df['low']
        close = df['close']

        highest_high = high.rolling(period).max()
        lowest_low = low.rolling(period).min()

        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)

        return williams_r.iloc[-1]

    def _calculate_smi(self, df: pd.DataFrame, period: int = 14, smooth: int = 3) -> Dict[str, float]:
        """计算Stochastic Momentum Index"""
        high = df['high']
        low = df['low']
        close = df['close']

        # 计算中点距离
        ll = low.rolling(period).min()
        hh = high.rolling(period).max()
        diff = close - (hh + ll) / 2

        # 双重平滑
        diff_smooth = diff.ewm(span=smooth, adjust=False).mean().ewm(span=smooth, adjust=False).mean()
        range_smooth = (hh - ll).ewm(span=smooth, adjust=False).mean().ewm(span=smooth, adjust=False).mean()

        smi = 100 * diff_smooth / (range_smooth / 2 + 1e-10)
        signal = smi.ewm(span=smooth, adjust=False).mean()

        return {
            'smi': smi.iloc[-1],
            'signal': signal.iloc[-1]
        }

    def _calculate_tsi(self, df: pd.DataFrame, long_period: int = 25,
                       short_period: int = 13, signal_period: int = 7) -> Dict[str, float]:
        """计算True Strength Index"""
        close = df['close']

        # 计算价格变动
        price_change = close.diff()

        # 双重EMA平滑
        double_smoothed = price_change.ewm(span=long_period, adjust=False).mean().ewm(span=short_period, adjust=False).mean()
        double_smoothed_abs = price_change.abs().ewm(span=long_period, adjust=False).mean().ewm(span=short_period, adjust=False).mean()

        tsi = 100 * double_smoothed / (double_smoothed_abs + 1e-10)
        signal = tsi.ewm(span=signal_period, adjust=False).mean()

        return {
            'tsi': tsi.iloc[-1],
            'signal': signal.iloc[-1]
        }

    # ========== 量价关系指标计算方法 ==========

    def _calculate_ad_line(self, df: pd.DataFrame) -> pd.Series:
        """计算Accumulation/Distribution Line"""
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']

        # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
        mfm = ((close - low) - (high - close)) / (high - low + 1e-10)

        # Money Flow Volume = MFM * Volume
        mfv = mfm * volume

        # A/D Line = cumulative sum of MFV
        ad_line = mfv.cumsum()

        return ad_line

    def _calculate_cmf(self, df: pd.DataFrame, period: int = 20) -> float:
        """计算Chaikin Money Flow"""
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']

        # Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low + 1e-10)

        # Money Flow Volume
        mfv = mfm * volume

        # CMF = sum(MFV, period) / sum(Volume, period)
        cmf = mfv.rolling(period).sum() / (volume.rolling(period).sum() + 1e-10)

        return cmf.iloc[-1]

    def _calculate_vwap_deviation(self, df: pd.DataFrame) -> float:
        """计算VWAP偏离度"""
        close = df['close']
        volume = df['volume']

        # VWAP = sum(Price * Volume) / sum(Volume)
        vwap = (close * volume).sum() / (volume.sum() + 1e-10)

        # 当前价格相对VWAP的偏离度
        deviation = (close.iloc[-1] - vwap) / vwap

        return deviation

    def _estimate_large_order_flow(self, df: pd.DataFrame, threshold: float = 1.5) -> float:
        """
        估算大单净流入比例

        使用成交量和价格变动估算：
        - 如果价格上涨且成交量 > 均值*threshold，视为大单流入
        - 如果价格下跌且成交量 > 均值*threshold，视为大单流出
        """
        close = df['close'].values
        volume = df['volume'].values

        # 计算最近10天的情况
        period = min(10, len(close) - 1)
        price_change = close[-period:] - close[-period-1:-1]
        volume_avg = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)

        large_volume_days = volume[-period:] > volume_avg * threshold

        # 计算大单净流入
        inflow = np.sum((price_change > 0) & large_volume_days)
        outflow = np.sum((price_change < 0) & large_volume_days)

        net_inflow_ratio = (inflow - outflow) / period

        return net_inflow_ratio

    # ========== 波动率指标计算方法 ==========

    def _calculate_bb_width(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> float:
        """计算Bollinger Bands宽度"""
        close = df['close']

        sma = close.rolling(period).mean()
        std = close.rolling(period).std()

        upper_band = sma + std_dev * std
        lower_band = sma - std_dev * std

        # BB宽度 = (上轨 - 下轨) / 中轨
        bb_width = (upper_band - lower_band) / (sma + 1e-10)

        return bb_width.iloc[-1]

    def _calculate_kc_width(self, df: pd.DataFrame, period: int = 20, multiplier: float = 2) -> float:
        """计算Keltner Channel宽度"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # 计算EMA
        ema = pd.Series(close).ewm(span=period, adjust=False).mean()

        # 计算ATR
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        atr = pd.Series(tr).rolling(period).mean()

        # KC宽度 = (2 * multiplier * ATR) / EMA
        kc_width = (2 * multiplier * atr.iloc[-1]) / (ema.iloc[-1] + 1e-10)

        return kc_width

    def _calculate_atr_percent(self, df: pd.DataFrame, period: int = 14) -> float:
        """计算ATR百分比"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # 计算True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        # 计算ATR
        atr = pd.Series(tr).rolling(period).mean().iloc[-1]

        # ATR% = ATR / 当前价格
        atr_percent = atr / close[-1]

        return atr_percent

    def _calculate_volatility_percentile(self, df: pd.DataFrame, period: int = 60) -> float:
        """计算历史波动率分位数"""
        close = df['close']

        # 计算20日历史波动率
        returns = close.pct_change()
        volatility = returns.rolling(20).std() * np.sqrt(252)  # 年化

        # 计算当前波动率在过去period天的分位数
        if len(volatility) < period:
            return 0.5

        current_vol = volatility.iloc[-1]
        historical_vols = volatility.iloc[-period:]

        percentile = (historical_vols < current_vol).sum() / period

        return percentile


if __name__ == "__main__":
    # 简单测试
    logging.basicConfig(level=logging.INFO)

    # 创建测试数据
    np.random.seed(42)
    n = 100
    test_df = pd.DataFrame({
        'open': 10 + np.random.randn(n).cumsum() * 0.1,
        'high': 10.5 + np.random.randn(n).cumsum() * 0.1,
        'low': 9.5 + np.random.randn(n).cumsum() * 0.1,
        'close': 10 + np.random.randn(n).cumsum() * 0.1,
        'volume': np.random.randint(1000000, 5000000, n)
    })

    extractor = TechnicalFeaturesV39()
    features = extractor.extract_features(test_df, 'TEST001')

    print("\n=== v3.9 技术特征测试 ===")
    for name, value in features.items():
        print(f"{name}: {value:.4f}" if not np.isnan(value) else f"{name}: NaN")

    print(f"\n总计: {len(features)}个特征")
    print(f"有效特征: {sum(1 for v in features.values() if not np.isnan(v))}个")
