#!/usr/bin/env python3
"""
挤压动量指标计算器 (Squeeze Momentum Indicator)
基于John Carter的TTM Squeeze指标实现

核心功能：
1. 布林带(Bollinger Bands)计算
2. 肯特纳通道(Keltner Channels)计算
3. 挤压状态识别
4. 动量计算
5. 挤压释放信号
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

class SqueezeMomentumCalculator:
    """挤压动量指标计算器"""
    
    def __init__(self, 
                 bb_length: int = 20,
                 bb_multiplier: float = 2.0,
                 kc_length: int = 20,
                 kc_multiplier: float = 1.5,
                 momentum_length: int = 20):
        """
        初始化挤压动量计算器
        
        Args:
            bb_length: 布林带计算周期
            bb_multiplier: 布林带标准差倍数
            kc_length: 肯特纳通道计算周期
            kc_multiplier: 肯特纳通道ATR倍数
            momentum_length: 动量计算周期
        """
        self.bb_length = bb_length
        self.bb_multiplier = bb_multiplier
        self.kc_length = kc_length
        self.kc_multiplier = kc_multiplier
        self.momentum_length = momentum_length
        
        self.logger = logging.getLogger(__name__)
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """计算平均真实范围(ATR)"""
        # 真实范围计算
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        # 取最大值作为真实范围
        true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        # 计算ATR (简单移动平均)
        atr = true_range.rolling(window=period, min_periods=1).mean()
        
        return atr
    
    def calculate_bollinger_bands(self, close: pd.Series) -> Dict[str, pd.Series]:
        """计算布林带"""
        # 中轨：简单移动平均
        middle = close.rolling(window=self.bb_length, min_periods=1).mean()
        
        # 标准差
        std = close.rolling(window=self.bb_length, min_periods=1).std()
        
        # 上轨和下轨
        upper = middle + (std * self.bb_multiplier)
        lower = middle - (std * self.bb_multiplier)
        
        return {
            'bb_upper': upper,
            'bb_middle': middle,
            'bb_lower': lower,
            'bb_width': upper - lower
        }
    
    def calculate_keltner_channels(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """计算肯特纳通道"""
        # 中轨：简单移动平均
        middle = close.rolling(window=self.kc_length, min_periods=1).mean()
        
        # ATR计算
        atr = self.calculate_atr(high, low, close, self.kc_length)
        
        # 上轨和下轨
        upper = middle + (atr * self.kc_multiplier)
        lower = middle - (atr * self.kc_multiplier)
        
        return {
            'kc_upper': upper,
            'kc_middle': middle,
            'kc_lower': lower,
            'kc_width': upper - lower
        }
    
    def calculate_linear_regression(self, series: pd.Series, period: int) -> pd.Series:
        """计算线性回归斜率"""
        def lr_slope(y_vals):
            if len(y_vals) < 2:
                return 0
            x_vals = np.arange(len(y_vals))
            # 简单线性回归斜率计算
            n = len(y_vals)
            sum_x = np.sum(x_vals)
            sum_y = np.sum(y_vals)
            sum_xy = np.sum(x_vals * y_vals)
            sum_x2 = np.sum(x_vals * x_vals)
            
            denominator = n * sum_x2 - sum_x * sum_x
            if abs(denominator) < 1e-10:
                return 0
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            return slope
        
        return series.rolling(window=period, min_periods=2).apply(lr_slope, raw=True)
    
    def calculate_momentum(self, close: pd.Series) -> pd.Series:
        """计算动量值"""
        # 价格偏离中轨的程度
        middle = close.rolling(window=self.momentum_length, min_periods=1).mean()
        deviation = close - middle
        
        # 对偏离度进行线性回归，得到动量
        momentum = self.calculate_linear_regression(deviation, self.momentum_length)
        
        return momentum
    
    def identify_squeeze_state(self, bb_data: Dict[str, pd.Series], kc_data: Dict[str, pd.Series]) -> pd.Series:
        """识别挤压状态"""
        # 挤压条件：布林带完全在肯特纳通道内部
        squeeze_condition = (bb_data['bb_upper'] <= kc_data['kc_upper']) & \
                          (bb_data['bb_lower'] >= kc_data['kc_lower'])
        
        return squeeze_condition
    
    def detect_squeeze_release(self, squeeze_state: pd.Series) -> pd.Series:
        """检测挤压释放信号"""
        # 挤压释放：从挤压状态转为非挤压状态
        squeeze_release = squeeze_state.shift(1) & (~squeeze_state)
        
        return squeeze_release.fillna(False)
    
    def calculate_momentum_acceleration(self, momentum: pd.Series) -> pd.Series:
        """计算动量加速度"""
        # 动量的变化率
        momentum_change = momentum.diff()
        
        # 加速度：动量变化的平滑
        acceleration = momentum_change.rolling(window=5, min_periods=1).mean()
        
        return acceleration
    
    def calculate_squeeze_momentum_indicators(self, 
                                            high: pd.Series, 
                                            low: pd.Series, 
                                            close: pd.Series) -> Dict[str, pd.Series]:
        """
        计算完整的挤压动量指标
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            
        Returns:
            包含所有挤压动量指标的字典
        """
        try:
            # 1. 计算布林带
            bb_data = self.calculate_bollinger_bands(close)
            
            # 2. 计算肯特纳通道
            kc_data = self.calculate_keltner_channels(high, low, close)
            
            # 3. 计算动量
            momentum = self.calculate_momentum(close)
            
            # 4. 识别挤压状态
            squeeze_state = self.identify_squeeze_state(bb_data, kc_data)
            
            # 5. 检测挤压释放
            squeeze_release = self.detect_squeeze_release(squeeze_state)
            
            # 6. 计算动量加速度
            momentum_acceleration = self.calculate_momentum_acceleration(momentum)
            
            # 7. 计算挤压强度（布林带宽度 / 肯特纳通道宽度）
            squeeze_intensity = bb_data['bb_width'] / kc_data['kc_width']
            squeeze_intensity = squeeze_intensity.fillna(1.0)
            
            # 8. 计算动量方向强度
            momentum_direction = np.where(momentum > 0, 1, -1)
            momentum_strength = abs(momentum)
            
            # 组合结果
            results = {
                # 布林带指标
                'bb_upper': bb_data['bb_upper'],
                'bb_middle': bb_data['bb_middle'],
                'bb_lower': bb_data['bb_lower'],
                'bb_width': bb_data['bb_width'],
                
                # 肯特纳通道指标
                'kc_upper': kc_data['kc_upper'],
                'kc_middle': kc_data['kc_middle'],
                'kc_lower': kc_data['kc_lower'],
                'kc_width': kc_data['kc_width'],
                
                # 挤压相关指标
                'squeeze_state': squeeze_state,
                'squeeze_release': squeeze_release,
                'squeeze_intensity': squeeze_intensity,
                
                # 动量相关指标
                'momentum': momentum,
                'momentum_direction': momentum_direction,
                'momentum_strength': momentum_strength,
                'momentum_acceleration': momentum_acceleration,
                
                # 综合评分要素
                'squeeze_days': self._calculate_squeeze_days(squeeze_state),
                'recent_releases': self._calculate_recent_releases(squeeze_release),
                'momentum_consistency': self._calculate_momentum_consistency(momentum)
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"计算挤压动量指标时出错: {e}")
            return {}
    
    def _calculate_squeeze_days(self, squeeze_state: pd.Series, window: int = 30) -> pd.Series:
        """计算最近N天内的挤压天数"""
        return squeeze_state.rolling(window=window, min_periods=1).sum()
    
    def _calculate_recent_releases(self, squeeze_release: pd.Series, window: int = 10) -> pd.Series:
        """计算最近N天内的挤压释放次数"""
        return squeeze_release.rolling(window=window, min_periods=1).sum()
    
    def _calculate_momentum_consistency(self, momentum: pd.Series, window: int = 10) -> pd.Series:
        """计算动量一致性（同方向持续性）"""
        momentum_sign = np.sign(momentum)
        
        def consistency_calc(signs):
            if len(signs) == 0:
                return 0
            # 计算连续同号的比例
            changes = np.diff(signs)
            if len(changes) == 0:
                return 1.0
            consistency = 1.0 - (np.sum(changes != 0) / len(changes))
            return consistency
        
        return momentum_sign.rolling(window=window, min_periods=2).apply(consistency_calc, raw=True)
    
    def get_current_signals(self, indicators: Dict[str, pd.Series]) -> Dict[str, any]:
        """获取当前的交易信号"""
        if not indicators or len(indicators) == 0:
            return {}
        
        # 获取最新值的辅助函数
        def get_latest_value(series_or_array):
            if hasattr(series_or_array, 'iloc'):
                return series_or_array.iloc[-1]
            else:
                return series_or_array[-1]
        
        signals = {
            'is_squeezed': bool(get_latest_value(indicators['squeeze_state'])),
            'just_released': bool(get_latest_value(indicators['squeeze_release'])),
            'momentum_value': float(get_latest_value(indicators['momentum'])),
            'momentum_direction': int(get_latest_value(indicators['momentum_direction'])),
            'momentum_strength': float(get_latest_value(indicators['momentum_strength'])),
            'momentum_acceleration': float(get_latest_value(indicators['momentum_acceleration'])),
            'squeeze_intensity': float(get_latest_value(indicators['squeeze_intensity'])),
            'squeeze_days': int(get_latest_value(indicators['squeeze_days'])),
            'recent_releases': int(get_latest_value(indicators['recent_releases'])),
            'momentum_consistency': float(get_latest_value(indicators['momentum_consistency']))
        }
        
        # 信号强度评估
        signals['signal_strength'] = self._evaluate_signal_strength(signals)
        signals['trading_signal'] = self._generate_trading_signal(signals)
        
        return signals
    
    def _evaluate_signal_strength(self, signals: Dict) -> float:
        """评估信号强度 (0-1)"""
        strength = 0.0
        
        # 挤压释放信号 (最重要)
        if signals['just_released']:
            strength += 0.4
        elif signals['is_squeezed'] and signals['squeeze_days'] > 5:
            strength += 0.2  # 长期挤压，潜在机会
        
        # 动量强度
        momentum_score = min(1.0, signals['momentum_strength'] * 10)
        strength += momentum_score * 0.3
        
        # 动量一致性
        strength += signals['momentum_consistency'] * 0.2
        
        # 动量加速度
        if signals['momentum_acceleration'] > 0 and signals['momentum_direction'] > 0:
            strength += 0.1
        elif signals['momentum_acceleration'] < 0 and signals['momentum_direction'] < 0:
            strength += 0.1
        
        return min(1.0, strength)
    
    def _generate_trading_signal(self, signals: Dict) -> str:
        """生成交易信号"""
        if signals['signal_strength'] > 0.7:
            if signals['momentum_direction'] > 0:
                return "STRONG_BUY"
            else:
                return "STRONG_SELL"
        elif signals['signal_strength'] > 0.5:
            if signals['momentum_direction'] > 0:
                return "BUY"
            else:
                return "SELL"
        elif signals['is_squeezed']:
            return "WAIT"  # 等待挤压释放
        else:
            return "HOLD"


def test_squeeze_momentum_calculator():
    """测试挤压动量计算器"""
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100)
    
    # 模拟价格数据
    base_price = 100
    returns = np.random.normal(0, 0.02, 100)
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # 创建OHLC数据
    close_prices = pd.Series(prices, index=dates)
    high_prices = close_prices * (1 + np.random.uniform(0, 0.03, 100))
    low_prices = close_prices * (1 - np.random.uniform(0, 0.03, 100))
    
    # 初始化计算器
    calculator = SqueezeMomentumCalculator()
    
    # 计算指标
    indicators = calculator.calculate_squeeze_momentum_indicators(
        high_prices, low_prices, close_prices
    )
    
    print("挤压动量指标计算完成！")
    print(f"数据点数量: {len(indicators['momentum'])}")
    print(f"挤压天数: {indicators['squeeze_state'].sum()}")
    print(f"挤压释放次数: {indicators['squeeze_release'].sum()}")
    
    # 获取最新信号
    signals = calculator.get_current_signals(indicators)
    print(f"\n当前交易信号: {signals}")
    
    return indicators, signals


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_results = test_squeeze_momentum_calculator()
    print("测试完成！")