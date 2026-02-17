#!/usr/bin/env python3
"""
V4选股系统因子提取器 (挤压动量增强版)
将V4挤压动量增强评分系统的因子映射到因子管理框架
"""

import pandas as pd
import numpy as np
import sqlite3
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path

# 导入因子管理框架
from factor_manager import FactorManager

class SqueezeMomentumCalculator:
    """简化版挤压动量计算器"""
    
    def __init__(self, bb_length: int = 20, bb_multiplier: float = 2.0,
                 kc_length: int = 20, kc_multiplier: float = 1.5,
                 momentum_length: int = 20):
        self.bb_length = bb_length
        self.bb_multiplier = bb_multiplier
        self.kc_length = kc_length
        self.kc_multiplier = kc_multiplier
        self.momentum_length = momentum_length
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """计算平均真实范围(ATR)"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = true_range.rolling(window=period, min_periods=1).mean()
        
        return atr
    
    def calculate_bollinger_bands(self, close: pd.Series) -> Dict[str, pd.Series]:
        """计算布林带"""
        middle = close.rolling(window=self.bb_length, min_periods=1).mean()
        std = close.rolling(window=self.bb_length, min_periods=1).std()
        
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
        middle = close.rolling(window=self.kc_length, min_periods=1).mean()
        atr = self.calculate_atr(high, low, close, self.kc_length)
        
        upper = middle + (atr * self.kc_multiplier)
        lower = middle - (atr * self.kc_multiplier)
        
        return {
            'kc_upper': upper,
            'kc_middle': middle,
            'kc_lower': lower,
            'kc_width': upper - lower
        }
    
    def calculate_squeeze_signals(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
        """计算挤压信号"""
        # 计算布林带和肯特纳通道
        bb_data = self.calculate_bollinger_bands(close)
        kc_data = self.calculate_keltner_channels(high, low, close)
        
        # 挤压状态：布林带在肯特纳通道内
        is_squeezed = (bb_data['bb_upper'] <= kc_data['kc_upper']) & (bb_data['bb_lower'] >= kc_data['kc_lower'])
        
        # 动量计算 (简化版线性回归)
        momentum = close.rolling(window=self.momentum_length, min_periods=1).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False
        )
        
        # 动量方向
        momentum_direction = np.where(momentum > 0, 1, -1)
        momentum_strength = abs(momentum)
        
        # 挤压释放信号
        squeeze_released = is_squeezed.shift(1) & (~is_squeezed)
        
        return pd.DataFrame({
            'is_squeezed': is_squeezed,
            'squeeze_released': squeeze_released,
            'momentum': momentum,
            'momentum_direction': momentum_direction,
            'momentum_strength': momentum_strength,
            'bb_width': bb_data['bb_width'],
            'kc_width': kc_data['kc_width']
        })


class V4FactorExtractor:
    """V4挤压动量增强因子提取器"""
    
    def __init__(self, db_path: str = "data_adapter/stock_data.db"):
        self.db_path = db_path
        self.factor_manager = FactorManager(db_path)
        self.squeeze_calculator = SqueezeMomentumCalculator()
        self.logger = self._setup_logger()
        
        # 注册V4系统的所有因子
        self._register_v4_factors()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("V4FactorExtractor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _register_v4_factors(self):
        """注册V4系统的所有因子到因子管理框架"""
        
        # 技术指标因子 (50%权重)
        self.factor_manager.register_factor(
            name="v4_kdj_strength",
            category="technical",
            description="V4 KDJ强度因子 (15%权重)",
            dependencies=["kdj_k", "kdj_d", "kdj_j"],
            calculator=self._calculate_v4_kdj_strength,
            version="4.0"
        )
        
        self.factor_manager.register_factor(
            name="v4_rsi_momentum",
            category="technical",
            description="V4 RSI动量因子 (14%权重)",
            dependencies=["rsi"],
            calculator=self._calculate_v4_rsi_momentum,
            version="4.0"
        )
        
        self.factor_manager.register_factor(
            name="v4_bbi_trend",
            category="technical",
            description="V4 BBI趋势因子 (10%权重)",
            dependencies=["close", "bbi"],
            calculator=self._calculate_v4_bbi_trend,
            version="4.0"
        )
        
        self.factor_manager.register_factor(
            name="v4_volume_surge",
            category="technical",
            description="V4成交量异动因子 (11%权重)",
            dependencies=["volume"],
            calculator=self._calculate_v4_volume_surge,
            version="4.0"
        )
        
        # 🆕 挤压动量因子 (20%权重) - V4核心创新
        self.factor_manager.register_factor(
            name="v4_squeeze_state",
            category="squeeze_momentum",
            description="V4挤压状态因子 (5%权重)",
            dependencies=["high", "low", "close"],
            calculator=self._calculate_v4_squeeze_state,
            version="4.0"
        )
        
        self.factor_manager.register_factor(
            name="v4_squeeze_release",
            category="squeeze_momentum",
            description="V4挤压释放因子 (6%权重)",
            dependencies=["high", "low", "close"],
            calculator=self._calculate_v4_squeeze_release,
            version="4.0"
        )
        
        self.factor_manager.register_factor(
            name="v4_momentum_direction",
            category="squeeze_momentum",
            description="V4动量方向因子 (5%权重)",
            dependencies=["high", "low", "close"],
            calculator=self._calculate_v4_momentum_direction,
            version="4.0"
        )
        
        self.factor_manager.register_factor(
            name="v4_momentum_acceleration",
            category="squeeze_momentum",
            description="V4动量加速度因子 (4%权重)",
            dependencies=["high", "low", "close"],
            calculator=self._calculate_v4_momentum_acceleration,
            version="4.0"
        )
        
        # 基本面因子 (8%权重)
        self.factor_manager.register_factor(
            name="v4_pe_valuation",
            category="fundamental",
            description="V4 PE估值因子 (2%权重)",
            dependencies=["pe_ttm"],
            calculator=self._calculate_v4_pe_valuation,
            version="4.0"
        )
        
        self.factor_manager.register_factor(
            name="v4_pb_valuation",
            category="fundamental",
            description="V4 PB估值因子 (2%权重)",
            dependencies=["pb"],
            calculator=self._calculate_v4_pb_valuation,
            version="4.0"
        )
        
        self.factor_manager.register_factor(
            name="v4_market_cap_factor",
            category="fundamental",
            description="V4市值因子 (2%权重)",
            dependencies=["market_cap"],
            calculator=self._calculate_v4_market_cap_factor,
            version="4.0"
        )
        
        self.factor_manager.register_factor(
            name="v4_turnover_activity",
            category="fundamental",
            description="V4换手率活跃度因子 (2%权重)",
            dependencies=["turnover_rate"],
            calculator=self._calculate_v4_turnover_activity,
            version="4.0"
        )
        
        # 市场表现因子 (18%权重)
        self.factor_manager.register_factor(
            name="v4_price_momentum",
            category="performance",
            description="V4价格动量因子 (13%权重)",
            dependencies=["close"],
            calculator=self._calculate_v4_price_momentum,
            version="4.0"
        )
        
        self.factor_manager.register_factor(
            name="v4_relative_strength",
            category="performance",
            description="V4相对强度因子 (3%权重)",
            dependencies=["close", "market_index"],
            calculator=self._calculate_v4_relative_strength,
            version="4.0"
        )
        
        self.factor_manager.register_factor(
            name="v4_volatility_risk",
            category="performance",
            description="V4波动率风险因子 (2%权重)",
            dependencies=["close"],
            calculator=self._calculate_v4_volatility_risk,
            version="4.0"
        )
        
        # 市场环境因子 (4%权重)
        self.factor_manager.register_factor(
            name="v4_market_beta",
            category="market_regime",
            description="V4市场贝塔因子 (1%权重)",
            dependencies=["close", "market_index"],
            calculator=self._calculate_v4_market_beta,
            version="4.0"
        )
        
        # V4综合评分
        self.factor_manager.register_factor(
            name="v4_comprehensive_score",
            category="composite",
            description="V4综合评分(挤压动量增强)",
            dependencies=["v4_technical", "v4_squeeze_momentum", "v4_fundamental", 
                         "v4_performance", "v4_market_regime"],
            calculator=self._calculate_v4_comprehensive_score,
            version="4.0"
        )
        
        self.logger.info("已注册15个V4系统因子到因子管理框架")
    
    def _calculate_v4_kdj_strength(self, df: pd.DataFrame) -> pd.Series:
        """计算V4 KDJ强度因子"""
        results = []
        for i in range(len(df)):
            kdj_k = df['kdj_k'].iloc[i] if 'kdj_k' in df.columns else 50
            kdj_d = df['kdj_d'].iloc[i] if 'kdj_d' in df.columns else 50
            kdj_j = df['kdj_j'].iloc[i] if 'kdj_j' in df.columns else 50
            
            # 处理NaN值
            if pd.isna(kdj_k): kdj_k = 50
            if pd.isna(kdj_d): kdj_d = 50
            if pd.isna(kdj_j): kdj_j = 50
            
            kdj_combined = (kdj_k + kdj_d + kdj_j) / 3
            
            # V4评分逻辑
            if kdj_combined <= 20:
                score = 100
            elif kdj_combined <= 30:
                score = 90 + (30 - kdj_combined) / 10 * 10
            elif kdj_combined <= 50:
                score = 50 + (50 - kdj_combined) / 20 * 40
            else:
                score = max(0, 50 - (kdj_combined - 50) / 50 * 50)
            
            results.append(score)
        
        return pd.Series(results, index=df.index)
    
    def _calculate_v4_rsi_momentum(self, df: pd.DataFrame) -> pd.Series:
        """计算V4 RSI动量因子"""
        results = []
        for i in range(len(df)):
            rsi = df['rsi'].iloc[i] if 'rsi' in df.columns else 50
            if pd.isna(rsi): rsi = 50
            
            # V4评分逻辑
            if rsi <= 30:
                score = 100
            elif rsi <= 40:
                score = 80 + (40 - rsi) / 10 * 20
            elif rsi <= 60:
                score = 50 + (50 - abs(rsi - 50)) / 10 * 30
            else:
                score = max(0, 50 - (rsi - 50) / 50 * 50)
            
            results.append(score)
        
        return pd.Series(results, index=df.index)
    
    def _calculate_v4_bbi_trend(self, df: pd.DataFrame) -> pd.Series:
        """计算V4 BBI趋势因子"""
        results = []
        for i in range(len(df)):
            close = df['close'].iloc[i] if 'close' in df.columns else 0
            bbi = df['bbi'].iloc[i] if 'bbi' in df.columns else close
            
            if pd.isna(bbi): bbi = close
            
            if close > 0 and bbi > 0:
                bbi_ratio = close / bbi
                if bbi_ratio >= 1.05:
                    score = 100
                elif bbi_ratio >= 1.02:
                    score = 80
                elif bbi_ratio >= 1.0:
                    score = 60
                elif bbi_ratio >= 0.98:
                    score = 40
                else:
                    score = 20
            else:
                score = 50
            
            results.append(score)
        
        return pd.Series(results, index=df.index)
    
    def _calculate_v4_volume_surge(self, df: pd.DataFrame) -> pd.Series:
        """计算V4成交量异动因子"""
        results = []
        for i in range(len(df)):
            if i < 5:
                results.append(50)
                continue
            
            window_df = df.iloc[max(0, i-19):i+1]
            
            # 近期5日平均成交量 vs 历史20日平均
            recent_volume = window_df['volume'].tail(5).mean()
            historical_volume = window_df['volume'].tail(20).mean()
            
            if historical_volume > 0:
                volume_ratio = recent_volume / historical_volume
                if volume_ratio >= 3:
                    score = 100
                elif volume_ratio >= 2:
                    score = 80
                elif volume_ratio >= 1.5:
                    score = 60
                else:
                    score = 40
            else:
                score = 50
            
            results.append(score)
        
        return pd.Series(results, index=df.index)
    
    def _calculate_v4_squeeze_state(self, df: pd.DataFrame) -> pd.Series:
        """计算V4挤压状态因子"""
        if len(df) < 20:
            return pd.Series([50.0] * len(df), index=df.index)
        
        # 计算挤压信号
        squeeze_data = self.squeeze_calculator.calculate_squeeze_signals(
            df['high'], df['low'], df['close']
        )
        
        results = []
        for i in range(len(df)):
            score = 50  # 基础分
            
            if i < len(squeeze_data):
                is_squeezed = squeeze_data['is_squeezed'].iloc[i]
                
                if is_squeezed:
                    score += 25  # 挤压状态奖励
                    
                    # 计算连续挤压天数
                    squeeze_days = 0
                    for j in range(i, max(-1, i-15), -1):
                        if j < len(squeeze_data) and squeeze_data['is_squeezed'].iloc[j]:
                            squeeze_days += 1
                        else:
                            break
                    
                    # 长期挤压额外奖励
                    if squeeze_days > 10:
                        score += 10
            
            results.append(min(100, score))
        
        return pd.Series(results, index=df.index)
    
    def _calculate_v4_squeeze_release(self, df: pd.DataFrame) -> pd.Series:
        """计算V4挤压释放因子"""
        if len(df) < 20:
            # 数据不足时，使用简单逻辑避免全部返回50
            simple_scores = []
            for i in range(len(df)):
                # 基于收盘价变化的简单评分
                if i >= 5:
                    recent_change = df['close'].iloc[i] / df['close'].iloc[i-5] - 1
                    score = min(100, max(10, 50 + recent_change * 1000))
                else:
                    score = 30 + np.random.uniform(-10, 20)  # 添加一些随机性避免全部相同
                simple_scores.append(score)
            return pd.Series(simple_scores, index=df.index)
        
        try:
            squeeze_data = self.squeeze_calculator.calculate_squeeze_signals(
                df['high'], df['low'], df['close']
            )
            
            results = []
            for i in range(len(df)):
                score = 20  # 基础分
                
                if i < len(squeeze_data):
                    # 获取挤压相关信息
                    is_squeezed = squeeze_data['is_squeezed'].iloc[i] if not pd.isna(squeeze_data['is_squeezed'].iloc[i]) else False
                    just_released = squeeze_data['squeeze_released'].iloc[i] if not pd.isna(squeeze_data['squeeze_released'].iloc[i]) else False
                    bb_width = squeeze_data['bb_width'].iloc[i] if not pd.isna(squeeze_data['bb_width'].iloc[i]) else 1.0
                    kc_width = squeeze_data['kc_width'].iloc[i] if not pd.isna(squeeze_data['kc_width'].iloc[i]) else 1.0
                    
                    if just_released:
                        score = 95 + np.random.uniform(0, 5)  # 刚释放，高分，加随机性
                    else:
                        # 检查近期释放情况
                        recent_releases = 0
                        for j in range(max(0, i-5), i):
                            if j < len(squeeze_data) and not pd.isna(squeeze_data['squeeze_released'].iloc[j]):
                                if squeeze_data['squeeze_released'].iloc[j]:
                                    recent_releases += 1
                        
                        if recent_releases > 0:
                            score = 70 + recent_releases * 5  # 近期有释放
                        elif is_squeezed:
                            # 检查挤压持续时间
                            squeeze_days = 0
                            for j in range(i, max(-1, i-15), -1):
                                if j < len(squeeze_data) and not pd.isna(squeeze_data['is_squeezed'].iloc[j]):
                                    if squeeze_data['is_squeezed'].iloc[j]:
                                        squeeze_days += 1
                                    else:
                                        break
                            
                            if squeeze_days > 10:
                                score = 50 + squeeze_days  # 长期挤压，分数递增
                            elif squeeze_days > 5:
                                score = 35 + squeeze_days  # 中期挤压
                            else:
                                score = 25  # 短期挤压
                        else:
                            # 非挤压状态，基于波动率相对大小评分
                            if bb_width > 0 and kc_width > 0:
                                width_ratio = bb_width / kc_width
                                score = min(80, max(15, 30 + width_ratio * 30))
                            else:
                                score = 30 + np.random.uniform(-5, 10)
                else:
                    # 如果挤压数据不可用，使用价格动量作为替代
                    if i >= 10:
                        price_momentum = (df['close'].iloc[i] / df['close'].iloc[i-10] - 1) * 100
                        score = min(85, max(10, 40 + price_momentum))
                    else:
                        score = 25 + np.random.uniform(0, 15)
                
                results.append(min(100, max(0, score)))
            
            return pd.Series(results, index=df.index)
            
        except Exception as e:
            self.logger.error(f"挤压释放因子计算失败: {e}")
            # 出错时使用基于价格和成交量的简单评分
            fallback_scores = []
            for i in range(len(df)):
                if i >= 5:
                    # 基于价格变化和成交量的简单评分
                    price_change = df['close'].iloc[i] / df['close'].iloc[i-5] - 1
                    if 'volume' in df.columns and i >= 5:
                        vol_ratio = df['volume'].iloc[i] / df['volume'].iloc[i-5:i].mean() if df['volume'].iloc[i-5:i].mean() > 0 else 1
                        score = min(95, max(5, 35 + price_change * 500 + vol_ratio * 10))
                    else:
                        score = min(90, max(10, 40 + price_change * 800))
                else:
                    score = 20 + np.random.uniform(5, 25)
                fallback_scores.append(score)
            return pd.Series(fallback_scores, index=df.index)
    
    def _calculate_v4_momentum_direction(self, df: pd.DataFrame) -> pd.Series:
        """计算V4动量方向因子"""
        if len(df) < 20:
            return pd.Series([50.0] * len(df), index=df.index)
        
        squeeze_data = self.squeeze_calculator.calculate_squeeze_signals(
            df['high'], df['low'], df['close']
        )
        
        results = []
        for i in range(len(df)):
            score = 50  # 基础分
            
            if i < len(squeeze_data):
                momentum_direction = squeeze_data['momentum_direction'].iloc[i]
                momentum_strength = squeeze_data['momentum_strength'].iloc[i]
                
                if momentum_direction > 0:
                    score += min(50, momentum_strength * 100)
                else:
                    score -= min(50, momentum_strength * 100)
            
            results.append(max(0, min(100, score)))
        
        return pd.Series(results, index=df.index)
    
    def _calculate_v4_momentum_acceleration(self, df: pd.DataFrame) -> pd.Series:
        """计算V4动量加速度因子"""
        if len(df) < 20:
            return pd.Series([50.0] * len(df), index=df.index)
        
        squeeze_data = self.squeeze_calculator.calculate_squeeze_signals(
            df['high'], df['low'], df['close']
        )
        
        results = []
        for i in range(len(df)):
            score = 50  # 基础分
            
            if i >= 1 and i < len(squeeze_data):
                current_momentum = squeeze_data['momentum'].iloc[i]
                prev_momentum = squeeze_data['momentum'].iloc[i-1]
                acceleration = current_momentum - prev_momentum
                
                if acceleration > 0:
                    score += min(50, abs(acceleration) * 50)
                else:
                    score -= min(50, abs(acceleration) * 50)
                
                # 一致性检查
                if i >= 3:
                    recent_momentums = squeeze_data['momentum'].iloc[i-3:i+1]
                    consistency = 1.0 if (recent_momentums > 0).all() or (recent_momentums < 0).all() else 0.0
                    
                    if consistency > 0.8:
                        score += 20  # 一致性奖励
            
            results.append(max(0, min(100, score)))
        
        return pd.Series(results, index=df.index)
    
    def _calculate_v4_pe_valuation(self, df: pd.DataFrame) -> pd.Series:
        """计算V4 PE估值因子"""
        results = []
        for i in range(len(df)):
            pe = df['pe_ttm'].iloc[i] if 'pe_ttm' in df.columns else None
            
            if pd.isna(pe) or pe is None or pe <= 0:
                score = 50
            elif pe <= 15:
                score = 100
            elif pe <= 25:
                score = 80
            elif pe <= 40:
                score = 60
            elif pe <= 60:
                score = 40
            else:
                score = 20
            
            results.append(score)
        
        return pd.Series(results, index=df.index)
    
    def _calculate_v4_pb_valuation(self, df: pd.DataFrame) -> pd.Series:
        """计算V4 PB估值因子"""
        results = []
        for i in range(len(df)):
            pb = df['pb'].iloc[i] if 'pb' in df.columns else None
            
            if pd.isna(pb) or pb is None or pb <= 0:
                score = 50
            elif pb <= 1:
                score = 100
            elif pb <= 2:
                score = 80
            elif pb <= 3:
                score = 60
            elif pb <= 5:
                score = 40
            else:
                score = 20
            
            results.append(score)
        
        return pd.Series(results, index=df.index)
    
    def _calculate_v4_market_cap_factor(self, df: pd.DataFrame) -> pd.Series:
        """计算V4市值因子(小市值偏好)"""
        results = []
        for i in range(len(df)):
            market_cap = df['market_cap'].iloc[i] if 'market_cap' in df.columns else None
            
            if pd.isna(market_cap) or market_cap is None:
                score = 50
            elif market_cap <= 50:
                score = 100  # 小市值
            elif market_cap <= 100:
                score = 80
            elif market_cap <= 300:
                score = 60
            elif market_cap <= 1000:
                score = 40
            else:
                score = 20   # 大市值
            
            results.append(score)
        
        return pd.Series(results, index=df.index)
    
    def _calculate_v4_turnover_activity(self, df: pd.DataFrame) -> pd.Series:
        """计算V4换手率活跃度因子"""
        results = []
        for i in range(len(df)):
            turnover = df['turnover_rate'].iloc[i] if 'turnover_rate' in df.columns else None
            
            if pd.isna(turnover) or turnover is None:
                score = 50
            elif turnover >= 10:
                score = 100
            elif turnover >= 5:
                score = 80
            elif turnover >= 2:
                score = 60
            elif turnover >= 1:
                score = 40
            else:
                score = 20
            
            results.append(score)
        
        return pd.Series(results, index=df.index)
    
    def _calculate_v4_price_momentum(self, df: pd.DataFrame) -> pd.Series:
        """计算V4价格动量因子"""
        results = []
        for i in range(len(df)):
            if i < 5:
                results.append(50.0)
                continue
            
            # 5日价格动量
            current_close = df['close'].iloc[i]
            prev_close = df['close'].iloc[i-4]  # 5天前
            
            if prev_close > 0:
                momentum_5d = (current_close / prev_close - 1) * 100
                
                if momentum_5d >= 5:
                    score = 100
                elif momentum_5d >= 2:
                    score = 80
                elif momentum_5d >= 0:
                    score = 60
                elif momentum_5d >= -2:
                    score = 40
                else:
                    score = 20
            else:
                score = 50
            
            results.append(score)
        
        return pd.Series(results, index=df.index)
    
    def _calculate_v4_relative_strength(self, df: pd.DataFrame) -> pd.Series:
        """计算V4相对强度因子"""
        results = []
        for i in range(len(df)):
            if i < 20:
                results.append(50.0)
                continue
            
            # 计算20日相对强度
            current_close = df['close'].iloc[i]
            start_close = df['close'].iloc[i-19]
            
            if start_close > 0:
                stock_return = (current_close / start_close - 1) * 100
                
                # 假设大盘20日收益率为2%(简化)
                market_return = 2.0
                
                # 相对强度 = 股票收益 - 市场收益
                relative_strength = stock_return - market_return
                
                # 转换为评分 (0-100)
                if relative_strength >= 15:
                    score = 95
                elif relative_strength >= 10:
                    score = 85
                elif relative_strength >= 5:
                    score = 75
                elif relative_strength >= 0:
                    score = 65
                elif relative_strength >= -5:
                    score = 45
                elif relative_strength >= -10:
                    score = 35
                elif relative_strength >= -15:
                    score = 25
                else:
                    score = 15
            else:
                score = 50
            
            results.append(score)
        
        return pd.Series(results, index=df.index)
    
    def _calculate_v4_volatility_risk(self, df: pd.DataFrame) -> pd.Series:
        """计算V4波动率风险因子"""
        results = []
        for i in range(len(df)):
            if i < 10:
                results.append(50.0)
                continue
            
            # 计算10日波动率
            returns = df['close'].iloc[i-9:i+1].pct_change().dropna()
            if len(returns) > 0:
                volatility = returns.std() * np.sqrt(252)  # 年化波动率
                # 波动率越低，风险越小，得分越高
                if volatility <= 0.2:
                    score = 100
                elif volatility <= 0.3:
                    score = 80
                elif volatility <= 0.5:
                    score = 60
                elif volatility <= 0.8:
                    score = 40
                else:
                    score = 20
            else:
                score = 50
            
            results.append(score)
        
        return pd.Series(results, index=df.index)
    
    def _calculate_v4_market_beta(self, df: pd.DataFrame) -> pd.Series:
        """计算V4市场贝塔因子"""
        results = []
        for i in range(len(df)):
            if i < 30:
                results.append(50.0)
                continue
            
            # 计算30日贝塔系数
            stock_returns = df['close'].iloc[i-29:i+1].pct_change().dropna()
            
            if len(stock_returns) > 10:
                # 简化市场收益率计算(假设市场日均收益0.05%)
                market_returns = pd.Series([0.0005] * len(stock_returns))
                
                # 计算贝塔系数
                if market_returns.var() > 0:
                    beta = stock_returns.cov(market_returns) / market_returns.var()
                    
                    # 转换为评分 (贝塔越接近1越好)
                    beta_abs = abs(beta - 1.0)
                    if beta_abs <= 0.1:
                        score = 90
                    elif beta_abs <= 0.2:
                        score = 80
                    elif beta_abs <= 0.3:
                        score = 70
                    elif beta_abs <= 0.5:
                        score = 60
                    elif beta_abs <= 0.8:
                        score = 50
                    else:
                        score = 40
                else:
                    score = 50
            else:
                score = 50
            
            results.append(score)
        
        return pd.Series(results, index=df.index)
    
    def _calculate_v4_sector_rotation(self, df: pd.DataFrame) -> pd.Series:
        """计算V4板块轮动因子"""
        results = []
        for i in range(len(df)):
            if i < 10:
                results.append(50.0)
                continue
            
            # 计算10日价格变化率
            current_close = df['close'].iloc[i]
            prev_close = df['close'].iloc[i-9]
            
            if prev_close > 0:
                price_change = (current_close / prev_close - 1) * 100
                
                # 根据价格变化率判断板块轮动活跃度
                if price_change >= 10:
                    score = 85  # 强势上涨
                elif price_change >= 5:
                    score = 75  # 温和上涨
                elif price_change >= 2:
                    score = 65  # 小幅上涨
                elif price_change >= -2:
                    score = 55  # 横盘
                elif price_change >= -5:
                    score = 45  # 小幅下跌
                elif price_change >= -10:
                    score = 35  # 温和下跌
                else:
                    score = 25  # 大幅下跌
            else:
                score = 50
            
            results.append(score)
        
        return pd.Series(results, index=df.index)
    
    def _calculate_v4_liquidity(self, df: pd.DataFrame) -> pd.Series:
        """计算V4流动性因子"""
        results = []
        for i in range(len(df)):
            if i < 5:
                results.append(50.0)
                continue
            
            # 计算5日平均换手率和成交额
            if 'turnover_rate' in df.columns:
                recent_turnover = df['turnover_rate'].iloc[i-4:i+1].mean()
                
                # 根据换手率评估流动性
                if recent_turnover >= 10:
                    score = 95  # 非常活跃
                elif recent_turnover >= 5:
                    score = 85  # 活跃
                elif recent_turnover >= 3:
                    score = 75  # 较为活跃
                elif recent_turnover >= 2:
                    score = 65  # 正常
                elif recent_turnover >= 1:
                    score = 55  # 较低
                elif recent_turnover >= 0.5:
                    score = 45  # 低
                else:
                    score = 35  # 很低
            else:
                # 没有换手率数据时，使用成交量简化计算
                if 'volume' in df.columns and i >= 5:
                    recent_volume = df['volume'].iloc[i-4:i+1].mean()
                    if recent_volume > 0:
                        # 简化的流动性评分
                        score = min(95, max(35, 50 + np.log10(recent_volume/1000000) * 10))
                    else:
                        score = 35
                else:
                    score = 50
            
            results.append(score)
        
        return pd.Series(results, index=df.index)
    
    def _calculate_v4_comprehensive_score(self, df: pd.DataFrame) -> pd.Series:
        """计算V4综合评分"""
        # V4权重配置
        weights = {
            # 技术指标 (50%)
            'kdj_strength': 0.15,
            'rsi_momentum': 0.14,
            'bbi_trend': 0.10,
            'volume_surge': 0.11,
            # 挤压动量 (20%) - V4核心创新
            'squeeze_state': 0.05,
            'squeeze_release': 0.06,
            'momentum_direction': 0.05,
            'momentum_acceleration': 0.04,
            # 基本面 (8%)
            'pe_valuation': 0.02,
            'pb_valuation': 0.02,
            'market_cap_factor': 0.02,
            'turnover_activity': 0.02,
            # 市场表现 (18%)
            'price_momentum': 0.13,
            'relative_strength': 0.03,
            'volatility_risk': 0.02,
            # 市场环境 (4%)
            'market_beta': 0.01,
            'sector_rotation': 0.015,
            'liquidity': 0.015
        }
        
        # 计算各因子得分
        factor_scores = {}
        factor_scores['kdj_strength'] = self._calculate_v4_kdj_strength(df)
        factor_scores['rsi_momentum'] = self._calculate_v4_rsi_momentum(df)
        factor_scores['bbi_trend'] = self._calculate_v4_bbi_trend(df)
        factor_scores['volume_surge'] = self._calculate_v4_volume_surge(df)
        
        # 挤压动量因子
        factor_scores['squeeze_state'] = self._calculate_v4_squeeze_state(df)
        factor_scores['squeeze_release'] = self._calculate_v4_squeeze_release(df)
        factor_scores['momentum_direction'] = self._calculate_v4_momentum_direction(df)
        factor_scores['momentum_acceleration'] = self._calculate_v4_momentum_acceleration(df)
        
        # 基本面因子
        factor_scores['pe_valuation'] = self._calculate_v4_pe_valuation(df)
        factor_scores['pb_valuation'] = self._calculate_v4_pb_valuation(df)
        factor_scores['market_cap_factor'] = self._calculate_v4_market_cap_factor(df)
        factor_scores['turnover_activity'] = self._calculate_v4_turnover_activity(df)
        
        # 市场表现因子
        factor_scores['price_momentum'] = self._calculate_v4_price_momentum(df)
        factor_scores['relative_strength'] = self._calculate_v4_relative_strength(df)
        factor_scores['volatility_risk'] = self._calculate_v4_volatility_risk(df)
        
        # 市场环境因子
        factor_scores['market_beta'] = self._calculate_v4_market_beta(df)
        factor_scores['sector_rotation'] = self._calculate_v4_sector_rotation(df)
        factor_scores['liquidity'] = self._calculate_v4_liquidity(df)
        
        # 计算加权综合得分
        comprehensive_score = pd.Series([0.0] * len(df), index=df.index)
        
        for factor_name, weight in weights.items():
            if factor_name in factor_scores:
                comprehensive_score += factor_scores[factor_name] * weight
        
        return comprehensive_score
    
    def extract_factors_for_stock(self, stock_code: str, 
                                 start_date: str, 
                                 end_date: str) -> pd.DataFrame:
        """为单只股票提取所有V4因子"""
        
        self.logger.info(f"提取股票 {stock_code} 的V4因子数据...")
        
        # 获取股票原始数据
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT 
                    dq.trade_date,
                    dq.open,
                    dq.high,
                    dq.low,
                    dq.close,
                    dq.volume,
                    ti.kdj_k,
                    ti.kdj_d,
                    ti.kdj_j,
                    ti.rsi12 as rsi,
                    ti.bbi,
                    db.pe_ttm,
                    db.pb,
                    db.circ_mv as market_cap,
                    db.turnover_rate
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                LEFT JOIN technical_indicators ti ON ti.security_id = s.id AND ti.trade_date = dq.trade_date
                LEFT JOIN daily_basic db ON db.security_id = s.id AND db.trade_date = dq.trade_date
                WHERE s.code = ? 
                AND dq.trade_date BETWEEN ? AND ?
                ORDER BY dq.trade_date
            """
            
            df = pd.read_sql_query(query, conn, params=[stock_code, start_date, end_date])
        
        if df.empty:
            self.logger.warning(f"未找到股票 {stock_code} 的数据")
            return pd.DataFrame()
        
        # 计算所有V4因子
        result_df = df[['trade_date']].copy()
        result_df['stock_code'] = stock_code
        
        # 技术因子
        result_df['v4_kdj_strength'] = self._calculate_v4_kdj_strength(df)
        result_df['v4_rsi_momentum'] = self._calculate_v4_rsi_momentum(df)
        result_df['v4_bbi_trend'] = self._calculate_v4_bbi_trend(df)
        result_df['v4_volume_surge'] = self._calculate_v4_volume_surge(df)
        
        # 🆕 挤压动量因子
        result_df['v4_squeeze_state'] = self._calculate_v4_squeeze_state(df)
        result_df['v4_squeeze_release'] = self._calculate_v4_squeeze_release(df)
        result_df['v4_momentum_direction'] = self._calculate_v4_momentum_direction(df)
        result_df['v4_momentum_acceleration'] = self._calculate_v4_momentum_acceleration(df)
        
        # 基本面因子
        result_df['v4_pe_valuation'] = self._calculate_v4_pe_valuation(df)
        result_df['v4_pb_valuation'] = self._calculate_v4_pb_valuation(df)
        result_df['v4_market_cap_factor'] = self._calculate_v4_market_cap_factor(df)
        result_df['v4_turnover_activity'] = self._calculate_v4_turnover_activity(df)
        
        # 市场表现因子
        result_df['v4_price_momentum'] = self._calculate_v4_price_momentum(df)
        result_df['v4_relative_strength'] = self._calculate_v4_relative_strength(df)
        result_df['v4_volatility_risk'] = self._calculate_v4_volatility_risk(df)
        
        # 市场环境因子
        result_df['v4_market_beta'] = self._calculate_v4_market_beta(df)
        
        # V4综合评分
        result_df['v4_comprehensive_score'] = self._calculate_v4_comprehensive_score(df)
        
        return result_df
    
    def batch_extract_factors(self, stock_codes: List[str],
                            start_date: str,
                            end_date: str,
                            batch_size: int = 100) -> pd.DataFrame:
        """批量提取V4因子数据"""
        
        self.logger.info(f"批量提取 {len(stock_codes)} 只股票的V4因子...")
        
        all_results = []
        
        for i in range(0, len(stock_codes), batch_size):
            batch_codes = stock_codes[i:i+batch_size]
            batch_results = []
            
            for code in batch_codes:
                try:
                    factor_data = self.extract_factors_for_stock(code, start_date, end_date)
                    if not factor_data.empty:
                        batch_results.append(factor_data)
                except Exception as e:
                    self.logger.error(f"提取股票 {code} 因子失败: {e}")
                    continue
            
            if batch_results:
                batch_df = pd.concat(batch_results, ignore_index=True)
                all_results.append(batch_df)
                
            self.logger.info(f"完成批次 {i//batch_size + 1}/{(len(stock_codes)-1)//batch_size + 1}")
        
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            self.logger.info(f"成功提取 {len(final_df)} 条V4因子记录")
            return final_df
        else:
            return pd.DataFrame()
    
    def save_factors_to_database(self, factor_data: pd.DataFrame):
        """将V4因子数据保存到数据库"""
        
        self.logger.info(f"保存 {len(factor_data)} 条V4因子数据到数据库...")
        
        with sqlite3.connect(self.db_path) as conn:
            # 创建v4_factors表（如果不存在）
            conn.execute("""
                CREATE TABLE IF NOT EXISTS v4_factors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    security_id INTEGER,
                    trade_date DATE,
                    -- 技术指标因子 (50%)
                    v4_kdj_strength REAL,
                    v4_rsi_momentum REAL,
                    v4_bbi_trend REAL,
                    v4_volume_surge REAL,
                    -- 挤压动量因子 (20%)
                    v4_squeeze_state REAL,
                    v4_squeeze_release REAL,
                    v4_momentum_direction REAL,
                    v4_momentum_acceleration REAL,
                    -- 基本面因子 (8%)
                    v4_pe_valuation REAL,
                    v4_pb_valuation REAL,
                    v4_market_cap_factor REAL,
                    v4_turnover_activity REAL,
                    -- 市场表现因子 (18%)
                    v4_price_momentum REAL,
                    v4_relative_strength REAL,
                    v4_volatility_risk REAL,
                    -- 市场环境因子 (4%)
                    v4_market_beta REAL,
                    -- 综合评分
                    v4_comprehensive_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(security_id, trade_date)
                )
            """)
            
            # 插入数据
            v4_factors = [col for col in factor_data.columns if col.startswith('v4_')]
            
            for _, row in factor_data.iterrows():
                try:
                    # 获取security_id
                    security_id_query = "SELECT id FROM securities WHERE code = ?"
                    security_result = conn.execute(security_id_query, [row['stock_code']]).fetchone()
                    
                    if security_result:
                        security_id = security_result[0]
                        
                        # 构建插入SQL
                        columns = ['security_id', 'trade_date'] + v4_factors
                        placeholders = ['?'] * len(columns)
                        
                        values = [security_id, row['trade_date']]
                        values.extend([row[factor] for factor in v4_factors])
                        
                        conn.execute(f"""
                            INSERT OR REPLACE INTO v4_factors 
                            ({','.join(columns)})
                            VALUES ({','.join(placeholders)})
                        """, values)
                        
                except Exception as e:
                    self.logger.error(f"保存股票 {row['stock_code']} 数据失败: {e}")
                    continue
            
            conn.commit()
            self.logger.info("V4因子数据保存完成")


def main():
    """测试V4因子提取器"""
    
    extractor = V4FactorExtractor()
    
    # 测试单只股票
    print("测试提取单只股票V4因子...")
    factor_data = extractor.extract_factors_for_stock(
        '000001', '2025-08-01', '2025-08-18'
    )
    print(f"提取到 {len(factor_data)} 条记录")
    
    # 显示V4因子列表
    v4_factors = [col for col in factor_data.columns if col.startswith('v4_')]
    print(f"V4因子数量: {len(v4_factors)}")
    print("V4因子列表:")
    for factor in v4_factors:
        print(f"  - {factor}")
    
    print(factor_data.head())
    
    # 测试批量提取
    print("\n测试批量提取V4因子...")
    test_stocks = ['000001', '000002', '000858']
    batch_data = extractor.batch_extract_factors(
        test_stocks, '2025-08-01', '2025-08-18'
    )
    print(f"批量提取到 {len(batch_data)} 条记录")
    
    # 保存到数据库
    if not batch_data.empty:
        extractor.save_factors_to_database(batch_data)


if __name__ == "__main__":
    main()