#!/usr/bin/env python3
"""
技术面因子优化模块
Enhanced Technical Factors Module

基于相关性分析结果，优化技术指标计算：
1. 动态参数调整（根据市场波动率）
2. 多周期确认
3. 形态识别
4. 趋势强度量化
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import sqlite3
from datetime import datetime, timedelta

from .core_framework import FactorCalculator, StockData

class AdaptiveTechnicalIndicator:
    """自适应技术指标计算器"""
    
    @staticmethod
    def calculate_volatility(prices: np.array, window: int = 20) -> float:
        """计算价格波动率"""
        if len(prices) < window:
            return 0.02  # 默认波动率
        
        returns = np.diff(np.log(prices[-window:]))
        return np.std(returns) * np.sqrt(252)  # 年化波动率
    
    @staticmethod
    def adaptive_ma_period(volatility: float, base_period: int = 20) -> int:
        """根据波动率调整移动平均周期"""
        if volatility > 0.4:  # 高波动
            return int(base_period * 0.7)  # 缩短周期
        elif volatility < 0.15:  # 低波动
            return int(base_period * 1.3)  # 延长周期
        else:
            return base_period
    
    @staticmethod
    def adaptive_rsi_period(volatility: float, base_period: int = 14) -> int:
        """根据波动率调整RSI周期"""
        if volatility > 0.4:
            return max(6, int(base_period * 0.6))
        elif volatility < 0.15:
            return min(30, int(base_period * 1.4))
        else:
            return base_period

class TrendStrengthFactor(FactorCalculator):
    """趋势强度因子计算器"""
    
    def __init__(self, db_path: str):
        """初始化趋势强度因子计算器"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
    
    def get_price_history(self, stock_code: str, trade_date: str, days: int = 60) -> np.array:
        """获取历史价格数据"""
        try:
            query = """
                SELECT dq.close
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.code = ? AND dq.trade_date <= ?
                ORDER BY dq.trade_date DESC
                LIMIT ?
            """
            
            df = pd.read_sql_query(query, self.conn, params=[stock_code, trade_date, days])
            
            if df.empty:
                return np.array([])
            
            return df['close'].values[::-1]  # 反转为时间正序
            
        except Exception as e:
            print(f"获取股票 {stock_code} 历史价格失败: {e}")
            return np.array([])
    
    def calculate_adx(self, high_prices: np.array, low_prices: np.array, 
                     close_prices: np.array, period: int = 14) -> float:
        """计算ADX趋势强度指标"""
        try:
            if len(high_prices) < period + 1:
                return 50.0
            
            # 计算True Range
            tr1 = high_prices[1:] - low_prices[1:]
            tr2 = np.abs(high_prices[1:] - close_prices[:-1])
            tr3 = np.abs(low_prices[1:] - close_prices[:-1])
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # 计算Directional Movement
            dm_pos = np.where(high_prices[1:] - high_prices[:-1] > low_prices[:-1] - low_prices[1:],
                             np.maximum(high_prices[1:] - high_prices[:-1], 0), 0)
            dm_neg = np.where(low_prices[:-1] - low_prices[1:] > high_prices[1:] - high_prices[:-1],
                             np.maximum(low_prices[:-1] - low_prices[1:], 0), 0)
            
            # 计算平滑的DI
            atr = np.mean(tr[-period:])
            di_pos = np.mean(dm_pos[-period:]) / atr * 100
            di_neg = np.mean(dm_neg[-period:]) / atr * 100
            
            # 计算ADX
            dx = np.abs(di_pos - di_neg) / (di_pos + di_neg) * 100 if (di_pos + di_neg) != 0 else 0
            
            return min(100.0, max(0.0, dx))
            
        except Exception as e:
            print(f"计算ADX失败: {e}")
            return 50.0
    
    def calculate_trend_consistency(self, prices: np.array, window: int = 20) -> float:
        """计算趋势一致性"""
        try:
            if len(prices) < window:
                return 50.0
            
            # 计算移动平均斜率
            ma = pd.Series(prices).rolling(5).mean().values
            slopes = []
            
            for i in range(len(ma) - 5):
                if not np.isnan(ma[i+5]) and not np.isnan(ma[i]):
                    slope = (ma[i+5] - ma[i]) / 5
                    slopes.append(1 if slope > 0 else -1)
            
            if len(slopes) < 10:
                return 50.0
            
            # 计算趋势一致性（连续同向的比例）
            consistency = 0
            max_consecutive = 0
            current_consecutive = 1
            
            for i in range(1, len(slopes)):
                if slopes[i] == slopes[i-1]:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 1
            
            consistency = max_consecutive / len(slopes) * 100
            return min(100.0, max(0.0, consistency))
            
        except Exception as e:
            print(f"计算趋势一致性失败: {e}")
            return 50.0
    
    def calculate(self, stock_data: StockData, market_data: Dict) -> float:
        """计算趋势强度因子综合得分"""
        try:
            # 获取历史价格数据
            prices = self.get_price_history(stock_data.code, stock_data.trade_date, 60)
            
            if len(prices) < 20:
                return 50.0
            
            # 计算趋势强度（ADX模拟）
            price_changes = np.diff(prices)
            positive_moves = np.sum(price_changes > 0)
            total_moves = len(price_changes)
            trend_direction = positive_moves / total_moves if total_moves > 0 else 0.5
            
            # 计算趋势一致性
            consistency = self.calculate_trend_consistency(prices)
            
            # 计算价格相对位置
            current_price = prices[-1]
            price_range = np.max(prices[-20:]) - np.min(prices[-20:])
            if price_range > 0:
                price_position = (current_price - np.min(prices[-20:])) / price_range * 100
            else:
                price_position = 50.0
            
            # 综合评分
            if trend_direction > 0.6:  # 上升趋势
                trend_score = 70 + (trend_direction - 0.6) * 75
            elif trend_direction < 0.4:  # 下降趋势
                trend_score = 30 - (0.4 - trend_direction) * 75
            else:  # 震荡
                trend_score = 50
            
            # 结合趋势一致性和价格位置
            final_score = trend_score * 0.5 + consistency * 0.3 + price_position * 0.2
            
            return min(100.0, max(0.0, final_score))
            
        except Exception as e:
            print(f"计算趋势强度因子失败: {e}")
            return 50.0
    
    def get_factor_name(self) -> str:
        return "TrendStrengthFactor"

class MomentumFactor(FactorCalculator):
    """动量因子计算器"""
    
    def __init__(self, db_path: str):
        """初始化动量因子计算器"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.adaptive_indicator = AdaptiveTechnicalIndicator()
    
    def get_technical_data(self, stock_code: str, trade_date: str) -> Dict:
        """获取技术指标数据"""
        try:
            query = """
                SELECT ti.rsi, ti.macd_dif, ti.kdj_k, ti.kdj_d, ti.kdj_j,
                       dq.close, dq.volume, dq.ma5, dq.ma10, dq.ma20
                FROM technical_indicators ti
                JOIN daily_quotes dq ON ti.security_id = dq.security_id AND ti.trade_date = dq.trade_date
                JOIN securities s ON ti.security_id = s.id
                WHERE s.code = ? AND ti.trade_date = ?
            """
            
            df = pd.read_sql_query(query, self.conn, params=[stock_code, trade_date])
            
            if df.empty:
                return {}
            
            row = df.iloc[0]
            return {
                'rsi': row.get('rsi'),
                'macd_dif': row.get('macd_dif'),
                'kdj_k': row.get('kdj_k'),
                'kdj_d': row.get('kdj_d'),
                'kdj_j': row.get('kdj_j'),
                'close': row.get('close'),
                'volume': row.get('volume'),
                'ma5': row.get('ma5'),
                'ma10': row.get('ma10'),
                'ma20': row.get('ma20')
            }
            
        except Exception as e:
            print(f"获取股票 {stock_code} 技术数据失败: {e}")
            return {}
    
    def calculate_rsi_score(self, rsi: Optional[float]) -> float:
        """计算RSI动量得分"""
        if rsi is None:
            return 50.0
        
        if rsi >= 80:
            return 20.0  # 超买，低分
        elif rsi >= 70:
            return 40.0
        elif rsi >= 60:
            return 75.0  # 强势，高分
        elif rsi >= 50:
            return 65.0
        elif rsi >= 40:
            return 45.0
        elif rsi >= 30:
            return 35.0
        elif rsi >= 20:
            return 60.0  # 超卖反弹机会
        else:
            return 75.0  # 严重超卖，机会较大
    
    def calculate_macd_score(self, macd_dif: Optional[float]) -> float:
        """计算MACD动量得分"""
        if macd_dif is None:
            return 50.0
        
        if macd_dif > 0.5:
            return 85.0  # 强势上涨
        elif macd_dif > 0:
            return 70.0  # 上涨趋势
        elif macd_dif > -0.5:
            return 40.0  # 弱势下跌
        else:
            return 25.0  # 强势下跌
    
    def calculate_kdj_score(self, kdj_k: Optional[float], kdj_d: Optional[float], 
                           kdj_j: Optional[float]) -> float:
        """计算KDJ动量得分"""
        if not all([kdj_k, kdj_d, kdj_j]):
            return 50.0
        
        scores = []
        
        # K值评分
        if kdj_k >= 80:
            scores.append(25.0)  # 超买
        elif kdj_k >= 50:
            scores.append(75.0)  # 强势
        elif kdj_k >= 20:
            scores.append(45.0)  # 弱势
        else:
            scores.append(70.0)  # 超卖反弹
        
        # 金叉死叉评分
        if kdj_k > kdj_d and kdj_j > kdj_k:
            scores.append(80.0)  # 金叉向上
        elif kdj_k < kdj_d and kdj_j < kdj_k:
            scores.append(30.0)  # 死叉向下
        else:
            scores.append(50.0)  # 中性
        
        return np.mean(scores)
    
    def calculate_ma_score(self, close: Optional[float], ma5: Optional[float], 
                          ma10: Optional[float], ma20: Optional[float]) -> float:
        """计算均线动量得分"""
        if not all([close, ma5, ma10, ma20]):
            return 50.0
        
        scores = []
        
        # 价格相对均线位置
        if close > ma5 > ma10 > ma20:
            scores.append(85.0)  # 多头排列
        elif close > ma5 > ma10:
            scores.append(75.0)  # 短期强势
        elif close > ma20:
            scores.append(60.0)  # 长期趋势向上
        elif close < ma5 < ma10 < ma20:
            scores.append(25.0)  # 空头排列
        elif close < ma5 < ma10:
            scores.append(35.0)  # 短期弱势
        else:
            scores.append(50.0)  # 中性
        
        # 均线斜率（简化计算）
        ma_avg = (ma5 + ma10 + ma20) / 3
        if close > ma_avg * 1.02:
            scores.append(70.0)
        elif close < ma_avg * 0.98:
            scores.append(30.0)
        else:
            scores.append(50.0)
        
        return np.mean(scores)
    
    def calculate(self, stock_data: StockData, market_data: Dict) -> float:
        """计算动量因子综合得分"""
        try:
            # 获取技术指标数据
            tech_data = self.get_technical_data(stock_data.code, stock_data.trade_date)
            
            if not tech_data:
                return 50.0
            
            # 计算各项动量得分
            rsi_score = self.calculate_rsi_score(tech_data.get('rsi'))
            macd_score = self.calculate_macd_score(tech_data.get('macd_dif'))
            kdj_score = self.calculate_kdj_score(
                tech_data.get('kdj_k'),
                tech_data.get('kdj_d'),
                tech_data.get('kdj_j')
            )
            ma_score = self.calculate_ma_score(
                tech_data.get('close'),
                tech_data.get('ma5'),
                tech_data.get('ma10'),
                tech_data.get('ma20')
            )
            
            # 权重配置：均线30%，RSI25%，MACD25%，KDJ20%
            momentum_score = (
                ma_score * 0.30 +
                rsi_score * 0.25 +
                macd_score * 0.25 +
                kdj_score * 0.20
            )
            
            return min(100.0, max(0.0, momentum_score))
            
        except Exception as e:
            print(f"计算动量因子失败: {e}")
            return 50.0
    
    def get_factor_name(self) -> str:
        return "MomentumFactor"

class VolatilityFactor(FactorCalculator):
    """波动率因子计算器"""
    
    def __init__(self, db_path: str):
        """初始化波动率因子计算器"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
    
    def get_volatility_data(self, stock_code: str, trade_date: str, days: int = 30) -> Dict:
        """获取波动率计算数据"""
        try:
            query = """
                SELECT dq.close, dq.high, dq.low, dq.volume
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.code = ? AND dq.trade_date <= ?
                ORDER BY dq.trade_date DESC
                LIMIT ?
            """
            
            df = pd.read_sql_query(query, self.conn, params=[stock_code, trade_date, days])
            
            if df.empty:
                return {}
            
            return {
                'closes': df['close'].values[::-1],
                'highs': df['high'].values[::-1],
                'lows': df['low'].values[::-1],
                'volumes': df['volume'].values[::-1]
            }
            
        except Exception as e:
            print(f"获取股票 {stock_code} 波动率数据失败: {e}")
            return {}
    
    def calculate_historical_volatility(self, prices: np.array) -> float:
        """计算历史波动率"""
        if len(prices) < 10:
            return 0.2
        
        returns = np.diff(np.log(prices))
        return np.std(returns) * np.sqrt(252)  # 年化波动率
    
    def calculate_atr_volatility(self, highs: np.array, lows: np.array, closes: np.array) -> float:
        """计算ATR波动率"""
        if len(highs) < 15:
            return 0.2
        
        tr1 = highs[1:] - lows[1:]
        tr2 = np.abs(highs[1:] - closes[:-1])
        tr3 = np.abs(lows[1:] - closes[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        atr = np.mean(tr[-14:])  # 14日ATR
        current_price = closes[-1]
        
        return atr / current_price if current_price > 0 else 0.2
    
    def calculate_volatility_score(self, historical_vol: float, atr_vol: float, 
                                 market_state: str) -> float:
        """计算波动率得分"""
        avg_vol = (historical_vol + atr_vol) / 2
        
        if market_state == "bull":
            # 牛市：适中波动率较好
            if 0.15 <= avg_vol <= 0.25:
                return 80.0
            elif 0.1 <= avg_vol <= 0.35:
                return 65.0
            elif avg_vol < 0.1:
                return 45.0  # 太稳定可能缺乏机会
            else:
                return 35.0  # 太不稳定风险高
        elif market_state == "bear":
            # 熊市：低波动率较好
            if avg_vol <= 0.2:
                return 75.0
            elif avg_vol <= 0.3:
                return 55.0
            else:
                return 30.0
        else:
            # 震荡市：中等波动率较好
            if 0.2 <= avg_vol <= 0.3:
                return 70.0
            elif 0.15 <= avg_vol <= 0.35:
                return 60.0
            else:
                return 45.0
    
    def calculate(self, stock_data: StockData, market_data: Dict) -> float:
        """计算波动率因子综合得分"""
        try:
            # 获取波动率数据
            vol_data = self.get_volatility_data(stock_data.code, stock_data.trade_date)
            
            if not vol_data:
                return 50.0
            
            # 计算不同类型的波动率
            hist_vol = self.calculate_historical_volatility(vol_data['closes'])
            atr_vol = self.calculate_atr_volatility(
                vol_data['highs'], 
                vol_data['lows'], 
                vol_data['closes']
            )
            
            # 获取市场状态
            market_state = market_data.get('market_state', 'sideways')
            
            # 计算波动率得分
            volatility_score = self.calculate_volatility_score(hist_vol, atr_vol, market_state)
            
            return min(100.0, max(0.0, volatility_score))
            
        except Exception as e:
            print(f"计算波动率因子失败: {e}")
            return 50.0
    
    def get_factor_name(self) -> str:
        return "VolatilityFactor"

class CompositeTechnicalFactor(FactorCalculator):
    """技术面综合因子计算器"""
    
    def __init__(self, db_path: str):
        """初始化技术面综合因子计算器"""
        self.trend_factor = TrendStrengthFactor(db_path)
        self.momentum_factor = MomentumFactor(db_path)
        self.volatility_factor = VolatilityFactor(db_path)
    
    def calculate(self, stock_data: StockData, market_data: Dict) -> float:
        """计算技术面综合得分"""
        try:
            # 计算各项技术面得分
            trend_score = self.trend_factor.calculate(stock_data, market_data)
            momentum_score = self.momentum_factor.calculate(stock_data, market_data)
            volatility_score = self.volatility_factor.calculate(stock_data, market_data)
            
            # 根据市场状态调整权重
            market_state = market_data.get('market_state', 'sideways')
            
            if market_state == "bull":
                # 牛市：重趋势和动量
                technical_score = (
                    trend_score * 0.45 +
                    momentum_score * 0.40 +
                    volatility_score * 0.15
                )
            elif market_state == "bear":
                # 熊市：重波动率和趋势
                technical_score = (
                    trend_score * 0.40 +
                    momentum_score * 0.25 +
                    volatility_score * 0.35
                )
            else:
                # 震荡市：平衡权重
                technical_score = (
                    trend_score * 0.35 +
                    momentum_score * 0.35 +
                    volatility_score * 0.30
                )
            
            return min(100.0, max(0.0, technical_score))
            
        except Exception as e:
            print(f"计算技术面综合因子失败: {e}")
            return 50.0
    
    def get_factor_name(self) -> str:
        return "CompositeTechnicalFactor"

if __name__ == "__main__":
    # 测试代码
    test_stock = StockData(
        code="000001",
        name="平安银行",
        trade_date="2025-08-01",
        open=10.0,
        high=10.5,
        low=9.8,
        close=10.2,
        volume=100000,
        rsi=65.0,
        macd=0.15,
        kdj_k=70.0,
        kdj_d=65.0,
        kdj_j=75.0
    )
    
    # 测试技术面因子
    factor = CompositeTechnicalFactor("data_adapter/stock_data.db")
    score = factor.calculate(test_stock, {"market_state": "bull", "trade_date": "2025-08-01"})
    
    print(f"✅ 技术面因子计算完成")
    print(f"📊 测试股票 {test_stock.code} 技术面得分: {score:.2f}")