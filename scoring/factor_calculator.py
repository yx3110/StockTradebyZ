#!/usr/bin/env python3
"""
因子计算模块
Factor Calculator Module

基于实际选股表现优化的多因子计算系统
"""

import pandas as pd
import numpy as np
import sqlite3
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class FactorCalculator:
    """因子计算器 - 基于实际数据优化的多因子计算"""
    
    def __init__(self, db_path: str = "data_adapter/stock_data.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
    
    def calculate_momentum_factor(self, stock_code: str, trade_date: str) -> float:
        """
        计算动量因子 (权重40%)
        
        基于实际数据发现：动量识别能力是预测股价的关键
        包含价格动量、成交量动量、技术动量和趋势一致性
        """
        try:
            query = """
                SELECT 
                    dq.close, 
                    dq.volume,
                    dq.trade_date,
                    ti.bbi,
                    ti.rsi6 as rsi,
                    ti.macd_dif as macd, 
                    ti.macd_dea as macds
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                JOIN technical_indicators ti ON ti.security_id = s.id AND ti.trade_date = dq.trade_date
                WHERE s.code = ? AND dq.trade_date <= ?
                ORDER BY dq.trade_date DESC
                LIMIT 20
            """
            
            df = pd.read_sql_query(query, self.conn, params=[stock_code, trade_date])
            
            if len(df) < 10:
                return 50.0
            
            df = df.sort_values('trade_date')
            
            # 1. 价格动量 (30%)
            price_momentum = self._calculate_price_momentum(df)
            
            # 2. 成交量动量 (25%) - 量价配合
            volume_momentum = self._calculate_volume_momentum(df)
            
            # 3. 技术动量 (25%) - MACD等
            tech_momentum = self._calculate_tech_momentum(df)
            
            # 4. 趋势一致性 (20%)
            trend_consistency = self._calculate_trend_consistency(df)
            
            # 综合动量分数
            momentum_score = (
                price_momentum * 0.30 + 
                volume_momentum * 0.25 + 
                tech_momentum * 0.25 + 
                trend_consistency * 0.20
            )
            
            return max(0, min(100, momentum_score))
            
        except Exception as e:
            print(f"计算动量因子失败 {stock_code}: {e}")
            return 50.0
    
    def _calculate_price_momentum(self, df: pd.DataFrame) -> float:
        """计算价格动量"""
        if len(df) < 5:
            return 50.0
        
        # 短期动量 (5日)
        short_return = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) * 100
        
        # 中期动量 (10日)
        if len(df) >= 10:
            medium_return = (df['close'].iloc[-1] / df['close'].iloc[-10] - 1) * 100
        else:
            medium_return = short_return
        
        # 动量加速度
        if len(df) >= 6:
            recent_momentum = (df['close'].iloc[-1] / df['close'].iloc[-3] - 1) * 100
            early_momentum = (df['close'].iloc[-3] / df['close'].iloc[-6] - 1) * 100
            acceleration = recent_momentum - early_momentum
        else:
            acceleration = 0
        
        # 综合评分
        momentum = short_return * 0.4 + medium_return * 0.4 + acceleration * 0.2
        
        # 转换为0-100分
        return 50 + np.tanh(momentum / 10) * 40
    
    def _calculate_volume_momentum(self, df: pd.DataFrame) -> float:
        """计算成交量动量"""
        if len(df) < 5:
            return 50.0
        
        # 近期成交量 vs 历史平均
        recent_volume = df['volume'].tail(3).mean()
        avg_volume = df['volume'].mean()
        
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        # 量价配合度
        price_change = df['close'].iloc[-1] / df['close'].iloc[-5] - 1
        volume_score = 50
        
        if price_change > 0 and volume_ratio > 1.2:  # 上涨配合放量
            volume_score = 70 + min(30, (volume_ratio - 1) * 30)
        elif price_change < 0 and volume_ratio < 0.8:  # 下跌伴随缩量
            volume_score = 30 + max(-20, volume_ratio * 25)
        
        return volume_score
    
    def _calculate_tech_momentum(self, df: pd.DataFrame) -> float:
        """计算技术指标动量"""
        if len(df) < 5:
            return 50.0
        
        score = 50
        
        # RSI动量
        current_rsi = df['rsi'].iloc[-1] if pd.notna(df['rsi'].iloc[-1]) else 50
        if 30 < current_rsi < 70:  # RSI在健康区间
            score += 10
        elif current_rsi < 30:  # 超卖
            score += 15
        elif current_rsi > 80:  # 超买
            score -= 10
        
        # MACD动量
        current_macd = df['macd'].iloc[-1] if pd.notna(df['macd'].iloc[-1]) else 0
        current_signal = df['macds'].iloc[-1] if pd.notna(df['macds'].iloc[-1]) else 0
        
        if current_macd > current_signal and current_macd > 0:  # 金叉且在零轴上方
            score += 20
        elif current_macd > current_signal:  # 仅金叉
            score += 10
        elif current_macd < current_signal and current_macd < 0:  # 死叉且在零轴下方
            score -= 15
        
        return max(0, min(100, score))
    
    def _calculate_trend_consistency(self, df: pd.DataFrame) -> float:
        """计算趋势一致性"""
        if len(df) < 5:
            return 50.0
        
        # 使用BBI和价格趋势
        current_price = df['close'].iloc[-1]
        bbi_current = df['bbi'].iloc[-1] if pd.notna(df['bbi'].iloc[-1]) else current_price
        
        # 计算价格趋势
        if len(df) >= 3:
            price_trend = (df['close'].iloc[-1] / df['close'].iloc[-3] - 1) * 100
            bbi_trend = (df['bbi'].iloc[-1] / df['bbi'].iloc[-3] - 1) * 100 if pd.notna(df['bbi'].iloc[-1]) and pd.notna(df['bbi'].iloc[-3]) else 0
        else:
            price_trend = 0
            bbi_trend = 0
        
        score = 50
        
        # 价格与BBI关系
        if current_price > bbi_current:
            score += 15  # 价格在BBI上方
        else:
            score -= 10  # 价格在BBI下方
        
        # 趋势一致性
        if price_trend > 1 and bbi_trend > 0:  # 双重上升
            score += 20
        elif price_trend < -1 and bbi_trend < 0:  # 双重下跌
            score -= 15
        elif abs(price_trend) < 1 and abs(bbi_trend) < 1:  # 横盘整理
            score += 5
        
        return max(0, min(100, score))
    
    def calculate_mean_reversion_factor(self, stock_code: str, trade_date: str) -> float:
        """
        计算均值回归因子 (权重25%)
        
        识别超跌反弹和高位回调机会
        """
        try:
            query = """
                SELECT 
                    dq.close, dq.high, dq.low,
                    ti.rsi6 as rsi,
                    ti.boll_upper, ti.boll_middle, ti.boll_lower
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                JOIN technical_indicators ti ON ti.security_id = s.id AND ti.trade_date = dq.trade_date
                WHERE s.code = ? AND dq.trade_date <= ?
                ORDER BY dq.trade_date DESC
                LIMIT 20
            """
            
            df = pd.read_sql_query(query, self.conn, params=[stock_code, trade_date])
            
            if len(df) < 10:
                return 50.0
            
            # 1. 相对历史位置
            current_price = df['close'].iloc[0]
            high_20 = df['high'].max()
            low_20 = df['low'].min()
            
            position_pct = (current_price - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5
            
            # 2. RSI背离
            rsi_current = df['rsi'].iloc[0] if pd.notna(df['rsi'].iloc[0]) else 50
            
            # 3. 布林带位置
            boll_upper = df['boll_upper'].iloc[0] if pd.notna(df['boll_upper'].iloc[0]) else current_price * 1.02
            boll_lower = df['boll_lower'].iloc[0] if pd.notna(df['boll_lower'].iloc[0]) else current_price * 0.98
            
            boll_position = (current_price - boll_lower) / (boll_upper - boll_lower) if boll_upper > boll_lower else 0.5
            
            # 回归评分
            reversion_score = 50
            
            # 超卖区域 (潜在反弹)
            if position_pct < 0.2 and rsi_current < 35:
                reversion_score = 75
            elif position_pct < 0.3 and rsi_current < 40:
                reversion_score = 65
            # 超买区域 (潜在回调)
            elif position_pct > 0.8 and rsi_current > 70:
                reversion_score = 25
            elif position_pct > 0.7 and rsi_current > 65:
                reversion_score = 35
            
            # 布林带调整
            if boll_position < 0.1:  # 接近下轨
                reversion_score += 10
            elif boll_position > 0.9:  # 接近上轨
                reversion_score -= 10
            
            return max(0, min(100, reversion_score))
            
        except Exception as e:
            print(f"计算均值回归因子失败 {stock_code}: {e}")
            return 50.0
    
    def calculate_volume_breakout_factor(self, stock_code: str, trade_date: str) -> float:
        """
        计算成交量突破因子 (权重20%)
        
        识别放量突破和缩量整理
        """
        try:
            query = """
                SELECT 
                    dq.close, dq.volume, dq.high, dq.low,
                    dq.trade_date
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.code = ? AND dq.trade_date <= ?
                ORDER BY dq.trade_date DESC
                LIMIT 15
            """
            
            df = pd.read_sql_query(query, self.conn, params=[stock_code, trade_date])
            
            if len(df) < 10:
                return 50.0
            
            df = df.sort_values('trade_date')
            
            # 1. 成交量突破
            recent_volume = df['volume'].tail(3).mean()
            avg_volume = df['volume'].head(10).mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # 2. 价格变化
            price_change = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) * 100
            
            # 3. 突破有效性
            breakout_score = 50
            
            # 放量上涨突破
            if price_change > 2 and volume_ratio > 1.5:
                breakout_score = 80
            elif price_change > 1 and volume_ratio > 1.2:
                breakout_score = 70
            elif price_change < -2 and volume_ratio > 1.5:  # 放量下跌
                breakout_score = 20
            
            # 缩量整理 (蓄势)
            if abs(price_change) < 1 and volume_ratio < 0.8:
                breakout_score = 60  # 轻微加分，等待突破
            
            return max(0, min(100, breakout_score))
            
        except Exception as e:
            print(f"计算成交量突破因子失败 {stock_code}: {e}")
            return 50.0
    
    def calculate_relative_performance_factor(self, stock_code: str, trade_date: str) -> float:
        """
        计算相对表现因子 (权重10%)
        
        相对大盘和行业的表现
        """
        try:
            # 获取股票收益率
            stock_query = """
                SELECT close
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.code = ? AND dq.trade_date <= ?
                ORDER BY dq.trade_date DESC
                LIMIT 10
            """
            
            stock_df = pd.read_sql_query(stock_query, self.conn, params=[stock_code, trade_date])
            
            if len(stock_df) < 5:
                return 50.0
            
            stock_return = (stock_df['close'].iloc[0] / stock_df['close'].iloc[-1] - 1) * 100
            
            # 获取大盘收益率 (使用上证指数作为基准)
            market_query = """
                SELECT close
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.code = '000001' AND dq.trade_date <= ?
                ORDER BY dq.trade_date DESC
                LIMIT 10
            """
            
            market_df = pd.read_sql_query(market_query, self.conn, params=[trade_date])
            
            if len(market_df) >= 5:
                market_return = (market_df['close'].iloc[0] / market_df['close'].iloc[-1] - 1) * 100
                relative_return = stock_return - market_return
                
                # 相对强度评分
                if relative_return > 5:
                    return 80
                elif relative_return > 2:
                    return 70
                elif relative_return > 0:
                    return 60
                elif relative_return > -2:
                    return 40
                else:
                    return 20
            
            return 50.0
            
        except Exception as e:
            print(f"计算相对表现因子失败 {stock_code}: {e}")
            return 50.0
    
    def calculate_stability_factor(self, stock_code: str, trade_date: str) -> float:
        """
        计算稳定性因子 (权重5%)
        
        价格波动稳定性和基本面健康度
        """
        try:
            query = """
                SELECT 
                    dq.close, dq.high, dq.low,
                    db.pe_ttm, db.turnover_rate
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                LEFT JOIN daily_basic db ON db.security_id = s.id AND db.trade_date = dq.trade_date
                WHERE s.code = ? AND dq.trade_date <= ?
                ORDER BY dq.trade_date DESC
                LIMIT 10
            """
            
            df = pd.read_sql_query(query, self.conn, params=[stock_code, trade_date])
            
            if len(df) < 5:
                return 50.0
            
            # 1. 价格波动率
            returns = df['close'].pct_change().dropna()
            if len(returns) > 2:
                volatility = returns.std() * np.sqrt(250) * 100  # 年化波动率
            else:
                volatility = 25.0
            
            # 2. 换手率健康度
            avg_turnover = df['turnover_rate'].mean() if df['turnover_rate'].notna().any() else 5.0
            
            # 3. PE合理性
            current_pe = df['pe_ttm'].iloc[0] if pd.notna(df['pe_ttm'].iloc[0]) else 25.0
            
            stability_score = 50
            
            # 波动率评分 (波动率适中更好)
            if 15 < volatility < 35:
                stability_score += 10
            elif volatility > 50:
                stability_score -= 15
            
            # 换手率评分
            if 2 < avg_turnover < 8:
                stability_score += 5
            elif avg_turnover > 15:
                stability_score -= 10
            
            # PE评分
            if 10 < current_pe < 30:
                stability_score += 5
            elif current_pe > 100 or current_pe < 0:
                stability_score -= 10
            
            return max(0, min(100, stability_score))
            
        except Exception as e:
            print(f"计算稳定性因子失败 {stock_code}: {e}")
            return 50.0
    
    def calculate_all_factors(self, stock_code: str, trade_date: str) -> Dict[str, float]:
        """计算所有因子分数"""
        return {
            'momentum': self.calculate_momentum_factor(stock_code, trade_date),
            'mean_reversion': self.calculate_mean_reversion_factor(stock_code, trade_date),
            'volume_breakout': self.calculate_volume_breakout_factor(stock_code, trade_date),
            'relative_performance': self.calculate_relative_performance_factor(stock_code, trade_date),
            'stability': self.calculate_stability_factor(stock_code, trade_date)
        }
    
    def __del__(self):
        """清理数据库连接"""
        if hasattr(self, 'conn'):
            self.conn.close()