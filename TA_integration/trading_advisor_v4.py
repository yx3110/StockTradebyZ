#!/usr/bin/env python3
"""
TradingAgents完整交易建议系统 v4
明确区分做空和卖出持仓的逻辑
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os

# 添加项目路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

class TechnicalAnalyzer:
    """技术分析器"""
    
    def __init__(self):
        self.data_dir = "/Users/yangxu/StockTradebyZ/full_securities_data"
    
    def get_stock_data(self, stock_code: str, days: int = 100) -> Optional[pd.DataFrame]:
        """获取股票数据"""
        try:
            # 查找股票数据文件
            stock_files = [
                f"{stock_code}_A股.csv",
                f"{stock_code}_ETF.csv", 
                f"{stock_code}_基金.csv"
            ]
            
            for filename in stock_files:
                filepath = os.path.join(self.data_dir, filename)
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date').tail(days)
                    return df
            
            print(f"未找到{stock_code}的数据文件")
            return None
            
        except Exception as e:
            print(f"读取{stock_code}数据失败: {e}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """计算技术指标"""
        if df is None or len(df) < 20:
            return {}
        
        try:
            # 基础价格信息
            current_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2] if len(df) > 1 else current_price
            
            # 移动平均线
            ma5 = df['close'].rolling(5).mean().iloc[-1] if len(df) >= 5 else current_price
            ma10 = df['close'].rolling(10).mean().iloc[-1] if len(df) >= 10 else current_price
            ma20 = df['close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else current_price
            ma60 = df['close'].rolling(60).mean().iloc[-1] if len(df) >= 60 else current_price
            
            # RSI
            rsi = self._calculate_rsi(df['close']) if len(df) >= 14 else 50
            
            # MACD
            macd_line, signal_line, histogram = self._calculate_macd(df['close'])
            
            # KDJ
            k_percent, d_percent = self._calculate_kdj(df)
            
            # 布林带
            upper_band, middle_band, lower_band = self._calculate_bollinger_bands(df['close'])
            
            # 成交量分析
            volume_ma5 = df['volume'].rolling(5).mean().iloc[-1] if len(df) >= 5 else df['volume'].iloc[-1]
            volume_ratio = df['volume'].iloc[-1] / volume_ma5 if volume_ma5 > 0 else 1
            
            # 价格变化
            price_change = (current_price - prev_price) / prev_price * 100
            
            # ATR (Average True Range)
            atr = self._calculate_atr(df) if len(df) >= 14 else 0
            
            return {
                'current_price': round(current_price, 2),
                'price_change_pct': round(price_change, 2),
                'ma5': round(ma5, 2),
                'ma10': round(ma10, 2), 
                'ma20': round(ma20, 2),
                'ma60': round(ma60, 2),
                'rsi': round(rsi, 2),
                'macd_line': round(macd_line, 4),
                'macd_signal': round(signal_line, 4),
                'macd_histogram': round(histogram, 4),
                'kdj_k': round(k_percent, 2),
                'kdj_d': round(d_percent, 2),
                'boll_upper': round(upper_band, 2),
                'boll_middle': round(middle_band, 2),
                'boll_lower': round(lower_band, 2),
                'volume_ratio': round(volume_ratio, 2),
                'atr': round(atr, 2)
            }
            
        except Exception as e:
            print(f"计算技术指标失败: {e}")
            return {}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """计算MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return (
            macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0,
            signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else 0,
            histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0
        )
    
    def _calculate_kdj(self, df: pd.DataFrame, period: int = 9) -> Tuple[float, float]:
        """计算KDJ"""
        high_max = df['high'].rolling(period).max()
        low_min = df['low'].rolling(period).min()
        
        rsv = (df['close'] - low_min) / (high_max - low_min) * 100
        k_percent = rsv.ewm(alpha=1/3).mean()
        d_percent = k_percent.ewm(alpha=1/3).mean()
        
        return (
            k_percent.iloc[-1] if not pd.isna(k_percent.iloc[-1]) else 50,
            d_percent.iloc[-1] if not pd.isna(d_percent.iloc[-1]) else 50
        )
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: float = 2) -> Tuple[float, float, float]:
        """计算布林带"""
        middle_band = prices.rolling(period).mean()
        std_dev = prices.rolling(period).std()
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        return (
            upper_band.iloc[-1] if not pd.isna(upper_band.iloc[-1]) else prices.iloc[-1],
            middle_band.iloc[-1] if not pd.isna(middle_band.iloc[-1]) else prices.iloc[-1],
            lower_band.iloc[-1] if not pd.isna(lower_band.iloc[-1]) else prices.iloc[-1]
        )
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """计算ATR (Average True Range)"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0

class TradingSignalGenerator:
    """交易信号生成器"""
    
    def generate_trading_signal(self, technical_data: Dict, sentiment_data: Dict, stock_info: Dict) -> Dict:
        """生成交易信号"""
        if not technical_data:
            return self._generate_fallback_signal(sentiment_data, stock_info)
        
        # 技术面信号
        tech_signal = self._analyze_technical_signals(technical_data)
        
        # 情绪面信号  
        sentiment_signal = self._analyze_sentiment_signals(sentiment_data)
        
        # 综合信号
        combined_signal = self._combine_signals(tech_signal, sentiment_signal)
        
        # 价格目标计算（区分买入、卖出持仓、做空）
        price_targets = self._calculate_price_targets_v4(technical_data, combined_signal)
        
        # 风险评估
        risk_assessment = self._assess_risk(technical_data, sentiment_data, combined_signal)
        
        return {
            'stock_code': stock_info.get('code', ''),
            'stock_name': stock_info.get('name', ''),
            'current_price': technical_data.get('current_price', 0),
            'signal': combined_signal['action'],
            'signal_strength': combined_signal['strength'],
            'confidence': combined_signal['confidence'],
            'technical_score': tech_signal['score'],
            'sentiment_score': sentiment_signal['score'],
            'price_targets': price_targets,
            'risk_assessment': risk_assessment,
            'reasoning': combined_signal['reasoning'],
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _analyze_technical_signals(self, tech_data: Dict) -> Dict:
        """增强版技术信号分析 - 包含KDJ和阻力位分析"""
        signals = []
        score = 0
        
        current_price = tech_data.get('current_price', 0)
        ma5 = tech_data.get('ma5', current_price)
        ma10 = tech_data.get('ma10', current_price)
        ma20 = tech_data.get('ma20', current_price)
        ma60 = tech_data.get('ma60', current_price)
        
        # 1. 移动平均线信号（加强分析）
        if current_price > ma5 > ma10 > ma20 > ma60:
            signals.append("完美多头排列")
            score += 3
        elif current_price > ma5 > ma10 > ma20:
            signals.append("多头排列")
            score += 2
        elif current_price < ma5 < ma10 < ma20 < ma60:
            signals.append("完美空头排列")
            score -= 3
        elif current_price < ma5 < ma10 < ma20:
            signals.append("空头排列")
            score -= 2
        elif current_price > ma5:
            signals.append("价格在短期均线上方")
            score += 1
        else:
            signals.append("价格在短期均线下方")
            score -= 1
        
        # 2. RSI信号
        rsi = tech_data.get('rsi', 50)
        if rsi > 80:
            signals.append(f"RSI严重超买({rsi:.1f})")
            score -= 2
        elif rsi > 70:
            signals.append(f"RSI超买({rsi:.1f})")
            score -= 1
        elif rsi < 20:
            signals.append(f"RSI严重超卖({rsi:.1f})")
            score += 2
        elif rsi < 30:
            signals.append(f"RSI超卖({rsi:.1f})")
            score += 1
        
        # 3. MACD信号（增强分析）
        macd_line = tech_data.get('macd_line', 0)
        macd_signal = tech_data.get('macd_signal', 0)
        macd_histogram = tech_data.get('macd_histogram', 0)
        
        if macd_line > macd_signal and macd_histogram > 0:
            if macd_histogram > 0.01:  # 强势金叉
                signals.append("MACD强势金叉")
                score += 2
            else:
                signals.append("MACD金叉")
                score += 1
        elif macd_line < macd_signal and macd_histogram < 0:
            if macd_histogram < -0.01:  # 强势死叉
                signals.append("MACD强势死叉")
                score -= 2
            else:
                signals.append("MACD死叉")
                score -= 1
        
        # 4. KDJ信号（详细分析）
        kdj_k = tech_data.get('kdj_k', 50)
        kdj_d = tech_data.get('kdj_d', 50)
        kdj_j = 3 * kdj_k - 2 * kdj_d  # 计算J值
        
        # KDJ位置分析
        if kdj_k > 80 and kdj_d > 80:
            signals.append(f"KDJ超买区({kdj_k:.1f})")
            score -= 1.5
        elif kdj_k < 20 and kdj_d < 20:
            signals.append(f"KDJ超卖区({kdj_k:.1f})")
            score += 1.5
        
        # KDJ交叉分析
        if kdj_k > kdj_d:
            if kdj_k < 50:  # 在低位金叉
                signals.append("KDJ低位金叉")
                score += 1
            else:
                signals.append("KDJ金叉")
                score += 0.5
        else:
            if kdj_k > 50:  # 在高位死叉
                signals.append("KDJ高位死叉")
                score -= 1
            else:
                signals.append("KDJ死叉")
                score -= 0.5
        
        # 5. 布林带支撑阻力分析
        boll_upper = tech_data.get('boll_upper', current_price * 1.1)
        boll_middle = tech_data.get('boll_middle', current_price)
        boll_lower = tech_data.get('boll_lower', current_price * 0.9)
        
        # 布林带位置分析
        boll_position = (current_price - boll_lower) / (boll_upper - boll_lower) if (boll_upper - boll_lower) > 0 else 0.5
        
        if current_price > boll_upper:
            signals.append("突破布林上轨")
            score += 0.5
        elif current_price < boll_lower:
            signals.append("跌破布林下轨")
            score -= 0.5
        elif boll_position > 0.8:
            signals.append("接近布林上轨")
            score -= 0.5  # 可能遇到阻力
        elif boll_position < 0.2:
            signals.append("接近布林下轨")
            score += 0.5  # 可能获得支撑
        
        # 6. 成交量信号
        volume_ratio = tech_data.get('volume_ratio', 1)
        if volume_ratio > 3:
            signals.append("异常放量")
            score += 1
        elif volume_ratio > 2:
            signals.append("放量")
            score += 0.5
        elif volume_ratio < 0.5:
            signals.append("缩量")
            score -= 0.5
        
        # 7. 阻力位和支撑位分析
        resistance_support = self._analyze_resistance_support(tech_data)
        if resistance_support:
            signals.extend(resistance_support['signals'])
            score += resistance_support['score']
        
        return {
            'score': score,
            'signals': signals,
            'kdj_analysis': {
                'k': kdj_k,
                'd': kdj_d,
                'j': kdj_j,
                'position': 'high' if kdj_k > 80 else ('low' if kdj_k < 20 else 'middle')
            },
            'boll_position': boll_position
        }
    
    def _analyze_resistance_support(self, tech_data: Dict) -> Dict:
        """分析阻力位和支撑位"""
        signals = []
        score = 0
        
        current_price = tech_data.get('current_price', 0)
        ma5 = tech_data.get('ma5', current_price)
        ma10 = tech_data.get('ma10', current_price)
        ma20 = tech_data.get('ma20', current_price)
        ma60 = tech_data.get('ma60', current_price)
        boll_upper = tech_data.get('boll_upper', current_price * 1.1)
        boll_lower = tech_data.get('boll_lower', current_price * 0.9)
        
        # 均线作为支撑阻力
        price_ma_distances = [
            ('MA5', abs(current_price - ma5) / current_price, ma5),
            ('MA10', abs(current_price - ma10) / current_price, ma10),
            ('MA20', abs(current_price - ma20) / current_price, ma20),
            ('MA60', abs(current_price - ma60) / current_price, ma60)
        ]
        
        # 寻找最近的均线作为支撑或阻力
        nearby_ma = min(price_ma_distances, key=lambda x: x[1])
        if nearby_ma[1] < 0.02:  # 距离均线2%以内
            ma_name, distance, ma_value = nearby_ma
            if current_price > ma_value:
                signals.append(f"接近{ma_name}支撑")
                score += 0.5
            else:
                signals.append(f"面临{ma_name}阻力")
                score -= 0.5
        
        # 布林带作为动态支撑阻力
        if abs(current_price - boll_upper) / current_price < 0.02:
            signals.append("接近布林上轨阻力")
            score -= 0.5
        elif abs(current_price - boll_lower) / current_price < 0.02:
            signals.append("接近布林下轨支撑")
            score += 0.5
        
        # 整数关口分析
        price_int = int(current_price)
        if abs(current_price - price_int) / current_price < 0.01:
            if price_int % 5 == 0 or price_int % 10 == 0:
                signals.append(f"面临{price_int}元整数关口")
                # 整数关口通常是阻力位
                score -= 0.3
        
        return {
            'signals': signals,
            'score': score
        }
    
    def _analyze_sentiment_signals(self, sentiment_data: Dict) -> Dict:
        """分析情绪信号"""
        if not sentiment_data or 'avg_sentiment' not in sentiment_data:
            return {'score': 0, 'signals': ["无情绪数据"]}
        
        signals = []
        score = 0
        
        avg_sentiment = sentiment_data.get('avg_sentiment', 0)
        total_posts = sentiment_data.get('total_posts', 0)
        filtered_posts = sentiment_data.get('filtered_posts', 0)
        
        # 调整情绪评分标准
        if avg_sentiment > 0.2:
            signals.append("市场情绪积极")
            score += 2
        elif avg_sentiment > 0:
            signals.append("市场情绪偏正面")
            score += 1
        elif avg_sentiment > -0.2:
            signals.append("市场情绪中性偏负")
            score -= 1
        else:
            signals.append("市场情绪悲观")
            score -= 2
        
        # 考虑数据量（调整标准）
        if filtered_posts >= 30:
            signals.append(f"讨论热度高({filtered_posts}条)")
            score += 0.5
        elif filtered_posts >= 10:
            signals.append(f"讨论热度中等({filtered_posts}条)")
            # 不加分也不减分
        elif filtered_posts >= 3:
            signals.append(f"讨论热度低({filtered_posts}条)")
            score -= 0.3
        else:
            signals.append(f"几乎无讨论({filtered_posts}条)")
            score -= 0.5
        
        return {
            'score': score,
            'signals': signals
        }
    
    def _combine_signals(self, tech_signal: Dict, sentiment_signal: Dict) -> Dict:
        """综合信号分析"""
        tech_score = tech_signal['score']
        sentiment_score = sentiment_signal['score']
        
        # 加权综合得分
        combined_score = tech_score * 0.7 + sentiment_score * 0.3
        
        # 生成交易建议（区分买入新仓、卖出平仓、避让观望）
        if combined_score > 1:
            action = "BUY"  # 建议买入建仓
            strength = min(10, int(combined_score + 5))
            confidence = "HIGH" if combined_score > 3 else "MEDIUM"
        elif combined_score < -1:
            action = "AVOID"  # 建议避让（不建议买入，持有者可考虑卖出）
            strength = min(10, int(abs(combined_score) + 5))
            confidence = "HIGH" if combined_score < -3 else "MEDIUM"
        else:
            action = "HOLD"  # 观望
            strength = 5
            confidence = "LOW"
        
        # 生成推理说明（优先显示关键技术信号）
        tech_signals = tech_signal['signals']
        
        # 优先显示KDJ和阻力位相关信号
        priority_signals = []
        other_signals = []
        
        for signal in tech_signals:
            if any(keyword in signal for keyword in ['KDJ', '阻力', '支撑', '关口', '布林']):
                priority_signals.append(signal)
            else:
                other_signals.append(signal)
        
        # 合并信号，优先级信号在前，最多显示6个
        display_signals = (priority_signals + other_signals)[:6]
        tech_reasoning = '; '.join(display_signals)
        
        sentiment_reasoning = '; '.join(sentiment_signal['signals'])
        reasoning = f"技术面: {tech_reasoning}; 情绪面: {sentiment_reasoning}"
        
        return {
            'action': action,
            'strength': strength,
            'confidence': confidence,
            'reasoning': reasoning
        }
    
    def _calculate_price_targets_v4(self, tech_data: Dict, signal: Dict) -> Dict:
        """V4版价格目标计算 - 更清晰的逻辑"""
        current_price = tech_data.get('current_price', 0)
        if current_price == 0:
            return {}
        
        # 获取技术指标
        atr = tech_data.get('atr', current_price * 0.02)  # 如果没有ATR，使用2%作为默认值
        boll_upper = tech_data.get('boll_upper', current_price * 1.1)
        boll_lower = tech_data.get('boll_lower', current_price * 0.9)
        ma20 = tech_data.get('ma20', current_price)
        
        if signal['action'] == 'BUY':
            # 买入建仓信号
            entry_price = current_price
            
            # 止损价：下方止损
            stop_loss = max(
                current_price - 2 * atr,  # 2倍ATR止损
                boll_lower * 0.98,  # 布林带下轨再下2%
                current_price * 0.95  # 最多5%止损
            )
            
            # 目标价：上方目标
            if signal['strength'] >= 8:
                target_price = min(
                    current_price + 4 * atr,  # 4倍ATR目标
                    boll_upper * 1.02,  # 布林带上轨再上2%
                    current_price * 1.15  # 最多15%目标
                )
            else:
                target_price = min(
                    current_price + 3 * atr,  # 3倍ATR目标
                    boll_upper,  # 布林带上轨
                    current_price * 1.10  # 最多10%目标
                )
                
        elif signal['action'] == 'AVOID':
            # 避让信号（适合持有者卖出，不建议新买入）
            # 如果已持有，建议的卖出价格
            entry_price = current_price  # 当前价格卖出
            
            # 目标价：预期会下跌到的位置
            if signal['strength'] >= 8:
                target_price = max(
                    current_price - 4 * atr,
                    boll_lower * 0.98,
                    current_price * 0.85  # 预期最多下跌15%
                )
            else:
                target_price = max(
                    current_price - 3 * atr,
                    boll_lower,
                    current_price * 0.90  # 预期最多下跌10%
                )
            
            # 止损价：如果不卖出，建议的止损位置
            stop_loss = max(
                current_price - 2 * atr,
                boll_lower * 0.98,
                current_price * 0.95
            )
                
        else:  # HOLD
            entry_price = current_price
            target_price = current_price
            stop_loss = current_price * 0.95
        
        # 计算风险收益比
        if signal['action'] == 'BUY':
            risk = abs(entry_price - stop_loss)
            reward = abs(target_price - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
        else:
            # 对于AVOID和HOLD，不计算传统的风险收益比
            risk_reward_ratio = 0
        
        return {
            'entry_price': round(entry_price, 2),
            'target_price': round(target_price, 2),
            'stop_loss': round(stop_loss, 2),
            'risk_reward_ratio': round(risk_reward_ratio, 2),
            'price_comment': self._get_price_comment(signal['action'])
        }
    
    def _get_price_comment(self, action: str) -> str:
        """获取价格说明"""
        if action == 'BUY':
            return "建议买入价格区间"
        elif action == 'AVOID':
            return "建议避让，持有者可考虑卖出"
        else:
            return "建议观望"
    
    def _assess_risk(self, tech_data: Dict, sentiment_data: Dict, signal: Dict) -> Dict:
        """风险评估"""
        risk_score = 0
        risk_factors = []
        
        # 技术面风险
        rsi = tech_data.get('rsi', 50)
        if rsi > 80 or rsi < 20:
            risk_score += 2
            risk_factors.append(f"RSI极端值({rsi:.0f})")
        elif rsi > 70 or rsi < 30:
            risk_score += 1
            risk_factors.append(f"RSI接近极端({rsi:.0f})")
        
        # 价格位置风险
        current_price = tech_data.get('current_price', 0)
        boll_upper = tech_data.get('boll_upper', current_price)
        boll_lower = tech_data.get('boll_lower', current_price)
        
        if current_price > boll_upper:
            risk_score += 1
            risk_factors.append("价格超出布林带上轨")
        elif current_price < boll_lower:
            risk_score += 1
            risk_factors.append("价格跌破布林带下轨")
        
        # 情绪面风险
        if sentiment_data:
            filtered_posts = sentiment_data.get('filtered_posts', 0)
            if filtered_posts < 20:
                risk_score += 1
                risk_factors.append("情绪数据样本不足")
        
        # 信号强度风险
        if signal['confidence'] == 'LOW':
            risk_score += 1
            risk_factors.append("信号置信度低")
        
        if risk_score <= 1:
            risk_level = "LOW"
            position_suggestion = "标准仓位(10-15%)"
        elif risk_score <= 3:
            risk_level = "MEDIUM"
            position_suggestion = "半仓操作(5-10%)"
        else:
            risk_level = "HIGH"
            position_suggestion = "轻仓试探(3-5%)"
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'position_suggestion': position_suggestion
        }
    
    def _generate_fallback_signal(self, sentiment_data: Dict, stock_info: Dict) -> Dict:
        """当没有技术数据时的备用信号"""
        return {
            'stock_code': stock_info.get('code', ''),
            'stock_name': stock_info.get('name', ''),
            'current_price': 0,
            'signal': 'HOLD',
            'signal_strength': 5,
            'confidence': 'LOW',
            'technical_score': 0,
            'sentiment_score': 0,
            'price_targets': {},
            'risk_assessment': {
                'risk_level': 'HIGH',
                'risk_score': 8,
                'risk_factors': ['缺少技术数据'],
                'position_suggestion': '观望'
            },
            'reasoning': '缺少技术分析数据',
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

def generate_trading_report(trading_results: List[Dict]) -> str:
    """生成交易建议报告"""
    # 按信号强度排序
    sorted_results = sorted(trading_results, key=lambda x: x['signal_strength'], reverse=True)
    
    # 分类统计
    buy_signals = [r for r in sorted_results if r['signal'] == 'BUY']
    avoid_signals = [r for r in sorted_results if r['signal'] == 'AVOID']
    hold_signals = [r for r in sorted_results if r['signal'] == 'HOLD']
    
    report_lines = [
        "# 🎯 TradingAgents完整交易建议报告 V4",
        "",
        "## 📊 分析概览",
        f"- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **分析股票**: {len(trading_results)}只",
        f"- **买入建议**: {len(buy_signals)}只",
        f"- **避让建议**: {len(avoid_signals)}只（不建议新买，持有者可考虑卖出）",
        f"- **观望建议**: {len(hold_signals)}只",
        f"- **分析引擎**: TradingAgents + 技术分析 + 情绪分析",
        f"- **版本**: V4 (明确交易逻辑)",
        "",
    ]
    
    # 买入推荐
    if buy_signals:
        report_lines.extend([
            "## 📈 买入推荐（建议新建仓）",
            ""
        ])
        for i, result in enumerate(buy_signals[:10], 1):
            report_lines.extend(format_stock_analysis(result, i))
    
    # 避让建议
    if avoid_signals:
        report_lines.extend([
            "## 🚫 避让建议（不建议买入，持有者可考虑卖出）", 
            ""
        ])
        for i, result in enumerate(avoid_signals[:10], 1):
            report_lines.extend(format_stock_analysis(result, i))
    
    # 观望
    if hold_signals:
        report_lines.extend([
            "## ⏸️ 观望建议",
            ""
        ])
        for i, result in enumerate(hold_signals[:5], 1):
            report_lines.extend(format_stock_analysis(result, i))
    
    # 操作说明和风险提示
    report_lines.extend([
        "",
        "## 📖 操作说明",
        "- **买入推荐(BUY)**: 适合新建仓位，给出买入价格、上涨目标、止损位",
        "- **避让建议(AVOID)**: 不建议新买入，如已持有可考虑在当前价格卖出",
        "- **观望建议(HOLD)**: 暂时观望，等待更明确的信号",
        "",
        "## ⚠️ 风险提示",
        "- 本报告仅供参考，投资有风险",
        "- 严格执行止损纪律，控制风险",
        "- 分散投资，不要集中在单一股票",
        "- 根据自身风险承受能力调整仓位",
        "",
        f"🤖 **报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Generated with [Claude Code](https://claude.ai/code)"
    ])
    
    return "\n".join(report_lines)

def format_stock_analysis(result: Dict, index: int) -> List[str]:
    """格式化单只股票的分析结果"""
    stock_code = result['stock_code']
    stock_name = result['stock_name']
    signal = result['signal']
    strength = result['signal_strength']
    current_price = result['current_price']
    
    lines = [
        f"### {index}. {stock_code} - {stock_name}",
        f"- **当前价格**: ¥{current_price}",
        f"- **操作建议**: {signal}",
        f"- **信号强度**: {strength}/10",
        f"- **置信度**: {result['confidence']}",
        ""
    ]
    
    # 技术指标详情
    tech_score = result.get('technical_score', 0)
    sentiment_score = result.get('sentiment_score', 0)
    lines.extend([
        "**技术面分析**",
        f"- 技术评分: {tech_score:.1f}/10",
        f"- 情绪评分: {sentiment_score:.1f}/10",
        ""
    ])
    
    # 价格目标
    price_targets = result.get('price_targets', {})
    if price_targets:
        price_comment = price_targets.get('price_comment', '')
        
        if signal == 'BUY':
            lines.extend([
                f"**{price_comment}**",
                f"- **建议买入价**: ¥{price_targets.get('entry_price', 0)}",
                f"- **上涨目标价**: ¥{price_targets.get('target_price', 0)}",
                f"- **止损价**: ¥{price_targets.get('stop_loss', 0)}",
            ])
            
            if price_targets.get('risk_reward_ratio', 0) > 0:
                lines.append(f"- **风险收益比**: 1:{price_targets.get('risk_reward_ratio', 0)}")
            lines.append("")
            
        elif signal == 'AVOID':
            lines.extend([
                f"**{price_comment}**",
                f"- **当前价格**: ¥{price_targets.get('entry_price', 0)}",
                f"- **预期下跌目标**: ¥{price_targets.get('target_price', 0)}",
                f"- **持仓止损建议**: ¥{price_targets.get('stop_loss', 0)}",
                ""
            ])
    
    # 风险评估
    risk_assessment = result.get('risk_assessment', {})
    if risk_assessment:
        lines.extend([
            f"- **风险等级**: {risk_assessment.get('risk_level', '')}",
            f"- **仓位建议**: {risk_assessment.get('position_suggestion', '')}",
        ])
        
        risk_factors = risk_assessment.get('risk_factors', [])
        if risk_factors:
            lines.append(f"- **风险因素**: {', '.join(risk_factors)}")
        
        lines.append("")
    
    # 分析依据
    reasoning = result.get('reasoning', '')
    if reasoning:
        lines.extend([
            f"**分析依据**: {reasoning}",
            ""
        ])
    
    lines.append("---")
    lines.append("")
    
    return lines

def main():
    """主函数"""
    print("🚀 TradingAgents完整交易建议系统 V4")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 初始化组件
    tech_analyzer = TechnicalAnalyzer()
    signal_generator = TradingSignalGenerator()
    
    # 读取情绪分析结果
    sentiment_file = f"/Users/yangxu/StockTradebyZ/reports/sentiment_analysis/分析结果_{datetime.now().strftime('%Y%m%d')}.json"
    
    try:
        with open(sentiment_file, 'r', encoding='utf-8') as f:
            sentiment_results = json.load(f)
        print(f"📊 读取情绪分析结果: {len(sentiment_results)}只股票")
    except FileNotFoundError:
        print(f"❌ 未找到情绪分析结果文件: {sentiment_file}")
        return
    
    # 处理每只股票
    trading_results = []
    processed = 0
    
    for stock_code, sentiment_data in sentiment_results.items():
        if 'error' in sentiment_data:
            continue
        
        processed += 1
        stock_info = sentiment_data.get('stock_info', {})
        stock_name = stock_info.get('name', stock_code)
        
        print(f"🔍 分析 {processed}. {stock_code} - {stock_name}")
        
        # 获取技术数据
        df = tech_analyzer.get_stock_data(stock_code)
        technical_data = tech_analyzer.calculate_technical_indicators(df)
        
        # 生成交易信号
        trading_result = signal_generator.generate_trading_signal(
            technical_data, sentiment_data, stock_info
        )
        
        trading_results.append(trading_result)
        
        # 显示简要结果
        signal = trading_result['signal']
        strength = trading_result['signal_strength']
        price = trading_result['current_price']
        
        print(f"   {signal} (强度:{strength}/10) 当前价:¥{price}")
    
    print(f"\n📊 分析完成，共处理 {len(trading_results)} 只股票")
    
    # 生成报告
    print("📝 生成交易建议报告...")
    trading_report = generate_trading_report(trading_results)
    
    # 确保报告目录存在
    os.makedirs("reports/trading_signals", exist_ok=True)
    
    # 保存报告
    report_file = f"reports/trading_signals/TradingAgents交易建议_{datetime.now().strftime('%Y%m%d')}_v4.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(trading_report)
    
    # 保存JSON数据
    json_file = f"reports/trading_signals/交易信号数据_{datetime.now().strftime('%Y%m%d')}_v4.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(trading_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 交易建议报告已生成!")
    print(f"📄 报告文件: {report_file}")
    print(f"💾 数据文件: {json_file}")
    
    # 统计信息
    buy_count = len([r for r in trading_results if r['signal'] == 'BUY'])
    avoid_count = len([r for r in trading_results if r['signal'] == 'AVOID'])
    hold_count = len([r for r in trading_results if r['signal'] == 'HOLD'])
    
    print(f"\n📈 交易建议统计:")
    print(f"   买入建仓: {buy_count}只")
    print(f"   避让卖出: {avoid_count}只")
    print(f"   观望等待: {hold_count}只")
    
    print(f"\n🎉 分析完成!")

if __name__ == "__main__":
    main()