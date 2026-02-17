#!/usr/bin/env python3
"""
TradingAgents V5 - 增强版交易建议生成器
增加详细的量价关系分析、BBI线分析和布林带分析
"""
import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import requests
import time
import os
import sys

# 添加项目路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# 导入数据源（情绪分析部分）
try:
    from data_sources.sentiment_integrator import ChineseSentimentIntegrator
    from xueqiu_config import get_xueqiu_cookie
except ImportError as e:
    print(f"警告: 无法导入中国市场情绪分析器 - {e}")
    get_xueqiu_cookie = lambda: None
    # 使用模拟情绪数据作为后备方案

class TechnicalAnalyzer:
    """技术分析器 - V5增强版"""
    
    def __init__(self):
        self.name = "TechnicalAnalyzer_V5"
    
    def calculate_bbi(self, prices: List[float], periods: List[int] = [3, 6, 12, 24]) -> float:
        """计算BBI（多空指标）"""
        if len(prices) < max(periods):
            return prices[-1] if prices else 0
        
        mas = []
        for period in periods:
            if len(prices) >= period:
                ma = sum(prices[-period:]) / period
                mas.append(ma)
        
        return sum(mas) / len(mas) if mas else prices[-1]
    
    def analyze_bbi_signals(self, current_price: float, bbi: float, prices: List[float]) -> Dict:
        """分析BBI信号"""
        signals = []
        score = 0
        
        if not prices or len(prices) < 2:
            return {'signals': signals, 'score': score}
        
        # 当前价格与BBI的关系
        bbi_distance = (current_price - bbi) / bbi if bbi > 0 else 0
        
        if current_price > bbi:
            if bbi_distance > 0.05:  # 价格明显高于BBI
                signals.append(f"价格强势突破BBI({bbi_distance*100:.1f}%)")
                score += 1.5
            else:
                signals.append("价格略高于BBI")
                score += 0.5
        else:
            if bbi_distance < -0.05:  # 价格明显低于BBI
                signals.append(f"价格明显跌破BBI({abs(bbi_distance)*100:.1f}%)")
                score -= 1.5
            else:
                signals.append("价格略低于BBI")
                score -= 0.5
        
        # BBI趋势分析（需要多期BBI数据，这里简化处理）
        if len(prices) >= 25:
            prev_bbi = self.calculate_bbi(prices[:-1])
            if bbi > prev_bbi:
                signals.append("BBI线上行")
                score += 0.5
            elif bbi < prev_bbi:
                signals.append("BBI线下行")
                score -= 0.5
        
        return {
            'signals': signals,
            'score': score,
            'bbi_value': bbi,
            'distance_pct': bbi_distance * 100
        }
    
    def analyze_volume_price_relationship(self, tech_data: Dict) -> Dict:
        """分析量价关系"""
        signals = []
        score = 0
        
        current_price = tech_data.get('current_price', 0)
        volume_ratio = tech_data.get('volume_ratio', 1)
        price_change_pct = tech_data.get('price_change_pct', 0)  # 价格变化百分比
        
        # 定义放量和缩量标准
        is_volume_increase = volume_ratio > 1.5  # 放量
        is_volume_decrease = volume_ratio < 0.7   # 缩量
        is_price_up = price_change_pct > 0.5      # 上涨
        is_price_down = price_change_pct < -0.5   # 下跌
        
        # 四种经典量价关系
        if is_price_up and is_volume_increase:
            # 放量上涨 - 强势信号
            if volume_ratio > 3:
                signals.append(f"放量大涨({volume_ratio:.1f}倍量)")
                score += 2.5
            elif volume_ratio > 2:
                signals.append(f"放量上涨({volume_ratio:.1f}倍量)")
                score += 2.0
            else:
                signals.append(f"温和放量上涨({volume_ratio:.1f}倍量)")
                score += 1.5
                
        elif is_price_up and is_volume_decrease:
            # 缩量上涨 - 需要观察后续
            signals.append(f"缩量上涨({volume_ratio:.1f}倍量)")
            if volume_ratio < 0.5:
                signals.append("缩量上涨或为主力锁仓")
                score += 0.5  # 可能是好信号，但需要谨慎
            else:
                score += 0.3
                
        elif is_price_down and is_volume_increase:
            # 放量下跌 - 可能见底信号
            if volume_ratio > 3:
                signals.append(f"放量大跌({volume_ratio:.1f}倍量)")
                signals.append("放量下跌或现恐慌性抛售")
                score -= 2.0  # 短期看空，但可能是反弹机会
            elif volume_ratio > 2:
                signals.append(f"放量下跌({volume_ratio:.1f}倍量)")
                score -= 1.5
            else:
                signals.append(f"温和放量下跌({volume_ratio:.1f}倍量)")
                score -= 1.0
                
        elif is_price_down and is_volume_decrease:
            # 缩量下跌 - 调整而非反转
            signals.append(f"缩量下跌({volume_ratio:.1f}倍量)")
            if volume_ratio < 0.5:
                signals.append("缩量下跌多为技术调整")
                score -= 0.5  # 相对温和的负面信号
            else:
                score -= 0.8
        
        # 特殊量价形态
        if volume_ratio > 5:
            signals.append("异常巨量交易")
            score += 1 if is_price_up else -1
        elif volume_ratio < 0.3:
            signals.append("地量交易")
            if not is_price_down:
                signals.append("地量不跌或为底部")
                score += 0.5
        
        # 成交量能量分析
        if volume_ratio > 2:
            signals.append("市场活跃度高")
        elif volume_ratio < 0.6:
            signals.append("市场活跃度低")
        
        return {
            'signals': signals,
            'score': score,
            'volume_ratio': volume_ratio,
            'price_change_pct': price_change_pct,
            'volume_pattern': self._classify_volume_pattern(volume_ratio, price_change_pct)
        }
    
    def _classify_volume_pattern(self, volume_ratio: float, price_change: float) -> str:
        """分类量价形态"""
        if price_change > 0.5 and volume_ratio > 1.5:
            return "放量上涨"
        elif price_change > 0.5 and volume_ratio < 0.7:
            return "缩量上涨"
        elif price_change < -0.5 and volume_ratio > 1.5:
            return "放量下跌"
        elif price_change < -0.5 and volume_ratio < 0.7:
            return "缩量下跌"
        elif volume_ratio > 2:
            return "放量横盘"
        elif volume_ratio < 0.6:
            return "缩量横盘"
        else:
            return "常量交易"
    
    def analyze_bollinger_enhanced(self, tech_data: Dict) -> Dict:
        """增强版布林带分析"""
        signals = []
        score = 0
        
        current_price = tech_data.get('current_price', 0)
        boll_upper = tech_data.get('boll_upper', current_price * 1.1)
        boll_middle = tech_data.get('boll_middle', current_price)
        boll_lower = tech_data.get('boll_lower', current_price * 0.9)
        volume_ratio = tech_data.get('volume_ratio', 1)
        
        if boll_upper <= boll_lower:
            return {'signals': signals, 'score': score}
        
        # 计算价格在布林带中的位置
        boll_position = (current_price - boll_lower) / (boll_upper - boll_lower)
        boll_width = (boll_upper - boll_lower) / boll_middle  # 布林带宽度
        
        # 布林带位置分析
        if boll_position > 0.95:
            signals.append("触及布林上轨")
            if volume_ratio > 1.5:
                signals.append("放量触及上轨或为突破")
                score += 0.5
            else:
                signals.append("缩量触及上轨需谨慎")
                score -= 1.0
        elif boll_position > 0.8:
            signals.append("接近布林上轨")
            score -= 0.5
        elif boll_position < 0.05:
            signals.append("触及布林下轨")
            if volume_ratio > 1.5:
                signals.append("放量触及下轨或为探底")
                score += 0.5
            else:
                signals.append("缩量触及下轨或为底部")
                score += 1.0
        elif boll_position < 0.2:
            signals.append("接近布林下轨")
            score += 0.5
        elif 0.4 <= boll_position <= 0.6:
            signals.append("运行在布林中轨附近")
        
        # 布林带宽度分析
        if boll_width > 0.15:
            signals.append("布林带开口较大")
            signals.append("市场波动性高")
        elif boll_width < 0.05:
            signals.append("布林带收窄")
            signals.append("市场或将变盘")
            score += 0.3  # 变盘前兆，可能是机会
        
        # 布林带突破分析
        if current_price > boll_upper:
            signals.append("突破布林上轨")
            if volume_ratio > 2:
                signals.append("放量突破上轨强势")
                score += 1.5
            else:
                signals.append("缩量突破上轨存疑")
                score += 0.5
        elif current_price < boll_lower:
            signals.append("跌破布林下轨")
            if volume_ratio > 2:
                signals.append("放量跌破下轨弱势")
                score -= 1.5
            else:
                signals.append("缩量跌破下轨超跌")
                score += 0.5
        
        return {
            'signals': signals,
            'score': score,
            'boll_position': boll_position,
            'boll_width': boll_width,
            'boll_pattern': self._classify_boll_pattern(boll_position, boll_width)
        }
    
    def _classify_boll_pattern(self, position: float, width: float) -> str:
        """分类布林带形态"""
        if position > 0.8 and width > 0.1:
            return "上轨压力"
        elif position < 0.2 and width > 0.1:
            return "下轨支撑"
        elif width < 0.05:
            return "布林收窄"
        elif width > 0.15:
            return "布林开口"
        elif 0.4 <= position <= 0.6:
            return "中轨震荡"
        else:
            return "常规形态"

class TradingSignalGenerator:
    """交易信号生成器 - V5版本"""
    
    def __init__(self):
        self.analyzer = TechnicalAnalyzer()
        try:
            # 获取雪球cookie配置
            xueqiu_cookie = get_xueqiu_cookie()
            self.sentiment_integrator = ChineseSentimentIntegrator(xueqiu_cookie=xueqiu_cookie)
        except:
            self.sentiment_integrator = None
        
    def generate_trading_signal(self, technical_data: Dict, sentiment_data: Dict, stock_info: Dict) -> Dict:
        """生成交易信号 - V5增强版"""
        try:
            # 技术分析
            tech_signal = self._analyze_technical_signals_v5(technical_data)
            
            # 情绪分析
            sentiment_signal = self._analyze_sentiment_signals(sentiment_data)
            
            # 综合信号
            combined_signal = self._combine_signals_v5(tech_signal, sentiment_signal, technical_data)
            
            # 价格目标计算
            price_targets = self._calculate_price_targets_v5(technical_data, combined_signal)
            
            # 风险评估
            risk_assessment = self._assess_risk_v5(tech_signal, sentiment_signal, combined_signal)
            
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
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                # V5新增详细分析
                'volume_analysis': tech_signal.get('volume_analysis', {}),
                'bbi_analysis': tech_signal.get('bbi_analysis', {}),
                'bollinger_analysis': tech_signal.get('bollinger_analysis', {})
            }
            
        except Exception as e:
            return {
                'stock_code': stock_info.get('code', ''),
                'stock_name': stock_info.get('name', ''),
                'error': f"分析失败: {str(e)}"
            }
    
    def _analyze_technical_signals_v5(self, tech_data: Dict) -> Dict:
        """V5技术信号分析 - 增强版"""
        signals = []
        score = 0
        
        current_price = tech_data.get('current_price', 0)
        if current_price == 0:
            return {'score': 0, 'signals': ["无价格数据"]}
        
        # 1. 均线信号（保持原有逻辑）
        ma5 = tech_data.get('ma5', current_price)
        ma10 = tech_data.get('ma10', current_price)
        ma20 = tech_data.get('ma20', current_price)
        ma60 = tech_data.get('ma60', current_price)
        
        if current_price > ma5 > ma10 > ma20 > ma60:
            signals.append("完美多头排列")
            score += 3
        elif current_price < ma5 < ma10 < ma20 < ma60:
            signals.append("完美空头排列")
            score -= 3
        elif current_price > ma20:
            signals.append("价格在短期均线上方")
            score += 1
        else:
            signals.append("价格在短期均线下方")
            score -= 1
        
        # 2. RSI信号
        rsi = tech_data.get('rsi', 50)
        if rsi > 80:
            signals.append(f"RSI超买({rsi:.1f})")
            score -= 1
        elif rsi < 20:
            signals.append(f"RSI超卖({rsi:.1f})")
            score += 1
        
        # 3. MACD信号
        macd_line = tech_data.get('macd_line', 0)
        macd_signal = tech_data.get('macd_signal', 0)
        macd_histogram = tech_data.get('macd_histogram', 0)
        
        if macd_line > macd_signal and macd_histogram > 0:
            if macd_histogram > 0.01:
                signals.append("MACD强势金叉")
                score += 2
            else:
                signals.append("MACD金叉")
                score += 1
        elif macd_line < macd_signal and macd_histogram < 0:
            if macd_histogram < -0.01:
                signals.append("MACD强势死叉")
                score -= 2
            else:
                signals.append("MACD死叉")
                score -= 1
        
        # 4. KDJ信号
        kdj_k = tech_data.get('kdj_k', 50)
        kdj_d = tech_data.get('kdj_d', 50)
        kdj_j = 3 * kdj_k - 2 * kdj_d
        
        if kdj_k > 80 and kdj_d > 80:
            signals.append(f"KDJ超买区({kdj_k:.1f})")
            score -= 1.5
        elif kdj_k < 20 and kdj_d < 20:
            signals.append(f"KDJ超卖区({kdj_k:.1f})")
            score += 1.5
        
        if kdj_k > kdj_d:
            if kdj_k < 50:
                signals.append("KDJ低位金叉")
                score += 1
            else:
                signals.append("KDJ金叉")
                score += 0.5
        else:
            if kdj_k > 50:
                signals.append("KDJ高位死叉")
                score -= 1
            else:
                signals.append("KDJ死叉")
                score -= 0.5
        
        # 5. V5新增：BBI分析
        prices = tech_data.get('price_history', [current_price])
        bbi_analysis = self.analyzer.analyze_bbi_signals(current_price, 
                                                        self.analyzer.calculate_bbi(prices), 
                                                        prices)
        signals.extend(bbi_analysis['signals'])
        score += bbi_analysis['score']
        
        # 6. V5新增：详细量价关系分析
        volume_analysis = self.analyzer.analyze_volume_price_relationship(tech_data)
        signals.extend(volume_analysis['signals'])
        score += volume_analysis['score']
        
        # 7. V5新增：增强版布林带分析
        bollinger_analysis = self.analyzer.analyze_bollinger_enhanced(tech_data)
        signals.extend(bollinger_analysis['signals'])
        score += bollinger_analysis['score']
        
        # 8. 阻力位和支撑位分析（保持原有逻辑）
        resistance_support = self._analyze_resistance_support(tech_data)
        if resistance_support:
            signals.extend(resistance_support['signals'])
            score += resistance_support['score']
        
        return {
            'score': score,
            'signals': signals,
            'kdj_analysis': {'k': kdj_k, 'd': kdj_d, 'j': kdj_j},
            'volume_analysis': volume_analysis,
            'bbi_analysis': bbi_analysis,
            'bollinger_analysis': bollinger_analysis
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
        filtered_posts = sentiment_data.get('filtered_posts', 0)
        
        # 情绪分析
        if avg_sentiment > 0.3:
            signals.append("市场情绪积极")
            score += 1.5
        elif avg_sentiment > 0.1:
            signals.append("市场情绪偏正面")
            score += 0.5
        elif avg_sentiment > -0.1:
            signals.append("市场情绪中性")
        elif avg_sentiment > -0.3:
            signals.append("市场情绪中性偏负")
            score -= 0.5
        elif avg_sentiment > -0.5:
            signals.append("市场情绪悲观")
            score -= 1.5
        else:
            signals.append("市场情绪极度悲观")
            score -= 2.0
        
        # 讨论热度分析（V4中已调整的阈值）
        if filtered_posts >= 30:
            signals.append(f"讨论热度高({filtered_posts}条)")
            score += 0.5
        elif filtered_posts >= 10:
            signals.append(f"讨论热度中等({filtered_posts}条)")
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
    
    def _combine_signals_v5(self, tech_signal: Dict, sentiment_signal: Dict, tech_data: Dict) -> Dict:
        """V5综合信号分析"""
        tech_score = tech_signal['score']
        sentiment_score = sentiment_signal['score']
        
        # 加权综合得分
        combined_score = tech_score * 0.7 + sentiment_score * 0.3
        
        # 生成交易建议
        if combined_score > 1:
            action = "BUY"
            strength = min(10, int(combined_score + 5))
            confidence = "HIGH" if combined_score > 3 else "MEDIUM"
        elif combined_score < -1:
            action = "AVOID"
            strength = min(10, int(abs(combined_score) + 5))
            confidence = "HIGH" if combined_score < -3 else "MEDIUM"
        else:
            action = "HOLD"
            strength = 5
            confidence = "LOW"
        
        # V5增强：优先显示关键技术信号（包含量价、BBI、布林带分析）
        tech_signals = tech_signal['signals']
        priority_signals = []
        other_signals = []
        
        for signal in tech_signals:
            # 优先显示量价关系、BBI、KDJ、阻力位、布林带相关信号
            if any(keyword in signal for keyword in ['量', 'BBI', 'KDJ', '阻力', '支撑', '关口', '布林', '放量', '缩量']):
                priority_signals.append(signal)
            else:
                other_signals.append(signal)
        
        # 合并信号，优先级信号在前，最多显示8个
        display_signals = (priority_signals + other_signals)[:8]
        tech_reasoning = '; '.join(display_signals)
        
        sentiment_reasoning = '; '.join(sentiment_signal['signals'])
        reasoning = f"技术面: {tech_reasoning}; 情绪面: {sentiment_reasoning}"
        
        return {
            'action': action,
            'strength': strength,
            'confidence': confidence,
            'reasoning': reasoning,
            'combined_score': combined_score
        }
    
    def _calculate_price_targets_v5(self, tech_data: Dict, signal: Dict) -> Dict:
        """V5价格目标计算"""
        current_price = tech_data.get('current_price', 0)
        if current_price == 0:
            return {}
        
        atr = tech_data.get('atr', current_price * 0.02)
        boll_upper = tech_data.get('boll_upper', current_price * 1.1)
        boll_lower = tech_data.get('boll_lower', current_price * 0.9)
        
        if signal['action'] == 'BUY':
            entry_price = current_price
            stop_loss = max(
                current_price - 2 * atr,
                boll_lower * 0.98,
                current_price * 0.95
            )
            target_price = min(
                current_price + 4 * atr,
                boll_upper * 1.02,
                current_price * 1.15
            )
            risk_reward_ratio = (target_price - entry_price) / (entry_price - stop_loss) if entry_price > stop_loss else 0
            price_comment = "建议买入价格区间"
            
        elif signal['action'] == 'AVOID':
            entry_price = current_price
            stop_loss = min(
                current_price + 2 * atr,
                boll_upper * 1.02,
                current_price * 1.05
            )
            target_price = max(
                current_price - 4 * atr,
                boll_lower * 0.98,
                current_price * 0.85
            )
            risk_reward_ratio = 0
            price_comment = "建议避让，持有者可考虑卖出"
            
        else:  # HOLD
            entry_price = current_price
            target_price = current_price
            stop_loss = max(
                current_price - 1.5 * atr,
                boll_lower,
                current_price * 0.95
            )
            risk_reward_ratio = 0
            price_comment = "建议观望"
        
        return {
            'entry_price': round(entry_price, 2),
            'target_price': round(target_price, 2),
            'stop_loss': round(stop_loss, 2),
            'risk_reward_ratio': round(risk_reward_ratio, 1),
            'price_comment': price_comment
        }
    
    def _assess_risk_v5(self, tech_signal: Dict, sentiment_signal: Dict, combined_signal: Dict) -> Dict:
        """V5风险评估"""
        risk_factors = []
        risk_score = 0
        
        # 基于技术分析的风险因素
        tech_score = tech_signal['score']
        if tech_score < -3:
            risk_factors.append("技术面极度弱势")
            risk_score += 2
        elif tech_score < -1:
            risk_factors.append("技术面偏弱")
            risk_score += 1
        
        # 基于情绪分析的风险因素
        sentiment_score = sentiment_signal['score']
        if sentiment_score < -1.5:
            risk_factors.append("市场情绪悲观")
            risk_score += 1
        
        # V5新增：基于量价关系的风险因素
        volume_analysis = tech_signal.get('volume_analysis', {})
        if volume_analysis.get('volume_pattern') == '放量下跌':
            risk_factors.append("放量下跌存在风险")
            risk_score += 1
        elif volume_analysis.get('volume_ratio', 1) < 0.3:
            risk_factors.append("成交量过低流动性风险")
            risk_score += 1
        
        # 置信度风险
        if combined_signal['confidence'] == 'LOW':
            risk_factors.append("信号置信度低")
            risk_score += 1
        
        # 数据完整性风险
        if sentiment_score == 0:
            risk_factors.append("情绪数据样本不足")
            risk_score += 1
        
        # 确定风险等级
        if risk_score >= 4:
            risk_level = "HIGH"
            position_suggestion = "轻仓操作(2-5%)"
        elif risk_score >= 2:
            risk_level = "MEDIUM"
            position_suggestion = "半仓操作(5-10%)"
        else:
            risk_level = "LOW"
            position_suggestion = "标准仓位(10-15%)"
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'position_suggestion': position_suggestion
        }

def generate_trading_report(trading_results: List[Dict]) -> str:
    """生成交易报告"""
    if not trading_results:
        return "# 📊 无交易数据"
    
    # 统计信息
    total_stocks = len(trading_results)
    buy_count = sum(1 for r in trading_results if r.get('signal') == 'BUY')
    avoid_count = sum(1 for r in trading_results if r.get('signal') == 'AVOID')
    hold_count = sum(1 for r in trading_results if r.get('signal') == 'HOLD')
    
    # 分类结果
    buy_stocks = [r for r in trading_results if r.get('signal') == 'BUY']
    avoid_stocks = [r for r in trading_results if r.get('signal') == 'AVOID']
    hold_stocks = [r for r in trading_results if r.get('signal') == 'HOLD']
    
    # 按信号强度排序
    buy_stocks.sort(key=lambda x: x.get('signal_strength', 0), reverse=True)
    avoid_stocks.sort(key=lambda x: x.get('signal_strength', 0), reverse=True)
    hold_stocks.sort(key=lambda x: x.get('signal_strength', 0), reverse=True)
    
    # 生成报告
    report = f"""# 🎯 TradingAgents完整交易建议报告

## 📊 分析概览
- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **分析股票**: {total_stocks}只
- **买入建议**: {buy_count}只
- **避让建议**: {avoid_count}只（不建议新买，持有者可考虑卖出）
- **观望建议**: {hold_count}只
- **分析引擎**: TradingAgents + 技术分析 + 情绪分析 + 量价关系分析
- **版本**: V5 (增强量价、BBI、布林带分析)

"""
    
    # 买入推荐
    if buy_stocks:
        report += "## 📈 买入推荐（建议新建仓）\n\n"
        for i, stock in enumerate(buy_stocks, 1):
            report += f"""### {i}. {stock['stock_code']} - {stock['stock_name']}
- **当前价格**: ¥{stock['current_price']}
- **操作建议**: {stock['signal']}
- **信号强度**: {stock['signal_strength']}/10
- **置信度**: {stock['confidence']}

**技术面分析**
- 技术评分: {stock['technical_score']}/10
- 情绪评分: {stock['sentiment_score']}/10

**{stock['price_targets']['price_comment']}**
- **建议买入价**: ¥{stock['price_targets']['entry_price']}
- **上涨目标价**: ¥{stock['price_targets']['target_price']}
- **止损价**: ¥{stock['price_targets']['stop_loss']}
- **风险收益比**: 1:{stock['price_targets']['risk_reward_ratio']}

- **风险等级**: {stock['risk_assessment']['risk_level']}
- **仓位建议**: {stock['risk_assessment']['position_suggestion']}
- **风险因素**: {', '.join(stock['risk_assessment']['risk_factors'])}

**分析依据**: {stock['reasoning']}

---

"""
    
    # 避让建议
    if avoid_stocks:
        report += "## 🚫 避让建议（不建议买入，持有者可考虑卖出）\n\n"
        for i, stock in enumerate(avoid_stocks, 1):
            report += f"""### {i}. {stock['stock_code']} - {stock['stock_name']}
- **当前价格**: ¥{stock['current_price']}
- **操作建议**: {stock['signal']}
- **信号强度**: {stock['signal_strength']}/10
- **置信度**: {stock['confidence']}

**技术面分析**
- 技术评分: {stock['technical_score']}/10
- 情绪评分: {stock['sentiment_score']}/10

**{stock['price_targets']['price_comment']}**
- **当前价格**: ¥{stock['price_targets']['entry_price']}
- **预期下跌目标**: ¥{stock['price_targets']['target_price']}
- **持仓止损建议**: ¥{stock['price_targets']['stop_loss']}

- **风险等级**: {stock['risk_assessment']['risk_level']}
- **仓位建议**: {stock['risk_assessment']['position_suggestion']}
- **风险因素**: {', '.join(stock['risk_assessment']['risk_factors'])}

**分析依据**: {stock['reasoning']}

---

"""
    
    # 观望建议
    if hold_stocks:
        report += "## ⏸️ 观望建议\n\n"
        for i, stock in enumerate(hold_stocks, 1):
            report += f"""### {i}. {stock['stock_code']} - {stock['stock_name']}
- **当前价格**: ¥{stock['current_price']}
- **操作建议**: {stock['signal']}
- **信号强度**: {stock['signal_strength']}/10
- **置信度**: {stock['confidence']}

**技术面分析**
- 技术评分: {stock['technical_score']}/10
- 情绪评分: {stock['sentiment_score']}/10

- **风险等级**: {stock['risk_assessment']['risk_level']}
- **仓位建议**: {stock['risk_assessment']['position_suggestion']}
- **风险因素**: {', '.join(stock['risk_assessment']['risk_factors'])}

**分析依据**: {stock['reasoning']}

---

"""
    
    # 操作说明和风险提示
    report += """
## 📖 操作说明
- **买入推荐(BUY)**: 适合新建仓位，给出买入价格、上涨目标、止损位
- **避让建议(AVOID)**: 不建议新买入，如已持有可考虑在当前价格卖出
- **观望建议(HOLD)**: 暂时观望，等待更明确的信号

## 🆕 V5版本新增功能
- **量价关系分析**: 详细分析放量上涨、缩量下跌等经典形态
- **BBI多空指标**: 综合3、6、12、24日均线的多空判断
- **增强布林带分析**: 布林带位置、宽度、突破等全面分析

## ⚠️ 风险提示
- 本报告仅供参考，投资有风险
- 严格执行止损纪律，控制风险
- 分散投资，不要集中在单一股票
- 根据自身风险承受能力调整仓位

🤖 **报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Generated with [Claude Code](https://claude.ai/code)
"""
    
    return report

if __name__ == "__main__":
    print("TradingAgents V5 - 增强版交易建议生成器")
    print("功能: 量价关系分析 + BBI分析 + 增强布林带分析")