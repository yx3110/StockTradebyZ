#!/usr/bin/env python3
"""
TradingAgents完整交易建议系统
集成技术分析、情绪分析，生成具体的买入/卖出建议和价格目标
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
            
            return {
                'current_price': round(current_price, 2),
                'price_change_pct': round(price_change, 2),
                'ma5': round(ma5, 2),
                'ma10': round(ma10, 2), 
                'ma20': round(ma20, 2),
                'rsi': round(rsi, 2),
                'macd_line': round(macd_line, 4),
                'macd_signal': round(signal_line, 4),
                'macd_histogram': round(histogram, 4),
                'kdj_k': round(k_percent, 2),
                'kdj_d': round(d_percent, 2),
                'boll_upper': round(upper_band, 2),
                'boll_middle': round(middle_band, 2),
                'boll_lower': round(lower_band, 2),
                'volume_ratio': round(volume_ratio, 2)
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

class TradingSignalGenerator:
    """交易信号生成器"""
    
    def __init__(self):
        pass
    
    def generate_trading_signal(self, technical_data: Dict, sentiment_data: Dict, stock_info: Dict) -> Dict:
        """生成交易信号"""
        if not technical_data:
            return self._generate_fallback_signal(sentiment_data, stock_info)
        
        # 技术面信号
        tech_signal = self._analyze_technical_signals(technical_data)
        
        # 情绪面信号
        sentiment_signal = self._analyze_sentiment_signals(sentiment_data)
        
        # 综合信号
        combined_signal = self._combine_signals(tech_signal, sentiment_signal, stock_info)
        
        # 价格目标计算
        price_targets = self._calculate_price_targets(technical_data, combined_signal)
        
        # 风险评估
        risk_assessment = self._assess_risk(technical_data, sentiment_data, combined_signal)
        
        return {
            'stock_code': stock_info.get('code', ''),
            'stock_name': stock_info.get('name', ''),
            'current_price': technical_data.get('current_price', 0),
            'signal': combined_signal['action'],  # BUY, SELL, HOLD
            'signal_strength': combined_signal['strength'],  # 1-10
            'confidence': combined_signal['confidence'],  # LOW, MEDIUM, HIGH
            'technical_score': tech_signal['score'],
            'sentiment_score': sentiment_signal['score'], 
            'price_targets': price_targets,
            'risk_assessment': risk_assessment,
            'reasoning': combined_signal['reasoning'],
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _analyze_technical_signals(self, tech_data: Dict) -> Dict:
        """分析技术信号"""
        signals = []
        score = 0
        
        current_price = tech_data.get('current_price', 0)
        
        # 移动平均线信号
        ma5 = tech_data.get('ma5', current_price)
        ma10 = tech_data.get('ma10', current_price)
        ma20 = tech_data.get('ma20', current_price)
        
        if current_price > ma5 > ma10 > ma20:
            signals.append("多头排列，趋势向上")
            score += 2
        elif current_price < ma5 < ma10 < ma20:
            signals.append("空头排列，趋势向下")
            score -= 2
        elif current_price > ma5:
            signals.append("价格在短期均线之上")
            score += 1
        else:
            signals.append("价格在短期均线之下")
            score -= 1
        
        # RSI信号
        rsi = tech_data.get('rsi', 50)
        if rsi > 70:
            signals.append(f"RSI超买({rsi:.1f})")
            score -= 1
        elif rsi < 30:
            signals.append(f"RSI超卖({rsi:.1f})")
            score += 1
        else:
            signals.append(f"RSI正常({rsi:.1f})")
        
        # MACD信号
        macd_line = tech_data.get('macd_line', 0)
        macd_signal = tech_data.get('macd_signal', 0)
        macd_histogram = tech_data.get('macd_histogram', 0)
        
        if macd_line > macd_signal and macd_histogram > 0:
            signals.append("MACD金叉向上")
            score += 1
        elif macd_line < macd_signal and macd_histogram < 0:
            signals.append("MACD死叉向下")
            score -= 1
        
        # KDJ信号
        kdj_k = tech_data.get('kdj_k', 50)
        kdj_d = tech_data.get('kdj_d', 50)
        
        if kdj_k > 80 and kdj_d > 80:
            signals.append("KDJ超买区域")
            score -= 1
        elif kdj_k < 20 and kdj_d < 20:
            signals.append("KDJ超卖区域")
            score += 1
        elif kdj_k > kdj_d:
            signals.append("KDJ金叉")
            score += 0.5
        else:
            signals.append("KDJ死叉")
            score -= 0.5
        
        # 布林带信号
        boll_upper = tech_data.get('boll_upper', current_price)
        boll_lower = tech_data.get('boll_lower', current_price)
        
        if current_price > boll_upper:
            signals.append("突破布林带上轨")
            score += 0.5
        elif current_price < boll_lower:
            signals.append("跌破布林带下轨")
            score -= 0.5
        
        # 成交量信号
        volume_ratio = tech_data.get('volume_ratio', 1)
        if volume_ratio > 2:
            signals.append("放量明显")
            score += 0.5
        elif volume_ratio < 0.5:
            signals.append("缩量明显")
            score -= 0.5
        
        return {
            'score': score,
            'signals': signals,
            'max_score': 8
        }
    
    def _analyze_sentiment_signals(self, sentiment_data: Dict) -> Dict:
        """分析情绪信号"""
        if not sentiment_data or 'avg_sentiment' not in sentiment_data:
            return {'score': 0, 'signals': ["无情绪数据"], 'max_score': 3}
        
        signals = []
        score = 0
        
        avg_sentiment = sentiment_data.get('avg_sentiment', 0)
        total_posts = sentiment_data.get('total_posts', 0)
        confidence = sentiment_data.get('confidence_level', 'low')
        
        # 情绪评分
        if avg_sentiment > 0.3:
            signals.append("市场情绪非常乐观")
            score += 3
        elif avg_sentiment > 0.1:
            signals.append("市场情绪偏乐观")
            score += 2
        elif avg_sentiment > -0.1:
            signals.append("市场情绪中性")
            score += 0
        elif avg_sentiment > -0.3:
            signals.append("市场情绪偏悲观")
            score -= 2
        else:
            signals.append("市场情绪非常悲观")
            score -= 3
        
        # 讨论热度
        if total_posts > 50:
            signals.append("讨论热度高")
            score += 0.5
        elif total_posts > 20:
            signals.append("讨论热度中等")
        else:
            signals.append("讨论热度低")
            score -= 0.5
        
        # 数据可信度调整
        if confidence == 'high':
            score *= 1.2
        elif confidence == 'low':
            score *= 0.8
        
        return {
            'score': score,
            'signals': signals,
            'max_score': 3
        }
    
    def _combine_signals(self, tech_signal: Dict, sentiment_signal: Dict, stock_info: Dict) -> Dict:
        """综合信号分析"""
        tech_score = tech_signal['score']
        sentiment_score = sentiment_signal['score']
        
        # 加权综合得分（技术面权重70%，情绪面权重30%）
        combined_score = tech_score * 0.7 + sentiment_score * 0.3
        max_combined_score = tech_signal['max_score'] * 0.7 + sentiment_signal['max_score'] * 0.3
        
        # 标准化到-10到10的范围
        if max_combined_score > 0:
            normalized_score = (combined_score / max_combined_score) * 10
        else:
            normalized_score = 0
        
        # 生成交易建议
        if normalized_score > 3:
            action = "BUY"
            strength = min(10, int(normalized_score + 2))
            confidence = "HIGH" if normalized_score > 6 else "MEDIUM"
        elif normalized_score < -3:
            action = "SELL" 
            strength = min(10, int(abs(normalized_score) + 2))
            confidence = "HIGH" if normalized_score < -6 else "MEDIUM"
        else:
            action = "HOLD"
            strength = 5
            confidence = "LOW"
        
        # 生成推理说明
        reasoning_parts = []
        reasoning_parts.extend([f"技术面: {sig}" for sig in tech_signal['signals'][:3]])
        reasoning_parts.extend([f"情绪面: {sig}" for sig in sentiment_signal['signals'][:2]])
        reasoning = "; ".join(reasoning_parts)
        
        return {
            'action': action,
            'strength': strength,
            'confidence': confidence,
            'combined_score': round(normalized_score, 2),
            'reasoning': reasoning
        }
    
    def _calculate_price_targets(self, tech_data: Dict, signal: Dict) -> Dict:
        """计算价格目标"""
        current_price = tech_data.get('current_price', 0)
        if current_price == 0:
            return {}
        
        # 基于布林带和移动平均线计算支撑阻力位
        boll_upper = tech_data.get('boll_upper', current_price * 1.1)
        boll_lower = tech_data.get('boll_lower', current_price * 0.9)
        ma20 = tech_data.get('ma20', current_price)
        
        if signal['action'] == 'BUY':
            # 买入建议的价格目标
            entry_price = max(current_price * 0.98, boll_lower)  # 稍微低于当前价格或布林带下轨
            target_price = min(current_price * 1.15, boll_upper)  # 目标价15%或布林带上轨
            stop_loss = max(entry_price * 0.92, boll_lower * 0.98)  # 止损8%或布林带下轨以下
            
        elif signal['action'] == 'SELL':
            # 卖出建议的价格目标
            entry_price = min(current_price * 1.02, boll_upper)  # 稍微高于当前价格
            target_price = max(current_price * 0.85, boll_lower)  # 目标价下跌15%
            stop_loss = min(entry_price * 1.08, boll_upper * 1.02)  # 止损8%
            
        else:  # HOLD
            entry_price = current_price
            target_price = current_price
            stop_loss = current_price * 0.9
        
        return {
            'entry_price': round(entry_price, 2),
            'target_price': round(target_price, 2),
            'stop_loss': round(stop_loss, 2),
            'risk_reward_ratio': round(abs(target_price - entry_price) / abs(entry_price - stop_loss), 2) if abs(entry_price - stop_loss) > 0 else 0
        }
    
    def _assess_risk(self, tech_data: Dict, sentiment_data: Dict, signal: Dict) -> Dict:
        """风险评估"""
        risk_factors = []
        risk_score = 0  # 0-10，越高风险越大
        
        # 技术面风险
        rsi = tech_data.get('rsi', 50)
        if rsi > 80:
            risk_factors.append("RSI严重超买")
            risk_score += 2
        elif rsi > 70:
            risk_factors.append("RSI超买")
            risk_score += 1
        elif rsi < 20:
            risk_factors.append("RSI严重超卖")
            risk_score += 2
        elif rsi < 30:
            risk_factors.append("RSI超卖")
            risk_score += 1
        
        # 价格位置风险
        current_price = tech_data.get('current_price', 0)
        boll_upper = tech_data.get('boll_upper', current_price)
        boll_lower = tech_data.get('boll_lower', current_price)
        
        if current_price > boll_upper:
            risk_factors.append("价格在布林带上轨以上")
            risk_score += 1
        elif current_price < boll_lower:
            risk_factors.append("价格在布林带下轨以下")
            risk_score += 1
        
        # 情绪面风险
        confidence = sentiment_data.get('confidence_level', 'low') if sentiment_data else 'low'
        if confidence == 'low':
            risk_factors.append("情绪数据置信度低")
            risk_score += 1
        
        total_posts = sentiment_data.get('total_posts', 0) if sentiment_data else 0
        if total_posts < 10:
            risk_factors.append("讨论热度不足")
            risk_score += 1
        
        # 信号强度风险
        if signal['confidence'] == 'LOW':
            risk_factors.append("交易信号置信度低")
            risk_score += 1
        
        # 风险等级
        if risk_score <= 2:
            risk_level = "LOW"
            position_suggestion = "标准仓位"
        elif risk_score <= 5:
            risk_level = "MEDIUM"
            position_suggestion = "半仓操作"
        else:
            risk_level = "HIGH"
            position_suggestion = "轻仓或观望"
        
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
            'reasoning': '缺少技术分析数据，无法给出明确建议',
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

class TradingReportGenerator:
    """交易报告生成器"""
    
    def __init__(self):
        pass
    
    def generate_trading_report(self, trading_results: List[Dict]) -> str:
        """生成交易建议报告"""
        # 按信号强度排序
        sorted_results = sorted(trading_results, key=lambda x: x['signal_strength'], reverse=True)
        
        # 分类统计
        buy_signals = [r for r in sorted_results if r['signal'] == 'BUY']
        sell_signals = [r for r in sorted_results if r['signal'] == 'SELL']
        hold_signals = [r for r in sorted_results if r['signal'] == 'HOLD']
        
        report_lines = [
            "# 🎯 TradingAgents完整交易建议报告",
            "",
            "## 📊 分析概览",
            f"- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **分析股票**: {len(trading_results)}只",
            f"- **买入建议**: {len(buy_signals)}只",
            f"- **卖出建议**: {len(sell_signals)}只", 
            f"- **持有建议**: {len(hold_signals)}只",
            f"- **分析引擎**: TradingAgents + 技术分析 + 情绪分析",
            "",
            "## 🚀 强烈买入推荐",
            ""
        ]
        
        # 强烈买入推荐（信号强度>=8的买入信号）
        strong_buys = [r for r in buy_signals if r['signal_strength'] >= 8]
        if strong_buys:
            for i, result in enumerate(strong_buys, 1):
                report_lines.extend(self._format_stock_analysis(result, i, "strong_buy"))
        else:
            report_lines.append("*暂无强烈买入推荐的股票*\n")
        
        # 买入推荐
        if buy_signals:
            report_lines.extend([
                "## 📈 买入推荐",
                ""
            ])
            for i, result in enumerate(buy_signals, 1):
                report_lines.extend(self._format_stock_analysis(result, i, "buy"))
        
        # 卖出建议
        if sell_signals:
            report_lines.extend([
                "## 📉 卖出建议",
                ""
            ])
            for i, result in enumerate(sell_signals, 1):
                report_lines.extend(self._format_stock_analysis(result, i, "sell"))
        
        # 持有观望
        if hold_signals:
            report_lines.extend([
                "## ⏸️ 持有观望",
                ""
            ])
            # 只显示前5个持有建议，避免报告过长
            for i, result in enumerate(hold_signals[:5], 1):
                report_lines.extend(self._format_stock_analysis(result, i, "hold"))
        
        # 总结和风险提示
        report_lines.extend([
            "## 📋 投资策略总结",
            "",
            "### 🎯 操作建议优先级",
            "1. **强烈买入**: 技术面和情绪面双重确认，高置信度",
            "2. **买入**: 技术指标积极，情绪面支持",
            "3. **持有**: 信号不明确，建议观望等待",
            "4. **卖出**: 技术面走弱或情绪面悲观",
            "",
            "### ⚠️ 风险管理",
            "- **仓位控制**: 根据风险等级调整仓位大小",
            "- **止损纪律**: 严格执行止损价格，控制单笔损失",
            "- **分散投资**: 不要将资金集中在单一股票",
            "- **市场环境**: 关注大盘走势和政策变化",
            "",
            "### 📊 使用说明",
            "- **信号强度**: 1-10分，越高越强烈",
            "- **置信度**: HIGH/MEDIUM/LOW，反映信号可靠性",
            "- **价格目标**: 基于技术分析计算的关键价位",
            "- **风险收益比**: 目标收益与潜在损失的比例",
            "",
            "---",
            "",
            f"🤖 **报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"📊 **数据来源**: 历史K线数据 + 东方财富股吧情绪数据",
            f"🔬 **分析模型**: 技术分析(70%) + 情绪分析(30%)",
            "",
            "Generated with [Claude Code](https://claude.ai/code)"
        ]
        
        return "\n".join(report_lines)
    
    def _format_stock_analysis(self, result: Dict, index: int, category: str) -> List[str]:
        """格式化单只股票的分析结果"""
        stock_code = result['stock_code']
        stock_name = result['stock_name']
        signal = result['signal']
        strength = result['signal_strength']
        confidence = result['confidence']
        current_price = result['current_price']
        
        # 信号图标
        signal_icons = {
            'strong_buy': '🔥',
            'buy': '📈',
            'sell': '📉', 
            'hold': '⏸️'
        }
        icon = signal_icons.get(category, '📊')
        
        lines = [
            f"### {icon} {index}. {stock_code} - {stock_name}",
            ""
        ]
        
        # 基本信息
        if current_price > 0:
            lines.extend([
                "**交易建议**",
                f"- **当前价格**: ¥{current_price}",
                f"- **操作建议**: {signal}",
                f"- **信号强度**: {strength}/10",
                f"- **置信度**: {confidence}",
                ""
            ])
        
        # 价格目标
        price_targets = result.get('price_targets', {})
        if price_targets:
            entry_price = price_targets.get('entry_price', 0)
            target_price = price_targets.get('target_price', 0)
            stop_loss = price_targets.get('stop_loss', 0)
            risk_reward = price_targets.get('risk_reward_ratio', 0)
            
            lines.extend([
                "**价格目标**",
                f"- **建议{signal.lower()}价**: ¥{entry_price}",
                f"- **目标价位**: ¥{target_price}",
                f"- **止损价位**: ¥{stop_loss}",
                f"- **风险收益比**: 1:{risk_reward}",
                ""
            ])
        
        # 风险评估
        risk_assessment = result.get('risk_assessment', {})
        if risk_assessment:
            risk_level = risk_assessment.get('risk_level', 'UNKNOWN')
            position_suggestion = risk_assessment.get('position_suggestion', '')
            risk_factors = risk_assessment.get('risk_factors', [])
            
            lines.extend([
                "**风险评估**",
                f"- **风险等级**: {risk_level}",
                f"- **仓位建议**: {position_suggestion}",
            ])
            
            if risk_factors:
                lines.append("- **风险因素**: " + "; ".join(risk_factors[:3]))
            
            lines.append("")
        
        # 分析依据
        reasoning = result.get('reasoning', '')
        if reasoning:
            lines.extend([
                "**分析依据**",
                f"{reasoning[:200]}{'...' if len(reasoning) > 200 else ''}",
                ""
            ])
        
        lines.extend([
            "---",
            ""
        ])
        
        return lines

def main():
    """主函数 - 生成完整交易建议"""
    print("🚀 TradingAgents完整交易建议系统")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 初始化组件
    tech_analyzer = TechnicalAnalyzer()
    signal_generator = TradingSignalGenerator()
    report_generator = TradingReportGenerator()
    
    # 读取之前的情绪分析结果
    sentiment_file = f"/Users/yangxu/StockTradebyZ/TA_integration/分析结果_{datetime.now().strftime('%Y%m%d')}.json"
    
    try:
        with open(sentiment_file, 'r', encoding='utf-8') as f:
            sentiment_results = json.load(f)
        print(f"📊 读取情绪分析结果: {len(sentiment_results)}只股票")
    except FileNotFoundError:
        print(f"❌ 未找到情绪分析结果文件: {sentiment_file}")
        print("请先运行 analyze_daily_report.py 生成情绪分析数据")
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
    trading_report = report_generator.generate_trading_report(trading_results)
    
    # 保存报告
    report_file = f"/Users/yangxu/StockTradebyZ/TA_integration/TradingAgents交易建议_{datetime.now().strftime('%Y%m%d')}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(trading_report)
    
    # 保存JSON数据
    json_file = f"/Users/yangxu/StockTradebyZ/TA_integration/交易信号数据_{datetime.now().strftime('%Y%m%d')}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(trading_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 交易建议报告已生成!")
    print(f"📄 报告文件: {report_file}")
    print(f"💾 数据文件: {json_file}")
    print(f"📊 报告长度: {len(trading_report)} 字符")
    
    # 统计信息
    buy_count = len([r for r in trading_results if r['signal'] == 'BUY'])
    sell_count = len([r for r in trading_results if r['signal'] == 'SELL'])
    hold_count = len([r for r in trading_results if r['signal'] == 'HOLD'])
    
    print(f"\n📈 交易建议统计:")
    print(f"   买入: {buy_count}只")
    print(f"   卖出: {sell_count}只")
    print(f"   持有: {hold_count}只")
    
    print(f"\n🎉 分析完成! 用时: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()