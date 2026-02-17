#!/usr/bin/env python3
"""
TradingAgents完整交易建议系统 v2
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
        
        # 价格目标计算
        price_targets = self._calculate_price_targets(technical_data, combined_signal)
        
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
        """分析技术信号"""
        signals = []
        score = 0
        
        current_price = tech_data.get('current_price', 0)
        ma5 = tech_data.get('ma5', current_price)
        ma10 = tech_data.get('ma10', current_price)
        ma20 = tech_data.get('ma20', current_price)
        
        # 移动平均线信号
        if current_price > ma5 > ma10 > ma20:
            signals.append("多头排列")
            score += 2
        elif current_price < ma5 < ma10 < ma20:
            signals.append("空头排列")
            score -= 2
        elif current_price > ma5:
            signals.append("价格在短期均线上方")
            score += 1
        else:
            signals.append("价格在短期均线下方")
            score -= 1
        
        # RSI信号
        rsi = tech_data.get('rsi', 50)
        if rsi > 70:
            signals.append(f"RSI超买({rsi:.1f})")
            score -= 1
        elif rsi < 30:
            signals.append(f"RSI超卖({rsi:.1f})")
            score += 1
        
        # MACD信号
        macd_line = tech_data.get('macd_line', 0)
        macd_signal = tech_data.get('macd_signal', 0)
        if macd_line > macd_signal:
            signals.append("MACD金叉")
            score += 1
        else:
            signals.append("MACD死叉")
            score -= 1
        
        return {
            'score': score,
            'signals': signals
        }
    
    def _analyze_sentiment_signals(self, sentiment_data: Dict) -> Dict:
        """分析情绪信号"""
        if not sentiment_data or 'avg_sentiment' not in sentiment_data:
            return {'score': 0, 'signals': ["无情绪数据"]}
        
        signals = []
        score = 0
        
        avg_sentiment = sentiment_data.get('avg_sentiment', 0)
        
        if avg_sentiment > 0.1:
            signals.append("市场情绪乐观")
            score += 2
        elif avg_sentiment > -0.1:
            signals.append("市场情绪中性")
            score += 0
        else:
            signals.append("市场情绪悲观")
            score -= 2
        
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
        
        # 生成交易建议
        if combined_score > 1:
            action = "BUY"
            strength = min(10, int(combined_score + 5))
            confidence = "HIGH" if combined_score > 3 else "MEDIUM"
        elif combined_score < -1:
            action = "SELL"
            strength = min(10, int(abs(combined_score) + 5))
            confidence = "HIGH" if combined_score < -3 else "MEDIUM"
        else:
            action = "HOLD"
            strength = 5
            confidence = "LOW"
        
        # 生成推理说明
        reasoning = f"技术面: {'; '.join(tech_signal['signals'][:2])}; 情绪面: {'; '.join(sentiment_signal['signals'])}"
        
        return {
            'action': action,
            'strength': strength,
            'confidence': confidence,
            'reasoning': reasoning
        }
    
    def _calculate_price_targets(self, tech_data: Dict, signal: Dict) -> Dict:
        """计算价格目标"""
        current_price = tech_data.get('current_price', 0)
        if current_price == 0:
            return {}
        
        if signal['action'] == 'BUY':
            entry_price = current_price * 0.98
            target_price = current_price * 1.15
            stop_loss = current_price * 0.92
        elif signal['action'] == 'SELL':
            entry_price = current_price * 1.02
            target_price = current_price * 0.85
            stop_loss = current_price * 1.08
        else:
            entry_price = current_price
            target_price = current_price
            stop_loss = current_price * 0.9
        
        return {
            'entry_price': round(entry_price, 2),
            'target_price': round(target_price, 2),
            'stop_loss': round(stop_loss, 2)
        }
    
    def _assess_risk(self, tech_data: Dict, sentiment_data: Dict, signal: Dict) -> Dict:
        """风险评估"""
        risk_score = 0
        
        # 技术面风险
        rsi = tech_data.get('rsi', 50)
        if rsi > 80 or rsi < 20:
            risk_score += 2
        elif rsi > 70 or rsi < 30:
            risk_score += 1
        
        # 信号强度风险
        if signal['confidence'] == 'LOW':
            risk_score += 1
        
        if risk_score <= 1:
            risk_level = "LOW"
            position_suggestion = "标准仓位"
        elif risk_score <= 3:
            risk_level = "MEDIUM"
            position_suggestion = "半仓操作"
        else:
            risk_level = "HIGH"
            position_suggestion = "轻仓或观望"
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
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
    ]
    
    # 买入推荐
    if buy_signals:
        report_lines.extend([
            "## 📈 买入推荐",
            ""
        ])
        for i, result in enumerate(buy_signals[:10], 1):  # 只显示前10个
            report_lines.extend(format_stock_analysis(result, i))
    
    # 卖出建议
    if sell_signals:
        report_lines.extend([
            "## 📉 卖出建议", 
            ""
        ])
        for i, result in enumerate(sell_signals[:10], 1):
            report_lines.extend(format_stock_analysis(result, i))
    
    # 持有观望
    if hold_signals:
        report_lines.extend([
            "## ⏸️ 持有观望",
            ""
        ])
        for i, result in enumerate(hold_signals[:5], 1):  # 只显示前5个
            report_lines.extend(format_stock_analysis(result, i))
    
    # 风险提示
    report_lines.extend([
        "",
        "## ⚠️ 风险提示",
        "- 本报告仅供参考，投资有风险",
        "- 严格执行止损纪律，控制风险",
        "- 分散投资，不要集中在单一股票",
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
    
    # 价格目标
    price_targets = result.get('price_targets', {})
    if price_targets:
        lines.extend([
            f"- **入场价**: ¥{price_targets.get('entry_price', 0)}",
            f"- **目标价**: ¥{price_targets.get('target_price', 0)}",
            f"- **止损价**: ¥{price_targets.get('stop_loss', 0)}",
            ""
        ])
    
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
    print("🚀 TradingAgents完整交易建议系统")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 初始化组件
    tech_analyzer = TechnicalAnalyzer()
    signal_generator = TradingSignalGenerator()
    
    # 读取情绪分析结果
    sentiment_file = f"/Users/yangxu/StockTradebyZ/TA_integration/分析结果_{datetime.now().strftime('%Y%m%d')}.json"
    
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
    
    # 统计信息
    buy_count = len([r for r in trading_results if r['signal'] == 'BUY'])
    sell_count = len([r for r in trading_results if r['signal'] == 'SELL'])
    hold_count = len([r for r in trading_results if r['signal'] == 'HOLD'])
    
    print(f"\n📈 交易建议统计:")
    print(f"   买入: {buy_count}只")
    print(f"   卖出: {sell_count}只")
    print(f"   持有: {hold_count}只")
    
    print(f"\n🎉 分析完成!")

if __name__ == "__main__":
    main()