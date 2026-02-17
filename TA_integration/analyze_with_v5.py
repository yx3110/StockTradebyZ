#!/usr/bin/env python3
"""
使用V5增强版系统分析股票
支持量价关系分析、BBI分析、增强布林带分析
"""
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import re

# 添加项目路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))

from trading_advisor_v5 import TradingSignalGenerator, generate_trading_report

def parse_daily_report(report_path: str) -> List[Dict[str, str]]:
    """解析每日选股报告，提取股票代码和名称"""
    stocks = []
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 匹配股票信息的正则表达式
        patterns = [
            r'###?\s*\d+\.\s*(\d{6})\s*-\s*([^#\n]+)',  # ### 1. 000001 - 平安银行
            r'(\d{6})\s*-\s*([^#\n,，]+)',              # 000001 - 平安银行
            r'(\d{6})[,，]\s*([^#\n,，]+)',             # 000001, 平安银行
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for code, name in matches:
                name = name.strip().replace('*', '').replace('ST', '').strip()
                if code and name and len(code) == 6:
                    stocks.append({'code': code, 'name': name})
        
        # 去重
        seen = set()
        unique_stocks = []
        for stock in stocks:
            key = stock['code']
            if key not in seen:
                seen.add(key)
                unique_stocks.append(stock)
        
        return unique_stocks
        
    except Exception as e:
        print(f"❌ 解析报告失败: {e}")
        return []

def load_real_technical_data(stock_code: str, stock_name: str) -> Dict:
    """从真实股票数据文件加载技术数据"""
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    # 查找股票数据文件
    data_paths = [
        f"/Users/yangxu/StockTradebyZ/full_securities_data/{stock_code}_A股.csv",
        f"/Users/yangxu/StockTradebyZ/full_securities_data/{stock_code}_ETF.csv",
        f"/Users/yangxu/StockTradebyZ/full_securities_data/{stock_code}_基金.csv"
    ]
    
    df = None
    for path in data_paths:
        if Path(path).exists():
            try:
                df = pd.read_csv(path)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                break
            except Exception as e:
                continue
    
    if df is None or len(df) < 60:
        print(f"   ⚠️ 无法读取{stock_code}的数据，使用模拟数据")
        return simulate_technical_data(stock_code, stock_name)
    
    # 取最近60天数据用于计算技术指标
    recent_data = df.tail(60).copy()
    
    # 计算移动平均线
    recent_data['ma5'] = recent_data['close'].rolling(5).mean()
    recent_data['ma10'] = recent_data['close'].rolling(10).mean()
    recent_data['ma20'] = recent_data['close'].rolling(20).mean()
    recent_data['ma60'] = recent_data['close'].rolling(60).mean()
    
    # 计算RSI
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    recent_data['rsi'] = calculate_rsi(recent_data['close'])
    
    # 计算MACD
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram
    
    macd_line, macd_signal, macd_histogram = calculate_macd(recent_data['close'])
    recent_data['macd_line'] = macd_line
    recent_data['macd_signal'] = macd_signal
    recent_data['macd_histogram'] = macd_histogram
    
    # 计算KDJ
    def calculate_kdj(df, period=9):
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        rsv = (df['close'] - low_min) / (high_max - low_min) * 100
        k = rsv.ewm(alpha=1/3).mean()
        d = k.ewm(alpha=1/3).mean()
        j = 3 * k - 2 * d
        return k, d, j
    
    kdj_k, kdj_d, kdj_j = calculate_kdj(recent_data)
    recent_data['kdj_k'] = kdj_k
    recent_data['kdj_d'] = kdj_d
    recent_data['kdj_j'] = kdj_j
    
    # 计算布林带
    def calculate_bollinger_bands(prices, period=20, std_dev=2):
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    boll_upper, boll_middle, boll_lower = calculate_bollinger_bands(recent_data['close'])
    recent_data['boll_upper'] = boll_upper
    recent_data['boll_middle'] = boll_middle
    recent_data['boll_lower'] = boll_lower
    
    # 计算ATR
    def calculate_atr(df, period=14):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    recent_data['atr'] = calculate_atr(recent_data)
    
    # 获取最新数据
    latest = recent_data.iloc[-1]
    prev = recent_data.iloc[-2] if len(recent_data) > 1 else latest
    
    # 计算涨跌幅
    price_change_pct = ((latest['close'] - prev['close']) / prev['close']) * 100
    
    # 计算成交量比率
    recent_volume_avg = recent_data['volume'].tail(5).mean()
    volume_ratio = latest['volume'] / recent_volume_avg if recent_volume_avg > 0 else 1.0
    
    tech_data = {
        'current_price': round(latest['close'], 2),
        'price_change_pct': round(price_change_pct, 2),
        'volume_ratio': round(volume_ratio, 2),
        
        # 均线
        'ma5': round(latest['ma5'], 2) if not pd.isna(latest['ma5']) else latest['close'],
        'ma10': round(latest['ma10'], 2) if not pd.isna(latest['ma10']) else latest['close'],
        'ma20': round(latest['ma20'], 2) if not pd.isna(latest['ma20']) else latest['close'],
        'ma60': round(latest['ma60'], 2) if not pd.isna(latest['ma60']) else latest['close'],
        
        # 技术指标
        'rsi': round(latest['rsi'], 1) if not pd.isna(latest['rsi']) else 50,
        'macd_line': round(latest['macd_line'], 4) if not pd.isna(latest['macd_line']) else 0,
        'macd_signal': round(latest['macd_signal'], 4) if not pd.isna(latest['macd_signal']) else 0,
        'macd_histogram': round(latest['macd_histogram'], 4) if not pd.isna(latest['macd_histogram']) else 0,
        'kdj_k': round(latest['kdj_k'], 1) if not pd.isna(latest['kdj_k']) else 50,
        'kdj_d': round(latest['kdj_d'], 1) if not pd.isna(latest['kdj_d']) else 50,
        
        # 布林带
        'boll_upper': round(latest['boll_upper'], 2) if not pd.isna(latest['boll_upper']) else latest['close'] * 1.1,
        'boll_middle': round(latest['boll_middle'], 2) if not pd.isna(latest['boll_middle']) else latest['close'],
        'boll_lower': round(latest['boll_lower'], 2) if not pd.isna(latest['boll_lower']) else latest['close'] * 0.9,
        
        # ATR
        'atr': round(latest['atr'], 2) if not pd.isna(latest['atr']) else latest['close'] * 0.02,
        
        # 价格历史（用于BBI计算）
        'price_history': [round(p, 2) for p in recent_data['close'].tail(30).tolist()]
    }
    
    return tech_data

def simulate_technical_data(stock_code: str, stock_name: str) -> Dict:
    """模拟技术数据（实际使用时应该从真实数据源获取）"""
    import random
    
    # 基础价格（根据股票代码特征模拟）
    base_price = 10 + (int(stock_code[-2:]) % 50) * 0.5
    
    # 模拟价格历史
    prices = []
    current = base_price
    for i in range(30):
        change = random.uniform(-0.03, 0.03)
        current = current * (1 + change)
        prices.append(current)
    
    current_price = prices[-1]
    
    # 模拟技术指标
    tech_data = {
        'current_price': round(current_price, 2),
        'price_change_pct': random.uniform(-5, 5),
        'volume_ratio': random.uniform(0.3, 4.0),
        
        # 均线
        'ma5': round(current_price * random.uniform(0.95, 1.05), 2),
        'ma10': round(current_price * random.uniform(0.93, 1.07), 2),
        'ma20': round(current_price * random.uniform(0.90, 1.10), 2),
        'ma60': round(current_price * random.uniform(0.85, 1.15), 2),
        
        # 技术指标
        'rsi': random.uniform(20, 80),
        'macd_line': random.uniform(-0.1, 0.1),
        'macd_signal': random.uniform(-0.08, 0.08),
        'macd_histogram': random.uniform(-0.05, 0.05),
        'kdj_k': random.uniform(10, 90),
        'kdj_d': random.uniform(10, 90),
        
        # 布林带
        'boll_upper': round(current_price * 1.1, 2),
        'boll_middle': round(current_price, 2),
        'boll_lower': round(current_price * 0.9, 2),
        
        # ATR
        'atr': round(current_price * 0.02, 2),
        
        # 价格历史（用于BBI计算）
        'price_history': [round(p, 2) for p in prices[-30:]]
    }
    
    return tech_data

def simulate_sentiment_data(stock_code: str) -> Dict:
    """模拟情绪数据"""
    import random
    
    filtered_posts = random.randint(0, 20)
    positive = random.randint(0, filtered_posts//2)
    negative = random.randint(0, filtered_posts//2)
    neutral = max(0, filtered_posts - positive - negative)
    
    if filtered_posts > 0:
        avg_sentiment = (positive - negative) / filtered_posts * 0.5
    else:
        avg_sentiment = 0
    
    return {
        'avg_sentiment': avg_sentiment,
        'filtered_posts': filtered_posts,
        'total_posts': filtered_posts + random.randint(0, 10),
        'sentiment_distribution': {
            'positive': positive,
            'neutral': neutral,
            'negative': negative
        }
    }

def analyze_stocks_with_v5(stocks: List[Dict], limit: int = None):
    """使用V5系统分析股票"""
    print(f"🚀 使用TradingAgents V5系统分析股票")
    print(f"📊 新功能: 量价关系分析 + BBI分析 + 增强布林带分析")
    print("="*80)
    
    if limit:
        stocks = stocks[:limit]
        print(f"⚠️  限制分析数量: {limit}只")
    
    print(f"📈 待分析股票: {len(stocks)}只")
    print()
    
    generator = TradingSignalGenerator()
    results = []
    
    for i, stock in enumerate(stocks, 1):
        print(f"📊 分析进度 ({i}/{len(stocks)}): {stock['code']} - {stock['name']}")
        
        try:
            # 获取真实技术数据
            tech_data = load_real_technical_data(stock['code'], stock['name'])
            sentiment_data = simulate_sentiment_data(stock['code'])
            
            # 生成交易信号
            result = generator.generate_trading_signal(tech_data, sentiment_data, stock)
            results.append(result)
            
            # 显示关键信息
            print(f"   🎯 建议: {result['signal']} (强度: {result['signal_strength']}/10)")
            
            # V5新功能展示
            if 'volume_analysis' in result:
                pattern = result['volume_analysis'].get('volume_pattern', '常量交易')
                ratio = result['volume_analysis'].get('volume_ratio', 1)
                print(f"   📊 量价: {pattern} ({ratio:.1f}倍量)")
            
            if 'bbi_analysis' in result and 'bbi_value' in result['bbi_analysis']:
                bbi = result['bbi_analysis']['bbi_value']
                distance = result['bbi_analysis'].get('distance_pct', 0)
                print(f"   📏 BBI: {bbi:.2f} (距离{distance:+.1f}%)")
            
            if 'bollinger_analysis' in result:
                boll_pattern = result['bollinger_analysis'].get('boll_pattern', '常规形态')
                position = result['bollinger_analysis'].get('boll_position', 0.5)
                print(f"   🎈 布林: {boll_pattern} (位置{position:.0%})")
            
            print()
            
        except Exception as e:
            print(f"   ❌ 分析失败: {e}")
            results.append({
                'stock_code': stock['code'],
                'stock_name': stock['name'],
                'error': str(e)
            })
    
    # 生成报告
    print("📄 生成V5增强版交易报告...")
    
    # 确保目录存在
    os.makedirs("reports/trading_signals", exist_ok=True)
    os.makedirs("reports/enhanced", exist_ok=True)
    
    today = datetime.now().strftime("%Y%m%d")
    
    # 保存Markdown报告
    report_content = generate_trading_report(results)
    report_file = f"reports/trading_signals/TradingAgents交易建议_{today}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # 保存JSON数据
    json_file = f"reports/trading_signals/交易信号数据_{today}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 统计结果
    success_count = sum(1 for r in results if 'error' not in r)
    buy_count = sum(1 for r in results if r.get('signal') == 'BUY')
    avoid_count = sum(1 for r in results if r.get('signal') == 'AVOID')
    hold_count = sum(1 for r in results if r.get('signal') == 'HOLD')
    
    print(f"\n✅ V5分析完成!")
    print(f"📊 分析统计:")
    print(f"   成功分析: {success_count}/{len(stocks)}只")
    print(f"   买入建议: {buy_count}只")
    print(f"   避让建议: {avoid_count}只")
    print(f"   观望建议: {hold_count}只")
    
    print(f"\n📄 输出文件:")
    print(f"   交易报告: {report_file}")
    print(f"   原始数据: {json_file}")
    
    # V5功能统计
    volume_analysis_count = sum(1 for r in results if 'volume_analysis' in r and 'error' not in r)
    bbi_analysis_count = sum(1 for r in results if 'bbi_analysis' in r and 'error' not in r)
    boll_analysis_count = sum(1 for r in results if 'bollinger_analysis' in r and 'error' not in r)
    
    print(f"\n🆕 V5新功能应用:")
    print(f"   量价关系分析: {volume_analysis_count}只")
    print(f"   BBI多空分析: {bbi_analysis_count}只")
    print(f"   增强布林带分析: {boll_analysis_count}只")
    
    return results

def main():
    """主函数"""
    print("🤖 TradingAgents V5 - 股票交易建议分析系统")
    print("🆕 新增功能: 量价关系 + BBI线 + 增强布林带分析")
    print("="*60)
    
    # 查找最新的选股报告
    report_paths = [
        "/Users/yangxu/StockTradebyZ/reports/daily_selection",
        "/Users/yangxu/StockTradebyZ/daily_result",
        "/Users/yangxu/StockTradebyZ/reports/daily_selection"
    ]
    
    report_file = None
    for path in report_paths:
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith('.md') and '选股' in f]
            if files:
                files.sort(reverse=True)  # 最新的在前
                report_file = os.path.join(path, files[0])
                break
    
    if not report_file:
        print("❌ 未找到选股报告，使用默认测试股票")
        stocks = [
            {'code': '000001', 'name': '平安银行'},
            {'code': '000002', 'name': '万科A'},
            {'code': '600036', 'name': '招商银行'},
            {'code': '000858', 'name': '五粮液'},
            {'code': '600519', 'name': '贵州茅台'}
        ]
    else:
        print(f"📋 使用选股报告: {report_file}")
        stocks = parse_daily_report(report_file)
        
        if not stocks:
            print("❌ 报告解析失败，使用默认测试股票")
            stocks = [
                {'code': '000001', 'name': '平安银行'},
                {'code': '000002', 'name': '万科A'},
                {'code': '600036', 'name': '招商银行'}
            ]
    
    # 分析股票（分析所有推荐股票）
    results = analyze_stocks_with_v5(stocks, limit=None)
    
    print(f"\n🎉 V5系统分析完成!")
    print(f"💡 提示: V5版本包含了您要求的所有功能:")
    print(f"   ✅ 交易量分析 (放量/缩量/量价关系)")
    print(f"   ✅ BBI多空指标")
    print(f"   ✅ 增强布林带分析")
    print(f"   ✅ KDJ指标分析")
    print(f"   ✅ 阻力位/支撑位分析")

if __name__ == "__main__":
    main()