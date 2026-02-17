#!/usr/bin/env python3
"""
测试知行选股策略
"""
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# 导入选股器
sys.path.append(str(Path(__file__).parent / "stock_selctor"))
from Selector import ZhiXingSelector, compute_kdj
from data_adapter.stock_data_loader import StockDataLoader

def test_zhixing_strategy():
    """测试知行策略逻辑"""
    print("=" * 50)
    print("测试知行选股策略")
    print("=" * 50)
    
    # 初始化数据加载器和选股器
    data_loader = StockDataLoader()
    zhixing_selector = ZhiXingSelector()
    
    # 加载少量股票数据进行测试
    print("1. 加载测试数据...")
    test_codes = ["000001", "000002", "000063", "300433", "002842"]  # 测试几只股票
    
    for code in test_codes:
        print(f"\n测试股票 {code}:")
        try:
            # 获取股票历史数据
            hist_data = data_loader.load_stock_data_by_code(code, days=200, target_date="2025-09-09")
            if hist_data is None or hist_data.empty:
                print(f"  ❌ 无法获取 {code} 的历史数据")
                continue
                
            print(f"  📊 获取到 {len(hist_data)} 天的历史数据")
            print(f"  📅 数据范围: {hist_data['date'].min()} 到 {hist_data['date'].max()}")
            
            # 筛选到2025-09-09的数据
            target_date = pd.Timestamp("2025-09-09")
            hist_filtered = hist_data[hist_data["date"] <= target_date]
            
            if len(hist_filtered) < 114:
                print(f"  ⚠️  数据不足，只有 {len(hist_filtered)} 天数据，需要至少114天")
                continue
                
            # 测试知行策略的各个组件
            print("  🔍 测试知行指标计算...")
            
            # 复制数据并计算指标
            test_data = hist_filtered.copy()
            
            # 计算KDJ指标
            test_data = compute_kdj(test_data)
            
            # 计算知行指标
            test_data = zhixing_selector._compute_zhixing_indicators(test_data)
            
            # 获取最新数据
            latest = test_data.iloc[-1]
            prev = test_data.iloc[-2] if len(test_data) > 1 else test_data.iloc[-1]
            
            print(f"  📈 最新价格: {latest['close']:.2f}")
            print(f"  📊 J值: {latest['J']:.2f} (阈值: {zhixing_selector.j_threshold})")
            
            # 计算涨幅
            price_change = (latest['close'] / prev['close'] - 1) * 100
            print(f"  📈 涨幅: {price_change:.2f}% (范围: {zhixing_selector.min_change_pct*100}% ~ {zhixing_selector.max_change_pct*100}%)")
            
            # 计算振幅
            amplitude = (latest['high'] - latest['low']) / prev['close'] * 100
            print(f"  📊 振幅: {amplitude:.2f}% (阈值: {zhixing_selector.max_amplitude_pct*100}%)")
            
            # 知行趋势线指标
            if not pd.isna(latest['zhixing_short_trend']) and not pd.isna(latest['zhixing_multi_kong']):
                print(f"  📈 知行短期趋势线: {latest['zhixing_short_trend']:.4f}")
                print(f"  📊 知行多空线: {latest['zhixing_multi_kong']:.4f}")
                print(f"  🔄 趋势线 > 多空线: {latest['zhixing_short_trend'] > latest['zhixing_multi_kong']}")
                
                threshold_price = latest['zhixing_multi_kong'] * zhixing_selector.close_threshold_pct
                print(f"  💰 收盘价 > 多空线*97%: {latest['close']:.4f} > {threshold_price:.4f} = {latest['close'] > threshold_price}")
            else:
                print(f"  ❌ 知行指标计算失败")
                continue
            
            # 详细检查每个条件
            print("  🧪 测试各个筛选条件...")
            j_condition = latest['J'] < zhixing_selector.j_threshold
            print(f"    条件1 - J值 < {zhixing_selector.j_threshold}: {j_condition} (J={latest['J']:.2f})")
            
            price_change_ratio = latest['close'] / prev['close']
            price_change_condition = price_change_ratio > (1 + zhixing_selector.min_change_pct) and price_change_ratio < (1 + zhixing_selector.max_change_pct)
            print(f"    条件2 - 涨幅范围: {price_change_condition} ({price_change_ratio:.4f} > {1 + zhixing_selector.min_change_pct:.4f} and < {1 + zhixing_selector.max_change_pct:.4f})")
            
            amplitude_decimal = (latest['high'] - latest['low']) / prev['close']
            amplitude_condition = amplitude_decimal < zhixing_selector.max_amplitude_pct
            print(f"    条件3 - 振幅 < {zhixing_selector.max_amplitude_pct:.4f}: {amplitude_condition} ({amplitude_decimal:.4f})")
            
            trend_condition = latest['zhixing_short_trend'] > latest['zhixing_multi_kong']
            print(f"    条件4 - 短期趋势线 > 多空线: {trend_condition} ({latest['zhixing_short_trend']:.4f} > {latest['zhixing_multi_kong']:.4f})")
            
            threshold_price = latest['zhixing_multi_kong'] * zhixing_selector.close_threshold_pct
            close_condition = latest['close'] > threshold_price
            print(f"    条件5 - 收盘价 > 多空线*97%: {close_condition} ({latest['close']:.4f} > {threshold_price:.4f})")
            
            # 测试完整过滤逻辑
            print("  🧪 测试完整过滤逻辑...")
            passes = zhixing_selector._passes_filters(hist_filtered)
            print(f"  ✅ 通过知行策略筛选: {'是' if passes else '否'}")
            
        except Exception as e:
            print(f"  ❌ 测试 {code} 时发生错误: {e}")
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)

if __name__ == "__main__":
    test_zhixing_strategy()