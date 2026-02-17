#!/usr/bin/env python3
"""
单独测试知行战法选股器
"""
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

# 导入相关模块
sys.path.append(str(Path(__file__).parent / "stock_selctor"))
from Selector import ZhiXingSelector
from data_adapter.stock_data_loader import StockDataLoader

def test_zhixing_selector_only():
    """单独测试知行战法选股功能"""
    print("=" * 50)
    print("知行战法独立选股测试")
    print("=" * 50)
    
    # 初始化选股器
    zhixing_selector = ZhiXingSelector()
    data_loader = StockDataLoader()
    
    print("1. 初始化知行选股器...")
    print(f"   参数设置:")
    print(f"   - J阈值: {zhixing_selector.j_threshold}")
    print(f"   - 涨幅范围: [{zhixing_selector.min_change_pct*100:.1f}%, {zhixing_selector.max_change_pct*100:.1f}%]")
    print(f"   - 振幅上限: {zhixing_selector.max_amplitude_pct*100:.1f}%")
    print(f"   - 收盘价阈值: {zhixing_selector.close_threshold_pct*100:.1f}%")
    
    print("\n2. 加载股票数据...")
    try:
        # 获取所有股票代码
        all_codes = data_loader.get_all_stock_codes()
        print(f"   获取到 {len(all_codes)} 只股票代码")
        
        # 加载前100只股票的数据进行测试
        test_codes = all_codes[:100]
        target_date = pd.Timestamp("2025-09-09")
        
        print(f"   测试前{len(test_codes)}只股票...")
        
        # 构建数据字典
        stock_data = {}
        for code in test_codes:
            hist = data_loader.load_stock_data_by_code(code, days=150, target_date="2025-09-09")
            if hist is not None and len(hist) >= 120:  # 确保有足够数据
                stock_data[code] = hist
        
        print(f"   成功加载 {len(stock_data)} 只股票的历史数据")
        
        print("\n3. 运行知行战法选股...")
        selected_stocks = zhixing_selector.select(target_date, stock_data)
        
        print(f"   ✅ 知行战法选中股票数量: {len(selected_stocks)}")
        if selected_stocks:
            print("   📈 选中股票列表:")
            for i, code in enumerate(selected_stocks, 1):
                print(f"      {i}. {code}")
        else:
            print("   📉 没有股票满足知行战法条件")
        
        # 如果有选中的股票，显示详细信息
        if selected_stocks:
            print("\n4. 选中股票详细分析:")
            for code in selected_stocks[:3]:  # 只显示前3只
                if code in stock_data:
                    hist = stock_data[code]
                    hist_filtered = hist[hist["date"] <= target_date]
                    if len(hist_filtered) >= 2:
                        latest = hist_filtered.iloc[-1]
                        prev = hist_filtered.iloc[-2]
                        
                        print(f"\n   股票 {code}:")
                        print(f"     最新价格: {latest['close']:.2f}")
                        print(f"     涨幅: {(latest['close']/prev['close']-1)*100:.2f}%")
                        print(f"     振幅: {(latest['high']-latest['low'])/prev['close']*100:.2f}%")
                        
                        # 计算KDJ和知行指标
                        from Selector import compute_kdj
                        test_data = hist_filtered.copy()
                        test_data = compute_kdj(test_data)
                        test_data = zhixing_selector._compute_zhixing_indicators(test_data)
                        latest_with_indicators = test_data.iloc[-1]
                        
                        print(f"     J值: {latest_with_indicators['J']:.2f}")
                        print(f"     短期趋势线: {latest_with_indicators['zhixing_short_trend']:.4f}")
                        print(f"     多空线: {latest_with_indicators['zhixing_multi_kong']:.4f}")
            
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)

if __name__ == "__main__":
    test_zhixing_selector_only()