#!/usr/bin/env python3
"""
调试TePu战法选股逻辑
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_selctor.Selector import BreakoutVolumeKDJSelector
from data_adapter.stock_data_loader import StockDataLoader
from data_adapter.database_manager import DatabaseManager
import pandas as pd

def debug_tepu_selector():
    """调试TePu战法选股过程"""
    print("=== 调试TePu战法选股逻辑 ===")
    
    # 初始化组件
    db_manager = DatabaseManager()
    data_loader = StockDataLoader()
    
    # 获取股票基本信息
    securities = db_manager.get_all_securities()
    print(f"总共有 {len(securities)} 只证券")
    
    # 加载数据
    print("加载股票数据...")
    stock_data = data_loader.load_all_stock_data(days=120)
    print(f"成功加载 {len(stock_data)} 只证券的数据")
    
    # 初始化TePu选股器
    config = {
        'params': {
            'j_threshold': 1,
            'up_threshold': 3.0,
            'volume_threshold': 0.6667,
            'price_range_pct': 1
        }
    }
    
    tepu_selector = BreakoutVolumeKDJSelector('TePu战法', config)
    
    # 测试几只我们知道涨幅很高的股票
    test_codes = ['839719', '300204', '300530']  # 宁新新材(30%), 舒泰神(20%), 领湃科技(20%)
    
    for code in test_codes:
        if code in stock_data:
            print(f"\n=== 测试股票 {code} ===")
            hist = stock_data[code]
            
            # 显示最后几天的数据
            print("最近5天数据:")
            recent_data = hist.tail(5)[['date', 'close', 'volume', 'pct_chg']]
            print(recent_data.to_string())
            
            # 检查是否通过选股
            try:
                result = tepu_selector.select([code], stock_data, target_date='2025-05-21')
                print(f"选股结果: {result}")
                
                # 详细分析为什么没通过
                print("\n详细检查:")
                
                # 检查数据完整性
                if len(hist) < 30:
                    print(f"❌ 数据不足: 只有 {len(hist)} 天数据，需要至少30天")
                    continue
                    
                # 获取最后一天数据
                last_row = hist.iloc[-1]
                print(f"目标日期: {last_row['date']}")
                print(f"收盘价: {last_row['close']}")
                print(f"成交量: {last_row['volume']}")
                print(f"涨跌幅: {last_row['pct_chg']}%")
                
                # 检查涨跌幅条件
                if last_row['pct_chg'] < config['params']['up_threshold']:
                    print(f"❌ 涨幅不足: {last_row['pct_chg']}% < {config['params']['up_threshold']}%")
                else:
                    print(f"✅ 涨幅符合: {last_row['pct_chg']}% >= {config['params']['up_threshold']}%")
                
                # 检查技术指标（如果有的话）
                if 'kdj_j' in hist.columns:
                    last_j = hist['kdj_j'].iloc[-1]
                    print(f"KDJ_J值: {last_j}")
                    if last_j < config['params']['j_threshold']:
                        print(f"❌ KDJ_J不足: {last_j} < {config['params']['j_threshold']}")
                    else:
                        print(f"✅ KDJ_J符合: {last_j} >= {config['params']['j_threshold']}")
                else:
                    print("⚠️ 缺少KDJ_J技术指标数据")
                    
            except Exception as e:
                print(f"❌ 选股过程出错: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    debug_tepu_selector()