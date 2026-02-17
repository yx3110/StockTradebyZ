#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试V3.8实时数据获取器的真实数据实现
"""

import sys
import os
from datetime import datetime, timedelta
import logging

# 添加项目路径
sys.path.append('/Users/yangxu/StockTradebyZ')

from data_adapter.database_manager import DatabaseManager
from incremental_learning.features.realtime_data_fetcher import RealtimeDataFetcher

def test_realtime_data_fetcher():
    """测试实时数据获取器的真实数据实现"""
    print("🧪 测试V3.8实时数据获取器的真实数据实现")
    print("=" * 50)

    # 配置日志
    logging.basicConfig(level=logging.INFO)

    try:
        # 初始化组件
        db_manager = DatabaseManager()
        fetcher = RealtimeDataFetcher(db_manager, cache_ttl=300)

        # 测试股票代码
        test_codes = ['000001', '600036', '002215']

        for code in test_codes:
            print(f"\n📊 测试股票: {code}")
            print("-" * 30)

            # 1. 测试当前价格数据
            print("1. 测试当前价格数据获取...")
            current_data = fetcher.get_current_price_data(code)
            if current_data:
                print(f"   ✅ 当前价格: {current_data.get('current_price', 'N/A')}")
                print(f"   ✅ 涨跌幅: {current_data.get('change_pct', 'N/A'):.3f}%")
                print(f"   ✅ 换手率: {current_data.get('turnover_rate', 'N/A'):.3f}%")
                print(f"   ✅ 数据日期: {current_data.get('trade_date', 'N/A')}")
            else:
                print("   ❌ 无法获取当前价格数据")

            # 2. 测试分钟级数据
            print("\n2. 测试分钟级数据获取...")
            start_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
            end_time = start_time + timedelta(hours=2)

            minute_data = fetcher.get_minute_data(code, start_time, end_time)
            if not minute_data.empty:
                print(f"   ✅ 分钟数据条数: {len(minute_data)}")
                print(f"   ✅ 价格范围: {minute_data['price'].min():.2f} - {minute_data['price'].max():.2f}")
                print(f"   ✅ 时间范围: {minute_data['datetime'].min()} - {minute_data['datetime'].max()}")
            else:
                print("   ❌ 无法获取分钟级数据")

        # 3. 测试市场快照
        print(f"\n🏛️ 测试市场快照数据")
        print("-" * 30)
        market_data = fetcher.get_market_snapshot()
        if market_data:
            market_index = market_data.get('market_index', {})
            print(f"   ✅ 上证指数: {market_index.get('sh_index', 'N/A')}")
            print(f"   ✅ 深证成指: {market_index.get('sz_index', 'N/A')}")
            print(f"   ✅ 创业板指: {market_index.get('cy_index', 'N/A')}")
            print(f"   ✅ 市场情绪: {market_data.get('market_sentiment', 'N/A'):.3f}")
            print(f"   ✅ 波动率指数: {market_data.get('volatility_index', 'N/A'):.3f}")
            print(f"   ✅ 数据日期: {market_data.get('trade_date', 'N/A')}")
        else:
            print("   ❌ 无法获取市场快照数据")

        # 4. 测试行业数据
        print(f"\n🏭 测试行业板块数据")
        print("-" * 30)
        test_sectors = ['银行', '软件服务', '电气设备']

        for sector in test_sectors:
            sector_data = fetcher.get_sector_data(sector)
            if sector_data:
                print(f"   ✅ {sector}行业:")
                print(f"     - 平均涨跌幅: {sector_data.get('change_pct', 'N/A'):.3f}%")
                print(f"     - 相对强度: {sector_data.get('relative_strength', 'N/A'):.3f}")
                print(f"     - 股票数量: {sector_data.get('stock_count', 'N/A')}")
                leading_stocks = sector_data.get('leading_stocks', [])
                if leading_stocks:
                    print(f"     - 龙头股票: {', '.join(leading_stocks[:3])}")
            else:
                print(f"   ❌ 无法获取{sector}行业数据")

        # 5. 测试缓存统计
        print(f"\n📈 缓存统计信息")
        print("-" * 30)
        cache_stats = fetcher.get_cache_stats()
        print(f"   ✅ 总缓存项数: {cache_stats['total_cache_items']}")
        print(f"   ✅ 缓存类型分布: {cache_stats['cache_types']}")
        print(f"   ✅ 内存使用: {cache_stats['memory_usage_mb']:.2f} MB")

        print(f"\n🎉 实时数据获取器测试完成！")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_realtime_data_fetcher()
    sys.exit(0 if success else 1)