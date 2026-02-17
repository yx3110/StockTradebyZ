#!/usr/bin/env python3
"""
测试扩展后的Tushare数据获取器
"""

import sys
import os
import logging

# 添加项目根路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pure_tushare_news_fetcher import PureTushareNewsFetcher

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_market_indices():
    """测试市场指数数据获取"""
    print("\n=== 测试市场指数数据获取 ===")
    
    fetcher = PureTushareNewsFetcher()
    
    try:
        indices_data = fetcher.get_market_indices_data(days=3)
        
        print(f"成功获取{len(indices_data)}个指数数据")
        
        for code, data in list(indices_data.items())[:5]:
            print(f"{code} {data['name']}: {data['latest_price']:.2f} ({data['change_pct']:+.2f}%)")
            
    except Exception as e:
        print(f"测试失败: {e}")

def test_sector_performance():
    """测试行业板块数据获取"""
    print("\n=== 测试申万二级行业板块数据获取 ===")
    
    fetcher = PureTushareNewsFetcher()
    
    try:
        sector_data = fetcher.get_sector_performance_data(days=3)
        
        print(f"成功获取{len(sector_data)}个行业板块数据")
        
        # 按涨跌幅排序显示前10个
        sectors_list = []
        for code, data in sector_data.items():
            sectors_list.append({
                'code': code,
                'name': data['name'],
                'change_pct': data['change_pct']
            })
        
        sectors_list.sort(key=lambda x: x['change_pct'], reverse=True)
        
        print("涨幅前10的行业:")
        for sector in sectors_list[:10]:
            print(f"{sector['code']} {sector['name']}: {sector['change_pct']:+.2f}%")
            
    except Exception as e:
        print(f"测试失败: {e}")

def test_news_headlines():
    """测试财经新闻获取"""
    print("\n=== 测试财经新闻获取 ===")
    
    fetcher = PureTushareNewsFetcher()
    
    try:
        news_data = fetcher.get_financial_news_headlines(days=2)
        
        print(f"成功获取{len(news_data)}条财经新闻")
        
        for i, news in enumerate(news_data[:5], 1):
            print(f"{i}. {news['title'][:50]}...")
            print(f"   重要性: {news['importance']} | 关键词: {', '.join(news['keywords'])}")
            
    except Exception as e:
        print(f"测试失败: {e}")

def test_comprehensive_market():
    """测试综合市场数据获取"""
    print("\n=== 测试综合市场数据获取 ===")
    
    fetcher = PureTushareNewsFetcher()
    
    try:
        market_data = fetcher.get_comprehensive_market_data(days=3)
        
        if 'error' in market_data:
            print(f"获取失败: {market_data['error']}")
            return
        
        summary = market_data['summary']
        sentiment = market_data['market_sentiment']
        ranking = market_data['sector_ranking']
        
        print("综合市场分析结果:")
        print(f"  指数数据: {summary['indices_count']}个")
        print(f"  行业数据: {summary['sectors_count']}个") 
        print(f"  新闻数据: {summary['news_count']}条")
        print(f"  市场情绪: {sentiment['sentiment_level']} (评分: {sentiment['sentiment_score']})")
        print(f"  情绪描述: {sentiment['description']}")
        
        if ranking['strong_sectors']:
            print("\n强势行业前5:")
            for sector in ranking['strong_sectors']:
                print(f"  {sector['name']}: {sector['change_pct']:+.2f}%")
        
        if ranking['weak_sectors']:
            print("\n弱势行业后5:")
            for sector in ranking['weak_sectors']:
                print(f"  {sector['name']}: {sector['change_pct']:+.2f}%")
                
    except Exception as e:
        print(f"测试失败: {e}")

def main():
    """主测试函数"""
    print("🚀 开始测试扩展后的Tushare数据获取器")
    
    # 测试各个功能模块
    test_market_indices()
    test_sector_performance()
    test_news_headlines()
    test_comprehensive_market()
    
    print("\n✅ 所有测试完成")

if __name__ == "__main__":
    main()