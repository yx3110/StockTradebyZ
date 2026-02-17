#!/usr/bin/env python3

from claude_driven_analyzer import ClaudeDrivenAnalyzer
import json

# 创建分析器实例
analyzer = ClaudeDrivenAnalyzer()

# 测试单个股票分析
stock_code = '002905'
stock_name = '002905'

print(f"开始分析股票: {stock_code} ({stock_name})")

try:
    # 获取新闻数据
    print("获取新闻数据...")
    news_data = analyzer.news_fetcher.get_stock_news(stock_code, stock_name)
    print(f"获取到 {len(news_data)} 条新闻")
    
    # 获取综合数据
    print("获取股票综合数据...")
    stock_data = analyzer.get_comprehensive_stock_data(stock_code, days=30)
    
    if stock_data:
        # 进行AI分析
        print("开始AI分析...")
        result = analyzer.analyze_stock_with_claude(stock_data)
    else:
        result = None
        print("无法获取股票数据")
    
    print("分析结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
except Exception as e:
    print(f"分析失败: {e}")
    import traceback
    traceback.print_exc()