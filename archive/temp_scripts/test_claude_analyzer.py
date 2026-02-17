#!/usr/bin/env python3

from claude_driven_analyzer import ClaudeDrivenAnalyzer

# 创建分析器实例
analyzer = ClaudeDrivenAnalyzer()

# 测试获取基础数据
stock_code = '002905'
print(f"测试股票: {stock_code}")

# 测试基础数据获取
fundamental_data = analyzer._get_fundamental_data(stock_code)
print(f"基础数据: {fundamental_data}")

# 测试技术数据获取
technical_data = analyzer._get_technical_data(stock_code, 250)
print(f"技术数据长度: {len(technical_data) if technical_data else 0}")

# 测试市场数据获取
market_data = analyzer._get_market_data(stock_code)
print(f"市场数据: {market_data}")

print("测试完成")