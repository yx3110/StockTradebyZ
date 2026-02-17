#!/usr/bin/env python3

from claude_driven_analyzer import ClaudeDrivenAnalyzer
import json

# 创建分析器实例
analyzer = ClaudeDrivenAnalyzer()

# 获取单个股票的综合数据
stock_code = '002905'
print(f"获取股票 {stock_code} 的综合数据...")

stock_data = analyzer.get_comprehensive_stock_data(stock_code, days=30)

if stock_data:
    print("股票数据结构:")
    for key, value in stock_data.items():
        if isinstance(value, dict):
            print(f"  {key}: {type(value)} - {len(value)} keys")
        elif isinstance(value, list):
            print(f"  {key}: {type(value)} - {len(value)} items")
        else:
            print(f"  {key}: {type(value)} - {value}")
    
    print("\n开始构建分析提示词...")
    
    try:
        # 直接调用内部方法测试
        analysis_prompt = analyzer._build_analysis_prompt(stock_data, None)
        print(f"提示词长度: {len(analysis_prompt)}")
        print("提示词前500字符:")
        print(analysis_prompt[:500])
        
        print("\n开始调用Claude API...")
        
        # 测试Claude API调用
        response = analyzer.claude_client.messages.create(
            model=analyzer.analysis_model,
            max_tokens=4000,
            temperature=0.1,
            messages=[
                {
                    "role": "user", 
                    "content": analysis_prompt
                }
            ]
        )
        
        print(f"Response type: {type(response)}")
        print(f"Response content type: {type(response.content)}")
        print(f"Response content length: {len(response.content) if response.content else 0}")
        
        if response.content:
            print(f"First content type: {type(response.content[0])}")
            print(f"First content text length: {len(response.content[0].text) if hasattr(response.content[0], 'text') else 'No text attribute'}")
            print("Response前500字符:")
            print(response.content[0].text[:500] if hasattr(response.content[0], 'text') else "No text")
        
    except Exception as e:
        print(f"调用失败: {e}")
        import traceback
        traceback.print_exc()

else:
    print("无法获取股票数据")