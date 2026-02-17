#!/usr/bin/env python3
"""
简单的雪球API测试脚本
"""
import requests
import json
import sys
from pathlib import Path

# 添加路径
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir.parent))

from xueqiu_config import get_xueqiu_cookie

def test_simple_xueqiu_request():
    """测试简单的雪球API请求"""
    
    cookie = get_xueqiu_cookie()
    print(f"Cookie长度: {len(cookie)}")
    
    # 设置session
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Referer': 'https://xueqiu.com/',
        'Cookie': cookie
    })
    
    # 测试简单的API调用
    test_symbol = "SH000001"  # 上证指数
    api_url = f"https://stock.xueqiu.com/v5/stock/timeline/list.json"
    
    params = {
        'symbol': test_symbol,
        'count': 10,
        'source': 'all'
    }
    
    try:
        print(f"测试API调用: {api_url}")
        print(f"参数: {params}")
        
        response = session.get(api_url, params=params, timeout=10)
        
        print(f"HTTP状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"响应数据结构: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                
                if 'error_code' in data:
                    print(f"错误代码: {data.get('error_code')}")
                    print(f"错误描述: {data.get('error_description', '无')}")
                
                if 'list' in data and data['list']:
                    print(f"获取到{len(data['list'])}条讨论数据")
                    
                    # 显示第一条数据的结构
                    first_item = data['list'][0]
                    print(f"第一条数据字段: {list(first_item.keys())}")
                else:
                    print("响应中没有找到list字段或数据为空")
                    
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}")
                print(f"响应内容前500字符: {response.text[:500]}")
        else:
            print(f"HTTP请求失败: {response.status_code}")
            print(f"响应内容: {response.text[:500]}")
            
    except Exception as e:
        print(f"请求异常: {e}")

if __name__ == "__main__":
    test_simple_xueqiu_request()