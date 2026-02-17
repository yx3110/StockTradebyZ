#!/usr/bin/env python3
"""
获取完整的指数和行业板块信息
通过Tushare API获取所有可用的指数代码和申万行业分类
"""

import tushare as ts
import json
import pandas as pd
import time
from pathlib import Path
import sys
import os

# 添加项目根路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载配置
with open("config.json", 'r', encoding='utf-8') as f:
    config = json.load(f)

# 初始化Tushare
ts.set_token(config['tushare']['token'])
pro = ts.pro_api()

def get_all_indices():
    """获取所有主要指数信息"""
    print("获取所有指数信息...")
    
    all_indices = {}
    
    # 获取各个市场的指数
    markets = ['SSE', 'SZSE', 'CSI', 'SW', 'MSCI', 'CNI']
    
    for market in markets:
        try:
            print(f"获取{market}市场指数...")
            df = pro.index_basic(market=market)
            time.sleep(0.2)
            
            if not df.empty:
                market_indices = {}
                for _, row in df.iterrows():
                    ts_code = row['ts_code']
                    name = row['name']
                    category = row.get('category', '')
                    
                    # 筛选重要指数
                    if any(keyword in name for keyword in ['上证', '深证', '创业板', '科创', '中证', '沪深', '全A', '综指']):
                        market_indices[ts_code] = {
                            'name': name,
                            'market': market,
                            'category': category,
                            'publisher': row.get('publisher', ''),
                            'base_date': row.get('base_date', ''),
                        }
                
                all_indices[market] = market_indices
                print(f"{market}市场重要指数: {len(market_indices)}个")
            
        except Exception as e:
            print(f"获取{market}市场指数失败: {e}")
            continue
    
    return all_indices

def get_sw_industries():
    """获取申万行业分类"""
    print("获取申万行业分类...")
    
    industries = {}
    
    # 获取申万一级行业
    try:
        print("获取申万一级行业...")
        df_l1 = pro.index_classify(level='L1', src='SW2021')
        time.sleep(0.2)
        
        if not df_l1.empty:
            l1_industries = {}
            for _, row in df_l1.iterrows():
                industry_code = row['industry_code']
                industry_name = row['industry_name']
                l1_industries[industry_code] = {
                    'name': industry_name,
                    'level': 'L1',
                    'parent_code': ''
                }
            industries['L1'] = l1_industries
            print(f"申万一级行业: {len(l1_industries)}个")
        
    except Exception as e:
        print(f"获取申万一级行业失败: {e}")
    
    # 获取申万二级行业（部分）
    try:
        print("获取申万二级行业...")
        df_l2 = pro.index_classify(level='L2', src='SW2021')
        time.sleep(0.2)
        
        if not df_l2.empty:
            l2_industries = {}
            for _, row in df_l2.head(50).iterrows():  # 限制数量
                industry_code = row['industry_code']
                industry_name = row['industry_name']
                parent_code = row.get('parent_code', '')
                l2_industries[industry_code] = {
                    'name': industry_name,
                    'level': 'L2',
                    'parent_code': parent_code
                }
            industries['L2'] = l2_industries
            print(f"申万二级行业(部分): {len(l2_industries)}个")
        
    except Exception as e:
        print(f"获取申万二级行业失败: {e}")
    
    return industries

def get_sw_index_codes():
    """获取申万行业指数代码"""
    print("获取申万行业指数代码...")
    
    try:
        # 获取申万行业指数
        df = pro.index_basic(market='SW', fields='ts_code,name,category,base_date')
        time.sleep(0.2)
        
        if df.empty:
            return {}
        
        sw_indices = {}
        for _, row in df.iterrows():
            ts_code = row['ts_code']
            name = row['name']
            category = row.get('category', '')
            
            # 筛选一级行业指数
            if '一级行业' in category or len(name) <= 8:
                sw_indices[ts_code] = {
                    'name': name,
                    'category': category,
                    'base_date': row.get('base_date', '')
                }
        
        print(f"申万行业指数: {len(sw_indices)}个")
        return sw_indices
        
    except Exception as e:
        print(f"获取申万行业指数失败: {e}")
        return {}

def main():
    """主函数"""
    print("开始获取完整的指数和行业信息...")
    
    # 获取所有指数
    all_indices = get_all_indices()
    
    # 获取申万行业分类
    sw_industries = get_sw_industries()
    
    # 获取申万行业指数
    sw_index_codes = get_sw_index_codes()
    
    # 整理结果
    result = {
        'update_time': pd.Timestamp.now().isoformat(),
        'indices_by_market': all_indices,
        'sw_industries_classification': sw_industries,
        'sw_industry_indices': sw_index_codes,
        'summary': {
            'total_markets': len(all_indices),
            'total_indices': sum(len(indices) for indices in all_indices.values()),
            'sw_l1_industries': len(sw_industries.get('L1', {})),
            'sw_l2_industries': len(sw_industries.get('L2', {})),
            'sw_industry_indices': len(sw_index_codes)
        }
    }
    
    # 保存结果
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "complete_indices_sectors.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== 获取完成 ===")
    print(f"市场数量: {result['summary']['total_markets']}")
    print(f"重要指数: {result['summary']['total_indices']}个")
    print(f"申万一级行业: {result['summary']['sw_l1_industries']}个")
    print(f"申万二级行业: {result['summary']['sw_l2_industries']}个")
    print(f"申万行业指数: {result['summary']['sw_industry_indices']}个")
    print(f"结果已保存到: reports/complete_indices_sectors.json")
    
    # 生成常用指数列表
    print(f"\n=== 重要A股指数 ===")
    important_indices = {}
    
    for market, indices in all_indices.items():
        for code, info in indices.items():
            name = info['name']
            if any(keyword in name for keyword in ['上证指数', '深证成指', '创业板指', '科创50', '沪深300', '中证500', '中证1000', '中证2000', '中证全指']):
                important_indices[code] = name
                print(f"{code}: {name}")
    
    print(f"\n=== 申万一级行业指数 ===")
    if sw_index_codes:
        for code, info in list(sw_index_codes.items())[:15]:  # 显示前15个
            print(f"{code}: {info['name']}")
    
    return result

if __name__ == "__main__":
    main()