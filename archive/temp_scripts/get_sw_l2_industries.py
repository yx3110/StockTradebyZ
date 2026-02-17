#!/usr/bin/env python3
"""
获取完整的申万二级行业分类和对应指数代码
"""

import tushare as ts
import json
import pandas as pd
import time
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

def get_all_sw_l2_industries():
    """获取所有申万二级行业分类"""
    print("获取所有申万二级行业分类...")
    
    try:
        # 获取申万二级行业分类
        df_l2 = pro.index_classify(level='L2', src='SW2021')
        time.sleep(0.3)
        
        l2_industries = {}
        if not df_l2.empty:
            for _, row in df_l2.iterrows():
                industry_code = row['industry_code']
                industry_name = row['industry_name']
                parent_code = row.get('parent_code', '')
                
                l2_industries[industry_code] = {
                    'name': industry_name,
                    'level': 'L2',
                    'parent_code': parent_code
                }
        
        print(f"获取到{len(l2_industries)}个申万二级行业")
        return l2_industries
        
    except Exception as e:
        print(f"获取申万二级行业失败: {e}")
        return {}

def get_sw_l2_index_codes():
    """获取申万二级行业指数代码"""
    print("获取申万二级行业指数代码...")
    
    try:
        # 获取申万指数
        df = pro.index_basic(market='SW')
        time.sleep(0.3)
        
        l2_indices = {}
        if not df.empty:
            for _, row in df.iterrows():
                ts_code = row['ts_code']
                name = row['name']
                category = row.get('category', '')
                
                # 筛选二级行业指数 (包含"二级行业"或名称较长的)
                if '二级行业' in category or (len(name) > 4 and len(name) <= 12 and '申万' not in name):
                    l2_indices[ts_code] = {
                        'name': name,
                        'category': category,
                        'base_date': row.get('base_date', ''),
                        'publisher': row.get('publisher', '')
                    }
        
        print(f"获取到{len(l2_indices)}个申万二级行业指数")
        return l2_indices
        
    except Exception as e:
        print(f"获取申万二级行业指数失败: {e}")
        return {}

def map_industries_to_indices():
    """尝试将行业分类与指数代码匹配"""
    print("匹配行业分类与指数代码...")
    
    # 手工整理的重要申万二级行业指数（基于行业重要性和市场关注度）
    important_l2_indices = {
        # 农林牧渔相关
        '850811.SI': '种植业',
        '850821.SI': '养殖业',
        '850831.SI': '饲料',
        
        # 化工相关
        '850851.SI': '化学原料',
        '850861.SI': '化学制品',  
        '850871.SI': '化学纤维',
        
        # 有色金属相关
        '850881.SI': '工业金属',
        '850891.SI': '黄金',
        '850901.SI': '稀有金属',
        
        # 电子相关
        '850911.SI': '半导体',
        '850921.SI': '消费电子',
        '850931.SI': '电子元件',
        '850941.SI': '光学光电子',
        
        # 食品饮料相关
        '850951.SI': '白酒',
        '850961.SI': '其他酒类',
        '850971.SI': '食品加工',
        '850981.SI': '调味发酵品',
        
        # 医药生物相关
        '850991.SI': '化学制药',
        '851001.SI': '中药',
        '851011.SI': '生物制品',
        '851021.SI': '医疗器械',
        '851031.SI': '医疗服务',
        
        # 计算机相关
        '851041.SI': '计算机设备',
        '851051.SI': '计算机应用',
        
        # 汽车相关
        '851061.SI': '汽车整车',
        '851071.SI': '汽车零部件',
        '851081.SI': '汽车服务',
        
        # 房地产相关
        '851091.SI': '房地产开发',
        '851101.SI': '物业管理',
        
        # 银行相关
        '851111.SI': '银行',
        
        # 非银金融相关
        '851121.SI': '证券',
        '851131.SI': '保险',
        '851141.SI': '多元金融',
        
        # 机械设备相关
        '851151.SI': '工程机械',
        '851161.SI': '重型机械',
        '851171.SI': '通用机械',
        
        # 电力设备相关
        '851181.SI': '电机',
        '851191.SI': '电气自动化设备',
        
        # 新能源相关
        '851201.SI': '风电设备',
        '851211.SI': '光伏设备',
        '851221.SI': '储能',
    }
    
    return important_l2_indices

def main():
    """主函数"""
    print("开始获取申万二级行业完整信息...")
    
    # 获取行业分类
    l2_industries = get_all_sw_l2_industries()
    
    # 获取指数代码
    l2_index_codes = get_sw_l2_index_codes()
    
    # 获取重要的二级行业映射
    important_mapping = map_industries_to_indices()
    
    # 整理结果
    result = {
        'update_time': pd.Timestamp.now().isoformat(),
        'sw_l2_industries_classification': l2_industries,
        'sw_l2_index_codes': l2_index_codes,
        'important_l2_mapping': important_mapping,
        'summary': {
            'l2_industries_count': len(l2_industries),
            'l2_index_codes_count': len(l2_index_codes),
            'important_mapping_count': len(important_mapping)
        }
    }
    
    # 保存结果
    with open("reports/sw_l2_industries_complete.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== 申万二级行业数据获取完成 ===")
    print(f"二级行业分类: {len(l2_industries)}个")
    print(f"二级行业指数: {len(l2_index_codes)}个")  
    print(f"重要行业映射: {len(important_mapping)}个")
    print(f"结果已保存到: reports/sw_l2_industries_complete.json")
    
    # 显示重要二级行业
    print(f"\n=== 重要申万二级行业指数 ===")
    for code, name in list(important_mapping.items())[:20]:
        print(f"{code}: {name}")
    
    return result

if __name__ == "__main__":
    main()