#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从缓存数据库重新分析权重优化结果
"""

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
import json

def load_cached_data():
    """加载缓存数据"""
    cache_db = "weight_optimization_cache.db"
    conn = sqlite3.connect(cache_db)
    
    # 加载指标数据
    query = """
    SELECT code, date, technical, fundamental, performance, 
           sentiment, risk_control, market_regime,
           return_1d, return_3d, return_5d, return_10d, return_20d
    FROM stock_indicators 
    WHERE return_5d IS NOT NULL
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df

def test_weight_combination(data, weights, name):
    """测试单个权重组合"""
    # 计算加权评分
    weighted_score = (
        data['technical'] * weights['technical'] +
        data['fundamental'] * weights['fundamental'] +
        data['performance'] * weights['performance'] +
        data['sentiment'] * weights['sentiment'] +
        data['risk_control'] * weights['risk_control'] +
        data['market_regime'] * weights['market_regime']
    ) * 100
    
    # 计算各期收益率的相关性和胜率
    results = {
        'name': name,
        'weights': weights,
        'total_weight': sum(weights.values()) * 100
    }
    
    # 对每个时间窗口进行分析
    for period in ['1d', '3d', '5d', '10d', '20d']:
        return_col = f'return_{period}'
        if return_col in data.columns:
            valid_data = data.dropna(subset=[return_col])
            
            if len(valid_data) > 100:
                # 计算相关性
                corr, p_value = stats.pearsonr(valid_data[return_col], weighted_score[:len(valid_data)])
                
                # 计算胜率 (高分股票未来上涨的概率)
                top_quartile = valid_data.nlargest(int(len(valid_data) * 0.25), weighted_score[:len(valid_data)])
                win_rate = (top_quartile[return_col] > 0).mean()
                
                # 计算平均收益
                avg_return = top_quartile[return_col].mean()
                
                # 计算夏普比率
                if top_quartile[return_col].std() > 0:
                    sharpe_ratio = avg_return / top_quartile[return_col].std()
                else:
                    sharpe_ratio = 0
                
                results[f'correlation_{period}'] = corr
                results[f'p_value_{period}'] = p_value
                results[f'win_rate_{period}'] = win_rate
                results[f'avg_return_{period}'] = avg_return
                results[f'sharpe_ratio_{period}'] = sharpe_ratio
                results[f'samples_{period}'] = len(valid_data)
    
    return results

def main():
    """主函数"""
    print("🔄 从缓存数据库重新分析权重优化结果...")
    
    # 加载数据
    data = load_cached_data()
    print(f"✅ 加载数据: {len(data):,} 条记录")
    print(f"📅 时间范围: {data['date'].min()} 到 {data['date'].max()}")
    
    # 定义要测试的权重组合
    weight_combinations = [
        # 原始v3.1权重
        {
            'technical': 0.60,
            'fundamental': 0.18, 
            'performance': 0.15,
            'sentiment': 0.05,
            'risk_control': 0.05,
            'market_regime': 0.05
        },
        # 技术面主导
        {
            'technical': 0.70,
            'fundamental': 0.15,
            'performance': 0.10,
            'sentiment': 0.02,
            'risk_control': 0.02,
            'market_regime': 0.01
        },
        # 基本面主导  
        {
            'technical': 0.40,
            'fundamental': 0.35,
            'performance': 0.15,
            'sentiment': 0.03,
            'risk_control': 0.04,
            'market_regime': 0.03
        },
        # 表现主导
        {
            'technical': 0.40,
            'fundamental': 0.15,
            'performance': 0.35,
            'sentiment': 0.03,
            'risk_control': 0.04,
            'market_regime': 0.03
        },
        # 均衡配置
        {
            'technical': 0.40,
            'fundamental': 0.20,
            'performance': 0.20,
            'sentiment': 0.07,
            'risk_control': 0.07,
            'market_regime': 0.06
        }
    ]
    
    names = ['原始v3.1', '技术主导', '基本面主导', '表现主导', '均衡配置']
    
    # 测试所有权重组合
    results = []
    for i, weights in enumerate(weight_combinations):
        result = test_weight_combination(data, weights, names[i])
        results.append(result)
    
    # 显示结果
    print("\n" + "="*80)
    print("🎯 权重优化对比分析结果")
    print("="*80)
    
    for result in results:
        print(f"\n🔧 【{result['name']}】")
        print("-" * 50)
        print("权重分布:")
        for dim, weight in result['weights'].items():
            print(f"  {dim:15s}: {weight:6.1%}")
        print(f"  {'总计':15s}: {result['total_weight']:6.1f}%")
        
        print("\n5日预测性能:")
        if 'correlation_5d' in result:
            print(f"  相关系数: {result['correlation_5d']:+.4f}")
            print(f"  胜率: {result['win_rate_5d']:.1%}")
            print(f"  平均收益: {result['avg_return_5d']:+.2f}%")
            print(f"  夏普比率: {result['sharpe_ratio_5d']:+.3f}")
            print(f"  样本数: {result['samples_5d']:,}")
        
        # 计算综合评分
        if 'win_rate_5d' in result and 'correlation_5d' in result:
            composite = abs(result['correlation_5d']) * 0.5 + result['win_rate_5d'] * 0.5
            print(f"  综合评分: {composite:.4f}")
    
    # 找出最佳方案
    print(f"\n🏆 最佳方案推荐:")
    valid_results = [r for r in results if 'win_rate_5d' in r]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['win_rate_5d'])
        print(f"  最高胜率方案: {best_result['name']} (胜率: {best_result['win_rate_5d']:.1%})")
        
        best_corr = max(valid_results, key=lambda x: abs(x['correlation_5d']))
        print(f"  最强相关方案: {best_corr['name']} (相关性: {best_corr['correlation_5d']:+.4f})")

if __name__ == "__main__":
    main()