#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正的权重测试 - 基于v3.1实际的直接加法逻辑
"""

import sys
import os
import json
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def corrected_weight_test():
    """基于v3.1直接加法逻辑的权重测试"""
    
    print("🔧 修正权重测试 - v3.1直接加法逻辑")
    print("=" * 60)
    
    # 原始权重分配 (基于v3.1配置)
    original_config_path = "scoring/v3/v3_1_optimized_config.json"
    with open(original_config_path, 'r', encoding='utf-8') as f:
        original_config = json.load(f)
    
    # 网格搜索优化权重
    optimized_config_path = "scoring/v3/fast_optimization_results/grid_search_optimized_config.json"  
    with open(optimized_config_path, 'r', encoding='utf-8') as f:
        optimized_config = json.load(f)
    
    print("📊 配置对比:")
    print("原始配置权重分布:")
    orig_category_weights = {}
    for category, weights in original_config['weights'].items():
        total = sum(v for k, v in weights.items() if not k.startswith('_'))
        orig_category_weights[category] = total
        print(f"  {category}: {total:.1%}")
    
    print("\n优化配置权重分布:")
    opt_category_weights = optimized_config['optimized_weights']
    for category, weight in opt_category_weights.items():
        print(f"  {category}: {weight:.1%}")
    
    print(f"\n权重总和对比:")
    print(f"  原始: {sum(orig_category_weights.values()):.1%}")
    print(f"  优化: {sum(opt_category_weights.values()):.1%}")
    
    # 加载测试数据
    test_file = "reports/daily_selection_v3.1/analysis_data_20250822.json"
    
    if not os.path.exists(test_file):
        print(f"❌ 测试数据文件不存在: {test_file}")
        return
    
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stocks = data['all_stocks_with_scores'][:100]
    print(f"\n📈 测试股票数量: {len(stocks)}")
    
    # 重新计算评分 - 使用v3.1的直接加法逻辑
    original_scores = []
    optimized_scores = []
    
    for stock in stocks:
        factor_scores = stock.get('factor_scores', {})
        
        # 原始总分 (v3.1实际逻辑：各维度分数直接相加)
        orig_score = sum(factor_scores.get(dim, 0) * 100 for dim in orig_category_weights.keys())
        
        # 优化总分 - 需要调整为直接加法逻辑
        # 方案1: 按比例调整各维度分数
        adjusted_factor_scores = {}
        for dim in opt_category_weights.keys():
            original_weight = orig_category_weights.get(dim, 0)
            optimized_weight = opt_category_weights[dim] 
            
            if original_weight > 0:
                scale_factor = optimized_weight / original_weight
                adjusted_factor_scores[dim] = factor_scores.get(dim, 0) * scale_factor * 100
            else:
                adjusted_factor_scores[dim] = factor_scores.get(dim, 0) * optimized_weight * 100
        
        opt_score = sum(adjusted_factor_scores.values())
        
        original_scores.append(orig_score)
        optimized_scores.append(opt_score)
    
    # 统计分析
    print(f"\n📊 评分统计对比 (修正后):")
    print(f"{'指标':15s} {'原始权重':>12s} {'优化权重':>12s} {'变化':>10s}")
    print("-" * 60)
    
    metrics = {
        '平均分': (np.mean(original_scores), np.mean(optimized_scores)),
        '最高分': (np.max(original_scores), np.max(optimized_scores)),
        '最低分': (np.min(original_scores), np.min(optimized_scores)),
        '标准差': (np.std(original_scores), np.std(optimized_scores)),
        '90分以上': (sum(s >= 90 for s in original_scores), sum(s >= 90 for s in optimized_scores)),
        '80-90分': (sum(80 <= s < 90 for s in original_scores), sum(80 <= s < 90 for s in optimized_scores)),
        '70-80分': (sum(70 <= s < 80 for s in original_scores), sum(70 <= s < 80 for s in optimized_scores)),
    }
    
    for metric, (orig, opt) in metrics.items():
        if metric in ['90分以上', '80-90分', '70-80分']:
            orig_pct = orig / len(stocks) * 100
            opt_pct = opt / len(stocks) * 100
            change = opt_pct - orig_pct
            print(f"{metric:15s} {orig_pct:>10.1f}% {opt_pct:>10.1f}% {change:>+8.1f}%")
        else:
            change = opt - orig
            change_pct = change / orig * 100 if orig != 0 else 0
            print(f"{metric:15s} {orig:>11.1f} {opt:>11.1f} {change_pct:>+8.1f}%")
    
    # 详细分数分布
    print(f"\n📈 分数分布对比 (修正后):")
    bins = [0, 50, 60, 70, 80, 90, 100]
    
    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]
        orig_count = sum(low <= s < high for s in original_scores)
        opt_count = sum(low <= s < high for s in optimized_scores)
        
        orig_pct = orig_count / len(stocks) * 100
        opt_pct = opt_count / len(stocks) * 100
        change = opt_pct - orig_pct
        
        print(f"  {low:2d}-{high:2d}分: 原始{orig_pct:5.1f}% → 优化{opt_pct:5.1f}% ({change:+5.1f}%)")
    
    # 相关性分析
    correlation = np.corrcoef(original_scores, optimized_scores)[0, 1]
    print(f"\n🔗 原始与优化评分相关性: {correlation:.3f}")
    
    # 具体案例分析
    print(f"\n📋 具体案例分析 (前10只股票):")
    print(f"{'股票代码':>8s} {'原始分':>8s} {'优化分':>8s} {'变化':>8s} {'排名变化':>10s}")
    print("-" * 50)
    
    # 计算排名
    orig_ranks = pd.Series(original_scores).rank(method='dense', ascending=False).astype(int)
    opt_ranks = pd.Series(optimized_scores).rank(method='dense', ascending=False).astype(int)
    
    for i in range(min(10, len(stocks))):
        stock_code = stocks[i]['stock_code']
        orig_score = original_scores[i]
        opt_score = optimized_scores[i] 
        score_change = opt_score - orig_score
        rank_change = orig_ranks[i] - opt_ranks[i]  # 正数表示排名上升
        
        print(f"{stock_code:>8s} {orig_score:>8.1f} {opt_score:>8.1f} {score_change:>+8.1f} {rank_change:>+8d}位")

if __name__ == "__main__":
    corrected_weight_test()