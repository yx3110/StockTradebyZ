#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速权重测试 - 直接对比优化前后的评分效果
"""

import sys
import os
import json
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入评分器
from scoring.v3.quantitative_scorer_v3_1 import QuantitativeScorerV31

def test_weight_comparison():
    """对比原始权重vs优化权重的评分效果"""
    
    print("🧪 快速权重对比测试")
    print("=" * 60)
    
    # 原始权重
    original_weights = {
        'technical': 0.60,
        'fundamental': 0.18,
        'performance': 0.15,
        'sentiment': 0.05,
        'risk_control': 0.05,
        'market_regime': 0.05
    }
    
    # 网格搜索优化权重
    optimized_weights = {
        'technical': 0.450,
        'fundamental': 0.100, 
        'performance': 0.200,
        'sentiment': 0.070,
        'risk_control': 0.070,
        'market_regime': 0.070
    }
    
    print("📊 权重对比:")
    for dim in original_weights:
        orig = original_weights[dim]
        opt = optimized_weights[dim]
        change = opt - orig
        print(f"  {dim:15s}: {orig:.1%} → {opt:.1%} ({change:+.1%})")
    
    print(f"\n🔍 使用现有数据测试权重效果...")
    
    # 加载已有的因子分数数据
    test_file = "reports/daily_selection_v3.1/analysis_data_20250822.json"
    
    if not os.path.exists(test_file):
        print(f"❌ 测试数据文件不存在: {test_file}")
        return
    
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'all_stocks_with_scores' not in data:
        print("❌ 数据格式错误")
        return
    
    stocks = data['all_stocks_with_scores'][:100]  # 取前100只股票测试
    print(f"📈 测试股票数量: {len(stocks)}")
    
    # 重新计算评分
    original_scores = []
    optimized_scores = []
    
    for stock in stocks:
        factor_scores = stock.get('factor_scores', {})
        
        # 原始权重计算
        orig_score = sum(
            original_weights[dim] * factor_scores.get(dim, 0) * 100
            for dim in original_weights
        )
        
        # 优化权重计算  
        opt_score = sum(
            optimized_weights[dim] * factor_scores.get(dim, 0) * 100
            for dim in optimized_weights
        )
        
        original_scores.append(orig_score)
        optimized_scores.append(opt_score)
    
    # 统计分析
    print(f"\n📊 评分统计对比:")
    print(f"{'指标':15s} {'原始权重':>12s} {'优化权重':>12s} {'变化':>10s}")
    print("-" * 60)
    
    metrics = {
        '平均分': (np.mean(original_scores), np.mean(optimized_scores)),
        '最高分': (np.max(original_scores), np.max(optimized_scores)),
        '最低分': (np.min(original_scores), np.min(optimized_scores)),
        '标准差': (np.std(original_scores), np.std(optimized_scores)),
        '90分以上': (sum(s >= 90 for s in original_scores), sum(s >= 90 for s in optimized_scores)),
        '80-90分': (sum(80 <= s < 90 for s in original_scores), sum(80 <= s < 90 for s in optimized_scores)),
    }
    
    for metric, (orig, opt) in metrics.items():
        if metric in ['90分以上', '80-90分']:
            orig_pct = orig / len(stocks) * 100
            opt_pct = opt / len(stocks) * 100
            change = opt_pct - orig_pct
            print(f"{metric:15s} {orig_pct:>10.1f}% {opt_pct:>10.1f}% {change:>+8.1f}%")
        else:
            change = opt - orig
            change_pct = change / orig * 100 if orig != 0 else 0
            print(f"{metric:15s} {orig:>11.1f} {opt:>11.1f} {change_pct:>+8.1f}%")
    
    # 分数分布分析
    print(f"\n📈 分数分布对比:")
    bins = [0, 60, 70, 80, 90, 100]
    
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
    
    # 排名变化分析
    orig_ranks = pd.Series(original_scores).rank(method='dense', ascending=False)
    opt_ranks = pd.Series(optimized_scores).rank(method='dense', ascending=False)
    rank_changes = opt_ranks - orig_ranks
    
    print(f"\n📊 排名变化分析:")
    print(f"  平均排名变化: {np.mean(rank_changes):+.1f}")
    print(f"  排名变化标准差: {np.std(rank_changes):.1f}")
    print(f"  最大排名提升: {np.min(rank_changes):+.0f} 位")
    print(f"  最大排名下降: {np.max(rank_changes):+.0f} 位")
    
    return {
        'original_scores': original_scores,
        'optimized_scores': optimized_scores,
        'correlation': correlation,
        'metrics': metrics
    }

if __name__ == "__main__":
    test_weight_comparison()