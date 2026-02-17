#!/usr/bin/env python3
"""
分析优化后的权重配置效果和预期收益
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

def analyze_weight_performance():
    print("📊 分析优化后权重的收益表现")
    
    # 优化后的权重
    optimized_weights = {
        'technical': 0.35,
        'squeeze_momentum': 0.06, 
        'fundamental': 0.10,
        'performance': 0.22,
        'sentiment': 0.10,
        'risk_control': 0.10,
        'market_regime': 0.05
    }
    
    # 原始v3.2权重（用于对比）
    original_weights = {
        'technical': 0.50,
        'squeeze_momentum': 0.08,
        'fundamental': 0.14, 
        'performance': 0.15,
        'sentiment': 0.06,
        'risk_control': 0.04,
        'market_regime': 0.03
    }
    
    print("🔍 权重对比分析:")
    print(f"{'维度':<20} {'原始权重':<10} {'优化权重':<10} {'变化':<10}")
    print("-" * 55)
    for dim in optimized_weights:
        orig = original_weights[dim]
        opt = optimized_weights[dim]
        change = opt - orig
        change_str = f"{change:+.2f}"
        print(f"{dim:<20} {orig:.2f}     {opt:.2f}     {change_str}")
    
    # 分析历史数据上的表现
    db_path = 'factor_optimization/standard_factors.db'
    
    print(f"\n📈 基于历史数据分析收益表现...")
    
    try:
        with sqlite3.connect(db_path) as conn:
            # 获取最近30天的数据进行分析
            query = """
            SELECT stock_code, trade_date,
                   technical_score, squeeze_momentum_score, fundamental_score,
                   performance_score, sentiment_score, risk_control_score, 
                   market_regime_score,
                   return_1d, return_3d, return_5d, return_10d, return_20d
            FROM standard_factors 
            WHERE trade_date >= date('now', '-30 days')
            AND technical_score IS NOT NULL
            AND return_1d IS NOT NULL
            ORDER BY trade_date DESC
            LIMIT 10000
            """
            
            df = pd.read_sql_query(query, conn)
            
        print(f"✅ 加载了 {len(df)} 条记录用于分析")
        
        if len(df) == 0:
            print("❌ 没有找到可用的历史数据")
            return
        
        # 计算两种权重下的综合得分
        def calculate_score(row, weights):
            score = (
                row['technical_score'] * weights['technical'] +
                row['squeeze_momentum_score'] * weights['squeeze_momentum'] +
                row['fundamental_score'] * weights['fundamental'] +
                row['performance_score'] * weights['performance'] +
                row['sentiment_score'] * weights['sentiment'] +
                row['risk_control_score'] * weights['risk_control'] +
                row['market_regime_score'] * weights['market_regime']
            )
            return score
        
        df['original_score'] = df.apply(lambda row: calculate_score(row, original_weights), axis=1)
        df['optimized_score'] = df.apply(lambda row: calculate_score(row, optimized_weights), axis=1)
        
        # 分析不同得分区间的收益表现
        print(f"\n📊 收益表现分析:")
        
        for score_col, weight_name in [('original_score', '原始权重'), ('optimized_score', '优化权重')]:
            print(f"\n🎯 {weight_name}下的表现:")
            
            # 按得分分成5个等级
            df['score_quintile'] = pd.qcut(df[score_col], 5, labels=['低', '中低', '中等', '中高', '高'])
            
            performance = df.groupby('score_quintile').agg({
                'return_1d': ['mean', 'std', 'count'],
                'return_3d': 'mean',
                'return_5d': 'mean', 
                'return_10d': 'mean',
                'return_20d': 'mean'
            }).round(4)
            
            print("得分等级  |  1日收益(%)  |  3日收益(%)  |  5日收益(%)  | 10日收益(%)  | 20日收益(%)")
            print("-" * 85)
            
            for quintile in ['低', '中低', '中等', '中高', '高']:
                if quintile in performance.index:
                    r1d = performance.loc[quintile, ('return_1d', 'mean')] * 100
                    r3d = performance.loc[quintile, ('return_3d', 'mean')] * 100  
                    r5d = performance.loc[quintile, ('return_5d', 'mean')] * 100
                    r10d = performance.loc[quintile, ('return_10d', 'mean')] * 100
                    r20d = performance.loc[quintile, ('return_20d', 'mean')] * 100
                    count = performance.loc[quintile, ('return_1d', 'count')]
                    
                    print(f"{quintile:<8}  | {r1d:>9.2f}   | {r3d:>9.2f}   | {r5d:>9.2f}   | {r10d:>10.2f}   | {r20d:>10.2f}   ({count}样本)")
        
        # 计算优化效果
        print(f"\n🚀 优化效果总结:")
        
        # 高分股票的表现对比
        top_20_original = df.nlargest(int(len(df) * 0.2), 'original_score')
        top_20_optimized = df.nlargest(int(len(df) * 0.2), 'optimized_score')
        
        orig_returns = {
            '1日': top_20_original['return_1d'].mean() * 100,
            '5日': top_20_original['return_5d'].mean() * 100,
            '20日': top_20_original['return_20d'].mean() * 100
        }
        
        opt_returns = {
            '1日': top_20_optimized['return_1d'].mean() * 100, 
            '5日': top_20_optimized['return_5d'].mean() * 100,
            '20日': top_20_optimized['return_20d'].mean() * 100
        }
        
        print("前20%高分股票平均收益对比:")
        for period in ['1日', '5日', '20日']:
            orig = orig_returns[period]
            opt = opt_returns[period]
            improvement = opt - orig
            print(f"{period}收益: 原始 {orig:.2f}% → 优化 {opt:.2f}% (改善 {improvement:+.2f}%)")
            
    except Exception as e:
        print(f"❌ 分析过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_weight_performance()