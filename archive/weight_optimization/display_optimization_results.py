#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
显示权重优化结果
"""

import sqlite3
import pandas as pd
import json

def display_optimization_results():
    """显示优化结果"""
    
    # 连接缓存数据库
    cache_db = "weight_optimization_cache.db"
    
    try:
        conn = sqlite3.connect(cache_db)
        
        # 检查是否有优化结果表
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"📊 缓存数据库包含的表: {tables}")
        
        # 检查数据量
        if 'stock_scoring_data' in tables:
            cursor.execute("SELECT COUNT(*) FROM stock_scoring_data")
            scoring_count = cursor.fetchone()[0]
            print(f"📈 评分数据量: {scoring_count:,} 条")
            
        if 'stock_returns_data' in tables:
            cursor.execute("SELECT COUNT(*) FROM stock_returns_data") 
            returns_count = cursor.fetchone()[0]
            print(f"💰 收益率数据量: {returns_count:,} 条")
        
        # 显示基本统计
        if 'stock_scoring_data' in tables:
            query = """
            SELECT 
                MIN(date) as start_date,
                MAX(date) as end_date,
                COUNT(DISTINCT code) as unique_stocks,
                COUNT(DISTINCT date) as unique_dates
            FROM stock_scoring_data
            """
            cursor.execute(query)
            result = cursor.fetchone()
            print(f"\n📅 数据时间范围: {result[0]} 到 {result[1]}")
            print(f"🏢 包含股票数量: {result[2]:,} 只")
            print(f"📊 交易日数量: {result[3]:,} 个")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ 无法读取缓存数据库: {e}")
    
    # 基于之前输出的结果分析
    print("\n" + "="*80)
    print("🎯 综合权重优化结果分析")
    print("="*80)
    
    print("\n✅ 成功完成的工作:")
    print("  • 评分数据加载: 24,802 条记录")
    print("  • 时间覆盖: 2024-01-02 到 2025-08-22")
    print("  • 股票覆盖: 4,759 只")
    print("  • 收益率计算: 2,740,922 条记录")
    print("  • 有效合并数据: 22,983 条记录")
    print("  • 权重组合测试: 7 个方案全部完成")
    
    print("\n🏆 最佳权重方案 (第1名):")
    print("  综合评分: 0.1255")
    print("  权重分布:")
    print("    • technical (技术面)      : 60.0%")
    print("    • fundamental (基本面)    : 20.0%")
    print("    • performance (表现)      : 20.0%")
    print("    • sentiment (情绪)        :  2.0%")
    print("    • risk_control (风控)     :  2.0%")
    print("    • market_regime (市况)    :  2.0%")
    print("    • 总计                     : 106.0%")
    
    print("\n📊 关键性能指标:")
    print("  • 5日相关性: -0.0366 (负相关需要进一步分析)")
    print("  • 5日胜率: 39.8% (低于随机水平)")
    
    print("\n⚠️ 分析结论:")
    print("  1. 当前所有测试的权重组合表现都不理想")
    print("  2. 相关性为负说明高分股票实际表现较差")
    print("  3. 胜率低于50%说明选股效果不佳")
    print("  4. 需要重新审视评分逻辑或者选择不同的优化策略")
    
    print("\n🔍 建议后续行动:")
    print("  • 检查评分逻辑是否存在反向信号")
    print("  • 尝试反向权重(低分股票)")
    print("  • 增加更多权重组合测试") 
    print("  • 分析不同市场环境下的表现")

if __name__ == "__main__":
    display_optimization_results()