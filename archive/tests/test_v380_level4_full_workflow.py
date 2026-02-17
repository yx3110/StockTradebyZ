#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V380 + Level 4 完整选股流程测试
验证集成系统的实际选股效果和质量评分差异化
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime
import json

sys.path.append('/Users/yangxu/StockTradebyZ')
from v380_level4_integrated_system import V380Level4IntegratedSystem

def test_full_stock_selection_workflow():
    """测试完整的股票选择流程"""
    print("🚀 V380 + Level 4 完整选股流程测试")
    print("=" * 60)

    # 1. 初始化集成系统
    print("1. 初始化V380 + Level 4集成系统...")
    system = V380Level4IntegratedSystem()
    print(f"✅ 系统初始化成功: {system.version} + Level 4")

    # 2. 测试股票池 (更大的股票池)
    test_stocks = [
        # 大盘蓝筹
        '000001.SZ', '000002.SZ', '000858.SZ', '002415.SZ',
        '600000.SH', '600036.SH', '600519.SH', '600887.SH',

        # 科技股
        '000725.SZ', '002230.SZ', '002415.SZ', '300059.SZ',
        '688981.SH', '002236.SZ', '300750.SZ',

        # 中小盘股
        '002215.SZ', '300363.SZ', '002050.SZ', '300033.SZ',
        '603259.SH', '688599.SH',

        # 周期股
        '600028.SH', '000858.SZ', '002142.SZ',
        '601088.SH', '600585.SH'
    ]

    test_date = '2025-09-23'
    print(f"\n2. 测试股票池: {len(test_stocks)}只股票")
    print(f"   测试日期: {test_date}")

    # 3. 运行预测
    print("\n3. 运行V380 + Level 4预测...")
    predictions = system.predict_scores_with_quality(test_stocks, test_date)

    if not predictions:
        print("❌ 预测失败，无结果返回")
        return

    # 4. 处理预测结果
    results = []
    for code, pred in predictions.items():
        if isinstance(pred, dict):
            result = {
                'code': code,
                'overall_score': pred.get('overall_score', 0),
                'quality_score': pred.get('quality_score', 0),
                'confidence_score': pred.get('confidence_score', 0),
                'short_term_score': pred.get('short_term_score', 0),
                'medium_term_score': pred.get('medium_term_score', 0),
                'long_term_score': pred.get('long_term_score', 0)
            }
        else:
            result = {
                'code': code,
                'overall_score': pred,
                'quality_score': 50,  # 默认中等质量
                'confidence_score': 0.5,
                'short_term_score': pred,
                'medium_term_score': pred,
                'long_term_score': pred
            }
        results.append(result)

    # 转换为DataFrame方便分析
    df = pd.DataFrame(results)

    # 5. 质量评分差异化分析
    print("\n4. 质量评分差异化分析:")
    print("=" * 40)

    quality_scores = df['quality_score'].values
    overall_scores = df['overall_score'].values

    quality_stats = {
        'count': len(quality_scores),
        'mean': np.mean(quality_scores),
        'std': np.std(quality_scores),
        'min': np.min(quality_scores),
        'max': np.max(quality_scores),
        'range': np.max(quality_scores) - np.min(quality_scores),
        'q25': np.percentile(quality_scores, 25),
        'q50': np.percentile(quality_scores, 50),
        'q75': np.percentile(quality_scores, 75)
    }

    print(f"  样本数量: {quality_stats['count']}")
    print(f"  均值: {quality_stats['mean']:.4f}")
    print(f"  标准差: {quality_stats['std']:.4f} {'✅' if quality_stats['std'] >= 0.15 else '❌'} (目标>=0.15)")
    print(f"  范围: [{quality_stats['min']:.4f}, {quality_stats['max']:.4f}] (宽度: {quality_stats['range']:.4f})")
    print(f"  分位数: Q25={quality_stats['q25']:.4f}, Q50={quality_stats['q50']:.4f}, Q75={quality_stats['q75']:.4f}")

    # 6. 质量评分分布分析
    print("\n5. 质量评分分布分析:")
    print("=" * 40)

    # 分等级统计
    high_quality = df[df['quality_score'] >= 0.7]
    medium_quality = df[(df['quality_score'] >= 0.3) & (df['quality_score'] < 0.7)]
    low_quality = df[df['quality_score'] < 0.3]

    print(f"  高质量股票 (>=0.7): {len(high_quality)}只 ({len(high_quality)/len(df)*100:.1f}%)")
    print(f"  中等质量股票 (0.3-0.7): {len(medium_quality)}只 ({len(medium_quality)/len(df)*100:.1f}%)")
    print(f"  低质量股票 (<0.3): {len(low_quality)}只 ({len(low_quality)/len(df)*100:.1f}%)")

    # 7. 综合评分vs质量评分相关性
    print("\n6. 综合评分 vs 质量评分关系:")
    print("=" * 40)

    correlation = np.corrcoef(overall_scores, quality_scores)[0, 1]
    print(f"  相关系数: {correlation:.4f}")
    print(f"  关系类型: {'正相关' if correlation > 0.3 else '负相关' if correlation < -0.3 else '弱相关'}")

    # 8. 顶级股票推荐 (基于质量评分)
    print("\n7. 基于质量评分的股票推荐:")
    print("=" * 40)

    # 按质量评分排序
    df_sorted = df.sort_values('quality_score', ascending=False)

    print("  🏆 顶级质量股票 (Top 5):")
    for i, row in df_sorted.head(5).iterrows():
        print(f"    {row['code']}: 质量={row['quality_score']:.3f}, 综合={row['overall_score']:.2f}, 置信度={row['confidence_score']:.3f}")

    print("\n  ⚠️ 低质量股票 (Bottom 5):")
    for i, row in df_sorted.tail(5).iterrows():
        print(f"    {row['code']}: 质量={row['quality_score']:.3f}, 综合={row['overall_score']:.2f}, 置信度={row['confidence_score']:.3f}")

    # 9. 多维度组合推荐
    print("\n8. 多维度组合推荐策略:")
    print("=" * 40)

    # 策略1: 高质量 + 高综合评分
    strategy1 = df[(df['quality_score'] >= 0.6) & (df['overall_score'] >= 70)]
    print(f"  📈 稳健增长策略 (高质量+高评分): {len(strategy1)}只")
    if len(strategy1) > 0:
        for _, row in strategy1.head(3).iterrows():
            print(f"    {row['code']}: Q={row['quality_score']:.3f}, S={row['overall_score']:.1f}")

    # 策略2: 高质量 + 中等评分 (价值投资)
    strategy2 = df[(df['quality_score'] >= 0.7) & (df['overall_score'] >= 40) & (df['overall_score'] < 70)]
    print(f"  💎 价值投资策略 (高质量+中等评分): {len(strategy2)}只")
    if len(strategy2) > 0:
        for _, row in strategy2.head(3).iterrows():
            print(f"    {row['code']}: Q={row['quality_score']:.3f}, S={row['overall_score']:.1f}")

    # 策略3: 高置信度选择
    strategy3 = df[(df['confidence_score'] >= 0.7) & (df['quality_score'] >= 0.5)]
    print(f"  🎯 高置信度策略: {len(strategy3)}只")
    if len(strategy3) > 0:
        for _, row in strategy3.head(3).iterrows():
            print(f"    {row['code']}: Q={row['quality_score']:.3f}, C={row['confidence_score']:.3f}")

    # 10. 保存测试结果
    print("\n9. 保存测试结果...")

    test_results = {
        'test_info': {
            'date': test_date,
            'stock_count': len(test_stocks),
            'system_version': f"{system.version} + Level 4"
        },
        'quality_score_stats': quality_stats,
        'correlation_analysis': {
            'overall_vs_quality_correlation': float(correlation)
        },
        'stock_recommendations': {
            'top_quality': df_sorted.head(5)[['code', 'quality_score', 'overall_score']].to_dict('records'),
            'strategy1_stable': strategy1[['code', 'quality_score', 'overall_score']].to_dict('records'),
            'strategy2_value': strategy2[['code', 'quality_score', 'overall_score']].to_dict('records'),
            'strategy3_confident': strategy3[['code', 'quality_score', 'confidence_score']].to_dict('records')
        },
        'full_results': df.to_dict('records')
    }

    # 保存到文件
    output_file = f"reports/v380_level4_workflow_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)

    print(f"✅ 测试结果已保存: {output_file}")

    # 11. 总结
    print("\n🎯 测试总结:")
    print("=" * 40)

    success_criteria = {
        'quality_differentiation': quality_stats['std'] >= 0.15,
        'range_coverage': quality_stats['range'] >= 0.5,
        'balanced_distribution': 0.2 <= quality_stats['mean'] <= 0.8
    }

    success_count = sum(success_criteria.values())

    print(f"  质量差异化: {'✅' if success_criteria['quality_differentiation'] else '❌'} (std={quality_stats['std']:.4f})")
    print(f"  范围覆盖: {'✅' if success_criteria['range_coverage'] else '❌'} (range={quality_stats['range']:.4f})")
    print(f"  分布平衡: {'✅' if success_criteria['balanced_distribution'] else '❌'} (mean={quality_stats['mean']:.4f})")

    if success_count == 3:
        print("\n🎉 V380 + Level 4集成系统测试 - 全面成功!")
        print("   Level 4 Quality Meta-learner完美解决了质量评分聚集问题")
        print("   系统已准备好用于实际生产环境")
    elif success_count >= 2:
        print("\n✅ V380 + Level 4集成系统测试 - 基本成功!")
        print("   主要功能正常，部分指标可进一步优化")
    else:
        print("\n⚠️ V380 + Level 4集成系统测试 - 需要优化")
        print("   建议检查模型参数和集成逻辑")

    return test_results

if __name__ == "__main__":
    test_full_stock_selection_workflow()