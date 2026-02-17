#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.8 原系统 vs V380+Level 4集成系统 性能对比分析
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime
import json
import matplotlib.pyplot as plt

sys.path.append('/Users/yangxu/StockTradebyZ')
from v380_advanced_incremental_ml_system import V380AdvancedIncrementalMLSystem
from v380_level4_integrated_system import V380Level4IntegratedSystem

def simulate_v38_quality_scores(predictions):
    """
    模拟V3.8原系统的固定质量评分算法
    使用固定权重: 40%置信度 + 30%评分 + 30%一致性
    """
    quality_scores = []

    for code, pred in predictions.items():
        if isinstance(pred, dict):
            confidence = pred.get('confidence_score', 0.5)
            overall_score = pred.get('overall_score', 50) / 100  # 标准化到0-1

            # 模拟一致性评分 (基于短期、中期、长期评分的一致性)
            short_term = pred.get('short_term_score', 50) / 100
            medium_term = pred.get('medium_term_score', 50) / 100
            long_term = pred.get('long_term_score', 50) / 100

            # 计算一致性 (方差的倒数)
            scores_variance = np.var([short_term, medium_term, long_term])
            consistency = 1.0 / (1.0 + scores_variance)  # 高一致性 = 低方差

            # V3.8固定权重公式
            quality_score = (
                confidence * 0.4 +       # 40%置信度
                overall_score * 0.3 +    # 30%综合评分
                consistency * 0.3        # 30%一致性
            )

            quality_scores.append(quality_score)
        else:
            # 简单评分情况
            quality_scores.append(0.6)  # 默认中等质量

    return np.array(quality_scores)

def compare_v38_vs_level4():
    """对比V3.8原系统与Level 4集成系统的性能差异"""
    print("🔍 V3.8 原系统 vs V380+Level 4 性能对比分析")
    print("=" * 70)

    # 1. 初始化两个系统
    print("1. 初始化对比系统...")

    # V3.8原系统
    v38_system = V380AdvancedIncrementalMLSystem()
    print("✅ V3.8原系统初始化完成")

    # V380+Level 4集成系统
    level4_system = V380Level4IntegratedSystem()
    print("✅ V380+Level 4集成系统初始化完成")

    # 2. 测试股票池
    test_stocks = [
        # 不同类型股票的代表性样本
        '000001.SZ', '000002.SZ', '000858.SZ', '002415.SZ',  # 大盘蓝筹
        '600000.SH', '600036.SH', '600519.SH', '600887.SH',  # 银行保险
        '000725.SZ', '002230.SZ', '300059.SZ', '688981.SH',  # 科技成长
        '002215.SZ', '300363.SZ', '002050.SZ', '603259.SH',  # 中小盘
        '600028.SH', '002142.SZ', '601088.SH', '600585.SH'   # 周期资源
    ]

    test_date = '2025-09-23'
    print(f"\n2. 测试设置: {len(test_stocks)}只股票，日期: {test_date}")

    # 3. 运行V3.8原系统预测
    print("\n3. 运行V3.8原系统预测...")
    v38_predictions = v38_system.predict_scores(test_stocks, test_date)

    # 模拟V3.8固定质量评分
    v38_quality_scores = simulate_v38_quality_scores(v38_predictions)

    print(f"✅ V3.8预测完成: {len(v38_predictions)}只股票")

    # 4. 运行Level 4集成系统预测
    print("\n4. 运行Level 4集成系统预测...")
    level4_predictions = level4_system.predict_scores_with_quality(test_stocks, test_date)

    # 提取Level 4质量评分
    level4_quality_scores = []
    for code in test_stocks:
        if code in level4_predictions and isinstance(level4_predictions[code], dict):
            level4_quality_scores.append(level4_predictions[code].get('quality_score', 0.5))
        else:
            level4_quality_scores.append(0.5)
    level4_quality_scores = np.array(level4_quality_scores)

    print(f"✅ Level 4预测完成: {len(level4_predictions)}只股票")

    # 5. 质量评分差异化对比
    print("\n5. 质量评分差异化对比:")
    print("=" * 50)

    # V3.8统计
    v38_stats = {
        'count': len(v38_quality_scores),
        'mean': np.mean(v38_quality_scores),
        'std': np.std(v38_quality_scores),
        'min': np.min(v38_quality_scores),
        'max': np.max(v38_quality_scores),
        'range': np.max(v38_quality_scores) - np.min(v38_quality_scores)
    }

    # Level 4统计
    level4_stats = {
        'count': len(level4_quality_scores),
        'mean': np.mean(level4_quality_scores),
        'std': np.std(level4_quality_scores),
        'min': np.min(level4_quality_scores),
        'max': np.max(level4_quality_scores),
        'range': np.max(level4_quality_scores) - np.min(level4_quality_scores)
    }

    print("📊 V3.8原系统 (固定权重算法):")
    print(f"   均值: {v38_stats['mean']:.4f}")
    print(f"   标准差: {v38_stats['std']:.4f} {'✅' if v38_stats['std'] >= 0.15 else '❌'} (目标>=0.15)")
    print(f"   范围: [{v38_stats['min']:.4f}, {v38_stats['max']:.4f}] (宽度: {v38_stats['range']:.4f})")

    print("\n🚀 Level 4集成系统 (机器学习算法):")
    print(f"   均值: {level4_stats['mean']:.4f}")
    print(f"   标准差: {level4_stats['std']:.4f} {'✅' if level4_stats['std'] >= 0.15 else '❌'} (目标>=0.15)")
    print(f"   范围: [{level4_stats['min']:.4f}, {level4_stats['max']:.4f}] (宽度: {level4_stats['range']:.4f})")

    # 6. 改进效果量化
    print("\n6. 改进效果量化:")
    print("=" * 50)

    improvements = {
        'std_improvement': float(level4_stats['std'] / v38_stats['std']) if v38_stats['std'] > 0 else float('inf'),
        'range_improvement': float(level4_stats['range'] / v38_stats['range']) if v38_stats['range'] > 0 else float('inf'),
        'differentiation_achieved': bool(level4_stats['std'] >= 0.15),
        'v38_differentiation_achieved': bool(v38_stats['std'] >= 0.15)
    }

    print(f"📈 标准差改进: {improvements['std_improvement']:.2f}倍")
    print(f"📈 范围扩展: {improvements['range_improvement']:.2f}倍")
    print(f"🎯 差异化目标: V3.8{'✅' if improvements['v38_differentiation_achieved'] else '❌'} → Level 4{'✅' if improvements['differentiation_achieved'] else '❌'}")

    # 7. 股票质量排名对比
    print("\n7. 股票质量排名对比 (Top 10):")
    print("=" * 50)

    # 创建对比数据框
    comparison_df = pd.DataFrame({
        'code': test_stocks[:len(v38_quality_scores)],
        'v38_quality': v38_quality_scores,
        'level4_quality': level4_quality_scores[:len(v38_quality_scores)]
    })

    # 按Level 4质量评分排序
    comparison_df = comparison_df.sort_values('level4_quality', ascending=False)

    print("代码\t\tV3.8质量\tLevel 4质量\t排名变化")
    print("-" * 50)
    for i, row in comparison_df.head(10).iterrows():
        # 计算在V3.8中的排名
        v38_rank = (comparison_df['v38_quality'] >= row['v38_quality']).sum()
        level4_rank = i + 1
        rank_change = v38_rank - level4_rank

        change_symbol = "↑" if rank_change > 0 else "↓" if rank_change < 0 else "→"
        print(f"{row['code']}\t{row['v38_quality']:.4f}\t\t{row['level4_quality']:.4f}\t\t{change_symbol}{abs(rank_change)}")

    # 8. 分布分析
    print("\n8. 质量评分分布分析:")
    print("=" * 50)

    # V3.8分布
    v38_high = np.sum(v38_quality_scores >= 0.7)
    v38_medium = np.sum((v38_quality_scores >= 0.3) & (v38_quality_scores < 0.7))
    v38_low = np.sum(v38_quality_scores < 0.3)

    # Level 4分布
    l4_high = np.sum(level4_quality_scores >= 0.7)
    l4_medium = np.sum((level4_quality_scores >= 0.3) & (level4_quality_scores < 0.7))
    l4_low = np.sum(level4_quality_scores < 0.3)

    print("质量等级分布对比:")
    print(f"高质量 (>=0.7): V3.8={v38_high}只 ({v38_high/len(v38_quality_scores)*100:.1f}%) vs Level 4={l4_high}只 ({l4_high/len(level4_quality_scores)*100:.1f}%)")
    print(f"中等质量 (0.3-0.7): V3.8={v38_medium}只 ({v38_medium/len(v38_quality_scores)*100:.1f}%) vs Level 4={l4_medium}只 ({l4_medium/len(level4_quality_scores)*100:.1f}%)")
    print(f"低质量 (<0.3): V3.8={v38_low}只 ({v38_low/len(v38_quality_scores)*100:.1f}%) vs Level 4={l4_low}只 ({l4_low/len(level4_quality_scores)*100:.1f}%)")

    # 9. 保存对比结果
    print("\n9. 保存对比分析结果...")

    comparison_results = {
        'comparison_info': {
            'date': test_date,
            'stock_count': len(test_stocks),
            'v38_version': v38_system.version,
            'level4_version': f"{level4_system.version} + Level 4"
        },
        'v38_stats': v38_stats,
        'level4_stats': level4_stats,
        'improvements': improvements,
        'distribution_comparison': {
            'v38_distribution': {'high': int(v38_high), 'medium': int(v38_medium), 'low': int(v38_low)},
            'level4_distribution': {'high': int(l4_high), 'medium': int(l4_medium), 'low': int(l4_low)}
        },
        'stock_comparison': comparison_df.to_dict('records')
    }

    # 保存到文件
    output_file = f"reports/v38_vs_level4_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)

    print(f"✅ 对比结果已保存: {output_file}")

    # 10. 总结评估
    print("\n🎯 性能对比总结:")
    print("=" * 50)

    if improvements['differentiation_achieved'] and not improvements['v38_differentiation_achieved']:
        print("🏆 Level 4集成系统 - 决定性胜利!")
        print("   ✅ 成功解决V3.8质量评分聚集问题")
        print(f"   ✅ 差异化能力提升{improvements['std_improvement']:.1f}倍")
        print("   ✅ 实现端到端机器学习质量评估")
        final_verdict = "Level 4 Complete Success"
    elif improvements['std_improvement'] >= 2.0:
        print("🥇 Level 4集成系统 - 显著优势!")
        print(f"   ✅ 差异化能力大幅提升{improvements['std_improvement']:.1f}倍")
        print("   ✅ 质量评分更加精准合理")
        final_verdict = "Level 4 Major Improvement"
    elif improvements['std_improvement'] >= 1.5:
        print("🥈 Level 4集成系统 - 明显改进!")
        print(f"   ✅ 差异化能力提升{improvements['std_improvement']:.1f}倍")
        final_verdict = "Level 4 Significant Improvement"
    else:
        print("🤔 Level 4集成系统 - 需要优化")
        print("   ⚠️ 改进效果有限，建议进一步调优")
        final_verdict = "Level 4 Needs Optimization"

    comparison_results['final_verdict'] = final_verdict

    return comparison_results

if __name__ == "__main__":
    comparison_results = compare_v38_vs_level4()