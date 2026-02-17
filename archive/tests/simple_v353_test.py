#!/usr/bin/env python3
"""
简化的V3.53测试脚本
直接测试V3.53评分器的核心功能
"""

import sys
import os
from datetime import datetime

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'scoring', 'v3.5'))

from quantitative_scorer_v3_53 import QuantitativeScorerV353MultiPeriod

def main():
    print("🧪 V3.53 简化测试")
    print("="*50)
    
    # 初始化V3.53评分器
    try:
        scorer = QuantitativeScorerV353MultiPeriod("./data_adapter/stock_data.db")
        print("✅ V3.53评分器初始化成功")
    except Exception as e:
        print(f"❌ V3.53评分器初始化失败: {e}")
        return
    
    # 构建测试股票数据
    test_stock_data = {
        'close': 10.50,
        'rsi6': 45.0,
        'kdj_k': 55.0,
        'kdj_d': 48.0,
        'bbi': 10.2,
        'ema12': 10.8,
        'ema26': 10.5,
        'ma5': 10.6,
        'ma10': 10.4,
        'ma20': 10.2,
        'pe_ttm': 18.5,
        'pb': 1.2,
        'market_cap': 500000000000,  # 5000亿
        'price_change_pct': 2.5,
        'volume_ratio_5d': 1.8,
        'volume_ratio_20d': 1.5,
        'volatility_20d': 0.025
    }
    
    test_date = datetime.now().strftime('%Y-%m-%d')
    
    # 测试各时间周期评分
    periods = ['1d', '3d', '5d', '10d', '15d', 'composite']
    
    print("\n📊 各时间周期评分测试:")
    print("-" * 50)
    
    results = {}
    
    for period in periods:
        try:
            score, details = scorer.calculate_multi_period_score(test_stock_data, test_date, period)
            results[period] = {'score': score, 'details': details}
            print(f"✅ {period:10s}: {score:.4f} ({details.get('scoring_method', 'Unknown')})")
        except Exception as e:
            print(f"❌ {period:10s}: 失败 - {e}")
    
    # 显示复合评分的详细信息
    if 'composite' in results:
        print("\n🔍 复合评分详细信息:")
        print("-" * 50)
        composite_details = results['composite']['details']
        
        if 'period_scores' in composite_details:
            period_scores = composite_details['period_scores']
            print("各时间周期评分:")
            for period, score in period_scores.items():
                print(f"  {period}: {score:.4f}")
        
        if 'period_weights' in composite_details:
            period_weights = composite_details['period_weights']
            print("\n时间周期重要性权重:")
            for period, weight in period_weights.items():
                print(f"  {period}: {weight:.1%}")
    
    # 测试配置导出
    print("\n💾 配置导出测试:")
    print("-" * 50)
    
    try:
        config_file = scorer.export_configuration()
        print(f"✅ 配置已导出: {config_file}")
        
        # 显示优化指标
        metrics = scorer.get_optimization_metrics()
        print("\n📈 系统指标:")
        for key, value in metrics.items():
            if key != 'expected_improvements':
                print(f"  {key}: {value}")
        
        if 'expected_improvements' in metrics:
            print("\n🎯 预期改进目标:")
            for target, value in metrics['expected_improvements'].items():
                print(f"  {target}: {value}")
                
    except Exception as e:
        print(f"❌ 配置导出失败: {e}")
    
    print("\n" + "="*50)
    print("🎉 V3.53简化测试完成!")
    
    # 成功率统计
    successful_tests = len([r for r in results.values() if r['score'] > 0])
    total_tests = len(periods)
    success_rate = successful_tests / total_tests * 100
    
    print(f"📊 测试成功率: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("✅ V3.53评分器工作正常！")
    elif success_rate >= 50:
        print("⚠️ V3.53评分器基本正常，有少量问题。")
    else:
        print("❌ V3.53评分器存在问题，需要修复。")

if __name__ == "__main__":
    main()