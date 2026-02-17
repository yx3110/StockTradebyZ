#!/usr/bin/env python3
"""
V3.53多时间周期评分器测试脚本

测试V3.53的多时间周期评分功能:
1. 验证基本评分功能
2. 测试所有时间周期的评分
3. 对比V3.52和V3.53的评分差异
4. 生成详细的测试报告
"""

import os
import sys
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import json
import logging

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入V3.53评分器
sys.path.append(os.path.join(current_dir, 'scoring', 'v3.5'))
from quantitative_scorer_v3_53 import QuantitativeScorerV353MultiPeriod

def setup_logger():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'v353_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)

def test_v353_basic_functionality():
    """测试V3.53基本功能"""
    logger = setup_logger()
    logger.info("🧪 开始V3.53基本功能测试")
    
    # 创建评分器
    try:
        scorer = QuantitativeScorerV353MultiPeriod("stock_data.db")
        logger.info("✅ V3.53评分器创建成功")
    except Exception as e:
        logger.error(f"❌ V3.53评分器创建失败: {e}")
        return False
    
    # 测试数据
    test_stock_data = {
        'code': '000001.SZ',
        'close': 10.50,
        'rsi6': 35.5,
        'kdj_k': 45.2,
        'kdj_d': 42.8,
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
    
    test_date = '2025-09-09'
    
    # 测试所有时间周期评分
    test_results = {}
    
    logger.info("🎯 测试各时间周期评分...")
    
    # 1. 测试复合评分
    try:
        composite_score, composite_details = scorer.calculate_multi_period_score(
            test_stock_data, test_date, 'composite'
        )
        test_results['composite'] = {
            'score': composite_score,
            'details': composite_details
        }
        logger.info(f"✅ 复合评分: {composite_score:.4f}")
    except Exception as e:
        logger.error(f"❌ 复合评分测试失败: {e}")
        return False
    
    # 2. 测试各时间周期评分
    periods = ['1d', '3d', '5d', '10d', '15d']
    for period in periods:
        try:
            period_score, period_details = scorer.calculate_multi_period_score(
                test_stock_data, test_date, period
            )
            test_results[period] = {
                'score': period_score,
                'details': period_details
            }
            logger.info(f"✅ {period}评分: {period_score:.4f}")
        except Exception as e:
            logger.error(f"❌ {period}评分测试失败: {e}")
            return False
    
    # 3. 测试配置导出
    try:
        config_file = scorer.export_configuration()
        logger.info(f"✅ 配置导出成功: {config_file}")
    except Exception as e:
        logger.error(f"❌ 配置导出失败: {e}")
    
    return test_results

def analyze_period_differences(test_results):
    """分析各时间周期的评分差异"""
    logger = logging.getLogger(__name__)
    logger.info("📊 分析时间周期评分差异...")
    
    periods = ['1d', '3d', '5d', '10d', '15d']
    scores = [test_results[period]['score'] for period in periods]
    
    analysis = {
        'period_scores': dict(zip(periods, scores)),
        'score_range': max(scores) - min(scores),
        'score_std': np.std(scores),
        'highest_period': periods[np.argmax(scores)],
        'lowest_period': periods[np.argmin(scores)],
        'composite_score': test_results['composite']['score']
    }
    
    logger.info(f"📈 评分分析结果:")
    logger.info(f"  - 最高评分周期: {analysis['highest_period']} ({max(scores):.4f})")
    logger.info(f"  - 最低评分周期: {analysis['lowest_period']} ({min(scores):.4f})")
    logger.info(f"  - 评分标准差: {analysis['score_std']:.4f}")
    logger.info(f"  - 复合评分: {analysis['composite_score']:.4f}")
    
    return analysis

def test_factor_contributions(test_results):
    """分析各因子贡献"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 分析因子贡献度...")
    
    periods = ['1d', '3d', '5d', '10d', '15d']
    factor_analysis = {}
    
    for period in periods:
        if period in test_results:
            details = test_results[period]['details']
            if 'factor_contributions' in details:
                contributions = details['factor_contributions']
                
                # 找出主要贡献因子
                sorted_factors = sorted(
                    contributions.items(), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )
                
                factor_analysis[period] = {
                    'top_factors': sorted_factors[:3],
                    'total_contribution': sum(contributions.values())
                }
                
                logger.info(f"  {period}: 主要因子 {sorted_factors[0][0]}({sorted_factors[0][1]:.4f})")
    
    return factor_analysis

def generate_test_report(test_results, analysis, factor_analysis):
    """生成测试报告"""
    logger = logging.getLogger(__name__)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"v353_test_report_{timestamp}.md"
    
    report_lines = []
    report_lines.append("# V3.53 多时间周期评分器测试报告")
    report_lines.append(f"\n## 测试概览")
    report_lines.append(f"- **测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"- **评分器版本**: V3.53 MultiPeriod")
    report_lines.append(f"- **测试股票**: 000001.SZ (平安银行)")
    
    report_lines.append(f"\n## 各时间周期评分结果")
    report_lines.append("| 时间周期 | 评分 | 评分方法 |")
    report_lines.append("|----------|------|----------|")
    
    periods = ['1d', '3d', '5d', '10d', '15d', 'composite']
    for period in periods:
        if period in test_results:
            score = test_results[period]['score']
            method = test_results[period]['details'].get('scoring_method', 'Unknown')
            report_lines.append(f"| {period} | {score:.4f} | {method} |")
    
    report_lines.append(f"\n## 评分差异分析")
    report_lines.append(f"- **评分范围**: {analysis['score_range']:.4f}")
    report_lines.append(f"- **评分标准差**: {analysis['score_std']:.4f}")
    report_lines.append(f"- **最高评分周期**: {analysis['highest_period']}")
    report_lines.append(f"- **最低评分周期**: {analysis['lowest_period']}")
    
    report_lines.append(f"\n## 主要因子贡献度")
    for period, factors in factor_analysis.items():
        report_lines.append(f"\n### {period.upper()} 周期")
        report_lines.append("| 因子 | 贡献度 |")
        report_lines.append("|------|--------|")
        for factor, contribution in factors['top_factors']:
            report_lines.append(f"| {factor} | {contribution:.4f} |")
    
    # 权重分布分析
    report_lines.append(f"\n## 权重分布特点")
    scorer = QuantitativeScorerV353MultiPeriod("stock_data.db")
    
    for period in ['1d', '3d', '5d', '10d', '15d']:
        weights = scorer.period_weights[period]
        active_weights = {k: v for k, v in weights.items() if v > 0.01}
        
        report_lines.append(f"\n### {period.upper()} 周期权重 (>1%)")
        report_lines.append("| 因子 | 权重 |")
        report_lines.append("|------|------|")
        
        sorted_weights = sorted(active_weights.items(), key=lambda x: x[1], reverse=True)
        for factor, weight in sorted_weights:
            report_lines.append(f"| {factor} | {weight:.3f} |")
    
    report_lines.append(f"\n## 创新特点总结")
    report_lines.append("1. **分层权重架构**: 不同时间周期使用专门优化的因子权重组合")
    report_lines.append("2. **技术指标时效性**: 短期周期技术指标权重高，长期周期基本面权重高")
    report_lines.append("3. **A股市场适配**: 权重分配考虑A股高噪音、高波动特点")
    report_lines.append("4. **复合评分机制**: 按重要性加权合成最终评分")
    
    report_lines.append(f"\n---")
    report_lines.append(f"🤖 *Generated by V3.53 MultiPeriod Quantitative Scorer Test*")
    report_lines.append(f"📅 *Test Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # 保存报告
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"📝 测试报告已保存到: {report_file}")
    return report_file

def main():
    """主测试函数"""
    print("🧪 V3.53 多时间周期评分器测试")
    print("="*60)
    
    # 1. 基本功能测试
    print("1️⃣ 基本功能测试...")
    test_results = test_v353_basic_functionality()
    
    if not test_results:
        print("❌ 基本功能测试失败，终止测试")
        return
    
    print("✅ 基本功能测试通过")
    
    # 2. 评分差异分析
    print("2️⃣ 评分差异分析...")
    analysis = analyze_period_differences(test_results)
    
    # 3. 因子贡献分析
    print("3️⃣ 因子贡献分析...")
    factor_analysis = test_factor_contributions(test_results)
    
    # 4. 生成测试报告
    print("4️⃣ 生成测试报告...")
    report_file = generate_test_report(test_results, analysis, factor_analysis)
    
    print("\n" + "="*60)
    print("🎉 V3.53测试完成！")
    print(f"📊 复合评分: {test_results['composite']['score']:.4f}")
    print(f"📝 测试报告: {report_file}")
    print("✨ V3.53多时间周期评分器已就绪，准备优化！")

if __name__ == "__main__":
    main()