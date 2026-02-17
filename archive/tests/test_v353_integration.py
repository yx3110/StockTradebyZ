#!/usr/bin/env python3
"""
测试V3.53集成到每日选股脚本的功能

验证:
1. V3.53评分器是否正确初始化
2. 能否正常进行股票评分
3. 报告格式是否正确显示多时间周期评分
4. 与V3.52版本的对比测试
"""

import sys
import os
import pandas as pd
from datetime import datetime
import logging

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from tomorrow_stock_selector import TomorrowStockSelector

def setup_logger():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_v353_initialization():
    """测试V3.53初始化"""
    logger = logging.getLogger(__name__)
    logger.info("🧪 测试V3.53初始化...")
    
    try:
        # 创建V3.53选股器
        selector = TomorrowStockSelector(scoring_version="v3.53", stocks_only=True)
        
        # 检查是否有v3.53评分引擎
        if hasattr(selector, 'scoring_engine_v353_multiperiod'):
            logger.info("✅ V3.53评分引擎初始化成功")
            return True
        else:
            logger.error("❌ V3.53评分引擎未初始化")
            return False
            
    except Exception as e:
        logger.error(f"❌ V3.53初始化失败: {e}")
        return False

def test_v353_scoring():
    """测试V3.53评分功能"""
    logger = logging.getLogger(__name__)
    logger.info("🎯 测试V3.53评分功能...")
    
    try:
        selector = TomorrowStockSelector(scoring_version="v3.53", stocks_only=True)
        
        # 测试几只股票的评分
        test_stocks = ['000001.SZ', '000002.SZ', '600000.SH']
        trade_date = datetime.now().strftime('%Y-%m-%d')
        
        scoring_results = []
        for stock_code in test_stocks:
            try:
                # 构建股票信息结构
                stock_info = {'code': stock_code}
                result = selector.calculate_comprehensive_score(stock_info, trade_date)
                if result and len(result) >= 2:
                    score, details = result
                    scoring_results.append({
                        'code': stock_code,
                        'score': score,
                        'method': details.get('scoring_method', 'Unknown'),
                        'period_scores': details.get('period_scores', {}),
                        'recommendation': details.get('recommendation', 'N/A')
                    })
                    logger.info(f"✅ {stock_code}: {score:.1f}分")
                else:
                    logger.warning(f"⚠️ {stock_code}: 评分失败")
                    
            except Exception as e:
                logger.error(f"❌ {stock_code}评分异常: {e}")
        
        if scoring_results:
            logger.info(f"✅ 成功评分 {len(scoring_results)} 只股票")
            return scoring_results
        else:
            logger.error("❌ 没有成功评分任何股票")
            return None
            
    except Exception as e:
        logger.error(f"❌ V3.53评分测试失败: {e}")
        return None

def test_v353_report_generation():
    """测试V3.53报告生成"""
    logger = logging.getLogger(__name__)
    logger.info("📝 测试V3.53报告生成...")
    
    try:
        selector = TomorrowStockSelector(scoring_version="v3.53", stocks_only=True)
        
        # 加载少量数据进行测试
        data = selector.load_data(limit=50)  # 只加载50只股票
        
        if not data:
            logger.error("❌ 无法加载股票数据")
            return False
        
        logger.info(f"✅ 加载了 {len(data)} 只股票数据")
        
        # 获取最新交易日
        latest_date = selector.get_latest_trading_date(data)
        logger.info(f"📅 最新交易日: {latest_date}")
        
        # 运行选股策略（简化测试）
        strategies_results = {}
        test_strategy = "少妇战法"
        
        # 模拟策略结果
        sample_stocks = list(data.keys())[:10]  # 取前10只股票
        strategies_results[test_strategy] = sample_stocks
        
        # 运行分析
        data_limited = {k: v for k, v in list(data.items())[:10]}  # 限制数据量
        analysis = selector.analyze_results(strategies_results, data_limited, latest_date)
        
        # 生成报告
        report = selector.generate_report(analysis, latest_date)
        
        # 检查报告是否包含多时间周期信息
        if "1日评分" in report and "3日评分" in report and "复合评分" in report:
            logger.info("✅ 报告包含多时间周期评分信息")
            
            # 保存测试报告
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = f"v353_integration_test_report_{timestamp}.md"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"📁 测试报告已保存: {report_file}")
            return True
        else:
            logger.warning("⚠️ 报告缺少多时间周期评分信息")
            return False
            
    except Exception as e:
        logger.error(f"❌ V3.53报告生成测试失败: {e}")
        return False

def compare_v352_vs_v353():
    """对比V3.52和V3.53的评分结果"""
    logger = logging.getLogger(__name__)
    logger.info("⚖️ 对比V3.52 vs V3.53评分结果...")
    
    try:
        # 创建两个版本的选股器
        selector_v352 = TomorrowStockSelector(scoring_version="v3.52", stocks_only=True)
        selector_v353 = TomorrowStockSelector(scoring_version="v3.53", stocks_only=True)
        
        # 测试股票
        test_stocks = ['000001.SZ', '000002.SZ', '600000.SH', '000858.SZ', '002415.SZ']
        trade_date = datetime.now().strftime('%Y-%m-%d')
        
        comparison_results = []
        
        for stock_code in test_stocks:
            try:
                # 构建股票信息
                stock_info = {'code': stock_code}
                
                # V3.52评分
                result_v352 = selector_v352.calculate_comprehensive_score(stock_info, trade_date)
                score_v352 = result_v352[0] if result_v352 and len(result_v352) > 0 else 0
                
                # V3.53评分
                result_v353 = selector_v353.calculate_comprehensive_score(stock_info, trade_date)
                score_v353 = result_v353[0] if result_v353 and len(result_v353) > 0 else 0
                
                # 记录对比结果
                comparison_results.append({
                    'code': stock_code,
                    'v352_score': score_v352,
                    'v353_score': score_v353,
                    'difference': score_v353 - score_v352,
                    'v353_periods': result_v353[1].get('period_scores', {}) if result_v353 and len(result_v353) > 1 else {}
                })
                
                logger.info(f"{stock_code}: V3.52={score_v352:.1f}, V3.53={score_v353:.1f}, 差异={score_v353-score_v352:+.1f}")
                
            except Exception as e:
                logger.error(f"❌ {stock_code}对比评分失败: {e}")
        
        if comparison_results:
            # 生成对比报告
            comparison_report = generate_comparison_report(comparison_results)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            comparison_file = f"v352_vs_v353_comparison_{timestamp}.md"
            
            with open(comparison_file, 'w', encoding='utf-8') as f:
                f.write(comparison_report)
            
            logger.info(f"📊 对比报告已保存: {comparison_file}")
            return comparison_results
        else:
            logger.error("❌ 没有成功的对比结果")
            return None
            
    except Exception as e:
        logger.error(f"❌ V3.52 vs V3.53对比测试失败: {e}")
        return None

def generate_comparison_report(comparison_results):
    """生成对比报告"""
    report_lines = []
    report_lines.append("# V3.52 vs V3.53 评分对比报告")
    report_lines.append(f"\n## 测试概览")
    report_lines.append(f"- **测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"- **测试股票**: {len(comparison_results)} 只")
    report_lines.append(f"- **对比版本**: V3.52 (全面优化) vs V3.53 (多时间周期IC优化)")
    
    report_lines.append(f"\n## 评分对比结果")
    report_lines.append("| 股票代码 | V3.52评分 | V3.53评分 | 评分差异 | V3.53优势 |")
    report_lines.append("|----------|----------|----------|----------|----------|")
    
    total_v352 = 0
    total_v353 = 0
    improvements = 0
    
    for result in comparison_results:
        code = result['code']
        v352_score = result['v352_score']
        v353_score = result['v353_score']
        difference = result['difference']
        
        total_v352 += v352_score
        total_v353 += v353_score
        
        if difference > 0:
            improvements += 1
            advantage = f"+{difference:.1f} ✅"
        elif difference < 0:
            advantage = f"{difference:.1f} ⚠️"
        else:
            advantage = "持平 ➖"
        
        report_lines.append(f"| {code} | {v352_score:.1f} | {v353_score:.1f} | {difference:+.1f} | {advantage} |")
    
    # 统计分析
    avg_v352 = total_v352 / len(comparison_results)
    avg_v353 = total_v353 / len(comparison_results)
    avg_improvement = avg_v353 - avg_v352
    improvement_rate = improvements / len(comparison_results) * 100
    
    report_lines.append(f"\n## 统计分析")
    report_lines.append(f"- **V3.52平均评分**: {avg_v352:.2f}")
    report_lines.append(f"- **V3.53平均评分**: {avg_v353:.2f}")
    report_lines.append(f"- **平均改善幅度**: {avg_improvement:+.2f}")
    report_lines.append(f"- **改善股票比例**: {improvement_rate:.1f}%")
    
    if avg_improvement > 0:
        overall_assessment = "🚀 V3.53显著优于V3.52"
    elif avg_improvement > -2:
        overall_assessment = "⚖️ V3.53与V3.52相当"
    else:
        overall_assessment = "⚠️ V3.53需要进一步优化"
    
    report_lines.append(f"- **整体评估**: {overall_assessment}")
    
    report_lines.append(f"\n## V3.53特色功能")
    report_lines.append("1. **多时间周期评分**: 1日、3日、5日、10日、15日分层权重")
    report_lines.append("2. **IC优化架构**: 基于6.5%的1日IC和5.7%的3日IC")
    report_lines.append("3. **自适应权重**: 不同周期使用最适合的因子组合")
    report_lines.append("4. **复合评分**: 按重要性加权融合多个时间周期")
    
    report_lines.append(f"\n---")
    report_lines.append(f"🤖 *Generated by V3.53 Integration Test*")
    report_lines.append(f"📅 *Test Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    return '\n'.join(report_lines)

def main():
    """主测试函数"""
    print("🧪 V3.53集成测试")
    print("="*60)
    
    logger = setup_logger()
    
    # 测试步骤
    tests = [
        ("V3.53初始化测试", test_v353_initialization),
        ("V3.53评分功能测试", test_v353_scoring),
        ("V3.53报告生成测试", test_v353_report_generation),
        ("V3.52 vs V3.53对比测试", compare_v352_vs_v353)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🔍 {test_name}")
        print("="*60)
        
        try:
            result = test_func()
            results[test_name] = result
            
            if result:
                print(f"✅ {test_name} - 通过")
            else:
                print(f"❌ {test_name} - 失败")
                
        except Exception as e:
            print(f"💥 {test_name} - 异常: {e}")
            results[test_name] = False
    
    # 总结
    print("\n" + "="*60)
    print("🎉 V3.53集成测试总结")
    print("="*60)
    
    passed_tests = sum(1 for result in results.values() if result)
    total_tests = len(tests)
    
    print(f"📊 测试通过率: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    if passed_tests == total_tests:
        print("\n🎯 V3.53集成完全成功！可以开始使用多时间周期评分系统。")
        print("\n使用方法:")
        print("  python3 tomorrow_stock_selector.py --scoring_version v3.53")
    elif passed_tests >= total_tests * 0.75:
        print("\n⚖️ V3.53集成基本成功，有少量问题需要修复。")
    else:
        print("\n⚠️ V3.53集成存在较多问题，需要检查代码。")

if __name__ == "__main__":
    main()