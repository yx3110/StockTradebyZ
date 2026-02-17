#!/usr/bin/env python3
"""
v3.41反向评分系统快速测试脚本
"""

import sys
import os
import importlib.util

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 动态导入v3.4评分系统
v34_path = os.path.join(current_dir, 'scoring/v3.4/quantitative_scorer_v3_4.py')
spec = importlib.util.spec_from_file_location("quantitative_scorer_v3_4", v34_path)
v34_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v34_module)
QuantitativeScorerV34 = v34_module.QuantitativeScorerV34

# 动态导入v3.41评分系统
v341_path = os.path.join(current_dir, 'scoring/v3.4/quantitative_scorer_v3_41.py')
spec = importlib.util.spec_from_file_location("quantitative_scorer_v3_41", v341_path)
v341_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v341_module)
QuantitativeScorerV341 = v341_module.QuantitativeScorerV341
import pandas as pd
from datetime import datetime

def test_reverse_scoring():
    """测试反向评分系统"""
    print("🔄 开始测试v3.41反向评分系统")
    print("=" * 60)
    
    # 初始化评分器
    v34_scorer = QuantitativeScorerV34()
    v341_scorer = QuantitativeScorerV341()
    
    # 测试股票列表
    test_stocks = ["000001", "000002", "002215", "300001"]
    test_date = "2025-09-02"  # 使用最新数据日期
    
    print(f"📅 测试日期: {test_date}")
    print(f"📊 测试股票: {test_stocks}")
    print("\n" + "=" * 60)
    
    results = []
    
    for stock in test_stocks:
        print(f"\n🔍 测试股票: {stock}")
        print("-" * 40)
        
        try:
            # v3.4原始评分
            v34_result = v34_scorer.calculate_quantitative_score(stock, test_date)
            v34_score = v34_result.get("quantitative_score", 0)
            
            # v3.41反向评分
            v341_result = v341_scorer.calculate_quantitative_score(stock, test_date)
            v341_score = v341_result.get("quantitative_score", 0)
            reversed_score = v341_result.get("reversed_score", 0)
            
            print(f"v3.4 原始评分:  {v34_score:.1f}")
            print(f"v3.41反转评分:  {reversed_score:.1f}")
            print(f"v3.41最终评分:  {v341_score:.1f}")
            
            # 检查风险信号
            if "risk_signals" in v341_result:
                risk = v341_result["risk_signals"]
                risk_count = sum([risk.get("is_st", False), 
                                risk.get("is_limit_up", False),
                                risk.get("is_limit_down", False),
                                risk.get("is_new_stock", False),
                                risk.get("is_low_liquidity", False)])
                if risk_count > 0:
                    print(f"⚠️  风险信号数量: {risk_count}")
                    print(f"⚠️  风险评分: {risk.get('risk_score', 0):.2f}")
            
            # 验证反转逻辑
            expected_reverse = 100 - v34_score
            if abs(reversed_score - expected_reverse) < 0.1:
                print("✅ 反转逻辑正确")
            else:
                print(f"❌ 反转逻辑错误: 期望{expected_reverse:.1f}, 实际{reversed_score:.1f}")
            
            results.append({
                "stock": stock,
                "v34_score": v34_score,
                "v341_score": v341_score,
                "reversed_score": reversed_score,
                "difference": v341_score - v34_score
            })
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            continue
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False, float_format='%.1f'))
        
        print(f"\n📈 平均差异: {df['difference'].mean():.1f}")
        print(f"📉 评分范围变化: {df['v34_score'].min():.1f}-{df['v34_score'].max():.1f} → {df['v341_score'].min():.1f}-{df['v341_score'].max():.1f}")
        
        # 检查反转效果
        correlation_before = df['v34_score'].corr(df['v341_score'])
        print(f"🔄 v3.4与v3.41相关性: {correlation_before:.3f}")
        
        if correlation_before < -0.5:
            print("✅ 反转效果明显，相关性为负")
        else:
            print("⚠️ 反转效果不够明显")
    
    # 显示反向逻辑说明
    print("\n" + "=" * 60)
    print("📝 反向逻辑说明")
    print("=" * 60)
    print(v341_scorer.explain_reverse_logic())
    
    return results

def compare_score_distributions():
    """比较评分分布"""
    print("\n🎯 比较评分分布（使用yesterday选股结果）")
    
    try:
        # 尝试加载最近的选股结果进行对比
        import glob
        latest_reports = glob.glob("reports/daily_selection_v3/*.md")
        if latest_reports:
            latest_report = sorted(latest_reports)[-1]
            print(f"📄 使用报告: {latest_report}")
            
            # 这里可以扩展为读取报告中的股票列表进行批量对比
            print("💡 建议: 在实际使用中可以批量测试昨日选股结果")
        else:
            print("📄 未找到历史选股报告，跳过分布对比")
            
    except Exception as e:
        print(f"⚠️ 分布对比失败: {e}")

if __name__ == "__main__":
    # 运行测试
    results = test_reverse_scoring()
    compare_score_distributions()
    
    print("\n🎉 v3.41反向评分系统测试完成!")
    print("💡 下一步: 使用tomorrow_stock_selector.py进行完整测试")