#!/usr/bin/env python3
"""
v3.41反向评分系统大规模测试脚本
使用昨日选股结果进行批量对比验证
"""

import sys
import os
import importlib.util
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob
import re

# 动态导入评分系统
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

v34_path = os.path.join(current_dir, 'scoring/v3.4/quantitative_scorer_v3_4.py')
spec = importlib.util.spec_from_file_location("quantitative_scorer_v3_4", v34_path)
v34_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v34_module)
QuantitativeScorerV34 = v34_module.QuantitativeScorerV34

v341_path = os.path.join(current_dir, 'scoring/v3.4/quantitative_scorer_v3_41.py')
spec = importlib.util.spec_from_file_location("quantitative_scorer_v3_41", v341_path)
v341_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v341_module)
QuantitativeScorerV341 = v341_module.QuantitativeScorerV341

from data_adapter.database_manager import DatabaseManager

def get_yesterday_stocks_from_reports():
    """从最近的选股报告中获取股票列表"""
    try:
        # 查找最近的选股报告
        report_dirs = [
            "reports/daily_selection_v3/*.md",
            "reports/daily_selection/*.md",
            "reports/daily_selection_v2/*.md"
        ]
        
        latest_report = None
        for pattern in report_dirs:
            reports = glob.glob(pattern)
            if reports:
                latest_report = sorted(reports)[-1]
                break
                
        if not latest_report:
            print("⚠️ 未找到历史选股报告，使用默认股票列表")
            return get_sample_stocks(), "default"
            
        print(f"📄 使用报告: {latest_report}")
        
        # 从报告中提取股票代码
        with open(latest_report, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 使用正则表达式提取股票代码（6位数字）
        stock_codes = re.findall(r'\b\d{6}\b', content)
        stock_codes = list(set(stock_codes))  # 去重
        
        # 过滤掉无效的股票代码（如日期等）
        valid_codes = []
        for code in stock_codes:
            if (code.startswith(('00', '30', '60', '68')) and 
                not code.startswith('202') and  # 排除年份2025等
                len(code) == 6):
                valid_codes.append(code)
                
        if len(valid_codes) < 10:
            print(f"⚠️ 提取的股票数量较少({len(valid_codes)})，使用默认列表")
            return get_sample_stocks(), "default"
            
        print(f"📊 从报告提取了 {len(valid_codes)} 只股票")
        return valid_codes[:100], latest_report  # 限制100只避免过长
        
    except Exception as e:
        print(f"⚠️ 读取报告失败: {e}")
        return get_sample_stocks(), "default"

def get_sample_stocks():
    """获取样本股票列表"""
    return [
        "000001", "000002", "000858", "002215", "002594", "300001", "300059", 
        "600000", "600036", "600519", "600893", "688001", "688009", "688036"
    ]

def batch_compare_scoring():
    """批量对比v3.4和v3.41评分系统"""
    print("🚀 开始大规模v3.41反向评分系统测试")
    print("=" * 70)
    
    # 获取测试股票
    test_stocks, source = get_yesterday_stocks_from_reports()
    test_date = "2025-09-02"
    
    print(f"📅 测试日期: {test_date}")
    print(f"📊 股票来源: {source}")
    print(f"🔢 测试数量: {len(test_stocks)} 只股票")
    print(f"📋 前10只股票: {test_stocks[:10]}")
    print("\n" + "=" * 70)
    
    # 初始化评分器
    v34_scorer = QuantitativeScorerV34()
    v341_scorer = QuantitativeScorerV341()
    
    results = []
    successful_count = 0
    error_count = 0
    risk_signals_count = 0
    
    print("🔄 开始批量评分...")
    for i, stock in enumerate(test_stocks):
        if i % 20 == 0:
            print(f"进度: {i+1}/{len(test_stocks)} ({(i+1)/len(test_stocks)*100:.1f}%)")
            
        try:
            # v3.4评分
            v34_result = v34_scorer.calculate_quantitative_score(stock, test_date)
            if "error" in v34_result:
                error_count += 1
                continue
                
            v34_score = v34_result.get("quantitative_score", 0)
            
            # v3.41评分
            v341_result = v341_scorer.calculate_quantitative_score(stock, test_date)
            if "error" in v341_result:
                error_count += 1
                continue
                
            v341_score = v341_result.get("quantitative_score", 0)
            reversed_score = v341_result.get("reversed_score", 0)
            
            # 检查风险信号
            risk_signals = v341_result.get("risk_signals", {})
            has_risk = risk_signals.get("risk_score", 0) > 0
            if has_risk:
                risk_signals_count += 1
            
            results.append({
                "stock_code": stock,
                "v34_score": v34_score,
                "v341_score": v341_score,
                "reversed_score": reversed_score,
                "difference": v341_score - v34_score,
                "risk_score": risk_signals.get("risk_score", 0),
                "has_risk": has_risk,
                "is_st": risk_signals.get("is_st", False),
                "is_limit_up": risk_signals.get("is_limit_up", False),
                "is_limit_down": risk_signals.get("is_limit_down", False)
            })
            
            successful_count += 1
            
        except Exception as e:
            print(f"❌ {stock} 评分失败: {e}")
            error_count += 1
            continue
    
    if not results:
        print("❌ 没有成功的评分结果")
        return
    
    # 分析结果
    df = pd.DataFrame(results)
    
    print(f"\n" + "=" * 70)
    print("📊 大规模测试结果分析")
    print("=" * 70)
    
    print(f"✅ 成功评分: {successful_count} 只")
    print(f"❌ 失败评分: {error_count} 只")
    print(f"⚠️  风险股票: {risk_signals_count} 只 ({risk_signals_count/successful_count*100:.1f}%)")
    
    # 基础统计
    print(f"\n📈 评分统计:")
    print(f"v3.4评分 - 均值: {df['v34_score'].mean():.1f}, 标准差: {df['v34_score'].std():.1f}")
    print(f"         范围: {df['v34_score'].min():.1f} - {df['v34_score'].max():.1f}")
    print(f"v3.41评分 - 均值: {df['v341_score'].mean():.1f}, 标准差: {df['v341_score'].std():.1f}")
    print(f"          范围: {df['v341_score'].min():.1f} - {df['v341_score'].max():.1f}")
    
    # 相关性分析
    correlation = df['v34_score'].corr(df['v341_score'])
    print(f"\n🔄 相关性分析:")
    print(f"v3.4 与 v3.41 相关系数: {correlation:.6f}")
    if correlation < -0.8:
        print("✅ 反转效果极强（相关性 < -0.8）")
    elif correlation < -0.5:
        print("✅ 反转效果明显（相关性 < -0.5）")
    elif correlation < 0:
        print("⚠️ 反转效果一般（负相关但不强）")
    else:
        print("❌ 反转失败（正相关）")
    
    # 评分分布分析
    print(f"\n📊 评分分布对比:")
    v34_bins = [0, 30, 50, 70, 90, 100]
    v34_dist = pd.cut(df['v34_score'], bins=v34_bins, labels=['<30', '30-50', '50-70', '70-90', '>90'])
    v341_dist = pd.cut(df['v341_score'], bins=v34_bins, labels=['<30', '30-50', '50-70', '70-90', '>90'])
    
    print("v3.4分布:")
    print(v34_dist.value_counts().sort_index())
    print("v3.41分布:")
    print(v341_dist.value_counts().sort_index())
    
    # 高风险股票分析
    if risk_signals_count > 0:
        risk_stocks = df[df['has_risk']]
        print(f"\n⚠️  高风险股票分析 ({len(risk_stocks)} 只):")
        st_count = risk_stocks['is_st'].sum()
        limit_up_count = risk_stocks['is_limit_up'].sum()
        limit_down_count = risk_stocks['is_limit_down'].sum()
        
        print(f"ST股票: {st_count} 只")
        print(f"涨停股票: {limit_up_count} 只")
        print(f"跌停股票: {limit_down_count} 只")
        print(f"平均风险评分: {risk_stocks['risk_score'].mean():.2f}")
        print(f"风险股票v3.41评分均值: {risk_stocks['v341_score'].mean():.1f}")
    
    # 保存详细结果
    output_file = f"reports/v341_large_scale_test_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 详细结果已保存: {output_file}")
    
    # 显示样例
    print(f"\n📋 样例结果 (前10只股票):")
    display_df = df[['stock_code', 'v34_score', 'v341_score', 'difference', 'risk_score']].head(10)
    print(display_df.to_string(index=False, float_format='%.1f'))
    
    return df

def analyze_score_effectiveness():
    """分析评分系统的有效性"""
    print(f"\n🎯 评分系统有效性分析")
    print("-" * 50)
    
    print("基于相关性报告的理论预期:")
    print("- v3.4高分股票(90+分)实际表现差(夏普比率-0.026)")  
    print("- v3.4低分股票(<60分)实际表现好(夏普比率1.223)")
    print("- v3.41反转后：原低分→高分，原高分→低分")
    print("- 预期v3.41与未来收益呈现正相关")
    
    print(f"\n💡 验证建议:")
    print("1. 使用历史3个月数据回测验证")
    print("2. 计算v3.41评分与实际收益的相关性") 
    print("3. 对比v3.4和v3.41的夏普比率")
    print("4. 集成到daily stock selector进行实盘验证")

if __name__ == "__main__":
    # 执行大规模测试
    results_df = batch_compare_scoring()
    
    if results_df is not None:
        analyze_score_effectiveness()
        
        print(f"\n🎉 大规模测试完成!")
        print(f"🚀 v3.41反向评分系统已准备就绪")
        print(f"💡 下一步: 集成到 tomorrow_stock_selector.py")