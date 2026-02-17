#!/usr/bin/env python3
"""
测试v3.5量化评分系统
验证知行指标集成和权重配置
"""

import sys
import os
from datetime import datetime

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from quantitative_scorer_v3_5 import QuantitativeScorerV35

def test_v35_scorer():
    """测试v3.5评分系统"""
    print("=" * 60)
    print("测试v3.5量化评分系统（知行指标版本）")
    print("=" * 60)
    
    try:
        # 初始化评分器
        scorer = QuantitativeScorerV35()
        print(f"✅ v3.5评分器初始化成功，版本: {scorer.version}")
        
        # 检查权重配置
        weights = scorer.config["weights"]
        technical_weights = weights["technical"]
        
        # 计算知行指标权重总和
        zhixing_total = (technical_weights["zhixing_trend"] + 
                        technical_weights["zhixing_multiavg"])
        
        print(f"\n📊 权重配置验证:")
        print(f"  知行短期趋势线权重: {technical_weights['zhixing_trend']:.1%}")
        print(f"  知行多空线权重: {technical_weights['zhixing_multiavg']:.1%}")
        print(f"  知行指标总权重: {zhixing_total:.1%}")
        
        # 验证总权重是否为100%
        total_weight = (
            sum(technical_weights.values()) +
            sum(weights["fundamental"].values()) +
            sum(weights["performance"].values()) +
            sum(weights["market_regime"].values())
        )
        print(f"  系统总权重: {total_weight:.1%}")
        
        if abs(total_weight - 1.0) < 0.001:
            print("  ✅ 权重配置正确")
        else:
            print("  ❌ 权重配置错误")
        
        if abs(zhixing_total - 0.20) < 0.001:
            print("  ✅ 知行指标权重为20%，符合要求")
        else:
            print(f"  ❌ 知行指标权重为{zhixing_total:.1%}，不符合20%要求")
        
        # 测试计算功能（使用测试股票）
        test_date = "2025-09-04"
        test_stocks = ["000001", "000002", "002215"]  # 测试几只股票
        
        print(f"\n🧪 功能测试 (日期: {test_date}):")
        
        success_count = 0
        for stock_code in test_stocks:
            try:
                result = scorer.calculate_quantitative_score(stock_code, test_date)
                
                if "error" in result:
                    print(f"  {stock_code}: ❌ {result.get('error', '未知错误')}")
                else:
                    print(f"  {stock_code}: ✅ 评分={result['quantitative_score']:.1f}")
                    
                    # 显示知行指标信息
                    if result.get('zhixing_signals'):
                        signals = result['zhixing_signals']
                        signal_strength = signals.get('signal_strength', '无信号')
                        print(f"           知行信号: {signal_strength}")
                        
                        if signals.get('golden_cross'):
                            print(f"           🔥 检测到金叉信号")
                        elif signals.get('death_cross'):
                            print(f"           ❄️ 检测到死叉信号")
                        elif signals.get('price_broken_multiavg'):
                            print(f"           📉 检测到价格跌破多空线")
                    
                    success_count += 1
                    
            except Exception as e:
                print(f"  {stock_code}: ❌ 计算异常: {e}")
        
        print(f"\n📈 测试结果: {success_count}/{len(test_stocks)} 只股票计算成功")
        
        if success_count > 0:
            print("✅ v3.5量化评分系统基本功能正常")
        else:
            print("❌ v3.5量化评分系统可能存在问题")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)

if __name__ == "__main__":
    test_v35_scorer()