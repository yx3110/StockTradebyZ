#!/usr/bin/env python3
"""
使用模拟数据运行权重优化测试
验证权重优化器是否能提升V3.5效果
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import json

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

def generate_mock_stock_data(num_stocks: int = 100, num_days: int = 50) -> tuple:
    """
    生成模拟股票数据用于测试
    
    Args:
        num_stocks: 股票数量
        num_days: 历史天数
        
    Returns:
        (历史特征数据, 未来收益数据)
    """
    print(f"📊 生成模拟数据：{num_stocks}只股票，{num_days}天历史数据")
    
    np.random.seed(42)  # 固定随机种子确保结果可重现
    
    historical_features = {}
    future_returns = {}
    
    for i in range(num_stocks):
        stock_code = f"{i+1:06d}"
        
        # 生成技术指标特征（使用真实的数值范围）
        features = {
            # KDJ指标 (0-100)
            'kdj_k': np.random.uniform(10, 90),
            'kdj_d': np.random.uniform(10, 90), 
            'kdj_j': np.random.uniform(-20, 120),
            
            # RSI指标 (0-100)
            'rsi': np.random.uniform(20, 80),
            
            # 价格相关
            'close': np.random.uniform(5, 50),
            'bbi': np.random.uniform(5, 50),
            'zhixing_trend': np.random.uniform(5, 50),
            'zhixing_multiavg': np.random.uniform(5, 50),
            
            # 成交量指标
            'volume_surge': np.random.lognormal(0, 0.5),  # 对数正态分布，均值约1
            
            # 价格动量 (-50% to +50%)
            'price_momentum': np.random.uniform(-0.5, 0.5),
            
            # 波动率 (0-10%)
            'volatility': np.random.uniform(0.01, 0.10),
            
            # 基本面指标
            'pe_ttm': np.random.uniform(8, 80),
            'pb': np.random.uniform(0.5, 5.0),
        }
        
        # 计算"真实"评分（基于一些逻辑规则）
        # 这将作为我们优化的目标
        true_score = calculate_true_score(features)
        
        # 生成与真实评分相关的未来收益（添加噪音）
        base_return = (true_score - 50) / 1000  # 评分越高，基础收益越高
        
        returns = {}
        for days in [1, 3, 5, 10]:
            # 收益随时间增加而增加方差
            noise_scale = days * 0.005
            returns[f'return_{days}d'] = base_return * days + np.random.normal(0, noise_scale)
        
        historical_features[stock_code] = features
        future_returns[stock_code] = returns
    
    print(f"✅ 模拟数据生成完成")
    return historical_features, future_returns

def calculate_true_score(features: Dict) -> float:
    """
    计算"真实"评分 - 模拟一个理想的评分函数
    我们的优化器应该能够发现这些权重
    """
    score = 50  # 基础分
    
    # KDJ低位给高分
    if features['kdj_k'] < 30 and features['kdj_d'] < 30:
        score += 20
    elif features['kdj_k'] < 50:
        score += 10
    
    # RSI超卖给高分
    if features['rsi'] < 30:
        score += 15
    elif features['rsi'] < 50:
        score += 8
    
    # 价格在趋势线之上给高分
    if features['close'] > features['bbi']:
        score += 8
    if features['close'] > features['zhixing_trend']:
        score += 7
    
    # 成交量放大给高分
    if features['volume_surge'] > 2.0:
        score += 12
    elif features['volume_surge'] > 1.5:
        score += 6
    
    # 价格动量给高分
    if features['price_momentum'] > 0.1:
        score += 10
    elif features['price_momentum'] > 0:
        score += 5
    
    # 低估值给高分
    if features['pe_ttm'] < 15:
        score += 8
    if features['pb'] < 1.5:
        score += 5
    
    # 低波动率给高分
    if features['volatility'] < 0.03:
        score += 5
    
    return max(0, min(100, score))

def run_optimization_test():
    """运行权重优化测试"""
    print("🚀 开始权重优化测试")
    
    try:
        # 直接导入模块，使用exec来避免导入问题
        qlib_optimizer_path = os.path.join(current_dir, 'qlib_weight_optimizer.py')
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("qlib_weight_optimizer", qlib_optimizer_path)
        qlib_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qlib_module)
        QlibWeightOptimizer = qlib_module.QlibWeightOptimizer
        
        # 创建优化器
        optimizer = QlibWeightOptimizer(optimization_period_days=30)
        
        # 生成模拟数据
        historical_features, future_returns = generate_mock_stock_data(
            num_stocks=200,  # 使用200只股票
            num_days=30
        )
        
        # 设置缓存数据
        optimizer.historical_data_cache = historical_features
        optimizer.future_returns_cache = future_returns
        
        print("📊 缓存数据设置完成")
        print(f"   历史特征: {len(historical_features)} 只股票")
        print(f"   未来收益: {len(future_returns)} 只股票")
        
        # 运行优化
        print("🎯 开始权重优化（20轮测试）...")
        results = optimizer.run_optimization(max_evals=20)
        
        # 分析结果
        print("\n" + "="*60)
        print("📊 优化结果分析")
        print("="*60)
        
        summary = results['optimization_summary']
        best_weights = results['best_weights']
        
        print(f"总优化轮次: {summary['total_trials']}")
        print(f"最佳相关性: {summary['best_correlation']:.4f}")
        print(f"V3.0基线相关性: {summary['v30_baseline_correlation']:.4f}")
        print(f"相对改进: {summary['improvement_vs_v30']:+.4f}")
        print(f"改进幅度: {(summary['improvement_vs_v30']/summary['v30_baseline_correlation']*100):+.1f}%")
        
        print("\n📊 最优权重配置:")
        print("-" * 40)
        
        # 按权重大小排序显示
        sorted_weights = sorted(best_weights.items(), key=lambda x: x[1], reverse=True)
        for param, weight in sorted_weights:
            print(f"{param:<20}: {weight:.4f}")
        
        print("\n🎯 测试结论:")
        if summary['best_correlation'] > 0.02:
            print("✅ 优化器成功找到了正相关性配置！")
            if summary['improvement_vs_v30'] > 0:
                print("🎉 相关性超越V3.0基线，优化效果显著！")
            else:
                print("⚠️ 相关性未超越V3.0基线，但已实现正相关")
        else:
            print("❌ 优化器未能找到强正相关性配置")
        
        # 保存测试结果
        test_results = {
            'test_type': 'mock_data_optimization',
            'test_date': datetime.now().isoformat(),
            'data_stats': {
                'num_stocks': len(historical_features),
                'num_days': 30
            },
            'optimization_results': results
        }
        
        os.makedirs("reports/qlib_optimization", exist_ok=True)
        results_file = f"reports/qlib_optimization/mock_optimization_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n📄 测试结果已保存: {results_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 优化测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def analyze_mock_vs_v35():
    """分析模拟优化结果与V3.5的对比"""
    print("\n🔍 分析模拟优化 vs V3.5对比")
    print("-" * 40)
    
    # V3.5的问题总结
    v35_issues = {
        '相关性': '从+0.05~+0.09变为-0.02~-0.04',
        '评分标准差': '从12.94降至9.73',
        '高分样本': '90+分股票样本为0', 
        'P值显著性': '30天收益P值升至0.2155'
    }
    
    print("V3.5问题诊断:")
    for issue, desc in v35_issues.items():
        print(f"  ❌ {issue}: {desc}")
    
    print("\n期望优化效果:")
    expectations = [
        "✅ 相关性恢复至+0.03以上",
        "✅ 评分标准差提升至12-15", 
        "✅ 高分股票样本数增加",
        "✅ P值显著性<0.05"
    ]
    
    for exp in expectations:
        print(f"  {exp}")
    
    print("\n💡 如果模拟测试成功，说明:")
    print("  1. 优化器算法正确")
    print("  2. V3.5权重配置确实存在问题")
    print("  3. 可以应用到真实数据获得改进")

def main():
    """主函数"""
    print("🧪 Qlib权重优化器模拟测试")
    print("="*60)
    
    # 分析期望
    analyze_mock_vs_v35()
    
    print("\n" + "="*60)
    
    # 运行优化测试
    success = run_optimization_test()
    
    print("\n" + "="*60)
    print("📝 总结")
    print("="*60)
    
    if success:
        print("🎉 模拟优化测试成功完成！")
        print("💡 下一步：可以用真实数据运行完整优化")
        print("   python3 qlib_integration/qlib_weight_optimizer.py")
    else:
        print("⚠️ 模拟测试失败，需要进一步调试")
    
    return success

if __name__ == "__main__":
    main()