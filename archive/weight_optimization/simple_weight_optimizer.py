#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版v3.1权重优化器
基于现有报告数据快速优化权重分配
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import glob

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class SimpleWeightOptimizer:
    """简化版权重优化器"""
    
    def __init__(self):
        # 权重搜索范围
        self.weight_ranges = {
            'technical': [0.50, 0.55, 0.60, 0.65],      # 技术指标
            'fundamental': [0.12, 0.15, 0.18, 0.22],   # 基本面  
            'performance': [0.10, 0.12, 0.15, 0.18],   # 市场表现
            'sentiment': [0.03, 0.05, 0.07, 0.10],     # 情绪指标
            'risk_control': [0.03, 0.05, 0.07, 0.10],  # 风险控制
        }
        
    def load_existing_reports_data(self) -> pd.DataFrame:
        """加载现有的报告数据"""
        print("📊 加载现有报告数据...")
        
        # 查找所有的分析数据文件
        v3_dir = "reports/daily_selection_v3"
        json_files = glob.glob(f"{v3_dir}/analysis_data_*.json")
        
        if not json_files:
            print("❌ 未找到v3分析数据文件")
            return pd.DataFrame()
        
        all_data = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 提取日期
                date = os.path.basename(json_file).replace('analysis_data_', '').replace('.json', '')
                date_formatted = f"{date[:4]}-{date[4:6]}-{date[6:]}"
                
                # 提取股票评分数据
                if 'all_stocks_with_scores' in data:
                    for stock in data['all_stocks_with_scores']:
                        stock_data = {
                            'date': date_formatted,
                            'code': stock['stock_code'],
                            'score': stock.get('score', 0),
                            'recommendation': stock.get('recommendation', '观望'),
                            **stock.get('factor_scores', {})
                        }
                        all_data.append(stock_data)
                        
            except Exception as e:
                print(f"⚠️ 读取文件失败 {json_file}: {e}")
                continue
        
        df = pd.DataFrame(all_data)
        print(f"📈 成功加载 {len(df)} 条股票评分记录，涵盖 {df['date'].nunique()} 个交易日")
        
        return df
    
    def generate_weight_combinations(self) -> list:
        """生成权重组合"""
        combinations = []
        
        for tech in self.weight_ranges['technical']:
            for fund in self.weight_ranges['fundamental']:
                for perf in self.weight_ranges['performance']:
                    for sent in self.weight_ranges['sentiment']:
                        for risk in self.weight_ranges['risk_control']:
                            # 计算market_regime权重
                            market = 1.0 - (tech + fund + perf + sent + risk)
                            
                            # 确保market权重在合理范围内
                            if 0.03 <= market <= 0.12:
                                combo = {
                                    'technical': tech,
                                    'fundamental': fund,
                                    'performance': perf,
                                    'sentiment': sent,
                                    'risk_control': risk,
                                    'market_regime': market
                                }
                                combinations.append(combo)
        
        print(f"🔍 生成了 {len(combinations)} 个权重组合")
        return combinations
    
    def calculate_score_with_weights(self, row: pd.Series, weights: dict) -> float:
        """使用新权重计算分数"""
        score = 0
        factor_mapping = {
            'technical': 'technical',
            'fundamental': 'fundamental', 
            'performance': 'performance',
            'sentiment': 'sentiment',
            'risk_control': 'risk_control',
            'market_regime': 'market_regime'
        }
        
        for weight_key, factor_key in factor_mapping.items():
            if weight_key in weights and factor_key in row:
                factor_score = row[factor_key] if not pd.isna(row[factor_key]) else 0.5
                score += factor_score * weights[weight_key]
        
        return score * 100  # 转换为百分制
    
    def evaluate_weights(self, df: pd.DataFrame, weights: dict) -> dict:
        """评估权重组合效果"""
        if len(df) < 100:
            return {'score': 0, 'details': 'insufficient_data'}
        
        # 重新计算所有股票的分数
        df['new_score'] = df.apply(lambda row: self.calculate_score_with_weights(row, weights), axis=1)
        
        metrics = {}
        
        # 1. 评分分布合理性
        score_90_plus = (df['new_score'] >= 90).mean()
        score_80_90 = ((df['new_score'] >= 80) & (df['new_score'] < 90)).mean()
        score_70_80 = ((df['new_score'] >= 70) & (df['new_score'] < 80)).mean()
        score_below_60 = (df['new_score'] < 60).mean()
        
        # 理想分布：90+<5%, 80-90<15%, 70-80<25%, <60>50%
        distribution_score = 1.0
        if score_90_plus > 0.05:
            distribution_score -= (score_90_plus - 0.05) * 5
        if score_80_90 > 0.15:
            distribution_score -= (score_80_90 - 0.15) * 3
        if score_below_60 < 0.50:
            distribution_score -= (0.50 - score_below_60) * 2
            
        distribution_score = max(0, distribution_score)
        
        # 2. 区分度 - 标准差越大越好
        score_std = df['new_score'].std()
        discrimination_score = min(score_std / 20, 1.0)  # 标准差20分为满分
        
        # 3. 高分股票推荐质量
        high_score_stocks = df[df['new_score'] >= 80]
        buy_ratio = 0
        if len(high_score_stocks) > 0:
            buy_ratio = (high_score_stocks['recommendation'].isin(['买入', '谨慎买入'])).mean()
        
        # 4. 综合评分
        overall_score = (distribution_score * 0.4 + 
                        discrimination_score * 0.3 + 
                        buy_ratio * 0.3)
        
        metrics = {
            'overall_score': overall_score,
            'distribution_score': distribution_score,
            'discrimination_score': discrimination_score,
            'buy_ratio': buy_ratio,
            'score_std': score_std,
            'score_mean': df['new_score'].mean(),
            'score_90_plus_pct': score_90_plus * 100,
            'score_80_90_pct': score_80_90 * 100,
            'score_70_80_pct': score_70_80 * 100,
            'score_below_60_pct': score_below_60 * 100,
            'high_score_count': len(high_score_stocks)
        }
        
        return metrics
    
    def optimize_weights(self) -> dict:
        """执行权重优化"""
        print("🚀 开始权重优化...")
        
        # 加载数据
        df = self.load_existing_reports_data()
        if len(df) < 100:
            raise ValueError("数据量不足，无法进行权重优化")
        
        # 生成权重组合
        combinations = self.generate_weight_combinations()
        
        print("⚡ 评估权重组合...")
        results = []
        
        for i, weights in enumerate(combinations):
            metrics = self.evaluate_weights(df, weights)
            results.append({
                'weights': weights,
                'metrics': metrics
            })
            
            if (i + 1) % 20 == 0:
                print(f"✅ 已评估 {i + 1}/{len(combinations)} 个组合")
        
        # 排序找到最佳权重
        results.sort(key=lambda x: x['metrics']['overall_score'], reverse=True)
        
        best_result = results[0]
        
        print("\n🏆 权重优化完成！")
        return {
            'best_weights': best_result['weights'],
            'best_metrics': best_result['metrics'],
            'top_results': results[:5],
            'total_tested': len(results),
            'optimization_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_summary': {
                'total_records': len(df),
                'date_range': f"{df['date'].min()} 到 {df['date'].max()}",
                'unique_stocks': df['code'].nunique(),
                'unique_dates': df['date'].nunique()
            }
        }
    
    def generate_optimized_config(self, optimization_result: dict) -> dict:
        """生成优化后的配置"""
        best_weights = optimization_result['best_weights']
        
        # 读取原始配置
        config_path = "scoring/v3/v3_1_optimized_config.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            original_config = json.load(f)
        
        # 创建新配置
        new_config = original_config.copy()
        new_config['version'] = "v3.1-SimpleGridOptimized"
        new_config['description'] = "基于简化网格搜索优化的v3.1配置"
        
        # 更新权重
        original_weights = {}
        for category, sub_weights in original_config['weights'].items():
            original_weights[category] = sum(v for k, v in sub_weights.items() if not k.startswith('_'))
        
        # 按比例调整子权重
        for category, new_weight in best_weights.items():
            if category in new_config['weights']:
                original_total = original_weights[category]
                scale_factor = new_weight / original_total
                
                for sub_key, sub_value in new_config['weights'][category].items():
                    if not sub_key.startswith('_'):
                        new_config['weights'][category][sub_key] = sub_value * scale_factor
        
        # 添加优化信息
        new_config['simple_grid_optimization'] = {
            'optimization_date': optimization_result['optimization_date'],
            'data_summary': optimization_result['data_summary'],
            'combinations_tested': optimization_result['total_tested'],
            'best_metrics': optimization_result['best_metrics'],
            'weight_changes': {}
        }
        
        # 计算权重变化
        for category in best_weights:
            if category in original_weights:
                new_config['simple_grid_optimization']['weight_changes'][category] = {
                    'original': f"{original_weights[category]:.1%}",
                    'optimized': f"{best_weights[category]:.1%}",
                    'change': f"{best_weights[category] - original_weights[category]:+.1%}"
                }
        
        return new_config

def main():
    """主函数"""
    print("🚀 简化版v3.1权重优化器")
    print("=" * 50)
    
    optimizer = SimpleWeightOptimizer()
    
    try:
        # 执行优化
        result = optimizer.optimize_weights()
        
        # 显示结果
        best_weights = result['best_weights']
        best_metrics = result['best_metrics']
        
        print("\n📊 最优权重配置:")
        print("-" * 30)
        total = 0
        for category, weight in best_weights.items():
            print(f"{category:15s}: {weight:6.1%}")
            total += weight
        print(f"{'总计':<15s}: {total:6.1%}")
        
        print(f"\n📈 优化效果:")
        print("-" * 30)
        print(f"综合评分: {best_metrics['overall_score']:.4f}")
        print(f"分布合理性: {best_metrics['distribution_score']:.4f}")
        print(f"区分度评分: {best_metrics['discrimination_score']:.4f}")
        print(f"高分买入率: {best_metrics['buy_ratio']:.4f}")
        print(f"评分标准差: {best_metrics['score_std']:.2f}")
        print(f"评分均值: {best_metrics['score_mean']:.2f}")
        
        print(f"\n📊 评分分布:")
        print("-" * 30)
        print(f"90+分股票: {best_metrics['score_90_plus_pct']:.1f}%")
        print(f"80-90分股票: {best_metrics['score_80_90_pct']:.1f}%")
        print(f"70-80分股票: {best_metrics['score_70_80_pct']:.1f}%")
        print(f"60分以下: {best_metrics['score_below_60_pct']:.1f}%")
        
        # 生成配置文件
        optimized_config = optimizer.generate_optimized_config(result)
        
        # 保存结果
        os.makedirs("scoring/v3/optimization_results", exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        result_file = f"scoring/v3/optimization_results/simple_weight_optimization_{timestamp}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        config_file = f"scoring/v3/optimization_results/v3_1_simple_optimized_config_{timestamp}.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(optimized_config, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 结果已保存:")
        print(f"优化数据: {result_file}")
        print(f"新配置: {config_file}")
        
        # 显示前5个最佳组合
        print(f"\n🏆 前5个最佳权重组合:")
        print("-" * 60)
        for i, r in enumerate(result['top_results'][:5], 1):
            weights = r['weights']
            score = r['metrics']['overall_score']
            print(f"{i}. 评分:{score:.4f} - 技术:{weights['technical']:.1%} 基本:{weights['fundamental']:.1%} "
                  f"表现:{weights['performance']:.1%} 情绪:{weights['sentiment']:.1%} "
                  f"风控:{weights['risk_control']:.1%} 市场:{weights['market_regime']:.1%}")
        
    except Exception as e:
        print(f"❌ 优化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()