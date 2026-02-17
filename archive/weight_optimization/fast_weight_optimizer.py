#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速权重优化器 - 基于预计算因子矩阵
避免重复生成报告，支持秒级权重迭代
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import glob
from itertools import product
from scipy.optimize import minimize
import concurrent.futures

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class FastWeightOptimizer:
    """快速权重优化器"""
    
    def __init__(self):
        self.dimensions = ['technical', 'fundamental', 'performance', 'sentiment', 'risk_control', 'market_regime']
        self.factor_matrix = {}
        self.historical_returns = {}
        
    def load_precomputed_factors(self, data_dir="reports/daily_selection_v3.1"):
        """一次性加载所有因子分数矩阵"""
        print("📊 加载预计算因子矩阵...")
        
        json_files = glob.glob(f"{data_dir}/analysis_data_*.json")
        all_data = []
        
        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                date = os.path.basename(json_file).replace('analysis_data_', '').replace('.json', '')
                
                if 'all_stocks_with_scores' in data:
                    for stock in data['all_stocks_with_scores']:
                        factor_scores = stock.get('factor_scores', {})
                        
                        record = {
                            'date': date,
                            'code': stock['stock_code'],
                            'total_score': stock.get('score', 0)
                        }
                        
                        # 添加各维度分数
                        for dim in self.dimensions:
                            record[dim] = factor_scores.get(dim, 0) * 100
                        
                        all_data.append(record)
                        
            except Exception as e:
                print(f"⚠️ 读取文件失败 {json_file}: {e}")
                continue
        
        df = pd.DataFrame(all_data)
        
        # 构建因子矩阵 - 每个维度一个数组
        for dim in self.dimensions:
            self.factor_matrix[dim] = df[dim].values
        
        # 存储日期和股票代码用于分析
        self.dates = df['date'].values
        self.codes = df['code'].values
        
        print(f"✅ 因子矩阵加载完成: {len(df)} 条记录")
        print(f"📅 时间范围: {df['date'].min()} 到 {df['date'].max()}")
        
        return df
    
    def calculate_weighted_scores(self, weights):
        """根据权重计算总分 - 毫秒级计算"""
        # 确保权重归一化
        total_weight = sum(weights.values())
        normalized_weights = {dim: w/total_weight for dim, w in weights.items()}
        
        # 矩阵运算计算加权总分
        weighted_score = np.zeros(len(self.factor_matrix[self.dimensions[0]]))
        
        for dim in self.dimensions:
            weighted_score += normalized_weights[dim] * self.factor_matrix[dim]
        
        return weighted_score
    
    def evaluate_weights_performance(self, weights, metric='correlation'):
        """评估权重组合的性能"""
        weighted_scores = self.calculate_weighted_scores(weights)
        
        if metric == 'correlation':
            # 这里可以添加与未来收益的相关性计算
            # 暂时用分数分布的合理性作为代理指标
            return self.score_distribution_quality(weighted_scores)
        
        elif metric == 'sharpe':
            # 可以添加基于历史回报的夏普比率计算
            pass
        
        return np.mean(weighted_scores)
    
    def score_distribution_quality(self, scores):
        """评估分数分布的合理性"""
        # 好的分数分布应该：
        # 1. 高分股票占比合适 (90+ < 5%, 80-90 < 15%)
        # 2. 分布相对均匀，避免过于集中
        
        score_90_plus = np.sum(scores >= 90) / len(scores)
        score_80_90 = np.sum((scores >= 80) & (scores < 90)) / len(scores)
        score_70_80 = np.sum((scores >= 70) & (scores < 80)) / len(scores)
        
        # 目标分布: 90+ < 5%, 80-90 < 15%, 70-80 < 25%
        target_penalty = 0
        if score_90_plus > 0.05:
            target_penalty += (score_90_plus - 0.05) * 10
        if score_80_90 > 0.15:
            target_penalty += (score_80_90 - 0.15) * 5
        
        # 分布均匀性 - 标准差适中
        score_std = np.std(scores)
        if score_std < 5:  # 过于集中
            target_penalty += (5 - score_std) * 0.5
        elif score_std > 20:  # 过于分散
            target_penalty += (score_std - 20) * 0.2
        
        # 返回质量分数 (越高越好)
        quality_score = 100 - target_penalty
        return max(0, quality_score)
    
    def grid_search_weights(self, step=0.05, max_workers=4):
        """网格搜索最优权重"""
        print("🔍 开始网格搜索权重优化...")
        
        # 定义搜索范围
        weight_ranges = {
            'technical': np.arange(0.45, 0.75, step),
            'fundamental': np.arange(0.05, 0.25, step), 
            'performance': np.arange(0.10, 0.25, step),
            'sentiment': np.arange(0.02, 0.10, step),
            'risk_control': np.arange(0.02, 0.10, step),
            'market_regime': np.arange(0.02, 0.10, step)
        }
        
        # 生成所有权重组合
        weight_combinations = []
        for combo in product(*weight_ranges.values()):
            weights = dict(zip(self.dimensions, combo))
            # 过滤权重和接近1.0的组合
            if 0.95 <= sum(weights.values()) <= 1.05:
                weight_combinations.append(weights)
        
        print(f"📊 搜索空间: {len(weight_combinations)} 个权重组合")
        
        # 并行评估权重组合
        best_weights = None
        best_score = -np.inf
        
        def evaluate_single_weight(weights):
            score = self.evaluate_weights_performance(weights)
            return weights, score
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(evaluate_single_weight, weight_combinations))
        
        # 找出最佳权重
        for weights, score in results:
            if score > best_score:
                best_score = score
                best_weights = weights
        
        print(f"✅ 网格搜索完成!")
        print(f"🎯 最佳性能分数: {best_score:.2f}")
        print(f"🏆 最佳权重组合:")
        for dim, weight in best_weights.items():
            print(f"   {dim}: {weight:.3f}")
        
        return best_weights, best_score
    
    def optimize_weights_scipy(self):
        """使用scipy优化器进行权重优化"""
        print("🔧 使用scipy优化器进行权重优化...")
        
        # 初始权重
        x0 = [0.60, 0.18, 0.15, 0.05, 0.05, 0.05]  # 对应各维度
        
        # 约束: 权重和为1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # 边界: 每个权重在合理范围内
        bounds = [(0.02, 0.8), (0.02, 0.3), (0.05, 0.3), 
                 (0.01, 0.15), (0.01, 0.15), (0.01, 0.15)]
        
        def objective(x):
            weights = dict(zip(self.dimensions, x))
            return -self.evaluate_weights_performance(weights)  # 负号因为要最大化
        
        result = minimize(objective, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = dict(zip(self.dimensions, result.x))
            optimal_score = -result.fun
            
            print(f"✅ scipy优化完成!")
            print(f"🎯 最佳性能分数: {optimal_score:.2f}")
            print(f"🏆 最佳权重组合:")
            for dim, weight in optimal_weights.items():
                print(f"   {dim}: {weight:.3f}")
            
            return optimal_weights, optimal_score
        else:
            print(f"❌ scipy优化失败: {result.message}")
            return None, None
    
    def save_optimized_config(self, optimal_weights, performance_score, method="grid_search"):
        """保存优化后的配置"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        optimized_config = {
            "version": f"v3.1-FastOptimized-{method}",
            "description": f"快速权重优化结果 ({method})",
            "optimization_info": {
                "method": method,
                "performance_score": performance_score,
                "optimization_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "data_records_used": len(self.factor_matrix[self.dimensions[0]])
            },
            "optimized_weights": optimal_weights,
            "weight_changes": {},
            "expected_improvements": [
                "基于预计算因子矩阵的快速优化",
                "秒级权重迭代验证",
                "分数分布质量优化",
                "支持大规模权重空间搜索"
            ]
        }
        
        # 保存配置文件
        os.makedirs("scoring/v3/fast_optimization_results", exist_ok=True)
        config_file = f"scoring/v3/fast_optimization_results/fast_optimized_config_{timestamp}.json"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(optimized_config, f, indent=2, ensure_ascii=False)
        
        print(f"💾 优化配置已保存: {config_file}")
        return config_file

def main():
    """主函数"""
    optimizer = FastWeightOptimizer()
    
    # 加载预计算因子矩阵
    df = optimizer.load_precomputed_factors()
    
    if len(df) < 1000:
        print("⚠️ 数据量较少，建议增加更多历史数据")
    
    print("\n" + "="*60)
    print("🚀 自动运行两种优化方法...")
    
    choice = '3'  # 自动选择运行两种方法
    
    if choice in ['1', '3']:
        # 网格搜索
        best_weights_grid, best_score_grid = optimizer.grid_search_weights(step=0.05)
        if best_weights_grid:
            optimizer.save_optimized_config(best_weights_grid, best_score_grid, "grid_search")
    
    if choice in ['2', '3']:
        # scipy优化
        best_weights_scipy, best_score_scipy = optimizer.optimize_weights_scipy()
        if best_weights_scipy:
            optimizer.save_optimized_config(best_weights_scipy, best_score_scipy, "scipy_optimize")
    
    print("\n🎉 快速权重优化完成!")

if __name__ == "__main__":
    main()