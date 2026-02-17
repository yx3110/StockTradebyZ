#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
权重向量回归测试和优化算法

实现用户需求的第二步：
"接着利用这些已经计算好的数据，对不同的权重进行回归。
比如你可以先测试20240204某个股票的六个feature值（我们可以叫做featurevector）
和六个weight的（weight vector），得出一个评分，
接着检测这个评分是否能反应后续的走势"
"""

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import argparse
import logging
from datetime import datetime
import time
import warnings

warnings.filterwarnings('ignore')

class WeightRegressionOptimizer:
    """权重向量回归测试和优化器"""
    
    def __init__(self, db_path="weight_optimization_cache.db"):
        """初始化优化器"""
        self.db_path = db_path
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 定义特征列名
        self.feature_columns = [
            'technical', 'fundamental', 'performance', 
            'sentiment', 'risk_control', 'market_regime'
        ]
        
        # 定义预测期间
        self.return_periods = ['return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d']
        
        # 当前最佳权重
        self.best_weights = None
        self.best_score = None
        
    def load_cached_data(self, sample_limit: Optional[int] = None) -> pd.DataFrame:
        """加载缓存数据"""
        self.logger.info("🔄 加载缓存数据...")
        
        conn = sqlite3.connect(self.db_path)
        
        # 基础查询
        query = f"""
        SELECT code, date, technical, fundamental, performance, 
               sentiment, risk_control, market_regime,
               return_1d, return_3d, return_5d, return_10d, return_20d
        FROM stock_indicators 
        WHERE return_5d IS NOT NULL
        """
        
        # 添加样本限制
        if sample_limit:
            query += f" ORDER BY RANDOM() LIMIT {sample_limit}"
        else:
            query += " ORDER BY date, code"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        self.logger.info(f"✅ 加载完成: {len(df):,} 条记录")
        self.logger.info(f"📅 时间范围: {df['date'].min()} 到 {df['date'].max()}")
        self.logger.info(f"🏢 股票数量: {df['code'].nunique():,} 只")
        
        return df
        
    def calculate_weighted_score(self, feature_vector: np.ndarray, weight_vector: np.ndarray) -> float:
        """计算加权评分
        
        Args:
            feature_vector: 特征向量 [technical, fundamental, performance, sentiment, risk_control, market_regime]
            weight_vector: 权重向量 [w1, w2, w3, w4, w5, w6]
        
        Returns:
            加权评分 (0-100)
        """
        if len(feature_vector) != len(weight_vector):
            raise ValueError(f"特征向量长度 {len(feature_vector)} 与权重向量长度 {len(weight_vector)} 不匹配")
        
        # 计算加权得分
        weighted_score = np.dot(feature_vector, weight_vector) * 100
        return weighted_score
        
    def test_weight_combination(self, data: pd.DataFrame, weights: Dict[str, float], 
                              target_period: str = 'return_5d', name: str = "测试组合") -> Dict:
        """测试单个权重组合的预测效果"""
        
        # 转换权重字典为向量
        weight_vector = np.array([
            weights.get('technical', 0),
            weights.get('fundamental', 0), 
            weights.get('performance', 0),
            weights.get('sentiment', 0),
            weights.get('risk_control', 0),
            weights.get('market_regime', 0)
        ])
        
        # 计算所有股票的加权评分
        feature_matrix = data[self.feature_columns].values
        weighted_scores = np.array([
            self.calculate_weighted_score(feature_vector, weight_vector)
            for feature_vector in feature_matrix
        ])
        
        # 获取目标收益率
        target_returns = data[target_period].values
        
        # 去除NaN值
        valid_mask = ~np.isnan(target_returns) & ~np.isnan(weighted_scores)
        valid_scores = weighted_scores[valid_mask]
        valid_returns = target_returns[valid_mask]
        
        if len(valid_returns) < 100:
            return {
                'name': name,
                'error': f'有效样本数量不足: {len(valid_returns)}'
            }
        
        # 计算相关性
        correlation, p_value = stats.pearsonr(valid_scores, valid_returns)
        
        # 计算胜率 (高分股票未来上涨的概率)
        score_quantiles = np.quantile(valid_scores, [0.75, 0.9, 0.95])
        
        results = {
            'name': name,
            'weights': weights,
            'total_weight': sum(weights.values()),
            'sample_size': len(valid_returns),
            'correlation': correlation,
            'p_value': p_value,
            'correlation_abs': abs(correlation)
        }
        
        # 计算不同分位数的胜率和收益
        for i, (threshold, label) in enumerate([(0.75, 'top25'), (0.9, 'top10'), (0.95, 'top5')]):
            top_mask = valid_scores >= score_quantiles[i]
            if np.sum(top_mask) > 10:  # 至少10个样本
                top_returns = valid_returns[top_mask]
                win_rate = (top_returns > 0).mean()
                avg_return = top_returns.mean()
                sharpe = avg_return / top_returns.std() if top_returns.std() > 0 else 0
                
                results[f'win_rate_{label}'] = win_rate
                results[f'avg_return_{label}'] = avg_return
                results[f'sharpe_{label}'] = sharpe
                results[f'sample_size_{label}'] = len(top_returns)
        
        # 计算综合评分
        if 'win_rate_top25' in results and abs(correlation) > 0.01:
            # 综合考虑相关性、胜率和样本量
            composite_score = (
                abs(correlation) * 0.4 +           # 相关性权重40%
                results['win_rate_top25'] * 0.4 +  # 胜率权重40%
                min(1.0, len(valid_returns) / 10000) * 0.2  # 样本量权重20%
            )
            results['composite_score'] = composite_score
        else:
            results['composite_score'] = 0.0
        
        return results
        
    def grid_search_optimization(self, data: pd.DataFrame, target_period: str = 'return_5d', 
                               grid_size: int = 10) -> List[Dict]:
        """网格搜索最优权重组合"""
        self.logger.info(f"🔍 开始网格搜索优化 (目标: {target_period})")
        self.logger.info(f"🔧 网格密度: {grid_size} × 6 = {grid_size**6:,} 种组合")
        
        # 生成权重网格
        weight_range = np.linspace(0, 1, grid_size)
        
        best_results = []
        total_combinations = 0
        valid_combinations = 0
        
        start_time = time.time()
        
        # 遍历所有权重组合
        for w1 in weight_range:
            for w2 in weight_range:
                for w3 in weight_range:
                    for w4 in weight_range:
                        for w5 in weight_range:
                            for w6 in weight_range:
                                total_combinations += 1
                                
                                # 跳过权重和为0的情况
                                total_weight = w1 + w2 + w3 + w4 + w5 + w6
                                if total_weight < 0.1:
                                    continue
                                
                                # 标准化权重
                                weights = {
                                    'technical': w1 / total_weight,
                                    'fundamental': w2 / total_weight,
                                    'performance': w3 / total_weight,
                                    'sentiment': w4 / total_weight,
                                    'risk_control': w5 / total_weight,
                                    'market_regime': w6 / total_weight
                                }
                                
                                # 测试权重组合
                                result = self.test_weight_combination(
                                    data, weights, target_period, 
                                    f"Grid_{valid_combinations+1:04d}"
                                )
                                
                                if 'error' not in result and result.get('composite_score', 0) > 0:
                                    best_results.append(result)
                                    valid_combinations += 1
                                
                                # 进度显示
                                if total_combinations % 10000 == 0:
                                    elapsed = time.time() - start_time
                                    self.logger.info(f"  进度: {total_combinations:,}/{grid_size**6:,} "
                                                   f"({total_combinations/grid_size**6*100:.1f}%) "
                                                   f"已用时: {elapsed:.1f}s")
        
        # 按综合评分排序
        best_results.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"✅ 网格搜索完成: {elapsed_time:.1f} 秒")
        self.logger.info(f"📊 测试组合: {total_combinations:,} (有效: {valid_combinations:,})")
        
        return best_results[:50]  # 返回前50个最佳组合
        
    def evolutionary_optimization(self, data: pd.DataFrame, target_period: str = 'return_5d',
                                max_iterations: int = 100) -> Dict:
        """进化算法优化权重"""
        self.logger.info(f"🧬 开始进化算法优化 (目标: {target_period})")
        
        def objective_function(weight_vector):
            """优化目标函数 - 最大化综合评分"""
            # 确保权重非负且总和为1
            weight_vector = np.abs(weight_vector)
            weight_sum = np.sum(weight_vector)
            if weight_sum == 0:
                return 1e6  # 返回大的惩罚值
            weight_vector = weight_vector / weight_sum
            
            weights = {
                'technical': weight_vector[0],
                'fundamental': weight_vector[1],
                'performance': weight_vector[2],
                'sentiment': weight_vector[3],
                'risk_control': weight_vector[4],
                'market_regime': weight_vector[5]
            }
            
            result = self.test_weight_combination(data, weights, target_period)
            
            if 'error' in result:
                return 1e6
            
            # 返回负的综合评分（因为要最小化）
            return -result.get('composite_score', 0)
        
        # 设定搜索范围 [0, 1] for each weight
        bounds = [(0, 1) for _ in range(6)]
        
        # 运行差分进化算法
        start_time = time.time()
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=max_iterations,
            popsize=15,
            seed=42,
            disp=True
        )
        
        elapsed_time = time.time() - start_time
        
        # 提取最优权重
        optimal_weights = result.x
        optimal_weights = optimal_weights / np.sum(optimal_weights)  # 标准化
        
        best_weights = {
            'technical': optimal_weights[0],
            'fundamental': optimal_weights[1], 
            'performance': optimal_weights[2],
            'sentiment': optimal_weights[3],
            'risk_control': optimal_weights[4],
            'market_regime': optimal_weights[5]
        }
        
        # 评估最优结果
        best_result = self.test_weight_combination(data, best_weights, target_period, "进化算法最优解")
        
        self.logger.info(f"✅ 进化算法完成: {elapsed_time:.1f} 秒")
        self.logger.info(f"🏆 最优综合评分: {best_result.get('composite_score', 0):.4f}")
        
        return best_result
        
    def bayesian_optimization(self, data: pd.DataFrame, target_period: str = 'return_5d',
                            n_calls: int = 100) -> Dict:
        """贝叶斯优化权重"""
        try:
            from skopt import gp_minimize
            from skopt.space import Real
        except ImportError:
            self.logger.warning("⚠️ skopt 未安装，跳过贝叶斯优化")
            return {}
        
        self.logger.info(f"🎯 开始贝叶斯优化 (目标: {target_period})")
        
        def objective_function(weights):
            """目标函数"""
            weight_dict = {
                'technical': weights[0],
                'fundamental': weights[1],
                'performance': weights[2], 
                'sentiment': weights[3],
                'risk_control': weights[4],
                'market_regime': weights[5]
            }
            
            result = self.test_weight_combination(data, weight_dict, target_period)
            
            if 'error' in result:
                return 1e6
            
            return -result.get('composite_score', 0)  # 负值因为要最小化
        
        # 定义搜索空间
        dimensions = [Real(0.0, 1.0, name=f'w_{i}') for i in range(6)]
        
        start_time = time.time()
        result = gp_minimize(
            objective_function,
            dimensions,
            n_calls=n_calls,
            random_state=42
        )
        
        elapsed_time = time.time() - start_time
        
        # 提取最优权重并标准化
        optimal_weights = np.array(result.x)
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        best_weights = {
            'technical': optimal_weights[0],
            'fundamental': optimal_weights[1],
            'performance': optimal_weights[2],
            'sentiment': optimal_weights[3], 
            'risk_control': optimal_weights[4],
            'market_regime': optimal_weights[5]
        }
        
        # 评估最优结果
        best_result = self.test_weight_combination(data, best_weights, target_period, "贝叶斯优化最优解")
        
        self.logger.info(f"✅ 贝叶斯优化完成: {elapsed_time:.1f} 秒") 
        self.logger.info(f"🏆 最优综合评分: {best_result.get('composite_score', 0):.4f}")
        
        return best_result
        
    def compare_optimization_methods(self, data: pd.DataFrame, target_period: str = 'return_5d') -> Dict:
        """比较不同优化方法"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("🔍 权重优化方法对比分析")
        self.logger.info(f"{'='*60}")
        
        results = {}
        
        # 1. 当前v3.1权重 (基准)
        current_weights = {
            'technical': 0.60,
            'fundamental': 0.15,
            'performance': 0.15,
            'sentiment': 0.05,
            'risk_control': 0.05,
            'market_regime': 0.05
        }
        
        results['baseline'] = self.test_weight_combination(
            data, current_weights, target_period, "当前v3.1权重"
        )
        
        # 2. 进化算法优化
        results['evolutionary'] = self.evolutionary_optimization(data, target_period, max_iterations=50)
        
        # 3. 贝叶斯优化
        results['bayesian'] = self.bayesian_optimization(data, target_period, n_calls=50)
        
        # 4. 简化网格搜索（小规模）
        results['grid_top'] = self.grid_search_optimization(data, target_period, grid_size=5)
        
        # 显示对比结果
        self.display_comparison_results(results, target_period)
        
        return results
        
    def display_comparison_results(self, results: Dict, target_period: str):
        """显示对比结果"""
        print(f"\n🏆 权重优化对比结果 (目标期间: {target_period})")
        print("=" * 80)
        
        # 处理网格搜索结果
        grid_results = results.get('grid_top', [])
        if grid_results and len(grid_results) > 0:
            results['grid'] = grid_results[0]  # 取最佳结果
        
        # 按综合评分排序
        method_results = []
        for method_name, result in results.items():
            if method_name == 'grid_top':  # 跳过原始列表
                continue
            if isinstance(result, dict) and 'composite_score' in result:
                method_results.append((method_name, result))
        
        method_results.sort(key=lambda x: x[1].get('composite_score', 0), reverse=True)
        
        # 显示结果表格
        headers = ["方法", "综合评分", "相关性", "胜率(Top25%)", "权重分布"]
        print(f"{'方法':<12} {'综合评分':<10} {'相关性':<10} {'胜率':<10} {'权重分布'}")
        print("-" * 80)
        
        for method_name, result in method_results:
            method_label = {
                'baseline': '当前v3.1',
                'evolutionary': '进化算法', 
                'bayesian': '贝叶斯优化',
                'grid': '网格搜索'
            }.get(method_name, method_name)
            
            composite = result.get('composite_score', 0)
            correlation = result.get('correlation', 0)
            win_rate = result.get('win_rate_top25', 0)
            
            # 简化权重显示
            weights = result.get('weights', {})
            weight_str = f"技术:{weights.get('technical', 0):.2f} 基本:{weights.get('fundamental', 0):.2f}"
            
            print(f"{method_label:<12} {composite:<10.4f} {correlation:<+10.4f} {win_rate:<10.1%} {weight_str}")
        
        # 显示最佳方案详细信息
        if method_results:
            best_method, best_result = method_results[0]
            print(f"\n🥇 最佳方案: {best_method}")
            print("-" * 40)
            
            weights = best_result.get('weights', {})
            for dim, weight in weights.items():
                print(f"  {dim:15s}: {weight:6.1%}")
            
            print(f"\n📊 性能指标:")
            print(f"  综合评分: {best_result.get('composite_score', 0):.4f}")
            print(f"  相关系数: {best_result.get('correlation', 0):+.4f}")
            print(f"  Top25%胜率: {best_result.get('win_rate_top25', 0):.1%}")
            if 'avg_return_top25' in best_result:
                print(f"  Top25%平均收益: {best_result.get('avg_return_top25', 0):+.2f}%")
            print(f"  样本数量: {best_result.get('sample_size', 0):,}")
        
        return method_results[0] if method_results else None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="权重向量回归测试和优化")
    parser.add_argument("--target-period", default="return_5d", 
                       choices=['return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d'],
                       help="目标收益期间")
    parser.add_argument("--sample-limit", type=int, help="数据样本限制")
    parser.add_argument("--method", default="compare",
                       choices=['grid', 'evolutionary', 'bayesian', 'compare'],
                       help="优化方法")
    
    args = parser.parse_args()
    
    # 创建优化器
    optimizer = WeightRegressionOptimizer()
    
    # 加载数据
    data = optimizer.load_cached_data(sample_limit=args.sample_limit)
    
    if data.empty:
        print("❌ 没有找到有效数据，请先运行数据生成步骤")
        return
    
    # 执行优化
    if args.method == "compare":
        results = optimizer.compare_optimization_methods(data, args.target_period)
    elif args.method == "grid":
        results = optimizer.grid_search_optimization(data, args.target_period)
    elif args.method == "evolutionary":
        results = optimizer.evolutionary_optimization(data, args.target_period)
    elif args.method == "bayesian":
        results = optimizer.bayesian_optimization(data, args.target_period)
    
    print(f"\n✅ 权重优化分析完成！")

if __name__ == "__main__":
    main()