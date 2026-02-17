#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
敏感市场环境乘数优化器

针对市场环境得分范围较小(0.031-0.040)的问题，
设计多种敏感的乘数映射函数，并优化映射参数。
"""

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Callable
import argparse
import logging
from datetime import datetime
import time
import warnings
import math

warnings.filterwarnings('ignore')

class SensitiveMarketMultiplierOptimizer:
    """敏感市场环境乘数优化器"""
    
    def __init__(self, db_path: str = "weight_optimization_cache.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # 预计算数据统计信息
        self.regime_min = 0.031
        self.regime_max = 0.040
        self.regime_range = self.regime_max - self.regime_min
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def load_cached_data(self, sample_limit: int = None) -> pd.DataFrame:
        """加载缓存数据"""
        self.logger.info("🔄 加载缓存数据...")
        
        query = """
        SELECT 
            code,
            date,
            technical,
            fundamental,
            performance,
            sentiment,
            risk_control,
            market_regime,
            return_1d,
            return_3d,
            return_5d,
            return_10d,
            return_20d
        FROM stock_indicators
        WHERE technical IS NOT NULL 
        AND fundamental IS NOT NULL
        AND performance IS NOT NULL
        AND sentiment IS NOT NULL  
        AND risk_control IS NOT NULL
        AND market_regime IS NOT NULL
        AND return_5d IS NOT NULL
        ORDER BY RANDOM()  -- 随机采样，确保时间分布均匀
        """
        
        if sample_limit:
            query += f" LIMIT {sample_limit}"
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                data = pd.read_sql_query(query, conn)
                
            if not data.empty:
                self.logger.info(f"✅ 加载完成: {len(data):,} 条记录")
                self.logger.info(f"📅 时间范围: {data['date'].min()} 到 {data['date'].max()}")
                self.logger.info(f"🏢 股票数量: {data['code'].nunique():,} 只")
                self.logger.info(f"📊 交易日数量: {data['date'].nunique():,} 天")
                
                # 显示市场环境得分分布
                regime_stats = data['market_regime'].describe()
                self.logger.info(f"📊 市场环境得分分布: 最小={regime_stats['min']:.4f}, 最大={regime_stats['max']:.4f}, 平均={regime_stats['mean']:.4f}")
                self.logger.info(f"📊 市场环境得分唯一值: {data['market_regime'].nunique()} 个")
            else:
                self.logger.warning("⚠️ 没有找到有效数据")
                
            return data
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            return pd.DataFrame()
    
    def normalize_regime_score(self, regime_score: float) -> float:
        """将市场环境得分标准化到0-1范围"""
        if self.regime_range == 0:
            return 0.5  # 如果没有变化，返回中性值
        return (regime_score - self.regime_min) / self.regime_range
    
    def linear_multiplier(self, regime_score: float, min_mult: float, max_mult: float) -> float:
        """线性乘数映射"""
        normalized = self.normalize_regime_score(regime_score)
        return min_mult + normalized * (max_mult - min_mult)
    
    def exponential_multiplier(self, regime_score: float, base: float, scale: float, offset: float) -> float:
        """指数乘数映射 - 对小变化更敏感"""
        normalized = self.normalize_regime_score(regime_score)
        # 将0-1映射到-1到1，然后应用指数函数
        centered = (normalized - 0.5) * 2  # -1 to 1
        multiplier = offset + scale * (base ** centered)
        return max(0.1, multiplier)  # 确保乘数为正
    
    def sigmoid_multiplier(self, regime_score: float, steepness: float, midpoint: float, 
                          min_mult: float, max_mult: float) -> float:
        """Sigmoid乘数映射 - 在中点附近变化敏感"""
        normalized = self.normalize_regime_score(regime_score)
        sigmoid_val = 1 / (1 + math.exp(-steepness * (normalized - midpoint)))
        return min_mult + sigmoid_val * (max_mult - min_mult)
    
    def power_multiplier(self, regime_score: float, power: float, min_mult: float, max_mult: float) -> float:
        """幂函数乘数映射 - 可调节敏感性曲线"""
        normalized = self.normalize_regime_score(regime_score)
        if power != 0:
            powered = normalized ** power
        else:
            powered = normalized
        return min_mult + powered * (max_mult - min_mult)
    
    def get_multiplier_function(self, multiplier_type: str, params: Dict) -> Callable:
        """根据类型和参数返回乘数函数"""
        if multiplier_type == "linear":
            return lambda score: self.linear_multiplier(score, params['min_mult'], params['max_mult'])
        elif multiplier_type == "exponential":
            return lambda score: self.exponential_multiplier(score, params['base'], params['scale'], params['offset'])
        elif multiplier_type == "sigmoid":
            return lambda score: self.sigmoid_multiplier(score, params['steepness'], params['midpoint'], 
                                                       params['min_mult'], params['max_mult'])
        elif multiplier_type == "power":
            return lambda score: self.power_multiplier(score, params['power'], params['min_mult'], params['max_mult'])
        else:
            return lambda score: 1.0  # 默认无乘数效果
    
    def calculate_stock_quality_score(self, quality_weights: Dict[str, float], 
                                    technical: float, fundamental: float, 
                                    performance: float, sentiment: float, 
                                    risk_control: float) -> float:
        """计算个股质量得分（5个因子加权）"""
        quality_score = (
            technical * quality_weights['technical'] +
            fundamental * quality_weights['fundamental'] + 
            performance * quality_weights['performance'] +
            sentiment * quality_weights['sentiment'] +
            risk_control * quality_weights['risk_control']
        )
        return quality_score
    
    def calculate_final_score(self, quality_weights: Dict[str, float], 
                            multiplier_func: Callable, row: pd.Series) -> float:
        """
        计算最终得分
        final_score = market_regime_multiplier × stock_quality_score
        """
        # 计算市场环境乘数
        market_multiplier = multiplier_func(row['market_regime'])
        
        # 计算个股质量得分
        quality_score = self.calculate_stock_quality_score(
            quality_weights,
            row['technical'],
            row['fundamental'], 
            row['performance'],
            row['sentiment'],
            row['risk_control']
        )
        
        # 最终得分 = 市场环境乘数 × 个股质量得分
        final_score = market_multiplier * quality_score
        
        return final_score
    
    def test_multiplier_combination(self, data: pd.DataFrame, quality_weights: Dict[str, float],
                                  multiplier_type: str, multiplier_params: Dict,
                                  target_period: str = 'return_5d', name: str = "测试") -> Dict:
        """测试乘数组合效果"""
        try:
            # 创建乘数函数
            multiplier_func = self.get_multiplier_function(multiplier_type, multiplier_params)
            
            # 计算每只股票的最终得分
            data['predicted_score'] = data.apply(
                lambda row: self.calculate_final_score(quality_weights, multiplier_func, row), axis=1
            )
            
            # 计算相关性
            correlation = data['predicted_score'].corr(data[target_period])
            if pd.isna(correlation):
                correlation = 0.0
                
            # 统计显著性检验
            try:
                _, p_value = stats.pearsonr(data['predicted_score'], data[target_period])
            except:
                p_value = 1.0
            
            # 分组回测 - 按日期分组，然后按得分排序
            results_by_date = []
            
            for date in data['date'].unique():
                daily_data = data[data['date'] == date].copy()
                if len(daily_data) < 10:  # 至少需要10只股票
                    continue
                    
                # 按得分排序
                daily_data = daily_data.sort_values('predicted_score', ascending=False)
                total_stocks = len(daily_data)
                
                # 计算Top25%, Top10%, Top5%的表现
                for top_pct, top_name in [(0.25, 'top25'), (0.10, 'top10'), (0.05, 'top5')]:
                    top_n = max(1, int(total_stocks * top_pct))
                    top_stocks = daily_data.head(top_n)
                    
                    if len(top_stocks) > 0:
                        avg_return = top_stocks[target_period].mean()
                        win_rate = (top_stocks[target_period] > 0).mean()
                        
                        results_by_date.append({
                            'date': date,
                            'category': top_name,
                            'avg_return': avg_return,
                            'win_rate': win_rate,
                            'sample_size': len(top_stocks)
                        })
            
            # 汇总结果
            results_df = pd.DataFrame(results_by_date)
            summary = {}
            
            for category in ['top25', 'top10', 'top5']:
                cat_data = results_df[results_df['category'] == category]
                if not cat_data.empty:
                    summary[f'win_rate_{category}'] = cat_data['win_rate'].mean()
                    summary[f'avg_return_{category}'] = cat_data['avg_return'].mean()
                    summary[f'sharpe_{category}'] = cat_data['avg_return'].mean() / (cat_data['avg_return'].std() + 1e-10)
                    summary[f'sample_size_{category}'] = int(cat_data['sample_size'].sum())
            
            # 综合评分：相关性 + 胜率 + 收益率
            composite_score = (
                abs(correlation) * 0.4 +  # 相关性权重40%
                summary.get('win_rate_top25', 0.5) * 0.3 +  # 胜率权重30%  
                min(0.3, max(-0.3, summary.get('avg_return_top25', 0)) / 0.1) * 0.3  # 收益率权重30%
            )
            
            # 计算乘数范围，用于分析
            sample_regimes = data['market_regime'].unique()
            multipliers = [multiplier_func(regime) for regime in sample_regimes]
            multiplier_range = max(multipliers) / min(multipliers) if min(multipliers) > 0 else 1.0
            
            result = {
                'name': name,
                'multiplier_type': multiplier_type,
                'multiplier_params': multiplier_params.copy(),
                'quality_weights': quality_weights.copy(),
                'sample_size': len(data),
                'correlation': correlation,
                'p_value': p_value,
                'correlation_abs': abs(correlation),
                'composite_score': composite_score,
                'multiplier_range': multiplier_range,
                'min_multiplier': min(multipliers),
                'max_multiplier': max(multipliers),
                **summary
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"测试乘数组合失败: {e}")
            return {'composite_score': 0}
    
    def compare_multiplier_functions(self, data: pd.DataFrame, target_period: str = 'return_5d') -> None:
        """对比不同乘数函数的效果"""
        self.logger.info("\n" + "="*60)
        self.logger.info("🔬 对比不同市场环境乘数函数的效果")
        self.logger.info("="*60)
        
        # 固定质量因子权重为均匀分布
        fixed_quality_weights = {
            'technical': 0.20,
            'fundamental': 0.20,
            'performance': 0.20,
            'sentiment': 0.20,
            'risk_control': 0.20
        }
        
        # 测试不同的乘数函数
        test_functions = [
            {
                'name': '线性映射(保守)',
                'type': 'linear',
                'params': {'min_mult': 0.8, 'max_mult': 1.2}
            },
            {
                'name': '线性映射(激进)',
                'type': 'linear', 
                'params': {'min_mult': 0.3, 'max_mult': 2.0}
            },
            {
                'name': '指数映射(敏感)',
                'type': 'exponential',
                'params': {'base': 2.0, 'scale': 0.5, 'offset': 1.0}
            },
            {
                'name': 'Sigmoid映射(中点敏感)',
                'type': 'sigmoid',
                'params': {'steepness': 10.0, 'midpoint': 0.5, 'min_mult': 0.2, 'max_mult': 2.5}
            },
            {
                'name': '幂函数映射(凸)',
                'type': 'power',
                'params': {'power': 2.0, 'min_mult': 0.5, 'max_mult': 1.8}
            },
            {
                'name': '幂函数映射(凹)',
                'type': 'power',
                'params': {'power': 0.5, 'min_mult': 0.5, 'max_mult': 1.8}
            },
            {
                'name': '无乘数(对照组)',
                'type': 'linear',
                'params': {'min_mult': 1.0, 'max_mult': 1.0}
            }
        ]
        
        results = []
        
        for func_config in test_functions:
            result = self.test_multiplier_combination(
                data, fixed_quality_weights, 
                func_config['type'], func_config['params'],
                target_period, func_config['name']
            )
            
            results.append(result)
            
            self.logger.info(f"📊 {func_config['name']}: "
                           f"相关性={result.get('correlation', 0):.4f} "
                           f"胜率={result.get('win_rate_top25', 0):.1%} "
                           f"乘数范围={result.get('multiplier_range', 1):.2f}x "
                           f"综合评分={result.get('composite_score', 0):.4f}")
        
        # 找到最佳函数
        best_result = max(results, key=lambda x: x['composite_score'])
        self.logger.info(f"\n🏆 最佳乘数函数: {best_result['name']}")
        self.logger.info(f"   相关系数: {best_result.get('correlation', 0):.4f}")
        self.logger.info(f"   综合评分: {best_result.get('composite_score', 0):.4f}")
        self.logger.info(f"   乘数范围: [{best_result.get('min_multiplier', 1):.3f}, {best_result.get('max_multiplier', 1):.3f}]")
        
        return results
    
    def optimize_best_multiplier_function(self, data: pd.DataFrame, 
                                        best_multiplier_type: str,
                                        target_period: str = 'return_5d',
                                        max_iterations: int = 30) -> Dict:
        """优化最佳乘数函数的参数"""
        self.logger.info(f"\n🧬 开始优化最佳乘数函数: {best_multiplier_type}")
        
        def objective_function(x):
            """目标函数"""
            # 前5个参数：质量因子权重
            quality_weights_array = np.array(x[:5])
            quality_weights_array = quality_weights_array / quality_weights_array.sum()
            
            quality_weights = {
                'technical': quality_weights_array[0],
                'fundamental': quality_weights_array[1], 
                'performance': quality_weights_array[2],
                'sentiment': quality_weights_array[3],
                'risk_control': quality_weights_array[4]
            }
            
            # 后续参数：乘数函数参数
            if best_multiplier_type == "linear":
                multiplier_params = {
                    'min_mult': x[5],
                    'max_mult': x[6]
                }
                if x[5] >= x[6]:  # 确保min < max
                    return 10.0
            elif best_multiplier_type == "exponential":
                multiplier_params = {
                    'base': x[5],
                    'scale': x[6], 
                    'offset': x[7]
                }
            elif best_multiplier_type == "sigmoid":
                multiplier_params = {
                    'steepness': x[5],
                    'midpoint': x[6],
                    'min_mult': x[7],
                    'max_mult': x[8]
                }
                if x[7] >= x[8]:  # 确保min < max
                    return 10.0
            elif best_multiplier_type == "power":
                multiplier_params = {
                    'power': x[5],
                    'min_mult': x[6],
                    'max_mult': x[7]
                }
                if x[6] >= x[7]:  # 确保min < max
                    return 10.0
            else:
                return 10.0  # 无效类型
            
            result = self.test_multiplier_combination(
                data, quality_weights, best_multiplier_type, multiplier_params, 
                target_period, "优化中"
            )
            return -result.get('composite_score', 0)
        
        # 根据乘数类型定义搜索边界
        if best_multiplier_type == "linear":
            bounds = [
                # 5个质量因子权重
                (0.01, 0.98), (0.01, 0.98), (0.01, 0.98), (0.01, 0.98), (0.01, 0.98),
                # 线性乘数参数: min_mult, max_mult
                (0.1, 0.9), (1.1, 3.0)
            ]
        elif best_multiplier_type == "exponential":
            bounds = [
                # 5个质量因子权重
                (0.01, 0.98), (0.01, 0.98), (0.01, 0.98), (0.01, 0.98), (0.01, 0.98),
                # 指数乘数参数: base, scale, offset
                (1.1, 5.0), (0.1, 2.0), (0.5, 2.0)
            ]
        elif best_multiplier_type == "sigmoid":
            bounds = [
                # 5个质量因子权重
                (0.01, 0.98), (0.01, 0.98), (0.01, 0.98), (0.01, 0.98), (0.01, 0.98),
                # Sigmoid乘数参数: steepness, midpoint, min_mult, max_mult
                (1.0, 20.0), (0.1, 0.9), (0.1, 0.8), (1.2, 3.0)
            ]
        elif best_multiplier_type == "power":
            bounds = [
                # 5个质量因子权重
                (0.01, 0.98), (0.01, 0.98), (0.01, 0.98), (0.01, 0.98), (0.01, 0.98),
                # 幂函数乘数参数: power, min_mult, max_mult
                (0.1, 3.0), (0.2, 0.8), (1.2, 2.5)
            ]
        else:
            self.logger.error(f"未知的乘数类型: {best_multiplier_type}")
            return {}
        
        # 运行优化
        start_time = time.time()
        self.logger.info(f"🔍 开始{len(bounds)}维参数空间搜索...")
        
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=max_iterations,
            popsize=12,
            seed=42,
            disp=True
        )
        
        elapsed_time = time.time() - start_time
        
        # 提取最优参数
        optimal_x = result.x
        
        # 质量因子权重
        quality_weights_array = np.array(optimal_x[:5])
        quality_weights_array = quality_weights_array / quality_weights_array.sum()
        
        optimal_quality_weights = {
            'technical': quality_weights_array[0],
            'fundamental': quality_weights_array[1], 
            'performance': quality_weights_array[2],
            'sentiment': quality_weights_array[3],
            'risk_control': quality_weights_array[4]
        }
        
        # 乘数函数参数
        if best_multiplier_type == "linear":
            optimal_multiplier_params = {
                'min_mult': optimal_x[5],
                'max_mult': optimal_x[6]
            }
        elif best_multiplier_type == "exponential":
            optimal_multiplier_params = {
                'base': optimal_x[5],
                'scale': optimal_x[6],
                'offset': optimal_x[7]
            }
        elif best_multiplier_type == "sigmoid":
            optimal_multiplier_params = {
                'steepness': optimal_x[5],
                'midpoint': optimal_x[6],
                'min_mult': optimal_x[7],
                'max_mult': optimal_x[8]
            }
        elif best_multiplier_type == "power":
            optimal_multiplier_params = {
                'power': optimal_x[5],
                'min_mult': optimal_x[6],
                'max_mult': optimal_x[7]
            }
        
        # 评估最优结果
        best_result = self.test_multiplier_combination(
            data, optimal_quality_weights, best_multiplier_type, optimal_multiplier_params,
            target_period, f"{best_multiplier_type}最优解"
        )
        
        self.logger.info(f"✅ 参数优化完成: {elapsed_time:.1f} 秒")
        self.logger.info(f"🏆 最优综合评分: {best_result.get('composite_score', 0):.4f}")
        
        # 打印详细结果
        self.logger.info("\n" + "="*60)
        self.logger.info("🎯 最优质量因子权重:")
        for name, weight in optimal_quality_weights.items():
            self.logger.info(f"  {name:15}: {weight:.4f} ({weight*100:.1f}%)")
        
        self.logger.info(f"\n🎪 最优{best_multiplier_type}乘数参数:")
        for name, value in optimal_multiplier_params.items():
            self.logger.info(f"  {name:15}: {value:.4f}")
        
        self.logger.info(f"\n📊 预测效果:")
        self.logger.info(f"  相关系数: {best_result.get('correlation', 0):.4f}")
        self.logger.info(f"  Top25%胜率: {best_result.get('win_rate_top25', 0):.1%}")
        self.logger.info(f"  Top25%收益: {best_result.get('avg_return_top25', 0):.2%}")
        self.logger.info(f"  乘数范围: [{best_result.get('min_multiplier', 1):.3f}, {best_result.get('max_multiplier', 1):.3f}]")
        
        return best_result

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="敏感市场环境乘数优化")
    parser.add_argument("--target-period", default="return_5d", 
                       choices=['return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d'],
                       help="目标收益期间")
    parser.add_argument("--sample-limit", type=int, default=30000, help="数据样本限制")
    parser.add_argument("--max-iterations", type=int, default=25, help="最大迭代次数")
    parser.add_argument("--compare-only", action="store_true", help="仅对比不同乘数函数")
    
    args = parser.parse_args()
    
    # 创建优化器
    optimizer = SensitiveMarketMultiplierOptimizer()
    
    # 加载数据
    data = optimizer.load_cached_data(sample_limit=args.sample_limit)
    
    if data.empty:
        print("❌ 没有找到有效数据，请先运行数据生成步骤")
        return
    
    # 对比不同乘数函数效果
    comparison_results = optimizer.compare_multiplier_functions(data, args.target_period)
    
    if not args.compare_only:
        # 找到最佳乘数函数类型
        best_result = max(comparison_results, key=lambda x: x['composite_score'])
        best_type = best_result['multiplier_type']
        
        # 优化最佳乘数函数
        final_result = optimizer.optimize_best_multiplier_function(
            data, best_type, args.target_period, args.max_iterations
        )
    
    print(f"\n✅ 敏感市场环境乘数优化完成！")

if __name__ == "__main__":
    main()