#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版市场环境乘数优化器

优化目标扩展为：
1. 质量因子权重 (5个参数): w1, w2, w3, w4, w5
2. 市场环境乘数参数 (2个参数): min_multiplier, max_multiplier

总计7个参数的联合优化
"""

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import argparse
import logging
from datetime import datetime
import time
import warnings

warnings.filterwarnings('ignore')

class EnhancedMarketRegimeOptimizer:
    """增强版市场环境乘数优化器"""
    
    def __init__(self, db_path: str = "weight_optimization_cache.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
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
        ORDER BY date, code
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
                
                # 显示市场环境得分分布
                regime_stats = data['market_regime'].describe()
                self.logger.info(f"📊 市场环境得分分布: 最小={regime_stats['min']:.3f}, 最大={regime_stats['max']:.3f}, 平均={regime_stats['mean']:.3f}")
            else:
                self.logger.warning("⚠️ 没有找到有效数据")
                
            return data
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            return pd.DataFrame()
    
    def calculate_market_regime_multiplier(self, market_regime_score: float, 
                                         min_multiplier: float, max_multiplier: float) -> float:
        """
        计算市场环境乘数 - 参数化版本
        
        Args:
            market_regime_score: 市场环境得分 (0-1)
            min_multiplier: 最小乘数 (熊市时)
            max_multiplier: 最大乘数 (牛市时)
        
        Returns:
            乘数因子
        """
        # 线性映射: market_regime_score从0-1映射到min_multiplier到max_multiplier
        multiplier = min_multiplier + market_regime_score * (max_multiplier - min_multiplier)
        return multiplier
    
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
    
    def calculate_final_score(self, params: Dict, row: pd.Series) -> float:
        """
        计算最终得分
        final_score = market_regime_multiplier × stock_quality_score
        
        Args:
            params: 包含quality_weights, min_multiplier, max_multiplier的参数字典
        """
        # 计算市场环境乘数
        market_multiplier = self.calculate_market_regime_multiplier(
            row['market_regime'], 
            params['min_multiplier'], 
            params['max_multiplier']
        )
        
        # 计算个股质量得分
        quality_score = self.calculate_stock_quality_score(
            params['quality_weights'],
            row['technical'],
            row['fundamental'], 
            row['performance'],
            row['sentiment'],
            row['risk_control']
        )
        
        # 最终得分 = 市场环境乘数 × 个股质量得分
        final_score = market_multiplier * quality_score
        
        return final_score
    
    def test_parameter_combination(self, data: pd.DataFrame, params: Dict, 
                                 target_period: str = 'return_5d', name: str = "测试") -> Dict:
        """测试参数组合效果"""
        try:
            # 计算每只股票的最终得分
            data['predicted_score'] = data.apply(
                lambda row: self.calculate_final_score(params, row), axis=1
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
            
            result = {
                'name': name,
                'params': params.copy(),
                'sample_size': len(data),
                'correlation': correlation,
                'p_value': p_value,
                'correlation_abs': abs(correlation),
                'composite_score': composite_score,
                **summary
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"测试参数组合失败: {e}")
            return {'composite_score': 0}
    
    def joint_optimization(self, data: pd.DataFrame, target_period: str = 'return_5d',
                          max_iterations: int = 50) -> Dict:
        """联合优化：5个质量因子权重 + 2个市场环境乘数参数"""
        self.logger.info(f"🧬 开始联合优化 (目标: {target_period})")
        self.logger.info("📊 优化参数: 5个质量因子权重 + 2个市场环境乘数参数")
        
        def objective_function(x):
            """
            目标函数
            x = [w1, w2, w3, w4, w5, min_multiplier, max_multiplier]
            """
            # 前5个参数：质量因子权重
            quality_weights_array = np.array(x[:5])
            quality_weights_array = quality_weights_array / quality_weights_array.sum()  # 归一化
            
            quality_weights = {
                'technical': quality_weights_array[0],
                'fundamental': quality_weights_array[1], 
                'performance': quality_weights_array[2],
                'sentiment': quality_weights_array[3],
                'risk_control': quality_weights_array[4]
            }
            
            # 后2个参数：市场环境乘数范围
            min_multiplier = x[5]
            max_multiplier = x[6]
            
            # 确保min < max
            if min_multiplier >= max_multiplier:
                return 10.0  # 惩罚无效参数
            
            params = {
                'quality_weights': quality_weights,
                'min_multiplier': min_multiplier,
                'max_multiplier': max_multiplier
            }
            
            result = self.test_parameter_combination(data, params, target_period, "优化中")
            return -result.get('composite_score', 0)  # 最小化负数 = 最大化正数
        
        # 定义搜索边界
        bounds = [
            # 5个质量因子权重: 每个权重在0.01-0.98之间
            (0.01, 0.98), (0.01, 0.98), (0.01, 0.98), (0.01, 0.98), (0.01, 0.98),
            # 最小乘数: 0.1-0.9 (熊市时的压缩程度)
            (0.1, 0.9),
            # 最大乘数: 1.1-3.0 (牛市时的放大程度)
            (1.1, 3.0)
        ]
        
        # 运行差分进化算法
        start_time = time.time()
        self.logger.info("🔍 开始7维参数空间搜索...")
        
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=max_iterations,
            popsize=15,
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
        
        # 市场环境乘数参数
        optimal_min_multiplier = optimal_x[5]
        optimal_max_multiplier = optimal_x[6]
        
        optimal_params = {
            'quality_weights': optimal_quality_weights,
            'min_multiplier': optimal_min_multiplier,
            'max_multiplier': optimal_max_multiplier
        }
        
        # 评估最优结果
        best_result = self.test_parameter_combination(data, optimal_params, target_period, "联合优化最优解")
        
        self.logger.info(f"✅ 联合优化完成: {elapsed_time:.1f} 秒")
        self.logger.info(f"🏆 最优综合评分: {best_result.get('composite_score', 0):.4f}")
        
        # 打印详细结果
        self.logger.info("\n" + "="*60)
        self.logger.info("🎯 最优质量因子权重:")
        for name, weight in optimal_quality_weights.items():
            self.logger.info(f"  {name:15}: {weight:.4f} ({weight*100:.1f}%)")
        
        self.logger.info(f"\n🎪 最优市场环境乘数参数:")
        self.logger.info(f"  最小乘数(熊市): {optimal_min_multiplier:.3f}")
        self.logger.info(f"  最大乘数(牛市): {optimal_max_multiplier:.3f}")
        self.logger.info(f"  乘数范围: {optimal_max_multiplier/optimal_min_multiplier:.2f}x")
        
        self.logger.info(f"\n📊 预测效果:")
        self.logger.info(f"  相关系数: {best_result.get('correlation', 0):.4f}")
        self.logger.info(f"  Top25%胜率: {best_result.get('win_rate_top25', 0):.1%}")
        self.logger.info(f"  Top25%平均收益: {best_result.get('avg_return_top25', 0):.2%}")
        
        return best_result
    
    def test_different_multiplier_ranges(self, data: pd.DataFrame, target_period: str = 'return_5d') -> None:
        """测试不同乘数范围的效果"""
        self.logger.info("\n" + "="*60)
        self.logger.info("🔬 测试不同市场环境乘数范围的效果")
        self.logger.info("="*60)
        
        # 固定质量因子权重为均匀分布
        fixed_quality_weights = {
            'technical': 0.20,
            'fundamental': 0.20,
            'performance': 0.20,
            'sentiment': 0.20,
            'risk_control': 0.20
        }
        
        # 测试不同的乘数范围
        test_ranges = [
            (0.5, 1.5, "保守范围"),    # 1.5-0.5 = 3x差异
            (0.3, 2.0, "中等范围"),    # 2.0-0.3 = 6.7x差异
            (0.1, 3.0, "激进范围"),    # 3.0-0.1 = 30x差异
            (0.8, 1.2, "极保守范围"),  # 1.2-0.8 = 1.5x差异
        ]
        
        results = []
        
        for min_mult, max_mult, desc in test_ranges:
            params = {
                'quality_weights': fixed_quality_weights,
                'min_multiplier': min_mult,
                'max_multiplier': max_mult
            }
            
            result = self.test_parameter_combination(data, params, target_period, desc)
            results.append({
                'description': desc,
                'min_multiplier': min_mult,
                'max_multiplier': max_mult,
                'range_ratio': max_mult / min_mult,
                'correlation': result.get('correlation', 0),
                'composite_score': result.get('composite_score', 0)
            })
            
            self.logger.info(f"📊 {desc}: 乘数范围[{min_mult:.1f}, {max_mult:.1f}] "
                           f"相关性={result.get('correlation', 0):.4f} "
                           f"综合评分={result.get('composite_score', 0):.4f}")
        
        # 找到最佳范围
        best_result = max(results, key=lambda x: x['composite_score'])
        self.logger.info(f"\n🏆 最佳乘数范围: {best_result['description']}")
        self.logger.info(f"   范围: [{best_result['min_multiplier']:.1f}, {best_result['max_multiplier']:.1f}]")
        self.logger.info(f"   倍数差异: {best_result['range_ratio']:.1f}x")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="增强版市场环境乘数优化")
    parser.add_argument("--target-period", default="return_5d", 
                       choices=['return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d'],
                       help="目标收益期间")
    parser.add_argument("--sample-limit", type=int, default=20000, help="数据样本限制")
    parser.add_argument("--max-iterations", type=int, default=30, help="最大迭代次数")
    parser.add_argument("--test-ranges", action="store_true", help="测试不同乘数范围效果")
    
    args = parser.parse_args()
    
    # 创建优化器
    optimizer = EnhancedMarketRegimeOptimizer()
    
    # 加载数据
    data = optimizer.load_cached_data(sample_limit=args.sample_limit)
    
    if data.empty:
        print("❌ 没有找到有效数据，请先运行数据生成步骤")
        return
    
    # 测试不同乘数范围效果
    if args.test_ranges:
        optimizer.test_different_multiplier_ranges(data, args.target_period)
    
    # 执行联合优化
    result = optimizer.joint_optimization(data, args.target_period, args.max_iterations)
    
    print(f"\n✅ 增强版市场环境乘数优化完成！")

if __name__ == "__main__":
    main()