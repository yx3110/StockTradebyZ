#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场环境乘数权重优化器

实现新的评分公式：
final_score = market_regime_multiplier × (technical×w1 + fundamental×w2 + performance×w3 + sentiment×w4 + risk_control×w5)

市场环境作为高级别乘数因子，其他5个因子作为个股质量评分
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

class MarketRegimeMultiplierOptimizer:
    """市场环境乘数权重优化器"""
    
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
            else:
                self.logger.warning("⚠️ 没有找到有效数据")
                
            return data
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            return pd.DataFrame()
    
    def calculate_market_regime_multiplier(self, market_regime_score: float) -> float:
        """
        将市场环境得分转换为乘数因子
        
        设计思路：
        - 市场环境好时(高分): 乘数 > 1，放大所有股票得分
        - 市场环境差时(低分): 乘数 < 1，压缩所有股票得分
        - 中性市场: 乘数 ≈ 1
        """
        # 将0-1的得分转换为0.5-1.5的乘数
        # 0.0 -> 0.5 (熊市，压缩50%)
        # 0.5 -> 1.0 (中性市场)
        # 1.0 -> 1.5 (牛市，放大50%)
        multiplier = 0.5 + market_regime_score
        return multiplier
    
    def calculate_stock_quality_score(self, weights: Dict[str, float], 
                                    technical: float, fundamental: float, 
                                    performance: float, sentiment: float, 
                                    risk_control: float) -> float:
        """计算个股质量得分（5个因子加权）"""
        quality_score = (
            technical * weights['technical'] +
            fundamental * weights['fundamental'] + 
            performance * weights['performance'] +
            sentiment * weights['sentiment'] +
            risk_control * weights['risk_control']
        )
        return quality_score
    
    def calculate_final_score(self, weights: Dict[str, float], row: pd.Series) -> float:
        """
        计算最终得分
        final_score = market_regime_multiplier × stock_quality_score
        """
        # 计算市场环境乘数
        market_multiplier = self.calculate_market_regime_multiplier(row['market_regime'])
        
        # 计算个股质量得分
        quality_score = self.calculate_stock_quality_score(
            weights,
            row['technical'],
            row['fundamental'], 
            row['performance'],
            row['sentiment'],
            row['risk_control']
        )
        
        # 最终得分 = 市场环境乘数 × 个股质量得分
        final_score = market_multiplier * quality_score
        
        return final_score
    
    def test_weight_combination(self, data: pd.DataFrame, weights: Dict[str, float], 
                              target_period: str = 'return_5d', name: str = "测试") -> Dict:
        """测试权重组合效果"""
        try:
            # 计算每只股票的最终得分
            data['predicted_score'] = data.apply(
                lambda row: self.calculate_final_score(weights, row), axis=1
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
                'optimal_weights': weights.copy(),
                'total_weight': sum(weights.values()),
                'sample_size': len(data),
                'correlation': correlation,
                'p_value': p_value,
                'correlation_abs': abs(correlation),
                'composite_score': composite_score,
                **summary
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"测试权重组合失败: {e}")
            return {'composite_score': 0}
    
    def evolutionary_optimization(self, data: pd.DataFrame, target_period: str = 'return_5d',
                                max_iterations: int = 50) -> Dict:
        """进化算法优化权重（只优化5个质量因子）"""
        self.logger.info(f"🧬 开始进化算法优化 (目标: {target_period})")
        self.logger.info("📊 新公式: final_score = market_regime_multiplier × (quality_factors_weighted_sum)")
        
        def objective_function(x):
            """目标函数"""
            # 归一化权重（确保和为1）
            weights_array = np.array(x)
            weights_array = weights_array / weights_array.sum()
            
            weights = {
                'technical': weights_array[0],
                'fundamental': weights_array[1], 
                'performance': weights_array[2],
                'sentiment': weights_array[3],
                'risk_control': weights_array[4]
            }
            
            result = self.test_weight_combination(data, weights, target_period, "优化中")
            return -result.get('composite_score', 0)  # 最小化负数 = 最大化正数
        
        # 定义搜索边界（每个权重在0.01-0.98之间）
        bounds = [(0.01, 0.98) for _ in range(5)]
        
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
        optimal_weights_array = result.x
        optimal_weights_array = optimal_weights_array / optimal_weights_array.sum()  # 标准化
        
        optimal_weights = {
            'technical': optimal_weights_array[0],
            'fundamental': optimal_weights_array[1], 
            'performance': optimal_weights_array[2],
            'sentiment': optimal_weights_array[3],
            'risk_control': optimal_weights_array[4]
        }
        
        # 评估最优结果
        best_result = self.test_weight_combination(data, optimal_weights, target_period, "进化算法最优解")
        
        self.logger.info(f"✅ 进化算法完成: {elapsed_time:.1f} 秒")
        self.logger.info(f"🏆 最优综合评分: {best_result.get('composite_score', 0):.4f}")
        
        # 打印详细结果
        self.logger.info("\n" + "="*60)
        self.logger.info("🎯 最优权重分配 (质量因子):")
        for name, weight in optimal_weights.items():
            self.logger.info(f"  {name:15}: {weight:.4f} ({weight*100:.1f}%)")
        
        self.logger.info(f"\n📊 预测效果:")
        self.logger.info(f"  相关系数: {best_result.get('correlation', 0):.4f}")
        self.logger.info(f"  Top25%胜率: {best_result.get('win_rate_top25', 0):.1%}")
        self.logger.info(f"  Top25%平均收益: {best_result.get('avg_return_top25', 0):.2%}")
        
        return best_result
    
    def compare_old_vs_new_formula(self, data: pd.DataFrame, target_period: str = 'return_5d') -> None:
        """对比旧公式vs新公式的效果"""
        self.logger.info("\n" + "="*60)
        self.logger.info("🔬 对比分析: 旧公式 vs 新公式")
        self.logger.info("="*60)
        
        # 1. 旧公式：6个因子平均权重
        old_weights_6_factors = {
            'technical': 1/6,
            'fundamental': 1/6,
            'performance': 1/6,
            'sentiment': 1/6,
            'risk_control': 1/6,
            'market_regime': 1/6  # 作为加法项
        }
        
        # 计算旧公式得分
        data['old_score'] = (
            data['technical'] * old_weights_6_factors['technical'] +
            data['fundamental'] * old_weights_6_factors['fundamental'] + 
            data['performance'] * old_weights_6_factors['performance'] +
            data['sentiment'] * old_weights_6_factors['sentiment'] +
            data['risk_control'] * old_weights_6_factors['risk_control'] +
            data['market_regime'] * old_weights_6_factors['market_regime']
        )
        
        # 2. 新公式：优化后的5个质量因子 + 市场环境乘数
        new_weights_5_factors = {
            'technical': 0.20,
            'fundamental': 0.20,
            'performance': 0.20,
            'sentiment': 0.20,
            'risk_control': 0.20
        }
        
        data['new_score'] = data.apply(
            lambda row: self.calculate_final_score(new_weights_5_factors, row), axis=1
        )
        
        # 对比结果
        old_corr = data['old_score'].corr(data[target_period])
        new_corr = data['new_score'].corr(data[target_period])
        
        self.logger.info(f"📊 旧公式相关性: {old_corr:.4f}")
        self.logger.info(f"📊 新公式相关性: {new_corr:.4f}")
        self.logger.info(f"🚀 相关性提升: {((new_corr/old_corr - 1) * 100):+.1f}%")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="市场环境乘数权重优化")
    parser.add_argument("--target-period", default="return_5d", 
                       choices=['return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d'],
                       help="目标收益期间")
    parser.add_argument("--sample-limit", type=int, default=20000, help="数据样本限制")
    parser.add_argument("--max-iterations", type=int, default=50, help="最大迭代次数")
    
    args = parser.parse_args()
    
    # 创建优化器
    optimizer = MarketRegimeMultiplierOptimizer()
    
    # 加载数据
    data = optimizer.load_cached_data(sample_limit=args.sample_limit)
    
    if data.empty:
        print("❌ 没有找到有效数据，请先运行数据生成步骤")
        return
    
    # 对比分析
    optimizer.compare_old_vs_new_formula(data, args.target_period)
    
    # 执行优化
    result = optimizer.evolutionary_optimization(data, args.target_period, args.max_iterations)
    
    print(f"\n✅ 市场环境乘数权重优化完成！")

if __name__ == "__main__":
    main()