#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v3.1评分系统权重优化器
使用网格搜索算法找到最优的权重分配

基于历史回测数据优化6个维度的权重：
- 技术指标、基本面、市场表现、情绪指标、风险控制、市场环境

目标：最大化预测准确性和夏普比率
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import sqlite3
from typing import Dict, List, Tuple, Optional

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scoring.v3.quantitative_scorer_v3_1 import QuantitativeScorerV31
from data_adapter.database_manager import DatabaseManager

class ScoringWeightOptimizer:
    """评分权重优化器"""
    
    def __init__(self, db_path: str = "stock_data.db"):
        self.db_path = db_path
        self.db_manager = DatabaseManager()
        
        # 权重搜索范围 (确保总权重=100%)
        self.weight_ranges = {
            'technical': [0.45, 0.50, 0.55, 0.60, 0.65, 0.70],      # 技术指标 45-70%
            'fundamental': [0.10, 0.12, 0.15, 0.18, 0.20, 0.25],   # 基本面 10-25%  
            'performance': [0.08, 0.10, 0.12, 0.15, 0.18, 0.20],   # 市场表现 8-20%
            'sentiment': [0.02, 0.03, 0.05, 0.07, 0.08, 0.10],     # 情绪指标 2-10%
            'risk_control': [0.02, 0.03, 0.05, 0.07, 0.08, 0.10],  # 风险控制 2-10%
            'market_regime': [0.03, 0.05, 0.07, 0.08, 0.10, 0.12]  # 市场环境 3-12%
        }
        
        # 回测参数
        self.lookback_days = [1, 3, 5, 10, 20]  # 预测天数
        self.min_stocks_per_day = 50  # 每天最少股票数
        
    def generate_weight_combinations(self) -> List[Dict[str, float]]:
        """生成所有权重组合，确保总权重=100%"""
        print("🔍 生成权重组合...")
        
        combinations = []
        ranges = self.weight_ranges
        
        # 生成所有可能的组合
        for tech in ranges['technical']:
            for fund in ranges['fundamental']:
                for perf in ranges['performance']:
                    for sent in ranges['sentiment']:
                        for risk in ranges['risk_control']:
                            # 计算剩余权重给market_regime
                            remaining = 1.0 - (tech + fund + perf + sent + risk)
                            
                            # 检查剩余权重是否在合理范围内
                            if ranges['market_regime'][0] <= remaining <= ranges['market_regime'][-1]:
                                combo = {
                                    'technical': tech,
                                    'fundamental': fund, 
                                    'performance': perf,
                                    'sentiment': sent,
                                    'risk_control': risk,
                                    'market_regime': remaining
                                }
                                # 验证总权重=1.0
                                if abs(sum(combo.values()) - 1.0) < 0.001:
                                    combinations.append(combo)
        
        print(f"📊 生成了 {len(combinations)} 个权重组合")
        return combinations
    
    def get_historical_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取历史股票数据用于回测"""
        print(f"📈 获取历史数据: {start_date} 到 {end_date}")
        
        query = """
        SELECT DISTINCT 
            s.code,
            dq.trade_date,
            dq.close,
            dq.price_change_pct,
            LAG(dq.close, 1) OVER (PARTITION BY s.code ORDER BY dq.trade_date) as prev_close,
            LAG(dq.close, 3) OVER (PARTITION BY s.code ORDER BY dq.trade_date) as close_3d_ago,
            LAG(dq.close, 5) OVER (PARTITION BY s.code ORDER BY dq.trade_date) as close_5d_ago,
            LAG(dq.close, 10) OVER (PARTITION BY s.code ORDER BY dq.trade_date) as close_10d_ago,
            LAG(dq.close, 20) OVER (PARTITION BY s.code ORDER BY dq.trade_date) as close_20d_ago
        FROM securities s
        INNER JOIN daily_quotes dq ON s.id = dq.security_id
        WHERE s.type = 'A股' 
            AND dq.trade_date >= ? 
            AND dq.trade_date <= ?
            AND dq.close > 0
        ORDER BY s.code, dq.trade_date
        """
        
        with self.db_manager.get_connection() as conn:
            df = pd.read_sql_query(query, conn, 
                                  params=[start_date.replace('-', ''), end_date.replace('-', '')])
        
        # 计算未来收益率
        for days in self.lookback_days:
            df[f'future_close_{days}d'] = df.groupby('code')['close'].shift(-days)
            df[f'future_return_{days}d'] = (df[f'future_close_{days}d'] - df['close']) / df['close']
        
        print(f"📊 获取了 {len(df)} 条历史记录，涉及 {df['code'].nunique()} 只股票")
        return df
    
    def calculate_scores_with_weights(self, stock_codes: List[str], date: str, weights: Dict[str, float]) -> pd.DataFrame:
        """使用指定权重计算股票评分"""
        scorer = QuantitativeScorerV31()
        
        # 临时修改权重配置
        original_weights = scorer.config['weights'].copy()
        
        # 更新权重（保持子权重结构不变，只调整大类权重）
        total_original = {}
        for category, sub_weights in original_weights.items():
            total_original[category] = sum(sub_weights.values())
        
        # 按比例调整子权重
        for category, new_weight in weights.items():
            if category in original_weights:
                scale_factor = new_weight / total_original[category]
                for sub_key in original_weights[category]:
                    scorer.config['weights'][category][sub_key] *= scale_factor
        
        results = []
        for code in stock_codes:
            try:
                result = scorer.calculate_stock_score(code, date)
                if 'error' not in result:
                    score_data = {
                        'code': code,
                        'date': date,
                        'total_score': result['total_score_100'],
                        **{f'{k}_score': v*100 for k, v in result['scores'].items()}
                    }
                    results.append(score_data)
            except Exception as e:
                continue
        
        # 恢复原始权重
        scorer.config['weights'] = original_weights
        
        return pd.DataFrame(results)
    
    def evaluate_weight_combination(self, weights: Dict[str, float], test_data: pd.DataFrame) -> Dict[str, float]:
        """评估权重组合的效果"""
        metrics = {}
        
        # 获取测试日期列表
        test_dates = sorted(test_data['trade_date'].unique())
        
        all_predictions = []
        
        for date in test_dates[:20]:  # 限制测试日期数量以提高速度
            # 获取当天的股票代码
            daily_stocks = test_data[test_data['trade_date'] == date]['code'].tolist()
            
            if len(daily_stocks) < self.min_stocks_per_day:
                continue
                
            # 计算评分
            date_str = f"{date[:4]}-{date[4:6]}-{date[6:]}"
            scores_df = self.calculate_scores_with_weights(daily_stocks[:100], date_str, weights)
            
            if len(scores_df) < 10:
                continue
                
            # 合并评分和收益数据
            daily_test = test_data[test_data['trade_date'] == date].copy()
            merged = scores_df.merge(daily_test, on='code', how='inner')
            
            if len(merged) < 10:
                continue
                
            # 计算各个时间窗口的相关性
            for days in self.lookback_days:
                if f'future_return_{days}d' in merged.columns:
                    valid_data = merged.dropna(subset=['total_score', f'future_return_{days}d'])
                    if len(valid_data) >= 10:
                        correlation = valid_data['total_score'].corr(valid_data[f'future_return_{days}d'])
                        if not np.isnan(correlation):
                            all_predictions.append({
                                'date': date,
                                'days': days,
                                'correlation': correlation,
                                'n_stocks': len(valid_data)
                            })
        
        if not all_predictions:
            return {'overall_score': -1, 'avg_correlation': 0, 'valid_dates': 0}
            
        pred_df = pd.DataFrame(all_predictions)
        
        # 计算综合评分
        avg_corr = pred_df['correlation'].mean()
        corr_std = pred_df['correlation'].std()
        valid_dates = len(pred_df['date'].unique())
        
        # 计算不同时间窗口的平均相关性
        for days in self.lookback_days:
            day_data = pred_df[pred_df['days'] == days]
            if len(day_data) > 0:
                metrics[f'correlation_{days}d'] = day_data['correlation'].mean()
            else:
                metrics[f'correlation_{days}d'] = 0
        
        # 综合评分：平均相关性 - 相关性方差（奖励稳定性）+ 有效日期奖励
        overall_score = avg_corr - corr_std * 0.5 + min(valid_dates / 20, 1) * 0.1
        
        metrics.update({
            'overall_score': overall_score,
            'avg_correlation': avg_corr,
            'correlation_std': corr_std,
            'valid_dates': valid_dates,
            'total_predictions': len(pred_df)
        })
        
        return metrics
    
    def optimize_weights(self, start_date: str = "2025-01-01", end_date: str = "2025-08-15") -> Dict:
        """执行权重优化"""
        print("🚀 开始权重优化...")
        print(f"📅 回测期间: {start_date} 到 {end_date}")
        
        # 获取历史数据
        test_data = self.get_historical_data(start_date, end_date)
        
        if len(test_data) < 1000:
            raise ValueError("历史数据不足，无法进行有效回测")
        
        # 生成权重组合
        weight_combinations = self.generate_weight_combinations()
        
        if len(weight_combinations) > 500:
            # 如果组合太多，随机采样
            np.random.seed(42)
            weight_combinations = np.random.choice(weight_combinations, 500, replace=False).tolist()
            print(f"📊 随机采样 {len(weight_combinations)} 个权重组合进行测试")
        
        # 并行评估权重组合
        print("⚡ 开始并行评估权重组合...")
        max_workers = min(multiprocessing.cpu_count(), 4)
        
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_weights = {
                executor.submit(self.evaluate_weight_combination, weights, test_data): weights 
                for weights in weight_combinations[:50]  # 限制数量以提高速度
            }
            
            completed = 0
            for future in as_completed(future_to_weights):
                weights = future_to_weights[future]
                completed += 1
                
                try:
                    metrics = future.result()
                    results.append({
                        'weights': weights,
                        'metrics': metrics
                    })
                    
                    if completed % 10 == 0:
                        print(f"✅ 已完成 {completed}/{len(future_to_weights)} 个权重组合评估")
                        
                except Exception as e:
                    print(f"❌ 权重组合评估失败: {e}")
        
        if not results:
            raise ValueError("没有成功评估的权重组合")
        
        # 找到最优权重
        results.sort(key=lambda x: x['metrics']['overall_score'], reverse=True)
        best_result = results[0]
        
        print("\n🏆 权重优化完成！")
        print("=" * 60)
        
        return {
            'best_weights': best_result['weights'],
            'best_metrics': best_result['metrics'],
            'all_results': results[:10],  # 保存前10个最佳结果
            'optimization_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'test_period': {'start': start_date, 'end': end_date},
            'total_combinations_tested': len(results)
        }
    
    def generate_optimized_config(self, optimization_result: Dict) -> Dict:
        """基于优化结果生成新的配置文件"""
        best_weights = optimization_result['best_weights']
        best_metrics = optimization_result['best_metrics']
        
        # 读取原始配置模板
        config_path = "scoring/v3/v3_1_optimized_config.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            original_config = json.load(f)
        
        # 创建优化后的配置
        optimized_config = original_config.copy()
        optimized_config['version'] = "v3.1-GridSearchOptimized"
        optimized_config['description'] = "基于网格搜索算法优化的v3.1评分配置"
        
        # 更新权重 - 保持原有子权重结构，按比例调整
        for category, new_weight in best_weights.items():
            if category in optimized_config['weights']:
                # 计算原始权重总和
                original_total = sum(optimized_config['weights'][category].values())
                scale_factor = new_weight / original_total
                
                # 按比例调整所有子权重
                for sub_key in optimized_config['weights'][category]:
                    if not sub_key.startswith('_'):  # 跳过注释
                        optimized_config['weights'][category][sub_key] *= scale_factor
        
        # 添加优化信息
        optimized_config['grid_search_optimization'] = {
            'optimization_date': optimization_result['optimization_date'],
            'test_period': optimization_result['test_period'],
            'combinations_tested': optimization_result['total_combinations_tested'],
            'best_metrics': best_metrics,
            'weight_changes': {
                category: {
                    'original': sum(original_config['weights'][category].values() 
                                  for k, v in original_config['weights'][category].items() 
                                  if not k.startswith('_')),
                    'optimized': weight,
                    'change': weight - sum(original_config['weights'][category].values() 
                                         for k, v in original_config['weights'][category].items() 
                                         if not k.startswith('_'))
                }
                for category, weight in best_weights.items()
            }
        }
        
        # 更新性能目标
        optimized_config['validation_metrics']['target_correlation'].update({
            f'{days}d': f"> {best_metrics.get(f'correlation_{days}d', 0.05):.3f}"
            for days in [1, 3, 5, 10, 20]
            if f'correlation_{days}d' in best_metrics
        })
        
        return optimized_config

def main():
    """主函数"""
    print("🚀 v3.1评分权重优化器启动")
    print("=" * 60)
    
    optimizer = ScoringWeightOptimizer()
    
    try:
        # 执行权重优化
        optimization_result = optimizer.optimize_weights()
        
        # 显示最优结果
        best_weights = optimization_result['best_weights']
        best_metrics = optimization_result['best_metrics']
        
        print(f"📊 最优权重配置:")
        print("-" * 40)
        for category, weight in best_weights.items():
            print(f"{category:15s}: {weight:6.1%}")
        
        print(f"\n📈 最优配置性能指标:")
        print("-" * 40)
        print(f"综合评分: {best_metrics['overall_score']:.4f}")
        print(f"平均相关性: {best_metrics['avg_correlation']:.4f}")
        print(f"相关性标准差: {best_metrics['correlation_std']:.4f}")
        print(f"有效测试日期: {best_metrics['valid_dates']}")
        
        for days in [1, 3, 5, 10, 20]:
            key = f'correlation_{days}d'
            if key in best_metrics:
                print(f"{days}天预测相关性: {best_metrics[key]:.4f}")
        
        # 生成优化后的配置文件
        optimized_config = optimizer.generate_optimized_config(optimization_result)
        
        # 保存结果
        results_dir = "scoring/v3/optimization_results"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存优化结果
        result_file = f"{results_dir}/weight_optimization_{timestamp}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(optimization_result, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存优化后的配置
        config_file = f"{results_dir}/v3_1_grid_optimized_config_{timestamp}.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(optimized_config, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 优化结果已保存:")
        print(f"优化数据: {result_file}")
        print(f"优化配置: {config_file}")
        
        print(f"\n🎯 建议:")
        print("1. 备份当前v3.1配置文件")
        print("2. 使用优化后的配置替换原配置")
        print("3. 进行小规模测试验证效果")
        print("4. 如果效果理想，应用到生产环境")
        
    except Exception as e:
        print(f"❌ 优化过程出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()