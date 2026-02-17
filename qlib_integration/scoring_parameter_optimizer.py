#!/usr/bin/env python3
"""
评分参数优化器 - 针对评分函数参数进行数据驱动优化
使用贝叶斯优化和交叉验证来找到最优的评分函数参数
"""

import os
import sys
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
import json
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr
from hyperopt import hp, fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

class ScoringParameterOptimizer:
    """评分函数参数优化器"""
    
    def __init__(self, db_path: str = None, cv_folds: int = 3):
        self.db_path = db_path or os.path.join(project_root, 'data_adapter/stock_data.db')
        self.cv_folds = cv_folds
        
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # 扩展参数搜索空间定义 - 包含知行指标参数
        self.param_space = {
            # RSI参数
            'rsi_optimal_min': hp.uniform('rsi_optimal_min', 30, 55),
            'rsi_optimal_max': hp.uniform('rsi_optimal_max', 45, 70),
            'rsi_good_range': hp.uniform('rsi_good_range', 5, 20),
            
            # KDJ_K参数
            'kdj_k_optimal_min': hp.uniform('kdj_k_optimal_min', 25, 55),
            'kdj_k_optimal_max': hp.uniform('kdj_k_optimal_max', 45, 75),
            'kdj_k_good_range': hp.uniform('kdj_k_good_range', 5, 25),
            
            # KDJ_D参数
            'kdj_d_optimal_min': hp.uniform('kdj_d_optimal_min', 30, 55),
            'kdj_d_optimal_max': hp.uniform('kdj_d_optimal_max', 45, 70),
            'kdj_d_good_range': hp.uniform('kdj_d_good_range', 5, 20),
            
            # BBI参数
            'bbi_optimal_min': hp.uniform('bbi_optimal_min', 0.98, 1.05),
            'bbi_optimal_max': hp.uniform('bbi_optimal_max', 1.03, 1.15),
            'bbi_good_range': hp.uniform('bbi_good_range', 0.02, 0.10),
            
            # 知行趋势参数 - 新增
            'zhixing_trend_optimal_ratio_min': hp.uniform('zhixing_trend_optimal_ratio_min', 0.95, 1.02),
            'zhixing_trend_optimal_ratio_max': hp.uniform('zhixing_trend_optimal_ratio_max', 1.00, 1.08),
            'zhixing_trend_good_range': hp.uniform('zhixing_trend_good_range', 0.02, 0.10),
            
            # 知行多均参数 - 新增
            'zhixing_multiavg_optimal_ratio_min': hp.uniform('zhixing_multiavg_optimal_ratio_min', 0.90, 1.05),
            'zhixing_multiavg_optimal_ratio_max': hp.uniform('zhixing_multiavg_optimal_ratio_max', 0.98, 1.12),
            'zhixing_multiavg_good_range': hp.uniform('zhixing_multiavg_good_range', 0.03, 0.15),
        }
        
        # 缓存数据
        self.data_cache = {}
        
    def load_historical_data(self, start_date: str, end_date: str, limit: int = 2000) -> pd.DataFrame:
        """加载历史数据用于优化"""
        cache_key = f"{start_date}_{end_date}_{limit}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        self.logger.info(f"加载历史数据: {start_date} 到 {end_date}, 限制 {limit} 只股票")
        
        with sqlite3.connect(self.db_path) as conn:
            # 获取活跃股票列表
            active_stocks_query = """
            SELECT DISTINCT s.code 
            FROM securities s 
            JOIN daily_quotes dq ON s.id = dq.security_id
            WHERE s.type = 'A股' 
                AND dq.trade_date BETWEEN ? AND ?
                AND dq.volume > 0
            ORDER BY RANDOM()
            LIMIT ?
            """
            active_stocks = pd.read_sql_query(active_stocks_query, conn, params=(start_date, end_date, limit))
            stock_codes = active_stocks['code'].tolist()
            
            self.logger.info(f"找到 {len(stock_codes)} 只活跃股票")
            
            # 构建数据查询
            placeholders = ','.join(['?' for _ in stock_codes])
            query = f"""
            SELECT 
                s.code,
                dq.trade_date,
                dq.close,
                dq.price_change_pct as next_1d_return,
                LAG(dq.price_change_pct, -1) OVER (PARTITION BY s.code ORDER BY dq.trade_date) as next_1d_return_actual,
                LAG(dq.price_change_pct, -2) OVER (PARTITION BY s.code ORDER BY dq.trade_date) as next_2d_return,
                LAG(dq.price_change_pct, -4) OVER (PARTITION BY s.code ORDER BY dq.trade_date) as next_5d_return,
                ti.rsi6,
                ti.kdj_k,
                ti.kdj_d,
                ti.bbi,
                db.pe_ttm,
                db.pb,
                db.total_mv as market_cap,
                dq.volume,
                (dq.close - LAG(dq.close, 4) OVER (PARTITION BY s.code ORDER BY dq.trade_date)) / 
                 LAG(dq.close, 4) OVER (PARTITION BY s.code ORDER BY dq.trade_date) * 100 as price_momentum_5d
            FROM securities s
            JOIN daily_quotes dq ON s.id = dq.security_id
            LEFT JOIN technical_indicators ti ON s.id = ti.security_id AND dq.trade_date = ti.trade_date
            LEFT JOIN daily_basic db ON s.id = db.security_id AND dq.trade_date = db.trade_date
            WHERE s.code IN ({placeholders})
                AND dq.trade_date BETWEEN ? AND ?
                AND ti.rsi6 IS NOT NULL
                AND ti.kdj_k IS NOT NULL
                AND ti.kdj_d IS NOT NULL
                AND ti.bbi IS NOT NULL
            ORDER BY s.code, dq.trade_date
            """
            
            df = pd.read_sql_query(query, conn, params=stock_codes + [start_date, end_date])
            
            # 计算price_to_bbi比率
            df['price_to_bbi'] = df['close'] / df['bbi']
            
            # 过滤掉缺失未来收益率的数据
            df = df.dropna(subset=['next_1d_return_actual'])
            
            self.logger.info(f"加载完成，共 {len(df)} 条记录")
            
            # 缓存数据
            self.data_cache[cache_key] = df
            return df
    
    def calculate_score_with_params(self, data: pd.Series, params: Dict) -> float:
        """使用给定参数计算综合评分"""
        
        # RSI评分
        rsi = data['rsi6']
        rsi_opt_min = params['rsi_optimal_min'] 
        rsi_opt_max = params['rsi_optimal_max']
        rsi_good_range = params['rsi_good_range']
        
        if rsi_opt_min <= rsi <= rsi_opt_max:
            rsi_score = 100.0
        elif abs(rsi - (rsi_opt_min + rsi_opt_max)/2) <= rsi_good_range:
            distance = abs(rsi - (rsi_opt_min + rsi_opt_max)/2)
            rsi_score = 85.0 + (1 - distance/rsi_good_range) * 15
        else:
            distance = min(abs(rsi - rsi_opt_min), abs(rsi - rsi_opt_max))
            rsi_score = max(30.0, 85.0 - distance * 2)
        
        # KDJ_K评分
        kdj_k = data['kdj_k']
        kdj_k_opt_min = params['kdj_k_optimal_min']
        kdj_k_opt_max = params['kdj_k_optimal_max'] 
        kdj_k_good_range = params['kdj_k_good_range']
        
        if kdj_k_opt_min <= kdj_k <= kdj_k_opt_max:
            kdj_k_score = 100.0
        elif abs(kdj_k - (kdj_k_opt_min + kdj_k_opt_max)/2) <= kdj_k_good_range:
            distance = abs(kdj_k - (kdj_k_opt_min + kdj_k_opt_max)/2)
            kdj_k_score = 80.0 + (1 - distance/kdj_k_good_range) * 20
        else:
            distance = min(abs(kdj_k - kdj_k_opt_min), abs(kdj_k - kdj_k_opt_max))
            kdj_k_score = max(25.0, 80.0 - distance * 1.5)
        
        # KDJ_D评分
        kdj_d = data['kdj_d']
        kdj_d_opt_min = params['kdj_d_optimal_min']
        kdj_d_opt_max = params['kdj_d_optimal_max']
        kdj_d_good_range = params['kdj_d_good_range']
        
        if kdj_d_opt_min <= kdj_d <= kdj_d_opt_max:
            kdj_d_score = 100.0
        elif abs(kdj_d - (kdj_d_opt_min + kdj_d_opt_max)/2) <= kdj_d_good_range:
            distance = abs(kdj_d - (kdj_d_opt_min + kdj_d_opt_max)/2)
            kdj_d_score = 80.0 + (1 - distance/kdj_d_good_range) * 20
        else:
            distance = min(abs(kdj_d - kdj_d_opt_min), abs(kdj_d - kdj_d_opt_max))
            kdj_d_score = max(25.0, 80.0 - distance * 1.5)
        
        # BBI评分
        price_to_bbi = data['price_to_bbi']
        if pd.isna(price_to_bbi) or price_to_bbi <= 0:
            bbi_score = 50.0
        else:
            bbi_opt_min = params['bbi_optimal_min']
            bbi_opt_max = params['bbi_optimal_max']
            bbi_good_range = params['bbi_good_range']
            
            if bbi_opt_min <= price_to_bbi <= bbi_opt_max:
                bbi_score = 100.0
            elif abs(price_to_bbi - (bbi_opt_min + bbi_opt_max)/2) <= bbi_good_range:
                distance = abs(price_to_bbi - (bbi_opt_min + bbi_opt_max)/2)
                bbi_score = 80.0 + (1 - distance/bbi_good_range) * 20
            else:
                distance = min(abs(price_to_bbi - bbi_opt_min), abs(price_to_bbi - bbi_opt_max))
                bbi_score = max(40.0, 80.0 - distance * 25)
        
        # 综合评分 (等权重)
        total_score = (rsi_score + kdj_k_score + kdj_d_score + bbi_score) / 4
        return total_score
    
    def objective_function(self, params: Dict, data: pd.DataFrame) -> float:
        """目标函数：最大化IC（信息系数）"""
        try:
            # 计算所有股票的评分
            scores = []
            returns = []
            
            for _, row in data.iterrows():
                score = self.calculate_score_with_params(row, params)
                scores.append(score)
                returns.append(row['next_1d_return_actual'])
            
            scores = np.array(scores)
            returns = np.array(returns)
            
            # 过滤掉nan值
            valid_mask = ~(np.isnan(scores) | np.isnan(returns))
            scores = scores[valid_mask]
            returns = returns[valid_mask]
            
            if len(scores) < 50:  # 样本数太少
                return 1.0  # 返回高损失值
            
            # 计算IC (Spearman相关系数)
            ic, p_value = spearmanr(scores, returns)
            
            if np.isnan(ic):
                return 1.0
                
            # 目标是最大化IC，所以返回负IC作为损失
            loss = -ic
            
            # 添加参数合理性约束
            penalty = 0
            if params['rsi_optimal_min'] >= params['rsi_optimal_max']:
                penalty += 10
            if params['kdj_k_optimal_min'] >= params['kdj_k_optimal_max']:
                penalty += 10
            if params['kdj_d_optimal_min'] >= params['kdj_d_optimal_max']:
                penalty += 10
            if params['bbi_optimal_min'] >= params['bbi_optimal_max']:
                penalty += 10
            
            return loss + penalty
            
        except Exception as e:
            self.logger.error(f"目标函数计算错误: {e}")
            return 1.0
    
    def cross_validation_optimize(self, data: pd.DataFrame, max_evals: int = 100) -> Tuple[Dict, Dict]:
        """使用交叉验证进行参数优化"""
        
        self.logger.info(f"开始交叉验证优化，最大评估次数: {max_evals}")
        
        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        # 按日期排序
        data = data.sort_values(['code', 'trade_date']).reset_index(drop=True)
        
        def cv_objective(params):
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(data)):
                train_data = data.iloc[train_idx]
                val_data = data.iloc[val_idx]
                
                # 在验证集上评估
                loss = self.objective_function(params, val_data)
                cv_scores.append(loss)
            
            avg_loss = np.mean(cv_scores)
            
            # 记录结果
            ic = -avg_loss if avg_loss < 1.0 else 0
            self.logger.info(f"参数评估完成 - 平均IC: {ic:.4f}")
            
            return {'loss': avg_loss, 'status': STATUS_OK}
        
        # 贝叶斯优化
        trials = Trials()
        best_params = fmin(
            fn=cv_objective,
            space=self.param_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            verbose=True
        )
        
        # 获取最佳结果
        best_trial = trials.best_trial
        best_ic = -best_trial['result']['loss'] if best_trial['result']['loss'] < 1.0 else 0
        
        self.logger.info(f"优化完成！最佳IC: {best_ic:.4f}")
        
        # 整理结果
        optimization_results = {
            'best_params': best_params,
            'best_ic': best_ic,
            'trials_count': len(trials.trials),
            'optimization_history': [(trial['result']['loss'], trial['misc']['vals']) for trial in trials.trials]
        }
        
        return best_params, optimization_results
    
    def run_optimization(self, 
                        start_date: str = "2024-01-01",
                        end_date: str = "2025-09-01", 
                        max_evals: int = 50,
                        sample_size: int = 1500) -> Dict:
        """运行完整的参数优化流程"""
        
        self.logger.info("="*60)
        self.logger.info("开始评分参数优化")
        self.logger.info(f"优化期间: {start_date} 到 {end_date}")
        self.logger.info(f"样本大小: {sample_size}")
        self.logger.info(f"最大评估次数: {max_evals}")
        self.logger.info("="*60)
        
        # 加载数据
        data = self.load_historical_data(start_date, end_date, sample_size)
        
        if data.empty:
            raise ValueError("无法加载有效数据")
        
        # 运行优化
        best_params, optimization_results = self.cross_validation_optimize(data, max_evals)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"qlib_integration/scoring_parameter_optimization_{timestamp}.json"
        
        final_results = {
            'optimization_date': timestamp,
            'data_period': {'start': start_date, 'end': end_date},
            'sample_size': len(data),
            'best_parameters': best_params,
            'optimization_results': optimization_results,
            'data_stats': {
                'total_records': len(data),
                'unique_stocks': data['code'].nunique(),
                'date_range': {
                    'start': data['trade_date'].min(),
                    'end': data['trade_date'].max()
                }
            }
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"结果已保存到: {results_file}")
        
        return final_results

def main():
    """主函数"""
    optimizer = ScoringParameterOptimizer(cv_folds=3)
    
    try:
        results = optimizer.run_optimization(
            start_date="2024-01-01",
            end_date="2025-09-01", 
            max_evals=30,  # 先用较少次数测试
            sample_size=1000
        )
        
        print("="*60)
        print("优化完成！")
        print(f"最佳IC: {results['optimization_results']['best_ic']:.4f}")
        print("最佳参数:")
        for key, value in results['best_parameters'].items():
            print(f"  {key}: {value:.4f}")
        print("="*60)
        
    except Exception as e:
        logging.error(f"优化过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()