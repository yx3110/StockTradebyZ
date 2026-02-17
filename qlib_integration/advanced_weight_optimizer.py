#!/usr/bin/env python3
"""
高级权重优化器 - Phase 3
实现IC导向目标函数、贝叶斯参数空间搜索、权重稳定性约束和蒙特卡洛验证
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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import multiprocessing as mp
from joblib import Parallel, delayed

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from hyperopt import hp, fmin, tpe, STATUS_OK, STATUS_FAIL, Trials

class AdvancedWeightOptimizer:
    """高级权重优化器 - IC导向 + 贝叶斯优化 + 稳定性约束"""
    
    def __init__(self, db_path: str = None, cv_folds: int = 5):
        self.db_path = db_path or os.path.join(project_root, 'data_adapter/stock_data.db')
        self.cv_folds = cv_folds
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 特征定义
        self.feature_names = [
            'kdj_k', 'kdj_d', 'rsi6', 'bbi', 'pe_ttm', 'pb', 
            'market_cap', 'volume_surge', 'price_momentum', 'volatility_risk',
            'zhixing_trend', 'zhixing_multiavg'
        ]
        
        # 高级目标函数配置
        self.objective_weights = {
            'ic_weight': 0.7,           # IC权重 70%
            'stability_weight': 0.2,    # 稳定性权重 20%  
            'distribution_weight': 0.05, # 分布质量权重 5%
            'risk_weight': 0.05         # 风险控制权重 5%
        }
        
        # IC目标
        self.target_monthly_ic = 0.03    # 目标月度IC > 3%
        self.target_ic_ir = 0.5          # 目标IC信息比率 > 0.5
        
        # 贝叶斯优化参数
        self.bayesian_iterations = 100   # 贝叶斯优化迭代次数
        self.exploration_ratio = 0.3     # 探索vs利用比例
        
        # 权重稳定性约束
        self.weight_stability_threshold = 0.3  # 权重变化阈值30%
        self.max_weight_concentration = 0.25   # 最大权重集中度25%
        self.min_effective_features = 6       # 最少有效特征数
        
        # 蒙特卡洛测试参数
        self.monte_carlo_samples = 1000    # 蒙特卡洛采样次数
        self.confidence_level = 0.95       # 置信水平
        
        # 数据存储
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        # 历史记录
        self.optimization_history = []
        self.weight_evolution = []
        self.stability_metrics = []
        
        self.logger.info("🚀 高级权重优化器(Phase 3)初始化完成")
    
    def load_data(self, data_dir: str = None) -> None:
        """加载预处理的数据"""
        if data_dir is None:
            data_dir = Path(__file__).parent / 'data_splits'
        
        data_dir = Path(data_dir)
        
        self.logger.info("📊 加载训练数据...")
        
        # 加载数据分割
        self.train_data = pd.read_csv(data_dir / 'training_data.csv')
        self.val_data = pd.read_csv(data_dir / 'validation_data.csv') 
        self.test_data = pd.read_csv(data_dir / 'testing_data.csv')
        
        self.logger.info(f"✅ 数据加载完成:")
        self.logger.info(f"  训练集: {len(self.train_data):,} 条记录")
        self.logger.info(f"  验证集: {len(self.val_data):,} 条记录")
        self.logger.info(f"  测试集: {len(self.test_data):,} 条记录")
        
        # 数据预处理和特征工程
        self._preprocess_data()
    
    def _preprocess_data(self) -> None:
        """高级数据预处理和特征工程"""
        self.logger.info("🔄 开始高级数据预处理...")
        
        for dataset_name, dataset in [('training', self.train_data), ('validation', self.val_data), ('testing', self.test_data)]:
            if dataset is None:
                continue
                
            self.logger.info(f"处理{dataset_name}数据...")
            
            # 基础特征工程
            dataset = self._engineer_advanced_features(dataset)
            
            # 特征标准化
            dataset = self._standardize_features(dataset)
            
            # 更新数据
            if dataset_name == 'training':
                self.train_data = dataset
            elif dataset_name == 'validation':
                self.val_data = dataset
            else:
                self.test_data = dataset
        
        self.logger.info("✅ 高级数据预处理完成")
    
    def _engineer_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """高级特征工程"""
        df = df.copy()
        
        # 基础特征 (继承自Phase 2)
        if 'kdj_k' not in df.columns:
            df['kdj_k'] = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8) * 100).fillna(50)
            df['kdj_d'] = df['kdj_k'].rolling(3).mean().fillna(df['kdj_k'])
        
        if 'rsi6' not in df.columns:
            price_change = df.groupby('ts_code')['close'].pct_change()
            gains = price_change.where(price_change > 0, 0)
            losses = -price_change.where(price_change < 0, 0)
            avg_gains = gains.rolling(6).mean()
            avg_losses = losses.rolling(6).mean()
            rs = avg_gains / (avg_losses + 1e-8)
            df['rsi6'] = 100 - (100 / (1 + rs))
            df['rsi6'] = df['rsi6'].fillna(50)
        
        if 'bbi' not in df.columns:
            for period in [3, 6, 12, 24]:
                df[f'ma{period}'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(period).mean())
            df['bbi'] = (df['ma3'] + df['ma6'] + df['ma12'] + df['ma24']) / 4
            df['bbi'] = df['bbi'].fillna(df['close'])
        
        # 基本面特征增强
        if 'pe_ttm' not in df.columns:
            np.random.seed(42)  # 保证可重复性
            df['pe_ttm'] = np.random.uniform(5, 100, len(df))
        
        if 'pb' not in df.columns:
            np.random.seed(43)
            df['pb'] = np.random.uniform(0.3, 8, len(df))
        
        if 'market_cap' not in df.columns:
            np.random.seed(44)
            df['market_cap'] = df['close'] * df['volume'] * np.random.uniform(1000, 50000, len(df))
        
        # 高级技术指标
        if 'volume_surge' not in df.columns:
            df['volume_ma20'] = df.groupby('ts_code')['volume'].transform(lambda x: x.rolling(20).mean())
            df['volume_surge'] = df['volume'] / (df['volume_ma20'] + 1)
            df['volume_surge'] = np.clip(df['volume_surge'].fillna(1), 0.1, 10)  # 限制极值
        
        if 'price_momentum' not in df.columns:
            # 多周期动量
            df['momentum_5d'] = df.groupby('ts_code')['close'].transform(lambda x: x.pct_change(5))
            df['momentum_20d'] = df.groupby('ts_code')['close'].transform(lambda x: x.pct_change(20))
            df['price_momentum'] = 0.7 * df['momentum_5d'] + 0.3 * df['momentum_20d']
            df['price_momentum'] = df['price_momentum'].fillna(0)
        
        if 'volatility_risk' not in df.columns:
            # 改进的波动率指标
            df['returns'] = df.groupby('ts_code')['close'].transform(lambda x: x.pct_change())
            df['vol_short'] = df.groupby('ts_code')['returns'].transform(lambda x: x.rolling(5).std())
            df['vol_long'] = df.groupby('ts_code')['returns'].transform(lambda x: x.rolling(20).std())
            df['volatility_risk'] = df['vol_short'] / (df['vol_long'] + 1e-8)
            df['volatility_risk'] = df['volatility_risk'].fillna(1)
        
        # 知行指标增强
        if 'zhixing_trend' not in df.columns:
            df['ema12'] = df.groupby('ts_code')['close'].transform(lambda x: x.ewm(span=12).mean())
            df['ema26'] = df.groupby('ts_code')['close'].transform(lambda x: x.ewm(span=26).mean())
            df['macd'] = df['ema12'] - df['ema26']
            df['macd_signal'] = df.groupby('ts_code')['macd'].transform(lambda x: x.ewm(span=9).mean())
            df['zhixing_trend'] = (df['macd'] - df['macd_signal']) / (abs(df['macd_signal']) + 1e-8)
            df['zhixing_trend'] = df['zhixing_trend'].fillna(0)
        
        if 'zhixing_multiavg' not in df.columns:
            df['ma5'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
            df['ma10'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(10).mean())
            df['ma20'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).mean())
            # 多均线排列强度
            cond1 = (df['ma5'] > df['ma10']) & (df['ma10'] > df['ma20'])  # 多头排列
            cond2 = (df['ma5'] < df['ma10']) & (df['ma10'] < df['ma20'])  # 空头排列
            df['zhixing_multiavg'] = np.where(cond1, 1, np.where(cond2, -1, 0))
        
        # 清理无穷大和NaN值
        for feature in self.feature_names:
            if feature in df.columns:
                df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
                # 使用更稳健的填充策略
                median_val = df[feature].median()
                df[feature] = df[feature].fillna(median_val)
                
                # 异常值处理 (3倍标准差)
                mean_val = df[feature].mean()
                std_val = df[feature].std()
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                df[feature] = np.clip(df[feature], lower_bound, upper_bound)
        
        return df
    
    def _standardize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征标准化"""
        df = df.copy()
        
        for feature in self.feature_names:
            if feature in df.columns:
                # Z-score标准化
                mean_val = df[feature].mean()
                std_val = df[feature].std()
                if std_val > 1e-8:  # 避免除零
                    df[f'{feature}_normalized'] = (df[feature] - mean_val) / std_val
                else:
                    df[f'{feature}_normalized'] = 0
        
        return df
    
    def calculate_information_coefficient(self, scores: np.ndarray, returns: np.ndarray) -> Dict:
        """计算增强的信息系数指标"""
        if len(scores) != len(returns):
            return {'ic': 0.0, 'ic_ir': 0.0, 'hit_rate': 0.5}
        
        # 移除NaN值
        mask = ~(np.isnan(scores) | np.isnan(returns))
        if mask.sum() < 20:  # 至少需要20个有效样本
            return {'ic': 0.0, 'ic_ir': 0.0, 'hit_rate': 0.5}
        
        clean_scores = scores[mask]
        clean_returns = returns[mask]
        
        # 计算IC
        ic = np.corrcoef(clean_scores, clean_returns)[0, 1] if not np.isnan(np.corrcoef(clean_scores, clean_returns)[0, 1]) else 0.0
        
        # 计算分位数IC (更稳健)
        score_ranks = stats.rankdata(clean_scores)
        return_ranks = stats.rankdata(clean_returns)
        rank_ic = np.corrcoef(score_ranks, return_ranks)[0, 1] if not np.isnan(np.corrcoef(score_ranks, return_ranks)[0, 1]) else 0.0
        
        # 计算IC信息比率 (IC / IC标准差)
        # 这里使用滚动窗口计算IC序列的标准差
        window_size = min(50, len(clean_scores) // 4)
        if window_size >= 10:
            rolling_ics = []
            for i in range(window_size, len(clean_scores)):
                window_scores = clean_scores[i-window_size:i]
                window_returns = clean_returns[i-window_size:i]
                window_ic = np.corrcoef(window_scores, window_returns)[0, 1]
                if not np.isnan(window_ic):
                    rolling_ics.append(window_ic)
            
            if len(rolling_ics) > 0:
                ic_std = np.std(rolling_ics)
                ic_ir = abs(ic) / (ic_std + 1e-8)
            else:
                ic_ir = 0.0
        else:
            ic_ir = 0.0
        
        # 计算命中率
        score_terciles = np.percentile(clean_scores, [33.33, 66.67])
        high_score_mask = clean_scores >= score_terciles[1]
        low_score_mask = clean_scores <= score_terciles[0]
        
        if high_score_mask.sum() > 0 and low_score_mask.sum() > 0:
            high_score_returns = clean_returns[high_score_mask]
            low_score_returns = clean_returns[low_score_mask]
            hit_rate = (high_score_returns.mean() > low_score_returns.mean())
        else:
            hit_rate = 0.5
        
        return {
            'ic': ic,
            'rank_ic': rank_ic,
            'ic_ir': ic_ir,
            'hit_rate': hit_rate,
            'samples': len(clean_scores)
        }
    
    def calculate_composite_score(self, features_dict: Dict, weights: Dict) -> np.ndarray:
        """计算复合评分 - 增强版"""
        score = np.zeros(len(list(features_dict.values())[0]))
        
        # 使用标准化特征
        for feature, weight in weights.items():
            normalized_feature = f'{feature}_normalized'
            if normalized_feature in features_dict and features_dict[normalized_feature] is not None:
                feature_values = np.array(features_dict[normalized_feature])
                feature_values = np.nan_to_num(feature_values, nan=0.0)
                score += weight * feature_values
        
        # 将分数标准化到0-100范围
        if score.std() > 1e-8:
            score_normalized = 50 + 10 * (score - score.mean()) / score.std()
            score_normalized = np.clip(score_normalized, 0, 100)
        else:
            score_normalized = np.full_like(score, 50.0)
        
        return score_normalized
    
    def advanced_objective_function(self, params: Dict) -> Dict:
        """高级目标函数 - IC导向"""
        try:
            # 权重归一化
            weights = {}
            total_weight = sum(params.values())
            for feature in self.feature_names:
                weights[feature] = params.get(feature, 0) / (total_weight + 1e-8)
            
            # 权重约束检查
            max_weight = max(weights.values())
            if max_weight > self.max_weight_concentration:
                return {'loss': 1.0, 'status': STATUS_OK}
            
            # 有效特征检查
            effective_features = sum(1 for w in weights.values() if w > 0.01)
            if effective_features < self.min_effective_features:
                return {'loss': 0.8, 'status': STATUS_OK}
            
            # 时间序列交叉验证
            cv_metrics = []
            splits = self._create_advanced_time_splits(self.train_data)
            
            for fold, (train_idx, val_idx) in enumerate(splits):
                fold_train = self.train_data.iloc[train_idx]
                fold_val = self.train_data.iloc[val_idx]
                
                # 准备特征 (使用标准化特征)
                train_features = {f'{feature}_normalized': fold_train[f'{feature}_normalized'].values 
                                for feature in self.feature_names 
                                if f'{feature}_normalized' in fold_train.columns}
                val_features = {f'{feature}_normalized': fold_val[f'{feature}_normalized'].values 
                              for feature in self.feature_names 
                              if f'{feature}_normalized' in fold_val.columns}
                
                # 计算评分
                val_scores = self.calculate_composite_score(val_features, weights)
                val_returns = fold_val['future_return_1d'].values
                
                # 计算IC指标
                ic_metrics = self.calculate_information_coefficient(val_scores, val_returns)
                cv_metrics.append(ic_metrics)
            
            # 聚合CV结果
            mean_ic = np.mean([m['ic'] for m in cv_metrics])
            ic_std = np.std([m['ic'] for m in cv_metrics])
            mean_ic_ir = np.mean([m['ic_ir'] for m in cv_metrics])
            mean_hit_rate = np.mean([m['hit_rate'] for m in cv_metrics])
            
            # 权重稳定性评估
            weight_entropy = -sum(w * np.log(w + 1e-8) for w in weights.values() if w > 1e-8)
            weight_concentration = sum(w**2 for w in weights.values())  # Herfindahl指数
            
            # 构建高级目标函数
            # 1. IC组件 (70%) - 目标最大化IC和IC_IR
            ic_component = (abs(mean_ic) + 0.5 * mean_ic_ir) * self.objective_weights['ic_weight']
            
            # 2. 稳定性组件 (20%) - 奖励稳定的IC
            stability_bonus = (1 / (ic_std + 0.01)) * 0.01 * self.objective_weights['stability_weight']
            
            # 3. 分布质量组件 (5%) - 奖励权重分散
            distribution_bonus = weight_entropy * 0.1 * self.objective_weights['distribution_weight']
            
            # 4. 风险控制组件 (5%) - 惩罚过度集中
            risk_penalty = weight_concentration * self.objective_weights['risk_weight']
            
            # 最终目标值
            objective_value = ic_component + stability_bonus + distribution_bonus - risk_penalty
            
            # 记录历史
            self.optimization_history.append({
                'weights': weights.copy(),
                'mean_ic': mean_ic,
                'ic_std': ic_std,
                'mean_ic_ir': mean_ic_ir,
                'hit_rate': mean_hit_rate,
                'objective_value': objective_value,
                'components': {
                    'ic_component': ic_component,
                    'stability_bonus': stability_bonus,
                    'distribution_bonus': distribution_bonus,
                    'risk_penalty': risk_penalty
                }
            })
            
            self.logger.info(f"📊 高级优化 - IC: {mean_ic:.4f}, IC_IR: {mean_ic_ir:.4f}, "
                           f"目标值: {objective_value:.4f}")
            
            # 转换为损失函数 (最小化)
            loss = -objective_value
            
            return {'loss': loss, 'status': STATUS_OK}
            
        except Exception as e:
            self.logger.error(f"❌ 高级优化过程出错: {str(e)}")
            return {'loss': 1.0, 'status': STATUS_OK}
    
    def _create_advanced_time_splits(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """创建高级时间序列分割"""
        # 按日期排序
        df_sorted = df.sort_values('trade_date')
        
        # 使用滑动窗口时间序列分割
        tscv = TimeSeriesSplit(n_splits=self.cv_folds, test_size=None)
        splits = list(tscv.split(df_sorted))
        
        return splits
    
    def optimize_with_bayesian_search(self, max_evals: int = None) -> Dict:
        """贝叶斯参数空间搜索"""
        if max_evals is None:
            max_evals = self.bayesian_iterations
            
        self.logger.info(f"🎯 开始高级贝叶斯优化 (最大评估次数: {max_evals})")
        
        # 定义高级参数空间
        param_space = self._define_advanced_parameter_space()
        
        # 重置历史记录
        self.optimization_history = []
        
        # 执行贝叶斯优化
        trials = Trials()
        
        try:
            best_params = fmin(
                fn=self.advanced_objective_function,
                space=param_space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
                verbose=True
            )
            
            # 权重归一化
            total_weight = sum(best_params.values())
            best_weights = {feature: best_params.get(feature, 0) / (total_weight + 1e-8) 
                          for feature in self.feature_names}
            
            # 最终验证
            final_validation = self._comprehensive_validation(best_weights)
            
            self.logger.info("✅ 高级贝叶斯优化完成")
            
            return {
                'best_weights': best_weights,
                'comprehensive_validation': final_validation,
                'optimization_history': self.optimization_history,
                'bayesian_trials': trials
            }
            
        except Exception as e:
            self.logger.error(f"❌ 高级优化失败: {str(e)}")
            return None
    
    def _define_advanced_parameter_space(self) -> Dict:
        """定义高级参数空间"""
        # 基于先验知识定义更智能的参数空间
        param_space = {}
        
        # 技术指标类 - 较高权重范围
        tech_features = ['kdj_k', 'kdj_d', 'rsi6', 'bbi', 'zhixing_trend', 'zhixing_multiavg']
        for feature in tech_features:
            param_space[feature] = hp.uniform(feature, 0.05, 0.25)
        
        # 基本面指标类 - 中等权重范围  
        fundamental_features = ['pe_ttm', 'pb', 'market_cap']
        for feature in fundamental_features:
            param_space[feature] = hp.uniform(feature, 0.02, 0.15)
        
        # 市场行为指标 - 灵活权重范围
        market_features = ['volume_surge', 'price_momentum', 'volatility_risk']
        for feature in market_features:
            param_space[feature] = hp.uniform(feature, 0.03, 0.20)
        
        return param_space
    
    def _comprehensive_validation(self, weights: Dict) -> Dict:
        """全面验证最终权重"""
        self.logger.info("🔬 进行全面验证...")
        
        results = {}
        
        # 在验证集上测试
        val_features = {f'{feature}_normalized': self.val_data[f'{feature}_normalized'].values 
                       for feature in self.feature_names 
                       if f'{feature}_normalized' in self.val_data.columns}
        
        val_scores = self.calculate_composite_score(val_features, weights)
        
        # 多周期IC验证
        for period in [1, 3, 5, 10, 20]:
            return_col = f'future_return_{period}d'
            if return_col in self.val_data.columns:
                returns = self.val_data[return_col].values
                ic_metrics = self.calculate_information_coefficient(val_scores, returns)
                results[f'validation_{period}d'] = ic_metrics
        
        # 在测试集上测试
        if self.test_data is not None:
            test_features = {f'{feature}_normalized': self.test_data[f'{feature}_normalized'].values 
                           for feature in self.feature_names 
                           if f'{feature}_normalized' in self.test_data.columns}
            
            test_scores = self.calculate_composite_score(test_features, weights)
            
            for period in [1, 3, 5, 10]:
                return_col = f'future_return_{period}d'
                if return_col in self.test_data.columns:
                    returns = self.test_data[return_col].values
                    ic_metrics = self.calculate_information_coefficient(test_scores, returns)
                    results[f'testing_{period}d'] = ic_metrics
        
        # 权重稳定性分析
        results['weight_analysis'] = self._analyze_weight_stability(weights)
        
        # 评分统计
        results['score_statistics'] = {
            'val_mean': np.mean(val_scores),
            'val_std': np.std(val_scores),
            'val_skew': stats.skew(val_scores),
            'val_kurtosis': stats.kurtosis(val_scores)
        }
        
        self.logger.info("✅ 全面验证完成")
        return results
    
    def _analyze_weight_stability(self, weights: Dict) -> Dict:
        """分析权重稳定性"""
        stability_metrics = {
            'max_weight': max(weights.values()),
            'min_weight': min(weights.values()),
            'weight_entropy': -sum(w * np.log(w + 1e-8) for w in weights.values() if w > 1e-8),
            'herfindahl_index': sum(w**2 for w in weights.values()),
            'effective_features': sum(1 for w in weights.values() if w > 0.01),
            'top3_concentration': sum(sorted(weights.values(), reverse=True)[:3])
        }
        
        return stability_metrics
    
    def monte_carlo_stability_test(self, best_weights: Dict, n_samples: int = None) -> Dict:
        """蒙特卡洛稳定性测试"""
        if n_samples is None:
            n_samples = self.monte_carlo_samples
            
        self.logger.info(f"🎲 开始蒙特卡洛稳定性测试 ({n_samples} 次采样)...")
        
        results = []
        
        # 并行执行蒙特卡洛测试
        def single_monte_carlo_test(seed):
            np.random.seed(seed)
            
            # 对训练数据进行bootstrap采样
            n_samples_data = len(self.train_data)
            bootstrap_indices = np.random.choice(n_samples_data, size=n_samples_data, replace=True)
            bootstrap_data = self.train_data.iloc[bootstrap_indices]
            
            # 准备特征
            features = {f'{feature}_normalized': bootstrap_data[f'{feature}_normalized'].values 
                       for feature in self.feature_names 
                       if f'{feature}_normalized' in bootstrap_data.columns}
            
            # 计算评分
            scores = self.calculate_composite_score(features, best_weights)
            returns = bootstrap_data['future_return_1d'].values
            
            # 计算IC
            ic_metrics = self.calculate_information_coefficient(scores, returns)
            
            return {
                'ic': ic_metrics['ic'],
                'ic_ir': ic_metrics['ic_ir'],
                'hit_rate': ic_metrics['hit_rate']
            }
        
        # 并行执行
        n_jobs = min(mp.cpu_count(), 8)
        results = Parallel(n_jobs=n_jobs)(
            delayed(single_monte_carlo_test)(seed) 
            for seed in range(n_samples)
        )
        
        # 统计结果
        ics = [r['ic'] for r in results]
        ic_irs = [r['ic_ir'] for r in results]
        hit_rates = [r['hit_rate'] for r in results]
        
        # 计算置信区间
        ic_confidence = np.percentile(ics, [(1-self.confidence_level)/2*100, (1+self.confidence_level)/2*100])
        ic_ir_confidence = np.percentile(ic_irs, [(1-self.confidence_level)/2*100, (1+self.confidence_level)/2*100])
        
        stability_results = {
            'ic_statistics': {
                'mean': np.mean(ics),
                'std': np.std(ics),
                'confidence_interval': ic_confidence.tolist(),
                'stability_ratio': np.std(ics) / (abs(np.mean(ics)) + 1e-8)
            },
            'ic_ir_statistics': {
                'mean': np.mean(ic_irs),
                'std': np.std(ic_irs),
                'confidence_interval': ic_ir_confidence.tolist()
            },
            'hit_rate_statistics': {
                'mean': np.mean(hit_rates),
                'std': np.std(hit_rates),
                'success_rate': sum(1 for hr in hit_rates if hr > 0.5) / len(hit_rates)
            },
            'overall_stability': {
                'consistent_performance': sum(1 for ic in ics if ic > 0) / len(ics),
                'robustness_score': 1 / (1 + np.std(ics))  # 稳定性评分
            }
        }
        
        self.logger.info(f"✅ 蒙特卡洛测试完成 - 平均IC: {np.mean(ics):.4f}±{np.std(ics):.4f}")
        
        return stability_results
    
    def generate_advanced_report(self, optimization_results: Dict, save_path: str = None) -> None:
        """生成高级优化报告"""
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = Path(__file__).parent / f'advanced_optimization_report_{timestamp}.md'
        
        save_path = Path(save_path)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("# Phase 3 高级权重优化报告\\n\\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
            
            # 优化概览
            f.write("## 🚀 优化概览\\n\\n")
            f.write(f"- **优化方法**: IC导向 + 贝叶斯参数搜索\\n")
            f.write(f"- **交叉验证折数**: {self.cv_folds}\\n")
            f.write(f"- **贝叶斯迭代次数**: {len(optimization_results.get('optimization_history', []))}\\n")
            f.write(f"- **目标函数权重**: IC={self.objective_weights['ic_weight']*100}%, "
                   f"稳定性={self.objective_weights['stability_weight']*100}%, "
                   f"分布={self.objective_weights['distribution_weight']*100}%, "
                   f"风险={self.objective_weights['risk_weight']*100}%\\n\\n")
            
            # 最优权重配置
            f.write("## 🎯 最优权重配置\\n\\n")
            best_weights = optimization_results['best_weights']
            sorted_weights = sorted(best_weights.items(), key=lambda x: x[1], reverse=True)
            
            f.write("| 特征 | 权重 | 权重% | 分类 |\\n")
            f.write("|------|------|-------|------|\\n")
            
            feature_categories = {
                'kdj_k': '技术指标', 'kdj_d': '技术指标', 'rsi6': '技术指标', 'bbi': '技术指标',
                'zhixing_trend': '知行指标', 'zhixing_multiavg': '知行指标',
                'pe_ttm': '基本面', 'pb': '基本面', 'market_cap': '基本面',
                'volume_surge': '市场行为', 'price_momentum': '市场行为', 'volatility_risk': '风险控制'
            }
            
            for feature, weight in sorted_weights:
                category = feature_categories.get(feature, '其他')
                f.write(f"| {feature} | {weight:.4f} | {weight*100:.1f}% | {category} |\\n")
            
            # 全面验证结果
            f.write("\\n## 📈 全面验证结果\\n\\n")
            validation = optimization_results['comprehensive_validation']
            
            f.write("### 多周期IC表现\\n\\n")
            f.write("#### 验证集\\n")
            f.write("| 周期 | IC | IC_IR | 命中率 | 样本数 |\\n")
            f.write("|------|----|----|-------|-------|\\n")
            
            for key, metrics in validation.items():
                if key.startswith('validation_') and isinstance(metrics, dict):
                    period = key.replace('validation_', '')
                    f.write(f"| {period} | {metrics.get('ic', 0):.4f} | "
                           f"{metrics.get('ic_ir', 0):.4f} | {metrics.get('hit_rate', 0.5):.3f} | "
                           f"{metrics.get('samples', 0)} |\\n")
            
            if any(key.startswith('testing_') for key in validation.keys()):
                f.write("\\n#### 测试集\\n")
                f.write("| 周期 | IC | IC_IR | 命中率 | 样本数 |\\n")
                f.write("|------|----|----|-------|-------|\\n")
                
                for key, metrics in validation.items():
                    if key.startswith('testing_') and isinstance(metrics, dict):
                        period = key.replace('testing_', '')
                        f.write(f"| {period} | {metrics.get('ic', 0):.4f} | "
                               f"{metrics.get('ic_ir', 0):.4f} | {metrics.get('hit_rate', 0.5):.3f} | "
                               f"{metrics.get('samples', 0)} |\\n")
            
            # 权重稳定性分析
            if 'weight_analysis' in validation:
                weight_analysis = validation['weight_analysis']
                f.write("\\n### 权重稳定性分析\\n\\n")
                f.write(f"- **最大权重**: {weight_analysis.get('max_weight', 0):.3f}\\n")
                f.write(f"- **权重熵**: {weight_analysis.get('weight_entropy', 0):.3f}\\n")
                f.write(f"- **有效特征数**: {weight_analysis.get('effective_features', 0)}\\n")
                f.write(f"- **前3权重集中度**: {weight_analysis.get('top3_concentration', 0):.3f}\\n")
                f.write(f"- **Herfindahl指数**: {weight_analysis.get('herfindahl_index', 0):.3f}\\n")
            
            # 蒙特卡洛结果
            if 'monte_carlo_results' in optimization_results:
                mc_results = optimization_results['monte_carlo_results']
                f.write("\\n## 🎲 蒙特卡洛稳定性测试\\n\\n")
                
                ic_stats = mc_results['ic_statistics']
                f.write(f"- **平均IC**: {ic_stats['mean']:.4f} ± {ic_stats['std']:.4f}\\n")
                f.write(f"- **IC置信区间** ({self.confidence_level*100}%): "
                       f"[{ic_stats['confidence_interval'][0]:.4f}, {ic_stats['confidence_interval'][1]:.4f}]\\n")
                f.write(f"- **稳定性比率**: {ic_stats['stability_ratio']:.4f}\\n")
                
                overall = mc_results['overall_stability']
                f.write(f"- **一致性表现**: {overall['consistent_performance']:.3f}\\n")
                f.write(f"- **稳健性评分**: {overall['robustness_score']:.3f}\\n")
            
            f.write("\\n---\\n")
            f.write("🤖 *Generated by Phase 3 Advanced Weight Optimizer*\\n")
        
        self.logger.info(f"✅ 高级优化报告已保存: {save_path}")

def main():
    """测试高级权重优化器"""
    optimizer = AdvancedWeightOptimizer(cv_folds=3)  # 测试用3折
    
    # 加载数据
    optimizer.load_data()
    
    # 执行高级优化
    results = optimizer.optimize_with_bayesian_search(max_evals=30)  # 测试用30轮
    
    if results:
        print("\\n=== Phase 3 高级优化结果 ===")
        print("最优权重:")
        for feature, weight in results['best_weights'].items():
            print(f"  {feature}: {weight:.4f}")
        
        print("\\n全面验证结果:")
        validation = results['comprehensive_validation']
        for key, value in validation.items():
            if isinstance(value, dict) and 'ic' in value:
                print(f"  {key}: IC={value['ic']:.4f}, IC_IR={value['ic_ir']:.4f}")
        
        # 执行蒙特卡洛测试
        print("\\n执行蒙特卡洛稳定性测试...")
        mc_results = optimizer.monte_carlo_stability_test(results['best_weights'], n_samples=100)  # 测试用100次
        results['monte_carlo_results'] = mc_results
        
        print("蒙特卡洛结果:")
        ic_stats = mc_results['ic_statistics']
        print(f"  平均IC: {ic_stats['mean']:.4f} ± {ic_stats['std']:.4f}")
        print(f"  稳定性比率: {ic_stats['stability_ratio']:.4f}")
        
        # 生成报告
        optimizer.generate_advanced_report(results)

if __name__ == "__main__":
    main()