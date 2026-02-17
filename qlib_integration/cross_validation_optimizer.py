#!/usr/bin/env python3
"""
交叉验证权重优化器
实现时间序列K折交叉验证、正则化、早停等防过拟合机制
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

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from hyperopt import hp, fmin, tpe, STATUS_OK, STATUS_FAIL, Trials

class CrossValidationOptimizer:
    """交叉验证权重优化器 - 防过拟合版本"""
    
    def __init__(self, db_path: str = None, cv_folds: int = 5):
        self.db_path = db_path or os.path.join(project_root, 'data_adapter/stock_data.db')
        self.cv_folds = cv_folds
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 优化参数
        self.feature_names = [
            'kdj_k', 'kdj_d', 'rsi6', 'bbi', 'pe_ttm', 'pb', 
            'market_cap', 'volume_surge', 'price_momentum', 'volatility_risk',
            'zhixing_trend', 'zhixing_multiavg'
        ]
        
        # 正则化参数
        self.l1_alpha = 0.01  # L1正则化系数
        self.l2_alpha = 0.01  # L2正则化系数
        self.weight_constraint = 0.3  # 单个权重上限
        
        # 早停参数
        self.patience = 5  # 早停耐心值
        self.min_delta = 0.001  # 最小改进阈值
        
        # 验证指标
        self.validation_metrics = []
        self.best_score = -np.inf
        self.patience_counter = 0
        
        # 数据存储
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        self.logger.info("🚀 交叉验证权重优化器初始化完成")
    
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
        """数据预处理和特征工程"""
        self.logger.info("🔄 开始数据预处理...")
        
        for dataset_name, dataset in [('training', self.train_data), ('validation', self.val_data), ('testing', self.test_data)]:
            if dataset is None:
                continue
                
            self.logger.info(f"处理{dataset_name}数据...")
            
            # 基础特征工程
            dataset = self._engineer_features(dataset)
            
            # 更新数据
            if dataset_name == 'training':
                self.train_data = dataset
            elif dataset_name == 'validation':
                self.val_data = dataset
            else:
                self.test_data = dataset
        
        self.logger.info("✅ 数据预处理完成")
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征工程"""
        df = df.copy()
        
        # 确保必要的列存在，如果不存在则创建模拟值
        if 'kdj_k' not in df.columns:
            # 基于价格计算简单的KDJ指标
            df['kdj_k'] = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8) * 100).fillna(50)
            df['kdj_d'] = df['kdj_k'].rolling(3).mean().fillna(df['kdj_k'])
        
        if 'rsi6' not in df.columns:
            # 简单RSI计算
            price_change = df.groupby('ts_code')['close'].pct_change()
            gains = price_change.where(price_change > 0, 0)
            losses = -price_change.where(price_change < 0, 0)
            avg_gains = gains.rolling(6).mean()
            avg_losses = losses.rolling(6).mean()
            rs = avg_gains / (avg_losses + 1e-8)
            df['rsi6'] = 100 - (100 / (1 + rs))
            df['rsi6'] = df['rsi6'].fillna(50)
        
        if 'bbi' not in df.columns:
            # BBI指标 = (MA3 + MA6 + MA12 + MA24) / 4
            for period in [3, 6, 12, 24]:
                df[f'ma{period}'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(period).mean())
            df['bbi'] = (df['ma3'] + df['ma6'] + df['ma12'] + df['ma24']) / 4
            df['bbi'] = df['bbi'].fillna(df['close'])
        
        if 'pe_ttm' not in df.columns:
            df['pe_ttm'] = np.random.uniform(10, 50, len(df))  # 模拟PE值
        
        if 'pb' not in df.columns:
            df['pb'] = np.random.uniform(0.5, 5, len(df))  # 模拟PB值
        
        if 'market_cap' not in df.columns:
            df['market_cap'] = df['close'] * df['volume'] * np.random.uniform(1000, 10000, len(df))
        
        # 派生特征
        if 'volume_surge' not in df.columns:
            df['volume_ma5'] = df.groupby('ts_code')['volume'].transform(lambda x: x.rolling(5).mean())
            df['volume_surge'] = df['volume'] / (df['volume_ma5'] + 1)
            df['volume_surge'] = df['volume_surge'].fillna(1)
        
        if 'price_momentum' not in df.columns:
            df['price_momentum'] = df.groupby('ts_code')['close'].transform(lambda x: x.pct_change(5))
            df['price_momentum'] = df['price_momentum'].fillna(0)
        
        if 'volatility_risk' not in df.columns:
            df['volatility_risk'] = df.groupby('ts_code')['close'].transform(
                lambda x: x.pct_change().rolling(20).std()
            )
            df['volatility_risk'] = df['volatility_risk'].fillna(df['volatility_risk'].median())
        
        # 知行指标（简化版）
        if 'zhixing_trend' not in df.columns:
            df['ema12'] = df.groupby('ts_code')['close'].transform(lambda x: x.ewm(span=12).mean())
            df['ema26'] = df.groupby('ts_code')['close'].transform(lambda x: x.ewm(span=26).mean())
            df['zhixing_trend'] = (df['ema12'] - df['ema26']) / df['ema26']
            df['zhixing_trend'] = df['zhixing_trend'].fillna(0)
        
        if 'zhixing_multiavg' not in df.columns:
            df['ma5'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
            df['ma20'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).mean())
            df['zhixing_multiavg'] = (df['ma5'] - df['ma20']) / df['ma20']
            df['zhixing_multiavg'] = df['zhixing_multiavg'].fillna(0)
        
        # 清理无穷大和NaN值
        for feature in self.feature_names:
            if feature in df.columns:
                df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
                df[feature] = df[feature].fillna(df[feature].median())
        
        return df
    
    def create_time_series_splits(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """创建时间序列交叉验证分割"""
        self.logger.info(f"🔀 创建时间序列{self.cv_folds}折交叉验证...")
        
        # 按日期排序
        df_sorted = df.sort_values('trade_date')
        
        # 使用时间序列分割
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        splits = list(tscv.split(df_sorted))
        
        self.logger.info(f"✅ 创建了{len(splits)}个时间序列分割")
        
        # 打印分割信息
        for i, (train_idx, val_idx) in enumerate(splits):
            train_dates = df_sorted.iloc[train_idx]['trade_date']
            val_dates = df_sorted.iloc[val_idx]['trade_date']
            self.logger.info(f"  分割{i+1}: 训练({train_dates.min()}到{train_dates.max()}) "
                           f"验证({val_dates.min()}到{val_dates.max()})")
        
        return splits
    
    def calculate_information_coefficient(self, scores: np.ndarray, returns: np.ndarray) -> float:
        """计算信息系数(IC)"""
        if len(scores) != len(returns):
            return 0.0
        
        # 移除NaN值
        mask = ~(np.isnan(scores) | np.isnan(returns))
        if mask.sum() < 10:  # 至少需要10个有效样本
            return 0.0
        
        clean_scores = scores[mask]
        clean_returns = returns[mask]
        
        # 计算Pearson相关系数
        correlation_matrix = np.corrcoef(clean_scores, clean_returns)
        ic = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0
        
        return ic
    
    def calculate_composite_score(self, features_dict: Dict, weights: Dict) -> np.ndarray:
        """计算复合评分"""
        score = np.zeros(len(list(features_dict.values())[0]))
        
        for feature, weight in weights.items():
            if feature in features_dict and features_dict[feature] is not None:
                feature_values = np.array(features_dict[feature])
                # 标准化特征值到0-100范围
                feature_normalized = ((feature_values - np.nanmin(feature_values)) / 
                                    (np.nanmax(feature_values) - np.nanmin(feature_values) + 1e-8) * 100)
                feature_normalized = np.nan_to_num(feature_normalized, nan=50.0)
                score += weight * feature_normalized
        
        return score
    
    def cross_validation_objective(self, params: Dict) -> Dict:
        """交叉验证目标函数"""
        try:
            # 权重归一化
            weights = {}
            total_weight = sum(params.values())
            for feature in self.feature_names:
                weights[feature] = params.get(feature, 0) / (total_weight + 1e-8)
            
            # 权重约束检查
            if any(w > self.weight_constraint for w in weights.values()):
                return {'loss': 1.0, 'status': STATUS_OK}
            
            cv_scores = []
            
            # 时间序列交叉验证
            splits = self.create_time_series_splits(self.train_data)
            
            for fold, (train_idx, val_idx) in enumerate(splits):
                # 分割数据
                fold_train = self.train_data.iloc[train_idx]
                fold_val = self.train_data.iloc[val_idx]
                
                # 准备特征
                train_features = {feature: fold_train[feature].values for feature in self.feature_names 
                                if feature in fold_train.columns}
                val_features = {feature: fold_val[feature].values for feature in self.feature_names 
                              if feature in fold_val.columns}
                
                # 计算评分
                val_scores = self.calculate_composite_score(val_features, weights)
                val_returns = fold_val['future_return_1d'].values
                
                # 计算IC
                fold_ic = self.calculate_information_coefficient(val_scores, val_returns)
                cv_scores.append(fold_ic)
            
            # 平均IC
            mean_ic = np.mean(cv_scores)
            ic_std = np.std(cv_scores)
            
            # 添加正则化惩罚
            l1_penalty = self.l1_alpha * sum(abs(w) for w in weights.values())
            l2_penalty = self.l2_alpha * sum(w**2 for w in weights.values())
            
            # 最终目标值（越大越好，转换为损失函数）
            objective_value = mean_ic - ic_std - l1_penalty - l2_penalty
            loss = -objective_value
            
            # 记录验证指标
            self.validation_metrics.append({
                'fold_ics': cv_scores,
                'mean_ic': mean_ic,
                'ic_std': ic_std,
                'objective_value': objective_value,
                'weights': weights.copy()
            })
            
            self.logger.info(f"📊 CV结果 - 平均IC: {mean_ic:.4f}, IC标准差: {ic_std:.4f}, 目标值: {objective_value:.4f}")
            
            # 早停检查
            if objective_value > self.best_score + self.min_delta:
                self.best_score = objective_value
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                self.logger.info(f"⏹️ 早停触发 - 连续{self.patience}轮无显著改进")
                # 返回一个信号让优化提前结束
                return {'loss': loss, 'status': STATUS_FAIL}
            
            return {'loss': loss, 'status': STATUS_OK}
            
        except Exception as e:
            self.logger.error(f"❌ 优化过程出错: {str(e)}")
            return {'loss': 1.0, 'status': STATUS_OK}
    
    def optimize_weights(self, max_evals: int = 50) -> Dict:
        """执行权重优化"""
        self.logger.info(f"🎯 开始交叉验证权重优化 (最大评估次数: {max_evals})")
        
        # 定义参数空间
        param_space = {}
        for feature in self.feature_names:
            param_space[feature] = hp.uniform(feature, 0.01, 0.3)
        
        # 重置验证指标
        self.validation_metrics = []
        self.best_score = -np.inf
        self.patience_counter = 0
        
        # 执行优化
        trials = Trials()
        
        try:
            best_params = fmin(
                fn=self.cross_validation_objective,
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
            
            # 计算最终验证结果
            final_results = self._validate_final_weights(best_weights)
            
            self.logger.info("✅ 交叉验证优化完成")
            
            return {
                'best_weights': best_weights,
                'validation_results': final_results,
                'cv_history': self.validation_metrics,
                'trials': trials
            }
            
        except Exception as e:
            self.logger.error(f"❌ 优化失败: {str(e)}")
            return None
    
    def _validate_final_weights(self, weights: Dict) -> Dict:
        """在验证集上验证最终权重"""
        self.logger.info("🔬 在验证集上测试最终权重...")
        
        # 验证集特征
        val_features = {feature: self.val_data[feature].values for feature in self.feature_names 
                       if feature in self.val_data.columns}
        
        # 计算验证集评分
        val_scores = self.calculate_composite_score(val_features, weights)
        
        # 不同周期的收益率验证
        results = {}
        for period in [1, 3, 5, 10]:
            return_col = f'future_return_{period}d'
            if return_col in self.val_data.columns:
                returns = self.val_data[return_col].values
                ic = self.calculate_information_coefficient(val_scores, returns)
                results[f'ic_{period}d'] = ic
        
        # 评分统计
        results['score_stats'] = {
            'mean': np.mean(val_scores),
            'std': np.std(val_scores),
            'min': np.min(val_scores),
            'max': np.max(val_scores)
        }
        
        self.logger.info(f"✅ 验证完成 - 平均IC: {np.mean(list(results[k] for k in results if k.startswith('ic_'))):.4f}")
        
        return results
    
    def generate_optimization_report(self, optimization_results: Dict, save_path: str = None) -> None:
        """生成优化报告"""
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = Path(__file__).parent / f'cross_validation_report_{timestamp}.md'
        
        save_path = Path(save_path)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("# 交叉验证权重优化报告\\n\\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
            
            # 优化概览
            f.write("## 📊 优化概览\\n\\n")
            f.write(f"- **交叉验证折数**: {self.cv_folds}\\n")
            f.write(f"- **评估轮次**: {len(self.validation_metrics)}\\n")
            f.write(f"- **早停机制**: 耐心值={self.patience}, 最小改进阈值={self.min_delta}\\n")
            f.write(f"- **正则化**: L1={self.l1_alpha}, L2={self.l2_alpha}\\n\\n")
            
            # 最优权重
            f.write("## 🎯 最优权重配置\\n\\n")
            best_weights = optimization_results['best_weights']
            
            # 按权重大小排序
            sorted_weights = sorted(best_weights.items(), key=lambda x: x[1], reverse=True)
            
            f.write("| 特征 | 权重 | 权重% |\\n")
            f.write("|------|------|-------|\\n")
            for feature, weight in sorted_weights:
                f.write(f"| {feature} | {weight:.4f} | {weight*100:.1f}% |\\n")
            
            # 验证结果
            f.write("\\n## 📈 验证结果\\n\\n")
            val_results = optimization_results['validation_results']
            
            f.write("### 信息系数(IC)\\n\\n")
            for key, value in val_results.items():
                if key.startswith('ic_'):
                    period = key.split('_')[1]
                    f.write(f"- **{period}**: {value:.4f}\\n")
            
            # 交叉验证历史
            f.write("\\n## 🔄 交叉验证历史\\n\\n")
            cv_history = optimization_results['cv_history']
            
            if cv_history:
                f.write("### 收敛过程\\n\\n")
                for i, record in enumerate(cv_history[:10]):  # 显示前10轮
                    f.write(f"- 第{i+1}轮: 平均IC={record['mean_ic']:.4f}, "
                           f"IC标准差={record['ic_std']:.4f}, 目标值={record['objective_value']:.4f}\\n")
            
            f.write("\\n---\\n")
            f.write("🤖 *Generated by Cross Validation Optimizer*\\n")
        
        self.logger.info(f"✅ 优化报告已保存: {save_path}")

def main():
    """主函数 - 测试交叉验证优化器"""
    optimizer = CrossValidationOptimizer(cv_folds=3)  # 测试用3折
    
    # 加载数据
    optimizer.load_data()
    
    # 执行优化
    results = optimizer.optimize_weights(max_evals=20)  # 测试用20轮
    
    if results:
        print("\\n=== 优化结果 ===")
        print("最优权重:")
        for feature, weight in results['best_weights'].items():
            print(f"  {feature}: {weight:.4f}")
        
        print("\\n验证结果:")
        for key, value in results['validation_results'].items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v:.4f}")
            else:
                print(f"  {key}: {value:.4f}")
        
        # 生成报告
        optimizer.generate_optimization_report(results)

if __name__ == "__main__":
    main()