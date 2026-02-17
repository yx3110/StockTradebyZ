#!/usr/bin/env python3
"""
多时间周期IC优化器 - V3.53专用

实现多时间周期权重联合优化:
- 针对1d/3d/5d/10d/15d分别计算IC
- 使用贝叶斯优化寻找每个时间周期的最优权重
- 联合优化多目标函数，平衡各时间周期表现
- 生成V3.53优化权重配置
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
from scipy.stats import spearmanr, pearsonr
from hyperopt import hp, fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

warnings.filterwarnings('ignore')

class MultiPeriodICOptimizer:
    """多时间周期IC联合优化器"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join(project_root, 'data_adapter', 'stock_data.db')
        
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # 时间周期配置
        self.periods = ['1d', '3d', '5d', '10d', '15d']
        self.period_importance = {
            '1d': 0.35,    # 35% - 短期交易最重要
            '3d': 0.25,    # 25% - T+1后续表现重要  
            '5d': 0.20,    # 20% - 周度表现有意义
            '10d': 0.15,   # 15% - 双周表现参考
            '15d': 0.05    # 5% - 月度趋势指导
        }
        
        # 12因子列表
        self.factors = [
            'rsi6', 'kdj_k', 'kdj_d', 'bbi', 'zhixing_trend', 'zhixing_multiavg',
            'pe_ttm', 'pb', 'market_cap', 'price_momentum', 'volume_surge', 'volatility_risk'
        ]
        
        # 贝叶斯优化配置
        self.optimization_config = {
            'max_evals': 200,           # 优化轮数
            'cv_folds': 3,              # 交叉验证折数
            'ic_threshold': 0.01,       # IC阈值 1%
            'stability_weight': 0.2     # 稳定性权重
        }
        
        # IC目标
        self.ic_targets = {
            '1d': 0.025,    # 1日IC目标: 2.5%
            '3d': 0.015,    # 3日IC目标: 1.5%
            '5d': 0.010,    # 5日IC目标: 1.0%
            '10d': 0.005,   # 10日IC目标: 0.5%
            '15d': 0.003    # 15日IC目标: 0.3%
        }
        
        # 数据存储
        self.data = None
        self.optimization_results = {}
        self.best_weights = {}
        self.ic_performance = {}
        
        self.logger.info("🚀 多时间周期IC优化器初始化完成")
    
    def load_and_prepare_data(self, start_date: str = '2022-01-01', 
                             end_date: str = '2025-09-09') -> None:
        """加载和预处理数据"""
        self.logger.info("📊 加载股票数据...")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 加载股票数据和技术指标
                query = """
                SELECT 
                    s.code, dq.trade_date,
                    dq.close, dq.price_change_pct,
                    ti.rsi6, ti.kdj_k, ti.kdj_d, ti.bbi,
                    ti.ma5, ti.ma10, ti.ma20, ti.ema12, ti.ema26,
                    db.pe_ttm, db.pb, db.total_mv as market_cap,
                    dq.volume, ti.volume_ratio_5d, ti.volume_ratio_20d,
                    ti.volatility_20d
                FROM securities s
                JOIN daily_quotes dq ON s.id = dq.security_id
                LEFT JOIN technical_indicators ti ON s.id = ti.security_id 
                    AND dq.trade_date = ti.trade_date
                LEFT JOIN daily_basic db ON s.id = db.security_id 
                    AND dq.trade_date = db.trade_date
                WHERE dq.trade_date >= ? AND dq.trade_date <= ?
                    AND s.type = 'A股'
                ORDER BY s.code, dq.trade_date
                """
                
                self.data = pd.read_sql_query(query, conn, params=[start_date, end_date])
        
        except Exception as e:
            self.logger.error(f"❌ 数据加载失败: {e}")
            raise
        
        if self.data.empty:
            raise ValueError("❌ 未加载到有效数据")
        
        self.logger.info(f"✅ 已加载 {len(self.data):,} 条记录")
        
        # 数据预处理
        self._preprocess_data()
    
    def _preprocess_data(self) -> None:
        """数据预处理"""
        self.logger.info("🔧 数据预处理...")
        
        # 计算多时间周期未来收益率
        self.data = self.data.sort_values(['code', 'trade_date']).reset_index(drop=True)
        
        for period in [1, 3, 5, 10, 15, 20]:
            self.data[f'future_return_{period}d'] = (
                self.data.groupby('code')['close']
                .pct_change(period)
                .shift(-period)
            )
        
        # 计算衍生因子
        self._calculate_derived_factors()
        
        # 去除缺失值
        self.data = self.data.dropna(subset=[
            'rsi6', 'kdj_k', 'kdj_d', 'pe_ttm', 'pb', 'future_return_1d'
        ])
        
        self.logger.info(f"✅ 预处理完成，有效数据 {len(self.data):,} 条")
    
    def _calculate_derived_factors(self) -> None:
        """计算衍生因子"""
        # 知行趋势
        self.data['zhixing_trend'] = np.where(
            (self.data['ema12'] > 0) & (self.data['ema26'] > 0),
            self.data['ema12'] / self.data['ema26'],
            1.0
        )
        
        # 知行多均
        self.data['avg_ma'] = (self.data['ma5'] + self.data['ma10'] + self.data['ma20']) / 3
        self.data['zhixing_multiavg'] = np.where(
            self.data['avg_ma'] > 0,
            self.data['close'] / self.data['avg_ma'],
            1.0
        )
        
        # 价格动量 (简化版)
        self.data['price_momentum'] = self.data['price_change_pct'].fillna(0)
        
        # 成交量激增
        self.data['volume_surge'] = (
            self.data['volume_ratio_5d'].fillna(1.0) * 0.7 +
            self.data['volume_ratio_20d'].fillna(1.0) * 0.3
        )
        
        # 波动性风险 (反向)
        self.data['volatility_risk'] = 1.0 - np.clip(self.data['volatility_20d'].fillna(0.025), 0, 0.1) / 0.1
    
    def optimize_period_weights(self, period: str, max_evals: int = 200) -> Dict:
        """优化单个时间周期的权重"""
        self.logger.info(f"🎯 优化 {period} 时间周期权重...")
        
        # 定义权重搜索空间 (Dirichlet分布，确保权重和为1)
        def objective(weights_raw):
            try:
                # 使用softmax确保权重和为1
                weights_exp = np.exp(weights_raw)
                weights = weights_exp / np.sum(weights_exp)
                
                # 构建权重字典
                weight_dict = dict(zip(self.factors, weights))
                
                # 计算IC
                ic_scores = []
                
                # 时间序列交叉验证
                tscv = TimeSeriesSplit(n_splits=3)
                data_grouped = self.data.groupby('trade_date')
                dates = sorted(self.data['trade_date'].unique())
                
                for train_idx, test_idx in tscv.split(dates):
                    train_dates = [dates[i] for i in train_idx]
                    test_dates = [dates[i] for i in test_idx]
                    
                    train_data = self.data[self.data['trade_date'].isin(train_dates)]
                    test_data = self.data[self.data['trade_date'].isin(test_dates)]
                    
                    if len(test_data) < 100:  # 至少100个样本
                        continue
                    
                    # 计算测试集评分
                    scores = self._calculate_weighted_scores(test_data, weight_dict)
                    returns = test_data[f'future_return_{period}'].values
                    
                    # 计算IC (使用Spearman相关系数)
                    if len(scores) > 0 and len(returns) > 0:
                        ic, _ = spearmanr(scores, returns)
                        if not np.isnan(ic):
                            ic_scores.append(ic)
                
                if len(ic_scores) == 0:
                    return {'loss': 1.0, 'status': STATUS_FAIL}
                
                # 平均IC
                avg_ic = np.mean(ic_scores)
                ic_std = np.std(ic_scores) if len(ic_scores) > 1 else 0.01
                
                # 稳定性惩罚
                stability_penalty = ic_std * 2
                
                # 权重分散度奖励
                entropy = -np.sum(weights * np.log(weights + 1e-8))
                diversity_bonus = entropy / np.log(len(self.factors)) * 0.1
                
                # 目标函数 (最小化负IC)
                objective_value = -(avg_ic - stability_penalty + diversity_bonus)
                
                return {
                    'loss': objective_value,
                    'status': STATUS_OK,
                    'eval_time': datetime.now().isoformat(),
                    'avg_ic': avg_ic,
                    'ic_std': ic_std,
                    'weights': weight_dict.copy()
                }
                
            except Exception as e:
                return {'loss': 1.0, 'status': STATUS_FAIL}
        
        # 定义搜索空间 (12个因子的权重)
        space = [hp.normal(f'w{i}', 0, 1) for i in range(len(self.factors))]
        
        # 贝叶斯优化
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            verbose=False
        )
        
        # 处理最优结果
        best_weights_exp = np.exp(list(best.values()))
        best_weights_normalized = best_weights_exp / np.sum(best_weights_exp)
        best_weight_dict = dict(zip(self.factors, best_weights_normalized))
        
        # 获取最佳试验结果
        best_trial = min(trials.trials, key=lambda t: t['result']['loss'])
        best_result = best_trial['result']
        
        result = {
            'period': period,
            'best_weights': best_weight_dict,
            'best_ic': best_result.get('avg_ic', 0),
            'ic_std': best_result.get('ic_std', 0),
            'optimization_trials': len(trials.trials),
            'best_objective': -best_result['loss']
        }
        
        self.logger.info(f"✅ {period} 优化完成: IC={result['best_ic']:.4f}")
        return result
    
    def _calculate_weighted_scores(self, data: pd.DataFrame, weights: Dict) -> np.ndarray:
        """计算加权评分"""
        scores = np.zeros(len(data))
        
        for factor, weight in weights.items():
            if weight > 0 and factor in data.columns:
                factor_values = data[factor].fillna(data[factor].median()).values
                
                # 标准化到0-1范围
                if factor_values.std() > 0:
                    factor_normalized = (factor_values - factor_values.min()) / (factor_values.max() - factor_values.min())
                else:
                    factor_normalized = np.full_like(factor_values, 0.5)
                
                scores += factor_normalized * weight
        
        return scores
    
    def optimize_all_periods(self) -> Dict:
        """优化所有时间周期权重"""
        self.logger.info("🚀 开始优化所有时间周期...")
        
        results = {}
        
        for period in self.periods:
            try:
                result = self.optimize_period_weights(
                    period, 
                    max_evals=self.optimization_config['max_evals']
                )
                results[period] = result
                self.best_weights[period] = result['best_weights']
                
                # 记录IC性能
                self.ic_performance[period] = {
                    'optimized_ic': result['best_ic'],
                    'ic_std': result['ic_std'],
                    'target_ic': self.ic_targets[period],
                    'improvement_needed': self.ic_targets[period] - result['best_ic']
                }
                
            except Exception as e:
                self.logger.error(f"❌ {period} 优化失败: {e}")
                results[period] = None
        
        self.optimization_results = results
        return results
    
    def calculate_composite_ic(self) -> float:
        """计算复合IC评分"""
        if not self.ic_performance:
            return 0.0
        
        composite_ic = sum(
            self.ic_performance[period]['optimized_ic'] * self.period_importance[period]
            for period in self.periods
            if period in self.ic_performance
        )
        
        return composite_ic
    
    def generate_v353_config(self) -> Dict:
        """生成V3.53配置"""
        config = {
            'version': 'V3.53_MultiPeriod_Optimized',
            'optimization_date': datetime.now().isoformat(),
            'period_weights': self.best_weights,
            'period_importance': self.period_importance,
            'ic_performance': self.ic_performance,
            'composite_ic': self.calculate_composite_ic(),
            'optimization_config': self.optimization_config,
            'ic_targets': self.ic_targets,
            'factors': self.factors,
            'periods': self.periods
        }
        
        return config
    
    def save_results(self, filepath: str = None) -> str:
        """保存优化结果"""
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"v353_multiperiod_optimization_{timestamp}.json"
        
        config = self.generate_v353_config()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"✅ 优化结果已保存到: {filepath}")
        return filepath
    
    def generate_comparison_report(self) -> str:
        """生成对比报告"""
        report = []
        report.append("# V3.53 多时间周期IC优化结果报告")
        report.append(f"\n## 优化概览\n")
        report.append(f"- **优化日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"- **数据期间**: {self.data['trade_date'].min()} 至 {self.data['trade_date'].max()}")
        report.append(f"- **样本数量**: {len(self.data):,} 条记录")
        report.append(f"- **复合IC**: {self.calculate_composite_ic():.4f}")
        
        report.append(f"\n## 各时间周期IC表现\n")
        report.append("| 时间周期 | 目标IC | 优化后IC | IC改善 | 权重重要性 | 状态 |")
        report.append("|----------|---------|----------|---------|-----------|------|")
        
        for period in self.periods:
            if period in self.ic_performance:
                perf = self.ic_performance[period]
                target = perf['target_ic']
                actual = perf['optimized_ic']
                improvement = actual - target if target > 0 else actual
                importance = self.period_importance[period]
                status = "✅ 达标" if actual >= target else "⚠️ 待改善"
                
                report.append(f"| {period} | {target:.3f} | {actual:.3f} | {improvement:+.3f} | {importance:.1%} | {status} |")
        
        report.append(f"\n## 权重分布分析\n")
        
        for period in self.periods:
            if period in self.best_weights:
                report.append(f"\n### {period.upper()} 时间周期权重")
                weights = self.best_weights[period]
                
                # 按权重排序
                sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
                
                report.append("| 因子 | 权重 | 排名 |")
                report.append("|------|------|------|")
                
                for rank, (factor, weight) in enumerate(sorted_weights, 1):
                    if weight > 0.01:  # 只显示权重>1%的因子
                        report.append(f"| {factor} | {weight:.3f} | {rank} |")
        
        return "\n".join(report)
    
    def visualize_results(self, save_path: str = None) -> None:
        """可视化优化结果"""
        if not self.ic_performance:
            self.logger.warning("⚠️ 无优化结果可视化")
            return
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. IC表现对比
        periods = list(self.ic_performance.keys())
        target_ics = [self.ic_targets[p] for p in periods]
        actual_ics = [self.ic_performance[p]['optimized_ic'] for p in periods]
        
        x = np.arange(len(periods))
        width = 0.35
        
        ax1.bar(x - width/2, target_ics, width, label='目标IC', alpha=0.7)
        ax1.bar(x + width/2, actual_ics, width, label='优化后IC', alpha=0.7)
        ax1.set_xlabel('时间周期')
        ax1.set_ylabel('IC值')
        ax1.set_title('IC表现对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(periods)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 时间周期重要性
        importance_values = [self.period_importance[p] for p in periods]
        ax2.pie(importance_values, labels=periods, autopct='%1.1f%%', startangle=90)
        ax2.set_title('时间周期重要性权重')
        
        # 3. 权重热力图 (选择几个主要因子)
        main_factors = ['rsi6', 'kdj_k', 'price_momentum', 'pb', 'pe_ttm', 'market_cap']
        weight_matrix = []
        
        for period in periods:
            if period in self.best_weights:
                period_weights = [self.best_weights[period].get(f, 0) for f in main_factors]
                weight_matrix.append(period_weights)
        
        if weight_matrix:
            im = ax3.imshow(weight_matrix, cmap='viridis', aspect='auto')
            ax3.set_xticks(range(len(main_factors)))
            ax3.set_xticklabels(main_factors, rotation=45)
            ax3.set_yticks(range(len(periods)))
            ax3.set_yticklabels(periods)
            ax3.set_title('主要因子权重热力图')
            plt.colorbar(im, ax=ax3)
        
        # 4. 复合IC趋势 (如果有历史数据)
        composite_ic = self.calculate_composite_ic()
        ax4.bar(['复合IC'], [composite_ic], color='green', alpha=0.7)
        ax4.axhline(y=0.015, color='red', linestyle='--', label='目标线(1.5%)')
        ax4.set_ylabel('IC值')
        ax4.set_title('复合IC表现')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"v353_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"📊 可视化结果已保存到: {save_path}")
        plt.show()


def run_v353_optimization():
    """运行V3.53多时间周期优化"""
    print("🚀 V3.53 多时间周期IC优化器")
    print("="*60)
    
    # 创建优化器
    optimizer = MultiPeriodICOptimizer()
    
    # 加载数据
    print("📊 加载数据...")
    optimizer.load_and_prepare_data()
    
    # 运行优化
    print("🎯 开始多时间周期优化...")
    results = optimizer.optimize_all_periods()
    
    # 保存结果
    print("💾 保存优化结果...")
    result_file = optimizer.save_results()
    
    # 生成报告
    print("📝 生成对比报告...")
    report = optimizer.generate_comparison_report()
    
    report_file = f"v353_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 报告已保存到: {report_file}")
    
    # 可视化结果
    print("📊 生成可视化图表...")
    optimizer.visualize_results()
    
    # 显示结果摘要
    print("\n" + "="*60)
    print("🎉 V3.53优化完成！")
    print(f"📈 复合IC: {optimizer.calculate_composite_ic():.4f}")
    print(f"📁 结果文件: {result_file}")
    print(f"📝 报告文件: {report_file}")
    

if __name__ == "__main__":
    run_v353_optimization()