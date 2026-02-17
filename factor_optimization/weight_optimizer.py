#!/usr/bin/env python3
"""
可扩展权重优化框架 - 基于 factor_optimization/standard_factors.db
支持动态添加新因子和版本迭代，完全基于标准化数据结构
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple, Optional, Any
import yaml
from multiprocessing import Pool, cpu_count
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

class WeightOptimizer:
    """可扩展权重优化框架"""
    
    def __init__(self, config_file: str = None, n_processes: int = None):
        """
        初始化优化器
        Args:
            config_file: 配置文件路径，支持JSON/YAML格式
            n_processes: 并行进程数，默认为CPU核心数-1
        """
        # 数据库路径
        project_root = os.path.dirname(os.path.dirname(__file__))
        self.factor_db_path = os.path.join(project_root, 'factor_optimization', 'standard_factors.db')
        
        # 并行配置
        self.n_processes = n_processes or max(1, cpu_count() - 1)
        
        # 加载配置
        self.config = self._load_config(config_file) if config_file else self._default_config()
        
        # 验证数据库表结构
        self.available_factors = self._discover_available_factors()
        
        # 设置日志
        self._setup_logging()
        
        print(f"🔍 发现可用因子: {len(self.available_factors)} 个")
        print(f"📊 维度配置: {len(self.config['dimensions'])} 个维度")
        print(f"🚀 并行进程数: {self.n_processes}")
        
        # 数据缓存
        self._data_cache = None
    
    def _load_config(self, config_file: str) -> Dict:
        """加载配置文件"""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                return json.load(f)
    
    def _default_config(self) -> Dict:
        """默认配置 - v3.2基础配置"""
        return {
            "version": "v3.2",
            "description": "基于standard_factors.db的可扩展权重优化",
            "dimensions": {
                "technical": {
                    "description": "技术指标维度",
                    "weight_range": [0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
                    "factors": ["kdj_strength", "rsi_momentum", "bbi_trend", "volume_surge"],
                    "sub_weights": {
                        "kdj_strength": [0.25, 0.30, 0.35],
                        "rsi_momentum": [0.20, 0.25, 0.30], 
                        "bbi_trend": [0.15, 0.20, 0.25],
                        "volume_surge": [0.20, 0.25, 0.30]
                    }
                },
                "squeeze_momentum": {
                    "description": "挤压动量维度",
                    "weight_range": [0.06, 0.08, 0.10, 0.12, 0.15],
                    "factors": ["squeeze_state", "squeeze_release", "momentum_direction", "momentum_consistency"],
                    "sub_weights": {
                        "squeeze_state": [0.15, 0.20, 0.25],
                        "squeeze_release": [0.35, 0.40, 0.45],
                        "momentum_direction": [0.20, 0.25, 0.30],
                        "momentum_consistency": [0.10, 0.15, 0.20]
                    }
                },
                "fundamental": {
                    "description": "基本面维度",
                    "weight_range": [0.10, 0.12, 0.14, 0.16, 0.20],
                    "factors": ["pe_valuation", "pb_valuation", "roe_profitability", "financial_quality", "market_cap", "turnover_activity"],
                    "sub_weights": {
                        "pe_valuation": [0.25, 0.30, 0.35],
                        "pb_valuation": [0.20, 0.25, 0.30],
                        "roe_profitability": [0.15, 0.20, 0.25],
                        "financial_quality": [0.10, 0.15, 0.20],
                        "market_cap": [0.05, 0.10, 0.15],
                        "turnover_activity": [0.05, 0.10, 0.15]
                    }
                },
                "performance": {
                    "description": "市场表现维度", 
                    "weight_range": [0.10, 0.12, 0.15, 0.18, 0.22],
                    "factors": ["price_momentum", "relative_strength", "volatility_risk"],
                    "sub_weights": {
                        "price_momentum": [0.60, 0.65, 0.70],
                        "relative_strength": [0.20, 0.25, 0.30],
                        "volatility_risk": [0.10, 0.15, 0.20]
                    }
                },
                "sentiment": {
                    "description": "情绪指标维度",
                    "weight_range": [0.02, 0.04, 0.06, 0.08, 0.10],
                    "factors": ["money_flow", "market_attention", "investor_emotion"],
                    "sub_weights": {
                        "money_flow": [0.40, 0.45, 0.50],
                        "market_attention": [0.30, 0.35, 0.40], 
                        "investor_emotion": [0.15, 0.20, 0.25]
                    }
                },
                "risk_control": {
                    "description": "风险控制维度",
                    "weight_range": [0.02, 0.04, 0.06, 0.08, 0.10],
                    "factors": ["stop_loss_risk", "max_drawdown", "risk_adjusted_return"],
                    "sub_weights": {
                        "stop_loss_risk": [0.40, 0.45, 0.50],
                        "max_drawdown": [0.25, 0.30, 0.35],
                        "risk_adjusted_return": [0.20, 0.25, 0.30]
                    }
                },
                "market_regime": {
                    "description": "市场环境维度",
                    "weight_range": [0.02, 0.03, 0.04, 0.05],
                    "factors": ["market_beta", "sector_rotation", "liquidity"],
                    "sub_weights": {
                        "market_beta": [0.40, 0.50, 0.60],
                        "sector_rotation": [0.25, 0.30, 0.35],
                        "liquidity": [0.10, 0.20, 0.30]
                    }
                }
            },
            "optimization": {
                "target_periods": ["return_5d", "return_10d", "return_20d"],
                "period_weights": {"return_5d": 0.2, "return_10d": 0.3, "return_20d": 0.5},
                "evaluation_metrics": {
                    "distribution_score": 0.25,
                    "discrimination_score": 0.20,
                    "prediction_score": 0.25,
                    "high_performance": 0.15,
                    "dimension_effectiveness": 0.10,
                    "balance_score": 0.05
                },
                "distribution_targets": {
                    "score_90_plus": 0.01,
                    "score_80_90": 0.05,
                    "score_70_80": 0.12,
                    "score_below_60": 0.65
                }
            }
        }
    
    def _discover_available_factors(self) -> Dict[str, List[str]]:
        """自动发现数据库中可用的因子"""
        if not os.path.exists(self.factor_db_path):
            raise FileNotFoundError(f"标准因子数据库不存在: {self.factor_db_path}")
        
        with sqlite3.connect(self.factor_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(standard_factors)")
            columns = cursor.fetchall()
        
        # 分类因子
        factor_categories = {
            "dimension_scores": [],
            "technical_factors": [],
            "squeeze_factors": [],
            "fundamental_factors": [],
            "performance_factors": [],
            "sentiment_factors": [],
            "risk_factors": [],
            "market_factors": [],
            "return_factors": []
        }
        
        for col_info in columns:
            col_name = col_info[1]
            
            if col_name in ['id', 'stock_code', 'trade_date', 'created_at']:
                continue
            elif col_name.endswith('_score') and not any(x in col_name for x in ['kdj', 'rsi', 'bbi', 'volume', 'squeeze', 'pe', 'pb']):
                factor_categories["dimension_scores"].append(col_name)
            elif col_name in ['kdj_strength', 'rsi_momentum', 'bbi_trend', 'volume_surge']:
                factor_categories["technical_factors"].append(col_name)
            elif col_name in ['squeeze_state', 'squeeze_release', 'momentum_direction', 'momentum_consistency']:
                factor_categories["squeeze_factors"].append(col_name)
            elif col_name in ['pe_valuation', 'pb_valuation', 'roe_profitability', 'financial_quality', 'market_cap', 'turnover_activity']:
                factor_categories["fundamental_factors"].append(col_name)
            elif col_name in ['price_momentum', 'relative_strength', 'volatility_risk']:
                factor_categories["performance_factors"].append(col_name)
            elif col_name in ['money_flow', 'market_attention', 'investor_emotion']:
                factor_categories["sentiment_factors"].append(col_name)
            elif col_name in ['stop_loss_risk', 'max_drawdown', 'risk_adjusted_return']:
                factor_categories["risk_factors"].append(col_name)
            elif col_name in ['market_beta', 'sector_rotation', 'liquidity']:
                factor_categories["market_factors"].append(col_name)
            elif col_name.startswith('return_'):
                factor_categories["return_factors"].append(col_name)
        
        return factor_categories
    
    def _setup_logging(self):
        """设置日志系统"""
        # 确保reports目录存在
        reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        # 配置日志
        log_file = os.path.join(reports_dir, f'optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, encoding='utf-8')
            ],
            force=True  # 强制重新配置
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"日志文件: {log_file}")
    
    def add_new_dimension(self, dimension_name: str, dimension_config: Dict) -> None:
        """动态添加新维度"""
        print(f"🆕 添加新维度: {dimension_name}")
        
        # 验证配置格式
        required_keys = ['description', 'weight_range', 'factors']
        if not all(key in dimension_config for key in required_keys):
            raise ValueError(f"维度配置缺少必要字段: {required_keys}")
        
        # 验证因子是否存在于数据库
        available_factors = set()
        for category in self.available_factors.values():
            available_factors.update(category)
        
        missing_factors = set(dimension_config['factors']) - available_factors
        if missing_factors:
            print(f"⚠️  警告: 以下因子在数据库中不存在: {missing_factors}")
        
        # 添加维度
        self.config['dimensions'][dimension_name] = dimension_config
        print(f"✅ 成功添加维度 {dimension_name}，包含因子: {dimension_config['factors']}")
    
    def add_new_factor_to_dimension(self, dimension_name: str, factor_name: str, 
                                   sub_weight_range: List[float] = None) -> None:
        """向现有维度添加新因子"""
        if dimension_name not in self.config['dimensions']:
            raise ValueError(f"维度 {dimension_name} 不存在")
        
        # 验证因子是否在数据库中存在
        available_factors = set()
        for category in self.available_factors.values():
            available_factors.update(category)
        
        if factor_name not in available_factors:
            print(f"⚠️  警告: 因子 {factor_name} 在数据库中不存在")
        
        # 添加因子
        self.config['dimensions'][dimension_name]['factors'].append(factor_name)
        
        if sub_weight_range and 'sub_weights' in self.config['dimensions'][dimension_name]:
            self.config['dimensions'][dimension_name]['sub_weights'][factor_name] = sub_weight_range
        
        print(f"✅ 成功向维度 {dimension_name} 添加因子 {factor_name}")
    
    def load_optimization_data(self, start_date: str, end_date: str, 
                             max_samples: Optional[int] = None,
                             target_trading_days: Optional[int] = None,
                             random_sample: bool = False) -> pd.DataFrame:
        """加载优化数据"""
        print(f"📊 加载优化数据 ({start_date} 到 {end_date})")
        if target_trading_days:
            print(f"🎯 目标交易日数: {target_trading_days} 天")
        elif max_samples:
            print(f"📊 样本限制: {max_samples} 条记录")
        
        print(f"🔧 连接数据库: {os.path.basename(self.factor_db_path)}")
        with sqlite3.connect(self.factor_db_path) as conn:
            # 如果指定了目标交易日数，先获取交易日
            if target_trading_days:
                # 获取指定范围内的所有交易日
                date_query = """
                SELECT DISTINCT trade_date 
                FROM standard_factors 
                WHERE trade_date >= ? AND trade_date <= ?
                ORDER BY trade_date
                """
                dates_df = pd.read_sql_query(date_query, conn, 
                                           params=[start_date, end_date])
                
                if len(dates_df) == 0:
                    raise ValueError(f"未找到 {start_date} 到 {end_date} 的交易日数据")
                
                all_dates = dates_df['trade_date'].tolist()
                
                if random_sample and len(all_dates) > target_trading_days:
                    # 随机采样指定数量的交易日
                    import random
                    random.seed(42)  # 固定随机种子
                    target_dates = random.sample(all_dates, target_trading_days)
                    target_dates.sort()
                    print(f"🎲 随机采样了 {len(target_dates)} 个交易日: {target_dates[0]} 到 {target_dates[-1]}")
                    print(f"📋 采样范围: 从{len(all_dates)}个交易日中随机选择")
                else:
                    # 取最近的N个交易日
                    target_dates = all_dates[-target_trading_days:] if len(all_dates) >= target_trading_days else all_dates
                    print(f"🗓️  选择了 {len(target_dates)} 个交易日: {target_dates[0]} 到 {target_dates[-1]}")
                
                # 使用IN子句查询这些日期的数据
                date_placeholders = ','.join(['?' for _ in target_dates])
            
            # 构建查询 - 选择所有相关列
            dimension_scores = ", ".join(self.available_factors["dimension_scores"])
            factor_columns = []
            
            for category in ["technical_factors", "squeeze_factors", "fundamental_factors", 
                           "performance_factors", "sentiment_factors", "risk_factors", "market_factors"]:
                factor_columns.extend(self.available_factors[category])
            
            return_columns = ", ".join(self.available_factors["return_factors"])
            all_factors = ", ".join(factor_columns)
            
            if target_trading_days:
                query = f"""
                SELECT stock_code, trade_date,
                       {dimension_scores},
                       {all_factors},
                       {return_columns}
                FROM standard_factors 
                WHERE trade_date IN ({date_placeholders})
                AND technical_score IS NOT NULL
                ORDER BY trade_date DESC, stock_code
                """
                query_params = target_dates
            else:
                query = f"""
                SELECT stock_code, trade_date,
                       {dimension_scores},
                       {all_factors},
                       {return_columns}
                FROM standard_factors 
                WHERE trade_date >= ? AND trade_date <= ?
                AND technical_score IS NOT NULL
                ORDER BY trade_date DESC, stock_code
                """
                query_params = [start_date, end_date]
                
                if max_samples:
                    query += f" LIMIT {max_samples}"
            
            print(f"🚀 执行查询中...")
            df = pd.read_sql_query(query, conn, params=query_params)
            print(f"💾 数据查询完成")
        
        if len(df) == 0:
            raise ValueError(f"未找到符合条件的数据")
        
        print(f"✅ 成功加载 {len(df)} 条记录")
        print(f"📈 覆盖 {df['stock_code'].nunique()} 只股票，{df['trade_date'].nunique()} 个交易日")
        
        return df
    
    def calculate_weighted_score(self, row: pd.Series, weights: Dict) -> float:
        """计算加权评分 - 支持灵活的权重结构"""
        total_score = 0.0
        total_weight = 0.0
        
        for dimension_name, dimension_weight in weights.items():
            if dimension_name not in self.config['dimensions']:
                continue
            
            dimension_config = self.config['dimensions'][dimension_name]
            dimension_score = 0.0
            dimension_total_weight = 0.0
            
            # 如果有子权重配置，使用子因子加权
            if 'sub_weights' in dimension_config and dimension_name in weights.get('sub_weights', {}):
                sub_weights = weights['sub_weights'][dimension_name]
                
                for factor_name, factor_weight in sub_weights.items():
                    if factor_name in row and pd.notna(row[factor_name]):
                        dimension_score += row[factor_name] * factor_weight
                        dimension_total_weight += factor_weight
            
            # 否则使用维度总分
            else:
                score_field = f"{dimension_name}_score"
                if score_field in row and pd.notna(row[score_field]):
                    dimension_score = row[score_field]
                    dimension_total_weight = 1.0
            
            # 归一化维度评分并加权
            if dimension_total_weight > 0:
                normalized_dimension_score = dimension_score / dimension_total_weight
                total_score += normalized_dimension_score * dimension_weight
                total_weight += dimension_weight
        
        return total_score / total_weight if total_weight > 0 else 50.0
    
    def evaluate_weights(self, df: pd.DataFrame, weights: Dict) -> Dict:
        """评估权重组合效果"""
        if len(df) < 100:
            return {'overall_score': 0, 'error': 'insufficient_data'}
        
        # 计算加权评分
        df_copy = df.copy()
        df_copy['weighted_score'] = df_copy.apply(
            lambda row: self.calculate_weighted_score(row, weights), axis=1
        )
        
        # 过滤异常值
        df_copy = df_copy[
            (df_copy['weighted_score'] >= 10) & 
            (df_copy['weighted_score'] <= 100)
        ]
        
        if len(df_copy) < 50:
            return {'overall_score': 0, 'error': 'insufficient_valid_data'}
        
        metrics = {}
        evaluation_config = self.config['optimization']['evaluation_metrics']
        
        # 1. 分布合理性
        distribution_targets = self.config['optimization']['distribution_targets']
        score_90_plus = (df_copy['weighted_score'] >= 90).mean()
        score_80_90 = ((df_copy['weighted_score'] >= 80) & (df_copy['weighted_score'] < 90)).mean()
        score_70_80 = ((df_copy['weighted_score'] >= 70) & (df_copy['weighted_score'] < 80)).mean()
        score_below_60 = (df_copy['weighted_score'] < 60).mean()
        
        distribution_score = 1.0
        distribution_score -= max(0, (score_90_plus - distribution_targets['score_90_plus']) * 50)
        distribution_score -= max(0, (score_80_90 - distribution_targets['score_80_90']) * 30)
        distribution_score -= max(0, (score_70_80 - distribution_targets['score_70_80']) * 20)
        distribution_score -= max(0, (distribution_targets['score_below_60'] - score_below_60) * 10)
        distribution_score = max(0, distribution_score)
        
        metrics['distribution_score'] = distribution_score
        
        # 2. 区分度
        score_std = df_copy['weighted_score'].std()
        discrimination_score = min(1.0, score_std / 15.0)
        metrics['discrimination_score'] = discrimination_score
        
        # 3. 预测能力
        target_periods = self.config['optimization']['target_periods']
        period_weights = self.config['optimization']['period_weights']
        
        prediction_score = 0.0
        total_period_weight = 0.0
        
        for period in target_periods:
            if period in df_copy.columns:
                valid_data = df_copy[[period, 'weighted_score']].dropna()
                if len(valid_data) > 20:
                    corr = valid_data[period].corr(valid_data['weighted_score'])
                    if pd.notna(corr):
                        period_weight = period_weights.get(period, 0.2)
                        prediction_score += abs(corr) * period_weight
                        total_period_weight += period_weight
        
        if total_period_weight > 0:
            prediction_score = prediction_score / total_period_weight
        
        metrics['prediction_score'] = prediction_score
        
        # 4. 高分表现
        high_score_stocks = df_copy[df_copy['weighted_score'] >= 75]
        if len(high_score_stocks) > 5 and 'return_10d' in df_copy.columns:
            avg_return = high_score_stocks['return_10d'].mean()
            high_performance = max(0, min(1.0, avg_return * 10 + 0.5))
        else:
            high_performance = 0.3
        
        metrics['high_performance'] = high_performance
        
        # 5. 维度有效性 (新维度的独立贡献)
        dimension_effectiveness = 0.5  # 默认值
        metrics['dimension_effectiveness'] = dimension_effectiveness
        
        # 6. 权重平衡性
        total_weight = sum(weights.values())
        weight_penalty = abs(total_weight - 1.0) * 2
        max_weight = max(weights.values()) if weights else 0
        if max_weight > 0.6:
            weight_penalty += (max_weight - 0.6) * 3
        
        balance_score = max(0, 1.0 - weight_penalty)
        metrics['balance_score'] = balance_score
        
        # 综合评分
        overall_score = (
            distribution_score * evaluation_config['distribution_score'] +
            discrimination_score * evaluation_config['discrimination_score'] +
            prediction_score * evaluation_config['prediction_score'] +
            high_performance * evaluation_config['high_performance'] +
            dimension_effectiveness * evaluation_config['dimension_effectiveness'] +
            balance_score * evaluation_config['balance_score']
        )
        
        metrics['overall_score'] = overall_score
        return metrics
    
    def _evaluate_weight_batch(self, weight_combinations: List[Dict], df_data: Dict) -> List[Dict]:
        """评估权重组合批次（用于并行处理）"""
        results = []
        for weights in weight_combinations:
            # 重建DataFrame
            df = pd.DataFrame(df_data)
            metrics = self.evaluate_weights(df, weights)
            
            if 'error' not in metrics:
                results.append({
                    'weights': weights,
                    'score': metrics['overall_score'],
                    'metrics': metrics
                })
        return results
    
    def optimize_weights_parallel(self, start_date: str = '2024-01-01', 
                                end_date: str = '2025-08-25',
                                max_samples: Optional[int] = None,
                                target_trading_days: Optional[int] = None,
                                early_stop_threshold: float = 0.95,
                                convergence_patience: int = 50) -> Dict:
        """并行权重优化（优化版）"""
        print(f"🚀 启动并行权重优化 (进程数: {self.n_processes})")
        print(f"📅 时间范围: {start_date} 到 {end_date}")
        print(f"📊 版本: {self.config['version']}")
        
        # 加载数据（一次性加载到内存）
        df = self.load_optimization_data(start_date, end_date, max_samples, target_trading_days, random_sample=True)
        print(f"💾 数据加载完成：{len(df)} 条记录")
        
        # 转换为字典格式以便序列化
        df_data = df.to_dict('records')
        
        # 生成权重组合
        dimension_names = list(self.config['dimensions'].keys())
        weight_ranges = [self.config['dimensions'][dim]['weight_range'] 
                        for dim in dimension_names]
        
        # 过滤权重和接近1.0的组合
        valid_combinations = []
        total_combinations = np.prod([len(r) for r in weight_ranges])
        
        print(f"🔍 生成有效权重组合...")
        for combination in product(*weight_ranges):
            weights = dict(zip(dimension_names, combination))
            total_weight = sum(weights.values())
            
            # 只保留权重和在合理范围内的组合
            if 0.95 <= total_weight <= 1.05:
                valid_combinations.append(weights)
        
        print(f"✅ 有效组合数: {len(valid_combinations)} / {total_combinations}")
        
        # 分批处理
        batch_size = max(1, len(valid_combinations) // (self.n_processes * 4))
        batches = [valid_combinations[i:i + batch_size] 
                  for i in range(0, len(valid_combinations), batch_size)]
        
        print(f"📦 分成 {len(batches)} 批次，每批 {batch_size} 个组合")
        
        # 并行处理
        best_score = -1
        best_weights = None
        best_metrics = None
        all_results = []
        processed_count = 0
        stagnant_count = 0
        
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            # 提交所有批次
            future_to_batch = {
                executor.submit(self._evaluate_weight_batch, batch, df_data): i 
                for i, batch in enumerate(batches)
            }
            
            print(f"⚡ 开始并行处理...")
            
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    processed_count += len(batches[batch_idx])
                    
                    # 更新最佳结果
                    for result in batch_results:
                        score = result['score']
                        if score > best_score:
                            best_score = score
                            best_weights = result['weights']
                            best_metrics = result['metrics']
                            stagnant_count = 0  # 重置停滞计数
                            
                            print(f"🎯 NEW BEST! 评分: {best_score:.4f}")
                            weights_str = " | ".join([f"{k}: {v:.1%}" for k, v in best_weights.items()])
                            print(f"   权重: {weights_str}")
                        else:
                            stagnant_count += 1
                    
                    # 进度报告
                    progress = processed_count / len(valid_combinations) * 100
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    eta = (len(valid_combinations) - processed_count) / rate / 60 if rate > 0 else 0
                    
                    print(f"📊 进度: {progress:.1f}% ({processed_count}/{len(valid_combinations)}) | "
                          f"⚡ 速度: {rate:.0f}/s | ⏰ 剩余: {eta:.1f}分钟 | 🏆 最佳: {best_score:.4f}")
                    
                    # 早停条件
                    if best_score >= early_stop_threshold:
                        print(f"🎉 达到早停阈值 {early_stop_threshold}，提前结束!")
                        break
                    
                    if stagnant_count >= convergence_patience:
                        print(f"📈 连续 {convergence_patience} 批次无改进，认为已收敛!")
                        break
                        
                except Exception as e:
                    print(f"❌ 批次 {batch_idx} 处理失败: {e}")
        
        if best_weights is None:
            raise ValueError("未找到有效的权重组合")
        
        # 整理结果
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        elapsed_total = time.time() - start_time
        print(f"\n🎯 并行优化完成!")
        print(f"⏱️  总用时: {elapsed_total/60:.1f} 分钟")
        print(f"📊 评估了 {len(all_results)} 个权重组合")
        print(f"🏆 最佳综合评分: {best_score:.4f}")
        print(f"📋 最佳权重配置:")
        for dim, weight in best_weights.items():
            print(f"   {dim}: {weight:.1%}")
        
        # 保存结果到reports目录
        reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = os.path.join(reports_dir, f'parallel_optimization_{timestamp}.json')
        
        result = {
            'success': True,
            'method': 'parallel_optimization',
            'version': self.config['version'],
            'best_weights': best_weights,
            'best_score': best_score,
            'best_metrics': best_metrics,
            'total_evaluated': len(all_results),
            'processing_time_minutes': elapsed_total / 60,
            'n_processes': self.n_processes,
            'early_stopped': best_score >= early_stop_threshold or stagnant_count >= convergence_patience,
            'optimization_results': all_results[:50],  # 保存前50个结果
            'data_stats': {
                'total_records': len(df),
                'unique_stocks': df['stock_code'].nunique(),
                'date_range': f"{df['trade_date'].min()} to {df['trade_date'].max()}",
                'available_factors': self.available_factors
            },
            'config': self.config,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"💾 优化结果已保存到: {result_file}")
        
        return result

    def optimize_weights(self, start_date: str = '2024-01-01', 
                        end_date: str = '2025-08-25',
                        max_samples: Optional[int] = None) -> Dict:
        """执行权重优化"""
        print(f"🚀 启动可扩展权重优化")
        print(f"📅 时间范围: {start_date} 到 {end_date}")
        print(f"📊 版本: {self.config['version']}")
        
        # 加载数据
        df = self.load_optimization_data(start_date, end_date, max_samples)
        
        # 生成权重组合
        dimension_names = list(self.config['dimensions'].keys())
        weight_ranges = [self.config['dimensions'][dim]['weight_range'] 
                        for dim in dimension_names]
        
        total_combinations = np.prod([len(r) for r in weight_ranges])
        print(f"🔍 开始权重搜索，总组合数: {total_combinations}")
        
        best_score = -1
        best_weights = None
        best_metrics = None
        optimization_results = []
        evaluated_count = 0
        skipped_count = 0
        
        import time
        start_time = time.time()
        last_log_time = start_time
        log_interval = 5  # 每5秒记录一次进度
        
        print(f"⏱️  优化开始时间: {time.strftime('%H:%M:%S')}")
        print(f"🎮 估计完成时间: {(total_combinations * 0.01 / 60):.1f} 分钟")
        print("="*60)
        
        for i, combination in enumerate(product(*weight_ranges)):
            weights = dict(zip(dimension_names, combination))
            
            # 权重和检查
            total_weight = sum(weights.values())
            if total_weight > 0.98:  # 允许小误差
                skipped_count += 1
                continue
            
            # 评估权重组合
            metrics = self.evaluate_weights(df, weights)
            
            if 'error' not in metrics:
                evaluated_count += 1
                score = metrics['overall_score']
                
                if score > best_score:
                    best_score = score
                    best_weights = weights.copy()
                    best_metrics = metrics.copy()
                    print(f"🎯 NEW BEST! 评分: {best_score:.4f}")
                    weights_str = " | ".join([f"{k}: {v:.1%}" for k, v in best_weights.items()])
                    print(f"   权重: {weights_str}")
                    print(f"   时间: {time.strftime('%H:%M:%S')} | 进度: {((i+1)/total_combinations)*100:.1f}%")
                    print("-"*60)
                
                optimization_results.append({
                    'weights': weights.copy(),
                    'score': score,
                    'metrics': metrics.copy()
                })
            
            # 时间间隔进度报告
            current_time = time.time()
            if current_time - last_log_time >= log_interval:
                elapsed = current_time - start_time
                progress = (i + 1) / total_combinations * 100
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta_seconds = (total_combinations - i - 1) / rate if rate > 0 else 0
                eta_minutes = eta_seconds / 60
                
                print(f"📊 [{time.strftime('%H:%M:%S')}] 进度: {progress:.2f}% ({i+1}/{total_combinations})")
                print(f"   ✅ 已评估: {evaluated_count} | ❌ 跳过: {skipped_count}")
                print(f"   ⚡ 速度: {rate:.1f} 组合/秒 | ⏰ 预计剩余: {eta_minutes:.1f} 分钟")
                print(f"   🏆 当前最佳评分: {best_score:.4f}")
                print()
                last_log_time = current_time
            
            # 每10%进度的详细报告
            if (i + 1) % max(1, total_combinations // 10) == 0:
                progress = (i + 1) / total_combinations * 100
                elapsed = time.time() - start_time
                print(f"🚀 里程碑: {progress:.0f}% 完成!")
                print(f"   ⏱️  已用时间: {elapsed/60:.1f} 分钟")
                print(f"   📈 最佳评分: {best_score:.4f}")
                print(f"   🔍 有效评估: {evaluated_count} 组合")
                print("="*60)
        
        if best_weights is None:
            raise ValueError("未找到有效的权重组合")
        
        print(f"\n🎯 优化完成!")
        print(f"📊 评估了 {evaluated_count} 个权重组合")
        print(f"🏆 最佳综合评分: {best_score:.4f}")
        print(f"📋 最佳权重配置:")
        for dim, weight in best_weights.items():
            print(f"   {dim}: {weight:.1%}")
        
        result = {
            'success': True,
            'version': self.config['version'],
            'best_weights': best_weights,
            'best_score': best_score,
            'best_metrics': best_metrics,
            'total_evaluated': evaluated_count,
            'optimization_results': sorted(optimization_results, 
                                         key=lambda x: x['score'], reverse=True)[:50],
            'data_stats': {
                'total_records': len(df),
                'unique_stocks': df['stock_code'].nunique(),
                'date_range': f"{df['trade_date'].min()} to {df['trade_date'].max()}",
                'available_factors': self.available_factors
            },
            'config': self.config,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return result
    
    def save_config(self, output_path: str) -> None:
        """保存当前配置"""
        with open(output_path, 'w', encoding='utf-8') as f:
            if output_path.endswith('.yaml') or output_path.endswith('.yml'):
                yaml.dump(self.config, f, ensure_ascii=False, indent=2)
            else:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        
        print(f"💾 配置已保存到: {output_path}")

def main():
    """主函数 - 支持命令行参数的并行优化版"""
    import argparse
    
    parser = argparse.ArgumentParser(description='并行权重优化器')
    parser.add_argument('--start-date', default='2024-01-01', 
                       help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2025-08-20',
                       help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='最大样本数量 (优先级低于trading-days)')
    parser.add_argument('--trading-days', type=int, default=100,
                       help='目标交易日数量')
    parser.add_argument('--random-sample', action='store_true',
                       help='使用随机采样而非连续日期')
    parser.add_argument('--processes', type=int, default=None,
                       help='并行进程数 (默认为CPU核心数-1)')
    parser.add_argument('--early-stop', type=float, default=0.80,
                       help='早停阈值 (0.0-1.0)')
    parser.add_argument('--patience', type=int, default=20,
                       help='收敛耐心值')
    
    args = parser.parse_args()
    
    print("🚀 启动并行权重优化系统")
    print(f"📅 时间范围: {args.start_date} 到 {args.end_date}")
    print(f"🎯 目标交易日: {args.trading_days}")
    print(f"🔧 早停阈值: {args.early_stop}")
    print(f"⏳ 收敛耐心: {args.patience}")
    
    # 初始化优化器
    optimizer = WeightOptimizer(n_processes=args.processes)
    
    try:
        # 使用并行优化
        result = optimizer.optimize_weights_parallel(
            start_date=args.start_date,
            end_date=args.end_date,
            max_samples=args.max_samples,
            target_trading_days=args.trading_days,
            early_stop_threshold=args.early_stop,
            convergence_patience=args.patience
        )
        
        print(f"\n✅ 优化成功完成!")
        print(f"🏆 最佳权重组合:")
        for dim, weight in result['best_weights'].items():
            print(f"   {dim}: {weight:.1%}")
        print(f"📈 最佳评分: {result['best_score']:.4f}")
        print(f"⏱️  用时: {result['processing_time_minutes']:.1f} 分钟")
        print(f"📊 评估组合数: {result['total_evaluated']}")
        
        return result
        
    except Exception as e:
        print(f"❌ 优化过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()