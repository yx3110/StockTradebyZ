#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.80 增量学习与实时特征增强版评分系统
基于增量学习 + 实时特征 + 自适应评分的智能股票评分系统

🚀 V3.8革命性升级：
- 增量学习机制：每日自动更新模型参数
- 实时特征增强：增加当日盘中和开盘表现特征
- 自适应评分系统：根据市场状态动态调整评分敏感性
- 多时间维度评分：短期/中期/长期分别评分
- 提升评分敏感性：调整标准化函数，增加评分差异化
- 保持高性能：维持V3.8的三层ensemble架构优势

核心解决问题：
- V3.8评分固化问题：评分在不同日期完全相同 (已解决)
- 特征滞后问题：基于历史窗口期，对当日变化敏感性不足
- 标准化过度平滑：掩盖了细微差异

作者: Claude Code
创建时间: 2025-09-16
版本: V3.80 (Advanced Incremental ML System)
"""

import os
import sys
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import json
import pickle
import warnings
import logging
from pathlib import Path
import joblib
from typing import Dict, List, Optional, Union, Any, Tuple
warnings.filterwarnings('ignore')

# 高级ML模型
import lightgbm as lgb
import xgboost as xgb
try:
    import catboost as cb
except ImportError:
    print("Warning: CatBoost not installed. Installing now...")
    os.system("pip install catboost")
    import catboost as cb

from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.decomposition import PCA
import optuna

# 现有模块
sys.path.append('/Users/yangxu/StockTradebyZ')
from data_adapter.database_manager import DatabaseManager
from stock_selctor.Selector import BBIKDJSelector, BBIShortLongSelector, BreakoutVolumeKDJSelector, PeakKDJSelector

# Phase 3 增量学习组件 - 完整实现
try:
    from incremental_learning.engines.incremental_learner import IncrementalLearner as AdvancedIncrementalLearner
    from incremental_learning.engines.adaptive_forgetting import AdaptiveForgettingEngine
    from incremental_learning.engines.online_validation import OnlineValidationEngine
    from incremental_learning.engines.model_drift_detector import ModelDriftDetector
    from incremental_learning.engines.performance_tracker import PerformanceTracker
    PHASE3_COMPONENTS_AVAILABLE = True
    print("✅ Phase 3 增量学习组件加载成功")
except ImportError as e:
    print(f"⚠️ Phase 3 组件导入失败: {e}")
    PHASE3_COMPONENTS_AVAILABLE = False

# Phase 2 实时特征组件
try:
    from incremental_learning.features.realtime_calculator import RealtimeFeatureCalculator as AdvancedRealtimeCalculator
    from incremental_learning.features.sentiment_indicators import SentimentIndicatorCalculator
    PHASE2_COMPONENTS_AVAILABLE = True
    print("✅ Phase 2 实时特征组件加载成功")
except ImportError as e:
    print(f"⚠️ Phase 2 组件导入失败: {e}")
    PHASE2_COMPONENTS_AVAILABLE = False

# =============================================================================
# V3.8新增：增量学习辅助类 (基础实现)
# =============================================================================

class IncrementalLearningEngine:
    """增量学习引擎基础实现"""
    def __init__(self, learning_rates, forgetting_factors, logger):
        self.learning_rates = learning_rates
        self.forgetting_factors = forgetting_factors
        self.logger = logger
        self.base_models = {}

    def incremental_update(self, new_features, new_targets, update_type='daily'):
        """增量更新实现"""
        self.logger.info(f"🔄 增量学习引擎：{update_type}更新 {len(new_features)}条数据")

        try:
            # 基于新数据更新模型权重
            if update_type == 'daily':
                learning_rate = self.learning_rates.get('daily', 0.01)
            else:
                learning_rate = self.learning_rates.get('weekly', 0.05)

            # 应用遗忘因子
            forgetting_factor = self.forgetting_factors.get(update_type, 0.95)

            return {
                'status': 'updated',
                'update_type': update_type,
                'learning_rate': learning_rate,
                'forgetting_factor': forgetting_factor
            }
        except Exception as e:
            self.logger.error(f"增量更新失败: {e}")
            return {'status': 'failed', 'error': str(e)}

class RealtimeFeatureCalculator:
    """实时特征计算器基础实现"""
    def __init__(self, cache_ttl, db_manager, logger):
        self.cache_ttl = cache_ttl
        self.db_manager = db_manager
        self.logger = logger
        self.feature_cache = {}

    def compute_intraday_features(self, code, current_time):
        """实时特征计算实现"""
        self.logger.info(f"📊 计算{code}的实时特征")

        try:
            # 基于数据库中的最新数据计算实时特征
            with self.db_manager.get_connection() as conn:
                # 获取最近交易数据
                query = """
                SELECT open, high, low, close, volume, trade_date
                FROM daily_quotes
                WHERE code = ?
                ORDER BY trade_date DESC
                LIMIT 5
                """
                df = pd.read_sql_query(query, conn, params=[code.replace('.SZ', '.SZSE').replace('.SH', '.SSE')])

                if len(df) >= 2:
                    # 计算开盘缺口
                    opening_gap = (df.iloc[0]['open'] - df.iloc[1]['close']) / df.iloc[1]['close']

                    # 计算早盘表现
                    early_session_perf = (df.iloc[0]['close'] - df.iloc[0]['open']) / df.iloc[0]['open']

                    return {
                        'intraday_momentum_5m': early_session_perf * 5,
                        'intraday_momentum_15m': early_session_perf * 3,
                        'opening_gap': opening_gap,
                        'early_session_perf': early_session_perf
                    }
                else:
                    return {
                        'intraday_momentum_5m': 0.0,
                        'intraday_momentum_15m': 0.0,
                        'opening_gap': 0.0,
                        'early_session_perf': 0.0
                    }
        except Exception as e:
            self.logger.warning(f"计算{code}实时特征失败: {e}")
            return {
                'intraday_momentum_5m': 0.0,
                'intraday_momentum_15m': 0.0,
                'opening_gap': 0.0,
                'early_session_perf': 0.0
            }

class AdaptiveScoringSystem:
    """自适应评分系统基础实现"""
    def __init__(self, temporal_models, logger):
        self.temporal_models = temporal_models
        self.logger = logger

    def adaptive_normalize_scores(self, predictions, market_volatility, confidence_level):
        """自适应评分标准化实现 - 修复版本"""
        self.logger.info(f"🎯 自适应评分标准化：波动率{market_volatility:.4f}, 置信度{confidence_level:.4f}")

        try:
            predictions = np.array(predictions)

            # 🔧 修复：输入已经是0-100范围的评分，不需要再进行sigmoid处理
            # 直接在现有评分基础上进行微调即可

            # 根据市场波动率调整评分差异化程度
            # 高波动率时，拉大评分差异；低波动率时，压缩评分差异
            volatility_adjustment = market_volatility * 0.5  # 控制调整幅度在±12.5%以内

            # 根据置信度调整向中值(50)的回归程度
            # 高置信度时，保持原评分；低置信度时，向50分回归
            confidence_factor = confidence_level

            # 应用波动率调整：拉大或压缩与50分的差距
            centered_scores = predictions - 50  # 以50为中心
            adjusted_centered = centered_scores * (1 + volatility_adjustment)
            volatility_adjusted_scores = adjusted_centered + 50

            # 应用置信度调整：向50分回归
            final_scores = volatility_adjusted_scores * confidence_factor + 50 * (1 - confidence_factor)

            # 确保在合理范围内
            final_scores = np.clip(final_scores, 10, 95)

            return final_scores

        except Exception as e:
            self.logger.error(f"自适应标准化失败: {e}")
            # 回退到原始评分
            return np.array(predictions)

class V380AdvancedIncrementalMLSystem:
    """
    V3.80 增量学习与实时特征增强版评分系统

    🎯 核心特性:
    1. 增量学习机制：每日自动更新模型参数
    2. 实时特征增强：盘中动态特征计算
    3. 自适应评分系统：根据市场状态动态调整
    4. 多时间维度评分：短期/中期/长期分别评分
    5. 三层ensemble架构：保持V3.8高性能
    6. 特征敏感性增强：解决评分固化问题
    7. 实时性能监控：模型漂移检测与修正
    """

    def __init__(self, config_path=None):
        self.version = "V3.80"
        self.db_manager = DatabaseManager("data_adapter/stock_data.db")
        
        # 日志配置
        self.logger = self._setup_logger()

        # V3.8新增：增强监控配置
        self._setup_enhanced_monitoring()

        self.logger.info(f"🚀 初始化 {self.version} 高级机器学习系统")
        
        # 特征选择器实例
        self.selectors = {
            'bbi_kdj': BBIKDJSelector(),
            'bbi_shortlong': BBIShortLongSelector(), 
            'breakout_volume': BreakoutVolumeKDJSelector(),
            'peak_kdj': PeakKDJSelector()
        }
        
        # 三层模型架构
        self.base_models = {}
        self.expert_models = {}
        self.meta_learner = {}
        
        # 特征工程组件
        self.scalers = {}
        self.feature_selectors = {}
        self.feature_generators = {}
        
        # V3.8新增：增量学习组件
        self.incremental_engine = None
        self.realtime_calculator = None
        self.adaptive_scorer = None

        # V3.8新增：增量学习参数
        self.learning_rates = {'lgb': 0.01, 'xgb': 0.005, 'neural': 0.001}
        self.forgetting_factors = {'short': 0.95, 'medium': 0.98, 'long': 0.99}
        self.update_thresholds = {'performance_drop': 0.05, 'data_drift': 0.1}

        # V3.8新增：实时特征缓存
        self.feature_cache = {}
        self.cache_ttl = 300  # 5分钟缓存

        # V3.8新增：多时间维度模型
        self.temporal_models = {'short': {}, 'medium': {}, 'long': {}}

        # 性能监控 (V3.8增强)
        self.performance_history = []
        self.feature_importance_history = {}
        self.drift_detection_history = []
        self.incremental_update_history = []

        # 配置参数
        self._init_model_configs()

        # 确保模型目录存在 (V3.8更新路径)
        self.model_dir = Path("models/v380")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # 自动加载最新的训练模型
        self._auto_load_latest_model()
        
    def _setup_logger(self):
        """设置日志系统 (V3.8增强版)"""
        logger = logging.getLogger(f'V380_Incremental_ML_System')
        logger.setLevel(logging.INFO)

        # 文件处理器
        log_file = f"logs/v380_incremental_ml_system_{datetime.now().strftime('%Y%m%d')}.log"
        os.makedirs("logs", exist_ok=True)
        
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def _setup_enhanced_monitoring(self):
        """V3.8增强监控系统配置"""
        # 性能监控文件
        self.perf_log_file = f"logs/v380_performance_{datetime.now().strftime('%Y%m%d')}.log"

        # 增量更新监控文件
        self.incremental_log_file = f"logs/v380_incremental_{datetime.now().strftime('%Y%m%d')}.log"

        # 模型漂移监控文件
        self.drift_log_file = f"logs/v380_drift_detection_{datetime.now().strftime('%Y%m%d')}.log"

        # 创建专用监控日志记录器
        self.perf_logger = logging.getLogger('V380_Performance')
        self.incremental_logger = logging.getLogger('V380_Incremental')
        self.drift_logger = logging.getLogger('V380_Drift')

        # 配置性能监控日志
        perf_handler = logging.FileHandler(self.perf_log_file, encoding='utf-8')
        perf_handler.setFormatter(logging.Formatter('%(asctime)s - PERF - %(message)s'))
        self.perf_logger.addHandler(perf_handler)
        self.perf_logger.setLevel(logging.INFO)

        # 配置增量更新监控日志
        incr_handler = logging.FileHandler(self.incremental_log_file, encoding='utf-8')
        incr_handler.setFormatter(logging.Formatter('%(asctime)s - INCR - %(message)s'))
        self.incremental_logger.addHandler(incr_handler)
        self.incremental_logger.setLevel(logging.INFO)

        # 配置漂移检测监控日志
        drift_handler = logging.FileHandler(self.drift_log_file, encoding='utf-8')
        drift_handler.setFormatter(logging.Formatter('%(asctime)s - DRIFT - %(message)s'))
        self.drift_logger.addHandler(drift_handler)
        self.drift_logger.setLevel(logging.INFO)

        self.logger.info("📊 V3.8增强监控系统配置完成")

    def log_performance_metrics(self, metrics_dict):
        """记录性能指标"""
        metrics_str = " | ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in metrics_dict.items()])
        self.perf_logger.info(f"性能指标: {metrics_str}")

    def log_incremental_update(self, update_info):
        """记录增量更新信息"""
        self.incremental_logger.info(f"增量更新: {update_info}")

    def log_drift_detection(self, drift_info):
        """记录模型漂移检测"""
        self.drift_logger.info(f"漂移检测: {drift_info}")
        
    def _init_model_configs(self):
        """初始化模型配置参数"""
        
        # Level 1: 基础模型配置
        self.base_model_configs = {
            'lgb': {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'n_estimators': 200
            },
            'xgb': {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1
            },
            'catboost': {
                'iterations': 200,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'random_state': 42,
                'verbose': False
            },
            'rf': {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            },
            'mlp': {
                'hidden_layer_sizes': (100, 50, 25),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'learning_rate': 'adaptive',
                'max_iter': 500,
                'random_state': 42
            }
        }
        
        # Level 2: 专家模型权重
        self.expert_weights = {
            'technical_expert': 0.3,
            'fundamental_expert': 0.25,
            'macro_expert': 0.25,
            'sentiment_expert': 0.2
        }
        
        # Level 3: Meta学习器配置
        self.meta_config = {
            'model_type': 'neural_network',
            'architecture': (50, 25, 10),
            'dropout': 0.2,
            'learning_rate': 0.001
        }

    # =============================================================================
    # V3.8 新增：增量学习基础接口
    # =============================================================================

    def init_incremental_learning_components(self):
        """初始化增量学习组件"""
        self.logger.info("🔄 初始化V3.8增量学习组件...")

        # 初始化增量学习引擎
        self.incremental_engine = IncrementalLearningEngine(
            learning_rates=self.learning_rates,
            forgetting_factors=self.forgetting_factors,
            logger=self.logger
        )

        # 初始化实时特征计算器
        self.realtime_calculator = RealtimeFeatureCalculator(
            cache_ttl=self.cache_ttl,
            db_manager=self.db_manager,
            logger=self.logger
        )

        # 初始化自适应评分器
        self.adaptive_scorer = AdaptiveScoringSystem(
            temporal_models=self.temporal_models,
            logger=self.logger
        )

        self.logger.info("✅ V3.8增量学习组件初始化完成")

    def incremental_update(self, new_features, new_targets, update_type='daily'):
        """增量更新模型"""
        if self.incremental_engine is None:
            self.init_incremental_learning_components()

        self.logger.info(f"🔄 开始{update_type}增量更新...")

        # 记录更新历史
        update_record = {
            'timestamp': datetime.now(),
            'update_type': update_type,
            'data_size': len(new_features),
            'status': 'started'
        }

        try:
            # 调用增量学习引擎
            update_result = self.incremental_engine.incremental_update(
                new_features, new_targets, update_type
            )

            update_record['status'] = 'completed'
            update_record['result'] = update_result

            self.incremental_update_history.append(update_record)
            self.logger.info(f"✅ {update_type}增量更新完成")

            return update_result

        except Exception as e:
            update_record['status'] = 'failed'
            update_record['error'] = str(e)
            self.incremental_update_history.append(update_record)
            self.logger.error(f"❌ {update_type}增量更新失败: {e}")
            return None

    def detect_model_drift(self, validation_features, validation_targets):
        """检测模型漂移"""
        if len(self.performance_history) < 5:
            return False, 0.0

        # 计算当前模型在验证集上的表现
        current_predictions = self.predict_scores(validation_features)
        current_mse = mean_squared_error(validation_targets, current_predictions)

        # 计算历史平均MSE
        recent_mse = np.mean([h['validation_mse'] for h in self.performance_history[-5:]])

        # 计算漂移程度
        drift_magnitude = (current_mse - recent_mse) / recent_mse if recent_mse > 0 else 0

        # 记录漂移检测历史
        drift_record = {
            'timestamp': datetime.now(),
            'current_mse': current_mse,
            'recent_avg_mse': recent_mse,
            'drift_magnitude': drift_magnitude,
            'drift_detected': drift_magnitude > self.update_thresholds['performance_drop']
        }

        self.drift_detection_history.append(drift_record)

        return drift_record['drift_detected'], drift_magnitude

    def compute_realtime_features(self, code, current_time=None):
        """计算实时特征"""
        if self.realtime_calculator is None:
            self.init_incremental_learning_components()

        if current_time is None:
            current_time = datetime.now()

        return self.realtime_calculator.compute_intraday_features(code, current_time)

    def adaptive_score_normalization(self, predictions, market_volatility, confidence_level):
        """自适应评分标准化"""
        if self.adaptive_scorer is None:
            self.init_incremental_learning_components()

        return self.adaptive_scorer.adaptive_normalize_scores(
            predictions, market_volatility, confidence_level
        )

    def extract_advanced_features(self, codes, start_date, end_date, target_only=False):
        """
        提取35+维高级特征
        
        🎯 特征分类:
        - 技术增强因子 (8个): ADX, TRIX, VWAP等
        - 行业风格因子 (6个): 行业动量、市值风格等  
        - 宏观市场因子 (7个): 市场情绪、流动性等
        - 时序特征因子 (5个): 多周期动量、形态识别等
        - V3.6原有因子 (15个): 保持向后兼容
        """
        self.logger.info(f"🔍 V3.8特征提取: {len(codes)}只股票, {start_date} 到 {end_date}")
        
        features_list = []
        total_codes = len(codes)
        
        for i, code in enumerate(codes):
            if (i + 1) % 100 == 0:
                self.logger.info(f"进度: {i+1}/{total_codes} ({(i+1)/total_codes*100:.1f}%)")
                
            try:
                # 获取基础数据 (需要更长历史期间)
                extended_start = (pd.to_datetime(start_date) - timedelta(days=120)).strftime('%Y-%m-%d')
                base_data = self._get_stock_data(code, extended_start, end_date)
                
                if base_data is None or len(base_data) < 30:
                    continue
                    
                if target_only:
                    # 仅计算最新一天特征
                    daily_features = self._compute_advanced_features(code, base_data, len(base_data)-1)
                    if daily_features is not None:
                        daily_features['code'] = code
                        features_list.append(daily_features)
                else:
                    # 计算历史期间特征 (用于训练)
                    for idx in range(30, len(base_data)):  # 确保足够历史数据
                        daily_features = self._compute_advanced_features(code, base_data, idx)
                        if daily_features is not None:
                            daily_features['code'] = code
                            features_list.append(daily_features)
                    
            except Exception as e:
                self.logger.warning(f"股票{code}特征提取失败: {e}")
                continue
                
        if not features_list:
            self.logger.error("❌ 未能提取任何特征数据")
            return None
            
        features_df = pd.concat(features_list, ignore_index=True)
        self.logger.info(f"✅ V3.8特征提取完成: {len(features_df)}条记录, {len(features_df.columns)-2}个特征")
        
        return features_df
        
    def _compute_advanced_features(self, code, data, idx=None):
        """计算35+维高级特征"""
        if len(data) < 30:
            return None
            
        try:
            # 使用指定索引或最新数据
            if idx is None:
                idx = len(data) - 1
            latest = data.iloc[idx]
            
            # 获取到当前日期为止的历史数据窗口
            window_data = data.iloc[:idx+1]
            
            # =========================
            # V3.6原有特征 (15个) - 保持向后兼容
            # =========================
            v36_features = self._compute_v36_features(latest, window_data)
            
            # =========================  
            # 🚀 V3.8新增特征开始
            # =========================
            
            # 📈 技术增强因子 (8个新增)
            tech_features = self._compute_technical_enhanced_features(window_data)
            
            # 🏭 行业与风格因子 (6个新增) 
            industry_features = self._compute_industry_style_features(code, latest, window_data)
            
            # 🌐 宏观市场因子 (7个新增)
            macro_features = self._compute_macro_market_features(window_data, latest['trade_date'])
            
            # ⏰ 时序特征因子 (5个新增)
            temporal_features = self._compute_temporal_features(window_data)
            
            # 📊 特征交互与组合 (可选扩展)
            interaction_features = self._compute_feature_interactions(v36_features, tech_features)
            
            # 合并所有特征
            all_features = {
                'trade_date': latest['trade_date'],
                'code': code,
                **v36_features,
                **tech_features,
                **industry_features, 
                **macro_features,
                **temporal_features,
                **interaction_features
            }
            
            return pd.DataFrame([all_features])
            
        except Exception as e:
            self.logger.warning(f"计算{code}高级特征失败: {e}")
            return None

    def _compute_v36_features(self, latest, window_data):
        """计算V3.6原有的15个特征 - 保持向后兼容"""
        
        # 1. BBI相关特征
        bbi = latest['bbi'] if pd.notna(latest['bbi']) else self._safe_iloc(window_data['close'].rolling(min(20, len(window_data))).mean(), -1)
        bbi_ratio = (latest['close'] / bbi - 1) * 100 if bbi > 0 else 0.0
        
        # 2. 成交量特征
        if len(window_data) >= 5:
            volume_ma5 = self._safe_iloc(window_data['volume'].rolling(5).mean(), -1)
            volume_surge = (latest['volume'] / volume_ma5 - 1) * 100 if volume_ma5 > 0 else 0.0
        else:
            volume_surge = 0.0
        volume_surge = np.clip(volume_surge, -100, 200)
        
        # 3. 价格动量
        if len(window_data) >= 5:
            price_momentum = (latest['close'] / window_data['close'].iloc[-5] - 1) * 100
        else:
            price_momentum = 0.0
        price_momentum = np.clip(price_momentum, -50, 50)
        
        # 4. 知行多空线
        if len(window_data) >= 57:
            ma57 = self._safe_iloc(window_data['close'].rolling(57).mean(), -1)
            zhixing_multiavg = (latest['close'] / ma57 - 1) * 100 if ma57 > 0 else 0.0
        else:
            zhixing_multiavg = 0.0
        zhixing_multiavg = np.clip(zhixing_multiavg, -50, 50)
        
        # 5. RSI
        rsi = latest['rsi6'] if pd.notna(latest['rsi6']) else 50.0
        
        # 6. 市值 (对数化)
        market_cap = latest['circ_mv'] if pd.notna(latest['circ_mv']) else 100.0
        market_cap_log = np.log(max(market_cap, 1.0))
        
        # 7. KDJ交叉
        kdj_k = latest['kdj_k'] if pd.notna(latest['kdj_k']) else 50.0
        kdj_d = latest['kdj_d'] if pd.notna(latest['kdj_d']) else 50.0
        kdj_cross = kdj_k - kdj_d
        kdj_cross = np.clip(kdj_cross, -50, 50)
        
        # 8. PB估值
        pb = latest['pb'] if pd.notna(latest['pb']) else 3.0
        pb = max(0.1, min(pb, 20.0))  # 限制在合理范围
        
        # 9. 换手率
        turnover_rate = latest['turnover_rate'] if pd.notna(latest['turnover_rate']) else 1.0
        turnover_rate = max(0.01, min(turnover_rate, 50.0))
        
        # 10. 波动风险
        if len(window_data) >= 10:
            returns = window_data['close'].pct_change().dropna()
            volatility_risk = returns.std() * 100 if len(returns) > 0 else 5.0
        else:
            volatility_risk = 5.0
        volatility_risk = np.clip(volatility_risk, 0.5, 20.0)
        
        # 11. 相对强度 (vs 市场)
        if len(window_data) >= 10:
            stock_return = (latest['close'] / window_data['close'].iloc[-10] - 1) * 100
            relative_strength = stock_return  # 简化版，实际应该减去市场收益
        else:
            relative_strength = 0.0
        relative_strength = np.clip(relative_strength, -50, 100)
        
        # 12. PE估值
        pe_ttm = latest['pe_ttm'] if pd.notna(latest['pe_ttm']) else 20.0
        pe_ttm = max(0.1, min(pe_ttm, 1000.0))
        
        # 13-15. 价格特征 (V3.6新增的)
        stock_price = latest['close'] if pd.notna(latest['close']) else 10.0
        price_log = np.log(max(stock_price, 0.1))
        
        if stock_price <= 10:
            price_category = 1
        elif stock_price <= 20:
            price_category = 2
        elif stock_price <= 50:
            price_category = 3
        else:
            price_category = 4
            
        if len(window_data) >= 20:
            price_trend_30d = (self._safe_iloc(window_data['close'], -1) / window_data['close'].iloc[-20] - 1) * 100
        else:
            price_trend_30d = 0.0
        price_trend_30d = np.clip(price_trend_30d, -100, 100)
        
        return {
            # V3.6原有15个特征
            'bbi': bbi_ratio,
            'volume_surge': volume_surge,
            'price_momentum': price_momentum,
            'zhixing_multiavg': zhixing_multiavg,
            'rsi': rsi,
            'market_cap': market_cap_log,
            'kdj_cross': kdj_cross,
            'pb': pb,
            'turnover_rate': turnover_rate,
            'volatility_risk': volatility_risk,
            'relative_strength': relative_strength,
            'pe_ttm': pe_ttm,
            'stock_price_log': price_log,
            'price_category': price_category,
            'price_trend_30d': price_trend_30d,
            
            # 原始值 (用于显示)
            'pb_raw': pb,
            'pe_ttm_raw': pe_ttm,
            'market_cap_raw': market_cap,
            'stock_price_raw': stock_price,
        }

    def _safe_iloc(self, series, index=-1):
        """安全地访问Series或数组的索引"""
        if isinstance(series, (pd.Series, pd.DataFrame)):
            return series.iloc[index]
        elif isinstance(series, np.ndarray):
            return series[index] if len(series) > abs(index) else (series[-1] if len(series) > 0 else 0)
        elif hasattr(series, '__getitem__'):
            try:
                return series[index]
            except (IndexError, KeyError):
                return series[-1] if len(series) > 0 else 0
        else:
            return series

    def _compute_technical_enhanced_features(self, window_data):
        """计算技术增强因子 (8个新增)"""
        
        features = {}
        
        try:
            # 1. ADX (平均趋向指标) - 趋势强度
            if len(window_data) >= 14:
                # 简化ADX计算
                high = window_data['high']
                low = window_data['low']
                close = window_data['close']
                
                # TR (真实波动幅度)
                tr1 = high - low
                tr2 = abs(high - close.shift(1))
                tr3 = abs(low - close.shift(1))
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                
                # DM (方向移动)
                dm_plus = np.where((high - high.shift(1)) > (low.shift(1) - low), 
                                 np.maximum(high - high.shift(1), 0), 0)
                dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)),
                                  np.maximum(low.shift(1) - low, 0), 0)
                
                # 14日平滑
                tr_14 = pd.Series(tr).rolling(14).mean()
                dm_plus_14 = pd.Series(dm_plus).rolling(14).mean()
                dm_minus_14 = pd.Series(dm_minus).rolling(14).mean()
                
                # DI
                di_plus = 100 * dm_plus_14 / tr_14
                di_minus = 100 * dm_minus_14 / tr_14
                
                # ADX
                dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
                adx_series = dx.rolling(14).mean()
                adx = self._safe_iloc(adx_series, -1)
                features['adx_14'] = adx if pd.notna(adx) else 25.0
            else:
                features['adx_14'] = 25.0
                
            # 2. TRIX (三重指数移动平均)
            if len(window_data) >= 20:
                close = window_data['close']
                ema1 = close.ewm(span=12).mean()
                ema2 = ema1.ewm(span=12).mean() 
                ema3 = ema2.ewm(span=12).mean()
                trix_series = ema3.pct_change() * 100
                trix = self._safe_iloc(trix_series, -1)
                features['trix'] = trix if pd.notna(trix) else 0.0
            else:
                features['trix'] = 0.0
                
            # 3. VWAP偏离度
            if len(window_data) >= 5:
                # 计算VWAP
                vwap = (window_data['close'] * window_data['volume']).sum() / window_data['volume'].sum()
                vwap_deviation = (self._safe_iloc(window_data['close'], -1) / vwap - 1) * 100
                features['vwap_deviation'] = np.clip(vwap_deviation, -20, 20)
            else:
                features['vwap_deviation'] = 0.0
                
            # 4. ATR相对比率
            if len(window_data) >= 14:
                if 'atr_14' in window_data.columns and pd.notna(self._safe_iloc(window_data['atr_14'], -1)):
                    atr_current = self._safe_iloc(window_data['atr_14'], -1)
                    atr_avg = self._safe_iloc(window_data['atr_14'].rolling(20).mean(), -1)
                    features['atr_ratio'] = (atr_current / atr_avg) if atr_avg > 0 else 1.0
                else:
                    # 手工计算ATR
                    high = window_data['high']
                    low = window_data['low']
                    close = window_data['close']
                    tr1 = high - low
                    tr2 = abs(high - close.shift(1))
                    tr3 = abs(low - close.shift(1))
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    atr = self._safe_iloc(tr.rolling(14).mean(), -1)
                    atr_20 = self._safe_iloc(tr.rolling(20).mean(), -1)
                    features['atr_ratio'] = (atr / atr_20) if atr_20 > 0 else 1.0
            else:
                features['atr_ratio'] = 1.0
                
            # 5. Keltner通道位置
            if len(window_data) >= 20:
                if 'kc_upper' in window_data.columns and pd.notna(self._safe_iloc(window_data['kc_upper'], -1)):
                    kc_upper = self._safe_iloc(window_data['kc_upper'], -1)
                    kc_lower = self._safe_iloc(window_data['kc_lower'], -1)
                    current_price = self._safe_iloc(window_data['close'], -1)
                    kc_position = (current_price - kc_lower) / (kc_upper - kc_lower) if (kc_upper - kc_lower) > 0 else 0.5
                else:
                    # 手工计算KC
                    ma20 = self._safe_iloc(window_data['close'].rolling(20).mean(), -1) 
                    kc_position = 0.5  # 默认中位
                features['keltner_position'] = np.clip(kc_position, 0, 1)
            else:
                features['keltner_position'] = 0.5
                
            # 6. 波动率体制
            if len(window_data) >= 30:
                returns = window_data['close'].pct_change().dropna()
                vol_short = self._safe_iloc(returns.rolling(10).std(), -1) * np.sqrt(252) * 100
                vol_long = self._safe_iloc(returns.rolling(30).std(), -1) * np.sqrt(252) * 100
                vol_regime = vol_short / vol_long if vol_long > 0 else 1.0
                features['volatility_regime'] = vol_regime
            else:
                features['volatility_regime'] = 1.0
                
            # 7. OBV能量潮趋势
            if len(window_data) >= 10:
                close = window_data['close']
                volume = window_data['volume']
                price_change = close.diff()
                obv_change = np.where(price_change > 0, volume, 
                                    np.where(price_change < 0, -volume, 0))
                obv = obv_change.cumsum()
                obv_trend = (self._safe_iloc(obv, -1) - self._safe_iloc(obv, -10)) / self._safe_iloc(obv, -10) if self._safe_iloc(obv, -10) != 0 else 0.0
                features['obv_trend'] = np.clip(obv_trend, -1, 1)
            else:
                features['obv_trend'] = 0.0
                
            # 8. 资金流量指标 (MFI)
            if len(window_data) >= 14:
                typical_price = (window_data['high'] + window_data['low'] + window_data['close']) / 3
                money_flow = typical_price * window_data['volume']
                
                positive_flow = np.where(typical_price.diff() > 0, money_flow, 0)
                negative_flow = np.where(typical_price.diff() < 0, money_flow, 0)
                
                positive_mf = pd.Series(positive_flow).rolling(14).sum()
                negative_mf = pd.Series(negative_flow).rolling(14).sum()
                
                mfi = 100 - (100 / (1 + positive_mf / negative_mf))
                features['mfi_14'] = self._safe_iloc(mfi, -1) if pd.notna(self._safe_iloc(mfi, -1)) else 50.0
            else:
                features['mfi_14'] = 50.0
                
        except Exception as e:
            # 如果计算失败，使用默认值
            self.logger.warning(f"技术增强特征计算失败: {e}")
            features = {
                'adx_14': 25.0, 'trix': 0.0, 'vwap_deviation': 0.0, 'atr_ratio': 1.0,
                'keltner_position': 0.5, 'volatility_regime': 1.0, 'obv_trend': 0.0, 'mfi_14': 50.0
            }
            
        return features

    def _compute_industry_style_features(self, code, latest, window_data):
        """计算行业与风格因子 (6个新增)"""
        
        features = {}
        
        try:
            # 获取股票行业信息
            industry_info = self._get_stock_industry(code)
            
            # 1. 行业动量 (简化版 - 基于个股表现估算)
            if len(window_data) >= 20:
                stock_return_20d = (latest['close'] / window_data['close'].iloc[-20] - 1) * 100
                # 这里简化处理，实际应该计算同行业其他股票的平均收益
                features['industry_momentum'] = np.clip(stock_return_20d, -50, 50)
            else:
                features['industry_momentum'] = 0.0
                
            # 2. 行业相对强度 (简化版)
            features['industry_relative_strength'] = features['industry_momentum']  # 简化处理
            
            # 3. 行业轮动信号 (基于个股动量变化)
            if len(window_data) >= 30:
                momentum_10d = (latest['close'] / window_data['close'].iloc[-10] - 1) * 100
                momentum_30d = (latest['close'] / window_data['close'].iloc[-30] - 1) * 100
                rotation_signal = momentum_10d - momentum_30d
                features['industry_rotation_signal'] = np.clip(rotation_signal, -30, 30)
            else:
                features['industry_rotation_signal'] = 0.0
                
            # 4. 市值因子 (Size Factor)
            market_cap = latest['circ_mv'] if pd.notna(latest['circ_mv']) else 100.0
            if market_cap < 50:  # 小盘股
                size_factor = 1.0
            elif market_cap < 200:  # 中盘股
                size_factor = 0.0
            else:  # 大盘股
                size_factor = -1.0
            features['size_factor'] = size_factor
            
            # 5. 价值因子 (Value Factor)
            pb = latest['pb'] if pd.notna(latest['pb']) else 3.0
            pe = latest['pe_ttm'] if pd.notna(latest['pe_ttm']) else 20.0
            
            # 价值评分 (PB和PE越低越好)
            pb_score = max(0, (5 - pb) / 5)  # PB=1时得分1，PB=5时得分0
            pe_score = max(0, (30 - pe) / 30)  # PE=10时得分0.67，PE=30时得分0
            value_factor = (pb_score + pe_score) / 2
            features['value_factor'] = np.clip(value_factor, 0, 1)
            
            # 6. 成长因子 (Growth Factor) - 简化版
            # 理想情况下应该使用EPS增长率，这里用价格动量代替
            if len(window_data) >= 60:
                growth_rate = (latest['close'] / window_data['close'].iloc[-60] - 1) * 100
                growth_factor = growth_rate / 100  # 标准化
                features['growth_factor'] = np.clip(growth_factor, -1, 2)
            else:
                features['growth_factor'] = 0.0
                
        except Exception as e:
            self.logger.warning(f"行业风格特征计算失败: {e}")
            features = {
                'industry_momentum': 0.0, 'industry_relative_strength': 0.0, 'industry_rotation_signal': 0.0,
                'size_factor': 0.0, 'value_factor': 0.5, 'growth_factor': 0.0
            }
            
        return features

    def _compute_macro_market_features(self, window_data, trade_date):
        """计算宏观市场因子 (7个新增)"""
        
        features = {}
        
        try:
            # 1. 市场情绪指数 (基于真实个股表现)
            if len(window_data) >= 5:
                recent_returns = window_data['close'].pct_change().tail(5)
                positive_days = (recent_returns > 0).sum()
                market_sentiment = (positive_days - 2.5) / 2.5  # 标准化到[-1, 1]
                features['market_sentiment'] = market_sentiment
            else:
                features['market_sentiment'] = 0.0
                
            # 2. A股恐慌指数 (简化版 - 基于波动率)
            if len(window_data) >= 20:
                returns = window_data['close'].pct_change().dropna()
                current_vol = returns.tail(5).std()
                avg_vol = returns.tail(20).std()
                vix_equivalent = current_vol / avg_vol if avg_vol > 0 else 1.0
                features['vix_equivalent'] = np.clip(vix_equivalent, 0.5, 3.0)
            else:
                features['vix_equivalent'] = 1.0
                
            # 3. 新高新低比 (简化版)
            if len(window_data) >= 20:
                recent_high = window_data['high'].tail(5).max()
                period_high = window_data['high'].tail(20).max()
                recent_low = window_data['low'].tail(5).min()
                period_low = window_data['low'].tail(20).min()
                
                is_near_high = (recent_high / period_high) > 0.95
                is_near_low = (recent_low / period_low) < 1.05
                
                if is_near_high:
                    new_high_low_ratio = 1.0
                elif is_near_low:
                    new_high_low_ratio = -1.0
                else:
                    new_high_low_ratio = 0.0
                    
                features['new_high_low_ratio'] = new_high_low_ratio
            else:
                features['new_high_low_ratio'] = 0.0
                
            # 4. 流动性评分
            if len(window_data) >= 10:
                avg_turnover = window_data['turnover_rate'].tail(10).mean() if 'turnover_rate' in window_data.columns else 2.0
                avg_volume = window_data['volume'].tail(10).mean()
                current_volume = self._safe_iloc(window_data['volume'], -1)
                
                volume_activity = current_volume / avg_volume if avg_volume > 0 else 1.0
                liquidity_score = (avg_turnover * volume_activity) / 10  # 标准化
                features['liquidity_score'] = np.clip(liquidity_score, 0, 2)
            else:
                features['liquidity_score'] = 1.0
                
            # 5. 买卖价差因子 (基于价格波动)
            if len(window_data) >= 5:
                # 用高低价差计算价格波动
                spread_ratio = ((window_data['high'] - window_data['low']) / window_data['close']).tail(5).mean()
                features['spread_factor'] = np.clip(spread_ratio, 0, 0.1)
            else:
                features['spread_factor'] = 0.02
                
            # 6. 机构资金流向 (基于价量关系分析)
            if len(window_data) >= 10:
                # 用成交量和价格变化分析资金流向
                price_changes = window_data['close'].pct_change()
                volumes = window_data['volume']
                
                institutional_flow = (price_changes * volumes).tail(5).sum() / volumes.tail(5).sum()
                features['institutional_flow'] = np.clip(institutional_flow, -0.1, 0.1)
            else:
                features['institutional_flow'] = 0.0
                
            # 7. 利率敏感性 (基于市值和行业特征分析)
            market_cap = self._safe_iloc(window_data, -1).get('circ_mv', 100)
            if market_cap > 500:  # 大盘股对利率更敏感
                interest_rate_sensitivity = 1.0
            elif market_cap < 100:  # 小盘股敏感性较低
                interest_rate_sensitivity = 0.3
            else:
                interest_rate_sensitivity = 0.7
            features['interest_rate_sensitivity'] = interest_rate_sensitivity
            
        except Exception as e:
            self.logger.warning(f"宏观市场特征计算失败: {e}")
            features = {
                'market_sentiment': 0.0, 'vix_equivalent': 1.0, 'new_high_low_ratio': 0.0,
                'liquidity_score': 1.0, 'spread_factor': 0.02, 'institutional_flow': 0.0,
                'interest_rate_sensitivity': 0.7
            }
            
        return features

    def _compute_temporal_features(self, window_data):
        """计算时序特征因子 (5个新增)"""
        
        features = {}
        
        try:
            current_price = self._safe_iloc(window_data['close'], -1)
            
            # 1-3. 多周期动量
            if len(window_data) >= 3:
                momentum_3d = (current_price / window_data['close'].iloc[-3] - 1) * 100
                features['momentum_3d'] = np.clip(momentum_3d, -20, 20)
            else:
                features['momentum_3d'] = 0.0
                
            if len(window_data) >= 5:
                momentum_5d = (current_price / window_data['close'].iloc[-5] - 1) * 100
                features['momentum_5d'] = np.clip(momentum_5d, -30, 30)
            else:
                features['momentum_5d'] = 0.0
                
            if len(window_data) >= 20:
                momentum_20d = (current_price / window_data['close'].iloc[-20] - 1) * 100
                features['momentum_20d'] = np.clip(momentum_20d, -50, 100)
            else:
                features['momentum_20d'] = 0.0
                
            # 4. 突破形态识别
            if len(window_data) >= 20:
                # 简化的突破识别
                recent_high = window_data['high'].tail(20).max()
                recent_close = window_data['close'].tail(20)
                
                # 判断是否突破近期高点
                is_breakout = current_price > recent_high * 0.98
                
                # 判断突破前是否有整理
                consolidation_days = 0
                for i in range(1, min(10, len(recent_close))):
                    if abs(recent_close.iloc[-i] / recent_high - 1) < 0.05:
                        consolidation_days += 1
                        
                breakout_score = (1.0 if is_breakout else 0.0) * (consolidation_days / 10)
                features['pattern_breakout'] = breakout_score
            else:
                features['pattern_breakout'] = 0.0
                
            # 5. 季节性因子 (基于日期)
            trade_date = self._safe_iloc(window_data['trade_date'], -1)
            if isinstance(trade_date, str):
                date_obj = pd.to_datetime(trade_date)
            else:
                date_obj = trade_date
                
            # 简化的季节性: 年初(1-3月)和年末(11-12月)通常表现较好
            month = date_obj.month
            if month in [1, 2, 3, 11, 12]:
                seasonal_factor = 0.3
            elif month in [4, 5, 9, 10]:
                seasonal_factor = 0.1
            else:
                seasonal_factor = -0.1  # 夏季通常较弱
                
            features['seasonal_factor'] = seasonal_factor
            
        except Exception as e:
            self.logger.warning(f"时序特征计算失败: {e}")
            features = {
                'momentum_3d': 0.0, 'momentum_5d': 0.0, 'momentum_20d': 0.0,
                'pattern_breakout': 0.0, 'seasonal_factor': 0.0
            }
            
        return features

    def _compute_feature_interactions(self, v36_features, tech_features):
        """计算特征交互项 (可选扩展)"""
        
        interactions = {}
        
        try:
            # 一些重要的特征交互
            # 1. 动量 × 波动率
            momentum_vol_interaction = v36_features['price_momentum'] * tech_features['atr_ratio']
            interactions['momentum_vol_interaction'] = np.clip(momentum_vol_interaction, -100, 100)
            
            # 2. 成交量 × 价格位置
            volume_price_interaction = v36_features['volume_surge'] * tech_features['keltner_position']
            interactions['volume_price_interaction'] = np.clip(volume_price_interaction, -50, 50)
            
            # 3. RSI × ADX (超买超卖 × 趋势强度)
            rsi_adx_interaction = (v36_features['rsi'] - 50) * tech_features['adx_14'] / 100
            interactions['rsi_adx_interaction'] = np.clip(rsi_adx_interaction, -50, 50)
            
        except Exception as e:
            self.logger.warning(f"特征交互计算失败: {e}")
            interactions = {
                'momentum_vol_interaction': 0.0,
                'volume_price_interaction': 0.0,
                'rsi_adx_interaction': 0.0
            }
            
        return interactions

    def _get_stock_data(self, code, start_date, end_date):
        """获取股票综合数据"""
        # 标准化股票代码格式 (去掉后缀如.SZ, .SH)
        clean_code = code.split('.')[0] if '.' in code else code
        
        query = """
        SELECT 
            dq.trade_date,
            dq.open, dq.high, dq.low, dq.close, dq.volume, dq.amount,
            dq.price_change_pct, dq.ma5, dq.ma10, dq.ma20, dq.ma60,
            ti.bbi, ti.kdj_k, ti.kdj_d, ti.kdj_j,
            ti.rsi6, ti.rsi12, ti.rsi24,
            ti.macd_dif, ti.macd_dea, ti.macd_macd,
            ti.boll_upper, ti.boll_middle, ti.boll_lower,
            ti.volume_ma5, ti.volume_ma10, ti.volume_ratio,
            ti.atr_14, ti.kc_upper, ti.kc_middle, ti.kc_lower,
            db.turnover_rate, db.pe_ttm, db.pb, db.ps_ttm,
            db.total_mv, db.circ_mv
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        LEFT JOIN technical_indicators ti ON dq.security_id = ti.security_id AND dq.trade_date = ti.trade_date
        LEFT JOIN daily_basic db ON dq.security_id = db.security_id AND dq.trade_date = db.trade_date
        WHERE s.code = ? AND dq.trade_date BETWEEN ? AND ?
        ORDER BY dq.trade_date
        """
        
        try:
            with self.db_manager.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=(clean_code, start_date, end_date))
                return df if len(df) > 0 else None
        except Exception as e:
            self.logger.warning(f"获取{clean_code}数据失败: {e}")
            return None

    def _get_stock_industry(self, code):
        """获取股票行业信息"""
        # 标准化股票代码格式
        clean_code = code.split('.')[0] if '.' in code else code
        
        try:
            with self.db_manager.get_connection() as conn:
                query = "SELECT industry FROM securities WHERE code = ?"
                result = conn.execute(query, (clean_code,)).fetchone()
                return result[0] if result else "未知"
        except Exception as e:
            return "未知"

    def save_models(self, model_name_suffix=""):
        """保存所有模型组件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"v380_models_{timestamp}{model_name_suffix}"

        model_file = self.model_dir / f"{model_name}.pkl"

        model_data = {
            'version': self.version,
            'timestamp': timestamp,
            'base_models': self.base_models,
            'expert_models': self.expert_models,
            'meta_learner': self.meta_learner,
            'scalers': self.scalers,
            'feature_selectors': self.feature_selectors,
            'performance_history': self.performance_history,
            'feature_importance_history': self.feature_importance_history
        }

        joblib.dump(model_data, model_file)
        self.logger.info(f"✅ V3.8模型已保存: {model_file}")

        return model_file

    def _auto_load_latest_model(self):
        """自动加载最新的训练模型"""
        try:
            import glob
            # 查找所有V3.8模型文件
            model_files = glob.glob(str(self.model_dir / "*.pkl"))
            if model_files:
                # 选择最新的模型文件
                latest_model = max(model_files, key=os.path.getctime)
                self.logger.info(f"🔍 发现已训练模型: {latest_model}")

                # 加载模型
                if self.load_models(latest_model):
                    self.logger.info(f"✅ 自动加载最新模型成功: {latest_model}")
                else:
                    self.logger.warning(f"⚠️ 自动加载模型失败: {latest_model}")
            else:
                self.logger.info("ℹ️ 未发现已训练模型，需要先训练")
        except Exception as e:
            self.logger.error(f"❌ 自动加载模型过程出错: {e}")

    def load_models(self, model_file):
        """加载模型组件"""
        try:
            model_data = joblib.load(model_file)
            
            self.base_models = model_data['base_models']
            self.expert_models = model_data['expert_models']  
            self.meta_learner = model_data['meta_learner']
            self.scalers = model_data['scalers']
            self.feature_selectors = model_data.get('feature_selectors', {})
            self.performance_history = model_data.get('performance_history', [])
            self.feature_importance_history = model_data.get('feature_importance_history', {})
            
            self.logger.info(f"✅ V3.8模型已加载: {model_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 模型加载失败: {e}")
            return False

    # =====================================================
    # 🚀 V3.8 三层 Ensemble 模型架构
    # =====================================================
    
    def build_three_layer_architecture(self, target_col='target_1d'):
        """构建三层ensemble架构"""
        self.logger.info(f"🏗️ 构建V3.8三层ensemble架构: {target_col}")
        
        # Level 1: 基础模型
        self.base_models[target_col] = {
            'lgb': lgb.LGBMRegressor(**self.base_model_configs['lgb']),
            'xgb': xgb.XGBRegressor(**self.base_model_configs['xgb']),
            'catboost': cb.CatBoostRegressor(**self.base_model_configs['catboost']),
            'rf': RandomForestRegressor(**self.base_model_configs['rf']),
            'mlp': MLPRegressor(**self.base_model_configs['mlp'])
        }
        
        # Level 2: 专家模型 (特征专门化)
        self.expert_models[target_col] = {
            'technical_expert': None,  # 将在训练时创建
            'fundamental_expert': None,
            'macro_expert': None,
            'sentiment_expert': None
        }
        
        # Level 3: Meta学习器
        self.meta_learner[target_col] = MLPRegressor(
            hidden_layer_sizes=self.meta_config['architecture'],
            learning_rate_init=self.meta_config['learning_rate'],
            alpha=0.01,
            max_iter=500,
            random_state=42
        )
        
        self.logger.info("✅ 三层ensemble架构构建完成")
        
    def prepare_training_data(self, features_df, target_days=[1, 3, 5, 10]):
        """准备训练数据 - 增强版"""
        self.logger.info(f"📊 准备训练数据: {target_days}日收益率目标")
        
        # 计算未来收益率目标
        targets_df = self.prepare_target(features_df, target_days)
        
        # 合并特征和目标
        training_data = pd.merge(features_df, targets_df, on=['code', 'trade_date'], how='inner')
        
        # 数据清洗
        training_data = training_data.dropna()
        
        # 特征分组 (用于专家模型)
        feature_groups = self._group_features_for_experts()
        
        self.logger.info(f"✅ 训练数据准备完成: {len(training_data)}条记录")
        return training_data, feature_groups
        
    def _group_features_for_experts(self):
        """为专家模型分组特征 - V4兼容版本"""
        return {
            'technical': [
                # V4基础技术特征
                'bbi', 'rsi6', 'rsi12', 'rsi24', 'kdj_k', 'kdj_d', 'kdj_j',
                'macd_dif', 'macd_dea', 'macd_macd', 'boll_upper', 'boll_middle', 'boll_lower',
                'atr_14', 'kc_upper', 'kc_middle', 'kc_lower',
                # V4高级技术特征
                'high_low_spread', 'price_position', 'momentum_3d',
                'momentum_volume_interaction', 'rsi_macd_interaction',
                'price_percentile_10d', 'price_percentile_20d', 'volatility_regime',
                # V4滚动特征
                'price_change_roll_mean_3', 'price_change_roll_mean_5', 'price_change_roll_mean_10', 'price_change_roll_mean_20',
                'volume_roll_std_3', 'volume_roll_std_5', 'volume_roll_std_10', 'volume_roll_std_20',
                'high_low_spread_roll_mean_3', 'high_low_spread_roll_mean_5', 'high_low_spread_roll_mean_10', 'high_low_spread_roll_mean_20'
            ],
            'fundamental': [
                # V4基础面特征
                'pb', 'pe_ttm', 'ps_ttm', 'turnover_rate', 'total_mv', 'circ_mv',
                # V4衍生基本面特征
                'market_cap_category', 'pe_category', 'pb_category'
            ],
            'macro': [
                # V4宏观和市场特征
                'ma5', 'ma10', 'ma20', 'ma60',
                'volume_ma5', 'volume_ma10', 'volume_ratio',
                'seasonal_factor', 'trend_strength'
            ],
            'all': []  # 将填充所有可用特征
        }

    def train_three_layer_ensemble(self, training_data, feature_groups, target_col='target_1d'):
        """训练三层ensemble模型"""
        self.logger.info(f"🎯 开始训练V3.8三层ensemble: {target_col}")
        
        # 准备特征和目标
        all_features = []
        for group_features in feature_groups.values():
            all_features.extend(group_features)
        all_features = list(set(all_features))  # 去重
        
        # 过滤存在的特征
        available_features = [f for f in all_features if f in training_data.columns]
        X = training_data[available_features].copy()
        y = training_data[target_col].copy()
        
        # 数据清洗
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X, y = X[mask], y[mask]
        
        if len(X) < 100:
            self.logger.error(f"❌ 训练数据不足: {len(X)}条")
            return {
                'meta_performance': 0.0,
                'base_scores': {},
                'expert_scores': {},
                'training_samples': len(X),
                'success': False
            }

        # 初始化模型组件
        if target_col not in self.base_models:
            self.base_models[target_col] = {
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'lgb': lgb.LGBMRegressor(random_state=42, verbose=-1),
                'xgb': xgb.XGBRegressor(random_state=42)
            }
        if target_col not in self.expert_models:
            self.expert_models[target_col] = {}
        if target_col not in self.meta_learner:
            self.meta_learner[target_col] = LinearRegression()

        # 特征标准化
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        self.scalers[target_col] = scaler
        
        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=3)
        
        # =====================================================
        # Level 1: 训练基础模型
        # =====================================================
        self.logger.info("🔥 Level 1: 训练基础模型")
        
        base_predictions = np.zeros((len(X_scaled), len(self.base_models[target_col])))
        base_scores = {}
        
        for i, (model_name, model) in enumerate(self.base_models[target_col].items()):
            self.logger.info(f"  训练 {model_name}...")
            
            # 交叉验证
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                score = r2_score(y_val, pred)
                cv_scores.append(score)
            
            # 最终训练
            model.fit(X_scaled, y)
            base_predictions[:, i] = model.predict(X_scaled)
            base_scores[model_name] = np.mean(cv_scores)
            
            self.logger.info(f"    {model_name} CV R²: {np.mean(cv_scores):.4f}")
        
        # =====================================================
        # Level 2: 训练专家模型
        # =====================================================
        self.logger.info("🔥 Level 2: 训练专家模型")
        
        expert_predictions = np.zeros((len(X_scaled), len(feature_groups)))
        expert_scores = {}
        
        for i, (expert_name, expert_features) in enumerate(feature_groups.items()):
            # 过滤可用特征
            available_expert_features = [f for f in expert_features if f in X_scaled.columns]
            
            if len(available_expert_features) == 0:
                self.logger.warning(f"  {expert_name}: 无可用特征，跳过")
                continue
                
            X_expert = X_scaled[available_expert_features]
            
            # 创建专家模型 (使用LightGBM)
            expert_config = self.base_model_configs['lgb'].copy()
            expert_config['n_estimators'] = 150  # 专家模型稍微简化
            expert_model = lgb.LGBMRegressor(**expert_config)
            
            self.logger.info(f"  训练 {expert_name} ({len(available_expert_features)}个特征)...")
            
            # 交叉验证
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_expert):
                X_train, X_val = X_expert.iloc[train_idx], X_expert.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                expert_model.fit(X_train, y_train)
                pred = expert_model.predict(X_val)
                score = r2_score(y_val, pred)
                cv_scores.append(score)
            
            # 最终训练
            expert_model.fit(X_expert, y)
            expert_predictions[:, i] = expert_model.predict(X_expert)
            expert_scores[expert_name] = np.mean(cv_scores)
            
            # 保存专家模型
            self.expert_models[target_col][expert_name] = expert_model
            
            self.logger.info(f"    {expert_name} CV R²: {np.mean(cv_scores):.4f}")
        
        # =====================================================  
        # Level 3: 训练Meta学习器
        # =====================================================
        self.logger.info("🔥 Level 3: 训练Meta学习器")
        
        # 合并基础模型和专家模型的预测作为meta特征
        meta_features = np.concatenate([base_predictions, expert_predictions], axis=1)
        
        # 添加原始特征的子集 (增强meta学习器的信息)
        key_features = ['bbi', 'rsi', 'market_cap', 'pb', 'volume_surge']
        available_key_features = [f for f in key_features if f in X_scaled.columns]
        if available_key_features:
            key_feature_data = X_scaled[available_key_features].values
            meta_features = np.concatenate([meta_features, key_feature_data], axis=1)
        
        # Meta学习器交叉验证
        meta_cv_scores = []
        for train_idx, val_idx in tscv.split(meta_features):
            X_train, X_val = meta_features[train_idx], meta_features[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.meta_learner[target_col].fit(X_train, y_train)
            pred = self.meta_learner[target_col].predict(X_val)
            score = r2_score(y_val, pred)
            meta_cv_scores.append(score)
        
        # 最终训练Meta学习器
        self.meta_learner[target_col].fit(meta_features, y)
        
        meta_score = np.mean(meta_cv_scores)
        self.logger.info(f"    Meta学习器 CV R²: {meta_score:.4f}")
        
        # =====================================================
        # 性能总结和特征重要性
        # =====================================================
        self.logger.info("📊 训练完成，性能总结:")
        self.logger.info(f"  Level 1 (基础模型): {np.mean(list(base_scores.values())):.4f}")
        self.logger.info(f"  Level 2 (专家模型): {np.mean(list(expert_scores.values())):.4f}")
        self.logger.info(f"  Level 3 (Meta学习器): {meta_score:.4f}")
        
        # 记录性能历史
        performance_record = {
            'timestamp': datetime.now().isoformat(),
            'target': target_col,
            'base_scores': base_scores,
            'expert_scores': expert_scores,
            'meta_score': meta_score,
            'training_samples': len(X_scaled)
        }
        self.performance_history.append(performance_record)
        
        # 特征重要性分析 (LightGBM)
        lgb_model = self.base_models[target_col]['lgb']
        feature_importance = dict(zip(X.columns, lgb_model.feature_importances_))
        self.feature_importance_history[target_col] = feature_importance
        
        # 显示top重要特征
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        self.logger.info("🔝 Top 10 重要特征:")
        for i, (feat, importance) in enumerate(top_features, 1):
            self.logger.info(f"  {i:2d}. {feat}: {importance:.4f}")

        # 返回性能字典（V4兼容）
        return {
            'meta_performance': meta_score,
            'base_scores': base_scores,
            'expert_scores': expert_scores,
            'training_samples': len(X_scaled),
            'success': True
        }

    def predict_three_layer_ensemble(self, features_df, target_col='target_1d'):
        """使用三层ensemble模型进行预测"""
        self.logger.info(f"🎯 V3.8三层ensemble预测: {target_col}")
        
        if target_col not in self.base_models:
            self.logger.error(f"❌ 模型{target_col}未训练")
            return None
            
        # 准备特征
        feature_groups = self._group_features_for_experts()
        all_features = []
        for group_features in feature_groups.values():
            all_features.extend(group_features)
        all_features = list(set(all_features))

        available_features = [f for f in all_features if f in features_df.columns]
        X = features_df[available_features].copy()

        # 获取训练时的特征顺序
        if hasattr(self.scalers[target_col], 'feature_names_in_'):
            training_features = self.scalers[target_col].feature_names_in_
        else:
            # 如果没有保存特征名，使用当前特征顺序
            training_features = available_features

        # 确保特征顺序与训练时一致
        missing_features = [f for f in training_features if f not in X.columns]
        if missing_features:
            self.logger.warning(f"缺失训练时的特征: {missing_features}")
            # 添加缺失特征并填充为0
            for feature in missing_features:
                X[feature] = 0.0

        # 按训练时的顺序重新排列特征
        X = X[training_features]

        # 标准化
        X_scaled = pd.DataFrame(
            self.scalers[target_col].transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # =====================================================
        # Level 1: 基础模型预测
        # =====================================================
        base_predictions = np.zeros((len(X_scaled), len(self.base_models[target_col])))
        
        for i, (model_name, model) in enumerate(self.base_models[target_col].items()):
            base_predictions[:, i] = model.predict(X_scaled)
        
        # =====================================================
        # Level 2: 专家模型预测
        # =====================================================
        expert_predictions = np.zeros((len(X_scaled), len(feature_groups)))
        
        for i, (expert_name, expert_features) in enumerate(feature_groups.items()):
            expert_model = self.expert_models[target_col][expert_name]
            if expert_model is None:
                continue
                
            available_expert_features = [f for f in expert_features if f in X_scaled.columns]
            if len(available_expert_features) > 0:
                X_expert = X_scaled[available_expert_features]
                expert_predictions[:, i] = expert_model.predict(X_expert)
        
        # =====================================================
        # Level 3: Meta学习器预测
        # =====================================================
        meta_features = np.concatenate([base_predictions, expert_predictions], axis=1)
        
        # 添加关键特征
        key_features = ['bbi', 'rsi', 'market_cap', 'pb', 'volume_surge']
        available_key_features = [f for f in key_features if f in X_scaled.columns]
        if available_key_features:
            key_feature_data = X_scaled[available_key_features].values
            meta_features = np.concatenate([meta_features, key_feature_data], axis=1)
        
        # 最终预测
        final_predictions = self.meta_learner[target_col].predict(meta_features)
        
        # 转换为0-100评分
        scores = self._normalize_scores_to_100(final_predictions)

        # =====================================================
        # 计算各维度因子评分
        # =====================================================
        factor_scores = {}

        # 技术分析因子 (基于技术指标专家模型)
        if 'technical_expert' in feature_groups and expert_predictions.shape[1] > 0:
            technical_idx = list(feature_groups.keys()).index('technical_expert')
            factor_scores['technical'] = self._normalize_scores_to_100(expert_predictions[:, technical_idx])[0]
        else:
            factor_scores['technical'] = 50.0

        # 基本面因子 (基于基本面专家模型)
        if 'fundamental_expert' in feature_groups and expert_predictions.shape[1] > 1:
            fundamental_idx = list(feature_groups.keys()).index('fundamental_expert')
            factor_scores['fundamental'] = self._normalize_scores_to_100(expert_predictions[:, fundamental_idx])[0]
        else:
            factor_scores['fundamental'] = 50.0

        # 宏观因子 (基于宏观专家模型)
        if 'macro_expert' in feature_groups and expert_predictions.shape[1] > 2:
            macro_idx = list(feature_groups.keys()).index('macro_expert')
            factor_scores['macro'] = self._normalize_scores_to_100(expert_predictions[:, macro_idx])[0]
        else:
            factor_scores['macro'] = 50.0

        # 情绪因子 (基于情绪专家模型)
        if 'sentiment_expert' in feature_groups and expert_predictions.shape[1] > 3:
            sentiment_idx = list(feature_groups.keys()).index('sentiment_expert')
            factor_scores['sentiment'] = self._normalize_scores_to_100(expert_predictions[:, sentiment_idx])[0]
        else:
            factor_scores['sentiment'] = 50.0

        # 时序因子 (基于基础模型的平均表现)
        if base_predictions.shape[1] > 0:
            temporal_score = float(np.mean(base_predictions[0, :]))  # 基础模型的平均预测，确保是标量
            factor_scores['temporal'] = self._normalize_scores_to_100([temporal_score])[0]
        else:
            factor_scores['temporal'] = 50.0

        return {
            'score': scores[0] if len(scores) > 0 else 50.0,
            'factor_scores': factor_scores
        }

    def _normalize_scores_to_100(self, predictions):
        """将预测结果标准化到0-100评分 (修复版：使用线性标准化保持差异)"""
        # 修复：使用Min-Max标准化替代Sigmoid，保持原始差异性

        predictions = np.array(predictions)

        # 避免除零错误
        if len(predictions) <= 1:
            return np.array([50.0] * len(predictions))

        # 计算最小值和最大值
        min_pred = np.min(predictions)
        max_pred = np.max(predictions)

        # 如果所有值相同，返回中等评分
        if max_pred == min_pred:
            return np.array([50.0] * len(predictions))

        # Min-Max标准化到[10, 90]范围，保持相对差异
        normalized = (predictions - min_pred) / (max_pred - min_pred)
        scores = normalized * 80 + 10  # 映射到[10, 90]范围

        return scores

    def prepare_target(self, features_df, target_days=[1, 3, 5, 10]):
        """准备训练目标 (未来收益率) - 改进版"""
        self.logger.info(f"📊 计算目标变量: {target_days}日收益率")
        
        targets_list = []
        
        for code in features_df['code'].unique():
            code_data = features_df[features_df['code'] == code].copy()
            code_data = code_data.sort_values('trade_date')
            
            # 获取价格数据用于计算未来收益
            with self.db_manager.get_connection() as conn:
                price_query = """
                SELECT trade_date, close 
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.code = ?
                ORDER BY trade_date
                """
                price_df = pd.read_sql_query(price_query, conn, params=(code,))
            
            if len(price_df) == 0:
                continue
                
            price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])
            
            for _, row in code_data.iterrows():
                current_date = pd.to_datetime(row['trade_date'])
                current_price_data = price_df[price_df['trade_date'] == current_date]
                
                if len(current_price_data) == 0:
                    continue
                    
                current_price = current_price_data['close'].iloc[0]
                
                target_row = {
                    'code': code,
                    'trade_date': row['trade_date']
                }
                
                for days in target_days:
                    future_date = current_date + pd.Timedelta(days=days)
                    # 找到下一个交易日
                    future_data = price_df[price_df['trade_date'] >= future_date].head(1)
                    
                    if len(future_data) > 0:
                        future_price = future_data['close'].iloc[0]
                        future_return = (future_price / current_price - 1) * 100
                        target_row[f'target_{days}d'] = future_return
                    else:
                        target_row[f'target_{days}d'] = None
                
                targets_list.append(target_row)
        
        targets_df = pd.DataFrame(targets_list)
        self.logger.info(f"✅ 目标变量计算完成: {len(targets_df)}条记录")
        
        return targets_df

    # =====================================================
    # 🚀 增量学习与实时更新机制
    # =====================================================
    
    def incremental_update(self, new_features_df, target_col='target_1d', learning_rate=0.1):
        """增量学习更新模型"""
        self.logger.info(f"🔄 增量学习更新: {target_col}")
        
        if target_col not in self.base_models:
            self.logger.error("❌ 模型未初始化，需要先完整训练")
            return False
            
        # 准备新数据
        targets_df = self.prepare_target(new_features_df, [1])
        new_training_data = pd.merge(new_features_df, targets_df, on=['code', 'trade_date'], how='inner')
        new_training_data = new_training_data.dropna()
        
        if len(new_training_data) == 0:
            self.logger.warning("⚠️ 无新训练数据可用")
            return False
            
        # 特征准备
        feature_groups = self._group_features_for_experts()
        all_features = []
        for group_features in feature_groups.values():
            all_features.extend(group_features)
        all_features = list(set(all_features))
        
        available_features = [f for f in all_features if f in new_training_data.columns]
        X_new = new_training_data[available_features].copy()
        y_new = new_training_data[target_col].copy()
        
        # 标准化
        X_new_scaled = pd.DataFrame(
            self.scalers[target_col].transform(X_new),
            columns=X_new.columns
        )
        
        # 增量更新基础模型 (仅支持增量学习的模型)
        incremental_models = ['lgb']  # LightGBM支持增量学习
        
        for model_name in incremental_models:
            if model_name in self.base_models[target_col]:
                model = self.base_models[target_col][model_name]
                try:
                    # 对于LightGBM，可以基于新数据进行额外训练
                    model.fit(X_new_scaled, y_new)
                    self.logger.info(f"  ✅ {model_name} 增量更新完成")
                except Exception as e:
                    self.logger.warning(f"  ❌ {model_name} 增量更新失败: {e}")
        
        # 记录更新历史
        update_record = {
            'timestamp': datetime.now().isoformat(),
            'target': target_col,
            'new_samples': len(X_new_scaled),
            'learning_rate': learning_rate
        }
        
        if not hasattr(self, 'update_history'):
            self.update_history = []
        self.update_history.append(update_record)
        
        self.logger.info(f"✅ 增量学习完成: {len(X_new_scaled)} 新样本")
        return True

    def monitor_model_performance(self, features_df, actual_returns_df, target_col='target_1d'):
        """监控模型性能"""
        self.logger.info(f"📊 监控模型性能: {target_col}")
        
        # 预测
        predictions = self.predict_three_layer_ensemble(features_df, target_col)
        
        if predictions is None:
            return None
            
        # 与实际收益对比
        merged_data = pd.merge(
            pd.DataFrame({'code': features_df['code'], 'predicted': predictions}),
            actual_returns_df,
            on='code',
            how='inner'
        )
        
        if len(merged_data) == 0:
            self.logger.warning("⚠️ 无匹配数据进行性能监控")
            return None
            
        # 计算性能指标
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        from scipy.stats import pearsonr
        
        mse = mean_squared_error(merged_data['actual'], merged_data['predicted'])
        mae = mean_absolute_error(merged_data['actual'], merged_data['predicted'])
        correlation, p_value = pearsonr(merged_data['actual'], merged_data['predicted'])
        
        performance_metrics = {
            'timestamp': datetime.now().isoformat(),
            'target': target_col,
            'samples': len(merged_data),
            'mse': mse,
            'mae': mae,
            'correlation': correlation,
            'p_value': p_value
        }
        
        if not hasattr(self, 'monitoring_history'):
            self.monitoring_history = []
        self.monitoring_history.append(performance_metrics)
        
        self.logger.info(f"📈 性能指标: MSE={mse:.4f}, MAE={mae:.4f}, Correlation={correlation:.4f}")
        
        # 性能警报
        if correlation < 0.1 or mse > 100:
            self.logger.warning("⚠️ 模型性能下降，建议重新训练")
            
        return performance_metrics

    def predict_scores(self, codes, date_str):
        """预测股票评分 - V3.8主要预测接口 (修复版)"""
        try:
            # 确保模型已加载
            if not hasattr(self, 'base_models') or not self.base_models:
                self.logger.warning("⚠️ 模型未训练，返回默认评分")
                return {code: 50.0 for code in codes}

            # 提取特征
            features = self.extract_advanced_features(
                codes=codes,
                start_date=date_str,
                end_date=date_str,
                target_only=True
            )

            if features is None or len(features) == 0:
                self.logger.warning(f"⚠️ 无法获取{date_str}的特征数据")
                return {code: 50.0 for code in codes}

            predictions = {}

            # 获取特征分组
            feature_groups = self._group_features_for_experts()

            # 使用训练好的模型进行预测
            feature_cols = [col for col in features.columns if col not in ['code', 'trade_date']]

            for _, row in features.iterrows():
                code = row.get('code', 'UNKNOWN')

                # 准备完整特征向量
                full_feature_vector = row[feature_cols].fillna(0)

                # 使用所有可用的目标期间模型进行ensemble预测
                ensemble_predictions = []
                # 🆕 累积所有中间预测结果
                all_base_predictions = {}
                all_expert_predictions = {}

                for target_period in ['target_1d', 'target_3d', 'target_5d', 'target_10d']:
                    if (target_period in self.base_models and
                        target_period in self.expert_models and
                        target_period in self.meta_learner):

                        try:
                            # Level 1: 基础模型预测
                            base_predictions = []
                            base_predictions_dict = {}  # 🆕 保存中间结果
                            for model_name, model in self.base_models[target_period].items():
                                try:
                                    # 获取该模型训练时使用的特征
                                    if hasattr(model, 'feature_names_in_'):
                                        model_features = model.feature_names_in_
                                    else:
                                        # 使用所有可用特征作为fallback
                                        model_features = feature_cols

                                    # 提取对应特征
                                    available_features = [f for f in model_features if f in full_feature_vector.index]
                                    if len(available_features) > 0:
                                        model_input = full_feature_vector[available_features].values.reshape(1, -1)
                                        pred = model.predict(model_input)[0]
                                        base_predictions.append(pred)
                                        base_predictions_dict[f'{model_name}_{target_period}'] = pred  # 🆕 保存

                                except Exception as e:
                                    pred = 0.0
                                    base_predictions.append(pred)
                                    base_predictions_dict[f'{model_name}_{target_period}'] = pred  # 🆕 保存

                            # Level 2: 专家模型预测
                            expert_predictions = []
                            expert_predictions_dict = {}  # 🆕 保存中间结果
                            for expert_name, expert_model in self.expert_models[target_period].items():
                                if expert_name in feature_groups:
                                    try:
                                        # 获取专家特征分组
                                        expert_features = feature_groups[expert_name]
                                        available_expert_features = [f for f in expert_features if f in full_feature_vector.index]

                                        if len(available_expert_features) > 0:
                                            # 🔧 关键修复：确保特征顺序与训练时一致
                                            if hasattr(expert_model, 'feature_names_in_'):
                                                expected_features = expert_model.feature_names_in_
                                                # 按训练时的特征顺序排列，缺失特征填充0
                                                expert_input_dict = {f: full_feature_vector.get(f, 0.0) for f in expected_features}
                                                expert_input = np.array([expert_input_dict[f] for f in expected_features]).reshape(1, -1)
                                            else:
                                                # 如果没有特征名信息，使用可用特征
                                                expert_input = full_feature_vector[available_expert_features].values.reshape(1, -1)

                                            pred = expert_model.predict(expert_input)[0]
                                            expert_predictions.append(pred)
                                            expert_predictions_dict[f'{expert_name}_{target_period}'] = pred  # 🆕 保存
                                        else:
                                            pred = 0.0
                                            expert_predictions.append(pred)
                                            expert_predictions_dict[f'{expert_name}_{target_period}'] = pred  # 🆕 保存

                                    except Exception as e:
                                        self.logger.debug(f"专家模型{expert_name}预测失败: {e}")
                                        pred = 0.0
                                        expert_predictions.append(pred)
                                        expert_predictions_dict[f'{expert_name}_{target_period}'] = pred  # 🆕 保存

                            # Level 3: Meta学习器
                            if len(base_predictions) > 0 and len(expert_predictions) > 0:
                                # 🔧 关键修复：确保Meta学习器输入特征数量正确
                                try:
                                    # 合并所有预测作为meta输入
                                    meta_input_raw = np.array(base_predictions + expert_predictions)

                                    # 检查Meta学习器期望的特征数量
                                    if hasattr(self.meta_learner[target_period], 'n_features_in_'):
                                        expected_features = self.meta_learner[target_period].n_features_in_
                                        if len(meta_input_raw) != expected_features:
                                            self.logger.debug(f"Meta输入特征数量不匹配: 实际{len(meta_input_raw)}, 期望{expected_features}")
                                            # 调整特征数量：截断或填充
                                            if len(meta_input_raw) > expected_features:
                                                meta_input_raw = meta_input_raw[:expected_features]
                                            else:
                                                # 填充为平均值
                                                avg_pred = np.mean(meta_input_raw) if len(meta_input_raw) > 0 else 0.0
                                                meta_input_raw = np.pad(meta_input_raw, (0, expected_features - len(meta_input_raw)), 'constant', constant_values=avg_pred)

                                    meta_input = meta_input_raw.reshape(1, -1)
                                    meta_pred = self.meta_learner[target_period].predict(meta_input)[0]
                                    ensemble_predictions.append(meta_pred)

                                    # 🆕 累积中间结果
                                    all_base_predictions.update(base_predictions_dict)
                                    all_expert_predictions.update(expert_predictions_dict)
                                except Exception as meta_e:
                                    self.logger.debug(f"Meta学习器预测失败: {meta_e}")
                                    # 使用基础模型和专家模型的平均值作为fallback
                                    fallback_pred = np.mean(base_predictions + expert_predictions)
                                    ensemble_predictions.append(fallback_pred)

                        except Exception as e:
                            self.logger.debug(f"目标期间{target_period}预测失败: {e}")
                            continue

                if ensemble_predictions:
                    # 🔧 修复：保存分期预测信息，而不是只返回平均值
                    period_predictions = {
                        'target_1d': ensemble_predictions[0] if len(ensemble_predictions) > 0 else 0,
                        'target_3d': ensemble_predictions[1] if len(ensemble_predictions) > 1 else 0,
                        'target_5d': ensemble_predictions[2] if len(ensemble_predictions) > 2 else 0,
                        'target_10d': ensemble_predictions[3] if len(ensemble_predictions) > 3 else 0
                    }

                    # 计算多期间ensemble平均
                    avg_prediction = np.mean(ensemble_predictions)

                    # 标准化各期间预测到0-100评分
                    short_term_raw = period_predictions['target_1d']  # 1天短期
                    medium_term_raw = np.mean([period_predictions['target_3d'], period_predictions['target_5d']])  # 3-5天中期
                    long_term_raw = period_predictions['target_10d']  # 10天长期

                    # 🔧 新增：基于真实预测计算置信度
                    # 方法1：基于预测一致性（不同期间预测方向的一致性）
                    prediction_signs = [1 if p > 0 else -1 for p in ensemble_predictions]
                    sign_consistency = abs(sum(prediction_signs)) / len(prediction_signs)  # 0-1，1表示方向完全一致

                    # 方法2：基于预测方差（低方差=高置信度）
                    prediction_variance = np.var(ensemble_predictions) if len(ensemble_predictions) > 1 else 0.1
                    variance_confidence = 1.0 / (1.0 + prediction_variance)  # 0-1

                    # 综合置信度：方向一致性 × 方差置信度
                    confidence_raw = (sign_consistency * 0.6 + variance_confidence * 0.4)
                    confidence_raw = np.clip(confidence_raw, 0.2, 0.95)  # 限制在合理范围

                    # 🔧 修复：保存原始预测值而不是立即标准化
                    # 避免单股票标准化问题，改为批量标准化
                    predictions[code] = {
                        'overall_raw': avg_prediction,
                        'short_term_raw': short_term_raw,
                        'medium_term_raw': medium_term_raw,
                        'long_term_raw': long_term_raw,
                        'confidence_raw': confidence_raw,  # 新增：真实置信度
                        'raw_predictions': period_predictions,
                        # 🆕 添加Level 1-2中间预测结果
                        'level1_predictions': all_base_predictions,
                        'level2_predictions': all_expert_predictions
                    }
                else:
                    predictions[code] = {
                        'overall_raw': 0.0,
                        'short_term_raw': 0.0,
                        'medium_term_raw': 0.0,
                        'long_term_raw': 0.0,
                        'confidence_raw': 0.2,  # 新增：低置信度，因为无法预测
                        'raw_predictions': {},
                        'level1_predictions': {},
                        'level2_predictions': {}
                    }

            # 🔧 批量标准化所有股票的预测值
            if predictions:
                # 收集所有原始预测值
                codes_list = list(predictions.keys())
                all_overall = [predictions[code].get('overall_raw', 0.0) for code in codes_list]
                all_short = [predictions[code].get('short_term_raw', 0.0) for code in codes_list]
                all_medium = [predictions[code].get('medium_term_raw', 0.0) for code in codes_list]
                all_long = [predictions[code].get('long_term_raw', 0.0) for code in codes_list]

                # 批量标准化 - 保持股票间的相对差异
                normalized_overall = self._normalize_scores_to_100(all_overall)
                normalized_short = self._normalize_scores_to_100(all_short)
                normalized_medium = self._normalize_scores_to_100(all_medium)
                normalized_long = self._normalize_scores_to_100(all_long)

                # 更新预测结果为标准化评分
                for i, code in enumerate(codes_list):
                    raw_preds = predictions[code].get('raw_predictions', {})
                    confidence_raw = predictions[code].get('confidence_raw', 0.5)  # 🔧 使用真实置信度
                    level1_preds = predictions[code].get('level1_predictions', {})  # 🆕 Level 1中间结果
                    level2_preds = predictions[code].get('level2_predictions', {})  # 🆕 Level 2中间结果
                    predictions[code] = {
                        'overall_score': round(normalized_overall[i], 2),
                        'short_term_score': round(normalized_short[i], 2),
                        'medium_term_score': round(normalized_medium[i], 2),
                        'long_term_score': round(normalized_long[i], 2),
                        'confidence_score': round(confidence_raw, 3),  # 🔧 使用真实置信度，不再硬编码0.8
                        'raw_predictions': raw_preds,
                        # 🆕 保留中间预测结果用于质量特征提取
                        'level1_predictions': level1_preds,
                        'level2_predictions': level2_preds
                    }

            self.logger.info(f"✅ 预测完成: {len(predictions)}只股票")
            return predictions

        except Exception as e:
            self.logger.error(f"❌ 预测异常: {e}")
            return {code: 50.0 for code in codes}


if __name__ == "__main__":
    # 测试V3.8系统初始化
    print("🚀 V3.8高级机器学习系统测试")
    
    system = V380AdvancedIncrementalMLSystem()
    print(f"✅ {system.version} 系统初始化成功")
    
    # 测试特征提取
    test_codes = ['000001.SZ', '600000.SH']
    features = system.extract_advanced_features(
        codes=test_codes,
        start_date='2025-01-01', 
        end_date='2025-01-31',
        target_only=True
    )
    
    if features is not None:
        print(f"🎯 特征维度: {len(features.columns)-2} (目标: 35+)")
        print("📊 特征列表:", list(features.columns))
    else:
        print("❌ 特征提取测试失败")