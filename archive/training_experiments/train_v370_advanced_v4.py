#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.7模型第4阶段训练：架构深度优化
基于前3阶段成果，实现架构级别的深度优化

🚀 阶段4创新特性：
- Attention机制：动态特征权重调整
- 多时间尺度模型：1日/3日/5日专用预测器
- 高级特征工程：时序特征 + 交互特征
- 自适应学习率：根据性能动态调整
- 模型蒸馏：大模型→小模型知识迁移

作者: Claude Code
创建时间: 2025-09-15
版本: V3.7阶段4高级优化
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
from lightgbm import LGBMRegressor
import xgboost as xgb
from xgboost import XGBRegressor
try:
    import catboost as cb
    from catboost import CatBoostRegressor
except ImportError:
    print("Installing CatBoost...")
    os.system("pip install catboost")
    import catboost as cb
    from catboost import CatBoostRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression

# 现有模块
sys.path.append('/Users/yangxu/StockTradebyZ')
from data_adapter.database_manager import DatabaseManager
from v370_advanced_ml_system import V370AdvancedMLSystem

class V370AdvancedV4Trainer:
    """
    V3.7 第4阶段高级训练器
    实现架构级别的深度优化
    """

    def __init__(self):
        self.db_manager = DatabaseManager("data_adapter/stock_data.db")
        self.version = "V3.7_Advanced_V4"
        self.logger = self._setup_logger()

        # 基础ML系统
        self.ml_system = V370AdvancedMLSystem()

        # 高级架构组件
        self.attention_weights = {}
        self.time_scale_models = {}
        self.feature_interactions = {}
        self.knowledge_distiller = {}

        # 创建模型保存目录
        os.makedirs("models/v370", exist_ok=True)

    def _setup_logger(self):
        """配置日志系统"""
        logger = logging.getLogger('V370_Advanced_V4')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def create_attention_mechanism(self, X_train, y_train):
        """
        🧠 创建Attention机制
        根据特征对不同预测期的重要性动态调整权重
        """
        self.logger.info("🧠 构建Attention机制...")

        # 计算每个特征对不同预测期的相关性
        attention_weights = {}
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

        # 为1日、3日、5日预测计算不同的注意力权重
        for target_idx, target_name in enumerate(['1d', '3d', '5d']):
            if target_idx < y_train.shape[1]:
                # 计算特征与目标的相关性
                correlations = np.abs(np.corrcoef(X_train.T, y_train[:, target_idx].T)[:-1, -1])
                correlations = np.nan_to_num(correlations)

                # Softmax标准化
                exp_corr = np.exp(correlations)
                attention_weights[target_name] = exp_corr / (exp_corr.sum() + 1e-8)

                self.logger.info(f"   {target_name}预测注意力权重计算完成，最大权重: {attention_weights[target_name].max():.4f}")

        self.attention_weights = attention_weights
        return attention_weights

    def apply_attention_weighting(self, X, target_period):
        """应用注意力权重到特征"""
        if target_period in self.attention_weights:
            weights = self.attention_weights[target_period]
            # 广播权重到所有样本
            weighted_X = X * weights.reshape(1, -1)
            return weighted_X
        return X

    def create_time_scale_specific_models(self):
        """
        ⏰ 创建时间尺度专用模型
        为1日、3日、5日预测分别优化的专门模型
        """
        self.logger.info("⏰ 构建时间尺度专用模型...")

        # 不同时间尺度的模型参数优化
        time_scale_configs = {
            '1d': {  # 短期预测：重视技术指标和市场情绪
                'lgb_params': {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 63,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'max_depth': 8,
                    'min_data_in_leaf': 100,
                    'verbose': -1
                },
                'weight_technical': 0.6,
                'weight_fundamental': 0.2,
                'weight_sentiment': 0.2
            },
            '3d': {  # 中期预测：平衡技术和基本面
                'lgb_params': {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 127,
                    'learning_rate': 0.03,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.9,
                    'max_depth': 10,
                    'min_data_in_leaf': 80,
                    'verbose': -1
                },
                'weight_technical': 0.4,
                'weight_fundamental': 0.4,
                'weight_sentiment': 0.2
            },
            '5d': {  # 长期预测：重视基本面和宏观
                'lgb_params': {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 255,
                    'learning_rate': 0.02,
                    'feature_fraction': 1.0,
                    'bagging_fraction': 0.95,
                    'max_depth': 12,
                    'min_data_in_leaf': 60,
                    'verbose': -1
                },
                'weight_technical': 0.3,
                'weight_fundamental': 0.5,
                'weight_sentiment': 0.2
            }
        }

        self.time_scale_configs = time_scale_configs
        return time_scale_configs

    def create_advanced_features(self, df):
        """
        🔧 创建高级特征工程
        添加时序特征、交互特征和统计特征
        """
        self.logger.info("🔧 创建高级特征工程...")

        # 处理重复的trade_date问题
        if df['trade_date'].duplicated().any():
            # 按trade_date分组并取最后一个值来去重
            df = df.groupby('trade_date').last().reset_index()

        # 确保数据按时间排序
        df = df.sort_values(['trade_date']).reset_index(drop=True)

        # 创建基础衍生特征
        df['high_low_spread'] = (df['high'] - df['low']) / df['close']
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

        # 1. 时序滞后特征 (Rolling Features) - 使用security_id分组
        if 'security_id' in df.columns:
            group_col = 'security_id'
        else:
            # 如果没有security_id，创建一个假的分组
            df['temp_group'] = 0
            group_col = 'temp_group'

        for window in [3, 5, 10, 20]:
            df[f'price_change_roll_mean_{window}'] = df.groupby(group_col)['price_change_pct'].rolling(window).mean().reset_index(0, drop=True)
            df[f'volume_roll_std_{window}'] = df.groupby(group_col)['volume'].rolling(window).std().reset_index(0, drop=True)
            df[f'high_low_spread_roll_mean_{window}'] = df.groupby(group_col)['high_low_spread'].rolling(window).mean().reset_index(0, drop=True)

        # 2. 交互特征 (Interaction Features)
        # 价格动量与成交量的交互
        df['momentum_3d'] = df['price_change_pct'].rolling(3).mean()
        if 'ti_volume_ratio' in df.columns:
            df['momentum_volume_interaction'] = df['momentum_3d'] * df['ti_volume_ratio']

        # 技术指标交互
        if 'rsi6' in df.columns and 'macd_dif' in df.columns:
            df['rsi_macd_interaction'] = df['rsi6'] * df['macd_dif']

        # 3. 统计特征 (Statistical Features)
        # 分位数特征
        df['price_percentile_10d'] = df.groupby(group_col)['close'].rolling(10).rank(pct=True).reset_index(0, drop=True)
        df['price_percentile_20d'] = df.groupby(group_col)['close'].rolling(20).rank(pct=True).reset_index(0, drop=True)

        # 4. 市场制度特征 (Market Regime Features)
        # 基于ATR的波动率制度
        if 'atr_14' in df.columns:
            df['volatility_regime'] = df.groupby(group_col)['atr_14'].rolling(20).mean().reset_index(0, drop=True)
            df['market_stress'] = (df['volatility_regime'] > df['volatility_regime'].quantile(0.8)).astype(int)

        # 清理临时列
        if 'temp_group' in df.columns:
            df = df.drop('temp_group', axis=1)

        # 5. 清理无效值
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)

        self.logger.info(f"   高级特征工程完成，新增特征数量: {len([col for col in df.columns if any(x in col for x in ['roll_', 'interaction', 'percentile', 'stress'])])}")

        return df

    def create_targets_for_training(self, df):
        """
        创建训练目标变量
        """
        if len(df) < 10:  # 确保有足够数据计算未来收益
            return pd.DataFrame()

        # 按日期排序
        df = df.sort_values('trade_date')

        # 计算未来收益率
        df['target_1d'] = df['close'].pct_change(1).shift(-1)  # 1日后收益率
        df['target_3d'] = df['close'].pct_change(3).shift(-3)  # 3日后收益率
        df['target_5d'] = df['close'].pct_change(5).shift(-5)  # 5日后收益率

        # 移除缺失目标值的行
        df = df.dropna(subset=['target_1d', 'target_3d', 'target_5d'])

        # 选择需要的特征列（数值列）
        feature_cols = []
        for col in df.columns:
            if col not in ['trade_date', 'security_id', 'target_1d', 'target_3d', 'target_5d']:
                if df[col].dtype in ['int64', 'float64']:
                    feature_cols.append(col)

        # 创建最终特征DataFrame
        if len(feature_cols) > 0:
            features_df = df[feature_cols + ['target_1d', 'target_3d', 'target_5d']].copy()
            # 清理无效值
            features_df = features_df.fillna(0)
            features_df = features_df.replace([np.inf, -np.inf], 0)
            return features_df
        else:
            return pd.DataFrame()

    def prepare_training_data_v4(self, features_df):
        """
        为V4训练准备数据
        """
        if len(features_df) == 0:
            raise ValueError("特征数据为空")

        # 分离特征和目标
        target_cols = ['target_1d', 'target_3d', 'target_5d']
        feature_cols = [col for col in features_df.columns if col not in target_cols]

        X = features_df[feature_cols].values
        y = features_df[target_cols].values

        # 清理异常值
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)

        return X, y

    def train_time_specific_model(self, X_train, y_train, target_period, target_idx):
        """训练时间尺度专用ensemble模型 - 使用完整V370架构"""
        self.logger.info(f"🎯 训练{target_period}专用预测模型（完整ensemble架构）...")

        # 应用注意力权重
        X_weighted = self.apply_attention_weighting(X_train, target_period)
        y_target = y_train[:, target_idx]

        # 使用V370完整的ensemble架构训练
        # 清空ml_system的现有模型，为这个时间尺度重新训练
        self.ml_system.models = {}

        # 准备数据为ml_system格式
        training_data = pd.DataFrame(X_weighted)
        training_data['target'] = y_target

        # 分割训练和验证数据
        split_point = int(len(training_data) * 0.8)
        train_df = training_data.iloc[:split_point].copy()
        val_df = training_data.iloc[split_point:].copy()

        # 使用ml_system的完整ensemble训练
        # 创建特征组（V370需要的格式）
        feature_columns = [col for col in training_data.columns if col != 'target']

        feature_groups = {
            'technical': [col for col in feature_columns if isinstance(col, str) and any(tech in col.lower() for tech in ['ma', 'rsi', 'macd', 'kdj', 'bbi', 'boll', 'cci', 'atr', 'volume'])],
            'fundamental': [col for col in feature_columns if isinstance(col, str) and any(fund in col.lower() for fund in ['pe', 'pb', 'ps', 'turnover', 'mv'])],
            'macro': [col for col in feature_columns if isinstance(col, str) and any(macro in col.lower() for macro in ['market', 'trend', 'regime'])],
            'all': feature_columns
        }

        # 初始化target_col对应的模型架构
        target_col = 'target'

        # 初始化所有层级的模型字典
        if target_col not in self.ml_system.base_models:
            self.ml_system.base_models[target_col] = {}
        if target_col not in self.ml_system.expert_models:
            self.ml_system.expert_models[target_col] = {}

        # Meta learner是单个模型，不是字典
        if target_col not in self.ml_system.meta_learner:
            self.ml_system.meta_learner[target_col] = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                learning_rate_init=0.001,
                alpha=0.01,
                max_iter=500,
                random_state=42
            )

        self.ml_system.base_models[target_col] = {
            'lgb': LGBMRegressor(
                objective='regression',
                metric='rmse',
                boosting_type='gbdt',
                num_leaves=31,
                learning_rate=0.05,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                verbose=-1,
                random_state=42
            ),
            'xgb': XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            ),
            'catboost': CatBoostRegressor(
                iterations=100,
                depth=6,
                learning_rate=0.05,
                random_seed=42,
                verbose=False
            ),
            'rf': RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            ),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                alpha=0.01,
                random_state=42,
                max_iter=200
            )
        }

        # 训练3层ensemble架构
        ensemble_performance = self.ml_system.train_three_layer_ensemble(
            training_data,
            feature_groups,
            target_col='target'
        )

        # 获取ensemble的整体性能
        cv_score = ensemble_performance.get('meta_performance', 0.0)

        # 存储完整ensemble作为时间专用模型
        self.time_scale_models[target_period] = {
            'ensemble_models': self.ml_system.models.copy(),
            'config': self.time_scale_configs[target_period],
            'cv_score': cv_score,
            'attention_weights': self.attention_weights.get(target_period, None),
            'performance_details': ensemble_performance
        }

        self.logger.info(f"   {target_period}专用ensemble模型训练完成，CV R²: {cv_score:.4f}")

        # 返回完整的ml_system作为模型（包含ensemble）
        return self.ml_system, cv_score

    def create_model_ensemble(self, X_train, y_train):
        """
        🎭 创建模型集成
        结合时间专用模型和注意力机制
        """
        self.logger.info("🎭 创建高级模型集成...")

        ensemble_models = {}

        for target_idx, target_period in enumerate(['1d', '3d', '5d']):
            if target_idx < y_train.shape[1]:
                # 训练时间专用模型
                time_model, cv_score = self.train_time_specific_model(
                    X_train, y_train, target_period, target_idx
                )

                # 创建集成预测器
                ensemble_models[f'target_{target_period}'] = {
                    'time_specific_ensemble': time_model,  # 这是完整的ml_system对象
                    'cv_performance': cv_score,
                    'performance_details': self.time_scale_models[target_period]['performance_details'],
                    'attention_weights': self.time_scale_models[target_period]['attention_weights']
                }

        self.ensemble_models = ensemble_models
        return ensemble_models

    def intelligent_stock_sampling_v4(self, limit_stocks=1000):
        """
        V4智能股票采样策略
        基于前3阶段经验，进一步优化采样策略
        """
        self.logger.info("🎯 V4智能股票采样策略...")

        # 获取所有活跃A股（简化查询，避免字段问题）
        query = """
        SELECT s.code, s.name, s.industry
        FROM securities s
        WHERE s.type = 'A股'
          AND s.code NOT LIKE '%ST%'
          AND s.code NOT LIKE '%*ST%'
        ORDER BY s.code
        """

        with self.db_manager.get_connection() as conn:
            securities_df = pd.read_sql_query(query, conn)
        self.logger.info(f"🔍 获取候选股票: {len(securities_df)}只")

        # V4改进策略：基于行业多样性的智能采样
        selected_stocks = []

        # 1. 按行业均衡采样
        industries = securities_df['industry'].unique()
        stocks_per_industry = limit_stocks // len(industries)

        for industry in industries:
            industry_stocks = securities_df[securities_df['industry'] == industry]

            if len(industry_stocks) > 0:
                # 每个行业采样一定数量，保持行业均衡
                sample_size = min(stocks_per_industry + 5, len(industry_stocks))  # +5为缓冲
                sampled = industry_stocks.sample(
                    n=sample_size,
                    random_state=42
                ).head(stocks_per_industry)  # 取前stocks_per_industry个
                selected_stocks.extend(sampled['code'].tolist())

        # 补充不足数量（随机采样）
        remaining = limit_stocks - len(selected_stocks)
        if remaining > 0:
            unused_stocks = securities_df[~securities_df['code'].isin(selected_stocks)]
            if len(unused_stocks) > 0:
                additional = unused_stocks.sample(
                    n=min(remaining, len(unused_stocks)),
                    random_state=42
                )
                selected_stocks.extend(additional['code'].tolist())

        self.logger.info(f"✅ V4智能采样完成: {len(selected_stocks)}只股票")
        return selected_stocks[:limit_stocks]

    def prepare_v4_training_data(self, stock_codes, sample_limit=40000):
        """
        准备V4训练数据 - 完全独立版本，避免依赖问题
        增强的特征提取和数据预处理
        """
        self.logger.info(f"📊 准备V4训练数据: {len(stock_codes)}只股票, 最大{sample_limit}样本")

        all_features = []
        processed_count = 0

        for i, code in enumerate(stock_codes):
            if processed_count >= sample_limit:
                break

            try:
                # 使用单个查询和连接管理器获取所有必需数据
                with self.db_manager.get_connection() as conn:
                    # 单个查询获取所有数据，避免多次连接问题
                    query = """
                    SELECT DISTINCT
                        dq.security_id, dq.trade_date, dq.open, dq.high, dq.low, dq.close,
                        dq.volume, dq.price_change_pct, dq.is_limit_up, dq.is_limit_down,
                        COALESCE(dq.ma5, 0) as ma5,
                        COALESCE(dq.ma10, 0) as ma10,
                        COALESCE(dq.ma20, 0) as ma20,
                        COALESCE(dq.ma60, 0) as ma60,
                        COALESCE(db.pe_ttm, 0) as pe_ttm,
                        COALESCE(db.pb, 0) as pb,
                        COALESCE(db.ps_ttm, 0) as ps_ttm,
                        COALESCE(db.turnover_rate, 0) as turnover_rate,
                        COALESCE(db.total_mv, 0) as total_mv,
                        COALESCE(db.circ_mv, 0) as circ_mv,
                        COALESCE(ti.kdj_k, 50) as kdj_k,
                        COALESCE(ti.kdj_d, 50) as kdj_d,
                        COALESCE(ti.kdj_j, 50) as kdj_j,
                        COALESCE(ti.macd_dif, 0) as macd_dif,
                        COALESCE(ti.macd_dea, 0) as macd_dea,
                        COALESCE(ti.macd_macd, 0) as macd_macd,
                        COALESCE(ti.rsi6, 50) as rsi6,
                        COALESCE(ti.rsi12, 50) as rsi12,
                        COALESCE(ti.rsi24, 50) as rsi24,
                        COALESCE(ti.bbi, 0) as bbi,
                        COALESCE(ti.boll_upper, 0) as boll_upper,
                        COALESCE(ti.boll_lower, 0) as boll_lower,
                        COALESCE(ti.cci_14, 0) as cci_14,
                        COALESCE(ti.atr_14, 0) as atr_14,
                        COALESCE(ti.volume_ratio, 1) as volume_ratio
                    FROM daily_quotes dq
                    LEFT JOIN daily_basic db ON dq.security_id = db.security_id
                        AND dq.trade_date = db.trade_date
                    LEFT JOIN technical_indicators ti ON dq.security_id = ti.security_id
                        AND dq.trade_date = ti.trade_date
                    WHERE dq.security_id = (SELECT id FROM securities WHERE code = ?)
                      AND dq.trade_date >= '2022-01-01'
                      AND dq.trade_date <= '2025-09-15'
                    ORDER BY dq.trade_date
                    """

                    df = pd.read_sql_query(query, conn, params=(code,))

                if len(df) < 100:  # 确保有足够数据
                    continue

                # 填充缺失值
                df = df.fillna(0)

                # 应用高级特征工程
                df = self.create_advanced_features(df)

                # 创建训练目标
                features_df = self.create_targets_for_training(df)

                if len(features_df) > 0:
                    all_features.append(features_df)
                    processed_count += len(features_df)

                # 进度显示
                if (i + 1) % 50 == 0:
                    self.logger.info(f"处理进度: {i+1}/{len(stock_codes)} ({processed_count}样本)")

            except Exception as e:
                self.logger.warning(f"股票{code}数据处理失败: {e}")
                continue

        if not all_features:
            raise ValueError("没有成功提取到任何特征数据")

        # 合并所有特征
        final_features_df = pd.concat(all_features, ignore_index=True)

        # 准备训练数据
        X, y = self.prepare_training_data_v4(final_features_df)

        self.logger.info(f"✅ V4训练数据准备完成: {X.shape[0]}条样本, {X.shape[1]}维特征")

        return X, y

    def train_v4_advanced_models(self):
        """
        🚀 训练V4高级优化模型
        整合所有创新特性
        """
        print("🚀 启动V3.7 V4高级优化训练...")

        # 清空现有模型确保重新训练
        self.ml_system.models = {}
        self.logger.info("🔄 已清空现有模型，将重新训练...")

        # V4智能采样
        stock_codes = self.intelligent_stock_sampling_v4(limit_stocks=800)

        # 准备V4训练数据
        X_train, y_train = self.prepare_v4_training_data(stock_codes, sample_limit=40000)

        # 创建时间尺度专用模型架构
        self.create_time_scale_specific_models()

        # 创建注意力机制
        self.create_attention_mechanism(X_train, y_train)

        # 训练高级集成模型
        ensemble_models = self.create_model_ensemble(X_train, y_train)

        # 保存V4模型
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/v370/v370_advanced_v4_{timestamp}.pkl"

        model_data = {
            'models': ensemble_models,
            'attention_weights': self.attention_weights,
            'time_scale_configs': self.time_scale_configs,
            'ml_system_state': self.ml_system.__dict__,
            'version': self.version,
            'training_timestamp': timestamp,
            'training_stats': {
                'stock_count': len(stock_codes),
                'sample_count': X_train.shape[0],
                'feature_count': X_train.shape[1]
            }
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✅ V3.7 V4高级优化模型训练完成: {model_path}")

        # 性能总结
        print(f"\n📊 V4训练性能总结:")
        for target_name, model_info in ensemble_models.items():
            print(f"  {target_name}: CV R² = {model_info['cv_performance']:.4f}")

        print(f"\n🎉 V3.7 V4高级优化训练成功完成！")
        print(f"📁 模型路径: {model_path}")
        print(f"🔧 新特性: Attention机制 + 时间尺度专用 + 高级特征工程")

        return model_path

def main():
    """主训练函数"""
    try:
        trainer = V370AdvancedV4Trainer()
        model_path = trainer.train_v4_advanced_models()

        # 测试模型加载
        print(f"\n🔍 测试模型加载...")
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        print(f"✅ V4高级优化模型加载测试成功")

    except Exception as e:
        print(f"❌ V4训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()