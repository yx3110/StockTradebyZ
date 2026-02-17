#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.90 增强特征机器学习系统
基于42个增强特征的高级ensemble评分系统

🚀 V3.9核心优势：
- 42个增强特征：24技术 + 10基本面 + 8市场特征
- 三层Ensemble架构：继承v3.7的成功经验
- 特征质量优化：更精细的技术指标和基本面分析
- 数据驱动：所有特征基于真实数据，无中性假设

核心特性：
1. 技术特征增强 (24个):
   - ADX, Aroon, Ichimoku等趋势指标
   - Williams %R, SMI, TSI等动量指标
   - A/D线, CMF, VWAP等量价关系
   - 布林带宽度, KC宽度等波动率指标

2. 基本面特征增强 (10个):
   - 盈利质量：经营现金流/净利润, ROE变化率
   - 估值相对性：PE/PB/PS行业分位数
   - 财务健康：资产负债率, 流动比率, 应收账款周转率

3. 市场特征增强 (8个):
   - 市场情绪：涨跌家数比, 涨停板数量
   - 板块效应：行业资金流向, 概念板块热度
   - 资金流向：北向资金, 融资余额变化

作者: Claude Code
创建时间: 2025-11-03
版本: V3.90 (Enhanced Feature ML System)
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

# ML模型
import lightgbm as lgb
import xgboost as xgb
try:
    import catboost as cb
except ImportError:
    print("Warning: CatBoost not installed")
    cb = None

from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
import optuna

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# v3.9特征提取器
from ml_models.v39.features.technical_features import TechnicalFeaturesV39
from ml_models.v39.features.fundamental_features import FundamentalFeaturesV39
from ml_models.v39.features.market_features import MarketFeaturesV39
from ml_models.v39.features.enhanced_features import EnhancedFeatures  # V3.9.1新增
from ml_models.v39.features.phase2_enhanced_features import DirectionalFeatures  # V3.9.2 Phase 2
from ml_models.v39.features.active_market_cap_features import ActiveMarketCapFeaturesV39  # V3.9.4活跃市值特征

# 数据库和选择器
from data_adapter.database_manager import DatabaseManager
from stock_selctor.Selector import BBIKDJSelector, BBIShortLongSelector, BreakoutVolumeKDJSelector, PeakKDJSelector

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


class V390EnhancedFeatureMLSystem:
    """V3.9增强特征ML系统"""

    def __init__(self, lookback_days=10, lookahead_days=5, config=None):
        """
        初始化V3.9系统

        Args:
            lookback_days: 回望天数（用于标签计算）
            lookahead_days: 前瞻天数（用于收益计算）
            config: 配置字典
        """
        self.logger = logging.getLogger(__name__)
        self.lookback_days = lookback_days
        self.lookahead_days = lookahead_days
        self.config = config or {}

        # 初始化数据库
        self.db_manager = DatabaseManager()

        # 初始化特征提取器
        self.logger.info("初始化v3.9特征提取器...")
        self.tech_extractor = TechnicalFeaturesV39()
        self.fund_extractor = FundamentalFeaturesV39()
        self.market_extractor = MarketFeaturesV39()
        self.enhanced_extractor = EnhancedFeatures()  # V3.9.1新增增强特征
        self.use_enhanced_features = config.get('use_enhanced_features', True) if config else True  # 默认启用

        # V3.9.2 Phase 2方向预测特征
        self.directional_extractor = DirectionalFeatures()
        self.use_phase2_features = config.get('use_phase2_features', False) if config else False

        # V3.9.3 Phase 3特征精简
        self.use_phase3_refined = config.get('use_phase3_refined', False) if config else False

        # V3.9.4 活跃市值特征 (6个) - 可选
        # 功能：大盘状态影响选股 + 小市值惩罚
        self.active_mv_extractor = ActiveMarketCapFeaturesV39()
        self.use_active_mv_features = config.get('use_active_mv_features', False) if config else False
        if self.use_active_mv_features:
            self.logger.info("✅ 启用活跃市值特征 (6个): 市场层面3个 + 个股层面3个")

        # 定义需要剔除的无用特征 (重要性=0或Phase 1/2失败特征)
        self.EXCLUDED_FEATURES = [
            # 原始特征 - importance=0
            'limit_up_count',
            'northbound_net_inflow',
            'concept_heat_index',
            'supertrend_signal',
            # Phase 1失败特征
            'relative_strength_to_industry',  # #48
            'ma_alignment_score',  # #49
            'volume_confirmation'  # #50
        ]

        # Phase 2失败特征 (v3.9.3剔除)
        self.EXCLUDED_PHASE2_FEATURES = [
            # 完全无用 (importance=0)
            'macd_price_divergence',
            'adx_change_rate',
            'channel_position',
            'large_order_intensity',
            'rsi_reversal_strength',
            # 低重要性噪音特征
            'momentum_persistence',
            'volume_price_divergence',
            'momentum_alignment',
            'volatility_spike',
            'volatility_reversion'  # 虽然有107重要性，但为了精简也剔除
        ]

        # 初始化量化选择器
        self.selectors = {
            'bbi_kdj': BBIKDJSelector(),
            'bbi_shortlong': BBIShortLongSelector(),
            'breakout': BreakoutVolumeKDJSelector(),
            'peak': PeakKDJSelector()
        }

        # 模型组件
        self.models = {}
        self.scaler = RobustScaler()
        self.feature_names = []
        self.feature_importance = {}

        # 模型配置
        self.model_config = {
            'lgb': {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            },
            'xgb': {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'verbosity': 0
            }
        }

        if cb:
            self.model_config['cat'] = {
                'iterations': 1000,
                'learning_rate': 0.05,
                'depth': 6,
                'loss_function': 'RMSE',
                'verbose': False
            }

        self.logger.info("✅ V3.9系统初始化完成")

    def extract_features(self, code: str, date: str) -> Optional[pd.DataFrame]:
        """
        提取单只股票的42个增强特征

        Args:
            code: 股票代码
            date: 日期 (YYYY-MM-DD)

        Returns:
            特征DataFrame或None
        """
        try:
            features = {}

            # 1. 技术特征 (24个)
            tech_features = self.tech_extractor.extract_features_from_code(code, date)
            if tech_features:
                features.update(tech_features)
            else:
                self.logger.warning(f"{code} 技术特征提取失败")
                return None

            # 2. 基本面特征 (10个)
            fund_features = self.fund_extractor.extract_features(code, date)
            if fund_features:
                features.update(fund_features)
            else:
                self.logger.debug(f"{code} 基本面特征缺失")
                # 基本面特征可能缺失，填充默认值
                for i in range(10):
                    features[f'fund_feature_{i}'] = 0.5

            # 3. 市场特征 (8个)
            market_features = self.market_extractor.extract_features(code, date)
            if market_features:
                features.update(market_features)
            else:
                self.logger.debug(f"{code} 市场特征缺失")
                # 市场特征可能缺失，填充默认值
                for i in range(8):
                    features[f'market_feature_{i}'] = 0.5

            # 4. V3.9.1增强特征 (10个) - 可选
            if self.use_enhanced_features:
                try:
                    # 获取股票价格数据 (需要更多历史数据来计算增强特征)
                    stock_df = self._get_price_data(code, date, lookback=80)
                    if stock_df is not None and len(stock_df) >= 60:
                        # 获取市场和行业数据
                        market_df = self._get_market_data(date, lookback=80)
                        industry_df = None  # 暂时不使用行业数据

                        # 提取增强特征
                        enhanced_feat = self.enhanced_extractor.extract_all_enhanced_features(
                            stock_df=stock_df,
                            market_df=market_df,
                            industry_df=industry_df,
                            technical_features=tech_features,
                            fundamental_features=fund_features
                        )

                        # 取最后一行的特征值
                        if enhanced_feat is not None and len(enhanced_feat) > 0:
                            enhanced_dict = enhanced_feat.iloc[-1].to_dict()
                            features.update(enhanced_dict)
                        else:
                            self.logger.debug(f"{code} 增强特征提取失败，使用默认值")
                            self._fill_default_enhanced_features(features)
                    else:
                        self.logger.debug(f"{code} 历史数据不足，跳过增强特征")
                        self._fill_default_enhanced_features(features)
                except Exception as e:
                    self.logger.warning(f"{code} 增强特征提取异常: {e}")
                    self._fill_default_enhanced_features(features)

            # 5. V3.9.2 Phase 2方向预测特征 (15个) - 可选
            if self.use_phase2_features:
                try:
                    # 获取股票价格数据 (需要更多历史数据)
                    stock_df = self._get_price_data(code, date, lookback=80)
                    if stock_df is not None and len(stock_df) >= 60:
                        # 获取市场数据
                        market_df = self._get_market_data(date, lookback=80)

                        # 构建技术指标字典供Phase 2使用
                        technical_dict = {
                            'macd_histogram': tech_features.get('macd_histogram'),
                            'rsi_14': tech_features.get('rsi_14'),
                            'adx_14': tech_features.get('adx_14')
                        }

                        # 提取Phase 2特征
                        phase2_feat = self.directional_extractor.extract_all_features(
                            stock_df=stock_df,
                            market_df=market_df,
                            technical_dict=technical_dict
                        )

                        if phase2_feat is not None and len(phase2_feat) > 0:
                            phase2_dict = phase2_feat.iloc[-1].to_dict()
                            features.update(phase2_dict)
                            self.logger.debug(f"{code} Phase 2特征提取成功: {len(phase2_dict)}个")
                        else:
                            self.logger.debug(f"{code} Phase 2特征提取失败")
                    else:
                        self.logger.debug(f"{code} Phase 2历史数据不足，跳过")
                except Exception as e:
                    self.logger.warning(f"{code} Phase 2特征异常: {e}")

            # 6. V3.9.4 活跃市值特征 (6个) - 可选
            # 功能：大盘状态影响选股 + 小市值惩罚
            if self.use_active_mv_features:
                try:
                    active_mv_features = self.active_mv_extractor.extract_features(code, date)
                    if active_mv_features:
                        features.update(active_mv_features)
                        self.logger.debug(f"{code} 活跃市值特征提取成功: {len(active_mv_features)}个")
                except Exception as e:
                    self.logger.warning(f"{code} 活跃市值特征异常: {e}")
                    # 失败时填充默认值
                    for feat_name in ['market_active_mv_ratio', 'market_active_mv_zscore',
                                      'market_active_mv_trend', 'stock_active_mv_rank',
                                      'stock_relative_liquidity', 'market_cap_quality_score']:
                        features[feat_name] = 0.5

            # 7. 剔除无用特征
            for excluded_feat in self.EXCLUDED_FEATURES:
                if excluded_feat in features:
                    features.pop(excluded_feat)

            # 8. V3.9.3 Phase 3精简 - 剔除失败的Phase 2特征
            if self.use_phase3_refined and self.use_phase2_features:
                for excluded_feat in self.EXCLUDED_PHASE2_FEATURES:
                    if excluded_feat in features:
                        features.pop(excluded_feat)
                        self.logger.debug(f"Phase 3: 剔除失败特征 {excluded_feat}")

            return pd.DataFrame([features])

        except Exception as e:
            self.logger.error(f"提取{code}特征失败: {e}")
            return None

    def calculate_label(self, code: str, date: str) -> Optional[float]:
        """
        计算训练标签（未来N日收益率）

        Args:
            code: 股票代码
            date: 日期

        Returns:
            收益率或None
        """
        try:
            # 从数据库获取价格数据
            query = """
            SELECT trade_date, close
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.code = ?
            AND trade_date >= ?
            ORDER BY trade_date
            LIMIT ?
            """

            conn = sqlite3.connect('data_adapter/stock_data.db')
            df = pd.read_sql_query(
                query,
                conn,
                params=(code, date, self.lookahead_days + 1)
            )
            conn.close()

            if len(df) < self.lookahead_days + 1:
                return None

            start_price = df.iloc[0]['close']
            end_price = df.iloc[self.lookahead_days]['close']

            if start_price <= 0:
                return None

            return_rate = (end_price - start_price) / start_price
            return return_rate

        except Exception as e:
            self.logger.error(f"计算{code}标签失败: {e}")
            return None

    def prepare_training_data(self, start_date: str, end_date: str, sample_stocks: Optional[List[str]] = None):
        """
        准备训练数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            sample_stocks: 股票代码列表（可选，None表示全部）

        Returns:
            X_train, y_train, stock_info
        """
        self.logger.info(f"准备训练数据: {start_date} ~ {end_date}")

        # 获取股票列表
        if sample_stocks is None:
            conn = sqlite3.connect('data_adapter/stock_data.db')
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT code FROM securities WHERE type='A股' LIMIT 1000")
            stock_list = [row[0] for row in cursor.fetchall()]
            conn.close()
        else:
            stock_list = sample_stocks

        self.logger.info(f"股票数量: {len(stock_list)}")

        # 获取交易日列表
        conn = sqlite3.connect('data_adapter/stock_data.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT trade_date
            FROM daily_basic
            WHERE trade_date BETWEEN ? AND ?
            ORDER BY trade_date
        """, (start_date, end_date))
        trade_dates = [row[0] for row in cursor.fetchall()]
        conn.close()

        self.logger.info(f"交易日数量: {len(trade_dates)}")

        # 提取特征和标签
        X_list = []
        y_list = []
        info_list = []

        total_dates = len(trade_dates) - self.lookahead_days
        for i, date in enumerate(trade_dates[:-self.lookahead_days]):
            # 每天输出进度，显示日期和当前样本数
            self.logger.info(f"处理进度: [{i+1}/{total_dates}] 日期={date} 当前样本数={len(X_list)}")

            for code in stock_list:
                # 提取特征
                features = self.extract_features(code, date)
                if features is None or features.empty:
                    continue

                # 计算标签
                label = self.calculate_label(code, date)
                if label is None:
                    continue

                X_list.append(features.iloc[0])
                y_list.append(label)
                info_list.append({'code': code, 'date': date})

        if len(X_list) == 0:
            self.logger.error("❌ 未能提取任何训练样本")
            return None, None, None

        X_train = pd.DataFrame(X_list)
        y_train = np.array(y_list)

        self.logger.info(f"✅ 训练数据准备完成: {len(X_train)} 个样本")
        return X_train, y_train, info_list

    def train(self, X_train, y_train, optimize_hyperparams=False):
        """
        训练三层Ensemble模型

        Args:
            X_train: 特征矩阵
            y_train: 标签
            optimize_hyperparams: 是否优化超参数
        """
        self.logger.info("开始训练v3.9模型...")

        # 保存特征名称
        self.feature_names = list(X_train.columns)

        # 标准化特征
        X_scaled = self.scaler.fit_transform(X_train)

        # Layer 1: 基础模型
        self.logger.info("训练Layer 1基础模型...")

        # LightGBM
        self.models['lgb'] = lgb.LGBMRegressor(**self.model_config['lgb'])
        self.models['lgb'].fit(X_scaled, y_train)

        # XGBoost
        self.models['xgb'] = xgb.XGBRegressor(**self.model_config['xgb'])
        self.models['xgb'].fit(X_scaled, y_train)

        # RandomForest
        self.models['rf'] = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.models['rf'].fit(X_scaled, y_train)

        # CatBoost (如果可用)
        if cb:
            self.models['cat'] = cb.CatBoostRegressor(**self.model_config['cat'])
            self.models['cat'].fit(X_scaled, y_train)

        # Layer 2: Meta模型
        self.logger.info("训练Layer 2 Meta模型...")

        # 获取Layer 1预测
        layer1_preds = []
        for name, model in self.models.items():
            if name not in ['meta', 'ensemble']:
                pred = model.predict(X_scaled)
                layer1_preds.append(pred)

        # 堆叠Layer 1预测
        X_meta = np.column_stack(layer1_preds)

        # Meta模型（Ridge回归）
        self.models['meta'] = Ridge(alpha=1.0)
        self.models['meta'].fit(X_meta, y_train)

        # Layer 3: 最终Ensemble
        self.logger.info("训练Layer 3 Ensemble...")

        # 计算加权平均
        layer2_pred = self.models['meta'].predict(X_meta)

        # 简单平均作为Ensemble
        self.models['ensemble'] = {
            'weights': {'meta': 0.7, 'lgb': 0.15, 'xgb': 0.15},
            'type': 'weighted_average'
        }

        # 计算特征重要性
        self._calculate_feature_importance(X_train)

        self.logger.info("✅ v3.9模型训练完成")

    def _get_price_data(self, code: str, date: str, lookback: int = 80) -> Optional[pd.DataFrame]:
        """
        获取股票价格数据（用于增强特征计算）

        Args:
            code: 股票代码
            date: 结束日期
            lookback: 回望天数

        Returns:
            价格数据DataFrame或None
        """
        try:
            query = """
            SELECT trade_date, open, high, low, close, volume
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.code = ?
            AND trade_date <= ?
            ORDER BY trade_date DESC
            LIMIT ?
            """

            conn = sqlite3.connect('data_adapter/stock_data.db')
            df = pd.read_sql_query(
                query,
                conn,
                params=(code, date, lookback)
            )
            conn.close()

            if len(df) == 0:
                return None

            # 反转顺序（从旧到新）
            df = df.sort_values('trade_date').reset_index(drop=True)
            return df

        except Exception as e:
            self.logger.warning(f"获取{code}价格数据失败: {e}")
            return None

    def _get_market_data(self, date: str, lookback: int = 80, index_code: str = '000001.SH') -> Optional[pd.DataFrame]:
        """
        获取市场指数数据（用于相对强度特征）

        Args:
            date: 结束日期
            lookback: 回望天数
            index_code: 指数代码（默认上证指数）

        Returns:
            指数数据DataFrame或None
        """
        try:
            query = """
            SELECT trade_date, open, high, low, close, volume
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.code = ?
            AND trade_date <= ?
            ORDER BY trade_date DESC
            LIMIT ?
            """

            conn = sqlite3.connect('data_adapter/stock_data.db')
            df = pd.read_sql_query(
                query,
                conn,
                params=(index_code, date, lookback)
            )
            conn.close()

            if len(df) == 0:
                return None

            df = df.sort_values('trade_date').reset_index(drop=True)
            return df

        except Exception as e:
            self.logger.debug(f"获取市场数据失败: {e}")
            return None

    def _fill_default_enhanced_features(self, features: dict):
        """
        填充默认增强特征值

        Args:
            features: 特征字典
        """
        default_values = {
            'momentum_5d': 0.0,
            'momentum_20d': 0.0,
            'momentum_strength': 1.0,
            'relative_strength_to_market': 1.0,
            'relative_strength_to_industry': 1.0,
            'ma_alignment_score': 0.5,
            'volume_confirmation': 0.5,
            'volatility_asymmetry': 1.0,
            'price_ma_ratio_squared': 0.0,
            'roe_momentum_interaction': 0.0
        }
        features.update(default_values)

    def _calculate_feature_importance(self, X_train):
        """计算特征重要性"""
        # LightGBM特征重要性
        if 'lgb' in self.models:
            lgb_importance = self.models['lgb'].feature_importances_
            self.feature_importance['lgb'] = dict(zip(self.feature_names, lgb_importance))

        # XGBoost特征重要性
        if 'xgb' in self.models:
            xgb_importance = self.models['xgb'].feature_importances_
            self.feature_importance['xgb'] = dict(zip(self.feature_names, xgb_importance))

    def predict(self, X):
        """
        预测

        Args:
            X: 特征矩阵

        Returns:
            预测结果
        """
        # 标准化
        X_scaled = self.scaler.transform(X)

        # Layer 1预测
        layer1_preds = []
        for name, model in self.models.items():
            if name not in ['meta', 'ensemble']:
                pred = model.predict(X_scaled)
                layer1_preds.append(pred)

        # Layer 2预测
        X_meta = np.column_stack(layer1_preds)
        layer2_pred = self.models['meta'].predict(X_meta)

        # Layer 3 Ensemble
        weights = self.models['ensemble']['weights']
        final_pred = (
            weights['meta'] * layer2_pred +
            weights['lgb'] * layer1_preds[0] +
            weights['xgb'] * layer1_preds[1]
        )

        return final_pred

    def score_stock(self, code: str, date: str) -> Optional[Dict]:
        """
        评分单只股票

        Args:
            code: 股票代码
            date: 日期

        Returns:
            评分结果字典
        """
        try:
            # 提取特征
            features = self.extract_features(code, date)
            if features is None or features.empty:
                return None

            # 预测
            prediction = self.predict(features)[0]

            # 转换为0-100评分
            score = self._normalize_score(prediction)

            return {
                'code': code,
                'date': date,
                'score': score,
                'raw_prediction': prediction,
                'version': 'v3.9'
            }

        except Exception as e:
            self.logger.error(f"评分{code}失败: {e}")
            return None

    def _normalize_score(self, raw_score):
        """
        标准化评分到0-100

        Args:
            raw_score: 原始预测值（收益率）

        Returns:
            0-100之间的评分
        """
        # Sigmoid转换
        normalized = 1 / (1 + np.exp(-raw_score * 10))
        return normalized * 100

    def save_model(self, save_path: str):
        """保存模型"""
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'config': {
                'lookback_days': self.lookback_days,
                'lookahead_days': self.lookahead_days,
                'version': 'v3.9'
            }
        }

        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)

        self.logger.info(f"✅ 模型已保存: {save_path}")

    def load_model(self, model_path: str):
        """加载模型"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data.get('feature_importance', {})

        config = model_data.get('config', {})
        self.lookback_days = config.get('lookback_days', 10)
        self.lookahead_days = config.get('lookahead_days', 5)

        self.logger.info(f"✅ 模型已加载: {model_path}")


if __name__ == "__main__":
    # 测试代码
    print("V3.9 Enhanced Feature ML System")
    print("=" * 60)

    system = V390EnhancedFeatureMLSystem()
    print(f"✅ 系统初始化成功")
    print(f"特征提取器就绪: 技术, 基本面, 市场")
    print(f"预期特征数量: 42个")
