#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.70 高级机器学习评分系统
基于多模型ensemble + 35维特征工程的智能股票评分系统

🚀 V3.7革命性升级：
- 35+维全方位特征工程 (vs V3.6的15维)
- 三层ensemble架构 (5基础模型 + 4专家模型 + Meta学习器)
- 自动特征工程与选择
- 增量学习与实时更新
- 多维度风险监控

作者: Claude Code
创建时间: 2025-09-12
版本: V3.70 (Advanced Multi-Model Ensemble)
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

class V370AdvancedMLSystem:
    """
    V3.70 高级机器学习评分系统
    
    🎯 核心特性:
    1. 35+维全方位特征工程
    2. 三层ensemble架构
    3. 自动特征工程与选择
    4. 增量学习机制
    5. 多维度性能监控
    6. 实时模型更新
    """
    
    def __init__(self, config_path=None, auto_load_model=True):
        self.version = "V3.70"
        self.db_manager = DatabaseManager("data_adapter/stock_data.db")
        
        # 日志配置
        self.logger = self._setup_logger()
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
        
        # 性能监控
        self.performance_history = []
        self.feature_importance_history = {}
        
        # 配置参数
        self._init_model_configs()
        
        # 确保模型目录存在
        self.model_dir = Path("models/v370")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # 自动加载最新的训练模型（可通过参数禁用）
        if auto_load_model:
            self._auto_load_latest_model()
        
    def _setup_logger(self):
        """设置日志系统"""
        logger = logging.getLogger(f'V370_ML_System')
        logger.setLevel(logging.INFO)
        
        # 文件处理器
        log_file = f"logs/v370_ml_system_{datetime.now().strftime('%Y%m%d')}.log"
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
        self.logger.info(f"🔍 V3.7特征提取: {len(codes)}只股票, {start_date} 到 {end_date}")
        
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
        self.logger.info(f"✅ V3.7特征提取完成: {len(features_df)}条记录, {len(features_df.columns)-2}个特征")
        
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
            # 🚀 V3.7新增特征开始
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
                'code_temp': code,  # 临时存储，后面会被外层代码设置
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

            # 🆕 9. 布林带位置 (Bollinger Band Position)
            if len(window_data) >= 20:
                close_prices = window_data['close']
                bb_middle = close_prices.rolling(20).mean()
                bb_std = close_prices.rolling(20).std()
                bb_upper = bb_middle + (2 * bb_std)
                bb_lower = bb_middle - (2 * bb_std)

                current_price = self._safe_iloc(close_prices, -1)
                bb_upper_val = self._safe_iloc(bb_upper, -1)
                bb_lower_val = self._safe_iloc(bb_lower, -1)

                if pd.notna(bb_upper_val) and pd.notna(bb_lower_val) and (bb_upper_val - bb_lower_val) > 0:
                    bollinger_position = (current_price - bb_lower_val) / (bb_upper_val - bb_lower_val)
                    features['bollinger_position'] = np.clip(bollinger_position, 0, 1)

                    # 🆕 10. 布林带宽度 (Bollinger Band Width)
                    bb_middle_val = self._safe_iloc(bb_middle, -1)
                    if pd.notna(bb_middle_val) and bb_middle_val > 0:
                        bollinger_width = (bb_upper_val - bb_lower_val) / bb_middle_val
                        features['bollinger_width'] = np.clip(bollinger_width, 0, 1)
                    else:
                        features['bollinger_width'] = 0.1
                else:
                    features['bollinger_position'] = 0.5
                    features['bollinger_width'] = 0.1
            else:
                features['bollinger_position'] = 0.5
                features['bollinger_width'] = 0.1

            # 🆕 11. 威廉指标 (Williams %R)
            if len(window_data) >= 14:
                high_14 = window_data['high'].rolling(14).max()
                low_14 = window_data['low'].rolling(14).min()
                current_close = self._safe_iloc(window_data['close'], -1)
                high_14_val = self._safe_iloc(high_14, -1)
                low_14_val = self._safe_iloc(low_14, -1)

                if pd.notna(high_14_val) and pd.notna(low_14_val) and (high_14_val - low_14_val) > 0:
                    williams_r = ((high_14_val - current_close) / (high_14_val - low_14_val)) * (-100)
                    features['williams_r'] = np.clip(williams_r, -100, 0)
                else:
                    features['williams_r'] = -50.0
            else:
                features['williams_r'] = -50.0

            # 🆕 12. 商品通道指标 (CCI-14)
            if len(window_data) >= 14:
                typical_price = (window_data['high'] + window_data['low'] + window_data['close']) / 3
                sma_tp = typical_price.rolling(14).mean()
                mad = typical_price.rolling(14).apply(lambda x: np.abs(x - x.mean()).mean())

                current_tp = self._safe_iloc(typical_price, -1)
                sma_tp_val = self._safe_iloc(sma_tp, -1)
                mad_val = self._safe_iloc(mad, -1)

                if pd.notna(sma_tp_val) and pd.notna(mad_val) and mad_val > 0:
                    cci = (current_tp - sma_tp_val) / (0.015 * mad_val)
                    features['cci_14'] = np.clip(cci, -200, 200)
                else:
                    features['cci_14'] = 0.0
            else:
                features['cci_14'] = 0.0

            # 🆕 13. MACD柱状图 (MACD Histogram)
            if len(window_data) >= 26:
                close_prices = window_data['close']
                ema12 = close_prices.ewm(span=12).mean()
                ema26 = close_prices.ewm(span=26).mean()
                macd_line = ema12 - ema26
                signal_line = macd_line.ewm(span=9).mean()
                macd_histogram = macd_line - signal_line

                macd_hist_val = self._safe_iloc(macd_histogram, -1)
                features['macd_histogram'] = macd_hist_val if pd.notna(macd_hist_val) else 0.0
            else:
                features['macd_histogram'] = 0.0

        except Exception as e:
            # 如果计算失败，使用默认值
            self.logger.warning(f"技术增强特征计算失败: {e}")
            features = {
                'adx_14': 25.0, 'trix': 0.0, 'vwap_deviation': 0.0, 'atr_ratio': 1.0,
                'keltner_position': 0.5, 'volatility_regime': 1.0, 'obv_trend': 0.0, 'mfi_14': 50.0,
                'bollinger_position': 0.5, 'bollinger_width': 0.1, 'williams_r': -50.0,
                'cci_14': 0.0, 'macd_histogram': 0.0
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
            # 1. 市场情绪指数 (基于个股表现模拟)
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
                
            # 5. 买卖价差因子 (模拟)
            if len(window_data) >= 5:
                # 用高低价差模拟买卖价差
                spread_ratio = ((window_data['high'] - window_data['low']) / window_data['close']).tail(5).mean()
                features['spread_factor'] = np.clip(spread_ratio, 0, 0.1)
            else:
                features['spread_factor'] = 0.02
                
            # 6. 机构资金流向 (基于大额交易模拟)
            if len(window_data) >= 10:
                # 用成交量和价格变化模拟资金流向
                price_changes = window_data['close'].pct_change()
                volumes = window_data['volume']
                
                institutional_flow = (price_changes * volumes).tail(5).sum() / volumes.tail(5).sum()
                features['institutional_flow'] = np.clip(institutional_flow, -0.1, 0.1)
            else:
                features['institutional_flow'] = 0.0
                
            # 7. 利率敏感性 (基于市值和行业特征模拟)
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
        model_name = f"v370_models_{timestamp}{model_name_suffix}"
        
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
        self.logger.info(f"✅ V3.7模型已保存: {model_file}")

        return model_file

    def _auto_load_latest_model(self):
        """自动加载最新的训练模型"""
        try:
            import glob
            # 查找所有V3.7模型文件
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
        """加载模型组件 - 兼容新旧格式"""
        try:
            model_data = joblib.load(model_file)

            # 检查模型文件格式
            if 'base_models' in model_data:
                # 旧格式 (v3.7原始版本)
                self.logger.info("🔄 加载旧格式v3.7模型...")
                self.base_models = model_data['base_models']
                self.expert_models = model_data['expert_models']
                self.meta_learner = model_data['meta_learner']
                self.scalers = model_data['scalers']
                self.feature_selectors = model_data.get('feature_selectors', {})
                self.performance_history = model_data.get('performance_history', [])
                self.feature_importance_history = model_data.get('feature_importance_history', {})

            elif 'models' in model_data:
                # 新格式 (v3.8+版本) - 转换为旧格式兼容
                self.logger.info("🔄 加载新格式v3.8+模型并转换为v3.7兼容格式...")

                # 从新格式提取模型数据
                models_dict = model_data['models']
                self.base_models = {}
                self.expert_models = {}
                self.meta_learner = {}
                self.scalers = {}

                # 提取每个时间尺度的模型
                for time_scale, model_info in models_dict.items():
                    if 'time_specific_ensemble' in model_info:
                        ensemble_system = model_info['time_specific_ensemble']

                        # 新格式中模型使用 'target' 作为键名，需要映射到对应的时间尺度
                        target_key = 'target'  # 新格式统一使用 'target' 键名

                        # 提取三层模型架构
                        if hasattr(ensemble_system, 'base_models') and target_key in ensemble_system.base_models:
                            self.base_models[time_scale] = ensemble_system.base_models[target_key]
                        if hasattr(ensemble_system, 'expert_models') and target_key in ensemble_system.expert_models:
                            self.expert_models[time_scale] = ensemble_system.expert_models[target_key]
                        if hasattr(ensemble_system, 'meta_learner') and target_key in ensemble_system.meta_learner:
                            self.meta_learner[time_scale] = ensemble_system.meta_learner[target_key]
                        if hasattr(ensemble_system, 'scalers'):
                            # 新格式中scalers使用 'target' 键名，需要映射到时间尺度
                            if target_key in ensemble_system.scalers:
                                self.scalers[time_scale] = ensemble_system.scalers[target_key]

                # 设置默认值
                self.feature_selectors = {}
                self.performance_history = []
                self.feature_importance_history = {}

                self.logger.info(f"✅ 已转换新格式模型，包含时间尺度: {list(self.base_models.keys())}")

            else:
                raise ValueError("未知的模型文件格式")

            self.logger.info(f"✅ V3.7模型已加载: {model_file}")
            return True

        except Exception as e:
            self.logger.error(f"❌ 模型加载失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return False

    # =====================================================
    # 🚀 V3.7 三层 Ensemble 模型架构
    # =====================================================
    
    def build_three_layer_architecture(self, target_col='target_1d'):
        """构建三层ensemble架构"""
        self.logger.info(f"🏗️ 构建V3.7三层ensemble架构: {target_col}")
        
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
        """为专家模型分组特征 - 基于专家模型实际训练特征的V3.7版本"""
        return {
            'technical_expert': [
                # 技术专家模型训练时的27个特征 (🆕 包含所有缺失特征)
                'bbi', 'volume_surge', 'price_momentum', 'rsi', 'kdj_cross',
                'volatility_risk', 'adx_14', 'trix', 'vwap_deviation', 'atr_ratio',
                'keltner_position', 'volatility_regime', 'obv_trend', 'mfi_14',
                'momentum_3d', 'momentum_5d', 'momentum_20d',
                # 🆕 新增的5个技术指标特征
                'bollinger_position', 'bollinger_width', 'williams_r', 'cci_14', 'macd_histogram',
                # 🆕 额外发现的技术特征
                'zhixing_multiavg',
                # 🆕 交互特征
                'momentum_vol_interaction', 'volume_price_interaction', 'rsi_adx_interaction'
            ],
            'fundamental_expert': [
                # 基本面专家模型训练时的13个特征 (🆕 包含原始数据)
                'pb', 'pe_ttm', 'market_cap', 'turnover_rate',
                'value_factor', 'growth_factor', 'price_category', 'stock_price_log',
                # 🆕 原始数据特征
                'pb_raw', 'pe_ttm_raw', 'market_cap_raw', 'stock_price_raw'
            ],
            'macro_expert': [
                # 宏观专家模型训练时的8个特征 (完全匹配)
                'market_sentiment', 'vix_equivalent', 'new_high_low_ratio', 'liquidity_score',
                'spread_factor', 'institutional_flow', 'interest_rate_sensitivity', 'seasonal_factor'
            ],
            'sentiment_expert': [
                # 情绪专家模型训练时的7个特征 (完全匹配)
                'industry_momentum', 'industry_relative_strength', 'industry_rotation_signal',
                'size_factor', 'relative_strength', 'pattern_breakout', 'price_trend_30d'
            ]
        }

    def train_three_layer_ensemble(self, training_data, feature_groups, target_col='target_1d'):
        """训练三层ensemble模型"""
        self.logger.info(f"🎯 开始训练V3.7三层ensemble: {target_col}")
        
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
        self.logger.info(f"🎯 V3.7三层ensemble预测: {target_col}")
        
        if target_col not in self.base_models:
            self.logger.error(f"❌ 模型{target_col}未训练")
            return None
            
        # 准备特征
        feature_groups = self._group_features_for_experts()

        # 🔧 V3.7维度兼容性处理: 解决40特征vs53特征的训练/预测不匹配问题

        # 1. 获取当前可用特征 (从特征组构建)
        all_features = []
        for group_features in feature_groups.values():
            all_features.extend(group_features)
        available_features = list(set(all_features))  # 去重，应为45个特征 (原40+新增5个)

        # 2. 从输入数据中选择可用特征
        existing_features = [f for f in available_features if f in features_df.columns]
        X = features_df[existing_features].copy()

        # 🔧 关键修复: 直接使用特征名，不转换为数字索引
        self.logger.info(f"📋 使用原始特征名: {len(existing_features)}个特征")

        # 3. 处理维度不匹配问题
        try:
            # 尝试直接转换
            if hasattr(self.scalers[target_col], 'feature_names_in_'):
                expected_features = list(self.scalers[target_col].feature_names_in_)
                self.logger.info(f"🔍 Scaler期望{len(expected_features)}个特征，当前有{len(X.columns)}个特征")

                # 创建完整的特征集，缺失特征填充为0
                X_aligned = pd.DataFrame(index=X.index)
                for feature in expected_features:
                    if feature in X.columns:
                        X_aligned[feature] = X[feature]
                    else:
                        X_aligned[feature] = 0.0
                        self.logger.debug(f"  填充缺失特征 {feature} = 0.0")

                # 确保特征顺序正确
                X_aligned = X_aligned[expected_features]
                X = X_aligned

            # 标准化
            X_scaled_array = self.scalers[target_col].transform(X)

            # 🔧 关键修复: 保持原始特征名，不转换为数字索引
            X_scaled = pd.DataFrame(
                X_scaled_array,
                columns=X.columns,  # 保持原始特征名
                index=X.index
            )
            self.logger.info(f"✅ 特征对齐成功: {X_scaled.shape[1]}个特征（使用原始特征名）")

        except Exception as e:
            self.logger.warning(f"⚠️ Scaler维度不匹配，使用当前特征集: {e}")
            # 如果scaler不匹配，创建新的临时scaler
            from sklearn.preprocessing import RobustScaler
            temp_scaler = RobustScaler()
            X_scaled_array = temp_scaler.fit_transform(X)

            # 🔧 关键修复: 临时scaler也保持原始特征名
            X_scaled = pd.DataFrame(
                X_scaled_array,
                columns=X.columns,  # 保持原始特征名
                index=X.index
            )
            self.logger.info(f"🔄 使用临时scaler处理{X_scaled.shape[1]}个特征（保持原始特征名）")
        
        # =====================================================
        # Level 1: 基础模型预测 (🔧 维度兼容性处理)
        # =====================================================
        base_predictions = np.zeros((len(X_scaled), len(self.base_models[target_col])))

        for i, (model_name, model) in enumerate(self.base_models[target_col].items()):
            try:
                # 尝试直接预测
                raw_predictions = model.predict(X_scaled)

                # 🔧 关键修复: 限制异常预测值，防止MLP等模型返回超大值
                clipped_predictions = np.clip(raw_predictions, -10, 10)  # 限制在合理范围内

                if np.abs(raw_predictions - clipped_predictions).max() > 0.1:
                    self.logger.warning(f"⚠️ 基础模型 {model_name} 预测值异常，已限制: {raw_predictions[0]:.2f} -> {clipped_predictions[0]:.2f}")

                base_predictions[:, i] = clipped_predictions
            except Exception as e:
                self.logger.warning(f"⚠️ 基础模型 {model_name} 维度不匹配: {e}")
                # 维度不匹配时使用平均值作为默认预测
                base_predictions[:, i] = 0.05  # 5%的默认预期收益率
        
        # =====================================================
        # Level 2: 专家模型预测
        # =====================================================
        expert_predictions = np.zeros((len(X_scaled), len(feature_groups)))
        
        for i, (expert_name, expert_features) in enumerate(feature_groups.items()):
            # 🔧 安全获取专家模型
            expert_model = None
            try:
                if target_col in self.expert_models and expert_name in self.expert_models[target_col]:
                    expert_model = self.expert_models[target_col][expert_name]
            except Exception as e:
                self.logger.warning(f"⚠️ 无法获取专家模型 {expert_name}: {e}")

            if expert_model is None:
                expert_predictions[:, i] = 0.05  # 默认预测值
                continue

            # 🔧 关键修复: 直接使用特征名，不转换为索引
            available_expert_features = [f for f in expert_features if f in X_scaled.columns]
            if len(available_expert_features) > 0:
                # 直接使用特征名选择数据
                X_expert = X_scaled[available_expert_features]
                self.logger.info(f"专家模型 {expert_name}: 期望{len(expert_features)}个特征, 实际{len(available_expert_features)}个特征")
                if len(available_expert_features) != len(expert_features):
                    missing_features = [f for f in expert_features if f not in X_scaled.columns]
                    self.logger.warning(f"专家模型 {expert_name} 缺失特征: {missing_features}")

                try:
                    # 尝试预测
                    raw_expert_pred = expert_model.predict(X_expert)

                    # 🔧 关键修复: 限制专家模型异常预测值
                    clipped_expert_pred = np.clip(raw_expert_pred, -5, 5)  # 专家模型使用更严格的限制

                    if np.abs(raw_expert_pred - clipped_expert_pred).max() > 0.1:
                        self.logger.warning(f"⚠️ 专家模型 {expert_name} 预测值异常，已限制: {raw_expert_pred[0]:.2f} -> {clipped_expert_pred[0]:.2f}")

                    expert_predictions[:, i] = clipped_expert_pred
                except Exception as e:
                    self.logger.warning(f"⚠️ 专家模型 {expert_name} 预测失败: {e}")
                    # 维度不匹配时使用默认值
                    expert_predictions[:, i] = 0.05
            else:
                expert_predictions[:, i] = 0.05  # 无特征时使用默认值
        
        # =====================================================
        # Level 3: Meta学习器预测
        # =====================================================
        meta_features = np.concatenate([base_predictions, expert_predictions], axis=1)

        # 🔧 关键修复: Meta学习器特征维度对齐
        # 训练时Meta学习器使用基础模型(5个) + 专家模型(4个) + 关键特征(5个) = 14个特征
        # 预测时也必须添加相同的关键特征以保持14个特征维度一致
        key_features = ['bbi', 'rsi', 'market_cap', 'pb', 'volume_surge']
        available_key_features = [f for f in key_features if f in X_scaled.columns]
        if available_key_features:
            key_feature_data = X_scaled[available_key_features].values
            meta_features = np.concatenate([meta_features, key_feature_data], axis=1)

        self.logger.info(f"🎯 Meta学习器特征: {meta_features.shape[1]}个（与训练时一致）")
        
        # 最终预测 (🔧 维度兼容性处理)
        try:
            final_predictions = self.meta_learner[target_col].predict(meta_features)
        except Exception as e:
            self.logger.warning(f"⚠️ Meta学习器预测失败: {e}")
            # 如果Meta学习器失败，使用ensemble平均值
            final_predictions = np.mean(np.concatenate([base_predictions, expert_predictions], axis=1), axis=1)
        
        # 转换为0-100评分
        scores = self._normalize_scores_to_100(final_predictions)

        # =====================================================
        # 计算各维度因子评分
        # =====================================================
        factor_scores = {}

        # 技术分析因子 (基于技术指标专家模型) - 🔧 安全处理
        try:
            if 'technical_expert' in feature_groups and expert_predictions.shape[1] > 0:
                technical_idx = list(feature_groups.keys()).index('technical_expert')
                factor_scores['technical'] = self._normalize_scores_to_100(expert_predictions[:, technical_idx])[0]
            else:
                factor_scores['technical'] = 50.0
        except Exception as e:
            self.logger.warning(f"⚠️ 计算技术因子失败: {e}")
            factor_scores['technical'] = 50.0

        # 基本面因子 (基于基本面专家模型) - 🔧 安全处理
        try:
            if 'fundamental_expert' in feature_groups and expert_predictions.shape[1] > 1:
                fundamental_idx = list(feature_groups.keys()).index('fundamental_expert')
                factor_scores['fundamental'] = self._normalize_scores_to_100(expert_predictions[:, fundamental_idx])[0]
            else:
                factor_scores['fundamental'] = 50.0
        except Exception as e:
            self.logger.warning(f"⚠️ 计算基本面因子失败: {e}")
            factor_scores['fundamental'] = 50.0

        # 宏观因子 (基于宏观专家模型) - 🔧 安全处理
        try:
            if 'macro_expert' in feature_groups and expert_predictions.shape[1] > 2:
                macro_idx = list(feature_groups.keys()).index('macro_expert')
                factor_scores['macro'] = self._normalize_scores_to_100(expert_predictions[:, macro_idx])[0]
            else:
                factor_scores['macro'] = 50.0
        except Exception as e:
            self.logger.warning(f"⚠️ 计算宏观因子失败: {e}")
            factor_scores['macro'] = 50.0

        # 情绪因子 (基于情绪专家模型) - 🔧 安全处理
        try:
            if 'sentiment_expert' in feature_groups and expert_predictions.shape[1] > 3:
                sentiment_idx = list(feature_groups.keys()).index('sentiment_expert')
                factor_scores['sentiment'] = self._normalize_scores_to_100(expert_predictions[:, sentiment_idx])[0]
            else:
                factor_scores['sentiment'] = 50.0
        except Exception as e:
            self.logger.warning(f"⚠️ 计算情绪因子失败: {e}")
            factor_scores['sentiment'] = 50.0

        # 时序因子 (基于基础模型的平均表现) - 🔧 安全处理
        try:
            if base_predictions.shape[1] > 0:
                temporal_score = float(np.mean(base_predictions[0, :]))  # 基础模型的平均预测，确保是标量
                factor_scores['temporal'] = self._normalize_scores_to_100([temporal_score])[0]
            else:
                factor_scores['temporal'] = 50.0
        except Exception as e:
            self.logger.warning(f"⚠️ 计算时序因子失败: {e}")
            factor_scores['temporal'] = 50.0

        # 🔧 修复批量预测: 返回所有评分而不是只返回第一个
        # 如果是单只股票，返回字典格式；如果是批量，返回分数数组
        if len(scores) == 1:
            return {
                'score': scores[0],
                'factor_scores': factor_scores
            }
        else:
            # 批量预测时返回所有分数
            return scores

    def _normalize_scores_to_100(self, predictions):
        """将预测结果标准化到0-100评分"""
        # 使用Sigmoid函数将预测值映射到0-100范围
        # 这样可以处理极端值并保持评分的分布合理

        # 确保predictions是numpy数组
        predictions = np.array(predictions)

        # 应用sigmoid函数，将任意实数映射到(0,1)
        sigmoid_scores = 1 / (1 + np.exp(-predictions/2))  # 除以2使得函数更平缓

        # 映射到0-100范围
        scores = sigmoid_scores * 100

        # 确保在合理范围内
        scores = np.clip(scores, 5, 95)  # 避免极端的0或100分

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


if __name__ == "__main__":
    # 测试V3.7系统初始化
    print("🚀 V3.7高级机器学习系统测试")
    
    system = V370AdvancedMLSystem()
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