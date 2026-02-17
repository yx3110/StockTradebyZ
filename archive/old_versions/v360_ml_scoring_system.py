#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.60 机器学习评分系统
基于LightGBM和XGBoost的智能股票评分系统

作者: Claude Code
创建时间: 2025-09-09
版本: V3.60 (Machine Learning Enhanced)
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
warnings.filterwarnings('ignore')

# ML模型
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import VotingRegressor

# 现有模块
sys.path.append('/Users/yangxu/StockTradebyZ')
from data_adapter.database_manager import DatabaseManager
from stock_selctor.Selector import BBIKDJSelector, BBIShortLongSelector, BreakoutVolumeKDJSelector, PeakKDJSelector

class V360MLScoringSystem:
    """
    V3.60 机器学习评分系统
    
    核心特性:
    1. 双模型ensemble (LightGBM + XGBoost)
    2. 时间序列交叉验证
    3. 自动特征工程
    4. 防过拟合机制
    5. 实时模型更新
    """
    
    def __init__(self, config_path=None):
        self.version = "V3.60"
        self.db_manager = DatabaseManager("data_adapter/stock_data.db")
        
        # 模型配置
        self.lgb_params = {
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
            'reg_lambda': 0.1
        }
        
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        }
        
        # 特征选择器实例
        self.selectors = {
            'bbi_kdj': BBIKDJSelector(),
            'bbi_shortlong': BBIShortLongSelector(), 
            'breakout_volume': BreakoutVolumeKDJSelector(),
            'peak_kdj': PeakKDJSelector()
        }
        
        # 模型存储
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # 日志
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """设置日志"""
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/v360_ml_system_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger('V360MLSystem')
        
    def extract_features(self, codes, start_date, end_date):
        """
        提取特征数据
        
        基于qlib_integration优化结果的12个核心特征:
        - 4个选择器的原始指标
        - 技术指标增强
        - 基本面特征
        - 市场行为特征
        """
        self.logger.info(f"提取特征数据: {len(codes)}只股票, {start_date} 到 {end_date}")
        
        # 如果是单日预测，需要扩展历史数据范围
        from datetime import datetime, timedelta
        if start_date == end_date:
            # 向前扩展60天确保有足够历史数据
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
            extended_start_date = (end_date_obj - timedelta(days=90)).strftime('%Y-%m-%d')
            self.logger.info(f"单日预测模式，扩展数据范围至: {extended_start_date} - {end_date}")
        else:
            extended_start_date = start_date
        
        features_list = []
        
        for i, code in enumerate(codes):
            if i % 100 == 0:
                self.logger.info(f"处理进度: {i}/{len(codes)}")
                
            try:
                # 获取基础数据
                base_data = self._get_stock_data(code, extended_start_date, end_date)
                if base_data is None or len(base_data) < 30:
                    self.logger.warning(f"股票{code}数据不足: {len(base_data) if base_data is not None else 0}天")
                    continue
                    
                # 如果是单日预测，只计算最后一天的特征
                if start_date == end_date:
                    # 计算最后一天的特征（基于历史数据）
                    daily_features = self._compute_core_features(code, base_data, len(base_data)-1)
                    if daily_features is not None:
                        daily_features['code'] = code
                        features_list.append(daily_features)
                else:
                    # 计算12个核心特征 - 针对每一天计算
                    for idx in range(20, len(base_data)):  # 从第20天开始，确保有足够历史数据
                        daily_features = self._compute_core_features(code, base_data, idx)
                        if daily_features is not None:
                            daily_features['code'] = code
                            features_list.append(daily_features)
                    
            except Exception as e:
                self.logger.warning(f"股票{code}特征提取失败: {e}")
                continue
                
        if not features_list:
            self.logger.error("未能提取任何特征数据")
            return None
            
        features_df = pd.concat(features_list, ignore_index=True)
        self.logger.info(f"特征提取完成: {len(features_df)}条记录")
        
        return features_df
    
    def _get_stock_data(self, code, start_date, end_date):
        """获取股票基础数据"""
        query = """
        SELECT dq.trade_date, dq.open, dq.high, dq.low, dq.close, dq.volume,
               dq.price_change_pct, db.pe_ttm, db.pb, db.total_mv as market_cap, db.turnover_rate,
               ti.ma14, ti.ma28, ti.ma57, ti.bbi, ti.rsi6, ti.rsi12, ti.kdj_k, ti.kdj_d,
               ti.zhixing_short_trend, ti.zhixing_multi_kong
        FROM daily_quotes dq
        LEFT JOIN daily_basic db ON dq.security_id = db.security_id AND dq.trade_date = db.trade_date
        LEFT JOIN technical_indicators ti ON dq.security_id = ti.security_id AND dq.trade_date = ti.trade_date
        JOIN securities s ON dq.security_id = s.id
        WHERE s.code = ? AND dq.trade_date BETWEEN ? AND ?
        ORDER BY dq.trade_date
        """
        
        try:
            with self.db_manager.get_connection() as conn:
                return pd.read_sql_query(query, conn, 
                                       params=[code, start_date, end_date])
        except Exception as e:
            self.logger.warning(f"获取{code}数据失败: {e}")
            return None
    
    def _compute_core_features(self, code, data, idx=None):
        """计算12个核心特征 (基于qlib优化结果) - 修复版本"""
        if len(data) < 5:
            return None
            
        try:
            # 使用指定索引或最新数据
            if idx is None:
                idx = len(data) - 1
            latest = data.iloc[idx]
            
            # 获取到当前日期为止的历史数据窗口
            window_data = data.iloc[:idx+1]
            
            # 1. BBI相关特征 (权重17.1%) - 直接使用数据库中的BBI
            bbi = latest['bbi'] if pd.notna(latest['bbi']) else window_data['close'].rolling(min(20, len(window_data))).mean().iloc[-1]
            bbi_ratio = (latest['close'] / bbi - 1) * 100 if bbi > 0 else 0.0  # 转换为百分比差异
            
            # 2. 成交量特征 (权重13.6%) - 改进计算逻辑
            if len(window_data) >= 5:
                volume_ma5 = window_data['volume'].rolling(5).mean().iloc[-1]
                volume_surge = (latest['volume'] / volume_ma5 - 1) * 100 if volume_ma5 > 0 else 0.0
            else:
                volume_surge = 0.0
            
            # 3. 价格动量 (权重13.4%) - 修复计算，使用合理的时间窗口
            if len(window_data) >= 5:
                price_momentum = ((window_data['close'].iloc[-1] / window_data['close'].iloc[-5]) - 1) * 100
            else:
                price_momentum = 0.0
            # 限制价格动量在合理范围内
            price_momentum = np.clip(price_momentum, -50, 50)
            
            # 4. 知行多均线 (权重11.3%) - 直接使用数据库中的知行多空线
            if pd.notna(latest['zhixing_multi_kong']):
                zhixing_multiavg = (latest['zhixing_multi_kong'] / latest['close'] - 1) * 100 if latest['close'] > 0 else 0.0
            else:
                # 如果数据库没有知行多空线，使用传统计算方法
                ma_14 = latest['ma14'] if pd.notna(latest['ma14']) else window_data['close'].rolling(min(14, len(window_data))).mean().iloc[-1]
                ma_28 = latest['ma28'] if pd.notna(latest['ma28']) else window_data['close'].rolling(min(28, len(window_data))).mean().iloc[-1]
                zhixing_multiavg = (ma_14 / ma_28 - 1) * 100 if ma_28 > 0 else 0.0
            
            # 5. RSI (权重10.7%) - 直接使用数据库中的RSI6
            if pd.notna(latest['rsi6']):
                rsi = latest['rsi6']
            elif pd.notna(latest['rsi12']):
                rsi = latest['rsi12']
            else:
                rsi = 50.0  # 默认中性值
            
            # 6. 市值因子 (权重8.9%) - 使用对数变换
            market_cap = latest['market_cap'] if pd.notna(latest['market_cap']) else 10000
            market_cap_log = np.log(max(market_cap, 1))  # 防止负数或0
            
            # 7. KDJ指标 (权重8.1%) - 直接使用数据库中的KDJ
            kdj_k = latest['kdj_k'] if pd.notna(latest['kdj_k']) else 50.0
            kdj_d = latest['kdj_d'] if pd.notna(latest['kdj_d']) else 50.0
            kdj_cross = kdj_k - kdj_d
            
            # 8. PB估值 (权重5.6%) - 改进PB处理
            pb = latest['pb'] if pd.notna(latest['pb']) and latest['pb'] > 0 else 2.0  # 默认合理PB值
            pb_inverse = 1.0 / pb if pb > 0 else 0.5  # PB倒数，高PB对应低分
            
            # 9. 换手率 (权重4.9%) - 直接使用
            turnover_rate = latest['turnover_rate'] if pd.notna(latest['turnover_rate']) else 1.0
            turnover_rate = max(0.01, min(turnover_rate, 50.0))  # 限制在合理范围
            
            # 10. 波动率风险 (权重3.2%) - 改进波动率计算
            if len(window_data) >= 10:
                # 使用更稳健的波动率计算：绝对价格变化的标准差
                price_changes = window_data['price_change_pct'].rolling(min(10, len(window_data))).std().iloc[-1]
                volatility_risk = abs(price_changes) * 100 if pd.notna(price_changes) else 2.0
            else:
                price_changes = window_data['price_change_pct'].std() if len(window_data) > 1 else 0.02
                volatility_risk = abs(price_changes) * 100 if pd.notna(price_changes) else 2.0
            # 不要过度限制波动率范围，保留真实的市场波动特征
            volatility_risk = max(0.1, min(volatility_risk, 50.0))  # 扩大上限到50%
            
            # 11. 相对强度 (权重2.1%) - 改进相对强度计算
            if len(window_data) >= 20:
                relative_strength = (window_data['close'].iloc[-1] / window_data['close'].iloc[-20] - 1) * 100
            elif len(window_data) >= 10:
                relative_strength = (window_data['close'].iloc[-1] / window_data['close'].iloc[-10] - 1) * 100
            else:
                relative_strength = 0.0
            relative_strength = np.clip(relative_strength, -50, 50)  # 限制在合理范围
            
            # 12. PE估值 (权重1.1%) - 改进PE处理
            pe_ttm = latest['pe_ttm'] if pd.notna(latest['pe_ttm']) and latest['pe_ttm'] > 0 else 20.0  # 默认合理PE值
            pe_inverse = 1.0 / pe_ttm if pe_ttm > 0 else 0.05  # PE倒数，高PE对应低分
            
            # 13. 股价绝对值特征 - 价格本身的影响
            stock_price = latest['close'] if pd.notna(latest['close']) else 10.0
            price_log = np.log(max(stock_price, 0.1))  # 价格对数化，处理不同价格区间
            
            # 14. 价格区间特征 - 判断是否为低价股/高价股
            if stock_price <= 5:
                price_category = 1  # 低价股
            elif stock_price <= 20:
                price_category = 2  # 中低价股
            elif stock_price <= 50:
                price_category = 3  # 中价股
            else:
                price_category = 4  # 高价股
            
            # 15. 价格趋势强度 - 更长期的价格走势
            if len(window_data) >= 20:
                price_trend_30d = (window_data['close'].iloc[-1] / window_data['close'].iloc[-20] - 1) * 100
            elif len(window_data) >= 10:
                price_trend_30d = (window_data['close'].iloc[-1] / window_data['close'].iloc[-10] - 1) * 100
            elif len(window_data) >= 5:
                price_trend_30d = (window_data['close'].iloc[-1] / window_data['close'].iloc[-5] - 1) * 100
            else:
                price_trend_30d = 0.0
            price_trend_30d = np.clip(price_trend_30d, -100, 100)

            features = pd.DataFrame([{
                'trade_date': latest['trade_date'],
                'bbi': bbi_ratio,  # BBI差异百分比
                'volume_surge': volume_surge,  # 成交量激增百分比
                'price_momentum': price_momentum,  # 价格动量百分比
                'zhixing_multiavg': zhixing_multiavg,  # 知行多空线差异百分比
                'rsi': rsi,  # RSI指标（0-100）
                'market_cap': market_cap_log,  # 市值对数
                'kdj_cross': kdj_cross,  # KDJ金叉死叉
                'pb': pb_inverse,  # PB倒数
                'turnover_rate': turnover_rate,  # 换手率
                'volatility_risk': volatility_risk,  # 波动率风险
                'relative_strength': relative_strength,  # 相对强度百分比
                'pe_ttm': pe_inverse,  # PE倒数
                'stock_price_log': price_log,  # 股价对数
                'price_category': price_category,  # 价格区间类别
                'price_trend_30d': price_trend_30d,  # 30日价格趋势
                # 原始值用于显示
                'pb_raw': pb,  # 原始PB值
                'pe_ttm_raw': pe_ttm,  # 原始PE值  
                'market_cap_raw': market_cap,  # 原始市值
                'stock_price_raw': latest['close'] if pd.notna(latest['close']) else 10.0,  # 原始股价
            }])
            
            return features
            
        except Exception as e:
            self.logger.warning(f"计算{code}特征失败: {e}")
            return None
    
    def prepare_target(self, features_df, target_days=[1, 3, 5, 10]):
        """准备训练目标 (未来收益率)"""
        self.logger.info(f"计算目标变量: {target_days}日收益率")
        
        targets = {}
        
        for code in features_df['code'].unique():
            code_data = features_df[features_df['code'] == code].copy()
            code_data = code_data.sort_values('trade_date')
            
            # 获取价格数据用于计算未来收益
            try:
                price_data = self._get_price_series(code, code_data['trade_date'].min(), 
                                                  code_data['trade_date'].max())
                if price_data is None:
                    continue
                    
                for days in target_days:
                    target_returns = []
                    for _, row in code_data.iterrows():
                        future_return = self._calculate_future_return(row['trade_date'], days, price_data)
                        target_returns.append(future_return)
                    
                    code_data[f'target_{days}d'] = target_returns
                    
                targets[code] = code_data
                
            except Exception as e:
                self.logger.warning(f"计算{code}目标变量失败: {e}")
                continue
        
        if not targets:
            return None
            
        result_df = pd.concat(targets.values(), ignore_index=True)
        result_df = result_df.dropna()
        
        self.logger.info(f"目标变量计算完成: {len(result_df)}条有效记录")
        return result_df
    
    def _get_price_series(self, code, start_date, end_date):
        """获取价格序列"""
        query = """
        SELECT trade_date, close
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id  
        WHERE s.code = ? AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date
        """
        
        try:
            with self.db_manager.get_connection() as conn:
                return pd.read_sql_query(query, conn,
                                       params=[code, start_date, end_date])
        except:
            return None
    
    def _calculate_future_return(self, current_date, days, price_data):
        """计算未来N日收益率"""
        try:
            current_idx = price_data[price_data['trade_date'] == current_date].index
            if len(current_idx) == 0:
                return 0.0
                
            current_idx = current_idx[0]
            future_idx = min(current_idx + days, len(price_data) - 1)
            
            current_price = price_data.iloc[current_idx]['close']
            future_price = price_data.iloc[future_idx]['close']
            
            return (future_price / current_price - 1) * 100
            
        except:
            return 0.0
    
    def train_models(self, features_df, target_col='target_1d'):
        """训练LightGBM和XGBoost模型"""
        self.logger.info(f"开始训练V3.60模型，目标变量: {target_col}")
        
        # 准备训练数据 - 更新为15个特征（包含新的价格特征）
        feature_cols = ['bbi', 'volume_surge', 'price_momentum', 'zhixing_multiavg', 
                       'rsi', 'market_cap', 'kdj_cross', 'pb', 'turnover_rate',
                       'volatility_risk', 'relative_strength', 'pe_ttm',
                       'stock_price_log', 'price_category', 'price_trend_30d']
        
        X = features_df[feature_cols].copy()
        y = features_df[target_col].copy()
        
        # 数据清洗
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X, y = X[mask], y[mask]
        
        if len(X) < 100:
            self.logger.error(f"训练数据不足: {len(X)}条")
            return False
        
        # 特征标准化
        self.scalers[target_col] = RobustScaler()
        X_scaled = self.scalers[target_col].fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
        
        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=3)
        
        lgb_scores, xgb_scores = [], []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            self.logger.info(f"训练第{fold+1}折")
            
            X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # LightGBM
            lgb_train = lgb.Dataset(X_train, label=y_train)
            lgb_val = lgb.Dataset(X_val, label=y_val)
            
            lgb_model = lgb.train(
                self.lgb_params,
                lgb_train,
                valid_sets=[lgb_val],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
            )
            
            lgb_pred = lgb_model.predict(X_val)
            lgb_score = np.sqrt(mean_squared_error(y_val, lgb_pred))
            lgb_scores.append(lgb_score)
            
            # XGBoost
            xgb_model = xgb.XGBRegressor(**self.xgb_params)
            xgb_model.set_params(early_stopping_rounds=50)
            xgb_model.fit(X_train, y_train,
                         eval_set=[(X_val, y_val)],
                         verbose=False)
            
            xgb_pred = xgb_model.predict(X_val)
            xgb_score = np.sqrt(mean_squared_error(y_val, xgb_pred))
            xgb_scores.append(xgb_score)
        
        # 训练最终模型
        self.logger.info("训练最终ensemble模型")
        
        # LightGBM最终模型
        lgb_final = lgb.Dataset(X_scaled, label=y)
        self.models[f'lgb_{target_col}'] = lgb.train(
            self.lgb_params,
            lgb_final,
            num_boost_round=1000
        )
        
        # XGBoost最终模型  
        self.models[f'xgb_{target_col}'] = xgb.XGBRegressor(**self.xgb_params)
        self.models[f'xgb_{target_col}'].fit(X_scaled, y)
        
        # 特征重要性
        lgb_importance = self.models[f'lgb_{target_col}'].feature_importance()
        xgb_importance = self.models[f'xgb_{target_col}'].feature_importances_
        
        self.feature_importance[target_col] = pd.DataFrame({
            'feature': feature_cols,
            'lgb_importance': lgb_importance,
            'xgb_importance': xgb_importance,
            'avg_importance': (lgb_importance + xgb_importance) / 2
        }).sort_values('avg_importance', ascending=False)
        
        # 性能统计
        avg_lgb_score = np.mean(lgb_scores)
        avg_xgb_score = np.mean(xgb_scores)
        
        self.logger.info(f"模型训练完成:")
        self.logger.info(f"  LightGBM RMSE: {avg_lgb_score:.4f}")
        self.logger.info(f"  XGBoost RMSE: {avg_xgb_score:.4f}")
        self.logger.info(f"  训练样本: {len(X_scaled)}")
        
        return True
    
    def predict_scores(self, features_df, target_col='target_1d'):
        """使用ensemble模型预测评分"""
        if f'lgb_{target_col}' not in self.models:
            self.logger.error(f"模型{target_col}未训练")
            return None
        
        feature_cols = ['bbi', 'volume_surge', 'price_momentum', 'zhixing_multiavg',
                       'rsi', 'market_cap', 'kdj_cross', 'pb', 'turnover_rate', 
                       'volatility_risk', 'relative_strength', 'pe_ttm',
                       'stock_price_log', 'price_category', 'price_trend_30d']
        
        X = features_df[feature_cols].copy()
        
        # 标准化
        X_scaled = self.scalers[target_col].transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
        
        # 双模型预测
        lgb_pred = self.models[f'lgb_{target_col}'].predict(X_scaled)
        xgb_pred = self.models[f'xgb_{target_col}'].predict(X_scaled)
        
        # Ensemble (权重可调)
        ensemble_pred = 0.6 * lgb_pred + 0.4 * xgb_pred
        
        # 转换为0-100评分
        scores = self._normalize_scores(ensemble_pred)
        
        return scores
    
    def _normalize_scores(self, predictions):
        """将预测值标准化为0-100评分 - 改进版本，避免极值聚集"""
        predictions = np.array(predictions)
        
        # 如果只有一个预测值，使用改进的映射逻辑
        if len(predictions) == 1:
            pred_val = predictions[0]
            
            # 使用更平滑的sigmoid映射，避免极值聚集
            # 预期预测值范围大约在 [-3, 3]，映射到 [5, 95]，避免极端0和100
            sigmoid_score = 1 / (1 + np.exp(-pred_val))  # sigmoid函数，输出0-1
            normalized_score = 5 + sigmoid_score * 90  # 映射到5-95，避免极值
            
            return np.array([np.clip(normalized_score, 5, 95)])
        
        # 多个预测值时使用改进的分位数标准化
        # 使用10%-90%分位数而不是5%-95%，减少极值影响
        p90 = np.percentile(predictions, 90)
        p10 = np.percentile(predictions, 10)
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # 防止分母为0
        if p90 == p10 or std_pred == 0:
            # 如果所有值相同或方差为0，使用基于均值的固定映射
            base_scores = []
            for pred in predictions:
                sigmoid_score = 1 / (1 + np.exp(-(pred - mean_pred)))
                normalized = 5 + sigmoid_score * 90
                base_scores.append(normalized)
            return np.array(base_scores)
        
        # 改进的标准化：结合Z-score和分位数映射
        normalized_scores = []
        for pred in predictions:
            # 计算相对于分位数的位置
            if pred <= p10:
                # 低于10%分位数，映射到5-25
                norm_score = 5 + (pred - p10) / (p10 - np.min(predictions) + 1e-8) * 20
                norm_score = max(5, norm_score)
            elif pred >= p90:
                # 高于90%分位数，映射到75-95
                norm_score = 75 + (pred - p90) / (np.max(predictions) - p90 + 1e-8) * 20
                norm_score = min(95, norm_score)
            else:
                # 中间值，映射到25-75
                norm_score = 25 + (pred - p10) / (p90 - p10) * 50
            
            normalized_scores.append(norm_score)
        
        normalized_scores = np.array(normalized_scores)
        return np.clip(normalized_scores, 5, 95)  # 最终限制在5-95范围
    
    def save_models(self, save_dir='models/v360'):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型
        for name, model in self.models.items():
            if 'lgb' in name:
                model.save_model(f'{save_dir}/{name}.txt')
            else:  # XGBoost
                with open(f'{save_dir}/{name}.pkl', 'wb') as f:
                    pickle.dump(model, f)
        
        # 保存scalers
        with open(f'{save_dir}/scalers.pkl', 'wb') as f:
            pickle.dump(self.scalers, f)
        
        # 保存特征重要性
        for target, importance in self.feature_importance.items():
            importance.to_csv(f'{save_dir}/feature_importance_{target}.csv', index=False)
        
        self.logger.info(f"模型保存完成: {save_dir}")
    
    def load_models(self, save_dir='models/v360'):
        """加载模型"""
        try:
            # 加载LightGBM模型
            for file in os.listdir(save_dir):
                if file.endswith('.txt') and 'lgb' in file:
                    model_name = file.replace('.txt', '')
                    self.models[model_name] = lgb.Booster(model_file=f'{save_dir}/{file}')
                elif file.endswith('.pkl') and 'xgb' in file:
                    model_name = file.replace('.pkl', '')
                    with open(f'{save_dir}/{file}', 'rb') as f:
                        self.models[model_name] = pickle.load(f)
            
            # 加载scalers
            with open(f'{save_dir}/scalers.pkl', 'rb') as f:
                self.scalers = pickle.load(f)
            
            # 加载特征重要性
            for file in os.listdir(save_dir):
                if file.startswith('feature_importance_') and file.endswith('.csv'):
                    target_name = file.replace('feature_importance_', '').replace('.csv', '')
                    self.feature_importance[target_name] = pd.read_csv(f'{save_dir}/{file}')
                    self.logger.info(f"加载特征重要性: {target_name}")
            
            self.logger.info(f"模型加载完成: {len(self.models)}个模型, {len(self.feature_importance)}个特征重要性")
            return True
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            return False

def main():
    """主函数 - V3.60系统测试"""
    print("🚀 V3.60 机器学习评分系统启动")
    print("="*50)
    
    # 初始化系统
    v360 = V360MLScoringSystem()
    
    # 测试参数 - 使用更多股票和更短时间窗口
    test_codes = ['000001', '000002', '000858', '002415', '300059', '000063', '000069', '000100', '000157', '000166']  
    start_date = '2025-01-01'  # 使用最近数据
    end_date = '2025-08-31'
    
    print(f"📊 测试配置:")
    print(f"  测试股票: {len(test_codes)}只")
    print(f"  数据期间: {start_date} ~ {end_date}")
    print(f"  模型版本: {v360.version}")
    
    # Step 1: 特征提取
    print("\n🔍 Step 1: 特征提取")
    features_df = v360.extract_features(test_codes, start_date, end_date)
    
    if features_df is None:
        print("❌ 特征提取失败")
        return
        
    print(f"✅ 特征提取完成: {len(features_df)}条记录")
    
    # Step 2: 目标变量计算
    print("\n📈 Step 2: 目标变量计算")
    training_data = v360.prepare_target(features_df)
    
    if training_data is None:
        print("❌ 目标变量计算失败")
        return
        
    print(f"✅ 训练数据准备完成: {len(training_data)}条记录")
    
    # Step 3: 模型训练
    print("\n🤖 Step 3: 模型训练")
    success = v360.train_models(training_data, target_col='target_1d')
    
    if not success:
        print("❌ 模型训练失败") 
        return
        
    print("✅ 模型训练完成")
    
    # Step 4: 评分预测
    print("\n🎯 Step 4: 评分预测")
    scores = v360.predict_scores(features_df, target_col='target_1d')
    
    if scores is None:
        print("❌ 评分预测失败")
        return
    
    # 结果展示
    results_df = features_df.copy()
    results_df['v360_score'] = scores
    
    print("✅ V3.60评分完成")
    print(f"📊 评分统计:")
    print(f"  平均分: {scores.mean():.2f}")
    print(f"  标准差: {scores.std():.2f}")
    print(f"  最高分: {scores.max():.2f}")
    print(f"  最低分: {scores.min():.2f}")
    
    # 保存结果
    output_file = f"reports/v360_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"📁 结果已保存: {output_file}")
    
    # 保存模型
    v360.save_models()
    
    print("\n🎉 V3.60系统测试完成!")

if __name__ == "__main__":
    main()