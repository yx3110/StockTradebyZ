#!/usr/bin/env python3
"""
全面评分参数优化器 - 优化所有评分函数的参数
使用贝叶斯优化和交叉验证来找到最优的评分函数参数

包含以下指标的参数优化：
1. RSI6 - 短期相对强弱指标
2. KDJ_K - KDJ随机指标K值
3. KDJ_D - KDJ随机指标D值
4. BBI - 牛熊指标
5. 知行趋势 - 双重EMA趋势线
6. 知行多均 - 多周期移动平均
7. PE_TTM - 市盈率
8. PB - 市净率
9. 市值 - 市场资本化
10. 价格动量 - 多周期价格变动
11. 成交量激增 - 量能突破确认
12. 波动性风险 - 风险调整评分
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

class ComprehensiveScoringOptimizer:
    """全面评分函数参数优化器"""
    
    def __init__(self, db_path: str = None, cv_folds: int = 3):
        self.db_path = db_path or os.path.join(project_root, 'data_adapter/stock_data.db')
        self.cv_folds = cv_folds
        
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # 全面参数搜索空间定义 - 包含所有指标参数
        self.param_space = {
            # 1. RSI参数 - 相对强弱指标
            'rsi_optimal_min': hp.uniform('rsi_optimal_min', 20, 50),
            'rsi_optimal_max': hp.uniform('rsi_optimal_max', 40, 70),
            'rsi_good_range': hp.uniform('rsi_good_range', 8, 25),
            
            # 2. KDJ_K参数 - 随机指标K值
            'kdj_k_optimal_min': hp.uniform('kdj_k_optimal_min', 20, 50),
            'kdj_k_optimal_max': hp.uniform('kdj_k_optimal_max', 40, 70),
            'kdj_k_good_range': hp.uniform('kdj_k_good_range', 5, 20),
            
            # 3. KDJ_D参数 - 随机指标D值
            'kdj_d_optimal_min': hp.uniform('kdj_d_optimal_min', 25, 55),
            'kdj_d_optimal_max': hp.uniform('kdj_d_optimal_max', 45, 75),
            'kdj_d_good_range': hp.uniform('kdj_d_good_range', 5, 20),
            
            # 4. BBI参数 - 牛熊指标
            'bbi_optimal_min': hp.uniform('bbi_optimal_min', 0.95, 1.05),
            'bbi_optimal_max': hp.uniform('bbi_optimal_max', 1.00, 1.15),
            'bbi_good_range': hp.uniform('bbi_good_range', 0.02, 0.15),
            
            # 5. 知行趋势参数 - 双重EMA趋势线
            'zhixing_trend_optimal_ratio_min': hp.uniform('zhixing_trend_optimal_ratio_min', 0.95, 1.02),
            'zhixing_trend_optimal_ratio_max': hp.uniform('zhixing_trend_optimal_ratio_max', 1.00, 1.08),
            'zhixing_trend_good_range': hp.uniform('zhixing_trend_good_range', 0.02, 0.15),
            
            # 6. 知行多均参数 - 多周期移动平均
            'zhixing_multiavg_optimal_ratio_min': hp.uniform('zhixing_multiavg_optimal_ratio_min', 0.90, 1.05),
            'zhixing_multiavg_optimal_ratio_max': hp.uniform('zhixing_multiavg_optimal_ratio_max', 1.00, 1.15),
            'zhixing_multiavg_good_range': hp.uniform('zhixing_multiavg_good_range', 0.05, 0.20),
            
            # 7. PE_TTM参数 - 市盈率
            'pe_optimal_min': hp.uniform('pe_optimal_min', 5, 20),
            'pe_optimal_max': hp.uniform('pe_optimal_max', 15, 35),
            'pe_good_range_low': hp.uniform('pe_good_range_low', 3, 10),  # 低于最优区间的容忍范围
            'pe_good_range_high': hp.uniform('pe_good_range_high', 10, 25), # 高于最优区间的容忍范围
            
            # 8. PB参数 - 市净率
            'pb_optimal_min': hp.uniform('pb_optimal_min', 0.8, 2.0),
            'pb_optimal_max': hp.uniform('pb_optimal_max', 1.5, 4.0),
            'pb_good_range_low': hp.uniform('pb_good_range_low', 0.2, 1.0),  # 低于最优区间的容忍范围
            'pb_good_range_high': hp.uniform('pb_good_range_high', 1.0, 3.0), # 高于最优区间的容忍范围
            
            # 9. 市值参数 - 市场资本化 (单位：亿元)
            'market_cap_optimal_min': hp.uniform('market_cap_optimal_min', 50, 200),   # 最优市值下限
            'market_cap_optimal_max': hp.uniform('market_cap_optimal_max', 200, 2000), # 最优市值上限
            'market_cap_small_cap_min': hp.uniform('market_cap_small_cap_min', 10, 80),   # 小盘股下限
            'market_cap_large_cap_max': hp.uniform('market_cap_large_cap_max', 1000, 8000), # 大盘股上限
            
            # 10. 价格动量参数 - 多周期价格变动
            'momentum_excellent_threshold': hp.uniform('momentum_excellent_threshold', 6, 12),  # 优秀动量阈值
            'momentum_good_threshold': hp.uniform('momentum_good_threshold', 2, 8),          # 良好动量阈值
            'momentum_negative_threshold': hp.uniform('momentum_negative_threshold', -6, -2), # 负动量阈值
            'momentum_weight_1d': hp.uniform('momentum_weight_1d', 0.2, 0.6),  # 1日权重
            'momentum_weight_5d': hp.uniform('momentum_weight_5d', 0.1, 0.4),  # 5日权重
            'momentum_weight_10d': hp.uniform('momentum_weight_10d', 0.1, 0.3), # 10日权重
            
            # 11. 成交量激增参数 - 量能突破确认
            'volume_surge_optimal_min': hp.uniform('volume_surge_optimal_min', 1.1, 1.5),  # 最优成交量倍数下限
            'volume_surge_optimal_max': hp.uniform('volume_surge_optimal_max', 1.8, 3.0),  # 最优成交量倍数上限
            'volume_surge_excellent_max': hp.uniform('volume_surge_excellent_max', 2.5, 4.0), # 优异成交量上限
            'volume_weight_5d': hp.uniform('volume_weight_5d', 0.4, 0.8),      # 5日平均权重
            'volume_weight_20d': hp.uniform('volume_weight_20d', 0.2, 0.6),    # 20日平均权重
        }
        
        # 数据缓存
        self.data_cache = None
        
    def load_historical_data(self, start_date: str = "2024-01-01", end_date: str = "2025-09-01", limit: int = 1000) -> pd.DataFrame:
        """加载历史数据用于优化"""
        self.logger.info(f"加载历史数据: {start_date} 到 {end_date}, 限制 {limit} 只股票")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 1. 获取活跃股票列表
            active_stocks_query = f"""
            SELECT DISTINCT s.code, s.name, s.type
            FROM securities s
            INNER JOIN daily_quotes dq ON s.code = dq.security_id
            WHERE s.type = 'A股'
              AND dq.trade_date BETWEEN '{start_date}' AND '{end_date}'
              AND dq.volume > 0
            ORDER BY s.code
            LIMIT {limit}
            """
            active_stocks = pd.read_sql(active_stocks_query, conn)
            self.logger.info(f"找到 {len(active_stocks)} 只活跃股票")
            
            # 2. 加载这些股票的完整数据
            stock_codes = "', '".join(active_stocks['code'].tolist())
            
            main_query = f"""
            SELECT 
                dq.security_id as stock_code,
                dq.trade_date,
                dq.close,
                dq.open,
                dq.high,
                dq.low,
                dq.volume,
                dq.price_change_pct as pct_chg,
                
                -- 技术指标
                ti.rsi6,
                ti.kdj_k,
                ti.kdj_d,
                ti.bbi,
                
                -- 基本面数据
                db.pe_ttm,
                db.pb,
                db.total_mv as market_cap,
                db.turnover_rate
                
            FROM daily_quotes dq
            LEFT JOIN technical_indicators ti ON dq.security_id = ti.security_id AND dq.trade_date = ti.trade_date
            LEFT JOIN daily_basic db ON dq.security_id = db.security_id AND dq.trade_date = db.trade_date
            WHERE dq.security_id IN ('{stock_codes}')
              AND dq.trade_date BETWEEN '{start_date}' AND '{end_date}'
              AND dq.volume > 0
              AND dq.close > 0
            ORDER BY dq.security_id, dq.trade_date
            """
            
            df = pd.read_sql(main_query, conn)
            conn.close()
            
            self.logger.info(f"加载完成，共 {len(df)} 条记录")
            
            # 3. 数据预处理
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            
            # 计算未来收益率（用于评估预测效果）
            df = df.sort_values(['stock_code', 'trade_date'])
            df['return_1d'] = df.groupby('stock_code')['close'].pct_change().shift(-1)  # 下一日收益率
            df['return_3d'] = df.groupby('stock_code')['close'].pct_change(periods=3).shift(-3)
            df['return_5d'] = df.groupby('stock_code')['close'].pct_change(periods=5).shift(-5)
            
            # 计算知行指标
            df = self._calculate_zhixing_indicators(df)
            
            # 计算动量指标
            df = self._calculate_momentum_indicators(df)
            
            # 计算成交量指标
            df = self._calculate_volume_indicators(df)
            
            # 去除缺失值
            df = df.dropna()
            
            self.data_cache = df
            return df
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_zhixing_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算知行指标"""
        def calculate_zhixing_trend(group):
            """计算知行趋势线: EMA(EMA(C,10),10)"""
            ema10_1 = group['close'].ewm(span=10, adjust=False).mean()
            ema10_2 = ema10_1.ewm(span=10, adjust=False).mean()
            return ema10_2
        
        def calculate_zhixing_multiavg(group):
            """计算知行多空线: (MA5+MA10+MA20+MA60)/4"""
            ma5 = group['close'].rolling(window=5, min_periods=1).mean()
            ma10 = group['close'].rolling(window=10, min_periods=1).mean()
            ma20 = group['close'].rolling(window=20, min_periods=1).mean()
            ma60 = group['close'].rolling(window=60, min_periods=1).mean()
            return (ma5 + ma10 + ma20 + ma60) / 4
        
        df['zhixing_trend'] = df.groupby('stock_code').apply(calculate_zhixing_trend).reset_index(0, drop=True)
        df['zhixing_multiavg'] = df.groupby('stock_code').apply(calculate_zhixing_multiavg).reset_index(0, drop=True)
        
        return df
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算动量指标"""
        def calculate_momentum(group):
            pct_chg_1d = group['pct_chg']
            pct_chg_5d = group['close'].pct_change(periods=5) * 100
            pct_chg_10d = group['close'].pct_change(periods=10) * 100
            pct_chg_20d = group['close'].pct_change(periods=20) * 100
            
            return pd.DataFrame({
                'pct_chg_5d': pct_chg_5d,
                'pct_chg_10d': pct_chg_10d,
                'pct_chg_20d': pct_chg_20d
            })
        
        momentum_df = df.groupby('stock_code').apply(calculate_momentum).reset_index(0, drop=True)
        df = pd.concat([df, momentum_df], axis=1)
        
        return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算成交量指标"""
        def calculate_volume_avg(group):
            avg_volume_5 = group['volume'].rolling(window=5, min_periods=1).mean()
            avg_volume_20 = group['volume'].rolling(window=20, min_periods=1).mean()
            return pd.DataFrame({
                'avg_volume_5': avg_volume_5,
                'avg_volume_20': avg_volume_20
            })
        
        volume_df = df.groupby('stock_code').apply(calculate_volume_avg).reset_index(0, drop=True)
        df = pd.concat([df, volume_df], axis=1)
        
        return df
    
    def calculate_comprehensive_score_with_params(self, data: pd.Series, params: Dict) -> float:
        """使用给定参数计算综合评分"""
        scores = {}
        weights = {
            'volatility_risk': 0.1532, 'market_cap': 0.1420, 'price_momentum': 0.1310,
            'pb': 0.1213, 'pe_ttm': 0.0875, 'rsi6': 0.0839, 'kdj_k': 0.0661,
            'bbi': 0.0566, 'kdj_d': 0.0529, 'zhixing_trend': 0.0478,
            'volume_surge': 0.0315, 'zhixing_multiavg': 0.0262
        }
        
        try:
            # 1. RSI评分
            rsi6 = data.get('rsi6', 50)
            if pd.notna(rsi6):
                rsi_min, rsi_max = params['rsi_optimal_min'], params['rsi_optimal_max']
                if rsi_min <= rsi6 <= rsi_max:
                    scores['rsi6'] = 100.0
                else:
                    distance = min(abs(rsi6 - rsi_min), abs(rsi6 - rsi_max))
                    penalty = distance / params['rsi_good_range'] * 50
                    scores['rsi6'] = max(30.0, 100.0 - penalty)
            else:
                scores['rsi6'] = 50.0
            
            # 2. KDJ_K评分
            kdj_k = data.get('kdj_k', 50)
            if pd.notna(kdj_k):
                kdj_k_min, kdj_k_max = params['kdj_k_optimal_min'], params['kdj_k_optimal_max']
                if kdj_k_min <= kdj_k <= kdj_k_max:
                    scores['kdj_k'] = 100.0
                else:
                    distance = min(abs(kdj_k - kdj_k_min), abs(kdj_k - kdj_k_max))
                    penalty = distance / params['kdj_k_good_range'] * 50
                    scores['kdj_k'] = max(25.0, 100.0 - penalty)
            else:
                scores['kdj_k'] = 50.0
            
            # 3. KDJ_D评分
            kdj_d = data.get('kdj_d', 50)
            if pd.notna(kdj_d):
                kdj_d_min, kdj_d_max = params['kdj_d_optimal_min'], params['kdj_d_optimal_max']
                if kdj_d_min <= kdj_d <= kdj_d_max:
                    scores['kdj_d'] = 100.0
                else:
                    distance = min(abs(kdj_d - kdj_d_min), abs(kdj_d - kdj_d_max))
                    penalty = distance / params['kdj_d_good_range'] * 50
                    scores['kdj_d'] = max(25.0, 100.0 - penalty)
            else:
                scores['kdj_d'] = 50.0
            
            # 4. BBI评分
            bbi = data.get('bbi', 0)
            close = data.get('close', 0)
            if bbi > 0 and close > 0:
                price_to_bbi = close / bbi
                bbi_min, bbi_max = params['bbi_optimal_min'], params['bbi_optimal_max']
                if bbi_min <= price_to_bbi <= bbi_max:
                    scores['bbi'] = 100.0
                else:
                    distance = min(abs(price_to_bbi - bbi_min), abs(price_to_bbi - bbi_max))
                    penalty = distance / params['bbi_good_range'] * 50
                    scores['bbi'] = max(40.0, 100.0 - penalty)
            else:
                scores['bbi'] = 50.0
            
            # 5. 知行趋势评分
            zhixing_trend = data.get('zhixing_trend', None)
            if pd.notna(zhixing_trend) and zhixing_trend > 0 and close > 0:
                trend_ratio = close / zhixing_trend
                trend_min = params['zhixing_trend_optimal_ratio_min']
                trend_max = params['zhixing_trend_optimal_ratio_max']
                if trend_min <= trend_ratio <= trend_max:
                    scores['zhixing_trend'] = 100.0
                else:
                    distance = min(abs(trend_ratio - trend_min), abs(trend_ratio - trend_max))
                    penalty = distance / params['zhixing_trend_good_range'] * 100
                    scores['zhixing_trend'] = max(30.0, 100.0 - penalty)
            else:
                scores['zhixing_trend'] = 70.0
            
            # 6. 知行多均评分
            zhixing_multiavg = data.get('zhixing_multiavg', None)
            if pd.notna(zhixing_multiavg) and zhixing_multiavg > 0 and close > 0:
                multiavg_ratio = close / zhixing_multiavg
                multiavg_min = params['zhixing_multiavg_optimal_ratio_min']
                multiavg_max = params['zhixing_multiavg_optimal_ratio_max']
                if multiavg_min <= multiavg_ratio <= multiavg_max:
                    scores['zhixing_multiavg'] = 100.0
                else:
                    distance = min(abs(multiavg_ratio - multiavg_min), abs(multiavg_ratio - multiavg_max))
                    penalty = distance / params['zhixing_multiavg_good_range'] * 80
                    scores['zhixing_multiavg'] = max(25.0, 100.0 - penalty)
            else:
                scores['zhixing_multiavg'] = 70.0
            
            # 7. PE评分
            pe = data.get('pe_ttm', 0)
            if pe > 0:
                pe_min, pe_max = params['pe_optimal_min'], params['pe_optimal_max']
                if pe_min <= pe <= pe_max:
                    scores['pe_ttm'] = 100.0
                elif pe < pe_min:
                    if pe >= pe_min - params['pe_good_range_low']:
                        scores['pe_ttm'] = 60.0 + (pe - (pe_min - params['pe_good_range_low'])) / params['pe_good_range_low'] * 40
                    else:
                        scores['pe_ttm'] = max(30.0, pe / pe_min * 60)
                else:  # pe > pe_max
                    if pe <= pe_max + params['pe_good_range_high']:
                        scores['pe_ttm'] = 100.0 - (pe - pe_max) / params['pe_good_range_high'] * 30
                    else:
                        scores['pe_ttm'] = max(30.0, 70.0 - (pe - pe_max) / params['pe_good_range_high'] * 40)
            else:
                scores['pe_ttm'] = 50.0
                
            # 8. PB评分
            pb = data.get('pb', 0)
            if pb > 0:
                pb_min, pb_max = params['pb_optimal_min'], params['pb_optimal_max']
                if pb_min <= pb <= pb_max:
                    scores['pb'] = 100.0
                elif pb < pb_min:
                    if pb >= pb_min - params['pb_good_range_low']:
                        scores['pb'] = 60.0 + (pb - (pb_min - params['pb_good_range_low'])) / params['pb_good_range_low'] * 40
                    else:
                        scores['pb'] = max(20.0, pb / pb_min * 60)
                else:  # pb > pb_max
                    if pb <= pb_max + params['pb_good_range_high']:
                        scores['pb'] = 100.0 - (pb - pb_max) / params['pb_good_range_high'] * 40
                    else:
                        scores['pb'] = max(20.0, 60.0 - (pb - pb_max) / params['pb_good_range_high'] * 40)
            else:
                scores['pb'] = 50.0
            
            # 9. 市值评分
            market_cap = data.get('market_cap', 0)
            if market_cap > 0:
                market_cap_yi = market_cap / 10000  # 转换为亿元
                cap_min, cap_max = params['market_cap_optimal_min'], params['market_cap_optimal_max']
                if cap_min <= market_cap_yi <= cap_max:
                    scores['market_cap'] = 100.0
                elif params['market_cap_small_cap_min'] <= market_cap_yi < cap_min:
                    scores['market_cap'] = 80.0 + (market_cap_yi - params['market_cap_small_cap_min']) / (cap_min - params['market_cap_small_cap_min']) * 20
                elif cap_max < market_cap_yi <= params['market_cap_large_cap_max']:
                    scores['market_cap'] = 100.0 - (market_cap_yi - cap_max) / (params['market_cap_large_cap_max'] - cap_max) * 30
                elif market_cap_yi < params['market_cap_small_cap_min']:
                    scores['market_cap'] = max(20.0, market_cap_yi / params['market_cap_small_cap_min'] * 80)
                else:
                    scores['market_cap'] = 70.0
            else:
                scores['market_cap'] = 50.0
            
            # 10. 价格动量评分
            pct_chg_1d = data.get('pct_chg', 0)
            pct_chg_5d = data.get('pct_chg_5d', 0)
            pct_chg_10d = data.get('pct_chg_10d', 0)
            pct_chg_20d = data.get('pct_chg_20d', 0)
            
            # 动态权重
            w1, w5, w10 = params['momentum_weight_1d'], params['momentum_weight_5d'], params['momentum_weight_10d']
            w20 = 1.0 - w1 - w5 - w10  # 剩余权重给20日
            w20 = max(0.05, w20)  # 至少5%权重
            
            momentum = (pct_chg_1d * w1 + pct_chg_5d * w5 + pct_chg_10d * w10 + pct_chg_20d * w20)
            
            if momentum > params['momentum_excellent_threshold']:
                scores['price_momentum'] = 100.0
            elif momentum > params['momentum_good_threshold']:
                scores['price_momentum'] = 80.0 + (momentum - params['momentum_good_threshold']) / (params['momentum_excellent_threshold'] - params['momentum_good_threshold']) * 20
            elif momentum > 0:
                scores['price_momentum'] = 60.0 + momentum / params['momentum_good_threshold'] * 20
            elif momentum > params['momentum_negative_threshold']:
                scores['price_momentum'] = 40.0 + (momentum - params['momentum_negative_threshold']) / (-params['momentum_negative_threshold']) * 20
            else:
                scores['price_momentum'] = max(0.0, 40.0 + momentum / params['momentum_negative_threshold'] * 40)
            
            # 11. 成交量激增评分
            volume = data.get('volume', 0)
            avg_volume_5 = data.get('avg_volume_5', volume)
            avg_volume_20 = data.get('avg_volume_20', volume)
            
            if avg_volume_20 > 0:
                volume_ratio_5 = volume / avg_volume_5 if avg_volume_5 > 0 else 1.0
                volume_ratio_20 = volume / avg_volume_20
                surge_score = volume_ratio_5 * params['volume_weight_5d'] + volume_ratio_20 * params['volume_weight_20d']
                
                surge_min, surge_max = params['volume_surge_optimal_min'], params['volume_surge_optimal_max']
                if surge_min <= surge_score <= surge_max:
                    scores['volume_surge'] = 100.0
                elif 1.0 <= surge_score < surge_min:
                    scores['volume_surge'] = 70.0 + (surge_score - 1.0) / (surge_min - 1.0) * 30
                elif surge_max < surge_score <= params['volume_surge_excellent_max']:
                    scores['volume_surge'] = 100.0 - (surge_score - surge_max) / (params['volume_surge_excellent_max'] - surge_max) * 20
                elif surge_score < 1.0:
                    scores['volume_surge'] = max(40.0, surge_score * 70)
                else:
                    scores['volume_surge'] = max(50.0, 80.0 - (surge_score - params['volume_surge_excellent_max']) * 15)
            else:
                scores['volume_surge'] = 50.0
            
            # 12. 波动性风险评分 (简化版)
            high = data.get('high', data.get('close', 0))
            low = data.get('low', data.get('close', 0))
            close = data.get('close', 0)
            
            if close > 0:
                daily_volatility = (high - low) / close
                volatility_score = max(0, 1 - daily_volatility * 2) * 100
                scores['volatility_risk'] = volatility_score
            else:
                scores['volatility_risk'] = 50.0
            
            # 计算加权总分
            final_score = sum(scores[key] * weights[key] for key in scores if key in weights)
            return final_score
            
        except Exception as e:
            return 50.0  # 出错时返回中性评分
    
    def evaluate_parameters(self, params: Dict) -> float:
        """评估参数组合的效果"""
        if self.data_cache is None:
            self.logger.error("数据未加载，请先运行load_historical_data")
            return 0.0
        
        try:
            # 确保参数范围合理
            if not self._validate_parameters(params):
                return 0.0
            
            # 计算每只股票每天的评分
            scores = []
            returns = []
            
            for idx, row in self.data_cache.iterrows():
                if pd.isna(row['return_1d']):  # 跳过没有未来收益的记录
                    continue
                    
                score = self.calculate_comprehensive_score_with_params(row, params)
                scores.append(score)
                returns.append(row['return_1d'])
            
            if len(scores) < 100:  # 数据点太少
                return 0.0
            
            # 计算信息系数 (IC) - 评分与未来收益的相关性
            ic, p_value = spearmanr(scores, returns)
            
            if pd.isna(ic):
                return 0.0
            
            # 返回负IC，因为hyperopt要最小化目标函数
            self.logger.info(f"参数评估完成 - 平均IC: {ic:.4f}")
            return -ic
            
        except Exception as e:
            self.logger.error(f"参数评估失败: {str(e)}")
            return 0.0
    
    def _validate_parameters(self, params: Dict) -> bool:
        """验证参数的合理性"""
        try:
            # 确保最小值小于最大值
            ranges_to_check = [
                ('rsi_optimal_min', 'rsi_optimal_max'),
                ('kdj_k_optimal_min', 'kdj_k_optimal_max'),
                ('kdj_d_optimal_min', 'kdj_d_optimal_max'),
                ('bbi_optimal_min', 'bbi_optimal_max'),
                ('zhixing_trend_optimal_ratio_min', 'zhixing_trend_optimal_ratio_max'),
                ('zhixing_multiavg_optimal_ratio_min', 'zhixing_multiavg_optimal_ratio_max'),
                ('pe_optimal_min', 'pe_optimal_max'),
                ('pb_optimal_min', 'pb_optimal_max'),
                ('market_cap_optimal_min', 'market_cap_optimal_max'),
            ]
            
            for min_key, max_key in ranges_to_check:
                if params[min_key] >= params[max_key]:
                    return False
            
            # 确保权重总和不超过1
            total_weight = (params['momentum_weight_1d'] + 
                          params['momentum_weight_5d'] + 
                          params['momentum_weight_10d'])
            if total_weight >= 0.95:  # 给20日权重留些空间
                return False
            
            return True
            
        except:
            return False
    
    def optimize(self, max_evals: int = 50) -> Dict:
        """运行参数优化"""
        self.logger.info("=" * 60)
        self.logger.info("开始全面评分参数优化")
        self.logger.info(f"优化期间: {self.data_cache['trade_date'].min()} 到 {self.data_cache['trade_date'].max()}")
        self.logger.info(f"样本大小: {len(self.data_cache)}")
        self.logger.info(f"最大评估次数: {max_evals}")
        self.logger.info("=" * 60)
        
        # 运行贝叶斯优化
        trials = Trials()
        best = fmin(fn=self.evaluate_parameters,
                   space=self.param_space,
                   algo=tpe.suggest,
                   max_evals=max_evals,
                   trials=trials,
                   verbose=True)
        
        best_ic = -trials.best_trial['result']['loss']
        self.logger.info(f"优化完成！最佳IC: {best_ic:.4f}")
        
        # 保存结果
        optimization_results = {
            'optimization_date': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'data_period': {
                'start': str(self.data_cache['trade_date'].min().date()),
                'end': str(self.data_cache['trade_date'].max().date())
            },
            'sample_size': len(self.data_cache),
            'best_parameters': best,
            'optimization_results': {
                'best_params': best,
                'best_ic': best_ic,
                'trials_count': len(trials.trials),
                'optimization_history': [(trial['result']['loss'], trial['misc']['vals']) 
                                       for trial in trials.trials]
            },
            'data_stats': {
                'total_records': len(self.data_cache),
                'unique_stocks': self.data_cache['stock_code'].nunique(),
                'date_range': {
                    'start': str(self.data_cache['trade_date'].min().date()),
                    'end': str(self.data_cache['trade_date'].max().date())
                }
            }
        }
        
        # 保存到文件
        output_file = f"qlib_integration/comprehensive_scoring_optimization_{optimization_results['optimization_date']}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(optimization_results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"结果已保存到: {output_file}")
        
        print("=" * 60)
        print("优化完成！")
        print(f"最佳IC: {best_ic:.4f}")
        print("最佳参数:")
        for key, value in best.items():
            print(f"  {key}: {value:.4f}")
        print("=" * 60)
        
        return optimization_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='全面评分参数优化器')
    parser.add_argument('--n-trials', type=int, default=50, help='优化试验次数')
    parser.add_argument('--sample-size', type=int, default=1000, help='样本股票数量')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='数据开始日期')
    parser.add_argument('--end-date', type=str, default='2025-09-01', help='数据结束日期')
    
    args = parser.parse_args()
    
    # 创建优化器
    optimizer = ComprehensiveScoringOptimizer()
    
    # 加载数据
    optimizer.load_historical_data(
        start_date=args.start_date,
        end_date=args.end_date,
        limit=args.sample_size
    )
    
    # 运行优化
    results = optimizer.optimize(max_evals=args.n_trials)