#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V3.53 多时间周期IC优化量化评分系统

革命性创新: 多时间周期分层权重架构
- 针对1日、3日、5日、10日、15日分别优化权重配置
- 每个时间周期使用最适合的因子组合
- 联合优化多目标IC函数，平衡短中长期预测能力
- 基于A股市场特点的时间周期重要性权重

核心突破:
1. 分层权重架构：不同时间周期使用不同因子权重
2. 多目标IC优化：联合优化5个时间周期的IC表现
3. 因子时间特征：基于因子在不同周期的有效性分配权重
4. A股市场适配：考虑A股高噪音、高波动特点

目标IC提升:
- 1日IC: 1.57% → 2.5%+ (提升60%)
- 3日IC: -1.13% → 1.5%+ (从负转正)
- 5日IC: -4.11% → 1.0%+ (大幅改善)
- 10日IC: -6.50% → 0.5%+ (至少转正)
- 15日IC: 新增 → 0.3%+ (建立基线)
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
import warnings
import json
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

class QuantitativeScorerV353MultiPeriod:
    """
    V3.53 多时间周期IC优化量化评分器
    
    革命性多时间周期分层权重架构:
    - 5个时间周期 (1d/3d/5d/10d/15d) 专门优化
    - 12个评分因子在不同周期的最优组合
    - 基于A股市场特点的权重分配策略
    """
    
    def __init__(self, db_path: str = "data_adapter/stock_data.db"):
        """
        初始化多时间周期评分器
        
        Args:
            db_path: SQLite数据库路径
        """
        self.db_path = db_path
        self.logger = self._setup_logger()
        
        # 多时间周期权重配置 - 核心创新
        self.period_weights = {
            # 1日预测：技术指标主导 (短线交易)
            '1d': {
                'rsi6': 0.15,           # RSI短期超买超卖
                'kdj_k': 0.12,          # KDJ短期动量
                'kdj_d': 0.08,          # KDJ平滑信号
                'volume_surge': 0.18,   # 量能突破确认
                'price_momentum': 0.20, # 价格动量最重要
                'volatility_risk': 0.15,# 波动性风险控制
                'bbi': 0.06,            # 趋势确认
                'zhixing_trend': 0.04,  # 短期趋势
                'pb': 0.01,             # 基本面权重极低
                'pe_ttm': 0.01,
                'market_cap': 0.0,      # 短期无关
                'zhixing_multiavg': 0.0
            },
            
            # 3日预测：动量+趋势结合 (短周期波段)
            '3d': {
                'price_momentum': 0.25, # 动量效应延续
                'zhixing_trend': 0.18,  # 趋势开始重要
                'bbi': 0.15,            # 牛熊指标有效
                'volume_surge': 0.12,   # 量能确认
                'rsi6': 0.10,           # RSI作用减弱
                'kdj_k': 0.08,          # KDJ作用减弱
                'volatility_risk': 0.12,# 风险控制重要
                'market_cap': 0.0,      # 仍无市值效应
                'kdj_d': 0.0,
                'pb': 0.0,
                'pe_ttm': 0.0,
                'zhixing_multiavg': 0.0
            },
            
            # 5日预测：趋势+基本面 (周度表现)
            '5d': {
                'zhixing_trend': 0.22,      # 趋势最重要
                'zhixing_multiavg': 0.18,   # 多均线趋势
                'bbi': 0.18,                # 牛熊指标强化
                'price_momentum': 0.15,     # 动量仍有效
                'pb': 0.12,                 # 估值开始起作用
                'pe_ttm': 0.08,            # PE作用较小
                'volatility_risk': 0.07,    # 风险权重降低
                'market_cap': 0.0,          # 市值效应微弱
                'rsi6': 0.0,               # 技术指标失效
                'kdj_k': 0.0,
                'kdj_d': 0.0,
                'volume_surge': 0.0
            },
            
            # 10日预测：基本面主导 (双周表现) 
            '10d': {
                'pb': 0.20,                 # 市净率最重要
                'pe_ttm': 0.18,            # 市盈率重要
                'market_cap': 0.20,         # 市值效应显现
                'zhixing_trend': 0.15,      # 长期趋势
                'zhixing_multiavg': 0.12,   # 多均线支撑
                'price_momentum': 0.10,     # 动量减弱
                'volatility_risk': 0.05,    # 风险权重最低
                'bbi': 0.0,                # 技术指标完全失效
                'rsi6': 0.0,
                'kdj_k': 0.0,
                'kdj_d': 0.0,
                'volume_surge': 0.0
            },
            
            # 15日预测：价值+长期趋势 (月度趋势)
            '15d': {
                'pb': 0.25,                 # 价值投资主导
                'pe_ttm': 0.20,            # 估值核心
                'market_cap': 0.18,         # 市值效应强化
                'zhixing_multiavg': 0.15,   # 长期趋势确认
                'zhixing_trend': 0.12,      # 趋势延续
                'volatility_risk': 0.10,    # 适度风险控制
                'price_momentum': 0.0,      # 动量完全失效
                'bbi': 0.0,                # 所有技术指标失效
                'rsi6': 0.0,
                'kdj_k': 0.0,
                'kdj_d': 0.0,
                'volume_surge': 0.0
            }
        }
        
        # 时间周期重要性权重 (基于A股特点)
        self.period_importance = {
            '1d': 0.35,    # 35% - 短期交易最重要
            '3d': 0.25,    # 25% - T+1后续表现重要
            '5d': 0.20,    # 20% - 周度表现有意义
            '10d': 0.15,   # 15% - 双周表现参考
            '15d': 0.05    # 5% - 月度趋势指导
        }
        
        # 继承v3.52的优化参数
        self.optimized_params = {
            # RSI参数
            'rsi_optimal_min': 23.39,
            'rsi_optimal_max': 43.08,
            'rsi_good_range': 19.14,
            
            # KDJ_K参数
            'kdj_k_optimal_min': 21.59,
            'kdj_k_optimal_max': 58.39,
            'kdj_k_good_range': 17.07,
            
            # KDJ_D参数
            'kdj_d_optimal_min': 36.93,
            'kdj_d_optimal_max': 49.11,
            'kdj_d_good_range': 10.89,
            
            # BBI参数
            'bbi_optimal_min': 0.9556,
            'bbi_optimal_max': 1.0522,
            'bbi_good_range': 0.1169,
            
            # 知行趋势参数
            'zhixing_trend_optimal_ratio_min': 0.9570,
            'zhixing_trend_optimal_ratio_max': 1.0407,
            'zhixing_trend_good_range': 0.1185,
            
            # 知行多均参数
            'zhixing_multiavg_optimal_ratio_min': 0.9747,
            'zhixing_multiavg_optimal_ratio_max': 1.0866,
            'zhixing_multiavg_good_range': 0.1481,
            
            # PE参数
            'pe_optimal_min': 15.57,
            'pe_optimal_max': 34.94,
            'pe_good_range_low': 7.88,
            'pe_good_range_high': 10.02,
            
            # PB参数
            'pb_optimal_min': 0.98,
            'pb_optimal_max': 3.57,
            'pb_good_range_low': 0.48,
            'pb_good_range_high': 1.04,
            
            # 市值参数 (单位：亿元)
            'market_cap_optimal_min': 131.82,
            'market_cap_optimal_max': 1760.58,
            'market_cap_small_cap_min': 37.69,
            'market_cap_large_cap_max': 7371.85,
            
            # 价格动量参数
            'momentum_excellent_threshold': 9.85,
            'momentum_good_threshold': 2.26,
            'momentum_negative_threshold': -3.42,
            'momentum_weight_1d': 0.295,
            'momentum_weight_5d': 0.316,
            'momentum_weight_10d': 0.138,
            'momentum_weight_20d': 0.251,
            
            # 成交量激增参数
            'volume_surge_optimal_min': 1.30,
            'volume_surge_optimal_max': 2.87,
            'volume_surge_excellent_max': 3.23,
            'volume_weight_5d': 0.786,
            'volume_weight_20d': 0.240
        }
        
        # 验证权重配置
        for period, weights in self.period_weights.items():
            total = sum(weights.values())
            if not (0.995 <= total <= 1.005):
                raise ValueError(f"{period}权重总和 {total:.4f}, 应约等于1.0")
        
        # 验证重要性权重
        total_importance = sum(self.period_importance.values())
        if not (0.995 <= total_importance <= 1.005):
            raise ValueError(f"时间周期重要性权重总和 {total_importance:.4f}, 应约等于1.0")
            
        self.logger.info("🚀 V3.53 多时间周期IC优化评分器初始化完成")
        self.logger.info(f"📊 覆盖{len(self.period_weights)}个时间周期，12个优化因子")
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志配置"""
        logger = logging.getLogger(f"{__name__}_v353_multiperiod")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def calculate_multi_period_score(self, stock_data: Dict, date: str, 
                                   target_period: str = 'composite') -> Tuple[float, Dict]:
        """
        计算多时间周期优化评分
        
        Args:
            stock_data: 股票指标数据字典
            date: 交易日期
            target_period: 目标时间周期 ('1d', '3d', '5d', '10d', '15d', 'composite')
            
        Returns:
            Tuple[最终评分, 详细分解]
        """
        try:
            if target_period == 'composite':
                return self._calculate_composite_score(stock_data, date)
            elif target_period in self.period_weights:
                return self._calculate_period_specific_score(stock_data, date, target_period)
            else:
                raise ValueError(f"不支持的时间周期: {target_period}")
                
        except Exception as e:
            self.logger.error(f"❌ 多时间周期评分计算失败: {e}")
            return 0.0, {}
    
    def _calculate_composite_score(self, stock_data: Dict, date: str) -> Tuple[float, Dict]:
        """计算加权复合评分"""
        period_scores = {}
        period_details = {}
        
        # 计算各时间周期评分
        for period in self.period_weights.keys():
            score, details = self._calculate_period_specific_score(stock_data, date, period)
            period_scores[period] = score
            period_details[period] = details
        
        # 加权复合评分
        composite_score = sum(
            period_scores[period] * self.period_importance[period]
            for period in self.period_weights.keys()
        )
        
        # 构建详细分解
        composite_details = {
            'composite_score': composite_score,
            'period_scores': period_scores,
            'period_weights': self.period_importance.copy(),
            'period_details': period_details,
            'scoring_method': 'V3.53_MultiPeriod_Composite'
        }
        
        return composite_score, composite_details
    
    def _calculate_period_specific_score(self, stock_data: Dict, date: str, 
                                       period: str) -> Tuple[float, Dict]:
        """计算特定时间周期的评分"""
        try:
            factor_scores = {}
            factor_contributions = {}
            weights = self.period_weights[period]
            
            # 计算各因子评分 (复用v3.52的优化算法)
            factors_to_calculate = [factor for factor, weight in weights.items() if weight > 0]
            
            for factor in factors_to_calculate:
                if factor == 'rsi6':
                    score = self._calculate_rsi_score_optimized(stock_data)
                elif factor == 'kdj_k':
                    score = self._calculate_kdj_k_score_optimized(stock_data)
                elif factor == 'kdj_d':
                    score = self._calculate_kdj_d_score_optimized(stock_data)
                elif factor == 'bbi':
                    score = self._calculate_bbi_score_optimized(stock_data)
                elif factor == 'zhixing_trend':
                    score = self._calculate_zhixing_trend_score_optimized(stock_data)
                elif factor == 'zhixing_multiavg':
                    score = self._calculate_zhixing_multiavg_score_optimized(stock_data)
                elif factor == 'pe_ttm':
                    score = self._calculate_pe_score_optimized(stock_data)
                elif factor == 'pb':
                    score = self._calculate_pb_score_optimized(stock_data)
                elif factor == 'market_cap':
                    score = self._calculate_market_cap_score_optimized(stock_data)
                elif factor == 'price_momentum':
                    score = self._calculate_price_momentum_score_optimized(stock_data)
                elif factor == 'volume_surge':
                    score = self._calculate_volume_surge_score_optimized(stock_data)
                elif factor == 'volatility_risk':
                    score = self._calculate_volatility_risk_score(stock_data)
                else:
                    score = 0.0
                
                factor_scores[factor] = score
                factor_contributions[factor] = score * weights[factor]
            
            # 计算期间总分
            period_score = sum(factor_contributions.values())
            
            # 构建详细信息
            period_details = {
                'period': period,
                'period_score': period_score,
                'factor_scores': factor_scores,
                'factor_contributions': factor_contributions,
                'weights_used': {k: v for k, v in weights.items() if v > 0},
                'scoring_method': f'V3.53_MultiPeriod_{period}'
            }
            
            return period_score, period_details
            
        except Exception as e:
            self.logger.error(f"❌ {period}周期评分计算失败: {e}")
            return 0.0, {}

    # 继承v3.52的所有优化评分算法
    def _calculate_rsi_score_optimized(self, data: Dict) -> float:
        """RSI6优化评分 (继承v3.52)"""
        try:
            rsi = data.get('rsi6', 50.0)
            if rsi is None or pd.isna(rsi):
                return 0.5
            
            params = self.optimized_params
            optimal_min = params['rsi_optimal_min']
            optimal_max = params['rsi_optimal_max']
            good_range = params['rsi_good_range']
            
            if optimal_min <= rsi <= optimal_max:
                return 1.0
            elif (optimal_min - good_range) <= rsi < optimal_min:
                return 0.8
            elif optimal_max < rsi <= (optimal_max + good_range):
                return 0.8
            elif rsi < 20 or rsi > 80:
                return 0.2
            else:
                return 0.4
        except:
            return 0.5
    
    def _calculate_kdj_k_score_optimized(self, data: Dict) -> float:
        """KDJ_K优化评分 (继承v3.52)"""
        try:
            kdj_k = data.get('kdj_k', 50.0)
            if kdj_k is None or pd.isna(kdj_k):
                return 0.5
            
            params = self.optimized_params
            optimal_min = params['kdj_k_optimal_min']
            optimal_max = params['kdj_k_optimal_max']
            good_range = params['kdj_k_good_range']
            
            if optimal_min <= kdj_k <= optimal_max:
                return 1.0
            elif (optimal_min - good_range) <= kdj_k < optimal_min:
                return 0.8
            elif optimal_max < kdj_k <= (optimal_max + good_range):
                return 0.8
            elif kdj_k < 10 or kdj_k > 90:
                return 0.2
            else:
                return 0.4
        except:
            return 0.5
    
    def _calculate_kdj_d_score_optimized(self, data: Dict) -> float:
        """KDJ_D优化评分 (继承v3.52)"""
        try:
            kdj_d = data.get('kdj_d', 50.0)
            if kdj_d is None or pd.isna(kdj_d):
                return 0.5
            
            params = self.optimized_params
            optimal_min = params['kdj_d_optimal_min']
            optimal_max = params['kdj_d_optimal_max']
            good_range = params['kdj_d_good_range']
            
            if optimal_min <= kdj_d <= optimal_max:
                return 1.0
            elif (optimal_min - good_range) <= kdj_d < optimal_min:
                return 0.8
            elif optimal_max < kdj_d <= (optimal_max + good_range):
                return 0.8
            elif kdj_d < 10 or kdj_d > 90:
                return 0.2
            else:
                return 0.4
        except:
            return 0.5
    
    def _calculate_bbi_score_optimized(self, data: Dict) -> float:
        """BBI优化评分 (继承v3.52)"""
        try:
            close = data.get('close', 0)
            bbi = data.get('bbi', 0)
            if not close or not bbi or pd.isna(close) or pd.isna(bbi):
                return 0.5
            
            ratio = close / bbi
            params = self.optimized_params
            optimal_min = params['bbi_optimal_min']
            optimal_max = params['bbi_optimal_max'] 
            good_range = params['bbi_good_range']
            
            if optimal_min <= ratio <= optimal_max:
                return 1.0
            elif (optimal_min - good_range) <= ratio < optimal_min:
                return 0.8
            elif optimal_max < ratio <= (optimal_max + good_range):
                return 0.8
            elif ratio < 0.85 or ratio > 1.20:
                return 0.2
            else:
                return 0.4
        except:
            return 0.5
    
    def _calculate_zhixing_trend_score_optimized(self, data: Dict) -> float:
        """知行趋势优化评分 (继承v3.52)"""
        try:
            ema12 = data.get('ema12', 0)
            ema26 = data.get('ema26', 0)
            if not ema12 or not ema26 or pd.isna(ema12) or pd.isna(ema26):
                return 0.5
            
            ratio = ema12 / ema26
            params = self.optimized_params
            optimal_min = params['zhixing_trend_optimal_ratio_min']
            optimal_max = params['zhixing_trend_optimal_ratio_max']
            good_range = params['zhixing_trend_good_range']
            
            if optimal_min <= ratio <= optimal_max:
                return 1.0
            elif (optimal_min - good_range) <= ratio < optimal_min:
                return 0.8
            elif optimal_max < ratio <= (optimal_max + good_range):
                return 0.8
            elif ratio < 0.90 or ratio > 1.15:
                return 0.2
            else:
                return 0.4
        except:
            return 0.5
    
    def _calculate_zhixing_multiavg_score_optimized(self, data: Dict) -> float:
        """知行多均优化评分 (继承v3.52)"""
        try:
            close = data.get('close', 0)
            ma5 = data.get('ma5', 0)
            ma10 = data.get('ma10', 0)
            ma20 = data.get('ma20', 0)
            
            if not all([close, ma5, ma10, ma20]) or any(pd.isna([close, ma5, ma10, ma20])):
                return 0.5
            
            avg_ma = (ma5 + ma10 + ma20) / 3
            ratio = close / avg_ma
            
            params = self.optimized_params
            optimal_min = params['zhixing_multiavg_optimal_ratio_min']
            optimal_max = params['zhixing_multiavg_optimal_ratio_max']
            good_range = params['zhixing_multiavg_good_range']
            
            if optimal_min <= ratio <= optimal_max:
                return 1.0
            elif (optimal_min - good_range) <= ratio < optimal_min:
                return 0.8
            elif optimal_max < ratio <= (optimal_max + good_range):
                return 0.8
            elif ratio < 0.85 or ratio > 1.25:
                return 0.2
            else:
                return 0.4
        except:
            return 0.5
    
    def _calculate_pe_score_optimized(self, data: Dict) -> float:
        """PE优化评分 (继承v3.52)"""
        try:
            pe = data.get('pe_ttm', 0)
            if not pe or pd.isna(pe) or pe <= 0:
                return 0.3
            
            params = self.optimized_params
            optimal_min = params['pe_optimal_min']
            optimal_max = params['pe_optimal_max']
            good_low = params['pe_good_range_low']
            good_high = params['pe_good_range_high']
            
            if optimal_min <= pe <= optimal_max:
                return 1.0
            elif good_low <= pe < optimal_min:
                return 0.8
            elif optimal_max < pe <= (optimal_max + good_high):
                return 0.8
            elif pe > 100 or pe < 5:
                return 0.2
            else:
                return 0.4
        except:
            return 0.3
    
    def _calculate_pb_score_optimized(self, data: Dict) -> float:
        """PB优化评分 (继承v3.52)"""
        try:
            pb = data.get('pb', 0)
            if not pb or pd.isna(pb) or pb <= 0:
                return 0.3
            
            params = self.optimized_params
            optimal_min = params['pb_optimal_min']
            optimal_max = params['pb_optimal_max']
            good_low = params['pb_good_range_low']
            good_high = params['pb_good_range_high']
            
            if optimal_min <= pb <= optimal_max:
                return 1.0
            elif good_low <= pb < optimal_min:
                return 0.8
            elif optimal_max < pb <= (optimal_max + good_high):
                return 0.8
            elif pb > 10 or pb < 0.3:
                return 0.2
            else:
                return 0.4
        except:
            return 0.3
    
    def _calculate_market_cap_score_optimized(self, data: Dict) -> float:
        """市值优化评分 (继承v3.52)"""
        try:
            market_cap = data.get('market_cap', 0)
            if not market_cap or pd.isna(market_cap):
                return 0.5
            
            # 转换为亿元
            market_cap_yi = market_cap / 1e8
            
            params = self.optimized_params
            optimal_min = params['market_cap_optimal_min']
            optimal_max = params['market_cap_optimal_max']
            small_min = params['market_cap_small_cap_min']
            large_max = params['market_cap_large_cap_max']
            
            if optimal_min <= market_cap_yi <= optimal_max:
                return 1.0
            elif small_min <= market_cap_yi < optimal_min:
                return 0.8
            elif optimal_max < market_cap_yi <= large_max:
                return 0.7
            elif market_cap_yi < small_min:
                return 0.4  # 超小盘风险
            elif market_cap_yi > large_max:
                return 0.6  # 超大盘成长性差
            else:
                return 0.5
        except:
            return 0.5
    
    def _calculate_price_momentum_score_optimized(self, data: Dict) -> float:
        """价格动量优化评分 (继承v3.52)"""
        try:
            chg_1d = data.get('price_change_pct', 0) or 0
            chg_5d = data.get('price_change_5d', 0) or 0
            chg_10d = data.get('price_change_10d', 0) or 0
            chg_20d = data.get('price_change_20d', 0) or 0
            
            params = self.optimized_params
            w1, w5, w10, w20 = (params['momentum_weight_1d'], params['momentum_weight_5d'],
                               params['momentum_weight_10d'], params['momentum_weight_20d'])
            
            composite_momentum = (chg_1d * w1 + chg_5d * w5 + 
                                chg_10d * w10 + chg_20d * w20)
            
            excellent_threshold = params['momentum_excellent_threshold']
            good_threshold = params['momentum_good_threshold']
            negative_threshold = params['momentum_negative_threshold']
            
            if composite_momentum >= excellent_threshold:
                return 1.0
            elif composite_momentum >= good_threshold:
                return 0.8
            elif composite_momentum >= 0:
                return 0.6
            elif composite_momentum >= negative_threshold:
                return 0.3
            else:
                return 0.1
        except:
            return 0.5
    
    def _calculate_volume_surge_score_optimized(self, data: Dict) -> float:
        """成交量激增优化评分 (继承v3.52)"""
        try:
            volume_ratio_5d = data.get('volume_ratio_5d', 1.0) or 1.0
            volume_ratio_20d = data.get('volume_ratio_20d', 1.0) or 1.0
            
            params = self.optimized_params
            w5 = params['volume_weight_5d']
            w20 = params['volume_weight_20d']
            
            composite_ratio = volume_ratio_5d * w5 + volume_ratio_20d * w20
            
            optimal_min = params['volume_surge_optimal_min']
            optimal_max = params['volume_surge_optimal_max']
            excellent_max = params['volume_surge_excellent_max']
            
            if optimal_min <= composite_ratio <= optimal_max:
                return 1.0
            elif composite_ratio <= excellent_max:
                return 0.8
            elif composite_ratio > excellent_max:
                return 0.6  # 过度放量风险
            elif composite_ratio < 0.8:
                return 0.3  # 量能不足
            else:
                return 0.5
        except:
            return 0.5
    
    def _calculate_volatility_risk_score(self, data: Dict) -> float:
        """波动性风险评分 (继承v3.52)"""
        try:
            volatility_20d = data.get('volatility_20d', 0.02) or 0.02
            
            if volatility_20d <= 0.015:  # 极低波动
                return 0.7
            elif volatility_20d <= 0.025:  # 低波动
                return 1.0
            elif volatility_20d <= 0.035:  # 中等波动
                return 0.8
            elif volatility_20d <= 0.05:   # 高波动
                return 0.5
            else:  # 极高波动
                return 0.2
        except:
            return 0.6
    
    def get_optimization_metrics(self) -> Dict:
        """获取优化相关指标"""
        return {
            'version': 'V3.53_MultiPeriod',
            'periods_supported': list(self.period_weights.keys()),
            'period_importance': self.period_importance.copy(),
            'total_factors': 12,
            'optimization_target': 'Multi-Period IC Maximization',
            'expected_improvements': {
                '1d_ic_target': '2.5%+',
                '3d_ic_target': '1.5%+', 
                '5d_ic_target': '1.0%+',
                '10d_ic_target': '0.5%+',
                '15d_ic_target': '0.3%+'
            },
            'innovation': 'Layered Multi-Period Weight Architecture'
        }
    
    def export_configuration(self, filepath: str = None) -> str:
        """导出配置到JSON文件"""
        config = {
            'version': 'V3.53_MultiPeriod',
            'creation_date': datetime.now().isoformat(),
            'period_weights': self.period_weights,
            'period_importance': self.period_importance,
            'optimized_params': self.optimized_params,
            'metrics': self.get_optimization_metrics()
        }
        
        if filepath is None:
            filepath = f"v353_multiperiod_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"✅ V3.53配置已导出到: {filepath}")
        return filepath
    
    def _get_investment_advice(self, score: float) -> Tuple[str, str]:
        """
        基于评分生成投资建议
        
        Args:
            score: 复合评分 (0-100)
            
        Returns:
            Tuple[str, str]: (投资建议, 置信度)
        """
        if score >= 80:
            return "买入", "高"
        elif score >= 70:
            return "谨慎买入", "中"
        elif score >= 60:
            return "观望", "低"
        else:
            return "回避", "低"


def create_v353_scorer(db_path: str = "data_adapter/stock_data.db") -> QuantitativeScorerV353MultiPeriod:
    """创建V3.53多时间周期评分器实例"""
    return QuantitativeScorerV353MultiPeriod(db_path=db_path)


if __name__ == "__main__":
    # 测试V3.53多时间周期评分器
    print("🚀 V3.53 多时间周期IC优化量化评分系统")
    print("="*60)
    
    scorer = create_v353_scorer()
    
    # 导出配置
    config_file = scorer.export_configuration()
    print(f"📁 配置已导出: {config_file}")
    
    # 显示优化指标
    metrics = scorer.get_optimization_metrics()
    print("\n📊 系统指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print("\n✅ V3.53系统初始化完成，准备优化多时间周期IC表现！")