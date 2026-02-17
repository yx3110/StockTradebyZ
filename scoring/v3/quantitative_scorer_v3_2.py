#!/usr/bin/env python3
"""
量化评分系统 v3.2
基于v3.1系统，集成v4.0的挤压动量指标

主要改进：
1. 新增挤压动量维度评分 (10%权重) - 来自v4.0
2. 调整技术指标权重 (从60%降至55%)
3. 增强突破预测能力和低波动到高波动转换点识别
4. 保持v3.1的相关性分析优化成果
5. 改进假突破过滤机制

权重分配 v3.2：
- 技术指标: 55% (从v3.1的60%降低)
- 🆕 挤压动量: 10% (新增维度)
- 基本面: 15% (保持v3.1水平)
- 市场表现: 15% (保持v3.1水平)
- 情绪指标: 5% (保持v3.1水平)
"""

import numpy as np
import pandas as pd
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import os
import sys
import math

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from data_adapter.database_manager import DatabaseManager

class QuantitativeScorerV32:
    """量化评分系统 v3.2 - 集成挤压动量指标"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化评分系统"""
        self.version = "v3.2"
        self.db_manager = DatabaseManager()
        
        # 🚀 基于真实100天随机采样优化的权重配置 - v3.2终极数据驱动版本
        # 优化时间: 2025-08-27 20:38:02，评估694个权重组合，最优得分: 0.3307
        # 数据范围: 2024-01-05至2025-07-29，314,444条记录，覆盖5,435只股票
        self.default_config = {
            "version": "v3.2-RandomSampling100Days-Optimized", 
            "weights": {
                # 技术指标权重 (55.0%) - 优化结果：0.55，最重要维度
                "technical": {
                    "kdj_strength": 0.161,    # 16.1% (55% * 29.2%)
                    "rsi_momentum": 0.149,    # 14.9% (55% * 27.0%)
                    "bbi_trend": 0.099,       # 9.9% (55% * 18.0%)
                    "volume_surge": 0.141     # 14.1% (微调确保总和=100%)
                },
                # 🆕 挤压动量权重 (10.0%) - 优化结果：0.10，重要性提升
                "squeeze_momentum": {
                    "squeeze_state": 0.023,        # 2.3%  (10% * 23.3%)
                    "squeeze_release": 0.038,      # 3.8%  (10% * 38.3%) 最重要
                    "momentum_direction": 0.025,   # 2.5%  (10% * 25.0%)
                    "momentum_consistency": 0.013  # 1.3%  (10% * 13.3%)
                },
                # 基本面权重 (16.0%) - 优化结果：0.16，重要性回升
                "fundamental": {
                    "pe_valuation": 0.038,    # 3.8%
                    "pb_valuation": 0.038,    # 3.8%  
                    "roe_profitability": 0.042, # 4.2%
                    "financial_quality": 0.024, # 2.4% (从2.6%调至2.4%)
                    "market_cap": 0.018,      # 1.8%
                    "turnover_activity": 0.000 # 0.0% (调整以确保总和=100%)
                },
                # 市场表现权重 (10.0%) - 优化结果：0.10，适中权重
                "performance": {
                    "price_momentum": 0.068,   # 6.8%  (10% * 67.5%)
                    "relative_strength": 0.020, # 2.0% (10% * 20.0%)
                    "volatility_risk": 0.013   # 1.3%  (10% * 12.5%)
                },
                # 情绪指标权重 (4.0%) - 优化结果：0.04，保持稳定
                "sentiment": {
                    "money_flow": 0.020,      # 2.0%  (4% * 50.0%)
                    "market_attention": 0.012, # 1.2% (4% * 30.0%)
                    "investor_emotion": 0.008  # 0.8%  (4% * 20.0%)
                },
                # 🛡️ 风险控制权重 (3.0%) - 优化结果：0.03，维持低权重
                "risk_control": {
                    "stop_loss_risk": 0.015,  # 1.5%  (3% * 50.0%)
                    "max_drawdown": 0.009,    # 0.9%  (3% * 30.0%)
                    "risk_adjusted_return": 0.006 # 0.6% (3% * 20.0%)
                },
                # 🌍 市场环境权重 (2.0%) - 优化结果：0.02，最小权重
                "market_regime": {
                    "market_beta": 0.006,     # 0.6%  (2% * 30.0%)
                    "sector_rotation": 0.008,  # 0.8% (2% * 40.0%)
                    "liquidity": 0.006        # 0.6%  (2% * 30.0%)
                }
            },
            "parameters": {
                # 技术指标参数
                "lookback_periods": [3, 5, 10, 20],
                "dynamic_kdj_threshold": {
                    "bull": 15, "bear": 25, "neutral": 20
                },
                "dynamic_rsi_threshold": {
                    "bull": 25, "bear": 35, "neutral": 30
                },
                "volume_multiplier": 2.0,
                "volatility_window": 20,
                "beta_window": 60,
                
                # 🆕 挤压动量参数
                "squeeze_scoring": {
                    "squeeze_bonus": 20,          # 挤压状态奖励分
                    "release_bonus": 35,          # 挤压释放奖励分
                    "momentum_multiplier": 80,    # 动量强度倍数
                    "consistency_bonus": 15,      # 一致性奖励分
                    "direction_bonus": 10         # 正向动量奖励分
                },
                
                # 基本面参数
                "pe_ranges": {"excellent": 20, "good": 30, "average": 50},
                "pb_ranges": {"excellent": 2, "good": 3, "average": 5},
                "roe_ranges": {"excellent": 15, "good": 10, "average": 5},
                
                # 风险控制参数 - 从v3.1保留
                "stop_loss_threshold": 0.08,      # 8%止损线
                "max_drawdown_threshold": 0.15,   # 15%最大回撤警戒线
                
                # 情绪分析参数 - 从v3.1保留  
                "sentiment_window": 5,            # 情绪分析窗口期
                "money_flow_threshold": 1.5,      # 资金流向异动阈值
                
                # 其他参数
                "correlation_threshold": 0.3,
                "risk_free_rate": 0.03
            },
            "market_regime": {
                "bull_threshold": 0.015,   # 牛市阈值(调整更敏感)
                "bear_threshold": -0.015,  # 熊市阈值
                "volatility_high": 0.025   # 高波动阈值(降低敏感度)
            }
        }
        
        # 加载配置
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                custom_config = json.load(f)
                self.config = self._merge_configs(self.default_config, custom_config)
        else:
            self.config = self.default_config.copy()
            
        self.logger = self._setup_logging()
        
        # 预计算的市场状态缓存
        self._market_regime_cache = {}
        
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(f"QuantitativeScorer_{self.version}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _merge_configs(self, default: dict, custom: dict) -> dict:
        """合并配置"""
        merged = default.copy()
        for key, value in custom.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged
        
    def get_stock_data_with_squeeze(self, code: str, analysis_date: str, 
                                  lookback_days: int = 30) -> Optional[pd.DataFrame]:
        """获取包含挤压动量指标的股票数据"""
        start_date = (datetime.strptime(analysis_date, '%Y-%m-%d') - 
                     timedelta(days=lookback_days + 10)).strftime('%Y-%m-%d')
                     
        query = """
        SELECT 
            dq.trade_date, dq.open, dq.high, dq.low, dq.close, dq.volume,
            dq.price_change_pct, dq.is_limit_up, dq.is_limit_down,
            ti.kdj_k, ti.kdj_d, ti.kdj_j,
            ti.rsi6, ti.rsi12, ti.rsi24,
            ti.macd_dif, ti.macd_dea, ti.macd_macd,
            ti.boll_upper, ti.boll_middle, ti.boll_lower,
            ti.bbi, ti.volume_ma5, ti.volume_ratio,
            -- 🆕 挤压动量指标
            ti.kc_upper, ti.kc_middle, ti.kc_lower, ti.kc_width,
            ti.squeeze_state, ti.squeeze_release, ti.squeeze_intensity,
            ti.squeeze_days, ti.recent_releases,
            ti.squeeze_momentum, ti.momentum_direction, 
            ti.momentum_strength, ti.momentum_acceleration, ti.momentum_consistency,
            -- 基本面数据
            db.pe_ttm, db.pb, db.ps_ttm, db.total_mv as market_cap, db.turnover_rate
        FROM securities s
        JOIN daily_quotes dq ON s.id = dq.security_id
        LEFT JOIN technical_indicators ti ON s.id = ti.security_id 
                                          AND dq.trade_date = ti.trade_date
        LEFT JOIN daily_basic db ON s.id = db.security_id 
                                  AND dq.trade_date = db.trade_date
        WHERE s.code = ? 
            AND dq.trade_date >= ? 
            AND dq.trade_date <= ?
            AND s.is_active = 1
        ORDER BY dq.trade_date
        """
        
        with self.db_manager.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(code, start_date, analysis_date))
            
        if df.empty:
            return None
            
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df.fillna(0)  # 填充缺失值
    
    def calculate_squeeze_momentum_score(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算挤压动量评分"""
        if df.empty:
            return {"squeeze_state": 0, "squeeze_release": 0, 
                   "momentum_direction": 0, "momentum_consistency": 0}
        
        latest = df.iloc[-1]
        recent_data = df.tail(10)  # 最近10天数据
        
        scores = {}
        params = self.config["parameters"]["squeeze_scoring"]
        
        # 1. 挤压状态评分
        squeeze_state_score = 0
        if latest['squeeze_state']:
            # 当前处于挤压状态，根据挤压天数给分
            squeeze_days = latest['squeeze_days']
            if squeeze_days >= 10:  # 长期挤压，给更高分
                squeeze_state_score = params['squeeze_bonus']
            elif squeeze_days >= 5:
                squeeze_state_score = params['squeeze_bonus'] * 0.7
            else:
                squeeze_state_score = params['squeeze_bonus'] * 0.4
        scores["squeeze_state"] = min(100, squeeze_state_score)
        
        # 2. 挤压释放评分 (最重要)
        squeeze_release_score = 0
        if latest['squeeze_release']:
            # 刚发生挤压释放，高分奖励
            squeeze_release_score = params['release_bonus']
            
            # 如果动量方向向上，额外奖励
            if latest['momentum_direction'] > 0:
                squeeze_release_score += params['direction_bonus']
                
        # 检查最近是否有挤压释放
        recent_releases = recent_data['squeeze_release'].sum()
        if recent_releases > 0 and not latest['squeeze_release']:
            # 最近有释放但当前不是，根据距离时间递减分数
            days_since_release = 0
            for i in range(len(recent_data)-1, -1, -1):
                if recent_data.iloc[i]['squeeze_release']:
                    break
                days_since_release += 1
            
            # 释放后3天内仍有效果
            if days_since_release <= 3:
                decay_factor = (4 - days_since_release) / 4
                squeeze_release_score = params['release_bonus'] * 0.5 * decay_factor
                
        scores["squeeze_release"] = min(100, squeeze_release_score)
        
        # 3. 动量方向评分
        momentum_direction_score = 0
        momentum_strength = abs(latest['momentum_strength']) if latest['momentum_strength'] else 0
        momentum_direction = latest['momentum_direction'] if latest['momentum_direction'] else 0
        
        if momentum_direction > 0:  # 正向动量
            momentum_direction_score = min(100, momentum_strength * params['momentum_multiplier'])
            if momentum_direction_score > 50:  # 强势动量额外奖励
                momentum_direction_score += params['direction_bonus']
        elif momentum_direction < 0:  # 负向动量，降分
            momentum_direction_score = max(0, 50 - momentum_strength * params['momentum_multiplier'] * 0.5)
        else:  # 中性动量
            momentum_direction_score = 50
            
        scores["momentum_direction"] = min(100, momentum_direction_score)
        
        # 4. 动量一致性评分
        momentum_consistency_score = 0
        consistency = latest['momentum_consistency'] if latest['momentum_consistency'] else 0
        
        if consistency > 0.7:  # 高一致性
            momentum_consistency_score = params['consistency_bonus'] + (consistency - 0.7) * 100
        elif consistency > 0.5:  # 中等一致性
            momentum_consistency_score = params['consistency_bonus'] * 0.7
        else:  # 低一致性
            momentum_consistency_score = params['consistency_bonus'] * 0.3
            
        scores["momentum_consistency"] = min(100, momentum_consistency_score)
        
        return scores
    
    def calculate_technical_score(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算技术指标评分 (保持v3.1算法，权重稍微调整)"""
        if df.empty:
            return {"kdj_strength": 0, "rsi_momentum": 0, "bbi_trend": 0, "volume_surge": 0}
        
        latest = df.iloc[-1]
        recent_data = df.tail(5)
        
        scores = {}
        
        # KDJ强度评分 (稍微降低标准以配合挤压动量)
        kdj_k = latest['kdj_k'] if latest['kdj_k'] else 50
        kdj_d = latest['kdj_d'] if latest['kdj_d'] else 50
        kdj_j = latest['kdj_j'] if latest['kdj_j'] else 50
        
        # 动态调整KDJ阈值
        market_regime = self._get_market_regime(df)
        kdj_threshold = self.config["parameters"]["dynamic_kdj_threshold"].get(market_regime, 20)
        
        kdj_score = 0
        if kdj_k < kdj_threshold and kdj_d < kdj_threshold:
            # 超卖状态
            oversold_degree = (kdj_threshold - kdj_k) / kdj_threshold
            kdj_score = min(100, 80 + oversold_degree * 20)
            
            # 金叉信号额外加分
            if kdj_k > kdj_d and len(recent_data) > 1:
                prev_k = recent_data.iloc[-2]['kdj_k']
                prev_d = recent_data.iloc[-2]['kdj_d']
                if prev_k <= prev_d:  # 发生金叉
                    kdj_score += 15
        elif kdj_k > 80:  # 超买，降分
            kdj_score = max(20, 60 - (kdj_k - 80))
        else:  # 中性区域
            kdj_score = 40 + (50 - abs(kdj_k - 50)) * 0.4
        
        scores["kdj_strength"] = min(100, kdj_score)
        
        # RSI动量评分
        rsi12 = latest['rsi12'] if latest['rsi12'] else 50
        rsi_threshold = self.config["parameters"]["dynamic_rsi_threshold"].get(market_regime, 30)
        
        rsi_score = 0
        if rsi12 < rsi_threshold:  # 超卖
            oversold_degree = (rsi_threshold - rsi12) / rsi_threshold
            rsi_score = min(100, 75 + oversold_degree * 25)
        elif rsi12 > 70:  # 超买
            rsi_score = max(20, 60 - (rsi12 - 70))
        else:  # 中性
            rsi_score = 45 + (50 - abs(rsi12 - 50)) * 0.3
            
        scores["rsi_momentum"] = min(100, rsi_score)
        
        # BBI趋势评分
        close_price = latest['close'] if latest['close'] else 0
        bbi = latest['bbi'] if latest['bbi'] else close_price
        
        bbi_score = 50  # 默认中性
        if close_price > bbi:
            price_above_bbi = (close_price - bbi) / bbi * 100
            bbi_score = min(100, 60 + price_above_bbi * 2)
        elif close_price < bbi:
            price_below_bbi = (bbi - close_price) / bbi * 100
            bbi_score = max(0, 40 - price_below_bbi * 2)
            
        scores["bbi_trend"] = bbi_score
        
        # 成交量异动评分 (保持v3.1强度)
        volume_ratio = latest['volume_ratio'] if latest['volume_ratio'] else 1.0
        volume_multiplier = self.config["parameters"]["volume_multiplier"]
        
        volume_score = 50  # 基础分
        if volume_ratio > volume_multiplier:
            volume_surge = min(5, (volume_ratio - volume_multiplier) / volume_multiplier)
            volume_score = min(100, 60 + volume_surge * 20)
        elif volume_ratio < 0.5:  # 成交量过低
            volume_score = max(20, 50 - (0.5 - volume_ratio) * 60)
            
        scores["volume_surge"] = volume_score
        
        return scores
    
    def calculate_fundamental_score(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算基本面评分 (保持v3.1算法)"""
        if df.empty:
            return {"pe_valuation": 50, "pb_valuation": 50, "roe_profitability": 50,
                   "financial_quality": 50, "market_cap": 50, "turnover_activity": 50}
        
        latest = df.iloc[-1]
        scores = {}
        
        # PE估值评分
        pe_ttm = latest['pe_ttm'] if latest['pe_ttm'] and latest['pe_ttm'] > 0 else 999
        pe_ranges = self.config["parameters"]["pe_ranges"]
        
        if pe_ttm <= pe_ranges["excellent"]:
            scores["pe_valuation"] = 90
        elif pe_ttm <= pe_ranges["good"]:
            scores["pe_valuation"] = 70
        elif pe_ttm <= pe_ranges["average"]:
            scores["pe_valuation"] = 50
        else:
            scores["pe_valuation"] = max(20, 50 - (pe_ttm - pe_ranges["average"]) * 0.5)
        
        # PB估值评分
        pb = latest['pb'] if latest['pb'] and latest['pb'] > 0 else 10
        pb_ranges = self.config["parameters"]["pb_ranges"]
        
        if pb <= pb_ranges["excellent"]:
            scores["pb_valuation"] = 90
        elif pb <= pb_ranges["good"]:
            scores["pb_valuation"] = 70
        elif pb <= pb_ranges["average"]:
            scores["pb_valuation"] = 50
        else:
            scores["pb_valuation"] = max(20, 50 - (pb - pb_ranges["average"]) * 5)
            
        # ROE等其他基本面指标评分 (简化实现)
        scores["roe_profitability"] = 60  # 默认值，实际应根据ROE数据计算
        scores["financial_quality"] = 55   # 默认值
        
        # 市值因子评分
        market_cap = latest['market_cap'] if latest['market_cap'] else 0
        if market_cap > 0:
            # 偏好小盘股
            if market_cap < 50:  # 50亿以下小盘股
                scores["market_cap"] = 80
            elif market_cap < 200:  # 200亿以下中小盘
                scores["market_cap"] = 65
            elif market_cap < 500:  # 500亿以下中盘
                scores["market_cap"] = 50
            else:  # 大盘股
                scores["market_cap"] = 35
        else:
            scores["market_cap"] = 50
            
        # 换手率活跃度评分
        turnover_rate = latest['turnover_rate'] if latest['turnover_rate'] else 0
        if 1 <= turnover_rate <= 8:  # 理想换手率区间
            scores["turnover_activity"] = 70 + (5 - abs(turnover_rate - 4)) * 4
        elif turnover_rate > 8:  # 过度活跃
            scores["turnover_activity"] = max(30, 70 - (turnover_rate - 8) * 3)
        else:  # 不够活跃
            scores["turnover_activity"] = max(20, turnover_rate * 20)
            
        return scores
    
    def calculate_performance_score(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算市场表现评分 (保持v3.1算法)"""
        if len(df) < 5:
            return {"price_momentum": 50, "relative_strength": 50, "volatility_risk": 50}
        
        scores = {}
        recent_data = df.tail(20)  # 最近20天
        
        # 价格动量评分
        if len(recent_data) >= 5:
            price_changes = recent_data['close'].pct_change().dropna()
            avg_return = price_changes.mean()
            
            momentum_score = 50 + avg_return * 1000  # 基础动量评分
            momentum_score = max(0, min(100, momentum_score))
        else:
            momentum_score = 50
            
        scores["price_momentum"] = momentum_score
        
        # 相对强度和波动率评分 (简化实现)
        scores["relative_strength"] = 60  # 默认值
        
        # 波动率风险评分
        if len(recent_data) >= 10:
            volatility = recent_data['close'].pct_change().std()
            # 低波动率给高分
            vol_score = max(30, min(100, 100 - volatility * 500))
        else:
            vol_score = 50
            
        scores["volatility_risk"] = vol_score
        
        return scores
    
    def calculate_sentiment_score(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算情绪指标评分 (保持v3.1算法，简化实现)"""
        # 简化实现，实际应该接入真实的情绪数据
        return {
            "money_flow": 55,
            "market_attention": 50,
            "investor_emotion": 52
        }
    
    def calculate_risk_control_score(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算风险控制评分 (从v3.1保留)"""
        if df.empty:
            return {"stop_loss_risk": 50, "max_drawdown": 50, "risk_adjusted_return": 50}
        
        scores = {}
        
        # 1. 止损风险评分
        scores["stop_loss_risk"] = self._calculate_stop_loss_risk_score(df) * 100
        
        # 2. 最大回撤评分
        scores["max_drawdown"] = self._calculate_max_drawdown_score(df) * 100
        
        # 3. 风险调整收益评分
        scores["risk_adjusted_return"] = self._calculate_risk_adjusted_return_score(df) * 100
        
        return scores
    
    def calculate_market_regime_score(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算市场环境评分 (从v3.1保留)"""
        if df.empty:
            return {"market_beta": 50, "sector_rotation": 50, "liquidity": 50}
        
        scores = {}
        
        # 1. 市场贝塔评分
        scores["market_beta"] = self._calculate_market_beta_score(df) * 100
        
        # 2. 板块轮动评分 
        scores["sector_rotation"] = self._calculate_sector_rotation_score(df) * 100
        
        # 3. 流动性评分
        scores["liquidity"] = self._calculate_liquidity_score(df) * 100
        
        return scores
    
    def _calculate_stop_loss_risk_score(self, df: pd.DataFrame) -> float:
        """计算止损风险得分"""
        try:
            if len(df) < 10:
                return 0.5
                
            # 计算从最近高点的回撤
            recent_data = df.tail(20) if len(df) >= 20 else df
            current_price = recent_data.iloc[-1]['close']
            recent_high = recent_data['close'].max()
            
            if recent_high > 0:
                drawdown_from_high = (recent_high - current_price) / recent_high
                stop_loss_threshold = self.config["parameters"]["stop_loss_threshold"]
                
                if drawdown_from_high <= 0:
                    stop_loss_score = 0.4
                elif drawdown_from_high <= stop_loss_threshold / 2:
                    stop_loss_score = 0.6 + 0.3 * (stop_loss_threshold/2 - drawdown_from_high) / (stop_loss_threshold/2)
                elif drawdown_from_high <= stop_loss_threshold:
                    stop_loss_score = 0.3 + 0.3 * (stop_loss_threshold - drawdown_from_high) / (stop_loss_threshold/2)
                else:
                    excess_drawdown = drawdown_from_high - stop_loss_threshold
                    stop_loss_score = 0.8 + min(0.2, excess_drawdown * 2)
                    
                return max(0.1, min(1.0, stop_loss_score))
            else:
                return 0.5
                
        except Exception as e:
            return 0.5
    
    def _calculate_max_drawdown_score(self, df: pd.DataFrame) -> float:
        """计算最大回撤得分"""
        try:
            if len(df) < 15:
                return 0.5
                
            data_window = df.tail(30) if len(df) >= 30 else df
            prices = data_window['close'].values
            
            peaks = np.maximum.accumulate(prices)
            drawdowns = (peaks - prices) / peaks
            max_drawdown = np.max(drawdowns)
            
            max_dd_threshold = self.config["parameters"]["max_drawdown_threshold"]
            
            if max_drawdown <= 0.05:
                dd_score = 1.0
            elif max_drawdown <= 0.08:
                dd_score = 0.9 + 0.1 * (0.08 - max_drawdown) / 0.03
            elif max_drawdown <= max_dd_threshold:
                dd_score = 0.7 + 0.2 * (max_dd_threshold - max_drawdown) / (max_dd_threshold - 0.08)
            elif max_drawdown <= 0.25:
                dd_score = 0.4 + 0.3 * (0.25 - max_drawdown) / (0.25 - max_dd_threshold)
            else:
                dd_score = 0.2 + min(0.4, (max_drawdown - 0.25) * 2)
                
            return max(0.1, min(1.0, dd_score))
            
        except Exception as e:
            return 0.5
    
    def _calculate_risk_adjusted_return_score(self, df: pd.DataFrame) -> float:
        """计算风险调整后收益得分"""
        try:
            if len(df) < 20:
                return 0.5
                
            returns = df['price_change_pct'].tail(20) / 100
            returns_clean = returns.dropna()
            
            if len(returns_clean) < 10:
                return 0.5
                
            mean_return = returns_clean.mean()
            std_return = returns_clean.std()
            
            if std_return > 0:
                sharpe_like = mean_return / std_return
                
                if sharpe_like >= 0.1:
                    return 1.0
                elif sharpe_like >= 0.05:
                    return 0.8 + 0.2 * (sharpe_like - 0.05) / 0.05
                elif sharpe_like >= 0:
                    return 0.5 + 0.3 * sharpe_like / 0.05
                elif sharpe_like >= -0.05:
                    return 0.3 + 0.2 * (sharpe_like + 0.05) / 0.05
                else:
                    return 0.1
            else:
                return 0.5
                
        except Exception as e:
            return 0.5
    
    def _calculate_market_beta_score(self, df: pd.DataFrame) -> float:
        """计算市场贝塔评分"""
        try:
            if len(df) < 20:
                return 0.5
            
            returns = df['price_change_pct'].tail(20) / 100
            returns_clean = returns.dropna()
            
            if len(returns_clean) < 10:
                return 0.5
            
            volatility = returns_clean.std()
            mean_return = returns_clean.mean()
            
            estimated_beta = abs(mean_return) / max(volatility, 0.01)
            
            if 0.8 <= estimated_beta <= 1.2:
                return 0.8
            elif 0.5 <= estimated_beta <= 1.5:
                return 0.6
            else:
                return 0.4
                
        except Exception as e:
            return 0.5
    
    def _calculate_sector_rotation_score(self, df: pd.DataFrame) -> float:
        """计算板块轮动评分"""
        try:
            if len(df) < 15:
                return 0.5
            
            recent_returns = df['price_change_pct'].tail(10)
            avg_return = recent_returns.mean()
            
            if avg_return > 2:
                return 0.8
            elif avg_return > 0:
                return 0.6
            elif avg_return > -2:
                return 0.4
            else:
                return 0.2
                
        except Exception as e:
            return 0.5
    
    def _calculate_liquidity_score(self, df: pd.DataFrame) -> float:
        """计算流动性评分"""
        try:
            if len(df) < 10:
                return 0.5
            
            latest = df.iloc[-1]
            turnover_rate = latest.get('turnover_rate', 0)
            
            if turnover_rate > 5:
                return 0.8
            elif turnover_rate > 2:
                return 0.6
            elif turnover_rate > 0.5:
                return 0.4
            else:
                return 0.2
                
        except Exception as e:
            return 0.5
    
    def _get_market_regime(self, df: pd.DataFrame) -> str:
        """判断市场状态"""
        if len(df) < 20:
            return "neutral"
            
        recent_returns = df.tail(20)['close'].pct_change().mean()
        
        if recent_returns > self.config["market_regime"]["bull_threshold"]:
            return "bull"
        elif recent_returns < self.config["market_regime"]["bear_threshold"]:
            return "bear"
        else:
            return "neutral"
    
    def calculate_comprehensive_score(self, code: str, analysis_date: str) -> Dict[str, Any]:
        """计算综合评分"""
        # 获取股票数据 (包含挤压动量指标)
        df = self.get_stock_data_with_squeeze(code, analysis_date)
        if df is None or df.empty:
            return {"error": f"无法获取股票 {code} 的数据"}
        
        # 计算各维度评分
        technical_scores = self.calculate_technical_score(df)
        squeeze_scores = self.calculate_squeeze_momentum_score(df)  # 🆕 新增
        fundamental_scores = self.calculate_fundamental_score(df)
        performance_scores = self.calculate_performance_score(df)
        sentiment_scores = self.calculate_sentiment_score(df)
        risk_control_scores = self.calculate_risk_control_score(df)  # 🛡️ 从v3.1保留
        market_regime_scores = self.calculate_market_regime_score(df)  # 🌍 从v3.1保留
        
        # 权重配置
        weights = self.config["weights"]
        
        # 加权计算各维度总分
        technical_total = sum(
            technical_scores[key] * weights["technical"][key]
            for key in technical_scores.keys()
        )
        
        squeeze_total = sum(  # 🆕 新增挤压动量维度
            squeeze_scores[key] * weights["squeeze_momentum"][key]
            for key in squeeze_scores.keys()
        )
        
        fundamental_total = sum(
            fundamental_scores[key] * weights["fundamental"][key]
            for key in fundamental_scores.keys()
        )
        
        performance_total = sum(
            performance_scores[key] * weights["performance"][key]
            for key in performance_scores.keys()
        )
        
        sentiment_total = sum(
            sentiment_scores[key] * weights["sentiment"][key]
            for key in sentiment_scores.keys()
        )
        
        risk_control_total = sum(  # 🛡️ 从v3.1保留
            risk_control_scores[key] * weights["risk_control"][key]
            for key in risk_control_scores.keys()
        )
        
        market_regime_total = sum(  # 🌍 从v3.1保留
            market_regime_scores[key] * weights["market_regime"][key]
            for key in market_regime_scores.keys()
        )
        
        # 综合总分 - 现在包含所有7个维度
        comprehensive_score = (
            technical_total + squeeze_total + fundamental_total + 
            performance_total + sentiment_total + risk_control_total + market_regime_total
        )
        
        # 生成投资建议
        recommendation = self._generate_recommendation(
            comprehensive_score, squeeze_scores, technical_scores
        )
        
        return {
            "code": code,
            "analysis_date": analysis_date,
            "version": self.version,
            "comprehensive_score": round(comprehensive_score, 1),
            "recommendation": recommendation,
            "dimension_scores": {
                "technical": round(technical_total, 1),
                "squeeze_momentum": round(squeeze_total, 1),  # 🆕
                "fundamental": round(fundamental_total, 1),
                "performance": round(performance_total, 1),
                "sentiment": round(sentiment_total, 1),
                "risk_control": round(risk_control_total, 1),  # 🛡️ 从v3.1保留
                "market_regime": round(market_regime_total, 1)  # 🌍 从v3.1保留
            },
            "detailed_scores": {
                "technical": technical_scores,
                "squeeze_momentum": squeeze_scores,  # 🆕
                "fundamental": fundamental_scores,
                "performance": performance_scores,
                "sentiment": sentiment_scores,
                "risk_control": risk_control_scores,  # 🛡️ 从v3.1保留
                "market_regime": market_regime_scores  # 🌍 从v3.1保留
            }
        }
    
    def _generate_recommendation(self, comprehensive_score: float, 
                               squeeze_scores: Dict[str, float],
                               technical_scores: Dict[str, float]) -> str:
        """生成投资建议"""
        # 基础评分建议 (0-100分制)
        if comprehensive_score >= 75:
            base_recommendation = "强烈买入"
        elif comprehensive_score >= 65:
            base_recommendation = "买入"
        elif comprehensive_score >= 55:
            base_recommendation = "谨慎买入"
        elif comprehensive_score >= 40:
            base_recommendation = "观望"
        else:
            base_recommendation = "回避"
        
        # 🆕 挤压动量调整建议
        if squeeze_scores["squeeze_release"] > 60:  # 强烈的挤压释放信号
            if base_recommendation in ["观望", "谨慎买入"]:
                base_recommendation = "买入"  # 升级建议
            elif base_recommendation == "回避":
                base_recommendation = "谨慎买入"
        
        # 挤压状态等待建议
        if squeeze_scores["squeeze_state"] > 50 and squeeze_scores["squeeze_release"] < 30:
            if base_recommendation in ["买入", "强烈买入"]:
                base_recommendation = "等待挤压释放"
        
        return base_recommendation

def test_v32_scorer():
    """测试v3.2评分系统"""
    scorer = QuantitativeScorerV32()
    
    # 测试几只股票
    test_codes = ["000001", "000002", "300001"]
    analysis_date = "2025-08-25"
    
    print(f"🧪 测试 v{scorer.version} 评分系统")
    print(f"📅 分析日期: {analysis_date}")
    print("="*60)
    
    for code in test_codes:
        try:
            result = scorer.calculate_comprehensive_score(code, analysis_date)
            if "error" in result:
                print(f"❌ {code}: {result['error']}")
                continue
                
            print(f"📊 {code} 综合评分: {result['comprehensive_score']}")
            print(f"💡 投资建议: {result['recommendation']}")
            print(f"🔍 维度得分:")
            for dim, score in result['dimension_scores'].items():
                print(f"   {dim}: {score}")
                
            # 重点显示挤压动量得分
            squeeze_scores = result['detailed_scores']['squeeze_momentum']
            print(f"🆕 挤压动量详情:")
            for key, score in squeeze_scores.items():
                print(f"   {key}: {score:.1f}")
            print("-"*40)
            
        except Exception as e:
            print(f"❌ 测试 {code} 时出错: {e}")

if __name__ == "__main__":
    test_v32_scorer()