#!/usr/bin/env python3
"""
量化评分系统 v3.4
基于v3.0成功经验的优化版本

主要改进：
1. 保留v3.0核心优势（正相关性、高分组夏普比率）
2. 微调权重分配（技术62%、基本面17%、表现18%、市场环境3%）
3. 新增ROE盈利能力和营收增长指标
4. 优化90+分股票识别算法
5. 保持Market Regime乘数机制
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

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from data_adapter.database_manager import DatabaseManager

class QuantitativeScorerV34:
    """量化评分系统 v3.4 - 基于v3.0优化"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化评分系统"""
        self.version = "v3.4"
        self.db_manager = DatabaseManager()
        
        # v3.4优化配置 - 基于v3.0成功权重微调
        self.default_config = {
            "version": "v3.4-Enhanced-Based-on-v3.0",
            "weights": {
                # 技术指标权重 (62%) - 从v3.0的65%微调至62%
                "technical": {
                    "kdj_strength": 0.18,     # 从0.20微调至0.18
                    "rsi_momentum": 0.16,     # 从0.18微调至0.16
                    "bbi_trend": 0.12,        # 保持不变
                    "volume_surge": 0.16      # 从0.15提升至0.16
                },
                # 基本面权重 (17%) - 从v3.0的10%大幅提升至17%
                "fundamental": {
                    "pe_valuation": 0.03,     # 从0.02提升至0.03
                    "pb_valuation": 0.03,     # 从0.02提升至0.03
                    "roe_profitability": 0.03, # 新增ROE盈利能力指标
                    "revenue_growth": 0.02,   # 新增营收增长指标
                    "market_cap": 0.03,       # 保持不变
                    "turnover_activity": 0.03  # 保持不变
                },
                # 市场表现权重 (18%) - 从v3.0的20%微调至18%
                "performance": {
                    "price_momentum": 0.12,   # 从0.15微调至0.12
                    "relative_strength": 0.04, # 从0.03提升至0.04
                    "volatility_risk": 0.02   # 保持不变
                },
                # 市场环境权重 (3%) - 从v3.0的5%降至3%，保持乘数机制
                "market_regime": {
                    "market_beta": 0.01,      # 保持不变
                    "sector_rotation": 0.01,  # 从0.02降至0.01
                    "liquidity": 0.01        # 从0.02降至0.01
                }
            },
            "parameters": {
                "lookback_periods": [5, 10, 20, 30],
                "kdj_threshold": 18,                   # 从20微调至18，更敏感
                "rsi_oversold": 28,                   # 从30微调至28，更敏感
                "volume_multiplier": 2.0,
                "volatility_window": 20,
                "beta_window": 60,
                "roe_threshold": 0.08,                # 新增ROE阈值8%
                "revenue_growth_threshold": 0.10      # 新增营收增长阈值10%
            },
            "market_regime": {
                # 保持v3.0成功的市场环境乘数机制
                "bull_threshold": 0.02,
                "bear_threshold": -0.02,
                "volatility_high": 0.03,
                "bull_multiplier": 1.25,    # 牛市加成25%
                "bear_multiplier": 0.40,    # 熊市减成60%
                "neutral_multiplier": 1.00  # 中性市场不变
            },
            "scoring_optimization": {
                # 优化90+分股票识别，降低阈值5%促进更多高分股票
                "high_score_threshold_adjustment": -5,
                "score_smoothing": True,    # 启用评分平滑
                "continuous_functions": True  # 使用连续评分函数
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
        
    def detect_market_regime(self, date: str) -> Dict[str, Any]:
        """检测市场环境 - 保持v3.0成功的乘数机制"""
        try:
            # 获取大盘指数数据
            query = """
            SELECT trade_date, close, price_change_pct
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.code = '000001.SH'
            AND trade_date <= ?
            ORDER BY trade_date DESC
            LIMIT 30
            """
            
            with self.db_manager.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=[date])
            
            if df.empty:
                return {
                    "regime": "neutral", 
                    "volatility": "normal", 
                    "trend": "sideways",
                    "multiplier": 1.0
                }
                
            # 计算市场特征
            recent_returns = df['price_change_pct'].head(20).mean() / 100
            volatility = df['price_change_pct'].head(20).std() / 100
            
            # 判断市场环境和应用乘数
            regime = "neutral"
            multiplier = self.config["market_regime"]["neutral_multiplier"]
            
            if recent_returns > self.config["market_regime"]["bull_threshold"]:
                regime = "bull"
                multiplier = self.config["market_regime"]["bull_multiplier"]
            elif recent_returns < self.config["market_regime"]["bear_threshold"]:
                regime = "bear"
                multiplier = self.config["market_regime"]["bear_multiplier"]
            
            volatility_level = "high" if volatility > self.config["market_regime"]["volatility_high"] else "normal"
            
            return {
                "regime": regime,
                "volatility": volatility_level,
                "trend": "bullish" if recent_returns > 0 else "bearish",
                "multiplier": multiplier,
                "recent_return": recent_returns,
                "volatility_value": volatility
            }
            
        except Exception as e:
            self.logger.warning(f"市场环境检测失败: {e}")
            return {
                "regime": "neutral", 
                "volatility": "normal", 
                "trend": "sideways",
                "multiplier": 1.0
            }
    
    def calculate_technical_score(self, stock_data: pd.DataFrame) -> float:
        """计算技术指标得分 - 优化评分函数连续性"""
        if stock_data.empty:
            return 0.0
            
        scores = []
        latest = stock_data.iloc[-1]
        
        # 1. KDJ强度得分 - 优化连续函数
        kdj_k = latest.get('kdj_k', 50)
        kdj_d = latest.get('kdj_d', 50)
        kdj_j = latest.get('kdj_j', 50)
        
        # 处理缺失值
        for val_name, val in [('kdj_k', kdj_k), ('kdj_d', kdj_d), ('kdj_j', kdj_j)]:
            if pd.isna(val):
                if val_name == 'kdj_k':
                    kdj_k = 50
                elif val_name == 'kdj_d':
                    kdj_d = 50
                else:
                    kdj_j = 50
        
        # KDJ综合评分 - 使用更平滑的连续函数
        kdj_combined = (kdj_k + kdj_d) / 2
        
        # 平滑的KDJ评分函数
        if kdj_combined <= 20:
            kdj_score = 1.0
        elif kdj_combined <= 30:
            kdj_score = 0.9 + 0.1 * (30 - kdj_combined) / 10
        elif kdj_combined <= 50:
            kdj_score = 0.5 + 0.4 * (50 - kdj_combined) / 20
        elif kdj_combined <= 70:
            kdj_score = 0.2 + 0.3 * (70 - kdj_combined) / 20
        else:
            kdj_score = max(0.05, 0.2 * (100 - kdj_combined) / 30)
        
        # J值极值加成（连续调整）
        j_bonus = 0
        if kdj_j < 0:
            j_bonus = min(0.15, abs(kdj_j) / 100)
        elif kdj_j > 100:
            j_bonus = -min(0.1, (kdj_j - 100) / 200)
        
        kdj_score = max(0, min(1, kdj_score + j_bonus))
        scores.append(kdj_score * self.config["weights"]["technical"]["kdj_strength"])
        
        # 2. RSI动量得分 - 优化连续函数
        rsi = latest.get('rsi', 50)
        if pd.isna(rsi):
            rsi = 50
        
        # 更平滑的RSI评分函数
        if rsi <= 20:
            rsi_score = 1.0
        elif rsi <= 28:  # 使用新的更敏感阈值
            rsi_score = 0.9 + 0.1 * (28 - rsi) / 8
        elif rsi <= 40:
            rsi_score = 0.7 + 0.2 * (40 - rsi) / 12
        elif rsi <= 50:
            rsi_score = 0.4 + 0.3 * (50 - rsi) / 10
        elif rsi <= 60:
            rsi_score = 0.2 + 0.2 * (60 - rsi) / 10
        else:
            rsi_score = max(0.02, 0.2 * (100 - rsi) / 40)
        
        # RSI趋势调整
        if len(stock_data) >= 3:
            rsi_values = stock_data['rsi'].tail(3).dropna()
            if len(rsi_values) >= 2:
                rsi_trend = rsi_values.diff().mean()
                if not pd.isna(rsi_trend):
                    if rsi < 40 and rsi_trend > 0:
                        rsi_score *= 1.1
                    elif rsi > 60 and rsi_trend < 0:
                        rsi_score *= 0.9
                        
        scores.append(rsi_score * self.config["weights"]["technical"]["rsi_momentum"])
        
        # 3. BBI趋势得分 - 保持v3.0成功逻辑
        close_price = latest.get('close', 0)
        bbi = latest.get('bbi', close_price)
        
        if pd.isna(bbi) or bbi <= 0:
            if len(stock_data) >= 11:
                bbi = stock_data['close'].tail(11).mean()
            else:
                bbi = close_price
                
        if bbi > 0 and close_price > 0:
            price_bbi_ratio = close_price / bbi
            
            # 连续BBI评分函数
            if price_bbi_ratio <= 0.90:
                bbi_score = 1.0
            elif price_bbi_ratio <= 0.95:
                bbi_score = 0.9 + 0.1 * (0.95 - price_bbi_ratio) / 0.05
            elif price_bbi_ratio <= 0.98:
                bbi_score = 0.7 + 0.2 * (0.98 - price_bbi_ratio) / 0.03
            elif price_bbi_ratio <= 1.02:
                bbi_score = 0.4 + 0.3 * (1.02 - price_bbi_ratio) / 0.04
            else:
                bbi_score = max(0.05, 0.2 * (1.2 - min(price_bbi_ratio, 1.2)) / 0.18)
                
            scores.append(bbi_score * self.config["weights"]["technical"]["bbi_trend"])
        else:
            scores.append(0)
        
        # 4. 成交量异动得分 - 权重微调
        volume = latest.get('volume', 0)
        volume_mean = stock_data['volume'].tail(20).mean() if len(stock_data) >= 20 else volume
        
        if volume_mean > 0:
            volume_ratio = volume / volume_mean
            if volume_ratio >= self.config["parameters"]["volume_multiplier"]:
                volume_score = min(1.0, volume_ratio / 3.0)
            else:
                volume_score = max(0.1, volume_ratio / 2.0)
        else:
            volume_score = 0.1
            
        scores.append(volume_score * self.config["weights"]["technical"]["volume_surge"])
        
        return sum(scores)
    
    def calculate_fundamental_score(self, stock_data: pd.DataFrame, basic_data: Optional[pd.Series] = None) -> float:
        """计算基本面得分 - 新增ROE和营收增长指标"""
        scores = []
        
        if basic_data is None:
            # 默认中性得分
            return sum([
                0.5 * self.config["weights"]["fundamental"]["pe_valuation"],
                0.5 * self.config["weights"]["fundamental"]["pb_valuation"], 
                0.5 * self.config["weights"]["fundamental"]["roe_profitability"],
                0.5 * self.config["weights"]["fundamental"]["revenue_growth"],
                0.5 * self.config["weights"]["fundamental"]["market_cap"],
                0.5 * self.config["weights"]["fundamental"]["turnover_activity"]
            ])
        
        # PE估值得分
        pe = basic_data.get('pe_ttm', 25)
        if pd.isna(pe) or pe <= 0:
            pe_score = 0.5
        elif pe <= 15:
            pe_score = 1.0
        elif pe <= 25:
            pe_score = 0.8 + 0.2 * (25 - pe) / 10
        elif pe <= 50:
            pe_score = 0.3 + 0.5 * (50 - pe) / 25
        else:
            pe_score = max(0.1, 0.3 * (100 - min(pe, 100)) / 50)
        
        scores.append(pe_score * self.config["weights"]["fundamental"]["pe_valuation"])
        
        # PB估值得分
        pb = basic_data.get('pb', 2.0)
        if pd.isna(pb) or pb <= 0:
            pb_score = 0.5
        elif pb <= 1.0:
            pb_score = 1.0
        elif pb <= 2.0:
            pb_score = 0.8 + 0.2 * (2.0 - pb) / 1.0
        elif pb <= 5.0:
            pb_score = 0.2 + 0.6 * (5.0 - pb) / 3.0
        else:
            pb_score = max(0.05, 0.2 * (10 - min(pb, 10)) / 5)
        
        scores.append(pb_score * self.config["weights"]["fundamental"]["pb_valuation"])
        
        # 新增：ROE盈利能力得分
        roe = basic_data.get('roe', 0.08) if basic_data is not None else 0.08  # 默认8%
        if pd.isna(roe) or roe is None:
            roe_score = 0.5
        elif roe >= 0.20:  # ROE >= 20%
            roe_score = 1.0
        elif roe >= 0.15:  # ROE >= 15%
            roe_score = 0.8 + 0.2 * (roe - 0.15) / 0.05
        elif roe >= 0.08:  # ROE >= 8%
            roe_score = 0.5 + 0.3 * (roe - 0.08) / 0.07
        elif roe >= 0.03:  # ROE >= 3%
            roe_score = 0.2 + 0.3 * (roe - 0.03) / 0.05
        else:
            roe_score = max(0.05, 0.2 * max(roe, 0) / 0.03)
        
        scores.append(roe_score * self.config["weights"]["fundamental"]["roe_profitability"])
        
        # 新增：营收增长得分（从数据库获取真实数据）
        revenue_growth_pct = basic_data.get('revenue_growth', 10.0) if basic_data is not None else 10.0  # 获取真实营收增长(%)
        if revenue_growth_pct is None or pd.isna(revenue_growth_pct):
            revenue_growth_pct = 10.0
        revenue_growth = revenue_growth_pct / 100.0  # 转换为小数形式
        if revenue_growth >= 0.30:  # 增长 >= 30%
            growth_score = 1.0
        elif revenue_growth >= 0.20:  # 增长 >= 20%
            growth_score = 0.8 + 0.2 * (revenue_growth - 0.20) / 0.10
        elif revenue_growth >= 0.10:  # 增长 >= 10%
            growth_score = 0.5 + 0.3 * (revenue_growth - 0.10) / 0.10
        elif revenue_growth >= 0:  # 正增长
            growth_score = 0.3 + 0.2 * revenue_growth / 0.10
        else:
            growth_score = max(0.05, 0.3 * (1 + revenue_growth))  # 负增长惩罚
        
        scores.append(growth_score * self.config["weights"]["fundamental"]["revenue_growth"])
        
        # 市值因子得分
        market_cap = basic_data.get('total_mv', 100)  # 亿元
        if pd.isna(market_cap) or market_cap <= 0:
            cap_score = 0.5
        elif market_cap <= 50:  # 小盘股偏好
            cap_score = 1.0
        elif market_cap <= 200:
            cap_score = 0.7 + 0.3 * (200 - market_cap) / 150
        elif market_cap <= 1000:
            cap_score = 0.3 + 0.4 * (1000 - market_cap) / 800
        else:
            cap_score = max(0.1, 0.3 * (5000 - min(market_cap, 5000)) / 4000)
        
        scores.append(cap_score * self.config["weights"]["fundamental"]["market_cap"])
        
        # 换手率活跃度
        turnover = basic_data.get('turnover_rate', 3.0)
        if pd.isna(turnover) or turnover <= 0:
            turnover_score = 0.3
        elif turnover <= 1.0:  # 换手率过低
            turnover_score = 0.2
        elif turnover <= 3.0:
            turnover_score = 0.5 + 0.3 * (turnover - 1.0) / 2.0
        elif turnover <= 8.0:
            turnover_score = 0.8 + 0.2 * (turnover - 3.0) / 5.0
        elif turnover <= 15.0:
            turnover_score = 0.6 + 0.2 * (15.0 - turnover) / 7.0
        else:  # 换手率过高
            turnover_score = max(0.2, 0.6 * (30 - min(turnover, 30)) / 15)
        
        scores.append(turnover_score * self.config["weights"]["fundamental"]["turnover_activity"])
        
        return sum(scores)
    
    def calculate_performance_score(self, stock_data: pd.DataFrame) -> float:
        """计算市场表现得分"""
        if stock_data.empty:
            return 0.0
            
        scores = []
        
        # 价格动量得分（优化阈值提高区分度）
        if len(stock_data) >= 5:
            recent_returns = stock_data['price_change_pct'].tail(5).mean() / 100
            if recent_returns > 0.03:  # > 3% (降低高分阈值)
                momentum_score = 1.0
            elif recent_returns > 0.01:  # > 1%
                momentum_score = 0.7 + 0.3 * (recent_returns - 0.01) / 0.02
            elif recent_returns > -0.01:  # -1% to 1%
                momentum_score = 0.4 + 0.3 * (recent_returns + 0.01) / 0.02
            elif recent_returns > -0.03:  # -1% to -3%
                momentum_score = 0.2 + 0.2 * (recent_returns + 0.03) / 0.02
            else:
                momentum_score = max(0.05, 0.2 * (recent_returns + 0.05) / 0.02)
        else:
            momentum_score = 0.3
            
        scores.append(momentum_score * self.config["weights"]["performance"]["price_momentum"])
        
        # 相对强度得分（优化阈值提高区分度）
        latest_change = stock_data['price_change_pct'].iloc[-1] / 100 if not stock_data.empty else 0
        if latest_change > 0.02:  # > 2% (降低高分阈值)
            relative_score = 1.0
        elif latest_change > 0.005:  # > 0.5%
            relative_score = 0.7 + 0.3 * (latest_change - 0.005) / 0.015
        elif latest_change > -0.005:  # -0.5% to 0.5%
            relative_score = 0.4 + 0.3 * (latest_change + 0.005) / 0.01
        elif latest_change > -0.02:  # -0.5% to -2%
            relative_score = 0.2 + 0.2 * (latest_change + 0.02) / 0.015
        else:
            relative_score = max(0.05, 0.2 * (latest_change + 0.05) / 0.03)
            
        scores.append(relative_score * self.config["weights"]["performance"]["relative_strength"])
        
        # 波动率风险得分
        if len(stock_data) >= 20:
            volatility = stock_data['price_change_pct'].tail(20).std() / 100
            if volatility <= 0.02:  # 低波动
                vol_score = 1.0
            elif volatility <= 0.05:
                vol_score = 0.7 + 0.3 * (0.05 - volatility) / 0.03
            elif volatility <= 0.08:
                vol_score = 0.4 + 0.3 * (0.08 - volatility) / 0.03
            else:
                vol_score = max(0.1, 0.4 * (0.15 - min(volatility, 0.15)) / 0.07)
        else:
            vol_score = 0.5
            
        scores.append(vol_score * self.config["weights"]["performance"]["volatility_risk"])
        
        return sum(scores)
    
    def calculate_market_regime_score(self, stock_data: pd.DataFrame, market_info: Dict) -> float:
        """计算市场环境得分"""
        scores = []
        
        # 市场贝塔得分（简化）
        beta_score = 0.5  # 默认中性
        scores.append(beta_score * self.config["weights"]["market_regime"]["market_beta"])
        
        # 板块轮动得分（简化）
        rotation_score = 0.5 if market_info["regime"] == "neutral" else 0.7
        scores.append(rotation_score * self.config["weights"]["market_regime"]["sector_rotation"])
        
        # 流动性因子得分（简化）
        liquidity_score = 0.6 if market_info["volatility"] == "normal" else 0.4
        scores.append(liquidity_score * self.config["weights"]["market_regime"]["liquidity"])
        
        return sum(scores)
    
    def calculate_quantitative_score(self, stock_code: str, date: str, 
                                   stock_data: pd.DataFrame = None, 
                                   basic_data: pd.Series = None) -> Dict[str, Any]:
        """计算量化评分"""
        try:
            # 如果没有提供数据，从数据库获取
            if stock_data is None:
                stock_data = self._get_stock_data(stock_code, date)
            if basic_data is None:
                basic_data = self._get_basic_data(stock_code, date)
            
            if stock_data.empty:
                return {
                    "stock_code": stock_code,
                    "date": date,
                    "quantitative_score": 0.0,
                    "version": self.version,
                    "error": "无股票数据"
                }
            
            # 检测市场环境
            market_info = self.detect_market_regime(date)
            
            # 计算各维度得分
            technical_score = self.calculate_technical_score(stock_data)
            fundamental_score = self.calculate_fundamental_score(stock_data, basic_data)
            performance_score = self.calculate_performance_score(stock_data)
            market_regime_score = self.calculate_market_regime_score(stock_data, market_info)
            
            # 基础得分
            base_score = (technical_score + fundamental_score + 
                         performance_score + market_regime_score)
            
            # 应用市场环境乘数（保持v3.0成功机制）
            final_score = base_score * market_info["multiplier"]
            
            # 应用评分优化
            if self.config.get("scoring_optimization", {}).get("high_score_threshold_adjustment"):
                adjustment = self.config["scoring_optimization"]["high_score_threshold_adjustment"]
                if final_score >= 0.85:  # 对高分进行调整
                    final_score = final_score * (1 + adjustment / 100)
            
            # 标准化到0-100分制
            # 原始分数通常在20-80范围内，标准化映射到0-100
            raw_score = final_score * 100
            quantitative_score = max(0, min(100, (raw_score - 20) * 100 / 60))
            
            return {
                "stock_code": stock_code,
                "date": date,
                "quantitative_score": round(quantitative_score, 1),
                "technical_score": round(max(0, min(100, (technical_score * 100 - 12) * 100 / 50)), 1),  # 技术评分标准化
                "fundamental_score": round(max(0, min(100, (fundamental_score * 100 - 5) * 100 / 15)), 1),  # 基本面标准化  
                "performance_score": round(max(0, min(100, (performance_score * 100 - 8) * 100 / 8)), 1),   # 表现标准化
                "market_regime_score": round(max(0, min(100, market_regime_score * 100 / 3 * 100)), 1),     # 市场环境标准化
                "market_regime": market_info["regime"],
                "market_multiplier": market_info["multiplier"],
                "version": self.version
            }
            
        except Exception as e:
            self.logger.error(f"计算量化评分失败 {stock_code} {date}: {e}")
            return {
                "stock_code": stock_code,
                "date": date,
                "quantitative_score": 0.0,
                "version": self.version,
                "error": str(e)
            }
    
    def _get_stock_data(self, stock_code: str, date: str, days: int = 60) -> pd.DataFrame:
        """获取股票数据"""
        try:
            query = """
            SELECT 
                dq.trade_date,
                dq.open, dq.high, dq.low, dq.close,
                dq.volume, dq.price_change_pct,
                ti.kdj_k, ti.kdj_d, ti.kdj_j,
                ti.rsi12 as rsi, ti.bbi
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            LEFT JOIN technical_indicators ti ON ti.security_id = s.id 
                AND ti.trade_date = dq.trade_date
            WHERE s.code = ? AND dq.trade_date <= ?
            ORDER BY dq.trade_date DESC
            LIMIT ?
            """
            
            with self.db_manager.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=[stock_code, date, days])
                
            return df.sort_values('trade_date')
            
        except Exception as e:
            self.logger.error(f"获取股票数据失败 {stock_code}: {e}")
            return pd.DataFrame()
    
    def _get_basic_data(self, stock_code: str, date: str) -> Optional[pd.Series]:
        """获取基本面数据 - 包含ROE和营收增长"""
        try:
            query = """
            SELECT 
                db.pe_ttm, db.pb, db.total_mv, db.turnover_rate,
                fi.roe, fi.or_yoy as revenue_growth
            FROM daily_basic db
            JOIN securities s ON db.security_id = s.id
            LEFT JOIN financial_indicator fi ON fi.security_id = s.id 
                AND fi.end_date = (
                    SELECT MAX(fi2.end_date) 
                    FROM financial_indicator fi2 
                    WHERE fi2.security_id = s.id AND fi2.end_date <= ?
                )
            WHERE s.code = ? AND db.trade_date <= ?
            ORDER BY db.trade_date DESC
            LIMIT 1
            """
            
            with self.db_manager.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=[date, stock_code, date])
                
            return df.iloc[0] if not df.empty else None
            
        except Exception as e:
            self.logger.warning(f"获取基本面数据失败 {stock_code}: {e}")
            return None

    def batch_calculate_scores(self, stock_codes: List[str], date: str) -> List[Dict[str, Any]]:
        """批量计算评分"""
        results = []
        
        for stock_code in stock_codes:
            result = self.calculate_quantitative_score(stock_code, date)
            results.append(result)
            
        return results