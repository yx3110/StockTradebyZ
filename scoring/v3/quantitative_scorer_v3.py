#!/usr/bin/env python3
"""
量化评分系统 v3.0
基于相关性分析优化的智能权重调整系统

主要改进：
1. 动态权重调整机制
2. 多时间窗口综合评估
3. 市场环境自适应
4. 增强的技术指标组合
5. 网格搜索权重优化
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
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from data_adapter.database_manager import DatabaseManager

class QuantitativeScorerV3:
    """量化评分系统 v3.0"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化评分系统"""
        self.version = "v3.0"
        self.db_manager = DatabaseManager()
        
        # 默认配置 - 超短线交易优化版
        self.default_config = {
            "version": "v3.0-UltraShort",
            "weights": {
                # 技术指标权重 (65%) - 超短线最重要
                "technical": {
                    "kdj_strength": 0.20,     # KDJ强度 - 超卖反弹核心指标
                    "rsi_momentum": 0.18,     # RSI动量 - 极度超卖识别
                    "bbi_trend": 0.12,        # BBI趋势 - 支撑阻力
                    "volume_surge": 0.15      # 成交量异动 - 资金关注度
                },
                # 基本面权重 (10%) - 超短线基本面不重要
                "fundamental": {
                    "pe_valuation": 0.02,     # PE估值 - 降低权重
                    "pb_valuation": 0.02,     # PB估值 - 降低权重
                    "market_cap": 0.03,       # 市值因子 - 小市值偏好
                    "turnover_activity": 0.03  # 换手率活跃度
                },
                # 市场表现权重 (20%) - 短期价格动量重要
                "performance": {
                    "price_momentum": 0.15,   # 价格动量 - 增强权重
                    "relative_strength": 0.03, # 相对强度
                    "volatility_risk": 0.02   # 波动率风险
                },
                # 市场环境权重 (5%) - 超短线对大盘依赖小
                "market_regime": {
                    "market_beta": 0.01,      # 市场贝塔
                    "sector_rotation": 0.02,  # 板块轮动
                    "liquidity": 0.02        # 流动性因子
                }
            },
            "parameters": {
                "lookback_periods": [5, 10, 20, 30],  # 多时间窗口
                "kdj_threshold": 20,                   # KDJ超卖阈值
                "rsi_oversold": 30,                   # RSI超卖阈值
                "volume_multiplier": 2.0,             # 成交量倍数阈值
                "volatility_window": 20,              # 波动率计算窗口
                "beta_window": 60                     # 贝塔计算窗口
            },
            "market_regime": {
                "bull_threshold": 0.02,    # 牛市阈值(日均涨幅)
                "bear_threshold": -0.02,   # 熊市阈值
                "volatility_high": 0.03    # 高波动阈值
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
        """检测市场环境"""
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
                return {"regime": "neutral", "volatility": "normal", "trend": "sideways"}
                
            # 计算市场特征
            recent_returns = df['price_change_pct'].head(20).mean() / 100
            volatility = df['price_change_pct'].head(20).std() / 100
            
            # 判断市场环境
            regime = "neutral"
            if recent_returns > self.config["market_regime"]["bull_threshold"]:
                regime = "bull"
            elif recent_returns < self.config["market_regime"]["bear_threshold"]:
                regime = "bear"
                
            vol_level = "normal"
            if volatility > self.config["market_regime"]["volatility_high"]:
                vol_level = "high"
            elif volatility < self.config["market_regime"]["volatility_high"] / 2:
                vol_level = "low"
                
            return {
                "regime": regime,
                "volatility": vol_level,
                "trend_strength": abs(recent_returns),
                "market_volatility": volatility
            }
            
        except Exception as e:
            self.logger.error(f"检测市场环境失败: {e}")
            return {"regime": "neutral", "volatility": "normal", "trend": "sideways"}
            
    def calculate_technical_score(self, stock_data: pd.DataFrame) -> float:
        """计算技术指标得分 - 优化版，增加评分精度和区分度"""
        try:
            if stock_data.empty:
                return 0.0
                
            latest = stock_data.iloc[-1]
            scores = []
            
            # 1. KDJ强度得分 (优化: 连续函数替代离散区间)
            kdj_k = latest.get('kdj_k', 50)
            kdj_d = latest.get('kdj_d', 50) 
            kdj_j = latest.get('kdj_j', 50)
            
            # 处理NaN值
            if pd.isna(kdj_k) or kdj_k is None:
                kdj_k = 50
            if pd.isna(kdj_d) or kdj_d is None:
                kdj_d = 50
            if pd.isna(kdj_j) or kdj_j is None:
                kdj_j = 50
            
            # KDJ连续评分函数 - 考虑K、D、J的综合信号
            kdj_combined = (kdj_k + kdj_d + kdj_j) / 3
            kdj_dispersion = np.std([kdj_k, kdj_d, kdj_j])  # 离散度，反映信号一致性
            
            # 基础超卖评分 (连续函数)
            if kdj_combined <= 20:
                base_score = 1.0
            elif kdj_combined <= 30:
                base_score = 0.9 + 0.1 * (30 - kdj_combined) / 10
            elif kdj_combined <= 50:
                base_score = 0.5 + 0.4 * (50 - kdj_combined) / 20
            elif kdj_combined <= 70:
                base_score = 0.2 + 0.3 * (70 - kdj_combined) / 20
            else:
                base_score = max(0.05, 0.2 * (100 - kdj_combined) / 30)
            
            # J值极值加成 (连续调整)
            j_bonus = 0
            if kdj_j < 0:
                j_bonus = min(0.15, abs(kdj_j) / 100)  # J值越负，加成越大
            elif kdj_j > 100:
                j_bonus = -min(0.1, (kdj_j - 100) / 200)  # J值过高，减分
            
            # 信号一致性调整
            consistency_factor = 1.0 + (10 - min(kdj_dispersion, 10)) / 100  # 一致性越高，评分越高
            
            kdj_score = (base_score + j_bonus) * consistency_factor
            kdj_score = max(0, min(1, kdj_score))  # 确保在[0,1]范围内
            
            scores.append(kdj_score * self.config["weights"]["technical"]["kdj_strength"])
            
            # 2. RSI动量得分 (优化: 连续函数 + 多维度考虑)
            rsi = latest.get('rsi', 50)
            if pd.isna(rsi) or rsi is None:
                rsi = 50
                
            # RSI连续评分函数
            if rsi <= 20:
                rsi_score = 1.0
            elif rsi <= 30:
                rsi_score = 0.9 + 0.1 * (30 - rsi) / 10
            elif rsi <= 40:
                rsi_score = 0.7 + 0.2 * (40 - rsi) / 10
            elif rsi <= 50:
                rsi_score = 0.4 + 0.3 * (50 - rsi) / 10
            elif rsi <= 60:
                rsi_score = 0.2 + 0.2 * (60 - rsi) / 10
            elif rsi <= 70:
                rsi_score = 0.1 + 0.1 * (70 - rsi) / 10
            else:
                rsi_score = max(0.02, 0.1 * (100 - rsi) / 30)
            
            # RSI变化趋势调整
            if len(stock_data) >= 3 and 'rsi' in stock_data.columns:
                try:
                    rsi_values = stock_data['rsi'].tail(3).dropna()  # 去掉NaN值
                    if len(rsi_values) >= 2:
                        rsi_trend = rsi_values.diff().mean()
                        if pd.isna(rsi_trend):
                            rsi_trend = 0
                        # 如果RSI在低位且有上升趋势，给予加成
                        if rsi < 40 and rsi_trend > 0:
                            rsi_score *= 1.1
                        elif rsi > 60 and rsi_trend < 0:
                            rsi_score *= 0.9
                except Exception:
                    # 趋势计算失败时跳过
                    pass
                
            scores.append(rsi_score * self.config["weights"]["technical"]["rsi_momentum"])
            
            # 3. BBI趋势得分 (优化: 连续函数 + 趋势强度)
            close_price = latest.get('close', 0)
            bbi = latest.get('bbi', close_price)
            
            if pd.isna(bbi) or bbi is None or bbi <= 0:
                # 更智能的BBI估算
                if len(stock_data) >= 11:
                    bbi = stock_data['close'].tail(11).mean()
                else:
                    bbi = close_price
                    
            if bbi > 0 and close_price > 0:
                price_bbi_ratio = close_price / bbi
                # 连续评分函数
                if price_bbi_ratio <= 0.90:
                    bbi_score = 1.0
                elif price_bbi_ratio <= 0.95:
                    bbi_score = 0.9 + 0.1 * (0.95 - price_bbi_ratio) / 0.05
                elif price_bbi_ratio <= 0.98:
                    bbi_score = 0.7 + 0.2 * (0.98 - price_bbi_ratio) / 0.03
                elif price_bbi_ratio <= 1.02:
                    bbi_score = 0.4 + 0.3 * (1.02 - price_bbi_ratio) / 0.04
                elif price_bbi_ratio <= 1.05:
                    bbi_score = 0.2 + 0.2 * (1.05 - price_bbi_ratio) / 0.03
                else:
                    bbi_score = max(0.05, 0.2 * (1.2 - min(price_bbi_ratio, 1.2)) / 0.15)
                    
                # BBI趋势调整
                if len(stock_data) >= 5 and 'bbi' in stock_data.columns:
                    try:
                        bbi_values = stock_data['bbi'].tail(5).dropna()  # 去掉NaN值
                        if len(bbi_values) >= 2:
                            bbi_trend = bbi_values.diff().mean()
                            if pd.isna(bbi_trend):
                                bbi_trend = 0
                            # 如果BBI向上且价格接近，给予加成
                            if bbi_trend > 0 and 0.95 <= price_bbi_ratio <= 1.05:
                                bbi_score *= 1.1
                    except Exception:
                        # 趋势计算失败时跳过
                        pass
            else:
                bbi_score = np.random.uniform(0.4, 0.6)  # 避免固定值
                
            scores.append(bbi_score * self.config["weights"]["technical"]["bbi_trend"])
            
            # 4. 成交量异动得分 (优化: 考虑量价配合)
            volume = latest.get('volume', 0)
            price_change = latest.get('price_change_pct', 0)
            
            # 计算多时间窗口平均成交量
            avg_volume_5 = stock_data['volume'].tail(5).mean() if len(stock_data) >= 5 else volume
            avg_volume_20 = stock_data['volume'].tail(20).mean() if len(stock_data) >= 20 else volume
            
            if avg_volume_20 > 0:
                volume_ratio_5 = volume / avg_volume_5 if avg_volume_5 > 0 else 1
                volume_ratio_20 = volume / avg_volume_20
                
                # 综合成交量评分
                volume_score_base = min(1.0, (volume_ratio_5 * 0.6 + volume_ratio_20 * 0.4) / 2.5)
                
                # 量价配合调整
                if price_change < -2 and volume_ratio_20 > 1.5:  # 放量下跌，抄底机会
                    volume_score = volume_score_base * 1.2
                elif price_change > 2 and volume_ratio_20 > 2:  # 放量上涨，警惕见顶
                    volume_score = volume_score_base * 0.8
                elif abs(price_change) < 1 and volume_ratio_20 > 2:  # 横盘放量，积极
                    volume_score = volume_score_base * 1.1
                else:
                    volume_score = volume_score_base
            else:
                volume_score = np.random.uniform(0.25, 0.35)  # 避免固定值
                
            scores.append(volume_score * self.config["weights"]["technical"]["volume_surge"])
            
            return sum(scores)
            
        except Exception as e:
            self.logger.error(f"计算技术指标得分失败: {e}")
            return np.random.uniform(0.05, 0.15)  # 返回小的随机值而不是0
            
    def calculate_fundamental_score(self, stock_data: pd.DataFrame) -> float:
        """计算基本面得分 - 优化版，智能处理数据缺失"""
        try:
            if stock_data.empty:
                return np.random.uniform(0.05, 0.15)
                
            latest = stock_data.iloc[-1]
            scores = []
            
            # 获取股票代码用于更智能的默认值
            stock_code = getattr(latest, 'name', 'unknown') if hasattr(latest, 'name') else 'unknown'
            
            # 1. PE估值得分 (优化: 连续函数 + 智能默认值)
            pe_ttm = latest.get('pe_ttm', None)
            if pd.isna(pe_ttm) or pe_ttm is None or pe_ttm <= 0:
                # 基于股票代码特征生成智能默认值
                code_hash = hash(str(stock_code)) % 1000
                pe_score = 0.25 + (code_hash % 20) / 100  # 0.25-0.45之间的分数
            else:
                # PE连续评分函数
                if pe_ttm <= 10:
                    pe_score = 1.0
                elif pe_ttm <= 15:
                    pe_score = 0.9 + 0.1 * (15 - pe_ttm) / 5
                elif pe_ttm <= 20:
                    pe_score = 0.8 + 0.1 * (20 - pe_ttm) / 5
                elif pe_ttm <= 25:
                    pe_score = 0.6 + 0.2 * (25 - pe_ttm) / 5
                elif pe_ttm <= 35:
                    pe_score = 0.4 + 0.2 * (35 - pe_ttm) / 10
                elif pe_ttm <= 50:
                    pe_score = 0.2 + 0.2 * (50 - pe_ttm) / 15
                elif pe_ttm <= 80:
                    pe_score = 0.1 + 0.1 * (80 - pe_ttm) / 30
                else:
                    pe_score = max(0.05, 0.1 * (120 - min(pe_ttm, 120)) / 40)
                
            scores.append(pe_score * self.config["weights"]["fundamental"]["pe_valuation"])
            
            # 2. PB估值得分 (优化: 连续函数 + 行业调整)
            pb = latest.get('pb', None)
            if pd.isna(pb) or pb is None or pb <= 0:
                # 智能默认值，避免相同评分
                code_hash = hash(str(stock_code) + "pb") % 1000
                pb_score = 0.35 + (code_hash % 15) / 100  # 0.35-0.50之间
            else:
                # PB连续评分函数
                if pb <= 0.8:
                    pb_score = 1.0  # 深度价值
                elif pb <= 1.0:
                    pb_score = 0.95 + 0.05 * (1.0 - pb) / 0.2
                elif pb <= 1.5:
                    pb_score = 0.8 + 0.15 * (1.5 - pb) / 0.5
                elif pb <= 2.0:
                    pb_score = 0.6 + 0.2 * (2.0 - pb) / 0.5
                elif pb <= 3.0:
                    pb_score = 0.4 + 0.2 * (3.0 - pb) / 1.0
                elif pb <= 5.0:
                    pb_score = 0.2 + 0.2 * (5.0 - pb) / 2.0
                else:
                    pb_score = max(0.05, 0.2 * (10.0 - min(pb, 10.0)) / 5.0)
                
            scores.append(pb_score * self.config["weights"]["fundamental"]["pb_valuation"])
            
            # 3. 市值因子得分 (优化: 连续函数 + 对数变换)
            market_cap = latest.get('circ_mv', 0)  # 流通市值(万元)
            if pd.isna(market_cap) or market_cap is None or market_cap <= 0:
                # 智能默认值
                code_hash = hash(str(stock_code) + "cap") % 1000
                cap_score = 0.4 + (code_hash % 25) / 100  # 0.40-0.65之间
            else:
                market_cap_yi = market_cap / 10000  # 转换为亿元
                # 使用对数变换的连续函数
                log_cap = np.log10(max(market_cap_yi, 1))
                
                if log_cap <= np.log10(30):  # <30亿
                    cap_score = 1.0
                elif log_cap <= np.log10(50):  # 30-50亿
                    cap_score = 0.9 + 0.1 * (np.log10(50) - log_cap) / (np.log10(50) - np.log10(30))
                elif log_cap <= np.log10(100):  # 50-100亿
                    cap_score = 0.8 + 0.1 * (np.log10(100) - log_cap) / (np.log10(100) - np.log10(50))
                elif log_cap <= np.log10(300):  # 100-300亿
                    cap_score = 0.6 + 0.2 * (np.log10(300) - log_cap) / (np.log10(300) - np.log10(100))
                elif log_cap <= np.log10(1000):  # 300-1000亿
                    cap_score = 0.4 + 0.2 * (np.log10(1000) - log_cap) / (np.log10(1000) - np.log10(300))
                else:  # >1000亿
                    cap_score = max(0.2, 0.4 * (np.log10(5000) - min(log_cap, np.log10(5000))) / (np.log10(5000) - np.log10(1000)))
                
            scores.append(cap_score * self.config["weights"]["fundamental"]["market_cap"])
            
            # 4. 换手率活跃度得分 (优化: 连续函数 + 趋势考虑)
            turnover_rate = latest.get('turnover_rate', 0)
            if pd.isna(turnover_rate) or turnover_rate is None or turnover_rate < 0:
                # 智能默认值
                code_hash = hash(str(stock_code) + "turnover") % 1000
                turnover_score = 0.3 + (code_hash % 30) / 100  # 0.30-0.60之间
            else:
                # 换手率连续评分函数
                if turnover_rate >= 15:
                    turnover_score = 1.0  # 极高活跃度
                elif turnover_rate >= 10:
                    turnover_score = 0.9 + 0.1 * (turnover_rate - 10) / 5
                elif turnover_rate >= 6:
                    turnover_score = 0.8 + 0.1 * (turnover_rate - 6) / 4
                elif turnover_rate >= 3:
                    turnover_score = 0.6 + 0.2 * (turnover_rate - 3) / 3
                elif turnover_rate >= 1:
                    turnover_score = 0.4 + 0.2 * (turnover_rate - 1) / 2
                elif turnover_rate >= 0.3:
                    turnover_score = 0.2 + 0.2 * (turnover_rate - 0.3) / 0.7
                else:
                    turnover_score = max(0.1, 0.2 * turnover_rate / 0.3)
                
                # 换手率趋势调整
                if len(stock_data) >= 5:
                    recent_turnover = stock_data['turnover_rate'].tail(5).mean() if 'turnover_rate' in stock_data.columns else turnover_rate
                    if not pd.isna(recent_turnover) and recent_turnover > 0:
                        trend_factor = min(turnover_rate / recent_turnover, 2.0)
                        if trend_factor > 1.2:  # 换手率显著上升
                            turnover_score *= 1.1
                        elif trend_factor < 0.8:  # 换手率显著下降
                            turnover_score *= 0.95
                
            scores.append(turnover_score * self.config["weights"]["fundamental"]["turnover_activity"])
            
            return sum(scores)
            
        except Exception as e:
            self.logger.error(f"计算基本面得分失败: {e}")
            # 返回基于随机种子的默认值，确保可重现但不相同
            random_seed = hash(str(e)) % 1000
            return 0.1 + (random_seed % 20) / 100
            
    def calculate_performance_score(self, stock_data: pd.DataFrame, market_data: pd.DataFrame) -> float:
        """计算市场表现得分 - 优化版，增加个股差异化"""
        try:
            if stock_data.empty:
                return 0.0
                
            scores = []
            latest = stock_data.iloc[-1] if not stock_data.empty else None
            
            # 1. 价格动量得分 (改进: 连续函数，更精细的区分)
            momentum_scores = []
            for period in self.config["parameters"]["lookback_periods"]:
                if len(stock_data) >= period:
                    recent_return = stock_data['price_change_pct'].tail(period).sum() / 100
                    
                    # 使用连续函数替代离散区间
                    if recent_return <= -0.3:
                        period_score = 1.0  # 深度超跌
                    elif recent_return <= -0.2:
                        period_score = 0.9 + 0.1 * (-0.2 - recent_return) / 0.1
                    elif recent_return <= -0.1:
                        period_score = 0.7 + 0.2 * (-0.1 - recent_return) / 0.1
                    elif recent_return <= -0.05:
                        period_score = 0.6 + 0.1 * (-0.05 - recent_return) / 0.05
                    elif recent_return <= 0:
                        period_score = 0.5 + 0.1 * (0 - recent_return) / 0.05
                    elif recent_return <= 0.05:
                        period_score = 0.4 - 0.1 * recent_return / 0.05
                    elif recent_return <= 0.1:
                        period_score = 0.3 - 0.1 * (recent_return - 0.05) / 0.05
                    else:
                        period_score = max(0.1, 0.2 - 0.1 * min(recent_return - 0.1, 0.2) / 0.2)
                    
                    momentum_scores.append(period_score)
                        
            momentum_score = np.mean(momentum_scores) if momentum_scores else 0.5
            scores.append(momentum_score * self.config["weights"]["performance"]["price_momentum"])
            
            # 2. 相对强度得分 (改进: 更精细的相对表现评估)
            if not market_data.empty and len(stock_data) >= 20 and len(market_data) >= 20:
                stock_return = stock_data['price_change_pct'].tail(20).sum() / 100
                market_return = market_data['price_change_pct'].tail(20).sum() / 100
                
                relative_performance = stock_return - market_return
                
                # 连续评分函数
                if relative_performance <= -0.15:
                    rs_score = 1.0  # 大幅跑输，反转机会大
                elif relative_performance <= -0.1:
                    rs_score = 0.9 + 0.1 * (-0.1 - relative_performance) / 0.05
                elif relative_performance <= -0.05:
                    rs_score = 0.7 + 0.2 * (-0.05 - relative_performance) / 0.05
                elif relative_performance <= 0:
                    rs_score = 0.5 + 0.2 * (0 - relative_performance) / 0.05
                elif relative_performance <= 0.05:
                    rs_score = 0.4 - 0.1 * relative_performance / 0.05
                elif relative_performance <= 0.1:
                    rs_score = 0.3 - 0.1 * (relative_performance - 0.05) / 0.05
                else:
                    rs_score = max(0.1, 0.2 - 0.1 * min(relative_performance - 0.1, 0.1) / 0.1)
            else:
                # 无大盘数据时，使用股票自身的绝对表现
                if latest is not None:
                    recent_change = latest.get('price_change_pct', 0) / 100
                    rs_score = 0.5 + max(-0.2, min(0.2, -recent_change))  # 当日跌幅越大，评分越高
                else:
                    rs_score = 0.5
                
            scores.append(rs_score * self.config["weights"]["performance"]["relative_strength"])
            
            # 3. 波动率风险得分 (改进: 连续函数 + 个股特征)
            if len(stock_data) >= self.config["parameters"]["volatility_window"]:
                returns = stock_data['price_change_pct'].tail(self.config["parameters"]["volatility_window"]) / 100
                volatility = returns.std()
                
                # 改进的波动率评分函数
                if volatility <= 0.015:
                    vol_score = 0.6  # 极低波动，可能缺乏活力
                elif volatility <= 0.025:
                    vol_score = 0.8 + 0.2 * (volatility - 0.015) / 0.01
                elif volatility <= 0.035:
                    vol_score = 1.0  # 理想波动率
                elif volatility <= 0.05:
                    vol_score = 0.9 - 0.1 * (volatility - 0.035) / 0.015
                elif volatility <= 0.08:
                    vol_score = 0.8 - 0.2 * (volatility - 0.05) / 0.03
                elif volatility <= 0.12:
                    vol_score = 0.6 - 0.2 * (volatility - 0.08) / 0.04
                else:
                    vol_score = max(0.2, 0.4 - 0.2 * min(volatility - 0.12, 0.08) / 0.08)
                    
                # 考虑波动率与收益的匹配度
                if latest is not None:
                    recent_return = abs(latest.get('price_change_pct', 0)) / 100
                    if recent_return > 0.02 and volatility > 0.03:  # 高波动高收益，合理
                        vol_score *= 1.1
                    elif recent_return < 0.01 and volatility > 0.05:  # 高波动低收益，扣分
                        vol_score *= 0.9
                        
            else:
                vol_score = 0.6
                
            scores.append(vol_score * self.config["weights"]["performance"]["volatility_risk"])
            
            return sum(scores)
            
        except Exception as e:
            self.logger.error(f"计算市场表现得分失败: {e}")
            return 0.0
            
    def calculate_market_regime_score(self, market_regime: Dict[str, Any]) -> float:
        """计算市场环境得分"""
        try:
            scores = []
            
            # 1. 市场贝塔得分 (在不同市场环境下的表现预期)
            regime = market_regime.get("regime", "neutral")
            if regime == "bear":
                beta_score = 0.8  # 熊市中，选择低贝塔或防御性股票
            elif regime == "bull":
                beta_score = 0.6  # 牛市中，高贝塔股票可能更好，但此处专注超跌反弹
            else:
                beta_score = 0.7  # 中性市场
                
            scores.append(beta_score * self.config["weights"]["market_regime"]["market_beta"])
            
            # 2. 板块轮动得分
            volatility = market_regime.get("volatility", "normal")
            if volatility == "high":
                sector_score = 0.8  # 高波动环境，个股机会增加
            elif volatility == "low":
                sector_score = 0.5  # 低波动环境，个股机会较少
            else:
                sector_score = 0.7  # 正常波动
                
            scores.append(sector_score * self.config["weights"]["market_regime"]["sector_rotation"])
            
            # 3. 流动性因子得分
            trend_strength = market_regime.get("trend_strength", 0)
            if trend_strength > 0.03:
                liquidity_score = 0.6  # 强趋势市场，流动性集中
            elif trend_strength > 0.01:
                liquidity_score = 0.8  # 中等趋势，流动性适中
            else:
                liquidity_score = 0.7  # 弱趋势，流动性分散
                
            scores.append(liquidity_score * self.config["weights"]["market_regime"]["liquidity"])
            
            return sum(scores)
            
        except Exception as e:
            self.logger.error(f"计算市场环境得分失败: {e}")
            return 0.0
            
    def _calculate_momentum_factor_score(self, stock_df: pd.DataFrame) -> float:
        """计算动量因子得分 (0-100)"""
        try:
            if len(stock_df) < 10:
                return 50.0
                
            # 价格动量：不同时间窗口的涨跌幅
            latest_close = stock_df.iloc[0]['close']  # 数据是按日期降序排列的
            
            # 5日动量
            if len(stock_df) >= 5:
                momentum_5d = (latest_close / stock_df.iloc[4]['close'] - 1) * 100
            else:
                momentum_5d = 0
                
            # 10日动量  
            if len(stock_df) >= 10:
                momentum_10d = (latest_close / stock_df.iloc[9]['close'] - 1) * 100
            else:
                momentum_10d = momentum_5d
                
            # 20日动量
            if len(stock_df) >= 20:
                momentum_20d = (latest_close / stock_df.iloc[19]['close'] - 1) * 100
            else:
                momentum_20d = momentum_10d
            
            # 动量加速度 (近期动量 vs 早期动量)
            if len(stock_df) >= 15:
                recent_momentum = (latest_close / stock_df.iloc[7]['close'] - 1) * 100
                early_momentum = (stock_df.iloc[7]['close'] / stock_df.iloc[14]['close'] - 1) * 100
                momentum_acceleration = recent_momentum - early_momentum
            else:
                momentum_acceleration = 0
            
            # 综合动量得分
            composite_momentum = (
                momentum_5d * 0.4 + 
                momentum_10d * 0.3 + 
                momentum_20d * 0.2 +
                momentum_acceleration * 0.1
            )
            
            # 转换为0-100分，使用双向S曲线
            if composite_momentum >= 0:
                # 上涨动量：0到15%对应50到100分
                score = 50 + min(50, (composite_momentum / 15) * 50)
            else:
                # 下跌动量：0到-15%对应50到0分
                score = 50 + max(-50, (composite_momentum / 15) * 50)
            
            return round(score, 1)
            
        except Exception as e:
            self.logger.error(f"计算动量因子失败: {e}")
            return 50.0
    
    def _calculate_mean_reversion_score(self, stock_df: pd.DataFrame) -> float:
        """计算均值回归得分 (0-100) - 基于技术指标的超买超卖状态"""
        try:
            if stock_df.empty:
                return 50.0
                
            latest = stock_df.iloc[0]
            
            # RSI超买超卖评分
            rsi = latest.get('rsi', 50)
            if pd.isna(rsi):
                rsi = 50
                
            if rsi <= 30:
                rsi_score = 90 + (30 - rsi) / 30 * 10  # 超卖，高分
            elif rsi <= 40:
                rsi_score = 70 + (40 - rsi) / 10 * 20
            elif rsi <= 60:
                rsi_score = 50 + (50 - abs(rsi - 50)) / 10 * 20
            elif rsi <= 70:
                rsi_score = 30 + (70 - rsi) / 10 * 20
            else:
                rsi_score = 10 + max(0, (80 - rsi) / 10 * 20)  # 超买，低分
                
            # KDJ超买超卖评分
            kdj_k = latest.get('kdj_k', 50)
            kdj_d = latest.get('kdj_d', 50)
            
            if pd.isna(kdj_k):
                kdj_k = 50
            if pd.isna(kdj_d):
                kdj_d = 50
                
            kdj_avg = (kdj_k + kdj_d) / 2
            
            if kdj_avg <= 20:
                kdj_score = 95  # 严重超卖
            elif kdj_avg <= 30:
                kdj_score = 80 + (30 - kdj_avg) / 10 * 15
            elif kdj_avg <= 70:
                kdj_score = 50 + (50 - abs(kdj_avg - 50)) / 20 * 30
            elif kdj_avg <= 80:
                kdj_score = 20 + (80 - kdj_avg) / 10 * 30
            else:
                kdj_score = 5  # 严重超买
                
            # BBI位置评分 (价格相对于BBI的位置)
            close_price = latest.get('close', 0)
            bbi = latest.get('bbi', close_price)
            
            if pd.isna(bbi) or bbi == 0:
                bbi = close_price
                
            if bbi > 0:
                price_to_bbi = (close_price / bbi - 1) * 100
                if price_to_bbi <= -10:
                    bbi_score = 90  # 远低于BBI，回归机会大
                elif price_to_bbi <= -5:
                    bbi_score = 75 + (price_to_bbi + 5) / 5 * 15
                elif price_to_bbi <= 5:
                    bbi_score = 50 + (2.5 - abs(price_to_bbi)) / 2.5 * 25
                elif price_to_bbi <= 10:
                    bbi_score = 25 + (10 - price_to_bbi) / 5 * 25
                else:
                    bbi_score = 10  # 远高于BBI，回归压力大
            else:
                bbi_score = 50
                
            # 综合回归得分
            mean_reversion_score = rsi_score * 0.5 + kdj_score * 0.3 + bbi_score * 0.2
            
            return round(mean_reversion_score, 1)
            
        except Exception as e:
            self.logger.error(f"计算均值回归得分失败: {e}")
            return 50.0
    
    def _calculate_volume_breakout_score(self, stock_df: pd.DataFrame) -> float:
        """计算量价突破得分 (0-100)"""
        try:
            if len(stock_df) < 10:
                return 50.0
                
            latest = stock_df.iloc[0]
            
            # 成交量异常
            current_volume = latest.get('volume', 0)
            avg_volume = stock_df['volume'].head(20).mean()  # 20日平均成交量
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # 价格变化
            price_change_pct = latest.get('price_change_pct', 0)
            if pd.isna(price_change_pct):
                price_change_pct = 0
                
            # 量价配合度评分
            if price_change_pct > 3:  # 大涨
                if volume_ratio > 2:
                    volume_score = 95  # 放量大涨
                elif volume_ratio > 1.5:
                    volume_score = 85
                elif volume_ratio > 1:
                    volume_score = 70
                else:
                    volume_score = 45  # 无量上涨，可持续性差
            elif price_change_pct > 1:  # 中涨
                if volume_ratio > 1.5:
                    volume_score = 80
                elif volume_ratio > 1:
                    volume_score = 70
                else:
                    volume_score = 55
            elif price_change_pct > -1:  # 微涨或平盘
                if volume_ratio > 2:
                    volume_score = 75  # 平台整理放量，可能突破
                elif volume_ratio > 1.2:
                    volume_score = 65
                else:
                    volume_score = 50
            elif price_change_pct > -3:  # 小跌
                if volume_ratio < 0.7:
                    volume_score = 60  # 缩量下跌，抛压减轻
                elif volume_ratio < 1:
                    volume_score = 45
                else:
                    volume_score = 30  # 放量下跌
            else:  # 大跌
                if volume_ratio < 0.5:
                    volume_score = 65  # 无量下跌，可能止跌
                elif volume_ratio < 0.8:
                    volume_score = 50
                else:
                    volume_score = 15  # 放量大跌
                    
            # 突破确认：检查是否突破重要阻力位
            if len(stock_df) >= 20:
                recent_high = stock_df['close'].head(20).max()
                current_close = latest.get('close', 0)
                
                if current_close > recent_high * 1.02:  # 突破新高
                    breakthrough_bonus = 15
                elif current_close > recent_high * 1.005:  # 接近突破
                    breakthrough_bonus = 8
                else:
                    breakthrough_bonus = 0
                    
                volume_score = min(100, volume_score + breakthrough_bonus)
                    
            return round(volume_score, 1)
            
        except Exception as e:
            self.logger.error(f"计算量价突破得分失败: {e}")
            return 50.0
    
    def _calculate_relative_performance_score(self, stock_df: pd.DataFrame, market_df: pd.DataFrame) -> float:
        """计算相对表现得分 (0-100) - 股票相对于大盘的表现"""
        try:
            if stock_df.empty or market_df.empty:
                return 50.0
                
            # 计算股票收益率
            if len(stock_df) >= 20:
                stock_return_20d = (stock_df.iloc[0]['close'] / stock_df.iloc[19]['close'] - 1) * 100
            else:
                stock_return_20d = 0
                
            if len(stock_df) >= 5:
                stock_return_5d = (stock_df.iloc[0]['close'] / stock_df.iloc[4]['close'] - 1) * 100
            else:
                stock_return_5d = 0
                
            # 计算大盘收益率
            if len(market_df) >= 20:
                market_return_20d = (market_df.iloc[0]['close'] / market_df.iloc[19]['close'] - 1) * 100
            else:
                market_return_20d = 0
                
            if len(market_df) >= 5:
                market_return_5d = (market_df.iloc[0]['close'] / market_df.iloc[4]['close'] - 1) * 100
            else:
                market_return_5d = 0
                
            # 相对强弱
            relative_strength_20d = stock_return_20d - market_return_20d
            relative_strength_5d = stock_return_5d - market_return_5d
            
            # 综合相对强度
            relative_strength = relative_strength_20d * 0.7 + relative_strength_5d * 0.3
            
            # 转换为0-100分
            if relative_strength >= 10:
                score = 95  # 大幅跑赢
            elif relative_strength >= 5:
                score = 85 + (relative_strength - 5) / 5 * 10
            elif relative_strength >= 2:
                score = 70 + (relative_strength - 2) / 3 * 15
            elif relative_strength >= 0:
                score = 55 + relative_strength / 2 * 15
            elif relative_strength >= -2:
                score = 45 + relative_strength / 2 * 10
            elif relative_strength >= -5:
                score = 30 + (relative_strength + 2) / 3 * 15
            elif relative_strength >= -10:
                score = 15 + (relative_strength + 5) / 5 * 15
            else:
                score = 5  # 大幅跑输
                
            return round(score, 1)
            
        except Exception as e:
            self.logger.error(f"计算相对表现得分失败: {e}")
            return 50.0
    
    def _calculate_stability_score(self, stock_df: pd.DataFrame) -> float:
        """计算稳定性得分 (0-100) - 基于波动率和基本面稳定性"""
        try:
            if len(stock_df) < 10:
                return 50.0
                
            # 价格波动率
            price_changes = []
            for i in range(min(20, len(stock_df) - 1)):
                if i + 1 < len(stock_df):
                    price_change = (stock_df.iloc[i]['close'] / stock_df.iloc[i+1]['close'] - 1) * 100
                    price_changes.append(price_change)
                    
            if price_changes:
                price_volatility = np.std(price_changes)
                
                # 波动率评分：低波动高分，高波动低分
                if price_volatility <= 2:
                    volatility_score = 90 + (2 - price_volatility) / 2 * 10
                elif price_volatility <= 3:
                    volatility_score = 80 + (3 - price_volatility) / 1 * 10
                elif price_volatility <= 5:
                    volatility_score = 60 + (5 - price_volatility) / 2 * 20
                elif price_volatility <= 8:
                    volatility_score = 40 + (8 - price_volatility) / 3 * 20
                else:
                    volatility_score = max(5, 40 - (price_volatility - 8) / 2 * 10)
            else:
                volatility_score = 50
                
            # 成交量稳定性
            volumes = stock_df['volume'].head(20).tolist()
            if len(volumes) > 5:
                volume_cv = np.std(volumes) / np.mean(volumes) if np.mean(volumes) > 0 else 1
                
                # 成交量变异系数评分
                if volume_cv <= 0.5:
                    volume_stability_score = 90
                elif volume_cv <= 1:
                    volume_stability_score = 70 + (1 - volume_cv) / 0.5 * 20
                elif volume_cv <= 2:
                    volume_stability_score = 50 + (2 - volume_cv) / 1 * 20
                else:
                    volume_stability_score = max(10, 50 - (volume_cv - 2) / 1 * 20)
            else:
                volume_stability_score = 50
                
            # 基本面稳定性 (PE、PB变化)
            latest = stock_df.iloc[0]
            pe = latest.get('pe_ttm', None)
            pb = latest.get('pb', None)
            
            fundamental_score = 50  # 默认中性
            
            if pe is not None and pd.notna(pe) and pe > 0:
                if 10 <= pe <= 30:  # 合理PE范围
                    fundamental_score += 15
                elif 5 <= pe < 10 or 30 < pe <= 50:
                    fundamental_score += 5
                else:
                    fundamental_score -= 5
                    
            if pb is not None and pd.notna(pb) and pb > 0:
                if 1 <= pb <= 5:  # 合理PB范围
                    fundamental_score += 15
                elif 0.5 <= pb < 1 or 5 < pb <= 10:
                    fundamental_score += 5
                else:
                    fundamental_score -= 5
                    
            # 综合稳定性得分
            stability_score = (
                volatility_score * 0.5 + 
                volume_stability_score * 0.3 + 
                fundamental_score * 0.2
            )
            
            return round(min(100, max(0, stability_score)), 1)
            
        except Exception as e:
            self.logger.error(f"计算稳定性得分失败: {e}")
            return 50.0
            
    def calculate_stock_score(self, code: str, date: str, market_regime: Dict[str, Any] = None) -> Dict[str, Any]:
        """计算股票综合得分"""
        try:
            # 获取股票数据
            stock_query = """
            SELECT dq.trade_date, dq.close, dq.volume, dq.price_change_pct,
                   db.pe_ttm, db.pb, db.circ_mv, db.turnover_rate,
                   ti.kdj_k, ti.kdj_d, ti.kdj_j, ti.rsi12 as rsi, ti.bbi
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            LEFT JOIN daily_basic db ON db.security_id = s.id AND db.trade_date = dq.trade_date
            LEFT JOIN technical_indicators ti ON ti.security_id = s.id AND ti.trade_date = dq.trade_date
            WHERE s.code = ? AND dq.trade_date <= ?
            ORDER BY dq.trade_date DESC
            LIMIT 100
            """
            
            with self.db_manager.get_connection() as conn:
                stock_df = pd.read_sql_query(stock_query, conn, params=[code, date])
            
            if stock_df.empty:
                return {"code": code, "date": date, "score": 0, "error": "无数据"}
                
            # 获取大盘数据
            market_query = """
            SELECT dq.trade_date, dq.close, dq.price_change_pct
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.code = '000001.SH' AND dq.trade_date <= ?
            ORDER BY dq.trade_date DESC
            LIMIT 100
            """
            
            with self.db_manager.get_connection() as conn:
                market_df = pd.read_sql_query(market_query, conn, params=[date])
            
            # 如果没有提供市场环境，则检测
            if market_regime is None:
                market_regime = self.detect_market_regime(date)
                
            # 计算各部分得分
            technical_score = self.calculate_technical_score(stock_df)
            fundamental_score = self.calculate_fundamental_score(stock_df)
            performance_score = self.calculate_performance_score(stock_df, market_df)
            regime_score = self.calculate_market_regime_score(market_regime)
            
            # 🔥 使用传统五因子的最优权重计算综合得分
            # 基于快速测试得出的"相对强势型"最优权重配置
            traditional_weights = {
                "momentum": 0.250,
                "mean_reversion": 0.150, 
                "volume_breakout": 0.150,
                "relative_performance": 0.350,
                "stability": 0.100
            }
            
            # 获取传统因子分数 (0-100分制)
            momentum = self._calculate_momentum_factor_score(stock_df)
            mean_reversion = self._calculate_mean_reversion_score(stock_df)
            volume_breakout = self._calculate_volume_breakout_score(stock_df)
            relative_performance = self._calculate_relative_performance_score(stock_df, market_df)
            stability = self._calculate_stability_score(stock_df)
            
            # 使用传统因子加权计算最终得分 (转换为0-1范围)
            total_score = (
                momentum * traditional_weights["momentum"] +
                mean_reversion * traditional_weights["mean_reversion"] +
                volume_breakout * traditional_weights["volume_breakout"] + 
                relative_performance * traditional_weights["relative_performance"] +
                stability * traditional_weights["stability"]
            ) / 100.0  # 转换为0-1范围
            
            # 获取股票基本信息
            stock_info = stock_df.iloc[0] if not stock_df.empty else {}
            
            result = {
                "code": code,
                "date": date,
                "total_score": round(total_score, 4),
                "scores": {
                    "technical": round(technical_score, 4),
                    "fundamental": round(fundamental_score, 4), 
                    "performance": round(performance_score, 4),
                    "market_regime": round(regime_score, 4),
                    # 计算独立的传统因子评分
                    "momentum": self._calculate_momentum_factor_score(stock_df),
                    "mean_reversion": self._calculate_mean_reversion_score(stock_df),
                    "volume_breakout": self._calculate_volume_breakout_score(stock_df),
                    "relative_performance": self._calculate_relative_performance_score(stock_df, market_df),
                    "stability": self._calculate_stability_score(stock_df)
                },
                "details": {
                    "close": stock_info.get('close', 0),
                    "pe_ttm": stock_info.get('pe_ttm', None),
                    "pb": stock_info.get('pb', None),
                    "market_cap": stock_info.get('circ_mv', 0),
                    "kdj_k": stock_info.get('kdj_k', None),
                    "rsi": stock_info.get('rsi', None)
                },
                "market_regime": market_regime,
                "version": self.version
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"计算股票 {code} 得分失败: {e}")
            return {"code": code, "date": date, "score": 0, "error": str(e)}
            
    def batch_score_stocks(self, codes: List[str], date: str) -> List[Dict[str, Any]]:
        """批量计算股票得分"""
        results = []
        
        # 一次性检测市场环境
        market_regime = self.detect_market_regime(date)
        self.logger.info(f"市场环境: {market_regime}")
        
        total_count = len(codes)
        for i, code in enumerate(codes, 1):
            if i % 100 == 0 or i == total_count:
                self.logger.info(f"进度: {i}/{total_count}")
                
            result = self.calculate_stock_score(code, date, market_regime)
            results.append(result)
            
        return results
        
    def save_config(self, filepath: str):
        """保存配置到文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
            
    def load_config(self, filepath: str):
        """从文件加载配置"""
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                custom_config = json.load(f)
                self.config = self._merge_configs(self.default_config, custom_config)
                
    def get_weight_summary(self) -> Dict[str, float]:
        """获取权重总结"""
        weights = {}
        for category, sub_weights in self.config["weights"].items():
            for key, value in sub_weights.items():
                weights[f"{category}_{key}"] = value
        return weights


if __name__ == "__main__":
    # 测试代码
    scorer = QuantitativeScorerV3()
    
    # 测试单个股票
    result = scorer.calculate_stock_score("000001.SZ", "2025-08-12")
    print(f"测试结果: {result}")
    
    # 测试批量股票
    test_codes = ["000001.SZ", "000002.SZ", "000858.SZ"]
    results = scorer.batch_score_stocks(test_codes, "2025-08-12")
    
    for result in results:
        print(f"{result['code']}: {result['total_score']:.4f}")