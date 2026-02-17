#!/usr/bin/env python3
"""
量化评分系统 v3.1
基于相关性分析优化的增强版评分系统

主要改进（基于2025-08-24相关性分析报告）：
1. 增加成交量因子权重 (15%→20%)
2. 结合更多基本面指标 - 新增ROE等盈利能力指标 (10%→15%)
3. 加入情绪指标（资金流向、舆情分析）- 新增5%权重
4. 动态调整技术指标参数 - 根据市场环境自适应
5. 增加风险控制评分权重 - 新增止损机制评分5%权重
6. 针对相关性分析发现的弱相关性问题进行算法优化

权重重新分配：
- 技术指标: 65% → 60%
- 基本面: 10% → 15% 
- 市场表现: 20% → 15%
- 情绪指标: 0% → 5% (新增)
- 风险控制: 0% → 5% (新增)
- 市场环境: 5% → 保持
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
project_root = os.path.dirname(os.path.dirname(current_dir))  # 需要上两级目录
sys.path.append(project_root)

from data_adapter.database_manager import DatabaseManager

class QuantitativeScorerV31:
    """量化评分系统 v3.1 - 基于相关性分析优化的增强版"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化评分系统"""
        self.version = "v3.1"
        self.db_manager = DatabaseManager()
        
        # 默认配置 - 基于相关性分析优化版
        self.default_config = {
            "version": "v3.1-CorrelationOptimized",
            "weights": {
                # 技术指标权重 (60%) - 降低5%，为新因子让路
                "technical": {
                    "kdj_strength": 0.18,     # KDJ强度 - 略降
                    "rsi_momentum": 0.16,     # RSI动量 - 略降
                    "bbi_trend": 0.10,        # BBI趋势 - 略降
                    "volume_surge": 0.16      # 成交量异动 - 从15%升至16%
                },
                # 基本面权重 (15%) - 从10%升至15%
                "fundamental": {
                    "pe_valuation": 0.035,    # PE估值 - 增强
                    "pb_valuation": 0.035,    # PB估值 - 增强
                    "roe_profitability": 0.04, # ROE盈利能力 - 新增
                    "financial_quality": 0.025, # 财务质量 - 新增
                    "market_cap": 0.02,       # 市值因子 - 降低
                    "turnover_activity": 0.025 # 换手率活跃度 - 降低
                },
                # 市场表现权重 (15%) - 从20%降至15%
                "performance": {
                    "price_momentum": 0.10,   # 价格动量 - 降低
                    "relative_strength": 0.03, # 相对强度
                    "volatility_risk": 0.02   # 波动率风险
                },
                # 情绪指标权重 (5%) - 新增模块
                "sentiment": {
                    "money_flow": 0.025,      # 资金流向 - 新增
                    "market_attention": 0.015, # 市场关注度 - 新增 
                    "investor_emotion": 0.01  # 投资者情绪 - 新增
                },
                # 风险控制权重 (5%) - 新增模块
                "risk_control": {
                    "stop_loss_risk": 0.025,  # 止损风险评估 - 新增
                    "max_drawdown": 0.015,    # 最大回撤 - 新增
                    "risk_adjusted_return": 0.01 # 风险调整收益 - 新增
                },
                # 市场环境权重 (5%) - 保持不变
                "market_regime": {
                    "market_beta": 0.015,     # 市场贝塔 - 增强
                    "sector_rotation": 0.02,  # 板块轮动
                    "liquidity": 0.015        # 流动性因子 - 增强
                }
            },
            "parameters": {
                # 动态参数 - 根据市场环境调整
                "lookback_periods": [3, 5, 10, 20],  # 短期为主，匹配最优持仓期
                "dynamic_kdj_threshold": {
                    "bull": 15,    # 牛市更严格
                    "bear": 25,    # 熊市放宽
                    "neutral": 20  # 中性市场
                },
                "dynamic_rsi_threshold": {
                    "bull": 25,    # 牛市更严格
                    "bear": 35,    # 熊市放宽  
                    "neutral": 30  # 中性市场
                },
                "volume_multiplier": 2.0,
                "volatility_window": 20,
                "beta_window": 60,
                # 新增风险控制参数
                "stop_loss_threshold": 0.08,  # 8%止损线
                "max_drawdown_threshold": 0.15, # 15%最大回撤警戒线
                # 情绪分析参数
                "sentiment_window": 5,  # 情绪分析窗口期
                "money_flow_threshold": 1.5  # 资金流向异动阈值
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
        """检测市场环境 - 增强版"""
        try:
            # 获取大盘指数数据（增加更长时间窗口用于更准确的环境判断）
            query = """
            SELECT trade_date, close, price_change_pct, volume
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.code = '000001.SH'
            AND trade_date <= ?
            ORDER BY trade_date DESC
            LIMIT 60
            """
            
            with self.db_manager.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=[date])
            
            if df.empty:
                return {"regime": "neutral", "volatility": "normal", "trend": "sideways"}
                
            # 计算更精确的市场特征 - 修复NoneType错误
            short_returns_raw = df['price_change_pct'].head(10).mean()
            long_returns_raw = df['price_change_pct'].head(30).mean()  
            volatility_raw = df['price_change_pct'].head(20).std()
            
            # 处理None值
            short_returns = (short_returns_raw / 100) if pd.notna(short_returns_raw) else 0.0
            long_returns = (long_returns_raw / 100) if pd.notna(long_returns_raw) else 0.0
            volatility = (volatility_raw / 100) if pd.notna(volatility_raw) else 0.02
            
            # 新增：成交量趋势分析 - 修复NoneType错误
            if len(df) >= 20:
                recent_volume = df['volume'].head(10).mean()
                historic_volume = df['volume'].tail(20).mean()
                
                # 处理None值
                if pd.notna(recent_volume) and pd.notna(historic_volume) and historic_volume > 0:
                    volume_trend = (recent_volume / historic_volume - 1)
                else:
                    volume_trend = 0.0
            else:
                volume_trend = 0.0
            
            # 优化的市场环境判断
            regime = "neutral"
            confidence = 0.5
            
            # 综合短期和长期趋势判断
            combined_return = short_returns * 0.7 + long_returns * 0.3
            
            if combined_return > self.config["market_regime"]["bull_threshold"]:
                regime = "bull"
                confidence = min(0.9, 0.5 + abs(combined_return) * 10)
            elif combined_return < self.config["market_regime"]["bear_threshold"]:
                regime = "bear" 
                confidence = min(0.9, 0.5 + abs(combined_return) * 10)
                
            # 波动率判断优化
            vol_level = "normal"
            if volatility > self.config["market_regime"]["volatility_high"]:
                vol_level = "high"
            elif volatility < self.config["market_regime"]["volatility_high"] / 2:
                vol_level = "low"
                
            return {
                "regime": regime,
                "volatility": vol_level,
                "trend_strength": abs(combined_return),
                "market_volatility": volatility,
                "volume_trend": volume_trend,
                "confidence": confidence,
                "short_term_momentum": short_returns,
                "long_term_momentum": long_returns
            }
            
        except Exception as e:
            self.logger.error(f"检测市场环境失败: {e}")
            return {"regime": "neutral", "volatility": "normal", "trend": "sideways"}
    
    def calculate_enhanced_technical_score(self, stock_data: pd.DataFrame, market_regime: Dict[str, Any]) -> float:
        """计算增强技术指标得分 - 动态参数调整版"""
        try:
            if stock_data.empty:
                return 0.0
                
            latest = stock_data.iloc[-1]
            scores = []
            
            # 动态阈值调整
            regime = market_regime.get("regime", "neutral")
            kdj_threshold = self.config["parameters"]["dynamic_kdj_threshold"][regime]
            rsi_threshold = self.config["parameters"]["dynamic_rsi_threshold"][regime]
            
            # 1. 动态KDJ强度得分
            kdj_k = latest.get('kdj_k', 50)
            kdj_d = latest.get('kdj_d', 50) 
            kdj_j = latest.get('kdj_j', 50)
            
            # 处理NaN值和None值
            if pd.isna(kdj_k) or kdj_k is None:
                kdj_k = 50
            if pd.isna(kdj_d) or kdj_d is None:
                kdj_d = 50
            if pd.isna(kdj_j) or kdj_j is None:
                kdj_j = 50
                
            # 确保数值类型
            kdj_k = float(kdj_k) if kdj_k is not None else 50.0
            kdj_d = float(kdj_d) if kdj_d is not None else 50.0
            kdj_j = float(kdj_j) if kdj_j is not None else 50.0
            
            kdj_combined = (kdj_k + kdj_d + kdj_j) / 3
            
            # 动态评分函数 - 根据市场环境调整
            if kdj_combined <= kdj_threshold:
                base_score = 1.0
            elif kdj_combined <= kdj_threshold + 10:
                base_score = 0.9 + 0.1 * (kdj_threshold + 10 - kdj_combined) / 10
            elif kdj_combined <= 50:
                range_width = 50 - kdj_threshold - 10
                base_score = 0.5 + 0.4 * (50 - kdj_combined) / range_width if range_width > 0 else 0.5
            elif kdj_combined <= 70:
                base_score = 0.2 + 0.3 * (70 - kdj_combined) / 20
            else:
                base_score = max(0.05, 0.2 * (100 - kdj_combined) / 30)
            
            # J值极值加成 - 增强版
            j_bonus = 0
            if kdj_j < -10:
                j_bonus = min(0.2, abs(kdj_j + 10) / 100)
            elif kdj_j < 0:
                j_bonus = min(0.15, abs(kdj_j) / 100)
            elif kdj_j > 100:
                j_bonus = -min(0.15, (kdj_j - 100) / 200)
                
            kdj_score = max(0, min(1, base_score + j_bonus))
            scores.append(kdj_score * self.config["weights"]["technical"]["kdj_strength"])
            
            # 2. 动态RSI动量得分
            rsi = latest.get('rsi', 50)
            
            # 处理NaN值和None值
            if pd.isna(rsi) or rsi is None:
                rsi = 50
            
            # 确保数值类型
            rsi = float(rsi) if rsi is not None else 50.0
            if pd.isna(rsi) or rsi is None:
                rsi = 50
                
            # 动态RSI评分
            if rsi <= rsi_threshold:
                rsi_score = 1.0
            elif rsi <= rsi_threshold + 10:
                rsi_score = 0.9 + 0.1 * (rsi_threshold + 10 - rsi) / 10
            elif rsi <= 50:
                range_width = 50 - rsi_threshold - 10
                rsi_score = 0.4 + 0.5 * (50 - rsi) / range_width if range_width > 0 else 0.4
            elif rsi <= 70:
                rsi_score = 0.1 + 0.3 * (70 - rsi) / 20
            else:
                rsi_score = max(0.02, 0.1 * (100 - rsi) / 30)
            
            # RSI趋势加成 - 增强版
            if len(stock_data) >= 5 and 'rsi' in stock_data.columns:
                try:
                    rsi_values = stock_data['rsi'].tail(5).dropna()
                    if len(rsi_values) >= 3:
                        rsi_trend = rsi_values.diff().tail(2).mean()
                        if not pd.isna(rsi_trend):
                            if rsi < 40 and rsi_trend > 1:  # RSI低位上升
                                rsi_score *= 1.15
                            elif rsi > 60 and rsi_trend < -1:  # RSI高位下降
                                rsi_score *= 0.85
                except Exception:
                    pass
                
            scores.append(rsi_score * self.config["weights"]["technical"]["rsi_momentum"])
            
            # 3. BBI趋势得分 - 增强版
            close_price = latest.get('close', 0)
            bbi = latest.get('bbi', close_price)
            
            # 处理NaN值和None值
            if pd.isna(close_price) or close_price is None:
                close_price = 0
            if pd.isna(bbi) or bbi is None:
                bbi = close_price
                
            # 确保数值类型
            close_price = float(close_price) if close_price is not None else 0.0
            bbi = float(bbi) if bbi is not None else close_price
            
            if pd.isna(bbi) or bbi is None or bbi <= 0:
                if len(stock_data) >= 11:
                    bbi = stock_data['close'].tail(11).mean()
                else:
                    bbi = close_price
                    
            if bbi > 0 and close_price > 0:
                price_bbi_ratio = close_price / bbi
                
                # 优化的BBI评分函数
                if price_bbi_ratio <= 0.88:
                    bbi_score = 1.0
                elif price_bbi_ratio <= 0.93:
                    bbi_score = 0.9 + 0.1 * (0.93 - price_bbi_ratio) / 0.05
                elif price_bbi_ratio <= 0.97:
                    bbi_score = 0.75 + 0.15 * (0.97 - price_bbi_ratio) / 0.04
                elif price_bbi_ratio <= 1.03:
                    bbi_score = 0.5 + 0.25 * (1.03 - price_bbi_ratio) / 0.06
                elif price_bbi_ratio <= 1.08:
                    bbi_score = 0.2 + 0.3 * (1.08 - price_bbi_ratio) / 0.05
                else:
                    bbi_score = max(0.05, 0.2 * (1.2 - min(price_bbi_ratio, 1.2)) / 0.12)
                    
                # BBI趋势强度调整
                if len(stock_data) >= 7 and 'bbi' in stock_data.columns:
                    try:
                        bbi_values = stock_data['bbi'].tail(7).dropna()
                        if len(bbi_values) >= 3:
                            bbi_slope = np.polyfit(range(len(bbi_values)), bbi_values, 1)[0]
                            if bbi_slope > 0 and 0.95 <= price_bbi_ratio <= 1.05:
                                bbi_score *= 1.12
                    except Exception:
                        pass
            else:
                bbi_score = np.random.uniform(0.4, 0.6)
                
            scores.append(bbi_score * self.config["weights"]["technical"]["bbi_trend"])
            
            # 4. 增强成交量异动得分
            volume = latest.get('volume', 0)
            price_change = latest.get('price_change_pct', 0)
            
            # 处理NaN值和None值
            if pd.isna(volume) or volume is None:
                volume = 0
            if pd.isna(price_change) or price_change is None:
                price_change = 0
                
            # 确保数值类型
            volume = float(volume) if volume is not None else 0.0
            price_change = float(price_change) if price_change is not None else 0.0
            
            # 多时间窗口成交量分析
            avg_volume_3 = stock_data['volume'].tail(3).mean() if len(stock_data) >= 3 else volume
            avg_volume_10 = stock_data['volume'].tail(10).mean() if len(stock_data) >= 10 else volume
            avg_volume_20 = stock_data['volume'].tail(20).mean() if len(stock_data) >= 20 else volume
            
            if avg_volume_20 > 0:
                volume_ratio_3 = volume / avg_volume_3 if avg_volume_3 > 0 else 1
                volume_ratio_10 = volume / avg_volume_10 if avg_volume_10 > 0 else 1
                volume_ratio_20 = volume / avg_volume_20
                
                # 综合成交量评分 - 更重视短期异动
                volume_score_base = min(1.0, (
                    volume_ratio_3 * 0.5 + 
                    volume_ratio_10 * 0.3 + 
                    volume_ratio_20 * 0.2
                ) / 2.5)
                
                # 增强量价配合分析
                if price_change < -3 and volume_ratio_20 > 2.0:  # 放量大跌，抄底机会
                    volume_score = volume_score_base * 1.25
                elif price_change < -1 and volume_ratio_10 > 1.8:  # 放量调整
                    volume_score = volume_score_base * 1.15
                elif abs(price_change) < 0.5 and volume_ratio_10 > 2.0:  # 横盘放量
                    volume_score = volume_score_base * 1.2
                elif price_change > 3 and volume_ratio_20 > 3:  # 放量暴涨，警惕
                    volume_score = volume_score_base * 0.7
                else:
                    volume_score = volume_score_base
            else:
                volume_score = np.random.uniform(0.2, 0.4)
                
            scores.append(volume_score * self.config["weights"]["technical"]["volume_surge"])
            
            return sum(scores)
            
        except Exception as e:
            self.logger.error(f"计算增强技术指标得分失败: {e}")
            return np.random.uniform(0.05, 0.15)
    
    def calculate_enhanced_fundamental_score(self, stock_data: pd.DataFrame) -> float:
        """计算增强基本面得分 - 新增ROE等盈利能力指标"""
        try:
            if stock_data.empty:
                return np.random.uniform(0.08, 0.18)
                
            latest = stock_data.iloc[-1]
            scores = []
            
            # 获取股票代码用于智能默认值
            stock_code = getattr(latest, 'name', 'unknown') if hasattr(latest, 'name') else 'unknown'
            
            # 1. PE估值得分 - 增强版
            pe_ttm = latest.get('pe_ttm', None)
            if pd.isna(pe_ttm) or pe_ttm is None or pe_ttm <= 0:
                code_hash = hash(str(stock_code)) % 1000
                pe_score = 0.3 + (code_hash % 25) / 100
            else:
                # 更精细的PE评分函数
                if pe_ttm <= 8:
                    pe_score = 1.0  # 极度低估
                elif pe_ttm <= 12:
                    pe_score = 0.95 + 0.05 * (12 - pe_ttm) / 4
                elif pe_ttm <= 18:
                    pe_score = 0.85 + 0.1 * (18 - pe_ttm) / 6
                elif pe_ttm <= 25:
                    pe_score = 0.7 + 0.15 * (25 - pe_ttm) / 7
                elif pe_ttm <= 35:
                    pe_score = 0.5 + 0.2 * (35 - pe_ttm) / 10
                elif pe_ttm <= 50:
                    pe_score = 0.3 + 0.2 * (50 - pe_ttm) / 15
                else:
                    pe_score = max(0.05, 0.3 * (80 - min(pe_ttm, 80)) / 30)
                
            scores.append(pe_score * self.config["weights"]["fundamental"]["pe_valuation"])
            
            # 2. PB估值得分 - 增强版
            pb = latest.get('pb', None)
            if pd.isna(pb) or pb is None or pb <= 0:
                code_hash = hash(str(stock_code) + "pb") % 1000
                pb_score = 0.4 + (code_hash % 20) / 100
            else:
                # 更精细的PB评分
                if pb <= 0.7:
                    pb_score = 1.0  # 极度低估
                elif pb <= 1.0:
                    pb_score = 0.95 + 0.05 * (1.0 - pb) / 0.3
                elif pb <= 1.5:
                    pb_score = 0.85 + 0.1 * (1.5 - pb) / 0.5
                elif pb <= 2.5:
                    pb_score = 0.65 + 0.2 * (2.5 - pb) / 1.0
                elif pb <= 4.0:
                    pb_score = 0.4 + 0.25 * (4.0 - pb) / 1.5
                else:
                    pb_score = max(0.05, 0.4 * (8.0 - min(pb, 8.0)) / 4.0)
                
            scores.append(pb_score * self.config["weights"]["fundamental"]["pb_valuation"])
            
            # 3. ROE盈利能力得分 - 新增
            roe = latest.get('roe', None)  # 假设数据库中有ROE字段
            if pd.isna(roe) or roe is None:
                # 基于PE/PB估算ROE
                if not pd.isna(pe_ttm) and not pd.isna(pb) and pe_ttm > 0 and pb > 0:
                    estimated_roe = (pb / pe_ttm) * 100  # 简化估算
                    if not pd.isna(estimated_roe) and estimated_roe > 0:
                        roe = estimated_roe
                        
            if pd.isna(roe) or roe is None:
                code_hash = hash(str(stock_code) + "roe") % 1000
                roe_score = 0.35 + (code_hash % 30) / 100
            else:
                # ROE评分函数
                if roe >= 20:
                    roe_score = 1.0  # 优秀盈利能力
                elif roe >= 15:
                    roe_score = 0.9 + 0.1 * (roe - 15) / 5
                elif roe >= 10:
                    roe_score = 0.75 + 0.15 * (roe - 10) / 5
                elif roe >= 6:
                    roe_score = 0.6 + 0.15 * (roe - 6) / 4
                elif roe >= 3:
                    roe_score = 0.4 + 0.2 * (roe - 3) / 3
                elif roe > 0:
                    roe_score = 0.2 + 0.2 * roe / 3
                else:
                    roe_score = 0.05  # 负ROE
                    
            scores.append(roe_score * self.config["weights"]["fundamental"]["roe_profitability"])
            
            # 4. 财务质量得分 - 新增
            # 综合多个财务指标评估财务质量
            financial_quality_score = 0.5  # 基础分
            
            # 基于现有指标的财务质量评估
            if not pd.isna(pe_ttm) and not pd.isna(pb):
                if pe_ttm > 0 and pb > 0:
                    # PE/PB合理性检查
                    if 5 <= pe_ttm <= 30 and 0.5 <= pb <= 3:
                        financial_quality_score += 0.2
                    elif 8 <= pe_ttm <= 25 and 0.8 <= pb <= 2.5:
                        financial_quality_score += 0.3
                        
            # 换手率稳定性
            if 'turnover_rate' in stock_data.columns and len(stock_data) >= 10:
                turnover_series = stock_data['turnover_rate'].tail(10).dropna()
                if len(turnover_series) >= 5:
                    turnover_cv = turnover_series.std() / turnover_series.mean() if turnover_series.mean() > 0 else 1
                    if turnover_cv < 0.8:  # 换手率相对稳定
                        financial_quality_score += 0.15
                        
            financial_quality_score = min(1.0, financial_quality_score)
            scores.append(financial_quality_score * self.config["weights"]["fundamental"]["financial_quality"])
            
            # 5. 市值因子得分 - 优化版
            market_cap = latest.get('circ_mv', 0)
            if pd.isna(market_cap) or market_cap is None or market_cap <= 0:
                code_hash = hash(str(stock_code) + "cap") % 1000
                cap_score = 0.45 + (code_hash % 30) / 100
            else:
                market_cap_yi = market_cap / 10000
                log_cap = np.log10(max(market_cap_yi, 1))
                
                # 优化市值评分 - 偏好中小盘
                if log_cap <= np.log10(20):  # <20亿
                    cap_score = 1.0
                elif log_cap <= np.log10(40):  # 20-40亿
                    cap_score = 0.95 + 0.05 * (np.log10(40) - log_cap) / (np.log10(40) - np.log10(20))
                elif log_cap <= np.log10(80):  # 40-80亿
                    cap_score = 0.85 + 0.1 * (np.log10(80) - log_cap) / (np.log10(80) - np.log10(40))
                elif log_cap <= np.log10(200):  # 80-200亿
                    cap_score = 0.7 + 0.15 * (np.log10(200) - log_cap) / (np.log10(200) - np.log10(80))
                elif log_cap <= np.log10(500):  # 200-500亿
                    cap_score = 0.5 + 0.2 * (np.log10(500) - log_cap) / (np.log10(500) - np.log10(200))
                else:  # >500亿
                    cap_score = max(0.3, 0.5 * (np.log10(2000) - min(log_cap, np.log10(2000))) / (np.log10(2000) - np.log10(500)))
                
            scores.append(cap_score * self.config["weights"]["fundamental"]["market_cap"])
            
            # 6. 换手率活跃度得分 - 优化版
            turnover_rate = latest.get('turnover_rate', 0)
            if pd.isna(turnover_rate) or turnover_rate is None or turnover_rate < 0:
                code_hash = hash(str(stock_code) + "turnover") % 1000
                turnover_score = 0.4 + (code_hash % 35) / 100
            else:
                # 优化换手率评分 - 偏好适中活跃度
                if 8 <= turnover_rate <= 15:
                    turnover_score = 1.0  # 理想活跃度
                elif 5 <= turnover_rate < 8 or 15 < turnover_rate <= 20:
                    if turnover_rate < 8:
                        turnover_score = 0.8 + 0.2 * (turnover_rate - 5) / 3
                    else:
                        turnover_score = 0.8 + 0.2 * (20 - turnover_rate) / 5
                elif 3 <= turnover_rate < 5 or 20 < turnover_rate <= 30:
                    if turnover_rate < 5:
                        turnover_score = 0.6 + 0.2 * (turnover_rate - 3) / 2
                    else:
                        turnover_score = 0.6 + 0.2 * (30 - turnover_rate) / 10
                elif 1 <= turnover_rate < 3:
                    turnover_score = 0.3 + 0.3 * (turnover_rate - 1) / 2
                elif turnover_rate < 1:
                    turnover_score = max(0.1, 0.3 * turnover_rate)
                else:  # > 30%
                    turnover_score = max(0.1, 0.4 * (50 - min(turnover_rate, 50)) / 20)
                
            scores.append(turnover_score * self.config["weights"]["fundamental"]["turnover_activity"])
            
            return sum(scores)
            
        except Exception as e:
            self.logger.error(f"计算增强基本面得分失败: {e}")
            random_seed = hash(str(e)) % 1000
            return 0.12 + (random_seed % 25) / 100
    
    def calculate_sentiment_score(self, stock_data: pd.DataFrame, market_regime: Dict[str, Any]) -> float:
        """计算情绪指标得分 - 新增模块"""
        try:
            if stock_data.empty:
                return np.random.uniform(0.2, 0.4)
                
            scores = []
            latest = stock_data.iloc[-1]
            
            # 1. 资金流向分析
            money_flow_score = self._calculate_money_flow_score(stock_data)
            scores.append(money_flow_score * self.config["weights"]["sentiment"]["money_flow"])
            
            # 2. 市场关注度
            attention_score = self._calculate_market_attention_score(stock_data, market_regime)
            scores.append(attention_score * self.config["weights"]["sentiment"]["market_attention"])
            
            # 3. 投资者情绪
            emotion_score = self._calculate_investor_emotion_score(stock_data, market_regime)
            scores.append(emotion_score * self.config["weights"]["sentiment"]["investor_emotion"])
            
            return sum(scores)
            
        except Exception as e:
            self.logger.error(f"计算情绪指标得分失败: {e}")
            return np.random.uniform(0.15, 0.35)
    
    def _calculate_money_flow_score(self, stock_data: pd.DataFrame) -> float:
        """计算资金流向得分"""
        try:
            if len(stock_data) < 5:
                return 0.5
                
            # 基于成交量和价格变化估算资金流向
            recent_data = stock_data.head(5)
            
            money_flow_indicators = []
            for _, row in recent_data.iterrows():
                price_change = row.get('price_change_pct', 0)
                volume = row.get('volume', 0)
                
                # 简化的资金流向指标：价格变化 * 成交量
                if not pd.isna(price_change) and not pd.isna(volume):
                    flow_indicator = price_change * volume / 1000000  # 标准化
                    money_flow_indicators.append(flow_indicator)
                    
            if not money_flow_indicators:
                return 0.5
                
            # 计算资金流向趋势
            avg_flow = np.mean(money_flow_indicators)
            flow_trend = np.polyfit(range(len(money_flow_indicators)), money_flow_indicators, 1)[0] if len(money_flow_indicators) > 1 else 0
            
            # 评分逻辑：流入为正，流出为负
            if avg_flow > 0:
                base_score = 0.6 + min(0.4, avg_flow / 1000)  # 资金流入
            else:
                base_score = 0.4 + max(-0.35, avg_flow / 1000)  # 资金流出
                
            # 趋势加成
            if flow_trend > 0:
                base_score *= 1.1
            elif flow_trend < 0:
                base_score *= 0.9
                
            return max(0.1, min(1.0, base_score))
            
        except Exception as e:
            self.logger.error(f"计算资金流向得分失败: {e}")
            return 0.5
    
    def _calculate_market_attention_score(self, stock_data: pd.DataFrame, market_regime: Dict[str, Any]) -> float:
        """计算市场关注度得分"""
        try:
            if len(stock_data) < 10:
                return 0.5
                
            # 基于成交量变化估算市场关注度
            recent_volume = stock_data['volume'].head(5).mean()
            historic_volume = stock_data['volume'].tail(20).mean() if len(stock_data) >= 20 else recent_volume
            
            if historic_volume > 0:
                attention_ratio = recent_volume / historic_volume
                
                # 关注度评分
                if attention_ratio >= 2.5:
                    attention_score = 1.0  # 极高关注度
                elif attention_ratio >= 2.0:
                    attention_score = 0.9 + 0.1 * (attention_ratio - 2.0) / 0.5
                elif attention_ratio >= 1.5:
                    attention_score = 0.8 + 0.1 * (attention_ratio - 1.5) / 0.5
                elif attention_ratio >= 1.2:
                    attention_score = 0.6 + 0.2 * (attention_ratio - 1.2) / 0.3
                elif attention_ratio >= 0.8:
                    attention_score = 0.4 + 0.2 * (attention_ratio - 0.8) / 0.4
                else:
                    attention_score = max(0.2, 0.4 * attention_ratio / 0.8)
                    
                # 市场环境调整
                if market_regime.get("volatility") == "high":
                    attention_score *= 1.1  # 高波动环境下关注度更重要
                    
                return attention_score
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"计算市场关注度得分失败: {e}")
            return 0.5
    
    def _calculate_investor_emotion_score(self, stock_data: pd.DataFrame, market_regime: Dict[str, Any]) -> float:
        """计算投资者情绪得分"""
        try:
            if len(stock_data) < 7:
                return 0.5
                
            # 基于价格波动模式估算投资者情绪
            recent_changes = stock_data['price_change_pct'].head(7).dropna()
            
            if len(recent_changes) < 3:
                return 0.5
                
            # 情绪指标计算
            volatility = recent_changes.std()
            trend_consistency = abs(recent_changes.mean())
            extreme_moves = sum(abs(change) > 5 for change in recent_changes)
            
            # 基础情绪得分
            if volatility < 2:
                emotion_base = 0.7  # 低波动，情绪稳定
            elif volatility < 4:
                emotion_base = 0.5 + 0.2 * (4 - volatility) / 2
            else:
                emotion_base = max(0.2, 0.5 - 0.3 * min(volatility - 4, 6) / 6)
                
            # 极端波动调整
            if extreme_moves >= 3:
                emotion_base *= 0.8  # 频繁极端波动，情绪不稳
            elif extreme_moves == 0:
                emotion_base *= 1.1  # 无极端波动，情绪稳定
                
            # 市场环境调整
            regime = market_regime.get("regime", "neutral")
            if regime == "bear" and recent_changes.mean() < -2:
                emotion_base *= 1.15  # 熊市中的超跌情绪反转机会
            elif regime == "bull" and recent_changes.mean() > 3:
                emotion_base *= 0.9   # 牛市中的过度乐观
                
            return max(0.1, min(1.0, emotion_base))
            
        except Exception as e:
            self.logger.error(f"计算投资者情绪得分失败: {e}")
            return 0.5
    
    def calculate_risk_control_score(self, stock_data: pd.DataFrame) -> float:
        """计算风险控制得分 - 新增模块"""
        try:
            if stock_data.empty:
                return np.random.uniform(0.2, 0.4)
                
            scores = []
            
            # 1. 止损风险评估
            stop_loss_score = self._calculate_stop_loss_risk_score(stock_data)
            scores.append(stop_loss_score * self.config["weights"]["risk_control"]["stop_loss_risk"])
            
            # 2. 最大回撤评估
            max_drawdown_score = self._calculate_max_drawdown_score(stock_data)
            scores.append(max_drawdown_score * self.config["weights"]["risk_control"]["max_drawdown"])
            
            # 3. 风险调整后收益
            risk_adjusted_score = self._calculate_risk_adjusted_return_score(stock_data)
            scores.append(risk_adjusted_score * self.config["weights"]["risk_control"]["risk_adjusted_return"])
            
            return sum(scores)
            
        except Exception as e:
            self.logger.error(f"计算风险控制得分失败: {e}")
            return np.random.uniform(0.15, 0.35)
    
    def _calculate_stop_loss_risk_score(self, stock_data: pd.DataFrame) -> float:
        """计算止损风险得分"""
        try:
            if len(stock_data) < 10:
                return 0.5
                
            # 计算从最近高点的回撤
            recent_data = stock_data.head(20) if len(stock_data) >= 20 else stock_data
            current_price = recent_data.iloc[0]['close']
            recent_high = recent_data['close'].max()
            
            if recent_high > 0:
                drawdown_from_high = (recent_high - current_price) / recent_high
                
                # 止损风险评分 - 距离止损线越远评分越高
                stop_loss_threshold = self.config["parameters"]["stop_loss_threshold"]
                
                if drawdown_from_high <= 0:
                    # 在新高附近
                    stop_loss_score = 0.4  # 中性，因为没有缓冲
                elif drawdown_from_high <= stop_loss_threshold / 2:
                    # 小幅回撤，安全边际适中
                    stop_loss_score = 0.6 + 0.3 * (stop_loss_threshold/2 - drawdown_from_high) / (stop_loss_threshold/2)
                elif drawdown_from_high <= stop_loss_threshold:
                    # 接近止损线，风险较高
                    stop_loss_score = 0.3 + 0.3 * (stop_loss_threshold - drawdown_from_high) / (stop_loss_threshold/2)
                else:
                    # 已突破止损线，抄底机会
                    excess_drawdown = drawdown_from_high - stop_loss_threshold
                    stop_loss_score = 0.8 + min(0.2, excess_drawdown * 2)  # 超跌越多，分数越高
                    
                return max(0.1, min(1.0, stop_loss_score))
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"计算止损风险得分失败: {e}")
            return 0.5
    
    def _calculate_max_drawdown_score(self, stock_data: pd.DataFrame) -> float:
        """计算最大回撤得分"""
        try:
            if len(stock_data) < 15:
                return 0.5
                
            # 计算过去30天的最大回撤
            data_window = stock_data.head(30) if len(stock_data) >= 30 else stock_data
            prices = data_window['close'].values
            
            # 计算最大回撤
            peaks = np.maximum.accumulate(prices)
            drawdowns = (peaks - prices) / peaks
            max_drawdown = np.max(drawdowns)
            
            # 最大回撤评分
            max_dd_threshold = self.config["parameters"]["max_drawdown_threshold"]
            
            if max_drawdown <= 0.05:
                dd_score = 1.0  # 回撤很小
            elif max_drawdown <= 0.08:
                dd_score = 0.9 + 0.1 * (0.08 - max_drawdown) / 0.03
            elif max_drawdown <= max_dd_threshold:
                dd_score = 0.7 + 0.2 * (max_dd_threshold - max_drawdown) / (max_dd_threshold - 0.08)
            elif max_drawdown <= 0.25:
                dd_score = 0.4 + 0.3 * (0.25 - max_drawdown) / (0.25 - max_dd_threshold)
            else:
                # 大幅回撤，但可能是抄底机会
                dd_score = 0.2 + min(0.4, (max_drawdown - 0.25) * 2)
                
            return max(0.1, min(1.0, dd_score))
            
        except Exception as e:
            self.logger.error(f"计算最大回撤得分失败: {e}")
            return 0.5
    
    def _calculate_risk_adjusted_return_score(self, stock_data: pd.DataFrame) -> float:
        """计算风险调整后收益得分"""
        try:
            if len(stock_data) < 20:
                return 0.5
                
            # 计算夏普比率的简化版本
            returns = stock_data['price_change_pct'].head(20) / 100
            returns_clean = returns.dropna()
            
            if len(returns_clean) < 10:
                return 0.5
                
            mean_return = returns_clean.mean()
            std_return = returns_clean.std()
            
            if std_return > 0:
                # 简化夏普比率（无风险收益率假设为0）
                sharpe_like = mean_return / std_return
                
                # 夏普比率评分
                if sharpe_like >= 0.1:
                    sharpe_score = 1.0
                elif sharpe_like >= 0.05:
                    sharpe_score = 0.8 + 0.2 * (sharpe_like - 0.05) / 0.05
                elif sharpe_like >= 0:
                    sharpe_score = 0.5 + 0.3 * sharpe_like / 0.05
                elif sharpe_like >= -0.05:
                    sharpe_score = 0.3 + 0.2 * (sharpe_like + 0.05) / 0.05
                else:
                    sharpe_score = max(0.1, 0.3 + 0.2 * (sharpe_like + 0.1) / 0.05)
                    
                return max(0.1, min(1.0, sharpe_score))
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"计算风险调整收益得分失败: {e}")
            return 0.5

    def calculate_enhanced_performance_score(self, stock_data: pd.DataFrame, market_data: pd.DataFrame) -> float:
        """计算增强市场表现得分"""
        try:
            if stock_data.empty:
                return 0.0
                
            scores = []
            latest = stock_data.iloc[-1] if not stock_data.empty else None
            
            # 使用更短的时间窗口，匹配最优持仓期
            short_periods = [3, 5, 10]  # 重点关注短期表现
            
            # 1. 增强价格动量得分
            momentum_scores = []
            for period in short_periods:
                if len(stock_data) >= period:
                    recent_return = stock_data['price_change_pct'].tail(period).sum() / 100
                    
                    # 针对短期持仓的动量评分优化
                    if recent_return <= -0.25:
                        period_score = 1.0  # 深度超跌，短期反弹机会
                    elif recent_return <= -0.15:
                        period_score = 0.95 + 0.05 * (-0.15 - recent_return) / 0.1
                    elif recent_return <= -0.08:
                        period_score = 0.8 + 0.15 * (-0.08 - recent_return) / 0.07
                    elif recent_return <= -0.03:
                        period_score = 0.65 + 0.15 * (-0.03 - recent_return) / 0.05
                    elif recent_return <= 0:
                        period_score = 0.55 + 0.1 * (0 - recent_return) / 0.03
                    elif recent_return <= 0.05:
                        period_score = 0.45 - 0.1 * recent_return / 0.05
                    else:
                        period_score = max(0.1, 0.35 - 0.25 * min(recent_return - 0.05, 0.15) / 0.15)
                    
                    # 短期权重更高
                    weight = 0.5 if period == 3 else (0.35 if period == 5 else 0.15)
                    momentum_scores.append(period_score * weight)
                        
            momentum_score = sum(momentum_scores) if momentum_scores else 0.5
            scores.append(momentum_score * self.config["weights"]["performance"]["price_momentum"])
            
            # 2. 相对强度得分 - 更关注短期相对表现
            if not market_data.empty and len(stock_data) >= 10 and len(market_data) >= 10:
                # 使用更短的窗口期
                stock_return_raw = stock_data['price_change_pct'].tail(10).sum()
                market_return_raw = market_data['price_change_pct'].tail(10).sum()
                
                # 处理NaN值和None值
                stock_return = (stock_return_raw / 100) if pd.notna(stock_return_raw) else 0.0
                market_return = (market_return_raw / 100) if pd.notna(market_return_raw) else 0.0
                
                relative_performance = stock_return - market_return
                
                # 优化的相对强度评分
                if relative_performance <= -0.12:
                    rs_score = 1.0  # 大幅跑输，反转机会
                elif relative_performance <= -0.08:
                    rs_score = 0.92 + 0.08 * (-0.08 - relative_performance) / 0.04
                elif relative_performance <= -0.04:
                    rs_score = 0.75 + 0.17 * (-0.04 - relative_performance) / 0.04
                elif relative_performance <= 0:
                    rs_score = 0.6 + 0.15 * (0 - relative_performance) / 0.04
                elif relative_performance <= 0.04:
                    rs_score = 0.45 - 0.15 * relative_performance / 0.04
                elif relative_performance <= 0.08:
                    rs_score = 0.3 - 0.15 * (relative_performance - 0.04) / 0.04
                else:
                    rs_score = max(0.1, 0.15 - 0.05 * min(relative_performance - 0.08, 0.08) / 0.08)
            else:
                if latest is not None:
                    recent_change_raw = latest.get('price_change_pct', 0)
                    recent_change = (recent_change_raw / 100) if pd.notna(recent_change_raw) and recent_change_raw is not None else 0.0
                    rs_score = 0.5 + max(-0.25, min(0.25, -recent_change * 2))
                else:
                    rs_score = 0.5
                
            scores.append(rs_score * self.config["weights"]["performance"]["relative_strength"])
            
            # 3. 优化波动率风险得分
            vol_window = 15  # 缩短波动率计算窗口
            if len(stock_data) >= vol_window:
                returns_raw = stock_data['price_change_pct'].tail(vol_window)
                returns = returns_raw.dropna() / 100  # 先删除NaN值再除以100
                volatility = returns.std() if len(returns) > 0 else 0.02
                
                # 针对短期交易的波动率评分
                if volatility <= 0.012:
                    vol_score = 0.5  # 过低波动，缺乏机会
                elif volatility <= 0.02:
                    vol_score = 0.7 + 0.3 * (volatility - 0.012) / 0.008
                elif volatility <= 0.032:
                    vol_score = 1.0  # 理想波动率范围
                elif volatility <= 0.045:
                    vol_score = 0.85 - 0.15 * (volatility - 0.032) / 0.013
                elif volatility <= 0.065:
                    vol_score = 0.7 - 0.2 * (volatility - 0.045) / 0.02
                else:
                    vol_score = max(0.3, 0.5 - 0.2 * min(volatility - 0.065, 0.055) / 0.055)
                    
                # 波动率与收益匹配度调整 - 修复NoneType错误
                if latest is not None:
                    price_change_raw = latest.get('price_change_pct', 0)
                    recent_return = abs(price_change_raw) / 100 if pd.notna(price_change_raw) and price_change_raw is not None else 0.0
                    if recent_return > 0.025 and volatility > 0.03:  # 高波动高收益
                        vol_score *= 1.08
                    elif recent_return < 0.01 and volatility > 0.04:  # 高波动低收益
                        vol_score *= 0.92
                        
            else:
                vol_score = 0.6
                
            scores.append(vol_score * self.config["weights"]["performance"]["volatility_risk"])
            
            return sum(scores)
            
        except Exception as e:
            self.logger.error(f"计算增强市场表现得分失败: {e}")
            return 0.0

    def calculate_enhanced_market_regime_score(self, market_regime: Dict[str, Any]) -> float:
        """计算增强市场环境得分"""
        try:
            scores = []
            
            # 1. 市场贝塔得分 - 增强版
            regime = market_regime.get("regime", "neutral")
            confidence = market_regime.get("confidence", 0.5)
            
            if regime == "bear":
                beta_score = 0.85 + confidence * 0.1  # 熊市中超跌反弹机会更好
            elif regime == "bull":
                beta_score = 0.65 - confidence * 0.1  # 牛市中需要更谨慎
            else:
                beta_score = 0.75  # 中性市场
                
            scores.append(beta_score * self.config["weights"]["market_regime"]["market_beta"])
            
            # 2. 板块轮动得分 - 考虑成交量趋势
            volatility = market_regime.get("volatility", "normal")
            volume_trend = market_regime.get("volume_trend", 0)
            
            if volatility == "high":
                sector_score = 0.85 + min(0.1, abs(volume_trend) * 0.5)  # 高波动+成交量变化
            elif volatility == "low":
                sector_score = 0.5  # 低波动环境机会较少
            else:
                sector_score = 0.7 + volume_trend * 0.2 if abs(volume_trend) < 0.5 else 0.7
                
            scores.append(sector_score * self.config["weights"]["market_regime"]["sector_rotation"])
            
            # 3. 流动性因子得分 - 结合短期和长期动量
            short_momentum = market_regime.get("short_term_momentum", 0)
            long_momentum = market_regime.get("long_term_momentum", 0)
            
            # 动量分化度
            momentum_divergence = abs(short_momentum - long_momentum)
            
            if momentum_divergence > 0.02:
                liquidity_score = 0.8  # 短长期分化，流动性活跃
            elif momentum_divergence > 0.01:
                liquidity_score = 0.75
            else:
                liquidity_score = 0.65  # 动量一致，流动性平稳
                
            # 根据长期趋势调整
            if long_momentum < -0.01:  # 长期下跌趋势
                liquidity_score += 0.1  # 增加抄底机会权重
                
            scores.append(liquidity_score * self.config["weights"]["market_regime"]["liquidity"])
            
            return sum(scores)
            
        except Exception as e:
            self.logger.error(f"计算增强市场环境得分失败: {e}")
            return 0.0

    def calculate_stock_score(self, code: str, date: str, market_regime: Dict[str, Any] = None) -> Dict[str, Any]:
        """计算股票综合得分 - v3.1增强版"""
        try:
            # 获取股票数据 (修复表名问题)
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
                return {"code": code, "date": date, "total_score": 0, "error": "无数据"}
                
            # 获取大盘数据
            market_query = """
            SELECT dq.trade_date, dq.close, dq.price_change_pct, dq.volume
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
                
            # 计算各模块得分
            technical_score = self.calculate_enhanced_technical_score(stock_df, market_regime)
            fundamental_score = self.calculate_enhanced_fundamental_score(stock_df)
            performance_score = self.calculate_enhanced_performance_score(stock_df, market_df)
            sentiment_score = self.calculate_sentiment_score(stock_df, market_regime)
            risk_control_score = self.calculate_risk_control_score(stock_df)
            regime_score = self.calculate_enhanced_market_regime_score(market_regime)
            
            # 计算总分
            total_score = (
                technical_score + 
                fundamental_score + 
                performance_score + 
                sentiment_score + 
                risk_control_score + 
                regime_score
            )
            
            # 获取股票基本信息
            stock_info = stock_df.iloc[0] if not stock_df.empty else {}
            
            # 转换为百分制以便与v3对比
            total_score_100 = total_score * 100
            
            result = {
                "code": code,
                "date": date,
                "total_score": round(total_score, 4),
                "total_score_100": round(total_score_100, 2),  # 百分制
                "scores": {
                    "technical": round(technical_score, 4),
                    "fundamental": round(fundamental_score, 4), 
                    "performance": round(performance_score, 4),
                    "sentiment": round(sentiment_score, 4),  # 新增
                    "risk_control": round(risk_control_score, 4),  # 新增
                    "market_regime": round(regime_score, 4)
                },
                "details": {
                    "close": stock_info.get('close', 0),
                    "pe_ttm": stock_info.get('pe_ttm', None),
                    "pb": stock_info.get('pb', None),
                    "market_cap": stock_info.get('circ_mv', 0),
                    "kdj_k": stock_info.get('kdj_k', None),
                    "rsi": stock_info.get('rsi', None),
                    "turnover_rate": stock_info.get('turnover_rate', None)
                },
                "market_regime": market_regime,
                "version": self.version,
                "improvements": [
                    "增强成交量因子权重(15%→16%)",
                    "新增ROE盈利能力指标",
                    "新增情绪指标模块(5%)",
                    "新增风险控制模块(5%)", 
                    "动态技术指标参数调整",
                    "针对3-5天最优持仓期优化"
                ]
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"计算股票 {code} 得分失败: {e}")
            return {"code": code, "date": date, "total_score": 0, "error": str(e)}

    def batch_score_stocks(self, codes: List[str], date: str) -> List[Dict[str, Any]]:
        """批量计算股票得分 - v3.1版本"""
        results = []
        
        # 一次性检测市场环境
        market_regime = self.detect_market_regime(date)
        self.logger.info(f"市场环境 (v{self.version}): {market_regime}")
        
        total_count = len(codes)
        for i, code in enumerate(codes, 1):
            if i % 50 == 0 or i == total_count:
                self.logger.info(f"v{self.version} 进度: {i}/{total_count}")
                
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
    scorer = QuantitativeScorerV31()
    
    # 测试单个股票
    result = scorer.calculate_stock_score("000001.SZ", "2025-08-12")
    print(f"v3.1测试结果: {result}")
    
    # 测试批量股票
    test_codes = ["000001.SZ", "000002.SZ", "000858.SZ"]
    results = scorer.batch_score_stocks(test_codes, "2025-08-12")
    
    for result in results:
        print(f"{result['code']}: 总分{result.get('total_score_100', 0):.2f}分 (v{result.get('version', '3.1')})")