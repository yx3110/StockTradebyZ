#!/usr/bin/env python3
"""
量化评分系统 v3.5 - 集成知行指标版本
基于v3.4成功版本，新增两个知行技术指标：
1. 知行短期趋势线: EMA(EMA(C,10),10) 
2. 知行多空线: (MA(CLOSE,M1)+MA(CLOSE,M2)+MA(CLOSE,M3)+MA(CLOSE,M4))/4

新增指标特点：
- 知行短期趋势线：双重指数平滑，更敏感反映短期趋势变化
- 知行多空线：多周期均线组合，判断多空平衡状态
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

class QuantitativeScorerV35:
    """量化评分系统 v3.5 - 集成知行指标版本"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化评分系统"""
        self.version = "v3.5"
        self.db_manager = DatabaseManager()
        
        # v3.5优化配置 - 知行指标权重大幅提升至20%
        self.default_config = {
            "version": "v3.5-Enhanced-With-ZhiXing-Indicators-20pct",
            "weights": {
                # 技术指标权重 (60%) - 为知行指标让出空间，从64%调整至60%
                "technical": {
                    "kdj_strength": 0.12,     # 从0.16降至0.12
                    "rsi_momentum": 0.10,     # 从0.14降至0.10
                    "bbi_trend": 0.08,        # 从0.10降至0.08
                    "volume_surge": 0.10,     # 从0.14降至0.10
                    # 知行指标权重大幅提升至20%
                    "zhixing_trend": 0.12,    # 知行短期趋势线 12% (从6%提升)
                    "zhixing_multiavg": 0.08  # 知行多空线 8% (从4%提升)
                },
                # 基本面权重 (14%) - 从16%降至14%
                "fundamental": {
                    "pe_valuation": 0.025,    # 从0.03降至0.025
                    "pb_valuation": 0.025,    # 从0.03降至0.025
                    "roe_profitability": 0.025, # 从0.03降至0.025
                    "revenue_growth": 0.02,   # 保持不变
                    "market_cap": 0.025,      # 从0.03降至0.025
                    "turnover_activity": 0.018  # 从0.02降至0.018
                },
                # 市场表现权重 (13%) - 从17%降至13%
                "performance": {
                    "price_momentum": 0.08,   # 从0.11降至0.08
                    "relative_strength": 0.03, # 从0.04降至0.03
                    "volatility_risk": 0.02   # 保持不变
                },
                # 市场环境权重 (3%) - 保持不变
                "market_regime": {
                    "market_beta": 0.01,
                    "sector_rotation": 0.01,
                    "liquidity": 0.01
                }
            },
            "parameters": {
                "lookback_periods": [5, 10, 20, 30],
                "kdj_threshold": 18,
                "rsi_oversold": 28,
                "volume_multiplier": 2.0,
                "volatility_window": 20,
                "beta_window": 60,
                "roe_threshold": 0.08,
                "revenue_growth_threshold": 0.10,
                # 知行指标参数
                "zhixing_trend_period": 10,      # 知行趋势线周期
                "zhixing_multiavg_periods": [5, 10, 20, 60]  # 知行多空线周期组合 (M1, M2, M3, M4)
            },
            "market_regime": {
                "bull_threshold": 0.02,
                "bear_threshold": -0.02,
                "volatility_high": 0.03,
                "bull_multiplier": 1.25,
                "bear_multiplier": 0.40,
                "neutral_multiplier": 1.00
            },
            "scoring_optimization": {
                "high_score_threshold_adjustment": -5,
                "score_smoothing": True,
                "continuous_functions": True
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
        
    def calculate_zhixing_trend(self, close_prices: np.array) -> Optional[float]:
        """
        计算知行短期趋势线: EMA(EMA(C,10),10)
        通达信公式: 知行短期趋势线:EMA(EMA(C,10),10),COLORFFFFFF,LINETHICK1;
        """
        try:
            if len(close_prices) < 20:  # 至少需要20天数据计算双重EMA
                return None
                
            # 第一层EMA(C, 10)
            ema1 = pd.Series(close_prices).ewm(span=10).mean().values
            
            # 第二层EMA(EMA1, 10) 
            ema2 = pd.Series(ema1).ewm(span=10).mean().values
            
            return ema2[-1]  # 返回最新值
            
        except Exception as e:
            self.logger.warning(f"计算知行短期趋势线失败: {e}")
            return None
    
    def calculate_zhixing_multiavg(self, close_prices: np.array, periods: List[int] = None) -> Optional[float]:
        """
        计算知行多空线: (MA(CLOSE,M1)+MA(CLOSE,M2)+MA(CLOSE,M3)+MA(CLOSE,M4))/4
        通达信公式: 知行多空线:(MA(CLOSE,M1)+MA(CLOSE,M2)+MA(CLOSE,M3)+MA(CLOSE,M4))/4;
        
        默认使用周期 [5, 10, 20, 60] 对应 M1, M2, M3, M4
        """
        try:
            if periods is None:
                periods = self.config["parameters"]["zhixing_multiavg_periods"]
                
            if len(close_prices) < max(periods):
                return None
                
            ma_values = []
            for period in periods:
                ma = pd.Series(close_prices).rolling(window=period).mean().iloc[-1]
                if not pd.isna(ma):
                    ma_values.append(ma)
            
            if len(ma_values) == len(periods):
                return sum(ma_values) / len(ma_values)
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"计算知行多空线失败: {e}")
            return None
    
    def calculate_zhixing_indicators_score(self, stock_data: pd.DataFrame) -> float:
        """
        计算知行指标得分
        
        正确的信号逻辑：
        1. 价格跌破多空线 → 买入信号（逆向思维，跌破是买入机会）
        2. 短期趋势线与多空线金叉 → 强买入信号
        3. 短期趋势线与多空线死叉 → 强卖出信号
        """
        if stock_data.empty:
            return 0.0
            
        scores = []
        close_prices = stock_data['close'].values
        latest_price = close_prices[-1]
        
        # 计算最新的知行指标值
        zhixing_trend = self.calculate_zhixing_trend(close_prices)
        zhixing_multiavg = self.calculate_zhixing_multiavg(close_prices)
        
        if zhixing_trend is None or zhixing_multiavg is None:
            return 0.5 * (self.config["weights"]["technical"]["zhixing_trend"] + 
                         self.config["weights"]["technical"]["zhixing_multiavg"])
        
        # 1. 检查价格跌破多空线信号 - 买入机会
        multiavg_ratio = latest_price / zhixing_multiavg if zhixing_multiavg > 0 else 1.0
        price_broken_multiavg = False
        if len(close_prices) >= 3:
            prev_multiavg = self.calculate_zhixing_multiavg(close_prices[:-1])
            if prev_multiavg is not None and prev_multiavg > 0:
                prev_price = close_prices[-2]
                # 判断是否从上方跌破多空线（买入信号）
                if prev_price >= prev_multiavg and latest_price < zhixing_multiavg:
                    price_broken_multiavg = True
        
        # 2. 检查趋势线与多空线的金叉死叉信号
        golden_cross = False  # 金叉：趋势线上穿多空线
        death_cross = False   # 死叉：趋势线下穿多空线
        
        if len(close_prices) >= 4:  # 需要足够数据判断交叉
            # 计算前一天的指标值
            prev_trend = self.calculate_zhixing_trend(close_prices[:-1])
            prev_multiavg = self.calculate_zhixing_multiavg(close_prices[:-1])
            
            if prev_trend is not None and prev_multiavg is not None:
                # 判断金叉：前一天趋势线<=多空线，今天趋势线>多空线
                if prev_trend <= prev_multiavg and zhixing_trend > zhixing_multiavg:
                    golden_cross = True
                # 判断死叉：前一天趋势线>=多空线，今天趋势线<多空线  
                elif prev_trend >= prev_multiavg and zhixing_trend < zhixing_multiavg:
                    death_cross = True
        
        # 3. 综合评分逻辑
        base_score = 0.5  # 基础分数
        
        # 强信号优先
        if golden_cross:
            # 金叉 - 最强买入信号
            final_score = 1.0
        elif death_cross:
            # 死叉 - 最强卖出信号
            final_score = 0.1
        elif price_broken_multiavg:
            # 价格跌破多空线 - 买入信号
            final_score = 0.85
        else:
            # 根据相对位置评分
            trend_multiavg_ratio = zhixing_trend / zhixing_multiavg if zhixing_multiavg > 0 else 1.0
            
            if trend_multiavg_ratio >= 1.03:  # 趋势线大幅高于多空线 3%+
                trend_score = 0.9
            elif trend_multiavg_ratio >= 1.01:  # 趋势线略高于多空线 1-3%
                trend_score = 0.7 + 0.2 * (trend_multiavg_ratio - 1.01) / 0.02
            elif trend_multiavg_ratio >= 0.99:  # 趋势线接近多空线 ±1%
                trend_score = 0.5 + 0.2 * (trend_multiavg_ratio - 0.99) / 0.02
            elif trend_multiavg_ratio >= 0.97:  # 趋势线略低于多空线 1-3%
                trend_score = 0.3 + 0.2 * (trend_multiavg_ratio - 0.97) / 0.02
            else:  # 趋势线大幅低于多空线 3%+
                trend_score = max(0.1, 0.3 * (trend_multiavg_ratio - 0.90) / 0.07)
                
            # 价格相对多空线的位置调整
            if multiavg_ratio <= 0.97:  # 价格低于多空线3%+ (潜在买入区域)
                price_adjustment = 0.1
            elif multiavg_ratio >= 1.05:  # 价格高于多空线5%+ (可能过热)
                price_adjustment = -0.05
            else:
                price_adjustment = 0
                
            final_score = min(1.0, max(0.05, trend_score + price_adjustment))
        
        # 分配到两个权重
        total_weight = (self.config["weights"]["technical"]["zhixing_trend"] + 
                       self.config["weights"]["technical"]["zhixing_multiavg"])
        
        return final_score * total_weight
    
    def _get_zhixing_signals(self, close_prices: np.array, zhixing_trend: float, zhixing_multiavg: float) -> Dict[str, Any]:
        """获取知行指标信号状态"""
        signals = {
            "golden_cross": False,
            "death_cross": False, 
            "price_broken_multiavg": False,
            "trend_above_multiavg": False,
            "signal_strength": "无信号"
        }
        
        if zhixing_trend is None or zhixing_multiavg is None:
            return signals
            
        latest_price = close_prices[-1]
        
        # 1. 检查价格跌破多空线信号
        if len(close_prices) >= 3:
            prev_multiavg = self.calculate_zhixing_multiavg(close_prices[:-1])
            if prev_multiavg is not None and prev_multiavg > 0:
                prev_price = close_prices[-2]
                if prev_price >= prev_multiavg and latest_price < zhixing_multiavg:
                    signals["price_broken_multiavg"] = True
                    signals["signal_strength"] = "买入信号"
        
        # 2. 检查趋势线与多空线的金叉死叉
        if len(close_prices) >= 4:
            prev_trend = self.calculate_zhixing_trend(close_prices[:-1])
            prev_multiavg = self.calculate_zhixing_multiavg(close_prices[:-1])
            
            if prev_trend is not None and prev_multiavg is not None:
                # 金叉：趋势线上穿多空线
                if prev_trend <= prev_multiavg and zhixing_trend > zhixing_multiavg:
                    signals["golden_cross"] = True
                    signals["signal_strength"] = "强买入信号"
                # 死叉：趋势线下穿多空线
                elif prev_trend >= prev_multiavg and zhixing_trend < zhixing_multiavg:
                    signals["death_cross"] = True
                    signals["signal_strength"] = "强卖出信号"
        
        # 3. 当前趋势线与多空线的相对位置
        signals["trend_above_multiavg"] = zhixing_trend > zhixing_multiavg
        
        # 4. 如果没有强信号，根据相对位置给出弱信号
        if signals["signal_strength"] == "无信号":
            multiavg_ratio = latest_price / zhixing_multiavg
            if multiavg_ratio <= 0.97:
                signals["signal_strength"] = "潜在买入区域"
            elif multiavg_ratio >= 1.05:
                signals["signal_strength"] = "可能过热区域"
            else:
                signals["signal_strength"] = "中性区域"
        
        return signals
    
    def detect_market_regime(self, date: str) -> Dict[str, Any]:
        """检测市场环境 - 保持v3.4成功的乘数机制"""
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
        """计算技术指标得分 - 集成知行指标"""
        if stock_data.empty:
            return 0.0
            
        scores = []
        latest = stock_data.iloc[-1]
        
        # 1. KDJ强度得分 - 权重从v3.4的0.18调整为0.16
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
        
        # KDJ综合评分
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
        
        # J值极值加成
        j_bonus = 0
        if kdj_j < 0:
            j_bonus = min(0.15, abs(kdj_j) / 100)
        elif kdj_j > 100:
            j_bonus = -min(0.1, (kdj_j - 100) / 200)
        
        kdj_score = max(0, min(1, kdj_score + j_bonus))
        scores.append(kdj_score * self.config["weights"]["technical"]["kdj_strength"])
        
        # 2. RSI动量得分 - 权重从v3.4的0.16调整为0.14
        rsi = latest.get('rsi', 50)
        if pd.isna(rsi):
            rsi = 50
        
        # RSI评分函数
        if rsi <= 20:
            rsi_score = 1.0
        elif rsi <= 28:
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
        
        # 3. BBI趋势得分 - 权重从v3.4的0.12调整为0.10
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
        
        # 4. 成交量异动得分 - 权重从v3.4的0.16调整为0.14
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
        
        # 5. 知行指标得分 - 新增部分 (总权重0.10 = 0.06 + 0.04)
        zhixing_score = self.calculate_zhixing_indicators_score(stock_data)
        scores.append(zhixing_score)
        
        return sum(scores)
    
    def calculate_fundamental_score(self, stock_data: pd.DataFrame, basic_data: Optional[pd.Series] = None) -> float:
        """计算基本面得分 - 权重微调"""
        scores = []
        
        if basic_data is None:
            # 默认中性得分 - 权重从v3.4微调
            return sum([
                0.5 * self.config["weights"]["fundamental"]["pe_valuation"],
                0.5 * self.config["weights"]["fundamental"]["pb_valuation"], 
                0.5 * self.config["weights"]["fundamental"]["roe_profitability"],
                0.5 * self.config["weights"]["fundamental"]["revenue_growth"],
                0.5 * self.config["weights"]["fundamental"]["market_cap"],
                0.5 * self.config["weights"]["fundamental"]["turnover_activity"]  # 权重从0.03调整为0.02
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
        
        # ROE盈利能力得分
        roe = basic_data.get('roe', 0.08) if basic_data is not None else 0.08
        if pd.isna(roe) or roe is None:
            roe_score = 0.5
        elif roe >= 0.20:
            roe_score = 1.0
        elif roe >= 0.15:
            roe_score = 0.8 + 0.2 * (roe - 0.15) / 0.05
        elif roe >= 0.08:
            roe_score = 0.5 + 0.3 * (roe - 0.08) / 0.07
        elif roe >= 0.03:
            roe_score = 0.2 + 0.3 * (roe - 0.03) / 0.05
        else:
            roe_score = max(0.05, 0.2 * max(roe, 0) / 0.03)
        
        scores.append(roe_score * self.config["weights"]["fundamental"]["roe_profitability"])
        
        # 营收增长得分
        revenue_growth_pct = basic_data.get('revenue_growth', 10.0) if basic_data is not None else 10.0
        if revenue_growth_pct is None or pd.isna(revenue_growth_pct):
            revenue_growth_pct = 10.0
        revenue_growth = revenue_growth_pct / 100.0
        if revenue_growth >= 0.30:
            growth_score = 1.0
        elif revenue_growth >= 0.20:
            growth_score = 0.8 + 0.2 * (revenue_growth - 0.20) / 0.10
        elif revenue_growth >= 0.10:
            growth_score = 0.5 + 0.3 * (revenue_growth - 0.10) / 0.10
        elif revenue_growth >= 0:
            growth_score = 0.3 + 0.2 * revenue_growth / 0.10
        else:
            growth_score = max(0.05, 0.3 * (1 + revenue_growth))
        
        scores.append(growth_score * self.config["weights"]["fundamental"]["revenue_growth"])
        
        # 市值因子得分
        market_cap = basic_data.get('total_mv', 100)
        if pd.isna(market_cap) or market_cap <= 0:
            cap_score = 0.5
        elif market_cap <= 50:
            cap_score = 1.0
        elif market_cap <= 200:
            cap_score = 0.7 + 0.3 * (200 - market_cap) / 150
        elif market_cap <= 1000:
            cap_score = 0.3 + 0.4 * (1000 - market_cap) / 800
        else:
            cap_score = max(0.1, 0.3 * (5000 - min(market_cap, 5000)) / 4000)
        
        scores.append(cap_score * self.config["weights"]["fundamental"]["market_cap"])
        
        # 换手率活跃度 - 权重从0.03调整为0.02
        turnover = basic_data.get('turnover_rate', 3.0)
        if pd.isna(turnover) or turnover <= 0:
            turnover_score = 0.3
        elif turnover <= 1.0:
            turnover_score = 0.2
        elif turnover <= 3.0:
            turnover_score = 0.5 + 0.3 * (turnover - 1.0) / 2.0
        elif turnover <= 8.0:
            turnover_score = 0.8 + 0.2 * (turnover - 3.0) / 5.0
        elif turnover <= 15.0:
            turnover_score = 0.6 + 0.2 * (15.0 - turnover) / 7.0
        else:
            turnover_score = max(0.2, 0.6 * (30 - min(turnover, 30)) / 15)
        
        scores.append(turnover_score * self.config["weights"]["fundamental"]["turnover_activity"])
        
        return sum(scores)
    
    def calculate_performance_score(self, stock_data: pd.DataFrame) -> float:
        """计算市场表现得分 - 权重从v3.4的18%调整为17%"""
        if stock_data.empty:
            return 0.0
            
        scores = []
        
        # 价格动量得分 - 权重从0.12调整为0.11
        if len(stock_data) >= 5:
            recent_returns = stock_data['price_change_pct'].tail(5).mean() / 100
            if recent_returns > 0.03:
                momentum_score = 1.0
            elif recent_returns > 0.01:
                momentum_score = 0.7 + 0.3 * (recent_returns - 0.01) / 0.02
            elif recent_returns > -0.01:
                momentum_score = 0.4 + 0.3 * (recent_returns + 0.01) / 0.02
            elif recent_returns > -0.03:
                momentum_score = 0.2 + 0.2 * (recent_returns + 0.03) / 0.02
            else:
                momentum_score = max(0.05, 0.2 * (recent_returns + 0.05) / 0.02)
        else:
            momentum_score = 0.3
            
        scores.append(momentum_score * self.config["weights"]["performance"]["price_momentum"])
        
        # 相对强度得分
        latest_change = stock_data['price_change_pct'].iloc[-1] / 100 if not stock_data.empty else 0
        if latest_change > 0.02:
            relative_score = 1.0
        elif latest_change > 0.005:
            relative_score = 0.7 + 0.3 * (latest_change - 0.005) / 0.015
        elif latest_change > -0.005:
            relative_score = 0.4 + 0.3 * (latest_change + 0.005) / 0.01
        elif latest_change > -0.02:
            relative_score = 0.2 + 0.2 * (latest_change + 0.02) / 0.015
        else:
            relative_score = max(0.05, 0.2 * (latest_change + 0.05) / 0.03)
            
        scores.append(relative_score * self.config["weights"]["performance"]["relative_strength"])
        
        # 波动率风险得分
        if len(stock_data) >= 20:
            volatility = stock_data['price_change_pct'].tail(20).std() / 100
            if volatility <= 0.02:
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
        
        # 市场贝塔得分
        beta_score = 0.5
        scores.append(beta_score * self.config["weights"]["market_regime"]["market_beta"])
        
        # 板块轮动得分
        rotation_score = 0.5 if market_info["regime"] == "neutral" else 0.7
        scores.append(rotation_score * self.config["weights"]["market_regime"]["sector_rotation"])
        
        # 流动性因子得分
        liquidity_score = 0.6 if market_info["volatility"] == "normal" else 0.4
        scores.append(liquidity_score * self.config["weights"]["market_regime"]["liquidity"])
        
        return sum(scores)
    
    def calculate_quantitative_score(self, stock_code: str, date: str, 
                                   stock_data: pd.DataFrame = None, 
                                   basic_data: pd.Series = None) -> Dict[str, Any]:
        """计算量化评分 - v3.5版本，集成知行指标"""
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
            
            # 应用市场环境乘数
            final_score = base_score * market_info["multiplier"]
            
            # 应用评分优化
            if self.config.get("scoring_optimization", {}).get("high_score_threshold_adjustment"):
                adjustment = self.config["scoring_optimization"]["high_score_threshold_adjustment"]
                if final_score >= 0.85:
                    final_score = final_score * (1 + adjustment / 100)
            
            # 标准化到0-100分制
            # final_score的理论最大值是0.898 (89.8%权重总和)
            # 将其线性映射到0-100分，只需要保证不小于0
            quantitative_score = max(0, (final_score / 0.898) * 100)
            
            # 计算知行指标值和信号状态
            close_prices = stock_data['close'].values
            latest_price = close_prices[-1]
            zhixing_trend = self.calculate_zhixing_trend(close_prices)
            zhixing_multiavg = self.calculate_zhixing_multiavg(close_prices)
            
            # 判断知行指标信号
            zhixing_signals = self._get_zhixing_signals(close_prices, zhixing_trend, zhixing_multiavg)
            
            return {
                "stock_code": stock_code,
                "date": date,
                "quantitative_score": round(quantitative_score, 1),
                "technical_score": round(max(0, (technical_score / 0.60) * 100), 1),  # 技术评分标准化(权重60%)
                "fundamental_score": round(max(0, (fundamental_score / 0.138) * 100), 1),  # 基本面标准化(权重13.8%)
                "performance_score": round(max(0, (performance_score / 0.13) * 100), 1),   # 表现标准化(权重13%)
                "market_regime_score": round(max(0, (market_regime_score / 0.03) * 100), 1),  # 市场环境标准化(权重3%)
                "market_regime": market_info["regime"],
                "market_multiplier": market_info["multiplier"],
                "zhixing_trend": round(zhixing_trend, 4) if zhixing_trend else None,
                "zhixing_multiavg": round(zhixing_multiavg, 4) if zhixing_multiavg else None,
                "zhixing_signals": zhixing_signals,
                "zhixing_score": round(max(0, (self.calculate_zhixing_indicators_score(stock_data) / 0.20) * 100), 1),  # 知行指标得分(权重20%)
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
        """获取基本面数据"""
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