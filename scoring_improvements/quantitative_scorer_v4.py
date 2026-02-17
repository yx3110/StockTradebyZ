#!/usr/bin/env python3
"""
量化评分系统 v4.0
基于v3.0系统，集成挤压动量指标(Squeeze Momentum Indicator)

主要改进：
1. 新增挤压动量维度评分 (20%权重)
2. 优化技术指标权重分配 (从65%降至50%)
3. 增强突破预测能力
4. 提高低波动到高波动转换点识别精度
5. 改进假突破过滤机制
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
from squeeze_momentum_calculator import SqueezeMomentumCalculator

class QuantitativeScorerV4:
    """量化评分系统 v4.0 - 集成挤压动量指标"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化评分系统"""
        self.version = "v4.0"
        self.db_manager = DatabaseManager()
        self.squeeze_calculator = SqueezeMomentumCalculator()
        
        # 默认配置 - v4.0 挤压动量增强版
        self.default_config = {
            "version": "v4.0-SqueezeMomentum",
            "weights": {
                # 技术指标权重 (50%) - 从v3的65%降低
                "technical": {
                    "kdj_strength": 0.15,     # KDJ强度 - 降低权重
                    "rsi_momentum": 0.14,     # RSI动量 - 降低权重
                    "bbi_trend": 0.10,        # BBI趋势 - 降低权重
                    "volume_surge": 0.11      # 成交量异动 - 降低权重
                },
                # 🆕 挤压动量权重 (20%) - 新增维度
                "squeeze_momentum": {
                    "squeeze_state": 0.05,        # 挤压状态
                    "squeeze_release": 0.06,      # 挤压释放信号
                    "momentum_direction": 0.05,   # 动量方向
                    "momentum_acceleration": 0.04 # 动量加速度
                },
                # 基本面权重 (8%) - 从v3的10%降低
                "fundamental": {
                    "pe_valuation": 0.02,     # PE估值
                    "pb_valuation": 0.02,     # PB估值
                    "market_cap": 0.02,       # 市值因子
                    "turnover_activity": 0.02  # 换手率活跃度
                },
                # 市场表现权重 (18%) - 从v3的20%降低
                "performance": {
                    "price_momentum": 0.13,   # 价格动量
                    "relative_strength": 0.03, # 相对强度
                    "volatility_risk": 0.02   # 波动率风险
                },
                # 市场环境权重 (4%) - 从v3的5%降低
                "market_regime": {
                    "market_beta": 0.01,      # 市场贝塔
                    "sector_rotation": 0.015,  # 板块轮动
                    "liquidity": 0.015        # 流动性因子
                }
            },
            "parameters": {
                "lookback_periods": [5, 10, 20, 30],  # 多时间窗口
                "kdj_threshold": 20,                   # KDJ超卖阈值
                "rsi_oversold": 30,                   # RSI超卖阈值
                "volume_multiplier": 2.0,             # 成交量倍数阈值
                "volatility_window": 20,              # 波动率计算窗口
                "beta_window": 60,                    # 贝塔计算窗口
                # 🆕 挤压动量参数
                "squeeze_bb_length": 20,              # 布林带周期
                "squeeze_bb_multiplier": 2.0,         # 布林带倍数
                "squeeze_kc_length": 20,              # 肯特纳通道周期
                "squeeze_kc_multiplier": 1.5,         # 肯特纳通道倍数
                "squeeze_momentum_length": 20         # 动量计算周期
            },
            "market_regime": {
                "bull_threshold": 0.02,    # 牛市阈值(日均涨幅)
                "bear_threshold": -0.02,   # 熊市阈值
                "volatility_high": 0.03    # 高波动阈值
            },
            # 🆕 挤压动量评分参数
            "squeeze_scoring": {
                "squeeze_bonus": 25,          # 挤压状态奖励分
                "release_bonus": 40,          # 挤压释放奖励分
                "momentum_multiplier": 100,   # 动量强度倍数
                "acceleration_multiplier": 50, # 加速度倍数
                "consistency_bonus": 20       # 一致性奖励分
            }
        }
        
        # 加载配置
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                custom_config = json.load(f)
                self.config = self._merge_configs(self.default_config, custom_config)
        else:
            self.config = self.default_config.copy()
            
        # 更新挤压动量计算器参数
        self.squeeze_calculator = SqueezeMomentumCalculator(
            bb_length=self.config["parameters"]["squeeze_bb_length"],
            bb_multiplier=self.config["parameters"]["squeeze_bb_multiplier"],
            kc_length=self.config["parameters"]["squeeze_kc_length"],
            kc_multiplier=self.config["parameters"]["squeeze_kc_multiplier"],
            momentum_length=self.config["parameters"]["squeeze_momentum_length"]
        )
            
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
    
    def get_stock_data(self, stock_code: str, date: str, days: int = 60) -> pd.DataFrame:
        """获取股票数据"""
        try:
            query = """
            SELECT dq.trade_date, dq.open, dq.high, dq.low, dq.close, dq.volume,
                   dq.price_change_pct, ti.kdj_k, ti.kdj_d, ti.kdj_j, ti.rsi12 as rsi, 
                   dq.ma5, dq.ma10, dq.ma20, ti.bbi, db.pe_ttm, db.pb, 
                   db.circ_mv as market_cap, db.turnover_rate
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            LEFT JOIN technical_indicators ti ON dq.security_id = ti.security_id 
                AND dq.trade_date = ti.trade_date
            LEFT JOIN daily_basic db ON dq.security_id = db.security_id 
                AND dq.trade_date = db.trade_date
            WHERE s.code = ? AND dq.trade_date <= ?
            ORDER BY dq.trade_date DESC
            LIMIT ?
            """
            
            with self.db_manager.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=[stock_code, date, days])
            
            if df.empty:
                self.logger.warning(f"未找到股票 {stock_code} 在 {date} 的数据")
                return pd.DataFrame()
                
            # 转换数据类型
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'price_change_pct',
                             'kdj_k', 'kdj_d', 'kdj_j', 'rsi', 'ma5', 'ma10', 'ma20', 'bbi',
                             'pe_ttm', 'pb', 'market_cap', 'turnover_rate']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            self.logger.error(f"获取股票数据失败: {e}")
            return pd.DataFrame()
    
    def calculate_squeeze_momentum_score(self, stock_data: pd.DataFrame) -> float:
        """🆕 计算挤压动量得分"""
        try:
            if stock_data.empty or len(stock_data) < 30:
                return 0.0
            
            # 准备OHLC数据
            high_prices = stock_data['high'].ffill()
            low_prices = stock_data['low'].ffill()
            close_prices = stock_data['close'].ffill()
            
            # 计算挤压动量指标
            indicators = self.squeeze_calculator.calculate_squeeze_momentum_indicators(
                high_prices, low_prices, close_prices
            )
            
            if not indicators:
                return 0.0
            
            # 获取当前信号
            signals = self.squeeze_calculator.get_current_signals(indicators)
            
            # 计算各子因子得分
            scores = []
            
            # 1. 挤压状态得分 (0-25分)
            squeeze_score = 0
            if signals['is_squeezed']:
                squeeze_score += self.config["squeeze_scoring"]["squeeze_bonus"]
                # 长期挤压额外奖励
                if signals['squeeze_days'] > 10:
                    squeeze_score += 10
            
            if signals['just_released']:
                squeeze_score += self.config["squeeze_scoring"]["release_bonus"]
            
            scores.append(min(100, squeeze_score) * self.config["weights"]["squeeze_momentum"]["squeeze_state"])
            
            # 2. 挤压释放得分 (0-30分)
            release_score = 0
            if signals['just_released']:
                release_score = 100  # 刚释放，满分
            elif signals['recent_releases'] > 0:
                release_score = 60   # 近期有释放
            elif signals['is_squeezed'] and signals['squeeze_days'] > 5:
                release_score = 40   # 长期挤压，蓄势待发
            
            scores.append(release_score * self.config["weights"]["squeeze_momentum"]["squeeze_release"])
            
            # 3. 动量方向得分 (0-25分)
            momentum_score = 50  # 基础分
            if signals['momentum_direction'] > 0:
                momentum_score += min(50, signals['momentum_strength'] * 
                                    self.config["squeeze_scoring"]["momentum_multiplier"])
            else:
                momentum_score -= min(50, signals['momentum_strength'] * 
                                    self.config["squeeze_scoring"]["momentum_multiplier"])
            
            momentum_score = max(0, min(100, momentum_score))
            scores.append(momentum_score * self.config["weights"]["squeeze_momentum"]["momentum_direction"])
            
            # 4. 动量加速度得分 (0-20分)
            acceleration_score = 50  # 基础分
            if signals['momentum_acceleration'] > 0:
                acceleration_score += min(50, abs(signals['momentum_acceleration']) * 
                                        self.config["squeeze_scoring"]["acceleration_multiplier"])
            else:
                acceleration_score -= min(50, abs(signals['momentum_acceleration']) * 
                                        self.config["squeeze_scoring"]["acceleration_multiplier"])
            
            # 一致性奖励
            if signals['momentum_consistency'] > 0.8:
                acceleration_score += self.config["squeeze_scoring"]["consistency_bonus"]
            
            acceleration_score = max(0, min(100, acceleration_score))
            scores.append(acceleration_score * self.config["weights"]["squeeze_momentum"]["momentum_acceleration"])
            
            total_score = sum(scores)
            
            self.logger.debug(f"挤压动量得分详情: 挤压={squeeze_score}, 释放={release_score}, "
                            f"动量={momentum_score}, 加速度={acceleration_score}, 总分={total_score}")
            
            return total_score
            
        except Exception as e:
            self.logger.error(f"计算挤压动量得分失败: {e}")
            return 0.0
    
    def calculate_technical_score(self, stock_data: pd.DataFrame) -> float:
        """计算技术指标得分 (继承v3逻辑，调整权重)"""
        try:
            if stock_data.empty:
                return 0.0
                
            latest = stock_data.iloc[0]  # 最新数据
            scores = []
            
            # 1. KDJ强度得分
            kdj_k = latest.get('kdj_k', 50)
            kdj_d = latest.get('kdj_d', 50) 
            kdj_j = latest.get('kdj_j', 50)
            
            # 处理NaN值
            if pd.isna(kdj_k): kdj_k = 50
            if pd.isna(kdj_d): kdj_d = 50
            if pd.isna(kdj_j): kdj_j = 50
            
            kdj_combined = (kdj_k + kdj_d + kdj_j) / 3
            
            if kdj_combined <= 20:
                kdj_score = 100
            elif kdj_combined <= 30:
                kdj_score = 90 + (30 - kdj_combined) / 10 * 10
            elif kdj_combined <= 50:
                kdj_score = 50 + (50 - kdj_combined) / 20 * 40
            else:
                kdj_score = max(0, 50 - (kdj_combined - 50) / 50 * 50)
            
            scores.append(kdj_score * self.config["weights"]["technical"]["kdj_strength"])
            
            # 2. RSI动量得分
            rsi = latest.get('rsi', 50)
            if pd.isna(rsi): rsi = 50
            
            if rsi <= 30:
                rsi_score = 100
            elif rsi <= 40:
                rsi_score = 80 + (40 - rsi) / 10 * 20
            elif rsi <= 60:
                rsi_score = 50 + (50 - abs(rsi - 50)) / 10 * 30
            else:
                rsi_score = max(0, 50 - (rsi - 50) / 50 * 50)
            
            scores.append(rsi_score * self.config["weights"]["technical"]["rsi_momentum"])
            
            # 3. BBI趋势得分
            close = latest.get('close', 0)
            bbi = latest.get('bbi', close)
            
            if pd.isna(bbi): bbi = close
            if close > 0 and bbi > 0:
                bbi_ratio = close / bbi
                if bbi_ratio >= 1.05:
                    bbi_score = 100
                elif bbi_ratio >= 1.02:
                    bbi_score = 80
                elif bbi_ratio >= 1.0:
                    bbi_score = 60
                elif bbi_ratio >= 0.98:
                    bbi_score = 40
                else:
                    bbi_score = 20
            else:
                bbi_score = 50
            
            scores.append(bbi_score * self.config["weights"]["technical"]["bbi_trend"])
            
            # 4. 成交量异动得分
            if len(stock_data) >= 5:
                recent_volume = stock_data['volume'].head(5).mean()
                historical_volume = stock_data['volume'].tail(20).mean()
                
                if historical_volume > 0:
                    volume_ratio = recent_volume / historical_volume
                    if volume_ratio >= 3:
                        volume_score = 100
                    elif volume_ratio >= 2:
                        volume_score = 80
                    elif volume_ratio >= 1.5:
                        volume_score = 60
                    else:
                        volume_score = 40
                else:
                    volume_score = 50
            else:
                volume_score = 50
            
            scores.append(volume_score * self.config["weights"]["technical"]["volume_surge"])
            
            return sum(scores)
            
        except Exception as e:
            self.logger.error(f"计算技术指标得分失败: {e}")
            return 0.0
    
    def calculate_fundamental_score(self, stock_data: pd.DataFrame) -> float:
        """计算基本面得分 (继承v3逻辑，调整权重)"""
        try:
            if stock_data.empty:
                return 0.0
                
            latest = stock_data.iloc[0]
            scores = []
            
            # 1. PE估值得分
            pe = latest.get('pe_ttm', None)
            if pd.isna(pe) or pe is None or pe <= 0:
                pe_score = 50
            elif pe <= 15:
                pe_score = 100
            elif pe <= 25:
                pe_score = 80
            elif pe <= 40:
                pe_score = 60
            elif pe <= 60:
                pe_score = 40
            else:
                pe_score = 20
            
            scores.append(pe_score * self.config["weights"]["fundamental"]["pe_valuation"])
            
            # 2. PB估值得分
            pb = latest.get('pb', None)
            if pd.isna(pb) or pb is None or pb <= 0:
                pb_score = 50
            elif pb <= 1:
                pb_score = 100
            elif pb <= 2:
                pb_score = 80
            elif pb <= 3:
                pb_score = 60
            elif pb <= 5:
                pb_score = 40
            else:
                pb_score = 20
            
            scores.append(pb_score * self.config["weights"]["fundamental"]["pb_valuation"])
            
            # 3. 市值因子得分 (小市值偏好)
            market_cap = latest.get('market_cap', None)
            if pd.isna(market_cap) or market_cap is None:
                cap_score = 50
            elif market_cap <= 50:
                cap_score = 100  # 小市值
            elif market_cap <= 100:
                cap_score = 80
            elif market_cap <= 300:
                cap_score = 60
            elif market_cap <= 1000:
                cap_score = 40
            else:
                cap_score = 20   # 大市值
            
            scores.append(cap_score * self.config["weights"]["fundamental"]["market_cap"])
            
            # 4. 换手率活跃度得分
            turnover = latest.get('turnover_rate', None)
            if pd.isna(turnover) or turnover is None:
                turnover_score = 50
            elif turnover >= 10:
                turnover_score = 100
            elif turnover >= 5:
                turnover_score = 80
            elif turnover >= 2:
                turnover_score = 60
            elif turnover >= 1:
                turnover_score = 40
            else:
                turnover_score = 20
            
            scores.append(turnover_score * self.config["weights"]["fundamental"]["turnover_activity"])
            
            return sum(scores)
            
        except Exception as e:
            self.logger.error(f"计算基本面得分失败: {e}")
            return 0.0
    
    def calculate_performance_score(self, stock_data: pd.DataFrame) -> float:
        """计算市场表现得分 (继承v3逻辑，调整权重)"""
        try:
            if stock_data.empty or len(stock_data) < 5:
                return 0.0
            
            scores = []
            
            # 价格动量得分
            latest_close = stock_data.iloc[0]['close']
            
            # 5日动量
            if len(stock_data) >= 5:
                momentum_5d = (latest_close / stock_data.iloc[4]['close'] - 1) * 100
            else:
                momentum_5d = 0
            
            # 动量得分转换
            if momentum_5d >= 5:
                momentum_score = 100
            elif momentum_5d >= 2:
                momentum_score = 80
            elif momentum_5d >= 0:
                momentum_score = 60
            elif momentum_5d >= -2:
                momentum_score = 40
            else:
                momentum_score = 20
            
            scores.append(momentum_score * self.config["weights"]["performance"]["price_momentum"])
            
            # 相对强度和波动率风险 (简化处理)
            scores.append(50 * self.config["weights"]["performance"]["relative_strength"])
            scores.append(50 * self.config["weights"]["performance"]["volatility_risk"])
            
            return sum(scores)
            
        except Exception as e:
            self.logger.error(f"计算市场表现得分失败: {e}")
            return 0.0
    
    def calculate_market_regime_score(self, date: str) -> float:
        """计算市场环境得分 (继承v3逻辑，调整权重)"""
        try:
            # 简化的市场环境评分
            scores = []
            
            # 市场贝塔得分
            scores.append(50 * self.config["weights"]["market_regime"]["market_beta"])
            
            # 板块轮动得分
            scores.append(50 * self.config["weights"]["market_regime"]["sector_rotation"])
            
            # 流动性得分
            scores.append(50 * self.config["weights"]["market_regime"]["liquidity"])
            
            return sum(scores)
            
        except Exception as e:
            self.logger.error(f"计算市场环境得分失败: {e}")
            return 0.0
    
    def calculate_comprehensive_score(self, stock_code: str, date: str) -> Dict[str, Any]:
        """计算综合评分"""
        try:
            # 获取股票数据
            stock_data = self.get_stock_data(stock_code, date)
            
            if stock_data.empty:
                return {
                    "stock_code": stock_code,
                    "date": date,
                    "total_score": 0,
                    "grade": "F",
                    "error": "无法获取股票数据"
                }
            
            # 计算各维度得分
            technical_score = self.calculate_technical_score(stock_data)
            squeeze_score = self.calculate_squeeze_momentum_score(stock_data)  # 🆕
            fundamental_score = self.calculate_fundamental_score(stock_data)
            performance_score = self.calculate_performance_score(stock_data)
            market_score = self.calculate_market_regime_score(date)
            
            # 计算总分
            total_score = (
                technical_score + 
                squeeze_score +      # 🆕 挤压动量得分
                fundamental_score + 
                performance_score + 
                market_score
            )
            
            # 评级
            if total_score >= 90:
                grade = "A+"
            elif total_score >= 85:
                grade = "A"
            elif total_score >= 80:
                grade = "A-"
            elif total_score >= 75:
                grade = "B+"
            elif total_score >= 70:
                grade = "B"
            elif total_score >= 65:
                grade = "B-"
            elif total_score >= 60:
                grade = "C+"
            elif total_score >= 55:
                grade = "C"
            elif total_score >= 50:
                grade = "C-"
            elif total_score >= 40:
                grade = "D"
            else:
                grade = "F"
            
            return {
                "stock_code": stock_code,
                "date": date,
                "total_score": round(total_score, 2),
                "grade": grade,
                "breakdown": {
                    "technical": round(technical_score, 2),
                    "squeeze_momentum": round(squeeze_score, 2),  # 🆕
                    "fundamental": round(fundamental_score, 2),
                    "performance": round(performance_score, 2),
                    "market_regime": round(market_score, 2)
                },
                "version": self.version
            }
            
        except Exception as e:
            self.logger.error(f"计算综合评分失败: {e}")
            return {
                "stock_code": stock_code,
                "date": date,
                "total_score": 0,
                "grade": "F",
                "error": str(e)
            }


def test_v4_scorer():
    """测试v4评分系统"""
    scorer = QuantitativeScorerV4()
    
    # 从数据库获取实际存在的股票
    with scorer.db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT s.code, s.name 
            FROM securities s
            JOIN daily_quotes dq ON s.id = dq.security_id
            WHERE s.type = 'A股'
            ORDER BY RANDOM()
            LIMIT 5
        """)
        stocks_data = cursor.fetchall()
        
        # 获取最新交易日期
        cursor.execute("""
            SELECT MAX(trade_date) as latest_date
            FROM daily_quotes
        """)
        result = cursor.fetchone()
        latest_date = result[0] if result and result[0] else "2025-08-14"
    
    test_stocks = stocks_data  # 保留股票代码和名称
    test_date = latest_date
    
    print(f"=== V4评分系统测试 ===")
    print(f"测试日期: {test_date}")
    print(f"版本: {scorer.version}")
    
    for stock_code, stock_name in test_stocks:
        print(f"\n--- 测试股票: {stock_code} {stock_name} ---")
        result = scorer.calculate_comprehensive_score(stock_code, test_date)
        
        if "error" in result:
            print(f"错误: {result['error']}")
            continue
            
        print(f"总分: {result['total_score']} ({result['grade']})")
        print("得分细分:")
        for dimension, score in result['breakdown'].items():
            print(f"  {dimension}: {score}")


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_v4_scorer()