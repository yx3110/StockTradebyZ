#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.1优化权重量化评分器

基于214万条历史数据优化的权重配置：
- Performance: 38.2% (表现因子最重要)
- Risk Control: 35.2% (风险控制次重要) 
- Fundamental: 25.1% (基本面适中权重)
- Technical: 0.9% (技术指标权重极低)
- Sentiment: 0.6% (情绪因子权重极低)
- Market Regime: 乘数因子 (0.360 - 1.296倍动态调整)

相比V3等权重配置，V3.1在所有持有期都表现更优，特别是10天和20天持有期
改进幅度达75%+和127%+。
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import time
import warnings

warnings.filterwarnings('ignore')

class QuantitativeScorerV31Optimized:
    """V3.1优化权重量化评分器"""
    
    def __init__(self, stock_db_path: str = "data_adapter/stock_data.db"):
        self.stock_db_path = stock_db_path
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # V3.1优化权重配置（基于214万条数据优化）
        self.optimized_weights = {
            'technical': 0.0088,      # 0.9% - 技术指标权重极低
            'fundamental': 0.2511,    # 25.1% - 基本面适中权重
            'performance': 0.3823,    # 38.2% - 表现因子最重要
            'sentiment': 0.0055,      # 0.6% - 情绪因子权重极低
            'risk_control': 0.3522    # 35.2% - 风险控制次重要
        }
        
        # 市场环境乘数配置
        self.market_regime_config = {
            'min_score': 0.3472,      # 最小市场环境得分
            'max_score': 0.8905,      # 最大市场环境得分  
            'min_multiplier': 0.360,  # 熊市乘数(压缩64%)
            'max_multiplier': 1.296   # 牛市乘数(放大30%)
        }
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def calculate_market_regime_multiplier(self, market_regime_score: float) -> float:
        """
        计算市场环境乘数
        
        将市场环境得分转换为乘数因子：
        - 熊市(低分): 乘数0.360, 压缩股票得分至36%
        - 牛市(高分): 乘数1.296, 放大股票得分至130%
        - 动态范围: 3.6倍调整能力
        """
        try:
            min_score = self.market_regime_config['min_score']
            max_score = self.market_regime_config['max_score']
            min_mult = self.market_regime_config['min_multiplier']
            max_mult = self.market_regime_config['max_multiplier']
            
            # 线性映射到乘数范围
            multiplier = min_mult + (market_regime_score - min_score) / (max_score - min_score) * (max_mult - min_mult)
            
            # 确保在有效范围内
            multiplier = max(min_mult, min(max_mult, multiplier))
            
            return multiplier
            
        except Exception as e:
            self.logger.error(f"计算市场环境乘数失败: {e}")
            return 1.0  # 默认中性乘数
    
    def calculate_quality_score(self, technical: float, fundamental: float, performance: float,
                               sentiment: float, risk_control: float) -> float:
        """
        计算个股质量得分
        
        使用优化权重计算5个质量因子的加权得分
        """
        try:
            quality_score = (
                technical * self.optimized_weights['technical'] +
                fundamental * self.optimized_weights['fundamental'] + 
                performance * self.optimized_weights['performance'] +
                sentiment * self.optimized_weights['sentiment'] +
                risk_control * self.optimized_weights['risk_control']
            )
            
            return quality_score
            
        except Exception as e:
            self.logger.error(f"计算质量得分失败: {e}")
            return 0.0
    
    def calculate_final_score(self, technical: float, fundamental: float, performance: float,
                             sentiment: float, risk_control: float, market_regime: float) -> float:
        """
        计算最终V3.1优化得分
        
        公式: final_score = market_regime_multiplier × quality_score
        其中:
        - market_regime_multiplier = 0.360 + (market_regime - 0.3472) / (0.8905 - 0.3472) * (1.296 - 0.360)
        - quality_score = performance×0.382 + risk_control×0.352 + fundamental×0.251 + technical×0.009 + sentiment×0.006
        """
        try:
            # 计算个股质量得分
            quality_score = self.calculate_quality_score(
                technical, fundamental, performance, sentiment, risk_control
            )
            
            # 计算市场环境乘数
            market_multiplier = self.calculate_market_regime_multiplier(market_regime)
            
            # 最终得分 = 市场环境乘数 × 个股质量得分
            final_score = market_multiplier * quality_score
            
            return final_score
            
        except Exception as e:
            self.logger.error(f"计算最终得分失败: {e}")
            return 0.0
    
    def get_stock_data_for_scoring(self, code: str, date_str: str) -> Optional[pd.Series]:
        """获取用于评分的股票数据"""
        try:
            query = """
            SELECT 
                s.code,
                dq.trade_date,
                dq.close,
                dq.price_change_pct,
                db.pe_ttm,
                db.pb,
                db.ps_ttm,
                db.total_mv as market_cap,
                db.turnover_rate,
                ti.ma_5,
                ti.ma_10,
                ti.ma_20,
                ti.rsi,
                ti.macd,
                ti.kdj_k,
                ti.kdj_d,
                ti.kdj_j,
                ti.bbi
            FROM securities s
            JOIN daily_quotes dq ON s.id = dq.security_id
            LEFT JOIN daily_basic db ON s.id = db.security_id AND db.trade_date = dq.trade_date
            LEFT JOIN technical_indicators ti ON s.id = ti.security_id AND ti.trade_date = dq.trade_date
            WHERE s.code = ? AND dq.trade_date = ?
            """
            
            with sqlite3.connect(self.stock_db_path) as conn:
                result = pd.read_sql_query(query, conn, params=[code, date_str])
            
            if not result.empty:
                return result.iloc[0]
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"获取股票 {code} 在 {date_str} 的数据失败: {e}")
            return None
    
    def calculate_technical_score(self, stock_data: pd.Series) -> float:
        """计算技术指标得分"""
        try:
            score = 0.0
            
            # 移动平均线趋势(30%)
            close = stock_data.get('close', 0)
            ma_5 = stock_data.get('ma_5', 0)
            ma_10 = stock_data.get('ma_10', 0)
            ma_20 = stock_data.get('ma_20', 0)
            
            if pd.notna(close) and pd.notna(ma_5) and pd.notna(ma_10) and pd.notna(ma_20):
                if close > ma_5 > ma_10 > ma_20:
                    score += 0.3
                elif close > ma_5 > ma_10:
                    score += 0.2
                elif close > ma_5:
                    score += 0.1
            
            # RSI指标(25%)
            rsi = stock_data.get('rsi', 50)
            if pd.notna(rsi):
                if 30 <= rsi <= 70:
                    score += 0.25
                elif 20 <= rsi < 30 or 70 < rsi <= 80:
                    score += 0.15
            
            # KDJ指标(25%)
            kdj_k = stock_data.get('kdj_k', 50)
            kdj_d = stock_data.get('kdj_d', 50)
            if pd.notna(kdj_k) and pd.notna(kdj_d):
                if kdj_k > kdj_d and kdj_k < 80:
                    score += 0.25
                elif kdj_k > kdj_d:
                    score += 0.15
            
            # MACD指标(20%)
            macd = stock_data.get('macd', 0)
            if pd.notna(macd) and macd > 0:
                score += 0.2
            
            return min(1.0, score)
            
        except Exception as e:
            self.logger.error(f"计算技术得分失败: {e}")
            return 0.5
    
    def calculate_fundamental_score(self, stock_data: pd.Series) -> float:
        """计算基本面得分"""
        try:
            score = 0.0
            
            # PE估值(40%)
            pe_ttm = stock_data.get('pe_ttm', 0)
            if pd.notna(pe_ttm) and pe_ttm > 0:
                if 10 <= pe_ttm <= 25:
                    score += 0.4
                elif 5 <= pe_ttm < 10 or 25 < pe_ttm <= 35:
                    score += 0.3
                elif pe_ttm < 50:
                    score += 0.2
            
            # PB估值(30%)  
            pb = stock_data.get('pb', 0)
            if pd.notna(pb) and pb > 0:
                if 1 <= pb <= 3:
                    score += 0.3
                elif 0.5 <= pb < 1 or 3 < pb <= 5:
                    score += 0.2
                elif pb < 8:
                    score += 0.1
            
            # PS估值(30%)
            ps_ttm = stock_data.get('ps_ttm', 0)
            if pd.notna(ps_ttm) and ps_ttm > 0:
                if ps_ttm <= 3:
                    score += 0.3
                elif ps_ttm <= 6:
                    score += 0.2
                elif ps_ttm <= 10:
                    score += 0.1
            
            return min(1.0, score)
            
        except Exception as e:
            self.logger.error(f"计算基本面得分失败: {e}")
            return 0.5
    
    def calculate_performance_score(self, code: str, date_str: str) -> float:
        """计算表现得分"""
        try:
            # 获取历史价格数据计算收益率
            query = """
            SELECT trade_date, close, price_change_pct
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.code = ? AND trade_date <= ?
            ORDER BY trade_date DESC
            LIMIT 60
            """
            
            with sqlite3.connect(self.stock_db_path) as conn:
                df = pd.read_sql_query(query, conn, params=[code, date_str])
            
            if len(df) < 20:
                return 0.5
            
            score = 0.0
            
            # 短期表现(5日，40%)
            short_returns = df['price_change_pct'].head(5).mean() if len(df) >= 5 else 0
            if pd.notna(short_returns):
                short_returns_normalized = short_returns / 100
                if short_returns_normalized > 0.02:
                    score += 0.4
                elif short_returns_normalized > 0:
                    score += 0.2
            
            # 中期表现(20日，35%)
            medium_returns = df['price_change_pct'].head(20).mean() if len(df) >= 20 else 0
            if pd.notna(medium_returns):
                medium_returns_normalized = medium_returns / 100
                if medium_returns_normalized > 0.01:
                    score += 0.35
                elif medium_returns_normalized > 0:
                    score += 0.2
            
            # 长期表现(60日，25%)
            long_returns = df['price_change_pct'].head(60).mean() if len(df) >= 60 else 0
            if pd.notna(long_returns):
                long_returns_normalized = long_returns / 100  
                if long_returns_normalized > 0.005:
                    score += 0.25
                elif long_returns_normalized > 0:
                    score += 0.15
            
            return min(1.0, score)
            
        except Exception as e:
            self.logger.error(f"计算表现得分失败: {e}")
            return 0.5
    
    def calculate_sentiment_score(self, code: str, date_str: str) -> float:
        """计算情绪得分（简化版本）"""
        try:
            # 基于换手率和成交量的市场情绪
            stock_data = self.get_stock_data_for_scoring(code, date_str)
            if stock_data is None:
                return 0.5
            
            score = 0.0
            
            # 换手率情绪(60%)
            turnover_rate = stock_data.get('turnover_rate', 0)
            if pd.notna(turnover_rate):
                if 2 <= turnover_rate <= 8:
                    score += 0.6
                elif 1 <= turnover_rate < 2 or 8 < turnover_rate <= 15:
                    score += 0.4
                elif turnover_rate < 20:
                    score += 0.2
            
            # 价格动能情绪(40%)
            price_change = stock_data.get('price_change_pct', 0)
            if pd.notna(price_change):
                if 0 < price_change <= 5:
                    score += 0.4
                elif -2 <= price_change <= 0:
                    score += 0.2
                elif price_change > 5:
                    score += 0.3
            
            return min(1.0, score)
            
        except Exception as e:
            self.logger.error(f"计算情绪得分失败: {e}")
            return 0.5
    
    def calculate_risk_control_score(self, code: str, date_str: str) -> float:
        """计算风险控制得分"""
        try:
            # 获取历史数据计算波动率
            query = """
            SELECT trade_date, close, price_change_pct
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.code = ? AND trade_date <= ?
            ORDER BY trade_date DESC
            LIMIT 30
            """
            
            with sqlite3.connect(self.stock_db_path) as conn:
                df = pd.read_sql_query(query, conn, params=[code, date_str])
            
            if len(df) < 10:
                return 0.5
            
            score = 0.0
            
            # 价格波动率控制(50%)
            returns = df['price_change_pct'].head(20) / 100 if len(df) >= 20 else df['price_change_pct'] / 100
            volatility = returns.std() if not returns.empty else 0.3
            
            if pd.notna(volatility):
                if volatility <= 0.02:
                    score += 0.5
                elif volatility <= 0.04:
                    score += 0.4
                elif volatility <= 0.06:
                    score += 0.3
                elif volatility <= 0.08:
                    score += 0.2
            
            # 回撤控制(30%)
            if len(df) >= 10:
                prices = df['close'].head(10).values
                peak = prices[0]
                max_drawdown = 0
                
                for price in prices:
                    if price > peak:
                        peak = price
                    drawdown = (peak - price) / peak
                    max_drawdown = max(max_drawdown, drawdown)
                
                if max_drawdown <= 0.05:
                    score += 0.3
                elif max_drawdown <= 0.1:
                    score += 0.2
                elif max_drawdown <= 0.15:
                    score += 0.1
            
            # 流动性风险(20%)
            stock_data = self.get_stock_data_for_scoring(code, date_str)
            if stock_data is not None:
                turnover_rate = stock_data.get('turnover_rate', 0)
                if pd.notna(turnover_rate) and turnover_rate >= 0.5:
                    score += 0.2
                elif pd.notna(turnover_rate) and turnover_rate >= 0.2:
                    score += 0.1
            
            return min(1.0, score)
            
        except Exception as e:
            self.logger.error(f"计算风险控制得分失败: {e}")
            return 0.5
    
    def calculate_market_regime_score(self, date_str: str) -> float:
        """计算市场环境得分（简化版本，调用增强版评分器）"""
        try:
            from enhanced_market_regime_scorer import EnhancedMarketRegimeScorer
            
            scorer = EnhancedMarketRegimeScorer()
            regime_info = scorer.detect_comprehensive_market_regime(date_str)
            market_score = scorer.calculate_dynamic_market_score(regime_info)
            
            return market_score
            
        except Exception as e:
            self.logger.warning(f"使用增强版市场环境评分失败，使用简化版本: {e}")
            return self.calculate_simple_market_regime_score(date_str)
    
    def calculate_simple_market_regime_score(self, date_str: str) -> float:
        """简化版市场环境得分"""
        try:
            # 获取上证指数数据
            query = """
            SELECT trade_date, close, price_change_pct
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.code = '000001.SH' AND trade_date <= ?
            ORDER BY trade_date DESC
            LIMIT 20
            """
            
            with sqlite3.connect(self.stock_db_path) as conn:
                df = pd.read_sql_query(query, conn, params=[date_str])
            
            if df.empty:
                return 0.6  # 默认中性偏多
            
            # 基于短期趋势的简化评分
            recent_returns = df['price_change_pct'].head(5).mean()
            medium_returns = df['price_change_pct'].head(10).mean()
            
            if pd.notna(recent_returns) and pd.notna(medium_returns):
                # 趋势得分
                trend_score = (recent_returns + medium_returns) / 200 + 0.5  # 归一化到0.5附近
                return max(0.35, min(0.85, trend_score))  # 限制在合理范围内
            
            return 0.6
            
        except Exception as e:
            self.logger.error(f"计算简化版市场环境得分失败: {e}")
            return 0.6
    
    def calculate_stock_score(self, code: str, date_str: str) -> Dict[str, float]:
        """
        计算股票的V3.1优化得分
        
        返回:
        - 各个因子得分
        - 最终优化得分
        - 权重配置信息
        """
        try:
            start_time = time.time()
            
            # 获取股票基础数据
            stock_data = self.get_stock_data_for_scoring(code, date_str)
            if stock_data is None:
                return self._get_default_scores()
            
            # 计算各个因子得分
            technical_score = self.calculate_technical_score(stock_data)
            fundamental_score = self.calculate_fundamental_score(stock_data)  
            performance_score = self.calculate_performance_score(code, date_str)
            sentiment_score = self.calculate_sentiment_score(code, date_str)
            risk_control_score = self.calculate_risk_control_score(code, date_str)
            market_regime_score = self.calculate_market_regime_score(date_str)
            
            # 计算最终V3.1优化得分
            final_score = self.calculate_final_score(
                technical_score, fundamental_score, performance_score,
                sentiment_score, risk_control_score, market_regime_score
            )
            
            # 计算市场环境乘数（用于分析）
            market_multiplier = self.calculate_market_regime_multiplier(market_regime_score)
            
            # 计算个股质量得分（用于分析）
            quality_score = self.calculate_quality_score(
                technical_score, fundamental_score, performance_score,
                sentiment_score, risk_control_score
            )
            
            elapsed_time = time.time() - start_time
            
            return {
                # 因子得分
                'technical': technical_score,
                'fundamental': fundamental_score,
                'performance': performance_score,
                'sentiment': sentiment_score,
                'risk_control': risk_control_score,
                'market_regime': market_regime_score,
                
                # 最终得分
                'final_score': final_score,
                
                # 分析信息
                'market_multiplier': market_multiplier,
                'quality_score': quality_score,
                
                # 权重信息
                'weights': self.optimized_weights.copy(),
                
                # 性能信息
                'calculation_time': elapsed_time,
                
                # 版本信息
                'version': 'v3.1_optimized',
                'optimization_basis': '214万条历史数据'
            }
            
        except Exception as e:
            self.logger.error(f"计算股票 {code} 在 {date_str} 的V3.1得分失败: {e}")
            return self._get_default_scores()
    
    def _get_default_scores(self) -> Dict[str, float]:
        """获取默认得分"""
        return {
            'technical': 0.5,
            'fundamental': 0.5,
            'performance': 0.5,
            'sentiment': 0.5,
            'risk_control': 0.5,
            'market_regime': 0.6,
            'final_score': 0.5,
            'market_multiplier': 1.0,
            'quality_score': 0.5,
            'weights': self.optimized_weights.copy(),
            'calculation_time': 0.0,
            'version': 'v3.1_optimized',
            'optimization_basis': '214万条历史数据'
        }
    
    def batch_calculate_scores(self, stock_codes: List[str], date_str: str) -> Dict[str, Dict[str, float]]:
        """批量计算多只股票的V3.1得分"""
        self.logger.info(f"开始批量计算 {len(stock_codes)} 只股票的V3.1优化得分...")
        
        results = {}
        start_time = time.time()
        
        for i, code in enumerate(stock_codes, 1):
            if i % 100 == 0:
                elapsed = time.time() - start_time
                self.logger.info(f"进度: {i}/{len(stock_codes)} ({i/len(stock_codes)*100:.1f}%), 耗时: {elapsed:.1f}秒")
            
            results[code] = self.calculate_stock_score(code, date_str)
        
        total_time = time.time() - start_time
        self.logger.info(f"批量计算完成！总耗时: {total_time:.1f}秒, 平均每只: {total_time/len(stock_codes):.3f}秒")
        
        return results

def main():
    """测试V3.1优化评分器"""
    scorer = QuantitativeScorerV31Optimized()
    
    # 测试单只股票
    test_code = "000001.SZ"
    test_date = "2024-08-20"
    
    print(f"🧪 测试V3.1优化评分器")
    print(f"股票代码: {test_code}")
    print(f"评分日期: {test_date}")
    print("="*50)
    
    result = scorer.calculate_stock_score(test_code, test_date)
    
    print("📊 评分结果:")
    print(f"技术指标: {result['technical']:.4f} (权重: {result['weights']['technical']*100:.1f}%)")
    print(f"基本面: {result['fundamental']:.4f} (权重: {result['weights']['fundamental']*100:.1f}%)")  
    print(f"表现: {result['performance']:.4f} (权重: {result['weights']['performance']*100:.1f}%)")
    print(f"情绪: {result['sentiment']:.4f} (权重: {result['weights']['sentiment']*100:.1f}%)")
    print(f"风控: {result['risk_control']:.4f} (权重: {result['weights']['risk_control']*100:.1f}%)")
    print(f"市场环境: {result['market_regime']:.4f} (乘数: {result['market_multiplier']:.3f})")
    print(f"")
    print(f"个股质量得分: {result['quality_score']:.4f}")
    print(f"🎯 最终V3.1得分: {result['final_score']:.4f}")
    print(f"")
    print(f"版本: {result['version']}")
    print(f"优化基础: {result['optimization_basis']}")
    print(f"计算耗时: {result['calculation_time']:.3f}秒")

if __name__ == "__main__":
    main()