"""
专业仓位管理分析器

集成V3.9.0 ML评分系统，提供多维度持仓分析和操作建议
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PositionAnalyzer:
    """
    专业持仓分析器

    功能:
    1. V3.9.0 ML评分集成
    2. 技术面分析 (趋势、超买超卖、支撑阻力)
    3. 风险指标计算 (ATR止损、波动率、Beta)
    4. Kelly准则仓位建议
    5. 综合操作建议生成
    """

    def __init__(self, stock_db_path: Path):
        """
        初始化分析器

        Args:
            stock_db_path: 主数据库路径
        """
        self.stock_db_path = stock_db_path
        self.ml_scorer = None
        self._init_ml_scorer()

    def _init_ml_scorer(self):
        """初始化ML评分系统 - 优先V3.9.4，回退V3.9.0"""
        self.ml_version = None

        # 优先尝试V3.9.4 (IC=0.1363, 比V3.9.0提升166%)
        try:
            from ml_models.v39.v394_production_scorer import V394ProductionScorer
            self.ml_scorer = V394ProductionScorer()
            self.ml_version = 'v3.9.4'
            logger.info("✅ V3.9.4 ML评分系统初始化成功 (48特征, IC=0.1363)")
            return
        except Exception as e:
            logger.warning(f"V3.9.4初始化失败，尝试V3.9.0: {e}")

        # 回退到V3.9.0
        try:
            from ml_models.v39.v390_production_scorer import V390ProductionScorer
            self.ml_scorer = V390ProductionScorer()
            self.ml_version = 'v3.9.0'
            logger.info("✅ V3.9.0 ML评分系统初始化成功 (42特征, IC=0.0489)")
        except Exception as e:
            logger.warning(f"⚠️ ML评分系统初始化失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            self.ml_scorer = None
            self.ml_version = None

    def analyze_position(self, code: str, avg_cost: float, quantity: int,
                         current_price: float, holding_days: int = None) -> Dict:
        """
        分析单只持仓

        Args:
            code: 股票代码
            avg_cost: 持仓成本
            quantity: 持仓数量
            current_price: 当前价格
            holding_days: 持仓天数

        Returns:
            分析结果字典
        """
        # 规范化代码
        code_clean = code.split('.')[0] if '.' in code else code

        # 处理None值 - 确保两个值都有效
        if current_price is None or current_price == 0:
            current_price = avg_cost

        if avg_cost is None or avg_cost == 0:
            avg_cost = current_price

        # 如果两个都是None/0，尝试从数据库获取价格
        if (current_price is None or current_price == 0) and (avg_cost is None or avg_cost == 0):
            try:
                conn = sqlite3.connect(self.stock_db_path)
                cursor = conn.execute("""
                    SELECT q.close FROM daily_quotes q
                    JOIN securities s ON q.security_id = s.id
                    WHERE s.code = ?
                    ORDER BY q.trade_date DESC LIMIT 1
                """, (code_clean,))
                row = cursor.fetchone()
                conn.close()
                if row and row[0]:
                    current_price = float(row[0])
                    avg_cost = current_price
            except:
                pass

        # 最终回退
        if current_price is None or current_price == 0:
            current_price = 10.0  # 最终默认值
            avg_cost = 10.0

        # 获取最新交易日期
        trade_date = self._get_latest_trade_date()

        # 1. ML评分
        ml_result = self._get_ml_score(code_clean, trade_date)

        # 2. 技术分析
        tech_analysis = self._get_technical_analysis(code_clean, trade_date)

        # 3. 风险指标
        risk_metrics = self._calculate_risk_metrics(code_clean, trade_date, avg_cost, current_price)

        # 4. 基本面数据
        fundamental = self._get_fundamental_data(code_clean, trade_date)

        # 5. 综合建议生成
        recommendation = self._generate_recommendation(
            ml_result, tech_analysis, risk_metrics, fundamental,
            avg_cost, current_price, quantity
        )

        return {
            'code': code,
            'trade_date': trade_date,
            'current_price': current_price,
            'avg_cost': avg_cost,
            'quantity': quantity,
            'market_value': current_price * quantity,
            'profit_loss_pct': (current_price - avg_cost) / avg_cost * 100,
            'holding_days': holding_days,
            # ML评分
            'ml_score': ml_result.get('score'),
            'ml_recommendation': ml_result.get('recommendation'),
            'predicted_return_5d': ml_result.get('predicted_return_5d'),
            'ml_confidence': ml_result.get('confidence'),
            # 技术分析
            'trend': tech_analysis.get('trend'),
            'trend_strength': tech_analysis.get('trend_strength'),
            'rsi': tech_analysis.get('rsi'),
            'macd_signal': tech_analysis.get('macd_signal'),
            'kdj_signal': tech_analysis.get('kdj_signal'),
            'support': tech_analysis.get('support'),
            'resistance': tech_analysis.get('resistance'),
            # 风险指标
            'atr': risk_metrics.get('atr'),
            'atr_pct': risk_metrics.get('atr_pct'),
            'volatility_20d': risk_metrics.get('volatility_20d'),
            'dynamic_stop_loss': risk_metrics.get('dynamic_stop_loss'),
            'dynamic_take_profit': risk_metrics.get('dynamic_take_profit'),
            'hard_stop_loss': risk_metrics.get('hard_stop_loss'),
            'kelly_position': risk_metrics.get('kelly_position'),
            # 基本面
            'pe_ttm': fundamental.get('pe_ttm'),
            'pb': fundamental.get('pb'),
            'turnover_rate': fundamental.get('turnover_rate'),
            'market_cap': fundamental.get('market_cap'),
            # 综合建议
            **recommendation
        }

    def _get_latest_trade_date(self) -> str:
        """获取最新交易日期"""
        try:
            conn = sqlite3.connect(self.stock_db_path)
            cursor = conn.execute("""
                SELECT MAX(trade_date) FROM daily_quotes
            """)
            result = cursor.fetchone()[0]
            conn.close()
            return result or datetime.now().strftime('%Y-%m-%d')
        except:
            return datetime.now().strftime('%Y-%m-%d')

    def _get_ml_score(self, code: str, trade_date: str) -> Dict:
        """获取ML评分"""
        if self.ml_scorer is None:
            return {
                'score': None,
                'recommendation': '无ML数据',
                'predicted_return_5d': None,
                'confidence': None
            }

        try:
            result = self.ml_scorer.predict_score(code, trade_date)
            if result:
                return result
        except Exception as e:
            logger.warning(f"ML评分获取失败 {code}: {e}")

        return {
            'score': None,
            'recommendation': '评分失败',
            'predicted_return_5d': None,
            'confidence': None
        }

    def _get_technical_analysis(self, code: str, trade_date: str) -> Dict:
        """获取技术分析数据"""
        try:
            conn = sqlite3.connect(self.stock_db_path)

            # 获取security_id
            cursor = conn.execute("SELECT id FROM securities WHERE code = ?", (code,))
            row = cursor.fetchone()
            if not row:
                conn.close()
                return self._default_tech_analysis()
            security_id = row[0]

            # 获取最近的技术指标
            df = pd.read_sql_query("""
                SELECT t.*, q.close, q.high, q.low
                FROM technical_indicators t
                JOIN daily_quotes q ON t.security_id = q.security_id AND t.trade_date = q.trade_date
                WHERE t.security_id = ? AND t.trade_date <= ?
                ORDER BY t.trade_date DESC
                LIMIT 20
            """, conn, params=(security_id, trade_date))

            conn.close()

            if df.empty:
                return self._default_tech_analysis()

            latest = df.iloc[0]

            # 趋势判断
            trend = self._judge_trend(df)

            # RSI分析
            rsi = latest.get('rsi_14', 50)

            # MACD信号
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            macd_sig = 'bullish' if macd > macd_signal else 'bearish'

            # KDJ信号
            kdj_k = latest.get('kdj_k', 50)
            kdj_d = latest.get('kdj_d', 50)
            kdj_sig = 'overbought' if kdj_k > 80 else ('oversold' if kdj_k < 20 else 'neutral')

            # 支撑阻力
            support = df['low'].tail(10).min()
            resistance = df['high'].tail(10).max()

            return {
                'trend': trend['direction'],
                'trend_strength': trend['strength'],
                'rsi': rsi,
                'rsi_signal': 'overbought' if rsi > 70 else ('oversold' if rsi < 30 else 'neutral'),
                'macd_signal': macd_sig,
                'kdj_k': kdj_k,
                'kdj_d': kdj_d,
                'kdj_signal': kdj_sig,
                'support': support,
                'resistance': resistance,
                'ma5': latest.get('ma_5'),
                'ma20': latest.get('ma_20'),
                'ma60': latest.get('ma_60'),
                'bbi': latest.get('bbi')
            }

        except Exception as e:
            logger.warning(f"技术分析获取失败 {code}: {e}")
            return self._default_tech_analysis()

    def _default_tech_analysis(self) -> Dict:
        return {
            'trend': 'unknown', 'trend_strength': 0,
            'rsi': None, 'rsi_signal': 'unknown',
            'macd_signal': 'unknown', 'kdj_signal': 'unknown',
            'kdj_k': None, 'kdj_d': None,
            'support': None, 'resistance': None,
            'ma5': None, 'ma20': None, 'ma60': None, 'bbi': None
        }

    def _judge_trend(self, df: pd.DataFrame) -> Dict:
        """判断趋势"""
        if len(df) < 5:
            return {'direction': 'unknown', 'strength': 0}

        close_prices = df['close'].values[::-1]  # 按时间正序

        # 使用MA判断趋势
        ma5 = np.mean(close_prices[-5:]) if len(close_prices) >= 5 else close_prices[-1]
        ma20 = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else ma5

        current = close_prices[-1]

        if current > ma5 > ma20:
            return {'direction': 'up', 'strength': min((current / ma20 - 1) * 100, 10)}
        elif current < ma5 < ma20:
            return {'direction': 'down', 'strength': min((1 - current / ma20) * 100, 10)}
        else:
            return {'direction': 'sideways', 'strength': 3}

    def _calculate_risk_metrics(self, code: str, trade_date: str,
                                 avg_cost: float, current_price: float) -> Dict:
        """计算风险指标"""
        try:
            conn = sqlite3.connect(self.stock_db_path)

            # 获取security_id
            cursor = conn.execute("SELECT id FROM securities WHERE code = ?", (code,))
            row = cursor.fetchone()
            if not row:
                conn.close()
                return self._default_risk_metrics(avg_cost, current_price)
            security_id = row[0]

            # 获取最近60天数据
            df = pd.read_sql_query("""
                SELECT trade_date, close, high, low, volume
                FROM daily_quotes
                WHERE security_id = ? AND trade_date <= ?
                ORDER BY trade_date DESC
                LIMIT 60
            """, conn, params=(security_id, trade_date))

            # 获取ATR
            atr_row = pd.read_sql_query("""
                SELECT atr_14 FROM technical_indicators
                WHERE security_id = ? AND trade_date <= ?
                ORDER BY trade_date DESC LIMIT 1
            """, conn, params=(security_id, trade_date))

            conn.close()

            if df.empty:
                return self._default_risk_metrics(avg_cost, current_price)

            # ATR
            atr = None
            if not atr_row.empty:
                atr_val = atr_row.iloc[0]['atr_14']
                if atr_val is not None and not pd.isna(atr_val):
                    atr = float(atr_val)
            if atr is None:
                atr = self._calculate_atr(df)
            if atr is None:
                atr = current_price * 0.02  # 默认2%的ATR
            atr_pct = atr / current_price * 100 if current_price and current_price > 0 else 2

            # 20日波动率
            returns = df['close'].pct_change().dropna()
            volatility_20d = returns.tail(20).std() * np.sqrt(252) * 100 if len(returns) >= 20 else 30

            # 动态止损 (基于ATR)
            atr_multiplier = 2.5  # 2.5倍ATR作为止损距离
            dynamic_stop_loss = current_price - atr * atr_multiplier

            # 硬止损 (8%固定)
            hard_stop_loss = avg_cost * 0.92

            # 取较高者作为止损
            stop_loss = max(dynamic_stop_loss, hard_stop_loss)

            # 动态止盈 (基于ML预测或ATR)
            # 默认3倍ATR作为止盈目标
            dynamic_take_profit = current_price + atr * 3

            # Kelly准则计算最优仓位
            kelly_position = self._calculate_kelly_position(avg_cost, current_price, stop_loss, dynamic_take_profit)

            return {
                'atr': round(atr, 2),
                'atr_pct': round(atr_pct, 2),
                'volatility_20d': round(volatility_20d, 1),
                'dynamic_stop_loss': round(stop_loss, 2),
                'hard_stop_loss': round(hard_stop_loss, 2),
                'dynamic_take_profit': round(dynamic_take_profit, 2),
                'kelly_position': round(kelly_position, 2)
            }

        except Exception as e:
            logger.warning(f"风险指标计算失败 {code}: {e}")
            return self._default_risk_metrics(avg_cost, current_price)

    def _default_risk_metrics(self, avg_cost: float, current_price: float) -> Dict:
        return {
            'atr': None, 'atr_pct': None, 'volatility_20d': None,
            'dynamic_stop_loss': round(avg_cost * 0.92, 2),
            'hard_stop_loss': round(avg_cost * 0.92, 2),
            'dynamic_take_profit': round(avg_cost * 1.20, 2),
            'kelly_position': 5
        }

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """计算ATR"""
        try:
            if df is None or df.empty or len(df) < 2:
                return None

            if len(df) < period + 1:
                close_val = df['close'].iloc[0]
                if close_val is not None and not pd.isna(close_val):
                    return float(close_val) * 0.02
                return None

            df = df.sort_values('trade_date')
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values

            tr = []
            for i in range(1, len(df)):
                h, l, c_prev = high[i], low[i], close[i-1]
                if any(pd.isna([h, l, c_prev])):
                    continue
                tr.append(max(
                    h - l,
                    abs(h - c_prev),
                    abs(l - c_prev)
                ))

            if not tr:
                return None
            return float(np.mean(tr[-period:]))
        except Exception as e:
            logger.warning(f"ATR计算失败: {e}")
            return None

    def _calculate_kelly_position(self, avg_cost: float, current_price: float,
                                   stop_loss: float, take_profit: float) -> float:
        """
        Kelly准则计算最优仓位百分比

        Kelly f* = (bp - q) / b
        其中: b = 盈利/亏损比, p = 胜率, q = 1-p
        """
        try:
            # 处理None值
            if any(v is None for v in [avg_cost, current_price, stop_loss, take_profit]):
                return 5.0  # 默认5%仓位

            # 潜在盈利和亏损
            potential_profit = take_profit - current_price
            potential_loss = current_price - stop_loss

            if potential_loss <= 0:
                return 5.0  # 默认5%仓位

            # 盈亏比
            reward_risk_ratio = potential_profit / potential_loss

            # 假设胜率基于ML模型的方向准确率 (67.30%)
            win_rate = 0.673
            lose_rate = 1 - win_rate

            # Kelly公式
            kelly = (reward_risk_ratio * win_rate - lose_rate) / reward_risk_ratio

            # 使用半Kelly (更保守)
            half_kelly = kelly * 0.5

            # 限制在1%-10%之间
            position_pct = float(np.clip(half_kelly * 100, 1, 10))

            return position_pct
        except Exception:
            return 5.0

    def _get_fundamental_data(self, code: str, trade_date: str) -> Dict:
        """获取基本面数据"""
        try:
            conn = sqlite3.connect(self.stock_db_path)

            cursor = conn.execute("SELECT id FROM securities WHERE code = ?", (code,))
            row = cursor.fetchone()
            if not row:
                conn.close()
                return {}
            security_id = row[0]

            df = pd.read_sql_query("""
                SELECT pe_ttm, pb, ps_ttm, total_mv, turnover_rate
                FROM daily_basic
                WHERE security_id = ? AND trade_date <= ?
                ORDER BY trade_date DESC
                LIMIT 1
            """, conn, params=(security_id, trade_date))

            conn.close()

            if df.empty:
                return {}

            row = df.iloc[0]
            return {
                'pe_ttm': row.get('pe_ttm'),
                'pb': row.get('pb'),
                'ps_ttm': row.get('ps_ttm'),
                'market_cap': row.get('total_mv'),
                'turnover_rate': row.get('turnover_rate')
            }

        except Exception as e:
            logger.warning(f"基本面数据获取失败 {code}: {e}")
            return {}

    def _generate_recommendation(self, ml_result: Dict, tech_analysis: Dict,
                                  risk_metrics: Dict, fundamental: Dict,
                                  avg_cost: float, current_price: float,
                                  quantity: int) -> Dict:
        """
        生成综合操作建议

        综合考虑:
        1. ML评分 (40%)
        2. 技术面 (30%)
        3. 盈亏状态 (20%)
        4. 风险指标 (10%)
        """
        profit_loss_pct = (current_price - avg_cost) / avg_cost * 100

        # 评分权重系统
        scores = {
            'ml': 0,        # -2到+2
            'tech': 0,      # -2到+2
            'pl': 0,        # -2到+2
            'risk': 0       # -2到+2
        }

        reasons = []

        # 1. ML评分分析 (权重40%)
        ml_score = ml_result.get('score')
        if ml_score is not None:
            if ml_score >= 65:
                scores['ml'] = 2
                reasons.append(f"ML强烈买入信号({ml_score:.1f}分)")
            elif ml_score >= 60:
                scores['ml'] = 1
                reasons.append(f"ML买入信号({ml_score:.1f}分)")
            elif ml_score >= 54:
                scores['ml'] = 0
                reasons.append(f"ML中性({ml_score:.1f}分)")
            elif ml_score >= 50:
                scores['ml'] = -1
                reasons.append(f"ML偏空({ml_score:.1f}分)")
            else:
                scores['ml'] = -2
                reasons.append(f"ML卖出信号({ml_score:.1f}分)")

        # 2. 技术面分析 (权重30%)
        trend = tech_analysis.get('trend', 'unknown')
        rsi = tech_analysis.get('rsi')
        kdj_signal = tech_analysis.get('kdj_signal')

        if trend == 'up':
            scores['tech'] += 1
            reasons.append("趋势向上")
        elif trend == 'down':
            scores['tech'] -= 1
            reasons.append("趋势向下")

        if rsi and rsi > 80:
            scores['tech'] -= 1
            reasons.append(f"RSI超买({rsi:.0f})")
        elif rsi and rsi < 20:
            scores['tech'] += 1
            reasons.append(f"RSI超卖({rsi:.0f})")

        if kdj_signal == 'overbought':
            scores['tech'] -= 0.5
        elif kdj_signal == 'oversold':
            scores['tech'] += 0.5

        # 3. 盈亏状态 (权重20%)
        if profit_loss_pct >= 20:
            scores['pl'] = -1
            reasons.append(f"盈利{profit_loss_pct:.1f}%,建议止盈")
        elif profit_loss_pct >= 10:
            scores['pl'] = 0
            reasons.append(f"盈利{profit_loss_pct:.1f}%")
        elif profit_loss_pct <= -10:
            scores['pl'] = -2
            reasons.append(f"亏损{abs(profit_loss_pct):.1f}%,触及止损")
        elif profit_loss_pct <= -5:
            scores['pl'] = -1
            reasons.append(f"亏损{abs(profit_loss_pct):.1f}%,需警惕")

        # 4. 风险指标 (权重10%)
        stop_loss = risk_metrics.get('dynamic_stop_loss', avg_cost * 0.92)
        if current_price <= stop_loss:
            scores['risk'] = -2
            reasons.append("已跌破动态止损位")

        # 综合评分
        total_score = (
            scores['ml'] * 0.40 +
            scores['tech'] * 0.30 +
            scores['pl'] * 0.20 +
            scores['risk'] * 0.10
        )

        # 生成操作建议
        if total_score >= 1.0:
            action = 'add'
            urgency = 'normal'
        elif total_score >= 0.3:
            action = 'hold'
            urgency = 'low'
        elif total_score >= -0.3:
            action = 'hold'
            urgency = 'normal'
        elif total_score >= -1.0:
            action = 'reduce'
            urgency = 'high'
        else:
            action = 'sell'
            urgency = 'critical'

        # 计算置信度 - 直接使用ML模型返回的置信度
        ml_confidence = ml_result.get('confidence')
        if ml_confidence is not None:
            # 将ML置信度(0.3-0.95)映射到显示置信度(30%-95%)
            confidence = float(ml_confidence)
        else:
            # 无ML数据时基于技术面和盈亏状态计算置信度
            base_confidence = 0.5
            # 趋势一致性加分
            if trend == 'up' and profit_loss_pct > 0:
                base_confidence += 0.1
            elif trend == 'down' and profit_loss_pct < 0:
                base_confidence += 0.1
            # RSI信号加分
            if rsi and (rsi < 30 or rsi > 70):
                base_confidence += 0.05
            confidence = min(base_confidence, 0.85)

        # 目标价位
        predicted_return = ml_result.get('predicted_return_5d', 0)
        if predicted_return:
            target_price = current_price * (1 + predicted_return)
        else:
            target_price = risk_metrics.get('dynamic_take_profit', avg_cost * 1.15)

        return {
            'action': action,
            'urgency': urgency,
            'confidence': round(confidence, 2),
            'reason': '; '.join(reasons) if reasons else '综合分析',
            'total_score': round(total_score, 2),
            'component_scores': scores,
            'target_price': round(target_price, 2) if target_price else None,
            'stop_loss_price': round(stop_loss, 2),
            'take_profit_price': round(risk_metrics.get('dynamic_take_profit', avg_cost * 1.20), 2),
            'suggested_position_pct': risk_metrics.get('kelly_position', 5)
        }

    def analyze_portfolio(self, positions: List[Dict]) -> List[Dict]:
        """
        分析整个投资组合

        Args:
            positions: 持仓列表

        Returns:
            分析结果列表
        """
        results = []

        for pos in positions:
            try:
                holding_days = None
                if pos.get('first_buy_date'):
                    try:
                        first_date = datetime.strptime(pos['first_buy_date'], '%Y-%m-%d')
                        holding_days = (datetime.now() - first_date).days
                    except:
                        pass

                # 确保参数有效
                avg_cost = pos.get('avg_cost')
                current_price = pos.get('current_price')
                quantity = pos.get('quantity', 0)

                # 处理None值
                if avg_cost is None or avg_cost == 0:
                    avg_cost = current_price
                if current_price is None or current_price == 0:
                    current_price = avg_cost
                if avg_cost is None or current_price is None:
                    # 尝试从数据库获取
                    code_clean = pos['code'].split('.')[0] if '.' in str(pos.get('code', '')) else pos.get('code', '')
                    try:
                        conn = sqlite3.connect(self.stock_db_path)
                        cursor = conn.execute("""
                            SELECT q.close FROM daily_quotes q
                            JOIN securities s ON q.security_id = s.id
                            WHERE s.code = ?
                            ORDER BY q.trade_date DESC LIMIT 1
                        """, (code_clean,))
                        row = cursor.fetchone()
                        conn.close()
                        if row and row[0]:
                            current_price = float(row[0])
                            avg_cost = current_price if avg_cost is None else avg_cost
                    except:
                        pass

                # 最终默认值
                if current_price is None:
                    current_price = 10.0
                if avg_cost is None:
                    avg_cost = current_price

                analysis = self.analyze_position(
                    code=pos['code'],
                    avg_cost=float(avg_cost),
                    quantity=int(quantity) if quantity else 0,
                    current_price=float(current_price),
                    holding_days=holding_days
                )

                # 添加股票名称
                analysis['name'] = pos.get('name', '')
                analysis['position_id'] = pos.get('id')

                results.append(analysis)

            except Exception as e:
                import traceback
                logger.error(f"分析持仓失败 {pos.get('code')}: {e}\n{traceback.format_exc()}")
                continue

        # 按综合评分排序 (从低到高，便于优先处理需要操作的)
        results.sort(key=lambda x: x.get('total_score', 0))

        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 测试
    analyzer = PositionAnalyzer(Path('data_adapter/stock_data.db'))

    result = analyzer.analyze_position(
        code='000001',
        avg_cost=15.0,
        quantity=1000,
        current_price=14.5,
        holding_days=30
    )

    print("\n分析结果:")
    for k, v in result.items():
        print(f"  {k}: {v}")
