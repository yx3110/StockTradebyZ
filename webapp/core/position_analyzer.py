"""
专业仓位管理分析器

集成V4.4.1 ML评分系统，提供绝对ML评分 + 六维风控 + 操作建议矩阵
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sqlite3
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# 绝对ML评分参数
SIGMOID_SCALE = 100  # sigmoid映射缩放因子


class PositionAnalyzer:
    """
    专业持仓分析器

    功能:
    1. V4.4.1 绝对ML评分 (sigmoid映射 + 多目标一致性加权)
    2. 六维风险评分 (盈亏/止损/波动率/技术/市场/集中度)
    3. 操作建议矩阵 (ML评分 × 风险评分)
    4. 组合级风控 (HHI集中度 + 行业暴露 + 预警)
    5. 技术面分析 + ATR止损 + Kelly仓位
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
        """初始化ML评分系统 - 优先V4.4.1，回退V3.9.4，最后V3.9.0"""
        self.ml_version = None

        # 优先尝试V4.4.1 (S级, 59特征, 6增强模块)
        try:
            from ml_models.v39.v44_production_scorer import V44ProductionScorer
            self.ml_scorer = V44ProductionScorer()
            if self.ml_scorer.models:
                self.ml_version = 'v4.4.1'
                logger.info("✅ V4.4.1 ML评分系统初始化成功 (59特征, S级, 6增强模块)")
                return
            else:
                logger.warning("V4.4.1模型文件缺失，尝试V3.9.4")
                self.ml_scorer = None
        except Exception as e:
            logger.warning(f"V4.4.1初始化失败，尝试V3.9.4: {e}")

        # 回退到V3.9.4 (IC=0.1363, 48特征)
        try:
            from ml_models.v39.v394_production_scorer import V394ProductionScorer
            self.ml_scorer = V394ProductionScorer()
            self.ml_version = 'v3.9.4'
            logger.info("✅ V3.9.4 ML评分系统初始化成功 (48特征, IC=0.1363)")
            return
        except Exception as e:
            logger.warning(f"V3.9.4初始化失败，尝试V3.9.0: {e}")

        # 最终回退到V3.9.0
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
                         current_price: float, holding_days: int = None,
                         ml_raw: Dict = None, concentration_pct: float = 10.0) -> Dict:
        """
        分析单只持仓

        Args:
            code: 股票代码
            avg_cost: 持仓成本
            quantity: 持仓数量
            current_price: 当前价格
            holding_days: 持仓天数
            ml_raw: 预计算的ML原始预测 (从批量调用传入)
            concentration_pct: 该持仓占组合比例%

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
            except Exception as e:
                logger.warning("get latest price for %s failed: %s", code_clean, e)

        # 最终回退
        if current_price is None or current_price == 0:
            current_price = 10.0  # 最终默认值
            avg_cost = 10.0

        # 获取最新交易日期
        trade_date = self._get_latest_trade_date()

        # 1. ML评分 (优先使用批量传入的raw预测)
        if ml_raw is None:
            ml_result = self._get_ml_score(code_clean, trade_date)
        else:
            ml_result = ml_raw

        # 2. 技术分析
        tech_analysis = self._get_technical_analysis(code_clean, trade_date)

        # 3. 风险指标
        risk_metrics = self._calculate_risk_metrics(code_clean, trade_date, avg_cost, current_price)

        # 4. 基本面数据
        fundamental = self._get_fundamental_data(code_clean, trade_date)

        # 5. 绝对ML评分 (sigmoid映射)
        absolute_ml_score = self._compute_absolute_ml_score(ml_result)
        exec_filter = ml_result.get('exec_filter')
        regime_info = ml_result.get('regime_info', {})
        score_source = 'ml'  # 评分来源: ml / momentum

        # ETF/基金: ML无数据时用动量评分替代
        if absolute_ml_score is None:
            absolute_ml_score = self._compute_momentum_score(code_clean, trade_date)
            if absolute_ml_score is not None:
                score_source = 'momentum'

        # Exec filter惩罚
        if exec_filter and exec_filter in ('limit_up', 'limit_up_t1') and absolute_ml_score is not None:
            absolute_ml_score = min(absolute_ml_score, 30)

        # 6. 盈亏百分比
        pnl_pct = (current_price - avg_cost) / avg_cost * 100

        # 7. 六维风险评分
        risk_result = self._compute_risk_score(
            pnl_pct=pnl_pct,
            current_price=current_price,
            stop_loss=risk_metrics.get('dynamic_stop_loss'),
            volatility_20d=risk_metrics.get('volatility_20d'),
            tech_analysis=tech_analysis,
            regime_info=regime_info,
            concentration_pct=concentration_pct
        )

        # 8. Kelly仓位 (需要ML评分和风险评分)
        kelly_position = self._calculate_kelly_position(
            avg_cost, current_price,
            risk_metrics.get('dynamic_stop_loss'),
            risk_metrics.get('dynamic_take_profit'),
            ml_score=absolute_ml_score,
            risk_score=risk_result['total']
        )
        risk_metrics['kelly_position'] = kelly_position

        # 9. 操作建议
        action_result = self._generate_action(
            ml_score=absolute_ml_score,
            risk_score=risk_result['total'],
            pnl_pct=pnl_pct,
            exec_filter=exec_filter,
            kelly_pct=kelly_position,
            concentration_pct=concentration_pct
        )

        # 10. 旧式综合建议 (兼容)
        recommendation = self._generate_recommendation(
            ml_result, tech_analysis, risk_metrics, fundamental,
            avg_cost, current_price, quantity
        )

        # 多目标预测百分比
        pred_targets = {}
        for days in [3, 5, 10, 15]:
            key = f'pred_{days}d'
            val = ml_result.get(key, 0)
            if val:
                pred_targets[days] = round(val * 100, 2)

        return {
            'code': code,
            'trade_date': trade_date,
            'current_price': current_price,
            'avg_cost': avg_cost,
            'quantity': quantity,
            'market_value': current_price * quantity,
            'profit_loss_pct': round(pnl_pct, 2),
            'holding_days': holding_days,
            # 绝对ML评分 (新)
            'ml_score': round(absolute_ml_score, 1) if absolute_ml_score is not None else None,
            'score_source': score_source,  # 'ml' 或 'momentum'
            'ml_recommendation': ml_result.get('recommendation'),
            'predicted_return_5d': ml_result.get('predicted_return_5d'),
            'ml_confidence': ml_result.get('confidence'),
            # V4.4.1 多目标预测
            'pred_3d': ml_result.get('pred_3d'),
            'pred_5d': ml_result.get('pred_5d', ml_result.get('predicted_return_5d')),
            'pred_10d': ml_result.get('pred_10d'),
            'pred_15d': ml_result.get('pred_15d'),
            'pred_targets': pred_targets,
            'exec_filter': exec_filter,
            'regime_info': regime_info,
            # 风险评分 (新)
            'risk_score': risk_result['total'],
            'risk_level': risk_result['level'],
            'risk_level_text': risk_result['level_text'],
            'risk_breakdown': risk_result['breakdown'],
            # 操作建议 (新)
            'action': action_result['action'],
            'action_cn': action_result['action_cn'],
            'action_reason': action_result['action_reason'],
            'action_color': action_result['color'],
            'reduce_pct': action_result.get('reduce_pct'),
            'add_pct': action_result.get('add_pct'),
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
            'industry': fundamental.get('industry', '未知'),
            # 旧式综合建议 (兼容)
            'urgency': recommendation.get('urgency', 'normal'),
            'confidence': recommendation.get('confidence', 0.5),
            'reason': action_result['action_reason'],
            'total_score': recommendation.get('total_score', 0),
            'component_scores': recommendation.get('component_scores'),
            'target_price': recommendation.get('target_price'),
            'stop_loss_price': recommendation.get('stop_loss_price'),
            'take_profit_price': recommendation.get('take_profit_price'),
            'suggested_position_pct': risk_metrics.get('kelly_position', 5)
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
        except Exception:
            return datetime.now().strftime('%Y-%m-%d')

    def _get_ml_score(self, code: str, trade_date: str) -> Dict:
        """获取ML评分 - 适配V4.4.1批量API和V3.9.x单股API"""
        if self.ml_scorer is None:
            return {
                'score': None,
                'recommendation': '无ML数据',
                'predicted_return_5d': None,
                'confidence': None
            }

        try:
            if self.ml_version == 'v4.4.1':
                # V4.4.1: 批量API predict_scores([code], date)
                results = self.ml_scorer.predict_scores([code], trade_date)
                if results and code in results:
                    r = results[code]
                    score = r.get('score')
                    pred_5d = r.get('pred_5d', 0)
                    # 生成recommendation文本
                    if score is not None:
                        if score >= 65:
                            rec = '强烈买入'
                        elif score >= 60:
                            rec = '买入'
                        elif score >= 54:
                            rec = '中性'
                        elif score >= 50:
                            rec = '偏空'
                        else:
                            rec = '卖出'
                    else:
                        rec = '评分失败'

                    return {
                        'score': score,
                        'recommendation': rec,
                        'predicted_return_5d': pred_5d,
                        'confidence': min(0.95, 0.5 + abs(score - 60) / 60) if score else None,
                        # V4.4.1 额外字段
                        'pred_3d': r.get('pred_3d', 0),
                        'pred_10d': r.get('pred_10d', 0),
                        'pred_15d': r.get('pred_15d', 0),
                        'exec_filter': r.get('exec_filter', 'unknown'),
                        'regime_info': r.get('regime_info', {}),
                    }
            else:
                # V3.9.x: 单股API predict_score(code, date)
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

    def _compute_absolute_ml_score(self, raw_preds: Dict) -> Optional[float]:
        """
        绝对ML评分: sigmoid映射 + 多目标一致性加权

        raw_preds含: pred_3d, pred_5d, pred_10d, pred_15d (industry-excess return)
        公式: score = 100 / (1 + exp(-SCALE * combined_pred)) * consistency_factor
        """
        pred_3d = raw_preds.get('pred_3d', 0) or 0
        pred_5d = raw_preds.get('pred_5d', 0) or 0
        pred_10d = raw_preds.get('pred_10d', 0) or 0
        pred_15d = raw_preds.get('pred_15d', 0) or 0

        # 如果全部为0，说明没有有效预测
        if pred_3d == 0 and pred_5d == 0 and pred_10d == 0 and pred_15d == 0:
            return None

        # 加权合成 (与V44 regime权重一致)
        combined_pred = 0.20 * pred_3d + 0.25 * pred_5d + 0.35 * pred_10d + 0.20 * pred_15d

        # Sigmoid映射到0-100
        base_score = 100.0 / (1.0 + math.exp(-SIGMOID_SCALE * combined_pred))

        # 多目标一致性加权
        targets = [pred_3d, pred_5d, pred_10d, pred_15d]
        agreement = sum(1 for t in targets if t > 0) / 4.0
        consistency_factor = 0.7 + 0.4 * agreement  # 0.7~1.1

        final_score = base_score * consistency_factor

        # Exec filter惩罚: 在调用处处理
        return float(np.clip(final_score, 0, 100))

    def _compute_momentum_score(self, code: str, trade_date: str) -> Optional[float]:
        """
        ETF/基金动量评分 (ML无数据时的替代方案)

        使用近5/10/20日涨跌幅 + 趋势一致性计算
        Returns: 0~100 分, 或 None
        """
        try:
            conn = sqlite3.connect(self.stock_db_path)
            cursor = conn.execute("SELECT id FROM securities WHERE code = ?", (code,))
            row = cursor.fetchone()
            if not row:
                conn.close()
                return None
            security_id = row[0]

            df = pd.read_sql_query("""
                SELECT close FROM daily_quotes
                WHERE security_id = ? AND trade_date <= ?
                ORDER BY trade_date DESC LIMIT 25
            """, conn, params=(security_id, trade_date))
            conn.close()

            if df.empty or len(df) < 6:
                return None

            closes = df['close'].values  # [newest, ..., oldest]

            # 近5/10/20日收益率
            ret_5d = (closes[0] - closes[min(5, len(closes)-1)]) / closes[min(5, len(closes)-1)] if len(closes) > 5 else 0
            ret_10d = (closes[0] - closes[min(10, len(closes)-1)]) / closes[min(10, len(closes)-1)] if len(closes) > 10 else ret_5d
            ret_20d = (closes[0] - closes[min(20, len(closes)-1)]) / closes[min(20, len(closes)-1)] if len(closes) > 20 else ret_10d

            # 加权动量: 短期权重大
            momentum = 0.5 * ret_5d + 0.3 * ret_10d + 0.2 * ret_20d

            # Sigmoid映射 (SCALE=30, 比ML温和)
            base_score = 100.0 / (1.0 + math.exp(-30 * momentum))

            # 趋势一致性
            rets = [ret_5d, ret_10d, ret_20d]
            agreement = sum(1 for r in rets if r > 0) / 3.0
            consistency = 0.8 + 0.2 * agreement  # 0.8~1.0

            return float(np.clip(base_score * consistency, 5, 95))

        except Exception as e:
            logger.warning(f"动量评分计算失败 {code}: {e}")
            return None

    def _compute_risk_score(self, pnl_pct: float, current_price: float,
                            stop_loss: float, volatility_20d: float,
                            tech_analysis: Dict, regime_info: Dict,
                            concentration_pct: float) -> Dict:
        """
        五维风险评分 (不含盈亏，避免成本锚定影响当下决策)

        每维度0~100 (0=最低风险, 100=最高风险), 加权汇总
        Returns: {total, level, level_text, breakdown: {stop_loss, volatility, technical, regime, concentration}}
        """
        breakdown = {}

        # 1. 止损距离 (25%)
        if current_price > 0 and stop_loss and stop_loss > 0:
            stop_distance_pct = (current_price - stop_loss) / current_price * 100
            if stop_distance_pct <= 1:
                breakdown['stop_loss'] = 95
            elif stop_distance_pct <= 3:
                breakdown['stop_loss'] = 70 + (3 - stop_distance_pct) / 2 * 25
            elif stop_distance_pct <= 5:
                breakdown['stop_loss'] = 50 + (5 - stop_distance_pct) / 2 * 20
            elif stop_distance_pct <= 8:
                breakdown['stop_loss'] = 30 + (8 - stop_distance_pct) / 3 * 20
            else:
                breakdown['stop_loss'] = max(10, 30 - (stop_distance_pct - 8) * 2)
        else:
            breakdown['stop_loss'] = 50  # 无止损数据

        # 2. 波动率 (20%)
        vol = volatility_20d or 30
        if vol < 20:
            breakdown['volatility'] = 20
        elif vol < 30:
            breakdown['volatility'] = 20 + (vol - 20) / 10 * 20
        elif vol < 40:
            breakdown['volatility'] = 40 + (vol - 30) / 10 * 15
        elif vol < 60:
            breakdown['volatility'] = 55 + (vol - 40) / 20 * 25
        else:
            breakdown['volatility'] = min(95, 80 + (vol - 60) / 20 * 15)

        # 3. 技术压力 (25%)
        tech_risk = 50  # 默认中等
        rsi = tech_analysis.get('rsi')
        macd_sig = tech_analysis.get('macd_signal', 'unknown')
        kdj_k = tech_analysis.get('kdj_k')

        if rsi is not None:
            if rsi > 80:
                tech_risk += 25
            elif rsi > 70:
                tech_risk += 15
            elif rsi < 20:
                tech_risk -= 25
            elif rsi < 30:
                tech_risk -= 15

        if macd_sig == 'bearish':
            tech_risk += 10
        elif macd_sig == 'bullish':
            tech_risk -= 10

        if kdj_k is not None:
            if kdj_k > 80:
                tech_risk += 10
            elif kdj_k < 20:
                tech_risk -= 10

        breakdown['technical'] = float(np.clip(tech_risk, 0, 100))

        # 4. 市场环境 (15%)
        regime = regime_info.get('regime', 'neutral') if regime_info else 'neutral'
        regime_map = {'bull': 20, 'neutral': 50, 'bear': 80}
        breakdown['regime'] = regime_map.get(regime, 50)

        # 5. 集中度 (15%)
        if concentration_pct < 5:
            breakdown['concentration'] = 10
        elif concentration_pct < 10:
            breakdown['concentration'] = 10 + (concentration_pct - 5) / 5 * 30
        elif concentration_pct < 15:
            breakdown['concentration'] = 40 + (concentration_pct - 10) / 5 * 30
        elif concentration_pct < 20:
            breakdown['concentration'] = 70 + (concentration_pct - 15) / 5 * 15
        else:
            breakdown['concentration'] = min(100, 85 + (concentration_pct - 20) / 10 * 15)

        # 加权汇总 (五维: 止损25% + 波动率20% + 技术25% + 市场15% + 集中度15%)
        total = (
            breakdown['stop_loss'] * 0.25 +
            breakdown['volatility'] * 0.20 +
            breakdown['technical'] * 0.25 +
            breakdown['regime'] * 0.15 +
            breakdown['concentration'] * 0.15
        )
        total = float(np.clip(total, 0, 100))

        # 风险等级
        if total <= 30:
            level, level_text = 'low', '低风险'
        elif total <= 50:
            level, level_text = 'medium', '中风险'
        elif total <= 70:
            level, level_text = 'high', '较高风险'
        else:
            level, level_text = 'critical', '高风险'

        return {
            'total': round(float(total), 1),
            'level': level,
            'level_text': level_text,
            'breakdown': {k: round(float(v), 1) for k, v in breakdown.items()}
        }

    def _generate_action(self, ml_score: Optional[float], risk_score: float,
                         pnl_pct: float, exec_filter: str = None,
                         kelly_pct: float = None, concentration_pct: float = None) -> Dict:
        """
        基于ML评分 × 风险评分的操作建议矩阵

        Returns: {action, action_cn, action_reason, color, reduce_pct, add_pct}
        reduce_pct: 建议减仓比例(0~100), 减仓/清仓/止损时有值
        add_pct: 建议加仓比例(相对当前持仓), 加仓时有值
        """
        base = {'reduce_pct': None, 'add_pct': None}

        # 特殊规则: 涨停锁定
        if exec_filter and exec_filter in ('limit_up', 'limit_up_t1'):
            return {**base, 'action': 'locked', 'action_cn': '锁定',
                    'action_reason': '涨停不可操作', 'color': 'secondary'}

        # 特殊规则: 深度亏损止损
        if pnl_pct < -15 and (ml_score is None or ml_score < 45):
            return {**base, 'action': 'stop_loss', 'action_cn': '止损',
                    'action_reason': f'浮亏{pnl_pct:.1f}%且ML看空',
                    'color': 'dark', 'reduce_pct': 100}

        # 加仓幅度: kelly目标仓位 vs 当前集中度的差额
        # kelly_pct=8%意味目标仓位8%, 当前concentration=3% → 还可加约(8-3)/3≈167%
        # 但要保守, 单次加仓上限50%, 且不能让集中度超过kelly目标
        def calc_add_pct():
            k = kelly_pct or 5.0
            c = concentration_pct or 5.0
            if c <= 0:
                c = 1.0
            # 目标仓位比当前高多少倍
            headroom = max(0, (k - c) / c)
            # ML越强加越多: ml_score 65→基础20%, 80→基础40%
            ml_factor = min(1.0, ((ml_score or 50) - 60) / 20)  # 0~1
            add = 20 + ml_factor * 30  # 20%~50%
            # 受kelly余量约束
            add = min(add, headroom * 100)
            # 风险越低加越多
            risk_factor = max(0.5, 1.0 - (risk_score or 40) / 100)
            add *= risk_factor
            return max(10, min(50, round(add / 10) * 10))  # 取整到10%, 范围10-50%

        # ETF/无ML数据: 仅基于风险
        if ml_score is None:
            if risk_score <= 40:
                return {**base, 'action': 'hold', 'action_cn': '持有', 'action_reason': '风险可控(无ML)', 'color': 'secondary'}
            elif risk_score <= 60:
                return {**base, 'action': 'hold', 'action_cn': '观望', 'action_reason': '风险偏高(无ML)', 'color': 'warning'}
            else:
                return {**base, 'action': 'reduce', 'action_cn': '减仓', 'action_reason': '高风险(无ML)',
                        'color': 'danger', 'reduce_pct': 50}

        # 二维矩阵
        if ml_score >= 65:
            if risk_score <= 40:
                ap = calc_add_pct()
                return {**base, 'action': 'add', 'action_cn': '加仓', 'action_reason': 'ML看好+风险低',
                        'color': 'success', 'add_pct': ap}
            elif risk_score <= 60:
                return {**base, 'action': 'hold', 'action_cn': '持有', 'action_reason': 'ML看好但风险中等', 'color': 'info'}
            else:
                return {**base, 'action': 'reduce', 'action_cn': '减仓', 'action_reason': 'ML看好但高风险',
                        'color': 'warning', 'reduce_pct': 30}
        elif ml_score >= 45:
            if risk_score <= 40:
                return {**base, 'action': 'hold', 'action_cn': '持有', 'action_reason': 'ML中性+风险可控', 'color': 'info'}
            elif risk_score <= 60:
                return {**base, 'action': 'watch', 'action_cn': '观望', 'action_reason': 'ML中性+风险偏高', 'color': 'warning'}
            else:
                return {**base, 'action': 'reduce', 'action_cn': '减仓', 'action_reason': 'ML中性+高风险',
                        'color': 'danger', 'reduce_pct': 50}
        else:
            # ML看空
            if risk_score <= 40:
                rp = 30 if ml_score >= 35 else 50
                return {**base, 'action': 'reduce', 'action_cn': '减仓', 'action_reason': 'ML看空',
                        'color': 'warning', 'reduce_pct': rp}
            elif risk_score <= 60:
                rp = 50 if ml_score >= 30 else 70
                return {**base, 'action': 'reduce', 'action_cn': '减仓', 'action_reason': 'ML看空+风险偏高',
                        'color': 'danger', 'reduce_pct': rp}
            else:
                return {**base, 'action': 'sell', 'action_cn': '清仓', 'action_reason': 'ML看空+高风险',
                        'color': 'dark', 'reduce_pct': 100}

    def _compute_portfolio_risk(self, analyzed_positions: List[Dict]) -> Dict:
        """
        组合级风控

        Returns: {health_score, hhi, sector_exposure, total_risk_exposure,
                  high_risk_count, warnings}
        """
        if not analyzed_positions:
            return {
                'health_score': 100, 'hhi': 0,
                'sector_exposure': {}, 'total_risk_exposure': 0,
                'high_risk_count': 0, 'warnings': []
            }

        total_mv = sum(p.get('market_value', 0) or 0 for p in analyzed_positions)
        if total_mv <= 0:
            total_mv = 1  # 避免除零

        warnings = []

        # HHI 赫芬达尔指数 (集中度)
        weights = [(p.get('market_value', 0) or 0) / total_mv for p in analyzed_positions]
        hhi = sum(w ** 2 for w in weights)

        if hhi > 0.25:
            warnings.append('持仓极度集中(HHI>{:.0f}%)，建议分散'.format(hhi * 100))
        elif hhi > 0.15:
            warnings.append('持仓较为集中(HHI={:.0f}%)'.format(hhi * 100))

        # 行业暴露
        sector_mv = {}
        for p in analyzed_positions:
            industry = p.get('industry', '未知')
            mv = p.get('market_value', 0) or 0
            sector_mv[industry] = sector_mv.get(industry, 0) + mv

        sector_exposure = {}
        for industry, mv in sorted(sector_mv.items(), key=lambda x: -x[1])[:5]:
            pct = mv / total_mv * 100
            sector_exposure[industry] = round(pct, 1)
            if pct > 30:
                warnings.append(f'{industry}占比{pct:.0f}%，建议分散')

        # 加权平均风险
        total_risk = 0
        high_risk_count = 0
        for p in analyzed_positions:
            risk = p.get('risk_score', 50) or 50
            w = (p.get('market_value', 0) or 0) / total_mv
            total_risk += risk * w
            if risk > 70:
                high_risk_count += 1

        if high_risk_count >= 3:
            warnings.append(f'{high_risk_count}个高风险持仓需关注')
        if total_risk > 60:
            warnings.append('组合整体风险偏高({:.0f}分)'.format(total_risk))

        # 健康分: 100 - 风险调整
        health_score = max(0, min(100, 100 - total_risk * 0.7 - hhi * 50 - high_risk_count * 5))

        return {
            'health_score': round(health_score, 0),
            'hhi': round(hhi, 4),
            'sector_exposure': sector_exposure,
            'total_risk_exposure': round(total_risk, 1),
            'high_risk_count': high_risk_count,
            'warnings': warnings
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

            # 硬止损 (8%固定，但不能高于现价)
            hard_stop_loss = avg_cost * 0.92

            # 如果已经深度亏损(hard_stop_loss > current_price)，硬止损失效
            # 改为基于现价的保护性止损
            if hard_stop_loss >= current_price:
                hard_stop_loss = current_price * 0.92  # 从现价下方8%

            # 取较高者作为止损 (确保不超过现价)
            stop_loss = min(max(dynamic_stop_loss, hard_stop_loss), current_price * 0.98)

            # 动态止盈 (基于ATR)
            dynamic_take_profit = current_price + atr * 3

            return {
                'atr': round(float(atr), 2),
                'atr_pct': round(float(atr_pct), 2),
                'volatility_20d': round(float(volatility_20d), 1),
                'dynamic_stop_loss': round(float(stop_loss), 2),
                'hard_stop_loss': round(float(hard_stop_loss), 2),
                'dynamic_take_profit': round(float(dynamic_take_profit), 2),
                'kelly_position': None  # 后续由analyze_position根据ML/风险评分计算
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
                                   stop_loss: float, take_profit: float,
                                   ml_score: float = None, risk_score: float = None) -> float:
        """
        Kelly准则计算最优仓位百分比

        Kelly f* = (bp - q) / b
        其中: b = 盈利/亏损比, p = 胜率, q = 1-p

        改进: ML评分调制胜率, 风险评分惩罚, 1/4 Kelly
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

            # 盈亏比 (上限3.0，避免不切实际的比率)
            reward_risk_ratio = min(potential_profit / potential_loss, 3.0)

            # ML评分调制胜率: score 0→30%, 50→50%, 100→70%
            if ml_score is not None:
                win_rate = 0.30 + 0.40 * (ml_score / 100.0)
            else:
                win_rate = 0.50  # 无评分时中性假设
            lose_rate = 1 - win_rate

            # Kelly公式
            kelly = (reward_risk_ratio * win_rate - lose_rate) / reward_risk_ratio

            if kelly <= 0:
                return 1.0  # 负Kelly = 最低仓位

            # 使用1/4 Kelly (保守)
            quarter_kelly = kelly * 0.25

            # 风险惩罚: risk_score > 50 时降低仓位
            if risk_score is not None and risk_score > 50:
                penalty = max(0.3, 1.0 - (risk_score - 50) / 100.0)
                quarter_kelly *= penalty

            # 限制在1%-10%之间
            position_pct = float(np.clip(quarter_kelly * 100, 1, 10))

            return round(position_pct, 1)
        except Exception:
            return 5.0

    def _get_fundamental_data(self, code: str, trade_date: str) -> Dict:
        """获取基本面数据 + 行业信息"""
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

            # 获取行业信息 (在securities表)
            industry_row = conn.execute(
                "SELECT industry FROM securities WHERE code = ?", (code,)
            ).fetchone()

            conn.close()

            result = {'industry': industry_row[0] if industry_row and industry_row[0] else '未知'}

            if not df.empty:
                row = df.iloc[0]
                result.update({
                    'pe_ttm': row.get('pe_ttm'),
                    'pb': row.get('pb'),
                    'ps_ttm': row.get('ps_ttm'),
                    'market_cap': row.get('total_mv'),
                    'turnover_rate': row.get('turnover_rate')
                })

            return result

        except Exception as e:
            logger.warning(f"基本面数据获取失败 {code}: {e}")
            return {'industry': '未知'}

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
        exec_filter = ml_result.get('exec_filter')

        # V4.4.1可执行性过滤: 涨停/近涨停直接标记
        if exec_filter and exec_filter in ('limit_up', 'limit_up_t1'):
            scores['ml'] = -2
            reasons.append(f"涨停不可买入({exec_filter})")
        elif exec_filter == 'near_limit_up_t1':
            scores['ml'] = -1
            reasons.append("T+1近涨停,追高风险")
        elif ml_score is not None:
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

        # V4.4.1市况信息: 熊市额外扣分
        regime_info = ml_result.get('regime_info', {})
        if regime_info.get('regime') == 'bear':
            scores['risk'] -= 0.5
            reasons.append(f"熊市环境(20d回报{regime_info.get('market_return_20d', 0)*100:.1f}%)")

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

    def analyze_portfolio(self, positions: List[Dict]) -> Dict:
        """
        分析整个投资组合 (批量ML评分 + 组合风控)

        Args:
            positions: 持仓列表

        Returns:
            {'positions': [...], 'portfolio_risk': {...}}
        """
        results = []
        trade_date = self._get_latest_trade_date()

        # 计算总市值用于集中度计算
        total_mv = 0
        position_mvs = {}
        for pos in positions:
            avg_cost = pos.get('avg_cost') or pos.get('current_price') or 10
            current_price = pos.get('current_price') or avg_cost
            quantity = pos.get('quantity', 0) or 0
            mv = float(current_price) * int(quantity)
            code = pos.get('code', '')
            position_mvs[code] = mv
            total_mv += mv
        if total_mv <= 0:
            total_mv = 1

        # 批量ML评分: 一次性调用predict_scores获取所有A股的ML预测
        all_codes = []
        for pos in positions:
            code = pos.get('code', '')
            code_clean = code.split('.')[0] if '.' in code else code
            all_codes.append(code_clean)

        ml_batch = {}
        if self.ml_scorer and all_codes:
            try:
                # 所有活跃版本 (v4.4.1 / v3.9.x / v3.95) 都暴露 predict_scores 批量接口;
                # fallback per-code 只在极老版本没有该方法时触发
                if hasattr(self.ml_scorer, 'predict_scores'):
                    ml_batch = self.ml_scorer.predict_scores(all_codes, trade_date)
                else:
                    for code in all_codes:
                        try:
                            result = self.ml_scorer.predict_score(code, trade_date)
                            if result:
                                ml_batch[code] = result
                        except Exception as e:
                            logger.warning("predict_score %s failed: %s", code, e)
                logger.info("批量ML评分完成: %d/%d", len(ml_batch), len(all_codes))
            except Exception as e:
                logger.warning("批量ML评分失败: %s", e)

        for pos in positions:
            try:
                holding_days = None
                if pos.get('first_buy_date'):
                    try:
                        first_date = datetime.strptime(pos['first_buy_date'], '%Y-%m-%d')
                        holding_days = (datetime.now() - first_date).days
                    except Exception as e:
                        logger.warning("parse first_buy_date %r failed: %s", pos.get('first_buy_date'), e)

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
                    except Exception as e:
                        logger.warning("price fallback for %s failed: %s", code_clean, e)

                if current_price is None:
                    current_price = 10.0
                if avg_cost is None:
                    avg_cost = current_price

                code = pos['code']
                code_clean = code.split('.')[0] if '.' in code else code

                # 获取批量ML结果
                ml_raw = ml_batch.get(code_clean)
                if ml_raw and self.ml_version == 'v4.4.1':
                    # 从V4.4.1 predict_scores结果构造ml_result
                    pred_5d = ml_raw.get('pred_5d', 0)
                    score = ml_raw.get('score')
                    ml_result = {
                        'score': score,
                        'recommendation': ml_raw.get('recommendation', ''),
                        'predicted_return_5d': pred_5d,
                        'confidence': min(0.95, 0.5 + abs((score or 60) - 60) / 60),
                        'pred_3d': ml_raw.get('pred_3d', 0),
                        'pred_5d': pred_5d,
                        'pred_10d': ml_raw.get('pred_10d', 0),
                        'pred_15d': ml_raw.get('pred_15d', 0),
                        'exec_filter': ml_raw.get('exec_filter', 'unknown'),
                        'regime_info': ml_raw.get('regime_info', {}),
                    }
                elif ml_raw:
                    ml_result = ml_raw
                else:
                    ml_result = None

                # 集中度
                concentration = position_mvs.get(code, 0) / total_mv * 100

                analysis = self.analyze_position(
                    code=code,
                    avg_cost=float(avg_cost),
                    quantity=int(quantity) if quantity else 0,
                    current_price=float(current_price),
                    holding_days=holding_days,
                    ml_raw=ml_result,
                    concentration_pct=concentration
                )

                # 添加股票名称
                analysis['name'] = pos.get('name', '')
                analysis['position_id'] = pos.get('id')

                results.append(analysis)

            except Exception as e:
                import traceback
                logger.error(f"分析持仓失败 {pos.get('code')}: {e}\n{traceback.format_exc()}")
                continue

        # 按风险评分排序 (高风险优先)
        results.sort(key=lambda x: -(x.get('risk_score', 0) or 0))

        # 组合级风控
        portfolio_risk = self._compute_portfolio_risk(results)

        return {
            'positions': results,
            'portfolio_risk': portfolio_risk
        }


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

    # 测试组合分析
    positions = [
        {'code': '000001', 'avg_cost': 15.0, 'quantity': 1000, 'current_price': 14.5, 'name': '平安银行'},
        {'code': '600519', 'avg_cost': 1800.0, 'quantity': 100, 'current_price': 1750.0, 'name': '贵州茅台'},
    ]
    portfolio_result = analyzer.analyze_portfolio(positions)
    print(f"\n组合风控: {portfolio_result['portfolio_risk']}")
    for p in portfolio_result['positions']:
        print(f"  {p['code']} {p.get('name','')}: ML={p['ml_score']}, 风险={p['risk_score']}, 建议={p.get('action_cn','')}")
