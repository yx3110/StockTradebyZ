"""
Portfolio Pilot Score (仓位领航评分) - 组合管理评分系统

参照北极星V2 (21指标/105分/6档) 设计的仓位管理评分体系:
- 4层 / 20指标 / 100分 (每项5分)
- 6档评级: S >= 80% / A+ >= 70% / A >= 60% / B >= 45% / C >= 30% / D < 30%

Layer 1: 持仓质量 (Position Quality) - 5指标/25分
Layer 2: 风险控制 (Risk Control) - 5指标/25分
Layer 3: 组合效率 (Portfolio Efficiency) - 5指标/25分
Layer 4: 执行纪律 (Execution Discipline) - 5指标/25分
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sqlite3
import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

# ==================== 评分阈值定义 ====================
# 每个指标: (pass, ok, good, great, target) 对应 1-5分
# direction: 'higher' = 越大越好, 'lower' = 越小越好, 'moderate' = 适中最好

METRIC_DEFINITIONS = {
    # Layer 1: 持仓质量
    'avg_ml_score': {
        'name': '平均ML评分',
        'layer': 1,
        'layer_name': '持仓质量',
        'thresholds': (40, 48, 55, 62, 70),
        'direction': 'higher',
        'unit': '分',
        'description': '持仓股票的平均ML评分(0-100)',
    },
    'signal_freshness': {
        'name': '信号新鲜度',
        'layer': 1,
        'layer_name': '持仓质量',
        'thresholds': (30, 20, 14, 8, 5),
        'direction': 'lower',
        'unit': '天',
        'description': '距最近买入信号的平均天数',
    },
    'profit_factor': {
        'name': '盈利因子',
        'layer': 1,
        'layer_name': '持仓质量',
        'thresholds': (1.0, 1.3, 1.6, 2.0, 2.5),
        'direction': 'higher',
        'unit': '',
        'description': '总盈利/总亏损 (>1说明赚多亏少)',
    },
    'win_rate': {
        'name': '持仓胜率',
        'layer': 1,
        'layer_name': '持仓质量',
        'thresholds': (40, 50, 60, 70, 80),
        'direction': 'higher',
        'unit': '%',
        'description': '盈利持仓占比',
    },
    'score_consistency': {
        'name': '持仓质量覆盖',
        'layer': 1,
        'layer_name': '持仓质量',
        'thresholds': (30, 50, 65, 80, 95),
        'direction': 'higher',
        'unit': '%',
        'description': 'ML评分≥45的持仓占比(高=质量均匀)',
    },

    # Layer 2: 风险控制
    'max_position_weight': {
        'name': '最大单仓占比',
        'layer': 2,
        'layer_name': '风险控制',
        'thresholds': (0.30, 0.22, 0.16, 0.12, 0.08),
        'direction': 'lower',
        'unit': '%',
        'description': '最大持仓占总市值比例(低=分散)',
    },
    'sector_hhi': {
        'name': '行业集中度',
        'layer': 2,
        'layer_name': '风险控制',
        'thresholds': (0.40, 0.30, 0.22, 0.16, 0.10),
        'direction': 'lower',
        'unit': 'HHI',
        'description': '行业Herfindahl指数(低=分散)',
    },
    'portfolio_volatility': {
        'name': '组合波动率',
        'layer': 2,
        'layer_name': '风险控制',
        'thresholds': (0.40, 0.32, 0.25, 0.20, 0.15),
        'direction': 'lower',
        'unit': '年化',
        'description': '加权平均年化波动率(20日)',
    },
    'stop_loss_coverage': {
        'name': '止损覆盖率',
        'layer': 2,
        'layer_name': '风险控制',
        'thresholds': (30, 50, 65, 80, 95),
        'direction': 'higher',
        'unit': '%',
        'description': '设置了止损价的持仓比例',
    },
    'current_drawdown': {
        'name': '当前回撤',
        'layer': 2,
        'layer_name': '风险控制',
        'thresholds': (-0.20, -0.12, -0.08, -0.05, -0.02),
        'direction': 'higher',  # closer to 0 is better
        'unit': '%',
        'description': '组合从峰值回撤幅度(接近0=良好)',
    },

    # Layer 3: 组合效率
    'effective_n': {
        'name': '有效持仓数',
        'layer': 3,
        'layer_name': '组合效率',
        'thresholds': (2.0, 3.0, 4.5, 6.0, 8.0),
        'direction': 'higher',
        'unit': '',
        'description': '1/Σ(wi²) 有效分散度',
    },
    'cash_ratio': {
        'name': '现金比例',
        'layer': 3,
        'layer_name': '组合效率',
        # moderate: 5-15% is good. Too high or too low is bad
        'thresholds': None,  # special handling
        'direction': 'moderate',
        'unit': '%',
        'target_range': (5, 15),
        'description': '现金占总资产比例(5-15%最佳)',
    },
    'risk_reward_ratio': {
        'name': '风险收益比',
        'layer': 3,
        'layer_name': '组合效率',
        'thresholds': (0.5, 1.0, 1.5, 2.0, 2.5),
        'direction': 'higher',
        'unit': '',
        'description': '预期收益/预期风险(高=效率好)',
    },
    'capital_utilization': {
        'name': '仓位利用率',
        'layer': 3,
        'layer_name': '组合效率',
        # moderate: 60-90% is good
        'thresholds': None,
        'direction': 'moderate',
        'unit': '%',
        'target_range': (60, 90),
        'description': '已投入资金/总资金(60-90%最佳)',
    },
    'turnover_efficiency': {
        'name': '换手效率',
        'layer': 3,
        'layer_name': '组合效率',
        # lower turnover is generally better (less cost drag)
        'thresholds': (40, 30, 22, 15, 8),
        'direction': 'lower',
        'unit': '%/月',
        'description': '月度换手率(低=摩擦成本低)',
    },

    # Layer 4: 执行纪律
    'recommendation_follow_rate': {
        'name': '建议执行率',
        'layer': 4,
        'layer_name': '执行纪律',
        'thresholds': (20, 40, 55, 70, 85),
        'direction': 'higher',
        'unit': '%',
        'description': '执行ML操作建议的比例',
    },
    'sl_tp_coverage': {
        'name': '止盈止损设置率',
        'layer': 4,
        'layer_name': '执行纪律',
        'thresholds': (20, 40, 60, 80, 95),
        'direction': 'higher',
        'unit': '%',
        'description': '同时设置了止盈止损的持仓比例',
    },
    'holding_period_score': {
        'name': '持仓时长合理性',
        'layer': 4,
        'layer_name': '执行纪律',
        # moderate: 5-20 days is optimal for short-term quant strategies
        'thresholds': None,
        'direction': 'moderate',
        'unit': '天',
        'target_range': (5, 20),
        'description': '平均持仓天数(5-20天最佳)',
    },
    'position_sizing_discipline': {
        'name': '仓位纪律',
        'layer': 4,
        'layer_name': '执行纪律',
        'thresholds': (40, 55, 70, 85, 95),
        'direction': 'higher',
        'unit': '%',
        'description': '符合仓位限制(≤10%)的持仓比例',
    },
    'regime_adaptiveness': {
        'name': '市场适应性',
        'layer': 4,
        'layer_name': '执行纪律',
        'thresholds': (20, 35, 50, 65, 80),
        'direction': 'higher',
        'unit': '分',
        'description': '仓位与市场状态匹配度(0-100)',
    },
}

# 评级阈值 (与北极星V2一致)
GRADE_THRESHOLDS = [
    (80, 'S', '卓越'),
    (70, 'A+', '优秀'),
    (60, 'A', '良好'),
    (45, 'B', '合格'),
    (30, 'C', '待改进'),
    (0, 'D', '需重建'),
]


class PortfolioScorer:
    """
    组合管理评分器

    基于当前持仓状态、交易记录、ML评分、市场数据
    计算 Portfolio Pilot Score (0-100)
    """

    def __init__(self, stock_db_path: Path, webapp_db_path: Path):
        self.stock_db_path = stock_db_path
        self.webapp_db_path = webapp_db_path

    def calculate_score(self,
                        positions: List[Dict],
                        trades: List[Dict],
                        recommendations: List[Dict],
                        snapshots: List[Dict],
                        portfolio_analysis: Optional[Dict] = None,
                        total_capital: float = 0,
                        cash_amount: float = 0) -> Dict[str, Any]:
        """
        计算完整的 Portfolio Pilot Score

        Args:
            positions: 当前持仓列表
            trades: 近期交易记录
            recommendations: 近期操作建议
            snapshots: 历史快照
            portfolio_analysis: PositionAnalyzer 分析结果 (含ML评分)
            total_capital: 总资金 (含现金)
            cash_amount: 现金金额

        Returns:
            完整评分报告
        """
        total_mv = sum(p.get('market_value') or 0 for p in positions)
        if total_capital <= 0:
            if cash_amount > 0:
                total_capital = total_mv + cash_amount
            else:
                # Auto-infer: assume 10% cash reserve
                total_capital = total_mv / 0.9 if total_mv > 0 else 0
                cash_amount = total_capital - total_mv

        # 提取分析结果中的ML评分
        ml_scores = {}
        analysis_positions = []
        if portfolio_analysis:
            analysis_positions = portfolio_analysis.get('positions', [])
            for ap in analysis_positions:
                code = ap.get('code', '')
                if ap.get('ml_score') is not None:
                    ml_scores[code] = ap

        # 计算每个指标
        metrics = {}

        # Layer 1: 持仓质量
        metrics['avg_ml_score'] = self._calc_avg_ml_score(positions, ml_scores)
        metrics['signal_freshness'] = self._calc_signal_freshness(positions, trades)
        metrics['profit_factor'] = self._calc_profit_factor(positions)
        metrics['win_rate'] = self._calc_win_rate(positions)
        metrics['score_consistency'] = self._calc_score_consistency(positions, ml_scores)

        # Layer 2: 风险控制
        metrics['max_position_weight'] = self._calc_max_position_weight(positions, total_mv)
        metrics['sector_hhi'] = self._calc_sector_hhi(positions, total_mv)
        metrics['portfolio_volatility'] = self._calc_portfolio_volatility(positions, total_mv)
        metrics['stop_loss_coverage'] = self._calc_stop_loss_coverage(positions, analysis_positions)
        metrics['current_drawdown'] = self._calc_current_drawdown(snapshots, total_mv)

        # Layer 3: 组合效率
        metrics['effective_n'] = self._calc_effective_n(positions, total_mv)
        metrics['cash_ratio'] = self._calc_cash_ratio(cash_amount, total_capital)
        metrics['risk_reward_ratio'] = self._calc_risk_reward_ratio(
            positions, ml_scores, metrics['portfolio_volatility'])
        metrics['capital_utilization'] = self._calc_capital_utilization(total_mv, total_capital)
        metrics['turnover_efficiency'] = self._calc_turnover_efficiency(trades, total_mv)

        # Layer 4: 执行纪律
        metrics['recommendation_follow_rate'] = self._calc_recommendation_follow_rate(recommendations)
        metrics['sl_tp_coverage'] = self._calc_sl_tp_coverage(positions, analysis_positions)
        metrics['holding_period_score'] = self._calc_holding_period_score(positions, trades)
        metrics['position_sizing_discipline'] = self._calc_position_sizing_discipline(positions, total_mv)
        metrics['regime_adaptiveness'] = self._calc_regime_adaptiveness(
            positions, total_mv, total_capital, analysis_positions)

        # 评分计算
        scored_metrics = {}
        for key, value in metrics.items():
            defn = METRIC_DEFINITIONS[key]
            score = self._score_metric(key, value)
            scored_metrics[key] = {
                'value': value,
                'score': score,
                'max_score': 5,
                'name': defn['name'],
                'layer': defn['layer'],
                'layer_name': defn['layer_name'],
                'unit': defn['unit'],
                'description': defn['description'],
                'direction': defn['direction'],
            }

        # 计算层级汇总
        layers = {}
        for layer_num in range(1, 5):
            layer_metrics = {k: v for k, v in scored_metrics.items() if v['layer'] == layer_num}
            layer_score = sum(m['score'] for m in layer_metrics.values())
            layer_max = sum(m['max_score'] for m in layer_metrics.values())
            layer_name = next(iter(layer_metrics.values()))['layer_name'] if layer_metrics else ''
            layers[layer_num] = {
                'name': layer_name,
                'score': layer_score,
                'max_score': layer_max,
                'pct': round(layer_score / layer_max * 100, 1) if layer_max > 0 else 0,
                'metrics': layer_metrics,
            }

        # 总分
        total_score = sum(m['score'] for m in scored_metrics.values())
        total_max = sum(m['max_score'] for m in scored_metrics.values())
        total_pct = round(total_score / total_max * 100, 1) if total_max > 0 else 0

        # 评级
        grade, grade_label = self._get_grade(total_pct)

        # 改进建议
        improvements = self._generate_improvements(scored_metrics, layers)

        # Diversification ratio diagnostic: DR = weighted_avg_vol / portfolio_vol
        dr = None
        portfolio_vol = metrics.get('portfolio_volatility', 0)
        if portfolio_vol > 0.01 and positions:
            weighted_avg_vol = sum(
                (p.get('market_value') or 0) / total_mv * self._get_stock_volatility(p.get('code', ''))
                for p in positions
            ) if total_mv > 0 else 0
            dr = round(weighted_avg_vol / portfolio_vol, 2) if weighted_avg_vol > 0 else None

        return {
            'total_score': total_score,
            'total_max': total_max,
            'total_pct': total_pct,
            'grade': grade,
            'grade_label': grade_label,
            'layers': layers,
            'metrics': scored_metrics,
            'improvements': improvements,
            'position_count': len(positions),
            'total_market_value': total_mv,
            'total_capital': total_capital,
            'cash_amount': cash_amount,
            'diversification_ratio': dr,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

    # ==================== Layer 1: 持仓质量 ====================

    def _calc_avg_ml_score(self, positions: List[Dict], ml_scores: Dict) -> float:
        """平均ML评分 (0-100)"""
        scores = []
        for p in positions:
            code = p.get('code', '')
            if code in ml_scores and ml_scores[code].get('ml_score') is not None:
                scores.append(ml_scores[code]['ml_score'])
        return np.mean(scores) if scores else 50.0  # default to neutral

    def _calc_signal_freshness(self, positions: List[Dict], trades: List[Dict]) -> float:
        """信号新鲜度: 距最近买入信号的平均天数"""
        if not positions:
            return 30.0
        today = datetime.now()
        days_list = []
        # Build latest buy date per code from trades
        buy_dates = {}
        for t in trades:
            if t.get('action') in ('buy', 'add'):
                code = t.get('code', '')
                tdate = t.get('trade_date', '')
                if tdate > buy_dates.get(code, ''):
                    buy_dates[code] = tdate

        for p in positions:
            code = p.get('code', '')
            last_buy = buy_dates.get(code) or p.get('first_buy_date')
            if last_buy:
                try:
                    dt = datetime.strptime(str(last_buy)[:10], '%Y-%m-%d')
                    days_list.append((today - dt).days)
                except ValueError:
                    days_list.append(30)
            else:
                days_list.append(30)
        return np.mean(days_list) if days_list else 30.0

    def _calc_profit_factor(self, positions: List[Dict]) -> float:
        """盈利因子: 总盈利 / 总亏损"""
        total_profit = 0.0
        total_loss = 0.0
        for p in positions:
            pl = p.get('profit_loss') or 0
            if pl > 0:
                total_profit += pl
            elif pl < 0:
                total_loss += abs(pl)
        if total_loss <= 0:
            return 3.0 if total_profit > 0 else 1.0  # no losses -> good
        return total_profit / total_loss

    def _calc_win_rate(self, positions: List[Dict]) -> float:
        """持仓胜率 (%)"""
        if not positions:
            return 50.0
        winners = sum(1 for p in positions if (p.get('profit_loss_pct') or 0) > 0)
        return winners / len(positions) * 100

    def _calc_score_consistency(self, positions: List[Dict], ml_scores: Dict) -> float:
        """持仓质量覆盖: ML评分>=45的持仓占比 (%)"""
        if not positions:
            return 50.0
        quality_count = 0
        total_with_score = 0
        for p in positions:
            code = p.get('code', '')
            if code in ml_scores and ml_scores[code].get('ml_score') is not None:
                total_with_score += 1
                if ml_scores[code]['ml_score'] >= 45:
                    quality_count += 1
        if total_with_score == 0:
            return 50.0
        return quality_count / total_with_score * 100

    # ==================== Layer 2: 风险控制 ====================

    def _calc_max_position_weight(self, positions: List[Dict], total_mv: float) -> float:
        """最大单仓占比"""
        if not positions or total_mv <= 0:
            return 0.0
        max_weight = max((p.get('market_value') or 0) / total_mv for p in positions)
        return max_weight

    def _calc_sector_hhi(self, positions: List[Dict], total_mv: float) -> float:
        """行业集中度 HHI"""
        if not positions or total_mv <= 0:
            return 1.0
        sector_weights = {}
        for p in positions:
            industry = self._get_stock_industry(p.get('code', ''))
            mv = p.get('market_value') or 0
            weight = mv / total_mv if total_mv > 0 else 0
            sector_weights[industry] = sector_weights.get(industry, 0) + weight
        hhi = sum(w ** 2 for w in sector_weights.values())
        return hhi

    def _calc_portfolio_volatility(self, positions: List[Dict], total_mv: float) -> float:
        """组合年化波动率: sqrt(w' Σ w) 协方差矩阵法"""
        if not positions or total_mv <= 0:
            return 0.30
        n = len(positions)
        weights = []
        returns_matrix = []  # each row = daily returns for one stock

        try:
            with sqlite3.connect(self.stock_db_path) as conn:
                cursor = conn.cursor()
                for p in positions:
                    code = p.get('code', '')
                    if '.' in code:
                        code = code.split('.')[0]
                    mv = p.get('market_value') or 0
                    weights.append(mv / total_mv if total_mv > 0 else 0)

                    cursor.execute(
                        "SELECT id FROM securities WHERE code = ? LIMIT 1", (code,))
                    row = cursor.fetchone()
                    if not row:
                        returns_matrix.append(None)
                        continue
                    sec_id = row[0]
                    cursor.execute("""
                        SELECT close FROM daily_quotes
                        WHERE security_id = ? AND close > 0
                        ORDER BY trade_date DESC LIMIT 21
                    """, (sec_id,))
                    rows = cursor.fetchall()
                    if len(rows) < 6:
                        returns_matrix.append(None)
                        continue
                    prices = [r[0] for r in reversed(rows)]
                    rets = [(prices[i] - prices[i-1]) / prices[i-1]
                            for i in range(1, len(prices))]
                    returns_matrix.append(rets)
        except Exception:
            return 0.30

        # Build aligned returns matrix (use min common length)
        valid = [(w, r) for w, r in zip(weights, returns_matrix) if r is not None]
        if not valid:
            return 0.30
        min_len = min(len(r) for _, r in valid)
        if min_len < 3:
            return 0.30

        w_arr = np.array([w for w, _ in valid])
        w_arr = w_arr / w_arr.sum()  # renormalize after dropping missing
        ret_arr = np.array([r[-min_len:] for _, r in valid])  # shape: (n_valid, min_len)

        cov_matrix = np.cov(ret_arr)
        if cov_matrix.ndim == 0:
            # Single stock: cov returns scalar
            portfolio_var = float(cov_matrix) * w_arr[0] ** 2
        else:
            portfolio_var = float(w_arr @ cov_matrix @ w_arr)

        daily_vol = math.sqrt(max(portfolio_var, 0))
        annual_vol = daily_vol * math.sqrt(252)
        return annual_vol

    def _calc_stop_loss_coverage(self, positions: List[Dict],
                                  analysis_positions: List[Dict]) -> float:
        """止损覆盖率 (%)"""
        if not positions:
            return 0.0
        # Check from analysis results
        sl_set = 0
        for p in positions:
            code = p.get('code', '')
            for ap in analysis_positions:
                if ap.get('code') == code and ap.get('stop_loss_price'):
                    sl_set += 1
                    break
        return sl_set / len(positions) * 100

    def _calc_current_drawdown(self, snapshots: List[Dict], current_mv: float) -> float:
        """当前回撤 (从峰值, 负值)"""
        if not snapshots or current_mv <= 0:
            return 0.0  # no history -> no drawdown
        peak_mv = current_mv
        for s in snapshots:
            mv = s.get('total_market_value') or 0
            if mv > peak_mv:
                peak_mv = mv
        if peak_mv <= 0:
            return 0.0
        drawdown = (current_mv - peak_mv) / peak_mv
        return drawdown  # negative or 0

    # ==================== Layer 3: 组合效率 ====================

    def _calc_effective_n(self, positions: List[Dict], total_mv: float) -> float:
        """有效持仓数 1/Σ(wi²)"""
        if not positions or total_mv <= 0:
            return 0.0
        sum_sq = sum(((p.get('market_value') or 0) / total_mv) ** 2 for p in positions)
        return 1.0 / sum_sq if sum_sq > 0 else 0.0

    def _calc_cash_ratio(self, cash_amount: float, total_capital: float) -> float:
        """现金比例 (%)"""
        if total_capital <= 0:
            return 0.0
        return cash_amount / total_capital * 100

    def _calc_risk_reward_ratio(self, positions: List[Dict], ml_scores: Dict,
                                portfolio_vol: float = 0.30) -> float:
        """风险收益比: Sharpe-like = 加权ML预期收益 / 组合波动率"""
        if not positions:
            return 1.0

        # Weighted average expected return from ML scores
        total_weight = 0.0
        weighted_return = 0.0
        for p in positions:
            code = p.get('code', '')
            mv = p.get('market_value') or 0
            if code in ml_scores:
                ml = ml_scores[code].get('ml_score', 50) or 50
                # Convert ML score (0-100) to annualized expected return proxy
                # Score 50 = 0% excess, 70 = ~20% annual, 30 = ~-20% annual
                expected_annual = (ml - 50) / 100.0  # range: -0.5 to +0.5
            else:
                expected_annual = 0.0
            weighted_return += mv * expected_annual
            total_weight += mv

        if total_weight <= 0 or portfolio_vol <= 0.01:
            return 1.0
        avg_return = weighted_return / total_weight
        # Shift to make ratio positive: add risk-free proxy ~3%
        return (avg_return + 0.03) / portfolio_vol

    def _calc_capital_utilization(self, total_mv: float, total_capital: float) -> float:
        """仓位利用率 (%)"""
        if total_capital <= 0:
            return 0.0
        return total_mv / total_capital * 100

    def _calc_turnover_efficiency(self, trades: List[Dict], total_mv: float) -> float:
        """月度换手率 (%): 仅计算卖出方向，去重(同code+date只计一次)"""
        if not trades or total_mv <= 0:
            return 0.0
        cutoff = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        seen = set()
        recent_volume = 0.0
        for t in trades:
            if t.get('trade_date', '') < cutoff:
                continue
            if t.get('action') not in ('sell', 'reduce'):
                continue
            dedup_key = (t.get('code', ''), t.get('trade_date', ''))
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            recent_volume += (t.get('amount') or
                              (t.get('quantity', 0) * t.get('price', 0)))
        return recent_volume / total_mv * 100 if total_mv > 0 else 0.0

    # ==================== Layer 4: 执行纪律 ====================

    def _calc_recommendation_follow_rate(self, recommendations: List[Dict]) -> float:
        """建议执行率 (%)"""
        if not recommendations:
            return 50.0  # default moderate when no history
        # Only count actionable recommendations (not 'hold')
        actionable = [r for r in recommendations if r.get('action') not in ('hold', None)]
        if not actionable:
            return 80.0  # all holds = no action needed = good
        executed = sum(1 for r in actionable if r.get('is_executed'))
        return executed / len(actionable) * 100

    def _calc_sl_tp_coverage(self, positions: List[Dict],
                              analysis_positions: List[Dict]) -> float:
        """止盈止损同时设置率 (%)"""
        if not positions:
            return 0.0
        both_set = 0
        for p in positions:
            code = p.get('code', '')
            for ap in analysis_positions:
                if ap.get('code') == code:
                    has_sl = ap.get('stop_loss_price') is not None
                    has_tp = ap.get('take_profit_price') is not None
                    if has_sl and has_tp:
                        both_set += 1
                    break
        return both_set / len(positions) * 100

    def _calc_holding_period_score(self, positions: List[Dict],
                                    trades: List[Dict] = None) -> float:
        """平均持仓天数 (优先从交易记录推算真实买入日)"""
        if not positions:
            return 10.0
        today = datetime.now()

        # Build first buy date per code from trades (more accurate than position field)
        trade_buy_dates = {}
        if trades:
            for t in trades:
                if t.get('action') in ('buy', 'add'):
                    code = t.get('code', '')
                    tdate = t.get('trade_date', '')
                    if tdate and (code not in trade_buy_dates or tdate < trade_buy_dates[code]):
                        trade_buy_dates[code] = tdate

        days_list = []
        for p in positions:
            code = p.get('code', '')
            # Prefer trade-inferred buy date, fallback to position field
            fbd = trade_buy_dates.get(code) or p.get('first_buy_date')
            if fbd:
                try:
                    dt = datetime.strptime(str(fbd)[:10], '%Y-%m-%d')
                    days_list.append(max((today - dt).days, 1))
                except ValueError:
                    days_list.append(10)
            else:
                days_list.append(10)
        return np.mean(days_list) if days_list else 10.0

    def _calc_position_sizing_discipline(self, positions: List[Dict],
                                          total_mv: float) -> float:
        """仓位纪律: 符合10%限制的持仓比例 (%)"""
        if not positions or total_mv <= 0:
            return 100.0
        compliant = 0
        for p in positions:
            mv = p.get('market_value') or 0
            weight = mv / total_mv
            if weight <= 0.10:
                compliant += 1
        return compliant / len(positions) * 100

    def _detect_market_regime(self) -> str:
        """从沪深300指数实际数据判断市场状态 (bull/bear/neutral)"""
        try:
            with sqlite3.connect(self.stock_db_path) as conn:
                cursor = conn.cursor()
                # CSI 300 index code: 000300.SH or 399300.SZ
                cursor.execute("""
                    SELECT dq.close FROM daily_quotes dq
                    JOIN securities s ON dq.security_id = s.id
                    WHERE s.code IN ('000300', '399300')
                    ORDER BY dq.trade_date DESC LIMIT 60
                """)
                rows = cursor.fetchall()
                if len(rows) < 20:
                    return 'neutral'
                prices = [r[0] for r in reversed(rows)]
                # 20-day return
                ret_20d = (prices[-1] - prices[-20]) / prices[-20]
                # 60-day MA trend
                ma20 = np.mean(prices[-20:])
                ma60 = np.mean(prices) if len(prices) >= 60 else ma20

                if ret_20d > 0.05 and prices[-1] > ma60:
                    return 'bull'
                elif ret_20d < -0.05 and prices[-1] < ma60:
                    return 'bear'
                else:
                    return 'neutral'
        except Exception:
            return 'neutral'

    def _calc_regime_adaptiveness(self, positions: List[Dict],
                                   total_mv: float, total_capital: float,
                                   analysis_positions: List[Dict]) -> float:
        """市场适应性: 仓位与市况匹配度 (0-100)"""
        if not analysis_positions:
            return 50.0  # neutral

        # Detect market regime: prefer real index data, fallback to analysis
        regime = self._detect_market_regime()
        if regime == 'neutral':
            # Fallback to analysis-provided regime
            for ap in analysis_positions:
                ri = ap.get('regime_info', {})
                if ri and ri.get('regime'):
                    regime = ri['regime']
                    break

        # Calculate exposure ratio
        exposure = total_mv / total_capital if total_capital > 0 else 0

        # Scoring based on regime-exposure alignment
        if regime == 'bull':
            # In bull market, higher exposure is better (70-95% ideal)
            if 0.70 <= exposure <= 0.95:
                score = 80
            elif 0.50 <= exposure < 0.70:
                score = 60
            elif exposure > 0.95:
                score = 65  # over-exposed
            else:
                score = 30  # under-invested in bull
        elif regime == 'bear':
            # In bear market, lower exposure is better (20-50% ideal)
            if 0.20 <= exposure <= 0.50:
                score = 80
            elif 0.50 < exposure <= 0.65:
                score = 55
            elif exposure < 0.20:
                score = 65  # too defensive
            else:
                score = 25  # over-exposed in bear
        else:
            # Neutral: moderate exposure (50-80% ideal)
            if 0.50 <= exposure <= 0.80:
                score = 75
            elif 0.40 <= exposure < 0.50 or 0.80 < exposure <= 0.90:
                score = 55
            else:
                score = 35
        return float(score)

    # ==================== 评分引擎 ====================

    def _score_metric(self, key: str, value: float) -> int:
        """根据阈值计算单项评分 (0-5)"""
        defn = METRIC_DEFINITIONS[key]

        if defn['direction'] == 'moderate':
            return self._score_moderate(value, defn.get('target_range', (0, 100)))

        thresholds = defn['thresholds']
        if thresholds is None:
            return 3  # default

        if defn['direction'] == 'higher':
            # pass=1, ok=2, good=3, great=4, target=5
            if value >= thresholds[4]:
                return 5
            elif value >= thresholds[3]:
                return 4
            elif value >= thresholds[2]:
                return 3
            elif value >= thresholds[1]:
                return 2
            elif value >= thresholds[0]:
                return 1
            else:
                return 0
        else:  # lower is better
            if value <= thresholds[4]:
                return 5
            elif value <= thresholds[3]:
                return 4
            elif value <= thresholds[2]:
                return 3
            elif value <= thresholds[1]:
                return 2
            elif value <= thresholds[0]:
                return 1
            else:
                return 0

    def _score_moderate(self, value: float, target_range: Tuple[float, float]) -> int:
        """评分适中型指标 (在范围内最优, 偏离扣分)"""
        low, high = target_range
        mid = (low + high) / 2
        range_width = high - low

        if low <= value <= high:
            # In target range: 4-5 points
            distance_from_mid = abs(value - mid) / (range_width / 2)
            return 5 if distance_from_mid < 0.3 else 4
        else:
            # Outside range: score by distance
            if value < low:
                deviation = (low - value) / max(low, 1)
            else:
                deviation = (value - high) / max(high, 1)

            if deviation < 0.2:
                return 3
            elif deviation < 0.5:
                return 2
            elif deviation < 1.0:
                return 1
            else:
                return 0

    def _get_grade(self, pct: float) -> Tuple[str, str]:
        """根据百分比获取评级"""
        for threshold, grade, label in GRADE_THRESHOLDS:
            if pct >= threshold:
                return grade, label
        return 'D', '需重建'

    # ==================== 建议自动匹配 ====================

    def auto_match_recommendations(self, trades: List[Dict],
                                    recommendations: List[Dict]) -> List[Dict]:
        """
        自动将交易记录匹配到操作建议 (code+action, 3天窗口内)
        返回更新后的 recommendations 列表 (is_executed 标记已更新)
        """
        if not trades or not recommendations:
            return recommendations

        # Action mapping: trade action -> recommendation action compatibility
        action_compat = {
            'buy': ('buy', 'add', '加仓', '建仓'),
            'add': ('buy', 'add', '加仓', '建仓'),
            'sell': ('sell', 'reduce', '减仓', '清仓', '止盈', '止损'),
            'reduce': ('sell', 'reduce', '减仓', '清仓', '止盈', '止损'),
        }

        for rec in recommendations:
            if rec.get('is_executed'):
                continue
            rec_code = rec.get('code', '')
            rec_date = rec.get('date', '')
            rec_action = (rec.get('action') or '').lower()
            if not rec_date or not rec_code:
                continue

            try:
                rec_dt = datetime.strptime(str(rec_date)[:10], '%Y-%m-%d')
            except ValueError:
                continue

            for t in trades:
                t_code = t.get('code', '')
                t_action = (t.get('action') or '').lower()
                t_date = t.get('trade_date', '')
                if not t_date:
                    continue

                # Code match
                if t_code != rec_code:
                    continue

                # Action compatibility
                compatible_actions = action_compat.get(t_action, ())
                if rec_action not in compatible_actions and t_action not in action_compat.get(rec_action, ()):
                    continue

                # Date window (trade within 3 days after recommendation)
                try:
                    t_dt = datetime.strptime(str(t_date)[:10], '%Y-%m-%d')
                except ValueError:
                    continue
                delta = (t_dt - rec_dt).days
                if 0 <= delta <= 3:
                    rec['is_executed'] = 1
                    # Persist to DB
                    self._mark_recommendation_executed(rec.get('id'))
                    break

        return recommendations

    def _mark_recommendation_executed(self, rec_id: Optional[int]):
        """标记建议已执行"""
        if not rec_id:
            return
        try:
            with sqlite3.connect(self.webapp_db_path) as conn:
                conn.execute(
                    "UPDATE recommendations SET is_executed = 1, executed_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (rec_id,))
                conn.commit()
        except Exception:
            pass

    # ==================== 改进建议生成 ====================

    def _generate_improvements(self, metrics: Dict, layers: Dict) -> List[Dict]:
        """生成改进建议, 按优先级排序"""
        improvements = []

        for key, m in metrics.items():
            if m['score'] <= 2:  # Low-scoring metrics
                defn = METRIC_DEFINITIONS[key]
                priority = 'high' if m['score'] == 0 else 'medium'

                suggestion = self._get_improvement_suggestion(key, m['value'], m['score'])
                improvements.append({
                    'metric': key,
                    'name': m['name'],
                    'layer': m['layer_name'],
                    'current_score': m['score'],
                    'current_value': m['value'],
                    'priority': priority,
                    'suggestion': suggestion,
                })

        # Sort by priority then score
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        improvements.sort(key=lambda x: (priority_order.get(x['priority'], 2), x['current_score']))

        return improvements[:5]  # Top 5 improvements

    def _get_improvement_suggestion(self, key: str, value: float, score: int) -> str:
        """生成具体改进建议"""
        suggestions = {
            'avg_ml_score': '考虑卖出ML评分低于45的持仓，用高评分标的替换',
            'signal_freshness': '部分持仓信号已过期，建议重新评估或设置退出计划',
            'profit_factor': '亏损仓位占比过大，考虑止损清理亏损持仓',
            'win_rate': '盈利持仓比例低，检查选股策略和进场时机',
            'score_consistency': '部分持仓ML评分低于45，建议替换为高评分标的',
            'max_position_weight': f'最大单仓占比{value:.0%}过高，建议分批减仓至10%以内',
            'sector_hhi': '行业过于集中，建议增加跨行业分散',
            'portfolio_volatility': '组合波动率偏高，考虑增加低波动标的或减少仓位',
            'stop_loss_coverage': '部分持仓未设止损，建议为所有持仓设置ATR止损',
            'current_drawdown': f'当前回撤{value:.1%}，考虑降低仓位并等待企稳信号',
            'effective_n': '有效分散度不足，建议增加不相关标的',
            'cash_ratio': '现金比例不在5-15%最佳区间，建议调整',
            'risk_reward_ratio': '风险收益比偏低，选择预期收益更高或风险更低的标的',
            'capital_utilization': '仓位利用率偏离60-90%最佳区间',
            'turnover_efficiency': '换手率过高增加摩擦成本，建议延长持仓周期',
            'recommendation_follow_rate': '建议执行率低，ML系统建议是经过严格验证的，建议提高执行力',
            'sl_tp_coverage': '多数持仓缺少止盈止损，建议系统性设置',
            'holding_period_score': '持仓时长偏离最佳区间(5-20天)，检查持仓周期策略',
            'position_sizing_discipline': '存在超过10%仓位上限的持仓，建议减仓至合规',
            'regime_adaptiveness': '仓位水平与当前市况不匹配，建议根据市况调整总仓位',
        }
        return suggestions.get(key, '建议优化此指标')

    # ==================== 辅助数据查询 ====================

    def _get_stock_industry(self, code: str) -> str:
        """获取股票行业 (from securities table)"""
        try:
            # 去掉后缀
            if '.' in code:
                code = code.split('.')[0]
            with sqlite3.connect(self.stock_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT industry FROM securities WHERE code = ? LIMIT 1",
                    (code,))
                row = cursor.fetchone()
                return row[0] if row and row[0] else '未知'
        except Exception:
            return '未知'

    def _get_stock_volatility(self, code: str, days: int = 20) -> float:
        """获取股票年化波动率 (从收盘价直接计算, 避免pct格式歧义)"""
        try:
            if '.' in code:
                code = code.split('.')[0]
            with sqlite3.connect(self.stock_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id FROM securities WHERE code = ? LIMIT 1", (code,))
                row = cursor.fetchone()
                if not row:
                    return 0.30
                sec_id = row[0]

                cursor.execute("""
                    SELECT close FROM daily_quotes
                    WHERE security_id = ? AND close > 0
                    ORDER BY trade_date DESC LIMIT ?
                """, (sec_id, days + 1))
                rows = cursor.fetchall()
                if len(rows) < 6:
                    return 0.30

                prices = [r[0] for r in reversed(rows)]
                returns = [(prices[i] - prices[i-1]) / prices[i-1]
                           for i in range(1, len(prices))]
                daily_vol = np.std(returns)
                annual_vol = daily_vol * math.sqrt(252)
                return annual_vol
        except Exception:
            return 0.30


def format_score_display(score_result: Dict) -> str:
    """格式化评分结果为文本摘要"""
    lines = []
    grade = score_result['grade']
    pct = score_result['total_pct']
    total = score_result['total_score']
    max_s = score_result['total_max']

    lines.append(f"Portfolio Pilot Score: {total}/{max_s} ({pct}%) [{grade}]")
    lines.append("")

    for layer_num in range(1, 5):
        layer = score_result['layers'].get(layer_num, {})
        lines.append(f"  L{layer_num} {layer['name']}: {layer['score']}/{layer['max_score']} ({layer['pct']}%)")

    if score_result.get('improvements'):
        lines.append("")
        lines.append("Top Improvements:")
        for imp in score_result['improvements'][:3]:
            lines.append(f"  [{imp['priority'].upper()}] {imp['name']}: {imp['suggestion']}")

    return '\n'.join(lines)
