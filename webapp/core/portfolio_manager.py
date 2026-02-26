"""
Portfolio Manager Engine - 智能组合管理引擎

提供:
- 建仓质量门控 (ML评分/仓位比例/行业集中度检查)
- 自动SL/TP计算 (ATR-based)
- 追踪止损更新 (trailing stop)
- 风险平价仓位计算 (inverse-volatility)
- 市况适配引擎 (regime-based exposure)
- 再平衡建议生成
- 一键风控更新
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


class PortfolioManager:
    """
    智能组合管理引擎

    编排: 质量门控 → SL/TP → 追踪止损 → 风险平价 → 市况适配 → 再平衡建议
    """

    def __init__(self, stock_db_path: Path, webapp_db_path: Path):
        self.stock_db_path = stock_db_path
        self.webapp_db_path = webapp_db_path

    # ==================== 2a. 建仓质量门控 ====================

    def validate_new_position(self, code: str, quantity: int, avg_cost: float,
                              total_capital: float, positions: List[Dict]) -> Dict:
        """
        建仓前质量门控检查

        Returns:
            {approved, warnings, blocks, ml_score, suggested_sl, suggested_tp}
        """
        warnings = []
        blocks = []
        ml_score = None

        # 1. ML评分检查
        try:
            ml_score = self._get_ml_score(code)
        except Exception:
            pass

        if ml_score is not None:
            if ml_score < 40:
                blocks.append(f'ML评分 {ml_score:.1f} < 40，建议放弃建仓')
            elif ml_score < 50:
                warnings.append(f'ML评分 {ml_score:.1f} 偏低(< 50)，谨慎建仓')

        # 2. 仓位比例检查
        position_value = quantity * avg_cost
        if total_capital > 0:
            weight = position_value / total_capital
            if weight > 0.10:
                max_qty = int(total_capital * 0.10 / avg_cost / 100) * 100
                blocks.append(
                    f'仓位占比 {weight:.1%} 超过10%上限，建议最多 {max_qty} 股')
            elif weight > 0.08:
                warnings.append(f'仓位占比 {weight:.1%} 接近10%上限')

        # 3. 行业集中度检查
        industry = self._get_stock_industry(code)
        same_industry = []
        for p in positions:
            p_industry = self._get_stock_industry(p.get('code', ''))
            if p_industry == industry and industry != '未知':
                same_industry.append(p.get('code', ''))

        if len(same_industry) >= 3:
            blocks.append(
                f'{industry}行业已有{len(same_industry)}只持仓，达到上限')
        elif len(same_industry) >= 2:
            warnings.append(
                f'{industry}行业已有{len(same_industry)}只持仓，注意分散')

        # 4. 计算建议SL/TP
        risk_levels = self.compute_initial_risk_levels(code, avg_cost, avg_cost)

        approved = len(blocks) == 0
        return {
            'approved': approved,
            'warnings': warnings,
            'blocks': blocks,
            'ml_score': ml_score,
            'industry': industry,
            'suggested_sl': risk_levels.get('stop_loss'),
            'suggested_tp': risk_levels.get('take_profit'),
            'suggested_trailing': risk_levels.get('trailing_stop'),
        }

    # ==================== 2b. 自动SL/TP计算 ====================

    def compute_initial_risk_levels(self, code: str, avg_cost: float,
                                     current_price: float) -> Dict:
        """
        基于ATR计算止损/止盈/追踪止损

        Returns:
            {stop_loss, take_profit, trailing_stop, atr}
        """
        atr = self._get_atr(code)
        if atr is None or atr <= 0:
            # Fallback: 8% stop loss, 15% take profit
            return {
                'stop_loss': round(avg_cost * 0.92, 2),
                'take_profit': round(current_price * 1.15, 2),
                'trailing_stop': round(avg_cost * 0.92, 2),
                'atr': None,
            }

        # ATR-based stop loss: max(price - 2.5*ATR, cost*0.92), cap at price*0.98
        raw_sl = current_price - 2.5 * atr
        sl = max(raw_sl, avg_cost * 0.92)
        sl = min(sl, current_price * 0.98)  # Don't set SL too close

        # Take profit: price + 3.0*ATR
        tp = current_price + 3.0 * atr

        return {
            'stop_loss': round(sl, 2),
            'take_profit': round(tp, 2),
            'trailing_stop': round(sl, 2),  # Initially same as SL
            'atr': round(atr, 4),
        }

    # ==================== 2c. 追踪止损更新 ====================

    def update_trailing_stops(self, positions: List[Dict]) -> List[Dict]:
        """
        更新所有持仓的追踪止损 (只升不降)

        Returns:
            触发止损的持仓列表 [{code, name, price, trailing_stop}]
        """
        triggered = []

        for p in positions:
            code = p.get('code', '')
            current_price = self._get_latest_price(code) or p.get('current_price')
            if not current_price or current_price <= 0:
                continue

            old_trailing = p.get('trailing_stop_price')
            if old_trailing is None:
                continue

            # Compute new ATR-based trailing
            atr = self._get_atr(code)
            if atr and atr > 0:
                new_candidate = current_price - 2.5 * atr
                new_trailing = max(old_trailing, new_candidate)
            else:
                new_trailing = old_trailing

            new_trailing = round(new_trailing, 2)

            # Check if triggered
            if current_price <= old_trailing:
                triggered.append({
                    'code': code,
                    'name': p.get('name', ''),
                    'current_price': current_price,
                    'trailing_stop': old_trailing,
                    'position_id': p.get('id'),
                })

            # Update DB if trailing moved up
            if new_trailing > old_trailing:
                self._update_position_field(
                    p['id'], 'trailing_stop_price', new_trailing)

        return triggered

    # ==================== 2d. 风险平价仓位计算 ====================

    def compute_risk_parity_weights(self, positions: List[Dict],
                                     total_capital: float) -> Dict[str, float]:
        """
        逆波动率加权 + 10% hard cap

        Returns:
            {code: target_weight_pct, ...}
        """
        if not positions or total_capital <= 0:
            return {}

        inv_vols = {}
        for p in positions:
            code = p.get('code', '')
            vol = self._get_stock_volatility(code)
            if vol > 0.01:
                inv_vols[code] = 1.0 / vol
            else:
                inv_vols[code] = 1.0 / 0.30  # Default 30% vol

        total_inv_vol = sum(inv_vols.values())
        if total_inv_vol <= 0:
            return {}

        # Raw weights
        weights = {code: iv / total_inv_vol for code, iv in inv_vols.items()}

        # Hard cap at 10%, re-normalize
        capped = {}
        excess = 0.0
        uncapped_total = 0.0
        for code, w in weights.items():
            if w > 0.10:
                capped[code] = 0.10
                excess += w - 0.10
            else:
                capped[code] = w
                uncapped_total += w

        # Redistribute excess proportionally to uncapped
        if excess > 0 and uncapped_total > 0:
            for code in capped:
                if capped[code] < 0.10:
                    capped[code] += excess * (capped[code] / uncapped_total)
                    capped[code] = min(capped[code], 0.10)

        # Save to DB
        for p in positions:
            code = p.get('code', '')
            if code in capped:
                target_pct = round(capped[code] * 100, 2)
                self._update_position_field(
                    p['id'], 'target_weight_pct', target_pct)

        return capped

    # ==================== 2e. 市况适配引擎 ====================

    def compute_regime_exposure(self, total_capital: float, current_mv: float,
                                 snapshots: List[Dict]) -> Dict:
        """
        基于市场状态计算目标仓位 + 回撤熔断

        Returns:
            {regime, target_exposure_pct, circuit_breaker_level, drawdown, peak_value}
        """
        regime = self._detect_market_regime()

        # Target exposure by regime
        regime_targets = {'bull': 85, 'neutral': 70, 'bear': 40}
        target_exposure = regime_targets.get(regime, 70)

        # Peak portfolio value for drawdown
        peak_value = current_mv
        for s in (snapshots or []):
            mv = s.get('total_market_value') or 0
            if mv > peak_value:
                peak_value = mv

        # Drawdown
        drawdown = 0.0
        if peak_value > 0:
            drawdown = (current_mv - peak_value) / peak_value

        # Circuit breaker
        circuit_breaker = 0
        if drawdown < -0.15:
            circuit_breaker = 2  # 熔断: 减仓40%
            target_exposure = max(target_exposure * 0.60, 20)
        elif drawdown < -0.10:
            circuit_breaker = 1  # 警戒: 减仓20%
            target_exposure = max(target_exposure * 0.80, 30)

        return {
            'regime': regime,
            'target_exposure_pct': round(target_exposure, 1),
            'circuit_breaker_level': circuit_breaker,
            'drawdown': round(drawdown, 4),
            'peak_value': round(peak_value, 2),
            'current_exposure_pct': round(
                current_mv / total_capital * 100, 1) if total_capital > 0 else 0,
        }

    # ==================== 2f. 再平衡建议 ====================

    def generate_rebalance_suggestions(self, positions: List[Dict],
                                        trades: List[Dict],
                                        total_mv: float) -> List[Dict]:
        """
        生成再平衡建议

        Returns:
            [{suggestion_type, code, reason, priority}, ...]
        """
        suggestions = []
        today = datetime.now()
        today_str = today.strftime('%Y-%m-%d')

        # Build latest buy date per code
        buy_dates = {}
        for t in (trades or []):
            if t.get('action') in ('buy', 'add'):
                code = t.get('code', '')
                tdate = t.get('trade_date', '')
                if tdate > buy_dates.get(code, ''):
                    buy_dates[code] = tdate

        for p in positions:
            code = p.get('code', '')

            # 1. 过期信号: 最近买入>14天
            last_buy = buy_dates.get(code) or p.get('first_buy_date')
            if last_buy:
                try:
                    dt = datetime.strptime(str(last_buy)[:10], '%Y-%m-%d')
                    days_held = (today - dt).days
                    if days_held > 14:
                        suggestions.append({
                            'suggestion_date': today_str,
                            'suggestion_type': 'stale_exit',
                            'code': code,
                            'reason': f'持仓 {days_held} 天未更新信号，建议重新评估',
                            'priority': 'normal',
                        })
                except ValueError:
                    pass

            # 2. 权重漂移: 当前权重 vs 目标偏离>3%
            if total_mv > 0:
                current_weight = (p.get('market_value') or 0) / total_mv
                target_weight = (p.get('target_weight_pct') or 0) / 100
                if target_weight > 0 and abs(current_weight - target_weight) > 0.03:
                    direction = '偏高' if current_weight > target_weight else '偏低'
                    suggestions.append({
                        'suggestion_date': today_str,
                        'suggestion_type': 'weight_rebalance',
                        'code': code,
                        'reason': f'当前权重 {current_weight:.1%} vs 目标 {target_weight:.1%} ({direction})',
                        'priority': 'normal',
                    })

        # 3. 行业超配: 任一行业>25%
        if total_mv > 0:
            sector_weights = {}
            for p in positions:
                industry = self._get_stock_industry(p.get('code', ''))
                mv = p.get('market_value') or 0
                sector_weights[industry] = sector_weights.get(industry, 0) + mv / total_mv

            for industry, weight in sector_weights.items():
                if weight > 0.25 and industry != '未知':
                    suggestions.append({
                        'suggestion_date': today_str,
                        'suggestion_type': 'sector_reduce',
                        'code': None,
                        'reason': f'{industry}行业占比 {weight:.1%} 超过25%，建议减持',
                        'priority': 'high',
                    })

        # 4. ML低分淘汰: ML评分<45
        for p in positions:
            code = p.get('code', '')
            try:
                ml_score = self._get_ml_score(code)
                if ml_score is not None and ml_score < 45:
                    suggestions.append({
                        'suggestion_date': today_str,
                        'suggestion_type': 'quality_exit',
                        'code': code,
                        'reason': f'ML评分 {ml_score:.1f} < 45，建议替换为高评分标的',
                        'priority': 'high',
                    })
            except Exception:
                pass

        return suggestions

    # ==================== 2g. 一键风控更新 ====================

    def run_daily_risk_update(self, db, positions: List[Dict],
                               total_capital: float, cash_amount: float,
                               trades: List[Dict] = None,
                               snapshots: List[Dict] = None) -> Dict:
        """
        编排所有风控子模块

        Args:
            db: DatabaseManager instance
            positions: 当前持仓
            total_capital: 总资金
            cash_amount: 现金
            trades: 交易记录
            snapshots: 历史快照

        Returns:
            完整风控状态报告
        """
        total_mv = sum(p.get('market_value') or 0 for p in positions)
        today_str = datetime.now().strftime('%Y-%m-%d')

        # 1. 更新追踪止损
        triggered = self.update_trailing_stops(positions)

        # 1b. 为缺少SL/TP的现有持仓补设
        backfilled = 0
        for p in positions:
            if not p.get('stop_loss_price'):
                code = p.get('code', '')
                avg_cost = p.get('avg_cost') or 0
                current_price = self._get_latest_price(code) or avg_cost
                if avg_cost > 0 and current_price > 0:
                    try:
                        risk = self.compute_initial_risk_levels(
                            code, avg_cost, current_price)
                        db.update_position_risk(p['id'],
                            stop_loss_price=risk['stop_loss'],
                            take_profit_price=risk['take_profit'],
                            trailing_stop_price=risk['trailing_stop'])
                        # Update local dict so downstream steps see it
                        p['stop_loss_price'] = risk['stop_loss']
                        p['take_profit_price'] = risk['take_profit']
                        p['trailing_stop_price'] = risk['trailing_stop']
                        backfilled += 1
                    except Exception:
                        pass

        # 2. 计算风险平价权重
        weights = self.compute_risk_parity_weights(positions, total_capital)

        # 3. 市况适配
        regime_info = self.compute_regime_exposure(
            total_capital, total_mv, snapshots or [])

        # 4. 生成再平衡建议
        suggestions = self.generate_rebalance_suggestions(
            positions, trades or [], total_mv)

        # 5. 清除当日旧pending建议，保存新建议
        db.clear_pending_rebalance_suggestions(today_str)
        for s in suggestions:
            db.add_rebalance_suggestion(s)

        # 6. 保存风控状态
        state = {
            'state_date': today_str,
            'market_regime': regime_info['regime'],
            'target_exposure_pct': regime_info['target_exposure_pct'],
            'circuit_breaker_level': regime_info['circuit_breaker_level'],
            'peak_portfolio_value': regime_info['peak_value'],
            'details': {
                'drawdown': regime_info['drawdown'],
                'current_exposure_pct': regime_info['current_exposure_pct'],
                'triggered_stops': len(triggered),
                'sl_tp_backfilled': backfilled,
                'new_suggestions': len(suggestions),
                'weights': {k: round(v * 100, 2) for k, v in weights.items()},
            }
        }
        db.save_portfolio_risk_state(state)

        # 7. 更新last_risk_update
        for p in positions:
            self._update_position_field(
                p['id'], 'last_risk_update', datetime.now().isoformat())

        return {
            'regime': regime_info,
            'triggered_stops': triggered,
            'weights': weights,
            'suggestions': suggestions,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

    # ==================== 辅助方法 ====================

    def _get_ml_score(self, code: str) -> Optional[float]:
        """获取最新ML评分 (from position_analyzer)"""
        try:
            from core.position_analyzer import PositionAnalyzer
            analyzer = PositionAnalyzer(self.stock_db_path)
            result = analyzer._get_ml_score(code)
            if result and isinstance(result, dict):
                return result.get('ml_score')
            elif isinstance(result, (int, float)):
                return float(result)
        except Exception:
            pass
        return None

    def _get_atr(self, code: str, period: int = 14) -> Optional[float]:
        """计算ATR (Average True Range)"""
        try:
            if '.' in code:
                code = code.split('.')[0]
            with sqlite3.connect(self.stock_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id FROM securities WHERE code = ? LIMIT 1", (code,))
                row = cursor.fetchone()
                if not row:
                    return None
                sec_id = row[0]

                cursor.execute("""
                    SELECT high, low, close FROM daily_quotes
                    WHERE security_id = ? AND close > 0
                    ORDER BY trade_date DESC LIMIT ?
                """, (sec_id, period + 1))
                rows = cursor.fetchall()
                if len(rows) < period + 1:
                    return None

                data = list(reversed(rows))
                true_ranges = []
                for i in range(1, len(data)):
                    high, low, close = data[i]
                    prev_close = data[i - 1][2]
                    tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                    true_ranges.append(tr)

                return sum(true_ranges) / len(true_ranges) if true_ranges else None
        except Exception:
            return None

    def _get_latest_price(self, code: str) -> Optional[float]:
        """获取最新收盘价"""
        try:
            if '.' in code:
                code = code.split('.')[0]
            with sqlite3.connect(self.stock_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT close FROM daily_quotes
                    WHERE security_id = (SELECT id FROM securities WHERE code = ?)
                    AND close > 0
                    ORDER BY trade_date DESC LIMIT 1
                """, (code,))
                row = cursor.fetchone()
                return row[0] if row else None
        except Exception:
            return None

    def _get_stock_volatility(self, code: str, days: int = 20) -> float:
        """获取年化波动率 (from close prices)"""
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
                returns = [(prices[i] - prices[i - 1]) / prices[i - 1]
                           for i in range(1, len(prices))]
                daily_vol = np.std(returns)
                return daily_vol * math.sqrt(252)
        except Exception:
            return 0.30

    def _get_stock_industry(self, code: str) -> str:
        """获取股票行业"""
        try:
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

    def _detect_market_regime(self) -> str:
        """从沪深300指数判断市场状态"""
        try:
            with sqlite3.connect(self.stock_db_path) as conn:
                cursor = conn.cursor()
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
                ret_20d = (prices[-1] - prices[-20]) / prices[-20]
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

    def _update_position_field(self, position_id: int, field: str, value):
        """更新单个持仓字段"""
        try:
            with sqlite3.connect(self.webapp_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f'UPDATE positions SET {field} = ? WHERE id = ?',
                    (value, position_id))
                conn.commit()
        except Exception as e:
            logger.warning(f'更新持仓字段 {field} 失败: {e}')
