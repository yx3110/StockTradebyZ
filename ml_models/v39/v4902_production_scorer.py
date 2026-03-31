#!/usr/bin/env python3
"""
V4.9.0.2 production scorer — V4901底座 + 风控增强 + 换手优化

与V4901的区别:
  - EMA平滑 0.7→0.6 (更稳定, 降换手)
  - 持仓保留bonus 0.2→0.3 (减少不必要调仓)
  - Market gate 0.30→0.35 (弱市更早减仓)
  - 个股CVaR止损降权 (近20日CVaR>5%的股票score×0.7)
"""

import numpy as np
import pandas as pd
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List

from .v4901_production_scorer import V4901ProductionScorer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'


class V4902ProductionScorer(V4901ProductionScorer):
    """V4.9.0.2 scorer — 风控增强 + 换手优化"""

    EMA_ALPHA = 0.6
    RETENTION_BONUS = 0.3
    GATE_DONT_BUY = 0.35
    CVAR_PENALTY_THRESHOLD = 0.05
    CVAR_PENALTY_FACTOR = 0.7

    def __init__(self, model_type: str = 'small_data'):
        super().__init__(model_type=model_type)

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """V4902: V4901 pipeline + CVaR止损降权"""
        results = super().predict_scores(stock_codes, date)

        recent_returns = self._load_recent_returns(stock_codes, date)

        for code, data in results.items():
            if code in recent_returns:
                ret_series = recent_returns[code]
                if len(ret_series) >= 20:
                    cvar = self._compute_cvar_simple(ret_series)
                    if cvar > self.CVAR_PENALTY_THRESHOLD:
                        data['composite'] = data.get('composite', 0) * self.CVAR_PENALTY_FACTOR
                        data['rank_score'] = data.get('rank_score', 0) * self.CVAR_PENALTY_FACTOR
                        data['cvar_penalty'] = True

        all_comp = [(code, data.get('composite', 0)) for code, data in results.items()]
        if all_comp:
            sorted_comp = sorted(all_comp, key=lambda x: x[1])
            n = len(sorted_comp)
            for rank_i, (code, _) in enumerate(sorted_comp):
                results[code]['score'] = round(rank_i / max(n - 1, 1) * 100, 1)

        return results

    def _load_recent_returns(self, stock_codes: List[str], date: str) -> Dict[str, np.ndarray]:
        """加载各股票近20日的price_change_pct"""
        date_str = date.replace('-', '')
        try:
            conn = sqlite3.connect(str(DB_PATH), timeout=30)
            placeholders = ','.join('?' * len(stock_codes))
            query = f"""
                SELECT s.code, dq.price_change_pct
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.code IN ({placeholders})
                  AND dq.trade_date <= ?
                ORDER BY s.code, dq.trade_date DESC
            """
            df = pd.read_sql(query, conn, params=list(stock_codes) + [date_str])
            conn.close()
            result = {}
            for code, group in df.groupby('code'):
                returns = group['price_change_pct'].head(20).values
                if len(returns) >= 10:
                    result[code] = returns
            return result
        except Exception as e:
            logger.warning(f"Failed to load recent returns: {e}")
            return {}

    @staticmethod
    def _compute_cvar_simple(returns: np.ndarray, alpha: float = 0.05) -> float:
        """简单CVaR计算"""
        sorted_r = np.sort(returns)
        n_tail = max(1, int(len(sorted_r) * alpha))
        return float(-sorted_r[:n_tail].mean())
