#!/usr/bin/env python3
"""
V5.0 production scorer — 因子残差alpha + 流动性惩罚

继承V4901的composite排序，新增:
  - 流动性惩罚 (ADV不足降权)
  - 回测参数默认优化 (sector_diversify=3, replace_threshold=0.006)
"""

import numpy as np
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List

from .v4901_production_scorer import V4901ProductionScorer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'


class V5ProductionScorer(V4901ProductionScorer):
    """V5.0 scorer — 因子残差alpha + 流动性惩罚"""

    MAX_PARTICIPATION = 0.02  # 单日不超过ADV的2%
    PENALTY_FLOOR = 0.1       # 流动性最差也保留10%得分
    DEFAULT_PORTFOLIO_VALUE = 1_000_000  # 默认100万组合

    def __init__(self, model_type: str = 'small_data', portfolio_value: float = None):
        self._portfolio_value = portfolio_value or self.DEFAULT_PORTFOLIO_VALUE
        super().__init__(model_type=model_type)

    def _load_models(self):
        """加载v5模型, fallback到v4901→v490→v485"""
        v5_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v5'
        v5_files = list(v5_dir.glob('v5_*.pkl')) if v5_dir.exists() else []
        if v5_files:
            self.model_dir = v5_dir
            latest = max(v5_files, key=lambda f: f.stat().st_mtime)
            self._load_model_from_file(latest, label='V5.0')
            return
        logger.warning("  V5模型未找到, fallback到V4901")
        super()._load_models()

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """V5: V4901 pipeline + 流动性惩罚"""
        results = super().predict_scores(stock_codes, date)

        # 加载ADV数据
        adv_map = self._load_adv(stock_codes, date)

        target_position = self._portfolio_value / 10  # Top10, 每只10%仓位

        for code, data in results.items():
            adv = adv_map.get(code, 0)
            if adv > 0:
                participation = target_position / adv
                penalty = np.clip(
                    1.0 - participation / self.MAX_PARTICIPATION,
                    self.PENALTY_FLOOR, 1.0
                )
            else:
                penalty = self.PENALTY_FLOOR

            data['liquidity_penalty'] = penalty
            data['rank_score'] = data.get('rank_score', 0) * penalty

        # 重新计算全局百分位score
        all_comp = [(code, data.get('rank_score', 0)) for code, data in results.items()]
        if all_comp:
            sorted_comp = sorted(all_comp, key=lambda x: x[1])
            n = len(sorted_comp)
            for rank_i, (code, _) in enumerate(sorted_comp):
                results[code]['score'] = round(rank_i / max(n - 1, 1) * 100, 1)

        return results

    def _load_adv(self, stock_codes: List[str], date: str) -> Dict[str, float]:
        """加载20日日均成交额 (ADV = volume * close * 100, volume单位是手)"""
        try:
            conn = sqlite3.connect(str(DB_PATH), timeout=30)
            placeholders = ','.join('?' for _ in stock_codes)
            query = f"""
                SELECT s.code,
                       AVG(dq.volume * dq.close * 100) AS adv_20d
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.code IN ({placeholders})
                  AND dq.trade_date <= ?
                  AND dq.trade_date >= date(?, '-30 day')
                  AND dq.volume > 0
                GROUP BY s.code
            """
            import pandas as pd
            params = list(stock_codes) + [date, date]
            df = pd.read_sql(query, conn, params=params)
            conn.close()
            return dict(zip(df['code'], df['adv_20d']))
        except Exception as e:
            logger.warning(f"  加载ADV失败: {e}")
            return {}
