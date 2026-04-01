#!/usr/bin/env python3
"""
V4.9.0.2 production scorer — V4901底座 + 自有模型 + 风控增强

与V4901的区别:
  - 使用V4902自己训练的模型 (v4902_multi_target_*.pkl)
  - 使用V4902自己的推荐阈值 (v4902/recommendation_thresholds.json)
  - Market gate 0.30→0.35 (弱市更早减仓)
  - 个股CVaR止损降权 (近20日CVaR@5% > 阈值 → composite降权)
  - 降权后重新校准推荐 (基于降权后composite的相对排名)
"""

import json
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

# V4902 门控阈值 (覆盖V490模块级常量)
V4902_GATE_DONT_BUY = 0.35
CVAR_PENALTY_THRESHOLD = 0.05   # 20日CVaR>5% → 降权 (3%/50%验证过激进)
CVAR_PENALTY_FACTOR = 0.7       # 降权30%


class V4902ProductionScorer(V4901ProductionScorer):
    """V4.9.0.2 scorer — 自有模型 + 风控增强"""

    def __init__(self, model_type: str = 'small_data'):
        self._v4902_model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v4902'
        super().__init__(model_type=model_type)
        # 覆盖V4901的阈值，加载V4902自己的
        self._load_v4902_thresholds()

    def _load_models(self):
        """加载v4902模型, fallback到v4901→v490→v485"""
        v4902_files = list(self._v4902_model_dir.glob('v4902_*.pkl'))
        if v4902_files:
            self.model_dir = self._v4902_model_dir
            latest = max(v4902_files, key=lambda f: f.stat().st_mtime)
            self._load_model_from_file(latest, label='V4.9.0.2')
            has_q95 = 'lgb_q95' in self.models.get('10d', {})
            logger.info(f"  Q95 in ensemble: {has_q95}")
            return
        logger.warning("  V4902模型未找到, fallback到V4901")
        super()._load_models()

    def _load_v4902_thresholds(self):
        """加载V4902自己的推荐阈值，覆盖V4901的"""
        th_path = self._v4902_model_dir / 'recommendation_thresholds.json'
        if th_path.exists():
            with open(th_path) as f:
                self._comp_thresholds = json.load(f)
            logger.info(f"  V4902 composite阈值: strong_buy≥{self._comp_thresholds.get('strong_buy', 'N/A')}")
        else:
            logger.info("  V4902阈值文件不存在，使用动态百分位推荐")
            self._comp_thresholds = None

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """V4902: V4901 pipeline + 推荐校准 + CVaR排序降权

        推荐基于原始composite绝对阈值(回测校准), CVaR只影响score排名。
        """
        # V4901 pipeline: 门控 + 基础预测 + Q95 + composite排序 + 推荐
        results = super().predict_scores(stock_codes, date)

        # 门控升级: V4902用更严的阈值
        confidence = getattr(self, '_last_gate_confidence', 1.0)
        if confidence < V4902_GATE_DONT_BUY:
            for code, data in results.items():
                data['gate_regime'] = 'dont_buy'

        # 用V4902校准的阈值重新设置推荐 (基于原始composite, CVaR之前)
        self._recalibrate_recommendations(results)

        # 未评分股票 → 推到排名底部
        unscored = 0
        for code, data in results.items():
            p10 = data.get('pred_10d', 0) or 0
            p15 = data.get('pred_15d', 0) or 0
            if abs(p10) < 1e-9 and abs(p15) < 1e-9:
                data['composite'] = -999.0
                data['rank_score'] = -999.0
                unscored += 1
        if unscored > 0:
            logger.info(f"  未评分股票: {unscored} 只 (推至底部)")

        # CVaR止损降权 — 只影响rank_score排序, 不改recommendation
        recent_returns = self._load_recent_returns(stock_codes, date)
        penalized = 0
        for code, data in results.items():
            if code in recent_returns:
                ret_series = recent_returns[code]
                if len(ret_series) >= 20:
                    cvar = self._compute_cvar_simple(ret_series)
                    if cvar > CVAR_PENALTY_THRESHOLD:
                        data['rank_score'] = data.get('rank_score', 0) * CVAR_PENALTY_FACTOR
                        data['cvar_penalty'] = True
                        penalized += 1
        if penalized > 0:
            logger.info(f"  CVaR排序降权: {penalized}/{len(results)} 只股票")

        # 重新用rank_score百分位排名 → score (0-100)
        all_rs = [(code, data.get('rank_score', 0)) for code, data in results.items()]
        if all_rs:
            sorted_rs = sorted(all_rs, key=lambda x: x[1])
            n = len(sorted_rs)
            for rank_i, (code, _) in enumerate(sorted_rs):
                results[code]['score'] = round(rank_i / max(n - 1, 1) * 100, 1)

        return results

    def _recalibrate_recommendations(self, results: Dict[str, Dict]):
        """降权后重新校准推荐 — 纯绝对阈值 (回测校准)"""
        th = self._comp_thresholds or {}
        th_strong = th.get('strong_buy', 999)
        th_buy = th.get('buy', 999)
        th_cautious = th.get('cautious', 999)
        th_hold = th.get('hold', 999)

        for code, data in results.items():
            # 未评分股票 → 观望
            p10 = data.get('pred_10d', 0) or 0
            p15 = data.get('pred_15d', 0) or 0
            if abs(p10) < 1e-9 and abs(p15) < 1e-9:
                data['recommendation'] = '观望'
                continue

            comp = data.get('composite', 0)
            if comp >= th_strong:
                data['recommendation'] = '强烈买入'
            elif comp >= th_buy:
                data['recommendation'] = '买入'
            elif comp >= th_cautious:
                data['recommendation'] = '谨慎买入'
            else:
                data['recommendation'] = '观望'

    def _load_recent_returns(self, stock_codes: List[str], date: str) -> Dict[str, np.ndarray]:
        """加载各股票近20日的price_change_pct"""
        # 标准化日期为 YYYY-MM-DD (DB格式)
        if len(date) == 8 and date.isdigit():
            date_db = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        else:
            date_db = date

        # strip交易所后缀 (000001.SZ → 000001)
        codes_stripped = [c.split('.')[0] if '.' in c else c for c in stock_codes]
        strip_to_orig = {}
        for orig, stripped in zip(stock_codes, codes_stripped):
            strip_to_orig[stripped] = orig

        try:
            conn = sqlite3.connect(str(DB_PATH), timeout=30)
            placeholders = ','.join('?' * len(codes_stripped))
            query = f"""
                SELECT s.code, dq.price_change_pct
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.code IN ({placeholders})
                  AND dq.trade_date <= ?
                ORDER BY s.code, dq.trade_date DESC
            """
            df = pd.read_sql(query, conn, params=list(codes_stripped) + [date_db])
            conn.close()
            result = {}
            for code, group in df.groupby('code'):
                returns = group['price_change_pct'].head(20).values.astype(float)
                if len(returns) >= 10:
                    orig_code = strip_to_orig.get(code, code)
                    result[orig_code] = returns
            return result
        except Exception as e:
            logger.warning(f"Failed to load recent returns: {e}")
            return {}

    @staticmethod
    def _compute_cvar_simple(returns: np.ndarray, alpha: float = 0.05) -> float:
        """简单CVaR计算: 取最差alpha分位的平均损失"""
        sorted_r = np.sort(returns)
        n_tail = max(1, int(len(sorted_r) * alpha))
        return float(-sorted_r[:n_tail].mean())
