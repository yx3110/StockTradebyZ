#!/usr/bin/env python3
"""
V4.9.0.1 production scorer — V4.8.5底座 + 去头尾加权训练 + composite排序

与V4.9.0的区别:
  - V490: head_rank排序 (Q95绝对值决定推荐)
  - V4901: composite排序 (pred_10d*0.6 + pred_15d*0.4)
  - V4901: 推荐阈值基于composite (grid_search_PF_weighted校准)

继承V485的Q95 Widen-then-Concentrate, 但排序用composite而非head_rank。
"""

import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List

from .v490_production_scorer import V490ProductionScorer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class V4901ProductionScorer(V490ProductionScorer):
    """V4.9.0.1 scorer — composite排序, 共享V490市场门控"""

    def __init__(self, model_type: str = 'small_data'):
        self._v4901_model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v4901'
        self._comp_thresholds = None
        super().__init__(model_type=model_type)
        self._load_comp_thresholds()

    def _load_models(self):
        """加载v4901模型, fallback到v490→v485"""
        v4901_files = list(self._v4901_model_dir.glob('v4901_*.pkl'))
        if v4901_files:
            self.model_dir = self._v4901_model_dir
            latest = max(v4901_files, key=lambda f: f.stat().st_mtime)
            self._load_model_from_file(latest, label='V4.9.0.1')
            has_q95 = 'lgb_q95' in self.models.get('10d', {})
            logger.info(f"  Q95 in ensemble: {has_q95}")
            return
        logger.warning("  V4901模型未找到, fallback到V490")
        super()._load_models()

    def _load_comp_thresholds(self):
        """加载composite推荐阈值"""
        th_path = self._v4901_model_dir / 'recommendation_thresholds.json'
        if th_path.exists():
            with open(th_path) as f:
                self._comp_thresholds = json.load(f)
            logger.info(f"  V4901 composite阈值: strong_buy≥{self._comp_thresholds.get('strong_buy', 'N/A')}")
        else:
            # 默认阈值
            self._comp_thresholds = {
                'strong_buy': 0.010,
                'buy': 0.007,
                'cautious': 0.004,
                'hold': 0.0
            }

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """V4901: 调用V490的完整pipeline, 然后用composite覆盖排序和推荐"""
        # V490 pipeline: 门控 + 基础预测 + Q95 Widen-then-Concentrate
        results = super().predict_scores(stock_codes, date)

        # 用composite覆盖V490的head_rank排序和Q95推荐
        th = self._comp_thresholds or {}
        th_strong = th.get('strong_buy', 0.010)
        th_buy = th.get('buy', 0.007)
        th_cautious = th.get('cautious', 0.004)

        for code, data in results.items():
            p10 = data.get('pred_10d', 0) or 0
            p15 = data.get('pred_15d', 0) or 0
            composite = 0.6 * p10 + 0.4 * p15
            data['composite'] = composite
            data['rank_score'] = composite  # 供selector排序用

            # composite-based推荐
            if composite >= th_strong:
                data['recommendation'] = '强烈买入'
            elif composite >= th_buy:
                data['recommendation'] = '买入'
            elif composite >= th_cautious:
                data['recommendation'] = '谨慎买入'
            else:
                data['recommendation'] = '观望'

            # score用composite百分位 (不用head_rank覆盖)
            # 留给selector的全局排序

        # 全局composite百分位 → score (0-100)
        all_comp = [(code, data.get('composite', 0)) for code, data in results.items()]
        if all_comp:
            sorted_comp = sorted(all_comp, key=lambda x: x[1])
            n = len(sorted_comp)
            for rank_i, (code, _) in enumerate(sorted_comp):
                results[code]['score'] = round(rank_i / max(n - 1, 1) * 100, 1)

        return results
