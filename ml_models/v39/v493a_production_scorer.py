#!/usr/bin/env python3
"""
V4.9.3a (消融A) production scorer — 仅删13特征 (61->48), 无BRAIN, 无浓度对策

继承V4901的composite排序和市场门控。
"""

import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List

from .v4901_production_scorer import V4901ProductionScorer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class V493AProductionScorer(V4901ProductionScorer):
    """消融A scorer — 仅特征裁剪, composite排序"""

    def __init__(self, model_type: str = 'small_data'):
        self._v493a_model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v493a'
        super().__init__(model_type=model_type)

    def _load_models(self):
        """加载v493a模型, fallback到v4901→v490→v485"""
        v493a_files = list(self._v493a_model_dir.glob('v493a_*.pkl'))
        if v493a_files:
            self.model_dir = self._v493a_model_dir
            latest = max(v493a_files, key=lambda f: f.stat().st_mtime)
            self._load_model_from_file(latest, label='V4.9.3a(Ablation-A)')
            has_q95 = 'lgb_q95' in self.models.get('10d', {})
            logger.info(f"  Q95 in ensemble: {has_q95}")
            th_path = self._v493a_model_dir / 'recommendation_thresholds.json'
            if th_path.exists():
                with open(th_path) as f:
                    self._comp_thresholds = json.load(f)
                logger.info(f"  V493A composite阈值: strong_buy>={self._comp_thresholds.get('strong_buy', 'N/A')}")
            return
        logger.warning("  V493A模型未找到, fallback到V4901")
        super()._load_models()
