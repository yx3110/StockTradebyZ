#!/usr/bin/env python3
"""
V4.8.7 production scorer -- 69 features + CatBoost YetiRank + RRF

Architecture:
  Base: V4.8.6 (V4.8.5 + 3 BRAIN + 5 V482 + CatBoost YetiRank NDCG@10)
  Scorer: RRF ensemble (k=60)
  Features: 69 (V4.8.5底座61 + 3 BRAIN Top-K + 5 V482 Top IC)

Fallback chain: v487 -> v486 -> v485 -> v484 -> v481 -> v475
"""

from pathlib import Path
from .v486_production_scorer import V486ProductionScorer

import logging
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class V487ProductionScorer(V486ProductionScorer):
    """V4.8.7 scorer -- 69 features + YetiRank + RRF (v487 model dir)"""

    def __init__(self, model_type: str = 'small_data'):
        self._v487_model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v487'
        super().__init__(model_type=model_type)

    def _load_models(self):
        """Try v487 model first, fallback to v486"""
        v487_files = list(self._v487_model_dir.glob('v487_*.pkl'))
        if v487_files:
            self.model_dir = self._v487_model_dir
            latest = max(v487_files, key=lambda f: f.stat().st_mtime)
            self._load_model_from_file(latest, label='V4.8.7')
            return
        super()._load_models()
