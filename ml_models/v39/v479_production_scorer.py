#!/usr/bin/env python3
"""
V4.7.9 production scorer -- Huber+DART + 240d decay + Top5% head weighting

Further optimizes top3 stock selection via head-weighted training.
Scorer: Inherits V4.7.6 post-processing (consistency bonus + vol discount)

Fallback chain: v479 model → v475 model
"""

from pathlib import Path
from .v477_production_scorer import V477ProductionScorer

import logging
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class V479ProductionScorer(V477ProductionScorer):
    """V4.7.9 scorer -- V4.7.9 model + V4.7.6 post-processing"""

    def __init__(self, model_type: str = 'small_data'):
        self._v479_model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v479'
        super().__init__(model_type=model_type)

    def _load_models(self):
        """Try v479 model first, fallback to v475"""
        v479_files = list(self._v479_model_dir.glob('v479_*.pkl'))
        if v479_files:
            self.model_dir = self._v479_model_dir
            latest = max(v479_files, key=lambda f: f.stat().st_mtime)
            self._load_model_from_file(latest)
            return
        super()._load_models()
