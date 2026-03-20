#!/usr/bin/env python3
"""
V4.7.8 production scorer -- Huber+DART(V4.7.7) + 365d decay(V4.7.5)

Combines V4.7.7's IC strength (Huber Loss) with V4.7.5's top3 accuracy (365d decay).
Scorer: Inherits V4.7.6 post-processing (consistency bonus + vol discount)

Fallback chain: v478 model → v475 model
"""

from pathlib import Path
from .v477_production_scorer import V477ProductionScorer

import logging
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class V478ProductionScorer(V477ProductionScorer):
    """V4.7.8 scorer -- V4.7.8 model + V4.7.6 post-processing"""

    def __init__(self, model_type: str = 'small_data'):
        self._v478_model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v478'
        super().__init__(model_type=model_type)

    def _load_models(self):
        """Try v478 model first, fallback to v475"""
        v478_files = list(self._v478_model_dir.glob('v478_*.pkl'))
        if v478_files:
            self.model_dir = self._v478_model_dir
            latest = max(v478_files, key=lambda f: f.stat().st_mtime)
            self._load_model_from_file(latest)
            return
        super()._load_models()
