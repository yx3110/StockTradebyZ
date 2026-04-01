#!/usr/bin/env python3
"""
V4.8.8 production scorer -- V4.8.7底座 + 基准超额标签训练

与V4.8.7使用相同的推理管线(RRF + 共识投票 + Head Refiner)，
但模型使用基准超额标签训练 + 熊市加权 + 单调性集成。

Fallback chain: v488 -> v487 -> v486 -> v485
"""

from pathlib import Path
from .v487_production_scorer import V487ProductionScorer

import logging
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class V488ProductionScorer(V487ProductionScorer):
    """V4.8.8 scorer -- same inference as V4.8.7, different trained model"""

    def __init__(self, model_type: str = 'small_data'):
        self._v488_model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v488'
        super().__init__(model_type=model_type)

    def _load_models(self):
        """Try v488 model first, fallback to v487"""
        v488_files = list(self._v488_model_dir.glob('v488_*.pkl'))
        if v488_files:
            self.model_dir = self._v488_model_dir
            latest = max(v488_files, key=lambda f: f.stat().st_mtime)
            self._load_model_from_file(latest, label='V4.8.8')
            return
        logger.info("  V4.8.8 model not found, falling back to V4.8.7")
        super()._load_models()
