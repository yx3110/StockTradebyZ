#!/usr/bin/env python3
"""
V4.8.5 production scorer -- 61 features (same as V4.8.4) + ETF support

Architecture:
  Model: V4.8.5 trained model (61 features, trained on A股+ETF)
  Scorer: Inherits V4.8.4, adds ETF scoring capability
  Training data: A股 + ETF (~10% more samples)

Key differences from V4.8.4:
  - Model trained on A股+ETF data (improved A股 ICIR +0.033)
  - predict_scores() accepts ETF codes (159xxx, 510xxx, etc.)
  - ETF scores marked with lower confidence (etf_confidence flag)

Fallback chain: v485 model -> v484 model -> v481 model -> v475 model
"""

import logging
from pathlib import Path
from typing import Dict, List

from .v484_production_scorer import V484ProductionScorer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class V485ProductionScorer(V484ProductionScorer):
    """V4.8.5 scorer -- 61 features (V4.8.4 architecture, trained on A股+ETF)"""

    def __init__(self, model_type: str = 'small_data'):
        self._v485_model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v485'
        super().__init__(model_type=model_type)

    def _load_models(self):
        """Try v485 model first, fallback to v484"""
        v485_files = list(self._v485_model_dir.glob('v485_*.pkl'))
        if v485_files:
            self.model_dir = self._v485_model_dir
            latest = max(v485_files, key=lambda f: f.stat().st_mtime)
            self._load_model_from_file(latest, label='V4.8.5')
            return
        super()._load_models()

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """V4.8.5 scoring — same as V4.8.4 but supports ETF codes.

        ETF codes get scored with the same model but results are flagged
        with etf=True for downstream consumers to handle appropriately.
        """
        results = super().predict_scores(stock_codes, date)

        # Flag ETF results
        etf_prefixes = ('510', '511', '512', '513', '515', '516', '517', '518',
                        '560', '561', '562', '563', '588',
                        '159', '160', '161', '162', '163', '164', '165',
                        '166', '167', '168', '169')
        for code, data in results.items():
            if code[:3] in etf_prefixes:
                data['etf'] = True
                data['etf_note'] = 'ETF预测信心较低(ICIR~0.31 vs A股~0.84)'
            else:
                data['etf'] = False

        return results
