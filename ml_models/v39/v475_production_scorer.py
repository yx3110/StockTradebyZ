#!/usr/bin/env python3
"""V4.7.5 Production Scorer - Asymmetric Top-Quantile Training

Same as V4.7.3 scorer, only loads from v475/ model directory.
The model itself was trained with top-quantile asymmetric sample weights.
"""

import logging
from pathlib import Path
from .v473_production_scorer import V473ProductionScorer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class V475ProductionScorer(V473ProductionScorer):
    """V4.7.5 scorer - identical to V4.7.3 scorer, loads v475 model"""

    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v475'
        super().__init__(model_dir=model_dir)
