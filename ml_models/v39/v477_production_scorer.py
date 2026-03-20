#!/usr/bin/env python3
"""
V4.7.7 production scorer -- V4.7.6 scorer (consistency+vol) + V4.7.7 model (Huber+DART+180d)

Model: V4.7.7 trained model (Huber Loss + DART + 180d time decay)
Scorer: Inherits V4.7.6 post-processing (consistency bonus + vol discount)

Fallback chain: v477 model → v475 model → v473 model
"""

from pathlib import Path
from typing import List, Optional
import pandas as pd

from .v476_production_scorer import V476ProductionScorer

import logging
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class V477ProductionScorer(V476ProductionScorer):
    """V4.7.7 scorer -- V4.7.7 model + V4.7.6 post-processing"""

    def __init__(self, model_type: str = 'small_data'):
        self._v477_model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v477'
        super().__init__(model_type=model_type)

    def _load_models(self):
        """Try v477 model first, fallback to v475"""
        v477_files = list(self._v477_model_dir.glob('v477_*.pkl'))
        if v477_files:
            # V4.7.7 model has same structure as V4.7.5
            # _load_v475_model globs for v475_*.pkl, so we need to load directly
            self.model_dir = self._v477_model_dir
            latest = max(v477_files, key=lambda f: f.stat().st_mtime)
            self._load_model_from_file(latest)
            return

        # No v477 model yet, fallback to V4.7.5 model via parent
        super()._load_models()

    def _load_model_from_file(self, model_path):
        """Load model from specific file (reuses V4.7.5 parsing logic)"""
        import joblib, pickle
        try:
            model_data = joblib.load(model_path)
        except Exception:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

        import numpy as np

        # Reuse V4.7.3's model structure parsing
        raw_models = model_data.get('models', {})
        self.models = {}
        self.weights = model_data.get('ensemble_weights', {})
        for target, target_data in raw_models.items():
            if isinstance(target_data, dict) and 'models' in target_data:
                self.models[target] = target_data['models']
                if not self.weights:
                    self.weights[f'label_{target}'] = target_data.get('weights', {})
            else:
                self.models[target] = target_data

        self.scaler = model_data.get('scaler')
        self.feature_cols = model_data.get('feature_names', model_data.get('feature_cols', []))
        self.market_feature_cols = model_data.get('market_features', model_data.get('market_feature_cols', []))
        self.target_weights = {
            'label_3d': 0.00, 'label_5d': 0.00, 'label_10d': 0.60, 'label_15d': 0.40
        }

        # Metadata
        self.cascade = False
        self.cascade_feature_names = None
        self.dual_stream = False
        self.rank_normalized = False
        self.robust_zscore = model_data.get('robust_zscore', True)
        self.extra_features_from_daily_basic = model_data.get('extra_features_from_daily_basic', None)
        self.extra_tech_features = model_data.get('extra_features_from_tech_indicators', None)
        self.stock_rank_cols = model_data.get('stock_feature_cols', None)

        self.extra_features_financial = model_data.get('extra_features_financial', [])
        self.extra_features_microstructure = model_data.get('extra_features_microstructure', [])
        self.extra_features_reversal = model_data.get('extra_features_reversal', [])
        self.extra_features_risk = model_data.get('extra_features_risk', [])

        # Winsorize bounds
        raw_bounds = model_data.get('winsorize_bounds')
        if raw_bounds:
            self.winsorize_bounds = {k: tuple(v) for k, v in raw_bounds.items()} if isinstance(raw_bounds, dict) else raw_bounds

        # V4.7.5 design: disable bear blend + isotonic
        self.bear_models = {}
        self.isotonic_calibration = {}

        # Global quantiles
        raw_quantiles = model_data.get('global_quantiles')
        if raw_quantiles is not None:
            self.global_quantiles = np.array(raw_quantiles)
        else:
            quantiles_path = self.model_dir / 'global_quantiles.npy'
            if quantiles_path.exists():
                self.global_quantiles = np.load(quantiles_path)

        # Recommendation thresholds: prefer JSON file (cross-model calibrated)
        # over pkl-embedded (model-specific, may not match actual prediction distribution)
        rec_path = self.model_dir / 'recommendation_thresholds.json'
        if rec_path.exists():
            import json as _json
            with open(rec_path, 'r') as f:
                self.recommendation_thresholds = _json.load(f)
        else:
            self.recommendation_thresholds = model_data.get('recommendation_thresholds')

        # ICIR weights: clip to [0.08, 0.50]
        self.weights = self._clip_icir_weights(self.weights)

        wf = model_data.get('walk_forward_metrics', {})
        gq_status = "continuous" if self.global_quantiles is not None else "cross-sectional"
        print(f"V4.7.7 loaded: {list(self.models.keys())} [{gq_status} scoring, {len(self.feature_cols)} features]")
        print(f"  file: {model_path.name}")
