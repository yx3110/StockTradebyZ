#!/usr/bin/env python3
"""
V4.8.0 production scorer -- V4.7.6 scorer + 270d time decay model

Target: North Star V3 breakthrough from 76.9% A+ toward S (80%)
Key metric: ic_decay_ratio (H2/H1) from 0.52 (1/5) → 0.70+ (3/5)

Architecture:
  Model: V4.8.0 trained model (270d time decay, MSE loss, no DART)
    - 270d half-life (vs V4.7.5=365d, V4.7.7=180d) — sweet spot
    - MSE loss preserved (not Huber) — keep alpha
    - No DART — no clear benefit, saves training time
    - Same 50 features as V4.7.5 (pruned from 70)
  Scorer: V4.7.6 post-processing (consistency bonus + vol discount)
    - Proven +3 V2 points, +DSR statistical significance

Fallback: V4.7.5 model if V4.8.0 model not available
"""

from pathlib import Path
from typing import List, Optional
import pandas as pd

from .v476_production_scorer import V476ProductionScorer

import logging
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class V480ProductionScorer(V476ProductionScorer):
    """V4.8.0 scorer -- 270d decay model + V4.7.6 post-processing"""

    def __init__(self, model_type: str = 'small_data'):
        self._v480_model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v480'
        super().__init__(model_type=model_type)

    def _load_models(self):
        """Try v480 model first, fallback to v475"""
        v480_files = list(self._v480_model_dir.glob('v480_*.pkl'))
        if v480_files:
            self.model_dir = self._v480_model_dir
            latest = max(v480_files, key=lambda f: f.stat().st_mtime)
            self._load_model_from_file(latest, label='V4.8.0')
            return

        # Fallback to V4.7.5 model
        super()._load_models()

    def _load_model_from_file(self, model_path, label='V4.8.0'):
        """Load model from specific file (reuses V4.7.5 parsing logic)"""
        import joblib, pickle, numpy as np
        try:
            model_data = joblib.load(model_path)
        except Exception:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

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
        self.target_weights = {'label_3d': 0.00, 'label_5d': 0.00, 'label_10d': 0.60, 'label_15d': 0.40}

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

        raw_bounds = model_data.get('winsorize_bounds')
        if raw_bounds:
            self.winsorize_bounds = {k: tuple(v) for k, v in raw_bounds.items()} if isinstance(raw_bounds, dict) else raw_bounds

        self.bear_models = {}
        self.isotonic_calibration = {}

        raw_quantiles = model_data.get('global_quantiles')
        if raw_quantiles is not None:
            self.global_quantiles = np.array(raw_quantiles)
        else:
            quantiles_path = self.model_dir / 'global_quantiles.npy'
            if quantiles_path.exists():
                self.global_quantiles = np.load(quantiles_path)

        rec_path = self.model_dir / 'recommendation_thresholds.json'
        if rec_path.exists():
            import json as _json
            with open(rec_path, 'r') as f:
                self.recommendation_thresholds = _json.load(f)
        else:
            self.recommendation_thresholds = model_data.get('recommendation_thresholds')

        self.weights = self._clip_icir_weights(self.weights)

        gq_status = "continuous" if self.global_quantiles is not None else "cross-sectional"
        print(f"{label} loaded: {list(self.models.keys())} [{gq_status} scoring, {len(self.feature_cols)} features]")
        print(f"  file: {model_path.name}")
