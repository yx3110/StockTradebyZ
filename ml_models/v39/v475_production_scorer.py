#!/usr/bin/env python3
"""
V4.7.5 production scorer -- V4.7.3 base + continuous scoring + built-in composite ranking

Changes from V4.7.3:
1. Continuous interpolation scoring: np.interp replaces np.searchsorted
   - Eliminates head tie-breaking (~5 stocks sharing same score -> unique scores)
   - Reduces turnover from discretization noise
2. Built-in composite multi-horizon ranking
   - Scorer directly outputs rank_score for report generation
   - No dependency on backtest layer for ranking
3. Adaptive target weights from OOS ICIR (if embedded in model)

Inherited from V4.7.3:
- No Meta-Learner, no Combined Isotonic (anti-compression design)
- ICIR weights clipped [0.08, 0.50]
- Bear Specialist + Per-target Isotonic
- Enhanced executability filters + liquidity discount
"""

import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
from typing import Dict, List, Optional

from .v473_production_scorer import V473ProductionScorer

import logging
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Composite ranking weights -- 10d+15d after multi-model ablation (2026-03-08)
# V4.7.5 ablation (511 days): 10d+15d = AnnRet +116.5%, Sharpe 1.873, ICIR 1.114
# vs pure 10d: AnnRet +98.2%, Sharpe 1.840, ICIR 1.067
# vs old (0.10/0.20/0.40/0.30): AnnRet +111.7%, Sharpe 1.793
# 3d/5d consistently weakest across all models, 15d adds value to 10d
COMPOSITE_WEIGHTS = {
    'pred_3d': 0.00,
    'pred_5d': 0.00,
    'pred_10d': 0.60,
    'pred_15d': 0.40,
}


class V475ProductionScorer(V473ProductionScorer):
    """V4.7.5 production scorer -- continuous scoring + composite ranking on V4.7.3 base"""

    def __init__(self, model_type: str = 'small_data'):
        self._v475_model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v475'
        super().__init__(model_type=model_type)

    def _load_models(self):
        """Override to use v475 model directory, fallback to v473"""
        self.model_dir = self._v475_model_dir
        if self.model_type == 'rolling':
            self._load_rolling_models()
        else:
            self._load_v475_model()

    def _load_v475_model(self):
        """Load v4.7.5 model -- same structure as V4.7.3, potentially fewer features"""
        model_files = list(self.model_dir.glob('v475_*.pkl'))
        fallback = False
        if not model_files:
            # Fallback: use V4.7.3 model with V4.7.5 scorer (Phase 1 validation)
            v473_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v473'
            model_files = list(v473_dir.glob('v473_*.pkl'))
            if not model_files:
                print("V4.7.5: no model found in v475/ or v473/")
                return
            fallback = True

        latest = max(model_files, key=lambda f: f.stat().st_mtime)
        try:
            model_data = joblib.load(latest)
        except Exception:
            with open(latest, 'rb') as f:
                model_data = pickle.load(f)

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

        # V4.7.5: adaptive target weights from OOS ICIR
        # DISABLED: shifts too much weight to 3d (noisy), degrades consistency & MaxDD
        adaptive_tw = model_data.get('adaptive_target_weights')
        # if adaptive_tw:
        #     self.target_weights = adaptive_tw

        # Metadata
        self.cascade = False
        self.cascade_feature_names = None
        self.dual_stream = False
        self.rank_normalized = False
        self.robust_zscore = model_data.get('robust_zscore', True)
        self.extra_features_from_daily_basic = model_data.get('extra_features_from_daily_basic', None)
        self.extra_tech_features = model_data.get('extra_features_from_tech_indicators', None)
        self.stock_rank_cols = model_data.get('stock_feature_cols', None)

        # V4.7.1 feature lists
        self.extra_features_financial = model_data.get('extra_features_financial', [])
        self.extra_features_microstructure = model_data.get('extra_features_microstructure', [])
        self.extra_features_reversal = model_data.get('extra_features_reversal', [])
        self.extra_features_risk = model_data.get('extra_features_risk', [])

        # Winsorize bounds
        raw_bounds = model_data.get('winsorize_bounds')
        if raw_bounds:
            self.winsorize_bounds = {k: tuple(v) for k, v in raw_bounds.items()} if isinstance(raw_bounds, dict) else raw_bounds

        # V4.7.5: disable bear blend + isotonic (ablation confirmed harmful)
        # Bear blend: IC worse 57.5% of days, only 42.5% better
        # Isotonic: step function compresses pred_Xd to 2-3 discrete values
        self.bear_models = {}
        self.isotonic_calibration = {}

        # No Meta-Learner, no Combined Isotonic (V4.7.3 design preserved)

        # Global quantiles
        raw_quantiles = model_data.get('global_quantiles')
        if raw_quantiles is not None:
            self.global_quantiles = np.array(raw_quantiles)
        else:
            quantiles_path = self.model_dir / 'global_quantiles.npy'
            if quantiles_path.exists():
                self.global_quantiles = np.load(quantiles_path)

        # Recommendation thresholds
        self.recommendation_thresholds = model_data.get('recommendation_thresholds')
        if not self.recommendation_thresholds:
            rec_path = self.model_dir / 'recommendation_thresholds.json'
            if rec_path.exists():
                import json as _json
                with open(rec_path, 'r') as f:
                    self.recommendation_thresholds = _json.load(f)

        # ICIR weights: clip to [0.08, 0.50] (inherited from V4.7.3)
        self.weights = self._clip_icir_weights(self.weights)

        wf = model_data.get('walk_forward_metrics', {})
        gq_status = "continuous" if self.global_quantiles is not None else "cross-sectional"
        src = "v473 fallback" if fallback else "v475"
        print(f"V4.7.5 loaded [{src}]: {list(self.models.keys())} [{gq_status} scoring, {len(self.feature_cols)} features]")
        print(f"  file: {latest.name}")
        print(f"  target_weights: {self.target_weights}")
        if adaptive_tw:
            print(f"  (adaptive from OOS ICIR)")
        if wf:
            for t, m in wf.items():
                print(f"  WF {t}: ICIR={m.get('mean_icir', 0):.4f}")

    # ========== V4.7.5 core: continuous interpolation scoring ==========

    def _to_global_score(self, combined_pred: np.ndarray) -> np.ndarray:
        """Continuous interpolation scoring -- replaces V4.7.3's searchsorted discretization.

        V4.7.3: searchsorted -> 1001 discrete values -> ~5 stocks tie at head
        V4.7.5: np.interp -> truly continuous float -> every stock gets unique score
        """
        if self.global_quantiles is not None and len(self.global_quantiles) > 1:
            percentile_grid = np.linspace(0, 100, len(self.global_quantiles))
            scores = np.interp(combined_pred, self.global_quantiles, percentile_grid)
            return np.clip(scores, 0, 100)
        else:
            # Fallback: cross-sectional percentile (already continuous)
            if len(combined_pred) > 1:
                from scipy import stats
                ranks = stats.rankdata(combined_pred)
                percentiles = (ranks - 1) / (len(ranks) - 1) * 100
                scores = 30 + percentiles * 0.6
            else:
                scores = np.array([60.0])
            return scores

    # ========== V4.7.5: built-in composite ranking ==========

    def _compute_composite_rank_score(self, results: Dict[str, Dict]) -> Dict[str, Dict]:
        """Multi-horizon weighted ranking fusion.

        For each pred_Xd, compute cross-sectional percentile rank [0,1],
        then weighted sum: 10d*0.60 + 15d*0.40 (3d/5d disabled).
        """
        codes = list(results.keys())
        if not codes:
            return results

        n = len(codes)
        if n <= 1:
            for code in codes:
                results[code]['rank_score'] = results[code].get('score', 50.0)
            return results

        from scipy.stats import rankdata

        rank_arrays = {}
        for field in COMPOSITE_WEIGHTS:
            values = np.array([results[c].get(field, 0) for c in codes])
            ranks = rankdata(values, method='average')
            rank_arrays[field] = (ranks - 1) / max(n - 1, 1)

        composite = np.zeros(n)
        for field, weight in COMPOSITE_WEIGHTS.items():
            composite += weight * rank_arrays[field]

        for i, code in enumerate(codes):
            results[code]['rank_score'] = float(composite[i])

        return results

    # ========== V4.7.5: disable regime weight adjustment ==========

    def _get_regime_target_weights(self, date: str) -> dict:
        """Override: fixed pure-10d weights, no regime adjustment.

        Ablation (1384 days): regime weights only better 13.9% of days vs fixed.
        """
        return dict(self.target_weights)

    # ========== Override predict_scores ==========

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """V4.7.5 scoring pipeline -- V4.7.3 base + continuous scoring.

        rank_score = 0.6*pred_10d + 0.4*pred_15d (ablation optimal).
        """
        results = super().predict_scores(stock_codes, date)
        for code, data in results.items():
            data['rank_score'] = 0.60 * data.get('pred_10d', 0) + 0.40 * data.get('pred_15d', 0)
        return results

    def predict_scores_from_preloaded(self, stock_codes: List[str], date: str,
                                       features_df: Optional[pd.DataFrame]) -> Dict[str, Dict]:
        """V4.7.5 batch scoring with preloaded features"""
        results = super().predict_scores_from_preloaded(stock_codes, date, features_df)
        for code, data in results.items():
            data['rank_score'] = 0.60 * data.get('pred_10d', 0) + 0.40 * data.get('pred_15d', 0)
        return results

    # ========== V4.7.5: recommendation overrides (10d+15d composite) ==========

    def _recommendation_from_composite(self, pred_3d: float, pred_5d: float,
                                        pred_10d: float, pred_15d: float = 0.0) -> str:
        """投资建议 -- 基于 0.6*10d + 0.4*15d composite"""
        composite = 0.6 * pred_10d + 0.4 * pred_15d

        t = self.recommendation_thresholds
        if t:
            if composite >= t['strong_buy']:
                return '强烈买入'
            elif composite >= t['buy']:
                return '买入'
            elif composite >= t['cautious']:
                return '谨慎买入'
            elif composite >= t['hold']:
                return '观望'
            else:
                return '回避'

        # fallback: 基于 composite 绝对值
        if composite >= 0.008:
            return '强烈买入'
        elif composite >= 0.005:
            return '买入'
        elif composite >= 0.002:
            return '谨慎买入'
        elif composite >= -0.001:
            return '观望'
        return '回避'

    def _risk_level_from_composite(self, pred_3d: float, pred_5d: float,
                                    pred_10d: float, pred_15d: float = 0.0) -> str:
        """风险等级 -- 基于 0.6*10d + 0.4*15d composite"""
        composite = 0.6 * pred_10d + 0.4 * pred_15d

        t = self.recommendation_thresholds
        if t:
            if composite >= t['buy']:
                return 'low'
            elif composite >= t['hold']:
                return 'medium'
            else:
                return 'high'

        if composite >= 0.005:
            return 'low'
        elif composite >= -0.001:
            return 'medium'
        return 'high'
