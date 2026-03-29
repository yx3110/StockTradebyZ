#!/usr/bin/env python3
"""
V4.8.5 production scorer -- 61 features + Q95 Widen-then-Concentrate

Architecture:
  Base: V4.8.4 (61 features, A股+ETF training)
  Head discrimination: Widen-then-Concentrate pipeline
    Stage 1: MSE ensemble → Top-30 candidates (宽选)
    Stage 2: Q95 quantile model reranks within Top-30 (精筛)

Output fields per stock:
  - score, pred_3d/5d/10d/15d, rank_score (from base pipeline)
  - q95_pred_10d: Q95 model prediction for 10d return right tail
  - head_rank: final rank after Widen-then-Concentrate (1=best)
  - in_head_pool: whether stock is in the Stage 1 Top-30 pool

Fallback chain: v485 model -> v484 model -> v481 model -> v475 model
"""

import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, List
from scipy.stats import rankdata

from .v484_production_scorer import V484ProductionScorer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Widen-then-Concentrate parameters
WIDEN_TOP_K = 30   # Stage 1: MSE ensemble selects top-K candidates
HEAD_SELECT = 10   # Stage 2: Q95 selects top-N from the pool


class V485ProductionScorer(V484ProductionScorer):
    """V4.8.5 scorer -- 61 features + Q95 Widen-then-Concentrate"""

    def __init__(self, model_type: str = 'small_data'):
        self._v485_model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v485'
        self._q95_model = None
        super().__init__(model_type=model_type)
        self._load_q95_model()

    def _load_models(self):
        """Try v485 model first, fallback to v484"""
        v485_files = list(self._v485_model_dir.glob('v485_*.pkl'))
        if v485_files:
            self.model_dir = self._v485_model_dir
            latest = max(v485_files, key=lambda f: f.stat().st_mtime)
            self._load_model_from_file(latest, label='V4.8.5')
            return
        super()._load_models()

    def _load_q95_model(self):
        """Load Q95 quantile model for head discrimination."""
        q95_files = sorted(self._v485_model_dir.glob('q95_model_*.pkl'))
        if q95_files:
            latest = q95_files[-1]
            try:
                data = joblib.load(latest)
                self._q95_model = data['models'].get('10d')
                if self._q95_model:
                    logger.info(f"  Q95 model loaded: {latest.name}")
            except Exception as e:
                logger.warning(f"Q95 model load failed: {e}")

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """V4.8.5 scoring with Q95 Widen-then-Concentrate.

        Pipeline:
        1. Run base MSE ensemble (inherited from V484) → pred_10d for all stocks
        2. If Q95 model available:
           a. Run Q95 on same features → q95_pred_10d
           b. Stage 1: Select Top-30 by MSE composite (widen)
           c. Stage 2: Rerank Top-30 by Q95 prediction (concentrate)
           d. Assign head_rank (1=best by Q95 within pool)
        3. Flag ETFs
        """
        results = super().predict_scores(stock_codes, date)

        # Q95 Widen-then-Concentrate
        per_model_preds = getattr(self, '_per_model_preds', {})
        pred_codes = getattr(self, '_last_pred_codes', [])

        if self._q95_model and pred_codes and '10d' in per_model_preds:
            # Get the feature matrix that was used for base predictions
            # We need to rerun Q95 on the same features
            # The features are cached in the parent's predict_scores call
            features_X = getattr(self, '_last_X', None)

            if features_X is not None and len(features_X) == len(pred_codes):
                try:
                    q95_pred = self._q95_model.predict(features_X)
                    n = len(pred_codes)

                    # MSE composite for Stage 1 ranking
                    mse_composite = np.array([
                        results.get(c, {}).get('pred_10d', 0) for c in pred_codes
                    ])
                    mse_rank = rankdata(-mse_composite, method='ordinal')

                    # Stage 1: Top-K by MSE
                    pool_mask = mse_rank <= WIDEN_TOP_K
                    pool_idx = np.where(pool_mask)[0]

                    # Stage 2: Rerank pool by Q95
                    if len(pool_idx) >= 3:
                        q95_in_pool = q95_pred[pool_idx]
                        q95_pool_rank = rankdata(-q95_in_pool, method='ordinal')

                        # Assign head_rank for pool stocks
                        for ii, idx in enumerate(pool_idx):
                            code = pred_codes[idx]
                            if code in results:
                                results[code]['q95_pred_10d'] = float(q95_pred[idx])
                                results[code]['head_rank'] = int(q95_pool_rank[ii])
                                results[code]['in_head_pool'] = True

                    # Non-pool stocks
                    for i, code in enumerate(pred_codes):
                        if code in results and 'in_head_pool' not in results[code]:
                            results[code]['q95_pred_10d'] = float(q95_pred[i])
                            results[code]['head_rank'] = WIDEN_TOP_K + int(mse_rank[i])
                            results[code]['in_head_pool'] = False

                except Exception as e:
                    logger.warning(f"Q95 prediction failed: {e}")

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
