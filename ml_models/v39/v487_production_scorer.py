#!/usr/bin/env python3
"""
V4.8.7 production scorer -- 69 features + RRF + 共识投票 + Head Refiner

Architecture:
  Base: V4.8.6 (V4.8.5 + 3 BRAIN + 5 V482 + CatBoost YetiRank NDCG@10)
  Ensemble: RRF (k=60) for base predictions
  Head discrimination:
    1. Consensus voting: per-model Top-K vote counting
    2. Head Refiner: LightGBM classifier trained on meta-features
       to identify true top performers (二阶段头部精筛)

Output fields per stock:
  - score, pred_3d/5d/10d/15d, rank_score (from base pipeline)
  - consensus_count, pred_confidence, consensus_score (consensus voting)
  - head_refiner_proba (probability of being true top performer)

Fallback chain: v487 -> v486 -> v485 -> v484 -> v481 -> v475
"""

import numpy as np
import joblib
from pathlib import Path
from scipy.stats import rankdata
from .v486_production_scorer import V486ProductionScorer

import logging
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CONSENSUS_TOP_K = 20
CONSENSUS_TARGET = '10d'


class V487ProductionScorer(V486ProductionScorer):
    """V4.8.7 scorer -- RRF + consensus voting + head refiner"""

    def __init__(self, model_type: str = 'small_data'):
        self._v487_model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v487'
        self._head_refiner = None
        self._head_refiner_meta = None
        super().__init__(model_type=model_type)
        self._load_head_refiner()

    def _load_models(self):
        """Try v487 model first, fallback to v486"""
        v487_files = list(self._v487_model_dir.glob('v487_*.pkl'))
        if v487_files:
            self.model_dir = self._v487_model_dir
            latest = max(v487_files, key=lambda f: f.stat().st_mtime)
            self._load_model_from_file(latest, label='V4.8.7')
            return
        super()._load_models()

    def _load_head_refiner(self):
        """Load head refiner model if available."""
        hr_files = list(self._v487_model_dir.glob('head_refiner_*.pkl'))
        if hr_files:
            latest = max(hr_files, key=lambda f: f.stat().st_mtime)
            try:
                data = joblib.load(latest)
                self._head_refiner = data['model']
                self._head_refiner_meta = data
                logger.info(f"  Head Refiner loaded: {latest.name} "
                            f"(AUC={data.get('test_auc', 0):.4f})")
            except Exception as e:
                logger.warning(f"Head Refiner load failed: {e}")

    def _compute_consensus_and_confidence(self, codes, per_model_preds):
        """Compute consensus voting and prediction confidence."""
        target = CONSENSUS_TARGET
        n = len(codes)

        if target not in per_model_preds or not per_model_preds[target]:
            return {code: {'consensus_count': 0, 'pred_confidence': 0.0,
                           'consensus_score': 0.0} for code in codes}

        preds = per_model_preds[target]
        if not preds:
            return {code: {'consensus_count': 0, 'pred_confidence': 0.0,
                           'consensus_score': 0.0} for code in codes}

        consensus_counts = np.zeros(n)
        pred_matrix = []

        for name, pred_arr in preds.items():
            if len(pred_arr) != n:
                continue
            pred_matrix.append(pred_arr)
            ranks = rankdata(-pred_arr, method='ordinal')
            consensus_counts += (ranks <= CONSENSUS_TOP_K).astype(float)

        pred_confidence = np.zeros(n)
        if len(pred_matrix) >= 2:
            pred_stack = np.array(pred_matrix)
            pred_mean = np.mean(pred_stack, axis=0)
            pred_std = np.std(pred_stack, axis=0)
            pred_confidence = pred_mean / np.maximum(pred_std, 1e-8)

        confidence_bonus = 1.0 / (1.0 + np.exp(-pred_confidence))
        consensus_score = consensus_counts + confidence_bonus * 0.5

        result = {}
        for i, code in enumerate(codes):
            result[code] = {
                'consensus_count': int(consensus_counts[i]),
                'pred_confidence': float(pred_confidence[i]),
                'consensus_score': float(consensus_score[i]),
            }
        return result

    def _compute_head_refiner_scores(self, codes, per_model_preds, features_df=None):
        """Run head refiner on the current cross-section.

        Builds meta-features from per-model predictions and runs the
        trained LightGBM classifier to get P(true_top) for each stock.
        """
        if self._head_refiner is None:
            return {}

        target = CONSENSUS_TARGET
        n = len(codes)

        if target not in per_model_preds or not per_model_preds[target]:
            return {}

        preds = per_model_preds[target]
        meta_info = self._head_refiner_meta
        meta_feature_names = meta_info.get('meta_feature_names', [])
        stock_feature_names = meta_info.get('stock_feature_names', [])

        # Get available model predictions
        model_names = list(self.models.get(target, {}).keys())
        avail_names = [nm for nm in model_names if nm in preds and len(preds[nm]) == n]
        if not avail_names:
            return {}

        pred_matrix = np.column_stack([preds[nm] for nm in avail_names])
        n_models = pred_matrix.shape[1]

        # Build meta-features (must match training exactly)
        meta = {}
        for nm in avail_names:
            meta[f'pred_{nm}'] = preds[nm]

        meta['pred_mean'] = np.mean(pred_matrix, axis=1)
        meta['pred_std'] = np.std(pred_matrix, axis=1)
        meta['pred_min'] = np.min(pred_matrix, axis=1)
        meta['pred_max'] = np.max(pred_matrix, axis=1)
        meta['pred_range'] = meta['pred_max'] - meta['pred_min']
        meta['pred_sharpe'] = meta['pred_mean'] / np.maximum(meta['pred_std'], 1e-8)

        # Consensus count
        consensus = np.zeros(n)
        for j in range(n_models):
            ranks = rankdata(-pred_matrix[:, j], method='ordinal')
            consensus += (ranks <= CONSENSUS_TOP_K).astype(float)
        meta['consensus_count'] = consensus
        meta['rank_in_day'] = rankdata(-meta['pred_mean'], method='average') / n

        # Rank agreement
        rank_mat = np.zeros((n, n_models))
        for j in range(n_models):
            rank_mat[:, j] = rankdata(-pred_matrix[:, j], method='average')
        meta['rank_std'] = np.std(rank_mat, axis=1)
        meta['rank_mean'] = np.mean(rank_mat, axis=1)

        # Assemble meta feature array
        meta_arr = np.column_stack([meta.get(col, np.zeros(n)) for col in meta_feature_names])

        # Stock features (if features_df available)
        if features_df is not None and len(features_df) == n:
            stock_vals = []
            for col in stock_feature_names:
                if col in features_df.columns:
                    stock_vals.append(features_df[col].fillna(0).values)
                else:
                    stock_vals.append(np.zeros(n))
            stock_arr = np.column_stack(stock_vals) if stock_vals else np.zeros((n, len(stock_feature_names)))
        else:
            stock_arr = np.zeros((n, len(stock_feature_names)))

        X_hr = np.column_stack([meta_arr, stock_arr])
        X_hr = np.nan_to_num(X_hr, nan=0.0, posinf=0.0, neginf=0.0)

        # Run classifier
        try:
            proba = self._head_refiner.predict(X_hr)
            return {codes[i]: float(proba[i]) for i in range(n)}
        except Exception as e:
            logger.warning(f"Head refiner prediction failed: {e}")
            return {}

    def predict_scores(self, stock_codes, date):
        """V4.8.7 scoring with consensus voting + head refiner."""
        from .v486_production_scorer import V486ProductionScorer
        results = V486ProductionScorer.predict_scores(self, stock_codes, date)

        per_model_preds = getattr(self, '_per_model_preds', {})
        pred_codes = getattr(self, '_last_pred_codes', [])

        if per_model_preds and pred_codes and CONSENSUS_TARGET in per_model_preds:
            first_pred = next(iter(per_model_preds[CONSENSUS_TARGET].values()), None)
            if first_pred is not None and len(first_pred) == len(pred_codes):
                # Consensus voting
                consensus_data = self._compute_consensus_and_confidence(
                    pred_codes, per_model_preds)
                for code, cdata in consensus_data.items():
                    if code in results:
                        results[code]['consensus_count'] = cdata['consensus_count']
                        results[code]['pred_confidence'] = cdata['pred_confidence']
                        results[code]['consensus_score'] = cdata['consensus_score']

                # Head refiner
                hr_scores = self._compute_head_refiner_scores(pred_codes, per_model_preds)
                for code, proba in hr_scores.items():
                    if code in results:
                        results[code]['head_refiner_proba'] = proba

        # ETF flagging
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
