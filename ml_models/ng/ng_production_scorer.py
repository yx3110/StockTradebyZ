#!/usr/bin/env python3
"""
NG Production Scorer — loads NG model, scores all stocks for a date.

Architecture:
  Model: NG trained model (62 features: 52 stock + 10 market)
  Data:  Features loaded directly from ng_feature_cache table
  Scoring pipeline:
    1. Load features from ng_feature_cache for the given date
    2. Cross-sectional Robust Z-Score on stock features
    3. Apply winsorization bounds from model
    4. Predict with ensemble for each target (3d, 5d, 10d)
    5. Composite ranking: 5d x 0.50 + 10d x 0.35 + 3d x 0.15
    6. Convert to percentile score (0-100)
    7. Assign recommendation: >=95 strong buy, >=85 buy, >=70 cautious buy

This scorer is self-contained — it does NOT inherit from V485ProductionScorer.
It reads directly from ng_feature_cache, avoiding the legacy feature pipeline.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')

# Import feature names from the trainer module
from .ng_trainer import (
    STOCK_FEATURE_NAMES,
    MARKET_FEATURE_NAMES,
    ALL_FEATURE_NAMES,
)

# Optional fast JSON
try:
    import orjson
    _json_loads = orjson.loads
except ImportError:
    _json_loads = json.loads

# Composite weights for ranking
COMPOSITE_WEIGHTS = {
    '3d': 0.15,
    '5d': 0.50,
    '10d': 0.35,
}

# Recommendation thresholds (percentile-based)
REC_THRESHOLDS = {
    'strong_buy': 95,
    'buy': 85,
    'cautious_buy': 70,
}


class NGProductionScorer:
    """NG Production Scorer — self-contained scoring from ng_feature_cache."""

    def __init__(self, db_path: str = None, model_path: str = None):
        self.db_path = db_path or DB_PATH
        self.model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'ng'

        # Model components
        self.models = {}       # {target: {model_name: model}}
        self.weights = {}      # {target: {model_name: weight}}
        self.feature_names = list(ALL_FEATURE_NAMES)
        self.stock_feature_cols = list(STOCK_FEATURE_NAMES)
        self.macro_feature_cols = list(MARKET_FEATURE_NAMES)
        self.winsorize_bounds = None
        self.global_quantiles = None
        self.recommendation_thresholds = None
        self.target_weights = dict(COMPOSITE_WEIGHTS)

        # Load model
        self._load_model(model_path)

    def _load_model(self, model_path: str = None):
        """Load the latest NG model from trained_models/ng/."""
        if model_path:
            path = Path(model_path)
        else:
            ng_files = sorted(
                self.model_dir.glob('ng_*.pkl'),
                key=lambda f: f.stat().st_mtime
            )
            if not ng_files:
                logger.warning("No NG model found in %s", self.model_dir)
                print(f"NG scorer: No model found in {self.model_dir}")
                return
            path = ng_files[-1]

        try:
            model_data = joblib.load(str(path))
        except Exception as e:
            logger.error("Failed to load NG model %s: %s", path, e)
            return

        # Parse model data
        raw_models = model_data.get('models', {})
        self.models = {}
        self.weights = {}
        for target, target_data in raw_models.items():
            if isinstance(target_data, dict) and 'models' in target_data:
                self.models[target] = target_data['models']
                self.weights[target] = target_data.get('weights', {})
            else:
                self.models[target] = target_data

        # Feature names from model (fallback to static list)
        self.feature_names = model_data.get('feature_names', list(ALL_FEATURE_NAMES))
        self.stock_feature_cols = model_data.get('stock_feature_cols', list(STOCK_FEATURE_NAMES))
        self.macro_feature_cols = model_data.get('macro_feature_cols', list(MARKET_FEATURE_NAMES))

        # Winsorize bounds
        raw_bounds = model_data.get('winsorize_bounds')
        if raw_bounds and isinstance(raw_bounds, dict):
            self.winsorize_bounds = {k: tuple(v) for k, v in raw_bounds.items()}
        elif raw_bounds and isinstance(raw_bounds, list):
            self.winsorize_bounds = raw_bounds
        else:
            self.winsorize_bounds = None

        # Global quantiles for percentile scoring
        raw_quantiles = model_data.get('global_quantiles')
        if raw_quantiles is not None:
            self.global_quantiles = np.array(raw_quantiles)
        else:
            gq_path = self.model_dir / 'global_quantiles.npy'
            if gq_path.exists():
                self.global_quantiles = np.load(gq_path)

        # Recommendation thresholds
        self.recommendation_thresholds = model_data.get('recommendation_thresholds')
        if self.recommendation_thresholds is None:
            rec_path = self.model_dir / 'recommendation_thresholds.json'
            if rec_path.exists():
                with open(rec_path, 'r') as f:
                    self.recommendation_thresholds = json.load(f)

        # Target weights from model
        stored_weights = model_data.get('target_weights', {})
        if stored_weights:
            # Convert label_Xd → Xd format if needed
            self.target_weights = {}
            for k, v in stored_weights.items():
                key = k.replace('label_', '') if k.startswith('label_') else k
                self.target_weights[key] = v

        # ICIR-clipped weights
        ensemble_weights = model_data.get('ensemble_weights', {})
        if ensemble_weights:
            for target_key in self.weights:
                ew_key = f'label_{target_key}'
                if ew_key in ensemble_weights:
                    self.weights[target_key] = ensemble_weights[ew_key]

        n_targets = len(self.models)
        n_features = len(self.feature_names)
        scoring_mode = "continuous" if self.global_quantiles is not None else "cross-sectional"
        print(f"NG scorer loaded: {list(self.models.keys())} "
              f"[{scoring_mode} scoring, {n_features} features]")
        print(f"  file: {path.name}")

    # ------------------------------------------------------------------
    # Feature loading from ng_feature_cache
    # ------------------------------------------------------------------

    def _load_features(self, stock_codes: List[str], date: str) -> Optional[pd.DataFrame]:
        """Load features from ng_feature_cache for a given date.

        Args:
            stock_codes: List of stock codes to score
            date: Trade date (YYYY-MM-DD or YYYYMMDD)

        Returns:
            DataFrame with columns: code, [62 feature names]
            or None if no data
        """
        # Normalize date format
        if isinstance(date, str) and len(date) == 8 and date.isdigit():
            date = f"{date[:4]}-{date[4:6]}-{date[6:]}"

        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            # Load all stocks for the date (for cross-sectional z-score)
            query = """
            SELECT code, features_json,
                   market_return_5d, market_return_20d, market_volatility_20d,
                   market_breadth, market_new_high_ratio, northbound_flow_5d,
                   market_volume_ratio, market_drawdown, vix_proxy,
                   market_momentum_diff
            FROM ng_feature_cache
            WHERE trade_date = ?
            """
            df_raw = pd.read_sql(query, conn, params=[date])
        finally:
            conn.close()

        if df_raw.empty:
            logger.warning("No ng_feature_cache data for date %s", date)
            return None

        # Parse features_json
        parsed = df_raw['features_json'].apply(_json_loads).tolist()
        df_stock = pd.DataFrame(parsed)

        # Ensure all stock feature columns exist
        for col in STOCK_FEATURE_NAMES:
            if col not in df_stock.columns:
                df_stock[col] = np.nan

        # Build result
        result = pd.DataFrame()
        result['code'] = df_raw['code'].values

        for col in STOCK_FEATURE_NAMES:
            result[col] = pd.to_numeric(df_stock[col], errors='coerce').fillna(0.0).values

        for col in MARKET_FEATURE_NAMES:
            if col in df_raw.columns:
                result[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0.0).values
            else:
                result[col] = 0.0

        return result

    # ------------------------------------------------------------------
    # Cross-sectional normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _robust_zscore(data: np.ndarray) -> np.ndarray:
        """Apply Robust Z-Score (MAD-based) across all rows for each column.

        Args:
            data: (n_samples, n_features) array

        Returns:
            Normalized array, clipped to [-3, 3]
        """
        result = data.copy()
        for col in range(data.shape[1]):
            values = data[:, col]
            median = np.nanmedian(values)
            mad = np.nanmedian(np.abs(values - median)) * 1.4826
            if mad < 1e-8:
                mad = 1e-8
            result[:, col] = np.clip((values - median) / mad, -3, 3)
        return result

    # ------------------------------------------------------------------
    # Ensemble prediction
    # ------------------------------------------------------------------

    def _ensemble_predict(self, X: np.ndarray, target: str) -> Optional[np.ndarray]:
        """Run ensemble prediction for a target.

        Args:
            X: Feature matrix (n_stocks, n_features)
            target: '3d', '5d', or '10d'

        Returns:
            Ensemble prediction array or None if target not available
        """
        if target not in self.models or not self.models[target]:
            return None

        preds = {}
        for name, model in self.models[target].items():
            try:
                if name == 'xgb':
                    import xgboost as xgb
                    preds[name] = model.predict(xgb.DMatrix(X))
                else:
                    preds[name] = model.predict(X)
            except Exception as e:
                logger.warning("NG predict failed for %s/%s: %s", target, name, e)
                continue

        if not preds:
            return None

        # Rescale rank models to regression scale (exclude quantile models)
        regression_names = [n for n in preds if n not in ('lgb_rank', 'lgb_listnet', 'lgb_q95')]
        rank_names = [n for n in preds if n in ('lgb_rank', 'lgb_listnet')]
        if regression_names and rank_names:
            reg_means = [np.mean(preds[n]) for n in regression_names]
            reg_stds = [max(np.std(preds[n]), 1e-8) for n in regression_names]
            t_mean, t_std = np.mean(reg_means), np.mean(reg_stds)
            for rn in rank_names:
                rp = preds[rn]
                rp_std = max(np.std(rp), 1e-8)
                preds[rn] = (rp - np.mean(rp)) / rp_std * t_std + t_mean

        # Weighted average (exclude Q95 from composite)
        COMPOSITE_EXCLUDE = ('lgb_q95',)
        target_w = self.weights.get(target, {})
        weighted_sum = np.zeros(X.shape[0])
        total_weight = 0.0

        for name, pred in preds.items():
            if name in COMPOSITE_EXCLUDE:
                continue
            w = target_w.get(name, 0.2)
            weighted_sum += w * pred
            total_weight += w

        if total_weight > 0:
            return weighted_sum / total_weight
        return weighted_sum

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _to_global_score(self, combined_pred: np.ndarray) -> np.ndarray:
        """Convert combined prediction to global percentile score (0-100).

        Uses global_quantiles from training if available, otherwise
        falls back to cross-sectional percentile.
        """
        if self.global_quantiles is not None and len(self.global_quantiles) > 0:
            gq = np.array(self.global_quantiles)
            scores = np.searchsorted(gq, combined_pred) / len(gq) * 100
            return np.clip(scores, 0, 100)
        else:
            # Cross-sectional percentile
            n = len(combined_pred)
            if n <= 1:
                return np.full(n, 50.0)
            ranks = rankdata(combined_pred, method='average')
            return (ranks - 1) / (n - 1) * 100

    def _get_recommendation(self, score: float) -> str:
        """Map score to recommendation string."""
        if score >= REC_THRESHOLDS['strong_buy']:
            return '强烈买入'
        elif score >= REC_THRESHOLDS['buy']:
            return '买入'
        elif score >= REC_THRESHOLDS['cautious_buy']:
            return '谨慎买入'
        else:
            return '观望'

    # ------------------------------------------------------------------
    # Main scoring API
    # ------------------------------------------------------------------

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """Score stocks for a given date.

        Args:
            stock_codes: List of stock codes to score
            date: Trade date (YYYY-MM-DD or YYYYMMDD)

        Returns:
            {code: {pred_3d, pred_5d, pred_10d, rank_score, score, recommendation}}
        """
        # Normalize date
        if isinstance(date, str) and len(date) == 8 and date.isdigit():
            date = f"{date[:4]}-{date[4:6]}-{date[6:]}"

        results = {}

        # Step 1: Load features (full cross-section for z-score)
        features_df = self._load_features(stock_codes, date)

        if features_df is None or len(features_df) == 0:
            for code in stock_codes:
                results[code] = {
                    'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0,
                    'rank_score': 0, 'recommendation': '观望',
                    'exec_filter': 'no_data',
                }
            return results

        # Step 2: Cross-sectional Robust Z-Score on stock features
        stock_cols_idx = [
            self.feature_names.index(c) for c in self.stock_feature_cols
            if c in self.feature_names
        ]
        feature_matrix = features_df[self.feature_names].values.copy()
        if stock_cols_idx:
            stock_block = feature_matrix[:, stock_cols_idx]
            stock_block = self._robust_zscore(stock_block)
            feature_matrix[:, stock_cols_idx] = stock_block

        # Step 3: Apply winsorization bounds
        if self.winsorize_bounds:
            if isinstance(self.winsorize_bounds, dict):
                for i, fname in enumerate(self.feature_names):
                    if fname in self.winsorize_bounds:
                        lo, hi = self.winsorize_bounds[fname]
                        feature_matrix[:, i] = np.clip(feature_matrix[:, i], lo, hi)
            elif isinstance(self.winsorize_bounds, list):
                for i, (lo, hi) in enumerate(self.winsorize_bounds):
                    if i < feature_matrix.shape[1]:
                        feature_matrix[:, i] = np.clip(feature_matrix[:, i], lo, hi)

        # Replace NaN/Inf
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        codes = features_df['code'].tolist()

        # Step 4: Predict for each target
        predictions = {}
        for target in ['3d', '5d', '10d']:
            pred = self._ensemble_predict(feature_matrix, target)
            if pred is not None:
                predictions[target] = pred
            else:
                predictions[target] = np.zeros(len(codes))

        # Step 5: Composite ranking
        w = self.target_weights
        combined_pred = (
            w.get('3d', 0.15) * predictions.get('3d', np.zeros(len(codes))) +
            w.get('5d', 0.50) * predictions.get('5d', np.zeros(len(codes))) +
            w.get('10d', 0.35) * predictions.get('10d', np.zeros(len(codes)))
        )

        # Step 6: Convert to score
        scores = self._to_global_score(combined_pred)

        # Build results for all stocks in cross-section
        all_results = {}
        for i, code in enumerate(codes):
            all_results[code] = {
                'pred_3d': float(predictions['3d'][i]),
                'pred_5d': float(predictions['5d'][i]),
                'pred_10d': float(predictions['10d'][i]),
                'rank_score': float(combined_pred[i]),
                'score': float(scores[i]),
                'recommendation': self._get_recommendation(float(scores[i])),
            }

        # Filter to requested codes, fill missing
        for code in stock_codes:
            if code in all_results:
                results[code] = all_results[code]
            else:
                results[code] = {
                    'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0,
                    'rank_score': 0, 'recommendation': '观望',
                    'exec_filter': 'no_data',
                }

        return results

    def predict_scores_from_preloaded(self, stock_codes: List[str], date: str,
                                       features_df: pd.DataFrame) -> Dict[str, Dict]:
        """Score stocks using a preloaded features DataFrame.

        This avoids the DB query — useful for batch report generation where
        features are already loaded.

        Args:
            stock_codes: List of stock codes to score
            date: Trade date string
            features_df: DataFrame with columns: code, [52 stock features]
                         (market features will be loaded from DB if missing)

        Returns:
            Same format as predict_scores
        """
        if features_df is None or len(features_df) == 0:
            return {code: {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0,
                           'pred_10d': 0, 'rank_score': 0,
                           'recommendation': '观望', 'exec_filter': 'no_data'}
                    for code in stock_codes}

        # Normalize date
        if isinstance(date, str) and len(date) == 8 and date.isdigit():
            date = f"{date[:4]}-{date[4:6]}-{date[6:]}"

        # Ensure all feature columns exist
        for col in self.feature_names:
            if col not in features_df.columns:
                features_df[col] = 0.0

        # Z-score stock features
        stock_cols_idx = [
            self.feature_names.index(c) for c in self.stock_feature_cols
            if c in self.feature_names
        ]
        feature_matrix = features_df[self.feature_names].values.copy()
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        if stock_cols_idx:
            stock_block = feature_matrix[:, stock_cols_idx]
            stock_block = self._robust_zscore(stock_block)
            feature_matrix[:, stock_cols_idx] = stock_block

        # Winsorize
        if self.winsorize_bounds:
            if isinstance(self.winsorize_bounds, dict):
                for i, fname in enumerate(self.feature_names):
                    if fname in self.winsorize_bounds:
                        lo, hi = self.winsorize_bounds[fname]
                        feature_matrix[:, i] = np.clip(feature_matrix[:, i], lo, hi)
            elif isinstance(self.winsorize_bounds, list):
                for i, (lo, hi) in enumerate(self.winsorize_bounds):
                    if i < feature_matrix.shape[1]:
                        feature_matrix[:, i] = np.clip(feature_matrix[:, i], lo, hi)

        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        codes = features_df['code'].tolist()

        # Predict
        predictions = {}
        for target in ['3d', '5d', '10d']:
            pred = self._ensemble_predict(feature_matrix, target)
            predictions[target] = pred if pred is not None else np.zeros(len(codes))

        w = self.target_weights
        combined_pred = (
            w.get('3d', 0.15) * predictions['3d'] +
            w.get('5d', 0.50) * predictions['5d'] +
            w.get('10d', 0.35) * predictions['10d']
        )
        scores = self._to_global_score(combined_pred)

        results = {}
        for i, code in enumerate(codes):
            if code in stock_codes:
                results[code] = {
                    'pred_3d': float(predictions['3d'][i]),
                    'pred_5d': float(predictions['5d'][i]),
                    'pred_10d': float(predictions['10d'][i]),
                    'rank_score': float(combined_pred[i]),
                    'score': float(scores[i]),
                    'recommendation': self._get_recommendation(float(scores[i])),
                }

        for code in stock_codes:
            if code not in results:
                results[code] = {
                    'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0,
                    'rank_score': 0, 'recommendation': '观望',
                    'exec_filter': 'no_data',
                }

        return results
