#!/usr/bin/env python3
"""
NG v1.1.0 Production Scorer — loads NG model, scores all stocks for a date.

v1.1.0 changes:
  - 68 features (58 stock + 10 market), was 62
  - ICIR adaptive composite weights from model (not hardcoded)
  - Labels are industry excess returns
  - Backward compatible: auto-detects v1.0.0 vs v1.1.0 models
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

from .ng_trainer import (
    STOCK_FEATURE_NAMES,
    MARKET_FEATURE_NAMES,
    ALL_FEATURE_NAMES,
    NG_VERSION,
)
from .ng_schema import get_table_name

try:
    import orjson
    _json_loads = orjson.loads
except ImportError:
    _json_loads = json.loads

# Default composite weights (overridden by model's ICIR adaptive weights)
DEFAULT_COMPOSITE_WEIGHTS = {
    '3d': 0.10,
    '5d': 0.20,
    '10d': 0.35,
    '15d': 0.35,
}

REC_THRESHOLDS = {
    'strong_buy': 95,
    'buy': 85,
    'cautious_buy': 70,
}


class NGProductionScorer:
    """NG v1.1.0 Production Scorer — self-contained scoring from ng_feature_cache."""

    def __init__(self, db_path: str = None, model_path: str = None):
        self.db_path = db_path or DB_PATH
        self.model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'ng'

        self.models = {}
        self.weights = {}
        self.feature_names = list(ALL_FEATURE_NAMES)
        self.stock_feature_cols = list(STOCK_FEATURE_NAMES)
        self.macro_feature_cols = list(MARKET_FEATURE_NAMES)
        self.winsorize_bounds = None
        self.global_quantiles = None
        self.recommendation_thresholds = None
        self.target_weights = dict(DEFAULT_COMPOSITE_WEIGHTS)
        self.cache_table = get_table_name(NG_VERSION)  # default, updated by model version

        self._load_model(model_path)

    def _load_model(self, model_path: str = None):
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

        # Feature names from model
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

        # Global quantiles
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

        # Target weights from model (ICIR adaptive in v1.1.0)
        stored_weights = model_data.get('target_weights', {})
        if stored_weights:
            self.target_weights = {}
            for k, v in stored_weights.items():
                key = k.replace('label_', '') if k.startswith('label_') else k
                self.target_weights[key] = v

        # ICIR-clipped ensemble weights
        ensemble_weights = model_data.get('ensemble_weights', {})
        if ensemble_weights:
            for target_key in self.weights:
                ew_key = f'label_{target_key}'
                if ew_key in ensemble_weights:
                    self.weights[target_key] = ensemble_weights[ew_key]

        model_version = model_data.get('version', 'unknown')
        # Auto-detect cache table from model version
        self.cache_table = get_table_name(model_version)
        n_targets = len(self.models)
        n_features = len(self.feature_names)
        scoring_mode = "continuous" if self.global_quantiles is not None else "cross-sectional"
        print(f"NG scorer loaded ({model_version}): {list(self.models.keys())} "
              f"[{scoring_mode} scoring, {n_features} features]")
        print(f"  file: {path.name}")
        print(f"  cache table: {self.cache_table}")
        print(f"  composite weights: {self.target_weights}")

    # ------------------------------------------------------------------
    # Feature loading
    # ------------------------------------------------------------------

    def _load_features(self, stock_codes: List[str], date: str) -> Optional[pd.DataFrame]:
        if isinstance(date, str) and len(date) == 8 and date.isdigit():
            date = f"{date[:4]}-{date[4:6]}-{date[6:]}"

        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            query = f"""
            SELECT code, features_json,
                   market_return_5d, market_return_20d, market_volatility_20d,
                   market_breadth, market_new_high_ratio, northbound_flow_5d,
                   market_volume_ratio, market_drawdown, vix_proxy,
                   market_momentum_diff
            FROM {self.cache_table}
            WHERE trade_date = ?
            """
            df_raw = pd.read_sql(query, conn, params=[date])
        finally:
            conn.close()

        if df_raw.empty:
            logger.warning("No %s data for date %s", self.cache_table, date)
            return None

        parsed = df_raw['features_json'].apply(_json_loads).tolist()
        df_stock = pd.DataFrame(parsed)

        # Ensure expected columns (handles both v1.0.0 and v1.1.0 data)
        for col in self.stock_feature_cols:
            if col not in df_stock.columns:
                df_stock[col] = np.nan

        result = pd.DataFrame()
        result['code'] = df_raw['code'].values

        for col in self.stock_feature_cols:
            result[col] = pd.to_numeric(df_stock[col], errors='coerce').fillna(0.0).values

        for col in self.macro_feature_cols:
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
        if self.global_quantiles is not None and len(self.global_quantiles) > 0:
            gq = np.array(self.global_quantiles)
            scores = np.searchsorted(gq, combined_pred) / len(gq) * 100
            return np.clip(scores, 0, 100)
        else:
            n = len(combined_pred)
            if n <= 1:
                return np.full(n, 50.0)
            ranks = rankdata(combined_pred, method='average')
            return (ranks - 1) / (n - 1) * 100

    def _get_recommendation(self, score: float) -> str:
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
        if isinstance(date, str) and len(date) == 8 and date.isdigit():
            date = f"{date[:4]}-{date[4:6]}-{date[6:]}"

        results = {}

        features_df = self._load_features(stock_codes, date)

        if features_df is None or len(features_df) == 0:
            for code in stock_codes:
                results[code] = {
                    'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0,
                    'rank_score': 0, 'recommendation': '观望',
                    'exec_filter': 'no_data',
                }
            return results

        # Cross-sectional Robust Z-Score on stock features
        stock_cols_idx = [
            self.feature_names.index(c) for c in self.stock_feature_cols
            if c in self.feature_names
        ]
        feature_matrix = features_df[self.feature_names].values.copy()
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

        # Predict for each target
        predictions = {}
        for target in ['3d', '5d', '10d', '15d']:
            pred = self._ensemble_predict(feature_matrix, target)
            if pred is not None:
                predictions[target] = pred
            else:
                predictions[target] = np.zeros(len(codes))

        # Composite ranking using ICIR adaptive weights
        w = self.target_weights
        combined_pred = np.zeros(len(codes), dtype=float)
        for target_key in ['3d', '5d', '10d', '15d']:
            combined_pred += w.get(target_key, 0.0) * predictions.get(target_key, np.zeros(len(codes)))

        scores = self._to_global_score(combined_pred)

        all_results = {}
        for i, code in enumerate(codes):
            all_results[code] = {
                'pred_3d': float(predictions['3d'][i]),
                'pred_5d': float(predictions['5d'][i]),
                'pred_10d': float(predictions['10d'][i]),
                'pred_15d': float(predictions.get('15d', np.zeros(len(codes)))[i]),
                'rank_score': float(combined_pred[i]),
                'score': float(scores[i]),
                'recommendation': self._get_recommendation(float(scores[i])),
            }

        # Build mapping: strip suffix (.SH/.SZ/.BJ) for matching against cache codes
        for code in stock_codes:
            short_code = code.split('.')[0] if '.' in code else code
            if code in all_results:
                results[code] = all_results[code]
            elif short_code in all_results:
                results[code] = all_results[short_code]
            else:
                results[code] = {
                    'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0,
                    'pred_15d': 0, 'rank_score': 0, 'recommendation': '观望',
                    'exec_filter': 'no_data',
                }

        return results

    def predict_scores_from_preloaded(self, stock_codes: List[str], date: str,
                                       features_df: pd.DataFrame) -> Dict[str, Dict]:
        if features_df is None or len(features_df) == 0:
            return {code: {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0,
                           'pred_10d': 0, 'pred_15d': 0, 'rank_score': 0,
                           'recommendation': '观望', 'exec_filter': 'no_data'}
                    for code in stock_codes}

        if isinstance(date, str) and len(date) == 8 and date.isdigit():
            date = f"{date[:4]}-{date[4:6]}-{date[6:]}"

        for col in self.feature_names:
            if col not in features_df.columns:
                features_df[col] = 0.0

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

        predictions = {}
        for target in ['3d', '5d', '10d', '15d']:
            pred = self._ensemble_predict(feature_matrix, target)
            predictions[target] = pred if pred is not None else np.zeros(len(codes))

        w = self.target_weights
        combined_pred = np.zeros(len(codes), dtype=float)
        for target_key in ['3d', '5d', '10d', '15d']:
            combined_pred += w.get(target_key, 0.0) * predictions.get(target_key, np.zeros(len(codes)))

        scores = self._to_global_score(combined_pred)

        results = {}
        for i, code in enumerate(codes):
            if code in stock_codes:
                results[code] = {
                    'pred_3d': float(predictions['3d'][i]),
                    'pred_5d': float(predictions['5d'][i]),
                    'pred_10d': float(predictions['10d'][i]),
                    'pred_15d': float(predictions.get('15d', np.zeros(len(codes)))[i]),
                    'rank_score': float(combined_pred[i]),
                    'score': float(scores[i]),
                    'recommendation': self._get_recommendation(float(scores[i])),
                }

        for code in stock_codes:
            if code not in results:
                results[code] = {
                    'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0,
                    'pred_15d': 0, 'rank_score': 0, 'recommendation': '观望',
                    'exec_filter': 'no_data',
                }

        return results
