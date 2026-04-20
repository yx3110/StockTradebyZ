#!/usr/bin/env python3
"""
NG v1.0.3 Production Scorer — loads NG model, scores all stocks for a date.

v1.0.3 changes:
  - Dynamic feature list from model pkl (moneyflow + interaction support)
  - 68+ features (58+ stock + 10 market), extensible via model pkl
  - ICIR adaptive composite weights from model (not hardcoded)
  - Labels are industry excess returns
  - Backward compatible: auto-detects v1.0.0 / v1.0.1 / v1.0.3 models
"""

import glob
import json
import logging
import os
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
    NG107_VERSION,
    NG107_MARKET_FEATURES,
    NG107_ALL_FEATURES,
    CONDITIONAL_IX_FEATURE_NAMES,
    EXTENDED_MARKET_FEATURE_NAMES,
    MONEYFLOW_FEATURE_NAMES,
    INTERACTION_FEATURE_NAMES,
)
from .ng_schema import get_table_name, version_ge

# Ensemble members whose raw outputs are ranks/probabilities rather than
# return-scale regressions. We rescale them into the regression heads' mean/std
# so composite weighting is on a common scale. Any new rank-style head must be
# added here — forgetting leaves it double-counted as a regression prediction.
RANK_HEAD_NAMES = frozenset({'lgb_rank', 'lgb_listnet', 'margin_rank', 'lgb_quintile'})
# Regression-shaped heads excluded from the rescale target (and from composite).
COMPOSITE_EXCLUDE_NAMES = frozenset({'lgb_q95'})

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
    """NG Production Scorer — self-contained scoring from ng_feature_cache."""

    def __init__(self, db_path: str = None, model_path: str = None, version: str = None):
        self.db_path = db_path or DB_PATH
        self.model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'ng'

        # Regime-conditional weight override (Phase B):
        # NG14X_REGIME_WEIGHTS_JSON=<path>.json with structure:
        #   {"weights": {"bull": {algo: w, ...}, "bear": {...}, "all": {...}}, ...}
        # When set, overrides per-target pkl weights with per-regime weights based
        # on today's AMV regime. Falls back to 'all' if regime unknown.
        self._regime_weights: Optional[Dict[str, Dict[str, float]]] = None
        self._regime_map: Optional[Dict[str, int]] = None  # trade_date → +1/-1
        self._active_regime_weights: Optional[Dict[str, float]] = None
        rw_path = os.environ.get('NG14X_REGIME_WEIGHTS_JSON')
        if rw_path and os.path.exists(rw_path):
            try:
                with open(rw_path, 'r') as _f:
                    self._regime_weights = json.load(_f).get('weights')
                print(f'NG14X regime weights loaded from {rw_path}: '
                      f'{list((self._regime_weights or {}).keys())}')
            except Exception as _e:
                logger.warning('regime weights load failed: %s', _e)

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
        self.downside_model = None
        self.lambda_risk = 0.5
        self.risk_filter_quantile = 0.20  # ng1.0.7: Pareto filter threshold
        self._ensemble_scorers = None  # NEW: multi-seed ensemble list

        if version and version_ge(version, 'ng1.0.4') and model_path is None:
            self._load_ensemble_models(version)
        else:
            self._load_model(model_path, version)

    def _load_model(self, model_path: str = None, version: str = None):
        if model_path:
            path = Path(model_path)
        else:
            glob_pat = f"{version.replace('.', '')}*.pkl" if version else 'ng*.pkl'
            ng_files = sorted(
                self.model_dir.glob(glob_pat),
                key=lambda f: f.stat().st_mtime
            )
            if not ng_files:
                logger.warning("No NG model found in %s matching %s", self.model_dir, glob_pat)
                print(f"NG scorer: No model found in {self.model_dir} matching {glob_pat}")
                return
            path = ng_files[-1]

        try:
            model_data = joblib.load(str(path))
        except Exception as e:
            logger.error("Failed to load NG model %s: %s", path, e)
            return

        # Warn if reproducibility metadata is missing (pre-2026-04-20 models)
        for _field in ('git_commit_hash', 'host', 'schema_version'):
            if _field not in model_data:
                logger.warning("pkl %s missing %s (pre-2026-04-20 model)", path.name, _field)

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

        # Target weights from model (ICIR adaptive in v1.0.3)
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
        # v1.0.2: downside model
        self.downside_model = model_data.get('downside_model')
        self.lambda_risk = model_data.get('lambda_risk', 0.5)
        # ng1.0.7: Pareto risk filter
        self.risk_filter_quantile = model_data.get('risk_filter_quantile', 0.20)
        n_targets = len(self.models)
        n_features = len(self.feature_names)
        scoring_mode = "continuous" if self.global_quantiles is not None else "cross-sectional"
        print(f"NG scorer loaded ({model_version}): {list(self.models.keys())} "
              f"[{scoring_mode} scoring, {n_features} features]")
        print(f"  file: {path.name}")
        print(f"  cache table: {self.cache_table}")
        print(f"  composite weights: {self.target_weights}")
        if self.downside_model is not None:
            print(f"  downside model: loaded (lambda_risk={self.lambda_risk})")

    def _load_ensemble_models(self, version: str):
        """Load all seed models for a given version and set up ensemble averaging."""
        ver_tag = version.replace('.', '')  # ng104

        seed_files = sorted(
            self.model_dir.glob(f'{ver_tag}_seed*_multi_target_*.pkl'),
            key=lambda f: f.stat().st_mtime
        )

        if not seed_files:
            # Fallback: look for single model without seed tag
            single_files = sorted(
                self.model_dir.glob(f'{ver_tag}*_multi_target_*.pkl'),
                key=lambda f: f.stat().st_mtime
            )
            if single_files:
                self._load_model(str(single_files[-1]))
                return
            # Final fallback: load latest ng model
            logger.warning(f"No {ver_tag} models found, falling back to latest")
            self._load_model(None)
            return

        # Load each seed model as an independent scorer
        self._ensemble_scorers = []
        for sf in seed_files:
            scorer = NGProductionScorer(db_path=self.db_path, model_path=str(sf))
            scorer._ensemble_scorers = None  # prevent recursion
            self._ensemble_scorers.append(scorer)

        # Copy metadata from first scorer
        first = self._ensemble_scorers[0]
        self.feature_names = first.feature_names
        self.stock_feature_cols = first.stock_feature_cols
        self.macro_feature_cols = first.macro_feature_cols
        self.cache_table = first.cache_table
        self.target_weights = first.target_weights
        self.winsorize_bounds = first.winsorize_bounds
        self.global_quantiles = first.global_quantiles
        self.models = first.models  # for single-stock fallback
        self.weights = first.weights

        print(f"  Multi-seed ensemble: {len(self._ensemble_scorers)} models loaded")

    # ------------------------------------------------------------------
    # Feature loading
    # ------------------------------------------------------------------

    def _load_features(self, stock_codes: List[str], date: str) -> Optional[pd.DataFrame]:
        if isinstance(date, str) and len(date) == 8 and date.isdigit():
            date = f"{date[:4]}-{date[4:6]}-{date[6:]}"

        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            # ng1.0.7: also load AMV columns from table — only if cache actually has them.
            # ng1.2.x reuses ng101/ng121 cache which omits AMV; the model may still list them in
            # macro_feature_cols (trained as NaN-filled), so skip the SELECT and fill with 0 downstream.
            extra_select = ""
            if any(c in self.macro_feature_cols for c in EXTENDED_MARKET_FEATURE_NAMES):
                cache_cols = {r[1] for r in conn.execute(f"PRAGMA table_info({self.cache_table})").fetchall()}
                if {'amv_var1', 'amv_macd', 'amv_regime_days'}.issubset(cache_cols):
                    extra_select = ", amv_var1, amv_macd, amv_regime_days"

            query = f"""
            SELECT code, features_json,
                   market_return_5d, market_return_20d, market_volatility_20d,
                   market_breadth, market_new_high_ratio, northbound_flow_5d,
                   market_volume_ratio, market_drawdown, vix_proxy,
                   market_momentum_diff{extra_select}
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

        # Ensure expected columns (handles v1.0.0 / v1.0.3 / v1.0.3+ data)
        # Fill missing moneyflow/interaction features with 0 for older cache entries
        for col in self.stock_feature_cols:
            if col not in df_stock.columns:
                df_stock[col] = 0.0

        result = pd.DataFrame()
        result['code'] = df_raw['code'].values

        for col in self.stock_feature_cols:
            result[col] = pd.to_numeric(df_stock[col], errors='coerce').fillna(0.0).values

        for col in self.macro_feature_cols:
            if col in df_raw.columns:
                result[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0.0).values
            elif col in df_stock.columns:
                # Extended market features stored in features_json (e.g., market_ret_60d)
                result[col] = pd.to_numeric(df_stock[col], errors='coerce').fillna(0.0).values
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

        rank_names = [n for n in preds if n in RANK_HEAD_NAMES]
        regression_names = [
            n for n in preds
            if n not in RANK_HEAD_NAMES and n not in COMPOSITE_EXCLUDE_NAMES
        ]
        if regression_names and rank_names:
            reg_means = [np.mean(preds[n]) for n in regression_names]
            reg_stds = [max(np.std(preds[n]), 1e-8) for n in regression_names]
            t_mean = np.mean(reg_means)
            # Floor on the target std too: if all regression heads are near-constant
            # in this batch, avg std → 0 collapses rank heads onto a single point.
            t_std = max(np.mean(reg_stds), 1e-8)
            for rn in rank_names:
                rp = preds[rn]
                rp_std = max(np.std(rp), 1e-8)
                preds[rn] = (rp - np.mean(rp)) / rp_std * t_std + t_mean

        # Regime-conditional weight override: if _active_regime_weights is set
        # (via _ensemble_predict_scores based on today's AMV regime), use those
        # instead of per-target pkl weights.
        target_w = self._active_regime_weights or self.weights.get(target, {})
        weighted_sum = np.zeros(X.shape[0])
        total_weight = 0.0

        for name, pred in preds.items():
            if name in COMPOSITE_EXCLUDE_NAMES:
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

    def _ensemble_predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """Average predictions from all seed models."""
        if isinstance(date, str) and len(date) == 8 and date.isdigit():
            date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        # Propagate regime-conditional weight override to each seed scorer.
        # Always propagate (including None) so children don't retain stale
        # weights from a previous call when the current date resolves to no
        # regime override (missing AMV row or sideways regime without 'all' key).
        self._apply_regime_weights(date)
        for sc in self._ensemble_scorers:
            sc._active_regime_weights = self._active_regime_weights

        # Load features once (all scorers share same cache table)
        features_df = self._load_features(stock_codes, date)

        all_results = []
        for scorer in self._ensemble_scorers:
            if features_df is not None and len(features_df) > 0:
                results = scorer.predict_scores_from_preloaded(
                    stock_codes, date, features_df)
            else:
                results = scorer.predict_scores(stock_codes, date)
            all_results.append(results)

        # Average numeric fields across seed models
        merged = {}
        for code in stock_codes:
            preds = [r.get(code, {}) for r in all_results
                     if code in r and r[code].get('exec_filter') != 'no_data']
            if not preds:
                merged[code] = {
                    'score': 50.0, 'pred_3d': 0, 'pred_5d': 0,
                    'pred_10d': 0, 'pred_15d': 0, 'rank_score': 0,
                    'recommendation': '观望', 'exec_filter': 'no_data',
                }
                continue

            avg = {}
            for key in ['pred_3d', 'pred_5d', 'pred_10d', 'pred_15d', 'rank_score', 'score']:
                vals = [float(p.get(key, 0) or 0) for p in preds]
                avg[key] = float(np.mean(vals)) if vals else 0.0
            avg['recommendation'] = self._get_recommendation(avg['score'])
            merged[code] = avg

        return merged

    def _apply_regime_weights(self, date: str) -> None:
        """Set self._active_regime_weights based on AMV regime for `date`."""
        self._active_regime_weights = None
        if not self._regime_weights:
            return
        if self._regime_map is None:
            try:
                import sqlite3 as _sq
                with _sq.connect(self.db_path, timeout=30) as _c:
                    rows = _c.execute(
                        'SELECT trade_date, amv_regime FROM market_amv'
                    ).fetchall()
                self._regime_map = dict(rows)
            except Exception as e:
                logger.warning('regime map load failed: %s', e)
                self._regime_map = {}
        reg = self._regime_map.get(date)
        key = 'bull' if reg == 1 else ('bear' if reg == -1 else 'all')
        self._active_regime_weights = self._regime_weights.get(key) \
            or self._regime_weights.get('all')

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        # Multi-seed ensemble mode
        if self._ensemble_scorers:
            return self._ensemble_predict_scores(stock_codes, date)

        if isinstance(date, str) and len(date) == 8 and date.isdigit():
            date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        self._apply_regime_weights(date)

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

        # v1.0.2: Risk discount (normalized to return scale)
        pred_downside = np.zeros(len(codes))
        if self.downside_model is not None:
            try:
                pred_downside = self.downside_model.predict(feature_matrix)
                pred_downside = np.clip(pred_downside, 0, None)
                # Normalize downside to same scale as combined_pred
                # Without this, downside (~0.024) overwhelms return (~0.00004)
                return_std = max(np.std(combined_pred), 1e-8)
                ds_std = max(np.std(pred_downside), 1e-8)
                pred_downside_scaled = pred_downside * (return_std / ds_std)
                combined_pred = combined_pred - self.lambda_risk * pred_downside_scaled
            except Exception as e:
                logger.warning("Downside prediction failed: %s", e)

        # ng1.0.7: Pareto risk filter — penalize stocks in worst N% by predicted maxdd
        risk_filtered = np.zeros(len(codes), dtype=bool)
        if self.risk_filter_quantile > 0 and np.any(pred_downside > 0):
            threshold = np.quantile(pred_downside, 1.0 - self.risk_filter_quantile)
            risk_filtered = pred_downside > threshold
            # Hard penalty: set filtered stocks to minimum score
            combined_pred[risk_filtered] = combined_pred.min() - 1.0
            n_filtered = np.sum(risk_filtered)
            if n_filtered > 0:
                logger.info(f"  Pareto filter: {n_filtered} stocks excluded "
                           f"(pred_maxdd > {threshold:.4f}, q={self.risk_filter_quantile})")

        scores = self._to_global_score(combined_pred)

        all_results = {}
        for i, code in enumerate(codes):
            all_results[code] = {
                'pred_3d': float(predictions['3d'][i]),
                'pred_5d': float(predictions['5d'][i]),
                'pred_10d': float(predictions['10d'][i]),
                'pred_15d': float(predictions.get('15d', np.zeros(len(codes)))[i]),
                'pred_downside_10d': float(pred_downside[i]),
                'rank_score': float(combined_pred[i]),
                'score': float(scores[i]),
                'recommendation': self._get_recommendation(float(scores[i])),
                'risk_filtered': bool(risk_filtered[i]),
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

        missing = {col: 0.0 for col in self.feature_names if col not in features_df.columns}
        if missing:
            features_df = features_df.assign(**missing)

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

        # v1.0.2: Risk discount (normalized to return scale)
        pred_downside = np.zeros(len(codes))
        if self.downside_model is not None:
            try:
                pred_downside = self.downside_model.predict(feature_matrix)
                pred_downside = np.clip(pred_downside, 0, None)
                # Normalize downside to same scale as combined_pred
                # Without this, downside (~0.024) overwhelms return (~0.00004)
                return_std = max(np.std(combined_pred), 1e-8)
                ds_std = max(np.std(pred_downside), 1e-8)
                pred_downside_scaled = pred_downside * (return_std / ds_std)
                combined_pred = combined_pred - self.lambda_risk * pred_downside_scaled
            except Exception as e:
                logger.warning("Downside prediction failed: %s", e)

        # ng1.0.7: Pareto risk filter (same logic as predict_scores)
        risk_filtered = np.zeros(len(codes), dtype=bool)
        if self.risk_filter_quantile > 0 and np.any(pred_downside > 0):
            threshold = np.quantile(pred_downside, 1.0 - self.risk_filter_quantile)
            risk_filtered = pred_downside > threshold
            combined_pred[risk_filtered] = combined_pred.min() - 1.0

        scores = self._to_global_score(combined_pred)

        # Build code→index mapping (cache codes have no suffix)
        stock_codes_short = {c.split('.')[0] if '.' in c else c for c in stock_codes}
        results = {}
        for i, code in enumerate(codes):
            if code in stock_codes or code in stock_codes_short:
                entry = {
                    'pred_3d': float(predictions['3d'][i]),
                    'pred_5d': float(predictions['5d'][i]),
                    'pred_10d': float(predictions['10d'][i]),
                    'pred_15d': float(predictions.get('15d', np.zeros(len(codes)))[i]),
                    'pred_downside_10d': float(pred_downside[i]),
                    'rank_score': float(combined_pred[i]),
                    'score': float(scores[i]),
                    'recommendation': self._get_recommendation(float(scores[i])),
                }
                results[code] = entry

        # Map results back to original stock_codes (handle .SH/.SZ/.BJ suffix)
        for code in stock_codes:
            if code not in results:
                short_code = code.split('.')[0] if '.' in code else code
                if short_code in results:
                    results[code] = results[short_code]
                else:
                    results[code] = {
                        'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0,
                        'pred_15d': 0, 'rank_score': 0, 'recommendation': '观望',
                        'exec_filter': 'no_data',
                    }

        return results


# ---------------------------------------------------------------------------
# NG 1.3.0 Dual-Head Scorer
# ---------------------------------------------------------------------------

from ml_models.ng.ng130_composite import DEFAULT_BETA, compute_composite


class NG130DualHeadScorer:
    """ng1.3.x dual-head scorer: 3 seeds × 2 heads, β composite.

    Loads 6 pkl files (ng130_seed{42,123,456}_{excess,downside}_multi_target_*.pkl),
    predicts each (seed, head) for the requested horizon, averages across seeds,
    then computes cross-sectional β composite per trade_date.
    """

    VERSION = 'ng1.3.0'

    def __init__(
        self,
        model_dir: Optional[str] = None,
        beta: Optional[float] = None,
        seeds: tuple = (42, 123, 456),
        industry_neutralize: bool = True,
    ):
        proj = Path(__file__).resolve().parent.parent.parent
        self.model_dir = model_dir or str(proj / 'ml_models' / 'trained_models' / 'ng')
        if beta is not None:
            self.beta = beta
        else:
            env_beta = os.environ.get('NG130_BETA')
            self.beta = float(env_beta) if env_beta is not None else DEFAULT_BETA
        self.seeds = seeds
        env_neut = os.environ.get('NG130_INDNEUT')
        if env_neut is not None:
            industry_neutralize = env_neut.lower() not in ('0', 'false', 'no')
        self.industry_neutralize = industry_neutralize
        self._db_path = str(proj / 'data_adapter' / 'stock_data.db')
        self._industry_map: Optional[Dict[str, str]] = None

        # Load 3 × 2 = 6 models
        self.models: Dict = {}
        for seed in seeds:
            for head in ('excess', 'downside'):
                pattern = os.path.join(
                    self.model_dir, f'ng130_seed{seed}_{head}_multi_target_*.pkl'
                )
                matches = sorted(glob.glob(pattern))
                if not matches:
                    raise FileNotFoundError(
                        f'No model for seed={seed} head={head} in {self.model_dir}'
                    )
                self.models[(seed, head)] = joblib.load(matches[-1])

        self.feature_names: List[str] = self.models[(seeds[0], 'excess')]['feature_names']
        print(
            f'NG130DualHeadScorer loaded: {len(self.models)} models, '
            f'{len(self.feature_names)} features, β={self.beta}, '
            f'industry_neutralize={self.industry_neutralize}'
        )

    def _load_industry_map(self) -> Dict[str, str]:
        """Lazy-load code → industry map from securities table."""
        if self._industry_map is not None:
            return self._industry_map
        import sqlite3
        with sqlite3.connect(self._db_path, timeout=30) as conn:
            rows = conn.execute(
                "SELECT code, COALESCE(industry, '__NA__') FROM securities"
            ).fetchall()
        self._industry_map = dict(rows)
        return self._industry_map

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_head(
        self, model_dict: dict, X: np.ndarray, horizon: str = '10d'
    ) -> np.ndarray:
        """Weighted ensemble prediction for one (seed, head) and one horizon."""
        target_cfg = model_dict['models'][horizon]
        boosters = target_cfg['models']
        weights = target_cfg.get('weights', {})

        preds: Dict[str, np.ndarray] = {}
        for algo, booster in boosters.items():
            try:
                if algo == 'xgb':
                    import xgboost as xgb
                    dmat = xgb.DMatrix(X, feature_names=self.feature_names)
                    preds[algo] = booster.predict(dmat)
                else:
                    preds[algo] = booster.predict(X)
            except Exception as e:
                logger.warning('NG130 %s predict failed: %s', algo, e)

        if not preds:
            raise RuntimeError(f'All boosters failed for horizon={horizon}')

        wsum = sum(weights.get(a, 0.0) for a in preds)
        if wsum <= 0:
            return np.mean(list(preds.values()), axis=0)
        return sum(weights.get(a, 0.0) * preds[a] for a in preds) / wsum

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, features_df: pd.DataFrame, horizon: str = '10d') -> pd.DataFrame:
        """Score rows in features_df.

        Args:
            features_df: Must contain all self.feature_names columns (NaN ok,
                filled with 0) plus a 'trade_date' column for cross-sectional
                composite ranking.
            horizon: one of '3d', '5d', '10d', '15d'.

        Returns:
            DataFrame with columns:
              trade_date, pred_excess, pred_downside, composite, composite_rank
        """
        X = features_df.reindex(columns=self.feature_names, fill_value=0.0).fillna(0.0).values

        pred_excess_list, pred_downside_list = [], []
        for seed in self.seeds:
            pred_excess_list.append(self._predict_head(self.models[(seed, 'excess')], X, horizon))
            pred_downside_list.append(self._predict_head(self.models[(seed, 'downside')], X, horizon))
        pred_excess_avg = np.mean(pred_excess_list, axis=0)
        pred_downside_avg = np.mean(pred_downside_list, axis=0)

        if 'trade_date' in features_df.columns:
            out = features_df[['trade_date']].copy()
        else:
            out = pd.DataFrame({'trade_date': 'unknown'}, index=features_df.index)
        out['pred_excess'] = pred_excess_avg
        out['pred_downside'] = pred_downside_avg

        # Industry-neutralize predictions: subtract (trade_date, industry) mean.
        # Hard-coded labels are already industry-excess for excess head, but predictions
        # drift back toward industry tilts during training. Subtract industry mean so
        # top-k picks aren't dominated by a few hot sectors (fixes L5 alpha collapse).
        if self.industry_neutralize and 'code' in features_df.columns:
            ind_map = self._load_industry_map()
            out['_industry'] = [ind_map.get(c, '__NA__') for c in features_df['code'].values]
            for col in ('pred_excess', 'pred_downside'):
                ind_mean = out.groupby(['trade_date', '_industry'])[col].transform('mean')
                out[col] = out[col] - ind_mean
            out = out.drop(columns=['_industry'])

        beta = self.beta

        # Per-date cross-sectional composite. Assign via aligned Series to preserve dtype.
        composite_series = pd.Series(index=out.index, dtype=float)
        for date, g in out.groupby('trade_date', sort=False):
            composite_series.loc[g.index] = compute_composite(
                g['pred_excess'], g['pred_downside'], beta,
            ).values
        out['composite'] = composite_series
        out['composite_rank'] = out.groupby('trade_date')['composite'].rank(
            ascending=False, method='first'
        )
        return out

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """Compatible with NGProductionScorer API: return {code: {score, pred_10d, ...}}.

        tomorrow_stock_selector.py 调用此 API 对批量 stock_codes 评分.
        内部: 从 ng130_feature_cache 读取 (code, date) 行 → predict() → 格式化返回.
        """
        import sqlite3
        import json

        if isinstance(date, str) and len(date) == 8 and date.isdigit():
            date = f'{date[:4]}-{date[4:6]}-{date[6:]}'

        proj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        db_path = os.path.join(proj, 'data_adapter', 'stock_data.db')

        placeholders = ','.join('?' * len(stock_codes))
        sql = f"""
            SELECT code, trade_date, features_json,
                   market_return_5d, market_return_20d, market_volatility_20d,
                   market_breadth, market_new_high_ratio, northbound_flow_5d,
                   market_volume_ratio, market_drawdown, vix_proxy,
                   market_momentum_diff
            FROM ng130_feature_cache
            WHERE trade_date = ? AND code IN ({placeholders})
        """
        with sqlite3.connect(db_path, timeout=30) as conn:
            df_raw = pd.read_sql(sql, conn, params=[date] + list(stock_codes))

        results: Dict[str, Dict] = {}
        default_result = {
            'score': 50.0, 'pred_excess': 0.0, 'pred_downside': 0.0,
            'composite': 0.5,
            'pred_3d': 0.0, 'pred_5d': 0.0, 'pred_10d': 0.0, 'pred_15d': 0.0,
            'rank_score': 0.5,
            'recommendation': '观望', 'exec_filter': 'no_data',
        }
        if df_raw.empty:
            return {code: dict(default_result) for code in stock_codes}

        feats = pd.DataFrame([json.loads(r) for r in df_raw['features_json']])
        market_cols = [
            'market_return_5d', 'market_return_20d', 'market_volatility_20d',
            'market_breadth', 'market_new_high_ratio', 'northbound_flow_5d',
            'market_volume_ratio', 'market_drawdown', 'vix_proxy',
            'market_momentum_diff',
        ]
        for c in market_cols:
            feats[c] = df_raw[c].values
        feats['trade_date'] = df_raw['trade_date'].values
        feats['code'] = df_raw['code'].values

        # Run predict() for each horizon so downstream consumers get the full
        # 3d/5d/10d/15d vector (batch_generate_v395_reports expects all four).
        per_horizon: Dict[str, pd.DataFrame] = {}
        for h in ('3d', '5d', '10d', '15d'):
            per_horizon[h] = self.predict(feats, horizon=h)
        codes_arr = feats['code'].values
        for df_h in per_horizon.values():
            df_h['code'] = codes_arr

        scored = per_horizon['10d']
        by_code = {c: i for i, c in enumerate(codes_arr)}

        for code in stock_codes:
            idx = by_code.get(code)
            if idx is None:
                results[code] = dict(default_result)
                continue
            r10 = scored.iloc[idx]
            composite = float(r10['composite'])
            results[code] = {
                'score': float(100 * composite),
                'pred_excess': float(r10['pred_excess']),
                'pred_downside': float(r10['pred_downside']),
                'composite': composite,
                'pred_3d': float(per_horizon['3d'].iloc[idx]['pred_excess']),
                'pred_5d': float(per_horizon['5d'].iloc[idx]['pred_excess']),
                'pred_10d': float(r10['pred_excess']),
                'pred_15d': float(per_horizon['15d'].iloc[idx]['pred_excess']),
                'rank_score': composite,
                'recommendation': '买入' if composite > 0.65 else ('关注' if composite > 0.5 else '观望'),
                'exec_filter': 'ok',
            }
        return results


# ---------------------------------------------------------------------------
# Scorer registry
# ---------------------------------------------------------------------------

SCORER_REGISTRY: Dict[str, tuple] = {
    'ng1.3.0': (NG130DualHeadScorer, {}),
}
