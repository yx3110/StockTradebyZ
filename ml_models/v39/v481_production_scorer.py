#!/usr/bin/env python3
"""
V4.8.1 production scorer -- 60 features (V4.7.5 base - 5 pruned + 15 new) + V4.7.6 post-processing

Architecture:
  Model: V4.8.1 trained model (60 features: 45 from V4.7.5 + 15 new factors)
  Scorer: Inherits V4.7.6 post-processing (consistency bonus + vol discount)

The 15 new factors are computed at inference time from daily_quotes and
technical_indicators DB tables, matching the training-time computation exactly.

New factors:
  1. atr_percentile       - ATR rank percentile (cross-sectional)
  2. vol_concentration    - Volume HHI (Herfindahl concentration index)
  3. intraday_ret_20d     - Cumulative intraday returns (open-to-close)
  4. industry_mom_rank    - Industry-relative momentum rank
  5. vwap_dev_20d         - VWAP deviation
  6. max_ret_20d          - Max daily return in 20d window
  7. gk_vol_20d           - Garman-Klass volatility
  8. abnormal_turnover    - Abnormal turnover ratio
  9. overnight_ret_20d    - Cumulative overnight returns (close-to-open)
  10. turnover_vol_20d    - Turnover volatility
  11. cci_14              - CCI (from technical_indicators)
  12. squeeze_mom_calc    - Squeeze momentum
  13. vol_price_div       - Volume-price divergence
  14. price_acceleration  - Price acceleration (2nd derivative)
  15. price_pos_volatility - Price position within volatility band

Fallback chain: v481 model -> v475 model
"""

import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from .v476_production_scorer import V476ProductionScorer

import logging
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# The 15 new factor column names
V481_NEW_FACTORS = [
    'atr_percentile',
    'vol_concentration',
    'intraday_ret_20d',
    'industry_mom_rank',
    'vwap_dev_20d',
    'max_ret_20d',
    'gk_vol_20d',
    'abnormal_turnover',
    'overnight_ret_20d',
    'turnover_vol_20d',
    'cci_14',
    'squeeze_mom_calc',
    'vol_price_div',
    'price_acceleration',
    'price_pos_volatility',
]


class V481ProductionScorer(V476ProductionScorer):
    """V4.8.1 scorer -- 60 features (V4.7.5 base - 5 pruned + 15 new) + V4.7.6 post-processing"""

    def __init__(self, model_type: str = 'small_data'):
        self._v481_model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v481'
        self._v481_factor_cache = {}  # date -> DataFrame of new factors
        self._industry_cache = {}     # code -> industry
        super().__init__(model_type=model_type)

    def _load_models(self):
        """Try v481 model first, fallback to v475"""
        v481_files = list(self._v481_model_dir.glob('v481_*.pkl'))
        if v481_files:
            self.model_dir = self._v481_model_dir
            latest = max(v481_files, key=lambda f: f.stat().st_mtime)
            self._load_model_from_file(latest, label='V4.8.1')
            return

        # No v481 model yet, fallback to V4.7.5 model via parent
        super()._load_models()

    def _load_model_from_file(self, model_path, label='V4.8.1'):
        """Load model from specific file (reuses V4.7.7 parsing logic)"""
        import joblib, pickle

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

        # Recommendation thresholds
        rec_path = self.model_dir / 'recommendation_thresholds.json'
        if rec_path.exists():
            import json as _json
            with open(rec_path, 'r') as f:
                self.recommendation_thresholds = _json.load(f)
        else:
            self.recommendation_thresholds = model_data.get('recommendation_thresholds')

        # ICIR weights: clip to [0.08, 0.50]
        self.weights = self._clip_icir_weights(self.weights)

        gq_status = "continuous" if self.global_quantiles is not None else "cross-sectional"
        print(f"{label} loaded: {list(self.models.keys())} [{gq_status} scoring, {len(self.feature_cols)} features]")
        print(f"  file: {model_path.name}")

    # ========== V4.8.1 core: compute 15 new factors from DB ==========

    def _compute_v481_new_factors(self, features_df: pd.DataFrame, date: str) -> pd.DataFrame:
        """Compute 15 new factors for V4.8.1 at inference time.

        Priority: v481_factor_cache DB table (pre-computed) > real-time calculation.
        DB cache is ~1000x faster (single SELECT vs 25-day rolling computation).

        Args:
            features_df: DataFrame with 'code' column
            date: trade date string (YYYY-MM-DD)

        Returns:
            features_df with 15 new factor columns added
        """
        if date in self._v481_factor_cache:
            df_factors = self._v481_factor_cache[date]
        else:
            # Try DB cache first (v481_factor_cache table)
            df_factors = self._load_v481_factors_from_db(date)
            if df_factors is None or len(df_factors) == 0:
                # Fallback: real-time calculation
                df_factors = self._build_v481_factors(date)
            self._v481_factor_cache[date] = df_factors

        if df_factors is not None and len(df_factors) > 0:
            features_df = features_df.merge(df_factors, on='code', how='left')

        # Fill missing factor values with 0
        for col in V481_NEW_FACTORS:
            if col in features_df.columns:
                features_df[col] = features_df[col].fillna(0.0)
            else:
                features_df[col] = 0.0

        return features_df

    def _load_v481_factors_from_db(self, date: str) -> Optional[pd.DataFrame]:
        """Load pre-computed V4.8.1 factors from v481_factor_cache DB table.

        ~1000x faster than real-time calculation. Table is pre-filled by
        /tmp/backfill_v481_factors.py or updated during daily data update.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"""
            SELECT code, {', '.join(V481_NEW_FACTORS)}
            FROM v481_factor_cache
            WHERE trade_date = ?
            """
            df = pd.read_sql_query(query, conn, params=[date])
            conn.close()
            if len(df) > 0:
                logger.debug(f"V4.8.1 factors loaded from DB cache: {len(df)} stocks for {date}")
                return df
            return None
        except Exception as e:
            logger.debug(f"V4.8.1 DB cache miss for {date}: {e}")
            return None

    def _build_v481_factors(self, date: str) -> Optional[pd.DataFrame]:
        """Build the 15 new factors for all A-stocks on a given date."""
        try:
            conn = sqlite3.connect(self.db_path)

            # Query 1: daily_quotes for last 25 trading days (OHLCV + turnover)
            query_ohlcv = """
            SELECT s.code, q.trade_date, q.open, q.high, q.low, q.close,
                   q.volume, q.price_change_pct
            FROM daily_quotes q
            JOIN securities s ON q.security_id = s.id
            WHERE s.type = 'A股'
              AND q.trade_date <= ?
              AND q.trade_date >= date(?, '-40 days')
            ORDER BY s.code, q.trade_date
            """
            df_ohlcv = pd.read_sql_query(query_ohlcv, conn, params=[date, date])

            # Query 2: technical_indicators for cci_14, atr_14
            query_tech = """
            SELECT s.code, ti.cci_14, ti.atr_14
            FROM technical_indicators ti
            JOIN securities s ON ti.security_id = s.id
            WHERE s.type = 'A股' AND ti.trade_date = ?
            """
            df_tech = pd.read_sql_query(query_tech, conn, params=[date])

            # Query 3: turnover_rate from daily_basic
            query_turnover = """
            SELECT s.code, db.turnover_rate, db.trade_date
            FROM daily_basic db
            JOIN securities s ON db.security_id = s.id
            WHERE s.type = 'A股'
              AND db.trade_date <= ?
              AND db.trade_date >= date(?, '-40 days')
            ORDER BY s.code, db.trade_date
            """
            df_turnover = pd.read_sql_query(query_turnover, conn, params=[date, date])

            # Query 4: industry info (cached)
            if not self._industry_cache:
                query_ind = """
                SELECT code, industry FROM securities WHERE type = 'A股' AND industry IS NOT NULL
                """
                df_ind = pd.read_sql_query(query_ind, conn)
                self._industry_cache = dict(zip(df_ind['code'], df_ind['industry']))

            conn.close()
        except Exception as e:
            logger.warning(f"V4.8.1 factor query failed: {e}")
            return None

        if len(df_ohlcv) == 0:
            return None

        # Build per-stock turnover lookup
        turnover_by_code = {}
        if len(df_turnover) > 0:
            for code, grp in df_turnover.groupby('code'):
                turnover_by_code[code] = grp.sort_values('trade_date')['turnover_rate'].values.astype(float)

        # Build cci/atr lookup
        tech_lookup = {}
        if len(df_tech) > 0:
            for _, row in df_tech.iterrows():
                tech_lookup[row['code']] = {
                    'cci_14': float(row['cci_14']) if pd.notna(row['cci_14']) else 0.0,
                    'atr_14': float(row['atr_14']) if pd.notna(row['atr_14']) else 0.0,
                }

        # Compute per-stock factors
        results_list = []
        # Collect all ATR values first for cross-sectional percentile
        all_atrs = {}

        for code, grp in df_ohlcv.groupby('code'):
            grp = grp.sort_values('trade_date')
            if len(grp) < 5:
                continue

            o = grp['open'].values.astype(float)
            h = grp['high'].values.astype(float)
            lo = grp['low'].values.astype(float)
            c = grp['close'].values.astype(float)
            v = grp['volume'].values.astype(float)
            pct = pd.to_numeric(grp['price_change_pct'], errors='coerce').fillna(0).values.astype(float)

            # Use last 20 days (or whatever is available, min_periods=5)
            n = min(20, len(c))
            c20 = c[-n:]
            o20 = o[-n:]
            h20 = h[-n:]
            lo20 = lo[-n:]
            v20 = v[-n:]
            pct20 = pct[-n:]

            row = {'code': code}

            # --- Factor computations ---

            # 1. atr_percentile: store ATR value, compute percentile later
            tech_info = tech_lookup.get(code, {})
            atr_val = tech_info.get('atr_14', 0.0)
            all_atrs[code] = atr_val

            # 2. vol_concentration: Volume HHI (Herfindahl)
            total_vol = np.sum(v20)
            if total_vol > 0 and len(v20) >= 5:
                shares = v20 / total_vol
                row['vol_concentration'] = float(np.sum(shares ** 2))
            else:
                row['vol_concentration'] = 0.0

            # 3. intraday_ret_20d: cumulative intraday returns (open-to-close)
            safe_open = np.where(o20 > 0, o20, 1e-8)
            intraday_rets = (c20 - o20) / safe_open
            row['intraday_ret_20d'] = float(np.sum(intraday_rets))

            # 4. industry_mom_rank: computed cross-sectionally later
            # Store 20d momentum for now
            if len(c) >= 21:
                row['_mom_20d'] = float(c[-1] / c[-21] - 1)
            elif len(c) >= 2:
                row['_mom_20d'] = float(c[-1] / c[0] - 1)
            else:
                row['_mom_20d'] = 0.0

            # 5. vwap_dev_20d: VWAP deviation
            # VWAP = sum(close * volume) / sum(volume) over 20d
            vwap_num = np.sum(c20 * v20)
            vwap_den = np.sum(v20)
            if vwap_den > 0 and c20[-1] > 0:
                vwap = vwap_num / vwap_den
                row['vwap_dev_20d'] = float((c20[-1] - vwap) / vwap)
            else:
                row['vwap_dev_20d'] = 0.0

            # 6. max_ret_20d: max daily return in 20d window
            row['max_ret_20d'] = float(np.max(pct20)) if len(pct20) > 0 else 0.0

            # 7. gk_vol_20d: Garman-Klass volatility
            if len(h20) >= 5:
                safe_lo = np.where(lo20 > 0, lo20, 1e-8)
                safe_c_prev = np.where(c20 > 0, c20, 1e-8)
                # GK = 0.5 * ln(H/L)^2 - (2ln2-1) * ln(C/O)^2
                log_hl = np.log(h20 / safe_lo)
                safe_o_gk = np.where(o20 > 0, o20, 1e-8)
                log_co = np.log(c20 / safe_o_gk)
                gk_daily = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
                row['gk_vol_20d'] = float(np.sqrt(np.mean(gk_daily)))
            else:
                row['gk_vol_20d'] = 0.0

            # 8. abnormal_turnover: current turnover / 20d avg turnover
            turnovers = turnover_by_code.get(code)
            if turnovers is not None and len(turnovers) >= 5:
                t_n = min(20, len(turnovers))
                t20 = turnovers[-t_n:]
                avg_turnover = np.mean(t20)
                if avg_turnover > 0:
                    row['abnormal_turnover'] = float(t20[-1] / avg_turnover)
                else:
                    row['abnormal_turnover'] = 0.0
                # 10. turnover_vol_20d: volatility of turnover
                row['turnover_vol_20d'] = float(np.std(t20)) if len(t20) >= 5 else 0.0
            else:
                row['abnormal_turnover'] = 0.0
                row['turnover_vol_20d'] = 0.0

            # 9. overnight_ret_20d: cumulative overnight returns (close[t-1] to open[t])
            if len(c20) >= 2 and len(o20) >= 2:
                prev_close = c20[:-1]
                next_open = o20[1:]
                safe_prev = np.where(prev_close > 0, prev_close, 1e-8)
                overnight_rets = (next_open - prev_close) / safe_prev
                row['overnight_ret_20d'] = float(np.sum(overnight_rets))
            else:
                row['overnight_ret_20d'] = 0.0

            # 11. cci_14: from technical_indicators
            row['cci_14'] = tech_info.get('cci_14', 0.0)

            # 12. squeeze_mom_calc: Squeeze momentum
            # squeeze_mom = (close - mean(close, 20)) / std(close, 20)
            if len(c20) >= 5:
                mean_c = np.mean(c20)
                std_c = np.std(c20)
                if std_c > 1e-8:
                    row['squeeze_mom_calc'] = float((c20[-1] - mean_c) / std_c)
                else:
                    row['squeeze_mom_calc'] = 0.0
            else:
                row['squeeze_mom_calc'] = 0.0

            # 13. vol_price_div: Volume-price divergence
            # Correlation between price changes and volume changes over 20d
            if len(pct20) >= 5 and len(v20) >= 5:
                v_pct = np.diff(v20) / np.where(v20[:-1] > 0, v20[:-1], 1e-8)
                p_pct = pct20[1:]  # align with volume changes
                min_len = min(len(v_pct), len(p_pct))
                if min_len >= 5:
                    corr = np.corrcoef(p_pct[:min_len], v_pct[:min_len])[0, 1]
                    row['vol_price_div'] = float(corr) if not np.isnan(corr) else 0.0
                else:
                    row['vol_price_div'] = 0.0
            else:
                row['vol_price_div'] = 0.0

            # 14. price_acceleration: 2nd derivative of price
            # acceleration = ret[-5:].mean() - ret[-10:-5].mean()
            if len(pct) >= 10:
                recent_5 = np.mean(pct[-5:])
                prev_5 = np.mean(pct[-10:-5])
                row['price_acceleration'] = float(recent_5 - prev_5)
            elif len(pct) >= 5:
                row['price_acceleration'] = float(np.mean(pct[-5:]))
            else:
                row['price_acceleration'] = 0.0

            # 15. price_pos_volatility: price position within volatility band
            # (close - low_20) / (high_20 - low_20), normalized by volatility
            if len(c20) >= 5:
                high_20 = np.max(h20)
                low_20 = np.min(lo20)
                band = high_20 - low_20
                if band > 1e-8:
                    row['price_pos_volatility'] = float((c20[-1] - low_20) / band)
                else:
                    row['price_pos_volatility'] = 0.5
            else:
                row['price_pos_volatility'] = 0.5

            results_list.append(row)

        if not results_list:
            return None

        df_factors = pd.DataFrame(results_list)

        # ---- Cross-sectional computations ----

        # 1. atr_percentile: rank percentile of ATR across all stocks
        if all_atrs:
            from scipy.stats import rankdata
            factor_codes = df_factors['code'].tolist()
            atr_values = np.array([all_atrs.get(c, 0.0) for c in factor_codes])
            if len(atr_values) > 1 and np.any(atr_values != 0):
                ranks = rankdata(atr_values, method='average')
                df_factors['atr_percentile'] = (ranks - 1) / max(len(ranks) - 1, 1)
            else:
                df_factors['atr_percentile'] = 0.5
        else:
            df_factors['atr_percentile'] = 0.5

        # 4. industry_mom_rank: rank momentum within industry
        if '_mom_20d' in df_factors.columns and self._industry_cache:
            df_factors['_industry'] = df_factors['code'].map(self._industry_cache)
            ind_ranks = []
            for _, grp in df_factors.groupby('_industry'):
                if len(grp) < 2:
                    ind_ranks.extend([(idx, 0.5) for idx in grp.index])
                    continue
                from scipy.stats import rankdata as _rd
                mom_vals = grp['_mom_20d'].values
                ranks = _rd(mom_vals, method='average')
                norm_ranks = (ranks - 1) / max(len(ranks) - 1, 1)
                ind_ranks.extend(zip(grp.index, norm_ranks))

            for idx, rank_val in ind_ranks:
                df_factors.at[idx, 'industry_mom_rank'] = rank_val

            # Clean up temp columns
            df_factors.drop(columns=['_mom_20d', '_industry'], inplace=True, errors='ignore')
        else:
            if '_mom_20d' in df_factors.columns:
                df_factors.drop(columns=['_mom_20d'], inplace=True, errors='ignore')
            df_factors['industry_mom_rank'] = 0.5

        # Keep only code + factor columns
        keep_cols = ['code'] + [c for c in V481_NEW_FACTORS if c in df_factors.columns]
        df_factors = df_factors[keep_cols]

        return df_factors

    # ========== Override predict_scores to inject new factors ==========

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """V4.8.1 scoring pipeline.

        Follows V4.7.3's pattern of adding extra features in predict_scores:
        1. Base V4.3 features (robust z-score + daily_basic + tech)
        2. V4.7.1 features (roe + daily_basic_extra + microstructure)
        3. V4.8.1 NEW: 15 new factors from DB
        4. Model prediction -> scoring -> post-processing (V4.7.6)
        """
        # Date format normalization
        if isinstance(date, str) and len(date) == 8 and date.isdigit():
            date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        results = {}

        # Step 1: Base features (V4.3)
        features_df = self._get_features(stock_codes, date, load_full_cross_section=True)
        if features_df is not None and len(features_df) > 0:
            features_df = self._robust_zscore_normalize_features(features_df)
            features_df = self._load_daily_basic_features(features_df, date)
            features_df = self._load_technical_features(features_df, date)

            # Step 1.5: V4.7.1 features
            features_df = self._load_financial_features(features_df, date)
            features_df = self._load_daily_basic_extra(features_df, date)
            features_df = self._compute_microstructure_features(features_df, date)

            # Step 1.8: V4.8.1 new factors
            features_df = self._compute_v481_new_factors(features_df, date)

            features_df = features_df[features_df['code'].isin(stock_codes)].copy()

        if features_df is None or len(features_df) == 0:
            for code in stock_codes:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0,
                                 'exec_filter': 'no_data'}
            return results

        # Prepare feature matrix
        exclude_cols = {'code', 'trade_date'}
        if self.feature_cols:
            missing = [c for c in self.feature_cols if c not in features_df.columns]
            if missing:
                if len(missing) > len(self.feature_cols) * 0.3:
                    logger.warning(f"V4.8.1: {len(missing)}/{len(self.feature_cols)} features missing: {missing[:5]}...")
                for col in missing:
                    features_df[col] = 0
            available_cols = self.feature_cols
        else:
            available_cols = [c for c in features_df.columns if c not in exclude_cols]

        X = features_df[available_cols].fillna(0).values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        codes = features_df['code'].tolist()
        self._last_pred_codes = codes  # Save for consensus/Q95 analysis
        self._last_X = X  # Save feature matrix for Q95 model

        # Step 2: Model predictions
        model_predictions_success = False
        predictions = {
            '3d': np.zeros(len(X)), '5d': np.zeros(len(X)),
            '10d': np.zeros(len(X)), '15d': np.zeros(len(X))
        }
        # Store per-model predictions for consensus/confidence analysis
        self._per_model_preds = {}

        for target in ['3d', '5d', '10d', '15d']:
            if target not in self.models or not self.models[target]:
                continue
            target_pred = np.zeros(len(X))
            total_weight = 0

            # Collect all sub-model predictions
            preds = {}
            for name, model in self.models[target].items():
                try:
                    if name == 'xgb':
                        import xgboost as xgb_lib
                        preds[name] = model.predict(xgb_lib.DMatrix(X))
                    else:
                        preds[name] = model.predict(X)
                except Exception:
                    continue

            # Rescale rank models to regression scale (exclude quantile models from stats)
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

            # Save per-model predictions before averaging
            self._per_model_preds[target] = dict(preds)

            # Exclude quantile models from composite (they serve Stage 2 only)
            COMPOSITE_EXCLUDE = ('lgb_q95',)

            target_w = self.weights.get(f'label_{target}', {})
            success_count = len(preds)
            for name, pred in preds.items():
                if name in COMPOSITE_EXCLUDE:
                    continue
                weight = target_w.get(name, 0.2)
                target_pred += weight * pred
                total_weight += weight

            if total_weight > 0:
                target_pred /= total_weight
                predictions[target] = target_pred
                if success_count > 0:
                    model_predictions_success = True

        # Step 3: ICIR weighted fusion (fixed 10d+15d weights)
        regime_weights = self._get_regime_target_weights(date)
        if model_predictions_success:
            combined_pred = (
                regime_weights.get('label_3d', 0.00) * predictions['3d'] +
                regime_weights.get('label_5d', 0.00) * predictions['5d'] +
                regime_weights.get('label_10d', 0.60) * predictions['10d'] +
                regime_weights.get('label_15d', 0.40) * predictions['15d']
            )
        else:
            combined_pred = self._calculate_fallback_scores(features_df, available_cols)
            predictions = self._estimate_predictions_from_features(features_df, available_cols)

        # Step 4: Continuous global percentile scoring
        scores = self._to_global_score(combined_pred)

        for i, code in enumerate(codes):
            results[code] = {
                'score': float(scores[i]),
                'pred_3d': float(predictions['3d'][i]) if i < len(predictions['3d']) else 0,
                'pred_5d': float(predictions['5d'][i]) if i < len(predictions['5d']) else 0,
                'pred_10d': float(predictions['10d'][i]) if i < len(predictions['10d']) else 0,
                'pred_15d': float(predictions.get('15d', np.zeros(1))[min(i, len(predictions.get('15d', [0]))-1)]) if '15d' in predictions else 0,
            }

        # Step 5: Enhanced executability filters
        results = self._apply_enhanced_executability_filters(results, date)

        # Compute rank_score (V4.7.5 composite: 0.6*10d + 0.4*15d)
        for code, data in results.items():
            data['rank_score'] = 0.60 * data.get('pred_10d', 0) + 0.40 * data.get('pred_15d', 0)

        # Fill missing codes
        for code in stock_codes:
            if code not in results:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0,
                                 'exec_filter': 'no_data'}

        # Module F: regime info (does not affect scores)
        regime_info = self._get_regime_info(date)
        for code in results:
            results[code]['regime_info'] = regime_info

        # V4.7.6 post-processing: consistency bonus + vol discount
        if len(results) >= 2:
            results = self._apply_consistency_bonus(results)
            results = self._apply_vol_discount(results, date)

        return results

    def predict_scores_from_preloaded(self, stock_codes: List[str], date: str,
                                       features_df: Optional[pd.DataFrame]) -> Dict[str, Dict]:
        """V4.8.1 batch scoring with preloaded features.

        Same pipeline as predict_scores but starting from preloaded features_df.
        """
        results = {}

        if features_df is None or len(features_df) == 0:
            for code in stock_codes:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0,
                                 'exec_filter': 'no_data'}
            return results

        # Date format normalization
        if isinstance(date, str) and len(date) == 8 and date.isdigit():
            date = f"{date[:4]}-{date[4:6]}-{date[6:]}"

        # Feature enrichment
        features_df = self._robust_zscore_normalize_features(features_df.copy())
        features_df = self._load_daily_basic_features(features_df, date)
        features_df = self._load_technical_features(features_df, date)

        # V4.7.1 features
        features_df = self._load_financial_features(features_df, date)
        features_df = self._load_daily_basic_extra(features_df, date)
        features_df = self._compute_microstructure_features(features_df, date)

        # V4.8.1 new factors
        features_df = self._compute_v481_new_factors(features_df, date)

        mask = features_df['code'].isin(stock_codes)
        filtered_df = features_df[mask].copy()

        if len(filtered_df) == 0:
            for code in stock_codes:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0,
                                 'exec_filter': 'no_data'}
            return results

        # Prepare feature matrix
        exclude_cols = {'code', 'trade_date'}
        if self.feature_cols:
            for col in self.feature_cols:
                if col not in filtered_df.columns:
                    filtered_df[col] = 0
            available_cols = self.feature_cols
        else:
            available_cols = [c for c in filtered_df.columns if c not in exclude_cols]

        X = filtered_df[available_cols].fillna(0).values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        codes = filtered_df['code'].tolist()

        model_predictions_success = False
        predictions = {
            '3d': np.zeros(len(X)), '5d': np.zeros(len(X)),
            '10d': np.zeros(len(X)), '15d': np.zeros(len(X))
        }

        for target in ['3d', '5d', '10d', '15d']:
            if target not in self.models or not self.models[target]:
                continue
            target_pred = np.zeros(len(X))
            total_weight = 0

            preds = {}
            for name, model in self.models[target].items():
                try:
                    if name == 'xgb':
                        import xgboost as xgb_lib
                        preds[name] = model.predict(xgb_lib.DMatrix(X))
                    else:
                        preds[name] = model.predict(X)
                except Exception:
                    continue

            regression_names = [n for n in preds if n not in ('lgb_rank', 'lgb_listnet')]
            rank_names = [n for n in preds if n in ('lgb_rank', 'lgb_listnet')]
            if regression_names and rank_names:
                reg_means = [np.mean(preds[n]) for n in regression_names]
                reg_stds = [max(np.std(preds[n]), 1e-8) for n in regression_names]
                t_mean, t_std = np.mean(reg_means), np.mean(reg_stds)
                for rn in rank_names:
                    rp = preds[rn]
                    rp_std = max(np.std(rp), 1e-8)
                    preds[rn] = (rp - np.mean(rp)) / rp_std * t_std + t_mean

            target_w = self.weights.get(f'label_{target}', {})
            success_count = len(preds)
            for name, pred in preds.items():
                weight = target_w.get(name, 0.2)
                target_pred += weight * pred
                total_weight += weight

            if total_weight > 0:
                target_pred /= total_weight
                predictions[target] = target_pred
                if success_count > 0:
                    model_predictions_success = True

        # ICIR weighted fusion
        regime_weights = self._get_regime_target_weights(date)
        if model_predictions_success:
            combined_pred = (
                regime_weights.get('label_3d', 0.00) * predictions['3d'] +
                regime_weights.get('label_5d', 0.00) * predictions['5d'] +
                regime_weights.get('label_10d', 0.60) * predictions['10d'] +
                regime_weights.get('label_15d', 0.40) * predictions['15d']
            )
        else:
            combined_pred = self._calculate_fallback_scores(filtered_df, available_cols)
            predictions = self._estimate_predictions_from_features(filtered_df, available_cols)

        # Continuous global percentile scoring
        scores = self._to_global_score(combined_pred)

        for i, code in enumerate(codes):
            results[code] = {
                'score': float(scores[i]),
                'pred_3d': float(predictions['3d'][i]) if i < len(predictions['3d']) else 0,
                'pred_5d': float(predictions['5d'][i]) if i < len(predictions['5d']) else 0,
                'pred_10d': float(predictions['10d'][i]) if i < len(predictions['10d']) else 0,
                'pred_15d': float(predictions.get('15d', np.zeros(1))[min(i, len(predictions.get('15d', [0]))-1)]) if '15d' in predictions else 0,
            }

        # Enhanced executability filters
        results = self._apply_enhanced_executability_filters(results, date)

        # Compute rank_score
        for code, data in results.items():
            data['rank_score'] = 0.60 * data.get('pred_10d', 0) + 0.40 * data.get('pred_15d', 0)

        # Fill missing codes
        for code in stock_codes:
            if code not in results:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0,
                                 'exec_filter': 'no_data'}

        # V4.7.6 post-processing
        if len(results) >= 2:
            results = self._apply_consistency_bonus(results)
            results = self._apply_vol_discount(results, date)

        return results
