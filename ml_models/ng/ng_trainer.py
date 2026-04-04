#!/usr/bin/env python3
"""
NG Trainer — inherits V485Trainer, overrides feature loading to use ng_feature_cache.

The NG (Next Generation) model uses a clean 62-factor set computed by
ng_feature_calculator and stored in ng_feature_cache.  This trainer:

  1. Reads ng_feature_cache (JSON stock features + market DB columns + labels)
  2. Parses features into a flat DataFrame with 62 named columns
  3. Delegates Walk-Forward, Ensemble, LambdaRank, Bear-Specialist, etc.
     entirely to V485Trainer (unchanged)
  4. Saves models to ml_models/trained_models/ng/

Key differences from V485Trainer:
  - Data source: ng_feature_cache (not v39_feature_cache)
  - Feature set: 62 NG factors (not 61 legacy factors)
  - label_15d: stored in cache but NaN-filled with label_10d fallback (V485 requires 4 targets)
  - No ETF data: NG cache contains A-shares only
  - No brain_roll_spread: NG factors are self-contained
"""

import json
import logging
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_models.training.train_v395_multi_target import V485Trainer

# Optional fast JSON
try:
    import orjson
    _json_loads = orjson.loads
except ImportError:
    _json_loads = json.loads

logger = logging.getLogger(__name__)

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')

# ---------------------------------------------------------------------------
# Feature name constants
# ---------------------------------------------------------------------------

STOCK_FEATURE_NAMES: List[str] = [
    # Trend state (12)
    'price_above_ma20', 'price_above_ma60', 'ma_alignment', 'trend_strength_20d',
    'new_high_20d', 'new_high_60d', 'days_since_breakout', 'adx_proxy',
    'macd_histogram', 'macd_acceleration', 'price_channel_position', 'cumulative_return_60d',
    # Pullback entry (10)
    'pullback_from_high', 'pullback_to_ma10', 'pullback_to_ma20', 'rsi_14',
    'kdj_j_value', 'volume_contraction', 'lower_shadow_ratio', 'consecutive_down_days',
    'bollinger_position', 'intraday_recovery',
    # Volume confirmation (8)
    'volume_ratio_5d', 'volume_price_corr', 'obv_trend', 'volume_breakout',
    'log_amount_ma5', 'turnover_rate', 'up_volume_ratio', 'volume_cv',
    # Fundamental quality (14)
    'roe_ttm', 'roe_change', 'revenue_growth', 'net_profit_margin', 'ocf_quality',
    'pe_ttm', 'pb', 'pe_percentile_60d', 'debt_to_assets', 'current_ratio',
    'log_market_cap', 'log_adv_20d', 'free_float_ratio', 'dv_ratio',
    # Industry (8)
    'industry_return_5d', 'industry_return_20d', 'industry_relative_strength',
    'industry_breadth', 'industry_volume_change', 'industry_rank_return_5d',
    'sw_index_return_5d', 'industry_hhi',
]

MARKET_FEATURE_NAMES: List[str] = [
    'market_return_5d', 'market_return_20d', 'market_volatility_20d',
    'market_breadth', 'market_new_high_ratio', 'northbound_flow_5d',
    'market_volume_ratio', 'market_drawdown', 'vix_proxy', 'market_momentum_diff',
]

ALL_FEATURE_NAMES: List[str] = STOCK_FEATURE_NAMES + MARKET_FEATURE_NAMES  # 62 total


# ---------------------------------------------------------------------------
# NGTrainer
# ---------------------------------------------------------------------------

class NGTrainer(V485Trainer):
    """NG Trainer — loads from ng_feature_cache, delegates training to V485."""

    VERSION_TAG = 'ng'

    # NG target weights: no label_15d
    TARGET_WEIGHTS = {
        'label_3d': 0.15,
        'label_5d': 0.50,
        'label_10d': 0.35,
    }

    def __init__(self, db_path: str = DB_PATH):
        super().__init__(db_path)
        # Override target weights
        self.target_weights = dict(self.TARGET_WEIGHTS)
        # Disable V485 ETF loading
        self._turbo_skip_etf = True
        # Stub market_calculator for V475 model_data serialization (line 7714)
        # NG computes market features in ng_cache_updater, not via market_calculator
        class _StubMC:
            class market_features:
                columns = ['date'] + list(MARKET_FEATURE_NAMES)
        self.market_calculator = _StubMC()

    # ------------------------------------------------------------------
    # Feature name accessors
    # ------------------------------------------------------------------

    @staticmethod
    def get_feature_names() -> List[str]:
        return list(ALL_FEATURE_NAMES)

    @staticmethod
    def get_stock_feature_names() -> List[str]:
        return list(STOCK_FEATURE_NAMES)

    @staticmethod
    def get_market_feature_names() -> List[str]:
        return list(MARKET_FEATURE_NAMES)

    # ------------------------------------------------------------------
    # Data loading — override the entire chain
    # ------------------------------------------------------------------

    def load_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load training data from ng_feature_cache.

        Returns DataFrame with columns:
          code, trade_date, [62 feature names], label_3d, label_5d, label_10d, label_15d
        (label_15d is filled with zeros for compatibility with V485's walk-forward)
        """
        logger.info("NG Trainer: Loading data from ng_feature_cache ...")

        conn = sqlite3.connect(self.db_path, timeout=30)

        # Build query
        date_filter = ""
        params = []
        if start_date:
            date_filter += " AND trade_date >= ?"
            params.append(start_date)
        if end_date:
            date_filter += " AND trade_date <= ?"
            params.append(end_date)

        query = f"""
        SELECT code, trade_date, features_json,
               label_3d, label_5d, label_10d, label_15d,
               market_return_5d, market_return_20d, market_volatility_20d,
               market_breadth, market_new_high_ratio, northbound_flow_5d,
               market_volume_ratio, market_drawdown, vix_proxy,
               market_momentum_diff
        FROM ng_feature_cache
        WHERE label_5d IS NOT NULL {date_filter}
        ORDER BY trade_date, code
        """

        df_raw = pd.read_sql(query, conn, params=params)
        conn.close()

        n_raw = len(df_raw)
        logger.info(f"  Raw rows from ng_feature_cache: {n_raw:,}")

        if n_raw == 0:
            logger.warning("  ng_feature_cache returned 0 rows!")
            return pd.DataFrame()

        # Parse features_json → individual stock feature columns
        logger.info("  Parsing features_json ...")
        parsed_rows = df_raw['features_json'].apply(_json_loads).tolist()
        df_stock_features = pd.DataFrame(parsed_rows)

        # Keep only the expected stock feature columns (ignore extra keys)
        for col in STOCK_FEATURE_NAMES:
            if col not in df_stock_features.columns:
                df_stock_features[col] = np.nan

        df_stock_features = df_stock_features[STOCK_FEATURE_NAMES]

        # Assemble result DataFrame
        result = pd.DataFrame()
        result['code'] = df_raw['code'].values
        result['trade_date'] = df_raw['trade_date'].values

        # Stock features
        for col in STOCK_FEATURE_NAMES:
            result[col] = df_stock_features[col].values

        # Market features (from DB columns)
        for col in MARKET_FEATURE_NAMES:
            if col in df_raw.columns:
                result[col] = df_raw[col].values
            else:
                result[col] = np.nan

        # Labels
        result['label_3d'] = pd.to_numeric(df_raw['label_3d'], errors='coerce').values
        result['label_5d'] = pd.to_numeric(df_raw['label_5d'], errors='coerce').values
        result['label_10d'] = pd.to_numeric(df_raw['label_10d'], errors='coerce').values
        # label_15d: keep as NaN where missing (prepare_features handles fallback)
        result['label_15d'] = pd.to_numeric(df_raw.get('label_15d'), errors='coerce').values

        # P0-2 fix: fill market features FIRST (ffill), then stock features (zero)
        # Market features: forward-fill (same value for all stocks on same day)
        result = result.sort_values('trade_date')
        for col in MARKET_FEATURE_NAMES:
            result[col] = pd.to_numeric(result[col], errors='coerce')
            result[col] = result[col].ffill()
        result = result.dropna(subset=MARKET_FEATURE_NAMES)

        # Stock features: fill NaN with 0 (unknown → neutral)
        for col in STOCK_FEATURE_NAMES:
            result[col] = pd.to_numeric(result[col], errors='coerce').fillna(0.0)

        # Drop rows with NaN in core labels (3d/5d/10d)
        result = result.dropna(subset=['label_3d', 'label_5d', 'label_10d'])

        result = result.sort_values(['trade_date', 'code']).reset_index(drop=True)

        # Set market_calculator.market_features for V475 serialization (line 7714)
        # NG doesn't use MarketStateCalculator; fake it with a DataFrame
        mkt_df = result[['trade_date'] + [c for c in MARKET_FEATURE_NAMES if c in result.columns]].drop_duplicates('trade_date')
        if self.market_calculator is not None:
            self.market_calculator.market_features = mkt_df

        n_stocks = result['code'].nunique()
        n_dates = result['trade_date'].nunique()
        logger.info(f"  NG load_data complete: {len(result):,} rows, "
                     f"{n_stocks:,} stocks, {n_dates} dates, 62 features")

        return result

    # ------------------------------------------------------------------
    # Feature preparation — override to match NG's 62-feature set
    # ------------------------------------------------------------------

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare NG feature matrix.

        Stock features get cross-sectional Robust Z-Score normalization.
        Market features are kept as raw values (same for all stocks on a date).

        Returns: (X, y_3d, y_5d, y_10d, y_15d, df)
        """
        logger.info("NG prepare_features: 62 factors (52 stock + 10 market)")

        self.stock_feature_cols = [c for c in STOCK_FEATURE_NAMES if c in df.columns]
        self.macro_feature_cols = [c for c in MARKET_FEATURE_NAMES if c in df.columns]
        self.feature_names = self.stock_feature_cols + self.macro_feature_cols

        logger.info(f"  Stock features: {len(self.stock_feature_cols)}, "
                     f"Market features: {len(self.macro_feature_cols)}")

        # Cross-sectional Robust Z-Score on stock features
        logger.info("  Applying cross-sectional Robust Z-Score to stock features ...")
        stock_data = df[self.stock_feature_cols].values.copy()
        dates_arr = df['trade_date'].values
        stock_data = self._robust_zscore_cross_section(stock_data, dates_arr)
        df[self.stock_feature_cols] = stock_data
        df[self.stock_feature_cols] = df[self.stock_feature_cols].fillna(0.0)

        self.rank_normalized = False
        self.robust_zscore = True
        self.dual_stream = False

        X = df[self.feature_names].values
        y_3d = df['label_3d'].values
        y_5d = df['label_5d'].values
        y_10d = df['label_10d'].values
        # label_15d: use actual values where available, fallback to label_10d
        # (V485 WF trains all 4 targets; all-zero labels crash Sharpe blend + CatBoost)
        y_15d_raw = pd.to_numeric(df['label_15d'], errors='coerce').values
        y_10d_vals = df['label_10d'].values.copy()
        y_15d = np.where(np.isnan(y_15d_raw), y_10d_vals, y_15d_raw)

        # Winsorization deferred to walk_forward_train per-window
        self.winsorize_bounds = None

        logger.info(f"  Feature matrix: {X.shape[0]:,} x {X.shape[1]}")
        return X, y_3d, y_5d, y_10d, y_15d, df

    # ------------------------------------------------------------------
    # Walk-Forward — delegates to V475 (via inheritance chain), saves to ng/
    # ------------------------------------------------------------------

    def walk_forward_train(self, start_date: str = None, end_date: str = None,
                           purge_days: int = 15, min_train_days: int = 900,
                           val_days: int = 120, test_days: int = 120,
                           step_days: int = 120):
        """NG Walk-Forward Training.

        Runs V475's walk_forward_train (which V485 inherits), then relocates
        the saved model to ml_models/trained_models/ng/.
        """
        import shutil

        logger.info("=" * 60)
        logger.info("NG Walk-Forward Training (62 clean factors, ng_feature_cache)")
        logger.info("=" * 60)
        logger.info(f"  Base: V4.8.5 machinery (6-model ensemble, LambdaRank, Bear Specialist)")
        logger.info(f"  Data: ng_feature_cache (52 stock + 10 market features)")
        logger.info(f"  Targets: 3d×{self.target_weights['label_3d']:.2f} + "
                     f"5d×{self.target_weights['label_5d']:.2f} + "
                     f"10d×{self.target_weights['label_10d']:.2f}")

        # V485's walk_forward_train calls V484's which calls V481's which calls V475's
        # V475 is the one that actually does the training and saves to v475/ dir
        # Then V481 renames to v481/, V484 renames to v484/, V485 renames to v485/
        # We let this chain run, then move the result to ng/
        model_data, history = super().walk_forward_train(
            start_date=start_date, end_date=end_date,
            purge_days=purge_days, min_train_days=min_train_days,
            val_days=val_days, test_days=test_days, step_days=step_days)

        # fast-check: skip model save
        if model_data.get('fast_check'):
            return model_data, history

        # Move from v485/ to ng/
        v485_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v485'
        ng_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'ng'
        ng_dir.mkdir(parents=True, exist_ok=True)

        v485_files = sorted(v485_dir.glob('v485_*.pkl'), key=lambda f: f.stat().st_mtime)
        if v485_files:
            latest = v485_files[-1]
            timestamp = latest.stem.replace('v485_multi_target_', '')
            new_path = ng_dir / f'ng_multi_target_{timestamp}.pkl'

            # Update model metadata
            model_data['version'] = 'ng'
            model_data['ng_innovations'] = {
                'base': 'V4.8.5 ensemble machinery',
                'data_source': 'ng_feature_cache (62 clean factors)',
                'feature_set': '52 stock + 10 market features',
                'stock_features': STOCK_FEATURE_NAMES,
                'market_features': MARKET_FEATURE_NAMES,
                'targets': ['3d', '5d', '10d'],
                'target_weights': self.target_weights,
                'no_label_15d': True,
            }
            model_data['feature_names'] = self.feature_names
            model_data['stock_feature_cols'] = self.stock_feature_cols
            model_data['macro_feature_cols'] = self.macro_feature_cols
            model_data['target_weights'] = self.target_weights

            joblib.dump(model_data, new_path)
            logger.info(f"\nNG model saved: {new_path}")
            logger.info(f"  Size: {new_path.stat().st_size / 1024 / 1024:.1f} MB")

            # Copy auxiliary files
            for aux in ['global_quantiles.npy', 'recommendation_thresholds.json']:
                src = v485_dir / aux
                if src.exists():
                    shutil.copy2(str(src), str(ng_dir / aux))

            # Clean up v485 artifacts
            latest.unlink()
            for hf in v485_dir.glob(f'training_history_{timestamp}*'):
                hf.unlink()

            # Save training history
            history['version'] = 'ng'
            history['base'] = 'NG (62 clean factors, ng_feature_cache)'
            history['ng_innovations'] = model_data['ng_innovations']

            import json as _json
            history_path = ng_dir / f'training_history_{timestamp}.json'
            with open(history_path, 'w', encoding='utf-8') as f:
                _json.dump(history, f, indent=2, ensure_ascii=False)
            latest_hist_path = ng_dir / 'training_history_latest.json'
            with open(latest_hist_path, 'w', encoding='utf-8') as f:
                _json.dump(history, f, indent=2, ensure_ascii=False)

            logger.info(f"\nNG training complete!")
            logger.info(f"  Features: {len(self.feature_names)}")
            logger.info(f"  Model: {new_path.name}")
        else:
            logger.warning("No v485 model file found to relocate to ng/")

        return model_data, history


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='NG Trainer')
    parser.add_argument('--start-date', default='2020-01-01')
    parser.add_argument('--end-date', default=None)
    parser.add_argument('--purge-days', type=int, default=15)
    parser.add_argument('--min-train-days', type=int, default=900)
    parser.add_argument('--val-days', type=int, default=120)
    parser.add_argument('--test-days', type=int, default=120)
    parser.add_argument('--step-days', type=int, default=120)
    parser.add_argument('--fast-check', action='store_true',
                        help='Fast check mode: 2 WF windows, no model save')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel WF workers')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    trainer = NGTrainer()

    if args.fast_check:
        trainer._fast_check = True
        trainer._fast_check_max_windows = 2
        trainer._fast_check_min_train = min(args.min_train_days, 600)
        trainer._fast_check_val_days = 60
        trainer._fast_check_test_days = 60
        trainer._fast_check_step_days = 60

    if args.parallel > 1:
        trainer._parallel_wf_workers = args.parallel

    model_data, history = trainer.walk_forward_train(
        start_date=args.start_date,
        end_date=args.end_date,
        purge_days=args.purge_days,
        min_train_days=args.min_train_days,
        val_days=args.val_days,
        test_days=args.test_days,
        step_days=args.step_days,
    )
