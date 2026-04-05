#!/usr/bin/env python3
"""
NG v1.1.0 Trainer — inherits V485Trainer, overrides feature loading to use ng_feature_cache.

v1.1.0 changes from v1.0.0 (ng1.0.0):
  - 58 stock features + 10 market = 68 total (was 52+10=62)
  - Labels are industry excess returns (not absolute)
  - ICIR adaptive composite weights (not hardcoded)
  - WF summary JSON generation for L4 scoring
  - Removed 11 low-efficiency factors, added 10 CS rank + 5 residual + 3 sector activity
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
from ml_models.ng.ng_schema import get_table_name

try:
    import orjson
    _json_loads = orjson.loads
except ImportError:
    _json_loads = json.loads

logger = logging.getLogger(__name__)

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')

# ---------------------------------------------------------------------------
# Feature name constants — NG v1.1.0 (68 total = 58 stock + 10 market)
# ---------------------------------------------------------------------------

STOCK_FEATURE_NAMES: List[str] = [
    # Trend state (5, was 12 — removed 7 low cross-sectional factors)
    'trend_strength_20d', 'days_since_breakout', 'adx_proxy',
    'pullback_from_high', 'volume_contraction',
    # Pullback entry (6, was 10 — removed bollinger_position, consecutive_down_days)
    'pullback_to_ma10', 'pullback_to_ma20', 'rsi_14',
    'kdj_j_value', 'lower_shadow_ratio', 'intraday_recovery',
    # Volume confirmation (7, was 8 — no change in v1.1.0 for this group)
    'volume_ratio_5d', 'volume_price_corr', 'obv_trend', 'volume_breakout',
    'log_amount_ma5', 'turnover_rate', 'up_volume_ratio', 'volume_cv',
    # Fundamental quality (14, unchanged)
    'roe_ttm', 'roe_change', 'revenue_growth', 'net_profit_margin', 'ocf_quality',
    'pe_ttm', 'pb', 'pe_percentile_60d', 'debt_to_assets', 'current_ratio',
    'log_market_cap', 'log_adv_20d', 'free_float_ratio', 'dv_ratio',
    # Industry rotation (11, was 8 — +3 sector activity features)
    'industry_return_5d', 'industry_return_20d', 'industry_relative_strength',
    'industry_breadth', 'industry_volume_change', 'industry_rank_return_5d',
    'sw_index_return_5d', 'industry_hhi',
    'sector_breadth_vs_market', 'sector_volume_vs_market', 'n_sectors_strong',
    # Cross-sectional rank (10, NEW in v1.1.0)
    'cs_rank_return_5d', 'cs_rank_return_20d', 'cs_rank_volume_surge',
    'cs_rank_turnover', 'cs_rank_rsi', 'cs_rank_new_high',
    'cs_rank_pullback', 'cs_rank_volatility', 'cs_rank_market_cap', 'cs_rank_pe',
    # Residual factors (5, NEW in v1.1.0)
    'residual_return_20d', 'residual_volume', 'idiosyncratic_volatility',
    'residual_skewness', 'relative_strength_vs_peers',
]

MARKET_FEATURE_NAMES: List[str] = [
    'market_return_5d', 'market_return_20d', 'market_volatility_20d',
    'market_breadth', 'market_new_high_ratio', 'northbound_flow_5d',
    'market_volume_ratio', 'market_drawdown', 'vix_proxy', 'market_momentum_diff',
]

ALL_FEATURE_NAMES: List[str] = STOCK_FEATURE_NAMES + MARKET_FEATURE_NAMES  # 68 total

# v1.0.0 constants (for backward compatibility reference)
NG_V1_VERSION = 'ng1.0.0'
NG_VERSION = 'ng1.0.1'


# ---------------------------------------------------------------------------
# NGTrainer
# ---------------------------------------------------------------------------

class NGTrainer(V485Trainer):
    """NG v1.1.0 Trainer — loads from ng_feature_cache, delegates training to V485."""

    VERSION_TAG = NG_VERSION

    # Default target weights — will be overridden by ICIR adaptive in walk_forward_train
    TARGET_WEIGHTS = {
        'label_3d': 0.10,
        'label_5d': 0.20,
        'label_10d': 0.35,
        'label_15d': 0.35,
    }

    def __init__(self, db_path: str = DB_PATH):
        super().__init__(db_path)
        self.target_weights = dict(self.TARGET_WEIGHTS)
        self._turbo_skip_etf = True
        self.cache_table = get_table_name(NG_VERSION)
        # Stub market_calculator for V475 model_data serialization
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
    # ICIR Adaptive Composite Weights
    # ------------------------------------------------------------------

    def _compute_icir_adaptive_weights(self, history: dict) -> dict:
        """
        Compute composite weights proportional to OOS ICIR for each target.
        Falls back to default weights if ICIR data unavailable.

        Returns dict like {'label_3d': 0.10, 'label_5d': 0.20, ...}
        """
        try:
            wf_windows = history.get('wf_windows', [])
            if not wf_windows:
                logger.warning("No WF windows in history, using default weights")
                return dict(self.TARGET_WEIGHTS)

            # Collect OOS IC values per target across windows
            target_ics: dict = {}
            for target in ['label_3d', 'label_5d', 'label_10d', 'label_15d']:
                ics = []
                for w in wf_windows:
                    metrics = w.get('test_metrics', w.get('oos_metrics', {}))
                    ic = metrics.get(f'{target}_ic', metrics.get(f'ic_{target}'))
                    if ic is not None and not np.isnan(ic):
                        ics.append(ic)
                if ics:
                    mean_ic = np.mean(ics)
                    std_ic = np.std(ics) if len(ics) > 1 else max(abs(mean_ic) * 0.5, 0.01)
                    icir = mean_ic / (std_ic + 1e-8)
                    target_ics[target] = max(icir, 0.0)  # Floor at 0
                else:
                    target_ics[target] = 0.0

            total = sum(target_ics.values())
            if total < 1e-8:
                logger.warning("All ICIR ≤ 0, using default weights")
                return dict(self.TARGET_WEIGHTS)

            weights = {k: v / total for k, v in target_ics.items()}
            logger.info(f"ICIR adaptive weights: {', '.join(f'{k}={v:.3f}' for k, v in weights.items())}")
            logger.info(f"  Raw ICIR: {', '.join(f'{k}={v:.3f}' for k, v in target_ics.items())}")
            return weights

        except Exception as e:
            logger.warning(f"ICIR adaptive weights failed: {e}, using defaults")
            return dict(self.TARGET_WEIGHTS)

    # ------------------------------------------------------------------
    # WF Summary Generation
    # ------------------------------------------------------------------

    def _generate_wf_summary(self, history: dict, model_dir: Path) -> dict:
        """
        Generate wf_summary.json for L4 scoring (WFER + OOS monthly IC).
        Returns the summary dict.
        """
        summary = {
            'version': NG_VERSION,
            'generated_at': datetime.now().isoformat(),
            'wf_windows': [],
            'aggregate': {},
        }

        try:
            wf_windows = history.get('wf_windows', [])
            if not wf_windows:
                logger.warning("No WF windows for summary generation")
                return summary

            # Per-window metrics
            window_summaries = []
            all_oos_ics = {'label_3d': [], 'label_5d': [], 'label_10d': [], 'label_15d': []}

            for i, w in enumerate(wf_windows):
                train_period = w.get('train_period', '')
                test_period = w.get('test_period', '')
                metrics = w.get('test_metrics', w.get('oos_metrics', {}))

                ws = {
                    'window_id': i,
                    'train_period': train_period,
                    'test_period': test_period,
                    'metrics': {},
                }

                for target in ['label_3d', 'label_5d', 'label_10d', 'label_15d']:
                    ic = metrics.get(f'{target}_ic', metrics.get(f'ic_{target}'))
                    if ic is not None and not np.isnan(ic):
                        ws['metrics'][f'{target}_ic'] = float(ic)
                        all_oos_ics[target].append(float(ic))

                # Train/test sample counts
                ws['n_train'] = w.get('n_train', 0)
                ws['n_test'] = w.get('n_test', 0)

                window_summaries.append(ws)

            summary['wf_windows'] = window_summaries

            # Aggregate metrics
            for target in ['label_3d', 'label_5d', 'label_10d', 'label_15d']:
                ics = all_oos_ics[target]
                if ics:
                    mean_ic = np.mean(ics)
                    std_ic = np.std(ics) if len(ics) > 1 else 0.01
                    icir = mean_ic / (std_ic + 1e-8)
                    summary['aggregate'][f'{target}_mean_ic'] = float(mean_ic)
                    summary['aggregate'][f'{target}_std_ic'] = float(std_ic)
                    summary['aggregate'][f'{target}_icir'] = float(icir)
                    summary['aggregate'][f'{target}_ic_positive_ratio'] = float(np.mean(np.array(ics) > 0))

            # WF Efficiency Ratio (WFER): mean OOS IC / mean train IC
            train_ics = []
            for w in wf_windows:
                train_m = w.get('train_metrics', {})
                ic = train_m.get('label_10d_ic', train_m.get('ic_label_10d'))
                if ic is not None and not np.isnan(ic):
                    train_ics.append(float(ic))

            oos_ics_10d = all_oos_ics.get('label_10d', [])
            if train_ics and oos_ics_10d:
                wfer = np.mean(oos_ics_10d) / (np.mean(train_ics) + 1e-8)
                summary['aggregate']['wfer'] = float(wfer)
            else:
                summary['aggregate']['wfer'] = None

            # OOS IC half-life (months until IC decays to half)
            if len(oos_ics_10d) >= 3:
                initial_ic = oos_ics_10d[0]
                half_target = initial_ic / 2
                half_life_months = None
                for j, ic in enumerate(oos_ics_10d[1:], 1):
                    if ic <= half_target:
                        half_life_months = j * 4  # ~4 months per window (120 days)
                        break
                summary['aggregate']['oos_ic_half_life_months'] = half_life_months
            else:
                summary['aggregate']['oos_ic_half_life_months'] = None

            summary['n_windows'] = len(wf_windows)
            summary['total_oos_days'] = sum(w.get('n_test', 0) for w in wf_windows)

            # Save to file
            summary_path = model_dir / 'wf_summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info(f"WF summary saved: {summary_path}")

        except Exception as e:
            logger.warning(f"WF summary generation failed: {e}")

        return summary

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load training data from ng_feature_cache.

        v1.1.0: Labels are now industry excess returns.
        Features include 10 CS rank + 5 residual + 3 sector activity.
        """
        logger.info(f"NG {NG_VERSION} Trainer: Loading data from {self.cache_table} ...")

        conn = sqlite3.connect(self.db_path, timeout=30)

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
        FROM {self.cache_table}
        WHERE label_5d IS NOT NULL {date_filter}
        ORDER BY trade_date, code
        """

        df_raw = pd.read_sql(query, conn, params=params)
        conn.close()

        n_raw = len(df_raw)
        logger.info(f"  Raw rows from {self.cache_table}: {n_raw:,}")

        if n_raw == 0:
            logger.warning(f"  {self.cache_table} returned 0 rows!")
            return pd.DataFrame()

        # Parse features_json
        logger.info("  Parsing features_json ...")
        parsed_rows = df_raw['features_json'].apply(_json_loads).tolist()
        df_stock_features = pd.DataFrame(parsed_rows)

        for col in STOCK_FEATURE_NAMES:
            if col not in df_stock_features.columns:
                df_stock_features[col] = np.nan

        df_stock_features = df_stock_features[STOCK_FEATURE_NAMES]

        # Assemble result
        result = pd.DataFrame()
        result['code'] = df_raw['code'].values
        result['trade_date'] = df_raw['trade_date'].values

        for col in STOCK_FEATURE_NAMES:
            result[col] = df_stock_features[col].values

        for col in MARKET_FEATURE_NAMES:
            if col in df_raw.columns:
                result[col] = df_raw[col].values
            else:
                result[col] = np.nan

        # Labels (industry excess returns in v1.1.0)
        result['label_3d'] = pd.to_numeric(df_raw['label_3d'], errors='coerce').values
        result['label_5d'] = pd.to_numeric(df_raw['label_5d'], errors='coerce').values
        result['label_10d'] = pd.to_numeric(df_raw['label_10d'], errors='coerce').values
        result['label_15d'] = pd.to_numeric(df_raw.get('label_15d'), errors='coerce').values

        # Market features: ffill
        result = result.sort_values('trade_date')
        for col in MARKET_FEATURE_NAMES:
            result[col] = pd.to_numeric(result[col], errors='coerce')
            result[col] = result[col].ffill()
        result = result.dropna(subset=MARKET_FEATURE_NAMES)

        # Stock features: fill NaN with 0
        for col in STOCK_FEATURE_NAMES:
            result[col] = pd.to_numeric(result[col], errors='coerce').fillna(0.0)

        result = result.dropna(subset=['label_3d', 'label_5d', 'label_10d'])
        result = result.sort_values(['trade_date', 'code']).reset_index(drop=True)

        # Stub market_calculator for V475 serialization
        mkt_df = result[['trade_date'] + [c for c in MARKET_FEATURE_NAMES if c in result.columns]].drop_duplicates('trade_date')
        if self.market_calculator is not None:
            self.market_calculator.market_features = mkt_df

        n_stocks = result['code'].nunique()
        n_dates = result['trade_date'].nunique()
        logger.info(f"  NG load_data complete: {len(result):,} rows, "
                     f"{n_stocks:,} stocks, {n_dates} dates, {len(ALL_FEATURE_NAMES)} features")

        return result

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare NG v1.1.0 feature matrix."""
        logger.info(f"NG {NG_VERSION} prepare_features: {len(ALL_FEATURE_NAMES)} factors "
                     f"({len(STOCK_FEATURE_NAMES)} stock + {len(MARKET_FEATURE_NAMES)} market)")

        self.stock_feature_cols = [c for c in STOCK_FEATURE_NAMES if c in df.columns]
        self.macro_feature_cols = [c for c in MARKET_FEATURE_NAMES if c in df.columns]
        self.feature_names = self.stock_feature_cols + self.macro_feature_cols

        logger.info(f"  Stock features: {len(self.stock_feature_cols)}, "
                     f"Market features: {len(self.macro_feature_cols)}")

        # Cross-sectional Robust Z-Score on stock features
        # Note: CS rank features are already [0,1] but z-scoring won't hurt much
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
        y_15d_raw = pd.to_numeric(df['label_15d'], errors='coerce').values
        y_10d_vals = df['label_10d'].values.copy()
        y_15d = np.where(np.isnan(y_15d_raw), y_10d_vals, y_15d_raw)

        self.winsorize_bounds = None

        logger.info(f"  Feature matrix: {X.shape[0]:,} x {X.shape[1]}")
        return X, y_3d, y_5d, y_10d, y_15d, df

    # ------------------------------------------------------------------
    # Walk-Forward Training
    # ------------------------------------------------------------------

    def walk_forward_train(self, start_date: str = None, end_date: str = None,
                           purge_days: int = 15, min_train_days: int = 900,
                           val_days: int = 120, test_days: int = 120,
                           step_days: int = 120):
        """NG v1.1.0 Walk-Forward Training with ICIR adaptive weights and WF summary."""
        import shutil

        logger.info("=" * 60)
        logger.info(f"NG {NG_VERSION} Walk-Forward Training")
        logger.info("=" * 60)
        logger.info(f"  Base: V4.8.5 machinery (6-model ensemble, LambdaRank, Bear Specialist)")
        logger.info(f"  Data: ng_feature_cache ({len(ALL_FEATURE_NAMES)} features)")
        logger.info(f"  Labels: INDUSTRY EXCESS returns (stock - industry median)")
        logger.info(f"  Initial weights: {', '.join(f'{k}={v:.2f}' for k, v in self.target_weights.items())}")

        model_data, history = super().walk_forward_train(
            start_date=start_date, end_date=end_date,
            purge_days=purge_days, min_train_days=min_train_days,
            val_days=val_days, test_days=test_days, step_days=step_days)

        # fast-check: skip model save
        if model_data.get('fast_check'):
            return model_data, history

        # Compute ICIR adaptive weights from WF history
        adaptive_weights = self._compute_icir_adaptive_weights(history)
        self.target_weights = adaptive_weights

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
            model_data['version'] = NG_VERSION
            model_data['ng_innovations'] = {
                'base': 'V4.8.5 ensemble machinery',
                'version': NG_VERSION,
                'prev_version': NG_V1_VERSION,
                'data_source': 'ng_feature_cache (industry excess labels)',
                'feature_set': f'{len(STOCK_FEATURE_NAMES)} stock + {len(MARKET_FEATURE_NAMES)} market features',
                'stock_features': STOCK_FEATURE_NAMES,
                'market_features': MARKET_FEATURE_NAMES,
                'targets': ['3d', '5d', '10d', '15d'],
                'target_weights': adaptive_weights,
                'label_type': 'industry_excess_return',
                'new_in_v110': {
                    'cs_rank_factors': 10,
                    'residual_factors': 5,
                    'sector_activity_factors': 3,
                    'removed_factors': 11,
                    'icir_adaptive_weights': True,
                    'wf_summary': True,
                },
            }
            model_data['feature_names'] = self.feature_names
            model_data['stock_feature_cols'] = self.stock_feature_cols
            model_data['macro_feature_cols'] = self.macro_feature_cols
            model_data['target_weights'] = adaptive_weights

            joblib.dump(model_data, new_path)
            logger.info(f"\nNG {NG_VERSION} model saved: {new_path}")
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
            history['version'] = NG_VERSION
            history['base'] = f'NG {NG_VERSION} ({len(ALL_FEATURE_NAMES)} features, industry excess labels)'
            history['ng_innovations'] = model_data['ng_innovations']
            history['adaptive_weights'] = adaptive_weights

            history_path = ng_dir / f'training_history_{timestamp}.json'
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            latest_hist_path = ng_dir / 'training_history_latest.json'
            with open(latest_hist_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

            # Generate WF summary (v1.1.0)
            wf_summary = self._generate_wf_summary(history, ng_dir)

            logger.info(f"\nNG {NG_VERSION} training complete!")
            logger.info(f"  Features: {len(self.feature_names)}")
            logger.info(f"  ICIR weights: {', '.join(f'{k}={v:.3f}' for k, v in adaptive_weights.items())}")
            logger.info(f"  Model: {new_path.name}")
            if wf_summary.get('aggregate', {}).get('wfer') is not None:
                logger.info(f"  WFER: {wf_summary['aggregate']['wfer']:.3f}")
        else:
            logger.warning("No v485 model file found to relocate to ng/")

        return model_data, history


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=f'NG {NG_VERSION} Trainer')
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
