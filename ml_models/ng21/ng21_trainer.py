#!/usr/bin/env python3
"""ng2.1 Trainer — Bull/Bear specialist via regime-filtered training data.

Subclass of NGTrainer. Two distinct usage modes:

  ng2.1-bull:
    --regime-filter bull
    Trains only on V11=bull days from market_regime_signals.
    Label horizon: 15d industry excess (longer than ng1.0.1 to capture trend).
    Target gates: V5.2 ≥ 75%, Sharpe ≥ 2.85, MaxDD ≤ -13%.

  ng2.1-bear:
    --regime-filter bear --dd-penalty-lambda {0.3,0.5,0.8}
    Trains only on V11=bear days. Applies DD-penalty label transform:
        y_dd = label_5d × (1 + λ) if label_5d < 0 else label_5d
    This is asymmetric loss: amplifies negatives so model learns to AVOID them.
    Target gates: V5.2 ≥ 78%, Sharpe ≥ 1.90, MaxDD ≤ -22%, turnover ≤ 70x.

Design rationale (from ng2.0b post-mortem):
  - Sample weighting amplifies alpha but destroys MaxDD profile (ng2.0b: -14pp).
  - Regime-filtering = strict data subset, no contamination from off-regime days.
  - DD-penalty in LABEL (not in features) keeps β_UMD clean (ng1.5.0 lesson).
  - All regime info enters via training data + selection-time risk overlay.
    Nothing regime-aware lives inside the model itself.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Optional

import numpy as np
import pandas as pd

from ml_models.ng.ng_trainer import NGTrainer

logger = logging.getLogger(__name__)


def _load_regime_map(db_path: str) -> dict:
    """Read trade_date → regime_v2 (∈ {+1, -1, 0/None}) from market_regime_signals.

    Uses the *baseline* calibration table (ng2.0a production setting).
    """
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute('PRAGMA busy_timeout=30000')
    try:
        rows = conn.execute(
            'SELECT trade_date, regime_v2 FROM market_regime_signals '
            'WHERE regime_v2 IS NOT NULL'
        ).fetchall()
    finally:
        conn.close()
    return {str(d): int(r) for d, r in rows}


class NG21Trainer(NGTrainer):
    """ng2.1 specialist trainer.

    Reuses ng1.0.1 feature cache (no schema fork). Adds:
      - regime-filter: drop rows whose trade_date doesn't match target regime
      - DD-penalty label transform (bear only, applied on label_5d)
      - Custom label horizon selection (bull→15d, bear→5d)
    """

    NG21_VERSION = 'ng2.1'

    def __init__(
        self,
        *args,
        regime_filter: Optional[str] = None,
        dd_penalty_lambda: float = 0.0,
        bull_label_horizon: str = 'label_15d',
        bear_label_horizon: str = 'label_5d',
        **kwargs,
    ):
        # Reuse ng1.0.1 schema — no new cache table.
        kwargs.setdefault('version', 'ng1.0.1')
        super().__init__(*args, **kwargs)

        if regime_filter not in (None, 'bull', 'bear'):
            raise ValueError(f"regime_filter must be None/bull/bear, got {regime_filter!r}")
        if dd_penalty_lambda < 0:
            raise ValueError(f"dd_penalty_lambda must be ≥ 0, got {dd_penalty_lambda}")

        self._regime_filter = regime_filter
        self._dd_penalty_lambda = float(dd_penalty_lambda)
        self._ng21_label_col = (
            bull_label_horizon if regime_filter == 'bull' else bear_label_horizon
        )

        # IMPORTANT: keep _ng_version = 'ng1.0.1' so all parent version_ge() checks
        # resolve to ng1.0.1 behavior (no auto regime_weight, no conditional labels,
        # no NG107 interaction features). Specialist identity is tracked separately
        # via _ng21_variant and stamped into model_data at save time.
        self._ng21_variant = f'ng2.1-{regime_filter}' if regime_filter else 'ng2.1'

        # Sanity: parent ng_trainer.py auto-enables regime_weight for >= ng1.0.7.
        # Defensive belt-and-suspenders: explicitly disable here.
        self._regime_weight = False
        self._regime_weight_mode = None

        logger.info(
            f"NG21Trainer init: variant={self._ng21_variant}, "
            f"regime_filter={regime_filter}, dd_penalty_lambda={dd_penalty_lambda}, "
            f"primary_label={self._ng21_label_col}, "
            f"regime_weight=DISABLED (specialist already filtered)"
        )

    # ------------------------------------------------------------------
    # Cache key: must include regime_filter + dd_penalty_lambda
    # ------------------------------------------------------------------
    def _compute_cache_key(self, start_date, end_date):
        """Critical: parent uses (class_name, _ng_version, dates, mtime, feat_hash).
        NG21 pins _ng_version='ng1.0.1' for both bull/bear, so without this
        override bear would reuse bull's cached preprocessed data (BUG observed
        2026-04-26: bear window 1 IC bit-identical to bull window 1).
        """
        base = super()._compute_cache_key(start_date, end_date)
        import hashlib
        suffix = f"{self._regime_filter}_{self._dd_penalty_lambda:.4f}"
        return hashlib.md5(f"{base}_{suffix}".encode()).hexdigest()[:12]

    # ------------------------------------------------------------------
    # compute_sample_weights override: neutralize V4.x base bear×2.0 boost
    # ------------------------------------------------------------------
    def compute_sample_weights(self, df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """For ng2.1 specialists, the regime is already filtered upstream.
        The V4.x base trainer's `market_return_20d < -5%` ×2.0 weighting either:
          - over-emphasizes a 5% subset of bull days (BAD: contaminates bull
            specialist with extreme-pullback samples)
          - over-emphasizes a 7% subset of bear days (acceptable but redundant)
        We strip that signal cleanly by hiding `market_return_20d` from the
        parent's weight computation, then restoring it. NGTrainer's own
        `compute_sample_weights` (already neutered via _regime_weight=False
        and _regime_weight_mode=None) is bypassed by this hide/restore trick.
        """
        if 'market_return_20d' in df.columns:
            saved = df['market_return_20d'].copy()
            df = df.drop(columns=['market_return_20d'])
            try:
                w = super().compute_sample_weights(df, y)
            finally:
                df['market_return_20d'] = saved.values
            return w
        return super().compute_sample_weights(df, y)

    # ------------------------------------------------------------------
    # walk_forward_train override: re-disable parent's auto regime_weight
    # ------------------------------------------------------------------
    def walk_forward_train(self, *args, **kwargs):
        # Parent NGTrainer.walk_forward_train flips _regime_weight=True if
        # version_ge(_ng_version, 'ng1.0.7'). We've pinned _ng_version='ng1.0.1'
        # but pin again right before super() in case any caller mutated it.
        self._regime_weight = False
        self._regime_weight_mode = None
        result = super().walk_forward_train(*args, **kwargs)

        # Stamp specialist variant into model_data for downstream scorer routing.
        model_data = None
        if isinstance(result, tuple) and len(result) >= 1:
            model_data = result[0]
            if isinstance(model_data, dict):
                model_data['ng21_variant'] = self._ng21_variant
                model_data['regime_filter'] = self._regime_filter
                model_data['dd_penalty_lambda'] = self._dd_penalty_lambda

        # Save-time pkl rename: parent saves as ng101_seed42_*.pkl (because we
        # pinned _ng_version='ng1.0.1' for correct behavior). Rename the most
        # recent ng101 pkl to ng21bull/ng21bear_seed*_*.pkl so it doesn't
        # overwrite the production ng1.0.1 model.
        if (
            model_data is not None
            and self._regime_filter
            and not model_data.get('fast_check')
        ):
            self._rename_saved_pkl()

        return result

    def _rename_saved_pkl(self):
        """Rename the pkl this run just saved to ng21<variant>_*.pkl.

        用父类记录的确切保存路径 (self._last_saved_pkl), 不再 mtime-glob 抢最新
        ng101 pkl — 旧实现在并发训练时可能劫持真 ng1.0.1 模型, 且 rename 失败被
        吞掉后 specialist 会永久伪装成生产 ng1.0.1 (scorer 现已按 ng21_variant
        字段拒载, 双保险)。
        """
        from pathlib import Path
        import shutil

        src_path = getattr(self, '_last_saved_pkl', None)
        if not src_path or not Path(src_path).exists():
            raise RuntimeError(
                f'NG21 rename failed: 找不到本次训练保存的 pkl '
                f'(_last_saved_pkl={src_path!r}) — 检查父类保存流程'
            )
        src = Path(src_path)
        # New name: ng21-bull_seed42_multi_target_TIMESTAMP.pkl (etc.)
        # Keep the dash so glob pattern from scorer (`ver.replace('.', '')`)
        # matches: 'ng2.1-bull'.replace('.', '') == 'ng21-bull'.
        tag = self._ng21_variant.replace('.', '')  # 'ng2.1-bull' → 'ng21-bull'
        new_name = src.name.replace('ng101', tag)
        dst = src.parent / new_name
        shutil.move(str(src), str(dst))
        self._last_saved_pkl = str(dst)
        logger.info(f'NG21 pkl renamed: {src.name} → {new_name}')

    # ------------------------------------------------------------------
    # Data loading: filter by regime, then optionally rewrite label
    # ------------------------------------------------------------------
    def load_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        df = super().load_data(start_date=start_date, end_date=end_date)
        if df.empty:
            return df

        n_before = len(df)

        # --- Regime filter ---
        if self._regime_filter in ('bull', 'bear'):
            target = +1 if self._regime_filter == 'bull' else -1
            regime_map = _load_regime_map(self.db_path)
            dates = df['trade_date'].astype(str).values
            keep_mask = np.array(
                [regime_map.get(d) == target for d in dates], dtype=bool
            )
            n_kept = int(keep_mask.sum())
            n_dropped_off_regime = n_before - n_kept
            df = df.loc[keep_mask].reset_index(drop=True)
            logger.info(
                f"  [ng2.1 regime_filter={self._regime_filter}] "
                f"kept {n_kept:,} / {n_before:,} rows ({100*n_kept/n_before:.1f}%); "
                f"dropped {n_dropped_off_regime:,} off-regime"
            )
            if n_kept == 0:
                raise RuntimeError(
                    f"Regime filter {self._regime_filter!r} produced 0 rows — "
                    f"check market_regime_signals coverage and date range."
                )

        # --- DD-penalty label transform (bear only) ---
        if self._dd_penalty_lambda > 0 and self._regime_filter == 'bear':
            df = self._apply_dd_penalty(df)

        return df

    def _apply_dd_penalty(self, df: pd.DataFrame) -> pd.DataFrame:
        """Asymmetric penalty: y_dd = y * (1 + λ × indicator(y < 0)).

        Effect: negative-return rows get amplified (more negative), so the model
        ranks them lower → tends to avoid drawdown stocks. Equivalent in spirit
        to ng1.0.4 RA-penalty but applied only to negatives (ng104 used a power
        transform on |label| which inflated turnover).

        Applied to all label horizons (label_3d/5d/10d/15d) so multi-target
        training keeps consistent ranking signal.
        """
        lam = self._dd_penalty_lambda
        label_cols = [c for c in ('label_3d', 'label_5d', 'label_10d', 'label_15d') if c in df.columns]
        for col in label_cols:
            y = df[col].values
            penalty = np.where(y < 0, -lam * y, 0.0)  # positive offset
            df[col] = y - penalty  # y_neg → y_neg × (1+λ); y_pos unchanged
        logger.info(
            f"  [ng2.1 DD-penalty λ={lam}] applied to {label_cols}; "
            f"asymmetric: y_neg → y_neg × (1+λ)"
        )
        return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    import random

    parser = argparse.ArgumentParser(description='NG v2.1 Specialist Trainer')
    parser.add_argument('--regime-filter', choices=['bull', 'bear'], required=True,
                        help='Train only on regime_v2 matching days. bull=+1, bear=-1.')
    parser.add_argument('--dd-penalty-lambda', type=float, default=0.0,
                        help='Bear-only DD-penalty (label_neg × (1+λ)). Recommended: 0.3/0.5/0.8.')
    parser.add_argument('--start-date', default='2020-01-01')
    parser.add_argument('--end-date', default=None)
    parser.add_argument('--purge-days', type=int, default=15)
    # Defaults lowered vs ng_trainer.py because regime-filter shrinks effective
    # date count by 40-60%. Bull 2020+ ~620 days, Bear 2020+ ~906 days.
    parser.add_argument('--min-train-days', type=int, default=300)
    parser.add_argument('--val-days', type=int, default=60)
    parser.add_argument('--test-days', type=int, default=60)
    parser.add_argument('--step-days', type=int, default=60)
    parser.add_argument('--fast-check', action='store_true',
                        help='Fast check: 2 WF windows, no model save')
    parser.add_argument('--target-parallel', type=int, default=1,
                        help='Targets trained concurrently per window (1=serial, 4=all parallel)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seeds for ensemble (overrides --seed)')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    def _build_trainer(seed: int) -> NG21Trainer:
        random.seed(seed)
        np.random.seed(seed)
        import ml_models.training.train_v395_multi_target as _tm
        _tm._GLOBAL_RANDOM_SEED = seed

        t = NG21Trainer(
            regime_filter=args.regime_filter,
            dd_penalty_lambda=args.dd_penalty_lambda,
        )
        if args.fast_check:
            t._fast_check = True
            t._fast_check_max_windows = 2
            # Fast-check: small windows. Post-filter dates are scarce.
            t._fast_check_min_train = min(args.min_train_days, 200)
            t._fast_check_val_days = 30
            t._fast_check_test_days = 30
            t._fast_check_step_days = 30
        if args.target_parallel > 1:
            t._target_parallel = args.target_parallel
        return t

    seed_list = (
        [int(s.strip()) for s in args.seeds.split(',')]
        if args.seeds else [args.seed]
    )

    logger.info('=' * 60)
    logger.info(
        f"NG2.1-{args.regime_filter} training | "
        f"seeds={seed_list} | dd_λ={args.dd_penalty_lambda} | "
        f"fast_check={args.fast_check}"
    )
    logger.info('=' * 60)

    for i, seed in enumerate(seed_list):
        logger.info(f"\n--- Seed {seed} ({i+1}/{len(seed_list)}) ---")
        trainer = _build_trainer(seed)
        trainer.walk_forward_train(
            start_date=args.start_date,
            end_date=args.end_date,
            purge_days=args.purge_days,
            min_train_days=args.min_train_days,
            val_days=args.val_days,
            test_days=args.test_days,
            step_days=args.step_days,
        )
