#!/usr/bin/env python3
"""ng2.1 v2 Trainer — Regime-Tailored Alpha Specialist (RTAS).

vs ng2.1 v1:
  v1: 硬 regime-filter (drop off-regime data) → sample efficiency 损失 → ICIR 下降
  v2: 全数据训练 + 在 regime 期 sample weight ×1.5 (软引导) +
      regime-specific feature set (Phase 0 数据驱动选出)
      + 不同 label horizon (bull=15d, bear=5d)
      + --wf-report-dir 必开, 拿真 fold OOS predictions for system-level eval

输入 feature 子集来自 ml_models/ng21/{bull,bear}_features.json (Phase 2 输出).
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml_models.ng.ng_trainer import NGTrainer

logger = logging.getLogger(__name__)
NG21V2_DIR = Path(__file__).resolve().parent
SAMPLE_WEIGHT_MULT = 1.5  # in-regime ×1.5, off-regime ×1.0


def _load_regime_map(db_path: str) -> dict:
    """trade_date → regime_v2 (+1 bull / -1 bear / 0 absent)."""
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


def _load_feature_subset(spec_name: str) -> list[str]:
    """Load Phase 2 feature list for given specialist (bull|bear)."""
    p = NG21V2_DIR / f'{spec_name}_features.json'
    if not p.exists():
        raise FileNotFoundError(
            f'{p} not found. Run scripts/ng21_phase2_select_features.py first.'
        )
    with open(p, 'r') as f:
        d = json.load(f)
    feats = d['features']
    logger.info(f'NG21v2 {spec_name} feature set: {len(feats)} features '
                f'(label={d["label_horizon"]})')
    return feats


class NG21v2Trainer(NGTrainer):
    """Regime-Tailored Alpha Specialist trainer.

    Key differences vs NG21Trainer (v1):
      1. NO hard regime filter — full data trained
      2. compute_sample_weights: in-regime samples ×1.5
      3. Feature subset loaded from Phase 2 JSON (regime-tailored)
      4. _wf_report_dir always set → fold OOS predictions saved
    """

    def __init__(
        self,
        *args,
        regime: Optional[str] = None,
        wf_report_dir: Optional[str] = None,
        bull_label_horizon: str = 'label_15d',
        bear_label_horizon: str = 'label_5d',
        **kwargs,
    ):
        # Pin parent's _ng_version to ng1.0.1 so version_ge() resolves to ng1.0.1
        # behavior (no ng1.0.7 regime weights, no conditional features)
        kwargs.setdefault('version', 'ng1.0.1')
        super().__init__(*args, **kwargs)

        if regime not in (None, 'bull', 'bear'):
            raise ValueError(f"regime must be bull/bear, got {regime!r}")

        self._ng21v2_regime = regime
        self._ng21v2_variant = f'ng2.1v2-{regime}' if regime else 'ng2.1v2'
        self._ng21v2_label_col = (
            bull_label_horizon if regime == 'bull' else bear_label_horizon
        )

        # Defensive: parent NG auto-enables regime_weight if version_ge(ng1.0.7)
        # — we pinned version='ng1.0.1' so this should be off, but pin again.
        self._regime_weight = False
        self._regime_weight_mode = None

        # Load feature subset (regime-tailored from Phase 2)
        if regime in ('bull', 'bear'):
            self._ng21v2_features = _load_feature_subset(regime)
            # Override stock_feature_cols to use ONLY the selected subset
            self.stock_feature_cols = list(self._ng21v2_features)
            logger.info(
                f'NG21v2 {regime}: stock_feature_cols overridden to '
                f'{len(self.stock_feature_cols)} regime-tailored features'
            )
        else:
            self._ng21v2_features = None

        # WF report dir for fold OOS predictions (system-level eval downstream)
        if wf_report_dir:
            self._wf_report_dir = wf_report_dir
            logger.info(f'NG21v2 wf_report_dir = {wf_report_dir}')

        # Load regime map for sample weighting
        self._ng21v2_regime_map = _load_regime_map(self.db_path)
        n_bull = sum(1 for r in self._ng21v2_regime_map.values() if r == 1)
        n_bear = sum(1 for r in self._ng21v2_regime_map.values() if r == -1)
        logger.info(
            f'NG21v2 init: variant={self._ng21v2_variant}, regime={regime}, '
            f'label={self._ng21v2_label_col}, '
            f'regime_map: bull={n_bull}, bear={n_bear}, '
            f'sample_weight_mult={SAMPLE_WEIGHT_MULT}'
        )

    # ------------------------------------------------------------------
    # Cache key: include regime + feature set hash (avoid v1 collision bug)
    # ------------------------------------------------------------------
    def _compute_cache_key(self, start_date, end_date):
        import hashlib
        base = super()._compute_cache_key(start_date, end_date)
        feat_str = '|'.join(self._ng21v2_features) if self._ng21v2_features else 'all'
        feat_hash = hashlib.md5(feat_str.encode()).hexdigest()[:8]
        suffix = f'v2_{self._ng21v2_regime}_{feat_hash}'
        return hashlib.md5(f'{base}_{suffix}'.encode()).hexdigest()[:12]

    # ------------------------------------------------------------------
    # Sample weighting: regime-soft (×1.5 in-regime) instead of hard filter
    # ------------------------------------------------------------------
    def compute_sample_weights(self, df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        # Hide market_return_20d to suppress V4.x bear×2.0 boost (v1 lesson)
        if 'market_return_20d' in df.columns:
            saved = df['market_return_20d'].copy()
            df = df.drop(columns=['market_return_20d'])
            try:
                w = super().compute_sample_weights(df, y)
            finally:
                df['market_return_20d'] = saved.values
        else:
            w = super().compute_sample_weights(df, y)

        # Apply soft regime weighting
        if self._ng21v2_regime in ('bull', 'bear'):
            target = 1 if self._ng21v2_regime == 'bull' else -1
            dates = df['trade_date'].astype(str).values
            in_regime = np.array(
                [self._ng21v2_regime_map.get(d) == target for d in dates],
                dtype=bool,
            )
            n_in = int(in_regime.sum())
            n_out = len(in_regime) - n_in
            w = w.copy()
            w[in_regime] *= SAMPLE_WEIGHT_MULT
            logger.info(
                f'  [ng21v2 soft regime weight] {self._ng21v2_regime}: '
                f'{n_in:,} in-regime ×{SAMPLE_WEIGHT_MULT}, {n_out:,} off-regime ×1.0'
            )
        return w

    # ------------------------------------------------------------------
    # walk_forward_train override: stamp metadata + rename pkl after save
    # ------------------------------------------------------------------
    def walk_forward_train(self, *args, **kwargs):
        self._regime_weight = False
        self._regime_weight_mode = None
        result = super().walk_forward_train(*args, **kwargs)

        model_data = None
        if isinstance(result, tuple) and len(result) >= 1:
            md = result[0]
            if isinstance(md, dict):
                model_data = md
                md['ng21v2_variant'] = self._ng21v2_variant
                md['ng21v2_regime'] = self._ng21v2_regime
                md['ng21v2_features'] = self._ng21v2_features

        if (
            model_data is not None
            and self._ng21v2_regime
            and not model_data.get('fast_check')
        ):
            self._rename_saved_pkl()
        return result

    def _rename_saved_pkl(self):
        from pathlib import Path
        import shutil

        try:
            project_root = Path(__file__).resolve().parents[2]
            ng_dir = project_root / 'ml_models' / 'trained_models' / 'ng'
            if not ng_dir.exists():
                return
            cands = sorted(
                ng_dir.glob('ng101_seed*_multi_target_*.pkl'),
                key=lambda p: p.stat().st_mtime,
            )
            if not cands:
                logger.warning('No ng101 pkl to rename')
                return
            src = cands[-1]
            tag = self._ng21v2_variant.replace('.', '')  # ng21v2-bull
            new_name = src.name.replace('ng101', tag)
            dst = src.parent / new_name
            shutil.move(str(src), str(dst))
            logger.info(f'NG21v2 pkl renamed: {src.name} → {new_name}')
        except Exception as e:
            logger.warning(f'NG21v2 pkl rename failed: {e}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    import random

    parser = argparse.ArgumentParser(description='NG v2.1 v2 — Regime-Tailored Alpha Specialist')
    parser.add_argument('--regime', choices=['bull', 'bear'], required=True,
                        help='Specialist target regime (bull|bear)')
    parser.add_argument('--wf-report-dir', required=True,
                        help='Output dir for per-fold OOS predictions (system-eval)')
    parser.add_argument('--start-date', default='2020-01-01')
    parser.add_argument('--end-date', default=None)
    parser.add_argument('--purge-days', type=int, default=15)
    parser.add_argument('--min-train-days', type=int, default=600)
    parser.add_argument('--val-days', type=int, default=120)
    parser.add_argument('--test-days', type=int, default=120)
    parser.add_argument('--step-days', type=int, default=120)
    parser.add_argument('--fast-check', action='store_true')
    parser.add_argument('--target-parallel', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=str, default=None)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    def _build(seed: int) -> NG21v2Trainer:
        random.seed(seed)
        np.random.seed(seed)
        import ml_models.training.train_v395_multi_target as _tm
        _tm._GLOBAL_RANDOM_SEED = seed
        # Per-seed subdir so each seed's WF predictions are preserved.
        # Without this, the trainer's "清空旧WF报告" step wipes the previous
        # seed's fold predictions when the next seed starts (verified bug).
        per_seed_dir = (
            str(Path(args.wf_report_dir) / f'seed{seed}')
            if args.wf_report_dir else None
        )
        t = NG21v2Trainer(regime=args.regime, wf_report_dir=per_seed_dir)
        if args.fast_check:
            t._fast_check = True
            t._fast_check_max_windows = 2
            t._fast_check_min_train = min(args.min_train_days, 400)
            t._fast_check_val_days = 60
            t._fast_check_test_days = 60
            t._fast_check_step_days = 60
        if args.target_parallel > 1:
            t._target_parallel = args.target_parallel
        return t

    seeds = (
        [int(s.strip()) for s in args.seeds.split(',')]
        if args.seeds else [args.seed]
    )

    logger.info('=' * 70)
    logger.info(f'NG2.1v2-{args.regime} | seeds={seeds} | '
                f'wf_report_dir={args.wf_report_dir} | fast={args.fast_check}')
    logger.info('=' * 70)

    for i, seed in enumerate(seeds):
        logger.info(f'\n--- Seed {seed} ({i+1}/{len(seeds)}) ---')
        t = _build(seed)
        t.walk_forward_train(
            start_date=args.start_date, end_date=args.end_date,
            purge_days=args.purge_days,
            min_train_days=args.min_train_days,
            val_days=args.val_days, test_days=args.test_days,
            step_days=args.step_days,
        )
