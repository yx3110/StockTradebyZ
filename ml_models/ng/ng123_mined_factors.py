"""ng1.2.3 mined alpha factors.

Selected via factor_mining_pipeline.py + Stage 2 cross-regime validation.
Per spec §4.2 (docs/superpowers/specs/2026-04-14-ng123-design.md).

MINED_FACTOR_SPEC is populated by Task 10 from stage2_status.json after
Stage 2 validation completes. Each entry specifies:
  - name: canonical name (with 'neg_' prefix if sign_flip=True)
  - sign_flip: multiply output by -1 (for IC<0 factors, treated as contrarian)
  - op/operand/window: formula per scripts.factor_mining_pipeline spec format
  - ic, icir: post-validation values
  - semantic: one-line description
"""
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.factor_mining_pipeline import generate_operands, compute_factor


__all__ = [
    'MINED_FACTOR_SPEC',
    'compute_mined_factor_value',
    'compute_all_mined_factors_for_stock',
    'get_mined_factor_names',
]


# ============================================================================
# MINED_FACTOR_SPEC — populated from Stage 2 results (initially empty).
# ============================================================================

MINED_FACTOR_SPEC: List[Dict] = [
    # Selected 2026-04-15 from stage2_status.json top-6.
    # All show negative IC in A-share → sign_flip=True (contrarian signals).
    # Name prefix `neg_` denotes sign-flipped for naming consistency with spec §4.2.
    {
        'name': 'neg_ts_cov_close_turnover_5',
        'type': 'binary_ts', 'op': 'ts_cov',
        'operand_a': 'close', 'operand_b': 'turnover', 'window': 5,
        'sign_flip': True,
        'ic': -0.090306, 'icir': -0.8026,
        'semantic': '5-day covariance(close, turnover) — contrarian microstructure',
    },
    {
        'name': 'neg_abs_ts_ret_volume_10',
        'type': 'depth2', 'elem_op': 'abs',
        'ts_op': 'ts_ret', 'operand': 'volume', 'window': 10,
        'sign_flip': True,
        'ic': -0.047846, 'icir': -0.7322,
        'semantic': '10-day |volume return| — contrarian volume-volatility',
    },
    {
        'name': 'neg_abs_ts_ret_volume_60',
        'type': 'depth2', 'elem_op': 'abs',
        'ts_op': 'ts_ret', 'operand': 'volume', 'window': 60,
        'sign_flip': True,
        'ic': -0.053732, 'icir': -0.7246,
        'semantic': '60-day |volume return| — contrarian long-horizon volume-volatility',
    },
    {
        'name': 'neg_ts_corr_turnover_high_10',
        'type': 'binary_ts', 'op': 'ts_corr',
        'operand_a': 'turnover', 'operand_b': 'high', 'window': 10,
        'sign_flip': True,
        'ic': -0.062888, 'icir': -0.6929,
        'semantic': '10-day corr(turnover, high) — contrarian liquidity-price',
    },
    {
        'name': 'neg_ts_corr_volume_high_10',
        'type': 'binary_ts', 'op': 'ts_corr',
        'operand_a': 'volume', 'operand_b': 'high', 'window': 10,
        'sign_flip': True,
        'ic': -0.062557, 'icir': -0.689,
        'semantic': '10-day corr(volume, high) — contrarian volume-price coupling',
    },
    {
        'name': 'neg_ts_cov_turnover_low_10',
        'type': 'binary_ts', 'op': 'ts_cov',
        'operand_a': 'turnover', 'operand_b': 'low', 'window': 10,
        'sign_flip': True,
        'ic': -0.069448, 'icir': -0.6671,
        'semantic': '10-day cov(turnover, low) — contrarian downside-turnover',
    },
]


def compute_mined_factor_value(spec: Dict, df_stock: pd.DataFrame) -> np.ndarray:
    """Compute a single mined factor for one stock's full OHLCV time series.

    Returns numpy array same length as df_stock. NaN where not computable.
    Applies sign flip if spec.sign_flip is True.
    """
    operands = generate_operands(df_stock)
    val_series = compute_factor(spec, operands)
    if val_series is None:
        return np.full(len(df_stock), np.nan)
    vals = np.asarray(val_series.values, dtype=np.float64)
    if spec.get('sign_flip', False):
        vals = -vals
    return vals


def compute_all_mined_factors_for_stock(df_stock: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Compute all mined factors for one stock's time series.

    Returns {name: array_of_length_len(df_stock)} dict.
    Returns empty dict if MINED_FACTOR_SPEC is empty (Stage 2 not yet run).
    """
    if not MINED_FACTOR_SPEC:
        return {}
    return {spec['name']: compute_mined_factor_value(spec, df_stock)
            for spec in MINED_FACTOR_SPEC}


def get_mined_factor_names() -> List[str]:
    """Return list of mined factor names (used by trainer + cache_updater)."""
    return [s['name'] for s in MINED_FACTOR_SPEC]
