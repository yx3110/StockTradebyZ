#!/usr/bin/env python3
"""
V4.8.3 production scorer -- ~102 features (V4.8.2 ~81 + 29 BRAIN factors)

Architecture:
  Model: V4.8.3 trained model (~102 features)
  Scorer: Inherits V4.8.2 + loads 29 BRAIN factors from brain_alpha_cache

BRAIN factors (29):
  Phase 1 - BRAIN verified (9): intraday_intensity, high_low_ratio, close_to_high,
    vol_ratio, vol_of_vol, momentum_decay5/10, vol_price_divergence, turnover_momentum
  Phase 2 - Academic+A-share+Microstructure (20): 52w_low_bounce, ma60_reversion,
    vol_asymmetry, roll_spread, extreme_day_freq, momentum_crash_hedge, loss_aversion,
    high_resistance, hl_spread, ret_autocorr, tail_risk, vwap_momentum, up_streak_ratio,
    hurst_proxy, post_limitup_ret, vol_price_coord, price_jerk, gap_strength,
    money_flow, vol_clustering

Fallback chain: v483 model -> v482 model -> v481 model -> v475 model
"""

import json
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from .v482_production_scorer import V482ProductionScorer, V482_NEW_FACTORS

import logging
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

V483_BRAIN_FACTORS = [
    'brain_intraday_intensity', 'brain_high_low_ratio', 'brain_close_to_high',
    'brain_vol_ratio', 'brain_vol_of_vol', 'brain_momentum_decay5',
    'brain_momentum_decay10', 'brain_vol_price_divergence', 'brain_turnover_momentum',
    'brain_52w_low_bounce', 'brain_ma60_reversion', 'brain_vol_asymmetry',
    'brain_roll_spread', 'brain_extreme_day_freq', 'brain_momentum_crash_hedge',
    'brain_loss_aversion', 'brain_high_resistance', 'brain_hl_spread',
    'brain_ret_autocorr', 'brain_tail_risk', 'brain_vwap_momentum',
    'brain_up_streak_ratio', 'brain_hurst_proxy', 'brain_post_limitup_ret',
    'brain_vol_price_coord', 'brain_price_jerk', 'brain_gap_strength',
    'brain_money_flow', 'brain_vol_clustering',
]


class V483ProductionScorer(V482ProductionScorer):
    """V4.8.3 scorer -- ~102 features (V4.8.2 + 29 BRAIN) + V4.7.6 post-processing"""

    def __init__(self, model_type: str = 'small_data'):
        self._v483_model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v483'
        self._brain_cache = {}  # date -> {code: {factor: value}}
        super().__init__(model_type=model_type)

    def _load_models(self):
        """Try v483 model first, fallback to v482"""
        v483_files = list(self._v483_model_dir.glob('v483_*.pkl'))
        if v483_files:
            self.model_dir = self._v483_model_dir
            latest = max(v483_files, key=lambda f: f.stat().st_mtime)
            self._load_model_from_file(latest, label='V4.8.3')
            return
        super()._load_models()

    def _load_brain_factors(self, date: str) -> Dict[str, Dict[str, float]]:
        """Load BRAIN factors from brain_alpha_cache for a given date.

        Returns: {code: {factor_name: value, ...}, ...}
        """
        if date in self._brain_cache:
            return self._brain_cache[date]

        result = {}
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT code, features_json FROM brain_alpha_cache WHERE trade_date = ?",
                (date,)
            )
            for code, fj in cursor:
                try:
                    result[code] = json.loads(fj)
                except Exception:
                    pass
            conn.close()
        except Exception as e:
            logger.warning(f"V4.8.3 brain_alpha_cache load failed for {date}: {e}")

        self._brain_cache[date] = result
        return result

    def _compute_v483_brain_factors(self, features_df: pd.DataFrame, date: str) -> pd.DataFrame:
        """Inject 29 BRAIN factors into features DataFrame."""
        brain_data = self._load_brain_factors(date)

        if brain_data:
            rows = []
            for code in features_df['code'].values:
                bd = brain_data.get(code, {})
                row = {'code': code}
                for factor in V483_BRAIN_FACTORS:
                    row[factor] = float(bd.get(factor, 0.0))
                rows.append(row)

            brain_df = pd.DataFrame(rows)
            features_df = features_df.merge(brain_df, on='code', how='left')

        # Ensure all columns exist and fill NaN
        for col in V483_BRAIN_FACTORS:
            if col in features_df.columns:
                features_df[col] = features_df[col].fillna(0.0)
            else:
                features_df[col] = 0.0

        return features_df

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """V4.8.3 scoring pipeline — V4.8.2 + 29 BRAIN factors.

        Hook: wrap V4.8.2's monkey-patched _compute_v481_new_factors
        to also add BRAIN factors at the end of the chain.
        """
        # Save original
        original_v481_compute = self._compute_v481_new_factors

        def _compute_all_factors(features_df, date_arg):
            # V4.8.1 + V4.8.2 factors (via parent chain)
            features_df = original_v481_compute(features_df, date_arg)
            features_df = self._compute_v482_new_factors(features_df, date_arg)
            # V4.8.3 BRAIN factors
            features_df = self._compute_v483_brain_factors(features_df, date_arg)
            return features_df

        self._compute_v481_new_factors = _compute_all_factors
        try:
            # Call grandparent (V481) predict_scores which calls _compute_v481_new_factors
            results = V482ProductionScorer.predict_scores(self, stock_codes, date)
        finally:
            self._compute_v481_new_factors = original_v481_compute

        return results
