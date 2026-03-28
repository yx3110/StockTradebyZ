#!/usr/bin/env python3
"""
V4.8.4 production scorer -- 61 features (V4.8.1 60 + brain_roll_spread)

Architecture:
  Model: V4.8.4 trained model (61 features)
  Scorer: Inherits V4.8.1 + loads brain_roll_spread from brain_alpha_cache
  Selection: Top-K Sharpe greedy (not global IC) — only 1 factor passed

Factor:
  brain_roll_spread = sqrt(max(0, -cov(Δclose, Δclose_lag1, 20d)))
  Source: Roll (1984) implied bid-ask spread
  TopK_Sharpe: 1.397 (baseline 0.016)

Fallback chain: v484 model -> v481 model -> v475 model
"""

import json
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from .v481_production_scorer import V481ProductionScorer

import logging
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class V484ProductionScorer(V481ProductionScorer):
    """V4.8.4 scorer -- 61 features (V4.8.1 + brain_roll_spread)"""

    def __init__(self, model_type: str = 'small_data'):
        self._v484_model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v484'
        self._brain_roll_cache = {}  # date -> {code: value}
        super().__init__(model_type=model_type)

    def _load_models(self):
        """Try v484 model first, fallback to v481"""
        v484_files = list(self._v484_model_dir.glob('v484_*.pkl'))
        if v484_files:
            self.model_dir = self._v484_model_dir
            latest = max(v484_files, key=lambda f: f.stat().st_mtime)
            self._load_model_from_file(latest, label='V4.8.4')
            return
        super()._load_models()

    def _load_brain_roll_spread(self, date: str) -> Dict[str, float]:
        """Load brain_roll_spread from cache for a given date."""
        if date in self._brain_roll_cache:
            return self._brain_roll_cache[date]

        result = {}
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT code, features_json FROM brain_alpha_cache WHERE trade_date = ?",
                (date,)
            )
            for code, fj in cursor:
                try:
                    parsed = json.loads(fj)
                    result[code] = float(parsed.get('brain_roll_spread', 0))
                except Exception:
                    pass
            conn.close()
        except Exception as e:
            logger.warning(f"V4.8.4 brain_roll_spread load failed for {date}: {e}")

        self._brain_roll_cache[date] = result
        return result

    def _compute_v484_roll_spread(self, features_df: pd.DataFrame, date: str) -> pd.DataFrame:
        """Inject brain_roll_spread into features DataFrame."""
        roll_data = self._load_brain_roll_spread(date)

        if roll_data:
            features_df['brain_roll_spread'] = features_df['code'].map(roll_data).fillna(0.0)
        else:
            features_df['brain_roll_spread'] = 0.0

        return features_df

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """V4.8.4 scoring — V4.8.1 + brain_roll_spread."""
        original_v481_compute = self._compute_v481_new_factors

        def _compute_v481_and_roll(features_df, date_arg):
            features_df = original_v481_compute(features_df, date_arg)
            features_df = self._compute_v484_roll_spread(features_df, date_arg)
            return features_df

        self._compute_v481_new_factors = _compute_v481_and_roll
        try:
            results = super().predict_scores(stock_codes, date)
        finally:
            self._compute_v481_new_factors = original_v481_compute

        return results
