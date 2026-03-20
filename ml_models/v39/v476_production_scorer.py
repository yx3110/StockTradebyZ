#!/usr/bin/env python3
"""
V4.7.6 production scorer -- V4.7.5 full pipeline + lightweight post-processing

Design principle: PRESERVE V4.7.5's complete pipeline (V4.4 bear blend, isotonic,
regime compression, executability filters, continuous scoring, composite ranking),
then ADD lightweight post-processing for:

1. Cross-Horizon Consistency Bonus: stocks where pred_10d and pred_15d agree
   get a small rank boost. This is free (no model re-run needed).

2. Volatility Discount: high-vol stocks get slight rank_score reduction.
   Targets L3 (Sharpe, MaxDD) and L4 (consistency).

Key difference from V4.7.6-v1: Does NOT override predict_scores.
Calls super().predict_scores() to get full V4.7.5 pipeline results,
then applies lightweight adjustments to rank_score only.

Parameters are intentionally conservative to avoid killing alpha:
- consistency_bonus_weight: 0.10 (10% of rank_score range)
- vol_discount_alpha: 0.05 (5% penalty per sigma above median)
"""

import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from .v475_production_scorer import V475ProductionScorer

import logging
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Production parameters (grid-searched 2026-03-17, optimal for 10d holding)
# Tested: c=[0.05,0.10,0.15,0.20] × v=[0.03,0.05,0.08] → all equivalent
# Using lightest effective setting to minimize interference with V4.7.5 alpha
CONSISTENCY_BONUS_WEIGHT = 0.05   # rank_score boost for cross-horizon agreement
VOL_DISCOUNT_ALPHA = 0.03         # per-sigma vol penalty


class V476ProductionScorer(V475ProductionScorer):
    """V4.7.6 scorer -- V4.7.5 full pipeline + lightweight rank adjustments"""

    def __init__(self, model_type: str = 'small_data'):
        self._v476_model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v476'
        self._vol_cache = {}  # date → {code: vol}
        # Parameters (can be overridden via env vars for grid search)
        import os
        self.consistency_bonus_weight = float(os.environ.get('V476_CONSISTENCY', CONSISTENCY_BONUS_WEIGHT))
        self.vol_discount_alpha = float(os.environ.get('V476_VOL_ALPHA', VOL_DISCOUNT_ALPHA))
        super().__init__(model_type=model_type)

    def _load_models(self):
        """Use V4.7.5 model — V4.7.6 adds scorer-only innovations, no model change needed.

        V4.7.6 training produced a v476 model, but diagnostic showed Top-K weighting
        caused overfitting. So we use V4.7.5 model + V4.7.6 scorer post-processing.
        If a v476 model exists, it's ignored in favor of the better v475 model.
        """
        super()._load_models()  # V475 → loads v475 model (50 features)

    # ========== Core: preserve V4.7.5 pipeline, add post-processing ==========

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """V4.7.6: full V4.7.5 pipeline + lightweight post-processing.

        Step 1: Run complete V4.7.5 predict_scores (bear blend, isotonic,
                regime compression, executability, continuous scoring, composite ranking)
        Step 2: Apply cross-horizon consistency bonus to rank_score
        Step 3: Apply volatility discount to rank_score
        """
        # Step 1: Full V4.7.5 pipeline
        results = super().predict_scores(stock_codes, date)

        if len(results) < 2:
            return results

        # Step 2: Cross-horizon consistency bonus
        results = self._apply_consistency_bonus(results)

        # Step 3: Volatility discount
        results = self._apply_vol_discount(results, date)

        return results

    def predict_scores_from_preloaded(self, stock_codes: List[str], date: str,
                                       features_df: Optional[pd.DataFrame]) -> Dict[str, Dict]:
        """V4.7.6 batch scoring with preloaded features"""
        results = super().predict_scores_from_preloaded(stock_codes, date, features_df)

        if len(results) < 2:
            return results

        results = self._apply_consistency_bonus(results)
        results = self._apply_vol_discount(results, date)

        return results

    # ========== Innovation 1: Cross-Horizon Consistency Bonus ==========

    def _apply_consistency_bonus(self, results: Dict[str, Dict]) -> Dict[str, Dict]:
        """Add small rank boost for stocks where 10d and 15d predictions agree.

        If both pred_10d and pred_15d are positive (or both negative),
        AND their relative ranking is similar, give a small boost.

        This is FREE — no model re-run needed, uses existing predictions.
        """
        if self.consistency_bonus_weight <= 0:
            return results

        codes = [c for c in results if results[c].get('rank_score', 0) > 0]
        if len(codes) < 5:
            return results

        # Compute cross-horizon agreement score
        pred_10d = np.array([results[c].get('pred_10d', 0) for c in codes])
        pred_15d = np.array([results[c].get('pred_15d', 0) for c in codes])

        # Agreement: both positive or both negative
        same_sign = (pred_10d * pred_15d > 0).astype(float)

        # Rank correlation within cross-section
        from scipy.stats import rankdata
        rank_10d = rankdata(pred_10d) / len(pred_10d)
        rank_15d = rankdata(pred_15d) / len(pred_15d)
        rank_diff = np.abs(rank_10d - rank_15d)

        # Consistency score: high when same sign AND similar ranking
        # Range [0, 1]: 1 = perfect agreement, 0 = disagreement
        consistency = same_sign * (1 - rank_diff)

        # Normalize to [0, 1]
        c_min, c_max = consistency.min(), consistency.max()
        if c_max - c_min > 1e-8:
            consistency_norm = (consistency - c_min) / (c_max - c_min)
        else:
            consistency_norm = np.zeros_like(consistency)

        # Apply as small additive bonus to rank_score
        rank_scores = np.array([results[c].get('rank_score', 0) for c in codes])
        rank_range = rank_scores.max() - rank_scores.min()
        if rank_range < 1e-10:
            return results

        bonus = self.consistency_bonus_weight * rank_range * consistency_norm

        for i, code in enumerate(codes):
            results[code]['rank_score'] += bonus[i]
            results[code]['consistency_bonus'] = float(bonus[i])

        return results

    # ========== Innovation 2: Volatility Discount ==========

    def _apply_vol_discount(self, results: Dict[str, Dict], date: str) -> Dict[str, Dict]:
        """Apply small rank_score discount for high-volatility stocks.

        Uses 20-day realized volatility. Only penalizes above-median vol stocks.
        Very conservative: ~5% rank_score reduction per sigma above median.
        """
        if self.vol_discount_alpha <= 0:
            return results

        codes = [c for c in results if results[c].get('rank_score', 0) > 0]
        if len(codes) < 5:
            return results

        # Get realized vol for these stocks
        raw_vols = self._get_realized_vols(codes, date)
        if raw_vols is None:
            return results

        # Cross-sectional normalization using MAD
        valid_mask = ~np.isnan(raw_vols)
        if valid_mask.sum() < 5:
            return results

        median_vol = np.nanmedian(raw_vols)
        mad_vol = np.nanmedian(np.abs(raw_vols[valid_mask] - median_vol)) * 1.4826
        if mad_vol < 1e-10:
            return results

        # Apply discount only to above-median vol stocks
        for i, code in enumerate(codes):
            if np.isnan(raw_vols[i]):
                continue

            norm_vol = (raw_vols[i] - median_vol) / mad_vol
            if norm_vol > 0:
                # Discount: reduce rank_score by alpha * norm_vol * rank_score
                discount = self.vol_discount_alpha * min(norm_vol, 3.0)
                old_rank = results[code]['rank_score']
                results[code]['rank_score'] *= (1 - discount)
                results[code]['vol_discount'] = float(discount)

        return results

    def _get_realized_vols(self, codes: List[str], date: str) -> Optional[np.ndarray]:
        """Compute 20-day realized volatility for each stock (with date-level cache)."""
        n = len(codes)
        vols = np.full(n, np.nan)

        # Check cache
        if date in self._vol_cache:
            cached = self._vol_cache[date]
            for i, code in enumerate(codes):
                if code in cached:
                    vols[i] = cached[code]
            if not np.all(np.isnan(vols)):
                return vols

        try:
            conn = sqlite3.connect(self.db_path)
            # Batch query all A-stocks for this date (cache-friendly)
            query = """
            SELECT s.code, q.close, q.trade_date
            FROM daily_quotes q
            JOIN securities s ON q.security_id = s.id
            WHERE s.type = 'A股'
              AND q.trade_date <= ?
              AND q.trade_date >= date(?, '-35 days')
            ORDER BY s.code, q.trade_date
            """
            df = pd.read_sql_query(query, conn, params=[date, date])
            conn.close()
        except Exception:
            return None

        if len(df) == 0:
            return None

        # Compute vols for ALL stocks and cache
        date_cache = {}
        for code, grp in df.groupby('code'):
            closes = grp.sort_values('trade_date')['close'].values.astype(float)
            if len(closes) < 6:
                continue
            recent = closes[-21:] if len(closes) >= 21 else closes
            daily_ret = np.diff(recent) / recent[:-1]
            date_cache[code] = float(np.std(daily_ret))

        self._vol_cache[date] = date_cache

        # Fill requested codes
        for i, code in enumerate(codes):
            if code in date_cache:
                vols[i] = date_cache[code]

        return vols
