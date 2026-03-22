#!/usr/bin/env python3
"""
V4.8.2 production scorer -- ~81 features (V4.8.1 60 + 21 new factors)

Architecture:
  Model: V4.8.2 trained model (~81 features)
  Scorer: Inherits V4.8.1/V4.7.6 post-processing

New factors (21):
  Phase 1 - Price-Volume (10):
    1.  industry_adj_str      - Industry-adjusted short-term reversal
    2.  turnover_reversal     - Turnover rate reversal (negative turnover)
    3.  max5_lottery          - MAX5 lottery effect (negative top-5 returns)
    4.  retail_crowding       - Retail crowding proxy
    5.  chaikin_mf_20d        - Chaikin Money Flow (20d)
    6.  residual_momentum     - Market-beta-adjusted residual momentum
    7.  ksft_5d               - K-bar close position (Qlib KSFT)
    8.  sumd_20d              - Gain-loss balance (Qlib SUMD)
    9.  obv_price_div         - OBV-Price divergence
    10. limit_proximity_5d    - Limit-up/down proximity

  Phase 2 - Financial Quality (6):
    11. cfp                   - Cash flow to price yield
    12. gpoa_approx           - Gross profit over assets (approx)
    13. accruals_quality      - Accruals quality
    14. delta_roe_yoy         - YoY ROE change
    15. delta_leverage_yoy    - YoY leverage change
    16. rev_growth_consistency - Revenue growth consistency

  Phase 3 - Academic (5):
    17. high_52w_ratio        - 52-week high ratio
    18. trend_rsquared_20d    - Trend R-squared
    19. imxd_20d              - IMAX-IMIN trend timing
    20. realized_skew_20d     - Realized skewness
    21. trend_strength_60d    - Trend strength t-statistic

Fallback chain: v482 model -> v481 model -> v475 model
"""

import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from .v481_production_scorer import V481ProductionScorer, V481_NEW_FACTORS

import logging
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# V4.8.2 new factor names (13 factors, dual-window IC-verified)
# Removed 8 FAIL factors: cfp, gpoa_approx, accruals_quality,
#   rev_growth_consistency, trend_rsquared_20d (both windows fail)
#   chaikin_mf_20d, ksft_5d, delta_leverage_yoy (3-year window fail)
V482_NEW_FACTORS = [
    'industry_adj_str', 'turnover_reversal', 'max5_lottery',
    'retail_crowding', 'residual_momentum',
    'sumd_20d', 'obv_price_div', 'limit_proximity_5d',
    'delta_roe_yoy',
    'high_52w_ratio', 'imxd_20d',
    'realized_skew_20d', 'trend_strength_60d',
]


class V482ProductionScorer(V481ProductionScorer):
    """V4.8.2 scorer -- ~81 features (V4.8.1 + 21 new) + V4.7.6 post-processing"""

    def __init__(self, model_type: str = 'small_data'):
        self._v482_model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v482'
        self._v482_factor_cache = {}  # date -> DataFrame
        self._fi_cache = {}           # code -> sorted quarterly DataFrame
        self._fi_loaded = False
        super().__init__(model_type=model_type)

    def _load_models(self):
        """Try v482 model first, fallback to v481"""
        v482_files = list(self._v482_model_dir.glob('v482_*.pkl'))
        if v482_files:
            self.model_dir = self._v482_model_dir
            latest = max(v482_files, key=lambda f: f.stat().st_mtime)
            self._load_model_from_file(latest, label='V4.8.2')
            return
        super()._load_models()

    def _compute_v482_new_factors(self, features_df: pd.DataFrame, date: str) -> pd.DataFrame:
        """Compute 21 new factors for V4.8.2 at inference time."""
        if date in self._v482_factor_cache:
            df_factors = self._v482_factor_cache[date]
        else:
            df_factors = self._build_v482_factors(features_df, date)
            self._v482_factor_cache[date] = df_factors

        if df_factors is not None and len(df_factors) > 0:
            features_df = features_df.merge(df_factors, on='code', how='left')

        for col in V482_NEW_FACTORS:
            if col in features_df.columns:
                features_df[col] = features_df[col].fillna(0.0)
            else:
                features_df[col] = 0.0

        return features_df

    def _load_financial_indicator_cache(self):
        """Load and cache quarterly financial data (one-time)"""
        if self._fi_loaded:
            return
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT s.code, fi.end_date, fi.roe
            FROM financial_indicator fi
            JOIN securities s ON fi.security_id = s.id
            WHERE fi.end_date >= '20180101'
            ORDER BY s.code, fi.end_date
            """
            df_fi = pd.read_sql(query, conn)
            conn.close()

            df_fi['end_date'] = df_fi['end_date'].str.replace('-', '')
            for code, grp in df_fi.groupby('code'):
                self._fi_cache[code] = grp.sort_values('end_date').reset_index(drop=True)
            self._fi_loaded = True
            logger.debug(f"Financial indicator cache loaded: {len(self._fi_cache)} stocks")
        except Exception as e:
            logger.warning(f"Failed to load financial indicators: {e}")
            self._fi_loaded = True

    def _build_v482_factors(self, features_df: pd.DataFrame, date: str) -> Optional[pd.DataFrame]:
        """Build the 21 new factors for V4.8.2"""
        try:
            conn = sqlite3.connect(self.db_path)

            # Query OHLCV: need 260 trading days for 52w high + 60d for residual momentum
            query_ohlcv = """
            SELECT s.code, q.trade_date, q.open, q.high, q.low, q.close,
                   q.volume, q.price_change_pct
            FROM daily_quotes q
            JOIN securities s ON q.security_id = s.id
            WHERE s.type = 'A股'
              AND q.trade_date <= ?
              AND q.trade_date >= date(?, '-370 days')
              AND q.volume > 0
            ORDER BY s.code, q.trade_date
            """
            df_ohlcv = pd.read_sql(query_ohlcv, conn, params=[date, date])

            # Turnover + market cap
            query_basic = """
            SELECT s.code, db.trade_date, db.turnover_rate, db.total_mv
            FROM daily_basic db
            JOIN securities s ON db.security_id = s.id
            WHERE db.trade_date <= ?
              AND db.trade_date >= date(?, '-90 days')
            """
            df_basic = pd.read_sql(query_basic, conn, params=[date, date])

            # Industry info
            if not self._industry_cache:
                query_ind = "SELECT code, industry FROM securities WHERE type = 'A股' AND industry IS NOT NULL"
                df_ind = pd.read_sql(query_ind, conn)
                self._industry_cache = dict(zip(df_ind['code'], df_ind['industry']))

            conn.close()
        except Exception as e:
            logger.warning(f"V4.8.2 data query failed: {e}")
            return None

        if len(df_ohlcv) == 0:
            return None

        # Merge turnover/mcap
        if len(df_basic) > 0:
            basic_today = df_basic[df_basic['trade_date'] == date].set_index('code')
            turnover_lookup = basic_today['turnover_rate'].to_dict() if 'turnover_rate' in basic_today.columns else {}
            mcap_lookup = basic_today['total_mv'].to_dict() if 'total_mv' in basic_today.columns else {}
        else:
            turnover_lookup = {}
            mcap_lookup = {}

        # Build per-code turnover time series
        turn_by_code = {}
        if len(df_basic) > 0:
            for code, grp in df_basic.groupby('code'):
                turn_by_code[code] = grp.sort_values('trade_date')['turnover_rate'].values.astype(float)

        # Market median return for residual momentum
        market_daily_ret = {}
        for td, grp in df_ohlcv.groupby('trade_date'):
            pcts = pd.to_numeric(grp['price_change_pct'], errors='coerce').dropna()
            if len(pcts) > 0:
                market_daily_ret[td] = float(pcts.median())

        # Industry median 5d return for industry_adj_str
        # Compute per-stock 5d returns
        code_ret5d = {}
        for code, grp in df_ohlcv.groupby('code'):
            grp = grp.sort_values('trade_date')
            c = grp['close'].values
            if len(c) >= 6:
                code_ret5d[code] = float(c[-1] / c[-6] - 1) if c[-6] > 0 else 0.0

        # Industry median
        ind_rets = {}
        for code, ret in code_ret5d.items():
            ind = self._industry_cache.get(code, '')
            if ind:
                ind_rets.setdefault(ind, []).append(ret)
        ind_med_ret5d = {ind: float(np.median(rets)) for ind, rets in ind_rets.items()}

        # Load financial data
        self._load_financial_indicator_cache()

        results_list = []

        for code, grp in df_ohlcv.groupby('code'):
            grp = grp.sort_values('trade_date')
            n = len(grp)
            if n < 25:
                continue

            c = grp['close'].values.astype(float)
            o = grp['open'].values.astype(float)
            h = grp['high'].values.astype(float)
            lo = grp['low'].values.astype(float)
            v = grp['volume'].values.astype(float)
            pct = pd.to_numeric(grp['price_change_pct'], errors='coerce').fillna(0).values.astype(float)
            dates_arr = grp['trade_date'].values
            industry = self._industry_cache.get(code, '')

            # Use last 20 days for short-term factors
            n20 = min(20, n)
            c20 = c[-n20:]
            o20 = o[-n20:]
            h20 = h[-n20:]
            lo20 = lo[-n20:]
            v20 = v[-n20:]
            pct20 = pct[-n20:]

            row = {'code': code}

            hl_range = h20 - lo20
            hl_safe = np.where(hl_range > 1e-8, hl_range, 1e-8)

            # === Phase 1: Price-Volume ===

            # 1. industry_adj_str
            stock_ret5d = code_ret5d.get(code, 0.0)
            ind_med = ind_med_ret5d.get(industry, 0.0)
            row['industry_adj_str'] = -(stock_ret5d - ind_med)

            # 2. turnover_reversal (cross-sectional rank done later)
            turns = turn_by_code.get(code)
            if turns is not None and len(turns) >= 5:
                row['_avg_turn_20d'] = float(np.mean(turns[-min(20, len(turns)):]))
            else:
                row['_avg_turn_20d'] = 0.0

            # 3. max5_lottery
            if len(pct20) >= 10:
                top5 = np.sort(pct20)[-5:]
                row['max5_lottery'] = -float(np.mean(top5))
            else:
                row['max5_lottery'] = 0.0

            # 4. retail_crowding (cross-sectional rank done later)
            row['_turnover'] = turnover_lookup.get(code, 0.0)
            row['_mcap'] = mcap_lookup.get(code, 0.0)

            # 6. residual_momentum
            if n >= 25:
                stock_r = pct[-25:]
                mkt_r = np.array([market_daily_ret.get(d, 0.0) for d in dates_arr[-25:]])
                mkt_var = np.var(mkt_r)
                if mkt_var > 1e-12:
                    beta = np.cov(stock_r, mkt_r)[0,1] / mkt_var
                else:
                    beta = 0.0
                residual = stock_r - beta * mkt_r
                row['residual_momentum'] = float(np.sum(residual[:20]))  # skip last 5
            else:
                row['residual_momentum'] = 0.0

            # 8. sumd_20d
            gains = np.sum(pct20[pct20 > 0])
            losses = np.sum(-pct20[pct20 < 0])
            denom = gains + losses
            row['sumd_20d'] = float((gains - losses) / max(denom, 1e-8))

            # 9. obv_price_div
            if n >= 20:
                obv_sign = np.where(pct > 0, 1, np.where(pct < 0, -1, 0))
                obv = np.cumsum(obv_sign * v)
                obv_now = obv[-1]
                obv_20ago = obv[-(min(20, n) + 1)] if n > 20 else obv[0]
                if abs(obv_20ago) > 1e-8:
                    obv_ret = obv_now / obv_20ago - 1
                else:
                    obv_ret = 0.0
                price_ret = float(c[-1] / c[-(min(20, n) + 1)] - 1) if n > 20 else 0.0
                row['obv_price_div'] = obv_ret - price_ret
            else:
                row['obv_price_div'] = 0.0

            # 10. limit_proximity_5d
            if code.startswith('3') or code.startswith('688'):
                limit_pct = 0.20
            elif code.startswith(('4', '8')):
                limit_pct = 0.30
            else:
                limit_pct = 0.10
            pct5 = pct[-min(5, n):]
            row['limit_proximity_5d'] = float(np.mean(np.abs(pct5) / limit_pct))

            # === Phase 2: Financial Quality (delta_roe_yoy only) ===
            fi_data = self._fi_cache.get(code)
            if fi_data is not None and len(fi_data) > 0:
                date_str = date.replace('-', '')
                valid = fi_data[fi_data['end_date'] <= date_str]
                if len(valid) > 0:
                    idx = valid.index[-1]
                    pos = fi_data.index.get_loc(idx)
                    if pos >= 4:
                        roe_now = float(fi_data.iloc[pos]['roe']) if pd.notna(fi_data.iloc[pos]['roe']) else np.nan
                        roe_prev = float(fi_data.iloc[pos-4]['roe']) if pd.notna(fi_data.iloc[pos-4]['roe']) else np.nan
                        row['delta_roe_yoy'] = roe_now - roe_prev if pd.notna(roe_now) and pd.notna(roe_prev) else 0.0
                    else:
                        row['delta_roe_yoy'] = 0.0
                else:
                    row['delta_roe_yoy'] = 0.0
            else:
                row['delta_roe_yoy'] = 0.0

            # === Phase 3: Academic ===

            # 17. high_52w_ratio
            n252 = min(252, n)
            max_252h = np.max(h[-n252:])
            row['high_52w_ratio'] = float(c[-1] / max(max_252h, 1e-8))

            # 19. imxd_20d
            if len(h20) >= 20 and len(lo20) >= 20:
                imax = np.argmax(h20) / 19.0
                imin = np.argmin(lo20) / 19.0
                row['imxd_20d'] = float(imax - imin)
            else:
                row['imxd_20d'] = 0.0

            # 20. realized_skew_20d
            if len(pct20) >= 10:
                mu = np.mean(pct20)
                sigma = np.std(pct20)
                if sigma > 1e-8:
                    row['realized_skew_20d'] = float(np.mean(((pct20 - mu) / sigma) ** 3))
                else:
                    row['realized_skew_20d'] = 0.0
            else:
                row['realized_skew_20d'] = 0.0

            # 21. trend_strength_60d
            n60 = min(60, n)
            if n60 >= 20:
                ret_60d = c[-1] / c[-n60] - 1
                vol_60d = np.std(pct[-n60:])
                row['trend_strength_60d'] = float(ret_60d / (max(vol_60d, 1e-8) * np.sqrt(n60)))
            else:
                row['trend_strength_60d'] = 0.0

            results_list.append(row)

        if not results_list:
            return None

        df_factors = pd.DataFrame(results_list)

        # Cross-sectional ranking for turnover_reversal and retail_crowding
        if '_avg_turn_20d' in df_factors.columns and len(df_factors) > 1:
            from scipy.stats import rankdata
            vals = df_factors['_avg_turn_20d'].values
            ranks = rankdata(vals, method='average') / len(vals)
            df_factors['turnover_reversal'] = 1.0 - ranks  # invert: low turnover = high factor

        if '_turnover' in df_factors.columns and '_mcap' in df_factors.columns and len(df_factors) > 1:
            from scipy.stats import rankdata
            turn_vals = df_factors['_turnover'].values
            mcap_vals = df_factors['_mcap'].values
            turn_ranks = rankdata(turn_vals, method='average') / len(turn_vals)
            mcap_ranks = rankdata(-mcap_vals, method='average') / len(mcap_vals)  # small cap = high rank
            df_factors['retail_crowding'] = 1.0 - (turn_ranks * mcap_ranks)

        # Clean up temp columns
        drop_cols = [c for c in df_factors.columns if c.startswith('_')]
        df_factors.drop(columns=drop_cols, inplace=True, errors='ignore')

        keep_cols = ['code'] + [c for c in V482_NEW_FACTORS if c in df_factors.columns]
        return df_factors[keep_cols]

    # ========== Override predict_scores to inject V4.8.2 factors ==========

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """V4.8.2 scoring pipeline — inherits V4.8.1 + injects 13 new factors.

        Hook: override _compute_v481_new_factors to chain V4.8.2 factors after V4.8.1.
        Then delegate to parent's predict_scores which handles the full pipeline.
        """
        # Monkey-patch: wrap _compute_v481_new_factors to also add V4.8.2 factors
        original_v481_compute = self._compute_v481_new_factors

        def _compute_v481_and_v482(features_df, date_arg):
            features_df = original_v481_compute(features_df, date_arg)
            features_df = self._compute_v482_new_factors(features_df, date_arg)
            return features_df

        self._compute_v481_new_factors = _compute_v481_and_v482
        try:
            results = super().predict_scores(stock_codes, date)
        finally:
            self._compute_v481_new_factors = original_v481_compute

        return results
