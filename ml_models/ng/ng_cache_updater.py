"""
Daily Selection NG v1.0.3 — Batch Feature Cache Updater
========================================================
v1.0.3 changes from v1.0.0:
  - Labels are now INDUSTRY EXCESS returns (stock_return - industry_median_return)
  - Cross-sectional rank factors computed per-industry
  - Residual factors (market/industry-neutralized)
  - 3 sector activity features
  - 11 low-efficiency factors removed

Usage:
    python3 ml_models/ng/ng_cache_updater.py --date 2025-06-13
    python3 ml_models/ng/ng_cache_updater.py --start-date 2025-01-01 --end-date 2025-06-30
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ml_models.ng.ng_schema import (
    create_table, get_table_name, DB_PATH, create_moneyflow_table,
    version_ge, get_schema_version, DEFAULT_VERSION,
    _is_1_2_branch, _is_1_3_branch, _is_1_5_branch, _version_in_range,
)
from ml_models.ng.ng130_downside_label import compute_all_downside_horizons
from ml_models.ng.ng130_moneyflow_factors import compute_ng130_mf_factors, NG130_MF_FACTORS
from ml_models.ng.ng_feature_calculator import (
    compute_stock_features,
    compute_fundamental_features,
    compute_market_features,
    compute_industry_features,
    compute_cross_sectional_rank_features,
    compute_residual_features,
    compute_moneyflow_features,
    compute_interaction_features,
    compute_smoothing_features,
    compute_extended_market_features,
    compute_conditional_interaction_features,
    compute_ng150_regime_stock_features,
    compute_ng150_regime_market_features,
    filter_ng123_features,
    get_ng123_drop_features,
)
from ml_models.ng.ng123_moneyflow_factors import (
    compute_all_moneyflow_factors,
    compute_stock_mf_scalars,
    compute_group_d_factors,
)
from ml_models.ng.ng123_label_transform import compute_path_min_kd, compute_downside_kd
from ml_models.ng.ng123_mined_factors import (
    MINED_FACTOR_SPEC,
    compute_mined_factor_value,
)
from scripts.factor_mining_pipeline import generate_operands, compute_factor as _compute_mined_factor
from sklearn.linear_model import LinearRegression
from fetch_data.label_utils import (
    compute_labels_from_future_prices,
    compute_vn_labels_from_future_prices,
)


def compute_maxdd_from_future_prices(
    base_open: float,
    future_closes: Dict[int, float],
    horizons: tuple = (3, 5, 10, 15),
) -> Dict[str, float]:
    """Compute max drawdown for each horizon from future close prices.
    MaxDD = min over t in [0..N] of (close_t / peak_so_far - 1)
    Returns dict like {'maxdd_3d': -0.05, ...}. Values in [-1, 0].
    """
    result = {}
    for h in horizons:
        prices = []
        for t in range(0, h + 1):
            if t in future_closes and not np.isnan(future_closes[t]):
                prices.append(future_closes[t])
        if not prices:
            result[f'maxdd_{h}d'] = np.nan
            continue
        peak = prices[0]
        max_dd = 0.0
        for p in prices:
            if p > peak:
                peak = p
            dd = p / (peak + 1e-8) - 1.0
            if dd < max_dd:
                max_dd = dd
        result[f'maxdd_{h}d'] = float(max_dd)
    return result


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOOKBACK_DAYS = 120
TECH_LOOKBACK_DAYS = 10
PE_HISTORY_DAYS = 60
FIN_LOOKBACK_DAYS = 400
CIRC_MV_MIN = 500000      # 50亿元 = 500000万元
MIN_DATA_DAYS = 60
BENCHMARK_CODE = '000300.SH'
LABEL_HORIZONS = [3, 5, 10, 15]


# ---------------------------------------------------------------------------
# JSON / float helpers
# ---------------------------------------------------------------------------

def _json_default(obj):
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


def _safe_float(val, default=np.nan) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _to_sql(v):
    """Convert a Python/NumPy scalar to a SQLite-safe value. NaN/Inf → NULL."""
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        if np.isnan(v) or np.isinf(v):
            return None
        return float(v)
    return v


# ---------------------------------------------------------------------------
# NGCacheUpdater
# ---------------------------------------------------------------------------

class NGCacheUpdater:
    """Batch compute NG factors and write to version-specific cache table."""

    def __init__(self, db_path: str = None, version: str = DEFAULT_VERSION):
        self.db_path = db_path or DB_PATH
        self.version = version
        self.schema_version = get_schema_version(version)
        self.table_name = get_table_name(version)
        create_table(self.db_path, version=self.schema_version)
        if version_ge(self.schema_version, 'ng1.0.3'):
            create_moneyflow_table(self.db_path)
        self._pro = None  # lazy-init Tushare API

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute('PRAGMA busy_timeout = 30000')
        conn.execute('PRAGMA journal_mode = WAL')
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Trading date utilities
    # ------------------------------------------------------------------

    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        conn = self._connect()
        try:
            rows = conn.execute(
                '''SELECT DISTINCT trade_date FROM daily_quotes
                   WHERE trade_date BETWEEN ? AND ?
                   ORDER BY trade_date''',
                (start_date, end_date)
            ).fetchall()
            return [r['trade_date'] for r in rows]
        finally:
            conn.close()

    def _get_future_dates(self, conn: sqlite3.Connection, date: str, n: int) -> List[str]:
        rows = conn.execute(
            '''SELECT DISTINCT trade_date FROM daily_quotes
               WHERE trade_date > ?
               ORDER BY trade_date LIMIT ?''',
            (date, n)
        ).fetchall()
        return [r['trade_date'] for r in rows]

    # ------------------------------------------------------------------
    # Bulk data loaders (unchanged from v1.0.0)
    # ------------------------------------------------------------------

    def _load_stock_universe(self, conn: sqlite3.Connection) -> Dict[int, dict]:
        rows = conn.execute(
            '''SELECT id, code, name, industry FROM securities WHERE type = 'A股' '''
        ).fetchall()
        return {r['id']: dict(r) for r in rows}

    def _load_price_data(
        self, conn: sqlite3.Connection, date: str, security_ids: List[int]
    ) -> Dict[int, List[dict]]:
        from datetime import datetime, timedelta
        dt = datetime.strptime(date, '%Y-%m-%d')
        lookback_start = (dt - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')

        rows = conn.execute(
            f'''SELECT security_id, trade_date, open, high, low, close,
                       volume, amount, ma5, ma10, ma20, ma60,
                       price_change_pct, is_limit_up, is_limit_down, is_st
                FROM daily_quotes
                WHERE trade_date BETWEEN ? AND ?
                  AND security_id IN ({','.join('?' * len(security_ids))})
                ORDER BY security_id, trade_date''',
            [lookback_start, date] + security_ids
        ).fetchall()

        result: Dict[int, List[dict]] = defaultdict(list)
        for r in rows:
            result[r['security_id']].append(dict(r))
        return dict(result)

    def _load_tech_indicators(
        self, conn: sqlite3.Connection, date: str, security_ids: List[int]
    ) -> Dict[int, List[dict]]:
        from datetime import datetime, timedelta
        dt = datetime.strptime(date, '%Y-%m-%d')
        lookback_start = (dt - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')

        rows = conn.execute(
            f'''SELECT security_id, trade_date, kdj_j, macd_macd, rsi12, rsi24,
                       boll_upper, boll_lower, atr_14
                FROM technical_indicators
                WHERE trade_date BETWEEN ? AND ?
                  AND security_id IN ({','.join('?' * len(security_ids))})
                ORDER BY security_id, trade_date''',
            [lookback_start, date] + security_ids
        ).fetchall()

        result: Dict[int, List[dict]] = defaultdict(list)
        for r in rows:
            result[r['security_id']].append(dict(r))
        return dict(result)

    def _load_daily_basic(
        self, conn: sqlite3.Connection, date: str, security_ids: List[int]
    ) -> Dict[int, dict]:
        rows = conn.execute(
            f'''SELECT security_id, pe_ttm, pb, turnover_rate, circ_mv,
                       dv_ratio, dv_ttm, total_share, float_share, free_share
                FROM daily_basic
                WHERE trade_date = ?
                  AND security_id IN ({','.join('?' * len(security_ids))})''',
            [date] + security_ids
        ).fetchall()
        return {r['security_id']: dict(r) for r in rows}

    def _load_pe_history(
        self, conn: sqlite3.Connection, date: str, security_ids: List[int]
    ) -> Dict[int, np.ndarray]:
        from datetime import datetime, timedelta
        dt = datetime.strptime(date, '%Y-%m-%d')
        lookback_start = (dt - timedelta(days=120)).strftime('%Y-%m-%d')

        rows = conn.execute(
            f'''SELECT security_id, trade_date, pe_ttm
                FROM daily_basic
                WHERE trade_date BETWEEN ? AND ?
                  AND security_id IN ({','.join('?' * len(security_ids))})
                ORDER BY security_id, trade_date''',
            [lookback_start, date] + security_ids
        ).fetchall()

        result: Dict[int, list] = defaultdict(list)
        for r in rows:
            val = r['pe_ttm']
            if val is not None:
                result[r['security_id']].append(float(val))

        return {sid: np.array(vals[-PE_HISTORY_DAYS:]) for sid, vals in result.items()}

    def _load_turnover_history(
        self, conn: sqlite3.Connection, date: str, security_ids: List[int]
    ) -> Dict[int, List[float]]:
        """Load per-date turnover_rate history (up to LOOKBACK_DAYS) for ng1.2.3 mined factors.

        Returns {security_id: [turnover_rate, ...]} ordered oldest→newest.
        Missing dates are NOT interpolated — caller should align by position with price rows.
        """
        from datetime import datetime, timedelta
        dt = datetime.strptime(date, '%Y-%m-%d')
        lookback_start = (dt - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')

        rows = conn.execute(
            f'''SELECT security_id, trade_date, turnover_rate
                FROM daily_basic
                WHERE trade_date BETWEEN ? AND ?
                  AND security_id IN ({','.join('?' * len(security_ids))})
                ORDER BY security_id, trade_date''',
            [lookback_start, date] + security_ids
        ).fetchall()

        result: Dict[int, List] = defaultdict(list)
        for r in rows:
            result[r['security_id']].append(
                (r['trade_date'], r['turnover_rate'])
            )
        return dict(result)

    def _load_financial_data(
        self, conn: sqlite3.Connection, date: str, security_ids: List[int]
    ) -> Dict[int, dict]:
        from datetime import datetime, timedelta
        dt = datetime.strptime(date, '%Y-%m-%d')
        lookback_start_dt = dt - timedelta(days=FIN_LOOKBACK_DAYS)
        date_int = int(date.replace('-', ''))
        lookback_start_int = int(lookback_start_dt.strftime('%Y%m%d'))

        rows = conn.execute(
            f'''SELECT security_id, ann_date, end_date, roe, profit_to_gr, or_yoy,
                       netprofit_margin, ocf_to_profit, debt_to_assets, current_ratio
                FROM financial_indicator
                WHERE ann_date BETWEEN ? AND ?
                  AND security_id IN ({','.join('?' * len(security_ids))})
                ORDER BY security_id, end_date DESC''',
            [lookback_start_int, date_int] + security_ids
        ).fetchall()

        grouped: Dict[int, List[dict]] = defaultdict(list)
        for r in rows:
            ann = r['ann_date']
            if ann is not None and int(ann) <= date_int:
                grouped[r['security_id']].append(dict(r))

        result = {}
        for sid, filings in grouped.items():
            if not filings:
                continue
            latest = filings[0]
            roe_prev = np.nan
            latest_end = str(latest['end_date'])
            latest_year = latest_end[:4] if latest_end else None
            for f in filings[1:]:
                f_end = str(f['end_date'])
                f_year = f_end[:4] if f_end else None
                if f_year and latest_year and f_year < latest_year:
                    roe_prev = _safe_float(f['roe'])
                    break

            result[sid] = {
                'roe': _safe_float(latest['roe']),
                'roe_prev_year': roe_prev,
                'profit_to_gr': _safe_float(latest['profit_to_gr']),
                'or_yoy': _safe_float(latest['or_yoy']),  # 真正的营收同比增长率
                'netprofit_margin': _safe_float(latest['netprofit_margin']),
                'ocf_to_profit': _safe_float(latest['ocf_to_profit']),
                'debt_to_assets': _safe_float(latest['debt_to_assets']),
                'current_ratio': _safe_float(latest['current_ratio']),
            }
        return result

    def _load_benchmark_data(
        self, conn: sqlite3.Connection, date: str
    ) -> np.ndarray:
        from datetime import datetime, timedelta
        dt = datetime.strptime(date, '%Y-%m-%d')
        lookback_start = (dt - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')

        rows = conn.execute(
            '''SELECT dq.close FROM daily_quotes dq
               JOIN securities s ON dq.security_id = s.id
               WHERE s.code = ? AND dq.trade_date BETWEEN ? AND ?
               ORDER BY dq.trade_date''',
            (BENCHMARK_CODE, lookback_start, date)
        ).fetchall()
        if not rows:
            return np.array([])
        return np.array([float(r['close']) for r in rows])

    def _load_benchmark_daily_returns(
        self, conn: sqlite3.Connection, date: str, n_days: int = 25
    ) -> np.ndarray:
        """Load daily log returns for CSI300 benchmark (for residual computation)."""
        from datetime import datetime, timedelta
        dt = datetime.strptime(date, '%Y-%m-%d')
        lookback_start = (dt - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')

        rows = conn.execute(
            '''SELECT dq.close FROM daily_quotes dq
               JOIN securities s ON dq.security_id = s.id
               WHERE s.code = ? AND dq.trade_date BETWEEN ? AND ?
               ORDER BY dq.trade_date''',
            (BENCHMARK_CODE, lookback_start, date)
        ).fetchall()
        if len(rows) < 2:
            return np.array([])
        closes = np.array([float(r['close']) for r in rows])
        log_rets = np.diff(np.log(closes + 1e-8))
        return log_rets[-n_days:] if len(log_rets) >= n_days else log_rets

    def _load_industry_5d_ret_history(
        self, conn: sqlite3.Connection, date: str, n_calendar_days: int = 95
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """ng1.5.0: Load per-industry mean 1d return time series (from closes),
        compute 5d rolling cumulative return per industry.

        Returns:
            industry_5d_hist: {industry_name: np.ndarray of 5d rets, last = latest date}
            market_5d_hist: np.ndarray of benchmark 5d rets aligned with industry series

        Used by ng150 `industry_regime_agreement` feature (60d sign agreement vs market).
        Computes 1d returns from close prices via LAG() (daily_quotes.price_change_pct
        is mostly NULL as of 2026-04). Single grouped query for all industries.
        """
        from datetime import datetime, timedelta
        dt = datetime.strptime(date, '%Y-%m-%d')
        hist_start = (dt - timedelta(days=n_calendar_days)).strftime('%Y-%m-%d')

        rows = conn.execute(
            '''WITH base AS (
                   SELECT dq.security_id, s.industry, dq.trade_date, dq.close,
                          LAG(dq.close) OVER (
                              PARTITION BY dq.security_id ORDER BY dq.trade_date
                          ) AS prev_close
                   FROM daily_quotes dq
                   JOIN securities s ON dq.security_id = s.id
                   WHERE s.type = 'A股' AND s.industry IS NOT NULL
                     AND dq.trade_date BETWEEN ? AND ?
               )
               SELECT trade_date, industry,
                      AVG((close - prev_close) / prev_close) AS avg_1d
               FROM base
               WHERE prev_close IS NOT NULL AND prev_close > 0
               GROUP BY trade_date, industry
               ORDER BY industry, trade_date''',
            (hist_start, date)
        ).fetchall()

        ind_by_date: Dict[str, Dict[str, float]] = defaultdict(dict)
        for r in rows:
            v = _safe_float(r['avg_1d'])
            if not np.isnan(v):
                ind_by_date[r['industry']][r['trade_date']] = v
        if not ind_by_date:
            return {}, np.array([])

        # Benchmark close trajectory for market 5d returns (aligned date index)
        bm_rows = conn.execute(
            '''SELECT dq.trade_date, dq.close FROM daily_quotes dq
               JOIN securities s ON dq.security_id = s.id
               WHERE s.code = ? AND dq.trade_date BETWEEN ? AND ?
               ORDER BY dq.trade_date''',
            (BENCHMARK_CODE, hist_start, date)
        ).fetchall()
        if len(bm_rows) < 10:
            return {}, np.array([])
        bm_dates = [r['trade_date'] for r in bm_rows]
        bm_closes = np.array([float(r['close']) for r in bm_rows])
        n = len(bm_closes)
        bm_5d = np.full(n, np.nan)
        for i in range(4, n):
            bm_5d[i] = bm_closes[i] / (bm_closes[i - 4] + 1e-8) - 1.0

        industry_5d_hist: Dict[str, np.ndarray] = {}
        for ind, dmap in ind_by_date.items():
            rets_1d = np.array([dmap.get(d, np.nan) for d in bm_dates])
            rets_5d = np.full(n, np.nan)
            for i in range(4, n):
                window = rets_1d[i - 4:i + 1]
                if np.all(np.isfinite(window)):
                    rets_5d[i] = float(np.prod(1.0 + window) - 1.0)
            industry_5d_hist[ind] = rets_5d
        return industry_5d_hist, bm_5d

    def _load_amv_var1_ma60(
        self, conn: sqlite3.Connection, date: str
    ) -> float:
        """ng1.5.0: 60-day MA of amv_var1 for amv_regime_bull_prob feature."""
        rows = conn.execute(
            '''SELECT var1 FROM market_amv
               WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT 60''',
            (date,)
        ).fetchall()
        if not rows:
            return np.nan
        vals = [_safe_float(r['var1']) for r in rows if _safe_float(r['var1']) is not None
                and not np.isnan(_safe_float(r['var1']))]
        if len(vals) < 10:
            return np.nan
        return float(np.mean(vals))

    def _load_northbound_data(
        self, conn: sqlite3.Connection, date: str
    ) -> Tuple[float, float]:
        from datetime import datetime, timedelta
        dt = datetime.strptime(date, '%Y-%m-%d')
        lookback_start = (dt - timedelta(days=60)).strftime('%Y-%m-%d')

        try:
            rows = conn.execute(
                '''SELECT trade_date, north_money FROM hsgt_daily
                   WHERE trade_date BETWEEN ? AND ?
                   ORDER BY trade_date''',
                (lookback_start, date)
            ).fetchall()
        except sqlite3.OperationalError:
            return 0.0, 1.0

        if not rows:
            return 0.0, 1.0

        vals = [_safe_float(r['north_money'], 0.0) for r in rows]
        arr = np.array(vals)
        net_buy_5d = float(np.sum(arr[-5:])) if len(arr) >= 5 else float(np.sum(arr))
        nb_std = float(arr.std()) if len(arr) >= 5 else 1.0
        if nb_std < 1e-8:
            nb_std = 1.0
        return net_buy_5d, nb_std

    def _load_market_amounts(
        self, conn: sqlite3.Connection, date: str
    ) -> np.ndarray:
        from datetime import datetime, timedelta
        dt = datetime.strptime(date, '%Y-%m-%d')
        lookback_start = (dt - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')

        rows = conn.execute(
            '''SELECT dq.trade_date, SUM(COALESCE(dq.amount, dq.volume)) as total_amount
               FROM daily_quotes dq
               JOIN securities s ON dq.security_id = s.id
               WHERE s.type = 'A股'
                 AND dq.trade_date BETWEEN ? AND ?
               GROUP BY dq.trade_date
               ORDER BY dq.trade_date''',
            (lookback_start, date)
        ).fetchall()
        if not rows:
            return np.array([])
        return np.array([float(r['total_amount'] or 0) for r in rows])

    # ------------------------------------------------------------------
    # Moneyflow data (v1.0.3)
    # ------------------------------------------------------------------

    def _fetch_and_store_moneyflow(self, conn: sqlite3.Connection, date: str) -> int:
        """Fetch moneyflow data from Tushare and store in moneyflow_daily table.
        Returns row count inserted."""
        date_str = date.replace('-', '')

        # Check if data already exists for this date
        existing = conn.execute(
            "SELECT COUNT(*) as cnt FROM moneyflow_daily WHERE trade_date = ?",
            (date,)
        ).fetchone()
        if existing and existing['cnt'] > 0:
            return 0

        # Lazy-init Tushare API (cached across calls)
        try:
            if self._pro is None:
                config_path = os.path.join(_PROJECT_ROOT, 'config.json')
                with open(config_path) as f:
                    cfg = json.load(f)
                import tushare as ts
                self._pro = ts.pro_api(cfg['tushare']['token'])
            df_mf = self._pro.moneyflow(trade_date=date_str)
        except Exception as e:
            print(f"    WARN: moneyflow fetch failed: {e}")
            return 0

        if df_mf is None or len(df_mf) == 0:
            return 0

        rows = []
        for _, row in df_mf.iterrows():
            rows.append((
                row['ts_code'], date,
                float(row.get('buy_sm_amount') or 0),
                float(row.get('sell_sm_amount') or 0),
                float(row.get('buy_md_amount') or 0),
                float(row.get('sell_md_amount') or 0),
                float(row.get('buy_lg_amount') or 0),
                float(row.get('sell_lg_amount') or 0),
                float(row.get('buy_elg_amount') or 0),
                float(row.get('sell_elg_amount') or 0),
                float(row.get('net_mf_amount') or 0),
            ))

        conn.executemany(
            """INSERT OR REPLACE INTO moneyflow_daily
               (code, trade_date, buy_sm_amount, sell_sm_amount,
                buy_md_amount, sell_md_amount, buy_lg_amount, sell_lg_amount,
                buy_elg_amount, sell_elg_amount, net_mf_amount)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            rows
        )
        conn.commit()
        return len(rows)

    def _load_moneyflow_data(
        self, conn: sqlite3.Connection, date: str,
        security_ids: List[int], universe: Dict[int, dict] = None, n_days: int = 20
    ) -> Dict[str, list]:
        """Load moneyflow data for given stocks up to n_days before date.
        Returns Dict[code -> List[dict]] sorted by date ascending."""
        from datetime import datetime, timedelta
        dt = datetime.strptime(date, '%Y-%m-%d')
        lookback_start = (dt - timedelta(days=n_days * 2)).strftime('%Y-%m-%d')

        # Use universe to get codes directly (avoid re-querying securities table)
        chunk_size = 900
        if universe:
            codes = [universe[sid]['code'] for sid in security_ids if sid in universe]
        else:
            sid_to_code: Dict[int, str] = {}
            for i in range(0, len(security_ids), chunk_size):
                chunk = security_ids[i:i + chunk_size]
                rows = conn.execute(
                    f"SELECT id, code FROM securities WHERE id IN ({','.join('?' * len(chunk))})",
                    chunk
                ).fetchall()
                for r in rows:
                    sid_to_code[r['id']] = r['code']
            codes = list(sid_to_code.values())
        if not codes:
            return {}

        # Query moneyflow_daily in chunks.
        # NOTE: securities.code is 6-digit ('000001') but moneyflow_daily.code has
        # exchange suffix ('000001.SZ'). JOIN on code_6 column (added 2026-04-18)
        # to avoid silent empty result.
        result: Dict[str, list] = defaultdict(list)
        for i in range(0, len(codes), chunk_size):
            chunk = codes[i:i + chunk_size]
            rows = conn.execute(
                f"""SELECT code_6 AS code, trade_date, buy_sm_amount, sell_sm_amount,
                           buy_md_amount, sell_md_amount, buy_lg_amount, sell_lg_amount,
                           buy_elg_amount, sell_elg_amount, net_mf_amount
                    FROM moneyflow_daily
                    WHERE trade_date BETWEEN ? AND ?
                      AND code_6 IN ({','.join('?' * len(chunk))})
                    ORDER BY code_6, trade_date""",
                [lookback_start, date] + chunk
            ).fetchall()
            for r in rows:
                result[r['code']].append(dict(r))

        # Trim to latest n_days per code
        for code in result:
            result[code] = result[code][-n_days:]

        return dict(result)

    def _load_future_prices(
        self, conn: sqlite3.Connection, future_dates: List[str],
        security_ids: List[int]
    ) -> Dict[int, Dict[str, dict]]:
        if not future_dates:
            return {}

        rows = conn.execute(
            f'''SELECT security_id, trade_date, open, close
                FROM daily_quotes
                WHERE trade_date IN ({','.join('?' * len(future_dates))})
                  AND security_id IN ({','.join('?' * len(security_ids))})''',
            future_dates + security_ids
        ).fetchall()

        result: Dict[int, Dict[str, dict]] = defaultdict(dict)
        for r in rows:
            result[r['security_id']][r['trade_date']] = {
                'open': _safe_float(r['open']),
                'close': _safe_float(r['close']),
            }
        return dict(result)

    # ------------------------------------------------------------------
    # Compute aggregates
    # ------------------------------------------------------------------

    def _compute_stock_returns(
        self, price_data: Dict[int, List[dict]], n_days: int
    ) -> Dict[int, float]:
        result = {}
        for sid, rows in price_data.items():
            if len(rows) < n_days + 1:
                continue
            c_now = _safe_float(rows[-1]['close'])
            c_prev = _safe_float(rows[-(n_days + 1)]['close'])
            if c_prev > 1e-8:
                result[sid] = c_now / c_prev - 1.0
            else:
                result[sid] = np.nan
        return result

    def _compute_1d_returns(self, price_data: Dict[int, List[dict]]) -> Dict[int, float]:
        return self._compute_stock_returns(price_data, 1)

    def _compute_highs_20d_ratio(self, price_data: Dict[int, List[dict]]) -> Dict[int, float]:
        result = {}
        for sid, rows in price_data.items():
            if len(rows) < 20:
                continue
            close = _safe_float(rows[-1]['close'])
            highs_20 = [_safe_float(r['high']) for r in rows[-20:]]
            max_high = max(highs_20) if highs_20 else 0
            if max_high > 1e-8:
                result[sid] = close / max_high
        return result

    def _compute_stock_daily_returns(
        self, price_data: Dict[int, List[dict]], n_days: int = 20
    ) -> Dict[int, np.ndarray]:
        """Compute daily log returns for each stock (for residual factors)."""
        result = {}
        for sid, rows in price_data.items():
            if len(rows) < n_days + 1:
                continue
            closes = np.array([_safe_float(r['close']) for r in rows])
            log_rets = np.diff(np.log(closes + 1e-8))
            result[sid] = log_rets[-n_days:] if len(log_rets) >= n_days else log_rets
        return result

    def _compute_stock_volatilities(
        self, price_data: Dict[int, List[dict]], n_days: int = 20
    ) -> Dict[int, float]:
        """Compute 20d return volatility (annualized) for each stock."""
        result = {}
        for sid, rows in price_data.items():
            if len(rows) < n_days + 1:
                continue
            closes = np.array([_safe_float(r['close']) for r in rows[-(n_days+1):]])
            log_rets = np.diff(np.log(closes + 1e-8))
            result[sid] = float(np.std(log_rets)) * np.sqrt(252)
        return result

    def _compute_industry_aggregates(
        self,
        universe: Dict[int, dict],
        price_data: Dict[int, List[dict]],
        returns_1d: Dict[int, float],
        returns_5d: Dict[int, float],
        returns_20d: Dict[int, float],
    ) -> Dict[str, dict]:
        """Group stocks by industry and compute industry-level aggregates."""
        industry_sids: Dict[str, List[int]] = defaultdict(list)
        for sid, info in universe.items():
            ind = info.get('industry') or 'unknown'
            industry_sids[ind].append(sid)

        result = {}
        for ind, sids in industry_sids.items():
            ind_ret1d = [returns_1d[s] for s in sids if s in returns_1d and not np.isnan(returns_1d[s])]
            ind_ret5d = [returns_5d[s] for s in sids if s in returns_5d and not np.isnan(returns_5d[s])]
            ind_ret20d = [returns_20d[s] for s in sids if s in returns_20d and not np.isnan(returns_20d[s])]

            amounts_5d = []
            amounts_20d = []
            for sid in sids:
                rows = price_data.get(sid, [])
                if len(rows) >= 5:
                    daily_amts = [_safe_float(r['amount'] if r['amount'] is not None else r['volume'], 0) for r in rows[-5:]]
                    amounts_5d.append(np.mean(daily_amts))
                if len(rows) >= 20:
                    daily_amts = [_safe_float(r['amount'] if r['amount'] is not None else r['volume'], 0) for r in rows[-20:]]
                    amounts_20d.append(np.mean(daily_amts))

            result[ind] = {
                'sids': sids,
                'returns_1d': np.array(ind_ret1d) if ind_ret1d else np.array([]),
                'returns_5d': np.array(ind_ret5d) if ind_ret5d else np.array([]),
                'returns_20d': np.array(ind_ret20d) if ind_ret20d else np.array([]),
                'amounts_5d': np.array(amounts_5d) if amounts_5d else np.array([]),
                'amounts_20d': np.array(amounts_20d) if amounts_20d else np.array([]),
                'mean_return_5d': float(np.mean(ind_ret5d)) if ind_ret5d else np.nan,
                'mean_return_20d': float(np.mean(ind_ret20d)) if ind_ret20d else np.nan,
                # v1.0.3: industry average volume (for residual_volume)
                'avg_volume_5d': float(np.mean(amounts_5d)) if amounts_5d else np.nan,
            }

        return result

    def _compute_labels(
        self, future_dates: List[str], future_prices: Dict[int, Dict[str, dict]],
        security_ids: List[int]
    ) -> Dict[int, Dict[str, float]]:
        """Compute ABSOLUTE forward-looking labels (before excess conversion)."""
        if not future_dates:
            return {}

        schema_ver = self.schema_version
        want_vn = (schema_ver == 'ng1.2.1')
        max_h = max(LABEL_HORIZONS)

        result = {}
        for sid in security_ids:
            fp = future_prices.get(sid, {})
            if not fp or future_dates[0] not in fp:
                continue

            base = fp[future_dates[0]].get('open', np.nan)
            if np.isnan(base) or base < 1e-8:
                continue

            # ng1.2.1 需要 1..max_h 连续每日 close 走 Sharpe path;
            # 其他分支只需要 horizon 对应的 close.
            if want_vn:
                future_closes = {}
                for n in range(1, max_h + 1):
                    if n < len(future_dates) and future_dates[n] in fp:
                        future_closes[n] = fp[future_dates[n]].get('close', np.nan)
            else:
                future_closes = {}
                for n in LABEL_HORIZONS:
                    if n < len(future_dates) and future_dates[n] in fp:
                        future_closes[n] = fp[future_dates[n]].get('close', np.nan)

            # compute_labels_from_future_prices only reads keys in LABEL_HORIZONS;
            # passing the dense dict for vn case is fine (extra keys ignored).
            labels = compute_labels_from_future_prices(
                base_open=base,
                future_closes=future_closes,
                horizons=tuple(LABEL_HORIZONS),
            )
            if want_vn:
                labels.update(compute_vn_labels_from_future_prices(
                    base_open=base,
                    future_closes=future_closes,
                    horizons=tuple(LABEL_HORIZONS),
                    path_horizon=10,
                ))
            result[sid] = labels

        # ng1.0.4: Compute max drawdown. ng1.2.x branches past ng1.0.1 so the
        # numeric version_ge check matches spuriously — gate with _is_1_2_branch.
        if version_ge(schema_ver, 'ng1.0.4') and not _is_1_2_branch(schema_ver):
            for sid in security_ids:
                fp = future_prices.get(sid, {})
                if not fp or future_dates[0] not in fp:
                    continue
                future_closes_for_dd = {}
                for n in range(max(LABEL_HORIZONS) + 1):
                    if n < len(future_dates) and future_dates[n] in fp:
                        future_closes_for_dd[n] = fp[future_dates[n]].get('close', np.nan)
                maxdd = compute_maxdd_from_future_prices(
                    base_open=fp[future_dates[0]].get('open', np.nan),
                    future_closes=future_closes_for_dd,
                    horizons=tuple(LABEL_HORIZONS),
                )
                if sid in result:
                    result[sid].update(maxdd)

        return result

    def _convert_labels_to_excess(
        self,
        labels_all: Dict[int, Dict[str, float]],
        universe: Dict[int, dict],
    ) -> Dict[int, Dict[str, float]]:
        """
        v1.0.3: Convert absolute labels to INDUSTRY EXCESS labels.
        label_Nd_excess = label_Nd_stock - median(label_Nd for same industry)
        """
        # Group by industry
        industry_labels: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        sid_to_industry: Dict[int, str] = {}

        for sid, labs in labels_all.items():
            info = universe.get(sid)
            if info is None:
                continue
            ind = info.get('industry') or 'unknown'
            sid_to_industry[sid] = ind
            for key, val in labs.items():
                if not np.isnan(val):
                    industry_labels[ind][key].append(val)

        # Compute industry medians
        industry_medians: Dict[str, Dict[str, float]] = {}
        for ind, label_dict in industry_labels.items():
            industry_medians[ind] = {}
            for key, vals in label_dict.items():
                industry_medians[ind][key] = float(np.median(vals)) if vals else 0.0

        # Convert to excess
        excess_labels: Dict[int, Dict[str, float]] = {}
        for sid, labs in labels_all.items():
            ind = sid_to_industry.get(sid, 'unknown')
            medians = industry_medians.get(ind, {})
            excess = {}
            for key, val in labs.items():
                if np.isnan(val):
                    excess[key] = np.nan
                else:
                    excess[key] = val - medians.get(key, 0.0)
            excess_labels[sid] = excess

        # v1.0.2: compute downside_10d from excess label_10d
        for sid, labs in excess_labels.items():
            label_10d = labs.get('label_10d', np.nan)
            if label_10d is not None and not np.isnan(label_10d):
                labs['downside_10d'] = max(0.0, -label_10d)
            else:
                labs['downside_10d'] = np.nan

        return excess_labels

    def _convert_labels_to_residual(
        self,
        excess_labels: Dict[int, Dict[str, float]],
        universe: Dict[int, dict],
        price_data: Dict[int, list],
        returns_20d: Dict[int, float],
        stock_volatilities: Dict[int, float],
    ) -> Dict[int, Dict[str, float]]:
        """
        v1.0.3: Convert industry-excess labels to STYLE RESIDUAL labels.
        Regress out [log_market_cap, momentum_20d, volatility_20d] cross-sectionally.
        Original excess labels stored as label_raw_Xd; residuals become label_Xd.
        """
        # Collect cross-sectional data
        sids_with_data = []
        X_list = []
        for sid, labs in excess_labels.items():
            info = universe.get(sid)
            if info is None:
                continue
            circ_mv = info.get('circ_mv', np.nan)
            if circ_mv is None or (isinstance(circ_mv, float) and np.isnan(circ_mv)):
                continue
            log_mcap = np.log(max(circ_mv, 1.0))
            mom_20d = returns_20d.get(sid, np.nan)
            vol_20d = stock_volatilities.get(sid, np.nan)
            if np.isnan(mom_20d) or np.isnan(vol_20d):
                continue
            sids_with_data.append(sid)
            X_list.append([log_mcap, mom_20d, vol_20d])

        # Need at least 100 stocks for meaningful regression
        if len(sids_with_data) < 100:
            # Keep excess labels as-is, just copy to label_raw
            for sid, labs in excess_labels.items():
                for h in LABEL_HORIZONS:
                    key = f'label_{h}d'
                    labs[f'label_raw_{h}d'] = labs.get(key, np.nan)
            return excess_labels

        X = np.array(X_list)

        # Run regression for each horizon
        for h in LABEL_HORIZONS:
            key = f'label_{h}d'
            raw_key = f'label_raw_{h}d'

            # First, save original excess as raw for ALL sids
            for sid, labs in excess_labels.items():
                labs[raw_key] = labs.get(key, np.nan)

            # Build y vector for regression (only sids_with_data)
            y = np.array([
                excess_labels[sid].get(key, np.nan) for sid in sids_with_data
            ])

            # Mask valid entries
            valid = ~np.isnan(y)
            if valid.sum() < 100:
                continue  # Not enough valid labels, keep excess as-is

            X_valid = X[valid]
            y_valid = y[valid]

            # Fit linear regression
            try:
                reg = LinearRegression()
                reg.fit(X_valid, y_valid)

                # Compute residuals for ALL sids_with_data
                y_pred_all = reg.predict(X)
                for i, sid in enumerate(sids_with_data):
                    labs = excess_labels[sid]
                    original = labs.get(key, np.nan)
                    if not np.isnan(original):
                        labs[key] = original - y_pred_all[i]
                    # else: keep as NaN
            except Exception:
                pass  # Keep excess labels if regression fails

        # Update downside_10d from residual label_10d
        for sid, labs in excess_labels.items():
            label_10d = labs.get('label_10d', np.nan)
            if label_10d is not None and not np.isnan(label_10d):
                labs['downside_10d'] = max(0.0, -label_10d)
            else:
                labs['downside_10d'] = np.nan

        return excess_labels

    def _convert_labels_to_risk_adjusted(
        self,
        labels_all: Dict[int, Dict[str, float]],
        penalty_power: float = 1.5,
    ) -> Dict[int, Dict[str, float]]:
        """ng1.0.4: Convert excess labels to risk-adjusted labels.
        ra_label_Nd = excess_label_Nd * (1 + maxDD_Nd) ^ penalty_power
        """
        for sid, labs in labels_all.items():
            for h in LABEL_HORIZONS:
                excess_key = f'label_{h}d'
                maxdd_key = f'maxdd_{h}d'
                ra_key = f'ra_label_{h}d'
                excess = labs.get(excess_key, np.nan)
                maxdd = labs.get(maxdd_key, np.nan)
                if excess is None or (isinstance(excess, (int, float)) and np.isnan(float(excess))):
                    labs[ra_key] = np.nan
                    continue
                if maxdd is None or (isinstance(maxdd, (int, float)) and np.isnan(float(maxdd))):
                    labs[ra_key] = np.nan
                    continue
                penalty = (1.0 + maxdd) ** penalty_power
                labs[ra_key] = float(excess * penalty)
        return labels_all

    # ------------------------------------------------------------------
    # Main entry point: process a single date
    # ------------------------------------------------------------------

    def update_single_date(self, date: str) -> int:
        """
        Compute NG v1.0.3 factors for all eligible A-stocks on the given date.
        Returns the number of stocks processed.
        """
        t0 = time.time()
        conn = self._connect()

        try:
            # 1. Load stock universe
            universe = self._load_stock_universe(conn)
            all_sids = list(universe.keys())
            if not all_sids:
                print(f"  [{date}] No A-stocks found in securities table")
                return 0

            chunk_size = 900

            # 2. Load price data
            print(f"  [{date}] Loading price data for {len(all_sids)} stocks...")
            price_data: Dict[int, List[dict]] = {}
            for i in range(0, len(all_sids), chunk_size):
                chunk = all_sids[i:i + chunk_size]
                chunk_data = self._load_price_data(conn, date, chunk)
                price_data.update(chunk_data)

            active_sids = [
                sid for sid, rows in price_data.items()
                if rows and rows[-1]['trade_date'] == date
            ]
            if not active_sids:
                print(f"  [{date}] No stocks traded on this date")
                return 0

            # 3. Load technical indicators
            print(f"  [{date}] Loading technical indicators...")
            tech_data: Dict[int, List[dict]] = {}
            for i in range(0, len(active_sids), chunk_size):
                chunk = active_sids[i:i + chunk_size]
                chunk_data = self._load_tech_indicators(conn, date, chunk)
                tech_data.update(chunk_data)

            # 4. Load daily_basic
            print(f"  [{date}] Loading daily_basic...")
            daily_basic: Dict[int, dict] = {}
            for i in range(0, len(active_sids), chunk_size):
                chunk = active_sids[i:i + chunk_size]
                chunk_data = self._load_daily_basic(conn, date, chunk)
                daily_basic.update(chunk_data)

            # 5. Load PE history
            print(f"  [{date}] Loading PE history...")
            pe_history: Dict[int, np.ndarray] = {}
            for i in range(0, len(active_sids), chunk_size):
                chunk = active_sids[i:i + chunk_size]
                chunk_data = self._load_pe_history(conn, date, chunk)
                pe_history.update(chunk_data)

            # 5.5. ng1.2.3: load turnover history for mined alpha factors
            # (4 of 6 mined factors need turnover as a time series, not just today's value)
            turnover_history: Dict[int, List] = {}
            if self.version == 'ng1.2.3':
                print(f"  [{date}] Loading turnover history (ng1.2.3 mined factors)...")
                for i in range(0, len(active_sids), chunk_size):
                    chunk = active_sids[i:i + chunk_size]
                    chunk_data = self._load_turnover_history(conn, date, chunk)
                    turnover_history.update(chunk_data)

            # 6. Load financial data
            print(f"  [{date}] Loading financial data...")
            fin_data: Dict[int, dict] = {}
            for i in range(0, len(active_sids), chunk_size):
                chunk = active_sids[i:i + chunk_size]
                chunk_data = self._load_financial_data(conn, date, chunk)
                fin_data.update(chunk_data)

            # 7. Load benchmark
            benchmark_closes = self._load_benchmark_data(conn, date)

            # v1.0.3: Load benchmark daily returns for residual factors
            benchmark_daily_returns = self._load_benchmark_daily_returns(conn, date, 25)

            # 8. Load northbound flow
            nb_5d, nb_std = self._load_northbound_data(conn, date)

            # 8.5. Load moneyflow data (v1.0.3)
            if version_ge(self.schema_version, 'ng1.0.3'):
                mf_count = self._fetch_and_store_moneyflow(conn, date)
                if mf_count > 0:
                    print(f"  [{date}] Fetched {mf_count} moneyflow records")
                mf_data = self._load_moneyflow_data(conn, date, active_sids, universe=universe, n_days=20)
            else:
                mf_data = {}

            # 8.6. ng1.2.3: pre-compute per-industry moneyflow peer scalars
            # Must run after mf_data is loaded (above) and before the per-stock loop.
            # Builds peer_mf_scalars_per_industry[industry] = {scalar_key → np.ndarray}
            # so Group D cs_rank factors can be computed per stock without re-scanning all peers.
            # stock_mf_scalars_per_code caches per-stock scalars to avoid a second
            # compute_stock_mf_scalars() call in the per-stock loop (Fix #2).
            peer_mf_scalars_per_industry: Dict[str, Dict[str, np.ndarray]] = {}
            stock_mf_scalars_per_code: Dict[str, Dict[str, float]] = {}
            if self.version in ('ng1.2.3', 'ng1.2.4'):
                _ind_scalars: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
                for _sid in active_sids:
                    _info = universe.get(_sid)
                    if _info is None:
                        continue
                    _code = _info['code']
                    _rows = mf_data.get(_code, [])
                    if not _rows:
                        continue
                    _scalars = compute_stock_mf_scalars(_rows)
                    stock_mf_scalars_per_code[_code] = _scalars  # cache for per-stock loop
                    _industry = _info.get('industry') or 'unknown'
                    for _k, _v in _scalars.items():
                        if not np.isnan(_v):
                            _ind_scalars[_industry][_k].append(_v)
                for _ind, _scalar_dict in _ind_scalars.items():
                    peer_mf_scalars_per_industry[_ind] = {
                        _k: np.array(_vlist) for _k, _vlist in _scalar_dict.items()
                    }

            # 9. Load total market amounts
            market_amounts = self._load_market_amounts(conn, date)

            # 10. Compute stock-level returns
            returns_1d = self._compute_1d_returns(price_data)
            returns_5d = self._compute_stock_returns(price_data, 5)
            returns_20d = self._compute_stock_returns(price_data, 20)
            highs_20d_ratio = self._compute_highs_20d_ratio(price_data)

            # v1.0.3: Stock-level daily returns and volatilities for residual/CS rank
            stock_daily_returns = self._compute_stock_daily_returns(price_data, 20)
            stock_volatilities = self._compute_stock_volatilities(price_data, 20)

            # 11. Compute market features
            all_ret1d = np.array([v for v in returns_1d.values() if not np.isnan(v)])
            all_h20_ratio = np.array([v for v in highs_20d_ratio.values() if not np.isnan(v)])

            market_feats = compute_market_features(
                benchmark_closes=benchmark_closes if len(benchmark_closes) > 0 else np.array([1.0]),
                all_stock_returns=all_ret1d,
                all_stock_highs_20d_ratio=all_h20_ratio,
                total_market_amount=market_amounts if len(market_amounts) > 0 else np.array([1.0]),
                northbound_net_buy_5d=nb_5d,
                northbound_std=nb_std,
            )

            # 11.4 ng1.5.x: Load industry 5d ret history (~65d × all industries)
            #      + benchmark 5d ret history for industry_regime_agreement feature.
            ng150_industry_5d_hist: Dict[str, np.ndarray] = {}
            ng150_benchmark_5d_hist: np.ndarray = np.array([])
            ng150_bench_1d_rets: np.ndarray = np.array([])
            ng150_amv_var1_ma60 = np.nan
            if _is_1_5_branch(self.schema_version):
                try:
                    (ng150_industry_5d_hist,
                     ng150_benchmark_5d_hist) = self._load_industry_5d_ret_history(conn, date)
                except Exception as _e:
                    print(f"  [{date}] WARN: ng150 industry history failed: {_e}")
                # Benchmark 1d log-returns over 60+ days from already-loaded benchmark_closes
                if len(benchmark_closes) >= 2:
                    ng150_bench_1d_rets = np.diff(np.log(benchmark_closes.astype(float) + 1e-8))
                try:
                    ng150_amv_var1_ma60 = self._load_amv_var1_ma60(conn, date)
                except Exception as _e:
                    print(f"  [{date}] WARN: ng150 amv ma60 failed: {_e}")

            # 11.5 ng1.0.7: Load AMV data and compute extended market features
            ext_market_feats = {}
            amv_row = None
            if version_ge(self.schema_version, 'ng1.0.7'):
                try:
                    amv_row_data = conn.execute(
                        'SELECT var1, amv_macd, amv_regime FROM market_amv WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT 1',
                        (date,)
                    ).fetchone()
                    if amv_row_data:
                        amv_var1_val = _safe_float(amv_row_data['var1'])
                        amv_macd_val = _safe_float(amv_row_data['amv_macd'])

                        # Compute regime_days: count consecutive days in current regime
                        amv_history = conn.execute(
                            'SELECT amv_regime FROM market_amv WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT 120',
                            (date,)
                        ).fetchall()
                        regime_days = 0
                        if amv_history:
                            current_regime = int(amv_history[0]['amv_regime'])
                            for r in amv_history:
                                if int(r['amv_regime']) == current_regime:
                                    regime_days += 1
                                else:
                                    break

                        # Compute breadth_history_5d: get dates then batch breadth
                        breadth_arr = np.array([])
                        date_rows = conn.execute(
                            '''SELECT DISTINCT trade_date FROM daily_quotes
                               WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT 6''',
                            (date,)
                        ).fetchall()
                        if len(date_rows) >= 2:
                            date_list = [r['trade_date'] for r in date_rows]
                            placeholders = ','.join('?' * len(date_list))
                            breadth_rows = conn.execute(
                                f'''SELECT dq.trade_date,
                                           AVG(CASE WHEN dq.price_change_pct > 0 THEN 1.0 ELSE 0.0 END) as breadth
                                    FROM daily_quotes dq
                                    JOIN securities s ON dq.security_id = s.id
                                    WHERE s.type = 'A股' AND dq.trade_date IN ({placeholders})
                                    GROUP BY dq.trade_date ORDER BY dq.trade_date''',
                                date_list
                            ).fetchall()
                            if breadth_rows:
                                breadth_arr = np.array([float(r['breadth']) for r in breadth_rows
                                                        if r['breadth'] is not None])

                        ext_market_feats = compute_extended_market_features(
                            benchmark_closes=benchmark_closes if len(benchmark_closes) > 0 else np.array([1.0]),
                            total_market_amount=market_amounts if len(market_amounts) > 0 else np.array([1.0]),
                            amv_var1=amv_var1_val,
                            amv_macd=amv_macd_val,
                            amv_regime_days=regime_days,
                            market_breadth_history_5d=breadth_arr,
                        )
                        amv_row = {'var1': amv_var1_val, 'macd': amv_macd_val, 'regime_days': regime_days}
                    else:
                        print(f"  [{date}] WARN: No AMV data found, ext_market_feats empty")
                except Exception as e:
                    print(f"  [{date}] WARN: AMV loading failed: {e}")

            # 12. Compute industry aggregates
            industry_agg = self._compute_industry_aggregates(
                universe, price_data, returns_1d, returns_5d, returns_20d
            )

            all_ind_ret5d = np.array([
                v['mean_return_5d'] for v in industry_agg.values()
                if not np.isnan(v['mean_return_5d'])
            ])

            # 13. Compute labels + convert to INDUSTRY EXCESS (v1.0.3)
            future_dates = self._get_future_dates(conn, date, max(LABEL_HORIZONS) + 1)
            future_prices: Dict[int, Dict[str, dict]] = {}
            if future_dates:
                for i in range(0, len(active_sids), chunk_size):
                    chunk = active_sids[i:i + chunk_size]
                    chunk_data = self._load_future_prices(conn, future_dates, chunk)
                    future_prices.update(chunk_data)

            # Compute absolute labels for ALL active stocks (including filtered ones)
            labels_abs = self._compute_labels(future_dates, future_prices, active_sids)
            # Convert to industry excess returns
            labels_all = self._convert_labels_to_excess(labels_abs, universe)

            # v1.0.3: Convert to style residual labels
            if version_ge(self.schema_version, 'ng1.0.3'):
                # Inject circ_mv into universe for residual regression
                for sid in active_sids:
                    db = daily_basic.get(sid)
                    if db:
                        universe[sid]['circ_mv'] = _safe_float(db.get('circ_mv'))
                labels_all = self._convert_labels_to_residual(
                    labels_all, universe, price_data, returns_20d, stock_volatilities
                )

            # ng1.0.4: Compute risk-adjusted labels
            if version_ge(self.schema_version, 'ng1.0.4'):
                pp = getattr(self, 'penalty_power', 1.5)
                labels_all = self._convert_labels_to_risk_adjusted(labels_all, penalty_power=pp)

            # ng1.0.7: Compute conditional labels
            # Bear market: rank_pct (relative positioning), Bull: industry excess
            # Smooth blend based on market_return_20d
            if version_ge(self.schema_version, 'ng1.0.7'):
                mkt_ret_20d = market_feats.get('market_return_20d', 0.0)
                if np.isnan(mkt_ret_20d):
                    mkt_ret_20d = 0.0
                # bear_weight: 0 when mkt_ret >= 0, 1 when mkt_ret <= -10%
                bear_weight = max(0.0, min(1.0, -mkt_ret_20d / 0.10))

                for h in [3, 5, 10, 15]:
                    excess_key = f'label_{h}d'

                    # Collect all excess labels for this horizon to compute rank
                    all_excess = {}
                    for sid, lbl in labels_all.items():
                        val = lbl.get(excess_key)
                        if val is not None and not np.isnan(val):
                            all_excess[sid] = val

                    if all_excess:
                        # Compute rank percentile (0..1, higher = better)
                        sorted_sids = sorted(all_excess.keys(), key=lambda s: all_excess[s])
                        n_total = len(sorted_sids)
                        rank_map = {sid: i / (n_total - 1) if n_total > 1 else 0.5
                                   for i, sid in enumerate(sorted_sids)}

                        for sid in labels_all:
                            excess_val = labels_all[sid].get(excess_key)
                            rank_val = rank_map.get(sid)
                            if excess_val is not None and rank_val is not None:
                                cond_val = (1.0 - bear_weight) * excess_val + bear_weight * (rank_val - 0.5) * 0.1
                                labels_all[sid][f'cond_label_{h}d'] = cond_val
                            else:
                                labels_all[sid][f'cond_label_{h}d'] = None

            # ---------------------------------------------------------------
            # v1.0.3: Pre-compute per-industry peer arrays for CS rank factors
            # ---------------------------------------------------------------
            # We need raw values for each metric, grouped by industry
            # First pass: compute raw metrics for all eligible stocks

            # Prepare data structures for CS rank
            # sid → raw metric values (computed during feature loop below)
            # We need a two-pass approach:
            #   Pass 1: compute raw values for all stocks
            #   Pass 2: compute CS ranks per industry + residual factors

            # Pass 1: Collect raw metric values for all eligible stocks
            print(f"  [{date}] Computing features (pass 1: raw values)...")

            eligible_stocks = {}  # sid → {raw metrics + features}
            skipped_mv = 0
            skipped_data = 0

            for sid in active_sids:
                rows = price_data.get(sid, [])
                if len(rows) < MIN_DATA_DAYS:
                    skipped_data += 1
                    continue

                db = daily_basic.get(sid)
                if db is None:
                    skipped_mv += 1
                    continue
                circ_mv = _safe_float(db.get('circ_mv'))
                if np.isnan(circ_mv) or circ_mv < CIRC_MV_MIN:
                    skipped_mv += 1
                    continue

                last_row = rows[-1]
                if last_row.get('is_st'):
                    skipped_data += 1
                    continue

                info = universe[sid]
                code = info['code']
                industry = info.get('industry') or 'unknown'

                closes = np.array([_safe_float(r['close']) for r in rows])
                opens = np.array([_safe_float(r['open']) for r in rows])
                highs = np.array([_safe_float(r['high']) for r in rows])
                lows = np.array([_safe_float(r['low']) for r in rows])
                volumes = np.array([_safe_float(r['volume']) for r in rows])
                amounts_raw = [r['amount'] for r in rows]
                if all(a is None for a in amounts_raw):
                    amounts = volumes.copy()
                else:
                    amounts = np.array([_safe_float(a, 0) for a in amounts_raw])
                ma5_arr = np.array([_safe_float(r['ma5']) for r in rows])
                ma10_arr = np.array([_safe_float(r['ma10']) for r in rows])
                ma20_arr = np.array([_safe_float(r['ma20']) for r in rows])
                ma60_arr = np.array([_safe_float(r['ma60']) for r in rows])

                # Technical indicators
                tech_rows = tech_data.get(sid, [])
                tech_today = None
                macd_arr = np.full(len(rows), np.nan)
                if tech_rows:
                    tech_by_date = {tr['trade_date']: tr for tr in tech_rows}
                    tech_today = tech_by_date.get(date)
                    for i_r, r in enumerate(rows):
                        td = r['trade_date']
                        if td in tech_by_date:
                            macd_arr[i_r] = _safe_float(tech_by_date[td].get('macd_macd'))

                atr_14 = _safe_float(tech_today['atr_14']) if tech_today else np.nan
                kdj_j = _safe_float(tech_today['kdj_j']) if tech_today else 50.0
                boll_upper = _safe_float(tech_today['boll_upper']) if tech_today else _safe_float(rows[-1]['close'])
                boll_lower = _safe_float(tech_today['boll_lower']) if tech_today else _safe_float(rows[-1]['close'])
                rsi12 = _safe_float(tech_today['rsi12']) if tech_today else 50.0
                rsi24 = _safe_float(tech_today['rsi24']) if tech_today else 50.0

                if np.isnan(atr_14):
                    if len(rows) >= 15:
                        tr_vals = []
                        for j in range(-14, 0):
                            h = _safe_float(rows[j]['high'])
                            l = _safe_float(rows[j]['low'])
                            c_prev = _safe_float(rows[j - 1]['close'])
                            tr_vals.append(max(h - l, abs(h - c_prev), abs(l - c_prev)))
                        atr_14 = float(np.mean(tr_vals))
                    else:
                        atr_14 = 1.0

                # --- Compute stock features (19, was 30 in v1.0.0) ---
                try:
                    stock_feats = compute_stock_features(
                        closes=closes, opens=opens, highs=highs, lows=lows,
                        volumes=volumes, amounts=amounts,
                        ma5=ma5_arr, ma10=ma10_arr, ma20=ma20_arr, ma60=ma60_arr,
                        atr_14=atr_14, macd_macd=macd_arr,
                        kdj_j=kdj_j, boll_upper=boll_upper, boll_lower=boll_lower,
                        rsi_12=rsi12, rsi_24=rsi24,
                    )
                except Exception as e:
                    print(f"    WARN: stock_features failed for {code}: {e}")
                    continue

                # ng1.0.4: Compute smoothing features (9)
                smooth_feats = {}
                if version_ge(self.schema_version, 'ng1.0.4'):
                    try:
                        smooth_feats = compute_smoothing_features(
                            closes=closes, opens=opens, highs=highs,
                            lows=lows,
                        )
                    except Exception as e:
                        print(f"    WARN: smoothing_features failed for {code}: {e}")
                        smooth_feats = {}

                # ng1.5.0: Regime-refined per-stock features (4)
                ng150_feats: Dict[str, float] = {}
                if _is_1_5_branch(self.schema_version):
                    try:
                        stock_1d_rets = (
                            np.diff(np.log(closes.astype(float) + 1e-8))
                            if len(closes) >= 2 else np.array([])
                        )
                        ind_5d_arr = ng150_industry_5d_hist.get(industry, np.array([]))
                        ng150_feats = compute_ng150_regime_stock_features(
                            closes=closes,
                            stock_returns_1d=stock_1d_rets,
                            benchmark_returns_1d=ng150_bench_1d_rets,
                            industry_returns_5d_history=ind_5d_arr,
                            benchmark_returns_5d_history=ng150_benchmark_5d_hist,
                        )
                    except Exception as e:
                        print(f"    WARN: ng150_regime_stock failed for {code}: {e}")
                        ng150_feats = {}

                # --- Compute fundamental features (14) ---
                fin = fin_data.get(sid, {})
                pe_hist = pe_history.get(sid, np.array([]))
                adv_20d = float(np.mean(amounts[-20:])) if len(amounts) >= 20 else _safe_float(amounts[-1])

                try:
                    fund_feats = compute_fundamental_features(
                        pe_ttm=_safe_float(db.get('pe_ttm')),
                        pb=_safe_float(db.get('pb')),
                        dv_ratio=_safe_float(db.get('dv_ratio')),
                        circ_mv=circ_mv,
                        free_share=_safe_float(db.get('free_share')),
                        total_share=_safe_float(db.get('total_share')),
                        turnover_rate=_safe_float(db.get('turnover_rate')),
                        adv_20d=adv_20d,
                        pe_ttm_history_60d=pe_hist,
                        roe=fin.get('roe', np.nan),
                        roe_prev_year=fin.get('roe_prev_year', np.nan),
                        profit_to_gr=fin.get('profit_to_gr', np.nan),
                        or_yoy=fin.get('or_yoy', np.nan),
                        netprofit_margin=fin.get('netprofit_margin', np.nan),
                        ocf_to_profit=fin.get('ocf_to_profit', np.nan),
                        debt_to_assets=fin.get('debt_to_assets', np.nan),
                        current_ratio=fin.get('current_ratio', np.nan),
                    )
                except Exception as e:
                    print(f"    WARN: fundamental_features failed for {code}: {e}")
                    fund_feats = {}

                # --- Compute industry features (11, was 8 in v1.0.0) ---
                ind_agg = industry_agg.get(industry, {})
                stock_ret_20d = returns_20d.get(sid, np.nan)
                sw_ret_5d = ind_agg.get('mean_return_5d', np.nan)

                try:
                    ind_feats = compute_industry_features(
                        stock_return_20d=stock_ret_20d if not np.isnan(stock_ret_20d) else 0.0,
                        industry_stock_returns_1d=ind_agg.get('returns_1d', np.array([])),
                        industry_stock_returns_5d=ind_agg.get('returns_5d', np.array([])),
                        industry_stock_returns_20d=ind_agg.get('returns_20d', np.array([])),
                        industry_amounts_5d=ind_agg.get('amounts_5d', np.array([])),
                        industry_amounts_20d=ind_agg.get('amounts_20d', np.array([])),
                        all_industry_returns_5d=all_ind_ret5d,
                        sw_index_return_5d=sw_ret_5d if not np.isnan(sw_ret_5d) else 0.0,
                        # v1.0.3: sector activity params
                        market_breadth=market_feats.get('market_breadth', np.nan),
                        market_volume_ratio=market_feats.get('market_volume_ratio', np.nan),
                    )
                except Exception as e:
                    print(f"    WARN: industry_features failed for {code}: {e}")
                    ind_feats = {}

                # --- Compute moneyflow features (8, v1.0.3) ---
                mf_feats = {}
                if version_ge(self.schema_version, 'ng1.0.3'):
                    code_for_mf = info['code']
                    mf_rows_for_stock = mf_data.get(code_for_mf, [])
                    if len(closes) >= 2:
                        price_changes = np.diff(closes) / (closes[:-1] + 1e-8)
                    else:
                        price_changes = np.array([0.0])
                    try:
                        mf_feats = compute_moneyflow_features(
                            mf_rows_for_stock, amounts, price_changes
                        )
                    except Exception as e:
                        print(f"    WARN: moneyflow_features failed for {code}: {e}")
                        mf_feats = {}

                # --- ng1.2.3/ng1.2.4: compute moneyflow factors ---
                ng123_mf_feats = {}
                if self.version in ('ng1.2.3', 'ng1.2.4'):
                    _mf_rows = mf_data.get(code, [])
                    _stock_scalars = stock_mf_scalars_per_code.get(
                        code, compute_stock_mf_scalars(_mf_rows)
                    )
                    _peer_scalars = peer_mf_scalars_per_industry.get(industry, {})
                    try:
                        ng123_mf_feats = compute_all_moneyflow_factors(
                            _mf_rows,
                            stock_scalars=_stock_scalars,
                            peer_scalars=_peer_scalars,
                            ng124_mode=(self.version == 'ng1.2.4'),
                        )
                    except Exception as e:
                        print(f"    WARN: ng123 moneyflow_factors failed for {code}: {e}")
                        ng123_mf_feats = {}

                # --- ng1.3.0: Compute Tier B moneyflow factors (3) ---
                ng130_mf_feats: Dict[str, float] = {}
                if _is_1_3_branch(self.schema_version):
                    _mf_rows_130 = mf_data.get(code, [])
                    try:
                        # cs_z_history_elg=None → log-sign self-transform fallback.
                        # Cross-sectional z-score pipeline deferred to future iteration.
                        ng130_mf_feats = compute_ng130_mf_factors(_mf_rows_130)
                    except Exception as e:
                        print(f"    WARN: ng130 mf factors failed for {code}: {e}")
                        ng130_mf_feats = {name: np.nan for name in NG130_MF_FACTORS}

                # --- ng1.2.3: compute 6 mined alpha factors ---
                ng123_mined_feats = {}
                if self.version == 'ng1.2.3' and MINED_FACTOR_SPEC:
                    # Build per-stock OHLCV df aligned with price rows.
                    # Merge daily_basic turnover_rate by trade_date for full time series.
                    _tr_history = turnover_history.get(sid, [])
                    _tr_by_date = {td: v for td, v in _tr_history}
                    _turnover_arr = np.array([
                        float(_tr_by_date[r['trade_date']])
                        if r['trade_date'] in _tr_by_date and _tr_by_date[r['trade_date']] is not None
                        else np.nan
                        for r in rows
                    ])
                    # Forward-fill NaN turnover (sparse coverage in daily_basic)
                    _last_valid = np.nan
                    for _i in range(len(_turnover_arr)):
                        if np.isfinite(_turnover_arr[_i]):
                            _last_valid = _turnover_arr[_i]
                        elif np.isfinite(_last_valid):
                            _turnover_arr[_i] = _last_valid
                    if len(rows) >= 60:
                        _df_stock = pd.DataFrame({
                            'open': opens,
                            'high': highs,
                            'low': lows,
                            'close': closes,
                            'volume': volumes,
                            'price_change_pct': np.array(
                                [_safe_float(r.get('price_change_pct', 0)) for r in rows]
                            ),
                            'turnover_rate': _turnover_arr,
                        })
                        # Call generate_operands ONCE and reuse across all 6 specs
                        # (vs 6 separate compute_mined_factor_value calls each rebuilding operands)
                        _operands = generate_operands(_df_stock)
                        for _spec in MINED_FACTOR_SPEC:
                            try:
                                _series = _compute_mined_factor(_spec, _operands)
                                if _series is None:
                                    _val = np.nan
                                else:
                                    _vals = np.asarray(_series.values, dtype=np.float64)
                                    if _spec.get('sign_flip', False):
                                        _vals = -_vals
                                    _val = float(_vals[-1]) if len(_vals) > 0 and np.isfinite(_vals[-1]) else np.nan
                                ng123_mined_feats[_spec['name']] = _val
                            except Exception:
                                ng123_mined_feats[_spec['name']] = np.nan
                    else:
                        # Not enough history — NaN-fill
                        for _spec in MINED_FACTOR_SPEC:
                            ng123_mined_feats[_spec['name']] = np.nan

                # Store raw values needed for CS rank (pass 2).
                # IMPORTANT: read raw values from UNFILTERED stock_feats/fund_feats/ind_feats.
                # The ng1.2.3 filter call below (filter_ng123_features) drops 'dv_ratio' and
                # other features that are also CS-rank inputs — reading after filtering would
                # silently produce NaN if the drop list ever expands further.
                ret_5d_val = returns_5d.get(sid, np.nan)
                ret_20d_val = returns_20d.get(sid, np.nan)
                vol_ratio = stock_feats.get('volume_ratio_5d', np.nan) if stock_feats else np.nan
                turnover = fund_feats.get('turnover_rate', np.nan) if fund_feats else np.nan
                rsi_val = stock_feats.get('rsi_14', np.nan) if stock_feats else np.nan
                new_high_dist = highs_20d_ratio.get(sid, np.nan)
                pullback_val = stock_feats.get('pullback_from_high', np.nan) if stock_feats else np.nan
                volatility_val = stock_volatilities.get(sid, np.nan)
                mcap_val = fund_feats.get('log_market_cap', np.nan) if fund_feats else np.nan
                pe_val = fund_feats.get('pe_ttm', np.nan) if fund_feats else np.nan
                pb_val = fund_feats.get('pb', np.nan) if fund_feats else np.nan
                dv_val = fund_feats.get('dv_ratio', np.nan) if fund_feats else np.nan
                avg_vol_5d = float(np.mean(amounts[-5:])) if len(amounts) >= 5 else np.nan

                # ng1.2.3: filter 12 drop-list features AFTER CS rank raw value snapshot.
                # Must read raw values from unfiltered dicts to avoid NaN corruption if
                # the drop list ever expands to include CS-rank inputs (e.g. dv_ratio is
                # already in both the drop list and the CS-rank inputs).
                if self.version == 'ng1.2.3':
                    # 1. Drop 12 weak features from ng1.0.1 base per spec §4.3
                    # Applied to stock, fund, and industry feature dicts since some
                    # dropped features live in ind_feats (e.g. industry_hhi).
                    stock_feats = filter_ng123_features(stock_feats)
                    fund_feats = filter_ng123_features(fund_feats)
                    ind_feats = filter_ng123_features(ind_feats)

                eligible_stocks[sid] = {
                    'code': code,
                    'industry': industry,
                    'stock_feats': stock_feats,
                    'fund_feats': fund_feats,
                    'ind_feats': ind_feats,
                    'mf_feats': mf_feats,
                    'smooth_feats': smooth_feats,
                    'ng150_feats': ng150_feats,          # ng1.5.0: 4 Tier B regime-refined
                    'ng123_mf_feats': ng123_mf_feats,   # ng1.2.3: 12-factor moneyflow
                    'ng123_mined_feats': ng123_mined_feats,  # ng1.2.3: 6 mined alpha factors
                    'ng130_mf_feats': ng130_mf_feats,   # ng1.3.0: 3 Tier B moneyflow factors
                    # Raw values for CS rank
                    'return_5d': ret_5d_val,
                    'return_20d': ret_20d_val,
                    'volume_surge': vol_ratio,
                    'turnover': turnover,
                    'rsi': rsi_val,
                    'new_high_dist': new_high_dist,
                    'pullback': pullback_val,
                    'volatility': volatility_val,
                    'market_cap': mcap_val,
                    'pe': pe_val,
                    'pb': pb_val,           # ng1.1.0 P2
                    'dv_ratio': dv_val,     # ng1.1.0 P2
                    # For residual factors
                    'daily_returns': stock_daily_returns.get(sid),
                    'avg_volume_5d': avg_vol_5d,
                    # ng1.2.3: today's close needed for downside_kd label computation
                    'today_close': _safe_float(rows[-1]['close']),
                }

            # ---------------------------------------------------------------
            # Pass 2: Compute CS rank + residual factors per industry
            # ---------------------------------------------------------------
            print(f"  [{date}] Computing features (pass 2: CS rank + residuals)...")

            # Group eligible stocks by industry
            industry_groups: Dict[str, List[int]] = defaultdict(list)
            for sid, data in eligible_stocks.items():
                industry_groups[data['industry']].append(sid)

            # Pre-build industry peer arrays
            CS_METRICS = [
                'return_5d', 'return_20d', 'volume_surge', 'turnover',
                'rsi', 'new_high_dist', 'pullback', 'volatility',
                'market_cap', 'pe', 'pb', 'dv_ratio',
            ]

            industry_peer_arrays: Dict[str, Dict[str, np.ndarray]] = {}
            for ind, sids in industry_groups.items():
                arrays = {}
                for metric in CS_METRICS:
                    arrays[metric] = np.array([
                        eligible_stocks[s][metric] for s in sids
                    ])
                industry_peer_arrays[ind] = arrays

            insert_rows = []
            # ng1.2.x / ng1.3.x / ng1.5.x branch guards — used in both per-stock column gating and INSERT dispatch.
            is_12 = _is_1_2_branch(self.schema_version)
            is_13 = _is_1_3_branch(self.schema_version)
            is_15 = _is_1_5_branch(self.schema_version)

            for sid, data in eligible_stocks.items():
                industry = data['industry']
                peers = industry_peer_arrays.get(industry, {})
                ind_agg = industry_agg.get(industry, {})

                # --- CS rank features (10 new) ---
                try:
                    cs_feats = compute_cross_sectional_rank_features(
                        stock_return_5d=data['return_5d'],
                        stock_return_20d=data['return_20d'],
                        stock_volume_surge=data['volume_surge'],
                        stock_turnover=data['turnover'],
                        stock_rsi=data['rsi'],
                        stock_new_high_dist=data['new_high_dist'],
                        stock_pullback=data['pullback'],
                        stock_volatility=data['volatility'],
                        stock_market_cap=data['market_cap'],
                        stock_pe=data['pe'],
                        peer_returns_5d=peers.get('return_5d', np.array([])),
                        peer_returns_20d=peers.get('return_20d', np.array([])),
                        peer_volume_surges=peers.get('volume_surge', np.array([])),
                        peer_turnovers=peers.get('turnover', np.array([])),
                        peer_rsis=peers.get('rsi', np.array([])),
                        peer_new_high_dists=peers.get('new_high_dist', np.array([])),
                        peer_pullbacks=peers.get('pullback', np.array([])),
                        peer_volatilities=peers.get('volatility', np.array([])),
                        peer_market_caps=peers.get('market_cap', np.array([])),
                        peer_pes=peers.get('pe', np.array([])),
                        # ng1.1.0 P2: new cs_rank dimensions
                        stock_pb=data.get('pb', np.nan),
                        stock_dv=data.get('dv_ratio', np.nan),
                        peer_pbs=peers.get('pb', np.array([])),
                        peer_dvs=peers.get('dv_ratio', np.array([])),
                    )
                except Exception as e:
                    print(f"    WARN: cs_rank failed for {data['code']}: {e}")
                    cs_feats = {}

                # --- Residual features (5 new) ---
                try:
                    res_feats = compute_residual_features(
                        stock_daily_returns=data['daily_returns'],
                        market_daily_returns=benchmark_daily_returns if len(benchmark_daily_returns) > 0 else None,
                        industry_return_20d=ind_agg.get('mean_return_20d', np.nan),
                        stock_avg_volume_5d=data['avg_volume_5d'],
                        industry_avg_volume_5d=ind_agg.get('avg_volume_5d', np.nan),
                        stock_return_20d=data['return_20d'],
                        market_return_20d=market_feats.get('market_return_20d', np.nan),
                    )
                except Exception as e:
                    print(f"    WARN: residual_features failed for {data['code']}: {e}")
                    res_feats = {}

                # --- Interaction features (8, v1.0.3) ---
                ix_feats = {}
                if version_ge(self.schema_version, 'ng1.0.3'):
                    try:
                        ix_feats = compute_interaction_features(
                            data['stock_feats'],
                            data.get('mf_feats', {}),
                            data['ind_feats'],
                            res_feats,
                            cs_feats,
                            fund_feats=data.get('fund_feats', {}),
                        )
                    except Exception as e:
                        print(f"    WARN: interaction_features failed for {data['code']}: {e}")
                        ix_feats = {}

                # --- ng1.0.7: Conditional interaction features (7) ---
                cond_ix_feats = {}
                if version_ge(self.schema_version, 'ng1.0.7') and ext_market_feats:
                    try:
                        cond_ix_feats = compute_conditional_interaction_features(
                            stock_feats=data['stock_feats'],
                            fund_feats=data.get('fund_feats', {}),
                            market_feats=market_feats,
                            ext_market_feats=ext_market_feats,
                            industry_feats=data['ind_feats'],
                            residual_feats=res_feats,
                        )
                    except Exception as e:
                        print(f"    WARN: conditional_ix failed for {data['code']}: {e}")
                        cond_ix_feats = {}

                # --- Merge all features ---
                all_feats = {}
                all_feats.update(data['stock_feats'])
                all_feats.update(data['fund_feats'])
                all_feats.update(data['ind_feats'])
                all_feats.update(cs_feats)
                all_feats.update(res_feats)
                if version_ge(self.schema_version, 'ng1.0.3') and self.version != 'ng1.2.3':
                    # ng1.2.3 replaces legacy mf_feats with ng123_mf_feats (below)
                    all_feats.update(data.get('mf_feats', {}))
                    all_feats.update(ix_feats)
                if version_ge(self.schema_version, 'ng1.0.4'):
                    all_feats.update(data.get('smooth_feats', {}))
                if (version_ge(self.schema_version, 'ng1.0.7')
                        and self.version not in ('ng1.2.3', 'ng1.2.4')
                        and not _is_1_3_branch(self.schema_version)
                        and not _is_1_5_branch(self.schema_version)):
                    # ext_market_feats stored in features_json for scorer access
                    # (only non-AMV features; AMV values already in dedicated columns)
                    # ng1.2.x / ng1.3.x / ng1.5.x branches do NOT inherit ng1.0.7 additions
                    for k, v in ext_market_feats.items():
                        if k not in ('amv_var1', 'amv_macd', 'amv_regime_days'):
                            all_feats[k] = v
                    all_feats.update(cond_ix_feats)
                # ng1.3.0: add only 3 AMV + 3 mf factors to features_json (no cond_ix, no other ext_market)
                if _is_1_3_branch(self.schema_version):
                    for k in ('amv_var1', 'amv_macd', 'amv_regime_days'):
                        if k in ext_market_feats:
                            all_feats[k] = ext_market_feats[k]
                    all_feats.update(data.get('ng130_mf_feats', {}))
                # ng1.5.x: add 3 AMV + 1 new market regime feature + 4 new stock regime features
                if _is_1_5_branch(self.schema_version):
                    for k in ('amv_var1', 'amv_macd', 'amv_regime_days'):
                        if k in ext_market_feats:
                            all_feats[k] = ext_market_feats[k]
                    ng150_mkt = compute_ng150_regime_market_features(
                        amv_var1=ext_market_feats.get('amv_var1', np.nan),
                        amv_macd=ext_market_feats.get('amv_macd', np.nan),
                        amv_var1_ma60=ng150_amv_var1_ma60,
                    )
                    all_feats.update(ng150_mkt)
                    all_feats.update(data.get('ng150_feats', {}))
                # ng1.2.3: add 12 moneyflow factors + 6 mined alpha factors to features_json
                # (mf_feats from ng1.0.3 path was already dropped via filter above;
                #  ng123_mf_feats replaces it with the new 12-factor set)
                if self.version == 'ng1.2.3':
                    all_feats.update(data.get('ng123_mf_feats', {}))
                    all_feats.update(data.get('ng123_mined_feats', {}))
                elif self.version == 'ng1.2.4':
                    all_feats.update(data.get('ng123_mf_feats', {}))

                # Clean NaN/Inf
                clean_feats = {}
                for k, v in all_feats.items():
                    if isinstance(v, (float, np.floating)):
                        if np.isnan(v) or np.isinf(v):
                            clean_feats[k] = None
                        else:
                            clean_feats[k] = float(v)
                    elif isinstance(v, np.integer):
                        clean_feats[k] = int(v)
                    else:
                        clean_feats[k] = v
                features_json = json.dumps(clean_feats)

                stock_labels = labels_all.get(sid, {})

                # ng1.2.3: compute 4-horizon downside_kd from future closes
                # base_row omits the legacy downside_10d position; downside values
                # go in ng123_cols using the new schema columns (downside_3d..15d).
                ng123_downside: Dict[str, float] = {}
                if self.version == 'ng1.2.3':
                    today_close = data.get('today_close', np.nan)
                    for _k in [3, 5, 10, 15]:
                        _future_closes_k = []
                        for _fd in future_dates[:_k + 1]:
                            if sid in future_prices and _fd in future_prices[sid]:
                                _fc = future_prices[sid][_fd].get('close')
                                if _fc is not None and not np.isnan(_fc):
                                    _future_closes_k.append(_fc)
                        _pm = compute_path_min_kd(today_close, np.array(_future_closes_k))
                        ng123_downside[f'downside_{_k}d'] = float(
                            compute_downside_kd(_pm)
                        )

                # ng1.3.x / ng1.5.x: compute 4-horizon downside labels (min-cumret semantics).
                # ng1.5.0 stores them in the same schema shape as ng1.3.0 even though the
                # trainer uses industry excess labels (Phase A per spec §2.2).
                ng130_downside: Dict[str, float] = {}
                if _is_1_3_branch(self.schema_version) or _is_1_5_branch(self.schema_version):
                    today_close = data.get('today_close', np.nan)
                    future_closes_list = []
                    for _fd in future_dates[:16]:  # t+1..t+15
                        if sid in future_prices and _fd in future_prices[sid]:
                            _fc = future_prices[sid][_fd].get('close')
                            if _fc is not None and not np.isnan(_fc):
                                future_closes_list.append(_fc)
                    ng130_downside = compute_all_downside_horizons(
                        float(today_close) if not np.isnan(today_close) else 0.0,
                        np.array(future_closes_list, dtype=np.float64),
                    )

                # ng1.2.3+ / ng1.3.x / ng1.5.x use label-only base_row (no legacy downside_10d position)
                if (self.version in ('ng1.2.3', 'ng1.2.4')
                        or _is_1_3_branch(self.schema_version)
                        or _is_1_5_branch(self.schema_version)):
                    base_row = (
                        data['code'],
                        date,
                        features_json,
                        _to_sql(stock_labels.get('label_3d')),
                        _to_sql(stock_labels.get('label_5d')),
                        _to_sql(stock_labels.get('label_10d')),
                        _to_sql(stock_labels.get('label_15d')),
                    )
                else:
                    base_row = (
                        data['code'],
                        date,
                        features_json,
                        _to_sql(stock_labels.get('label_3d')),
                        _to_sql(stock_labels.get('label_5d')),
                        _to_sql(stock_labels.get('label_10d')),
                        _to_sql(stock_labels.get('label_15d')),
                        _to_sql(stock_labels.get('downside_10d')),
                    )

                if version_ge(self.schema_version, 'ng1.0.3'):
                    raw_cols = (
                        _to_sql(stock_labels.get('label_raw_3d')),
                        _to_sql(stock_labels.get('label_raw_5d')),
                        _to_sql(stock_labels.get('label_raw_10d')),
                        _to_sql(stock_labels.get('label_raw_15d')),
                    )
                else:
                    raw_cols = ()

                if version_ge(self.schema_version, 'ng1.0.4') and not is_12 and not is_13 and not is_15:
                    ng104_cols = (
                        _to_sql(stock_labels.get('maxdd_3d')),
                        _to_sql(stock_labels.get('maxdd_5d')),
                        _to_sql(stock_labels.get('maxdd_10d')),
                        _to_sql(stock_labels.get('maxdd_15d')),
                        _to_sql(stock_labels.get('ra_label_3d')),
                        _to_sql(stock_labels.get('ra_label_5d')),
                        _to_sql(stock_labels.get('ra_label_10d')),
                        _to_sql(stock_labels.get('ra_label_15d')),
                    )
                else:
                    ng104_cols = ()

                if version_ge(self.schema_version, 'ng1.0.7') and not is_12 and not is_13 and not is_15:
                    ng107_cols = (
                        _to_sql(stock_labels.get('cond_label_3d')),
                        _to_sql(stock_labels.get('cond_label_5d')),
                        _to_sql(stock_labels.get('cond_label_10d')),
                        _to_sql(stock_labels.get('cond_label_15d')),
                        _to_sql(amv_row.get('var1') if amv_row else None),
                        _to_sql(amv_row.get('macd') if amv_row else None),
                        _to_sql(amv_row.get('regime_days', 0) / 60.0 if amv_row else None),
                    )
                else:
                    ng107_cols = ()

                if is_12 and _version_in_range(self.schema_version, 'ng1.2.1', 'ng1.2.3'):
                    ng121_cols = (
                        _to_sql(stock_labels.get('vn_label_3d')),
                        _to_sql(stock_labels.get('vn_label_5d')),
                        _to_sql(stock_labels.get('vn_label_10d')),
                        _to_sql(stock_labels.get('vn_label_15d')),
                        _to_sql(stock_labels.get('path_mean_10d')),
                        _to_sql(stock_labels.get('path_std_10d')),
                        _to_sql(stock_labels.get('downside_std_10d')),
                    )
                else:
                    ng121_cols = ()

                # ng1.2.3: 4-horizon downside_kd columns (ng1.2.4 has no downside)
                if is_12 and _version_in_range(self.schema_version, 'ng1.2.3', 'ng1.2.4'):
                    ng123_cols = (
                        _to_sql(ng123_downside.get('downside_3d')),
                        _to_sql(ng123_downside.get('downside_5d')),
                        _to_sql(ng123_downside.get('downside_10d')),
                        _to_sql(ng123_downside.get('downside_15d')),
                    )
                else:
                    ng123_cols = ()

                # ng1.3.x / ng1.5.x: 4-horizon downside labels (min-cumret semantics)
                if is_13 or is_15:
                    ng130_cols = (
                        _to_sql(ng130_downside.get('downside_3d')),
                        _to_sql(ng130_downside.get('downside_5d')),
                        _to_sql(ng130_downside.get('downside_10d')),
                        _to_sql(ng130_downside.get('downside_15d')),
                    )
                else:
                    ng130_cols = ()

                market_cols = (
                    _to_sql(market_feats.get('market_return_5d')),
                    _to_sql(market_feats.get('market_return_20d')),
                    _to_sql(market_feats.get('market_volatility_20d')),
                    _to_sql(market_feats.get('market_breadth')),
                    _to_sql(market_feats.get('market_new_high_ratio')),
                    _to_sql(market_feats.get('northbound_flow_5d')),
                    _to_sql(market_feats.get('market_volume_ratio')),
                    _to_sql(market_feats.get('market_drawdown')),
                    _to_sql(market_feats.get('vix_proxy')),
                    _to_sql(market_feats.get('market_momentum_diff')),
                )

                insert_rows.append(
                    base_row + raw_cols + ng104_cols + ng107_cols
                    + ng121_cols + ng123_cols + ng130_cols + market_cols
                )

            # Write to database
            if insert_rows:
                conn.row_factory = None
                if is_13 or is_15:
                    # ng1.3.x / ng1.5.x: 25 columns — base(3) + labels(4) + label_raw(4) + downside_4(4) + market(10)
                    # Same column structure as ng1.2.3 but different downside semantics (min-cumret)
                    conn.executemany(
                        f'''INSERT OR REPLACE INTO {self.table_name}
                           (code, trade_date, features_json,
                            label_3d, label_5d, label_10d, label_15d,
                            label_raw_3d, label_raw_5d, label_raw_10d, label_raw_15d,
                            downside_3d, downside_5d, downside_10d, downside_15d,
                            market_return_5d, market_return_20d, market_volatility_20d,
                            market_breadth, market_new_high_ratio, northbound_flow_5d,
                            market_volume_ratio, market_drawdown, vix_proxy,
                            market_momentum_diff)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                        insert_rows
                    )
                elif is_12 and _version_in_range(self.schema_version, 'ng1.2.3', 'ng1.2.4'):
                    # 25 columns: base(3) + labels(4) + label_raw(4) + downside_kd(4) + market(10)
                    conn.executemany(
                        f'''INSERT OR REPLACE INTO {self.table_name}
                           (code, trade_date, features_json,
                            label_3d, label_5d, label_10d, label_15d,
                            label_raw_3d, label_raw_5d, label_raw_10d, label_raw_15d,
                            downside_3d, downside_5d, downside_10d, downside_15d,
                            market_return_5d, market_return_20d, market_volatility_20d,
                            market_breadth, market_new_high_ratio, northbound_flow_5d,
                            market_volume_ratio, market_drawdown, vix_proxy,
                            market_momentum_diff)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                        insert_rows
                    )
                elif is_12 and version_ge(self.schema_version, 'ng1.2.4'):
                    # 21 columns: base(3) + labels(4) + label_raw(4) + market(10)
                    # No downside columns (ng1.2.4 uses industry-excess label without penalty)
                    conn.executemany(
                        f'''INSERT OR REPLACE INTO {self.table_name}
                           (code, trade_date, features_json,
                            label_3d, label_5d, label_10d, label_15d,
                            label_raw_3d, label_raw_5d, label_raw_10d, label_raw_15d,
                            market_return_5d, market_return_20d, market_volatility_20d,
                            market_breadth, market_new_high_ratio, northbound_flow_5d,
                            market_volume_ratio, market_drawdown, vix_proxy,
                            market_momentum_diff)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                        insert_rows
                    )
                elif is_12 and _version_in_range(self.schema_version, 'ng1.2.1', 'ng1.2.3'):
                    # 29 columns: base(3) + labels(5, includes downside_10d)
                    #   + label_raw(4) + vn_label(4) + path_stats(3) + market(10)
                    conn.executemany(
                        f'''INSERT OR REPLACE INTO {self.table_name}
                           (code, trade_date, features_json,
                            label_3d, label_5d, label_10d, label_15d, downside_10d,
                            label_raw_3d, label_raw_5d, label_raw_10d, label_raw_15d,
                            vn_label_3d, vn_label_5d, vn_label_10d, vn_label_15d,
                            path_mean_10d, path_std_10d, downside_std_10d,
                            market_return_5d, market_return_20d, market_volatility_20d,
                            market_breadth, market_new_high_ratio, northbound_flow_5d,
                            market_volume_ratio, market_drawdown, vix_proxy,
                            market_momentum_diff)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                        insert_rows
                    )
                elif version_ge(self.schema_version, 'ng1.0.7') and not is_12:
                    # 37 columns: ng1.0.4 (30) + 7 ng1.0.7 columns (cond_label + amv)
                    conn.executemany(
                        f'''INSERT OR REPLACE INTO {self.table_name}
                           (code, trade_date, features_json,
                            label_3d, label_5d, label_10d, label_15d, downside_10d,
                            label_raw_3d, label_raw_5d, label_raw_10d, label_raw_15d,
                            maxdd_3d, maxdd_5d, maxdd_10d, maxdd_15d,
                            ra_label_3d, ra_label_5d, ra_label_10d, ra_label_15d,
                            cond_label_3d, cond_label_5d, cond_label_10d, cond_label_15d,
                            amv_var1, amv_macd, amv_regime_days,
                            market_return_5d, market_return_20d, market_volatility_20d,
                            market_breadth, market_new_high_ratio, northbound_flow_5d,
                            market_volume_ratio, market_drawdown, vix_proxy,
                            market_momentum_diff)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                        insert_rows
                    )
                elif version_ge(self.schema_version, 'ng1.0.4') and not is_12:
                    # 30 columns: ng1.0.3 columns + 8 ng1.0.4 columns (maxdd + ra_label)
                    conn.executemany(
                        f'''INSERT OR REPLACE INTO {self.table_name}
                           (code, trade_date, features_json,
                            label_3d, label_5d, label_10d, label_15d, downside_10d,
                            label_raw_3d, label_raw_5d, label_raw_10d, label_raw_15d,
                            maxdd_3d, maxdd_5d, maxdd_10d, maxdd_15d,
                            ra_label_3d, ra_label_5d, ra_label_10d, ra_label_15d,
                            market_return_5d, market_return_20d, market_volatility_20d,
                            market_breadth, market_new_high_ratio, northbound_flow_5d,
                            market_volume_ratio, market_drawdown, vix_proxy,
                            market_momentum_diff)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                        insert_rows
                    )
                elif version_ge(self.schema_version, 'ng1.0.3'):
                    # 22 columns: includes downside_10d + 4 label_raw columns
                    conn.executemany(
                        f'''INSERT OR REPLACE INTO {self.table_name}
                           (code, trade_date, features_json,
                            label_3d, label_5d, label_10d, label_15d, downside_10d,
                            label_raw_3d, label_raw_5d, label_raw_10d, label_raw_15d,
                            market_return_5d, market_return_20d, market_volatility_20d,
                            market_breadth, market_new_high_ratio, northbound_flow_5d,
                            market_volume_ratio, market_drawdown, vix_proxy,
                            market_momentum_diff)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                        insert_rows
                    )
                elif version_ge(self.schema_version, 'ng1.0.2'):
                    # 18 columns: includes downside_10d at position 7
                    conn.executemany(
                        f'''INSERT OR REPLACE INTO {self.table_name}
                           (code, trade_date, features_json,
                            label_3d, label_5d, label_10d, label_15d, downside_10d,
                            market_return_5d, market_return_20d, market_volatility_20d,
                            market_breadth, market_new_high_ratio, northbound_flow_5d,
                            market_volume_ratio, market_drawdown, vix_proxy,
                            market_momentum_diff)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                        insert_rows
                    )
                else:
                    # 17 columns: drop downside_10d (index 7) for backward compat
                    rows_no_ds = [r[:7] + r[8:] for r in insert_rows]
                    conn.executemany(
                        f'''INSERT OR REPLACE INTO {self.table_name}
                           (code, trade_date, features_json,
                            label_3d, label_5d, label_10d, label_15d,
                            market_return_5d, market_return_20d, market_volatility_20d,
                            market_breadth, market_new_high_ratio, northbound_flow_5d,
                            market_volume_ratio, market_drawdown, vix_proxy,
                            market_momentum_diff)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                        rows_no_ds
                    )
                conn.commit()

            elapsed = time.time() - t0
            print(f"  [{date}] Done: {len(insert_rows)} stocks written "
                  f"(skipped: {skipped_mv} mv_filter, {skipped_data} data_filter) "
                  f"in {elapsed:.1f}s")
            return len(insert_rows)

        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Backfill
    # ------------------------------------------------------------------

    def backfill(self, start_date: str, end_date: str) -> int:
        dates = self.get_trading_dates(start_date, end_date)
        print(f"NG {self.version} [{self.table_name}] Backfilling {len(dates)} dates from {start_date} to {end_date}")

        total = 0
        for i, d in enumerate(dates):
            print(f"[{i + 1}/{len(dates)}] Processing {d}...")
            try:
                count = self.update_single_date(d)
                total += count
            except Exception as e:
                print(f"  ERROR on {d}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\nBackfill complete: {total} total records across {len(dates)} dates")
        return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='NG Feature Cache Updater (version-aware)')
    parser.add_argument('--date', help='Single date to process (YYYY-MM-DD)')
    parser.add_argument('--start-date', help='Backfill start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Backfill end date (YYYY-MM-DD)')
    parser.add_argument('--db-path', help='Override database path')
    parser.add_argument('--version', default='ng1.0.3', help='NG version (default: ng1.0.3)')
    parser.add_argument('--penalty-power', type=float, default=1.5,
                        help='Risk-adjusted label penalty power (default: 1.5, ng1.0.4)')
    args = parser.parse_args()

    updater = NGCacheUpdater(db_path=args.db_path, version=args.version)
    updater.penalty_power = args.penalty_power

    if args.date:
        count = updater.update_single_date(args.date)
        print(f"\nProcessed {count} stocks for {args.date}")
    elif args.start_date and args.end_date:
        updater.backfill(args.start_date, args.end_date)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
