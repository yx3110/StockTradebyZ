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

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ml_models.ng.ng_schema import create_table, get_table_name, DB_PATH, create_moneyflow_table
from ml_models.ng.ng_feature_calculator import (
    compute_stock_features,
    compute_fundamental_features,
    compute_market_features,
    compute_industry_features,
    compute_cross_sectional_rank_features,
    compute_residual_features,
    compute_moneyflow_features,
    compute_interaction_features,
)
from sklearn.linear_model import LinearRegression
from fetch_data.label_utils import compute_labels_from_future_prices


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


# ---------------------------------------------------------------------------
# NGCacheUpdater
# ---------------------------------------------------------------------------

class NGCacheUpdater:
    """Batch compute NG factors and write to version-specific cache table."""

    def __init__(self, db_path: str = None, version: str = 'ng1.0.3'):
        self.db_path = db_path or DB_PATH
        self.version = version
        self.table_name = get_table_name(version)
        create_table(self.db_path, version=version)
        if version >= 'ng1.0.3':
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

    def _load_financial_data(
        self, conn: sqlite3.Connection, date: str, security_ids: List[int]
    ) -> Dict[int, dict]:
        from datetime import datetime, timedelta
        dt = datetime.strptime(date, '%Y-%m-%d')
        lookback_start_dt = dt - timedelta(days=FIN_LOOKBACK_DAYS)
        date_int = int(date.replace('-', ''))
        lookback_start_int = int(lookback_start_dt.strftime('%Y%m%d'))

        rows = conn.execute(
            f'''SELECT security_id, ann_date, end_date, roe, profit_to_gr,
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

        # Query moneyflow_daily in chunks
        result: Dict[str, list] = defaultdict(list)
        for i in range(0, len(codes), chunk_size):
            chunk = codes[i:i + chunk_size]
            rows = conn.execute(
                f"""SELECT code, trade_date, buy_sm_amount, sell_sm_amount,
                           buy_md_amount, sell_md_amount, buy_lg_amount, sell_lg_amount,
                           buy_elg_amount, sell_elg_amount, net_mf_amount
                    FROM moneyflow_daily
                    WHERE trade_date BETWEEN ? AND ?
                      AND code IN ({','.join('?' * len(chunk))})
                    ORDER BY code, trade_date""",
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

        result = {}
        for sid in security_ids:
            fp = future_prices.get(sid, {})
            if not fp or future_dates[0] not in fp:
                continue

            base = fp[future_dates[0]].get('open', np.nan)
            if np.isnan(base) or base < 1e-8:
                continue

            # 构造 {horizon: close_price} 字典
            future_closes = {}
            for n in LABEL_HORIZONS:
                if n < len(future_dates) and future_dates[n] in fp:
                    future_closes[n] = fp[future_dates[n]].get('close', np.nan)

            labels = compute_labels_from_future_prices(
                base_open=base,
                future_closes=future_closes,
                horizons=tuple(LABEL_HORIZONS),
            )
            result[sid] = labels

        # ng1.0.4: Compute max drawdown for each security
        if getattr(self, 'version', '') >= 'ng1.0.4':
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
                if isinstance(excess, float) and np.isnan(excess):
                    labs[ra_key] = np.nan
                    continue
                if isinstance(maxdd, float) and np.isnan(maxdd):
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
            if self.version >= 'ng1.0.3':
                mf_count = self._fetch_and_store_moneyflow(conn, date)
                if mf_count > 0:
                    print(f"  [{date}] Fetched {mf_count} moneyflow records")
                mf_data = self._load_moneyflow_data(conn, date, active_sids, universe=universe, n_days=20)
            else:
                mf_data = {}

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
            if self.version >= 'ng1.0.3':
                # Inject circ_mv into universe for residual regression
                for sid in active_sids:
                    db = daily_basic.get(sid)
                    if db:
                        universe[sid]['circ_mv'] = _safe_float(db.get('circ_mv'))
                labels_all = self._convert_labels_to_residual(
                    labels_all, universe, price_data, returns_20d, stock_volatilities
                )

            # ng1.0.4: Compute risk-adjusted labels
            if self.version >= 'ng1.0.4':
                pp = getattr(self, 'penalty_power', 1.5)
                labels_all = self._convert_labels_to_risk_adjusted(labels_all, penalty_power=pp)

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
                if self.version >= 'ng1.0.4':
                    from ml_models.ng.ng_feature_calculator import compute_smoothing_features
                    try:
                        smooth_feats = compute_smoothing_features(
                            closes=closes, opens=opens, highs=highs,
                            lows=lows, volumes=volumes,
                        )
                    except Exception as e:
                        print(f"    WARN: smoothing_features failed for {code}: {e}")
                        smooth_feats = {}

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
                if self.version >= 'ng1.0.3':
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

                # Store raw values needed for CS rank (pass 2)
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
                avg_vol_5d = float(np.mean(amounts[-5:])) if len(amounts) >= 5 else np.nan

                eligible_stocks[sid] = {
                    'code': code,
                    'industry': industry,
                    'stock_feats': stock_feats,
                    'fund_feats': fund_feats,
                    'ind_feats': ind_feats,
                    'mf_feats': mf_feats,
                    'smooth_feats': smooth_feats,
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
                    # For residual factors
                    'daily_returns': stock_daily_returns.get(sid),
                    'avg_volume_5d': avg_vol_5d,
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
                'market_cap', 'pe',
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
                if self.version >= 'ng1.0.3':
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

                # --- Merge all features ---
                all_feats = {}
                all_feats.update(data['stock_feats'])
                all_feats.update(data['fund_feats'])
                all_feats.update(data['ind_feats'])
                all_feats.update(cs_feats)
                all_feats.update(res_feats)
                if self.version >= 'ng1.0.3':
                    all_feats.update(data.get('mf_feats', {}))
                    all_feats.update(ix_feats)
                if self.version >= 'ng1.0.4':
                    all_feats.update(data.get('smooth_feats', {}))

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

                # Labels (industry excess returns)
                stock_labels = labels_all.get(sid, {})

                def _to_sql(v):
                    if v is None:
                        return None
                    if isinstance(v, (float, np.floating)):
                        if np.isnan(v) or np.isinf(v):
                            return None
                        return float(v)
                    return v

                # Build base row tuple
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

                # v1.0.3: add label_raw columns
                if self.version >= 'ng1.0.3':
                    raw_cols = (
                        _to_sql(stock_labels.get('label_raw_3d')),
                        _to_sql(stock_labels.get('label_raw_5d')),
                        _to_sql(stock_labels.get('label_raw_10d')),
                        _to_sql(stock_labels.get('label_raw_15d')),
                    )
                else:
                    raw_cols = ()

                # ng1.0.4: add maxDD + RA label columns
                if self.version >= 'ng1.0.4':
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

                insert_rows.append(base_row + raw_cols + ng104_cols + market_cols)

            # Write to database
            if insert_rows:
                conn.row_factory = None
                if self.version >= 'ng1.0.4':
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
                elif self.version >= 'ng1.0.3':
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
                elif self.version >= 'ng1.0.2':
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
