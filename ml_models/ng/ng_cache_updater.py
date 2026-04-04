"""
Daily Selection NG — Batch Feature Cache Updater
=================================================
Orchestrates DB data loading and 62-factor computation for all eligible A-stocks.

For a given date:
  1. Load 90-day lookback of OHLCV + MA data for all A-stocks
  2. Load technical indicators for the lookback period
  3. Load daily_basic for the date (and 60-day PE history)
  4. Load latest financial indicators (look back 400 days)
  5. Compute market features (CSI300 benchmark, breadth, northbound)
  6. Compute industry aggregates (group by securities.industry)
  7. Compute labels (open[T+1] -> close[T+N])
  8. Write 62-factor JSON + labels to ng_feature_cache

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

# Add project root to path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ml_models.ng.ng_schema import create_table, DB_PATH
from ml_models.ng.ng_feature_calculator import (
    compute_stock_features,
    compute_fundamental_features,
    compute_market_features,
    compute_industry_features,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOOKBACK_DAYS = 120       # Calendar days for price/volume lookback (~80 trading days)
TECH_LOOKBACK_DAYS = 10   # Extra tech indicator lookback for MACD acceleration
PE_HISTORY_DAYS = 60      # Trading days for PE percentile history
FIN_LOOKBACK_DAYS = 400   # Calendar days to find latest financial filing
CIRC_MV_MIN = 500000      # 50亿元 = 500000万元
MIN_DATA_DAYS = 60        # Minimum trading days for a stock to be eligible
BENCHMARK_CODE = '000300.SH'  # CSI300

# Label horizons (trading days after T+1 open)
LABEL_HORIZONS = [3, 5, 10, 15]


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------

def _json_default(obj):
    """Handle numpy types and NaN for JSON serialization."""
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
    """Convert DB value to float, handling None."""
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
    """Batch compute 62 NG factors and write to ng_feature_cache."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        create_table(self.db_path)

    def _connect(self) -> sqlite3.Connection:
        """Create a DB connection with proper settings."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute('PRAGMA busy_timeout = 30000')
        conn.execute('PRAGMA journal_mode = WAL')
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Trading date utilities
    # ------------------------------------------------------------------

    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """Query distinct trading dates from daily_quotes."""
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
        """Get the next n trading dates after `date`."""
        rows = conn.execute(
            '''SELECT DISTINCT trade_date FROM daily_quotes
               WHERE trade_date > ?
               ORDER BY trade_date LIMIT ?''',
            (date, n)
        ).fetchall()
        return [r['trade_date'] for r in rows]

    # ------------------------------------------------------------------
    # Bulk data loaders
    # ------------------------------------------------------------------

    def _load_stock_universe(self, conn: sqlite3.Connection) -> Dict[int, dict]:
        """Load A-stock security info: {security_id: {code, name, industry}}."""
        rows = conn.execute(
            '''SELECT id, code, name, industry FROM securities WHERE type = 'A股' '''
        ).fetchall()
        return {r['id']: dict(r) for r in rows}

    def _load_price_data(
        self, conn: sqlite3.Connection, date: str, security_ids: List[int]
    ) -> Dict[int, List[dict]]:
        """
        Load 90 calendar days of OHLCV+MA data for all given securities.
        Returns {security_id: [row_dicts sorted by trade_date ASC]}.
        """
        # Compute lookback start date
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
        """
        Load technical indicators for lookback period.
        We need multiple days of MACD for macd_acceleration.
        Returns {security_id: [rows sorted by trade_date ASC]}.
        """
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
        """Load daily_basic for the given date. Returns {security_id: row_dict}."""
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
        """Load 60-day PE TTM history for each security. Returns {sec_id: np.array}."""
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
        """
        Load the latest financial filing for each security (looking back 400 days).
        Also load the previous year's filing for ROE change computation.
        Returns {security_id: {roe, roe_prev_year, profit_to_gr, ...}}.

        Note: ann_date and end_date are stored as integers (YYYYMMDD) in the DB.
        """
        from datetime import datetime, timedelta
        dt = datetime.strptime(date, '%Y-%m-%d')
        lookback_start_dt = dt - timedelta(days=FIN_LOOKBACK_DAYS)
        # Convert to integer format matching DB storage
        date_int = int(date.replace('-', ''))
        lookback_start_int = int(lookback_start_dt.strftime('%Y%m%d'))

        # Get all filings in the lookback window
        rows = conn.execute(
            f'''SELECT security_id, ann_date, end_date, roe, profit_to_gr,
                       netprofit_margin, ocf_to_profit, debt_to_assets, current_ratio
                FROM financial_indicator
                WHERE ann_date BETWEEN ? AND ?
                  AND security_id IN ({','.join('?' * len(security_ids))})
                ORDER BY security_id, end_date DESC''',
            [lookback_start_int, date_int] + security_ids
        ).fetchall()

        # Group by security, pick latest and second-latest by end_date
        grouped: Dict[int, List[dict]] = defaultdict(list)
        for r in rows:
            # Only include filings announced on or before report date
            ann = r['ann_date']
            if ann is not None and int(ann) <= date_int:
                grouped[r['security_id']].append(dict(r))

        result = {}
        for sid, filings in grouped.items():
            if not filings:
                continue
            latest = filings[0]  # Already sorted DESC by end_date
            # Find previous year's filing (different end_date year)
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
        """Load CSI300 closing prices for 90-day lookback."""
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

    def _load_northbound_data(
        self, conn: sqlite3.Connection, date: str
    ) -> Tuple[float, float]:
        """
        Load northbound flow data. Returns (net_buy_5d_sum, historical_std).
        north_money is daily total in 万元.
        """
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
            # Table may not exist
            return 0.0, 1.0

        if not rows:
            return 0.0, 1.0

        vals = [_safe_float(r['north_money'], 0.0) for r in rows]
        arr = np.array(vals)

        # 5-day sum (latest 5 values)
        net_buy_5d = float(np.sum(arr[-5:])) if len(arr) >= 5 else float(np.sum(arr))
        # Historical std of daily values
        nb_std = float(arr.std()) if len(arr) >= 5 else 1.0
        if nb_std < 1e-8:
            nb_std = 1.0

        return net_buy_5d, nb_std

    def _load_market_amounts(
        self, conn: sqlite3.Connection, date: str
    ) -> np.ndarray:
        """Load total market daily turnover for last 90 days (sum across all A-stocks).
        Uses COALESCE(amount, volume) since amount is often NULL in this DB."""
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

    def _load_future_prices(
        self, conn: sqlite3.Connection, future_dates: List[str],
        security_ids: List[int]
    ) -> Dict[int, Dict[str, dict]]:
        """
        Load open and close prices for future dates.
        Returns {security_id: {date_str: {'open': ..., 'close': ...}}}.
        """
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
        """Compute n-day returns for each stock. Returns {sec_id: return}."""
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
        """Compute 1-day returns for all stocks."""
        return self._compute_stock_returns(price_data, 1)

    def _compute_highs_20d_ratio(self, price_data: Dict[int, List[dict]]) -> Dict[int, float]:
        """Compute close / max(high, 20d) for all stocks."""
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

    def _compute_industry_aggregates(
        self,
        universe: Dict[int, dict],
        price_data: Dict[int, List[dict]],
        returns_1d: Dict[int, float],
        returns_5d: Dict[int, float],
        returns_20d: Dict[int, float],
    ) -> Dict[str, dict]:
        """
        Group stocks by industry and compute industry-level aggregates.
        Returns {industry: {returns_1d: array, returns_5d: array, ...}}.
        """
        # Group security IDs by industry
        industry_sids: Dict[str, List[int]] = defaultdict(list)
        for sid, info in universe.items():
            ind = info.get('industry') or 'unknown'
            industry_sids[ind].append(sid)

        # Compute per-industry aggregates
        result = {}
        for ind, sids in industry_sids.items():
            ind_ret1d = [returns_1d[s] for s in sids if s in returns_1d and not np.isnan(returns_1d[s])]
            ind_ret5d = [returns_5d[s] for s in sids if s in returns_5d and not np.isnan(returns_5d[s])]
            ind_ret20d = [returns_20d[s] for s in sids if s in returns_20d and not np.isnan(returns_20d[s])]

            # Industry amounts for last 5d and 20d (use COALESCE of amount/volume)
            amounts_5d = []
            amounts_20d = []
            for sid in sids:
                rows = price_data.get(sid, [])
                if len(rows) >= 5:
                    amt = sum(_safe_float(r['amount'] if r['amount'] is not None else r['volume'], 0) for r in rows[-5:])
                    amounts_5d.append(amt)
                if len(rows) >= 20:
                    amt = sum(_safe_float(r['amount'] if r['amount'] is not None else r['volume'], 0) for r in rows[-20:])
                    amounts_20d.append(amt)

            result[ind] = {
                'returns_1d': np.array(ind_ret1d) if ind_ret1d else np.array([]),
                'returns_5d': np.array(ind_ret5d) if ind_ret5d else np.array([]),
                'returns_20d': np.array(ind_ret20d) if ind_ret20d else np.array([]),
                'amounts_5d': np.array(amounts_5d) if amounts_5d else np.array([]),
                'amounts_20d': np.array(amounts_20d) if amounts_20d else np.array([]),
                'mean_return_5d': float(np.mean(ind_ret5d)) if ind_ret5d else np.nan,
            }

        return result

    def _compute_labels(
        self, future_dates: List[str], future_prices: Dict[int, Dict[str, dict]],
        security_ids: List[int]
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute forward-looking labels: label_Nd = close[T+1+N] / open[T+1] - 1.
        Returns {security_id: {'label_3d': ..., 'label_5d': ..., 'label_10d': ...}}.
        """
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

            labels = {}
            for n in LABEL_HORIZONS:
                # T+1+N means we need future_dates[n] (0-indexed: T+1 is [0])
                if n < len(future_dates) and future_dates[n] in fp:
                    future_close = fp[future_dates[n]].get('close', np.nan)
                    if not np.isnan(future_close) and future_close > 0:
                        labels[f'label_{n}d'] = future_close / base - 1.0
                    else:
                        labels[f'label_{n}d'] = np.nan
                else:
                    labels[f'label_{n}d'] = np.nan
            result[sid] = labels

        return result

    # ------------------------------------------------------------------
    # Main entry point: process a single date
    # ------------------------------------------------------------------

    def update_single_date(self, date: str) -> int:
        """
        Compute 62 NG factors for all eligible A-stocks on the given date.
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

            # Split security_ids into chunks for SQL IN clause (SQLite limit ~999 params)
            chunk_size = 900

            # 2. Load price data (90-day lookback) for all stocks
            print(f"  [{date}] Loading price data for {len(all_sids)} stocks...")
            price_data: Dict[int, List[dict]] = {}
            for i in range(0, len(all_sids), chunk_size):
                chunk = all_sids[i:i + chunk_size]
                chunk_data = self._load_price_data(conn, date, chunk)
                price_data.update(chunk_data)

            # Filter: only stocks that have data on this date
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

            # 5. Load PE history (60d)
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

            # 7. Load benchmark (CSI300)
            benchmark_closes = self._load_benchmark_data(conn, date)

            # 8. Load northbound flow
            nb_5d, nb_std = self._load_northbound_data(conn, date)

            # 9. Load total market amounts
            market_amounts = self._load_market_amounts(conn, date)

            # 10. Compute stock-level returns for market/industry features
            returns_1d = self._compute_1d_returns(price_data)
            returns_5d = self._compute_stock_returns(price_data, 5)
            returns_20d = self._compute_stock_returns(price_data, 20)
            highs_20d_ratio = self._compute_highs_20d_ratio(price_data)

            # 11. Compute market features (once for all stocks)
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

            # All-industry 5d returns for percentile ranking
            all_ind_ret5d = np.array([
                v['mean_return_5d'] for v in industry_agg.values()
                if not np.isnan(v['mean_return_5d'])
            ])

            # 13. Compute labels (future returns)
            future_dates = self._get_future_dates(conn, date, max(LABEL_HORIZONS) + 1)
            future_prices: Dict[int, Dict[str, dict]] = {}
            if future_dates:
                for i in range(0, len(active_sids), chunk_size):
                    chunk = active_sids[i:i + chunk_size]
                    chunk_data = self._load_future_prices(conn, future_dates, chunk)
                    future_prices.update(chunk_data)

            labels_all = self._compute_labels(future_dates, future_prices, active_sids)

            # 14. Apply market cap filter and compute per-stock features
            print(f"  [{date}] Computing features...")
            insert_rows = []
            skipped_mv = 0
            skipped_data = 0

            for sid in active_sids:
                rows = price_data.get(sid, [])
                if len(rows) < MIN_DATA_DAYS:
                    skipped_data += 1
                    continue

                # Market cap filter
                db = daily_basic.get(sid)
                if db is None:
                    skipped_mv += 1
                    continue
                circ_mv = _safe_float(db.get('circ_mv'))
                if np.isnan(circ_mv) or circ_mv < CIRC_MV_MIN:
                    skipped_mv += 1
                    continue

                info = universe[sid]
                code = info['code']
                industry = info.get('industry') or 'unknown'

                # Prepare arrays from price_data
                closes = np.array([_safe_float(r['close']) for r in rows])
                opens = np.array([_safe_float(r['open']) for r in rows])
                highs = np.array([_safe_float(r['high']) for r in rows])
                lows = np.array([_safe_float(r['low']) for r in rows])
                volumes = np.array([_safe_float(r['volume']) for r in rows])
                # amount is often NULL in DB; fall back to volume (which often
                # stores turnover value in this DB schema)
                amounts_raw = [r['amount'] for r in rows]
                if all(a is None for a in amounts_raw):
                    amounts = volumes.copy()  # Use volume as amount proxy
                else:
                    amounts = np.array([_safe_float(a, 0) for a in amounts_raw])
                ma5_arr = np.array([_safe_float(r['ma5']) for r in rows])
                ma10_arr = np.array([_safe_float(r['ma10']) for r in rows])
                ma20_arr = np.array([_safe_float(r['ma20']) for r in rows])
                ma60_arr = np.array([_safe_float(r['ma60']) for r in rows])

                # Technical indicators - build MACD array from lookback data
                tech_rows = tech_data.get(sid, [])
                tech_today = None
                macd_arr = np.full(len(rows), np.nan)

                if tech_rows:
                    # Build date->tech mapping
                    tech_by_date = {tr['trade_date']: tr for tr in tech_rows}
                    # Get today's tech values
                    tech_today = tech_by_date.get(date)
                    # Fill MACD array aligned with price dates
                    for i_r, r in enumerate(rows):
                        td = r['trade_date']
                        if td in tech_by_date:
                            macd_arr[i_r] = _safe_float(tech_by_date[td].get('macd_macd'))

                # Extract today's technical scalars (with defaults)
                atr_14 = _safe_float(tech_today['atr_14']) if tech_today else np.nan
                kdj_j = _safe_float(tech_today['kdj_j']) if tech_today else 50.0
                boll_upper = _safe_float(tech_today['boll_upper']) if tech_today else _safe_float(rows[-1]['close'])
                boll_lower = _safe_float(tech_today['boll_lower']) if tech_today else _safe_float(rows[-1]['close'])
                rsi12 = _safe_float(tech_today['rsi12']) if tech_today else 50.0
                rsi24 = _safe_float(tech_today['rsi24']) if tech_today else 50.0

                if np.isnan(atr_14):
                    # Fallback ATR: average true range over last 14 days
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

                # --- Compute stock features (1-30) ---
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

                # --- Compute fundamental features (31-44) ---
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

                # --- Compute industry features (55-62) ---
                ind_agg = industry_agg.get(industry, {})
                stock_ret_20d = returns_20d.get(sid, np.nan)

                # SW index return: use industry mean as proxy
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
                    )
                except Exception as e:
                    print(f"    WARN: industry_features failed for {code}: {e}")
                    ind_feats = {}

                # --- Merge all features ---
                all_feats = {}
                all_feats.update(stock_feats)
                all_feats.update(fund_feats)   # Overrides turnover_rate placeholder
                all_feats.update(market_feats)
                all_feats.update(ind_feats)

                # Clean NaN/Inf before JSON serialization
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

                # Labels
                stock_labels = labels_all.get(sid, {})

                # Build insert row (clean NaN to None for SQLite)
                def _to_sql(v):
                    if v is None:
                        return None
                    if isinstance(v, (float, np.floating)):
                        if np.isnan(v) or np.isinf(v):
                            return None
                        return float(v)
                    return v

                insert_rows.append((
                    code,
                    date,
                    features_json,
                    _to_sql(stock_labels.get('label_3d')),
                    _to_sql(stock_labels.get('label_5d')),
                    _to_sql(stock_labels.get('label_10d')),
                    _to_sql(stock_labels.get('label_15d')),
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
                ))

            # 15. Write to database
            if insert_rows:
                conn.row_factory = None  # Reset for executemany
                conn.executemany(
                    '''INSERT OR REPLACE INTO ng_feature_cache
                       (code, trade_date, features_json,
                        label_3d, label_5d, label_10d, label_15d,
                        market_return_5d, market_return_20d, market_volatility_20d,
                        market_breadth, market_new_high_ratio, northbound_flow_5d,
                        market_volume_ratio, market_drawdown, vix_proxy,
                        market_momentum_diff)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                    insert_rows
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
        """Loop over trading dates and update each one."""
        dates = self.get_trading_dates(start_date, end_date)
        print(f"Backfilling {len(dates)} trading dates from {start_date} to {end_date}")

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
    parser = argparse.ArgumentParser(description='NG Feature Cache Updater')
    parser.add_argument('--date', help='Single date to process (YYYY-MM-DD)')
    parser.add_argument('--start-date', help='Backfill start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Backfill end date (YYYY-MM-DD)')
    parser.add_argument('--db-path', help='Override database path')
    args = parser.parse_args()

    updater = NGCacheUpdater(db_path=args.db_path)

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
