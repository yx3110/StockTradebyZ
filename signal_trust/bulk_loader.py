"""Bulk data pre-loader for rebuild. Loads daily_quotes + daily_basic once and provides O(1) enrichment lookups."""
import logging
from bisect import bisect_left

from .db import connect
from .constants import HOLD_DAYS, MARKET_CAP_BUCKETS

logger = logging.getLogger(__name__)


class BulkEnricher:
    """Holds pre-loaded market data; provides per-record lookups."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        # code -> (sorted dates list, closes list) — parallel lists for O(log n) bisect
        self._close_by_code: dict[str, tuple[list[str], list[float]]] = {}
        # (code, date) -> circ_mv in 万元
        self._circ_mv: dict[tuple[str, str], float] = {}
        # (code, date) -> 30-day rolling mean amount
        self._liq_mean: dict[tuple[str, str], float] = {}
        # all distinct trade dates, ascending
        self._all_trade_dates: list[str] = []

    def load(self, min_date: str | None = None) -> None:
        """Load all required data from daily_quotes + daily_basic. Optional min_date reduces memory."""
        self._load_quotes(min_date)
        self._load_circ_mv(min_date)

    def _load_quotes(self, min_date: str | None) -> None:
        logger.info("Bulk 预加载 daily_quotes...")
        conn = connect(self.db_path)
        try:
            where = " WHERE dq.trade_date >= ?" if min_date else ""
            params: list = [min_date] if min_date else []
            rows = conn.execute(
                "SELECT s.code, dq.trade_date, dq.close, "
                "       COALESCE(dq.amount, dq.volume * dq.close) AS amount "
                "FROM daily_quotes dq JOIN securities s ON s.id = dq.security_id"
                + where
                + " ORDER BY s.code, dq.trade_date",
                params,
            ).fetchall()
        finally:
            conn.close()
        logger.info(f"  daily_quotes: {len(rows):,} 行")

        # Accumulate raw lists per code
        tmp_close: dict[str, list[tuple[str, float]]] = {}
        tmp_amount: dict[str, list[tuple[str, float]]] = {}
        all_dates_set: set[str] = set()

        for r in rows:
            code = r["code"]
            date = r["trade_date"]
            all_dates_set.add(date)
            if r["close"] is not None:
                tmp_close.setdefault(code, []).append((date, r["close"]))
            if r["amount"] is not None:
                tmp_amount.setdefault(code, []).append((date, r["amount"]))

        # Convert to paired lists (already ordered due to ORDER BY)
        for code, seq in tmp_close.items():
            self._close_by_code[code] = (
                [d for d, _ in seq],
                [c for _, c in seq],
            )

        # Global trade-date calendar
        self._all_trade_dates = sorted(all_dates_set)
        logger.info(f"  股票: {len(self._close_by_code):,}; 交易日: {len(self._all_trade_dates):,}")

        # Pre-compute 30-day rolling mean amount per (code, date)
        logger.info("预计算 30日均成交额滚动值...")
        for code, seq in tmp_amount.items():
            window: list[float] = []
            window_sum = 0.0
            for date, amount in seq:
                window.append(amount)
                window_sum += amount
                if len(window) > 30:
                    window_sum -= window.pop(0)
                self._liq_mean[(code, date)] = window_sum / len(window)
        logger.info(f"  liquidity_mean 键: {len(self._liq_mean):,}")

    def _load_circ_mv(self, min_date: str | None) -> None:
        logger.info("Bulk 预加载 daily_basic (circ_mv)...")
        conn = connect(self.db_path)
        try:
            where = " WHERE db.trade_date >= ?" if min_date else ""
            params: list = [min_date] if min_date else []
            rows = conn.execute(
                "SELECT s.code, db.trade_date, db.circ_mv "
                "FROM daily_basic db JOIN securities s ON s.id = db.security_id"
                + where,
                params,
            ).fetchall()
        finally:
            conn.close()
        for r in rows:
            if r["circ_mv"] is not None:
                self._circ_mv[(r["code"], r["trade_date"])] = r["circ_mv"]
        logger.info(f"  circ_mv 键: {len(self._circ_mv):,}")

    # ------------------------------------------------------------------
    # Lookup API
    # ------------------------------------------------------------------

    def actual_10d(self, code: str, trade_date: str) -> float | None:
        """Return (close[T+10] - close[T]) / close[T], or None if unavailable."""
        pair = self._close_by_code.get(code)
        if pair is None:
            return None
        dates, closes = pair
        idx = bisect_left(dates, trade_date)
        if idx >= len(dates) or dates[idx] != trade_date:
            return None
        if idx + HOLD_DAYS >= len(dates):
            return None
        p0 = closes[idx]
        pN = closes[idx + HOLD_DAYS]
        if p0 is None or pN is None or p0 == 0:
            return None
        return (pN - p0) / p0

    def sample_end_date(self, trade_date: str) -> str | None:
        """Return the trading date HOLD_DAYS market days after trade_date, or None."""
        idx = bisect_left(self._all_trade_dates, trade_date)
        if idx >= len(self._all_trade_dates) or self._all_trade_dates[idx] != trade_date:
            return None
        if idx + HOLD_DAYS >= len(self._all_trade_dates):
            return None
        return self._all_trade_dates[idx + HOLD_DAYS]

    def market_cap_bucket(self, code: str, trade_date: str) -> str:
        """Map circ_mv to bucket label using MARKET_CAP_BUCKETS."""
        mv = self._circ_mv.get((code, trade_date))
        if mv is None:
            return "未知"
        for lo, hi, label in MARKET_CAP_BUCKETS:
            if lo <= mv < hi:
                return label
        return "未知"

    def liquidity_bucket(
        self, code: str, trade_date: str, thresholds: tuple[float, float, float]
    ) -> str:
        """Map 30-day mean amount to bucket label."""
        m = self._liq_mean.get((code, trade_date))
        if m is None:
            return "未知"
        p25, p50, p75 = thresholds
        if m < p25:
            return "低"
        elif m < p50:
            return "中低"
        elif m < p75:
            return "中高"
        return "高"

    def compute_liquidity_thresholds(self, as_of_date: str) -> tuple[float, float, float]:
        """Compute p25/p50/p75 of per-stock 30-day mean amount, as of as_of_date."""
        vals = sorted(v for (c, d), v in self._liq_mean.items() if d <= as_of_date)
        if len(vals) < 4:
            return (1e8, 3e8, 1e9)
        n = len(vals)
        return (vals[n // 4], vals[n // 2], vals[3 * n // 4])
