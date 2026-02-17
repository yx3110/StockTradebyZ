from __future__ import annotations

import logging
import sqlite3
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, List, Optional

import pandas as pd

# --------------------------- 日志配置 --------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("db_manager")

# --------------------------- SQL 常量 --------------------------- #
_DDL_KLINES = """
CREATE TABLE IF NOT EXISTS klines (
    code   TEXT NOT NULL,
    date   TEXT NOT NULL,
    open   REAL,
    close  REAL,
    high   REAL,
    low    REAL,
    volume REAL,
    PRIMARY KEY (code, date)
)
"""

_DDL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_klines_code ON klines(code)",
    "CREATE INDEX IF NOT EXISTS idx_klines_date ON klines(date)",
]

_UPSERT_SQL = (
    "INSERT OR REPLACE INTO klines (code, date, open, close, high, low, volume) "
    "VALUES (?, ?, ?, ?, ?, ?, ?)"
)

_DEFAULT_BATCH_SIZE = 5000


class DatabaseManager:
    """SQLite connection manager with per-thread connection reuse and WAL mode.

    Each thread gets its own :class:`sqlite3.Connection` object stored in
    thread-local storage so connections are never shared between threads.
    WAL mode is enabled on every new connection, enabling concurrent reads
    during writes.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Use ``':memory:'`` for an
        in-memory database (useful for testing).
    batch_size:
        Number of rows to insert per ``executemany`` batch when writing
        stock data.
    """

    def __init__(self, db_path: str | Path, batch_size: int = _DEFAULT_BATCH_SIZE) -> None:
        self.db_path = str(db_path)
        self.batch_size = batch_size
        self._local = threading.local()
        self.init_db()

    # ------------------------------------------------------------------
    # Internal connection management
    # ------------------------------------------------------------------

    def _get_thread_connection(self) -> sqlite3.Connection:
        """Return the current thread's SQLite connection, creating it if needed."""
        conn: Optional[sqlite3.Connection] = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # Performance & WAL pragmas
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")
            self._local.conn = conn
            logger.debug("New SQLite connection created for thread %s → %s",
                         threading.current_thread().name, self.db_path)
        return conn

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that yields the thread-local connection.

        Commits on successful exit, rolls back on exception.

        Yields
        ------
        sqlite3.Connection
        """
        conn = self._get_thread_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def init_db(self) -> None:
        """Create the ``klines`` table and indexes if they do not exist."""
        with self.get_connection() as conn:
            conn.execute(_DDL_KLINES)
            for ddl in _DDL_INDEXES:
                conn.execute(ddl)
        logger.info("Database initialised: %s", self.db_path)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save_stock_data(self, code: str, df: pd.DataFrame) -> None:
        """Upsert OHLCV rows for *code* from a pandas DataFrame.

        Parameters
        ----------
        code:
            Stock code, e.g. ``'000001'``.
        df:
            DataFrame with columns ``date, open, close, high, low, volume``.
            The ``date`` column may be :class:`pandas.Timestamp` or string;
            it will be stored as an ISO-8601 date string (``YYYY-MM-DD``).
        """
        if df is None or df.empty:
            logger.debug("save_stock_data(%s): empty DataFrame, skipping.", code)
            return

        required = {"date", "open", "close", "high", "low", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")

        def _to_date_str(val) -> str:
            if isinstance(val, str):
                return val[:10]
            return pd.Timestamp(val).strftime("%Y-%m-%d")

        rows = [
            (
                code,
                _to_date_str(row.date),
                float(row.open)   if pd.notna(row.open)   else None,
                float(row.close)  if pd.notna(row.close)  else None,
                float(row.high)   if pd.notna(row.high)   else None,
                float(row.low)    if pd.notna(row.low)    else None,
                float(row.volume) if pd.notna(row.volume) else None,
            )
            for row in df.itertuples(index=False)
        ]

        with self.get_connection() as conn:
            for start in range(0, len(rows), self.batch_size):
                chunk = rows[start : start + self.batch_size]
                conn.executemany(_UPSERT_SQL, chunk)

        logger.debug("save_stock_data(%s): %d rows written.", code, len(rows))

    def upsert_klines(self, code: str, df: pd.DataFrame) -> None:
        """Alias for :meth:`save_stock_data` for use by incremental updaters.

        Parameters
        ----------
        code:
            Stock code, e.g. ``'000001'``.
        df:
            DataFrame with columns ``date, open, close, high, low, volume``.
        """
        self.save_stock_data(code, df)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_last_date(self, code: str) -> Optional[str]:
        """Return the latest stored date for *code* as a ``YYYY-MM-DD`` string.

        Returns ``None`` if no data exists for the given code.

        Parameters
        ----------
        code:
            Stock code, e.g. ``'000001'``.

        Returns
        -------
        str or None
            Last stored date in ``YYYY-MM-DD`` format, or ``None`` if the
            code has no data in the database.
        """
        sql = "SELECT MAX(date) FROM klines WHERE code = ?"
        with self.get_connection() as conn:
            cur = conn.execute(sql, (code,))
            row = cur.fetchone()
        if row is None or row[0] is None:
            return None
        return str(row[0])[:10]

    def load_stock_data(self, code: str) -> pd.DataFrame:
        """Load all OHLCV rows for *code* sorted by date.

        Returns
        -------
        pd.DataFrame
            Columns: ``date (datetime64), open, close, high, low, volume``.
            Empty DataFrame if the code has no data.
        """
        sql = (
            "SELECT date, open, close, high, low, volume "
            "FROM klines WHERE code = ? ORDER BY date"
        )
        with self.get_connection() as conn:
            cur = conn.execute(sql, (code,))
            rows = cur.fetchall()

        if not rows:
            return pd.DataFrame(columns=["date", "open", "close", "high", "low", "volume"])

        df = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume"])
        df["date"] = pd.to_datetime(df["date"])
        for col in ("open", "close", "high", "low", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.reset_index(drop=True)

    def load_all_stocks_data(
        self, codes: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Load OHLCV data for multiple stocks.

        Parameters
        ----------
        codes:
            List of stock codes to load.  If ``None``, loads all codes
            present in the database.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping from stock code to its OHLCV DataFrame.
        """
        if codes is None:
            codes = self.get_all_codes()

        result: Dict[str, pd.DataFrame] = {}
        for code in codes:
            df = self.load_stock_data(code)
            if not df.empty:
                result[code] = df
        return result

    def get_all_codes(self) -> List[str]:
        """Return a sorted list of all distinct stock codes in the database."""
        sql = "SELECT DISTINCT code FROM klines ORDER BY code"
        with self.get_connection() as conn:
            cur = conn.execute(sql)
            return [row[0] for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close_all(self) -> None:
        """Close the current thread's connection, if open.

        Call this from each thread when it finishes to release file handles.
        Note: connections in other threads are *not* affected; each thread
        must call ``close_all()`` itself (or the OS will reclaim them on
        process exit).
        """
        conn: Optional[sqlite3.Connection] = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception as exc:  # pragma: no cover
                logger.warning("Error closing connection: %s", exc)
            finally:
                self._local.conn = None
            logger.debug("Connection closed for thread %s.", threading.current_thread().name)
