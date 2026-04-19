"""CRUD for saved_queries on webapp.db + idempotent seed."""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS saved_queries (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL UNIQUE,
    sql          TEXT    NOT NULL,
    tags         TEXT,
    description  TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_run_at  TIMESTAMP,
    run_count    INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_saved_queries_tags ON saved_queries(tags);
"""


_COLS = [
    "id", "name", "sql", "tags", "description",
    "created_at", "updated_at", "last_run_at", "run_count",
]


def apply_migration(webapp_db_path: str | Path) -> None:
    """Idempotent: create saved_queries table if missing."""
    webapp_db_path = Path(webapp_db_path)
    webapp_db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(webapp_db_path, timeout=30.0)
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        conn.executescript(MIGRATION_SQL)
        conn.commit()
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row | tuple) -> dict[str, Any]:
    return dict(zip(_COLS, row))


def _connect(db: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db, timeout=30.0)
    conn.execute("PRAGMA busy_timeout=30000")
    conn.row_factory = sqlite3.Row
    return conn


def create_query(
    db: str | Path, *, name: str, sql: str,
    tags: str | None, description: str | None,
) -> dict[str, Any]:
    conn = _connect(db)
    try:
        try:
            cur = conn.execute(
                "INSERT INTO saved_queries (name, sql, tags, description) "
                "VALUES (?, ?, ?, ?)",
                (name, sql, tags, description),
            )
            conn.commit()
        except sqlite3.IntegrityError as e:
            raise ValueError(f"saved query name already exists: {name}") from e
        return get_query(db, cur.lastrowid, _conn=conn)
    finally:
        conn.close()


def get_query(
    db: str | Path, query_id: int, *, _conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    conn = _conn or _connect(db)
    try:
        row = conn.execute(
            "SELECT id,name,sql,tags,description,created_at,updated_at,"
            "last_run_at,run_count FROM saved_queries WHERE id=?",
            (query_id,),
        ).fetchone()
        if row is None:
            raise LookupError(f"saved query id={query_id} not found")
        return _row_to_dict(tuple(row))
    finally:
        if _conn is None:
            conn.close()


def list_queries(
    db: str | Path, tag: str | None = None,
) -> list[dict[str, Any]]:
    conn = _connect(db)
    try:
        if tag:
            rows = conn.execute(
                "SELECT id,name,sql,tags,description,created_at,updated_at,"
                "last_run_at,run_count FROM saved_queries "
                "WHERE ',' || tags || ',' LIKE ? ORDER BY updated_at DESC",
                (f"%,{tag},%",),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id,name,sql,tags,description,created_at,updated_at,"
                "last_run_at,run_count FROM saved_queries "
                "ORDER BY updated_at DESC"
            ).fetchall()
        return [_row_to_dict(tuple(r)) for r in rows]
    finally:
        conn.close()


def update_query(
    db: str | Path, query_id: int, **fields: Any,
) -> dict[str, Any]:
    allowed = {"name", "sql", "tags", "description"}
    changes = {k: v for k, v in fields.items() if k in allowed}
    if not changes:
        return get_query(db, query_id)
    set_clause = ", ".join(f"{k}=?" for k in changes)
    params = list(changes.values()) + [query_id]
    conn = _connect(db)
    try:
        conn.execute(
            f"UPDATE saved_queries SET {set_clause}, "
            "updated_at=CURRENT_TIMESTAMP WHERE id=?",
            params,
        )
        conn.commit()
        return get_query(db, query_id, _conn=conn)
    finally:
        conn.close()


def delete_query(db: str | Path, query_id: int) -> None:
    conn = _connect(db)
    try:
        conn.execute("DELETE FROM saved_queries WHERE id=?", (query_id,))
        conn.commit()
    finally:
        conn.close()


def touch_query(db: str | Path, query_id: int) -> None:
    conn = _connect(db)
    try:
        conn.execute(
            "UPDATE saved_queries SET run_count=run_count+1, "
            "last_run_at=CURRENT_TIMESTAMP WHERE id=?",
            (query_id,),
        )
        conn.commit()
    finally:
        conn.close()


# -- seed ---------------------------------------------------------------------

_SEED: list[tuple[str, str, str, str]] = [
    (
        "preset: single-stock all-features",
        "SELECT * FROM ng101_feature_cache\n"
        "WHERE code = '600519.SH'\n"
        "  AND trade_date >= date('now', '-60 days')\n"
        "ORDER BY trade_date DESC",
        "preset,stock",
        "Mode A: every ng101 feature for a single stock over 60 days",
    ),
    (
        "preset: cross-section pred_10d top50",
        "SELECT code, trade_date, label_10d, features_json\n"
        "FROM ng101_feature_cache\n"
        "WHERE trade_date = (SELECT MAX(trade_date) FROM ng101_feature_cache)\n"
        "ORDER BY label_10d DESC\n"
        "LIMIT 50",
        "preset,cross",
        "Mode B: cross-section — top 50 by label_10d on latest day",
    ),
    (
        "preset: model compare ng101 vs ng110",
        "SELECT a.code, a.trade_date,\n"
        "       a.label_10d AS ng101_label10, b.label_10d AS ng110_label10\n"
        "FROM ng101_feature_cache a\n"
        "JOIN ng110_feature_cache b\n"
        "  ON a.code = b.code AND a.trade_date = b.trade_date\n"
        "WHERE a.trade_date = (SELECT MAX(trade_date) FROM ng101_feature_cache)\n"
        "ORDER BY a.label_10d DESC\n"
        "LIMIT 100",
        "preset,compare",
        "Mode C: ng101 vs ng110 label_10d on latest day",
    ),
    (
        "preset: signal_trust 🔴 tagged stocks",
        "SELECT code, as_of_date, n_samples, direction_hit_rate, trust_tag\n"
        "FROM signal_trust_scores\n"
        "WHERE trust_tag = '🔴'\n"
        "ORDER BY n_samples DESC\n"
        "LIMIT 50",
        "preset",
        "Stocks flagged 🔴 (low-trust) by signal_trust system",
    ),
]


def seed_default_queries(db: str | Path) -> int:
    """Insert SEED rows when saved_queries is empty. Returns rows inserted."""
    conn = _connect(db)
    try:
        (count,) = conn.execute("SELECT COUNT(*) FROM saved_queries").fetchone()
        if count > 0:
            return 0
        inserted = 0
        for name, sql, tags, description in _SEED:
            try:
                conn.execute(
                    "INSERT INTO saved_queries (name, sql, tags, description) "
                    "VALUES (?, ?, ?, ?)",
                    (name, sql, tags, description),
                )
                inserted += 1
            except sqlite3.IntegrityError as e:
                logger.warning(f"seed insert skipped for '{name}': {e}")
        conn.commit()
        return inserted
    finally:
        conn.close()
