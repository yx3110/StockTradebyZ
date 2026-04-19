"""CRUD for saved_queries on webapp.db. See Task 7 for CRUD helpers."""
from __future__ import annotations

import sqlite3
from pathlib import Path


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
