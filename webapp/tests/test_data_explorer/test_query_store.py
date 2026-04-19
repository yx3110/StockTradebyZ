"""Tests for query_store (Task 1 covers migration; Task 7 covers CRUD)."""
import sqlite3
from pathlib import Path

from core.data_explorer.query_store import apply_migration


def test_migration_creates_table(tmp_webapp_db: Path) -> None:
    apply_migration(tmp_webapp_db)
    conn = sqlite3.connect(tmp_webapp_db)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    conn.close()
    assert "saved_queries" in tables


def test_migration_is_idempotent(tmp_webapp_db: Path) -> None:
    apply_migration(tmp_webapp_db)
    apply_migration(tmp_webapp_db)
    apply_migration(tmp_webapp_db)
    conn = sqlite3.connect(tmp_webapp_db)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(saved_queries)")]
    conn.close()
    assert cols == [
        "id", "name", "sql", "tags", "description",
        "created_at", "updated_at", "last_run_at", "run_count",
    ]
