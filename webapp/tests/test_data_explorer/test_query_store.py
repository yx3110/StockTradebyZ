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


# --- Task 7: CRUD + seed ----------------------------------------------------

import pytest

from core.data_explorer.query_store import (
    create_query, delete_query, get_query, list_queries,
    seed_default_queries, touch_query, update_query,
)


def _setup(tmp_webapp_db):
    apply_migration(tmp_webapp_db)
    return tmp_webapp_db


def test_create_and_get_and_list(tmp_webapp_db):
    db = _setup(tmp_webapp_db)
    q = create_query(db, name="my-q", sql="SELECT 1", tags="test", description="d")
    assert q["id"] > 0
    assert q["name"] == "my-q"

    fetched = get_query(db, q["id"])
    assert fetched["sql"] == "SELECT 1"

    rows = list_queries(db)
    assert len(rows) == 1


def test_create_name_unique(tmp_webapp_db):
    db = _setup(tmp_webapp_db)
    create_query(db, name="dup", sql="SELECT 1", tags=None, description=None)
    with pytest.raises(ValueError) as exc:
        create_query(db, name="dup", sql="SELECT 2", tags=None, description=None)
    assert "already exists" in str(exc.value)


def test_update_fields(tmp_webapp_db):
    db = _setup(tmp_webapp_db)
    q = create_query(db, name="n", sql="S1", tags=None, description=None)
    updated = update_query(db, q["id"], sql="S2", tags="new")
    assert updated["sql"] == "S2"
    assert updated["tags"] == "new"


def test_delete(tmp_webapp_db):
    db = _setup(tmp_webapp_db)
    q = create_query(db, name="n", sql="S", tags=None, description=None)
    delete_query(db, q["id"])
    assert list_queries(db) == []


def test_touch_increments_run_count(tmp_webapp_db):
    db = _setup(tmp_webapp_db)
    q = create_query(db, name="n", sql="S", tags=None, description=None)
    touch_query(db, q["id"])
    touch_query(db, q["id"])
    fetched = get_query(db, q["id"])
    assert fetched["run_count"] == 2
    assert fetched["last_run_at"] is not None


def test_list_filters_by_tag(tmp_webapp_db):
    db = _setup(tmp_webapp_db)
    create_query(db, name="a", sql="S", tags="preset,stock", description=None)
    create_query(db, name="b", sql="S", tags="preset,cross", description=None)
    create_query(db, name="c", sql="S", tags="user", description=None)

    preset_rows = list_queries(db, tag="preset")
    assert {r["name"] for r in preset_rows} == {"a", "b"}


def test_seed_only_on_empty_table(tmp_webapp_db):
    db = _setup(tmp_webapp_db)
    inserted1 = seed_default_queries(db)
    inserted2 = seed_default_queries(db)
    assert inserted1 > 0
    assert inserted2 == 0  # idempotent
