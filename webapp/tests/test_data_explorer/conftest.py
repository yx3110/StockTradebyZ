"""Shared fixtures for data_explorer tests."""
import json
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def tmp_webapp_db(tmp_path: Path) -> Path:
    """Empty webapp.db-style SQLite file for query_store tests."""
    return tmp_path / "webapp.db"


@pytest.fixture
def tmp_stock_db(tmp_path: Path) -> Path:
    """Tiny stock_data.db-style SQLite file with a couple of tables for query_runner tests."""
    db_path = tmp_path / "stock.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE daily_quotes (
            code TEXT, trade_date DATE, close REAL
        );
        INSERT INTO daily_quotes VALUES
          ('600519.SH', '2026-04-17', 1700.0),
          ('600519.SH', '2026-04-18', 1720.0),
          ('000858.SZ', '2026-04-18', 150.0);

        CREATE TABLE ng101_feature_cache (
            code TEXT, trade_date DATE, label_10d REAL, features_json TEXT
        );
        INSERT INTO ng101_feature_cache VALUES
          ('600519.SH', '2026-04-18', 0.05, '{"pb": 8.1, "roe": 31.2}'),
          ('000858.SZ', '2026-04-18', 0.03, '{"pb": 4.2, "roe": 22.1}');

        CREATE TABLE securities (code TEXT, name TEXT, type TEXT);
        INSERT INTO securities VALUES ('600519.SH', 'Maotai', 'A股');
        """
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def sample_df_with_json() -> pd.DataFrame:
    """DataFrame shaped like a feature_cache query result."""
    return pd.DataFrame(
        {
            "code": ["600519.SH", "000858.SZ"],
            "trade_date": ["2026-04-18", "2026-04-18"],
            "label_10d": [0.05, 0.03],
            "features_json": [
                json.dumps({"pb": 8.1, "roe": 31.2}),
                json.dumps({"pb": 4.2, "roe": 22.1}),
            ],
        }
    )
