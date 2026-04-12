"""数据库连接与 schema 迁移。"""
import sqlite3
from pathlib import Path
from .constants import DEFAULT_DB_PATH


SAMPLES_SCHEMA = """
CREATE TABLE IF NOT EXISTS signal_trust_samples (
    code TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    sample_end_date TEXT NOT NULL,
    pred_10d REAL NOT NULL,
    actual_10d REAL,
    version TEXT NOT NULL,
    market_cap_bucket TEXT,
    industry TEXT,
    liquidity_bucket TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (code, trade_date)
);
"""

SAMPLES_INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_sts_code ON signal_trust_samples(code);",
    "CREATE INDEX IF NOT EXISTS idx_sts_trade_date ON signal_trust_samples(trade_date);",
    "CREATE INDEX IF NOT EXISTS idx_sts_end_date ON signal_trust_samples(sample_end_date);",
    "CREATE INDEX IF NOT EXISTS idx_sts_mc ON signal_trust_samples(market_cap_bucket);",
    "CREATE INDEX IF NOT EXISTS idx_sts_ind ON signal_trust_samples(industry);",
    "CREATE INDEX IF NOT EXISTS idx_sts_liq ON signal_trust_samples(liquidity_bucket);",
]

SCORES_SCHEMA = """
CREATE TABLE IF NOT EXISTS signal_trust_scores (
    code TEXT PRIMARY KEY,
    as_of_date TEXT NOT NULL,
    n_samples INTEGER NOT NULL,
    direction_hit_rate REAL,
    systematic_bias REAL,
    high_pred_realize_rate REAL,
    trust_tag TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def connect(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """标准连接：busy_timeout + row_factory."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def migrate(db_path: str = DEFAULT_DB_PATH) -> None:
    """幂等创建两张表和所有索引。"""
    conn = connect(db_path)
    try:
        conn.execute(SAMPLES_SCHEMA)
        for sql in SAMPLES_INDICES:
            conn.execute(sql)
        conn.execute(SCORES_SCHEMA)
        conn.commit()
    finally:
        conn.close()
