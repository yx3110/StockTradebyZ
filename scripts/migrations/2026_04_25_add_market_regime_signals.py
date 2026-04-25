"""Migration: create market_regime_signals table for ng2.0a.

Idempotent: safe to re-run, will not drop or alter existing data.
"""
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / 'data_adapter' / 'stock_data.db'

DDL = """
CREATE TABLE IF NOT EXISTS market_regime_signals (
    trade_date TEXT PRIMARY KEY,
    v11_var1 REAL,
    v11_ma60 REAL,
    v11_macd REAL,
    v11_bull INTEGER,
    v11_streak INTEGER,
    b1_pct_above_ma20 REAL,
    b1_pct_above_ma60 REAL,
    b1_adv_dec_ratio REAL,
    b1_score REAL,
    b1_bull INTEGER,
    b1_streak INTEGER,
    b2_rv_60d REAL,
    b2_rv_percentile_252 REAL,
    b2_bull INTEGER,
    b2_streak INTEGER,
    vote_count INTEGER,
    regime_v2_raw INTEGER,
    regime_v2_streak INTEGER,
    regime_v2 INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

INDEX_DDL = "CREATE INDEX IF NOT EXISTS idx_mrs_regime_v2 ON market_regime_signals(regime_v2);"


def main():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute('PRAGMA busy_timeout=30000')
    try:
        conn.executescript(DDL)
        conn.execute(INDEX_DDL)
        conn.commit()
        print(f'OK: market_regime_signals created in {DB_PATH}')
    finally:
        conn.close()


if __name__ == '__main__':
    main()
