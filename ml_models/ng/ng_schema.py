"""Create and manage the ng_feature_cache table."""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data_adapter', 'stock_data.db')

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ng_feature_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,
    features_json TEXT NOT NULL,
    label_3d REAL,
    label_5d REAL,
    label_10d REAL,
    label_15d REAL,
    market_return_5d REAL,
    market_return_20d REAL,
    market_volatility_20d REAL,
    market_breadth REAL,
    market_new_high_ratio REAL,
    northbound_flow_5d REAL,
    market_volume_ratio REAL,
    market_drawdown REAL,
    vix_proxy REAL,
    market_momentum_diff REAL,
    UNIQUE(code, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_ng_fc_date ON ng_feature_cache(trade_date);
CREATE INDEX IF NOT EXISTS idx_ng_fc_code_date ON ng_feature_cache(code, trade_date);
"""


def create_table(db_path: str = None):
    """Create ng_feature_cache table if not exists."""
    path = db_path or DB_PATH
    with sqlite3.connect(path, timeout=30) as conn:
        conn.executescript(SCHEMA_SQL)
    print(f"ng_feature_cache table ready: {path}")


if __name__ == '__main__':
    create_table()
