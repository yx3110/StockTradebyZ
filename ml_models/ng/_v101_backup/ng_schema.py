"""
Create and manage the ng_feature_cache table.

v1.1.0 notes:
  - label_3d/5d/10d/15d now store INDUSTRY EXCESS returns
    (stock_return - industry_median_return), not absolute returns.
  - features_json contains 58 stock features (was 52 in v1.0.0):
    +10 cs_rank_*, +5 residual_*, +3 sector_*, -11 removed low-efficiency.
"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data_adapter', 'stock_data.db')

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ng_feature_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,
    features_json TEXT NOT NULL,       -- 58 stock + 11 industry features in JSON (v1.1.0)
    label_3d REAL,                     -- Forward 3d INDUSTRY EXCESS return (v1.1.0)
    label_5d REAL,                     -- Forward 5d INDUSTRY EXCESS return (v1.1.0)
    label_10d REAL,                    -- Forward 10d INDUSTRY EXCESS return (v1.1.0)
    label_15d REAL,                    -- Forward 15d INDUSTRY EXCESS return (v1.1.0)
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
