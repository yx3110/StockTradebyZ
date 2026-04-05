"""
NG Feature Cache Schema — version-specific tables for backward compatibility.

Table naming convention:
  - ng_feature_cache       → ng1.0.0 (original, 62 features, absolute labels)
  - ng101_feature_cache    → ng1.0.1 (69 features, industry excess labels)
  - ng102_feature_cache    → ng1.0.2 (future)

All tables share the same column schema; only the content differs.
Old version tables are never deleted.
"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data_adapter', 'stock_data.db')

# Version → table name mapping
VERSION_TABLE_MAP = {
    'ng1.0.0': 'ng_feature_cache',
    'ng1.0.1': 'ng101_feature_cache',
}

DEFAULT_VERSION = 'ng1.0.1'


def get_table_name(version: str = None) -> str:
    """Get cache table name for a given NG version."""
    ver = version or DEFAULT_VERSION
    return VERSION_TABLE_MAP.get(ver, f'ng{ver.replace(".", "").replace("ng", "")}_feature_cache')


def _schema_sql(table_name: str) -> str:
    """Generate CREATE TABLE SQL for a given table name."""
    return f"""
CREATE TABLE IF NOT EXISTS {table_name} (
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

CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {table_name}(trade_date);
CREATE INDEX IF NOT EXISTS idx_{table_name}_code_date ON {table_name}(code, trade_date);
"""


def create_table(db_path: str = None, version: str = None):
    """Create feature cache table for the given version (if not exists)."""
    path = db_path or DB_PATH
    table_name = get_table_name(version)
    with sqlite3.connect(path, timeout=30) as conn:
        conn.executescript(_schema_sql(table_name))
    print(f"{table_name} table ready: {path}")


# Backward compatibility: also expose SCHEMA_SQL for ng_feature_cache (v1.0.0)
SCHEMA_SQL = _schema_sql('ng_feature_cache')


if __name__ == '__main__':
    import sys
    ver = sys.argv[1] if len(sys.argv) > 1 else None
    create_table(version=ver)
