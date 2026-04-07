"""
NG Feature Cache Schema — version-specific tables for backward compatibility.

Table naming convention:
  - ng_feature_cache       → ng1.0.0 (original, 62 features, absolute labels)
  - ng101_feature_cache    → ng1.0.1 (69 features, industry excess labels)
  - ng102_feature_cache    → ng1.0.2
  - ng103_feature_cache    → ng1.0.3 (66 features, drop 3 flipping factors)

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
    'ng1.0.2': 'ng102_feature_cache',
    'ng1.0.3': 'ng103_feature_cache',
    'ng1.0.4': 'ng104_feature_cache',
}

DEFAULT_VERSION = 'ng1.0.3'


def get_table_name(version: str = None) -> str:
    """Get cache table name for a given NG version."""
    ver = version or DEFAULT_VERSION
    return VERSION_TABLE_MAP.get(ver, f'ng{ver.replace(".", "").replace("ng", "")}_feature_cache')


def _schema_sql(table_name: str, version: str = None) -> str:
    """Generate CREATE TABLE SQL for a given table name and version."""
    ver = version or DEFAULT_VERSION
    extra_cols = ''
    if ver >= 'ng1.0.2':
        extra_cols = '\n    downside_10d REAL,'
    if ver >= 'ng1.0.3':
        extra_cols += '\n    label_raw_3d REAL,'
        extra_cols += '\n    label_raw_5d REAL,'
        extra_cols += '\n    label_raw_10d REAL,'
        extra_cols += '\n    label_raw_15d REAL,'
    if ver >= 'ng1.0.4':
        extra_cols += '\n    maxdd_3d REAL,'
        extra_cols += '\n    maxdd_5d REAL,'
        extra_cols += '\n    maxdd_10d REAL,'
        extra_cols += '\n    maxdd_15d REAL,'
        extra_cols += '\n    ra_label_3d REAL,'
        extra_cols += '\n    ra_label_5d REAL,'
        extra_cols += '\n    ra_label_10d REAL,'
        extra_cols += '\n    ra_label_15d REAL,'
    return f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,
    features_json TEXT NOT NULL,
    label_3d REAL,
    label_5d REAL,
    label_10d REAL,
    label_15d REAL,{extra_cols}
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
    ver = version or DEFAULT_VERSION
    table_name = get_table_name(ver)
    with sqlite3.connect(path, timeout=30) as conn:
        conn.executescript(_schema_sql(table_name, version=ver))
    print(f"{table_name} table ready: {path}")


# Backward compatibility: also expose SCHEMA_SQL for ng_feature_cache (v1.0.0)
SCHEMA_SQL = _schema_sql('ng_feature_cache')


MONEYFLOW_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS moneyflow_daily (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,
    buy_sm_amount REAL,
    sell_sm_amount REAL,
    buy_md_amount REAL,
    sell_md_amount REAL,
    buy_lg_amount REAL,
    sell_lg_amount REAL,
    buy_elg_amount REAL,
    sell_elg_amount REAL,
    net_mf_amount REAL,
    UNIQUE(code, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_moneyflow_daily_date ON moneyflow_daily(trade_date);
CREATE INDEX IF NOT EXISTS idx_moneyflow_daily_code_date ON moneyflow_daily(code, trade_date);
"""


def create_moneyflow_table(db_path: str = None):
    """Create moneyflow_daily table (if not exists)."""
    path = db_path or DB_PATH
    with sqlite3.connect(path, timeout=30) as conn:
        conn.executescript(MONEYFLOW_SCHEMA_SQL)
    print(f"moneyflow_daily table ready: {path}")


if __name__ == '__main__':
    import sys
    ver = sys.argv[1] if len(sys.argv) > 1 else None
    create_table(version=ver)
