"""
NG Feature Cache Schema — version-specific tables.

Table naming convention:
  - ng_feature_cache       → ng1.0.0 (original, 62 features, absolute labels)
  - ng101_feature_cache    → ng1.0.1 (69 features, industry excess labels)
  - ng102_feature_cache    → ng1.0.2 (adds downside_10d single column)
  - ng103_feature_cache    → ng1.0.3 (adds label_raw_*)
  - ng104_feature_cache    → ng1.0.4 (adds maxdd_*, ra_label_*)
  - ng107_feature_cache    → ng1.0.7 (adds cond_label_*, amv_*)
  - ng121_feature_cache    → ng1.2.1 (adds vn_label_*, path_*, downside_std_10d)
  - ng123_feature_cache    → ng1.2.3 (adds 4-horizon downside_*, drops legacy)

Lineage:
  - ng1.0.x: linear (each version inherits previous columns)
  - ng1.1.x: reuses ng1.0.1 schema (no own table)
  - ng1.2.x: branches from ng1.0.1 with own independent schemas per sub-version
    (ng1.2.0/2.2 reuse ng1.0.1 cache; ng1.2.1/2.3 have own schemas)

Old version tables are never deleted (backward compat).
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
    'ng1.0.7': 'ng107_feature_cache',
    'ng1.1.0': 'ng101_feature_cache',  # 基于ng1.0.1(69feat)精简, 复用ng101缓存
    'ng1.2.0': 'ng101_feature_cache',  # Margin Ranking Loss, 复用ng101缓存(仅训练层改动)
    'ng1.2.1': 'ng121_feature_cache',  # Vol-Normalized Rank Label, 独立缓存(含vn_label列)
    'ng1.2.2': 'ng101_feature_cache',  # Return-Weighted CE Quintiles, 复用ng101缓存(训练层转换)
    'ng1.2.3': 'ng123_feature_cache',  # 三轴重构: -12 弱特征 + 12 moneyflow + 6 mined + downside label
    'ng1.2.4': 'ng124_feature_cache',  # ng1.0.1 + 2 top mf factors only (极保守增量)
    'ng1.3.0': 'ng130_feature_cache',  # Multi-task 双头 (excess + downside) + β composite
    'ng1.4.0': 'ng130_feature_cache',  # ng1.0.1 底座 + 4 downside + 3 AMV (70 features, 无 dual-head)
    'ng1.4.1': 'ng130_feature_cache',  # ng1.4.0 - 4 downside (ablation)
    'ng1.4.2': 'ng130_feature_cache',  # ng1.4.0 - 3 AMV (ablation)
    'ng1.5.0': 'ng150_feature_cache',  # ng1.4.0 底座 + 5 Tier B regime-refined features (78 total)
    'ng1.6.1': 'ng101_feature_cache',  # F2: ng1.0.1 特征 + cross-sectional factor-residual labels (动态)
    'ng1.6.2': 'ng101_feature_cache',  # P1.3 Step B: ng1.0.1 特征 + risk-adjusted label override (Calmar/Sortino) from CSV
    'ng1.7.0': 'ng101_feature_cache',  # ng1.0.1 基座 (66) + 4 altdata 因子 (JOIN altdata_factor_cache 在 trainer/scorer 侧)
    'ng2.1-bull': 'ng101_feature_cache',  # ng2.1 specialist: bull-only training subset, same 66 features
    'ng2.1-bear': 'ng101_feature_cache',  # ng2.1 specialist: bear-only training subset + DD-penalty label
}

DEFAULT_VERSION = 'ng1.0.3'

PRODUCTION_VERSION = 'ng1.0.6'  # MOE v1: ng1.0.1 bull + ng1.0.4 bear; 4-25 真零泄漏 OOS Pre-2020 验证 v1 (Top10 +0.87%) > v2 ng1.0.62 (+0.49%) 78%, v1→v2 切换被回滚

# Versions that reuse another version's cache schema (table columns + feature semantics)
# ng1.1.0 is a pruning+bugfix iteration on top of ng1.0.1, uses identical ng101_feature_cache schema
# ng1.2.0/ng1.2.2 are training-layer variants on ng1.0.1 (same features, different loss/label transform)
# ng1.2.1 adds new label columns (vn_label_Nd) but keeps ng1.0.1 feature set
SCHEMA_VERSION_MAP = {
    'ng1.1.0': 'ng1.0.1',
    'ng1.2.0': 'ng1.0.1',
    'ng1.2.1': 'ng1.2.1',  # new schema (adds vn_label columns)
    'ng1.2.2': 'ng1.0.1',
    'ng1.2.3': 'ng1.2.3',  # own schema (adds downside_kd label cols)
    'ng1.2.4': 'ng1.2.4',  # own schema (ng1.0.1 base + label_raw, NO downside cols)
    'ng1.3.0': 'ng1.3.0',  # own schema (ng1.0.1 base + downside_{3,5,10,15}d + amv_* reuse from market_amv)
    'ng1.4.0': 'ng1.3.0',  # reuses ng130 cache schema (same label + downside cols, trainer filters features)
    'ng1.4.1': 'ng1.3.0',  # ablation variant, same schema as ng140
    'ng1.4.2': 'ng1.3.0',  # ablation variant, same schema as ng140
    'ng1.5.0': 'ng1.5.0',  # own schema (ng1.4.0 base + 5 Tier B regime-refined features in features_json)
    'ng1.6.1': 'ng1.0.1',  # reuses ng101 cache; only label transform differs (factor residualization)
    'ng1.6.2': 'ng1.0.1',  # reuses ng101 cache; labels overridden by Calmar/Sortino CSV (P1.3 Step B)
    'ng1.7.0': 'ng1.0.1',  # reuses ng101 cache; 4 altdata factors joined at trainer/scorer layer
    'ng2.1-bull': 'ng1.0.1',  # ng2.1 specialist; identical feature schema to ng1.0.1
    'ng2.1-bear': 'ng1.0.1',  # ng2.1 specialist; identical feature schema to ng1.0.1
}


def get_schema_version(version: str) -> str:
    """Return the underlying cache schema version (handles reuse like ng1.1.0 → ng1.0.1)."""
    return SCHEMA_VERSION_MAP.get(version, version)


def version_ge(v1: str, v2: str) -> bool:
    """Safe semantic version comparison for 'ngX.Y.Z' strings."""
    def _parts(v):
        return tuple(int(x) for x in v.lstrip('ng').split('.'))
    try:
        return _parts(v1) >= _parts(v2)
    except (ValueError, AttributeError):
        return v1 >= v2  # fallback to string comparison


def get_table_name(version: str = None) -> str:
    """Get cache table name for a given NG version."""
    ver = version or DEFAULT_VERSION
    return VERSION_TABLE_MAP.get(ver, f'ng{ver.replace(".", "").replace("ng", "")}_feature_cache')


def _is_1_2_branch(ver: str) -> bool:
    """ng1.2.x branches from ng1.0.1 — skip ng1.0.4/ng1.0.7 linear additions."""
    return ver.startswith('ng1.2.')


def _is_1_3_branch(ver: str) -> bool:
    """ng1.3.x branches from ng1.0.1 — adds multi-task downside labels, does NOT inherit ng1.0.4/0.7 linear columns unless needed."""
    return ver.startswith('ng1.3.')


def _is_1_4_branch(ver: str) -> bool:
    """ng1.4.x reuses ng1.3.0 schema (same cache table) but with ng1.0.1 single-head training."""
    return ver.startswith('ng1.4.')


def _is_1_5_branch(ver: str) -> bool:
    """ng1.5.x = ng1.4.0 base + 5 Tier B regime-refined features (own schema).
    Stores label_raw_* + downside_* (same as ng1.3.x cache) + 5 new features in features_json.
    """
    return ver.startswith('ng1.5.')


def _is_1_6_branch(ver: str) -> bool:
    """ng1.6.x = ng1.0.1 base + label-engineering variants (no schema change).
    Reuses ng101_feature_cache; trainer dynamically residualizes labels.
    """
    return ver.startswith('ng1.6.')


def _version_in_range(ver: str, lo: str, hi: str) -> bool:
    """True iff lo <= ver < hi (inclusive lower, exclusive upper).

    Use this for ng1.2.x sub-version blocks where each version has its own
    schema and must be bounded above by the next sub-version (e.g., ng1.2.1
    block must stop firing when ver=ng1.2.3+).
    """
    return version_ge(ver, lo) and not version_ge(ver, hi)


def _real_cols(*names: str) -> str:
    """Build SQL DDL fragment for REAL columns. Centralizes the repeated
    `extra_cols += '\\n    {col} REAL,'` pattern across schema blocks.
    """
    return ''.join(f'\n    {n} REAL,' for n in names)


def _schema_sql(table_name: str, version: str = None) -> str:
    """Generate CREATE TABLE SQL for a given table name and version.

    Version lineage:
      - ng1.0.x: linear (each adds more columns)
      - ng1.1.x: reuses ng1.0.1 schema
      - ng1.2.x: branches from ng1.0.1 (does NOT inherit ng1.0.4/ng1.0.7 columns)
      - ng1.3.x: branches from ng1.0.1 (adds downside_{3,5,10,15}d; AMV joined from market_amv)
    """
    ver = version or DEFAULT_VERSION
    is_12 = _is_1_2_branch(ver)
    is_13 = _is_1_3_branch(ver)
    is_15 = _is_1_5_branch(ver)
    extra_cols = ''

    # ng1.5.x: same schema shape as ng1.3.x (label_raw_* + downside_*), but features_json
    # carries 5 additional Tier B regime-refined features. amv_* joined from market_amv
    # at read time, new Tier B features stored inside features_json only.
    if is_15:
        extra_cols = _real_cols('label_raw_3d', 'label_raw_5d', 'label_raw_10d', 'label_raw_15d')
        extra_cols += _real_cols('downside_3d', 'downside_5d', 'downside_10d', 'downside_15d')
    # ng1.3.x: branches from ng1.0.1 (does NOT inherit ng1.0.4/ng1.0.7/ng1.2.x columns).
    # Only adds label_raw_{3,5,10,15}d (ablation) and downside_{3,5,10,15}d (L3 multi-task
    # downside head labels). amv_* features are joined from market_amv at read time, not
    # stored as inline columns.
    elif is_13:
        extra_cols = _real_cols('label_raw_3d', 'label_raw_5d', 'label_raw_10d', 'label_raw_15d')
        extra_cols += _real_cols('downside_3d', 'downside_5d', 'downside_10d', 'downside_15d')
    else:
        # Existing ng1.0.x / ng1.1.x / ng1.2.x lineage handling.
        # ng1.2.x branches from ng1.0.1 and does NOT inherit ng1.0.4/ng1.0.7
        # additions (maxdd/ra_label/cond_label/amv). The ng1.0.2 block is also
        # gated on `not is_12` to avoid a downside_10d naming conflict with
        # ng1.2.3's new 4-horizon downside_kd columns. Each linear-lineage block
        # below gates on `not is_12` for that reason.
        if version_ge(ver, 'ng1.0.2') and not is_12:
            extra_cols = _real_cols('downside_10d')
        if version_ge(ver, 'ng1.0.3'):
            extra_cols += _real_cols('label_raw_3d', 'label_raw_5d', 'label_raw_10d', 'label_raw_15d')
        if version_ge(ver, 'ng1.0.4') and not is_12:
            extra_cols += _real_cols(
                'maxdd_3d', 'maxdd_5d', 'maxdd_10d', 'maxdd_15d',
                'ra_label_3d', 'ra_label_5d', 'ra_label_10d', 'ra_label_15d',
            )
        if version_ge(ver, 'ng1.0.7') and not is_12:
            extra_cols += _real_cols(
                'cond_label_3d', 'cond_label_5d', 'cond_label_10d', 'cond_label_15d',
                'amv_var1', 'amv_macd', 'amv_regime_days',
            )
        # ng1.2.1 adds Sharpe-style path-based labels (ng1.2.x branch only, NOT ng1.2.3+)
        # ng1.2.3 spec §3.3 explicitly excludes vn_label_* / path_* / downside_std_10d
        if is_12 and _version_in_range(ver, 'ng1.2.1', 'ng1.2.3'):
            extra_cols += _real_cols(
                'vn_label_3d', 'vn_label_5d', 'vn_label_10d', 'vn_label_15d',
                'path_mean_10d', 'path_std_10d', 'downside_std_10d',
            )
        # ng1.2.3 adds soft-downside label columns (per spec section 5).
        # Upper-bound guard: ng1.2.4 does NOT inherit downside cols (no penalty label).
        if is_12 and _version_in_range(ver, 'ng1.2.3', 'ng1.2.4'):
            extra_cols += _real_cols('downside_3d', 'downside_5d', 'downside_10d', 'downside_15d')
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
