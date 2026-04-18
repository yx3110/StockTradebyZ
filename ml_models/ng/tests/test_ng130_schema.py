"""Tests for ng1.3.0 schema registration and SQL generation."""
import sqlite3
import tempfile
import os
import pytest

from ml_models.ng.ng_schema import (
    get_table_name, get_schema_version, _is_1_3_branch,
    _schema_sql, create_table,
)


def test_version_table_map_ng130():
    assert get_table_name('ng1.3.0') == 'ng200_feature_cache'


def test_schema_version_map_ng130():
    assert get_schema_version('ng1.3.0') == 'ng1.3.0'


def test_is_1_3_branch():
    assert _is_1_3_branch('ng1.3.0') is True
    assert _is_1_3_branch('ng1.3.1') is True
    assert _is_1_3_branch('ng1.0.1') is False
    assert _is_1_3_branch('ng1.2.3') is False


def test_ng130_schema_has_downside_columns():
    sql = _schema_sql('ng200_feature_cache', version='ng1.3.0')
    for horizon in [3, 5, 10, 15]:
        assert f'downside_{horizon}d REAL' in sql


def test_ng130_does_not_inherit_ng104_ra_labels():
    """ng1.3.x branches from ng1.0.1, should NOT have maxdd_*/ra_label_* from ng1.0.4."""
    sql = _schema_sql('ng200_feature_cache', version='ng1.3.0')
    assert 'maxdd_3d' not in sql
    assert 'ra_label_3d' not in sql


def test_ng130_does_not_inherit_ng107_cond_labels():
    """ng1.3.x should NOT have cond_label_*/amv_* inline columns (AMV from market_amv join)."""
    sql = _schema_sql('ng200_feature_cache', version='ng1.3.0')
    assert 'cond_label_3d' not in sql
    assert 'amv_var1 REAL' not in sql


def test_create_ng200_table_succeeds():
    """End-to-end: CREATE TABLE for ng1.3.0 doesn't crash."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    try:
        create_table(db_path=db_path, version='ng1.3.0')
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur]
            assert 'ng200_feature_cache' in tables
    finally:
        os.unlink(db_path)
