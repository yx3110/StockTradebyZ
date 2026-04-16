"""Unit tests for ng1.2.3 schema isolation.

Catches the regression class where adding a new ng1.2.x version accidentally
inherits unwanted columns from sibling versions. The bug fixed in commit
f1e025aa (vn_label_* leaking into ng1.2.3) would have been caught by these
tests on the next iteration.
"""
import pytest

from ml_models.ng.ng_schema import _schema_sql, _version_in_range


# --- _version_in_range helper ----------------------------------------------

@pytest.mark.parametrize('ver, lo, hi, expected', [
    ('ng1.2.1', 'ng1.2.1', 'ng1.2.3', True),   # lower-bound inclusive
    ('ng1.2.2', 'ng1.2.1', 'ng1.2.3', True),   # mid-range
    ('ng1.2.3', 'ng1.2.1', 'ng1.2.3', False),  # upper-bound exclusive
    ('ng1.2.0', 'ng1.2.1', 'ng1.2.3', False),  # below lower bound
    ('ng1.2.5', 'ng1.2.1', 'ng1.2.3', False),  # above upper bound
    ('ng1.0.7', 'ng1.0.4', 'ng1.0.8', True),   # linear lineage range
    ('ng1.0.8', 'ng1.0.4', 'ng1.0.8', False),  # linear lineage upper-exclusive
])
def test_version_in_range(ver, lo, hi, expected):
    assert _version_in_range(ver, lo, hi) is expected


# --- ng1.2.3 must EXCLUDE these columns (per spec §3.3) ---------------------

@pytest.mark.parametrize('forbidden_col', [
    'vn_label_3d', 'vn_label_5d', 'vn_label_10d', 'vn_label_15d',
    'path_mean_10d', 'path_std_10d', 'downside_std_10d',  # ng1.2.1 leak
    'maxdd_3d', 'maxdd_5d', 'maxdd_10d', 'maxdd_15d',  # ng1.0.4 lineage
    'ra_label_3d', 'ra_label_5d', 'ra_label_10d', 'ra_label_15d',
    'cond_label_3d', 'cond_label_10d',  # ng1.0.7 lineage
    'amv_var1', 'amv_macd', 'amv_regime_days',  # ng1.0.7 lineage
])
def test_ng123_excludes(forbidden_col):
    sql = _schema_sql('ng123_feature_cache', version='ng1.2.3')
    assert forbidden_col not in sql, \
        f"ng1.2.3 schema must NOT contain {forbidden_col} (per spec §3.3)"


# --- ng1.2.3 must INCLUDE these columns -------------------------------------

@pytest.mark.parametrize('required_col', [
    'downside_3d', 'downside_5d', 'downside_10d', 'downside_15d',
    'label_3d', 'label_5d', 'label_10d', 'label_15d',
    'label_raw_3d', 'label_raw_5d', 'label_raw_10d', 'label_raw_15d',
    'features_json',
    'market_return_5d', 'market_volatility_20d', 'market_breadth',
])
def test_ng123_includes(required_col):
    sql = _schema_sql('ng123_feature_cache', version='ng1.2.3')
    assert required_col in sql, \
        f"ng1.2.3 schema must contain {required_col}"


# --- ng1.2.1 schema must still work (regression check for our guard) -------

def test_ng121_still_has_vn_label():
    """Verify the upper-bound guard on the ng1.2.1 block did not break ng1.2.1 itself."""
    sql = _schema_sql('ng121_feature_cache', version='ng1.2.1')
    for col in ('vn_label_3d', 'vn_label_10d', 'path_mean_10d', 'path_std_10d'):
        assert col in sql, f"ng1.2.1 schema must still contain {col}"


# --- Linear lineage versions still work (not affected by our changes) -------

def test_ng104_has_maxdd_columns():
    """ng1.0.4 (linear lineage) must still get maxdd_* columns."""
    sql = _schema_sql('ng104_feature_cache', version='ng1.0.4')
    for col in ('maxdd_3d', 'maxdd_10d', 'ra_label_3d', 'ra_label_10d'):
        assert col in sql


def test_ng102_linear_has_legacy_downside_10d():
    """ng1.0.2 (linear lineage) must still get the legacy single-column downside_10d."""
    sql = _schema_sql('ng102_feature_cache', version='ng1.0.2')
    assert 'downside_10d' in sql


# --- ng1.2.4 must EXCLUDE these columns (no downside penalty, no legacy) ------

@pytest.mark.parametrize('forbidden_col', [
    'downside_3d', 'downside_5d', 'downside_15d',  # ng1.2.3 downside_kd
    'vn_label_3d', 'vn_label_5d', 'vn_label_10d', 'vn_label_15d',  # ng1.2.1
    'path_mean_10d', 'path_std_10d', 'downside_std_10d',  # ng1.2.1
    'maxdd_3d', 'maxdd_10d',  # ng1.0.4 lineage
    'ra_label_3d', 'ra_label_10d',  # ng1.0.4 lineage
    'cond_label_3d', 'cond_label_10d',  # ng1.0.7 lineage
    'amv_var1', 'amv_macd', 'amv_regime_days',  # ng1.0.7 lineage
])
def test_ng124_excludes(forbidden_col):
    sql = _schema_sql('ng124_feature_cache', version='ng1.2.4')
    assert forbidden_col not in sql, \
        f"ng1.2.4 schema must NOT contain {forbidden_col}"


@pytest.mark.parametrize('required_col', [
    'label_3d', 'label_5d', 'label_10d', 'label_15d',
    'label_raw_3d', 'label_raw_5d', 'label_raw_10d', 'label_raw_15d',
    'features_json',
    'market_return_5d', 'market_volatility_20d', 'market_breadth',
])
def test_ng124_includes(required_col):
    sql = _schema_sql('ng124_feature_cache', version='ng1.2.4')
    assert required_col in sql, \
        f"ng1.2.4 schema must contain {required_col}"


def test_ng123_does_not_have_legacy_downside_10d_block():
    """The ng1.2.3 downside_10d should come from the new 4-col block, not the ng1.0.2 legacy block.
    Match 'downside_10d REAL' (DDL-specific) so a stray comment containing 'downside_10d'
    won't false-positive; if both code paths emit the column, count would be 2."""
    sql = _schema_sql('ng123_feature_cache', version='ng1.2.3')
    assert 'downside_3d' in sql, "ng1.2.3 must have downside_3d (sanity for 4-col block)"
    assert sql.count('downside_10d REAL') == 1, \
        "ng1.2.3 must have exactly 1 DDL definition for downside_10d (not 2)"
