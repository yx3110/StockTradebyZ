"""Unit tests for ng1.2.3 drop-list filter helpers."""
from ml_models.ng.ng_feature_calculator import (
    NG123_DROP_FEATURES,
    get_ng123_drop_features,
    filter_ng123_features,
)


def test_drop_features_count_is_12():
    assert len(NG123_DROP_FEATURES) == 12, \
        f"Spec §4.3 requires exactly 12 drop-list features, got {len(NG123_DROP_FEATURES)}"


def test_drop_features_are_known():
    """Pin the exact 12 features per spec §4.3."""
    expected = {
        'lower_shadow_ratio', 'volume_cv', 'volume_contraction', 'volume_price_corr',
        'industry_hhi', 'industry_volume_change', 'n_sectors_strong',
        'peg_proxy', 'pb_roe_ratio', 'dv_ratio', 'up_volume_ratio', 'ocf_quality',
    }
    assert set(NG123_DROP_FEATURES) == expected


def test_get_ng123_drop_features_returns_frozenset():
    r = get_ng123_drop_features()
    assert isinstance(r, frozenset)
    assert len(r) == 12


def test_filter_removes_drop_features():
    d = {
        'volume_ratio_5d': 1.0,      # NOT dropped
        'lower_shadow_ratio': 0.5,   # dropped
        'rsi_14': 50.0,              # NOT dropped
        'peg_proxy': 10.0,           # dropped
        'industry_hhi': 0.1,         # dropped
    }
    r = filter_ng123_features(d)
    assert 'volume_ratio_5d' in r
    assert 'rsi_14' in r
    assert 'lower_shadow_ratio' not in r
    assert 'peg_proxy' not in r
    assert 'industry_hhi' not in r
    assert len(r) == 2


def test_filter_preserves_empty_dict():
    assert filter_ng123_features({}) == {}


def test_filter_passthrough_when_no_overlap():
    d = {'volume_ratio_5d': 1.0, 'rsi_14': 50.0}
    r = filter_ng123_features(d)
    assert r == d  # no dropped features present → identical output
