"""Regression tests for ng1.3.0 NGTrainer._compute_cache_key.

Guards Round 1 bug: cache key that doesn't include head caused V485Trainer to
reuse excess-labeled X/y when training downside head, producing bit-identical
models. See commit ee9ccb86 for the fix.
"""
from ml_models.ng.ng_trainer import NGTrainer


_START = '2020-01-01'
_END = '2026-04-17'


def test_cache_key_differentiates_heads_ng130():
    """ng1.3.0 excess vs downside must produce different cache keys."""
    t_ex = NGTrainer(version='ng1.3.0', head='excess')
    t_dn = NGTrainer(version='ng1.3.0', head='downside')
    key_ex = t_ex._compute_cache_key(_START, _END)
    key_dn = t_dn._compute_cache_key(_START, _END)
    assert key_ex != key_dn, (
        f'ng1.3.0 excess vs downside MUST differ to prevent bit-identical models. '
        f'Got both={key_ex}'
    )


def test_cache_key_stable_within_head():
    """Same head/version/dates yields same key (idempotent)."""
    t1 = NGTrainer(version='ng1.3.0', head='excess')
    t2 = NGTrainer(version='ng1.3.0', head='excess')
    assert t1._compute_cache_key(_START, _END) == t2._compute_cache_key(_START, _END)


def test_cache_key_unchanged_for_non_ng130_versions():
    """ng1.0.1 default 'excess' head produces no head_suffix (backward compat)."""
    t = NGTrainer(version='ng1.0.1')
    assert t._head == 'excess'
    key = t._compute_cache_key(_START, _END)
    assert len(key) == 12  # md5 truncation


def test_cache_key_differs_across_versions():
    """Different versions always produce different keys (prevents cache reuse across schemas)."""
    t101 = NGTrainer(version='ng1.0.1')
    t130 = NGTrainer(version='ng1.3.0', head='excess')
    assert t101._compute_cache_key(_START, _END) != t130._compute_cache_key(_START, _END)


def test_cache_key_differs_across_dates():
    """Different date ranges produce different keys."""
    t = NGTrainer(version='ng1.3.0', head='downside')
    k1 = t._compute_cache_key('2020-01-01', '2024-01-01')
    k2 = t._compute_cache_key('2020-01-01', '2026-04-17')
    assert k1 != k2
