"""P2.8b: regression — trust tags from signal_trust_scores have Chinese
descriptor suffixes ('🟢可信', '🔴高风险'). Booster must match by leading emoji."""
from __future__ import annotations

import pytest

from stock_selctor.post_rank_booster import apply_post_rank_booster, _trust_lookup


def test_trust_lookup_with_descriptor():
    assert _trust_lookup('🟢可信') == 1.00
    assert _trust_lookup('🟡存疑') == 0.85
    assert _trust_lookup('🔴高风险') == 0.60
    assert _trust_lookup('⚪数据不足') == 1.00


def test_trust_lookup_with_emoji_only():
    assert _trust_lookup('🟢') == 1.00
    assert _trust_lookup('🔴') == 0.60


def test_trust_lookup_none_or_empty():
    assert _trust_lookup(None) == 1.00
    assert _trust_lookup('') == 1.00


def test_trust_lookup_unknown_tag():
    assert _trust_lookup('未分类') == 1.00


def test_apply_booster_with_descriptor_tag():
    """End-to-end: 🔴高风险 should still penalize × 0.6."""
    picks = [{'code': '001', 'rank_score': 100,
              'strategy_hits': [], 'trust_tag': '🔴高风险'}]
    out = apply_post_rank_booster(picks, regime='bull')
    assert out[0]['rank_score_boosted'] == 60.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
