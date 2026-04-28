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


# ─────────────── NG return-scale regression (P2.8 prod bug 2026-04-28) ───────────────

def test_skip_zero_score_picks():
    """Strategy hits with rank_score=0 (ML did not score) should NOT outrank
    legitimately ML-scored picks.

    Regression: 2026-04-28 production found 暴力K hits with rank_score=0
    promoted to rank_score_boosted=8 above real ML picks (rank_score≈0.003).
    """
    picks = [
        # ML-scored top picks
        {'code': '002371', 'rank_score': 0.0028, 'strategy_hits': [], 'trust_tag': '⚪'},
        {'code': '000725', 'rank_score': 0.0023, 'strategy_hits': [], 'trust_tag': '⚪'},
        # Strategy hits without ML scoring
        {'code': '000151', 'rank_score': 0.0, 'strategy_hits': ['暴力K战法'], 'trust_tag': '⚪'},
        {'code': '002677', 'rank_score': 0.0, 'strategy_hits': ['暴力K战法'], 'trust_tag': '⚪'},
    ]
    out = apply_post_rank_booster(picks, regime='bear')
    # ML-scored picks must come first
    assert out[0]['code'] == '002371'
    assert out[1]['code'] == '000725'
    # Zero-score picks should still have rank_score_boosted = 0 (no bonus applied)
    zero_picks = [p for p in out if p['code'] in ('000151', '002677')]
    assert all(p['rank_score_boosted'] == 0.0 for p in zero_picks)
    assert all(p['_booster_strategy_bonus'] == 0.0 for p in zero_picks)


def test_bonus_scaled_to_ng_return_magnitude():
    """When rank_score is in [0, 1) range (NG predicted-return), bonus magnitude
    auto-scales to be proportional, not in 0-100 pts."""
    picks = [
        {'code': 'A', 'rank_score': 0.005, 'strategy_hits': [], 'trust_tag': '⚪'},
        {'code': 'B', 'rank_score': 0.001, 'strategy_hits': ['暴力K战法'], 'trust_tag': '⚪'},
    ]
    out = apply_post_rank_booster(picks, regime='bear')
    # pos_max = 0.005, scale = 0.005/100 = 5e-5; raw bonus 8 → scaled 8 × 5e-5 = 4e-4
    # B becomes 0.001 + 4e-4 = 0.0014, still less than A's 0.005
    a = next(p for p in out if p['code'] == 'A')
    b = next(p for p in out if p['code'] == 'B')
    assert a['rank_score_boosted'] > b['rank_score_boosted']
    # bonus should be in 1e-4 range, not 8
    assert 0 < b['_booster_strategy_bonus'] < 0.01


def test_bonus_scale_legacy_unchanged():
    """0-100 score scale: bonus_scale = 1.0 (no auto-scaling)."""
    picks = [
        {'code': 'A', 'rank_score': 80, 'strategy_hits': [], 'trust_tag': '⚪'},
        {'code': 'B', 'rank_score': 75, 'strategy_hits': ['暴力K战法'], 'trust_tag': '⚪'},
    ]
    out = apply_post_rank_booster(picks, regime='bear')
    # B = 75 + 8 = 83 > A = 80
    b = next(p for p in out if p['code'] == 'B')
    assert b['rank_score_boosted'] == 83.0
    assert b['_booster_strategy_bonus'] == 8.0


def test_skip_when_score_zero_disabled():
    """Opt-out: when skip_when_score_zero=False, zero-score picks DO get bonus."""
    picks = [
        {'code': 'A', 'rank_score': 0.005, 'strategy_hits': [], 'trust_tag': '⚪'},
        {'code': 'B', 'rank_score': 0.0, 'strategy_hits': ['暴力K战法'], 'trust_tag': '⚪'},
    ]
    out = apply_post_rank_booster(picks, regime='bear', skip_when_score_zero=False)
    b = next(p for p in out if p['code'] == 'B')
    # Now B gets bonus = 8 × 0.005/100 = 4e-4, ranks below A still
    assert b['_booster_strategy_bonus'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
