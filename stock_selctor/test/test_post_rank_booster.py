"""P2.8: tests for post-rank booster (8-strategy + signal trust)."""
from __future__ import annotations

import pytest

from stock_selctor.post_rank_booster import (
    apply_post_rank_booster, STRATEGY_BONUS_BY_REGIME, DEAD_STRATEGIES, TRUST_MULT,
)


def _pick(code, score, strategies=None, trust='⚪'):
    return {
        'code': code, 'rank_score': score,
        'strategy_hits': strategies or [], 'trust_tag': trust,
    }


def test_no_booster_passthrough():
    picks = [_pick('001', 100), _pick('002', 80)]
    out = apply_post_rank_booster(picks, regime='bull')
    assert [p['code'] for p in out] == ['001', '002']
    assert all(p['rank_score_boosted'] == p['rank_score'] for p in out)


def test_bull_strategy_bonus():
    picks = [_pick('001', 100, ['少负战法']), _pick('002', 105)]
    out = apply_post_rank_booster(picks, regime='bull')
    # 001 = (100+8)*1.0 = 108, 002 = 105 → 001 wins
    assert out[0]['code'] == '001'
    assert out[0]['rank_score_boosted'] == 108.0


def test_bear_strategy_swaps_top():
    picks = [_pick('001', 100, ['少负战法']), _pick('002', 95, ['暴力K战法'])]
    out = apply_post_rank_booster(picks, regime='bear')
    # bear: 少负 has no bonus (not in bear table), 暴力K=+8 → 002=(95+8)=103, 001=100
    assert out[0]['code'] == '002'
    assert out[0]['rank_score_boosted'] == 103.0
    assert out[1]['rank_score_boosted'] == 100.0


def test_dead_strategies_get_no_bonus():
    """知行/填坑/TePu/上穿60放量 are wiki-confirmed deadweight."""
    picks = [_pick('001', 100, ['知行战法', 'TePu战法', '填坑战法', '上穿60放量战法'])]
    out = apply_post_rank_booster(picks, regime='bull')
    assert out[0]['rank_score_boosted'] == 100.0
    assert out[0]['_booster_strategy_bonus'] == 0.0


def test_red_trust_penalizes_score():
    picks = [_pick('001', 100, trust='🔴'), _pick('002', 80)]
    out = apply_post_rank_booster(picks, regime='bull')
    # 001 = 100 * 0.60 = 60, 002 = 80 → 002 wins
    assert out[0]['code'] == '002'
    assert out[1]['code'] == '001'
    assert out[1]['rank_score_boosted'] == 60.0


def test_yellow_trust_mild_penalty():
    picks = [_pick('001', 100, trust='🟡')]
    out = apply_post_rank_booster(picks, regime='bull')
    assert out[0]['rank_score_boosted'] == 85.0


def test_combined_bonus_and_penalty():
    """少负 +8 then 🔴 ×0.6 → (100+8)*0.6 = 64.8."""
    picks = [_pick('001', 100, ['少负战法'], trust='🔴')]
    out = apply_post_rank_booster(picks, regime='bull')
    assert abs(out[0]['rank_score_boosted'] - 64.8) < 1e-6


def test_top_n_trim():
    picks = [_pick(f'{i:03d}', 100 - i) for i in range(20)]
    out = apply_post_rank_booster(picks, regime='bull', top_n=5)
    assert len(out) == 5
    assert [p['code'] for p in out] == ['000', '001', '002', '003', '004']


def test_unknown_regime_no_bonus():
    picks = [_pick('001', 100, ['少负战法'])]
    out = apply_post_rank_booster(picks, regime='sideways')  # not configured
    assert out[0]['rank_score_boosted'] == 100.0


def test_input_picks_not_mutated():
    picks = [_pick('001', 100, ['少负战法'])]
    apply_post_rank_booster(picks, regime='bull')
    assert 'rank_score_boosted' not in picks[0]


def test_constants_align_with_wiki():
    """Sanity: dead strategies absent from any regime bonus table."""
    for regime, table in STRATEGY_BONUS_BY_REGIME.items():
        for strat in DEAD_STRATEGIES:
            assert strat not in table, f"{strat} should not grant bonus in {regime}"


def test_trust_table_complete():
    """All 4 trust tags must have multipliers."""
    for tag in ('🟢', '🟡', '🔴', '⚪'):
        assert tag in TRUST_MULT


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
