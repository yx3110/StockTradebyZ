"""P0.1: L3 vol-target sizing + L5 SL persistence — failing tests first."""
from __future__ import annotations

import pytest

from stock_selctor.ng21_risk_overlay import RiskDecision, compute_position_size


def _bull_decision():
    return RiskDecision(
        regime='bull', top_n=10, industry_cap=3,
        vol_target_annual=0.25, cash_ceiling=0.20, cash_floor=0.0,
        stop_loss=-0.08, trailing_stop=None, crisis_active=False,
        pos_cap_per_stock=0.10, rebalance_freq_days=15,
    )


def _bear_decision(crisis: bool = False):
    return RiskDecision(
        regime='bear', top_n=5 if crisis else 10, industry_cap=2,
        vol_target_annual=0.15, cash_ceiling=0.50, cash_floor=0.70 if crisis else 0.0,
        stop_loss=-0.04, trailing_stop=-0.06, crisis_active=crisis,
        pos_cap_per_stock=0.05 if crisis else 0.10, rebalance_freq_days=5,
    )


def test_bull_vt_satisfied_equal_weight():
    """Bull, est_vol(0.20) ≤ target(0.25) → 满仓等权 0.10/票, 总和 1.0."""
    picks = [{'code': f'{i:06d}.SZ'} for i in range(10)]
    sized = compute_position_size(picks, _bull_decision(), est_portfolio_vol=0.20)
    weights = [s['position_size'] for s in sized]
    assert all(abs(w - 0.10) < 1e-6 for w in weights)
    assert abs(sum(weights) - 1.0) < 1e-6
    assert all(s['stop_loss_pct'] == -0.08 for s in sized)
    assert all(s['regime'] == 'bull' for s in sized)
    assert all('trailing_stop_pct' not in s for s in sized)


def test_vt_scale_down_when_high_vol():
    """est_vol(0.30) > target(0.15) → 缩仓 50%, 总和 ≈ 0.50."""
    picks = [{'code': f'{i:06d}.SZ'} for i in range(10)]
    sized = compute_position_size(picks, _bear_decision(crisis=False), est_portfolio_vol=0.30)
    total = sum(s['position_size'] for s in sized)
    assert 0.45 < total < 0.55


def test_bear_crisis_caps_and_cash_floor():
    """熊+crisis: 单票 ≤ 5%, 总仓 ≤ 30% (cash_floor=70%)."""
    picks = [{'code': f'{i:06d}.SZ'} for i in range(5)]
    sized = compute_position_size(picks, _bear_decision(crisis=True), est_portfolio_vol=0.40)
    weights = [s['position_size'] for s in sized]
    assert all(w <= 0.05 + 1e-9 for w in weights)
    assert sum(weights) <= 0.30 + 1e-9
    assert all(s['stop_loss_pct'] == -0.04 for s in sized)
    assert all(s['trailing_stop_pct'] == -0.06 for s in sized)
    assert all(s['crisis_active'] is True for s in sized)


def test_pos_cap_caps_individual_when_few_picks():
    """只有 5 票 → 等权应为 20%, 但 pos_cap=10% → 实际 10%/票, 总和 0.50."""
    picks = [{'code': f'{i:06d}.SZ'} for i in range(5)]
    sized = compute_position_size(picks, _bull_decision(), est_portfolio_vol=0.20)
    weights = [s['position_size'] for s in sized]
    assert all(abs(w - 0.10) < 1e-6 for w in weights)
    assert abs(sum(weights) - 0.50) < 1e-6


def test_empty_picks_returns_empty():
    sized = compute_position_size([], _bull_decision(), est_portfolio_vol=0.20)
    assert sized == []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
