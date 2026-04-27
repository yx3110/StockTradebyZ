"""P0.1: integration test — confirm selector's overlay branch wires
apply_overlay_to_picks + compute_position_size + estimate_portfolio_vol end-to-end
without needing a full daily SELECT run. Mocks the DB-touching estimate_portfolio_vol.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from stock_selctor.ng21_risk_overlay import (
    RiskDecision, apply_overlay_to_picks, compute_position_size,
)


@pytest.fixture
def bull_decision():
    return RiskDecision(
        regime='bull', top_n=10, industry_cap=3,
        vol_target_annual=0.25, cash_ceiling=0.20, cash_floor=0.0,
        stop_loss=-0.08, trailing_stop=None, crisis_active=False,
        pos_cap_per_stock=0.10, rebalance_freq_days=15,
    )


def _fake_picks(n=15, scores=None, industries=None):
    out = []
    for i in range(n):
        out.append({
            'code': f'{i:06d}.SZ',
            'rank_score': (scores[i] if scores else 100 - i),
            'industry': (industries[i] if industries else f'ind_{i % 5}'),
        })
    return out


def test_pipeline_apply_overlay_then_size(bull_decision):
    """L1+L2 截断 + L3 sizing 链式: 15 票进, 经 overlay 截到 top_n=10 等权 0.10 = 100%."""
    picks = _fake_picks(15)
    kept, dropped = apply_overlay_to_picks(picks, bull_decision)
    assert len(kept) == 10  # top_n cap
    sized = compute_position_size(kept, bull_decision, est_portfolio_vol=0.20)
    assert all('position_size' in s for s in sized)
    assert all('stop_loss_pct' in s for s in sized)
    assert all(s['regime'] == 'bull' for s in sized)
    total = sum(s['position_size'] for s in sized)
    assert abs(total - 1.0) < 1e-6


def test_pipeline_industry_cap_then_size(bull_decision):
    """同行业 5 票, cap=3 → 留 3 + 其它行业票, sizing 正确."""
    industries = ['CHEM'] * 5 + [f'ind_{i}' for i in range(10)]
    picks = _fake_picks(15, industries=industries)
    kept, dropped = apply_overlay_to_picks(picks, bull_decision)
    chem_kept = [s for s in kept if s.get('industry') == 'CHEM']
    assert len(chem_kept) <= 3
    sized = compute_position_size(kept, bull_decision, est_portfolio_vol=0.20)
    assert len(sized) == len(kept)


def test_score_floor_drops_low_score(bull_decision):
    """L1 score < 30 → drop. 5 票 score >= 30, 10 票 score < 30 → 全保留 5 票."""
    scores = [80, 70, 60, 50, 40] + [25, 20, 15, 10, 5] * 2
    picks = _fake_picks(15, scores=scores)
    kept, dropped = apply_overlay_to_picks(picks, bull_decision)
    assert len(kept) == 5
    assert all(s.get('rank_score', 0) >= 30 for s in kept)


@patch('stock_selctor.ng21_risk_overlay.sqlite3')
def test_estimate_portfolio_vol_db_failure_falls_back(mock_sqlite):
    """DB error 时返回保守 0.25, 不抛."""
    from stock_selctor.ng21_risk_overlay import estimate_portfolio_vol
    import sqlite3 as real_sqlite3

    class FakeConn:
        def execute(self, *a, **k):
            raise real_sqlite3.Error("simulated")
        def close(self):
            pass

    mock_sqlite.connect.return_value = FakeConn()
    mock_sqlite.Error = real_sqlite3.Error
    picks = [{'code': f'{i:06d}.SZ'} for i in range(5)]
    vol = estimate_portfolio_vol(picks, '/dummy', '2026-04-24')
    assert vol == 0.25


def test_estimate_portfolio_vol_empty_picks():
    from stock_selctor.ng21_risk_overlay import estimate_portfolio_vol
    assert estimate_portfolio_vol([], '/dummy', '2026-04-24') == 0.25


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
