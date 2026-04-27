"""P2.1 单测: V_ALPHA 评分卡."""
from __future__ import annotations

import pytest

from backtest.north_star_metrics import (
    V_ALPHA_METRICS,
    compute_v_alpha_score,
    format_v_alpha_report,
)


def test_v_alpha_weights_sum_to_1():
    """权重和应 = 1.0."""
    total = sum(m['weight'] for m in V_ALPHA_METRICS.values())
    assert abs(total - 1.0) < 1e-6, f"weight sum = {total}"


def test_v_alpha_strong_alpha_grade_high():
    """强 alpha (in-sample) 应得高分."""
    metrics = {
        'daily_ic': 0.05,
        'icir': 0.55,
        'ic_positive_pct': 65,
        'excess_annual_return': 0.50,
        'information_ratio': 1.20,
        'excess_win_rate': 75,
        'wfer': 0.45,
        'oos_ic_half_life': 6,
    }
    result = compute_v_alpha_score(metrics, n_trading_days=500)
    assert result['total_pct'] > 60.0, f"strong alpha should get >60%, got {result['total_pct']}"
    assert result['grade'] in ('A', 'A+', 'S')


def test_v_alpha_weak_alpha_grade_low():
    """弱 alpha (Pre-2020 OOS 退化) 应得低分."""
    metrics = {
        'daily_ic': 0.01,
        'icir': 0.08,
        'ic_positive_pct': 52,
        'excess_annual_return': 0.05,
        'information_ratio': 0.20,
        'excess_win_rate': 51,
        'wfer': 0.10,
        'oos_ic_half_life': 1,
    }
    result = compute_v_alpha_score(metrics, n_trading_days=500)
    assert result['total_pct'] < 30.0, f"weak alpha should get <30%, got {result['total_pct']}"
    assert result['grade'] in ('D', 'C')


def test_v_alpha_skip_min_days():
    """n_trading_days 不足时, min_days 标注的 metric 应被跳过."""
    metrics = {
        'daily_ic': 0.05, 'icir': 0.55, 'ic_positive_pct': 65,
        'excess_annual_return': 0.50,    # min_days=200
        'information_ratio': 1.20,        # min_days=120
        'excess_win_rate': 75,
        'wfer': 0.45,
        'oos_ic_half_life': 6,
    }
    result = compute_v_alpha_score(metrics, n_trading_days=90)
    skipped_names = [s[0] for s in result['skipped']]
    assert 'excess_annual_return' in skipped_names
    assert 'information_ratio' in skipped_names


def test_v_alpha_format_no_crash():
    """report formatter 不应 crash."""
    metrics = {'daily_ic': 0.04, 'icir': 0.4, 'ic_positive_pct': 60,
               'excess_annual_return': 0.20, 'information_ratio': 0.8, 'excess_win_rate': 60,
               'wfer': 0.35, 'oos_ic_half_life': 3}
    result = compute_v_alpha_score(metrics, n_trading_days=500)
    text = format_v_alpha_report(result, 'TEST')
    assert 'V_ALPHA' in text
    assert '总分' in text


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
