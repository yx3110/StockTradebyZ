"""Unit tests for ng1.2.3 moneyflow factors."""
import numpy as np
import pytest

from ml_models.ng.ng123_moneyflow_factors import (
    aggregate_moneyflow_window,
    EMPTY_MF_RESULT,
)


def _mk_row(buy_sm=0, sell_sm=0, buy_md=0, sell_md=0,
            buy_lg=0, sell_lg=0, buy_elg=0, sell_elg=0):
    return {
        'buy_sm_amount': buy_sm, 'sell_sm_amount': sell_sm,
        'buy_md_amount': buy_md, 'sell_md_amount': sell_md,
        'buy_lg_amount': buy_lg, 'sell_lg_amount': sell_lg,
        'buy_elg_amount': buy_elg, 'sell_elg_amount': sell_elg,
    }


def test_aggregate_5d_simple():
    """5 days with consistent +100 net_elg each day → sum_net_elg=500."""
    rows = [_mk_row(buy_elg=200, sell_elg=100)] * 5  # net_elg = 100/day
    agg = aggregate_moneyflow_window(rows, n_days=5)
    assert agg['sum_net_elg'] == 500


def test_aggregate_total_amount():
    """sum(buy_total + sell_total) over window."""
    rows = [_mk_row(buy_sm=10, sell_sm=10, buy_lg=20, sell_lg=20)] * 3
    agg = aggregate_moneyflow_window(rows, n_days=3)
    # Per row: total = (10+10) + (20+20) = 60. Over 3 days: 180.
    assert agg['sum_total_amount'] == 180


def test_aggregate_empty_returns_empty():
    """No rows → all NaN scalars + empty arrays."""
    agg = aggregate_moneyflow_window([], n_days=5)
    # Scalar NaN fields
    for k in ['sum_net_sm', 'sum_net_md', 'sum_net_lg', 'sum_net_elg',
              'sum_total_amount']:
        assert np.isnan(agg[k]), f"{k} should be NaN"
    # Array fields (use np.testing for safety)
    import numpy.testing as npt
    for k in ['daily_net_sm', 'daily_net_md', 'daily_net_lg', 'daily_net_elg']:
        npt.assert_array_equal(agg[k], np.array([], dtype=agg[k].dtype))
    assert agg['n_days_actual'] == 0


def test_aggregate_fewer_rows_than_n_days():
    """Only 3 rows but n_days=5 → use what's available."""
    rows = [_mk_row(buy_elg=100, sell_elg=50)] * 3
    agg = aggregate_moneyflow_window(rows, n_days=5)
    assert agg['sum_net_elg'] == 150  # 50 * 3
    assert agg['n_days_actual'] == 3


def test_aggregate_takes_last_n():
    """Function uses LAST n_days rows (most recent)."""
    rows = [_mk_row(buy_elg=10, sell_elg=0)] * 3 + [_mk_row(buy_elg=100, sell_elg=0)] * 5
    agg = aggregate_moneyflow_window(rows, n_days=5)
    # Last 5 rows: each has buy_elg=100. sum_net_elg = 500.
    assert agg['sum_net_elg'] == 500


def test_daily_signs_derived_from_daily_net():
    """Sign of net_elg derivable from daily_net_elg via np.sign."""
    rows = [
        _mk_row(buy_elg=100, sell_elg=50),   # net +50, sign +1
        _mk_row(buy_elg=50, sell_elg=100),   # net -50, sign -1
        _mk_row(buy_elg=100, sell_elg=100),  # net 0, sign 0
    ]
    agg = aggregate_moneyflow_window(rows, n_days=3)
    signs = np.sign(agg['daily_net_elg']).astype(np.int8)
    assert signs.tolist() == [1, -1, 0]


def test_daily_net_arrays_exposed():
    """daily_net_* arrays exist, have correct length, and match per-day values."""
    import numpy.testing as npt
    rows = [
        _mk_row(buy_sm=100, sell_sm=50,    # net_sm=+50
                buy_md=200, sell_md=100,   # net_md=+100
                buy_lg=300, sell_lg=200,   # net_lg=+100
                buy_elg=400, sell_elg=300) # net_elg=+100
    ] * 3
    agg = aggregate_moneyflow_window(rows, n_days=3)
    assert agg['n_days_actual'] == 3
    npt.assert_array_equal(agg['daily_net_sm'],  np.full(3, 50.0))
    npt.assert_array_equal(agg['daily_net_md'],  np.full(3, 100.0))
    npt.assert_array_equal(agg['daily_net_lg'],  np.full(3, 100.0))
    npt.assert_array_equal(agg['daily_net_elg'], np.full(3, 100.0))
    # sum_net_elg must equal sum of daily array
    assert agg['sum_net_elg'] == float(agg['daily_net_elg'].sum())


def test_aggregate_n_days_zero_returns_empty():
    """n_days=0 should return EMPTY_MF_RESULT (degenerate input guard)."""
    rows = [_mk_row(buy_elg=100, sell_elg=50)] * 5
    agg = aggregate_moneyflow_window(rows, n_days=0)
    assert agg['n_days_actual'] == 0
    assert np.isnan(agg['sum_net_elg'])


def test_aggregate_n_days_negative_returns_empty():
    """n_days=-1 should return EMPTY_MF_RESULT."""
    rows = [_mk_row(buy_elg=100)] * 5
    agg = aggregate_moneyflow_window(rows, n_days=-1)
    assert agg['n_days_actual'] == 0
    assert np.isnan(agg['sum_net_elg'])


# --- Group A: Net Flow Magnitude (4 factors) ---------------------------------

def test_mf_net_elg_5d_ratio_basic():
    """5 days each net_elg=+100, total_amount=+1100 per day → ratio = 500/5500."""
    from ml_models.ng.ng123_moneyflow_factors import compute_group_a_factors
    rows = [_mk_row(buy_elg=200, sell_elg=100, buy_sm=400, sell_sm=400)] * 5
    # Per row: net_elg = 100, total = 200+100+400+400 = 1100
    # 5d: sum_net_elg = 500, sum_total = 5500 → ratio = 500/5500 ≈ 0.0909
    res = compute_group_a_factors(rows)
    assert abs(res['mf_net_elg_5d_ratio'] - 500/5500) < 1e-6


def test_mf_net_elg_5d_ratio_zero_total():
    """Edge case: all amounts zero → NaN (not div by zero)."""
    from ml_models.ng.ng123_moneyflow_factors import compute_group_a_factors
    rows = [_mk_row()] * 5  # all zeros
    res = compute_group_a_factors(rows)
    assert np.isnan(res['mf_net_elg_5d_ratio'])


def test_mf_net_elg_20d_ratio():
    """20d ratio aggregates over 20 days when available."""
    from ml_models.ng.ng123_moneyflow_factors import compute_group_a_factors
    rows = [_mk_row(buy_elg=100, sell_elg=50, buy_sm=50, sell_sm=50)] * 25
    # Last 20 used: net_elg=50/d * 20 = 1000; total=(100+50+50+50)/d * 20 = 5000
    # ratio = 1000/5000 = 0.2
    res = compute_group_a_factors(rows)
    assert abs(res['mf_net_elg_20d_ratio'] - 0.2) < 1e-6


def test_mf_net_lg_5d_ratio():
    """Large-order net flow ratio (parallel to elg)."""
    from ml_models.ng.ng123_moneyflow_factors import compute_group_a_factors
    rows = [_mk_row(buy_lg=300, sell_lg=200, buy_sm=100, sell_sm=100)] * 5
    # Per row: net_lg=100, total=300+200+100+100=700; 5d: 500/3500≈0.1429
    res = compute_group_a_factors(rows)
    assert abs(res['mf_net_lg_5d_ratio'] - 500/3500) < 1e-6


def test_mf_smart_net_share_20d():
    """Share of (net_elg+net_lg) over total absolute daily net flow."""
    from ml_models.ng.ng123_moneyflow_factors import compute_group_a_factors
    # net_elg=+100, net_lg=+50, net_md=-30, net_sm=-20 per day, 20 days
    rows = [_mk_row(buy_elg=200, sell_elg=100, buy_lg=150, sell_lg=100,
                    buy_md=70, sell_md=100, buy_sm=80, sell_sm=100)] * 20
    # Per day: net = +100, +50, -30, -20 → smart sum = +150, abs_per_day = 200
    # 20d: smart_num = 3000, abs_daily_total = 20*200 = 4000 → share = 0.75
    res = compute_group_a_factors(rows)
    assert abs(res['mf_smart_net_share_20d'] - 0.75) < 1e-6


def test_group_a_empty_input():
    """No rows → all 4 NaN."""
    from ml_models.ng.ng123_moneyflow_factors import compute_group_a_factors
    res = compute_group_a_factors([])
    assert all(np.isnan(res[k]) for k in
               ['mf_net_elg_5d_ratio', 'mf_net_elg_20d_ratio',
                'mf_net_lg_5d_ratio', 'mf_smart_net_share_20d'])


def test_mf_smart_net_share_20d_with_sign_flips():
    """Days with sign flips: must use sum(|daily_net|), NOT abs(sum_net).

    If the implementation uses abs(sum_net_*) the denominator collapses to 0
    and the result is NaN rather than 0. This test pins the correct behavior.
    """
    from ml_models.ng.ng123_moneyflow_factors import compute_group_a_factors
    # 10 days alternating: +100/-100 for elg, +50/-50 for lg, etc.
    # daily_net_elg: [+100, -100, +100, -100, ...] → sum_net_elg = 0
    # but sum(|daily_net_elg|) = 1000  (per-day absolute values)
    # Wrong impl: abs(sum_net_elg)=0 → denominator=0 → NaN
    # Correct impl: sum(|daily_net_elg|)=1000 → denominator=4000 → share=0
    rows_pos = _mk_row(buy_elg=200, sell_elg=100, buy_lg=150, sell_lg=100,
                       buy_md=70, sell_md=100, buy_sm=80, sell_sm=100)
    rows_neg = _mk_row(buy_elg=100, sell_elg=200, buy_lg=100, sell_lg=150,
                       buy_md=100, sell_md=70, buy_sm=100, sell_sm=80)
    rows = [rows_pos, rows_neg] * 10  # 20 days alternating
    res = compute_group_a_factors(rows)
    # smart numerator = 0 (sign cancellation), denominator = 20*(100+50+30+20)=4000
    # share = 0 / 4000 = 0
    assert abs(res['mf_smart_net_share_20d']) < 1e-9, \
        f"Expected ~0 (sign cancellation in numerator), got {res['mf_smart_net_share_20d']}"
