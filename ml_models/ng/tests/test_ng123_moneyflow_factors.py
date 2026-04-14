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
    assert agg['sum_buy_elg'] == 1000
    assert agg['sum_sell_elg'] == 500


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
              'sum_buy_elg', 'sum_sell_elg', 'sum_total_amount']:
        assert np.isnan(agg[k]), f"{k} should be NaN"
    # Array fields (use np.testing for safety)
    import numpy.testing as npt
    for k in ['daily_sign_net_elg', 'daily_sign_net_lg', 'daily_sign_net_sm',
              'daily_net_sm', 'daily_net_md', 'daily_net_lg', 'daily_net_elg']:
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


def test_daily_signs():
    """Daily sign array length matches n_days, in correct order (oldest → newest)."""
    rows = [
        _mk_row(buy_elg=100, sell_elg=50),   # net +50, sign +1
        _mk_row(buy_elg=50, sell_elg=100),   # net -50, sign -1
        _mk_row(buy_elg=100, sell_elg=100),  # net 0, sign 0
    ]
    agg = aggregate_moneyflow_window(rows, n_days=3)
    assert agg['daily_sign_net_elg'].tolist() == [1, -1, 0]


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
