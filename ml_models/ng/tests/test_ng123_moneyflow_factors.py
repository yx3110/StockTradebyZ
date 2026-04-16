"""Unit tests for ng1.2.3 moneyflow factors."""
import numpy as np
import pytest

from ml_models.ng.ng123_moneyflow_factors import (
    ACCEPTED_MF_FACTORS,
    aggregate_moneyflow_window,
    EMPTY_MF_RESULT,
    compute_group_a_factors,
    compute_group_b_factors,
    compute_group_c_factors,
    compute_stock_mf_scalars,
    compute_group_d_factors,
    compute_all_moneyflow_factors,
    _SHORT_WINDOW,
    _LONG_WINDOW,
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
    rows = [_mk_row(buy_elg=200, sell_elg=100, buy_sm=400, sell_sm=400)] * 5
    # Per row: net_elg = 100, total = 200+100+400+400 = 1100
    # 5d: sum_net_elg = 500, sum_total = 5500 → ratio = 500/5500 ≈ 0.0909
    res = compute_group_a_factors(rows)
    assert abs(res['mf_net_elg_5d_ratio'] - 500/5500) < 1e-6


def test_mf_net_elg_5d_ratio_zero_total():
    """Edge case: all amounts zero → NaN (not div by zero)."""
    rows = [_mk_row()] * 5  # all zeros
    res = compute_group_a_factors(rows)
    assert np.isnan(res['mf_net_elg_5d_ratio'])


def test_mf_net_elg_20d_ratio():
    """20d ratio aggregates over 20 days when available."""
    rows = [_mk_row(buy_elg=100, sell_elg=50, buy_sm=50, sell_sm=50)] * 25
    # Last 20 used: net_elg=50/d * 20 = 1000; total=(100+50+50+50)/d * 20 = 5000
    # ratio = 1000/5000 = 0.2
    res = compute_group_a_factors(rows)
    assert abs(res['mf_net_elg_20d_ratio'] - 0.2) < 1e-6


def test_mf_net_lg_5d_ratio():
    """Large-order net flow ratio (parallel to elg)."""
    rows = [_mk_row(buy_lg=300, sell_lg=200, buy_sm=100, sell_sm=100)] * 5
    # Per row: net_lg=100, total=300+200+100+100=700; 5d: 500/3500≈0.1429
    res = compute_group_a_factors(rows)
    assert abs(res['mf_net_lg_5d_ratio'] - 500/3500) < 1e-6


def test_mf_smart_net_share_20d():
    """Share of (net_elg+net_lg) over total absolute daily net flow."""
    # net_elg=+100, net_lg=+50, net_md=-30, net_sm=-20 per day, 20 days
    rows = [_mk_row(buy_elg=200, sell_elg=100, buy_lg=150, sell_lg=100,
                    buy_md=70, sell_md=100, buy_sm=80, sell_sm=100)] * 20
    # Per day: net = +100, +50, -30, -20 → smart sum = +150, abs_per_day = 200
    # 20d: smart_num = 3000, abs_daily_total = 20*200 = 4000 → share = 0.75
    res = compute_group_a_factors(rows)
    assert abs(res['mf_smart_net_share_20d'] - 0.75) < 1e-6


def test_group_a_empty_input():
    """No rows → all 4 NaN."""
    res = compute_group_a_factors([])
    assert all(np.isnan(res[k]) for k in
               ['mf_net_elg_5d_ratio', 'mf_net_elg_20d_ratio',
                'mf_net_lg_5d_ratio', 'mf_smart_net_share_20d'])


def test_mf_smart_net_share_20d_with_sign_flips():
    """Days with sign flips: must use sum(|daily_net|), NOT abs(sum_net).

    If the implementation uses abs(sum_net_*) the denominator collapses to 0
    and the result is NaN rather than 0. This test pins the correct behavior.
    """
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


def test_compute_group_a_factors_partial_window():
    """4 rows for a 5d-and-20d factor pair: both compute from 4 rows; documented behavior."""
    # 4 rows; both 5d and 20d factors will use these 4 rows
    rows = [_mk_row(buy_elg=200, sell_elg=100, buy_sm=400, sell_sm=400)] * 4
    res = compute_group_a_factors(rows)
    # Per row: net_elg=100, total=1100; 4 rows: net=400, total=4400
    expected = 400 / 4400
    assert abs(res['mf_net_elg_5d_ratio'] - expected) < 1e-6
    # 20d factor uses same 4 rows → identical value (documented behavior)
    assert abs(res['mf_net_elg_20d_ratio'] - expected) < 1e-6
    assert abs(res['mf_net_elg_5d_ratio'] - res['mf_net_elg_20d_ratio']) < 1e-9


# --- Group B: Persistence (2 factors) --------------------------------------

def test_mf_elg_persistence_20d_all_positive():
    """20 days all net_elg > 0 → persistence = +1.0"""
    rows = [_mk_row(buy_elg=100, sell_elg=50)] * 20
    res = compute_group_b_factors(rows)
    assert abs(res['mf_elg_persistence_20d'] - 1.0) < 1e-9


def test_mf_elg_persistence_20d_mixed():
    """10 positive + 10 negative → persistence = 0."""
    pos = _mk_row(buy_elg=100, sell_elg=50)
    neg = _mk_row(buy_elg=50, sell_elg=100)
    rows = [pos] * 10 + [neg] * 10
    res = compute_group_b_factors(rows)
    assert abs(res['mf_elg_persistence_20d'] - 0.0) < 1e-9


def test_mf_smart_consistency_5d_all_aligned():
    """All 5 days net_elg sign = net_lg sign → consistency = 1.0"""
    rows = [_mk_row(buy_elg=100, sell_elg=50, buy_lg=80, sell_lg=40)] * 5
    res = compute_group_b_factors(rows)
    assert abs(res['mf_smart_consistency_5d'] - 1.0) < 1e-9


def test_mf_smart_consistency_5d_misaligned():
    """3 aligned + 2 misaligned → consistency = 0.6"""
    aligned = _mk_row(buy_elg=100, sell_elg=50, buy_lg=80, sell_lg=40)   # both +
    misaligned = _mk_row(buy_elg=100, sell_elg=50, buy_lg=40, sell_lg=80)  # elg+, lg-
    rows = [aligned] * 3 + [misaligned] * 2
    res = compute_group_b_factors(rows)
    assert abs(res['mf_smart_consistency_5d'] - 0.6) < 1e-9


# --- Group C: Divergence + Acceleration (2 factors) ------------------------

def test_mf_smart_retail_divergence_5d_smart_in_retail_out():
    """net_elg_5d > 0, net_sm_5d < 0 → divergence = sign(+) - sign(-) = +2"""
    rows = [_mk_row(buy_elg=200, sell_elg=100,   # net_elg = +100
                    buy_sm=50, sell_sm=200)] * 5  # net_sm = -150
    res = compute_group_c_factors(rows)
    assert res['mf_smart_retail_divergence_5d'] == 2


def test_mf_smart_retail_divergence_5d_aligned():
    """Both positive → divergence = 0"""
    rows = [_mk_row(buy_elg=200, sell_elg=100, buy_sm=200, sell_sm=100)] * 5
    res = compute_group_c_factors(rows)
    assert res['mf_smart_retail_divergence_5d'] == 0


def test_mf_elg_acceleration_5_20():
    """5d ratio - 20d ratio: if 5d more positive, acceleration positive."""
    # 15 days neutral (net_elg=0), 5 days positive (net_elg=+100)
    neutral = _mk_row(buy_elg=100, sell_elg=100, buy_sm=0, sell_sm=0)   # total=200
    positive = _mk_row(buy_elg=200, sell_elg=100, buy_sm=0, sell_sm=0)  # net=+100, total=300
    rows = [neutral] * 15 + [positive] * 5
    res = compute_group_c_factors(rows)
    # 5d: net_elg_sum=500, total=1500 → ratio=0.333
    # 20d: net_elg_sum=500, total=15*200+5*300=4500 → ratio=0.111
    # acc = 0.333 - 0.111 = 0.222
    assert abs(res['mf_elg_acceleration_5_20'] - (500/1500 - 500/4500)) < 1e-6


# --- Fix #3 (m-2): Empty-input tests for Groups B and C --------------------

def test_compute_group_b_factors_empty_input():
    """Empty rows → both factors NaN."""
    res = compute_group_b_factors([])
    assert np.isnan(res['mf_elg_persistence_20d'])
    assert np.isnan(res['mf_smart_consistency_5d'])


def test_compute_group_c_factors_empty_input():
    """Empty rows → both factors NaN."""
    res = compute_group_c_factors([])
    assert np.isnan(res['mf_smart_retail_divergence_5d'])
    assert np.isnan(res['mf_elg_acceleration_5_20'])


# --- Fix #4 (m-3): Bearish divergence test for Factor 7 --------------------

def test_mf_smart_retail_divergence_5d_smart_out_retail_in():
    """net_elg_5d < 0, net_sm_5d > 0 → divergence = sign(-) - sign(+) = -2 (bearish)"""
    rows = [_mk_row(buy_elg=50, sell_elg=200,  # net_elg = -150
                    buy_sm=200, sell_sm=50)] * 5  # net_sm = +150
    res = compute_group_c_factors(rows)
    assert res['mf_smart_retail_divergence_5d'] == -2


# --- Fix #6 (partial-window tests to PIN spec-literal divisor behavior) ----

def test_mf_elg_persistence_20d_partial_window():
    """3 days all positive → persistence = 3/20 = 0.15 (spec literal divisor)."""
    rows = [_mk_row(buy_elg=100, sell_elg=50)] * 3
    res = compute_group_b_factors(rows)
    assert abs(res['mf_elg_persistence_20d'] - 3/20) < 1e-9


def test_mf_smart_consistency_5d_partial_window():
    """3 days all aligned → consistency = 3/5 = 0.6 (spec literal divisor)."""
    rows = [_mk_row(buy_elg=100, sell_elg=50, buy_lg=80, sell_lg=40)] * 3
    res = compute_group_b_factors(rows)
    assert abs(res['mf_smart_consistency_5d'] - 3/5) < 1e-9


# --- Group D: Cross-Sectional Industry Ranks (4 factors) --------------------

def test_compute_stock_mf_scalars_for_cs_rank():
    """Helper that returns scalar values needed for cs_rank wrapper."""
    rows = [_mk_row(buy_elg=200, sell_elg=100, buy_sm=400, sell_sm=400)] * 20
    s = compute_stock_mf_scalars(rows)
    assert 'net_elg_5d_ratio' in s
    assert 'net_elg_20d_ratio' in s
    assert 'smart_net_share_20d' in s
    assert 'persistence_20d' in s


def test_compute_stock_mf_scalars_empty_input():
    """Empty rows → all 4 scalars NaN."""
    s = compute_stock_mf_scalars([])
    assert all(np.isnan(s[k]) for k in s)


def test_compute_group_d_factors_basic():
    """cs_rank factors return percentile rank in [0, 1]."""
    stock_scalars = {
        'net_elg_5d_ratio': 0.10,
        'net_elg_20d_ratio': 0.05,
        'smart_net_share_20d': 0.30,
        'persistence_20d': 0.40,
    }
    peer_scalars = {
        'net_elg_5d_ratio': np.array([0.02, 0.05, 0.08, 0.10, 0.15]),
        'net_elg_20d_ratio': np.array([0.01, 0.03, 0.05, 0.05, 0.10]),
        'smart_net_share_20d': np.array([-0.20, 0.10, 0.30, 0.40, 0.50]),
        'persistence_20d': np.array([-0.50, 0.0, 0.40, 0.60, 0.80]),
    }
    res = compute_group_d_factors(stock_scalars, peer_scalars)
    assert 'cs_rank_mf_net_elg_5d' in res
    assert 'cs_rank_mf_net_elg_20d' in res
    assert 'cs_rank_mf_smart_net_share_20d' in res
    assert 'cs_rank_mf_elg_persistence_20d' in res
    for v in res.values():
        assert 0.0 <= v <= 1.0
    # Stock at 0.10 with peers [0.02, 0.05, 0.08, 0.10, 0.15] → 3/5 strictly less
    assert abs(res['cs_rank_mf_net_elg_5d'] - 3/5) < 1e-9


def test_compute_group_d_factors_empty_peers():
    """No peers → return 0.5 (neutral)."""
    stock_scalars = {
        'net_elg_5d_ratio': 0.10, 'net_elg_20d_ratio': 0.05,
        'smart_net_share_20d': 0.30, 'persistence_20d': 0.40,
    }
    peer_scalars = {
        'net_elg_5d_ratio': np.array([]),
        'net_elg_20d_ratio': np.array([]),
        'smart_net_share_20d': np.array([]),
        'persistence_20d': np.array([]),
    }
    res = compute_group_d_factors(stock_scalars, peer_scalars)
    for v in res.values():
        assert v == 0.5


def test_compute_all_moneyflow_factors_returns_6_accepted():
    """Default returns only the 6 Stage 1-accepted factors (ng101 median+ bar)."""
    rows = [_mk_row(buy_elg=200, sell_elg=100, buy_sm=400, sell_sm=400)] * 20
    res = compute_all_moneyflow_factors(rows)
    expected_keys = {
        'mf_net_elg_20d_ratio', 'cs_rank_mf_net_elg_20d',
        'mf_net_elg_5d_ratio', 'cs_rank_mf_net_elg_5d',
        'mf_smart_net_share_20d', 'cs_rank_mf_smart_net_share_20d',
    }
    assert set(res.keys()) == expected_keys, \
        f"Mismatch: {set(res.keys()) ^ expected_keys}"


def test_compute_all_moneyflow_factors_returns_12_when_accepted_only_false():
    """accepted_only=False returns all 12 factors (diagnostic mode)."""
    rows = [_mk_row(buy_elg=200, sell_elg=100, buy_sm=400, sell_sm=400)] * 20
    res = compute_all_moneyflow_factors(rows, accepted_only=False)
    assert len(res) == 12, f"Expected all 12 factors, got {len(res)}"


def test_compute_all_moneyflow_factors_empty_input():
    """Empty rows → 3 Group A kept factors NaN, 3 Group D kept factors 0.5 (empty peers default)."""
    res = compute_all_moneyflow_factors([])
    # 3 Group A factors accepted
    for k in ['mf_net_elg_5d_ratio', 'mf_net_elg_20d_ratio', 'mf_smart_net_share_20d']:
        assert np.isnan(res[k]), f"{k} should be NaN"
    # 3 Group D factors accepted
    for k in ['cs_rank_mf_net_elg_5d', 'cs_rank_mf_net_elg_20d', 'cs_rank_mf_smart_net_share_20d']:
        assert res[k] == 0.5, f"{k} should be 0.5 (empty peers)"


def test_compute_group_d_factors_nan_stock_value():
    """Stock value NaN with valid peers → 0.5 (neutral, not 0.0).

    Pins the behavioral fix in _industry_percentile_rank_safe vs the original
    in ng_feature_calculator (which would return 0.0 for NaN < anything).
    """
    stock_scalars = {
        'net_elg_5d_ratio': np.nan,
        'net_elg_20d_ratio': 0.05,
        'smart_net_share_20d': 0.30,
        'persistence_20d': 0.40,
    }
    peer_scalars = {
        'net_elg_5d_ratio': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        'net_elg_20d_ratio': np.array([0.01, 0.03, 0.05, 0.08, 0.10]),
        'smart_net_share_20d': np.array([-0.20, 0.10, 0.30, 0.40, 0.50]),
        'persistence_20d': np.array([-0.50, 0.0, 0.40, 0.60, 0.80]),
    }
    res = compute_group_d_factors(stock_scalars, peer_scalars)
    # NaN value with valid peers → neutral 0.5
    assert res['cs_rank_mf_net_elg_5d'] == 0.5, \
        f"NaN stock value should give neutral 0.5, got {res['cs_rank_mf_net_elg_5d']}"
    # Other factors compute normally
    assert res['cs_rank_mf_net_elg_20d'] != 0.5  # 0.05 is somewhere in peer range


def test_compute_stock_mf_scalars_parity_with_groups():
    """Pin the contract: scalar values match Group A/B factor outputs.

    This is the parity that compute_all_moneyflow_factors's inline extraction
    relies on. If a future refactor changes Group A/B formulas without updating
    compute_stock_mf_scalars, the batch path (Task 8) would silently use stale
    values for cs_rank computation.
    """
    rows = [_mk_row(buy_elg=200, sell_elg=100, buy_lg=150, sell_lg=80,
                     buy_md=70, sell_md=100, buy_sm=80, sell_sm=100)] * 20
    a = compute_group_a_factors(rows)
    b = compute_group_b_factors(rows)
    s = compute_stock_mf_scalars(rows)
    assert s['net_elg_5d_ratio'] == a['mf_net_elg_5d_ratio']
    assert s['net_elg_20d_ratio'] == a['mf_net_elg_20d_ratio']
    assert s['smart_net_share_20d'] == a['mf_smart_net_share_20d']
    assert s['persistence_20d'] == b['mf_elg_persistence_20d']


# ---------------------------------------------------------------------------
# ng1.2.4: top-2 factor mode
# ---------------------------------------------------------------------------

def test_ng124_returns_2_factors():
    """ng124_mode=True returns exactly the top-2 P90 factors."""
    rows = [_mk_row(buy_elg=200, sell_elg=100, buy_sm=400, sell_sm=400)] * 20
    res = compute_all_moneyflow_factors(rows, ng124_mode=True)
    expected = {'mf_net_elg_20d_ratio', 'cs_rank_mf_net_elg_20d'}
    assert set(res.keys()) == expected


def test_ng124_mode_overrides_accepted_only_false():
    """ng124_mode=True overrides accepted_only=False (still returns 2)."""
    rows = [_mk_row(buy_elg=200, sell_elg=100, buy_sm=400, sell_sm=400)] * 20
    res = compute_all_moneyflow_factors(rows, accepted_only=False, ng124_mode=True)
    assert len(res) == 2
