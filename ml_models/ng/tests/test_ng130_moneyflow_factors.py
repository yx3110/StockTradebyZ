"""Tests for ng1.3.0 Tier B moneyflow factors."""
import numpy as np
import pytest


def _mock_mf_data(days=20, net_elg=0.0, net_lg=0.0, total_amount=1e8):
    """Build mock moneyflow records sorted oldest→newest."""
    return [
        {
            'buy_elg_amount': total_amount * 0.1 + max(net_elg, 0),
            'sell_elg_amount': total_amount * 0.1 - min(net_elg, 0),
            'buy_lg_amount': total_amount * 0.15 + max(net_lg, 0),
            'sell_lg_amount': total_amount * 0.15 - min(net_lg, 0),
            'buy_md_amount': total_amount * 0.15,
            'sell_md_amount': total_amount * 0.15,
            'buy_sm_amount': total_amount * 0.1,
            'sell_sm_amount': total_amount * 0.1,
            'net_mf_amount': net_elg + net_lg,
        }
        for _ in range(days)
    ]


def test_elg_net_inflow_20d_z_positive_inflow():
    from ml_models.ng.ng130_moneyflow_factors import compute_ng130_mf_factors
    records = _mock_mf_data(days=20, net_elg=1e7)
    z_score_history = [0.0, 0.5, 1.0, 1.5, 2.0]
    result = compute_ng130_mf_factors(records, cs_z_history_elg=z_score_history)
    assert result['elg_net_inflow_20d_z'] > 0


def test_elg_net_inflow_20d_z_outflow():
    from ml_models.ng.ng130_moneyflow_factors import compute_ng130_mf_factors
    records = _mock_mf_data(days=20, net_elg=-1e7)
    z_score_history = [0.0, 0.5, 1.0, 1.5, 2.0]
    result = compute_ng130_mf_factors(records, cs_z_history_elg=z_score_history)
    assert result['elg_net_inflow_20d_z'] < 0


def test_mf_main_ratio_20d_range():
    from ml_models.ng.ng130_moneyflow_factors import compute_ng130_mf_factors
    records = _mock_mf_data(days=20, net_elg=2e6, net_lg=1e6, total_amount=1e8)
    result = compute_ng130_mf_factors(records, cs_z_history_elg=[0])
    assert -1.0 <= result['mf_main_ratio_20d'] <= 1.0
    assert abs(result['mf_main_ratio_20d'] - 0.0015) < 1e-4


def test_mf_concentration_20d_stable():
    from ml_models.ng.ng130_moneyflow_factors import compute_ng130_mf_factors
    records = _mock_mf_data(days=20, net_elg=1e6)
    result = compute_ng130_mf_factors(records, cs_z_history_elg=[0])
    assert result['mf_concentration_20d'] < 0.1


def test_mf_concentration_20d_volatile():
    from ml_models.ng.ng130_moneyflow_factors import compute_ng130_mf_factors
    records = []
    for i in range(20):
        sign = 1 if i % 2 == 0 else -1
        records.append({
            'buy_elg_amount': 5e7 + sign * 5e7,
            'sell_elg_amount': 5e7 - sign * 5e7,
            'buy_lg_amount': 1e7, 'sell_lg_amount': 1e7,
            'buy_md_amount': 1e7, 'sell_md_amount': 1e7,
            'buy_sm_amount': 1e7, 'sell_sm_amount': 1e7,
            'net_mf_amount': sign * 1e8,
        })
    result = compute_ng130_mf_factors(records, cs_z_history_elg=[0])
    assert result['mf_concentration_20d'] > 1.0


def test_empty_records_returns_nan():
    from ml_models.ng.ng130_moneyflow_factors import compute_ng130_mf_factors
    result = compute_ng130_mf_factors([], cs_z_history_elg=[])
    assert np.isnan(result['elg_net_inflow_20d_z'])
    assert np.isnan(result['mf_main_ratio_20d'])
    assert np.isnan(result['mf_concentration_20d'])


def test_all_factor_names_present():
    from ml_models.ng.ng130_moneyflow_factors import compute_ng130_mf_factors, NG130_MF_FACTORS
    records = _mock_mf_data(days=20, net_elg=1e6)
    result = compute_ng130_mf_factors(records, cs_z_history_elg=[0])
    for name in NG130_MF_FACTORS:
        assert name in result
    assert NG130_MF_FACTORS == ('elg_net_inflow_20d_z', 'mf_main_ratio_20d', 'mf_concentration_20d')
