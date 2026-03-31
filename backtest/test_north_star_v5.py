"""北极星V5单元测试"""
import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFactorReturns:
    """因子收益构建测试"""

    def test_build_factor_returns_columns(self):
        """构建结果应包含4因子列"""
        from backtest.factor_returns import build_factor_returns
        df = build_factor_returns('2025-01-01', '2025-01-31')
        assert 'MKT' in df.columns
        assert 'SMB' in df.columns
        assert 'HML' in df.columns
        assert 'UMD' in df.columns
        assert len(df) > 0

    def test_factor_returns_no_extreme_values(self):
        """因子日收益应在合理范围 (-15%, +15%)"""
        from backtest.factor_returns import build_factor_returns
        df = build_factor_returns('2025-01-01', '2025-03-31')
        for col in ['MKT', 'SMB', 'HML', 'UMD']:
            assert df[col].abs().max() < 0.15, f"{col} has extreme value"

    def test_factor_returns_low_correlation(self):
        """因子间相关性应 < 0.5"""
        from backtest.factor_returns import build_factor_returns
        df = build_factor_returns('2024-01-01', '2025-01-01')
        corr = df[['SMB', 'HML', 'UMD']].corr()
        for i in range(3):
            for j in range(i+1, 3):
                assert abs(corr.iloc[i, j]) < 0.5, \
                    f"High correlation: {corr.columns[i]} vs {corr.columns[j]} = {corr.iloc[i,j]:.3f}"


class TestScoreMetricV5:
    def test_at_target_returns_5(self):
        from backtest.north_star_metrics import score_metric_v5
        t = {'pass': 0.03, 'ok': 0.04, 'good': 0.05, 'great': 0.06, 'target': 0.08, 'direction': 'higher'}
        assert score_metric_v5(0.08, t) == 5.0

    def test_above_target_capped_at_5(self):
        from backtest.north_star_metrics import score_metric_v5
        t = {'pass': 0.03, 'ok': 0.04, 'good': 0.05, 'great': 0.06, 'target': 0.08, 'direction': 'higher'}
        assert score_metric_v5(0.12, t) == 5.0

    def test_at_pass_returns_1(self):
        from backtest.north_star_metrics import score_metric_v5
        t = {'pass': 0.03, 'ok': 0.04, 'good': 0.05, 'great': 0.06, 'target': 0.08, 'direction': 'higher'}
        assert score_metric_v5(0.03, t) == 1.0

    def test_below_pass_is_fraction(self):
        from backtest.north_star_metrics import score_metric_v5
        t = {'pass': 0.03, 'ok': 0.04, 'good': 0.05, 'great': 0.06, 'target': 0.08, 'direction': 'higher'}
        score = score_metric_v5(0.015, t)
        assert 0 < score < 1.0

    def test_zero_returns_zero(self):
        from backtest.north_star_metrics import score_metric_v5
        t = {'pass': 0.03, 'ok': 0.04, 'good': 0.05, 'great': 0.06, 'target': 0.08, 'direction': 'higher'}
        assert score_metric_v5(0.0, t) == 0.0

    def test_midpoint_interpolation(self):
        from backtest.north_star_metrics import score_metric_v5
        t = {'pass': 0.03, 'ok': 0.04, 'good': 0.05, 'great': 0.06, 'target': 0.08, 'direction': 'higher'}
        score = score_metric_v5(0.059, t)
        assert 3.5 < score < 4.0

    def test_direction_lower(self):
        from backtest.north_star_metrics import score_metric_v5
        t = {'pass': 2.0, 'ok': 1.5, 'good': 1.0, 'great': 0.8, 'target': 0.6, 'direction': 'lower'}
        assert score_metric_v5(0.6, t) == 5.0
        assert score_metric_v5(2.0, t) == 1.0

    def test_direction_lower_interpolation(self):
        from backtest.north_star_metrics import score_metric_v5
        t = {'pass': 2.0, 'ok': 1.5, 'good': 1.0, 'great': 0.8, 'target': 0.6, 'direction': 'lower'}
        score = score_metric_v5(0.9, t)
        assert 3.0 < score < 4.0

    def test_direction_lower_worse_than_pass(self):
        from backtest.north_star_metrics import score_metric_v5
        t = {'pass': 2.0, 'ok': 1.5, 'good': 1.0, 'great': 0.8, 'target': 0.6, 'direction': 'lower'}
        score = score_metric_v5(3.0, t)
        assert 0 < score < 1.0


class TestNewL3Metrics:
    def test_cvar_normal_distribution(self):
        from backtest.north_star_metrics import compute_cvar
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
        cvar = compute_cvar(returns, alpha=0.05)
        assert 0.02 < cvar < 0.06

    def test_cvar_all_positive(self):
        from backtest.north_star_metrics import compute_cvar
        returns = pd.Series([0.01, 0.02, 0.015, 0.005, 0.03] * 100)
        cvar = compute_cvar(returns, alpha=0.05)
        assert cvar < 0.01

    def test_max_dd_duration_known_sequence(self):
        from backtest.north_star_metrics import compute_max_dd_duration
        cum_ret = pd.Series([1.0, 1.1, 1.2, 1.15, 1.10, 1.05, 1.08, 1.12, 1.25])
        duration = compute_max_dd_duration(cum_ret)
        assert duration >= 4

    def test_max_dd_duration_no_drawdown(self):
        from backtest.north_star_metrics import compute_max_dd_duration
        cum_ret = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4])
        assert compute_max_dd_duration(cum_ret) == 0

    def test_underwater_ratio_always_up(self):
        from backtest.north_star_metrics import compute_underwater_ratio
        cum_ret = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4])
        assert compute_underwater_ratio(cum_ret) == 0.0

    def test_underwater_ratio_always_down(self):
        from backtest.north_star_metrics import compute_underwater_ratio
        cum_ret = pd.Series([1.0, 0.9, 0.8, 0.7, 0.6])
        assert compute_underwater_ratio(cum_ret) == 0.8


class TestNewL1Metrics:
    def test_ic_autocorrelation_persistent_signal(self):
        from backtest.north_star_metrics import compute_ic_autocorrelation
        np.random.seed(42)
        ic = [0.05]
        for _ in range(199):
            ic.append(0.7 * ic[-1] + 0.3 * np.random.normal(0.05, 0.02))
        assert compute_ic_autocorrelation(pd.Series(ic), lag=1) > 0.4

    def test_ic_autocorrelation_random_signal(self):
        from backtest.north_star_metrics import compute_ic_autocorrelation
        np.random.seed(42)
        assert abs(compute_ic_autocorrelation(pd.Series(np.random.normal(0, 0.05, 200)), lag=1)) < 0.2

    def test_transfer_coefficient_perfect(self):
        from backtest.north_star_metrics import compute_transfer_coefficient
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert compute_transfer_coefficient(s, s) > 0.99

    def test_transfer_coefficient_partial(self):
        from backtest.north_star_metrics import compute_transfer_coefficient
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        a = pd.Series([2, 1, 4, 3, 6, 5, 8, 7, 10, 9])
        tc = compute_transfer_coefficient(s, a)
        assert 0.5 < tc < 1.0


class TestNewL4Metrics:
    def test_wfer_basic(self):
        from backtest.north_star_metrics import compute_wfer
        assert compute_wfer({'is_sharpe': [2.0, 2.0, 2.0], 'oos_sharpe': [1.0, 1.0, 1.0]}) == pytest.approx(0.5, abs=0.01)

    def test_wfer_perfect(self):
        from backtest.north_star_metrics import compute_wfer
        assert compute_wfer({'is_sharpe': [1.5, 1.8, 2.0], 'oos_sharpe': [1.5, 1.8, 2.0]}) == pytest.approx(1.0, abs=0.01)

    def test_wfer_negative_is(self):
        from backtest.north_star_metrics import compute_wfer
        assert compute_wfer({'is_sharpe': [-0.5, 0.2, 0.1], 'oos_sharpe': [0.5, 0.3, 0.2]}) is None

    def test_oos_ic_half_life_no_decay(self):
        from backtest.north_star_metrics import compute_oos_ic_half_life
        assert compute_oos_ic_half_life({'oos_monthly_ics': [[0.05]*4, [0.06]*4]}) == 12.0

    def test_oos_ic_half_life_fast_decay(self):
        from backtest.north_star_metrics import compute_oos_ic_half_life
        hl = compute_oos_ic_half_life({'oos_monthly_ics': [[0.08, 0.04, 0.02, 0.01], [0.10, 0.05, 0.03, 0.01]]})
        assert 0 < hl < 3.0

    def test_oos_ic_half_life_no_data(self):
        from backtest.north_star_metrics import compute_oos_ic_half_life
        assert compute_oos_ic_half_life({}) is None


class TestFactorAttribution:
    def test_pure_market_portfolio(self):
        from backtest.north_star_metrics import compute_factor_attribution
        np.random.seed(42)
        n = 500
        mkt = np.random.normal(0.0005, 0.015, n)
        factor_df = pd.DataFrame({
            'MKT': mkt, 'SMB': np.random.normal(0, 0.008, n),
            'HML': np.random.normal(0, 0.008, n), 'UMD': np.random.normal(0, 0.008, n)
        })
        portfolio = pd.Series(mkt + np.random.normal(0, 0.001, n))
        result = compute_factor_attribution(portfolio, factor_df)
        assert abs(result['betas']['mkt'] - 1.0) < 0.15
        assert result['factor_r_squared'] > 0.8

    def test_pure_alpha_portfolio(self):
        from backtest.north_star_metrics import compute_factor_attribution
        np.random.seed(42)
        n = 500
        factor_df = pd.DataFrame({
            'MKT': np.random.normal(0, 0.015, n), 'SMB': np.random.normal(0, 0.008, n),
            'HML': np.random.normal(0, 0.008, n), 'UMD': np.random.normal(0, 0.008, n),
        })
        portfolio = pd.Series(np.random.normal(0.002, 0.01, n))
        result = compute_factor_attribution(portfolio, factor_df)
        assert result['factor_r_squared'] < 0.3
        assert result['residual_alpha_t'] > 2.0

    def test_small_cap_tilted(self):
        from backtest.north_star_metrics import compute_factor_attribution
        np.random.seed(42)
        n = 500
        smb = np.random.normal(0, 0.008, n)
        factor_df = pd.DataFrame({
            'MKT': np.random.normal(0.0005, 0.015, n), 'SMB': smb,
            'HML': np.random.normal(0, 0.008, n), 'UMD': np.random.normal(0, 0.008, n),
        })
        portfolio = pd.Series(1.5 * smb + np.random.normal(0.001, 0.005, n))
        result = compute_factor_attribution(portfolio, factor_df)
        assert result['smb_beta'] > 1.0

    def test_result_keys(self):
        from backtest.north_star_metrics import compute_factor_attribution
        np.random.seed(42)
        n = 100
        factor_df = pd.DataFrame({
            'MKT': np.random.normal(0, 0.01, n), 'SMB': np.random.normal(0, 0.01, n),
            'HML': np.random.normal(0, 0.01, n), 'UMD': np.random.normal(0, 0.01, n),
        })
        portfolio = pd.Series(np.random.normal(0.001, 0.01, n))
        result = compute_factor_attribution(portfolio, factor_df)
        for key in ['residual_alpha', 'residual_alpha_annual', 'residual_alpha_t',
                     'factor_r_squared', 'betas', 'max_factor_loading', 'smb_beta', 'mom_beta']:
            assert key in result
