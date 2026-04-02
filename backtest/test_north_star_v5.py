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

    def test_zero_below_pass_gets_fractional_score(self):
        from backtest.north_star_metrics import score_metric_v5
        t = {'pass': 0.03, 'ok': 0.04, 'good': 0.05, 'great': 0.06, 'target': 0.08, 'direction': 'higher'}
        # value=0 is below pass=0.03 but within extrapolation range → fractional score (0 < s < 1)
        s = score_metric_v5(0.0, t)
        assert 0 < s < 1.0, f"0.0 below pass should get fractional score, got {s}"
        # NaN and None should return exactly 0.0
        assert score_metric_v5(None, t) == 0.0
        assert score_metric_v5(float('nan'), t) == 0.0

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


class TestV5Score:
    def test_perfect_scores(self):
        from backtest.north_star_metrics import compute_v5_score, NORTH_STAR_TARGETS_V5
        metric_values = {}
        for name, info in NORTH_STAR_TARGETS_V5.items():
            target = info['target']
            if info['direction'] == 'higher':
                # For positive targets: go 10% above. For negative targets (e.g. -0.08):
                # "better" means closer to 0, i.e. multiply by 0.9 (less negative).
                if target >= 0:
                    metric_values[name] = target * 1.1
                else:
                    metric_values[name] = target * 0.9
            else:
                # direction='lower': better means smaller value
                if target >= 0:
                    metric_values[name] = target * 0.9
                else:
                    metric_values[name] = target * 1.1
        result = compute_v5_score(metric_values, n_trading_days=600)
        assert result['final_pct'] >= 99.0
        assert result['grade'] == 'S'

    def test_all_zeros(self):
        from backtest.north_star_metrics import compute_v5_score
        result = compute_v5_score({}, n_trading_days=600)
        assert result['final_pct'] == 0.0
        assert result['grade'] == 'D'

    def test_length_penalty_v5(self):
        from backtest.north_star_metrics import compute_backtest_length_factor_v5
        assert compute_backtest_length_factor_v5(500) == 1.0
        assert compute_backtest_length_factor_v5(600) == 1.0
        factor_250 = compute_backtest_length_factor_v5(250)
        assert 0.6 < factor_250 < 0.7
        assert compute_backtest_length_factor_v5(60) == 0.0
        assert compute_backtest_length_factor_v5(30) == 0.0

    def test_six_layers(self):
        from backtest.north_star_metrics import compute_v5_score, NORTH_STAR_TARGETS_V5
        metric_values = {name: 0.01 for name in NORTH_STAR_TARGETS_V5}
        result = compute_v5_score(metric_values, n_trading_days=600)
        assert len(result['layer_details']) == 6
        for layer_id in [1, 2, 3, 4, 5, 6]:
            assert layer_id in result['layer_details']

    def test_continuous_scores_in_result(self):
        from backtest.north_star_metrics import compute_v5_score
        metric_values = {'daily_ic': 0.055}
        result = compute_v5_score(metric_values, n_trading_days=600)
        ic_score = result['metric_scores']['daily_ic']
        assert isinstance(ic_score[0], float)
        assert 3.0 < ic_score[0] < 4.0

    def test_auto_select_benchmark(self):
        from backtest.north_star_metrics import auto_select_benchmark
        assert auto_select_benchmark(100) == '000300.SH'   # 大盘
        assert auto_select_benchmark(60) == '000905.SH'    # 中盘 (was CSI300 with old threshold)
        assert auto_select_benchmark(20) == '000905.SH'    # 中盘
        assert auto_select_benchmark(8) == '000852.SH'     # 中小盘
        assert auto_select_benchmark(3) == '932000.CSI'    # 小盘

    def test_39_metrics(self):
        """V5应有恰好39个指标"""
        from backtest.north_star_metrics import NORTH_STAR_TARGETS_V5
        assert len(NORTH_STAR_TARGETS_V5) == 39
        layer_counts = {}
        for info in NORTH_STAR_TARGETS_V5.values():
            l = info['layer']
            layer_counts[l] = layer_counts.get(l, 0) + 1
        assert layer_counts == {1: 10, 2: 5, 3: 7, 4: 6, 5: 5, 6: 6}

    def test_max_score_195(self):
        from backtest.north_star_metrics import compute_v5_score, NORTH_STAR_TARGETS_V5
        metric_values = {name: 0.01 for name in NORTH_STAR_TARGETS_V5}
        result = compute_v5_score(metric_values, n_trading_days=600)
        assert result['max_score'] == 195.0


class TestV51L3Stability:
    def test_hurst_random_walk(self):
        from backtest.north_star_metrics import compute_hurst_exponent
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 1000))
        h = compute_hurst_exponent(returns)
        assert 0.35 < h < 0.65, f"Random walk H={h}, expected ~0.5"

    def test_hurst_trending(self):
        from backtest.north_star_metrics import compute_hurst_exponent
        np.random.seed(42)
        rw = np.cumsum(np.random.normal(0.001, 0.005, 1000))
        returns = pd.Series(np.diff(rw))
        h = compute_hurst_exponent(returns)
        assert h > 0.51, f"Trending H={h}, expected >0.51"

    def test_hurst_short_series(self):
        from backtest.north_star_metrics import compute_hurst_exponent
        returns = pd.Series(np.random.normal(0, 0.01, 50))
        assert compute_hurst_exponent(returns) == 0.5

    def test_regime_transition_dd_stable(self):
        from backtest.north_star_metrics import compute_regime_transition_dd
        np.random.seed(42)
        strategy_ret = pd.Series(np.random.normal(0.001, 0.01, 500))
        benchmark_ret = pd.Series(np.concatenate([
            np.random.normal(0.002, 0.01, 200),
            np.random.normal(-0.003, 0.015, 100),
            np.random.normal(0.001, 0.01, 200),
        ]))
        ratio = compute_regime_transition_dd(strategy_ret, benchmark_ret)
        assert ratio is not None
        assert ratio < 5.0

    def test_regime_transition_dd_short(self):
        from backtest.north_star_metrics import compute_regime_transition_dd
        ret = pd.Series(np.random.normal(0, 0.01, 50))
        bench = pd.Series(np.random.normal(0, 0.01, 50))
        assert compute_regime_transition_dd(ret, bench) is None


class TestV51L4Advanced:
    def test_cscv_pbo_random_strategy(self):
        from backtest.north_star_metrics import compute_cscv_pbo
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 500))
        pbo = compute_cscv_pbo(returns, n_subperiods=16, max_combinations=200)
        assert 0.2 < pbo < 0.8, f"Random PBO={pbo}"

    def test_cscv_pbo_strong_strategy(self):
        from backtest.north_star_metrics import compute_cscv_pbo
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.003, 0.01, 500))
        pbo = compute_cscv_pbo(returns, n_subperiods=16, max_combinations=200)
        assert pbo < 0.6

    def test_cscv_pbo_short_series(self):
        from backtest.north_star_metrics import compute_cscv_pbo
        returns = pd.Series(np.random.normal(0, 0.02, 100))
        assert compute_cscv_pbo(returns) is None

    def test_effective_n_uncorrelated(self):
        from backtest.north_star_metrics import compute_effective_n_corr
        np.random.seed(42)
        holdings = pd.DataFrame({f'stock_{i}': np.random.normal(0, 0.02, 100) for i in range(10)})
        assert compute_effective_n_corr(holdings) > 5.0

    def test_effective_n_highly_correlated(self):
        from backtest.north_star_metrics import compute_effective_n_corr
        np.random.seed(42)
        base = np.random.normal(0, 0.02, 100)
        holdings = pd.DataFrame({f'stock_{i}': base + np.random.normal(0, 0.003, 100) for i in range(10)})
        assert compute_effective_n_corr(holdings) < 3.0

    def test_effective_n_single_stock(self):
        from backtest.north_star_metrics import compute_effective_n_corr
        holdings = pd.DataFrame({'stock_0': np.random.normal(0, 0.02, 100)})
        assert compute_effective_n_corr(holdings) == 1.0


class TestV51L7Capacity:
    def test_capacity_high_liquidity(self):
        from backtest.north_star_metrics import compute_strategy_capacity
        picks = pd.DataFrame({
            'code': ['000001.SZ', '600519.SH', '000858.SZ'],
            'adv_20d_value': [5e8, 3e8, 2e8],
            'daily_vol': [0.02, 0.015, 0.025],
        })
        cap = compute_strategy_capacity(picks, gross_annual_return=0.30, avg_turnover=0.3)
        assert cap > 200, f"High liquidity capacity={cap}M"

    def test_capacity_low_liquidity(self):
        from backtest.north_star_metrics import compute_strategy_capacity
        picks = pd.DataFrame({
            'code': ['000001.SZ', '600519.SH', '000858.SZ'],
            'adv_20d_value': [5e6, 3e6, 2e6],
            'daily_vol': [0.03, 0.035, 0.04],
        })
        cap = compute_strategy_capacity(picks, gross_annual_return=0.30, avg_turnover=0.3)
        assert cap < 200, f"Low liquidity capacity={cap}M"

    def test_participation_rate(self):
        from backtest.north_star_metrics import compute_participation_rate_p90
        picks = pd.DataFrame({
            'code': ['A', 'B', 'C'],
            'adv_20d_value': [1e8, 5e7, 2e7],
        })
        p90 = compute_participation_rate_p90(picks, assumed_aum_mn=10, n_positions=3)
        assert 0 < p90 < 1.0

    def test_liquidity_adj_sharpe(self):
        from backtest.north_star_metrics import compute_liquidity_adj_sharpe
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.015, 252))
        raw_sharpe = returns.mean() / returns.std() * np.sqrt(252)
        la_sharpe = compute_liquidity_adj_sharpe(returns, impact_cost_annual=0.02)
        assert la_sharpe < raw_sharpe + 0.1
        assert la_sharpe > 0


class TestV51Score:
    def test_46_metrics(self):
        from backtest.north_star_metrics import NORTH_STAR_TARGETS_V51
        assert len(NORTH_STAR_TARGETS_V51) == 46
        layer_counts = {}
        for info in NORTH_STAR_TARGETS_V51.values():
            l = info['layer']
            layer_counts[l] = layer_counts.get(l, 0) + 1
        assert layer_counts == {1: 10, 2: 5, 3: 9, 4: 8, 5: 5, 6: 6, 7: 3}

    def test_max_score_230(self):
        from backtest.north_star_metrics import compute_v51_score, NORTH_STAR_TARGETS_V51
        metric_values = {name: 0.01 for name in NORTH_STAR_TARGETS_V51}
        result = compute_v51_score(metric_values, n_trading_days=600)
        assert result['max_score'] == 230.0

    def test_seven_layers(self):
        from backtest.north_star_metrics import compute_v51_score, NORTH_STAR_TARGETS_V51
        metric_values = {name: 0.01 for name in NORTH_STAR_TARGETS_V51}
        result = compute_v51_score(metric_values, n_trading_days=600)
        assert len(result['layer_details']) == 7

    def test_perfect_scores(self):
        from backtest.north_star_metrics import compute_v51_score, NORTH_STAR_TARGETS_V51
        metric_values = {}
        for name, info in NORTH_STAR_TARGETS_V51.items():
            t = info['target']
            if info['direction'] == 'higher':
                metric_values[name] = t * 1.1 if t >= 0 else t * 0.5
            else:
                metric_values[name] = t * 0.9 if t > 0 else t * 1.1
        result = compute_v51_score(metric_values, n_trading_days=600)
        assert result['final_pct'] >= 99.0
        assert result['grade'] == 'S'

    def test_weights_sum_to_100(self):
        from backtest.north_star_metrics import V51_LAYER_WEIGHTS
        assert abs(sum(V51_LAYER_WEIGHTS.values()) - 1.0) < 0.001
