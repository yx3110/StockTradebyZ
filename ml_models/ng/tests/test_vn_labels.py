"""Unit tests for ng1.2.1 vol-normalized Sharpe-style labels."""
import numpy as np
import pytest

from fetch_data.label_utils import compute_vn_labels_from_future_prices


HORIZONS = (3, 5, 10, 15)


def _linear_price_path(base: float, daily_ret: float, n: int):
    """Return {day: close} for a constant-daily-return path over n days."""
    return {d: base * (1.0 + daily_ret) ** d for d in range(1, n + 1)}


def test_constant_up_path_zero_std_gets_floored():
    # Every daily return identical → std=0 → floor triggers → vn_label finite
    future = _linear_price_path(10.0, 0.01, 15)
    out = compute_vn_labels_from_future_prices(10.0, future, HORIZONS)
    assert np.isfinite(out['vn_label_3d'])
    assert out['path_std_10d'] < 1e-8
    # downside_std_10d: no negative days → 0.0
    assert out['downside_std_10d'] == 0.0


def test_missing_future_day_blocks_path():
    # Drop day 5 → path breaks at day 4, only vn_label_3d computable
    future = _linear_price_path(10.0, 0.01, 15)
    del future[5]
    out = compute_vn_labels_from_future_prices(10.0, future, HORIZONS)
    assert np.isfinite(out['vn_label_3d'])
    assert np.isnan(out['vn_label_5d'])
    assert np.isnan(out['vn_label_10d'])
    assert np.isnan(out['vn_label_15d'])
    # path stats require full 10-day window
    assert np.isnan(out['path_mean_10d'])
    assert np.isnan(out['path_std_10d'])


def test_all_negative_days_downside_equals_total_std():
    # Monotone decline → all daily rets negative → downside_std ≈ path_std
    future = _linear_price_path(10.0, -0.02, 15)
    out = compute_vn_labels_from_future_prices(10.0, future, HORIZONS)
    # Constant daily ret still → std=0 → floor kicks in
    assert out['path_std_10d'] < 1e-8
    assert out['downside_std_10d'] < 1e-8
    # Cumulative return is negative so vn_label is negative
    assert out['vn_label_10d'] < 0


def test_mixed_path_sharpe_sign():
    # First half up 5%, second half down 3% → cumret > 0 but volatile
    future = {}
    base = 10.0
    for d in range(1, 16):
        ret = 0.05 if d <= 5 else -0.03
        future[d] = (future.get(d - 1, base)) * (1.0 + ret)
    out = compute_vn_labels_from_future_prices(base, future, HORIZONS)
    # path_std should be meaningfully > 0 (mixed signs)
    assert out['path_std_10d'] > 1e-3
    # downside_std captures only negative days
    assert out['downside_std_10d'] > 0
    assert out['downside_std_10d'] < out['path_std_10d']


def test_invalid_base_open_returns_all_nan():
    out = compute_vn_labels_from_future_prices(0.0, {1: 10.0, 10: 11.0}, HORIZONS)
    for k, v in out.items():
        assert np.isnan(v), f"{k} expected NaN, got {v}"

    out2 = compute_vn_labels_from_future_prices(np.nan, {1: 10.0}, HORIZONS)
    for v in out2.values():
        assert np.isnan(v)


def test_output_keys_match_schema():
    future = _linear_price_path(10.0, 0.01, 15)
    out = compute_vn_labels_from_future_prices(10.0, future, HORIZONS)
    expected = {
        'vn_label_3d', 'vn_label_5d', 'vn_label_10d', 'vn_label_15d',
        'path_mean_10d', 'path_std_10d', 'downside_std_10d',
    }
    assert set(out.keys()) == expected


def test_sharpe_sign_matches_cumulative_return():
    # When daily ret > 0 → vn_label > 0; when < 0 → < 0
    up = _linear_price_path(10.0, 0.005, 15)
    out_up = compute_vn_labels_from_future_prices(10.0, up, HORIZONS)
    assert out_up['vn_label_10d'] > 0

    down = _linear_price_path(10.0, -0.005, 15)
    out_down = compute_vn_labels_from_future_prices(10.0, down, HORIZONS)
    assert out_down['vn_label_10d'] < 0


def test_realistic_stochastic_path():
    rng = np.random.default_rng(42)
    daily = rng.normal(0.001, 0.015, size=15)
    prices = {}
    p = 10.0
    for d, r in enumerate(daily, 1):
        p *= (1 + r)
        prices[d] = p
    out = compute_vn_labels_from_future_prices(10.0, prices, HORIZONS)
    # Finite everywhere
    for k, v in out.items():
        assert np.isfinite(v), f"{k} not finite: {v}"
    # Sanity: std bounds (daily sigma=0.015 over 10 days → std ~0.015)
    assert 0.005 < out['path_std_10d'] < 0.05


def test_min_sigma_floor_respected():
    # Path with tiny stdev + big positive cumret — ensure floor prevents huge vn_label
    future = _linear_price_path(10.0, 1e-5, 15)  # 0.001% daily
    out = compute_vn_labels_from_future_prices(10.0, future, HORIZONS, min_sigma=1e-3)
    # Even with big floor, 15d cumret is ~1.5e-4 → vn_label small, not huge
    assert abs(out['vn_label_15d']) < 1.0


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
