"""Unit tests for ng1.2.2 return-weighted quintile CE classification."""
import numpy as np
import pytest

from ml_models.ng.ng_quintile_ce import (
    N_CLASSES,
    RETURN_CAP,
    build_quintile_labels,
    build_return_weights,
    make_quintile_dataset,
    strong_buy_prob,
)


def test_balanced_group_produces_equal_frequency():
    y = np.arange(100, dtype=np.float64)
    dates = np.array(['D1'] * 100)
    classes = build_quintile_labels(y, dates)
    # 100 rows / 5 classes = 20 per class
    counts = np.bincount(classes, minlength=N_CLASSES)
    assert counts.tolist() == [20, 20, 20, 20, 20]
    # Monotonic: lowest y → class 0, highest y → class 4
    assert classes[0] == 0
    assert classes[-1] == N_CLASSES - 1


def test_multiple_dates_binned_independently():
    # Date 1: values [0..9], Date 2: values [100..109] (disjoint ranges)
    y = np.concatenate([np.arange(10), np.arange(100, 110)]).astype(np.float64)
    dates = np.array(['D1'] * 10 + ['D2'] * 10)
    classes = build_quintile_labels(y, dates)
    # Each date should produce 2 per class (10/5)
    for d in ('D1', 'D2'):
        mask = dates == d
        counts = np.bincount(classes[mask], minlength=N_CLASSES)
        assert counts.tolist() == [2, 2, 2, 2, 2], f"date {d} not balanced: {counts}"


def test_small_group_marked_invalid():
    y = np.array([1.0, 2.0, 3.0, 4.0])  # only 4 < N_CLASSES=5
    dates = np.array(['D1'] * 4)
    classes = build_quintile_labels(y, dates)
    assert np.all(classes == -1), "undersized groups should be all -1"


def test_nan_kept_as_sentinel():
    y = np.array([1.0, np.nan, 2.0, 3.0, 4.0, 5.0])
    dates = np.array(['D1'] * 6)
    classes = build_quintile_labels(y, dates)
    # NaN row preserves -1; 5 valid rows get quintiled
    nan_idx = 1
    assert classes[nan_idx] == -1
    valid_mask = np.arange(6) != nan_idx
    assert np.all(classes[valid_mask] >= 0)
    # 5 valid → 1 per class
    counts = np.bincount(classes[valid_mask], minlength=N_CLASSES)
    assert counts.tolist() == [1, 1, 1, 1, 1]


def test_tie_degradation_to_ranks():
    # All y equal → edges collapse, degrade to rank-based
    y = np.array([0.5] * 10)
    dates = np.array(['D1'] * 10)
    classes = build_quintile_labels(y, dates)
    # Rank-based: floor(rank*5/10) per row → 2 per class
    counts = np.bincount(classes, minlength=N_CLASSES)
    assert counts.tolist() == [2, 2, 2, 2, 2]


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        build_quintile_labels(np.arange(5), np.array(['D1'] * 6))


def test_output_order_preserved():
    # Classes should be placed back in input order, not sorted order
    rng = np.random.default_rng(0)
    n = 50
    y = rng.normal(size=n)
    # Shuffle the date assignment to ensure we test the scatter-back step
    dates = rng.choice(['A', 'B', 'C'], size=n)
    classes = build_quintile_labels(y, dates)
    # Undersized groups (possible if one date got <5) → -1; others ∈ {0..4}
    for d in np.unique(dates):
        mask = dates == d
        grp = classes[mask]
        if mask.sum() >= N_CLASSES:
            assert np.all((grp >= 0) & (grp < N_CLASSES))


def test_return_weights_clip_and_abs():
    y = np.array([-0.5, -0.1, 0.0, 0.1, 0.5])
    w = build_return_weights(y, cap=0.25)
    assert w[0] == 0.25  # clipped from -0.5
    assert w[1] == 0.1
    assert w[2] == 0.0
    assert w[3] == 0.1
    assert w[4] == 0.25  # clipped from 0.5


def test_return_weights_nan_becomes_zero():
    y = np.array([0.05, np.nan, -0.03])
    w = build_return_weights(y)
    assert w[0] == 0.05
    assert w[1] == 0.0
    assert w[2] == 0.03


def test_return_weights_invalid_cap():
    with pytest.raises(ValueError):
        build_return_weights(np.array([0.1]), cap=0)
    with pytest.raises(ValueError):
        build_return_weights(np.array([0.1]), cap=-0.05)


def test_strong_buy_prob_returns_last_class():
    proba = np.array([
        [0.1, 0.2, 0.3, 0.2, 0.2],
        [0.5, 0.2, 0.1, 0.1, 0.1],
        [0.05, 0.05, 0.1, 0.2, 0.6],
    ])
    sb = strong_buy_prob(proba)
    assert sb.tolist() == [0.2, 0.1, 0.6]


def test_strong_buy_prob_shape_validation():
    with pytest.raises(ValueError, match="must be"):
        strong_buy_prob(np.array([0.1, 0.2, 0.3, 0.2, 0.2]))  # 1D
    with pytest.raises(ValueError, match="must be"):
        strong_buy_prob(np.zeros((5, 3)))  # wrong n_classes


def test_make_quintile_dataset_alignment():
    rng = np.random.default_rng(42)
    n = 100
    y = rng.normal(0, 0.05, size=n)
    # 5 dates with 20 stocks each
    dates = np.repeat(['D1', 'D2', 'D3', 'D4', 'D5'], 20)
    classes, weights, valid = make_quintile_dataset(y, dates)
    assert len(classes) == n
    assert len(weights) == n
    assert len(valid) == n
    # All 20-per-date groups are valid (>=5)
    assert valid.sum() == n
    # Weights: non-negative, bounded by cap
    assert np.all(weights >= 0)
    assert np.all(weights <= RETURN_CAP)


def test_make_quintile_dataset_drops_small_groups():
    # Date with 3 rows (too small) + date with 10 rows
    y = np.concatenate([np.arange(3, dtype=np.float64), np.arange(10, dtype=np.float64)])
    dates = np.array(['D1'] * 3 + ['D2'] * 10)
    classes, weights, valid = make_quintile_dataset(y, dates)
    assert valid.sum() == 10  # only D2 rows valid
    assert np.all(classes[:3] == -1)
    assert np.all(classes[3:] >= 0)


def test_quintile_wrapper_end_to_end():
    """Train a tiny multiclass LGB, wrap it, round-trip pickle, verify predict."""
    import pickle
    import lightgbm as lgb
    from ml_models.ng.ng_quintile_ce import QuintileStrongBuyModel

    rng = np.random.default_rng(0)
    n, p = 600, 8
    X = rng.normal(size=(n, p))
    # Synthetic signal so the model learns something non-trivial
    score = X[:, 0] - 0.5 * X[:, 1]
    y_cls = np.digitize(score, np.quantile(score, [0.2, 0.4, 0.6, 0.8])).astype(np.int32)
    w = rng.uniform(0.0, 0.25, size=n)

    dtrain = lgb.Dataset(X, label=y_cls, weight=w)
    booster = lgb.train(
        {'objective': 'multiclass', 'num_class': N_CLASSES, 'verbose': -1, 'num_leaves': 15},
        dtrain, num_boost_round=20,
    )
    wrapper = QuintileStrongBuyModel(booster)

    out = wrapper.predict(X)
    assert out.shape == (n,)
    assert np.all(out >= 0) and np.all(out <= 1)

    restored = pickle.loads(pickle.dumps(wrapper))
    out2 = restored.predict(X)
    assert np.array_equal(out, out2)


def test_classes_dtype_and_range():
    y = np.arange(25, dtype=np.float64)
    dates = np.array(['D1'] * 25)
    classes = build_quintile_labels(y, dates)
    assert classes.dtype == np.int8
    assert classes.min() >= -1
    assert classes.max() <= N_CLASSES - 1


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
