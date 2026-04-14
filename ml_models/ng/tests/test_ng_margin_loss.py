"""Unit tests for ng1.2.0 margin ranking loss."""
import numpy as np
import pytest

from ml_models.ng.ng_margin_loss import make_margin_objective, make_margin_eval_metric


class _FakeData:
    """Minimal stand-in for LightGBM Dataset used only by the callables."""
    def __init__(self, label, group):
        self._label = np.asarray(label, dtype=np.float64)
        self._group = np.asarray(group, dtype=np.int64) if group is not None else None

    def get_label(self):
        return self._label

    def get_group(self):
        return self._group


def test_perfect_ranking_zero_gradient():
    # y_true = [3, 2, 1], preds perfectly ranked with wide gap > margin
    y = np.array([3.0, 2.0, 1.0])
    p = np.array([1.0, 0.5, 0.0])  # all gaps 0.5 > 0.05 margin
    obj = make_margin_objective(margin=0.05)
    grad, hess = obj(p, _FakeData(y, [3]))
    assert np.allclose(grad, 0), f"expected zero grad, got {grad}"
    # Hess clamped at 1.0 when no active pairs
    assert np.all(hess >= 1.0)


def test_inverted_ranking_pushes_correctly():
    # y_true = [1, 2, 3] but preds inverted [3, 2, 1]: all pairs active
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([3.0, 2.0, 1.0])
    obj = make_margin_objective(margin=0.05)
    grad, hess = obj(p, _FakeData(y, [3]))
    # Sample 0 (y=1): appears as worse in pairs (1,0) and (2,0) → grad += 2
    # Sample 2 (y=3): appears as better in pairs (2,0) and (2,1) → grad -= 2
    # Sample 1 (y=2): in one pair as worse (2,1), one as better (1,0) → 0
    assert grad[0] == 2
    assert grad[1] == 0
    assert grad[2] == -2


def test_margin_boundary():
    # Pair (0,1): y_0 > y_1, p_diff = margin exactly
    # Active only if p_diff < margin (strict), so should be INACTIVE
    y = np.array([1.0, 0.0])
    p = np.array([0.05, 0.0])
    obj = make_margin_objective(margin=0.05)
    grad, _ = obj(p, _FakeData(y, [2]))
    assert np.allclose(grad, 0)


def test_single_group_vs_two_groups():
    # Same data, but grouping should isolate pairs within each group
    y = np.array([3.0, 1.0, 3.0, 1.0])
    p = np.array([0.0, 1.0, 0.0, 1.0])  # all 4 inverted
    obj = make_margin_objective(margin=0.05)

    grad1, _ = obj(p, _FakeData(y, [4]))
    # Within one group of 4: pairs (0,1),(0,3),(2,1),(2,3) active → each sample touched twice
    # Sample 0 (y=3, best): loses 2 pairs as "better" → grad = -2
    # Sample 1 (y=1, worst): in 2 pairs as "worse" → grad = +2
    assert grad1[0] == -2 and grad1[2] == -2
    assert grad1[1] == 2 and grad1[3] == 2

    grad2, _ = obj(p, _FakeData(y, [2, 2]))
    # Two groups of 2: pair (0,1) in group1, pair (2,3) in group2
    # Each sample touched once
    assert grad2[0] == -1 and grad2[2] == -1
    assert grad2[1] == 1 and grad2[3] == 1


def test_empty_group():
    grad, hess = make_margin_objective()(np.array([]), _FakeData([], []))
    assert len(grad) == 0 and len(hess) == 0


def test_size_1_group_skipped():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.0, 3.0])
    grad, hess = make_margin_objective()(p, _FakeData(y, [1, 1, 1]))
    assert np.allclose(grad, 0)
    assert np.all(hess == 1.0)


def test_hessian_positivity():
    np.random.seed(42)
    y = np.random.rand(100)
    p = np.random.rand(100)
    _, hess = make_margin_objective(margin=0.05)(p, _FakeData(y, [100]))
    assert np.all(hess >= 1.0), "hessian must stay positive for LightGBM"


def test_eval_metric_zero_for_perfect_rank():
    y = np.array([3.0, 2.0, 1.0])
    p = np.array([10.0, 5.0, 0.0])  # huge gaps
    name, val, higher_better = make_margin_eval_metric(0.05)(p, _FakeData(y, [3]))
    assert name == 'margin_loss'
    assert val == 0.0
    assert higher_better is False


def test_eval_metric_positive_for_inverted():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([3.0, 2.0, 1.0])
    _, val, _ = make_margin_eval_metric(0.05)(p, _FakeData(y, [3]))
    assert val > 0


def test_invalid_margin_rejected():
    with pytest.raises(ValueError):
        make_margin_objective(margin=0)
    with pytest.raises(ValueError):
        make_margin_objective(margin=-0.1)


def test_grad_symmetry_property():
    """In zero-mean group, sum of gradients should be zero (pairwise cancellation)."""
    np.random.seed(7)
    y = np.random.rand(50)
    p = np.random.rand(50)
    grad, _ = make_margin_objective(margin=0.05)(p, _FakeData(y, [50]))
    assert abs(grad.sum()) < 1e-10


def test_objective_is_picklable():
    """joblib.dump of a trained model needs the objective to survive pickle."""
    import pickle
    obj = make_margin_objective(margin=0.07)
    eval_fn = make_margin_eval_metric(margin=0.07)
    # Round-trip and verify the restored callable still works
    restored_obj = pickle.loads(pickle.dumps(obj))
    restored_eval = pickle.loads(pickle.dumps(eval_fn))
    y = np.array([3.0, 1.0, 2.0])
    p = np.array([0.0, 1.0, 0.5])
    g1, h1 = obj(p, _FakeData(y, [3]))
    g2, h2 = restored_obj(p, _FakeData(y, [3]))
    assert np.array_equal(g1, g2)
    assert np.array_equal(h1, h2)
    _, v1, _ = eval_fn(p, _FakeData(y, [3]))
    _, v2, _ = restored_eval(p, _FakeData(y, [3]))
    assert v1 == v2


def test_eval_metric_margin_validation():
    with pytest.raises(ValueError):
        make_margin_eval_metric(margin=0)
    with pytest.raises(ValueError):
        make_margin_eval_metric(margin=-0.1)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
