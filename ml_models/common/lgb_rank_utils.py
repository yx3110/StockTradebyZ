"""Shared LightGBM ranking utilities across NG variants.

Owns:
  - RANK_BASE_PARAMS: the common 11-key hyperparameter dict that both
    lambdarank (V4.7.1 lineage) and margin_rank (ng1.2.0) use.
  - build_groups_per_date: run-length group sizes preserving input order.

Rationale: LightGBM's ranking API requires per-row groups to be contiguous in
row order, not sorted alphabetically. np.unique would silently reorder them.

The older `train_v395_multi_target.py:4940-4977` uses np.unique to build groups.
That code is correct under its precondition (data is pre-sorted by date before
entering the WF training loop), but is fragile: a non-sorted input would
silently produce group sizes in np.unique's sorted order, misaligned with the
actual row order. Prefer this helper for any new ranking training path.
"""
from typing import List, Optional, Sequence


RANK_BASE_PARAMS = {
    'learning_rate': 0.02,
    'num_leaves': 20,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'reg_alpha': 1.0,
    'reg_lambda': 5.0,
    'min_data_in_leaf': 500,
    'min_gain_to_split': 0.01,
    'path_smooth': 10.0,
    'verbose': -1,
}


def build_groups_per_date(dates: Optional[Sequence]) -> List[int]:
    """Return LightGBM group sizes, one per contiguous run of identical dates.

    Args:
        dates: sequence (list / ndarray) of trading-date values, already sorted
            in row order. Contiguous duplicates form one group.

    Returns:
        List of group sizes. Sum of elements equals len(dates).
    """
    if dates is None:
        return []
    groups: List[int] = []
    current = None
    count = 0
    for d in dates:
        if d != current:
            if count > 0:
                groups.append(count)
            current, count = d, 1
        else:
            count += 1
    if count > 0:
        groups.append(count)
    return groups
