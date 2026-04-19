"""Expand features_json column in a DataFrame into flat scalar columns.

feature_cache tables store ~66 stock-level features as a single JSON blob per
row; users expect to SELECT / filter / plot individual features without
calling json_extract() everywhere, so we normalize on the Python side.
"""
from __future__ import annotations

import json
from typing import Tuple

import pandas as pd


def expand(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """Return (df_with_json_columns_flat, warnings).

    - If 'features_json' not in df, return df unchanged, no warnings.
    - On any JSON parse failure, return df unchanged with one warning string.
    - Idempotent (safe to call multiple times; no-op after first call).
    """
    if "features_json" not in df.columns:
        return df, []
    try:
        normalized = pd.json_normalize(df["features_json"].apply(json.loads))
    except (json.JSONDecodeError, TypeError) as e:
        return df, [f"features_json expansion failed: {e}"]
    out = pd.concat(
        [
            df.drop(columns=["features_json"]).reset_index(drop=True),
            normalized.reset_index(drop=True),
        ],
        axis=1,
    )
    return out, []
