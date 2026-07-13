"""Heuristic chart-type picker.

Rules evaluated in order; first match wins. Returns None when nothing fits.
Spec: docs/superpowers/specs/2026-04-19-webapp-data-explorer-design.md §5.4
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd
from pandas.api.types import is_numeric_dtype


_CATEGORICAL_NAMES = {"code", "industry", "name", "tag", "trust_tag"}
_DATE_NAMES = {"trade_date", "date", "ann_date", "end_date", "as_of_date"}


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if is_numeric_dtype(df[c])]


def _categorical_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c in _CATEGORICAL_NAMES]


def _date_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if c in _DATE_NAMES:
            return c
    return None


def suggest(df: pd.DataFrame) -> Optional[dict]:
    if df.empty:
        return None

    nums = _numeric_cols(df)
    cats = _categorical_cols(df)
    date_col = _date_col(df)

    # R1 + R2: trade_date + numeric
    if date_col and nums:
        # R2: multiple distinct codes → ambiguous (no small-multiples in v1)
        if "code" in df.columns and df["code"].nunique() > 1:
            return None
        # R1: single (or no) code → line
        return {"type": "line", "x": date_col, "y": nums[0]}

    # R3: exactly 2 numerics (no date, no category)
    if len(nums) == 2 and not cats and not date_col:
        r = float(df[nums[0]].corr(df[nums[1]]))
        # 常量列 / 全 NaN / 不足 2 个有效点 → corr 为 NaN, 直接 round 会得到 NaN,
        # 经 jsonify(allow_nan) 序列化成非法 JSON token 'NaN', 前端 resp.json() 抛错整个响应失败
        pearson = None if math.isnan(r) else round(r, 4)
        return {
            "type": "scatter",
            "x": nums[0],
            "y": nums[1],
            "annotations": {"pearson_r": pearson},
        }

    # R4 / R5: 1 categorical + 1 numeric
    if len(cats) == 1 and len(nums) == 1:
        if len(df) <= 50:
            return {"type": "bar", "x": cats[0], "y": nums[0]}
        return {"type": "histogram", "x": nums[0]}

    # R6: single numeric only
    if len(nums) == 1 and not cats and not date_col:
        return {"type": "histogram", "x": nums[0]}

    return None
