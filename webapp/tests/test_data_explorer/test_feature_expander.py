"""Tests for feature_expander.expand()."""
import pandas as pd
import pytest

from core.data_explorer.feature_expander import expand


def test_expands_features_json_into_columns(sample_df_with_json: pd.DataFrame) -> None:
    df_out, warnings = expand(sample_df_with_json)
    assert "features_json" not in df_out.columns
    assert "pb" in df_out.columns
    assert "roe" in df_out.columns
    assert df_out.loc[0, "pb"] == 8.1
    assert df_out.loc[1, "roe"] == 22.1
    assert warnings == []


def test_passthrough_when_no_features_json_column() -> None:
    df = pd.DataFrame({"code": ["A"], "close": [1.0]})
    df_out, warnings = expand(df)
    assert list(df_out.columns) == ["code", "close"]
    assert warnings == []


def test_malformed_json_returns_warning_and_preserves_column() -> None:
    df = pd.DataFrame(
        {"code": ["A"], "features_json": ['{"broken":']}
    )
    df_out, warnings = expand(df)
    assert "features_json" in df_out.columns  # preserved
    assert len(warnings) == 1
    assert "features_json expansion failed" in warnings[0]


def test_preserves_non_feature_columns_alongside_expanded() -> None:
    df = pd.DataFrame(
        {
            "code": ["A", "B"],
            "label_10d": [0.1, 0.2],
            "features_json": ['{"x": 1}', '{"x": 2}'],
        }
    )
    df_out, _ = expand(df)
    assert list(df_out.columns) == ["code", "label_10d", "x"]
    assert df_out["x"].tolist() == [1, 2]
