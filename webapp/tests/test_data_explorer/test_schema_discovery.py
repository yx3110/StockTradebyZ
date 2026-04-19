"""Tests for schema_discovery.discover() + classify_table()."""
from pathlib import Path

from core.data_explorer.schema_discovery import (
    classify_table,
    discover,
)


def test_classify_known_tables() -> None:
    assert classify_table("daily_quotes") == "raw"
    assert classify_table("daily_basic") == "raw"
    assert classify_table("financial_indicator") == "raw"
    assert classify_table("technical_indicators") == "technical"
    assert classify_table("ng101_feature_cache") == "feature_cache"
    assert classify_table("v39_feature_cache") == "feature_cache"
    assert classify_table("alpha158_feature_cache") == "feature_cache"
    assert classify_table("worldquant_factors") == "feature_cache"
    assert classify_table("v492_factor_cache") == "factor"
    assert classify_table("factor_daily_returns") == "factor"
    assert classify_table("market_amv") == "market_state"
    assert classify_table("signal_trust_scores") == "market_state"
    assert classify_table("moneyflow_daily") == "moneyflow"
    assert classify_table("hsgt_daily") == "moneyflow"
    assert classify_table("securities") == "meta"
    assert classify_table("sw_industry") == "meta"
    assert classify_table("backtest_trades") == "backtest"
    assert classify_table("stock_signals") == "backtest"


def test_classify_unknown_returns_other() -> None:
    assert classify_table("some_brand_new_table") == "other"


def test_discover_returns_categorized_dict(tmp_stock_db: Path) -> None:
    result = discover(tmp_stock_db)

    # conftest creates daily_quotes, ng101_feature_cache, securities
    categories = set(result.keys())
    assert "raw" in categories
    assert "feature_cache" in categories
    assert "meta" in categories

    raw_tables = [t["table"] for t in result["raw"]]
    assert "daily_quotes" in raw_tables

    ng = next(t for t in result["feature_cache"] if t["table"] == "ng101_feature_cache")
    assert ng["has_features_json"] is True
    assert "code" in [c["name"] for c in ng["columns"]]
    assert ng["row_count"] == 2
    assert ng["date_range"] == ("2026-04-18", "2026-04-18")


def test_discover_without_trade_date_handles_date_range_none(tmp_stock_db: Path) -> None:
    # securities has no trade_date; date_range must be None, not crash
    result = discover(tmp_stock_db)
    sec = next(t for t in result["meta"] if t["table"] == "securities")
    assert sec["date_range"] is None
    assert sec["has_features_json"] is False
