"""Scan sqlite_master and classify tables into category buckets.

In-process TTL cache (5 min) keyed by db path. Invalidated via refresh=True.
"""
from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Optional


_CACHE: dict[str, tuple[float, dict]] = {}
_TTL_S = 300

_RAW = {
    "daily_quotes", "daily_basic", "financial_indicator",
    "financial_indicator_backup", "financial_statement_raw",
    "index_daily", "market_indices", "sw_index_daily",
    "daily_basic_temp",
}
_TECHNICAL = {"technical_indicators", "technical_overview"}
_MARKET_STATE = {
    "market_amv",
    "signal_trust_scores", "signal_trust_scores_history", "signal_trust_samples",
}
_MONEYFLOW = {"moneyflow_daily", "hsgt_daily"}
_META = {
    "securities", "stock_basic_info", "sw_industry",
    "schema_version", "data_update_log", "latest_quotes",
}
_BACKTEST = {"backtest_trades", "backtest_results", "stock_signals"}
# names that contain the word "factor" but are NOT feature_cache-shaped
_FACTOR_EXTRA = {"factor_daily_returns"}


def classify_table(name: str) -> str:
    if name in _RAW:
        return "raw"
    if name in _TECHNICAL:
        return "technical"
    if name in _MARKET_STATE:
        return "market_state"
    if name in _MONEYFLOW:
        return "moneyflow"
    if name in _META:
        return "meta"
    if name in _BACKTEST:
        return "backtest"
    if name in _FACTOR_EXTRA:
        return "factor"
    # pattern-based
    if re.search(r"_factor(_cache)?$", name):
        return "factor"
    if name.endswith("_feature_cache") or name in {
        "worldquant_factors", "latest_worldquant_factors",
        "active_mv_feature_cache", "neural_embedding_cache",
        "brain_alpha_cache", "alpha158_feature_cache",
        "ng_feature_cache",  # legacy ng v1.0.0
    }:
        return "feature_cache"
    return "other"


def discover(db_path: str | Path, refresh: bool = False) -> dict[str, list[dict]]:
    key = str(db_path)
    now = time.time()
    if not refresh and key in _CACHE:
        ts, val = _CACHE[key]
        if now - ts < _TTL_S:
            return val

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0)
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        tables = [
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        result: dict[str, list[dict]] = {}
        for t in tables:
            cols = [
                {"name": c[1], "type": c[2]}
                for c in conn.execute(f'PRAGMA table_info("{t}")')
            ]
            col_names = {c["name"] for c in cols}
            has_json = "features_json" in col_names

            # row_count — may be slow on 10M-row tables but acceptable at 5min TTL
            row_count = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]

            # date_range — only if a known date column exists
            date_col = next(
                (d for d in ("trade_date", "end_date", "ann_date", "as_of_date")
                 if d in col_names),
                None,
            )
            date_range: Optional[tuple[str, str]] = None
            if date_col and row_count > 0:
                row = conn.execute(
                    f'SELECT MIN("{date_col}"), MAX("{date_col}") FROM "{t}"'
                ).fetchone()
                if row and row[0] and row[1]:
                    date_range = (str(row[0]), str(row[1]))

            category = classify_table(t)
            result.setdefault(category, []).append(
                {
                    "table": t,
                    "columns": cols,
                    "row_count": row_count,
                    "date_range": date_range,
                    "has_features_json": has_json,
                }
            )
    finally:
        conn.close()

    _CACHE[key] = (now, result)
    return result
