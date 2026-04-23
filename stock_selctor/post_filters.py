"""Post-scoring hard filters applied after ML ranking, before final Top-N display.

Two filters:
1. exclude_unreliable_by_trust: drop 🔴 stocks using signal_trust_scores
2. cap_industry_concentration: limit per-industry count in the ranked list

Plus one annotator (doesn't change the list, adds a tag for T+1 execution timing):
3. annotate_horizon_alignment: label 🟢 ALIGN / 🟡 MIXED / 🔴 DIVERGE based on
   whether the 3d/5d/10d/15d predictions agree on direction.

All pure functions; no side effects on caller state beyond in-place dict updates
for the annotator.
"""
from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from typing import Iterable

logger = logging.getLogger(__name__)

RED_TAG_MARKER = "高风险"
UNKNOWN_INDUSTRY = {"未知", "Unknown", "unknown", "", None}


def _get_code(stock: dict) -> str:
    return stock.get("stock_code") or stock.get("code") or ""


def _get_rank(stock: dict) -> float:
    v = stock.get("rank_score")
    if v is None:
        v = stock.get("composite")
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def _is_a_share(stock: dict) -> bool:
    stype = stock.get("stock_type") or stock.get("type") or ""
    # Treat blank as A-share for safety (existing filter *ST logic also does)
    return stype in ("", "A股") or stype.startswith("A")


def _load_red_codes(db_path: str, codes: Iterable[str]) -> set[str]:
    codes = [c for c in codes if c]
    if not codes:
        return set()
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout = 30000")
    except sqlite3.OperationalError as e:
        logger.warning(f"signal_trust 读取失败, 跳过过滤: {e}")
        return set()
    try:
        # Chunk for SQLite parameter limit (999)
        red = set()
        for i in range(0, len(codes), 900):
            chunk = codes[i : i + 900]
            placeholders = ",".join("?" * len(chunk))
            try:
                rows = conn.execute(
                    f"SELECT code, trust_tag FROM signal_trust_scores "
                    f"WHERE code IN ({placeholders})",
                    chunk,
                ).fetchall()
            except sqlite3.OperationalError as e:
                logger.warning(f"signal_trust_scores 查询失败: {e}")
                return set()
            for code, tag in rows:
                if tag and RED_TAG_MARKER in tag:
                    red.add(code)
        return red
    finally:
        conn.close()


def exclude_unreliable_by_trust(
    stocks: list[dict],
    db_path: str,
) -> tuple[list[dict], list[dict]]:
    """Remove A-share stocks tagged 🔴高风险 in signal_trust_scores.

    ETFs/indices are NOT filtered (no fundamental basis for trust scoring).
    Returns (kept, dropped). If DB unavailable, returns (stocks, []).
    """
    a_codes = [_get_code(s) for s in stocks if _is_a_share(s)]
    red = _load_red_codes(db_path, a_codes)
    if not red:
        return stocks, []

    kept, dropped = [], []
    for s in stocks:
        if _is_a_share(s) and _get_code(s) in red:
            dropped.append(s)
        else:
            kept.append(s)
    return kept, dropped


def cap_industry_concentration(
    stocks: list[dict],
    max_per_industry: int,
) -> tuple[list[dict], list[dict]]:
    """Limit the ranked list to at most N stocks per industry.

    Input is consumed in given order (caller must pre-sort by rank_score desc).
    "未知" industry (ETFs/indices) bypasses the cap.
    Returns (kept, dropped).
    """
    if max_per_industry <= 0:
        return stocks, []

    counts: dict[str, int] = defaultdict(int)
    kept, dropped = [], []
    for s in stocks:
        ind = s.get("industry") or "未知"
        if ind in UNKNOWN_INDUSTRY:
            kept.append(s)
            continue
        if counts[ind] < max_per_industry:
            counts[ind] += 1
            kept.append(s)
        else:
            dropped.append(s)
    return kept, dropped


def apply_post_filters(
    stocks: list[dict],
    db_path: str,
    enable_trust_filter: bool = True,
    industry_cap: int = 3,
) -> dict:
    """Run both filters and return a summary dict for logging.

    Caller should use `result['stocks']` as the new list. Input `stocks` must be
    pre-sorted descending by rank_score so industry cap preserves the highest-ranked
    stocks per industry.
    """
    summary = {
        "input_count": len(stocks),
        "trust_dropped": [],
        "industry_dropped": [],
        "stocks": stocks,
    }

    if enable_trust_filter:
        stocks, trust_dropped = exclude_unreliable_by_trust(stocks, db_path)
        summary["trust_dropped"] = trust_dropped

    if industry_cap > 0:
        stocks, ind_dropped = cap_industry_concentration(stocks, industry_cap)
        summary["industry_dropped"] = ind_dropped

    summary["stocks"] = stocks
    summary["output_count"] = len(stocks)
    return summary


EXEC_TAG_ALIGN = "🟢ALIGN"      # all 4 horizons positive → T+1 buy ok
EXEC_TAG_MIXED = "🟡MIXED"       # 10d/15d positive but 3d or 5d negative → wait for pullback
EXEC_TAG_DIVERGE = "🔴DIVERGE"   # 10d positive but 15d ≤ 0 → unstable signal, caution
EXEC_TAG_WEAK = "⚪WEAK"         # 10d ≤ 0 → model doesn't even see alpha; tag for audit
EXEC_TAG_NO_DATA = "⚪NO_DATA"


def _horizon_alignment(pred_3d: float, pred_5d: float, pred_10d: float, pred_15d: float) -> str:
    """Classify T+1 execution timing based on sign agreement across 4 horizons.

    Priority: WEAK → DIVERGE → ALIGN → MIXED. Logic reflects the idea that 10d
    is the primary alpha signal; 15d cross-validates persistence; 3d/5d are
    short-term execution-window indicators (not alpha gates).
    """
    if pred_10d <= 0:
        return EXEC_TAG_WEAK
    if pred_15d <= 0:
        return EXEC_TAG_DIVERGE
    if pred_3d > 0 and pred_5d > 0:
        return EXEC_TAG_ALIGN
    return EXEC_TAG_MIXED


def annotate_horizon_alignment(stocks: list[dict]) -> dict:
    """Attach an `exec_tag` field to every stock with 4-horizon predictions.

    Stocks missing any of pred_3d/5d/10d/15d get `⚪NO_DATA`. Returns a count
    summary for logging. Mutates the input dicts in place.
    """
    counts: dict[str, int] = defaultdict(int)
    for s in stocks:
        try:
            p3 = float(s.get("pred_3d", 0) or 0)
            p5 = float(s.get("pred_5d", 0) or 0)
            p10 = float(s.get("pred_10d", 0) or 0)
            p15 = float(s.get("pred_15d", 0) or 0)
        except (TypeError, ValueError):
            s["exec_tag"] = EXEC_TAG_NO_DATA
            counts[EXEC_TAG_NO_DATA] += 1
            continue
        # No model output at all → NO_DATA
        if p3 == 0 and p5 == 0 and p10 == 0 and p15 == 0:
            s["exec_tag"] = EXEC_TAG_NO_DATA
            counts[EXEC_TAG_NO_DATA] += 1
            continue
        tag = _horizon_alignment(p3, p5, p10, p15)
        s["exec_tag"] = tag
        counts[tag] += 1
    return dict(counts)


def format_drop_log(summary: dict, top_preview: int = 10) -> str:
    def _fmt(stocks: list[dict]) -> list[str]:
        return [
            f"    {_get_code(s)} {s.get('stock_name','?')[:8]:<10} "
            f"{s.get('industry','?')[:8]:<10} rs={_get_rank(s):.4f}"
            for s in stocks[:top_preview]
        ]

    lines = [
        f"post-filter: {summary['input_count']} → {summary['output_count']} "
        f"(trust 剔除 {len(summary['trust_dropped'])}, "
        f"行业限流 {len(summary['industry_dropped'])})"
    ]
    if summary["trust_dropped"]:
        lines.append("  🔴 Trust 剔除:")
        lines.extend(_fmt(summary["trust_dropped"]))
    if summary["industry_dropped"]:
        lines.append("  🏭 行业限流剔除:")
        lines.extend(_fmt(summary["industry_dropped"]))
    return "\n".join(lines)
