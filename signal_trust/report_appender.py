"""为日报 JSON 追加 trust_* 字段."""
import json
import logging
import sqlite3
from pathlib import Path

from .db import connect
from .constants import TAG_NO_DATA

logger = logging.getLogger(__name__)


def _query_scores(db_path: str, codes: list[str]) -> dict[str, dict]:
    if not codes:
        return {}
    try:
        conn = connect(db_path)
    except sqlite3.OperationalError as e:
        logger.warning(f"连接 DB 失败: {e}")
        return {}
    try:
        placeholders = ",".join("?" * len(codes))
        try:
            rows = conn.execute(
                f"SELECT * FROM signal_trust_scores WHERE code IN ({placeholders})",
                codes,
            ).fetchall()
        except sqlite3.OperationalError as e:
            # scores 表不存在 → 优雅降级
            logger.warning(f"signal_trust_scores 表不可用: {e}")
            return {}
        return {r["code"]: dict(r) for r in rows}
    finally:
        conn.close()


def append_trust_tags(report_json_path: str, db_path: str, top_n: int = 50) -> int:
    """
    为日报 JSON 的 Top-N 股票追加 trust_tag/trust_samples/trust_details.
    原子写 (临时文件 + rename).
    返回被追加的股票数.
    """
    p = Path(report_json_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    stocks = data.get("all_stocks_with_scores", [])
    # 按 rank_score 倒序取 Top-N
    ranked = sorted(
        stocks,
        key=lambda s: float(s.get("rank_score", 0) or 0),
        reverse=True,
    )[:top_n]
    codes = [s["stock_code"] for s in ranked if "stock_code" in s]
    scores = _query_scores(db_path, codes)

    for s in ranked:
        code = s.get("stock_code")
        if not code:
            continue
        sc = scores.get(code)
        if sc is None:
            s["trust_tag"] = TAG_NO_DATA
            s["trust_samples"] = 0
            s["trust_details"] = None
        else:
            s["trust_tag"] = sc["trust_tag"]
            s["trust_samples"] = sc["n_samples"]
            s["trust_details"] = {
                "direction_hit_rate": sc["direction_hit_rate"],
                "systematic_bias": sc["systematic_bias"],
                "high_pred_realize_rate": sc["high_pred_realize_rate"],
                "as_of_date": sc["as_of_date"],
            }

    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(p)
    return len(ranked)
