"""可信度聚合器."""
import logging
from .db import connect
from .constants import (
    MIN_SAMPLES, REALIZE_ACTUAL_THRESHOLD,
    GREEN_HIT_MIN, GREEN_BIAS_MIN, GREEN_REALIZE_MIN,
    RED_HIT_MAX, RED_BIAS_MAX, RED_REALIZE_MAX,
    TAG_GREEN, TAG_YELLOW, TAG_RED, TAG_NO_DATA,
)

logger = logging.getLogger(__name__)


def trust_tag(hit: float | None, bias: float | None, realize: float | None, n: int) -> str:
    if n < MIN_SAMPLES:
        return TAG_NO_DATA
    if hit is None or bias is None or realize is None:
        return TAG_NO_DATA
    if hit < RED_HIT_MAX or bias < RED_BIAS_MAX or realize < RED_REALIZE_MAX:
        return TAG_RED
    if hit >= GREEN_HIT_MIN and bias >= GREEN_BIAS_MIN and realize >= GREEN_REALIZE_MIN:
        return TAG_GREEN
    return TAG_YELLOW


def compute_scores(db_path: str, as_of_date: str, codes: list[str] | None = None) -> dict[str, dict]:
    """
    对 codes 列表(None=全库)每只股票聚合三指标 + 标签. 返回 {code: {...}}.
    泄露防护: WHERE sample_end_date < as_of_date AND actual_10d IS NOT NULL.
    """
    conn = connect(db_path)
    try:
        params: list = [as_of_date]
        where = "sample_end_date < ? AND actual_10d IS NOT NULL"
        if codes is not None:
            if not codes:
                return {}
            placeholders = ",".join("?" * len(codes))
            where += f" AND code IN ({placeholders})"
            params.extend(codes)
        # 一次查所有样本, Python 端聚合(样本量百万级也 OK)
        rows = conn.execute(
            f"SELECT code, pred_10d, actual_10d FROM signal_trust_samples WHERE {where}",
            params,
        ).fetchall()
    finally:
        conn.close()

    by_code: dict[str, list] = {}
    for r in rows:
        by_code.setdefault(r["code"], []).append((r["pred_10d"], r["actual_10d"]))

    result: dict[str, dict] = {}
    for code, pairs in by_code.items():
        n = len(pairs)
        if n == 0:
            continue
        hits = sum(1 for p, a in pairs if (p > 0) == (a > 0))
        biases = [a - p for p, a in pairs]
        realizes = sum(1 for _, a in pairs if a > REALIZE_ACTUAL_THRESHOLD)
        hit = hits / n
        bias = sum(biases) / n
        realize = realizes / n
        result[code] = {
            "code": code,
            "as_of_date": as_of_date,
            "n_samples": n,
            "direction_hit_rate": hit,
            "systematic_bias": bias,
            "high_pred_realize_rate": realize,
            "trust_tag": trust_tag(hit, bias, realize, n),
        }
    # 对 codes 中有指定但无样本的, 写一个 no-data 记录
    if codes is not None:
        for c in codes:
            if c not in result:
                result[c] = {
                    "code": c, "as_of_date": as_of_date, "n_samples": 0,
                    "direction_hit_rate": None, "systematic_bias": None,
                    "high_pred_realize_rate": None, "trust_tag": TAG_NO_DATA,
                }
    return result


def upsert_scores(db_path: str, scores: dict[str, dict]) -> int:
    if not scores:
        return 0
    conn = connect(db_path)
    try:
        for s in scores.values():
            conn.execute(
                "INSERT OR REPLACE INTO signal_trust_scores "
                "(code, as_of_date, n_samples, direction_hit_rate, systematic_bias, "
                " high_pred_realize_rate, trust_tag, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
                (s["code"], s["as_of_date"], s["n_samples"],
                 s["direction_hit_rate"], s["systematic_bias"],
                 s["high_pred_realize_rate"], s["trust_tag"]),
            )
        conn.commit()
        return len(scores)
    finally:
        conn.close()
