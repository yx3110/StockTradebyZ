"""周度全局失效统计."""
import logging
from .db import connect
from .constants import (
    REALIZE_ACTUAL_THRESHOLD,
    GREEN_HIT_MIN, GREEN_BIAS_MIN, GREEN_REALIZE_MIN,
    RED_HIT_MAX, RED_BIAS_MAX, RED_REALIZE_MAX,
)

logger = logging.getLogger(__name__)

_ALLOWED_BUCKETS = {"market_cap_bucket", "industry", "liquidity_bucket"}


def aggregate_by_bucket(db_path: str, bucket_column: str, as_of_date: str) -> list[dict]:
    """按 bucket_column GROUP BY，返回每桶的样本数/三指标。

    Args:
        db_path: SQLite 数据库路径。
        bucket_column: 分组列名，必须是 market_cap_bucket / industry / liquidity_bucket 之一。
        as_of_date: 截止日期（ISO 格式），只取 sample_end_date < as_of_date 的已完成样本。

    Returns:
        list of dicts with keys: bucket, n_samples, direction_hit_rate,
        systematic_bias, high_pred_realize_rate.
    """
    if bucket_column not in _ALLOWED_BUCKETS:
        raise ValueError(f"bucket_column must be one of {_ALLOWED_BUCKETS}, got {bucket_column!r}")

    # bucket_column is validated against a whitelist — safe to interpolate
    sql = (
        f"SELECT {bucket_column} AS bucket, "
        f"  COUNT(*) AS n, "
        f"  AVG(CASE WHEN (pred_10d > 0) = (actual_10d > 0) THEN 1.0 ELSE 0.0 END) AS hit, "
        f"  AVG(actual_10d - pred_10d) AS bias, "
        f"  AVG(CASE WHEN actual_10d > ? THEN 1.0 ELSE 0.0 END) AS realize "
        f"FROM signal_trust_samples "
        f"WHERE sample_end_date < ? AND actual_10d IS NOT NULL "
        f"  AND {bucket_column} IS NOT NULL "
        f"GROUP BY {bucket_column} "
        f"ORDER BY n DESC"
    )
    conn = connect(db_path)
    try:
        rows = conn.execute(sql, (REALIZE_ACTUAL_THRESHOLD, as_of_date)).fetchall()
        return [
            {
                "bucket": r["bucket"],
                "n_samples": r["n"],
                "direction_hit_rate": r["hit"],
                "systematic_bias": r["bias"],
                "high_pred_realize_rate": r["realize"],
            }
            for r in rows
        ]
    finally:
        conn.close()


def _tag_symbol(hit, bias, realize) -> str:
    """返回单字符状态标识。"""
    if hit is None:
        return "⚪"
    if hit < RED_HIT_MAX or bias < RED_BIAS_MAX or realize < RED_REALIZE_MAX:
        return "⚠️"
    if hit >= GREEN_HIT_MIN and bias >= GREEN_BIAS_MIN and realize >= GREEN_REALIZE_MIN:
        return "✓"
    return "🟡"


def _section(title: str, agg: list[dict], order: list[str] | None = None) -> str:
    """生成 Markdown 分组表格（含标题行）。"""
    lines = [
        f"### {title}",
        "",
        "| 组 | 样本数 | 方向命中 | 系统偏差 | 兑现率 | 标识 |",
        "|----|--------|---------|---------|-------|------|",
    ]
    if order is not None:
        agg = sorted(
            agg,
            key=lambda r: order.index(r["bucket"]) if r["bucket"] in order else 999,
        )
    for r in agg:
        sym = _tag_symbol(r["direction_hit_rate"], r["systematic_bias"], r["high_pred_realize_rate"])
        lines.append(
            f"| {r['bucket']} | {r['n_samples']:,} | "
            f"{r['direction_hit_rate']:.1%} | {r['systematic_bias']:+.2%} | "
            f"{r['high_pred_realize_rate']:.1%} | {sym} |"
        )
    return "\n".join(lines)


def format_markdown_report(db_path: str, as_of_date: str) -> str:
    """生成完整的周度信号可信度 Markdown 报告。

    Args:
        db_path: SQLite 数据库路径。
        as_of_date: 报告截止日期。

    Returns:
        Markdown 字符串。
    """
    market_cap = aggregate_by_bucket(db_path, "market_cap_bucket", as_of_date)
    industry = aggregate_by_bucket(db_path, "industry", as_of_date)
    liquidity = aggregate_by_bucket(db_path, "liquidity_bucket", as_of_date)

    # 行业挑 Top5/Bottom5（按方向命中率，仅取样本≥50）
    ind_filtered = [r for r in industry if r["n_samples"] >= 50]
    ind_sorted = sorted(ind_filtered, key=lambda r: r["direction_hit_rate"] or 0)
    ind_bottom = ind_sorted[:5]
    ind_top = ind_sorted[-5:][::-1]

    parts = [
        "# 信号可信度 · 全局失效诊断",
        f"\n**截止日**: {as_of_date}\n",
        _section("按市值分组", market_cap, order=["微盘", "小盘", "中盘", "大盘", "未知"]),
        "",
        _section("按流动性分组", liquidity, order=["低", "中低", "中高", "高", "未知"]),
        "",
        "### 按行业分组 (样本≥50 的前 5 / 后 5)",
        "",
        "**⚠️ 失效最严重**:",
        _section("失效 Top5", ind_bottom).split("\n", 2)[2],
        "",
        "**✓ 最可靠**:",
        _section("可靠 Top5", ind_top).split("\n", 2)[2],
    ]
    return "\n".join(parts)
