"""样本池构建器。"""
import json
import logging
import math
import re
from pathlib import Path
from typing import Iterator
from .constants import PRED_THRESHOLD, VERSION_PRIORITY, HOLD_DAYS, MARKET_CAP_BUCKETS
from .db import connect

logger = logging.getLogger(__name__)

# 优先级映射：越小越高
_PRIORITY_MAP = {v: i for i, v in enumerate(VERSION_PRIORITY)}
_UNKNOWN_PRIORITY = len(VERSION_PRIORITY) + 1

_DIR_VERSION_RE = re.compile(r"^daily_selection_(.+?)$")


def _extract_version(dir_name: str) -> str | None:
    m = _DIR_VERSION_RE.match(dir_name)
    if not m:
        return None
    version = m.group(1)
    # 剥离后缀如 _fullmarket / _pre2020 / _wf_oos / _fast / _ensemble_3seed
    for suffix in ("_fullmarket", "_pre2020", "_wf_oos", "_fast",
                   "_ensemble_3seed", "_ensemble_5seed", "_ensemble", "_90d", "_fixed"):
        if version.endswith(suffix):
            version = version[: -len(suffix)]
    return version


def scan_reports(report_parent_dirs: list[str]) -> Iterator[dict]:
    """
    扫描 `reports/` 下所有 `daily_selection_*` 子目录, 产出满足 pred_10d > 阈值的记录.

    report_parent_dirs: 形如 ['reports'] 的父目录列表(测试时用 tmp_path 对应目录).
    Yields: {code, trade_date, pred_10d, version}
    """
    for parent in report_parent_dirs:
        p = Path(parent)
        if not p.exists():
            logger.warning(f"目录不存在: {p}")
            continue
        for sub in sorted(p.iterdir()):
            if not sub.is_dir():
                continue
            version = _extract_version(sub.name)
            if not version:
                continue
            # 跳过带有分析后缀的报告目录(它们是离线实验, 不应污染生产可信度)
            if any(x in sub.name for x in ("_pre2020", "_wf_oos", "_fast")):
                continue
            for json_file in sorted(sub.glob("analysis_data_*.json")):
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"跳过坏文件 {json_file}: {e}")
                    continue
                trade_date = data.get("analysis_date")
                if not trade_date:
                    # 从文件名推断: analysis_data_20260412.json → 2026-04-12
                    m = re.search(r"(\d{4})(\d{2})(\d{2})", json_file.stem)
                    if m:
                        trade_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                    else:
                        continue
                for stock in data.get("all_stocks_with_scores", []):
                    pred = stock.get("pred_10d")
                    try:
                        pred_val = float(pred) if pred is not None else 0.0
                    except (TypeError, ValueError):
                        continue
                    if math.isnan(pred_val) or pred_val <= PRED_THRESHOLD:
                        continue
                    code = stock.get("stock_code")
                    if not code:
                        continue
                    yield {
                        "code": code,
                        "trade_date": trade_date,
                        "pred_10d": pred_val,
                        "version": version,
                    }


def dedupe_by_version(records: list[dict]) -> list[dict]:
    """按 (code, trade_date) 去重, 保留版本优先级最高的记录."""
    best: dict[tuple[str, str], dict] = {}
    for r in records:
        key = (r["code"], r["trade_date"])
        cur = best.get(key)
        new_p = _PRIORITY_MAP.get(r["version"], _UNKNOWN_PRIORITY)
        if cur is None or new_p < _PRIORITY_MAP.get(cur["version"], _UNKNOWN_PRIORITY):
            best[key] = r
    return list(best.values())


def streaming_dedupe(records_iter) -> dict[tuple[str, str], dict]:
    """
    流式版 dedupe_by_version: 边扫边去重, 内存 O(唯一 (code, trade_date) 对) 而非 O(总记录数).
    对 67K JSON/154GB 规模的首次建库必需.
    Returns: {(code, trade_date): record} — 调用方用 .values() 拿列表.
    """
    best: dict[tuple[str, str], dict] = {}
    n_seen = 0
    for r in records_iter:
        n_seen += 1
        if n_seen % 100000 == 0:
            logger.info(f"  流式扫描: {n_seen:,} 条原始记录, {len(best):,} 条去重后")
        key = (r["code"], r["trade_date"])
        new_p = _PRIORITY_MAP.get(r["version"], _UNKNOWN_PRIORITY)
        cur = best.get(key)
        if cur is None or new_p < _PRIORITY_MAP.get(cur["version"], _UNKNOWN_PRIORITY):
            best[key] = r
    logger.info(f"  流式扫描完成: {n_seen:,} 条原始记录 → {len(best):,} 条去重后")
    return best


def compute_actual_10d(db_path: str, code: str, trade_date: str) -> float | None:
    """
    用 daily_quotes 查 T 日 close 和 T+HOLD_DAYS 个交易日后的 close.
    返回实际收益率, 查不到返回 None.
    """
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT id FROM securities WHERE code = ?", (code,)
        ).fetchone()
        if row is None:
            return None
        sid = row["id"]
        quotes = conn.execute(
            "SELECT trade_date, close FROM daily_quotes "
            "WHERE security_id = ? AND trade_date >= ? "
            "ORDER BY trade_date ASC LIMIT ?",
            (sid, trade_date, HOLD_DAYS + 1),
        ).fetchall()
        if len(quotes) < HOLD_DAYS + 1:
            return None
        if quotes[0]["trade_date"] != trade_date:
            # T 日本身也需是交易日
            return None
        p0 = quotes[0]["close"]
        pN = quotes[HOLD_DAYS]["close"]
        if p0 is None or pN is None or p0 == 0:
            return None
        return (pN - p0) / p0
    finally:
        conn.close()


def compute_sample_end_date(db_path: str, trade_date: str) -> str | None:
    """
    给定 trade_date, 返回 sample_end_date = 市场第 HOLD_DAYS 个交易日后的日期.
    用 daily_quotes 中的不重复交易日来推(大盘行情日期一致).
    若无法确定(如未来日期), 返回 None.
    """
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT trade_date FROM daily_quotes "
            "WHERE trade_date >= ? ORDER BY trade_date ASC LIMIT ?",
            (trade_date, HOLD_DAYS + 1),
        ).fetchall()
        if len(rows) < HOLD_DAYS + 1:
            return None
        return rows[HOLD_DAYS]["trade_date"]
    finally:
        conn.close()


def compute_market_cap_bucket(db_path: str, code: str, trade_date: str) -> str:
    """查 daily_basic.circ_mv（万元），按 MARKET_CAP_BUCKETS 分档。"""
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT db.circ_mv FROM daily_basic db "
            "JOIN securities s ON s.id = db.security_id "
            "WHERE s.code = ? AND db.trade_date = ?",
            (code, trade_date),
        ).fetchone()
        if row is None or row["circ_mv"] is None:
            return "未知"
        mv = row["circ_mv"]
        for lo, hi, label in MARKET_CAP_BUCKETS:
            if lo <= mv < hi:
                return label
        return "未知"
    finally:
        conn.close()


def compute_liquidity_bucket(
    db_path: str, code: str, trade_date: str, thresholds: tuple[float, float, float]
) -> str:
    """thresholds = (p25, p50, p75); 基于该股 30 日日均成交额分档。"""
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT AVG(amount) AS m FROM ("
            "  SELECT amount FROM daily_quotes dq "
            "  JOIN securities s ON s.id = dq.security_id "
            "  WHERE s.code = ? AND dq.trade_date <= ? "
            "  ORDER BY dq.trade_date DESC LIMIT 30"
            ")",
            (code, trade_date),
        ).fetchone()
        if row is None or row["m"] is None:
            return "未知"
        m = row["m"]
        p25, p50, p75 = thresholds
        if m < p25:
            return "低"
        elif m < p50:
            return "中低"
        elif m < p75:
            return "中高"
        else:
            return "高"
    finally:
        conn.close()


def compute_liquidity_thresholds(db_path: str, as_of_date: str) -> tuple[float, float, float]:
    """基于所有股票的 30 日均成交额 p25/p50/p75 (与 compute_liquidity_bucket 一致)."""
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "WITH ranked AS ("
            "  SELECT security_id, amount, "
            "    ROW_NUMBER() OVER (PARTITION BY security_id ORDER BY trade_date DESC) AS rn "
            "  FROM daily_quotes "
            "  WHERE trade_date <= ? AND trade_date >= date(?, '-60 days') "
            ") "
            "SELECT security_id, AVG(amount) AS m "
            "FROM ranked "
            "WHERE rn <= 30 "
            "GROUP BY security_id "
            "HAVING COUNT(*) >= 15",
            (as_of_date, as_of_date),
        ).fetchall()
        vals = sorted(float(r["m"]) for r in rows if r["m"] is not None)
        if len(vals) < 4:
            return (1e8, 3e8, 1e9)
        n = len(vals)
        return (vals[n // 4], vals[n // 2], vals[3 * n // 4])
    finally:
        conn.close()


def _industry_lookup(db_path: str, codes: list[str]) -> dict[str, str]:
    """批量查询股票行业，返回 {code: industry}。"""
    if not codes:
        return {}
    conn = connect(db_path)
    try:
        placeholders = ",".join("?" * len(codes))
        rows = conn.execute(
            f"SELECT code, industry FROM securities WHERE code IN ({placeholders})",
            codes,
        ).fetchall()
        return {r["code"]: (r["industry"] or "未分类") for r in rows}
    finally:
        conn.close()


def upsert_samples(db_path: str, rows: list[dict], update_actual: bool = False) -> int:
    """
    写入样本表，幂等。
    update_actual=True 时允许覆盖已有的 actual_10d（用于回填）。
    返回处理的行数。
    """
    if not rows:
        return 0
    conn = connect(db_path)
    try:
        n = 0
        for r in rows:
            if update_actual:
                conn.execute(
                    "INSERT INTO signal_trust_samples "
                    "(code, trade_date, sample_end_date, pred_10d, actual_10d, version, "
                    " market_cap_bucket, industry, liquidity_bucket) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(code, trade_date) DO UPDATE SET "
                    "  actual_10d = excluded.actual_10d",
                    (r["code"], r["trade_date"], r["sample_end_date"],
                     r["pred_10d"], r.get("actual_10d"), r["version"],
                     r.get("market_cap_bucket"), r.get("industry"), r.get("liquidity_bucket")),
                )
            else:
                conn.execute(
                    "INSERT OR IGNORE INTO signal_trust_samples "
                    "(code, trade_date, sample_end_date, pred_10d, actual_10d, version, "
                    " market_cap_bucket, industry, liquidity_bucket) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (r["code"], r["trade_date"], r["sample_end_date"],
                     r["pred_10d"], r.get("actual_10d"), r["version"],
                     r.get("market_cap_bucket"), r.get("industry"), r.get("liquidity_bucket")),
                )
            n += 1
        conn.commit()
        return n
    finally:
        conn.close()
