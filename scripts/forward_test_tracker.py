"""
Forward Test Tracker (P0.1) — 真 OOS 闭环.

每日生产报告 + 真实 forward 收益累积成 panel, 用来:
1. 校验 in-sample 评估 (V5.2/IC) 是否能预测真实未来
2. 生产模型切换前的 paper-trade gate (forward IC ≥ in-sample × 0.6, N ≥ 20)
3. 长期跟踪生产模型衰减

子命令:
- scan: 扫报告目录, 用 D+holding_days 日的真实收益打分, 增量 append 到 parquet
- report: 输出 weekly 汇总 (forward IC, Top-N 实际收益, win rate)
- gate: 给定 scoring_version + in_sample_ic 参考值, 输出 PASS/FAIL

数据存储: reports/forward_test/forward_samples.parquet
schema: (scoring_version, report_date, stock_code, top_n_rank, rank_score,
         pred_10d, composite, forward_ret_5d, forward_ret_10d, forward_ret_15d)
唯一键: (scoring_version, report_date, stock_code) — 重跑 scan 不会双写
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data_adapter" / "stock_data.db"
SAMPLES_PATH = ROOT / "reports" / "forward_test" / "forward_samples.csv"
WEEKLY_REPORT_PATH = ROOT / "reports" / "forward_test" / "weekly_report.md"

DEFAULT_TOP_N = 200          # 每日报告留前 200 算 IC (cheaper than 全市场, 足够 IC 显著)
DEFAULT_HOLDING_DAYS = 10
HORIZONS = (5, 10, 15)


# ──────────────────────── 工具 ────────────────────────

def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _next_business_day(date_str: str, days_ahead: int = 1) -> str:
    """T+N business day (粗略, 用 pd.bdate_range)."""
    d = pd.Timestamp(date_str)
    # bdate_range 包含起点, 所以多取一位
    bd = pd.bdate_range(d + pd.Timedelta(days=1), periods=days_ahead)
    return bd[-1].strftime("%Y-%m-%d")


def _parse_report_date(filename: str) -> Optional[str]:
    """从 'analysis_data_YYYYMMDD.json' 提取 'YYYY-MM-DD'."""
    base = os.path.basename(filename)
    if not base.startswith("analysis_data_") or not base.endswith(".json"):
        return None
    digits = base.replace("analysis_data_", "").replace(".json", "")
    if len(digits) != 8 or not digits.isdigit():
        return None
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"


# ──────────────────────── DB 批量取 forward 收益 ────────────────────────

def _fetch_forward_returns(
    pairs: list[tuple[str, str]],   # [(stock_code, buy_date_yyyy_mm_dd), ...]
    holding_days: int,
    db_path: Path = DB_PATH,
) -> dict[tuple[str, str], dict[int, float]]:
    """
    对每个 (code, buy_date), 拿 buy_date 后 max(HORIZONS)+5 业务日的 close,
    返回 {(code, buy_date): {5: ret_5d, 10: ret_10d, 15: ret_15d}}.

    业务规则:
    - buy_date 当天的 close 作为成本基准 (实盘是 T+1 close 但简化用 T close)
    - forward_ret_Nd = close[buy_date + N business days] / close[buy_date] - 1
    - 跨 trading_date gap (停牌/退市) 时 close 缺失 → 该 horizon 输出 NaN
    """
    if not pairs:
        return {}

    max_h = max(HORIZONS)
    # 每个 pair 单独 query 太慢, 按 code 聚合 → 一次查全部 trade_date
    code_to_dates: dict[str, list[str]] = {}
    for code, buy_date in pairs:
        code_to_dates.setdefault(code, []).append(buy_date)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout=30000")

    out: dict[tuple[str, str], dict[int, float]] = {}
    cur = conn.cursor()
    for code, buy_dates in code_to_dates.items():
        min_date = min(buy_dates)
        # max horizon + buffer days for non-trading days
        max_date = (pd.Timestamp(max(buy_dates)) +
                    pd.Timedelta(days=max_h + 10)).strftime("%Y-%m-%d")
        rows = cur.execute(
            """
            SELECT q.trade_date, q.close
            FROM daily_quotes q JOIN securities s ON s.id = q.security_id
            WHERE s.code = ? AND q.trade_date BETWEEN ? AND ?
            ORDER BY q.trade_date
            """,
            (code, min_date, max_date),
        ).fetchall()
        if not rows:
            for buy_date in buy_dates:
                out[(code, buy_date)] = {h: np.nan for h in HORIZONS}
            continue

        date_to_close = {d: c for d, c in rows if c is not None}
        sorted_dates = sorted(date_to_close.keys())

        for buy_date in buy_dates:
            row_horizons: dict[int, float] = {h: np.nan for h in HORIZONS}
            # buy_date close (找最近的可交易日, ≥ buy_date)
            buy_close = None
            buy_idx = None
            for i, d in enumerate(sorted_dates):
                if d >= buy_date:
                    buy_close = date_to_close[d]
                    buy_idx = i
                    break
            if buy_close is None or buy_close <= 0:
                out[(code, buy_date)] = row_horizons
                continue

            for h in HORIZONS:
                target_idx = buy_idx + h
                if target_idx < len(sorted_dates):
                    target_close = date_to_close[sorted_dates[target_idx]]
                    if target_close and target_close > 0:
                        row_horizons[h] = target_close / buy_close - 1
            out[(code, buy_date)] = row_horizons
    conn.close()
    return out


# ──────────────────────── scan ────────────────────────

def _load_existing_samples() -> pd.DataFrame:
    if not SAMPLES_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(SAMPLES_PATH, dtype={"stock_code": str})


def _save_samples(df: pd.DataFrame) -> None:
    _ensure_dir(SAMPLES_PATH)
    df.to_csv(SAMPLES_PATH, index=False)


def scan_reports(
    report_dirs: list[Path],
    scoring_version: str,
    top_n: int = DEFAULT_TOP_N,
    holding_days: int = DEFAULT_HOLDING_DAYS,
    rescan: bool = False,
) -> pd.DataFrame:
    """扫描 report_dirs, 抽 top-N 股, 拉 forward 收益, 增量 append."""
    existing = _load_existing_samples()
    today = pd.Timestamp.today().normalize()
    # 等待完整 forward 窗口: 最长 horizon (15d) + 周末缓冲 5 天
    cutoff_date = today - pd.Timedelta(days=max(HORIZONS) + 5)

    seen_keys: set[tuple[str, str, str]] = set()
    if not existing.empty and not rescan:
        seen_keys = set(zip(
            existing["scoring_version"], existing["report_date"], existing["stock_code"],
        ))

    new_rows: list[dict] = []
    pairs_to_fetch: list[tuple[str, str]] = []     # (code, buy_date)
    pending_meta: list[dict] = []                  # 等 forward ret 的元信息

    for rdir in report_dirs:
        rdir = Path(rdir)
        if not rdir.exists():
            print(f"[skip] {rdir} 不存在", file=sys.stderr)
            continue
        for json_file in sorted(rdir.glob("analysis_data_*.json")):
            report_date = _parse_report_date(json_file.name)
            if not report_date:
                continue
            if pd.Timestamp(report_date) > cutoff_date:
                continue  # 还没攒够 forward 窗口
            try:
                with open(json_file) as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[skip] {json_file}: {e}", file=sys.stderr)
                continue

            stocks = data.get("all_stocks_with_scores", [])
            if not stocks:
                continue
            # 按 composite (rank_score) 降序取 top_n
            sorted_stocks = sorted(
                stocks,
                key=lambda s: float(s.get("composite", s.get("rank_score", 0)) or 0),
                reverse=True,
            )[:top_n]

            buy_date = _next_business_day(report_date, days_ahead=1)

            for rank, s in enumerate(sorted_stocks, start=1):
                code = s.get("stock_code", "").strip()
                if not code:
                    continue
                key = (scoring_version, report_date, code)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                pending_meta.append({
                    "scoring_version": scoring_version,
                    "report_date": report_date,
                    "buy_date": buy_date,
                    "stock_code": code,
                    "top_n_rank": rank,
                    "rank_score": float(s.get("rank_score", 0) or 0),
                    "pred_5d": float(s.get("pred_5d", 0) or 0),
                    "pred_10d": float(s.get("pred_10d", 0) or 0),
                    "pred_15d": float(s.get("pred_15d", 0) or 0),
                    "composite": float(s.get("composite", 0) or 0),
                    "score": float(s.get("score", 0) or 0),
                })
                pairs_to_fetch.append((code, buy_date))

    if not pending_meta:
        print("[scan] 无新样本可加")
        return existing

    print(f"[scan] 拉 {len(pending_meta)} 个 forward 收益 (从 {len(report_dirs)} 个目录)...")
    fwd_map = _fetch_forward_returns(pairs_to_fetch, holding_days)

    for meta in pending_meta:
        key = (meta["stock_code"], meta["buy_date"])
        horizons = fwd_map.get(key, {h: np.nan for h in HORIZONS})
        meta["forward_ret_5d"] = horizons.get(5, np.nan)
        meta["forward_ret_10d"] = horizons.get(10, np.nan)
        meta["forward_ret_15d"] = horizons.get(15, np.nan)
        new_rows.append(meta)

    new_df = pd.DataFrame(new_rows)
    if existing.empty:
        out = new_df
    else:
        out = pd.concat([existing, new_df], ignore_index=True)
    out = out.drop_duplicates(
        subset=["scoring_version", "report_date", "stock_code"], keep="last"
    )
    _save_samples(out)
    print(f"[scan] 新增 {len(new_df)} 行, 总计 {len(out)} 行 → {SAMPLES_PATH}")
    return out


# ──────────────────────── report ────────────────────────

def _pick_score_col(df: pd.DataFrame) -> str:
    """挑非 trivial 的 score 列: composite > rank_score > pred_10d > score."""
    for col in ("composite", "rank_score", "pred_10d", "score"):
        if col in df.columns and df[col].nunique() > 5:
            return col
    return "rank_score"


def _compute_forward_ic(df: pd.DataFrame, horizon: int = 10) -> dict:
    """每日截面 Spearman IC + ICIR."""
    col = f"forward_ret_{horizon}d"
    if col not in df.columns or df[col].isna().all():
        return {"ic_mean": np.nan, "icir": np.nan, "n_days": 0,
                "score_col": "n/a"}

    score_col = _pick_score_col(df)
    import warnings as _warnings
    ic_per_day = []
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")  # 安静处理 constant-input
        for date, group in df.groupby("report_date"):
            sub = group.dropna(subset=[col, score_col])
            if len(sub) < 20 or sub[score_col].nunique() < 3:
                continue
            ic, _ = spearmanr(sub[score_col], sub[col])
            if not np.isnan(ic):
                ic_per_day.append(ic)
    if not ic_per_day:
        return {"ic_mean": np.nan, "icir": np.nan, "n_days": 0,
                "score_col": score_col}
    ic_arr = np.array(ic_per_day)
    return {
        "ic_mean": float(ic_arr.mean()),
        "ic_std": float(ic_arr.std()),
        "icir": float(ic_arr.mean() / ic_arr.std()) if ic_arr.std() > 1e-8 else 0.0,
        "ic_positive_pct": float((ic_arr > 0).mean()),
        "n_days": len(ic_arr),
        "score_col": score_col,
    }


def _compute_top_n_perf(df: pd.DataFrame, top_n: int = 10, horizon: int = 10) -> dict:
    """每日 Top-N 平均收益 + 胜率."""
    col = f"forward_ret_{horizon}d"
    if col not in df.columns:
        return {"top_n_mean_ret": np.nan, "top_n_win_rate": np.nan, "n_days": 0}
    daily_ret = []
    for date, group in df.groupby("report_date"):
        top = group.nsmallest(top_n, "top_n_rank")  # rank 1 是最优
        valid = top.dropna(subset=[col])
        if len(valid) < max(3, top_n // 2):
            continue
        daily_ret.append(valid[col].mean())
    if not daily_ret:
        return {"top_n_mean_ret": np.nan, "top_n_win_rate": np.nan, "n_days": 0}
    arr = np.array(daily_ret)
    return {
        "top_n_mean_ret_per_period": float(arr.mean()),
        "top_n_win_rate": float((arr > 0).mean()),
        "top_n_sharpe": float(arr.mean() / arr.std()) if arr.std() > 1e-8 else 0.0,
        "n_days": len(arr),
    }


def report_summary(scoring_version: Optional[str] = None,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> str:
    df = _load_existing_samples()
    if df.empty:
        return "[report] 无 forward samples, 先跑 scan"

    if scoring_version:
        df = df[df["scoring_version"] == scoring_version]
        if df.empty:
            return f"[report] {scoring_version} 无样本"
    if start_date:
        df = df[df["report_date"] >= start_date]
    if end_date:
        df = df[df["report_date"] <= end_date]
    if df.empty:
        return f"[report] {scoring_version} 在 {start_date}→{end_date} 无样本"

    lines = [
        f"# Forward Test Weekly Report",
        f"_生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        f"**总样本数**: {len(df)}",
        f"**覆盖 scoring_version**: {sorted(df['scoring_version'].unique())}",
        f"**报告日期范围**: {df['report_date'].min()} → {df['report_date'].max()}",
        f"**报告日期数**: {df['report_date'].nunique()}",
        "",
    ]

    for ver in sorted(df["scoring_version"].unique()):
        sub = df[df["scoring_version"] == ver]
        lines.append(f"## {ver}")
        lines.append("")
        lines.append(f"- 报告天数: {sub['report_date'].nunique()}")
        lines.append(f"- 总样本数: {len(sub)}")
        lines.append("")
        lines.append("| Horizon | Forward IC | ICIR | IC>0% | N天 | Top-10 平均收益 | Top-10 胜率 |")
        lines.append("|---|---|---|---|---|---|---|")
        for h in HORIZONS:
            ic = _compute_forward_ic(sub, h)
            top = _compute_top_n_perf(sub, top_n=10, horizon=h)
            lines.append(
                f"| {h}d "
                f"| {ic['ic_mean']:+.4f} "
                f"| {ic.get('icir', 0):+.3f} "
                f"| {ic.get('ic_positive_pct', 0):.1%} "
                f"| {ic['n_days']} "
                f"| {top.get('top_n_mean_ret_per_period', 0):+.2%} "
                f"| {top.get('top_n_win_rate', 0):.1%} |"
            )
        lines.append("")

    return "\n".join(lines)


def write_weekly_report(scoring_version: Optional[str] = None) -> Path:
    text = report_summary(scoring_version)
    _ensure_dir(WEEKLY_REPORT_PATH)
    WEEKLY_REPORT_PATH.write_text(text)
    return WEEKLY_REPORT_PATH


# ──────────────────────── gate ────────────────────────

def gate_decision(
    scoring_version: str,
    in_sample_ic: float,
    horizon: int = 10,
    min_n_days: int = 20,
    forward_ic_floor_ratio: float = 0.6,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> tuple[bool, str]:
    """
    生产切换 gate.

    PASS 条件:
    - forward N ≥ min_n_days (默认 20)
    - forward IC ≥ in_sample_ic × forward_ic_floor_ratio (默认 60%)
    - forward IC > 0 (符号一致)
    """
    df = _load_existing_samples()
    if df.empty:
        return False, "[gate] 无 forward samples"
    sub = df[df["scoring_version"] == scoring_version]
    if start_date:
        sub = sub[sub["report_date"] >= start_date]
    if end_date:
        sub = sub[sub["report_date"] <= end_date]
    if sub.empty:
        return False, f"[gate] {scoring_version} 无样本 (start={start_date}, end={end_date})"

    ic = _compute_forward_ic(sub, horizon)
    n = ic["n_days"]
    ic_mean = ic["ic_mean"]
    threshold = in_sample_ic * forward_ic_floor_ratio

    if np.isnan(ic_mean):
        return False, f"[gate] IC 不可计算 (n_days={n})"

    reasons = []
    if n < min_n_days:
        reasons.append(f"N={n} < {min_n_days}")
    if ic_mean <= 0:
        reasons.append(f"forward IC={ic_mean:+.4f} ≤ 0")
    if ic_mean < threshold:
        reasons.append(
            f"forward IC={ic_mean:+.4f} < threshold {threshold:+.4f} "
            f"(in_sample={in_sample_ic:+.4f} × {forward_ic_floor_ratio:.1f})"
        )

    passed = not reasons
    msg_lines = [
        f"[gate] {scoring_version} {horizon}d forward test",
        f"  N天数: {n} (min {min_n_days})",
        f"  Forward IC: {ic_mean:+.4f} (ICIR={ic.get('icir', 0):+.3f}, IC>0={ic.get('ic_positive_pct', 0):.1%})",
        f"  In-sample IC ref: {in_sample_ic:+.4f} → threshold {threshold:+.4f}",
        f"  ratio forward/in_sample: {ic_mean/in_sample_ic:.2%}" if in_sample_ic > 1e-8 else "",
        f"  结论: {'✅ PASS' if passed else '❌ FAIL'}",
    ]
    if reasons:
        msg_lines.append(f"  原因: {'; '.join(reasons)}")
    return passed, "\n".join(filter(None, msg_lines))


# ──────────────────────── CLI ────────────────────────

def _default_report_dirs(scoring_version: str) -> list[Path]:
    """根据 scoring_version 推断默认报告目录."""
    # ng1.0.6 → daily_selection_ng106_fullmarket; ng2.0a → daily_selection_ng2_0a_fullmarket
    short = scoring_version.replace(".", "").replace("v", "v")  # 'ng106', 'ng2.0a'→'ng20a'
    short = short.replace("ng20a", "ng2_0a").replace("ng21", "ng2_1")
    candidates = [
        ROOT / f"reports/daily_selection_{short}_fullmarket",
        ROOT / f"reports/daily_selection_{short}",
        ROOT / "reports/daily_selection_fullmarket",          # 默认生产
        ROOT / "reports/daily_selection",
    ]
    return [p for p in candidates if p.exists()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", help="扫报告 → append forward samples")
    p_scan.add_argument("--scoring-version", required=True)
    p_scan.add_argument("--report-dir", action="append",
                         help="报告目录 (可重复). 不传则按 scoring_version 推断")
    p_scan.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    p_scan.add_argument("--holding-days", type=int, default=DEFAULT_HOLDING_DAYS)
    p_scan.add_argument("--rescan", action="store_true",
                         help="忽略已存在样本, 全量重扫 (谨慎用)")

    p_report = sub.add_parser("report", help="输出 weekly summary")
    p_report.add_argument("--scoring-version", default=None,
                           help="只看一个版本; 默认所有")
    p_report.add_argument("--start-date", default=None,
                           help="只评估 ≥ 此日期的样本 (区分 forward / in-sample)")
    p_report.add_argument("--end-date", default=None)
    p_report.add_argument("--write", action="store_true",
                           help="同时写到 reports/forward_test/weekly_report.md")

    p_gate = sub.add_parser("gate", help="生产切换 gate (PASS/FAIL)")
    p_gate.add_argument("--scoring-version", required=True)
    p_gate.add_argument("--in-sample-ic", type=float, required=True,
                         help="参考 in-sample IC (来自训练日志或 wiki)")
    p_gate.add_argument("--horizon", type=int, default=10)
    p_gate.add_argument("--min-n-days", type=int, default=20)
    p_gate.add_argument("--floor-ratio", type=float, default=0.6)
    p_gate.add_argument("--start-date", default=None,
                         help="只评估 ≥ 此日期的样本")
    p_gate.add_argument("--end-date", default=None,
                         help="只评估 ≤ 此日期的样本")

    args = ap.parse_args()

    if args.cmd == "scan":
        report_dirs = (
            [Path(d) for d in args.report_dir]
            if args.report_dir
            else _default_report_dirs(args.scoring_version)
        )
        if not report_dirs:
            print(f"[error] 找不到 {args.scoring_version} 的报告目录, 用 --report-dir 指定",
                  file=sys.stderr)
            sys.exit(1)
        print(f"[scan] 扫描目录: {[str(d) for d in report_dirs]}")
        scan_reports(
            report_dirs,
            args.scoring_version,
            top_n=args.top_n,
            holding_days=args.holding_days,
            rescan=args.rescan,
        )

    elif args.cmd == "report":
        text = report_summary(args.scoring_version, args.start_date, args.end_date)
        print(text)
        if args.write:
            _ensure_dir(WEEKLY_REPORT_PATH)
            WEEKLY_REPORT_PATH.write_text(text)
            print(f"\n[written] {WEEKLY_REPORT_PATH}")

    elif args.cmd == "gate":
        passed, msg = gate_decision(
            args.scoring_version,
            in_sample_ic=args.in_sample_ic,
            horizon=args.horizon,
            min_n_days=args.min_n_days,
            forward_ic_floor_ratio=args.floor_ratio,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        print(msg)
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
