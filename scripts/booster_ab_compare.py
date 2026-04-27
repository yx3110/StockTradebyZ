"""P2.8b: end-to-end A/B comparison of post-rank booster on real report JSON.

Loads a recent generation of ng1.0.6 reports, joins each pick with:
  - strategy_hits from stock_signals table (zh names)
  - trust_tag from signal_trust system (best-effort, defaults to ⚪)
  - regime from market_regime_signals.v11_bull
Runs apply_post_rank_booster, then computes Top-N realized forward 10d
return for both ranking schemes from forward_samples.csv.

Output:
  reports/diagnostics/booster_ab.md — table of dates × (top10_baseline,
  top10_boosted, Δ).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DB = ROOT / "data_adapter" / "stock_data.db"
SAMPLES = ROOT / "reports" / "forward_test" / "forward_samples.csv"

from stock_selctor.post_rank_booster import apply_post_rank_booster


def load_strategy_hits(start: str, end: str) -> dict[tuple[str, str], list[str]]:
    """Map (date, code) → list of strategy zh names."""
    conn = sqlite3.connect(str(DB), timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        rows = conn.execute(
            """
            SELECT ss.signal_date, s.code, ss.strategy_name
              FROM stock_signals ss JOIN securities s ON s.id = ss.security_id
             WHERE ss.signal_date BETWEEN ? AND ?
            """, (start, end),
        ).fetchall()
    finally:
        conn.close()
    out: dict[tuple[str, str], list[str]] = {}
    for d, c, sn in rows:
        out.setdefault((d, c), []).append(sn)
    return out


def load_regime(start: str, end: str) -> dict[str, str]:
    conn = sqlite3.connect(str(DB), timeout=30)
    try:
        rows = conn.execute(
            "SELECT trade_date, v11_bull FROM market_regime_signals "
            "WHERE trade_date BETWEEN ? AND ?",
            (start, end),
        ).fetchall()
    finally:
        conn.close()
    return {r[0]: ("bull" if r[1] == 1 else "bear") for r in rows}


def load_trust_tags() -> dict[str, str]:
    """Best-effort trust tag lookup — schema varies across deployments;
    fall back to '⚪' (neutral) if table absent or empty."""
    candidates = [
        ("signal_trust_scores", "code", "trust_tag"),
        ("signal_trust_tags", "code", "tag"),
        ("signal_trust", "code", "trust_tag"),
    ]
    conn = sqlite3.connect(str(DB), timeout=30)
    try:
        for table, code_col, tag_col in candidates:
            try:
                rows = conn.execute(f"SELECT {code_col}, {tag_col} FROM {table}").fetchall()
                if rows:
                    return {r[0]: (r[1] or "⚪") for r in rows}
            except sqlite3.OperationalError:
                continue
    finally:
        conn.close()
    return {}


def load_report(path: Path, regime: dict[str, str], strat_hits: dict, trust: dict) -> tuple[str, str, list[dict]]:
    """Return (date, regime, picks) where picks is list of dicts with rank_score etc."""
    d = json.loads(path.read_text())
    date = d["analysis_date"]
    rg = regime.get(date, "bull")
    picks = []
    for s in d.get("all_stocks_with_scores", []):
        c = s.get("stock_code")
        if not c:
            continue
        picks.append({
            "code": c,
            "rank_score": float(s.get("rank_score") or s.get("score") or 0.0),
            "strategy_hits": strat_hits.get((date, c), []),
            "trust_tag": trust.get(c, "⚪"),
        })
    return date, rg, picks


def realized_top_n_return(panel: pd.DataFrame, picks: list[dict], date: str, top_n: int,
                          score_field: str, ret_col: str) -> float:
    if not picks:
        return float("nan")
    top_codes = [p["code"] for p in picks[:top_n]]
    g = panel[(panel["report_date"] == date) & (panel["stock_code"].isin(top_codes))]
    if g.empty:
        return float("nan")
    return float(g[ret_col].mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", default="reports/daily_selection_ng106")
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-04-01")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--horizon", choices=["5d", "10d", "15d"], default="10d")
    ap.add_argument("--out", default="reports/diagnostics/booster_ab.md")
    args = ap.parse_args()

    print(f"[1/4] loading lookups...")
    strat_hits = load_strategy_hits(args.start, args.end)
    print(f"  strategy_hits: {len(strat_hits):,} (date, code) pairs")
    regime = load_regime(args.start, args.end)
    print(f"  regime: {len(regime)} dates")
    trust = load_trust_tags()
    print(f"  trust tags: {len(trust)} codes (0 if signal_trust table absent)")

    print(f"[2/4] loading forward_samples for realized returns...")
    panel = pd.read_csv(SAMPLES, dtype={"stock_code": str})
    panel = panel[panel["scoring_version"] == "ng1.0.6"]
    panel = panel[(panel["report_date"] >= args.start) & (panel["report_date"] <= args.end)]
    ret_col = f"forward_ret_{args.horizon}"
    print(f"  panel rows: {len(panel):,}")

    print(f"[3/4] running A/B over {args.report_dir}...")
    files = sorted(Path(args.report_dir).glob("analysis_data_*.json"))
    rows = []
    for path in files:
        date_digits = path.stem.replace("analysis_data_", "")
        if not (date_digits.isdigit() and len(date_digits) == 8):
            continue
        date = f"{date_digits[:4]}-{date_digits[4:6]}-{date_digits[6:8]}"
        if not (args.start <= date <= args.end):
            continue
        d, rg, picks = load_report(path, regime, strat_hits, trust)
        if not picks:
            continue
        # Baseline: rank by rank_score (already sorted in JSON typically)
        base_picks = sorted(picks, key=lambda p: -p["rank_score"])
        boost_picks = apply_post_rank_booster(picks, regime=rg)
        ret_base = realized_top_n_return(panel, base_picks, d, args.top_n, "rank_score", ret_col)
        ret_boost = realized_top_n_return(panel, boost_picks, d, args.top_n, "rank_score_boosted", ret_col)
        # How many top-10 picks differ?
        base_set = set(p["code"] for p in base_picks[:args.top_n])
        boost_set = set(p["code"] for p in boost_picks[:args.top_n])
        overlap = len(base_set & boost_set)
        # average bonus / mult applied to top-10
        avg_bonus = float(np.mean([p.get("_booster_strategy_bonus", 0.0)
                                    for p in boost_picks[:args.top_n]]))
        avg_mult = float(np.mean([p.get("_booster_trust_mult", 1.0)
                                   for p in boost_picks[:args.top_n]]))
        rows.append({
            "date": d, "regime": rg, "n_picks": len(picks),
            "overlap_top_n": overlap,
            "avg_bonus": avg_bonus, "avg_trust_mult": avg_mult,
            f"ret_base_{args.horizon}": ret_base,
            f"ret_boost_{args.horizon}": ret_boost,
            f"delta_{args.horizon}": (ret_boost - ret_base) if not np.isnan(ret_base + ret_boost) else np.nan,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no overlapping report dates with forward_samples")
    print(f"\nA/B sample: {len(df)} dates")
    delta_col = f"delta_{args.horizon}"
    valid = df.dropna(subset=[delta_col])
    print(f"\nrealized {args.horizon} fwd return:")
    print(f"  baseline mean: {valid[f'ret_base_{args.horizon}'].mean():+.4%}")
    print(f"  boosted mean:  {valid[f'ret_boost_{args.horizon}'].mean():+.4%}")
    print(f"  Δ mean:        {valid[delta_col].mean():+.4%}")
    print(f"  Δ win rate:    {(valid[delta_col] > 0).mean():.1%}")
    print(f"  avg overlap:   {df['overlap_top_n'].mean():.1f} / {args.top_n}")
    print(f"  avg bonus:     {df['avg_bonus'].mean():.2f}")
    print(f"  avg trust:     {df['avg_trust_mult'].mean():.3f}")

    print(f"[4/4] writing {args.out}...")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Booster A/B (P2.8b) — ng1.0.6 Top-{args.top_n} {args.horizon}",
        "",
        f"**窗口**: {args.start} ~ {args.end}  ",
        f"**样本**: {len(df)} 日 / {len(valid)} 有 forward 数据  ",
        f"**Trust 表**: {'已读' if trust else '缺失 → 全 ⚪ neutral'}  ",
        "",
        "## 总体",
        "",
        f"- baseline 平均 {args.horizon} 收益: **{valid[f'ret_base_{args.horizon}'].mean():+.4%}**",
        f"- boosted 平均 {args.horizon} 收益:  **{valid[f'ret_boost_{args.horizon}'].mean():+.4%}**",
        f"- Δ 平均:                     **{valid[delta_col].mean():+.4%}**",
        f"- Δ 胜率 (boosted > baseline): **{(valid[delta_col] > 0).mean():.1%}**",
        f"- top-{args.top_n} 重合度:     {df['overlap_top_n'].mean():.1f} / {args.top_n}",
        f"- 平均策略 bonus:              {df['avg_bonus'].mean():.2f} pts",
        f"- 平均 trust 乘子:             {df['avg_trust_mult'].mean():.3f}",
        "",
        "## 各 regime 拆分",
        "",
    ]
    for rg in ("bull", "bear"):
        sub = valid[valid["regime"] == rg]
        if sub.empty:
            continue
        lines += [
            f"### {rg.upper()} ({len(sub)} 天)",
            f"- baseline: {sub[f'ret_base_{args.horizon}'].mean():+.4%}",
            f"- boosted:  {sub[f'ret_boost_{args.horizon}'].mean():+.4%}",
            f"- Δ:        {sub[delta_col].mean():+.4%}  (胜率 {(sub[delta_col] > 0).mean():.1%})",
            "",
        ]

    lines += [
        "## 解读",
        "",
        "Booster 的设计目标不是 alpha 提升而是:",
        "1. **过滤已知失效信号** (🔴 trust × 0.6 / 死策略不给 bonus)",
        "2. **强化与 regime 一致的量化信号** (牛市加少负, 熊市加暴力K)",
        "",
        "成功标准: Δ 胜率 ≥ 50% AND Δ 均值不显著为负. 失败 (Δ < 0 显著)",
        "时回退至无 booster baseline 即可, 不影响主信号.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
