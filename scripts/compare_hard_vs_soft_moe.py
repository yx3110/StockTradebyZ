"""P2.7c: Compare hard-switch (ng1.0.6) vs soft-MOE (ng1.0.6+soft) reports.

For each common date, compute:
  - top-10 overlap (#picks present in both)
  - rank-correlation of full universe scores
  - regime spec: hard says bull/bear, soft P_bull
  - whether soft picks include any ST stocks (trust filter is NOT applied
    in batch_generate_ng_soft.py; should be in real production wiring)

Output: reports/diagnostics/hard_vs_soft_moe.md
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data_adapter" / "stock_data.db"
HARD_DIR = ROOT / "reports" / "daily_selection_ng106_fullmarket"
SOFT_DIR = ROOT / "reports" / "daily_selection_ng106_soft"
OUT = ROOT / "reports" / "diagnostics" / "hard_vs_soft_moe.md"


def load_st_set() -> set[str]:
    conn = sqlite3.connect(str(DB), timeout=30)
    try:
        rows = conn.execute(
            "SELECT code FROM securities WHERE name LIKE '%ST%'",
        ).fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows}


def load_report(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_regime(date: str) -> tuple[int, int]:
    conn = sqlite3.connect(str(DB), timeout=30)
    try:
        row = conn.execute(
            "SELECT v11_bull, regime_v2 FROM market_regime_signals WHERE trade_date = ?",
            (date,),
        ).fetchone()
    finally:
        conn.close()
    return (row[0], row[1]) if row else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=10)
    args = ap.parse_args()

    st_set = load_st_set()
    print(f"loaded {len(st_set)} ST/stale-ST codes from securities")

    hard_files = sorted(HARD_DIR.glob("analysis_data_*.json"))
    soft_files = sorted(SOFT_DIR.glob("analysis_data_*.json"))
    soft_dates = {p.stem.replace("analysis_data_", "") for p in soft_files}

    rows = []
    for hp in hard_files:
        d8 = hp.stem.replace("analysis_data_", "")
        if d8 not in soft_dates:
            continue
        date = f"{d8[:4]}-{d8[4:6]}-{d8[6:]}"
        hard = load_report(hp)
        soft = load_report(SOFT_DIR / f"analysis_data_{d8}.json")
        if not hard or not soft:
            continue
        v11_bull, regime_v2 = load_regime(date)

        h_all = hard.get("all_stocks_with_scores", [])
        s_all = soft.get("all_stocks_with_scores", [])
        h_top = [s.get("stock_code") for s in h_all[:args.top_n]]
        s_top = [s.get("stock_code") for s in s_all[:args.top_n]]
        overlap = len(set(h_top) & set(s_top))

        # Rank correlation on overlap of full universe
        h_score = {s.get("stock_code"): float(s.get("rank_score", 0) or 0) for s in h_all}
        s_score = {s.get("stock_code"): float(s.get("rank_score", 0) or 0) for s in s_all}
        common_codes = list(set(h_score.keys()) & set(s_score.keys()))
        if len(common_codes) >= 30:
            h_arr = np.array([h_score[c] for c in common_codes])
            s_arr = np.array([s_score[c] for c in common_codes])
            spearman, _ = spearmanr(h_arr, s_arr)
        else:
            spearman = np.nan

        # ST in soft top-10
        soft_st = [c for c in s_top if c in st_set]

        rows.append({
            "date": date,
            "v11_hard_bull": v11_bull,
            "p_bull_soft": soft.get("p_bull"),
            "n_hard": len(h_all),
            "n_soft": len(s_all),
            "top10_overlap": overlap,
            "spearman_rs": spearman,
            "soft_st_count": len(soft_st),
            "soft_top1": s_top[0] if s_top else None,
            "hard_top1": h_top[0] if h_top else None,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no overlapping (hard, soft) reports found")

    print(df.to_string(index=False))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Hard vs Soft MOE Compare (P2.7c)",
        "",
        f"**Top-N**: {args.top_n}  ",
        f"**Reports compared**: {len(df)} 天  ",
        "",
        "## 每日对比",
        "",
        "| 日期 | hard_v11 | P_bull(soft) | top10 重合 | rank corr | soft 中 ST 票 | hard top1 | soft top1 |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for _, r in df.iterrows():
        bull_hard = "🐂" if r["v11_hard_bull"] == 1 else "🐻"
        sp = "—" if pd.isna(r["spearman_rs"]) else f"{r['spearman_rs']:+.3f}"
        lines.append(
            f"| {r['date']} | {bull_hard} | {r['p_bull_soft']:.3f} | "
            f"{r['top10_overlap']}/{args.top_n} | {sp} | {r['soft_st_count']} | "
            f"{r['hard_top1']} | {r['soft_top1']} |"
        )

    avg_overlap = df["top10_overlap"].mean()
    avg_corr = df["spearman_rs"].mean()
    avg_st = df["soft_st_count"].mean()
    lines += [
        "",
        "## 总结",
        "",
        f"- 平均 top-{args.top_n} 重合: **{avg_overlap:.1f}/{args.top_n}** ({avg_overlap/args.top_n:.0%})",
        f"- 平均全市场 rank correlation: **{avg_corr:+.3f}**",
        f"- soft 输出含 ST 股票数 (生产应被 post_filters 拦截): **{avg_st:.1f}/天**",
        "",
        "## 解读",
        "",
        "- **重合度低** = 两种 MOE 选股策略差异显著, 不是同一逻辑加噪声; soft 提供真正的 alternative.",
        "- **rank correlation** 反映两 scorer 在共同股票池的排序一致性 (-1 完全反向, +1 完全一致).",
        "- **ST 出现** = batch_generate_ng_soft.py 没接 post_filters / ST 过滤, 真生产接入时必须加.",
        "",
        "## 后续",
        "",
        "- 真生产接入需要在 selector 层接 dual scorer + post_filters 链",
        "- A/B 评估需要 forward returns 闭环 (运行 forward_test_tracker scan 后对比 IC/Top-N alpha)",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
