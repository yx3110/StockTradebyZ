"""P1.4: Pre-2020 因子风格衰减诊断.

为每个 ng1.0.1 stock-level 特征对比 2018-2019 vs 2020-2026 两个窗口的
forward 10d Spearman IC, 输出:
- 符号翻转因子 (sign flip): 在两个窗口里 IC 同号性破坏
- |IC| 衰减榜 Top 15: 跨窗口绝对 IC 损失最大的因子

输出到 reports/diagnostics/pre2020_factor_decay.md.

设计动机: ng1.0.6 v1 在 2018-2019 OOS 净年化 +0.7% 是所有 NG 唯一为正,
但 V5.2 仅 41% C. 这说明部分因子在 2018 风格里反向, 模型整体侥幸但
仍有改进空间. 找出反向因子, 后续可在 ng1.6.2/ng2.2 训练里给负权重
约束或剔除.
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
OUT = ROOT / "reports" / "diagnostics" / "pre2020_factor_decay.md"


def load_panel(start: str, end: str, table: str = "ng101_feature_cache") -> pd.DataFrame:
    """Load feature_cache rows; expand features_json into one column per factor."""
    conn = sqlite3.connect(str(DB), timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        df = pd.read_sql_query(
            f"SELECT trade_date, code, features_json FROM {table} "
            "WHERE trade_date BETWEEN ? AND ?",
            conn,
            params=[start, end],
        )
    finally:
        conn.close()
    if df.empty:
        return df
    feats_records = df["features_json"].apply(json.loads).tolist()
    feats = pd.DataFrame.from_records(feats_records)
    return pd.concat([df[["trade_date", "code"]].reset_index(drop=True), feats], axis=1)


def fwd_returns(min_date: str, max_date: str, n: int = 10) -> pd.DataFrame:
    """forward N-day return panel keyed by (code, trade_date)."""
    conn = sqlite3.connect(str(DB), timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        df = pd.read_sql_query(
            "SELECT s.code, dq.trade_date, dq.close FROM daily_quotes dq "
            "JOIN securities s ON s.id = dq.security_id "
            "WHERE dq.trade_date BETWEEN ? AND ? AND s.type = 'A股'",
            conn,
            params=[min_date, max_date],
        )
    finally:
        conn.close()
    if df.empty:
        return df
    df = df.sort_values(["code", "trade_date"])
    df["fwd_close"] = df.groupby("code")["close"].shift(-n)
    df["fwd_ret"] = df["fwd_close"] / df["close"] - 1.0
    return df[["code", "trade_date", "fwd_ret"]].dropna()


def daily_ic_series(panel_with_ret: pd.DataFrame, factor: str) -> np.ndarray:
    out = []
    for d, g in panel_with_ret.groupby("trade_date"):
        g2 = g[[factor, "fwd_ret"]].dropna()
        if len(g2) < 30:
            continue
        ic, _ = spearmanr(g2[factor], g2["fwd_ret"])
        if not np.isnan(ic):
            out.append(float(ic))
    return np.asarray(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-start", default="2018-01-01")
    ap.add_argument("--pre-end", default="2019-12-31")
    ap.add_argument("--post-start", default="2020-01-01")
    ap.add_argument("--post-end", default="2026-04-26")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--table", default="ng101_feature_cache")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    print(f"[1/4] loading pre window {args.pre_start}~{args.pre_end} from {args.table}...")
    pre = load_panel(args.pre_start, args.pre_end, args.table)
    print(f"  → {len(pre):,} rows")
    print(f"[2/4] loading post window {args.post_start}~{args.post_end}...")
    post = load_panel(args.post_start, args.post_end, args.table)
    print(f"  → {len(post):,} rows")
    if pre.empty or post.empty:
        raise SystemExit("missing rows in one window — feature cache backfill required")

    print(f"[3/4] computing forward {args.horizon}d returns...")
    pre_ret = fwd_returns(args.pre_start, args.pre_end, args.horizon)
    post_ret = fwd_returns(args.post_start, args.post_end, args.horizon)
    pre_join = pre.merge(pre_ret, on=["code", "trade_date"])
    post_join = post.merge(post_ret, on=["code", "trade_date"])
    print(f"  pre joined: {len(pre_join):,}, post joined: {len(post_join):,}")

    factors = [c for c in pre.columns if c not in ("trade_date", "code")]
    print(f"[4/4] scanning {len(factors)} factors...")

    rows = []
    for i, f in enumerate(factors, 1):
        if i % 10 == 0:
            print(f"  ...{i}/{len(factors)}")
        if f not in post.columns:
            continue
        ic_pre = daily_ic_series(pre_join, f)
        ic_post = daily_ic_series(post_join, f)
        if len(ic_pre) < 50 or len(ic_post) < 50:
            continue
        m_pre = ic_pre.mean()
        m_post = ic_post.mean()
        rows.append({
            "factor": f,
            "ic_pre": m_pre,
            "ic_post": m_post,
            "icir_pre": m_pre / (ic_pre.std(ddof=1) + 1e-9),
            "icir_post": m_post / (ic_post.std(ddof=1) + 1e-9),
            "n_pre": len(ic_pre),
            "n_post": len(ic_post),
            "sign_flip": int(np.sign(m_pre) != np.sign(m_post) and abs(m_pre) > 0.005 and abs(m_post) > 0.005),
            "abs_decay": abs(m_post) - abs(m_pre),  # negative = stronger pre, weaker post
            "delta": m_post - m_pre,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no factors passed sample-count gate (>= 50 daily ICs each)")

    flip = df[df["sign_flip"] == 1].sort_values("ic_pre", key=lambda s: s.abs(), ascending=False)
    decay = df.assign(loss=df["abs_decay"]).sort_values("loss").head(15)
    grow = df.assign(gain=df["abs_decay"]).sort_values("gain", ascending=False).head(15)

    lines = [
        "# Pre-2020 因子风格衰减诊断 (P1.4)",
        "",
        f"- 对比窗口: **{args.pre_start} ~ {args.pre_end}** vs **{args.post_start} ~ {args.post_end}**",
        f"- 标签: forward {args.horizon}d Spearman IC (按日均值)",
        f"- 总特征数: {len(df)} (含两端各 ≥50 个有效 IC)",
        f"- 符号翻转因子数: **{len(flip)}**",
        "",
        "## ⚠️ 符号翻转因子 (跨窗口 IC 同号性破坏, |IC|>0.005)",
        "",
    ]
    if not flip.empty:
        lines.append("| factor | ic_pre | ic_post | icir_pre | icir_post |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in flip.iterrows():
            lines.append(
                f"| `{r['factor']}` | {r['ic_pre']:+.4f} | {r['ic_post']:+.4f} | "
                f"{r['icir_pre']:+.3f} | {r['icir_post']:+.3f} |"
            )
    else:
        lines.append("无 (所有因子 IC 同号或 |IC| 太小不构成翻转)")

    lines += [
        "",
        "## 📉 |IC| 衰减最大 Top 15 (ic_post - ic_pre 绝对值损失)",
        "",
        "| factor | ic_pre | ic_post | abs_loss |",
        "|---|---:|---:|---:|",
    ]
    for _, r in decay.iterrows():
        lines.append(
            f"| `{r['factor']}` | {r['ic_pre']:+.4f} | {r['ic_post']:+.4f} | "
            f"{abs(r['ic_post']) - abs(r['ic_pre']):+.4f} |"
        )

    lines += [
        "",
        "## 📈 |IC| 增强最大 Top 15 (post 阶段更强)",
        "",
        "| factor | ic_pre | ic_post | abs_gain |",
        "|---|---:|---:|---:|",
    ]
    for _, r in grow.iterrows():
        lines.append(
            f"| `{r['factor']}` | {r['ic_pre']:+.4f} | {r['ic_post']:+.4f} | "
            f"{abs(r['ic_post']) - abs(r['ic_pre']):+.4f} |"
        )

    lines += [
        "",
        "## 解读建议",
        "",
        "- **符号翻转因子**: 是 Pre-2020 V5.2 弱的主要原因 — 训练只学到 2020+ 时代的方向. ng1.6.2/ng2.2 训练里可考虑剔除或加 L1 惩罚.",
        "- **|IC| 衰减因子**: post 时代权重应下调; 优先保留 |IC| 增强 / 跨窗口稳定的因子.",
        '- **配合 P1.3 Calmar 标签**: 风险调整 label 会自然削弱"高收益高回撤"因子的权重, 可能压制部分 sign_flip 暴露.',
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out_path}")
    print(f"  sign-flip factors: {len(flip)}")
    if not flip.empty:
        print(f"  worst: {flip.iloc[0]['factor']} pre={flip.iloc[0]['ic_pre']:+.3f} → post={flip.iloc[0]['ic_post']:+.3f}")


if __name__ == "__main__":
    main()
