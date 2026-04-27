"""P0.2: forward OOS 90 日滚动 dashboard.

Reads reports/forward_test/forward_samples.csv (built by forward_test_tracker.py
scan) and emits markdown with rolling-window forward IC / ICIR / Top-N hit
rate / annualized return.

Why rolling: in-sample inflation 8x means we cannot trust V5.2 alone for
production gating. Rolling 90d forward window gives a real-money signal of
model decay; gate thresholds in forward_test_tracker gate sub-command should
read these numbers, not training-set ICs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_PATH = ROOT / "reports" / "forward_test" / "forward_samples.csv"
OUT_PATH = ROOT / "reports" / "forward_test" / "dashboard.md"


def load_panel() -> pd.DataFrame:
    if not SAMPLES_PATH.exists():
        raise SystemExit(f"missing {SAMPLES_PATH} — run forward_test_tracker scan first")
    df = pd.read_csv(SAMPLES_PATH, dtype={"stock_code": str})
    df['report_date'] = pd.to_datetime(df['report_date'])
    return df


def daily_metrics(df: pd.DataFrame, score_col: str, ret_col: str) -> pd.DataFrame:
    """One row per report_date: spearman IC, mean ret, top-10 hit rate."""
    rows = []
    for d, g in df.groupby('report_date'):
        g2 = g[[score_col, ret_col]].dropna()
        if len(g2) < 10:
            continue
        ic, _ = spearmanr(g2[score_col], g2[ret_col])
        if np.isnan(ic):
            continue
        top10 = g2.nlargest(10, score_col)[ret_col]
        rows.append({
            'date': d,
            'ic': ic,
            'top10_ret': top10.mean(),
            'top10_hit': (top10 > 0).mean(),
            'n': len(g2),
        })
    return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)


def rolling_window(daily: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """Step day-by-day; window is by calendar days, not row count."""
    out_rows = []
    for i, row in daily.iterrows():
        end = row['date']
        start = end - pd.Timedelta(days=window_days)
        win = daily[(daily['date'] > start) & (daily['date'] <= end)]
        if len(win) < 5:
            continue
        ic_mean = win['ic'].mean()
        ic_std = win['ic'].std(ddof=1) if len(win) > 1 else np.nan
        out_rows.append({
            'date': end,
            'forward_ic_mean': ic_mean,
            'forward_icir': ic_mean / (ic_std + 1e-9) if pd.notna(ic_std) else np.nan,
            'top10_avg_ret': win['top10_ret'].mean(),
            'top10_hit_rate': win['top10_hit'].mean(),
            'n_days': len(win),
        })
    return pd.DataFrame(out_rows)


def fmt_pct(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:+.2%}"


def fmt_4(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:+.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scoring-version', default='ng1.0.6')
    ap.add_argument('--window-days', type=int, default=90)
    ap.add_argument('--horizon', choices=['5d', '10d', '15d'], default='10d')
    ap.add_argument('--score-col', default='rank_score',
                    help="ranking score column (rank_score / composite / pred_10d)")
    ap.add_argument('--out', default=str(OUT_PATH))
    args = ap.parse_args()

    df = load_panel()
    df = df[df['scoring_version'] == args.scoring_version].copy()
    if df.empty:
        raise SystemExit(f"no rows for scoring_version={args.scoring_version!r}")
    ret_col = f'forward_ret_{args.horizon}'
    if ret_col not in df.columns:
        raise SystemExit(f"missing column {ret_col}")
    if args.score_col not in df.columns:
        # graceful fallback to alternative score columns
        for cand in ('rank_score', 'composite', 'pred_10d'):
            if cand in df.columns and df[cand].notna().any():
                args.score_col = cand
                break

    daily = daily_metrics(df, args.score_col, ret_col)
    if daily.empty:
        raise SystemExit("no daily ICs computed (samples too sparse?)")
    rolling = rolling_window(daily, args.window_days)
    if rolling.empty:
        raise SystemExit("not enough rolling-window observations")

    last = rolling.iloc[-1]
    overall_ic = daily['ic'].mean()
    overall_icir = overall_ic / (daily['ic'].std(ddof=1) + 1e-9)

    lines = [
        f"# Forward OOS Dashboard — {args.scoring_version}",
        "",
        f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Score column**: `{args.score_col}` | **Horizon**: {args.horizon} | **Window**: {args.window_days} 日滚动  ",
        f"**Panel size**: {len(df):,} 行 / {daily['date'].nunique()} 报告日  ",
        "",
        "## 全样本基线",
        "",
        f"- Forward IC (mean): **{fmt_4(overall_ic)}**",
        f"- Forward ICIR (mean/std): **{fmt_4(overall_icir)}**",
        f"- IC > 0 比例: **{(daily['ic'] > 0).mean():.1%}**",
        f"- Top-10 平均 {args.horizon} 收益: **{fmt_pct(daily['top10_ret'].mean())}**",
        f"- Top-10 胜率: **{daily['top10_hit'].mean():.1%}**",
        "",
        f"## 最新 {args.window_days} 日滚动",
        "",
        f"- 截止: {last['date'].strftime('%Y-%m-%d')} (覆盖 {int(last['n_days'])} 个报告日)",
        f"- Forward IC: **{fmt_4(last['forward_ic_mean'])}**",
        f"- Forward ICIR: **{fmt_4(last['forward_icir'])}**",
        f"- Top-10 平均 {args.horizon} 收益: **{fmt_pct(last['top10_avg_ret'])}**",
        f"- Top-10 胜率: **{last['top10_hit_rate']:.1%}**",
        "",
        "## 历史轨迹 (最近 30 个观测点)",
        "",
        "| 截止日期 | rolling_IC | rolling_ICIR | Top10 收益 | Top10 胜率 | n_days |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in rolling.tail(30).iterrows():
        lines.append(
            f"| {r['date'].strftime('%Y-%m-%d')} | "
            f"{fmt_4(r['forward_ic_mean'])} | "
            f"{fmt_4(r['forward_icir'])} | "
            f"{fmt_pct(r['top10_avg_ret'])} | "
            f"{r['top10_hit_rate']:.1%} | "
            f"{int(r['n_days'])} |"
        )

    lines += [
        "",
        "## 模型衰减预警",
        "",
    ]
    if len(rolling) >= 60:
        first_30 = rolling.head(30)['forward_ic_mean'].mean()
        last_30 = rolling.tail(30)['forward_ic_mean'].mean()
        delta = last_30 - first_30
        flag = "⚠️ ALERT" if delta < -0.02 else ("⚪ stable" if abs(delta) < 0.01 else "🟢 improving" if delta > 0 else "🟡 mild decay")
        lines += [
            f"- 首 30 个观测窗口平均 Forward IC: {fmt_4(first_30)}",
            f"- 末 30 个观测窗口平均 Forward IC: {fmt_4(last_30)}",
            f"- Δ: **{fmt_4(delta)}** {flag}",
            "",
            "门槛: Δ < -0.02 触发 ALERT (模型可能失效, 检查特征/重训)",
        ]
    else:
        lines.append("- 数据不足 60 个滚动观测点, 暂不评估衰减")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"  panel: {len(df):,} rows / {daily['date'].nunique()} days")
    print(f"  baseline forward IC ({args.horizon}): {fmt_4(overall_ic)} (ICIR {fmt_4(overall_icir)})")
    print(f"  latest {args.window_days}d window: IC {fmt_4(last['forward_ic_mean'])} / "
          f"top10_ret {fmt_pct(last['top10_avg_ret'])} / hit {last['top10_hit_rate']:.1%}")


if __name__ == "__main__":
    main()
