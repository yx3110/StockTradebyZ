#!/usr/bin/env python3
"""Multi-model candidate ranker: unified Top-50 scan + 6-factor scoring across NG versions.

Extends single_stock_review with:
  * multi-version Top-50 union (ng1.0.1 + ng1.0.6 by default)
  * unified 6-factor scoring (win_rate 35% + mean 25% + sample 10% + cross-model 10% + tech 10% + valuation 10%)
  * Tier bucketing (S/A+/A/B/C) — 0-100 weighted-score scale, distinct from north_star V2_GRADE_THRESHOLDS

Usage:
    python3 scripts/candidate_ranker.py --date 2026-04-17 --versions ng1.0.1,ng1.0.6

Output:
    reports/candidate_ranking/{versions}_{date}.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from single_stock_review import (
    _mean_win,
    compute_forward_returns,
    fetch_snapshot,
    find_todays_report,
    load_price_series,
    load_top_n,
    scan_historical_hits,
    version_tag,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / 'reports' / 'candidate_ranking'
TOP_N = 50

# (label, icon, min_score). First match wins; order from highest.
TIERS = [
    ('S', '🏆', 70),
    ('A+', '🥇', 60),
    ('A', '🥈', 50),
    ('B', '🥉', 40),
    ('C', '❌', 0),
]


def score_candidate(
    w10: float, m10: float, n10: int, is_cross: bool, snap: dict
) -> tuple[float, dict]:
    """6-factor weighted score. Returns (total, breakdown dict)."""
    # 零历史样本 = 未知, 给"中性 0 分"而不是免费 10 分 (过去会超过 w10=28.6% 的
    # 真实统计, 制造 IPO / 冷门股排名虚高).
    win_score = 0 if n10 == 0 else w10 * 0.35
    mean_score = 0 if n10 == 0 else max(0, min(25, m10 + 5))

    if n10 <= 0:
        sample_score = 2
    elif n10 < 10:
        sample_score = 3
    elif n10 < 20:
        sample_score = 6
    elif n10 < 50:
        sample_score = 8
    else:
        sample_score = 10

    cross_score = 10 if is_cross else 0

    j = float(snap.get('kdj_j') or 50)
    pos = float(snap.get('range20_pos_pct') or 50)
    if j < 30:
        tech = 6
    elif j < 60:
        tech = 10
    elif j < 85:
        tech = 7
    elif j < 100:
        tech = 4
    else:
        tech = 2
    if pos < 30:
        tech -= 1
    elif pos > 70:
        tech -= 2
    tech_score = max(0, min(10, tech))

    val = 5
    pe = snap.get('pe_ttm')
    pb = snap.get('pb')
    dv = snap.get('dv_ratio') or 0
    if pe and 0 < pe < 20:
        val += 4
    elif pe and pe < 40:
        val += 2
    elif pe and pe > 80:
        val -= 2
    if pb and pb < 1.5:
        val += 3
    elif pb and pb < 3:
        val += 1
    elif pb and pb > 6:
        val -= 2
    if dv > 2:
        val += 3
    elif dv > 1:
        val += 1
    name = snap.get('name') or ''
    # `ST`/`*ST`/`退`/`PT` 都按 ST 处理 (DB name 已由 refresh_stock_names 同步
    # Tushare namechange, 历史滞后问题已根治).
    if any(tag in name for tag in ('ST', '退', 'PT')):
        val -= 5
    if n10 == 0:
        val -= 3
    val_score = max(0, min(10, val))

    total = win_score + mean_score + sample_score + cross_score + tech_score + val_score
    return total, {
        'win': win_score, 'mean': mean_score, 'n': sample_score,
        'cross': cross_score, 'tech': tech_score, 'val': val_score,
    }


def tier_for(score: float) -> str:
    for label, _, min_s in TIERS:
        if score >= min_s:
            return label
    return TIERS[-1][0]


def _fmt_row(i: int, c: dict) -> str:
    name = (c['snap'].get('name') or '?')[:8]
    industry = (c['snap'].get('industry') or '?')[:6]
    ws = f"{c['w10']:.0f}%" if c['n10'] > 0 else '-'
    ms = f"{c['m10']:+.1f}%" if c['n10'] > 0 else '-'
    d = c['det']
    return (
        f"| {i} | {tier_for(c['score'])} | {c['code']} | {name} | {industry} | "
        f"**{c['score']:.1f}** | {d['win']:.1f} | {d['mean']:.1f} | {d['n']:.1f} | "
        f"{d['cross']:.1f} | {d['tech']:.1f} | {d['val']:.1f} | {c['source']} | "
        f"{ws} ({c['n10']}) | {ms} |"
    )


def _fmt_bullet(c: dict) -> str:
    name = c['snap'].get('name') or '?'
    ind = c['snap'].get('industry') or '?'
    ws = f"{c['w10']:.0f}%" if c['n10'] > 0 else '样本不足'
    ms = f"{c['m10']:+.1f}%" if c['n10'] > 0 else ''
    return (f"- **{c['code']} {name}** ({ind}) · 总分 {c['score']:.1f} · "
            f"来源 {c['source']} · 10d {ws} {ms} (n={c['n10']})")


def render_markdown(date_str: str, versions: list[str], ranked: list[dict]) -> str:
    lines = [
        f"# 📊 多模型候选综合排名 · {date_str}",
        "",
        f"_模型: {' + '.join(versions)} · Top-{TOP_N} 合并 · 6因子评分 "
        "(胜率35%+均值25%+样本10%+交集10%+技术10%+估值10%)_",
        "",
        "## 评分分布",
        "",
        "| Tier | 总分 | 含义 |",
        "|------|------|------|",
    ]
    tier_desc = {'S': '强力推荐', 'A+': '首选', 'A': '次选', 'B': '谨慎观察', 'C': '跳过'}
    for label, icon, min_s in TIERS:
        next_min = next((t[2] for t in TIERS if t[2] > min_s), None)
        threshold = f"≥{min_s}" if next_min is None else f"{min_s}-{next_min}"
        if min_s == 0:
            threshold = f"<{next(t[2] for t in TIERS if t[2] > 0)}"
        lines.append(f"| {icon} {label} | {threshold} | {tier_desc[label]} |")

    lines += [
        "",
        "---",
        "",
        "## 完整排名",
        "",
        "| # | Tier | 代码 | 名称 | 行业 | 总分 | 胜率 | 均值 | 样本 | 交集 | 技术 | 估值 | 来源 | 10d 胜率(n) | 10d 均值 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, c in enumerate(ranked, 1):
        lines.append(_fmt_row(i, c))

    lines += ["", "---", "", "## Tier 汇总 (含业务解读请交 Claude 做 narrative)"]
    for label, icon, min_s in TIERS:
        next_min = next((t[2] for t in TIERS if t[2] > min_s), 9999)
        bucket = [c for c in ranked if min_s <= c['score'] < next_min]
        if not bucket:
            continue
        header_range = f"≥{min_s}" if min_s == TIERS[0][2] else f"{min_s}-{next_min}"
        if min_s == 0:
            header_range = f"<{next(t[2] for t in TIERS if t[2] > 0)}"
        lines.append(f"\n### {icon} {label} ({header_range})")
        lines.extend(_fmt_bullet(c) for c in bucket)

    lines += [
        "",
        "---",
        "",
        "_生成自 `scripts/candidate_ranker.py`。"
        "若需业务解读 + 首选/次选分桶建议, 请 Claude 对本报告做 narrative enhancement._",
    ]
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=True, help='YYYY-MM-DD')
    ap.add_argument('--versions', default='ng1.0.1,ng1.0.6',
                    help='Comma-separated NG versions')
    args = ap.parse_args()

    date_str = args.date.replace('-', '')
    versions = [v.strip() for v in args.versions.split(',') if v.strip()]

    picks_by_version: dict[str, dict[str, int]] = {}
    for v in versions:
        report = find_todays_report(v, date_str)
        top = load_top_n(report, TOP_N)
        picks_by_version[v] = {s['stock_code']: rank for rank, s in enumerate(top, 1)}
        print(f"[{v}] loaded Top-{TOP_N} from {report.name}")

    union_codes_raw = sorted(set().union(*picks_by_version.values()))
    # Drop ETF / LOF / REIT codes (1xxxxx, 5xxxxx). 只保留 A 股 (0/3/6/688/8 开头)
    union_codes = [c for c in union_codes_raw
                   if c and not (c[0] in ('1', '5'))]
    dropped = len(union_codes_raw) - len(union_codes)
    if dropped:
        print(f"  dropped {dropped} non-A-share codes (ETF/LOF/REIT)")
    print(f"Union: {len(union_codes)} unique codes across {len(versions)} versions")

    merged_hits: dict[str, list[tuple[str, int, float]]] = {c: [] for c in union_codes}
    seen_dates_per_code: dict[str, set[str]] = {c: set() for c in union_codes}
    for v in versions:
        hits = scan_historical_hits(v, union_codes, max_rank=TOP_N, exclude_date=date_str)
        for c, events in hits.items():
            for ev in events:
                if ev[0] not in seen_dates_per_code[c]:
                    merged_hits[c].append(ev)
                    seen_dates_per_code[c].add(ev[0])

    price, dates_sorted = load_price_series(union_codes)
    fwd = compute_forward_returns(merged_hits, price, dates_sorted)
    snapshots = fetch_snapshot(union_codes)

    ranked = []
    for code in union_codes:
        versions_hit = [v for v, picks in picks_by_version.items() if code in picks]
        is_cross = len(versions_hit) > 1
        source = ' / '.join(f"{v} r{picks_by_version[v][code]}" for v in versions_hit)
        m10_frac, w10, n10 = _mean_win(fwd.get(code, {}).get(10, []))
        snap = snapshots.get(code, {})
        total, det = score_candidate(w10, m10_frac, n10, is_cross, snap)
        ranked.append({
            'code': code, 'source': source, 'is_cross': is_cross,
            'w10': w10, 'm10': m10_frac, 'n10': n10,
            'snap': snap, 'score': total, 'det': det,
        })

    ranked.sort(key=lambda c: -c['score'])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = '_'.join(version_tag(v) for v in versions)
    md_path = OUT_DIR / f'{tag}_{date_str}.md'
    md_path.write_text(render_markdown(args.date, versions, ranked))

    json_path = OUT_DIR / f'{tag}_{date_str}.json'
    json_path.write_text(json.dumps([
        {k: v for k, v in c.items() if k != 'snap'} | {
            'name': c['snap'].get('name'), 'industry': c['snap'].get('industry'),
        }
        for c in ranked
    ], indent=2, ensure_ascii=False))

    print(f"\n✅ Wrote {md_path}")
    print(f"   {json_path}")
    print(f"\nTop 10:")
    for i, c in enumerate(ranked[:10], 1):
        name = c['snap'].get('name') or '?'
        print(f"  {i:>2}. [{tier_for(c['score'])}] {c['code']} {name} "
              f"总分={c['score']:.1f} · 10d={c['w10']:.0f}%/{c['m10']:+.1f}% "
              f"(n={c['n10']}) · {c['source']}")


if __name__ == '__main__':
    main()
