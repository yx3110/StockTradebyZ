#!/usr/bin/env python3
"""Single-stock review: historical backtest + tech/fundamental snapshot for a day's top picks.

Usage:
    python3 scripts/single_stock_review.py --date 2026-04-14 --version ng1.0.1 --top-n 10

Output:
    reports/single_stock_review/{version}_{date}.md
    reports/single_stock_review/{version}_{date}.json
"""
from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'
OUT_DIR = PROJECT_ROOT / 'reports' / 'single_stock_review'
HORIZONS = (3, 5, 10, 15)


def version_tag(version: str) -> str:
    """Normalize ng1.0.1 → ng101 (for dir glob + model file search)."""
    return version.replace('.', '').replace('_', '')


# Versions whose report dir diverges from the default tag (MOE variants, etc).
# tomorrow_stock_selector.py writes ng1.0.62 → daily_selection_ng106v2[_fullmarket].
# ng2.0a writes daily_selection_ng2_0a[_fullmarket] (underscore preserved in dir name).
VERSION_DIR_ALIASES: dict[str, list[str]] = {
    'ng1.0.62': ['ng106v2'],
    'ng1.0.6': ['ng106'],
    'ng2.0a': ['ng2_0a'],
}


def _candidate_tags(version: str) -> list[str]:
    tags = [version_tag(version)]
    for alias in VERSION_DIR_ALIASES.get(version, []):
        if alias not in tags:
            tags.append(alias)
    return tags


def find_todays_report(version: str, date_str: str) -> Path:
    """Locate the day's report JSON. date_str: YYYYMMDD."""
    candidates: list[Path] = []
    for tag in _candidate_tags(version):
        candidates.extend([
            PROJECT_ROOT / f'reports/daily_selection_{tag}_fullmarket/analysis_data_{date_str}.json',
            PROJECT_ROOT / f'reports/daily_selection_{tag}/analysis_data_{date_str}.json',
        ])
    candidates.extend([
        PROJECT_ROOT / f'reports/daily_selection_{version}_fast/analysis_data_{date_str}.json',
        PROJECT_ROOT / f'reports/daily_selection_{version}_fullmarket/analysis_data_{date_str}.json',
    ])
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No report for {version} on {date_str}. Tried:\n  " + '\n  '.join(str(c) for c in candidates)
    )


def load_top_n(report_path: Path, top_n: int) -> list[dict]:
    with open(report_path) as f:
        data = json.load(f)
    stocks = data.get('all_stocks_with_scores', [])
    if not stocks:
        raise ValueError(f"{report_path} has no all_stocks_with_scores")

    def score_key(s):
        return float(s.get('composite') or s.get('rank_score') or 0)

    stocks_sorted = sorted(stocks, key=score_key, reverse=True)
    return stocks_sorted[:top_n]


def scan_historical_hits(
    version: str, target_codes: list[str], max_rank: int, exclude_date: str
) -> dict[str, list[tuple[str, int, float]]]:
    """Scan historical reports once; return all hits with rank <= max_rank per code.

    Callers filter by rank threshold as needed. Returns dict[code] -> [(date, rank, pred_10d)].
    """
    search_dirs: list[Path] = []
    for tag in _candidate_tags(version):
        search_dirs.append(PROJECT_ROOT / f'reports/daily_selection_{tag}')
        search_dirs.append(PROJECT_ROOT / f'reports/daily_selection_{tag}_fullmarket')
    search_dirs.extend([
        PROJECT_ROOT / f'reports/daily_selection_{version}_fast',
        PROJECT_ROOT / f'reports/daily_selection_{version}_fullmarket',
    ])
    targets = set(target_codes)
    hits: dict[str, list[tuple[str, int, float]]] = defaultdict(list)

    seen_dates: set[str] = set()
    for d in search_dirs:
        if not d.exists():
            continue
        for fpath in sorted(d.glob('analysis_data_*.json')):
            date_str = fpath.stem.replace('analysis_data_', '')
            if date_str == exclude_date or date_str in seen_dates:
                continue
            seen_dates.add(date_str)
            try:
                with open(fpath) as f:
                    data = json.load(f)
            except Exception:
                continue
            stocks = data.get('all_stocks_with_scores', [])
            for rank, s in enumerate(stocks[:max_rank], 1):
                code = s.get('stock_code')
                if code in targets:
                    hits[code].append((date_str, rank, float(s.get('pred_10d') or 0)))
    return hits


def filter_hits_by_rank(
    hits: dict[str, list[tuple[str, int, float]]], max_rank: int
) -> dict[str, list[tuple[str, int, float]]]:
    return {c: [t for t in h if t[1] <= max_rank] for c, h in hits.items()}


def load_price_series(
    codes: list[str],
) -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
    """Load adj_close (fallback close) time series per code. Returns (price, dates_sorted)."""
    if not codes:
        return {}, {}
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute('PRAGMA busy_timeout=30000')
    placeholders = ','.join('?' * len(codes))
    q = f"""
    SELECT s.code AS code, d.trade_date, d.adj_close, d.close
    FROM daily_quotes d
    JOIN securities s ON s.id = d.security_id
    WHERE s.code IN ({placeholders})
    ORDER BY s.code, d.trade_date
    """
    rows = conn.execute(q, codes).fetchall()
    conn.close()

    dates_sorted: dict[str, list[str]] = defaultdict(list)
    price: dict[str, dict[str, float]] = defaultdict(dict)
    for code, td, adj_close, close in rows:
        td_str = td.strftime('%Y%m%d') if hasattr(td, 'strftime') else str(td).replace('-', '')
        px = adj_close if adj_close is not None else close
        if px is None:
            continue
        price[code][td_str] = float(px)
        dates_sorted[code].append(td_str)
    for c in dates_sorted:
        dates_sorted[c].sort()
    return price, dates_sorted


def compute_forward_returns(
    hits: dict[str, list[tuple[str, int, float]]],
    price: dict[str, dict[str, float]],
    dates_sorted: dict[str, list[str]],
) -> dict[str, dict[int, list[float]]]:
    """For each hit (code, date), compute forward adj_close returns over HORIZONS. T+1 entry."""
    out: dict[str, dict[int, list[float]]] = {}
    for code, events in hits.items():
        ds = dates_sorted.get(code, [])
        if not ds:
            continue
        fwd: dict[int, list[float]] = {h: [] for h in HORIZONS}
        for date_str, _rank, _p10 in events:
            i = bisect.bisect_right(ds, date_str)
            if i >= len(ds):
                continue
            entry_px = price[code].get(ds[i])
            if not entry_px:
                continue
            for h in HORIZONS:
                j = i + h
                if j >= len(ds):
                    continue
                exit_px = price[code].get(ds[j])
                if exit_px:
                    fwd[h].append(exit_px / entry_px - 1)
        out[code] = fwd
    return out


def fetch_snapshot(codes: list[str]) -> dict[str, dict[str, Any]]:
    """Latest tech + fundamental snapshot per code."""
    if not codes:
        return {}
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute('PRAGMA busy_timeout=30000')
    conn.row_factory = sqlite3.Row
    out: dict[str, dict[str, Any]] = {}
    for code in codes:
        q = """
        SELECT s.code, s.name, s.industry, s.area, s.list_date,
               d.trade_date, d.close, d.price_change_pct, d.high, d.low,
               d.ma5, d.ma10, d.ma20, d.ma60,
               t.kdj_k, t.kdj_d, t.kdj_j, t.rsi6, t.rsi12, t.rsi24,
               t.macd_dif, t.macd_dea, t.macd_macd,
               t.bbi, t.atr_14, t.volume_ratio,
               t.boll_upper, t.boll_lower, t.zhixing_short_trend,
               b.pe_ttm, b.pb, b.ps_ttm, b.total_mv, b.circ_mv,
               b.turnover_rate, b.dv_ratio
        FROM securities s
        LEFT JOIN daily_quotes d ON d.security_id=s.id
          AND d.trade_date=(SELECT MAX(trade_date) FROM daily_quotes WHERE security_id=s.id)
        LEFT JOIN technical_indicators t ON t.security_id=s.id AND t.trade_date=d.trade_date
        LEFT JOIN daily_basic b ON b.security_id=s.id AND b.trade_date=d.trade_date
        WHERE s.code=? AND s.type='A股'
        """
        r = conn.execute(q, (code,)).fetchone()
        if r:
            out[code] = dict(r)
        # 20-day range
        hi_lo = conn.execute(
            """
            SELECT MAX(high) hi, MIN(low) lo FROM daily_quotes d
            JOIN securities s ON s.id=d.security_id
            WHERE s.code=? AND d.trade_date >= date(
                (SELECT MAX(trade_date) FROM daily_quotes d2
                 JOIN securities s2 ON s2.id=d2.security_id WHERE s2.code=?),
                '-30 days'
            )
            """,
            (code, code),
        ).fetchone()
        if r and hi_lo and hi_lo[0] and hi_lo[1] and out[code].get('close'):
            lo, hi = float(hi_lo[1]), float(hi_lo[0])
            close = float(out[code]['close'])
            out[code]['range20_low'] = lo
            out[code]['range20_high'] = hi
            out[code]['range20_pos_pct'] = (close - lo) / (hi - lo) * 100 if hi > lo else 0
    conn.close()
    return out


def _fmt(v, dec=2, suffix=''):
    if v is None:
        return '-'
    try:
        return f'{float(v):.{dec}f}{suffix}'
    except (ValueError, TypeError):
        return '-'


def _mean_win(arr: list[float]) -> tuple[float, float, int]:
    if not arr:
        return 0.0, 0.0, 0
    mean = sum(arr) / len(arr) * 100
    win = sum(1 for x in arr if x > 0) / len(arr) * 100
    return mean, win, len(arr)


def build_backtest_row(code: str, name: str, fwd: dict[int, list[float]]) -> str:
    parts = [f'| {code} | {name} | {len(fwd.get(HORIZONS[0], []))} |']
    for h in HORIZONS:
        mean, win, n = _mean_win(fwd.get(h, []))
        parts.append(f' {mean:+.2f}% ({win:.0f}%) |' if n else ' - |')
    return ''.join(parts)


def auto_comment(snap: dict) -> str:
    """Rule-based one-line commentary on the tech/fundamental snapshot."""
    notes = []
    close = float(snap.get('close') or 0)
    ma20 = float(snap.get('ma20') or 0)
    ma60 = float(snap.get('ma60') or 0)
    j = float(snap.get('kdj_j') or 0)
    rsi6 = float(snap.get('rsi6') or 0)
    hist = float(snap.get('macd_macd') or 0)
    pb = float(snap.get('pb') or 0)
    pos = float(snap.get('range20_pos_pct') or 50)

    if close and ma60 and close > ma60:
        notes.append('站上MA60')
    elif close and ma60:
        notes.append(f'MA60下{(close/ma60-1)*100:+.1f}%')
    if j >= 90 or rsi6 >= 85:
        notes.append('KDJ/RSI严重超买')
    elif j >= 80 or rsi6 >= 75:
        notes.append('短期偏热')
    if hist > 0:
        notes.append('MACD hist翻红')
    if pb and pb < 1.5:
        notes.append(f'PB={pb:.2f}低估')
    if pos < 35:
        notes.append('区间低位')
    elif pos > 70:
        notes.append('区间高位')
    return '；'.join(notes) if notes else '-'


def build_markdown(
    date_str_h: str, version: str, top_n: list[dict],
    fwd_top10: dict, fwd_top50: dict, snapshots: dict,
) -> str:
    date_display = f'{date_str_h[:4]}-{date_str_h[4:6]}-{date_str_h[6:]}'
    lines = [
        f'# 📊 {version} 重点股回顾 · {date_display}',
        '',
        f'_top-{len(top_n)} 当日推荐的历史表现回测 + 技术面/基本面快照_',
        '',
        '---',
        '',
        '## Part 1 · 当日 Top 推荐',
        '',
        f'| # | 代码 | 名称 | 行业 | 收盘 | comp | pred_10d | pred_15d | ATR% | R:R |',
        f'|--|--|--|--|--|--|--|--|--|--|',
    ]
    for i, s in enumerate(top_n, 1):
        lines.append(
            f"| {i} | {s.get('stock_code','')} | {s.get('stock_name','')} | "
            f"{s.get('industry','') or '-'} | "
            f"{_fmt(s.get('close_price'))} | "
            f"{_fmt(s.get('composite'), 3)} | "
            f"{float(s.get('pred_10d') or 0)*100:+.2f}% | "
            f"{float(s.get('pred_15d') or 0)*100:+.2f}% | "
            f"{float(s.get('atr_pct') or 0)*100:.1f}% | "
            f"{_fmt(s.get('risk_reward_ratio'))} |"
        )

    lines += [
        '',
        '---',
        '',
        '## Part 2 · 历史 Top10 入选后前向收益回测',
        '',
        '_扫描全部历史报告（不含今日），每次入选 Top10 后按次日 adj_close 入场，前向 N 日计算收益。_',
        '',
        '| 代码 | 名称 | 样本 | 3d mean(win%) | 5d | 10d | 15d |',
        '|--|--|--|--|--|--|--|',
    ]
    agg = {h: [] for h in HORIZONS}
    for s in top_n:
        code = s.get('stock_code')
        name = s.get('stock_name', '')
        fwd = fwd_top10.get(code, {h: [] for h in HORIZONS})
        for h in HORIZONS:
            agg[h].extend(fwd.get(h, []))
        lines.append(build_backtest_row(code, name, fwd))
    agg_parts = []
    for h in HORIZONS:
        mean, win, n = _mean_win(agg[h])
        agg_parts.append(f' {mean:+.2f}% ({win:.0f}%) |' if n else ' - |')
    lines.append(
        f"| **汇总** | **ALL** | **{len(agg[HORIZONS[0]])}** |" + ''.join(agg_parts)
    )

    lines += [
        '',
        '### 扩展 Top50 样本',
        '',
        '_当 Top10 样本不足，用 Top50 估计更稳健的前向收益统计。_',
        '',
        '| 代码 | 名称 | 样本 | 3d mean(win%) | 5d | 10d | 15d |',
        '|--|--|--|--|--|--|--|',
    ]
    for s in top_n:
        code = s.get('stock_code')
        name = s.get('stock_name', '')
        fwd = fwd_top50.get(code, {h: [] for h in HORIZONS})
        lines.append(build_backtest_row(code, name, fwd))

    lines += [
        '',
        '---',
        '',
        '## Part 3 · 技术面 / 基本面快照',
        '',
    ]
    for s in top_n:
        code = s.get('stock_code')
        snap = snapshots.get(code, {})
        name = snap.get('name') or s.get('stock_name', '')
        industry = snap.get('industry') or '-'
        area = snap.get('area') or '-'
        close = snap.get('close') or 0
        chg = float(snap.get('price_change_pct') or 0) * 100
        tmv = (snap.get('total_mv') or 0) / 10000
        cmv = (snap.get('circ_mv') or 0) / 10000
        ma20 = snap.get('ma20') or 0
        ma60 = snap.get('ma60') or 0
        bias_20 = (float(close) / float(ma20) - 1) * 100 if close and ma20 else 0
        bias_60 = (float(close) / float(ma60) - 1) * 100 if close and ma60 else 0
        comment = auto_comment(snap)

        lines.append(f'### {code} {name} — {industry} ({area})')
        lines.append('')
        lines.append(
            f'**报价** {_fmt(close)}（{chg:+.2f}%）'
            f'  H/L {_fmt(snap.get("high"))}/{_fmt(snap.get("low"))}'
            f'  ATR14={_fmt(snap.get("atr_14"))}  量比={_fmt(snap.get("volume_ratio"))}'
        )
        lines.append('')
        lines.append(
            f'**均线** MA5 {_fmt(snap.get("ma5"))} · MA10 {_fmt(snap.get("ma10"))} · '
            f'MA20 {_fmt(ma20)} ({bias_20:+.1f}%) · MA60 {_fmt(ma60)} ({bias_60:+.1f}%) · '
            f'BBI {_fmt(snap.get("bbi"))}'
        )
        lines.append('')
        lines.append(
            f'**动量** KDJ K/D/J = {_fmt(snap.get("kdj_k"),1)}/{_fmt(snap.get("kdj_d"),1)}/'
            f'{_fmt(snap.get("kdj_j"),1)}'
            f' · RSI6/12/24 = {_fmt(snap.get("rsi6"),1)}/{_fmt(snap.get("rsi12"),1)}/'
            f'{_fmt(snap.get("rsi24"),1)}'
            f' · MACD DIF {_fmt(snap.get("macd_dif"),3)} / DEA {_fmt(snap.get("macd_dea"),3)} / '
            f'hist {_fmt(snap.get("macd_macd"),3)}'
        )
        lines.append('')
        r20_lo = snap.get('range20_low')
        r20_hi = snap.get('range20_high')
        r20_pos = snap.get('range20_pos_pct')
        if r20_lo and r20_hi:
            lines.append(
                f'**区间** 20日 {_fmt(r20_lo)} ~ {_fmt(r20_hi)}，'
                f'当前位置 {_fmt(r20_pos,0)}%'
            )
            lines.append('')
        lines.append(
            f'**估值** PE(TTM)={snap.get("pe_ttm") or "-"}'
            f' · PB={_fmt(snap.get("pb"))}'
            f' · PS={_fmt(snap.get("ps_ttm"))}'
            f' · 股息率={_fmt(snap.get("dv_ratio"))}%'
            f' · 总市值 {tmv:.1f}亿 / 流通 {cmv:.1f}亿'
            f' · 换手 {_fmt(snap.get("turnover_rate"))}%'
        )
        lines.append('')
        lines.append(f'**自动点评**：{comment}')
        lines.append('')

    lines += [
        '---',
        '',
        '_生成自 `scripts/single_stock_review.py`。若需业务解读和组合级建议，请交由 Claude 对本报告做 narrative enhancement。_',
    ]
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=True, help='YYYY-MM-DD')
    ap.add_argument('--version', required=True, help='e.g. ng1.0.1, ng1.0.6')
    ap.add_argument('--top-n', type=int, default=10)
    ap.add_argument('--output-dir', default=str(OUT_DIR))
    args = ap.parse_args()

    date_str = args.date.replace('-', '')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[1/5] locating report for {args.version} on {date_str}...')
    report_path = find_todays_report(args.version, date_str)
    print(f'       found: {report_path}')

    print(f'[2/5] extracting top {args.top_n}...')
    top_n = load_top_n(report_path, args.top_n)
    codes = [s['stock_code'] for s in top_n]
    print(f'       codes: {codes}')

    print('[3/5] scanning historical top50 hits (one pass)...')
    hits_top50 = scan_historical_hits(args.version, codes, max_rank=50, exclude_date=date_str)
    hits_top10 = filter_hits_by_rank(hits_top50, max_rank=10)
    for code in codes:
        print(f'       {code}: top10={len(hits_top10.get(code, []))} top50={len(hits_top50.get(code, []))}')

    print('[4/5] computing forward returns (shared price cache)...')
    price, dates_sorted = load_price_series(codes)
    fwd_top10 = compute_forward_returns(hits_top10, price, dates_sorted)
    fwd_top50 = compute_forward_returns(hits_top50, price, dates_sorted)

    print('[5/5] fetching tech+fundamental snapshots...')
    snapshots = fetch_snapshot(codes)

    md = build_markdown(date_str, args.version, top_n, fwd_top10, fwd_top50, snapshots)
    md_path = out_dir / f'{args.version}_{date_str}.md'
    md_path.write_text(md, encoding='utf-8')

    json_payload = {
        'date': args.date, 'version': args.version, 'top_n': args.top_n,
        'source_report': str(report_path),
        'top_picks': top_n,
        'backtest_top10': {c: {str(h): v for h, v in d.items()} for c, d in fwd_top10.items()},
        'backtest_top50': {c: {str(h): v for h, v in d.items()} for c, d in fwd_top50.items()},
        'snapshots': {c: {k: (str(v) if hasattr(v, 'isoformat') else v) for k, v in s.items()}
                      for c, s in snapshots.items()},
        'hits_top10': {c: [list(t) for t in h] for c, h in hits_top10.items()},
        'hits_top50': {c: [list(t) for t in h] for c, h in hits_top50.items()},
    }
    json_path = out_dir / f'{args.version}_{date_str}.json'
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2, default=str))

    print(f'\n✅ wrote:\n  {md_path}\n  {json_path}')


if __name__ == '__main__':
    main()
