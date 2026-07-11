#!/usr/bin/env python3
"""Post-process ng2.1 merged report dir to apply L1-L5 risk overlay.

Applied transforms (per regime, read from market_regime_signals):
  L1: score floor 30 — drop stocks with rank_score < 30
  L2: industry cap — bull=3, bear=2 (per regime)
  L4: crisis hard-stop — bear regime + B2_RV_pct ≥ 90% + hs300 chg ≤ -3%
       → top_n → 5 (achieved by demoting #6+ via composite=0)

The reports preserve original ordering and full all_stocks_with_scores list,
but we re-rank by overlay logic so V5.2 eval picks correctly.

Usage:
  python3 scripts/ng21_apply_overlay_to_reports.py \\
      --src reports/daily_selection_ng21_2020_2026_merged \\
      --dst reports/daily_selection_ng21_2020_2026_overlay
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DB = PROJECT / 'data_adapter' / 'stock_data.db'

L1_SCORE_FLOOR = 30
INDUSTRY_CAP = {'bull': 3, 'bear': 2}
CRISIS_TOP_N = 5
CRISIS_RV_PCT = 0.90
CRISIS_INDEX_DROP = -0.03


def load_regime_full(start: str, end: str) -> dict:
    """trade_date → (regime_label, b2_rv_pct, hs300_chg)."""
    conn = sqlite3.connect(str(DB), timeout=30)
    conn.execute('PRAGMA busy_timeout=30000')

    # Regime + b2_rv_pct
    rows = conn.execute(
        'SELECT trade_date, regime_v2, b2_rv_percentile_252 FROM market_regime_signals '
        'WHERE regime_v2 IS NOT NULL AND trade_date >= ? AND trade_date <= ?',
        (start, end),
    ).fetchall()

    out = {}
    for d, r, rv in rows:
        out[d] = {'regime': 'bull' if r == 1 else 'bear',
                  'b2_rv_pct': float(rv) if rv is not None else 0.0,
                  'hs300_chg': 0.0}

    # 沪深300 daily change
    chg_rows = conn.execute(
        '''SELECT trade_date, (close - prev_close) / prev_close
           FROM (SELECT dq.close, LAG(dq.close) OVER (ORDER BY dq.trade_date) AS prev_close,
                        dq.trade_date
                 FROM daily_quotes dq JOIN securities s ON s.id = dq.security_id
                 WHERE s.code = '000300.SH')
           WHERE trade_date >= ? AND trade_date <= ?''',
        (start, end),
    ).fetchall()
    for d, chg in chg_rows:
        if d in out:
            out[d]['hs300_chg'] = float(chg) if chg is not None else 0.0
    conn.close()
    return out


def crisis_active(meta: dict) -> bool:
    return (meta['regime'] == 'bear'
            and meta['b2_rv_pct'] >= CRISIS_RV_PCT
            and meta['hs300_chg'] <= CRISIS_INDEX_DROP)


def apply_overlay(stocks: list, regime: str, crisis: bool) -> list:
    """Return new list with overlay applied. Preserves original order
    in all_stocks_with_scores; demotes overlay-failed stocks by zeroing rank_score
    so V5.2 eval naturally drops them.
    """
    cap = INDUSTRY_CAP[regime]
    top_n_target = CRISIS_TOP_N if crisis else 10

    # Sort current rank_score descending
    sorted_stocks = sorted(
        stocks,
        key=lambda s: float(s.get('rank_score', 0) or 0),
        reverse=True,
    )

    # Track per-industry count and survivors
    industry_count: dict[str, int] = defaultdict(int)
    survivors_codes: set[str] = set()
    n_kept = 0

    # Detect WF fold preds (placeholder score=50): skip L1 floor for those.
    use_l1_floor = any(float(s.get('score', 0) or 0) != 50.0 for s in sorted_stocks[:5])

    for s in sorted_stocks:
        if use_l1_floor:
            l1_score = float(s.get('score', 0) or 0)
            if l1_score < L1_SCORE_FLOOR:
                continue  # L1 drop
        # else: WF fold preds, score is placeholder, skip L1 — rank_score sort handles it
        ind = s.get('industry') or 'UNKNOWN'
        if industry_count[ind] >= cap:
            continue  # L2 drop
        industry_count[ind] += 1
        survivors_codes.add(s.get('stock_code') or s.get('code') or '')
        n_kept += 1
        if n_kept >= top_n_target:
            break

    # Build new list: keep original order; for non-survivors, set rank_score=0
    new = []
    for s in stocks:
        code = s.get('stock_code') or s.get('code') or ''
        if code in survivors_codes:
            new.append(s)
        else:
            s2 = dict(s)
            # Zero out scores so V5.2 eval doesn't pick them
            s2['rank_score'] = 0.0
            s2['composite'] = 0.0
            s2['score'] = 0.0
            new.append(s2)
    return new


def process(src_dir: Path, dst_dir: Path, regime_meta: dict) -> dict:
    dst_dir.mkdir(parents=True, exist_ok=True)
    counts = {'total': 0, 'bull': 0, 'bear': 0, 'crisis': 0, 'no_regime': 0}
    for f in sorted(src_dir.glob('analysis_data_*.json')):
        ymd = f.stem.replace('analysis_data_', '')
        iso = f'{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}'
        meta = regime_meta.get(iso)
        counts['total'] += 1
        if meta is None:
            shutil.copy2(f, dst_dir / f.name)
            counts['no_regime'] += 1
            continue
        with open(f, 'r') as fh:
            d = json.load(fh)
        stocks = d.get('all_stocks_with_scores', [])
        is_crisis = crisis_active(meta)
        new_stocks = apply_overlay(stocks, meta['regime'], is_crisis)
        d['all_stocks_with_scores'] = new_stocks
        d['_ng21_overlay'] = {
            'regime': meta['regime'],
            'crisis': is_crisis,
            'b2_rv_pct': meta['b2_rv_pct'],
            'hs300_chg': meta['hs300_chg'],
            'industry_cap': INDUSTRY_CAP[meta['regime']],
            'top_n_target': CRISIS_TOP_N if is_crisis else 10,
        }
        with open(dst_dir / f.name, 'w') as fh:
            json.dump(d, fh)
        counts[meta['regime']] += 1
        if is_crisis:
            counts['crisis'] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--start', default='2020-01-01')
    ap.add_argument('--end', default='2026-04-30')
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        print(f'ERR: {src} missing', file=sys.stderr)
        sys.exit(2)

    meta = load_regime_full(args.start, args.end)
    print(f'Loaded regime meta for {len(meta)} dates')

    counts = process(src, dst, meta)
    print(f'Processed: total={counts["total"]}, bull={counts["bull"]}, '
          f'bear={counts["bear"]}, crisis_days={counts["crisis"]}, '
          f'no_regime={counts["no_regime"]}')


if __name__ == '__main__':
    main()
