#!/usr/bin/env python3
"""ng2.1 Stage 4 evaluation helper: regime-merge + V5.2 eval.

Reads V11 regime from market_regime_signals (baseline calibration), copies
ng2.1-bull report to merged dir for bull-regime days and ng2.1-bear for
bear-regime days, then runs run_north_star_eval.py for V5.2 scoring.

Usage:
  python3 scripts/ng21_merge_and_eval.py \\
      --bull-dir reports/daily_selection_ng21_bull_2020_2026 \\
      --bear-dir reports/daily_selection_ng21_bear_2020_2026 \\
      --merged-dir reports/daily_selection_ng21_2020_2026_merged \\
      --start 2020-01-02 --end 2026-04-24 \\
      --label "ng2.1-WF-OOS"
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DB = PROJECT / 'data_adapter' / 'stock_data.db'


def load_regime_map(start: str, end: str) -> dict:
    conn = sqlite3.connect(str(DB), timeout=30)
    conn.execute('PRAGMA busy_timeout=30000')
    rows = conn.execute(
        'SELECT trade_date, regime_v2 FROM market_regime_signals '
        'WHERE regime_v2 IS NOT NULL AND trade_date >= ? AND trade_date <= ?',
        (start, end),
    ).fetchall()
    conn.close()
    return {d: int(r) for d, r in rows}


def merge(bull_dir: Path, bear_dir: Path, merged: Path,
          regime: dict) -> dict:
    merged.mkdir(parents=True, exist_ok=True)
    n_bull = n_bear = n_skip = 0
    for f in sorted(bull_dir.glob('analysis_data_*.json')):
        ymd = f.stem.replace('analysis_data_', '')
        iso = f'{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}'
        r = regime.get(iso)
        if r == 1:
            shutil.copy2(f, merged / f.name)
            n_bull += 1
        elif r == -1:
            bear_f = bear_dir / f.name
            if bear_f.exists():
                shutil.copy2(bear_f, merged / f.name)
                n_bear += 1
            else:
                n_skip += 1
        else:
            n_skip += 1
    return {'bull': n_bull, 'bear': n_bear, 'skip': n_skip,
            'total': len(list(merged.glob('*.json')))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bull-dir', required=True)
    ap.add_argument('--bear-dir', required=True)
    ap.add_argument('--merged-dir', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--label', default='ng2.1')
    ap.add_argument('--top-n', type=int, default=10)
    ap.add_argument('--focus-days', type=int, default=10)
    ap.add_argument('--rank-field', default='composite')
    ap.add_argument('--skip-merge', action='store_true',
                    help='Skip merge step, only run eval on existing dir')
    args = ap.parse_args()

    bull_dir = Path(args.bull_dir)
    bear_dir = Path(args.bear_dir)
    merged = Path(args.merged_dir)

    if not args.skip_merge:
        if not bull_dir.exists() or not bear_dir.exists():
            print(f'ERR: bull/bear dir missing', file=sys.stderr)
            sys.exit(2)
        regime = load_regime_map(args.start, args.end)
        bull_n = sum(1 for r in regime.values() if r == 1)
        bear_n = sum(1 for r in regime.values() if r == -1)
        print(f'Regime days {args.start}~{args.end}: bull={bull_n}, bear={bear_n}, total={len(regime)}')
        stats = merge(bull_dir, bear_dir, merged, regime)
        print(f'Merged: bull-regime files={stats["bull"]}, bear-regime files={stats["bear"]}, '
              f'skipped={stats["skip"]}, total in dir={stats["total"]}')

    # Run V5.2 eval
    cmd = [
        sys.executable, str(PROJECT / 'backtest' / 'run_north_star_eval.py'),
        '--backtest',
        '--report-dir', str(merged),
        '--label', args.label,
        '--top-n', str(args.top_n),
        '--focus-days', str(args.focus_days),
        '--rank-field', args.rank_field,
        '--start-date', args.start,
        '--end-date', args.end,
    ]
    print(f'\n→ Running: {" ".join(cmd)}\n')
    subprocess.run(cmd, check=False)


if __name__ == '__main__':
    main()
