#!/usr/bin/env python3
"""ng2.1 v2 Phase 4: 拼装 system-level WF-OOS reports + V5.2 eval.

输入: bull/bear specialist 各自的 fold OOS predictions (dir 来自 trainer --wf-report-dir).
输出: reports/{label}_system_oos/ 含按 V11 regime 拼接的 analysis_data_*.json,
     可被 backtest/run_north_star_eval.py 直接评估.

用法:
  python3 scripts/ng21v2_phase4_system_eval.py \\
      --bull-dir reports/ng21v2_bull_wf_oos \\
      --bear-dir reports/ng21v2_bear_wf_oos \\
      --merged-dir reports/ng21v2_system_wfoos \\
      --label "ng2.1v2-system-WFOOS"

  # baseline:
  python3 scripts/ng21v2_phase4_system_eval.py \\
      --bull-dir reports/ng101_baseline_wf_oos \\
      --bear-dir reports/ng104_baseline_wf_oos \\
      --merged-dir reports/ng2.0a_system_wfoos \\
      --label "ng2.0a-baseline-WFOOS"
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


def load_regime_map() -> dict:
    conn = sqlite3.connect(str(DB), timeout=30)
    conn.execute('PRAGMA busy_timeout=30000')
    rows = conn.execute(
        'SELECT trade_date, regime_v2 FROM market_regime_signals '
        'WHERE regime_v2 IS NOT NULL'
    ).fetchall()
    conn.close()
    return {str(d): int(r) for d, r in rows}


def merge_by_regime(bull_dir: Path, bear_dir: Path, merged: Path,
                    regime: dict) -> dict:
    """For each date with a fold OOS prediction, copy the regime-matching specialist's report."""
    merged.mkdir(parents=True, exist_ok=True)
    n_bull = n_bear = n_no_match = 0
    available_bull_files = {f.name for f in bull_dir.glob('analysis_data_*.json')}
    available_bear_files = {f.name for f in bear_dir.glob('analysis_data_*.json')}
    all_dates = available_bull_files | available_bear_files

    for fname in sorted(all_dates):
        ymd = fname.replace('analysis_data_', '').replace('.json', '')
        iso = f'{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}'
        r = regime.get(iso)
        if r == 1 and fname in available_bull_files:
            shutil.copy2(bull_dir / fname, merged / fname)
            n_bull += 1
        elif r == -1 and fname in available_bear_files:
            shutil.copy2(bear_dir / fname, merged / fname)
            n_bear += 1
        else:
            n_no_match += 1
    return {'bull': n_bull, 'bear': n_bear, 'no_match': n_no_match,
            'total': len(list(merged.glob('*.json')))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bull-dir', required=True)
    ap.add_argument('--bear-dir', required=True)
    ap.add_argument('--merged-dir', required=True)
    ap.add_argument('--label', default='ng2.1v2')
    ap.add_argument('--top-n', type=int, default=10)
    ap.add_argument('--focus-days', type=int, default=10)
    ap.add_argument('--rank-field', default='composite')
    ap.add_argument('--start-date', default='auto')
    ap.add_argument('--end-date', default='auto')
    ap.add_argument('--skip-merge', action='store_true')
    args = ap.parse_args()

    bull_dir = Path(args.bull_dir)
    bear_dir = Path(args.bear_dir)
    merged = Path(args.merged_dir)

    if not args.skip_merge:
        if not bull_dir.exists() or not bear_dir.exists():
            print(f'ERR: bull/bear dir missing', file=sys.stderr)
            sys.exit(2)
        regime = load_regime_map()
        print(f'V11 regime map: {len(regime)} dates loaded')
        stats = merge_by_regime(bull_dir, bear_dir, merged, regime)
        print(f'Merged: bull-regime={stats["bull"]}, bear-regime={stats["bear"]}, '
              f'no-match={stats["no_match"]}, total in dir={stats["total"]}')
        if stats['total'] == 0:
            print('No reports merged — nothing to evaluate', file=sys.stderr)
            sys.exit(3)

    # Run V5.2 eval
    cmd = [
        sys.executable, str(PROJECT / 'backtest' / 'run_north_star_eval.py'),
        '--backtest',
        '--report-dir', str(merged),
        '--label', args.label,
        '--top-n', str(args.top_n),
        '--focus-days', str(args.focus_days),
        '--rank-field', args.rank_field,
    ]
    if args.start_date != 'auto':
        cmd += ['--start-date', args.start_date]
    if args.end_date != 'auto':
        cmd += ['--end-date', args.end_date]
    print(f'\n→ {" ".join(cmd)}\n')
    subprocess.run(cmd, check=False)


if __name__ == '__main__':
    main()
