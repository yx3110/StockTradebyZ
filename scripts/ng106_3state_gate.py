"""3-state regime switch backtest (bull/sideways/bear) for ng1.0.6 v2.

Refines binary AMV gate to 3-state:
  bull:     amv_regime == +1 and |amv_macd| >= SIDEWAYS_THRESH
  bear:     amv_regime == -1 and |amv_macd| >= SIDEWAYS_THRESH
  sideways: |amv_macd| < SIDEWAYS_THRESH (ambiguous/transition days)

Usage:
  python3 scripts/ng106_3state_gate.py \
    --bull-dir reports/daily_selection_ng1.0.7_fast \
    --bear-dir reports/daily_selection_ng104_ensemble_3seed \
    --side-dir reports/daily_selection_ng1.0.1_fast \
    --start-date 2024-01-01 --end-date 2026-04-17 \
    --sideways-thresh 0.5 --out reports/ng106_3state_test
"""
import argparse
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB = REPO / 'data_adapter' / 'stock_data.db'


def load_regime_3state(thresh: float) -> dict:
    """trade_date -> 'bull' | 'bear' | 'sideways'"""
    with sqlite3.connect(str(DB), timeout=30) as conn:
        rows = conn.execute(
            'SELECT trade_date, amv_regime, amv_macd FROM market_amv'
        ).fetchall()
    out = {}
    for dt, regime, macd in rows:
        date_s = str(dt)[:10] if isinstance(dt, str) else dt.strftime('%Y-%m-%d')
        if macd is None:
            continue
        if abs(macd) < thresh:
            out[date_s] = 'sideways'
        elif regime == 1:
            out[date_s] = 'bull'
        elif regime == -1:
            out[date_s] = 'bear'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bull-dir', required=True)
    ap.add_argument('--bear-dir', required=True)
    ap.add_argument('--side-dir', required=True)
    ap.add_argument('--start-date', default='2024-01-01')
    ap.add_argument('--end-date', default='2026-04-17')
    ap.add_argument('--sideways-thresh', type=float, default=0.5)
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', default='ng106-3state')
    args = ap.parse_args()

    regime = load_regime_3state(args.sideways_thresh)
    n_bull = sum(1 for v in regime.values() if v == 'bull')
    n_bear = sum(1 for v in regime.values() if v == 'bear')
    n_side = sum(1 for v in regime.values() if v == 'sideways')
    print(f'Regime 3-state (|macd|<{args.sideways_thresh}): '
          f'bull={n_bull}, bear={n_bear}, sideways={n_side}')

    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    src_for = {
        'bull': Path(args.bull_dir),
        'bear': Path(args.bear_dir),
        'sideways': Path(args.side_dir),
    }

    copied = {'bull': 0, 'bear': 0, 'sideways': 0}
    for date_s, tag in regime.items():
        if date_s < args.start_date or date_s > args.end_date:
            continue
        date_compact = date_s.replace('-', '')
        src = src_for[tag] / f'analysis_data_{date_compact}.json'
        if src.exists():
            shutil.copy2(str(src), str(out_dir / f'analysis_data_{date_compact}.json'))
            copied[tag] += 1
    print(f'Copied: bull={copied["bull"]}, bear={copied["bear"]}, sideways={copied["sideways"]}')

    # North star V5.2
    print(f'\nRunning north_star V5.2...')
    cmd = [
        sys.executable, 'backtest/run_north_star_eval.py',
        '--backtest', '--report-dir', str(out_dir),
        '--label', args.label,
        '--top-n', '10', '--focus-days', '10',
        '--rank-field', 'composite', '--score-version', 'v52',
        '--start-date', args.start_date, '--end-date', args.end_date,
    ]
    res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        raise SystemExit(res.returncode)
    # Grab V5.2 score
    import re
    m = re.search(r'原始总分:\s*([\d.]+)\s*/\s*[\d.]+\s*\(未加权([\d.]+)%\)',
                  res.stdout.split('V5.2')[-1] if 'V5.2' in res.stdout else res.stdout)
    if m:
        print(f'V5.2 raw: {m.group(1)}/295 ({m.group(2)}%)')
    else:
        print('could not parse V5.2')
        print(res.stdout[-2000:])


if __name__ == '__main__':
    main()
