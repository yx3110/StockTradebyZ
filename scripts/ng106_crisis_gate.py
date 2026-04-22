"""3-expert gate with crisis overlay (bull/bear/crisis).

Gate:
  crisis: 000001.SH rolling 120d max-drawdown < -8%  (exogenous threshold)
  bull:   not crisis AND amv_regime == +1
  bear:   not crisis AND amv_regime == -1

Crisis expert priority over AMV regime (stress conditions dominate).

Usage:
  python3 scripts/ng106_crisis_gate.py \
    --bull-dir reports/daily_selection_ng1.0.7_fast \
    --bear-dir reports/daily_selection_ng104_ensemble_3seed \
    --crisis-dir reports/daily_selection_ng1.0.1_fast \
    --start-date 2024-01-01 --end-date 2026-04-17 \
    --out reports/ng106_crisis_test
"""
import argparse
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB = REPO / 'data_adapter' / 'stock_data.db'


def load_regime_crisis() -> dict:
    """trade_date -> 'bull' | 'bear' | 'crisis'  (crisis has priority)"""
    with sqlite3.connect(str(DB), timeout=30) as conn:
        amv_rows = conn.execute(
            'SELECT trade_date, amv_regime FROM market_amv'
        ).fetchall()
        crisis_rows = conn.execute(
            'SELECT trade_date, is_crisis FROM crisis_regime'
        ).fetchall()
    crisis_map = {str(d)[:10]: bool(c) for d, c in crisis_rows}
    out = {}
    for d, regime in amv_rows:
        date_s = str(d)[:10]
        if crisis_map.get(date_s, False):
            out[date_s] = 'crisis'
        elif regime == 1:
            out[date_s] = 'bull'
        elif regime == -1:
            out[date_s] = 'bear'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bull-dir', required=True)
    ap.add_argument('--bear-dir', required=True)
    ap.add_argument('--crisis-dir', required=True)
    ap.add_argument('--start-date', default='2024-01-01')
    ap.add_argument('--end-date', default='2026-04-17')
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', default='ng106-crisis')
    args = ap.parse_args()

    regime = load_regime_crisis()
    n_bull = sum(1 for v in regime.values() if v == 'bull')
    n_bear = sum(1 for v in regime.values() if v == 'bear')
    n_cri = sum(1 for v in regime.values() if v == 'crisis')
    print(f'Regime: bull={n_bull}, bear={n_bear}, crisis={n_cri}')

    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    src_for = {
        'bull': Path(args.bull_dir),
        'bear': Path(args.bear_dir),
        'crisis': Path(args.crisis_dir),
    }

    copied = {'bull': 0, 'bear': 0, 'crisis': 0}
    for date_s, tag in regime.items():
        if date_s < args.start_date or date_s > args.end_date:
            continue
        date_compact = date_s.replace('-', '')
        src = src_for[tag] / f'analysis_data_{date_compact}.json'
        if src.exists():
            shutil.copy2(str(src), str(out_dir / f'analysis_data_{date_compact}.json'))
            copied[tag] += 1
    print(f'Copied: bull={copied["bull"]}, bear={copied["bear"]}, crisis={copied["crisis"]}')

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
    import re
    block = res.stdout.split('V5.2')[-1] if 'V5.2' in res.stdout else res.stdout
    m = re.search(r'原始总分:\s*([\d.]+)\s*/\s*[\d.]+\s*\(未加权([\d.]+)%\)', block)
    if m:
        print(f'V5.2 raw: {m.group(1)}/295 ({m.group(2)}%)')
    else:
        print('could not parse V5.2')
        print(res.stdout[-1500:])


if __name__ == '__main__':
    main()
