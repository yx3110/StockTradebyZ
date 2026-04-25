"""Compute Top-10 daily overlap between ng2.0a v2 (multi-beta vote) and v1 baseline (V11)."""
import json
from pathlib import Path

import pandas as pd

V2_DIR = Path('reports/daily_selection_regime_switch_v2')
V1_DIR = Path('reports/daily_selection_regime_switch_v1')


def load_top10(report_dir: Path, date_compact: str) -> set:
    # Try both common filename patterns
    for fname in (f'analysis_data_{date_compact}.json', f'选股报告_{date_compact}.json'):
        p = report_dir / fname
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            stocks = data.get('all_stocks_with_scores', [])[:10]
            return {s.get('stock_code') or s.get('code') for s in stocks
                    if s.get('stock_code') or s.get('code')}
    return set()


def main():
    if not V1_DIR.exists() or not V2_DIR.exists():
        raise SystemExit(f'Missing dirs: V1={V1_DIR.exists()}, V2={V2_DIR.exists()}')

    files_v2 = sorted(V2_DIR.glob('analysis_data_*.json'))
    overlaps = []
    for fp in files_v2:
        date_compact = fp.stem.replace('analysis_data_', '')
        if date_compact < '20200101' or date_compact > '20260425':
            continue
        s_v1 = load_top10(V1_DIR, date_compact)
        s_v2 = load_top10(V2_DIR, date_compact)
        if not s_v1 or not s_v2:
            continue
        overlap_pct = 100.0 * len(s_v1 & s_v2) / 10.0
        overlaps.append((date_compact, overlap_pct))

    if not overlaps:
        raise SystemExit('No comparable dates found')

    df = pd.DataFrame(overlaps, columns=['date', 'overlap_pct'])
    avg = df['overlap_pct'].mean()
    median = df['overlap_pct'].median()
    pct_below_50 = 100.0 * (df['overlap_pct'] < 50).sum() / len(df)
    print(f'Top-10 overlap (ng2.0a v2 vs v1 baseline):')
    print(f'  comparable dates: {len(df)}')
    print(f'  mean overlap: {avg:.2f}%')
    print(f'  median overlap: {median:.2f}%')
    print(f'  % days with overlap < 50%: {pct_below_50:.2f}%')
    if avg < 50:
        print('  GATE: avg overlap < 50% (list drift) — flag for review')
    else:
        print('  GATE: Top-10 overlap PASS')


if __name__ == '__main__':
    main()
