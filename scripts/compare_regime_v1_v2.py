"""Step A validation: compare V11 (regime_v1) vs ng2.0a multi-beta vote (regime_v2).

Outputs:
  1. Regime distribution: % bull / % bear, total trading days
  2. Flip count: regime transitions in 2020-2026
  3. Agreement matrix (V11 × ng2.0a): how often they agree
  4. 2018-2019 sanity check: did ng2.0a identify 2018Q4 bear & 2019Q1 bull rebound?
  5. PASS/ABORT decision per spec section 6.1

Usage: python3 scripts/compare_regime_v1_v2.py
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).resolve().parents[1] / 'data_adapter' / 'stock_data.db'


def load_regimes(start: str, end: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    q = """
        SELECT mrs.trade_date AS date,
               ma.amv_regime AS regime_v1,
               mrs.regime_v2 AS regime_v2,
               mrs.v11_bull, mrs.b1_bull, mrs.b2_bull, mrs.vote_count
        FROM market_regime_signals mrs
        JOIN market_amv ma ON ma.trade_date = mrs.trade_date
        WHERE mrs.trade_date BETWEEN ? AND ?
        ORDER BY mrs.trade_date
    """
    df = pd.read_sql(q, conn, params=(start, end))
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')


def count_flips(regime: pd.Series) -> int:
    return int((regime.diff().fillna(0) != 0).sum())


def regime_distribution(regime: pd.Series) -> dict:
    n = len(regime)
    return {
        'total_days': n,
        'bull_days': int((regime == 1).sum()),
        'bear_days': int((regime == -1).sum()),
        'pct_bull': round(100.0 * (regime == 1).sum() / n, 2) if n else 0,
        'pct_bear': round(100.0 * (regime == -1).sum() / n, 2) if n else 0,
    }


def agreement_matrix(v1: pd.Series, v2: pd.Series) -> dict:
    n = len(v1)
    both_bull = int(((v1 == 1) & (v2 == 1)).sum())
    both_bear = int(((v1 == -1) & (v2 == -1)).sum())
    v1_bull_v2_bear = int(((v1 == 1) & (v2 == -1)).sum())
    v1_bear_v2_bull = int(((v1 == -1) & (v2 == 1)).sum())
    agree = both_bull + both_bear
    return {
        'total': n,
        'agree': agree,
        'agree_pct': round(100.0 * agree / n, 2) if n else 0,
        'both_bull': both_bull,
        'both_bear': both_bear,
        'v1_bull_v2_bear': v1_bull_v2_bear,
        'v1_bear_v2_bull': v1_bear_v2_bull,
    }


def sanity_2018_2019(df_18_19: pd.DataFrame) -> dict:
    """Did v2 identify 2018 bear (Q4 selloff) + 2019 Q1 rebound bull?"""
    q4_2018 = df_18_19.loc['2018-10-01':'2018-12-31']
    q1_2019 = df_18_19.loc['2019-01-01':'2019-03-31']
    return {
        '2018_q4_bear_days_v2': int((q4_2018['regime_v2'] == -1).sum()),
        '2018_q4_total': len(q4_2018),
        '2018_q4_bear_days_v1': int((q4_2018['regime_v1'] == -1).sum()),
        '2019_q1_bull_days_v2': int((q1_2019['regime_v2'] == 1).sum()),
        '2019_q1_total': len(q1_2019),
        '2019_q1_bull_days_v1': int((q1_2019['regime_v1'] == 1).sum()),
    }


def main():
    print('=' * 70)
    print('Step A Validation: regime_v1 (V11) vs regime_v2 (ng2.0a multi-beta vote)')
    print('=' * 70)

    df = load_regimes('2020-01-01', '2026-04-25')
    print(f'\n[Main window 2020-2026: {len(df)} trading days]\n')

    dist_v1 = regime_distribution(df['regime_v1'])
    dist_v2 = regime_distribution(df['regime_v2'])
    print('1. Regime distribution:')
    print(f'   V11 (v1):    bull={dist_v1["pct_bull"]}%  bear={dist_v1["pct_bear"]}%  total={dist_v1["total_days"]}d')
    print(f'   ng2.0a (v2): bull={dist_v2["pct_bull"]}%  bear={dist_v2["pct_bear"]}%  total={dist_v2["total_days"]}d')
    pct_diff = abs(dist_v1['pct_bull'] - dist_v2['pct_bull'])
    print(f'   Δ%bull: {pct_diff:.2f}pp')

    flips_v1 = count_flips(df['regime_v1'])
    flips_v2 = count_flips(df['regime_v2'])
    print('\n2. Flip count (transitions):')
    print(f'   V11 (v1):    {flips_v1} flips')
    print(f'   ng2.0a (v2): {flips_v2} flips')
    flip_ratio = flips_v2 / flips_v1 if flips_v1 else float('inf')
    print(f'   ratio v2/v1: {flip_ratio:.2f}x')

    agree = agreement_matrix(df['regime_v1'], df['regime_v2'])
    print('\n3. Agreement matrix (V11 × ng2.0a):')
    print(f'   agree: {agree["agree"]}/{agree["total"]} = {agree["agree_pct"]}%')
    print(f'   both_bull={agree["both_bull"]}, both_bear={agree["both_bear"]}')
    print(f'   v1_bull_v2_bear={agree["v1_bull_v2_bear"]}, v1_bear_v2_bull={agree["v1_bear_v2_bull"]}')

    df_18_19 = load_regimes('2018-01-01', '2019-12-31')
    sanity = sanity_2018_2019(df_18_19)
    print('\n4. 2018-2019 sanity check:')
    print(f'   2018 Q4 (selloff expected): v2 bear days = {sanity["2018_q4_bear_days_v2"]}/{sanity["2018_q4_total"]} '
          f'(v1: {sanity["2018_q4_bear_days_v1"]})')
    print(f'   2019 Q1 (rebound expected): v2 bull days = {sanity["2019_q1_bull_days_v2"]}/{sanity["2019_q1_total"]} '
          f'(v1: {sanity["2019_q1_bull_days_v1"]})')

    print('\n5. PASS/ABORT decision (spec gates):')
    issues = []
    if flip_ratio > 1.5:
        issues.append(f'ABORT: flip ratio {flip_ratio:.2f}x > 1.5x (whipsaw)')
    if pct_diff > 25:
        issues.append(f'ABORT: %bull diff {pct_diff:.2f}pp > 25pp')
    if agree['agree_pct'] < 50:
        issues.append(f'ABORT: agreement {agree["agree_pct"]}% < 50%')
    if sanity['2018_q4_bear_days_v2'] == 0:
        issues.append('FAIL sanity: ng2.0a missed 2018 Q4 bear')
    if sanity['2019_q1_bull_days_v2'] == 0:
        issues.append('FAIL sanity: ng2.0a missed 2019 Q1 bull rebound')

    if issues:
        print('   STEP A NOT PASSING:')
        for i in issues:
            print(f'     - {i}')
        verdict = 'ABORT'
    else:
        print('   Step A primary gates PASS — proceed to Step B end-to-end backtest')
        verdict = 'PASS'

    report_path = Path('reports') / 'ng2_0a_step_a_report.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(f'# ng2.0a Step A Report\n\n')
        f.write(f'**Date**: 2026-04-25\n\n')
        f.write(f'## Distributions (2020-2026)\n')
        f.write(f'- V11: bull {dist_v1["pct_bull"]}%, bear {dist_v1["pct_bear"]}%\n')
        f.write(f'- ng2.0a: bull {dist_v2["pct_bull"]}%, bear {dist_v2["pct_bear"]}%\n\n')
        f.write(f'## Flips (2020-2026)\n- V11: {flips_v1}\n- ng2.0a: {flips_v2}\n- ratio: {flip_ratio:.2f}x\n\n')
        f.write(f'## Agreement\n- {agree["agree_pct"]}% agree ({agree["agree"]}/{agree["total"]})\n')
        f.write(f'- both_bull={agree["both_bull"]}, both_bear={agree["both_bear"]}\n')
        f.write(f'- v1_bull_v2_bear={agree["v1_bull_v2_bear"]}, v1_bear_v2_bull={agree["v1_bear_v2_bull"]}\n\n')
        f.write(f'## 2018-2019 Sanity\n- 2018 Q4 v2 bear: {sanity["2018_q4_bear_days_v2"]}/{sanity["2018_q4_total"]}\n')
        f.write(f'- 2019 Q1 v2 bull: {sanity["2019_q1_bull_days_v2"]}/{sanity["2019_q1_total"]}\n\n')
        f.write(f'## Verdict: {verdict}\n')
        if issues:
            for i in issues:
                f.write(f'- {i}\n')
    print(f'\nReport saved to {report_path}')

    if issues:
        sys.exit(1)


if __name__ == '__main__':
    main()
