#!/usr/bin/env python3
"""ng2.1 Phase 0 诊断: 每个 ng1.0.1 feature 的 bull-only vs bear-only IC.

判断 regime specialist 路线在数学上是否可能赢:
  - regime_divergence = |bull_IC - bear_IC| / max(|bull_IC|, |bear_IC|, 0.01)
  - sign_flip: bull_IC 和 bear_IC 符号是否相反 (强 divergence 信号)

输出按 |bull_IC - bear_IC| 排序, 找出"在哪个 regime 强信号 / 哪个 regime 弱信号"的因子。
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = Path(__file__).resolve().parents[1]
DB = PROJECT / 'data_adapter' / 'stock_data.db'

START = '2020-01-02'
END = '2026-04-24'

LABEL_COLS = ['label_5d', 'label_10d', 'label_15d']


def load_data() -> pd.DataFrame:
    """Load ng101_feature_cache + V11 regime."""
    conn = sqlite3.connect(str(DB), timeout=60)
    print(f'Loading ng101_feature_cache {START}~{END} ...')
    df = pd.read_sql(
        f"""SELECT code, trade_date, features_json,
                   label_5d, label_10d, label_15d,
                   market_return_5d, market_return_20d, market_volatility_20d,
                   market_breadth, market_new_high_ratio, northbound_flow_5d,
                   market_volume_ratio, market_drawdown, vix_proxy, market_momentum_diff
            FROM ng101_feature_cache
            WHERE trade_date >= ? AND trade_date <= ? AND label_10d IS NOT NULL
            ORDER BY trade_date, code""",
        conn, params=(START, END),
    )
    print(f'  rows: {len(df):,}, dates: {df["trade_date"].nunique()}')

    print('Loading V11 regime ...')
    reg = pd.read_sql(
        'SELECT trade_date, regime_v2 FROM market_regime_signals '
        'WHERE regime_v2 IS NOT NULL AND trade_date >= ? AND trade_date <= ?',
        conn, params=(START, END),
    )
    conn.close()
    df = df.merge(reg, on='trade_date', how='left')
    n_bull = (df['regime_v2'] == 1).sum()
    n_bear = (df['regime_v2'] == -1).sum()
    n_other = df['regime_v2'].isna().sum()
    print(f'  bull rows={n_bull:,}, bear rows={n_bear:,}, no-regime={n_other:,}')

    print('Parsing features_json ...')
    parsed = pd.DataFrame(df['features_json'].apply(json.loads).tolist())
    df = pd.concat([df.drop(columns=['features_json']).reset_index(drop=True),
                    parsed.reset_index(drop=True)], axis=1)
    print(f'  parsed columns: {parsed.shape[1]} stock features')
    return df


def compute_daily_ic_per_feature(df: pd.DataFrame, feat: str, label: str) -> tuple[pd.Series, pd.Series]:
    """Spearman IC per trade_date for given feature×label, separated by regime.

    Returns: (bull_daily_ic, bear_daily_ic)
    """
    if feat not in df.columns:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    sub = df[[feat, label, 'trade_date', 'regime_v2']].dropna()
    if len(sub) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    def _daily_ic(g: pd.DataFrame) -> float:
        if len(g) < 50:
            return np.nan
        x = g[feat].values
        y = g[label].values
        if np.std(x) < 1e-9 or np.std(y) < 1e-9:
            return np.nan
        return stats.spearmanr(x, y).correlation

    bull = sub[sub['regime_v2'] == 1].groupby('trade_date').apply(_daily_ic).dropna()
    bear = sub[sub['regime_v2'] == -1].groupby('trade_date').apply(_daily_ic).dropna()
    return bull, bear


def main():
    df = load_data()

    feature_cols = [
        c for c in df.columns
        if c not in ('code', 'trade_date', 'regime_v2',
                     'label_5d', 'label_10d', 'label_15d',
                     'market_return_5d', 'market_return_20d', 'market_volatility_20d',
                     'market_breadth', 'market_new_high_ratio', 'northbound_flow_5d',
                     'market_volume_ratio', 'market_drawdown', 'vix_proxy',
                     'market_momentum_diff')
    ]
    print(f'\nAnalyzing {len(feature_cols)} stock features × {len(LABEL_COLS)} labels ...')

    rows = []
    for label in LABEL_COLS:
        for feat in feature_cols:
            bull_ic, bear_ic = compute_daily_ic_per_feature(df, feat, label)
            if len(bull_ic) < 30 or len(bear_ic) < 30:
                continue
            bull_mean = bull_ic.mean()
            bear_mean = bear_ic.mean()
            bull_std = bull_ic.std()
            bear_std = bear_ic.std()
            bull_icir = bull_mean / bull_std if bull_std > 1e-9 else np.nan
            bear_icir = bear_mean / bear_std if bear_std > 1e-9 else np.nan
            denom = max(abs(bull_mean), abs(bear_mean), 0.005)
            divergence = abs(bull_mean - bear_mean) / denom
            sign_flip = (bull_mean * bear_mean) < 0
            rows.append({
                'feature': feat, 'label': label,
                'bull_IC': bull_mean, 'bear_IC': bear_mean,
                'bull_ICIR': bull_icir, 'bear_ICIR': bear_icir,
                'bull_n_days': len(bull_ic), 'bear_n_days': len(bear_ic),
                '|Δ|': abs(bull_mean - bear_mean),
                'divergence': divergence,
                'sign_flip': sign_flip,
            })

    res = pd.DataFrame(rows)
    if res.empty:
        print('NO RESULTS — check data load')
        return

    print('\n=' * 1, '=' * 80)
    print('Phase 0: Regime Divergence Analysis')
    print('=' * 80)

    for label in LABEL_COLS:
        sub = res[res['label'] == label].sort_values('|Δ|', ascending=False).reset_index(drop=True)
        if sub.empty:
            continue
        print(f'\n--- label={label} ---')
        print(f'Top 20 by |bull_IC - bear_IC|:')
        print(sub.head(20)[['feature', 'bull_IC', 'bear_IC', 'bull_ICIR', 'bear_ICIR', '|Δ|', 'divergence', 'sign_flip']].to_string(index=False, float_format=lambda x: f'{x:+.4f}' if isinstance(x, float) else str(x)))

        # Categorize
        strong_bull = sub[(sub['bull_IC'].abs() > 0.025) & (sub['divergence'] > 0.5)]
        strong_bear = sub[(sub['bear_IC'].abs() > 0.025) & (sub['divergence'] > 0.5)]
        sign_flips = sub[sub['sign_flip']]
        universal = sub[(sub['bull_IC'].abs() > 0.02) & (sub['bear_IC'].abs() > 0.02) & (sub['divergence'] < 0.3)]

        print(f'\n{label} 分类统计:')
        print(f'  Strong bull-favored (|bull_IC|>0.025, divergence>50%): {len(strong_bull)}')
        print(f'  Strong bear-favored (|bear_IC|>0.025, divergence>50%): {len(strong_bear)}')
        print(f'  Sign-flip features (bull_IC×bear_IC<0): {len(sign_flips)}')
        print(f'  Universal (both |IC|>0.02, divergence<30%): {len(universal)}')

    # Output to CSV
    out = PROJECT / 'reports' / 'ng21_phase0_regime_divergence.csv'
    res.to_csv(out, index=False)
    print(f'\nFull results saved: {out}')

    # Summary verdict for label_10d
    sub10 = res[res['label'] == 'label_10d']
    n_strong = ((sub10['bull_IC'].abs() > 0.025) & (sub10['divergence'] > 0.5)).sum() + \
               ((sub10['bear_IC'].abs() > 0.025) & (sub10['divergence'] > 0.5)).sum()
    n_flips = sub10['sign_flip'].sum()
    print('\n' + '=' * 80)
    print('VERDICT (10d label):')
    print(f'  strong regime-divergent features: {n_strong}')
    print(f'  sign-flip features: {n_flips}')
    if n_strong >= 10:
        print('  ✅ specialist 路线数学上成立 → 进 Phase 1-4')
    elif n_strong >= 5:
        print('  ⚠️ 边界情况 → 直接 Phase 3 (用现有 66 + alt-data 30, 跳过 alpha158)')
    else:
        print('  ❌ specialist 路线在当前 feat 上数学不成立')
    print('=' * 80)


if __name__ == '__main__':
    main()
