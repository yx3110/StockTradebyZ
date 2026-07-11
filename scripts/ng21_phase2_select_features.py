#!/usr/bin/env python3
"""ng2.1 v2 Phase 2: 用 Phase 0 IC 数据做 per-regime feature selection.

从 ng21_phase0_regime_divergence.csv 读 64 stock features 的 bull/bear IC,
为 bull 和 bear specialist 各选 ~50 个最适合的 stock feature.

输出:
  ml_models/ng21/bull_features.json
  ml_models/ng21/bear_features.json
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
CSV = PROJECT / 'reports' / 'ng21_phase0_regime_divergence.csv'

# Bull specialist 主标签 15d (拉长抓 trend)
BULL_LABEL = 'label_15d'
# Bear specialist 主标签 5d (短期避雷)
BEAR_LABEL = 'label_5d'

# 每个 specialist 选多少 stock feature
TARGET_N = 50
# Universal threshold (跨 regime IC 都超过这个的算 universal, 强制保留在两个 specialist 里)
UNIVERSAL_IC_THRESHOLD = 0.018


def select_for_regime(df: pd.DataFrame, target: str, label: str, n: int) -> list[str]:
    """target: 'bull' or 'bear'. label: label_5d/10d/15d.

    Score = |target_IC| × bonus(divergence, sign_flip).
    Score 高 = 在 target regime 上 alpha 强且与 off-regime 有差异.
    """
    sub = df[df['label'] == label].copy()
    target_ic = f'{target}_IC'
    other_ic = 'bear_IC' if target == 'bull' else 'bull_IC'

    def score_row(r):
        ti = r[target_ic]
        oi = r[other_ic]
        base = abs(ti)
        # Divergence bonus: 越分化越好
        div_bonus = 1.0 + min(r['divergence'], 2.0) * 0.3
        # Sign-flip bonus: target IC 方向正确且 off-regime 反向 = 强 specialist 信号
        if r['sign_flip']:
            div_bonus *= 1.5
        # Penalize if off-regime IC has same direction AND larger magnitude (说明这是 off-regime 的因子)
        if abs(oi) > abs(ti) and (ti * oi) > 0:
            div_bonus *= 0.3
        return base * div_bonus

    sub['select_score'] = sub.apply(score_row, axis=1)
    sub = sub.sort_values('select_score', ascending=False)

    # 优先 select_score 排序 → 取前 n
    selected = sub.head(n)['feature'].tolist()
    return selected


def main():
    df = pd.read_csv(CSV)
    print(f'Loaded {len(df)} rows from {CSV}')

    bull_feats = select_for_regime(df, 'bull', BULL_LABEL, TARGET_N)
    bear_feats = select_for_regime(df, 'bear', BEAR_LABEL, TARGET_N)

    # 强制保留 universal features (跨 regime 都强) 在 BOTH 列表里
    df10 = df[df['label'] == 'label_10d']
    universal = df10[
        (df10['bull_IC'].abs() > UNIVERSAL_IC_THRESHOLD)
        & (df10['bear_IC'].abs() > UNIVERSAL_IC_THRESHOLD)
        & (df10['divergence'] < 0.3)
    ]['feature'].tolist()
    print(f'Universal features (force-include in both): {len(universal)}')
    print(f'  {universal}')

    # 合并 (去重)
    bull_final = list(dict.fromkeys(bull_feats + universal))[:TARGET_N + 5]
    bear_final = list(dict.fromkeys(bear_feats + universal))[:TARGET_N + 5]

    overlap = set(bull_final) & set(bear_final)
    only_bull = set(bull_final) - overlap
    only_bear = set(bear_final) - overlap

    print(f'\n=== Bull specialist features ({len(bull_final)}) ===')
    print(f'  Top 10: {bull_final[:10]}')
    print(f'\n=== Bear specialist features ({len(bear_final)}) ===')
    print(f'  Top 10: {bear_final[:10]}')
    print(f'\n=== Overlap ({len(overlap)}, will be in both) ===')
    print(f'  {sorted(overlap)[:15]} ...')
    print(f'\n=== Only in bull ({len(only_bull)}) ===')
    print(f'  {sorted(only_bull)}')
    print(f'\n=== Only in bear ({len(only_bear)}) ===')
    print(f'  {sorted(only_bear)}')

    out_dir = PROJECT / 'ml_models' / 'ng21'
    out_dir.mkdir(parents=True, exist_ok=True)

    bull_out = out_dir / 'bull_features.json'
    bear_out = out_dir / 'bear_features.json'

    with open(bull_out, 'w') as f:
        json.dump({
            'specialist': 'ng2.1.v2-bull',
            'label_horizon': BULL_LABEL,
            'n_features': len(bull_final),
            'features': bull_final,
            'overlap_with_bear': sorted(overlap),
            'unique_to_bull': sorted(only_bull),
        }, f, indent=2, ensure_ascii=False)
    with open(bear_out, 'w') as f:
        json.dump({
            'specialist': 'ng2.1.v2-bear',
            'label_horizon': BEAR_LABEL,
            'n_features': len(bear_final),
            'features': bear_final,
            'overlap_with_bull': sorted(overlap),
            'unique_to_bear': sorted(only_bear),
        }, f, indent=2, ensure_ascii=False)

    print(f'\nSaved: {bull_out}, {bear_out}')


if __name__ == '__main__':
    main()
