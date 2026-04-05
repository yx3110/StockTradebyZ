#!/usr/bin/env python3
"""快速诊断V5残差标签下各特征IC — 解析features_json版本"""
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import sqlite3
import json
from scipy.stats import spearmanr

DB_PATH = 'data_adapter/stock_data.db'

def main():
    conn = sqlite3.connect(DB_PATH, timeout=30)

    # 只加载最近300天（平衡速度和覆盖）
    print("加载最近300天特征(解析JSON)...")
    raw = pd.read_sql("""
        SELECT code, trade_date, features_json, label_10d,
               market_return_20d, market_volatility_20d,
               market_momentum_20d, market_momentum_5d
        FROM v39_feature_cache
        WHERE trade_date >= '2025-01-01'
          AND label_10d IS NOT NULL
          AND features_json IS NOT NULL
        ORDER BY trade_date, code
    """, conn)
    conn.close()
    print(f"  {len(raw):,} 样本, {raw['trade_date'].nunique()} 交易日")

    # 解析features_json
    print("解析features_json...")
    parsed = raw['features_json'].apply(json.loads)
    feat_df = pd.DataFrame(parsed.tolist())
    feat_df['code'] = raw['code'].values
    feat_df['trade_date'] = raw['trade_date'].values
    feat_df['label_10d'] = raw['label_10d'].values

    # 加入宏观特征
    for col in ['market_return_20d', 'market_volatility_20d', 'market_momentum_20d', 'market_momentum_5d']:
        if col in raw.columns:
            feat_df[col] = raw[col].values

    # 特征列表（排除meta列）
    exclude = {'code', 'trade_date', 'label_10d', 'label_3d', 'label_5d', 'label_15d'}
    feature_cols = [c for c in feat_df.columns if c not in exclude]
    print(f"  {len(feature_cols)} 个特征")

    # 计算截面IC
    print(f"\n计算截面IC (vs label_10d)...")
    dates = sorted(feat_df['trade_date'].unique())
    print(f"  {len(dates)} 交易日")

    results = []
    for feat in feature_cols:
        daily_ics = []
        for d in dates:
            sub = feat_df.loc[feat_df['trade_date'] == d, [feat, 'label_10d']].dropna()
            if len(sub) < 100:
                continue
            # 检查是否常数列
            if sub[feat].std() < 1e-10:
                continue
            ic, _ = spearmanr(sub[feat], sub['label_10d'])
            if not np.isnan(ic):
                daily_ics.append(ic)

        if len(daily_ics) < 20:
            results.append({'feature': feat, 'ic': 0, 'icir': 0, 'n': len(daily_ics), 'ic_pos': 0,
                           'ic_early': 0, 'ic_late': 0})
            continue

        mean_ic = np.mean(daily_ics)
        std_ic = np.std(daily_ics)
        mid = len(daily_ics) // 2
        results.append({
            'feature': feat,
            'ic': mean_ic,
            'icir': mean_ic / std_ic if std_ic > 1e-8 else 0,
            'n': len(daily_ics),
            'ic_pos': np.mean([x > 0 for x in daily_ics]) * 100,
            'ic_early': np.mean(daily_ics[:mid]) if mid > 0 else 0,
            'ic_late': np.mean(daily_ics[mid:]) if mid > 0 else 0,
        })

    rdf = pd.DataFrame(results).sort_values('icir', ascending=False)

    # 输出
    print(f"\n{'='*95}")
    print(f"{'特征':<35} {'IC':>8} {'ICIR':>7} {'IC>0%':>6} {'IC前半':>8} {'IC后半':>8} {'衰减':>6} {'判定':>6}")
    print(f"{'='*95}")

    strong, weak, harmful = [], [], []
    for _, r in rdf.iterrows():
        feat = r['feature']
        icir = r['icir']
        decay = r['ic_late'] - r['ic_early']

        if r['n'] < 20:
            verdict = "🗑️ 无数据"
            weak.append(feat)
        elif icir > 0.20:
            verdict = "✅ 强"
            strong.append(feat)
        elif icir > 0.05:
            verdict = "⚠️ 弱+"
            weak.append(feat)
        elif icir > -0.05:
            verdict = "🗑️ 噪声"
            weak.append(feat)
        elif icir > -0.15:
            verdict = "⚠️ 弱-"
            harmful.append(feat)
        else:
            verdict = "❌ 有害"
            harmful.append(feat)
        print(f"{feat:<35} {r['ic']:>8.4f} {icir:>7.3f} {r['ic_pos']:>5.1f}% {r['ic_early']:>8.4f} {r['ic_late']:>8.4f} {decay:>+6.3f} {verdict}")

    print(f"\n{'='*95}")
    print(f"汇总: ✅强={len(strong)} ⚠️弱={len(weak)} ❌有害={len(harmful)}")

    print(f"\n✅ 强特征 保留 ({len(strong)}):")
    for f in strong:
        r = rdf[rdf['feature']==f].iloc[0]
        print(f"  {f}: IC={r['ic']:.4f} ICIR={r['icir']:.3f}")

    print(f"\n❌ 有害/弱负特征 建议删除 ({len(harmful)}):")
    for f in harmful:
        r = rdf[rdf['feature']==f].iloc[0]
        print(f"  {f}: IC={r['ic']:.4f} ICIR={r['icir']:.3f}")

    print(f"\n🗑️ 噪声特征 可删 ({len(weak)}):")
    for f in weak:
        r = rdf[rdf['feature']==f].iloc[0]
        print(f"  {f}: IC={r['ic']:.4f} ICIR={r['icir']:.3f}")

if __name__ == '__main__':
    main()
