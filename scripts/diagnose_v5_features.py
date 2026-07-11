#!/usr/bin/env python3
"""诊断V5残差标签下各特征的预测力

输出每个特征对label_10d残差的截面IC均值/ICIR，用于裁剪决策。
同时分析WF窗口2/3 IC为负的根因。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import sqlite3
from scipy.stats import spearmanr
from backtest.factor_returns import load_or_build_factors

DB_PATH = 'data_adapter/stock_data.db'

def load_v5_data():
    """加载V5训练数据（含残差标签）"""
    conn = sqlite3.connect(DB_PATH, timeout=30)

    # 加载v39 feature cache
    print("加载特征缓存...")
    df = pd.read_sql("""
        SELECT * FROM v39_feature_cache
        WHERE trade_date >= '2020-01-01' AND trade_date <= '2026-04-01'
          AND label_10d IS NOT NULL
    """, conn)
    conn.close()

    print(f"  {len(df):,} 样本, {df['trade_date'].nunique()} 交易日")
    return df

def residualize_labels(df):
    """对标签做因子残差化（简化版：截面回归）"""
    print("\n因子残差化标签...")
    factors = load_or_build_factors('2019-06-01', '2026-04-01', DB_PATH)
    factors.index = factors.index.astype(str).str[:10]

    # 加载个股日收益用于beta估计
    conn = sqlite3.connect(DB_PATH, timeout=30)
    stock_ret = pd.read_sql("""
        SELECT s.code, dq.trade_date, CAST(dq.price_change_pct AS REAL) AS ret
        FROM daily_quotes dq JOIN securities s ON dq.security_id = s.id
        WHERE s.type = 'A股' AND dq.trade_date >= '2019-06-01' AND dq.trade_date <= '2026-04-01'
          AND dq.volume > 0
        ORDER BY s.code, dq.trade_date
    """, conn)
    conn.close()
    stock_ret['trade_date'] = stock_ret['trade_date'].astype(str).str[:10]
    stock_ret['ret'] = pd.to_numeric(stock_ret['ret'], errors='coerce')
    stock_ret = stock_ret.dropna(subset=['ret'])

    # pivot
    pivot = stock_ret.pivot_table(index='trade_date', columns='code', values='ret', aggfunc='first')
    common_dates = pivot.index.intersection(factors.index)
    pivot = pivot.loc[common_dates]
    factors_aligned = factors.loc[common_dates]

    # 估计每只股票rolling 120天beta
    print("  估计因子beta (rolling 120天)...")
    codes = df['code'].unique()
    code_betas = {}
    factor_cols = ['MKT', 'SMB', 'HML', 'UMD']

    for code in codes:
        if code not in pivot.columns:
            continue
        series = pivot[code].dropna()
        if len(series) < 60:
            continue
        recent = series.tail(120)
        f_recent = factors_aligned.loc[recent.index, factor_cols]
        valid = f_recent.notna().all(axis=1) & recent.notna()
        if valid.sum() < 60:
            continue
        y = recent[valid].values
        X = f_recent[valid].values
        X_c = np.column_stack([np.ones(len(y)), X])
        try:
            betas = np.linalg.lstsq(X_c, y, rcond=None)[0]
            code_betas[code] = betas[1:]
        except Exception:
            continue

    print(f"  {len(code_betas)} 只股票有beta")

    if code_betas:
        all_b = np.array(list(code_betas.values()))
        median_beta = np.median(all_b, axis=0)
    else:
        median_beta = np.zeros(4)

    # 残差化label_10d
    dates_list = sorted(factors.index.tolist())
    date_idx = {d: i for i, d in enumerate(dates_list)}

    for label_col in ['label_3d', 'label_5d', 'label_10d', 'label_15d']:
        N = int(label_col.split('_')[1].replace('d', ''))
        factor_cum = {}
        for d in df['trade_date'].unique():
            d_str = str(d)[:10]
            if d_str not in date_idx:
                continue
            idx = date_idx[d_str]
            start = idx + 1
            end = min(idx + N + 1, len(dates_list))
            if start >= len(dates_list):
                continue
            future = dates_list[start:end]
            if not future:
                continue
            factor_cum[d] = factors.loc[future, factor_cols].sum().values

        count = 0
        for i in df.index:
            code = df.at[i, 'code']
            date = df.at[i, 'trade_date']
            beta = code_betas.get(code, median_beta)
            cum_f = factor_cum.get(date)
            if cum_f is None:
                continue
            expected = np.dot(beta, cum_f)
            df.at[i, label_col] = df.at[i, label_col] - expected
            count += 1
        print(f"  {label_col}: 残差化 {count:,} 样本")

    return df

def compute_feature_ic(df, label_col='label_10d'):
    """计算每个特征对残差标签的截面IC"""
    exclude = ['code', 'trade_date', 'label_3d', 'label_5d', 'label_10d', 'label_15d',
               'is_limit_up', 'is_limit_down']
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'float32', 'int64']]

    print(f"\n计算 {len(feature_cols)} 个特征 vs {label_col} 的截面IC...")

    dates = sorted(df['trade_date'].unique())
    # 用最近500天（更贴近OOS期间）
    recent_dates = dates[-500:]

    results = []
    for feat in feature_cols:
        daily_ics = []
        for d in recent_dates:
            mask = df['trade_date'] == d
            sub = df.loc[mask, [feat, label_col]].dropna()
            if len(sub) < 50:
                continue
            ic, _ = spearmanr(sub[feat], sub[label_col])
            if not np.isnan(ic):
                daily_ics.append(ic)

        if len(daily_ics) < 30:
            results.append({'feature': feat, 'ic': 0, 'icir': 0, 'ic_std': 0, 'n_days': 0})
            continue

        mean_ic = np.mean(daily_ics)
        ic_std = np.std(daily_ics)
        icir = mean_ic / ic_std if ic_std > 1e-8 else 0
        results.append({
            'feature': feat,
            'ic': mean_ic,
            'icir': icir,
            'ic_std': ic_std,
            'n_days': len(daily_ics),
            'ic_positive_pct': np.mean([ic > 0 for ic in daily_ics]) * 100,
        })

    result_df = pd.DataFrame(results).sort_values('icir', ascending=False)
    return result_df

def analyze_temporal_stability(df, label_col='label_10d'):
    """分析特征IC的时间稳定性（前半 vs 后半）"""
    exclude = ['code', 'trade_date', 'label_3d', 'label_5d', 'label_10d', 'label_15d',
               'is_limit_up', 'is_limit_down']
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'float32', 'int64']]

    dates = sorted(df['trade_date'].unique())
    mid = len(dates) // 2
    early_dates = set(dates[:mid])
    late_dates = set(dates[mid:])

    print(f"\n时间稳定性分析: 前半({dates[0]}~{dates[mid-1]}) vs 后半({dates[mid]}~{dates[-1]})")

    results = []
    for feat in feature_cols:
        for period, period_dates in [('early', early_dates), ('late', late_dates)]:
            daily_ics = []
            for d in period_dates:
                mask = df['trade_date'] == d
                sub = df.loc[mask, [feat, label_col]].dropna()
                if len(sub) < 50:
                    continue
                ic, _ = spearmanr(sub[feat], sub[label_col])
                if not np.isnan(ic):
                    daily_ics.append(ic)

            if daily_ics:
                results.append({
                    'feature': feat,
                    'period': period,
                    'ic': np.mean(daily_ics),
                    'icir': np.mean(daily_ics) / (np.std(daily_ics) + 1e-8),
                })

    return pd.DataFrame(results)

def main():
    df = load_v5_data()
    df = residualize_labels(df)

    # 1. 特征IC排名
    ic_df = compute_feature_ic(df, 'label_10d')

    print("\n" + "=" * 80)
    print("特征IC排名 (对残差label_10d, 最近500天)")
    print("=" * 80)
    print(f"\n{'特征':<35} {'IC':>8} {'ICIR':>8} {'IC>0%':>7} {'判定':>6}")
    print("-" * 70)

    strong = []
    weak = []
    harmful = []

    for _, row in ic_df.iterrows():
        feat = row['feature']
        ic = row['ic']
        icir = row['icir']
        ic_pos = row.get('ic_positive_pct', 0)

        if abs(icir) < 0.05:
            verdict = "🗑️ 删"
            weak.append(feat)
        elif icir < -0.15:
            verdict = "❌ 有害"
            harmful.append(feat)
        elif icir > 0.15:
            verdict = "✅ 保留"
            strong.append(feat)
        else:
            verdict = "⚠️ 弱"
            weak.append(feat)

        print(f"{feat:<35} {ic:>8.4f} {icir:>8.3f} {ic_pos:>6.1f}% {verdict}")

    print(f"\n汇总: ✅保留={len(strong)}, ⚠️弱/🗑️删={len(weak)}, ❌有害={len(harmful)}")
    print(f"\n建议删除的特征 ({len(weak)+len(harmful)}个):")
    for f in harmful:
        print(f"  ❌ {f} (负ICIR, 有害)")
    for f in weak:
        print(f"  🗑️ {f} (ICIR≈0, 无用)")

    # 2. 时间稳定性
    stab_df = analyze_temporal_stability(df, 'label_10d')
    if not stab_df.empty:
        pivot_stab = stab_df.pivot(index='feature', columns='period', values='icir')
        if 'early' in pivot_stab.columns and 'late' in pivot_stab.columns:
            pivot_stab['decay'] = pivot_stab['late'] - pivot_stab['early']
            pivot_stab = pivot_stab.sort_values('decay')

            print("\n" + "=" * 80)
            print("时间衰减最严重的特征 (ICIR early→late)")
            print("=" * 80)
            for feat in pivot_stab.head(15).index:
                row = pivot_stab.loc[feat]
                print(f"  {feat:<35} early={row['early']:>7.3f} → late={row['late']:>7.3f}  衰减={row['decay']:>7.3f}")

            print("\n时间最稳定的特征:")
            stable = pivot_stab[pivot_stab['late'] > 0.10].sort_values('late', ascending=False)
            for feat in stable.head(15).index:
                row = pivot_stab.loc[feat]
                print(f"  {feat:<35} early={row['early']:>7.3f} → late={row['late']:>7.3f}  ✅稳定")

if __name__ == '__main__':
    main()
