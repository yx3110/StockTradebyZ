#!/usr/bin/env python3
"""
V4.9.0.2 推荐阈值校准脚本 (复用V4901方法: grid_search_PF_weighted)

从 reports/daily_selection_v4902/ 加载batch JSON报告,
匹配实际收益, 网格搜索最优 strong_buy/buy/cautious 阈值。
"""

import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REPORT_DIR = PROJECT_ROOT / 'reports' / 'daily_selection_v4902'
MODEL_DIR = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v4902'
END_DATE = '2026-03-16'  # 留15个交易日给forward return


def phase1_load_and_match():
    """加载报告 + 匹配实际收益"""
    print("=" * 70)
    print("Phase 1: 加载报告 + 匹配实际10d收益")
    print("=" * 70)

    from backtest.backtest_report_based import batch_get_all_future_returns

    json_files = sorted(REPORT_DIR.glob('analysis_data_*.json'))
    print(f"  找到 {len(json_files)} 份JSON报告")

    all_stocks_by_date = {}
    for jf in json_files:
        date_str = jf.stem.replace('analysis_data_', '')
        date_dash = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        if date_dash > END_DATE:
            continue
        try:
            with open(jf) as f:
                data = json.load(f)
            stocks = data.get('all_stocks_with_scores', [])
            if stocks:
                all_stocks_by_date[date_dash] = stocks
        except Exception:
            continue

    report_dates = sorted(all_stocks_by_date.keys())
    print(f"  加载了 {len(report_dates)} 个交易日")
    print(f"  日期范围: {report_dates[0]} ~ {report_dates[-1]}")

    all_future_rets = batch_get_all_future_returns(report_dates,
                                                    holding_days_list=[3, 5, 10, 15])
    print(f"  已获取实际收益数据")

    records = []
    for date, stocks in all_stocks_by_date.items():
        buy_date_rets = all_future_rets.get(date, {})
        for stock in stocks:
            code = stock.get('stock_code', stock.get('code', ''))
            if not code:
                continue
            fwd = buy_date_rets.get(code, {})
            fwd_10d = fwd.get('return_10d')
            if fwd_10d is None:
                continue

            p10 = stock.get('pred_10d', 0) or 0
            p15 = stock.get('pred_15d', 0) or 0
            comp = 0.6 * p10 + 0.4 * p15

            records.append({
                'date': date,
                'code': code,
                'composite': comp,
                'pred_10d': p10,
                'pred_15d': p15,
                'rank_score': stock.get('rank_score', comp),
                'fwd_3d': fwd.get('return_3d'),
                'fwd_5d': fwd.get('return_5d'),
                'fwd_10d': fwd_10d,
                'fwd_15d': fwd.get('return_15d'),
            })

    df = pd.DataFrame(records)
    print(f"  有效样本: {len(df)} stock-days, {df['date'].nunique()} 个交易日")
    print(f"  composite分布: P50={df['composite'].median():.6f}, "
          f"P90={df['composite'].quantile(0.9):.6f}, "
          f"P95={df['composite'].quantile(0.95):.6f}, "
          f"P99={df['composite'].quantile(0.99):.6f}, "
          f"Max={df['composite'].max():.6f}")
    return df


def phase2_grid_search(df: pd.DataFrame):
    """网格搜索最优阈值 (PF-weighted, 与V4901相同方法)"""
    print("\n" + "=" * 70)
    print("Phase 2: 网格搜索最优阈值 (PF-weighted)")
    print("=" * 70)

    n_dates = df['date'].nunique()

    pcts = {}
    for p in [50, 70, 80, 90, 95, 97, 98, 99, 99.5, 99.8, 99.9]:
        pcts[p] = df['composite'].quantile(p / 100)
    print(f"  分布分位数:")
    for p, v in pcts.items():
        n_above = (df['composite'] >= v).sum()
        per_day = n_above / n_dates
        print(f"    P{p}: {v:.6f} ({per_day:.1f}/day)")

    sb_range = np.linspace(pcts[98], pcts[99.9], 40)
    buy_range = np.linspace(pcts[90], pcts[99], 30)
    cautious_range = np.linspace(pcts[70], pcts[95], 20)

    print(f"\n  搜索范围: strong_buy=[{pcts[98]:.5f}, {pcts[99.9]:.5f}]")
    print(f"            buy=[{pcts[90]:.5f}, {pcts[99]:.5f}]")
    print(f"            cautious=[{pcts[70]:.5f}, {pcts[95]:.5f}]")

    # ---- Step 1: strong_buy ----
    print(f"\n  Step 1: 搜索 strong_buy 阈值...")
    sb_results = []
    for sb_th in sb_range:
        mask = df['composite'] >= sb_th
        rets = df.loc[mask, 'fwd_10d'].dropna()
        n = len(rets)
        if n < 30:
            continue

        gp = rets[rets > 0].sum()
        gl = abs(rets[rets <= 0].sum())
        pf = gp / max(gl, 1e-8)
        wr = (rets > 0).mean()
        per_day = n / n_dates
        n_dates_with = df.loc[mask, 'date'].nunique()
        coverage = n_dates_with / n_dates

        pf_cap = min(pf, 5.0)

        if per_day < 0.3:
            cnt_pen = 0.4
        elif per_day <= 1:
            cnt_pen = 0.7 + 0.3 * (per_day - 0.3) / 0.7
        elif per_day <= 5:
            cnt_pen = 1.0
        elif per_day <= 10:
            cnt_pen = 1.0 - 0.3 * (per_day - 5) / 5
        elif per_day <= 20:
            cnt_pen = 0.7 - 0.3 * (per_day - 10) / 10
        else:
            cnt_pen = 0.3

        if coverage >= 0.5:
            cov_pen = 1.0
        elif coverage >= 0.3:
            cov_pen = 0.7 + 0.3 * (coverage - 0.3) / 0.2
        elif coverage >= 0.15:
            cov_pen = 0.4 + 0.3 * (coverage - 0.15) / 0.15
        else:
            cov_pen = 0.2

        score = pf_cap * wr * cnt_pen * cov_pen
        sb_results.append({
            'threshold': sb_th, 'score': score,
            'pf': pf, 'wr': wr, 'per_day': per_day,
            'coverage': coverage, 'avg_ret': rets.mean() * 100, 'n': n
        })

    sb_results.sort(key=lambda x: x['score'], reverse=True)
    for r in sb_results[:5]:
        print(f"    sb≥{r['threshold']:.6f}: score={r['score']:.3f} PF={r['pf']:.2f} WR={r['wr']:.1%} "
              f"{r['per_day']:.1f}/day cov={r['coverage']:.0%} avg={r['avg_ret']:+.2f}%")
    best_sb = sb_results[0]['threshold'] if sb_results else pcts[99]

    # ---- Step 2: buy ----
    print(f"\n  Step 2: 固定sb≥{best_sb:.6f}, 搜索 buy 阈值...")
    buy_results = []
    for buy_th in buy_range:
        if buy_th >= best_sb:
            continue
        mask = (df['composite'] >= buy_th) & (df['composite'] < best_sb)
        rets = df.loc[mask, 'fwd_10d'].dropna()
        n = len(rets)
        if n < 50:
            continue

        gp = rets[rets > 0].sum()
        gl = abs(rets[rets <= 0].sum())
        pf = gp / max(gl, 1e-8)
        wr = (rets > 0).mean()
        per_day = n / n_dates

        if per_day < 2:
            cnt_pen = 0.5
        elif per_day <= 5:
            cnt_pen = 0.7
        elif per_day <= 20:
            cnt_pen = 1.0
        elif per_day <= 50:
            cnt_pen = 0.8
        else:
            cnt_pen = 0.5

        score = min(pf, 4.0) * wr * cnt_pen
        buy_results.append({
            'threshold': buy_th, 'score': score,
            'pf': pf, 'wr': wr, 'per_day': per_day, 'n': n
        })

    buy_results.sort(key=lambda x: x['score'], reverse=True)
    for r in buy_results[:5]:
        print(f"    buy≥{r['threshold']:.6f}: score={r['score']:.3f} PF={r['pf']:.2f} WR={r['wr']:.1%} {r['per_day']:.1f}/day")
    best_buy = buy_results[0]['threshold'] if buy_results else pcts[95]

    # ---- Step 3: cautious ----
    print(f"\n  Step 3: 固定buy≥{best_buy:.6f}, 搜索 cautious 阈值...")
    cau_results = []
    for c_th in cautious_range:
        if c_th >= best_buy:
            continue
        mask = (df['composite'] >= c_th) & (df['composite'] < best_buy)
        rets = df.loc[mask, 'fwd_10d'].dropna()
        n = len(rets)
        if n < 100:
            continue

        gp = rets[rets > 0].sum()
        gl = abs(rets[rets <= 0].sum())
        pf = gp / max(gl, 1e-8)
        wr = (rets > 0).mean()

        cau_results.append({
            'threshold': c_th, 'pf': pf, 'wr': wr, 'n': n,
            'per_day': n / n_dates
        })

    best_cautious = pcts[70]
    for r in sorted(cau_results, key=lambda x: x['threshold']):
        if r['pf'] >= 1.0:
            best_cautious = r['threshold']
            print(f"    cautious≥{r['threshold']:.6f}: PF={r['pf']:.2f} WR={r['wr']:.1%} {r['per_day']:.1f}/day")
            break
    if not cau_results:
        print(f"    使用默认: {best_cautious:.6f}")

    # 合并
    results = []
    for r in sb_results[:5]:
        sb_th = r['threshold']
        for br in buy_results[:3]:
            if br['threshold'] < sb_th:
                results.append({
                    'strong_buy': round(sb_th, 6),
                    'buy': round(br['threshold'], 6),
                    'cautious': round(best_cautious, 6),
                    'score': round(r['score'], 4),
                    'sb_pf': round(r['pf'], 3),
                    'sb_wr': round(r['wr'], 4),
                    'sb_per_day': round(r['per_day'], 2),
                    'sb_coverage': round(r['coverage'], 3),
                    'sb_avg_ret': round(r['avg_ret'], 3),
                    'sb_n': r['n'],
                    'buy_pf': round(br['pf'], 3),
                    'buy_wr': round(br['wr'], 4),
                    'buy_n': br['n'],
                })
                break

    if not results:
        results = [{
            'strong_buy': round(pcts[99], 6), 'buy': round(pcts[95], 6),
            'cautious': round(pcts[80], 6), 'score': 0,
            'sb_pf': 0, 'sb_wr': 0, 'sb_per_day': 0, 'sb_coverage': 0,
            'sb_avg_ret': 0, 'sb_n': 0, 'buy_pf': 0, 'buy_wr': 0, 'buy_n': 0,
        }]
    return pd.DataFrame(results).sort_values('score', ascending=False)


def phase3_save(df_results: pd.DataFrame, df_cal: pd.DataFrame):
    """保存校准结果"""
    print("\n" + "=" * 70)
    print("Phase 3: 保存校准结果")
    print("=" * 70)

    print("\n  Top 10 组合:")
    print(f"  {'score':>6} | {'sb_th':>9} {'sb_PF':>6} {'sb_WR':>6} {'sb/day':>6} {'sb_cov':>6} {'sb_ret':>7} | {'buy_th':>9} {'buy_PF':>6} {'buy_WR':>6} | cautious")
    print("  " + "-" * 105)
    for _, row in df_results.head(10).iterrows():
        print(f"  {row['score']:6.3f} | {row['strong_buy']:9.6f} {row['sb_pf']:6.2f} {row['sb_wr']:6.1%} {row['sb_per_day']:6.1f} {row['sb_coverage']:6.0%} {row['sb_avg_ret']:+6.2f}% | "
              f"{row['buy']:9.6f} {row['buy_pf']:6.2f} {row['buy_wr']:6.1%} | {row['cautious']:.6f}")

    best = df_results.iloc[0]
    n_dates = df_cal['date'].nunique()
    date_range = f"{df_cal['date'].min()} ~ {df_cal['date'].max()}"
    hold_th = max(0.0, float(df_cal['composite'].quantile(0.30)))

    thresholds = {
        'strong_buy': float(best['strong_buy']),
        'buy': float(best['buy']),
        'cautious': float(best['cautious']),
        'hold': hold_th,
        '_calibration': {
            'method': 'grid_search_PF_weighted',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'data_range': f"{date_range} ({n_dates} trading days)",
            'optimal_max_per_day': round(float(best['sb_per_day']), 1),
            'optimization_score': float(best['score']),
            'strong_buy_10d_stats': f"PF={best['sb_pf']:.3f}, WR={best['sb_wr']:.1%}, "
                                    f"{best['sb_per_day']:.1f}/day, {best['sb_coverage']:.0%} coverage, "
                                    f"avg_ret={best['sb_avg_ret']:+.2f}%",
            'buy_10d_stats': f"PF={best['buy_pf']:.3f}, WR={best['buy_wr']:.1%}, n={int(best['buy_n'])}",
        }
    }

    out_path = MODEL_DIR / 'recommendation_thresholds.json'
    with open(out_path, 'w') as f:
        json.dump(thresholds, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ 已保存到: {out_path}")
    print(f"  strong_buy ≥ {thresholds['strong_buy']:.6f}")
    print(f"  buy        ≥ {thresholds['buy']:.6f}")
    print(f"  cautious   ≥ {thresholds['cautious']:.6f}")
    print(f"  hold       ≥ {thresholds['hold']:.6f}")
    return thresholds


if __name__ == '__main__':
    df_cal = phase1_load_and_match()
    df_results = phase2_grid_search(df_cal)
    phase3_save(df_results, df_cal)
    print("\n校准完成!")
