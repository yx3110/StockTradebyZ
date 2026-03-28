#!/usr/bin/env python3
"""
诊断: 为什么IC/ICIR高但回测收益低?

系统性检查6个假设:
  H1: 训练vs推理特征值不一致 (train-test skew)
  H2: IC在全局高但头部低 (head IC problem)
  H3: 头部预测同质化 (top-K都是同类股票)
  H4: Global quantiles校准失效 (prediction distribution shift)
  H5: 换手率过高吃掉alpha
  H6: 新增因子在WF OOS窗口退化

用法:
    python3 scripts/diagnose_ic_vs_return.py
"""

import sys
import os
import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, rankdata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')


def load_reports(report_dir, rank_field='auto'):
    """加载报告"""
    from backtest import backtest_report_based as brb
    import io, contextlib
    brb.DB_PATH = DB_PATH
    with contextlib.redirect_stdout(io.StringIO()):
        return brb.load_reports(report_dir, rank_field=rank_field)


def h1_train_test_feature_mismatch():
    """H1: 检查训练pipeline和scorer pipeline是否产生相同的特征值"""
    print("\n" + "="*70)
    print("H1: 训练 vs 推理 特征值一致性检查")
    print("="*70)

    import joblib
    model_path = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v486'
    pkl_files = sorted(model_path.glob('v486_*.pkl'), key=lambda f: f.stat().st_mtime)
    if not pkl_files:
        print("  ❌ 无V4.8.6模型")
        return

    m = joblib.load(pkl_files[-1])
    train_features = m.get('feature_names', [])
    print(f"  训练特征数: {len(train_features)}")
    print(f"  特征列表: {train_features[:10]}...")

    # 用scorer预测一天, 看它用了哪些特征
    try:
        from ml_models.v39.v486_production_scorer import V486ProductionScorer
        scorer = V486ProductionScorer()
        # 预测少量股票
        test_codes = ['000001.SZ', '600519.SH', '000858.SZ']
        results = scorer.predict_scores(test_codes, '2025-12-31')
        print(f"  Scorer预测: {len(results)} stocks")
        for code, data in results.items():
            print(f"    {code}: pred_10d={data.get('pred_10d',0):.6f}, score={data.get('score',0):.1f}")
        print("  ✅ Scorer可以正常预测")
    except Exception as e:
        print(f"  ❌ Scorer预测失败: {e}")


def h2_head_ic_analysis(reports):
    """H2: IC在全局高但头部低? 分析不同分位段的IC"""
    print("\n" + "="*70)
    print("H2: 全局IC vs 头部IC (top-quintile) 分析")
    print("="*70)

    conn = sqlite3.connect(DB_PATH)
    dates = sorted(reports.keys())

    global_ics = []
    top20_ics = []
    top10_ics = []
    top5_ics = []

    for date in dates:
        stocks = reports[date]
        codes = list(stocks.keys())
        preds = np.array([stocks[c].get('pred_10d', 0) for c in codes])

        # 获取实际10d收益
        future_query = """
        SELECT s.code,
               (q2.close / q1.close - 1) as ret_10d
        FROM daily_quotes q1
        JOIN securities s ON q1.security_id = s.id
        JOIN daily_quotes q2 ON q2.security_id = q1.security_id
        WHERE q1.trade_date = ?
          AND q2.trade_date = (
            SELECT trade_date FROM daily_quotes
            WHERE security_id = q1.security_id AND trade_date > q1.trade_date
            ORDER BY trade_date LIMIT 1 OFFSET 9
          )
          AND s.code IN ({})
        """.format(','.join(f"'{c}'" for c in codes))

        try:
            df_ret = pd.read_sql(future_query, conn, params=[date])
        except Exception:
            continue

        if len(df_ret) < 100:
            continue

        # 合并
        code_to_ret = dict(zip(df_ret['code'], df_ret['ret_10d']))
        valid_codes = [c for c in codes if c in code_to_ret]
        if len(valid_codes) < 100:
            continue

        p = np.array([stocks[c].get('pred_10d', 0) for c in valid_codes])
        r = np.array([code_to_ret[c] for c in valid_codes])

        # 全局IC
        try:
            ic, _ = spearmanr(p, r)
            if np.isfinite(ic):
                global_ics.append(ic)
        except:
            continue

        # Top 20% IC
        threshold_80 = np.percentile(p, 80)
        mask_top20 = p >= threshold_80
        if mask_top20.sum() >= 20:
            try:
                ic_top20, _ = spearmanr(p[mask_top20], r[mask_top20])
                if np.isfinite(ic_top20):
                    top20_ics.append(ic_top20)
            except:
                pass

        # Top 10% IC
        threshold_90 = np.percentile(p, 90)
        mask_top10 = p >= threshold_90
        if mask_top10.sum() >= 20:
            try:
                ic_top10, _ = spearmanr(p[mask_top10], r[mask_top10])
                if np.isfinite(ic_top10):
                    top10_ics.append(ic_top10)
            except:
                pass

        # Top 5% IC
        threshold_95 = np.percentile(p, 95)
        mask_top5 = p >= threshold_95
        if mask_top5.sum() >= 10:
            try:
                ic_top5, _ = spearmanr(p[mask_top5], r[mask_top5])
                if np.isfinite(ic_top5):
                    top5_ics.append(ic_top5)
            except:
                pass

    conn.close()

    print(f"  分析天数: {len(global_ics)}")
    if global_ics:
        print(f"  全局 IC:    mean={np.mean(global_ics):.4f}, ICIR={np.mean(global_ics)/max(np.std(global_ics),1e-8):.3f}")
    if top20_ics:
        print(f"  Top 20% IC: mean={np.mean(top20_ics):.4f}, ICIR={np.mean(top20_ics)/max(np.std(top20_ics),1e-8):.3f}")
    if top10_ics:
        print(f"  Top 10% IC: mean={np.mean(top10_ics):.4f}, ICIR={np.mean(top10_ics)/max(np.std(top10_ics),1e-8):.3f}")
    if top5_ics:
        print(f"  Top 5% IC:  mean={np.mean(top5_ics):.4f}, ICIR={np.mean(top5_ics)/max(np.std(top5_ics),1e-8):.3f}")

    if global_ics and top10_ics:
        ratio = np.mean(top10_ics) / max(abs(np.mean(global_ics)), 1e-8)
        print(f"\n  头部/全局 IC比: {ratio:.2f}")
        if ratio < 0.3:
            print("  ⚠️ 头部IC远低于全局IC → 模型在头部无区分力!")
        elif ratio < 0.7:
            print("  ⚠️ 头部IC明显低于全局IC → 头部区分力衰减")
        else:
            print("  ✅ 头部IC与全局IC接近 → 头部区分力正常")


def h3_top_stock_homogeneity(reports):
    """H3: Top-8选股是否同质化 (同行业/同类型)"""
    print("\n" + "="*70)
    print("H3: Top-8选股同质化分析 (行业集中度)")
    print("="*70)

    conn = sqlite3.connect(DB_PATH)
    # 加载行业信息
    ind_df = pd.read_sql("SELECT code, industry FROM securities WHERE type='A股' AND industry IS NOT NULL", conn)
    code_to_ind = dict(zip(ind_df['code'], ind_df['industry']))
    conn.close()

    dates = sorted(reports.keys())
    ind_counts = []  # 每天top8的不同行业数
    repeat_rates = []  # 连续两天top8重叠率

    prev_top8 = set()
    for date in dates:
        stocks = reports[date]
        top8 = sorted(stocks, key=lambda c: stocks[c].get('pred_10d', 0), reverse=True)[:8]

        # 行业分布
        industries = [code_to_ind.get(c, 'unknown') for c in top8]
        n_unique_ind = len(set(industries))
        ind_counts.append(n_unique_ind)

        # 重叠率
        top8_set = set(top8)
        if prev_top8:
            overlap = len(top8_set & prev_top8) / 8.0
            repeat_rates.append(overlap)
        prev_top8 = top8_set

    print(f"  分析天数: {len(dates)}")
    print(f"  Top8行业多样性: mean={np.mean(ind_counts):.1f}/8 unique industries")
    if repeat_rates:
        print(f"  日间重叠率: mean={np.mean(repeat_rates)*100:.1f}% (高=低换手, 低=高换手)")
        print(f"  隐含年化换手: {(1-np.mean(repeat_rates))*252:.0f}x")

    if np.mean(ind_counts) < 4:
        print("  ⚠️ 行业过度集中! Top8平均不到4个行业")
    if repeat_rates and np.mean(repeat_rates) < 0.3:
        print("  ⚠️ 日间重叠率极低 → 每天完全换股, 换手率过高!")


def h4_prediction_distribution(reports, reports_v475=None):
    """H4: 预测值分布是否异常"""
    print("\n" + "="*70)
    print("H4: 预测值分布对比 (V4.8.6 vs V4.7.5)")
    print("="*70)

    dates = sorted(reports.keys())

    # V486分布
    all_preds_486 = []
    top8_preds_486 = []
    for d in dates[:100]:
        stocks = reports[d]
        preds = [stocks[c].get('pred_10d', 0) for c in stocks]
        all_preds_486.extend(preds)
        top8 = sorted(preds, reverse=True)[:8]
        top8_preds_486.extend(top8)

    p486 = np.array(all_preds_486)
    t486 = np.array(top8_preds_486)
    print(f"  V4.8.6 pred_10d: mean={np.mean(p486):.6f}, std={np.std(p486):.6f}")
    print(f"  V4.8.6 top8:     mean={np.mean(t486):.6f}, range=[{np.min(t486):.6f}, {np.max(t486):.6f}]")
    print(f"  V4.8.6 top8 spread: {np.mean([np.max(t486[i*8:(i+1)*8])-np.min(t486[i*8:(i+1)*8]) for i in range(len(t486)//8)]):.6f}")

    if reports_v475:
        dates_475 = sorted(reports_v475.keys())
        all_preds_475 = []
        top8_preds_475 = []
        for d in dates_475[:100]:
            stocks = reports_v475[d]
            preds = [stocks[c].get('pred_10d', 0) for c in stocks]
            all_preds_475.extend(preds)
            top8 = sorted(preds, reverse=True)[:8]
            top8_preds_475.extend(top8)

        p475 = np.array(all_preds_475)
        t475 = np.array(top8_preds_475)
        print(f"\n  V4.7.5 pred_10d: mean={np.mean(p475):.6f}, std={np.std(p475):.6f}")
        print(f"  V4.7.5 top8:     mean={np.mean(t475):.6f}, range=[{np.min(t475):.6f}, {np.max(t475):.6f}]")
        print(f"  V4.7.5 top8 spread: {np.mean([np.max(t475[i*8:(i+1)*8])-np.min(t475[i*8:(i+1)*8]) for i in range(len(t475)//8)]):.6f}")

        print(f"\n  比较:")
        print(f"  pred_10d std: V486={np.std(p486):.6f} vs V475={np.std(p475):.6f} (比值={np.std(p486)/max(np.std(p475),1e-8):.2f})")
        print(f"  top8 mean:    V486={np.mean(t486):.6f} vs V475={np.mean(t475):.6f}")

        if np.std(p486) < np.std(p475) * 0.5:
            print("  ⚠️ V4.8.6预测方差远小于V4.7.5 → 预测值压缩, 头部无区分!")


def h5_turnover_analysis(reports):
    """H5: 换手率分析"""
    print("\n" + "="*70)
    print("H5: 换手率分析")
    print("="*70)

    dates = sorted(reports.keys())
    turnovers = []
    prev_top8 = set()

    for d in dates:
        stocks = reports[d]
        top8 = set(sorted(stocks, key=lambda c: stocks[c].get('pred_10d', 0), reverse=True)[:8])
        if prev_top8:
            changed = len(top8 - prev_top8)
            turnovers.append(changed / 8.0)
        prev_top8 = top8

    if turnovers:
        print(f"  日换手率: mean={np.mean(turnovers)*100:.1f}%, median={np.median(turnovers)*100:.1f}%")
        print(f"  年化换手: {np.mean(turnovers)*252:.0f}x")
        print(f"  年化交易成本 (0.3%/次): {np.mean(turnovers)*252*0.003*100:.1f}%")

        if np.mean(turnovers) > 0.5:
            print("  ⚠️ 日换手率>50% → 大部分alpha被交易成本吃掉!")


def h6_wf_window_analysis():
    """H6: WF各窗口的模型质量是否退化"""
    print("\n" + "="*70)
    print("H6: WF窗口分析 (模型内部)")
    print("="*70)

    import joblib
    model_path = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v486'
    pkl_files = sorted(model_path.glob('v486_*.pkl'), key=lambda f: f.stat().st_mtime)
    if not pkl_files:
        print("  ❌ 无模型")
        return

    m = joblib.load(pkl_files[-1])

    # WF metrics
    wf = m.get('walk_forward_metrics', {})
    if wf:
        print("  WF OOS Metrics:")
        for t in ['3d', '5d', '10d', '15d']:
            key = f'label_{t}' if f'label_{t}' in wf else t
            if key in wf:
                d = wf[key]
                print(f"    {t}: IC={d.get('mean_ic',0):.4f}, ICIR={d.get('mean_icir',0):.4f}")

    # Ensemble weights
    ew = m.get('ensemble_weights', {})
    if not ew:
        # 尝试从models中提取
        models_data = m.get('models', {})
        if '10d' in models_data and isinstance(models_data['10d'], dict):
            ew = {'10d': models_data['10d'].get('weights', {})}

    if ew:
        print("\n  Ensemble权重 (10d):")
        w10 = ew.get('10d', ew.get('label_10d', {}))
        if isinstance(w10, dict):
            for name, w in sorted(w10.items(), key=lambda x: -x[1]):
                bar = '█' * int(w * 50)
                print(f"    {name:>10}: {w:.4f} {bar}")

            # 检查权重集中度
            weights = list(w10.values())
            max_w = max(weights)
            if max_w > 0.5:
                print(f"  ⚠️ 权重过度集中: {max(w10, key=w10.get)} 占 {max_w*100:.0f}%")

    # Adaptive target weights
    atw = m.get('adaptive_target_weights', {})
    if atw:
        print(f"\n  自适应目标权重: {atw}")
        w10d = atw.get('label_10d', 0)
        if w10d < 0.25:
            print(f"  ⚠️ 10d权重仅{w10d:.2f} → 模型不信任10d预测")


def main():
    print("=" * 70)
    print("IC高但回测低 — 系统性诊断")
    print("=" * 70)

    # 加载报告
    v486_dir = str(PROJECT_ROOT / 'reports' / 'daily_selection_v486_exp')
    v475_dir = str(PROJECT_ROOT / 'reports' / 'daily_selection_v4.7.5')

    print("\n加载报告...", flush=True)
    reports_486 = load_reports(v486_dir)
    reports_475 = load_reports(v475_dir) if os.path.exists(v475_dir) else None
    print(f"  V4.8.6: {len(reports_486)} days")
    if reports_475:
        print(f"  V4.7.5: {len(reports_475)} days")

    # 运行诊断
    h1_train_test_feature_mismatch()
    h2_head_ic_analysis(reports_486)
    h3_top_stock_homogeneity(reports_486)
    h4_prediction_distribution(reports_486, reports_475)
    h5_turnover_analysis(reports_486)
    h6_wf_window_analysis()

    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70)


if __name__ == '__main__':
    main()
