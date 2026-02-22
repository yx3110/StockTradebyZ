#!/usr/bin/env python3
"""
ML模型分数阈值评估工具

分析V3.9和V3.95模型的分数分布，找到最优阈值使每天筛选出指定数量的高概率股票。
基于回测picks CSV数据计算不同阈值下的3d/5d/10d/15d预期收益和胜率。

用法:
    # 自动查找最新的picks CSV文件进行分析 (默认目标<=10只/天)
    python3 backtest/evaluate_score_threshold.py

    # 指定目标日均股数
    python3 backtest/evaluate_score_threshold.py --target 5

    # 指定CSV文件
    python3 backtest/evaluate_score_threshold.py \
        --v39-csv reports/backtest/report_backtest_v39新模型_20260222_141956_picks.csv \
        --v395-csv reports/backtest/report_backtest_v395新模型_20260222_141956_picks.csv

    # 只分析单个模型
    python3 backtest/evaluate_score_threshold.py --v39-csv some_file.csv

    # 先生成最新picks数据再分析
    python3 backtest/evaluate_score_threshold.py --regenerate

    # 指定报告目录生成picks后分析
    python3 backtest/evaluate_score_threshold.py \
        --v39-dir reports/daily_selection_v3.9_model20260222 \
        --v395-dir reports/daily_selection_v3.95_model20260221
"""

import sys
import os
import argparse
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from glob import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


def find_latest_picks_csv(pattern, exclude_pattern=None):
    """在reports/backtest/下找到最新的picks CSV文件"""
    search_dir = os.path.join(PROJECT_ROOT, 'reports', 'backtest')
    candidates = sorted(glob(os.path.join(search_dir, pattern)), key=os.path.getmtime, reverse=True)
    if exclude_pattern:
        candidates = [c for c in candidates if exclude_pattern not in os.path.basename(c)]
    return candidates[0] if candidates else None


def analyze_model(csv_path, model_name, target_count=10):
    """分析单个模型的分数分布和阈值收益，返回分析结果字典"""
    df = pd.read_csv(csv_path)
    total_days = df['date'].nunique()
    total_records = len(df)
    avg_per_day = total_records / total_days

    result = {
        'model_name': model_name,
        'csv_path': csv_path,
        'total_days': total_days,
        'total_records': total_records,
        'avg_per_day': avg_per_day,
        'score_min': df['score'].min(),
        'score_max': df['score'].max(),
        'score_mean': df['score'].mean(),
        'score_median': df['score'].median(),
        'score_std': df['score'].std(),
        'percentiles': {},
        'threshold_results': [],
        'fine_results': [],
        'daily_distributions': [],
        'recommendation': {},
    }

    # 分位数
    for pct in [50, 60, 70, 75, 80, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]:
        val = df['score'].quantile(pct / 100)
        result['percentiles'][pct] = val

    # 粗扫阈值
    score_start = df['score'].quantile(0.80)
    score_end = df['score'].max()

    for threshold in np.arange(score_start, score_end + 0.5, 0.5):
        tr = _calc_threshold_stats(df, threshold, total_days)
        if tr is None:
            break
        result['threshold_results'].append(tr)

    # 细粒度分析：日均target_count附近的范围
    target_candidates = [r for r in result['threshold_results']
                         if target_count * 0.3 <= r['avg_count'] <= target_count * 1.5]

    if target_candidates:
        low_th = min(r['threshold'] for r in target_candidates) - 1
        high_th = max(r['threshold'] for r in target_candidates) + 1

        for threshold in np.arange(low_th, high_th, 0.25):
            tr = _calc_threshold_stats(df, threshold, total_days)
            if tr is None:
                continue
            if tr['avg_count'] < target_count * 0.2 or tr['avg_count'] > target_count * 1.5:
                continue
            result['fine_results'].append(tr)

    # 日度分布
    for tr in result['threshold_results']:
        if target_count * 0.3 <= tr['avg_count'] <= target_count * 1.2:
            daily_counts = df[df['score'] >= tr['threshold']].groupby('date').size()
            zero_days = total_days - len(daily_counts)
            dist = {
                'threshold': tr['threshold'],
                'mean': daily_counts.mean(),
                'median': daily_counts.median(),
                'max': daily_counts.max(),
                'zero_days': zero_days,
                'le3_days': (daily_counts <= 3).sum() + zero_days,
                'le_target_days': (daily_counts <= target_count).sum() + zero_days,
                'total_days': total_days,
            }
            result['daily_distributions'].append(dist)

    # 推荐阈值
    candidates = [r for r in result['fine_results']
                  if r['avg_count'] <= target_count and r['valid_5d'] >= 10]
    if not candidates:
        candidates = [r for r in result['threshold_results']
                      if r['avg_count'] <= target_count and r['valid_5d'] >= 10]

    if candidates:
        best_5d_return = max(candidates, key=lambda x: x['avg_5d'])
        best_5d_wr = max(candidates, key=lambda x: x['wr_5d'])
        best_combined = max(candidates, key=lambda x: x['avg_5d'] * x['wr_5d'] / 100)

        # 找到日均最接近target_count/2的阈值（在收益正的前提下）
        good_candidates = [r for r in candidates if r['avg_5d'] > 0]
        if good_candidates:
            best_balanced = min(good_candidates,
                                key=lambda x: abs(x['avg_count'] - target_count * 0.6))
        else:
            best_balanced = best_combined

        result['recommendation'] = {
            'best_5d_return': best_5d_return,
            'best_5d_wr': best_5d_wr,
            'best_combined': best_combined,
            'best_balanced': best_balanced,
        }

    return result


def _calc_threshold_stats(df, threshold, total_days):
    """计算指定阈值下的统计指标"""
    subset = df[df['score'] >= threshold]
    if len(subset) == 0:
        return None

    avg_count = len(subset) / total_days

    valid_3d = subset['return_3d'].dropna()
    valid_5d = subset['return_5d'].dropna()
    valid_10d = subset['return_10d'].dropna()

    # return_15d 列可能不存在
    if 'return_15d' in subset.columns:
        valid_15d = subset['return_15d'].dropna()
    else:
        valid_15d = pd.Series(dtype=float)

    if len(valid_5d) < 5:
        return None

    return {
        'threshold': threshold,
        'avg_count': avg_count,
        'total': len(subset),
        'valid_5d': len(valid_5d),
        'avg_3d': valid_3d.mean() * 100 if len(valid_3d) > 0 else float('nan'),
        'wr_3d': (valid_3d > 0).mean() * 100 if len(valid_3d) > 0 else float('nan'),
        'avg_5d': valid_5d.mean() * 100,
        'wr_5d': (valid_5d > 0).mean() * 100,
        'avg_10d': valid_10d.mean() * 100 if len(valid_10d) > 0 else float('nan'),
        'wr_10d': (valid_10d > 0).mean() * 100 if len(valid_10d) > 0 else float('nan'),
        'avg_15d': valid_15d.mean() * 100 if len(valid_15d) > 0 else float('nan'),
        'wr_15d': (valid_15d > 0).mean() * 100 if len(valid_15d) > 0 else float('nan'),
    }


def analyze_dual_model(v39_csv, v395_csv, target_count=10):
    """双模型交叉验证分析"""
    df39 = pd.read_csv(v39_csv)
    df395 = pd.read_csv(v395_csv)

    merged = pd.merge(df39, df395, on=['date', 'code'], suffixes=('_v39', '_v395'))
    total_days = merged['date'].nunique()

    results = []
    v39_percentiles = [70, 75, 80, 85, 90, 95]
    v395_percentiles = [70, 75, 80, 85, 90, 95]

    for v39_pct in v39_percentiles:
        for v395_pct in v395_percentiles:
            v39_th = df39['score'].quantile(v39_pct / 100)
            v395_th = df395['score'].quantile(v395_pct / 100)

            subset = merged[(merged['score_v39'] >= v39_th) & (merged['score_v395'] >= v395_th)]
            if len(subset) < 5:
                continue

            avg_count = len(subset) / total_days
            if avg_count > target_count * 1.5:
                continue

            # 使用v39的return列（两个模型的return应该一致）
            ret_5d_col = 'return_5d_v39' if 'return_5d_v39' in subset.columns else 'return_5d'
            ret_10d_col = 'return_10d_v39' if 'return_10d_v39' in subset.columns else 'return_10d'
            ret_3d_col = 'return_3d_v39' if 'return_3d_v39' in subset.columns else 'return_3d'

            valid_3d = subset[ret_3d_col].dropna()
            valid_5d = subset[ret_5d_col].dropna()
            valid_10d = subset[ret_10d_col].dropna()

            if len(valid_5d) < 5:
                continue

            results.append({
                'v39_threshold': v39_th,
                'v395_threshold': v395_th,
                'v39_pct': v39_pct,
                'v395_pct': v395_pct,
                'avg_count': avg_count,
                'total': len(subset),
                'avg_3d': valid_3d.mean() * 100,
                'wr_3d': (valid_3d > 0).mean() * 100,
                'avg_5d': valid_5d.mean() * 100,
                'wr_5d': (valid_5d > 0).mean() * 100,
                'avg_10d': valid_10d.mean() * 100,
                'wr_10d': (valid_10d > 0).mean() * 100,
            })

    return {
        'total_days': total_days,
        'total_merged': len(merged),
        'results': results,
    }


def generate_report(v39_result, v395_result, dual_result, target_count, output_path):
    """生成Markdown分析报告"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    lines = []
    lines.append("# ML模型分数阈值评估报告\n")
    lines.append(f"**生成时间**: {timestamp}")
    lines.append(f"**目标日均股数**: ≤ {target_count} 只\n")

    # ========== 推荐阈值总结 ==========
    lines.append("## 推荐阈值总结\n")

    if v395_result and v395_result.get('recommendation'):
        rec = v395_result['recommendation']
        bb = rec.get('best_balanced', rec.get('best_combined'))
        lines.append(f"### V3.95 推荐阈值: **>= {bb['threshold']:.1f} 分**\n")
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 日均股数 | {bb['avg_count']:.1f} 只 |")
        lines.append(f"| 3d 平均收益 | {bb['avg_3d']:+.2f}% |")
        lines.append(f"| 3d 胜率 | {bb['wr_3d']:.1f}% |")
        lines.append(f"| 5d 平均收益 | {bb['avg_5d']:+.2f}% |")
        lines.append(f"| 5d 胜率 | {bb['wr_5d']:.1f}% |")
        lines.append(f"| 10d 平均收益 | {bb['avg_10d']:+.2f}% |")
        lines.append(f"| 10d 胜率 | {bb['wr_10d']:.1f}% |")
        if not np.isnan(bb.get('avg_15d', float('nan'))):
            lines.append(f"| 15d 平均收益 | {bb['avg_15d']:+.2f}% |")
            lines.append(f"| 15d 胜率 | {bb['wr_15d']:.1f}% |")
        lines.append(f"| 样本数 | {bb['total']} 条 (有效5d: {bb['valid_5d']}) |")
        lines.append("")

    if v39_result and v39_result.get('recommendation'):
        rec = v39_result['recommendation']
        bb = rec.get('best_balanced', rec.get('best_combined'))
        lines.append(f"### V3.9 推荐阈值: **>= {bb['threshold']:.1f} 分**\n")
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 日均股数 | {bb['avg_count']:.1f} 只 |")
        lines.append(f"| 3d 平均收益 | {bb['avg_3d']:+.2f}% |")
        lines.append(f"| 3d 胜率 | {bb['wr_3d']:.1f}% |")
        lines.append(f"| 5d 平均收益 | {bb['avg_5d']:+.2f}% |")
        lines.append(f"| 5d 胜率 | {bb['wr_5d']:.1f}% |")
        lines.append(f"| 10d 平均收益 | {bb['avg_10d']:+.2f}% |")
        lines.append(f"| 10d 胜率 | {bb['wr_10d']:.1f}% |")
        if not np.isnan(bb.get('avg_15d', float('nan'))):
            lines.append(f"| 15d 平均收益 | {bb['avg_15d']:+.2f}% |")
            lines.append(f"| 15d 胜率 | {bb['wr_15d']:.1f}% |")
        lines.append(f"| 样本数 | {bb['total']} 条 (有效5d: {bb['valid_5d']}) |")
        lines.append("")

    # ========== 详细分析 ==========
    for model_result in [v395_result, v39_result]:
        if model_result is None:
            continue

        name = model_result['model_name']
        lines.append(f"\n## {name} 详细分析\n")

        # 基本信息
        lines.append(f"### 基本信息\n")
        lines.append(f"- 数据来源: `{os.path.basename(model_result['csv_path'])}`")
        lines.append(f"- 交易日数: {model_result['total_days']}")
        lines.append(f"- 总记录数: {model_result['total_records']}")
        lines.append(f"- 日均股票数: {model_result['avg_per_day']:.1f}")
        lines.append(f"- 分数范围: {model_result['score_min']:.2f} ~ {model_result['score_max']:.2f}")
        lines.append(f"- 分数均值: {model_result['score_mean']:.2f}, 中位数: {model_result['score_median']:.2f}, 标准差: {model_result['score_std']:.2f}")
        lines.append("")

        # 分位数
        lines.append(f"### 分数分位数\n")
        lines.append(f"| 分位 | 分数阈值 |")
        lines.append(f"|------|---------|")
        for pct, val in model_result['percentiles'].items():
            lines.append(f"| Top {100 - pct}% | >= {val:.2f} |")
        lines.append("")

        # 阈值分析表
        lines.append(f"### 阈值-收益分析\n")
        lines.append(f"| 阈值 | 日均数 | 总数 | 3d均值 | 3d胜率 | 5d均值 | 5d胜率 | 10d均值 | 10d胜率 | 15d均值 | 15d胜率 |")
        lines.append(f"|-----:|-------:|-----:|-------:|-------:|-------:|-------:|--------:|--------:|--------:|--------:|")
        for tr in model_result['threshold_results']:
            marker = " **" if target_count * 0.3 <= tr['avg_count'] <= target_count else ""
            marker_end = "**" if marker else ""
            avg_15d_str = f"{tr['avg_15d']:+.2f}%" if not np.isnan(tr.get('avg_15d', float('nan'))) else "N/A"
            wr_15d_str = f"{tr['wr_15d']:.1f}%" if not np.isnan(tr.get('wr_15d', float('nan'))) else "N/A"
            lines.append(
                f"| {marker}>= {tr['threshold']:.1f}{marker_end} | {tr['avg_count']:.1f} | {tr['total']} | "
                f"{tr['avg_3d']:+.2f}% | {tr['wr_3d']:.1f}% | "
                f"{tr['avg_5d']:+.2f}% | {tr['wr_5d']:.1f}% | "
                f"{tr['avg_10d']:+.2f}% | {tr['wr_10d']:.1f}% | "
                f"{avg_15d_str} | {wr_15d_str} |")
        lines.append("")

        # 细粒度分析
        if model_result['fine_results']:
            lines.append(f"### 细粒度分析 (日均{target_count * 0.3:.0f}-{target_count}只范围)\n")
            lines.append(f"| 阈值 | 日均数 | 3d均值 | 3d胜率 | 5d均值 | 5d胜率 | 10d均值 | 10d胜率 |")
            lines.append(f"|-----:|-------:|-------:|-------:|-------:|-------:|--------:|--------:|")
            for tr in model_result['fine_results']:
                lines.append(
                    f"| >= {tr['threshold']:.2f} | {tr['avg_count']:.1f} | "
                    f"{tr['avg_3d']:+.2f}% | {tr['wr_3d']:.1f}% | "
                    f"{tr['avg_5d']:+.2f}% | {tr['wr_5d']:.1f}% | "
                    f"{tr['avg_10d']:+.2f}% | {tr['wr_10d']:.1f}% |")
            lines.append("")

        # 日度分布
        if model_result['daily_distributions']:
            lines.append(f"### 日度分布统计\n")
            lines.append(f"| 阈值 | 均值 | 中位数 | 最大 | 0只天数 | ≤3只天数 | ≤{target_count}只天数 |")
            lines.append(f"|-----:|-----:|-------:|-----:|--------:|---------:|----------:|")
            td = model_result['total_days']
            for d in model_result['daily_distributions']:
                lines.append(
                    f"| >= {d['threshold']:.1f} | {d['mean']:.1f} | {d['median']:.0f} | {d['max']} | "
                    f"{d['zero_days']}/{td} ({d['zero_days'] / td * 100:.0f}%) | "
                    f"{d['le3_days']}/{td} ({d['le3_days'] / td * 100:.0f}%) | "
                    f"{d['le_target_days']}/{td} ({d['le_target_days'] / td * 100:.0f}%) |")
            lines.append("")

        # 推荐
        if model_result.get('recommendation'):
            rec = model_result['recommendation']
            lines.append(f"### 推荐阈值\n")
            for label, key in [("最高5d收益", "best_5d_return"),
                                ("最高5d胜率", "best_5d_wr"),
                                ("综合最优", "best_combined"),
                                ("平衡推荐", "best_balanced")]:
                r = rec.get(key)
                if r:
                    lines.append(
                        f"- **{label}**: >= {r['threshold']:.1f} 分, "
                        f"日均{r['avg_count']:.1f}只, "
                        f"5d均{r['avg_5d']:+.2f}% WR{r['wr_5d']:.1f}%, "
                        f"10d均{r['avg_10d']:+.2f}% WR{r['wr_10d']:.1f}%")
            lines.append("")

    # ========== 双模型交叉 ==========
    if dual_result and dual_result['results']:
        lines.append(f"\n## 双模型交叉验证\n")
        lines.append(f"- 共同覆盖: {dual_result['total_merged']} 条记录, {dual_result['total_days']} 交易日\n")

        lines.append(f"| V3.9阈值 | V3.95阈值 | 日均数 | 总数 | 5d均值 | 5d胜率 | 10d均值 | 10d胜率 |")
        lines.append(f"|---------:|---------:|-------:|-----:|-------:|-------:|--------:|--------:|")

        for r in dual_result['results']:
            if r['avg_count'] > target_count * 1.2:
                continue
            lines.append(
                f"| >= {r['v39_threshold']:.1f} (Top{100 - r['v39_pct']}%) "
                f"| >= {r['v395_threshold']:.1f} (Top{100 - r['v395_pct']}%) "
                f"| {r['avg_count']:.1f} | {r['total']} | "
                f"{r['avg_5d']:+.2f}% | {r['wr_5d']:.1f}% | "
                f"{r['avg_10d']:+.2f}% | {r['wr_10d']:.1f}% |")
        lines.append("")

    report_text = '\n'.join(lines)

    # 保存报告
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    return report_text


def print_summary(v39_result, v395_result, target_count):
    """打印简洁的终端总结"""
    print(f"\n{'=' * 70}")
    print(f"  ML模型分数阈值评估 (目标: 日均 ≤ {target_count} 只)")
    print(f"{'=' * 70}")

    for model_result in [v395_result, v39_result]:
        if model_result is None:
            continue

        name = model_result['model_name']
        print(f"\n📊 {name}")
        print(f"  分数范围: {model_result['score_min']:.1f} ~ {model_result['score_max']:.1f}, "
              f"均值: {model_result['score_mean']:.1f}, 标准差: {model_result['score_std']:.1f}")
        print(f"  交易日: {model_result['total_days']}, 日均股票: {model_result['avg_per_day']:.0f}")

        if model_result.get('recommendation'):
            rec = model_result['recommendation']
            bb = rec.get('best_balanced', rec.get('best_combined'))
            print(f"\n  🎯 推荐阈值: >= {bb['threshold']:.1f} 分")
            print(f"     日均: {bb['avg_count']:.1f} 只")
            print(f"     3d:  {bb['avg_3d']:+.2f}% (胜率 {bb['wr_3d']:.1f}%)")
            print(f"     5d:  {bb['avg_5d']:+.2f}% (胜率 {bb['wr_5d']:.1f}%)")
            print(f"     10d: {bb['avg_10d']:+.2f}% (胜率 {bb['wr_10d']:.1f}%)")
            if not np.isnan(bb.get('avg_15d', float('nan'))):
                print(f"     15d: {bb['avg_15d']:+.2f}% (胜率 {bb['wr_15d']:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='ML模型分数阈值评估')
    parser.add_argument('--target', type=int, default=10,
                        help='目标日均股票数量 (默认: 10)')
    parser.add_argument('--v39-csv', type=str, default=None,
                        help='V3.9 picks CSV文件路径')
    parser.add_argument('--v395-csv', type=str, default=None,
                        help='V3.95 picks CSV文件路径')
    parser.add_argument('--v39-dir', type=str, default=None,
                        help='V3.9 报告目录 (用于生成picks CSV)')
    parser.add_argument('--v395-dir', type=str, default=None,
                        help='V3.95 报告目录 (用于生成picks CSV)')
    parser.add_argument('--regenerate', action='store_true',
                        help='使用backtest_report_based.py重新生成picks数据')
    parser.add_argument('--output', type=str, default=None,
                        help='输出报告路径 (默认: reports/backtest/阈值评估报告_YYYYMMDD.md)')
    parser.add_argument('--no-dual', action='store_true',
                        help='跳过双模型交叉分析')
    args = parser.parse_args()

    target_count = args.target

    # 确定CSV文件路径
    v39_csv = args.v39_csv
    v395_csv = args.v395_csv

    # 如果指定了报告目录，先生成picks
    if args.v39_dir or args.v395_dir or args.regenerate:
        backtest_script = os.path.join(PROJECT_ROOT, 'backtest', 'backtest_report_based.py')

        if args.v39_dir:
            v39_label = os.path.basename(args.v39_dir)
            print(f"正在为 {v39_label} 生成回测数据...")
            result = subprocess.run(
                ['python3', backtest_script,
                 '--report-dir', args.v39_dir,
                 '--label', v39_label],
                capture_output=True, text=True, cwd=PROJECT_ROOT
            )
            if result.returncode != 0:
                print(f"回测生成失败: {result.stderr}")
            else:
                # 查找刚生成的CSV
                v39_csv = find_latest_picks_csv(f'report_backtest_{v39_label}_*_picks.csv')
                if v39_csv:
                    print(f"  找到: {v39_csv}")

        if args.v395_dir:
            v395_label = os.path.basename(args.v395_dir)
            print(f"正在为 {v395_label} 生成回测数据...")
            result = subprocess.run(
                ['python3', backtest_script,
                 '--report-dir', args.v395_dir,
                 '--label', v395_label],
                capture_output=True, text=True, cwd=PROJECT_ROOT
            )
            if result.returncode != 0:
                print(f"回测生成失败: {result.stderr}")
            else:
                v395_csv = find_latest_picks_csv(f'report_backtest_{v395_label}_*_picks.csv')
                if v395_csv:
                    print(f"  找到: {v395_csv}")

        if args.regenerate and not args.v39_dir and not args.v395_dir:
            print("正在生成四模型回测数据...")
            result = subprocess.run(
                ['python3', backtest_script, '--all'],
                capture_output=True, text=True, cwd=PROJECT_ROOT
            )
            if result.returncode != 0:
                print(f"回测生成失败: {result.stderr}")

    # 自动查找最新CSV (注意: v39 pattern会匹配v395，需要排除)
    if v39_csv is None:
        for pattern in ['report_backtest_*v39*新*_picks.csv',
                        'report_backtest_*v3.9*_picks.csv',
                        'report_backtest_*v39*_picks.csv',
                        'ml_backtest_v39_*_picks.csv']:
            v39_csv = find_latest_picks_csv(pattern, exclude_pattern='v395')
            if v39_csv:
                break

    if v395_csv is None:
        for pattern in ['report_backtest_*v395*新*_picks.csv',
                        'report_backtest_*v3.95*_picks.csv',
                        'report_backtest_*v395*_picks.csv',
                        'ml_backtest_v395_*_picks.csv']:
            v395_csv = find_latest_picks_csv(pattern)
            if v395_csv:
                break

    if not v39_csv and not v395_csv:
        print("错误: 未找到任何picks CSV文件")
        print("请先运行回测生成picks数据:")
        print("  python3 backtest/backtest_report_based.py --all")
        print("或指定CSV文件:")
        print("  python3 backtest/evaluate_score_threshold.py --v39-csv <path> --v395-csv <path>")
        sys.exit(1)

    # 运行分析
    v39_result = None
    v395_result = None
    dual_result = None

    if v39_csv:
        print(f"\n分析 V3.9: {os.path.basename(v39_csv)}")
        v39_result = analyze_model(v39_csv, "V3.9", target_count)

    if v395_csv:
        print(f"分析 V3.95: {os.path.basename(v395_csv)}")
        v395_result = analyze_model(v395_csv, "V3.95", target_count)

    if v39_csv and v395_csv and not args.no_dual:
        print("分析双模型交叉...")
        dual_result = analyze_dual_model(v39_csv, v395_csv, target_count)

    # 输出报告
    if args.output:
        output_path = args.output
    else:
        date_str = datetime.now().strftime('%Y%m%d')
        output_path = os.path.join(PROJECT_ROOT, 'reports', 'backtest',
                                   f'阈值评估报告_{date_str}.md')

    report_text = generate_report(v39_result, v395_result, dual_result, target_count, output_path)
    print(f"\n报告已保存: {output_path}")

    # 终端摘要
    print_summary(v39_result, v395_result, target_count)

    print(f"\n详细报告: {output_path}")


if __name__ == '__main__':
    main()
