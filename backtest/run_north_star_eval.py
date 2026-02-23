#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
北极星指标快速评估脚本

在模型重训练后，快速生成报告、运行回测、输出评分卡。

用法:
    # 1. 模型训练完成后，生成报告 + 回测
    python3 backtest/run_north_star_eval.py --generate-reports --backtest

    # 2. 仅回测已有报告
    python3 backtest/run_north_star_eval.py --backtest --report-dir reports/daily_selection_v3.95_robust_zscore

    # 3. 多模型对比
    python3 backtest/run_north_star_eval.py --compare

作者: Claude Code
创建时间: 2026-02-23
"""

import sys
import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')


def generate_reports(scoring_version='v3.95', start_date='2025-09-01', end_date='2026-02-13'):
    """批量生成选股报告 (并行版)"""
    import subprocess
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print(f"\n{'='*60}")
    print(f"  批量生成 {scoring_version} 报告: {start_date} → {end_date}")
    print(f"{'='*60}\n")

    dates = _get_trading_dates(start_date, end_date)
    print(f"  共 {len(dates)} 个交易日")

    # 跳过已有报告
    if scoring_version == 'v3.95':
        report_dir = PROJECT_ROOT / 'reports' / 'daily_selection_v3.95_robust_zscore'
    else:
        report_dir = PROJECT_ROOT / 'reports' / f'daily_selection_{scoring_version}'

    existing = set()
    if report_dir.exists():
        for f in report_dir.glob('*.md'):
            # 提取日期 from 选股分析报告_YYYYMMDD.md
            name = f.stem
            if '_' in name:
                date_str = name.split('_')[-1]
                if len(date_str) == 8:
                    existing.add(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}")

    dates_todo = [d for d in dates if d not in existing]
    print(f"  已有 {len(existing)} 份报告, 需生成 {len(dates_todo)} 份")

    if not dates_todo:
        print("  所有报告已存在, 跳过")
        return

    def gen_one(date):
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / 'tomorrow_stock_selector.py'),
            date,
            '--scoring-version', scoring_version,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                                     cwd=str(PROJECT_ROOT))
            if result.returncode != 0:
                return date, f"error: {result.stderr[:100]}"
            return date, "ok"
        except subprocess.TimeoutExpired:
            return date, "timeout"
        except Exception as e:
            return date, str(e)

    # 串行执行(避免DB锁冲突)
    done = 0
    for date in dates_todo:
        done += 1
        if done % 10 == 0 or done == 1:
            print(f"  [{done}/{len(dates_todo)}] {date}")
        _, status = gen_one(date)
        if status != "ok":
            print(f"    ⚠️ {date}: {status}")

    print(f"\n  报告生成完成 ({done} 份)")


def run_backtest(report_dir, label, top_n=20, benchmark='000905.SH', focus_days=10,
                 retention_bonus=0.0):
    """运行单个模型的回测"""
    from backtest import backtest_report_based as brb
    from backtest import north_star_metrics as nsm

    # 确保DB路径正确
    nsm.DB_PATH = DB_PATH
    brb.DB_PATH = DB_PATH

    reports = brb.load_reports(report_dir)
    if not reports:
        print(f"  ⚠️ 无报告: {report_dir}")
        return None

    result = brb.run_single_backtest(
        reports, label, top_n=top_n,
        benchmark_code=benchmark, focus_days=focus_days,
        retention_bonus=retention_bonus
    )
    return result


def run_comparison(top_n=20, benchmark='000905.SH', focus_days=10):
    """多模型对比"""
    from backtest import backtest_report_based as brb
    from backtest import north_star_metrics as nsm
    nsm.DB_PATH = DB_PATH
    brb.DB_PATH = DB_PATH

    report_dirs = {
        'v3.9': str(PROJECT_ROOT / 'reports' / 'daily_selection_v3.9'),
        'v3.95-RZ': str(PROJECT_ROOT / 'reports' / 'daily_selection_v3.95_robust_zscore'),
    }

    # 检查是否有新的Phase 2报告
    phase2_dir = str(PROJECT_ROOT / 'reports' / 'daily_selection_v3.95_phase2')
    if os.path.isdir(phase2_dir) and os.listdir(phase2_dir):
        report_dirs['v3.95-Phase2'] = phase2_dir

    results = {}
    for label, dir_path in report_dirs.items():
        if not os.path.isdir(dir_path):
            print(f"  跳过 {label}: 目录不存在")
            continue
        reports = brb.load_reports(dir_path)
        if not reports:
            print(f"  跳过 {label}: 无报告")
            continue
        print(f"\n{'#'*80}")
        results[label] = brb.run_single_backtest(
            reports, label, top_n=top_n,
            benchmark_code=benchmark, focus_days=focus_days
        )

    if len(results) > 1:
        brb.compare_results(results)


def _get_trading_dates(start_date, end_date):
    """获取交易日列表"""
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT DISTINCT trade_date
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.code = '000001.SH'
          AND dq.trade_date >= ? AND dq.trade_date <= ?
        ORDER BY trade_date
    """
    dates = [row[0] for row in conn.execute(query, (start_date, end_date))]
    conn.close()
    return dates


def main():
    parser = argparse.ArgumentParser(description='北极星指标快速评估')
    parser.add_argument('--generate-reports', action='store_true', help='生成选股报告')
    parser.add_argument('--backtest', action='store_true', help='运行回测')
    parser.add_argument('--compare', action='store_true', help='多模型对比')
    parser.add_argument('--report-dir', type=str, default=None, help='报告目录')
    parser.add_argument('--label', type=str, default='v3.95', help='标签名')
    parser.add_argument('--top-n', type=int, default=20, help='Top N选股')
    parser.add_argument('--benchmark', type=str, default='000905.SH', help='基准指数')
    parser.add_argument('--focus-days', type=int, default=10, help='重点持仓天数')
    parser.add_argument('--start-date', type=str, default='2025-09-01', help='开始日期')
    parser.add_argument('--end-date', type=str, default='2026-02-13', help='结束日期')
    parser.add_argument('--scoring-version', type=str, default='v3.95', help='评分版本')
    parser.add_argument('--retention-bonus', type=float, default=0.0,
                        help='持仓保留加分比例 (0.0-1.0)')
    args = parser.parse_args()

    if args.generate_reports:
        generate_reports(args.scoring_version, args.start_date, args.end_date)

    if args.backtest:
        if args.report_dir:
            run_backtest(args.report_dir, args.label, args.top_n, args.benchmark,
                        args.focus_days, args.retention_bonus)
        else:
            # 默认回测v3.95 RobustZScore
            default_dir = str(PROJECT_ROOT / 'reports' / 'daily_selection_v3.95_robust_zscore')
            run_backtest(default_dir, 'v3.95-RZ', args.top_n, args.benchmark,
                        args.focus_days, args.retention_bonus)

    if args.compare:
        run_comparison(args.top_n, args.benchmark, args.focus_days)

    if not args.generate_reports and not args.backtest and not args.compare:
        print("请指定操作: --generate-reports, --backtest, 或 --compare")
        parser.print_help()


if __name__ == '__main__':
    main()
