#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
北极星指标快速评估脚本 (V2 升级版)

在模型重训练后，快速生成报告、运行回测、输出评分卡。

用法:
    # 1. 模型训练完成后，生成报告 + 回测
    python3 backtest/run_north_star_eval.py --generate-reports --backtest

    # 2. 仅回测已有报告
    python3 backtest/run_north_star_eval.py --backtest --report-dir reports/daily_selection_v3.95_robust_zscore

    # 3. 多模型对比 (含V2评分卡)
    python3 backtest/run_north_star_eval.py --compare

    # 4. 扩展窗口回测 (合并多目录报告)
    python3 backtest/run_north_star_eval.py --extended \
        --report-dir reports/daily_selection_v4.3 \
        --extended-dir reports/daily_selection_v4.3_extended \
        --label "V4.3"

    # 5. 生成扩展期报告 (2024-01~2025-08)
    python3 backtest/run_north_star_eval.py --generate-extended --scoring-version v4.3

    # 6. 市况分析 (附加到回测结果)
    python3 backtest/run_north_star_eval.py --backtest --regime-analysis \
        --report-dir reports/daily_selection_v4.3

作者: Claude Code
创建时间: 2026-02-23
更新: 2026-02-24 (V2升级: 21项指标, 6档评分, 扩展窗口, 市况分析)
"""

import sys
import os
import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')


def generate_reports(scoring_version='v3.95', start_date='2025-09-01', end_date='2026-02-13'):
    """批量生成选股报告 (并行版)"""
    import subprocess

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
    """多模型对比 (含V2评分卡)"""
    from backtest import backtest_report_based as brb
    from backtest import north_star_metrics as nsm
    nsm.DB_PATH = DB_PATH
    brb.DB_PATH = DB_PATH

    report_dirs = {
        'v3.9': str(PROJECT_ROOT / 'reports' / 'daily_selection_v3.9'),
        'v3.95-RZ': str(PROJECT_ROOT / 'reports' / 'daily_selection_v3.95_robust_zscore'),
    }

    # 检查更多模型目录
    extra_dirs = {
        'v3.95-Phase2': 'daily_selection_v3.95_phase2',
        'v3.96-merged': 'daily_selection_v3.95_merged',
        'v4.3': 'daily_selection_v4.3',
        'v4.4': 'daily_selection_v4.4',
        'v5.0': 'daily_selection_v5.0',
    }
    for label, dirname in extra_dirs.items():
        dir_path = str(PROJECT_ROOT / 'reports' / dirname)
        if os.path.isdir(dir_path) and os.listdir(dir_path):
            report_dirs[label] = dir_path

    results = []
    for label, dir_path in report_dirs.items():
        if not os.path.isdir(dir_path):
            print(f"  跳过 {label}: 目录不存在")
            continue
        reports = brb.load_reports(dir_path)
        if not reports:
            print(f"  跳过 {label}: 无报告")
            continue
        print(f"\n{'#'*80}")
        result = brb.run_single_backtest(
            reports, label, top_n=top_n,
            benchmark_code=benchmark, focus_days=focus_days
        )
        if result:
            results.append(result)

    if len(results) >= 2:
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                brb.compare_results(results[i], results[j], focus_days)

        brb.generate_report(results, benchmark_code=benchmark, focus_days=focus_days)


def merge_report_dirs(dirs: list, merged_dir: str) -> int:
    """
    合并多个报告目录到一个目录

    Args:
        dirs: 源目录列表
        merged_dir: 目标合并目录

    Returns:
        合并后的文件总数
    """
    merged_path = Path(merged_dir)
    merged_path.mkdir(parents=True, exist_ok=True)

    total = 0
    existing = set(f.name for f in merged_path.iterdir())

    for src_dir in dirs:
        src_path = Path(src_dir)
        if not src_path.exists():
            print(f"  ⚠️ 源目录不存在: {src_dir}")
            continue

        for f in src_path.iterdir():
            if f.is_file() and f.name not in existing:
                shutil.copy2(f, merged_path / f.name)
                existing.add(f.name)
                total += 1

    print(f"  合并完成: {total} 个新文件 → {merged_dir} (总计 {len(existing)} 文件)")
    return len(existing)


def run_extended_backtest(report_dir, extended_dir, label, top_n=20,
                           benchmark='000905.SH', focus_days=10, retention_bonus=0.0):
    """
    扩展窗口回测: 合并现有报告+扩展期报告，运行V2评分

    Args:
        report_dir: 现有报告目录 (e.g. reports/daily_selection_v4.3)
        extended_dir: 扩展期报告目录 (e.g. reports/daily_selection_v4.3_extended)
        label: 标签名
    """
    from backtest import backtest_report_based as brb
    from backtest import north_star_metrics as nsm
    nsm.DB_PATH = DB_PATH
    brb.DB_PATH = DB_PATH

    # 合并目录
    merged_dir = str(Path(report_dir).parent / (Path(report_dir).name + '_merged_extended'))
    dirs_to_merge = [report_dir]
    if extended_dir and os.path.isdir(extended_dir):
        dirs_to_merge.append(extended_dir)

    n_files = merge_report_dirs(dirs_to_merge, merged_dir)
    if n_files == 0:
        print(f"  ⚠️ 合并后无报告")
        return None

    # 回测合并后的报告
    reports = brb.load_reports(merged_dir)
    if not reports:
        print(f"  ⚠️ 加载报告失败: {merged_dir}")
        return None

    print(f"\n  扩展回测: {label} ({len(reports)} 交易日)")
    result = brb.run_single_backtest(
        reports, f"{label} (扩展)", top_n=top_n,
        benchmark_code=benchmark, focus_days=focus_days,
        retention_bonus=retention_bonus
    )
    return result


def run_regime_analysis(report_dir, label, benchmark='000905.SH', focus_days=10, top_n=20):
    """
    市况分析: 分牛/熊/震荡计算IC/ICIR

    输出分市况指标作为回测报告的补充信息（不纳入V2评分卡）
    """
    from backtest import backtest_report_based as brb
    from backtest import north_star_metrics as nsm
    nsm.DB_PATH = DB_PATH
    brb.DB_PATH = DB_PATH

    # 先运行回测获取daily IC
    reports = brb.load_reports(report_dir)
    if not reports:
        print(f"  ⚠️ 无报告: {report_dir}")
        return

    result = brb.run_single_backtest(
        reports, label, top_n=top_n,
        benchmark_code=benchmark, focus_days=focus_days
    )
    if not result:
        return

    # 加载基准
    all_dates = sorted(reports.keys())
    benchmark_ret = nsm.load_benchmark_returns(
        benchmark, start_date=all_dates[0], end_date=all_dates[-1]
    )
    if benchmark_ret.empty:
        print("  ⚠️ 无基准数据")
        return

    # 分类市况
    regime = nsm.classify_market_regime(benchmark_ret)
    if regime.empty:
        print("  ⚠️ 无法分类市况 (数据不足)")
        return

    # 统计市况分布
    regime_counts = regime.value_counts()
    print(f"\n  {'═'*60}")
    print(f"  市况分析: {label} (基准: {benchmark})")
    print(f"  {'═'*60}")
    for r_name in ['bull', 'bear', 'neutral']:
        n = regime_counts.get(r_name, 0)
        pct = n / len(regime) * 100 if len(regime) > 0 else 0
        cn_name = {'bull': '牛市', 'bear': '熊市', 'neutral': '震荡'}[r_name]
        print(f"    {cn_name}: {n}天 ({pct:.1f}%)")

    # 分市况IC
    for days in [5, 10, 15]:
        ic_df = result.get('daily_ic_series', {}).get(days, None)
        if ic_df is None or ic_df.empty:
            continue

        regime_metrics = nsm.compute_regime_conditional_metrics(ic_df, regime)
        if not regime_metrics:
            continue

        print(f"\n  {days}日持仓 分市况IC:")
        for r_name in ['bull', 'bear', 'neutral']:
            m = regime_metrics.get(r_name, {})
            cn_name = {'bull': '牛市', 'bear': '熊市', 'neutral': '震荡'}[r_name]
            ic = m.get('ic', 0)
            icir = m.get('icir', 0)
            n = m.get('n_days', 0)
            ic_pos = m.get('ic_positive_pct', 0)
            print(f"    {cn_name}: IC={ic:+.4f}, ICIR={icir:+.3f}, IC>0={ic_pos:.0f}% ({n}天)")

    print(f"  {'═'*60}")


def generate_extended_reports(scoring_version='v3.95',
                               start_date='2024-01-01', end_date='2025-08-31'):
    """生成扩展期报告 (2024-01~2025-08)"""
    # 根据版本选择不同的生成方式
    if scoring_version in ('v3.95', 'v3.96'):
        # 使用batch_generate_v395_reports.py
        import subprocess
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / 'backtest' / 'batch_generate_v395_reports.py'),
            '--start-date', start_date,
            '--end-date', end_date,
            '--version', 'v3.95',
            '--output-dir', str(PROJECT_ROOT / 'reports' / f'daily_selection_{scoring_version}_extended'),
        ]
        print(f"  生成 {scoring_version} 扩展报告: {start_date} → {end_date}")
        print(f"  命令: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    elif scoring_version in ('v4.3', 'v4.4'):
        # 使用batch_generate_v395_reports.py (支持v4.3/v4.4)
        import subprocess
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / 'backtest' / 'batch_generate_v395_reports.py'),
            '--version', scoring_version,
            '--start-date', start_date,
            '--end-date', end_date,
            '--output-dir', str(PROJECT_ROOT / 'reports' / f'daily_selection_{scoring_version}_extended'),
        ]
        print(f"  生成 {scoring_version} 扩展报告: {start_date} → {end_date}")
        print(f"  命令: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    else:
        # 通用: 逐日生成
        generate_reports(scoring_version, start_date, end_date)


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
    parser = argparse.ArgumentParser(description='北极星指标快速评估 (V2)')
    parser.add_argument('--generate-reports', action='store_true', help='生成选股报告')
    parser.add_argument('--backtest', action='store_true', help='运行回测 (含V2评分卡)')
    parser.add_argument('--compare', action='store_true', help='多模型对比')
    parser.add_argument('--extended', action='store_true',
                        help='扩展窗口回测 (合并多目录)')
    parser.add_argument('--generate-extended', action='store_true',
                        help='生成扩展期报告 (2024-01~2025-08)')
    parser.add_argument('--regime-analysis', action='store_true',
                        help='市况分析 (分牛/熊/震荡)')
    parser.add_argument('--report-dir', type=str, default=None, help='报告目录')
    parser.add_argument('--extended-dir', type=str, default=None,
                        help='扩展期报告目录 (用于--extended)')
    parser.add_argument('--label', type=str, default='v3.95', help='标签名')
    parser.add_argument('--top-n', type=int, default=20, help='Top N选股')
    parser.add_argument('--benchmark', type=str, default='000905.SH', help='基准指数')
    parser.add_argument('--focus-days', type=int, default=10, help='重点持仓天数')
    parser.add_argument('--start-date', type=str, default='2025-09-01', help='开始日期')
    parser.add_argument('--end-date', type=str, default='2026-02-13', help='结束日期')
    parser.add_argument('--extended-start', type=str, default='2024-01-01',
                        help='扩展期开始日期')
    parser.add_argument('--extended-end', type=str, default='2025-08-31',
                        help='扩展期结束日期')
    parser.add_argument('--scoring-version', type=str, default='v3.95', help='评分版本')
    parser.add_argument('--retention-bonus', type=float, default=0.0,
                        help='持仓保留加分比例 (0.0-1.0)')
    args = parser.parse_args()

    if args.generate_reports:
        generate_reports(args.scoring_version, args.start_date, args.end_date)

    if args.generate_extended:
        generate_extended_reports(args.scoring_version, args.extended_start, args.extended_end)

    if args.backtest:
        if args.report_dir:
            result = run_backtest(args.report_dir, args.label, args.top_n, args.benchmark,
                                  args.focus_days, args.retention_bonus)
        else:
            # 默认回测v3.95 RobustZScore
            default_dir = str(PROJECT_ROOT / 'reports' / 'daily_selection_v3.95_robust_zscore')
            result = run_backtest(default_dir, 'v3.95-RZ', args.top_n, args.benchmark,
                                  args.focus_days, args.retention_bonus)

    if args.regime_analysis:
        report_dir = args.report_dir or str(
            PROJECT_ROOT / 'reports' / 'daily_selection_v3.95_robust_zscore')
        run_regime_analysis(report_dir, args.label, args.benchmark,
                            args.focus_days, args.top_n)

    if args.extended:
        if not args.report_dir:
            print("  ⚠️ --extended 需要 --report-dir")
        else:
            run_extended_backtest(
                args.report_dir, args.extended_dir, args.label,
                args.top_n, args.benchmark, args.focus_days, args.retention_bonus
            )

    if args.compare:
        run_comparison(args.top_n, args.benchmark, args.focus_days)

    if not any([args.generate_reports, args.generate_extended, args.backtest,
                args.compare, args.extended, args.regime_analysis]):
        print("请指定操作: --generate-reports, --backtest, --compare, --extended, "
              "--generate-extended, --regime-analysis")
        parser.print_help()


if __name__ == '__main__':
    main()
