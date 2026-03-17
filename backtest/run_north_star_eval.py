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


def generate_reports(scoring_version='v3.95', start_date='auto', end_date='auto'):
    """批量生成选股报告 (并行版)"""
    import subprocess

    # auto 日期: 从数据库获取最新可用范围
    if start_date == 'auto' or end_date == 'auto':
        all_dates = _get_trading_dates('2020-01-01', '2030-12-31')
        if all_dates:
            if start_date == 'auto':
                start_date = all_dates[0]
            if end_date == 'auto':
                end_date = all_dates[-1]
        else:
            print("  ⚠️ 无法从数据库检测日期范围")
            return

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
                 retention_bonus=0.0, score_floor=0.0, min_holdings=3,
                 risk_control=False,
                 vol_target=0.0, cppi_floor=0.0, cppi_multiplier=3.0,
                 sector_diversify=0, rank_field='auto', hold_buffer=0):
    """运行单个模型的回测"""
    from backtest import backtest_report_based as brb
    from backtest import north_star_metrics as nsm

    # 确保DB路径正确
    nsm.DB_PATH = DB_PATH
    brb.DB_PATH = DB_PATH

    reports = brb.load_reports(report_dir, rank_field=rank_field)
    if not reports:
        print(f"  ⚠️ 无报告: {report_dir}")
        return None

    # 打印评估窗口摘要
    all_dates = sorted(reports.keys())
    print(f"  评估窗口: {all_dates[0]} → {all_dates[-1]} ({len(all_dates)} 交易日)")

    result = brb.run_single_backtest(
        reports, label, top_n=top_n,
        benchmark_code=benchmark, focus_days=focus_days,
        retention_bonus=retention_bonus,
        score_floor=score_floor, min_holdings=min_holdings,
        risk_control=risk_control,
        vol_target=vol_target, cppi_floor=cppi_floor,
        cppi_multiplier=cppi_multiplier,
        sector_diversify=sector_diversify,
        hold_buffer=hold_buffer
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
        'v4.4.2': 'daily_selection_v4.4.2',
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
                           benchmark='000905.SH', focus_days=10, retention_bonus=0.0,
                           score_floor=0.0, min_holdings=3, risk_control=False,
                           vol_target=0.0, cppi_floor=0.0, cppi_multiplier=3.0,
                           sector_diversify=0, rank_field='auto', hold_buffer=0):
    """
    扩展窗口回测: 合并现有报告+扩展期报告，运行V2评分

    Args:
        report_dir: 现有报告目录 (e.g. reports/daily_selection_v4.3)
        extended_dir: 扩展期报告目录 (e.g. reports/daily_selection_v4.3_extended)
        label: 标签名
        rank_field: 排名字段 ('auto'=优先pred_10d, 'score'=全局百分位, 'composite'=多周期)
        hold_buffer: 持仓缓冲区倍数
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
    reports = brb.load_reports(merged_dir, rank_field=rank_field)
    if not reports:
        print(f"  ⚠️ 加载报告失败: {merged_dir}")
        return None

    ext_dates = sorted(reports.keys())
    print(f"\n  扩展回测: {label} ({len(reports)} 交易日)")
    print(f"  评估窗口: {ext_dates[0]} → {ext_dates[-1]} ({len(ext_dates)} 交易日)")
    result = brb.run_single_backtest(
        reports, f"{label} (扩展)", top_n=top_n,
        benchmark_code=benchmark, focus_days=focus_days,
        retention_bonus=retention_bonus,
        score_floor=score_floor, min_holdings=min_holdings,
        risk_control=risk_control,
        vol_target=vol_target, cppi_floor=cppi_floor,
        cppi_multiplier=cppi_multiplier,
        sector_diversify=sector_diversify,
        hold_buffer=hold_buffer
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
                               start_date='2024-01-01', end_date='auto'):
    """生成扩展期报告 (2024-01~扩展期结束)"""
    if end_date == 'auto':
        # 默认: 使用数据库中可用的最新日期
        all_dates = _get_trading_dates('2020-01-01', '2030-12-31')
        if all_dates:
            end_date = all_dates[-1]
        else:
            end_date = '2025-08-31'
            print(f"  ⚠️ 无法自动检测日期, 使用默认 {end_date}")
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


def _detect_report_date_range(report_dir):
    """
    自动检测报告目录中的日期范围

    扫描 analysis_data_*.json 和 选股分析报告_*.md 文件提取日期。

    Returns:
        (min_date, max_date, count) 或 (None, None, 0) 如果无报告
    """
    report_path = Path(report_dir)
    if not report_path.exists():
        return None, None, 0

    dates = set()

    # 扫描 JSON 报告
    for f in report_path.glob('analysis_data_*.json'):
        date_str = f.stem.replace('analysis_data_', '')
        if len(date_str) == 8 and date_str.isdigit():
            dates.add(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}")

    # 扫描 Markdown 报告
    for f in report_path.glob('选股分析报告_*.md'):
        date_str = f.stem.replace('选股分析报告_', '')
        if len(date_str) == 8 and date_str.isdigit():
            dates.add(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}")

    if not dates:
        return None, None, 0

    sorted_dates = sorted(dates)
    return sorted_dates[0], sorted_dates[-1], len(sorted_dates)


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
    parser.add_argument('--start-date', type=str, default='auto',
                        help='开始日期 (default: auto, 从报告目录检测)')
    parser.add_argument('--end-date', type=str, default='auto',
                        help='结束日期 (default: auto, 从报告目录检测)')
    parser.add_argument('--extended-start', type=str, default='2024-01-01',
                        help='扩展期开始日期')
    parser.add_argument('--extended-end', type=str, default='auto',
                        help='扩展期结束日期 (default: auto, 标准窗口start前一天)')
    parser.add_argument('--scoring-version', type=str, default='v3.95', help='评分版本')
    parser.add_argument('--retention-bonus', type=float, default=0.0,
                        help='持仓保留加分比例 (0.0-1.0)')
    parser.add_argument('--score-floor', type=float, default=0.0,
                        help='评分门槛 (Module J): 低于此分不入选，空位算现金 (default: 0)')
    parser.add_argument('--min-holdings', type=int, default=3,
                        help='最少持仓数 (default: 3)')
    parser.add_argument('--risk-control', action='store_true',
                        help='启用V4.4.2组合风控 (熊市减仓+行业集中度)')
    parser.add_argument('--vol-target', type=float, default=0.0,
                        help='V4.5: 年化波动率目标 (0=关闭, 推荐0.12)')
    parser.add_argument('--cppi-floor', type=float, default=0.0,
                        help='V4.5: CPPI最大回撤容忍度 (0=关闭, 推荐0.10)')
    parser.add_argument('--cppi-multiplier', type=float, default=3.0,
                        help='V4.5: CPPI乘数 (默认3.0)')
    parser.add_argument('--sector-diversify', type=int, default=0,
                        help='行业分散: 单行业最多N只 (0=关闭, 推荐2)')
    parser.add_argument('--rank-field', type=str, default='auto',
                        help='排名字段: auto=优先pred_10d, score=全局百分位, composite=多周期融合')
    parser.add_argument('--hold-buffer', type=float, default=0,
                        help='持仓缓冲区倍数 (0=关闭, 推荐2-3). 现有持仓在top_n*(1+buffer)内保留')
    parser.add_argument('--score-version', type=str, default='both', choices=['v2', 'v3', 'both'],
                        help='评分卡版本: v2=传统等权, v3=加权+统计鲁棒性, both=两者都打印 (default: both)')
    parser.add_argument('--n-trials', type=int, default=10,
                        help='DSR多重测试校正: 尝试过的策略变体数 (default: 10)')
    args = parser.parse_args()

    # ── auto 日期解析 ──
    # 如果指定了 --report-dir 且日期为 auto，从报告目录自动检测
    resolved_start = args.start_date
    resolved_end = args.end_date
    resolved_ext_end = args.extended_end

    if args.report_dir and (resolved_start == 'auto' or resolved_end == 'auto'):
        min_d, max_d, n = _detect_report_date_range(args.report_dir)
        if n > 0:
            if resolved_start == 'auto':
                resolved_start = min_d
            if resolved_end == 'auto':
                resolved_end = max_d
            print(f"  📅 自动检测报告日期: {min_d} → {max_d} ({n} 份报告)")
        else:
            print(f"  ⚠️ 无法从 {args.report_dir} 检测日期范围, 使用数据库最新")
            all_dates = _get_trading_dates('2020-01-01', '2030-12-31')
            if all_dates:
                if resolved_start == 'auto':
                    resolved_start = all_dates[0]
                if resolved_end == 'auto':
                    resolved_end = all_dates[-1]

    # extended-end auto: 标准窗口 start 的前一天
    if resolved_ext_end == 'auto' and resolved_start != 'auto':
        from datetime import datetime as dt, timedelta
        try:
            start_dt = dt.strptime(resolved_start, '%Y-%m-%d')
            ext_end_dt = start_dt - timedelta(days=1)
            resolved_ext_end = ext_end_dt.strftime('%Y-%m-%d')
        except ValueError:
            resolved_ext_end = '2025-08-31'

    if args.generate_reports:
        generate_reports(args.scoring_version, resolved_start, resolved_end)

    if args.generate_extended:
        generate_extended_reports(args.scoring_version, args.extended_start, resolved_ext_end)

    if args.backtest:
        if args.report_dir:
            result = run_backtest(args.report_dir, args.label, args.top_n, args.benchmark,
                                  args.focus_days, args.retention_bonus,
                                  args.score_floor, args.min_holdings,
                                  args.risk_control,
                                  args.vol_target, args.cppi_floor, args.cppi_multiplier,
                                  args.sector_diversify, args.rank_field,
                                  args.hold_buffer)
        else:
            # 默认回测v3.95 RobustZScore
            default_dir = str(PROJECT_ROOT / 'reports' / 'daily_selection_v3.95_robust_zscore')
            result = run_backtest(default_dir, 'v3.95-RZ', args.top_n, args.benchmark,
                                  args.focus_days, args.retention_bonus,
                                  args.score_floor, args.min_holdings,
                                  args.risk_control,
                                  args.vol_target, args.cppi_floor, args.cppi_multiplier,
                                  args.sector_diversify, args.rank_field,
                                  args.hold_buffer)

    if args.regime_analysis:
        report_dir = args.report_dir or str(
            PROJECT_ROOT / 'reports' / 'daily_selection_v3.95_robust_zscore')
        run_regime_analysis(report_dir, args.label, args.benchmark,
                            args.focus_days, args.top_n)

    if args.extended:
        if not args.report_dir:
            print("  ⚠️ --extended 需要 --report-dir")
        else:
            # 如果有 extended-dir，也检测其日期范围
            if args.extended_dir and resolved_ext_end == 'auto':
                _, ext_max, ext_n = _detect_report_date_range(args.extended_dir)
                if ext_n > 0:
                    resolved_ext_end = ext_max
                    print(f"  📅 扩展报告日期检测: → {ext_max} ({ext_n} 份)")

            run_extended_backtest(
                args.report_dir, args.extended_dir, args.label,
                args.top_n, args.benchmark, args.focus_days, args.retention_bonus,
                args.score_floor, args.min_holdings, args.risk_control,
                args.vol_target, args.cppi_floor, args.cppi_multiplier,
                args.sector_diversify, args.rank_field, args.hold_buffer
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
