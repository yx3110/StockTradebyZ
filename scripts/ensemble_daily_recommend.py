#!/usr/bin/env python3
"""
多模型综合选股推荐 (基于北极星V4评分选择最优模型)

用法:
  # 自动选择北极星V4最优的4个模型，生成综合推荐
  python3 scripts/ensemble_daily_recommend.py --date 2026-03-18

  # 指定模型数量
  python3 scripts/ensemble_daily_recommend.py --date 2026-03-18 --top-models 3

  # 跳过北极星评估，直接指定模型
  python3 scripts/ensemble_daily_recommend.py --date 2026-03-18 --models v4.7.3 v4.7.5 v4.7.6 v4.7.7
"""

import argparse
import json
import os
import subprocess
import sys
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
CACHE_FILE = PROJECT_ROOT / 'reports' / 'ensemble_recommend' / 'north_star_v4_cache.json'

# ── 版本 → merged_extended报告目录的特殊映射 (目录名与版本不一致的) ──
_REPORT_DIR_OVERRIDES = {
    'v4.4':   'daily_selection_v4.4_v2_merged_extended',
    'v4.4.2': 'daily_selection_v4.4_v2_merged_extended',
}

# 版本 → scorer参数的特殊映射 (选股脚本参数与版本不一致的)
_SCORER_OVERRIDES = {
    'v4.4': 'v4.4.2',
}

# 弃用版本 (不参与自动评估)
_DEPRECATED_VERSIONS = {
    'v2', 'v3', 'v3.1', 'v3.2', 'v3.3', 'v3.4', 'v3.41',
    'v3.5', 'v3.51', 'v3.52', 'v3.53', 'v3.6', 'v3.7',
    'v3.8', 'v3.81', 'v3.94', 'v3.95',
    'v4', 'v4.0', 'v4.2',
}

# 最低报告数量 (少于此数不做V4评估)
MIN_REPORT_COUNT = 100


def discover_available_models() -> dict:
    """
    从 tomorrow_stock_selector.py 的 --scoring-version choices 自动发现所有可用模型。

    Returns:
        dict: {version: report_dir_or_None} — report_dir 为 merged_extended 目录名，
              如果没有回测数据则为 None
    """
    selector_path = PROJECT_ROOT / 'tomorrow_stock_selector.py'
    try:
        content = selector_path.read_text()
        # 提取 choices=[...] 列表
        match = re.search(r"choices=\[(.*?)\]", content, re.DOTALL)
        if not match:
            print("  ⚠️ 无法从 tomorrow_stock_selector.py 解析版本列表")
            return {}
        choices_str = match.group(1)
        versions = re.findall(r"'([^']+)'", choices_str)
    except Exception as e:
        print(f"  ⚠️ 读取选股脚本失败: {e}")
        return {}

    # 过滤弃用版本
    versions = [v for v in versions if v not in _DEPRECATED_VERSIONS]

    # 查找报告目录 (优先 merged_extended, 其次主目录)
    reports_root = PROJECT_ROOT / 'reports'
    candidates = {}

    for version in versions:
        report_dir = None
        best_count = 0

        # 候选目录列表 (优先级从高到低)
        if version in _REPORT_DIR_OVERRIDES:
            search_dirs = [_REPORT_DIR_OVERRIDES[version]]
        else:
            search_dirs = [
                f'daily_selection_{version}_merged_extended',
                f'daily_selection_{version}',
            ]

        for dirname in search_dirs:
            full_path = reports_root / dirname
            if full_path.is_dir():
                json_count = len(list(full_path.glob('analysis_data_*.json')))
                if json_count >= MIN_REPORT_COUNT and json_count > best_count:
                    report_dir = dirname
                    best_count = json_count

        if report_dir:
            candidates[version] = report_dir

    return candidates


def get_scorer_version(version: str) -> str:
    """获取 tomorrow_stock_selector.py 的 --scoring-version 参数。"""
    return _SCORER_OVERRIDES.get(version, version)


def _load_v4_cache() -> dict:
    """加载北极星V4评分缓存。"""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_v4_cache(cache: dict):
    """保存北极星V4评分缓存。"""
    CACHE_FILE.parent.mkdir(exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def evaluate_north_star_v4(version: str, report_dir: str,
                            top_n: int = 10, focus_days: int = 10,
                            min_turnover_rate: float = 1.0,
                            replace_threshold: float = 0.05) -> dict:
    """
    运行北极星V4评估，返回加权百分比分数。

    Args:
        min_turnover_rate: 最低换手率过滤 (默认1.0%, 剔除不可交易的低流动性股票)
        replace_threshold: 替换门槛 (默认0.05, 新股评分需高出旧持仓5%才替换, 降低换手率)
    """
    full_dir = str(PROJECT_ROOT / 'reports' / report_dir)
    if not os.path.isdir(full_dir):
        return None

    # 检查是否有足够的数据文件
    json_count = len(list(Path(full_dir).glob('analysis_data_*.json')))
    if json_count < 100:
        print(f"  ⚠️ {version}: 仅 {json_count} 个数据文件，跳过")
        return None

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from backtest import backtest_report_based as brb
        from backtest import north_star_metrics as nsm

        nsm.DB_PATH = DB_PATH
        brb.DB_PATH = DB_PATH

        reports = brb.load_reports(full_dir, rank_field='composite')
        if not reports or len(reports) < 100:
            return None

        result = brb.run_single_backtest(
            reports, version, top_n=top_n,
            benchmark_code='000905.SH', focus_days=focus_days,
            cppi_floor=0.05, cppi_multiplier=20,
            min_turnover_rate=min_turnover_rate,
            replace_threshold=replace_threshold,
        )

        if not result:
            return None

        # 从result中提取V4评分需要的指标
        summary = result.get('summary', {})
        s = summary.get(focus_days, {})
        if not s:
            return None

        # 计算V4分数
        ic_mono_val = s.get('ic_monotonicity_v3')
        if ic_mono_val is None or ic_mono_val == 0:
            ic_mono_val = s.get('ic_monotonicity', 0)

        metric_value_map = {
            'daily_ic': s.get('ic_mean', 0),
            'icir': s.get('icir', 0),
            'ic_positive_pct': s.get('ic_positive_pct', 0),
            'ic_monotonicity': ic_mono_val,
            'ic_time_stability': s.get('ic_time_stability', 999),
            'signal_half_life': s.get('signal_half_life', 0),
            'bear_icir': s.get('bear_icir'),
            'ic_decay_ratio': s.get('ic_decay_ratio', 0),
            'annual_turnover': s.get('annual_turnover', 0),
            'annual_cost_drag': s.get('annual_cost_drag', 0),
            'net_gross_ratio': s.get('net_gross_ratio', 0),
            'limit_up_fail_rate': s.get('limit_up_fail_rate', 0),
            'liquidity_coverage': s.get('liquidity_coverage', 0),
            'max_drawdown': s.get('max_drawdown', 0),
            'sharpe_ratio': s.get('sharpe_ratio', 0),
            'sortino_ratio': s.get('sortino_ratio', 0),
            'calmar_ratio': s.get('calmar_ratio', 0),
            'worst_rolling_60d_icir': s.get('worst_rolling_60d_icir'),
            'tail_ratio': s.get('tail_ratio', 0),
            'max_consecutive_loss_periods': s.get('max_consecutive_loss_periods', 0),
            'annual_return': s.get('annual_return', 0),
            'monthly_win_rate': s.get('monthly_win_rate', 0),
            'half_period_consistency': s.get('half_period_consistency', 0),
            'probabilistic_sharpe': s.get('probabilistic_sharpe', 0),
            'deflated_sharpe': s.get('deflated_sharpe', 0),
            'excess_annual_return': s.get('excess_annual_return', 0),
            'information_ratio': s.get('information_ratio', 0),
            'excess_win_rate': s.get('excess_win_rate', 0),
            'excess_max_drawdown': s.get('excess_max_drawdown', 0),
            'bear_excess_return': s.get('bear_excess_return'),
            'up_capture_ratio': s.get('up_capture_ratio', 0),
        }

        n_trading_days = len(reports)
        v4_result = nsm.compute_v4_score(metric_value_map, n_trading_days)

        return {
            'version': version,
            'final_pct': v4_result['final_pct'],
            'raw_pct': v4_result['raw_pct'],
            'grade': v4_result['grade'],
            'total_score': v4_result['total_score'],
            'max_score': v4_result['max_score'],
            'n_days': n_trading_days,
            'layer_details': v4_result['layer_details'],
        }

    except Exception as e:
        print(f"  ❌ {version} 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def select_top_models(top_n: int = 4, force_refresh: bool = False) -> list:
    """评估所有候选模型，返回北极星V4分数最高的top_n个版本。

    自动从 tomorrow_stock_selector.py 发现可用模型版本。
    使用缓存避免重复评估。缓存基于报告目录中JSON文件数量判断是否过期。
    """
    model_candidates = discover_available_models()
    print("=" * 70)
    print(f"🔍 北极星V4评估: 发现 {len(model_candidates)} 个可评估模型，选择 Top {top_n}")
    print("=" * 70)

    cache = {} if force_refresh else _load_v4_cache()
    results = []
    evaluated = 0

    for version, report_dir in model_candidates.items():
        full_dir = str(PROJECT_ROOT / 'reports' / report_dir)
        json_count = len(list(Path(full_dir).glob('analysis_data_*.json'))) if os.path.isdir(full_dir) else 0

        # 检查缓存: 版本存在且报告数量未变 → 使用缓存
        cached = cache.get(version)
        if cached and cached.get('n_days', 0) == json_count and not force_refresh:
            results.append(cached)
            print(f"  📦 {version}: V4加权={cached['final_pct']:.1f}% "
                  f"({cached['grade']}) [缓存, {cached['n_days']}天]")
            continue

        print(f"\n📊 评估 {version}... ({json_count} 个数据文件)")
        result = evaluate_north_star_v4(version, report_dir)
        if result:
            results.append(result)
            cache[version] = result
            evaluated += 1
            # 增量保存缓存 (每完成一个模型就保存)
            _save_v4_cache(cache)
            print(f"  ✅ {version}: V4加权={result['final_pct']:.1f}% "
                  f"({result['grade']}) "
                  f"[{result['total_score']}/{result['max_score']}] "
                  f"({result['n_days']}天)")
        else:
            print(f"  ⚠️ {version}: 无法评估")

    if evaluated > 0:
        print(f"\n  💾 缓存已更新 ({evaluated} 个新评估)")

    # 按V4加权分数排序
    results.sort(key=lambda x: x['final_pct'], reverse=True)

    print(f"\n{'=' * 70}")
    print(f"🏆 北极星V4排名:")
    print(f"{'=' * 70}")
    print(f"{'排名':<5}{'版本':<10}{'V4加权%':<10}{'等级':<6}{'得分':<12}{'天数':<8}"
          f"{'L1信号':<10}{'L2效率':<10}{'L3风控':<10}{'L4鲁棒':<10}{'L5超额':<10}")
    print("-" * 105)

    for i, r in enumerate(results):
        ld = r.get('layer_details', {})
        l_strs = []
        for layer_id in [1, 2, 3, 4, 5]:
            # JSON keys are strings, Python keys are ints
            d = ld.get(layer_id) or ld.get(str(layer_id), {})
            l_strs.append(f"{d.get('score', 0)}/{d.get('max', 0)}")

        marker = "⭐" if i < top_n else ""
        print(f"{i+1:<5}{r['version']:<10}{r['final_pct']:<10.1f}{r['grade']:<6}"
              f"{r['total_score']}/{r['max_score']:<10}"
              f"{'  '.join(l_strs[:2]):>18}  {'  '.join(l_strs[2:]):>28} {marker}")

    selected = [r['version'] for r in results[:top_n]]
    print(f"\n✅ 选中模型: {', '.join(selected)}")
    return selected


def generate_fullmarket_report(version: str, date: str) -> bool:
    """为指定版本生成全市场报告。"""
    scorer = get_scorer_version(version)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / 'tomorrow_stock_selector.py'),
        date,
        '--scoring-version', scorer,
        '--full-market',
    ]
    print(f"  🔄 生成 {version} 全市场报告...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  ❌ {version} 报告生成失败: {result.stderr[-200:]}")
        return False
    print(f"  ✅ {version} 报告生成完成")
    return True


def parse_report_top_stocks(version: str, date_str: str, top_n: int = 20) -> list:
    """从报告中解析Top N股票。"""
    scorer = get_scorer_version(version)
    # 查找报告目录
    report_dir = PROJECT_ROOT / 'reports' / f'daily_selection_{scorer.replace(".", ".")}_fullmarket'
    report_file = report_dir / f'选股分析报告_{date_str}.md'

    if not report_file.exists():
        # 尝试其他命名模式
        for d in PROJECT_ROOT.glob(f'reports/daily_selection_*{scorer.replace(".", "")}*_fullmarket'):
            f = d / f'选股分析报告_{date_str}.md'
            if f.exists():
                report_file = f
                break

    if not report_file.exists():
        print(f"  ⚠️ 找不到 {version} 的报告: {report_file}")
        return []

    with open(report_file, 'r') as f:
        content = f.read()

    stocks = []
    for line in content.split('\n'):
        if not line.startswith('|'):
            continue
        parts = [p.strip() for p in line.split('|')]
        parts = [p for p in parts if p]
        if len(parts) < 7:
            continue
        try:
            rank = int(parts[0])
        except (ValueError, IndexError):
            continue
        if rank > top_n:
            continue

        code = parts[1]
        name = parts[2]
        strategy = parts[3]
        composite = parts[4]
        advice = parts[5]
        pred_10d = parts[6]

        stocks.append({
            'rank': rank,
            'code': code,
            'name': name,
            'strategy': strategy,
            'composite': composite,
            'pred_10d': pred_10d,
        })

    return stocks


def cross_reference_and_recommend(all_results: dict, top_n: int = 10,
                                   v4_scores: dict = None):
    """最佳模型+共识≥2: 用V4最佳模型排名选股，但要求至少被2个模型同时看好。

    回测验证此方案 Sharpe 最优 (1.536, 年化+64.9%)，优于等权(1.453)和纯单模型(1.511)。
    """
    all_stocks = {}

    for version, stocks in all_results.items():
        for s in stocks:
            code = s['code']
            if code not in all_stocks:
                all_stocks[code] = {
                    'name': s['name'],
                    'versions': {},
                    'count': 0,
                }
            all_stocks[code]['versions'][version] = {
                'rank': s['rank'],
                'pred_10d': s['pred_10d'],
                'composite': s['composite'],
                'strategy': s['strategy'],
            }
            all_stocks[code]['count'] += 1

    versions = list(all_results.keys())
    n_models = len(versions)

    # 确定V4最佳模型 (第一个版本，因为select_top_models已按V4分数排序)
    if v4_scores:
        best_model = max(v4_scores, key=v4_scores.get)
    else:
        best_model = versions[0]

    # 最佳模型+共识≥2: 只保留被≥2个模型看好的股票，按最佳模型排名排序
    consensus_stocks = [(code, info) for code, info in all_stocks.items()
                        if info['count'] >= 2]

    def sort_key(item):
        code, info = item
        # 按最佳模型的排名排序 (如果最佳模型没选中，用平均排名兜底)
        best_info = info['versions'].get(best_model)
        if best_info:
            return best_info['rank']
        else:
            return 100 + sum(v['rank'] for v in info['versions'].values()) / len(info['versions'])

    sorted_stocks = sorted(consensus_stocks, key=sort_key)

    # 如果共识股票不够top_n，补充最佳模型独有的高排名股票
    if len(sorted_stocks) < top_n:
        selected_codes = {code for code, _ in sorted_stocks}
        best_only = [(code, info) for code, info in all_stocks.items()
                     if code not in selected_codes and best_model in info['versions']]
        best_only.sort(key=lambda x: x[1]['versions'][best_model]['rank'])
        sorted_stocks.extend(best_only[:top_n - len(sorted_stocks)])

    print(f"\n{'=' * 120}")
    print(f"🎯 综合推荐 Top {top_n} (最佳模型+共识≥2, 主排序={best_model})")
    print(f"  模型: {', '.join(versions)}")
    print(f"{'=' * 120}")

    header = f"{'排名':<5}{'代码':<12}{'名称':<12}{'命中':<6}{'均排':<7}"
    for v in versions:
        header += f"{v:<14}"
    header += "策略"
    print(header)
    print("-" * 120)

    recommended = []
    for i, (code, info) in enumerate(sorted_stocks[:top_n * 2]):
        if i >= top_n * 2:
            break
        avg_rank = sum(v['rank'] for v in info['versions'].values()) / len(info['versions'])

        cols = []
        strategies = set()
        for v in versions:
            vi = info['versions'].get(v)
            if vi:
                cols.append(f"#{vi['rank']} {vi['pred_10d']}")
                if vi['strategy'] and vi['strategy'] != '-':
                    strategies.add(vi['strategy'])
            else:
                cols.append('-')

        marker = '⭐' if info['count'] >= n_models - 1 else ('✅' if info['count'] >= 2 else '')
        strat_str = ','.join(strategies) if strategies else '-'
        if len(strat_str) > 20:
            strat_str = strat_str[:18] + '..'

        line = f"{i+1:<5}{code:<12}{info['name']:<12}{info['count']}{marker:<5}{avg_rank:<7.1f}"
        for c in cols:
            line += f"{c:<14}"
        line += strat_str
        print(line)

        if i < top_n:
            recommended.append({
                'rank': i + 1,
                'code': code,
                'name': info['name'],
                'hit_count': info['count'],
                'avg_rank': avg_rank,
                'versions': info['versions'],
                'strategies': list(strategies),
            })

    return recommended


def main():
    parser = argparse.ArgumentParser(description='多模型综合选股推荐 (北极星V4自动选模型)')
    parser.add_argument('--date', type=str, required=True,
                        help='分析日期 (YYYY-MM-DD)')
    parser.add_argument('--top-models', type=int, default=4,
                        help='选择北极星V4最优的N个模型 (默认4)')
    parser.add_argument('--top-stocks', type=int, default=20,
                        help='每个模型取Top N股票 (默认20)')
    parser.add_argument('--recommend-n', type=int, default=10,
                        help='最终推荐Top N (默认10)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='直接指定模型版本，跳过北极星评估')
    parser.add_argument('--skip-report-gen', action='store_true',
                        help='跳过报告生成（使用已有报告）')
    parser.add_argument('--force-refresh', action='store_true',
                        help='强制重新评估所有模型（忽略缓存）')
    args = parser.parse_args()

    date_str = args.date.replace('-', '')  # YYYY-MM-DD → YYYYMMDD
    date_display = args.date

    # Step 1: 选择模型
    if args.models:
        selected_models = args.models
        print(f"📌 使用指定模型: {', '.join(selected_models)}")
    else:
        selected_models = select_top_models(top_n=args.top_models,
                                            force_refresh=args.force_refresh)

    if not selected_models:
        print("❌ 没有可用的模型")
        sys.exit(1)

    # Step 2: 生成全市场报告
    if not args.skip_report_gen:
        print(f"\n{'=' * 70}")
        print(f"📝 为 {len(selected_models)} 个模型生成 {date_display} 全市场报告")
        print(f"{'=' * 70}")

        for version in selected_models:
            success = generate_fullmarket_report(version, date_display)
            if not success:
                print(f"  ⚠️ {version} 报告生成失败，继续其他模型")

    # Step 3: 解析报告
    print(f"\n{'=' * 70}")
    print(f"📊 解析各模型 Top {args.top_stocks} 股票")
    print(f"{'=' * 70}")

    all_results = {}
    for version in selected_models:
        stocks = parse_report_top_stocks(version, date_str, top_n=args.top_stocks)
        if stocks:
            all_results[version] = stocks
            print(f"  ✅ {version}: {len(stocks)} 只股票")
        else:
            print(f"  ⚠️ {version}: 未找到股票数据")

    if len(all_results) < 2:
        print("❌ 有效模型不足2个，无法交叉验证")
        sys.exit(1)

    # Step 4: 交叉对比 (最佳模型+共识≥2)
    # 从缓存获取V4分数用于确定最佳模型
    cache = _load_v4_cache()
    v4_scores = {v: cache[v]['final_pct'] for v in all_results if v in cache}
    recommended = cross_reference_and_recommend(
        all_results, top_n=args.recommend_n, v4_scores=v4_scores)

    # Step 5: 保存结果
    output_dir = PROJECT_ROOT / 'reports' / 'ensemble_recommend'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f'综合推荐_{date_str}.json'

    best_model = max(v4_scores, key=v4_scores.get) if v4_scores else list(all_results.keys())[0]
    output_data = {
        'date': date_display,
        'models': list(all_results.keys()),
        'best_model': best_model,
        'method': 'best_model+consensus>=2',
        'north_star_version': 'v4',
        'recommended': recommended,
        'generated_at': datetime.now().isoformat(),
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果已保存: {output_file}")

    return recommended


if __name__ == '__main__':
    main()
