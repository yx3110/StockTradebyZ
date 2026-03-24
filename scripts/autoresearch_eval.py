#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autoresearch 快速评估脚本

用于 autoresearch 自动迭代循环的评估脚本。
读取 autoresearch_params.json 中的参数，运行回测，输出单个数字指标。

用法:
    # 输出北极星V4加权总分 (0-100 scale)
    python3 scripts/autoresearch_eval.py

    # 指定报告目录
    python3 scripts/autoresearch_eval.py --report-dir reports/daily_selection_v4.7.5

    # 输出原始分数 (非百分比)
    python3 scripts/autoresearch_eval.py --raw

输出:
    仅输出一个数字到 stdout (北极星V4加权百分比分数)
    所有日志输出到 stderr
"""

import sys
import os
import json
import time
import hashlib
import pickle
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
CACHE_DIR = PROJECT_ROOT / 'scripts' / '.eval_cache'

# 默认参数 (可被 autoresearch_params.json 覆盖)
DEFAULT_PARAMS = {
    # === 回测参数 ===
    'report_dir': 'reports/daily_selection_v4.7.5',
    'label': 'autoresearch',
    'top_n': 10,
    'focus_days': 10,
    'benchmark': '000905.SH',
    'rank_field': 'auto',           # auto|score|composite|pred_10d|pred_5d
    'retention_bonus': 0.0,         # 0.0-0.5
    'score_floor': 0.0,             # 0-50
    'min_holdings': 3,              # 1-10
    'hold_buffer': 0,               # 0-3
    'cppi_floor': 0.0,              # 0.0-0.20
    'cppi_multiplier': 3.0,         # 1.0-5.0
    'sector_diversify': 0,          # 0=off, 2-4
    'vol_target': 0.0,              # 0=off, 0.10-0.20
    'risk_control': False,

    # === Composite 权重 (仅 rank_field=composite 时生效) ===
    'composite_weights': {
        'pred_3d': 0.00,
        'pred_5d': 0.00,
        'pred_10d': 0.60,
        'pred_15d': 0.40,
    },

    # === 评估输出 ===
    'metric': 'v4_weighted_pct',    # v2_score|v2_pct|v4_weighted_pct|sharpe|annual_return
}

PARAMS_FILE = PROJECT_ROOT / 'scripts' / 'autoresearch_params.json'


def load_params():
    """加载参数: 默认值 + autoresearch_params.json 覆盖"""
    params = DEFAULT_PARAMS.copy()

    if PARAMS_FILE.exists():
        with open(PARAMS_FILE) as f:
            overrides = json.load(f)
        # 深合并 composite_weights
        if 'composite_weights' in overrides:
            params['composite_weights'] = {**params['composite_weights'], **overrides.pop('composite_weights')}
        params.update(overrides)
        print(f"[eval] 已加载参数: {PARAMS_FILE}", file=sys.stderr)
    else:
        print(f"[eval] 使用默认参数 (无 {PARAMS_FILE.name})", file=sys.stderr)

    return params


def patch_composite_weights(weights):
    """动态修改 backtest_report_based 的 COMPOSITE_WEIGHTS"""
    from backtest import backtest_report_based as brb
    if hasattr(brb, 'COMPOSITE_WEIGHTS'):
        brb.COMPOSITE_WEIGHTS = weights
        print(f"[eval] Composite 权重已修改: {weights}", file=sys.stderr)


def _get_cache_key(report_dir, rank_field, composite_weights=None):
    """生成缓存键 (基于报告目录+排名方式+权重)"""
    key_str = f"{report_dir}|{rank_field}"
    if rank_field == 'composite' and composite_weights:
        key_str += f"|{json.dumps(composite_weights, sort_keys=True)}"
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


def _load_or_build_cache(report_dir, rank_field, composite_weights, benchmark_code):
    """加载或构建缓存 (reports + future_returns + market_data)

    缓存的数据在迭代间完全不变 (仅参数如top_n/score_floor变化)。
    首次运行 ~35s，后续 ~2s。
    """
    import io
    import contextlib
    from backtest import backtest_report_based as brb
    from backtest import north_star_metrics as nsm

    nsm.DB_PATH = DB_PATH
    brb.DB_PATH = DB_PATH

    cache_key = _get_cache_key(report_dir, rank_field, composite_weights)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f'eval_{cache_key}.pkl'

    # 检查缓存是否有效 (存在 + 比报告目录新)
    if cache_file.exists():
        report_dir_path = Path(report_dir)
        cache_mtime = cache_file.stat().st_mtime
        # 找最新的报告文件时间
        latest_report = max(
            (f.stat().st_mtime for f in report_dir_path.glob('analysis_data_*.json')),
            default=0)
        if cache_mtime > latest_report:
            t0 = time.time()
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            print(f"[eval] 缓存命中: {cache_file.name} ({time.time()-t0:.1f}s)", file=sys.stderr)
            # 恢复模块级缓存 (避免 run_single_backtest 重新查询DB)
            cache_key_tuple = tuple(sorted(cached['reports'].keys()))
            brb._future_returns_cache[cache_key_tuple] = cached['future_returns']
            brb._next_trading_dates_cache[cache_key_tuple] = cached['next_trading_dates']
            # 恢复 benchmark + metric_data 缓存
            if 'benchmark_cache' in cached:
                nsm._benchmark_cache.update(cached['benchmark_cache'])
            if 'metric_data_cache' in cached:
                nsm._metric_data_cache.update(cached['metric_data_cache'])
            return cached['reports']

    # 缓存未命中: 正常加载
    print(f"[eval] 构建缓存...", file=sys.stderr)
    t0 = time.time()

    if rank_field == 'composite' and composite_weights:
        patch_composite_weights(composite_weights)

    with contextlib.redirect_stdout(io.StringIO()):
        reports = brb.load_reports(report_dir, rank_field=rank_field)

    if not reports:
        return None

    # 预加载 future_returns (触发批量SQL查询)
    all_dates = sorted(reports.keys())
    future_returns = brb.batch_get_all_future_returns(all_dates, brb.HOLDING_DAYS)
    next_trading_dates = brb._batch_get_next_trading_dates(all_dates)

    # 存入模块级缓存
    cache_key_tuple = tuple(all_dates)
    brb._future_returns_cache[cache_key_tuple] = future_returns
    brb._next_trading_dates_cache[cache_key_tuple] = next_trading_dates

    # 预加载 metric data + benchmark (也缓存到磁盘)
    all_dates = sorted(reports.keys())
    # 触发 benchmark 加载
    nsm.load_benchmark_returns(benchmark_code, start_date=min(all_dates), end_date=max(all_dates))
    # 计算 buy_dates 并预加载 metric data
    from backtest.backtest_report_based import _batch_get_next_trading_dates
    buy_dates = sorted(set(
        next_trading_dates.get(d) for d in all_dates
        if next_trading_dates.get(d)
    ))
    if buy_dates:
        nsm.batch_load_all_metric_data(buy_dates)

    # 保存到磁盘
    cached = {
        'reports': reports,
        'future_returns': future_returns,
        'next_trading_dates': next_trading_dates,
        'benchmark_cache': nsm._benchmark_cache.copy(),
        'metric_data_cache': nsm._metric_data_cache.copy(),
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(cached, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[eval] 缓存已保存: {cache_file.name} ({time.time()-t0:.1f}s)", file=sys.stderr)

    return reports


def run_eval(params):
    """运行回测并返回指标"""
    import io
    import contextlib
    from backtest import backtest_report_based as brb
    from backtest import north_star_metrics as nsm

    nsm.DB_PATH = DB_PATH
    brb.DB_PATH = DB_PATH

    report_dir = params['report_dir']
    if not os.path.isabs(report_dir):
        report_dir = str(PROJECT_ROOT / report_dir)

    # 加载报告 (带磁盘缓存)
    reports = _load_or_build_cache(
        report_dir, params['rank_field'],
        params.get('composite_weights'),
        params['benchmark'])

    if not reports:
        print(f"[eval] ERROR: 无报告: {report_dir}", file=sys.stderr)
        return None

    all_dates = sorted(reports.keys())
    print(f"[eval] 报告: {len(all_dates)} 天 ({all_dates[0]} → {all_dates[-1]})", file=sys.stderr)

    # 运行回测 (将 backtest 的 stdout 重定向到 stderr)
    with contextlib.redirect_stdout(sys.stderr):
        result = brb.run_single_backtest(
            reports,
            params['label'],
            top_n=params['top_n'],
            benchmark_code=params['benchmark'],
            focus_days=params['focus_days'],
            retention_bonus=params['retention_bonus'],
            score_floor=params['score_floor'],
            min_holdings=params['min_holdings'],
            risk_control=params['risk_control'],
            vol_target=params['vol_target'],
            cppi_floor=params['cppi_floor'],
            cppi_multiplier=params['cppi_multiplier'],
            sector_diversify=params['sector_diversify'],
            hold_buffer=params['hold_buffer'],
        )

    if not result:
        print(f"[eval] ERROR: 回测失败", file=sys.stderr)
        return None

    # 提取指标
    summary = result.get('summary', {})
    focus = summary.get(params['focus_days'], {})

    metric_name = params['metric']

    if metric_name == 'v2_score':
        # V2 原始分数 (0-105)
        return _compute_v2_score(focus, len(all_dates))

    elif metric_name == 'v2_pct':
        # V2 百分比 (0-100)
        score = _compute_v2_score(focus, len(all_dates))
        return score / 105 * 100 if score is not None else None

    elif metric_name == 'v4_weighted_pct':
        # V4 加权百分比 (考虑层级权重, 0-100)
        return _compute_v4_weighted_pct(focus, len(all_dates))

    elif metric_name == 'sharpe':
        return focus.get('sharpe_ratio', 0)

    elif metric_name == 'annual_return':
        return focus.get('annual_return', 0) * 100  # 百分比

    else:
        print(f"[eval] WARNING: 未知指标 '{metric_name}', 使用 v4_weighted_pct", file=sys.stderr)
        return _compute_v4_weighted_pct(focus, len(all_dates))


def _compute_v2_score(s, n_trading_days):
    """计算 V2 北极星总分 (0-105)"""
    from backtest.north_star_metrics import NORTH_STAR_TARGETS_V2, score_metric_v2

    metric_value_map = {
        'daily_ic':              s.get('ic_mean', 0),
        'icir':                  s.get('icir', 0),
        'ic_positive_pct':       s.get('ic_positive_pct', 0),
        'ic_monotonicity':       s.get('ic_monotonicity', 0),
        'ic_time_stability':     s.get('ic_time_stability', 999),
        'signal_half_life':      s.get('signal_half_life', 0),
        'annual_turnover':       s.get('annual_turnover', 0),
        'annual_cost_drag':      s.get('annual_cost_drag', 0),
        'net_gross_ratio':       s.get('net_gross_ratio', 0),
        'limit_up_fail_rate':    s.get('limit_up_fail_rate', 0),
        'liquidity_coverage':    s.get('liquidity_coverage', 0),
        'max_drawdown':          s.get('max_drawdown', 0),
        'sharpe_ratio':          s.get('sharpe_ratio', 0),
        'sortino_ratio':         s.get('sortino_ratio', 0),
        'calmar_ratio':          s.get('calmar_ratio', 0),
        'worst_rolling_60d_icir': s.get('worst_rolling_60d_icir', None),
        'annual_return':         s.get('annual_return', 0),
        'monthly_win_rate':      s.get('monthly_win_rate', 0),
        'half_period_consistency': s.get('half_period_consistency', 0),
        'cap_balance_ratio':     s.get('cap_balance_ratio', 0),
        'median_market_cap_bn':  s.get('median_market_cap_bn', 0),
    }

    total_score = 0
    for metric_key, target_info in NORTH_STAR_TARGETS_V2.items():
        current = metric_value_map.get(metric_key)
        if current is None:
            continue
        score, _ = score_metric_v2(current, target_info)
        total_score += score

    return total_score


def _compute_v4_weighted_pct(s, n_trading_days):
    """计算 V4 加权百分比分数 (0-100)

    V4权重: L1=35%, L2=15%, L3=20%, L4=15%, L5=15%
    这里只算 L1-L4 (不需要超额收益数据), 权重重归一化
    L1=41.2%, L2=17.6%, L3=23.5%, L4=17.6%
    """
    from backtest.north_star_metrics import NORTH_STAR_TARGETS_V2, score_metric_v2

    metric_value_map = {
        'daily_ic':              s.get('ic_mean', 0),
        'icir':                  s.get('icir', 0),
        'ic_positive_pct':       s.get('ic_positive_pct', 0),
        'ic_monotonicity':       s.get('ic_monotonicity', 0),
        'ic_time_stability':     s.get('ic_time_stability', 999),
        'signal_half_life':      s.get('signal_half_life', 0),
        'annual_turnover':       s.get('annual_turnover', 0),
        'annual_cost_drag':      s.get('annual_cost_drag', 0),
        'net_gross_ratio':       s.get('net_gross_ratio', 0),
        'limit_up_fail_rate':    s.get('limit_up_fail_rate', 0),
        'liquidity_coverage':    s.get('liquidity_coverage', 0),
        'max_drawdown':          s.get('max_drawdown', 0),
        'sharpe_ratio':          s.get('sharpe_ratio', 0),
        'sortino_ratio':         s.get('sortino_ratio', 0),
        'calmar_ratio':          s.get('calmar_ratio', 0),
        'worst_rolling_60d_icir': s.get('worst_rolling_60d_icir', None),
        'annual_return':         s.get('annual_return', 0),
        'monthly_win_rate':      s.get('monthly_win_rate', 0),
        'half_period_consistency': s.get('half_period_consistency', 0),
        'cap_balance_ratio':     s.get('cap_balance_ratio', 0),
        'median_market_cap_bn':  s.get('median_market_cap_bn', 0),
    }

    # V4 layer weights (L1-L4 only, renormalized)
    layer_weights = {1: 0.412, 2: 0.176, 3: 0.235, 4: 0.176}

    layer_scores = {}
    layer_maxes = {}

    for metric_key, target_info in NORTH_STAR_TARGETS_V2.items():
        current = metric_value_map.get(metric_key)
        if current is None:
            continue
        layer = target_info['layer']
        score, _ = score_metric_v2(current, target_info)

        layer_scores[layer] = layer_scores.get(layer, 0) + score
        layer_maxes[layer] = layer_maxes.get(layer, 0) + 5

    # 加权百分比
    weighted_pct = 0
    for layer_id, weight in layer_weights.items():
        if layer_id in layer_scores and layer_maxes[layer_id] > 0:
            layer_pct = layer_scores[layer_id] / layer_maxes[layer_id]
            weighted_pct += weight * layer_pct

    return round(weighted_pct * 100, 2)


def main():
    parser = argparse.ArgumentParser(description='Autoresearch 评估脚本')
    parser.add_argument('--report-dir', help='报告目录 (覆盖 params.json)')
    parser.add_argument('--raw', action='store_true', help='输出原始分数')
    parser.add_argument('--metric', choices=['v2_score', 'v2_pct', 'v4_weighted_pct', 'sharpe', 'annual_return'],
                       help='指标类型 (覆盖 params.json)')
    parser.add_argument('--params', help='参数文件路径')
    args = parser.parse_args()

    # 加载参数
    global PARAMS_FILE
    if args.params:
        PARAMS_FILE = Path(args.params)

    params = load_params()

    # CLI 覆盖
    if args.report_dir:
        params['report_dir'] = args.report_dir
    if args.metric:
        params['metric'] = args.metric

    # 运行评估
    score = run_eval(params)

    if score is None:
        print("ERROR", file=sys.stderr)
        sys.exit(1)

    # 输出单个数字到 stdout
    if args.raw or params['metric'] == 'v2_score':
        print(f"{score:.0f}")
    else:
        print(f"{score:.2f}")


if __name__ == '__main__':
    main()
