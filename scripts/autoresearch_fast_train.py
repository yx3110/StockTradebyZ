#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autoresearch Phase 2/3: 快速代理训练 + 报告生成 + 评估

Phase 2 (fast): 跳过WF, 3年数据, ~30-45min → 报告 → 评估
Phase 3 (full): 完整WF, 6年数据, ~6h → 报告 → 评估

用法:
    # Phase 2: 快速训练+评估 (输出单个数字)
    python3 scripts/autoresearch_fast_train.py

    # Phase 3: 全量训练验证
    python3 scripts/autoresearch_fast_train.py --full

    # 跳过训练, 只生成报告+评估 (指定已有模型)
    python3 scripts/autoresearch_fast_train.py --skip-train --model-path ml_models/trained_models/v475/xxx.pkl
"""

import sys
import os
import json
import re
import time
import subprocess
import shutil
import contextlib
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')

TRAIN_PARAMS_FILE = PROJECT_ROOT / 'scripts' / 'autoresearch_train_params.json'
BACKTEST_PARAMS_FILE = PROJECT_ROOT / 'scripts' / 'autoresearch_params.json'

DEFAULT_TRAIN_PARAMS = {
    'start_date': '2022-01-01',
    'end_date': '2026-02-13',
    'purge_days': 15,
    'sharpe_blend': 0.30,
    'extra_prune_features': [],
    'mode': 'fast',
    'output_version': 'v475_exp',
}


def load_train_params():
    params = DEFAULT_TRAIN_PARAMS.copy()
    if TRAIN_PARAMS_FILE.exists():
        with open(TRAIN_PARAMS_FILE) as f:
            overrides = json.load(f)
        params.update(overrides)
        print(f"[train] 已加载: {TRAIN_PARAMS_FILE}", file=sys.stderr)
    return params


def train_model(params):
    """用 subprocess 调用训练脚本，返回模型路径"""
    train_script = str(PROJECT_ROOT / 'ml_models' / 'training' / 'train_v395_multi_target.py')

    train_version = params.get('train_version', '--v475')
    cmd = [
        sys.executable, train_script,
        train_version,
        '--start-date', params['start_date'],
        '--end-date', params['end_date'],
        '--purge-days', str(params['purge_days']),
        '--sharpe-blend', str(params.get('sharpe_blend', 0.30)),
    ]

    if params['mode'] == 'fast':
        cmd.append('--skip-wf')
        print(f"[train] 快速模式: --skip-wf", file=sys.stderr)

    print(f"[train] CMD: {' '.join(cmd)}", file=sys.stderr)

    start = time.time()
    result = subprocess.run(cmd, capture_output=False, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - start

    print(f"[train] 完成: {elapsed/60:.1f}min, exit={result.returncode}", file=sys.stderr)

    if result.returncode != 0:
        return None

    # Find latest model — check version-specific dirs first, then fallback
    version_tag = train_version.lstrip('-').lstrip('-')  # '--v486' -> 'v486'
    search_dirs = [version_tag]
    # V486 inherits V484 which inherits V481 — model is saved by rename chain
    # During fast mode (skip-wf), model lands in parent dir first
    if version_tag in ('v486', 'v487'):
        search_dirs = [version_tag, 'v487', 'v486', 'v485', 'v484', 'v481', 'v47']
    elif version_tag == 'v475':
        search_dirs = ['v475', 'v47']
    else:
        search_dirs = [version_tag, 'v47', 'v475']
    for search_dir in search_dirs:
        model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / search_dir
        if model_dir.exists():
            models = sorted(model_dir.glob('*.pkl'), key=lambda f: f.stat().st_mtime, reverse=True)
            if models:
                latest = models[0]
                # Only return if created within last hour
                age = time.time() - latest.stat().st_mtime
                if age < 7200:  # 2 hours
                    print(f"[train] 模型: {latest} ({latest.stat().st_size/1024/1024:.1f}MB)", file=sys.stderr)
                    return str(latest)

    print(f"[train] ERROR: 未找到新模型", file=sys.stderr)
    return None


def generate_reports(model_path, params):
    """使用 batch_generate 生成报告"""
    output_version = params['output_version']
    report_dir = PROJECT_ROOT / 'reports' / f'daily_selection_{output_version}'

    # Clear old reports for fresh eval
    if report_dir.exists():
        shutil.rmtree(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Copy model to expected location so scorer can find it
    # For V486: scorer looks in v486/ for v486_*.pkl
    batch_version = params.get('batch_version', 'v4.7.5')
    scorer_dir_name = batch_version.replace('v', 'v').replace('.', '')  # 'v4.8.6' -> 'v486'
    scorer_dir_name = scorer_dir_name.replace('v486', 'v486').replace('v475', 'v475')  # keep as-is
    # Map batch_version to scorer directory name
    ver_map = {'v4.8.6': 'v486', 'v4.8.5': 'v485', 'v4.8.4': 'v484', 'v4.7.5': 'v475'}
    scorer_dir = ver_map.get(batch_version, output_version)

    exp_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / scorer_dir
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Rename model file to match scorer's glob pattern (e.g., v486_*.pkl)
    src_name = Path(model_path).name
    model_prefix = f'{scorer_dir}_multi_target_'
    if not src_name.startswith(model_prefix):
        # Extract timestamp from original name (e.g., v47_multi_target_20260324_040017.pkl)
        ts_match = re.search(r'(\d{8}_\d{6})', src_name)
        ts = ts_match.group(1) if ts_match else datetime.now().strftime('%Y%m%d_%H%M%S')
        target_name = f'{model_prefix}{ts}.pkl'
    else:
        target_name = src_name

    target_model = exp_dir / target_name
    if not target_model.exists():
        shutil.copy2(model_path, target_model)
        print(f"[reports] 模型已复制: {target_model.name}", file=sys.stderr)

    # Also copy auxiliary files if available
    src_dir = Path(model_path).parent
    for aux in ['global_quantiles.npy', 'recommendation_thresholds.json']:
        aux_src = src_dir / aux
        aux_dst = exp_dir / aux
        if aux_src.exists() and not aux_dst.exists():
            shutil.copy2(str(aux_src), str(aux_dst))

    # Use batch_generate script
    gen_script = str(PROJECT_ROOT / 'backtest' / 'batch_generate_v395_reports.py')
    cmd = [
        sys.executable, gen_script,
        '--version', params.get('batch_version', 'v4.7.5'),
        '--start-date', params.get('report_start_date', '2024-01-01'),
        '--end-date', params['end_date'],
        '--output-dir', str(report_dir),
        '--force',
    ]

    print(f"[reports] 生成报告: {report_dir}", file=sys.stderr)
    start = time.time()
    result = subprocess.run(cmd, capture_output=False, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - start
    print(f"[reports] 完成: {elapsed/60:.1f}min", file=sys.stderr)

    n_reports = len(list(report_dir.glob('analysis_data_*.json')))
    print(f"[reports] {n_reports} 份报告", file=sys.stderr)

    return str(report_dir) if n_reports > 0 else None


def evaluate(report_dir):
    """运行回测评估"""
    from backtest import backtest_report_based as brb
    from backtest import north_star_metrics as nsm

    nsm.DB_PATH = DB_PATH
    brb.DB_PATH = DB_PATH

    bp = json.load(open(BACKTEST_PARAMS_FILE)) if BACKTEST_PARAMS_FILE.exists() else {}

    with contextlib.redirect_stdout(sys.stderr):
        reports = brb.load_reports(report_dir, rank_field=bp.get('rank_field', 'auto'))

    if not reports:
        return None

    dates = sorted(reports.keys())
    print(f"[eval] {len(dates)} 天 ({dates[0]}→{dates[-1]})", file=sys.stderr)

    with contextlib.redirect_stdout(sys.stderr):
        result = brb.run_single_backtest(
            reports, 'experiment',
            top_n=bp.get('top_n', 10),
            benchmark_code=bp.get('benchmark', '000905.SH'),
            focus_days=bp.get('focus_days', 15),
            score_floor=bp.get('score_floor', 30.0),
            cppi_floor=bp.get('cppi_floor', 0.05),
            cppi_multiplier=bp.get('cppi_multiplier', 20.0),
        )

    if not result:
        return None

    s = result.get('summary', {}).get(bp.get('focus_days', 15), {})
    return _v4_weighted_pct(s)


def _v4_weighted_pct(s):
    """V4加权百分比 (0-100)"""
    from backtest.north_star_metrics import NORTH_STAR_TARGETS_V2, score_metric_v2

    mv = {
        'daily_ic': s.get('ic_mean', 0), 'icir': s.get('icir', 0),
        'ic_positive_pct': s.get('ic_positive_pct', 0),
        'ic_monotonicity': s.get('ic_monotonicity', 0),
        'ic_time_stability': s.get('ic_time_stability', 999),
        'signal_half_life': s.get('signal_half_life', 0),
        'annual_turnover': s.get('annual_turnover', 0),
        'annual_cost_drag': s.get('annual_cost_drag', 0),
        'net_gross_ratio': s.get('net_gross_ratio', 0),
        'limit_up_fail_rate': s.get('limit_up_fail_rate', 0),
        'liquidity_coverage': s.get('liquidity_coverage', 0),
        'max_drawdown': s.get('max_drawdown', 0),
        'sharpe_ratio': s.get('sharpe_ratio', 0),
        'sortino_ratio': s.get('sortino_ratio', 0),
        'calmar_ratio': s.get('calmar_ratio', 0),
        'worst_rolling_60d_icir': s.get('worst_rolling_60d_icir', None),
        'annual_return': s.get('annual_return', 0),
        'monthly_win_rate': s.get('monthly_win_rate', 0),
        'half_period_consistency': s.get('half_period_consistency', 0),
        'cap_balance_ratio': s.get('cap_balance_ratio', 0),
        'median_market_cap_bn': s.get('median_market_cap_bn', 0),
    }

    lw = {1: 0.412, 2: 0.176, 3: 0.235, 4: 0.176}
    ls, lm = {}, {}
    for k, ti in NORTH_STAR_TARGETS_V2.items():
        v = mv.get(k)
        if v is None: continue
        layer = ti['layer']
        sc, _ = score_metric_v2(v, ti)
        ls[layer] = ls.get(layer, 0) + sc
        lm[layer] = lm.get(layer, 0) + 5

    wp = sum(lw.get(l, 0) * ls[l] / lm[l] for l in ls if lm[l] > 0)
    return round(wp * 100, 2)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Phase 3: 全量WF训练')
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--skip-reports', action='store_true')
    parser.add_argument('--model-path', help='指定已有模型')
    args = parser.parse_args()

    params = load_train_params()
    if args.full:
        params['mode'] = 'full'
        params['start_date'] = '2020-01-01'

    t0 = time.time()
    model_path = args.model_path

    # Step 1: Train
    if not args.skip_train and not model_path:
        model_path = train_model(params)
        if not model_path:
            print("ERROR: train failed", file=sys.stderr)
            sys.exit(1)

    # Step 2: Reports
    report_dir = str(PROJECT_ROOT / 'reports' / f"daily_selection_{params['output_version']}")
    if not args.skip_reports and model_path:
        report_dir = generate_reports(model_path, params)
        if not report_dir:
            print("ERROR: reports failed", file=sys.stderr)
            sys.exit(1)

    # Step 3: Evaluate
    score = evaluate(report_dir)
    print(f"\n[total] {(time.time()-t0)/60:.1f}min", file=sys.stderr)

    if score is None:
        print("ERROR", file=sys.stderr)
        sys.exit(1)

    print(f"{score:.2f}")


if __name__ == '__main__':
    main()
