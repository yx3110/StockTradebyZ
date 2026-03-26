#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iterative Pipeline — L1/L2/L3/L4 Orchestrator

Ties L1 fast training to L2 mini-report generation and North Star evaluation,
then optionally escalates to L3/L4 full training via subprocess.

Functions:
    run_l1(params)              — Thin wrapper around L1FastTrainer
    run_l2(params, l1_result)  — Mini-report generation + NS evaluation
    run_l3(params)             — Full training + NS eval via subprocess (L3 gate >= 60)
    run_l4(params)             — Full training + NS eval via subprocess (final, always pass)
    run_auto_gate(params)      — Automatic L1→L2→L3→L4 pipeline with gating
    run_batch(param_files)     — Batch mode: L1+L2 all, promote top N to L3/L4
    append_comparison(result)  — Append row to iteration_comparison.tsv
    _update_calibration(...)   — Update l2_calibration.json with new data pair
    _check_l2_gate(mini_ns, metrics) — Gate conditions
    _calibrate_ns(mini_ns_raw)       — Linear calibration from l2_calibration.json
    _get_version_flag(params)        — Map base_version → training --flag
    _get_report_version(params)      — Map base_version → report version string

作者: Claude Code
创建时间: 2026-03-26
"""

import sys
import os
import json
import time
import csv
import subprocess
import tempfile
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Project root setup
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
CALIBRATION_FILE = str(PROJECT_ROOT / 'scripts' / 'l2_calibration.json')
COMPARISON_FILE = str(PROJECT_ROOT / 'scripts' / 'iteration_comparison.tsv')

logger = logging.getLogger(__name__)


# ==================================================================
# Public API
# ==================================================================

def run_l1(params: dict) -> dict:
    """Thin wrapper around L1FastTrainer.

    Args:
        params: Parameter dict with 'variant_name', 'training', 'features' keys.

    Returns:
        L1 result dict with keys: variant_name, level, duration_sec, metrics,
        gate_pass, model_path, feature_cols.
    """
    from scripts.l1_fast_trainer import L1FastTrainer
    trainer = L1FastTrainer(params)
    return trainer.train()


def run_l2(params: dict, l1_result: dict, n_days: int = 60) -> dict:
    """L2 mini-report generation + North Star evaluation.

    Flow:
    1. Load L1 model from l1_result['model_path']
    2. Get recent trading dates from v39_feature_cache (last n_days)
    3. Preload features via fast_preload_feature_cache()
    4. Load securities info (stock name, industry)
    5. For each date: predict with L1 LGB models, cross-sectional rank,
       build JSON report, save to temp dir
    6. Run backtest on those reports via run_backtest()
    7. Compute NS scores via compute_ns_scores()
    8. Apply L2 calibration
    9. Check L2 gate conditions
    10. Return structured result

    Args:
        params:    Same params dict passed to run_l1(). Uses
                   params['scoring'] for top_n, focus_days, rank_field.
        l1_result: Return value of run_l1().
        n_days:    Number of recent trading days to use for mini-backtest.

    Returns:
        L2 result dict.
    """
    t0 = time.time()

    variant_name = params.get('variant_name', l1_result.get('variant_name', 'l2_eval'))
    scoring_cfg = params.get('scoring', {})
    top_n = scoring_cfg.get('top_n', 10)
    focus_days = scoring_cfg.get('focus_days', 10)
    rank_field = scoring_cfg.get('rank_field', 'pred_10d')

    print(f"\n{'='*60}")
    print(f"[L2] 开始 L2 评估: {variant_name}")
    print(f"[L2] top_n={top_n}, focus_days={focus_days}, rank_field={rank_field}, n_days={n_days}")
    print(f"{'='*60}")

    # ── 1. Load L1 model ──────────────────────────────────────────
    import joblib
    model_path = l1_result.get('model_path')
    if not model_path or not Path(model_path).exists():
        return _error_result(variant_name, f"model_path not found: {model_path}")

    payload = joblib.load(model_path)
    lgb_models = payload.get('models', {})   # {'label_5d': Booster, 'label_10d': Booster}
    feature_cols = payload.get('feature_cols', l1_result.get('feature_cols', []))

    if not lgb_models:
        return _error_result(variant_name, "No models in payload")

    print(f"[L2] 已加载模型: {list(lgb_models.keys())}  特征数: {len(feature_cols)}")

    # ── 2. Get recent trading dates ───────────────────────────────
    from backtest.batch_generate_v395_reports import get_trading_dates
    all_dates = get_trading_dates('2020-01-01', '2030-12-31')
    if not all_dates:
        return _error_result(variant_name, "No trading dates available")

    selected_dates = all_dates[-n_days:]
    print(f"[L2] 评估日期: {selected_dates[0]} → {selected_dates[-1]} ({len(selected_dates)} 天)")

    # ── 3. Preload features ───────────────────────────────────────
    from backtest.batch_generate_v395_reports import fast_preload_feature_cache
    print("[L2] 预加载特征缓存 ...")
    features_by_date = fast_preload_feature_cache(selected_dates)

    # ── 4. Load securities info ───────────────────────────────────
    from backtest.batch_generate_v395_reports import load_securities_info
    sec_info = load_securities_info()   # {code: {name, industry, area}}

    # ── 5. Generate mini JSON reports ────────────────────────────
    with tempfile.TemporaryDirectory(prefix='l2_reports_') as tmp_dir:
        tmp_path = Path(tmp_dir)
        n_reports = 0

        for date in selected_dates:
            features_df = features_by_date.get(date)
            if features_df is None or len(features_df) == 0:
                continue

            report = _generate_single_report(
                date=date,
                features_df=features_df,
                lgb_models=lgb_models,
                feature_cols=feature_cols,
                sec_info=sec_info,
                rank_field=rank_field,
            )
            if report is None:
                continue

            date_compact = date.replace('-', '')
            out_path = tmp_path / f'analysis_data_{date_compact}.json'
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False)
            n_reports += 1

        if n_reports == 0:
            return _error_result(variant_name, "No reports generated (all dates empty)")

        print(f"[L2] 生成报告: {n_reports} 份  目录: {tmp_dir}")

        # ── 6. Run backtest ────────────────────────────────────────
        from backtest.run_north_star_eval import run_backtest
        bt_result = run_backtest(
            report_dir=str(tmp_dir),
            label=variant_name,
            top_n=top_n,
            focus_days=focus_days,
            rank_field=rank_field,
        )

        if bt_result is None:
            return _error_result(variant_name, "run_backtest() returned None")

        # ── 7. Compute NS scores ───────────────────────────────────
        from backtest.backtest_report_based import compute_ns_scores
        summary = bt_result.get('summary', {})
        ns = compute_ns_scores(
            summary=summary,
            focus_days=focus_days,
            n_trading_days=n_reports,
        )

        mini_ns_raw = ns.get('v2_score', 0)
        mini_ns_grade = ns.get('v2_grade', 'D')

        # Extract key metrics
        s = summary.get(focus_days, {})
        metrics = {
            'ic_10d': s.get('ic_mean', 0.0),
            'icir_10d': s.get('icir', 0.0),
            'annual_return_gross': s.get('annual_return', 0.0),
            'max_drawdown': s.get('max_drawdown', 0.0),
            'sharpe': s.get('sharpe_ratio', 0.0),
        }

        # ── 8. Apply calibration ──────────────────────────────────
        mini_ns_calibrated = _calibrate_ns(mini_ns_raw)

        # ── 9. Gate check ─────────────────────────────────────────
        gate_pass = _check_l2_gate(mini_ns_raw, metrics)

        duration = time.time() - t0

        print(f"\n[L2] 结果摘要:")
        print(f"  mini_ns_raw        = {mini_ns_raw}")
        print(f"  mini_ns_calibrated = {mini_ns_calibrated:.2f}")
        print(f"  mini_ns_grade      = {mini_ns_grade}")
        print(f"  ic_10d             = {metrics['ic_10d']:.4f}")
        print(f"  icir_10d           = {metrics['icir_10d']:.3f}")
        print(f"  annual_return_gross= {metrics['annual_return_gross']:.3f}")
        print(f"  max_drawdown       = {metrics['max_drawdown']:.3f}")
        print(f"  sharpe             = {metrics['sharpe']:.3f}")
        print(f"  gate_pass          = {gate_pass}")
        print(f"  duration           = {duration:.1f}s")

        return {
            'variant_name': variant_name,
            'level': 'L2',
            'duration_sec': duration,
            'mini_ns_raw': mini_ns_raw,
            'mini_ns_calibrated': mini_ns_calibrated,
            'mini_ns_grade': mini_ns_grade,
            'metrics': metrics,
            'gate_pass': gate_pass,
            'reports_dir': None,   # tmp dir has been deleted
        }


# ==================================================================
# Internal helpers
# ==================================================================

def _generate_single_report(
    date: str,
    features_df: pd.DataFrame,
    lgb_models: dict,
    feature_cols: list,
    sec_info: dict,
    rank_field: str = 'pred_10d',
) -> Optional[dict]:
    """Generate a single-date JSON report dict in the format expected by load_reports().

    Args:
        date:         Trade date string 'YYYY-MM-DD'.
        features_df:  Pre-loaded features DataFrame with 'code' column.
        lgb_models:   {'label_5d': Booster, 'label_10d': Booster, ...}
        feature_cols: Feature column names used at training time.
        sec_info:     {code: {name, industry, area}} from load_securities_info().
        rank_field:   Which field to use for rank_score ('pred_10d', 'pred_5d', 'score').

    Returns:
        dict matching analysis_data JSON schema, or None if no predictions.
    """
    if features_df is None or len(features_df) == 0:
        return None

    df = features_df.copy()

    # ── Robust Z-Score normalization (same as L1 training) ────────
    exclude_cols = {'code', 'trade_date'}
    norm_cols = [c for c in feature_cols if c in df.columns and c not in exclude_cols]

    if norm_cols:
        for col in norm_cols:
            med = df[col].median()
            mad = (df[col] - med).abs().median()
            if mad > 1e-10:
                df[col] = (df[col] - med) / (mad * 1.4826)
            else:
                df[col] = 0.0
        df[norm_cols] = df[norm_cols].clip(-5, 5).fillna(0)

    # ── Prepare feature matrix ────────────────────────────────────
    available_cols = [c for c in feature_cols if c in df.columns]
    if not available_cols:
        return None

    X = df[available_cols].fillna(0).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    codes = df['code'].tolist()
    n = len(codes)
    if n == 0:
        return None

    # ── Predict with each LGB model ───────────────────────────────
    pred_5d = np.zeros(n)
    pred_10d = np.zeros(n)

    model_5d = lgb_models.get('label_5d')
    model_10d = lgb_models.get('label_10d')

    if model_5d is not None:
        try:
            pred_5d = model_5d.predict(X)
        except Exception as e:
            logger.warning(f"[L2] pred_5d failed for {date}: {e}")

    if model_10d is not None:
        try:
            pred_10d = model_10d.predict(X)
        except Exception as e:
            logger.warning(f"[L2] pred_10d failed for {date}: {e}")

    # ── Cross-sectional rank → score [0, 100] ─────────────────────
    # Use pred_10d for primary ranking
    rank_arr = _crosssectional_rank_score(pred_10d)

    # ── Build stock list ──────────────────────────────────────────
    stock_list = []
    for i, code in enumerate(codes):
        info = sec_info.get(code, {})
        score_val = float(rank_arr[i])

        # Determine rank_score based on rank_field
        if rank_field == 'pred_10d':
            rs = float(pred_10d[i])
        elif rank_field == 'pred_5d':
            rs = float(pred_5d[i])
        elif rank_field == 'score':
            rs = score_val
        else:
            rs = float(pred_10d[i])

        entry = {
            'stock_code': code,
            'stock_name': info.get('name', f'Stock_{code}'),
            'industry': info.get('industry', ''),
            'score': round(score_val, 2),
            'predicted_return_5d': round(float(pred_5d[i]), 6),
            'pred_5d': round(float(pred_5d[i]), 6),
            'pred_10d': round(float(pred_10d[i]), 6),
            'rank_score': round(rs, 6),
            'strategies': ['ML_Score'],
            'analysis_date': date.replace('-', ''),
        }
        stock_list.append(entry)

    if not stock_list:
        return None

    # Sort by rank_score descending
    stock_list.sort(key=lambda x: x['rank_score'], reverse=True)

    date_compact = date.replace('-', '')
    return {
        'analysis_date': date_compact,
        'scoring_version': 'l1_fast',
        'total_scored_stocks': len(stock_list),
        'all_stocks_with_scores': stock_list,
    }


def _crosssectional_rank_score(predictions: np.ndarray) -> np.ndarray:
    """Convert raw predictions to cross-sectional percentile scores [0, 100].

    Args:
        predictions: 1D array of prediction values.

    Returns:
        1D array of scores in [0, 100].
    """
    n = len(predictions)
    if n == 0:
        return np.array([])
    ranks = predictions.argsort().argsort()   # 0-indexed ranks
    scores = ranks / max(n - 1, 1) * 100.0
    return scores


def _check_l2_gate(mini_ns: float, metrics: dict) -> bool:
    """Check L2 gate conditions.

    Gate conditions (all must pass):
    - mini_ns_raw (V2 score) >= 40
    - ic_10d >= 0.05
    - max_drawdown >= -0.30

    Args:
        mini_ns:  Raw V2 North Star score (0-105).
        metrics:  Dict with ic_10d, max_drawdown, etc.

    Returns:
        True if all conditions are met.
    """
    ic_10d = metrics.get('ic_10d', 0.0)
    max_dd = metrics.get('max_drawdown', -1.0)

    checks = {
        'mini_ns_raw >= 40': mini_ns >= 40,
        'ic_10d >= 0.05': ic_10d >= 0.05,
        'max_drawdown >= -0.30': max_dd >= -0.30,
    }

    all_pass = all(checks.values())

    print("\n[L2] 门控检查:")
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    return all_pass


def _calibrate_ns(mini_ns_raw: float) -> float:
    """Apply linear calibration to raw NS score.

    Reads scripts/l2_calibration.json for calibration pairs.
    If fewer than 5 pairs exist, returns raw value unchanged.
    Otherwise applies: calibrated = slope * raw + intercept

    calibration.json format:
    {
        "pairs": [[raw1, actual1], [raw2, actual2], ...],
        "slope": 1.0,
        "intercept": 0.0
    }

    Args:
        mini_ns_raw: Raw V2 North Star score.

    Returns:
        Calibrated score (float).
    """
    cal_path = Path(CALIBRATION_FILE)
    if not cal_path.exists():
        return float(mini_ns_raw)

    try:
        with open(cal_path, 'r', encoding='utf-8') as f:
            cal = json.load(f)
    except Exception as e:
        logger.warning(f"[L2] Failed to load calibration file: {e}")
        return float(mini_ns_raw)

    pairs = cal.get('pairs', [])
    if len(pairs) < 5:
        return float(mini_ns_raw)

    # Use pre-computed slope/intercept if available
    slope = cal.get('slope', None)
    intercept = cal.get('intercept', None)

    if slope is None or intercept is None:
        # Compute linear regression from pairs
        xs = np.array([p[0] for p in pairs], dtype=float)
        ys = np.array([p[1] for p in pairs], dtype=float)
        if len(xs) < 2 or np.std(xs) < 1e-10:
            return float(mini_ns_raw)
        slope = np.cov(xs, ys)[0, 1] / np.var(xs)
        intercept = np.mean(ys) - slope * np.mean(xs)

    calibrated = slope * mini_ns_raw + intercept
    return float(calibrated)


# ==================================================================
# Version mapping helpers
# ==================================================================

# Canonical (short) key → training flag  (e.g. 'v475' → '--v475')
_VERSION_FLAG_MAP = {
    'v395': '--v395',
    'v3.95': '--v395',   # normalised alias
    'v43': '--v43',
    'v4.3': '--v43',
    'v44': '--v44',
    'v4.4': '--v44',
    'v46': '--v46',
    'v4.6': '--v46',
    'v47': '--v47',
    'v4.7': '--v47',
    'v471': '--v471',
    'v4.7.1': '--v471',
    'v472': '--v472',
    'v4.7.2': '--v472',
    'v473': '--v473',
    'v4.7.3': '--v473',
    'v474': '--v474',
    'v4.7.4': '--v474',
    'v475': '--v475',
    'v4.7.5': '--v475',
    'v476': '--v476',
    'v4.7.6': '--v476',
    'v477': '--v477',
    'v4.7.7': '--v477',
    'v478': '--v478',
    'v4.7.8': '--v478',
    'v479': '--v479',
    'v4.7.9': '--v479',
    'v480': '--v480',
    'v4.8.0': '--v480',
    'v481': '--v481',
    'v4.8.1': '--v481',
    'v482': '--v482',
    'v4.8.2': '--v482',
    'v483': '--v483',
    'v4.8.3': '--v483',
    'v484': '--v484',
    'v4.8.4': '--v484',
    'v485': '--v485',
    'v4.8.5': '--v485',
    'v486': '--v486',
    'v4.8.6': '--v486',
}

# Canonical (short) key → human-readable report version string
_VERSION_REPORT_MAP = {
    'v395': 'v3.95',
    'v3.95': 'v3.95',
    'v43': 'v4.3',
    'v4.3': 'v4.3',
    'v44': 'v4.4',
    'v4.4': 'v4.4',
    'v46': 'v4.6',
    'v4.6': 'v4.6',
    'v47': 'v4.7',
    'v4.7': 'v4.7',
    'v471': 'v4.7.1',
    'v4.7.1': 'v4.7.1',
    'v472': 'v4.7.2',
    'v4.7.2': 'v4.7.2',
    'v473': 'v4.7.3',
    'v4.7.3': 'v4.7.3',
    'v474': 'v4.7.4',
    'v4.7.4': 'v4.7.4',
    'v475': 'v4.7.5',
    'v4.7.5': 'v4.7.5',
    'v476': 'v4.7.6',
    'v4.7.6': 'v4.7.6',
    'v477': 'v4.7.7',
    'v4.7.7': 'v4.7.7',
    'v478': 'v4.7.8',
    'v4.7.8': 'v4.7.8',
    'v479': 'v4.7.9',
    'v4.7.9': 'v4.7.9',
    'v480': 'v4.8.0',
    'v4.8.0': 'v4.8.0',
    'v481': 'v4.8.1',
    'v4.8.1': 'v4.8.1',
    'v482': 'v4.8.2',
    'v4.8.2': 'v4.8.2',
    'v483': 'v4.8.3',
    'v4.8.3': 'v4.8.3',
    'v484': 'v4.8.4',
    'v4.8.4': 'v4.8.4',
    'v485': 'v4.8.5',
    'v4.8.5': 'v4.8.5',
    'v486': 'v4.8.6',
    'v4.8.6': 'v4.8.6',
}


def _get_version_flag(params: dict) -> str:
    """Map params['base_version'] to the training script flag (e.g. '--v475').

    Falls back to '--v475' if the version is unknown.
    """
    raw = params.get('base_version', 'v475').strip().lower()
    return _VERSION_FLAG_MAP.get(raw, f'--{raw}')


def _get_report_version(params: dict) -> str:
    """Map params['base_version'] to the human-readable report version (e.g. 'v4.7.5').

    Falls back to the raw string if not found in the map.
    """
    raw = params.get('base_version', 'v475').strip().lower()
    return _VERSION_REPORT_MAP.get(raw, raw)


# ==================================================================
# L3 / L4 — full training + evaluation via subprocess
# ==================================================================

def _run_subprocess_training(
    params: dict,
    start_date: str,
    level: str,
    variant_suffix: str,
) -> dict:
    """Common implementation for L3 and L4: train then evaluate.

    Steps:
    1. Build & run training subprocess (train_v395_multi_target.py)
    2. Build & run batch report subprocess (batch_generate_v395_reports.py)
    3. Run NS evaluation via run_backtest() + compute_ns_scores()

    Args:
        params:         Full params dict.
        start_date:     Training start date string ('YYYY-MM-DD').
        level:          'L3' or 'L4'.
        variant_suffix: Suffix appended to the output report dir.

    Returns:
        Result dict with standard keys.
    """
    t0 = time.time()
    variant_name = params.get('variant_name', f'{level}_eval')
    training_cfg = params.get('training', {})
    scoring_cfg = params.get('scoring', {})

    top_n = scoring_cfg.get('top_n', 10)
    focus_days = scoring_cfg.get('focus_days', 10)
    rank_field = scoring_cfg.get('rank_field', 'pred_10d')

    version_flag = _get_version_flag(params)
    report_version = _get_report_version(params)

    print(f"\n{'='*60}")
    print(f"[{level}] 开始 {level} 评估: {variant_name}")
    print(f"[{level}] base_version={params.get('base_version')}, version_flag={version_flag}")
    print(f"[{level}] start_date={start_date}")
    print(f"{'='*60}")

    # ── 1. Training subprocess ────────────────────────────────────
    train_script = str(PROJECT_ROOT / 'ml_models' / 'training' / 'train_v395_multi_target.py')

    purge_days = training_cfg.get('purge_days', 15)
    sharpe_blend = training_cfg.get('sharpe_blend', 0.3)

    train_cmd = [
        sys.executable, train_script,
        version_flag,
        '--start-date', start_date,
        '--purge-days', str(purge_days),
        '--sharpe-blend', str(sharpe_blend),
    ]

    # Pass through hyperparameter overrides
    num_leaves = training_cfg.get('num_leaves')
    if num_leaves:
        train_cmd += ['--num-leaves', str(num_leaves)]
    min_data = training_cfg.get('min_data_in_leaf')
    if min_data:
        train_cmd += ['--min-data-in-leaf', str(min_data)]

    print(f"[{level}] 训练命令: {' '.join(train_cmd)}")
    try:
        result = subprocess.run(
            train_cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=36000,  # 10 hours max (V486 WF训练可能需要6-8h)
        )
        if result.returncode != 0:
            stderr_tail = (result.stderr or '')[-2000:]
            msg = f"Training failed (rc={result.returncode}): {stderr_tail}"
            print(f"[{level}] {msg}")
            return _error_result_level(variant_name, msg, level)
        # Print training stdout for visibility
        if result.stdout:
            for line in result.stdout.strip().split('\n')[-10:]:
                print(f"  [train] {line}")
    except subprocess.TimeoutExpired:
        return _error_result_level(variant_name, "Training subprocess timed out (>2h)", level)
    except Exception as e:
        return _error_result_level(variant_name, f"Training subprocess error: {e}", level)

    print(f"[{level}] 训练完成")

    # ── 2. Batch report subprocess ────────────────────────────────
    report_dir = str(PROJECT_ROOT / 'reports' / f'daily_selection_{report_version}_{variant_suffix}')
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    batch_script = str(PROJECT_ROOT / 'backtest' / 'batch_generate_v395_reports.py')
    batch_cmd = [
        sys.executable, batch_script,
        '--version', report_version,
        '--output-dir', report_dir,
        '--start-date', start_date,
        '--force',
    ]

    print(f"[{level}] 批量报告命令: {' '.join(batch_cmd)}")
    try:
        result = subprocess.run(
            batch_cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=36000,
        )
        if result.returncode != 0:
            stderr_tail = (result.stderr or '')[-2000:]
            msg = f"Report gen failed (rc={result.returncode}): {stderr_tail}"
            print(f"[{level}] {msg}")
            return _error_result_level(variant_name, msg, level)
    except subprocess.TimeoutExpired:
        return _error_result_level(variant_name, "Batch report subprocess timed out (>2h)", level)
    except Exception as e:
        return _error_result_level(variant_name, f"Batch report subprocess error: {e}", level)

    print(f"[{level}] 报告生成完成 → {report_dir}")

    # ── 3. NS evaluation ──────────────────────────────────────────
    from backtest.run_north_star_eval import run_backtest
    from backtest.backtest_report_based import compute_ns_scores

    json_files = list(Path(report_dir).glob('analysis_data_*.json'))
    n_reports = len(json_files)

    if n_reports == 0:
        return _error_result_level(variant_name, f"No JSON reports found in {report_dir}", level)

    bt_result = run_backtest(
        report_dir=report_dir,
        label=variant_name,
        top_n=top_n,
        focus_days=focus_days,
        rank_field=rank_field,
    )

    if bt_result is None:
        return _error_result_level(variant_name, "run_backtest() returned None", level)

    summary = bt_result.get('summary', {})
    ns = compute_ns_scores(
        summary=summary,
        focus_days=focus_days,
        n_trading_days=n_reports,
    )

    ns_score = ns.get('v2_score', 0)
    ns_grade = ns.get('v2_grade', 'D')

    s = summary.get(focus_days, {})
    metrics = {
        'ic_10d': s.get('ic_mean', 0.0),
        'icir_10d': s.get('icir', 0.0),
        'annual_return_gross': s.get('annual_return', 0.0),
        'max_drawdown': s.get('max_drawdown', 0.0),
        'sharpe': s.get('sharpe_ratio', 0.0),
    }

    duration = time.time() - t0

    print(f"\n[{level}] 结果摘要:")
    print(f"  ns_score  = {ns_score}")
    print(f"  ns_grade  = {ns_grade}")
    print(f"  ic_10d    = {metrics['ic_10d']:.4f}")
    print(f"  icir_10d  = {metrics['icir_10d']:.3f}")
    print(f"  duration  = {duration:.1f}s")

    return {
        'variant_name': variant_name,
        'level': level,
        'duration_sec': duration,
        'ns_score': ns_score,
        'ns_grade': ns_grade,
        'metrics': metrics,
        'reports_dir': report_dir,
        'n_reports': n_reports,
    }


def run_l3(params: dict) -> dict:
    """L3: Full training + NS evaluation via subprocess.

    Uses a reduced training window (default start 2023-01-01) to finish
    faster while still producing a meaningful signal quality estimate.

    Gate condition: ns_score >= 60 (A level).

    Args:
        params: Parameter dict.  Uses params['training']['l3_start_date']
                (default '2023-01-01').

    Returns:
        Result dict with gate_pass key.
    """
    training_cfg = params.get('training', {})
    start_date = training_cfg.get('l3_start_date', '2023-01-01')
    variant_name = params.get('variant_name', 'l3_eval')
    report_version = _get_report_version(params)
    variant_suffix = f'l3_{variant_name}'

    result = _run_subprocess_training(
        params=params,
        start_date=start_date,
        level='L3',
        variant_suffix=variant_suffix,
    )

    if 'error' in result:
        result['gate_pass'] = False
        return result

    ns_score = result.get('ns_score', 0)
    gate_pass = ns_score >= 60

    result['gate_pass'] = gate_pass

    print(f"\n[L3] 门控检查:")
    status = "PASS" if gate_pass else "FAIL"
    print(f"  [{status}] ns_score >= 60  (actual: {ns_score})")

    return result


def run_l4(params: dict) -> dict:
    """L4: Full training + NS evaluation via subprocess (final level).

    Uses the full training window (default start 2020-01-01).
    gate_pass is always True (this is the final level).

    Args:
        params: Parameter dict.  Uses params['training']['start_date']
                (default '2020-01-01').

    Returns:
        Result dict with gate_pass=True.
    """
    training_cfg = params.get('training', {})
    start_date = training_cfg.get('start_date', '2020-01-01')
    variant_name = params.get('variant_name', 'l4_eval')
    variant_suffix = f'l4_{variant_name}'

    result = _run_subprocess_training(
        params=params,
        start_date=start_date,
        level='L4',
        variant_suffix=variant_suffix,
    )

    # L4 is the final level — always gate_pass=True
    result['gate_pass'] = True

    print(f"\n[L4] Final level — gate_pass=True")

    return result


# ==================================================================
# Auto-Gate / Batch
# ==================================================================

def run_auto_gate(params: dict, max_level: str = 'L4') -> dict:
    """Run L1 → L2 → L3 → L4 with automatic gating.

    Stops at the first failed gate or when max_level is reached.
    Calls append_comparison() after each level.
    If L2 and L4 both complete, calls _update_calibration() to record
    the (mini_ns, full_ns) pair for future calibration.

    Args:
        params:    Full parameter dict.
        max_level: Maximum level to run ('L1', 'L2', 'L3', 'L4').

    Returns:
        {'final': last_result, 'all_results': [list of all level results]}
    """
    level_order = ['L1', 'L2', 'L3', 'L4']
    max_idx = level_order.index(max_level.upper()) if max_level.upper() in level_order else 3

    all_results = []
    l1_result = None
    l2_result = None
    l4_result = None

    variant_name = params.get('variant_name', 'auto_gate')
    print(f"\n{'='*60}")
    print(f"[AutoGate] 开始自动门控流水线: {variant_name}  max_level={max_level}")
    print(f"{'='*60}")

    # ── L1 ────────────────────────────────────────────────────────
    print(f"\n[AutoGate] 运行 L1 ...")
    l1_result = run_l1(params)
    all_results.append(l1_result)
    append_comparison(l1_result)

    if not l1_result.get('gate_pass', False):
        print(f"[AutoGate] L1 门控失败, 停止流水线")
        return {'final': l1_result, 'all_results': all_results}

    if max_idx < 1:
        return {'final': l1_result, 'all_results': all_results}

    # ── L2 ────────────────────────────────────────────────────────
    l2_days = params.get('l2_days', 60)
    print(f"\n[AutoGate] 运行 L2 (n_days={l2_days}) ...")
    l2_result = run_l2(params, l1_result, n_days=l2_days)
    all_results.append(l2_result)
    append_comparison(l2_result)

    if not l2_result.get('gate_pass', False):
        print(f"[AutoGate] L2 门控失败, 停止流水线")
        return {'final': l2_result, 'all_results': all_results}

    if max_idx < 2:
        return {'final': l2_result, 'all_results': all_results}

    # ── L3 ────────────────────────────────────────────────────────
    print(f"\n[AutoGate] 运行 L3 ...")
    l3_result = run_l3(params)
    all_results.append(l3_result)
    append_comparison(l3_result)

    if not l3_result.get('gate_pass', False):
        print(f"[AutoGate] L3 门控失败, 停止流水线")
        return {'final': l3_result, 'all_results': all_results}

    if max_idx < 3:
        return {'final': l3_result, 'all_results': all_results}

    # ── L4 ────────────────────────────────────────────────────────
    print(f"\n[AutoGate] 运行 L4 ...")
    l4_result = run_l4(params)
    all_results.append(l4_result)
    append_comparison(l4_result)

    # Update calibration with (mini_ns, full_ns) pair
    if l2_result is not None and l4_result is not None:
        mini_ns = l2_result.get('mini_ns_raw', 0)
        full_ns = l4_result.get('ns_score', 0)
        _update_calibration(mini_ns, full_ns, variant_name)

    print(f"\n[AutoGate] 流水线完成 — 最终级别: L4")
    return {'final': l4_result, 'all_results': all_results}


def run_batch(
    param_files: List[str],
    promote_top: int = 2,
    max_level: str = 'L3',
) -> list:
    """Batch mode: run L1+L2 for all param files, then promote top N to L3/L4.

    Phase 1: Run L1+L2 for all variants.
    Phase 2: Sort by mini_ns (L2 raw score), promote top N to L3 (and L4
             if max_level='L4').

    Args:
        param_files:  List of paths to JSON parameter files.
        promote_top:  Number of top variants to promote to L3/L4.
        max_level:    Maximum level to run for promoted variants.

    Returns:
        List of all result dicts (one per level per variant).
    """
    all_results = []
    l2_results_with_params = []

    print(f"\n{'='*60}")
    print(f"[Batch] 开始批量流水线: {len(param_files)} 个变体  promote_top={promote_top}  max_level={max_level}")
    print(f"{'='*60}")

    # ── Phase 1: L1 + L2 for all ─────────────────────────────────
    for pf in param_files:
        try:
            with open(pf, 'r', encoding='utf-8') as f:
                params = json.load(f)
        except Exception as e:
            print(f"[Batch] 无法加载参数文件 {pf}: {e}")
            continue

        print(f"\n[Batch] === 变体: {params.get('variant_name', pf)} ===")

        l1_result = run_l1(params)
        all_results.append(l1_result)
        append_comparison(l1_result)

        if not l1_result.get('gate_pass', False):
            print(f"[Batch] L1 失败, 跳过 L2")
            continue

        l2_days = params.get('l2_days', 60)
        l2_result = run_l2(params, l1_result, n_days=l2_days)
        all_results.append(l2_result)
        append_comparison(l2_result)

        if l2_result.get('gate_pass', False):
            l2_results_with_params.append((l2_result, params))

    # ── Phase 2: Promote top N by mini_ns ────────────────────────
    l2_results_with_params.sort(
        key=lambda x: x[0].get('mini_ns_raw', 0),
        reverse=True,
    )
    top_variants = l2_results_with_params[:promote_top]

    print(f"\n[Batch] Phase 2: 推进 Top {len(top_variants)} 变体 → {max_level}")
    for l2_res, params in top_variants:
        vname = params.get('variant_name', '?')
        mini_ns = l2_res.get('mini_ns_raw', 0)
        print(f"  - {vname}  mini_ns={mini_ns}")

    for l2_res, params in top_variants:
        l3_result = run_l3(params)
        all_results.append(l3_result)
        append_comparison(l3_result)

        if max_level.upper() == 'L4' and l3_result.get('gate_pass', False):
            l4_result = run_l4(params)
            all_results.append(l4_result)
            append_comparison(l4_result)

            # Update calibration
            mini_ns = l2_res.get('mini_ns_raw', 0)
            full_ns = l4_result.get('ns_score', 0)
            _update_calibration(mini_ns, full_ns, params.get('variant_name', '?'))

    print(f"\n[Batch] 批量流水线完成. 共 {len(all_results)} 条结果")
    return all_results


# ==================================================================
# Comparison table
# ==================================================================

_COMPARISON_COLUMNS = [
    'timestamp', 'variant', 'level', 'duration', 'gate',
    'ic_10d', 'icir_10d', 'mini_ns', 'ns_200d', 'ns_full', 'grade',
]


def append_comparison(result: dict) -> None:
    """Append one row to scripts/iteration_comparison.tsv.

    Creates the file with a header if it doesn't exist.

    Args:
        result: A result dict from run_l1(), run_l2(), run_l3(), or run_l4().
    """
    cmp_path = Path(COMPARISON_FILE)
    write_header = not cmp_path.exists()

    metrics = result.get('metrics', {})
    level = result.get('level', '?')

    # Determine which NS field to use
    mini_ns = ''
    ns_200d = ''
    ns_full = ''

    if level in ('L1', 'L2'):
        mini_ns = result.get('mini_ns_raw', result.get('mini_ns', ''))
    elif level in ('L3', 'L4'):
        ns_200d = result.get('ns_score', '') if level == 'L3' else ''
        ns_full = result.get('ns_score', '') if level == 'L4' else ''

    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'variant': result.get('variant_name', ''),
        'level': level,
        'duration': f"{result.get('duration_sec', 0):.1f}",
        'gate': 'PASS' if result.get('gate_pass', False) else 'FAIL',
        'ic_10d': f"{metrics.get('ic_10d', 0.0):.4f}",
        'icir_10d': f"{metrics.get('icir_10d', 0.0):.3f}",
        'mini_ns': str(mini_ns),
        'ns_200d': str(ns_200d),
        'ns_full': str(ns_full),
        'grade': result.get('ns_grade', result.get('mini_ns_grade', '')),
    }

    with open(cmp_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=_COMPARISON_COLUMNS, delimiter='\t')
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[Comparison] 已追加行到 {cmp_path}  ({level}, gate={row['gate']})")


# ==================================================================
# Calibration
# ==================================================================

def _update_calibration(mini_ns: float, full_ns: float, variant: str) -> None:
    """Update l2_calibration.json with a new (mini_ns, full_ns) data pair.

    If >= 5 pairs exist, fits a linear regression (numpy polyfit) and saves
    slope / intercept / r_squared.

    Args:
        mini_ns: Raw mini NS score from L2.
        full_ns: Full NS score from L4.
        variant: Variant name (for record-keeping).
    """
    cal_path = Path(CALIBRATION_FILE)

    if cal_path.exists():
        try:
            with open(cal_path, 'r', encoding='utf-8') as f:
                cal = json.load(f)
        except Exception:
            cal = {}
    else:
        cal = {}

    pairs = cal.get('pairs', [])
    entries = cal.get('entries', [])

    # Append new pair  [mini_ns, full_ns]
    pairs.append([float(mini_ns), float(full_ns)])
    entries.append({
        'variant': variant,
        'mini_ns': float(mini_ns),
        'full_ns': float(full_ns),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    })

    cal['pairs'] = pairs
    cal['entries'] = entries

    # Fit linear regression if enough data
    if len(pairs) >= 5:
        xs = np.array([p[0] for p in pairs], dtype=float)
        ys = np.array([p[1] for p in pairs], dtype=float)

        if np.std(xs) > 1e-10:
            coeffs = np.polyfit(xs, ys, 1)
            slope = float(coeffs[0])
            intercept = float(coeffs[1])

            # Compute R²
            y_pred = slope * xs + intercept
            ss_res = np.sum((ys - y_pred) ** 2)
            ss_tot = np.sum((ys - np.mean(ys)) ** 2)
            r_squared = float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

            cal['slope'] = slope
            cal['intercept'] = intercept
            cal['r_squared'] = r_squared
            cal['n_pairs'] = len(pairs)

            print(f"[Calibration] 更新线性校准: slope={slope:.4f}, intercept={intercept:.2f}, R²={r_squared:.4f}  (n={len(pairs)})")
        else:
            print(f"[Calibration] mini_ns 方差过小, 跳过回归拟合")
    else:
        print(f"[Calibration] 已记录第 {len(pairs)} 对校准数据 (需要≥5对才拟合回归)")

    with open(cal_path, 'w', encoding='utf-8') as f:
        json.dump(cal, f, indent=2, ensure_ascii=False)


# ==================================================================
# Internal utilities
# ==================================================================

def _error_result(variant_name: str, msg: str) -> dict:
    """Build a standard error result dict (L2)."""
    logger.error(f"[L2] Error: {msg}")
    print(f"[L2] ERROR: {msg}")
    return {
        'variant_name': variant_name,
        'level': 'L2',
        'duration_sec': 0.0,
        'mini_ns_raw': 0,
        'mini_ns_calibrated': 0.0,
        'mini_ns_grade': 'D',
        'metrics': {
            'ic_10d': 0.0,
            'icir_10d': 0.0,
            'annual_return_gross': 0.0,
            'max_drawdown': -1.0,
            'sharpe': 0.0,
        },
        'gate_pass': False,
        'reports_dir': None,
        'error': msg,
    }


def _error_result_level(variant_name: str, msg: str, level: str) -> dict:
    """Build a standard error result dict for L3/L4."""
    logger.error(f"[{level}] Error: {msg}")
    print(f"[{level}] ERROR: {msg}")
    return {
        'variant_name': variant_name,
        'level': level,
        'duration_sec': 0.0,
        'ns_score': 0,
        'ns_grade': 'D',
        'metrics': {
            'ic_10d': 0.0,
            'icir_10d': 0.0,
            'annual_return_gross': 0.0,
            'max_drawdown': -1.0,
            'sharpe': 0.0,
        },
        'gate_pass': False,
        'reports_dir': None,
        'n_reports': 0,
        'error': msg,
    }


# ==================================================================
# CLI entry point
# ==================================================================

def main():
    """Main CLI entry point.

    Usage:
        # Run a specific level
        python3 scripts/iterative_pipeline.py --level L1 --params params.json
        python3 scripts/iterative_pipeline.py --level L2 --params params.json
        python3 scripts/iterative_pipeline.py --level L3 --params params.json
        python3 scripts/iterative_pipeline.py --level L4 --params params.json

        # Auto-gate: L1 → L2 → L3 → L4 (stop at first failure)
        python3 scripts/iterative_pipeline.py --auto-gate --params params.json
        python3 scripts/iterative_pipeline.py --auto-gate --params params.json --max-level L2

        # Batch: L1+L2 for all, promote top 2 to L3
        python3 scripts/iterative_pipeline.py --batch --params a.json b.json --promote-top 2
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Iterative Pipeline — L1/L2/L3/L4 Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mutually exclusive mode group (required)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--level',
        choices=['L1', 'L2', 'L3', 'L4'],
        help='Run a specific pipeline level',
    )
    mode_group.add_argument(
        '--auto-gate',
        action='store_true',
        help='Auto-gate mode: L1 → L2 → L3 → L4, stop at first failure',
    )
    mode_group.add_argument(
        '--batch',
        action='store_true',
        help='Batch mode: L1+L2 all, promote top N to L3/L4',
    )

    # Common arguments
    parser.add_argument(
        '--params',
        nargs='+',
        required=True,
        metavar='PARAMS_JSON',
        help='One or more JSON parameter file paths',
    )
    parser.add_argument(
        '--max-level',
        choices=['L1', 'L2', 'L3', 'L4'],
        default='L4',
        help='Maximum level to run (default: L4)',
    )
    parser.add_argument(
        '--promote-top',
        type=int,
        default=2,
        help='Number of top variants to promote in batch mode (default: 2)',
    )
    parser.add_argument(
        '--l2-days',
        type=int,
        default=60,
        help='Number of recent trading days for L2 mini-backtest (default: 60)',
    )

    args = parser.parse_args()

    # ── Load params ───────────────────────────────────────────────
    def load_params(path: str) -> dict:
        with open(path, 'r', encoding='utf-8') as f:
            p = json.load(f)
        # Inject l2_days from CLI if not already set
        if 'l2_days' not in p:
            p['l2_days'] = args.l2_days
        return p

    # ── Dispatch to mode ──────────────────────────────────────────
    if args.level:
        # Single level mode — use first params file only
        params = load_params(args.params[0])
        level = args.level

        if level == 'L1':
            result = run_l1(params)
            append_comparison(result)
            _print_result(result)
            _print_comparison_table()
            sys.exit(0 if result.get('gate_pass') else 1)

        elif level == 'L2':
            # L2 needs an L1 result — run L1 silently first
            print("[CLI] L2 requires L1 model. Running L1 first ...")
            l1_result = run_l1(params)
            if not l1_result.get('gate_pass'):
                print("[CLI] L1 gate failed, cannot proceed to L2")
                sys.exit(1)
            result = run_l2(params, l1_result, n_days=args.l2_days)
            append_comparison(result)
            _print_result(result)
            _print_comparison_table()
            sys.exit(0 if result.get('gate_pass') else 1)

        elif level == 'L3':
            result = run_l3(params)
            append_comparison(result)
            _print_result(result)
            _print_comparison_table()
            sys.exit(0 if result.get('gate_pass') else 1)

        elif level == 'L4':
            result = run_l4(params)
            append_comparison(result)
            _print_result(result)
            _print_comparison_table()
            sys.exit(0)

    elif args.auto_gate:
        params = load_params(args.params[0])
        final_result = run_auto_gate(params, max_level=args.max_level)
        _print_comparison_table()
        sys.exit(0 if final_result['final'].get('gate_pass') else 1)

    elif args.batch:
        all_results = run_batch(
            param_files=args.params,
            promote_top=args.promote_top,
            max_level=args.max_level,
        )
        _print_comparison_table()
        sys.exit(0)


def _print_result(result: dict) -> None:
    """Pretty-print a single result dict."""
    print(f"\n{'─'*50}")
    print(f"  变体:      {result.get('variant_name', '?')}")
    print(f"  级别:      {result.get('level', '?')}")
    print(f"  耗时:      {result.get('duration_sec', 0):.1f}s")
    print(f"  门控:      {'PASS' if result.get('gate_pass') else 'FAIL'}")
    m = result.get('metrics', {})
    print(f"  IC(10d):   {m.get('ic_10d', 0.0):.4f}")
    print(f"  ICIR(10d): {m.get('icir_10d', 0.0):.3f}")
    for key in ('mini_ns_raw', 'ns_score'):
        if key in result:
            print(f"  NS分数:    {result[key]}")
    print(f"{'─'*50}")


def _print_comparison_table() -> None:
    """Print the comparison table path and contents."""
    cmp_path = Path(COMPARISON_FILE)
    print(f"\n{'='*60}")
    print(f"对比表: {cmp_path}")
    print(f"{'='*60}")

    if not cmp_path.exists():
        print("  (尚无数据)")
        return

    try:
        df = pd.read_csv(cmp_path, sep='\t')
        print(df.to_string(index=False))
    except Exception as e:
        print(f"  无法读取对比表: {e}")


if __name__ == '__main__':
    main()
