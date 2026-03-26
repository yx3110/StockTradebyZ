#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iterative Pipeline — L1/L2 Orchestrator

Ties L1 fast training to L2 mini-report generation and North Star evaluation.

Functions:
    run_l1(params)              — Thin wrapper around L1FastTrainer
    run_l2(params, l1_result)  — Mini-report generation + NS evaluation
    _check_l2_gate(mini_ns, metrics) — Gate conditions
    _calibrate_ns(mini_ns_raw)       — Linear calibration from l2_calibration.json

作者: Claude Code
创建时间: 2026-03-26
"""

import sys
import os
import json
import time
import tempfile
import logging
from pathlib import Path
from typing import Dict, Optional, Any

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
# Internal utilities
# ==================================================================

def _error_result(variant_name: str, msg: str) -> dict:
    """Build a standard error result dict."""
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


# ==================================================================
# CLI entry point
# ==================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Iterative Pipeline L1+L2')
    parser.add_argument('--variant', default='pipeline_test', help='变体名称')
    parser.add_argument('--start-date', default=None, help='L1 训练起始日期')
    parser.add_argument('--num-boost-round', type=int, default=150, help='LGB 迭代次数')
    parser.add_argument('--n-days', type=int, default=60, help='L2 评估天数')
    parser.add_argument('--top-n', type=int, default=10, help='Top N 选股')
    parser.add_argument('--focus-days', type=int, default=10, help='重点持仓天数')
    parser.add_argument('--rank-field', default='pred_10d', help='排名字段')
    parser.add_argument('--skip-l2', action='store_true', help='仅运行 L1 不运行 L2')
    args = parser.parse_args()

    params = {
        'variant_name': args.variant,
        'training': {
            'l1_num_boost_round': args.num_boost_round,
            'purge_days': 10,
        },
        'scoring': {
            'top_n': args.top_n,
            'focus_days': args.focus_days,
            'rank_field': args.rank_field,
        },
    }
    if args.start_date:
        start = args.start_date.replace('-', '')
        if len(start) == 8:
            start = f"{start[:4]}-{start[4:6]}-{start[6:8]}"
        params['training']['l1_start_date'] = start

    l1_result = run_l1(params)
    print(f"\n[L1] gate_pass={l1_result['gate_pass']}")

    if args.skip_l2:
        sys.exit(0 if l1_result['gate_pass'] else 1)

    if not l1_result['gate_pass']:
        print("[L1] L1 gate failed, skipping L2")
        sys.exit(1)

    l2_result = run_l2(params, l1_result, n_days=args.n_days)
    print(f"\n[L2] gate_pass={l2_result['gate_pass']}")
    sys.exit(0 if l2_result['gate_pass'] else 1)
