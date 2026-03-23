#!/usr/bin/env python3
"""
Autoresearch evaluation script for stop/target parameter optimization.
Uses pre-cached scores (from cache_scores_for_eval.py) for fast iteration.

Outputs a SINGLE number to stdout:
  composite = excess_annual_return_pct + sharpe * 5

Where excess = strategy_annual_return - benchmark_annual_return (中证2000)

Usage:
    # First time: build score cache (~30 min)
    python3 scripts/cache_scores_for_eval.py --version v4.8.1

    # Each iteration: fast eval (~30 sec)
    python3 scripts/eval_stop_target.py
"""
import sys
import os
import json
import time
import pickle
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'
PARAMS_PATH = PROJECT_ROOT / 'scripts' / 'stop_target_params.json'

# ── Config ──
VERSION = 'v4.8.1'
CACHE_PATH = PROJECT_ROOT / 'scripts' / f'.score_cache_{VERSION}.pkl'
HOLD_DAYS = 10
BENCHMARK_CODE = '932000.CSI'  # 中证2000


def load_params() -> dict:
    with open(PARAMS_PATH) as f:
        return json.load(f)


def compute_stop_target_buy(
    stock_code, close_price, pred_3d, pred_5d, pred_10d, pred_15d,
    risk_level, recommendation, params,
):
    """Parameterized stop/target/buy/position calculation."""
    if close_price <= 0:
        return {'buy_price': 0, 'stop_loss': 0, 'target': 0, 'position_pct': 0}

    primary_pred = pred_10d if pred_10d != 0 else pred_5d
    confidence = {'low': 0.85, 'medium': 0.6, 'high': 0.3}.get(risk_level, 0.5)

    is_wide = stock_code.startswith('30') or stock_code.startswith('688')
    daily_limit = 0.20 if is_wide else 0.10

    # ── Buy price ──
    buy_price = round(close_price * (1 - params['buy_discount']), 2)

    # ── Stop loss ──
    base_stop_pct = params['stop_pct_wide'] if is_wide else params['stop_pct_normal']
    enhanced_stop = close_price * (1 - base_stop_pct)

    if primary_pred < params['ml_tighten_threshold']:
        tighten_pct = min(abs(primary_pred) * params['ml_tighten_factor'], params['ml_tighten_cap'])
        enhanced_stop = enhanced_stop * (1 + tighten_pct)
    elif primary_pred > params['ml_loosen_threshold'] and confidence >= params['ml_loosen_confidence_min']:
        loosen_pct = min(primary_pred * params['ml_loosen_factor'], params['ml_loosen_cap'])
        enhanced_stop = enhanced_stop * (1 - loosen_pct)

    bearish_count = sum(1 for p in [pred_3d, pred_5d, pred_10d, pred_15d]
                        if p < params['bearish_pred_threshold'])
    if bearish_count >= params['bearish_count_threshold']:
        enhanced_stop = max(enhanced_stop, close_price * (1 - base_stop_pct + params['bearish_tighten']))

    min_stop = close_price * (1 - daily_limit)
    enhanced_stop = max(enhanced_stop, min_stop)
    enhanced_stop = max(enhanced_stop, buy_price * params['stop_lower_bound'])
    enhanced_stop = min(enhanced_stop, buy_price * params['stop_upper_bound'])

    # ── Target price ──
    max_t = params['target_max_wide'] if is_wide else params['target_max_normal']
    min_t = params['target_min_wide'] if is_wide else params['target_min_normal']
    ml_target_pct = max(min(primary_pred * params['target_ml_scale'], max_t), min_t)
    tech_target_pct = ml_target_pct

    if confidence >= params['blend_high_conf_threshold'] and primary_pred > params['blend_high_pred_threshold']:
        ml_w = params['blend_high_ml_w']
    elif confidence >= params['blend_mid_conf_threshold']:
        ml_w = params['blend_mid_ml_w']
    else:
        ml_w = params['blend_low_ml_w']
    tech_w = 1.0 - ml_w

    blended_pct = ml_target_pct * ml_w + tech_target_pct * tech_w
    blended_target = close_price * (1 + blended_pct)

    final_risk = buy_price - enhanced_stop
    final_reward = blended_target - buy_price
    if final_risk > 0 and final_reward / final_risk > params['rr_cap']:
        blended_target = buy_price + final_risk * params['rr_fallback']

    # ── Position size ──
    if recommendation in ('回避', '卖出'):
        position_pct = 0
    else:
        base = {
            '强烈买入': params['pos_strong_buy'],
            '买入': params['pos_buy'],
            '谨慎买入': params['pos_cautious'],
            '观望': params['pos_hold'],
        }.get(recommendation, params['pos_hold'])
        risk_adj = {
            'low': 0,
            'medium': params['pos_risk_adj_medium'],
            'high': params['pos_risk_adj_high'],
        }.get(risk_level, params['pos_risk_adj_medium'])
        position_pct = max(base + risk_adj, 0)

    return {
        'buy_price': round(buy_price, 2),
        'stop_loss': round(enhanced_stop, 2),
        'target': round(blended_target, 2),
        'position_pct': position_pct,
    }


def simulate_trade(code, analysis_date, buy_price, stop_loss, target,
                   all_quotes, hold_days=10):
    """Simulate a single trade using preloaded quote data."""
    try:
        code_quotes = all_quotes.loc[code]
    except KeyError:
        return None

    future = code_quotes[code_quotes.index > analysis_date].head(hold_days + 1)
    if future.empty:
        return None
    if buy_price <= 0 or stop_loss <= 0 or target <= 0:
        return None

    # Limit buy check
    entry_day = None
    actual_entry = None
    for idx in range(len(future)):
        row = future.iloc[idx]
        if row['low'] <= buy_price:
            entry_day = idx + 1
            actual_entry = min(row['open'], buy_price)
            break
        elif row['open'] <= buy_price * 1.005:
            entry_day = idx + 1
            actual_entry = row['open']
            break

    if entry_day is None:
        return {'outcome': 'no_fill', 'trade_return': 0, 'hold_return': 0}

    # Stop/target check
    remaining = future.iloc[entry_day:]
    target_hit_day = None
    stop_hit_day = None
    for idx in range(len(remaining)):
        row = remaining.iloc[idx]
        day_after = idx + 1
        if target_hit_day is None and row['high'] >= target:
            target_hit_day = day_after
        if stop_hit_day is None and row['low'] <= stop_loss:
            stop_hit_day = day_after

    exit_price = future.iloc[-1]['close']

    if target_hit_day and stop_hit_day:
        outcome = 'target_first' if target_hit_day <= stop_hit_day else 'stop_first'
    elif target_hit_day:
        outcome = 'target_only'
    elif stop_hit_day:
        outcome = 'stop_only'
    else:
        outcome = 'neither'

    hold_return = (exit_price - actual_entry) / actual_entry if actual_entry > 0 else 0

    if outcome in ('target_first', 'target_only'):
        trade_return = (target - actual_entry) / actual_entry
    elif outcome in ('stop_first', 'stop_only'):
        trade_return = (stop_loss - actual_entry) / actual_entry
    else:
        trade_return = hold_return

    return {'outcome': outcome, 'trade_return': trade_return, 'hold_return': hold_return}


def get_benchmark_return(start_date, end_date):
    """Get benchmark annualized return."""
    conn = sqlite3.connect(str(DB_PATH))
    query = """
        SELECT dq.trade_date, dq.close
        FROM daily_quotes dq JOIN securities s ON dq.security_id = s.id
        WHERE s.code = ? AND dq.trade_date >= ? AND dq.trade_date <= ?
        ORDER BY dq.trade_date
    """
    df = pd.read_sql_query(query, conn, params=(BENCHMARK_CODE, start_date, end_date))
    conn.close()
    if len(df) < 2:
        return 0.0
    total_return = df['close'].iloc[-1] / df['close'].iloc[0] - 1
    n_years = len(df) / 252
    return (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0


def run_eval():
    """Fast eval using pre-cached scores."""
    t0 = time.time()

    if not CACHE_PATH.exists():
        print("-999", flush=True)
        print("[eval] ERROR: Score cache not found. Run: "
              f"python3 scripts/cache_scores_for_eval.py --version {VERSION}",
              file=sys.stderr, flush=True)
        return

    params = load_params()

    # Load cache
    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)

    all_quotes = cache['all_quotes_index']
    daily_entries = cache['daily_entries']
    dates = cache['dates']

    if not daily_entries:
        print("-999", flush=True)
        return

    # Compute stop/target/buy for each stock and simulate trades
    all_results = []
    for entry in daily_entries:
        date = entry['date']
        for stock in entry['stocks']:
            prices = compute_stop_target_buy(
                stock['code'], stock['close'],
                stock['pred_3d'], stock['pred_5d'],
                stock['pred_10d'], stock['pred_15d'],
                stock['risk_level'], stock['recommendation'],
                params,
            )
            sim = simulate_trade(
                stock['code'], date,
                prices['buy_price'], prices['stop_loss'], prices['target'],
                all_quotes, HOLD_DAYS,
            )
            if sim is None:
                continue
            all_results.append({
                'analysis_date': date,
                'trade_return': sim['trade_return'],
                'hold_return': sim['hold_return'],
                'outcome': sim['outcome'],
                'position_pct': prices['position_pct'],
            })

    if not all_results:
        print("-999", flush=True)
        return

    df = pd.DataFrame(all_results)
    filled = df[df['outcome'] != 'no_fill']

    if len(filled) == 0:
        print("-999", flush=True)
        return

    # ── Strategy metrics (position-weighted) ──
    def pos_weighted_return(g):
        weights = g['position_pct'].values.astype(float)
        if weights.sum() == 0:
            return g['trade_return'].mean()
        return np.average(g['trade_return'].values, weights=weights)

    daily = filled.groupby('analysis_date').apply(
        pos_weighted_return, include_groups=False
    ).sort_index()
    cost_per_trade = 0.003
    daily_net = daily - cost_per_trade
    cumulative = (1 + daily_net).cumprod()
    total_return = cumulative.iloc[-1] - 1
    n_days = len(cumulative)
    n_years = n_days / 252

    strategy_ann = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    daily_excess_rf = daily_net - 0.02 / 252
    sharpe = (daily_excess_rf.mean() / daily_excess_rf.std() * np.sqrt(252)
              if daily_excess_rf.std() > 0 else 0)

    # ── Win rate & expected return (for info) ──
    wins = filled['outcome'].isin(['target_first', 'target_only']).sum()
    losses = filled['outcome'].isin(['stop_first', 'stop_only']).sum()
    n_decided = wins + losses
    win_rate = wins / n_decided * 100 if n_decided > 0 else 0

    # ── Benchmark ──
    benchmark_ann = get_benchmark_return(dates[0], dates[-1])

    # ── Composite metric ──
    excess_ann_pct = (strategy_ann - benchmark_ann) * 100
    composite = excess_ann_pct + sharpe * 5

    elapsed = time.time() - t0

    # Single number to stdout
    print(f"{composite:.2f}", flush=True)
    # Details to stderr
    print(f"[eval] {elapsed:.1f}s | excess={excess_ann_pct:.1f}% "
          f"(strategy={strategy_ann*100:.1f}% - bench={benchmark_ann*100:.1f}%) "
          f"sharpe={sharpe:.3f} win_rate={win_rate:.1f}% "
          f"trades={len(filled)} composite={composite:.2f}",
          file=sys.stderr, flush=True)


if __name__ == '__main__':
    run_eval()
