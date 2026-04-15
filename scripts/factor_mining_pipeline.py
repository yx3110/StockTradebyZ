#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动因子挖掘 + 筛选 Pipeline

WorldQuant风格: 算子×操作数×窗口 组合生成候选因子 → IC/ICIR/头部IC筛选 → 去冗余

用法:
    # 快速模式 (~10min, 500+候选)
    python3 scripts/factor_mining_pipeline.py

    # 深度模式 (~30min, 2000+候选, depth=2)
    python3 scripts/factor_mining_pipeline.py --depth 2

    # 指定时间范围
    python3 scripts/factor_mining_pipeline.py --start-date 2023-01-01 --end-date 2026-02-13

输出:
    scripts/mined_factors_results.json  — 通过筛选的因子列表+公式+IC
    stdout: 排名表
"""

import sys
import os
import json
import time
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product, combinations
from scipy.stats import spearmanr, rankdata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')

# ============================================================
# 1. 算子定义
# ============================================================

def ts_mean(x, d):
    return x.rolling(d, min_periods=max(d//2, 3)).mean()

def ts_std(x, d):
    return x.rolling(d, min_periods=max(d//2, 3)).std()

def ts_rank(x, d):
    return x.rolling(d, min_periods=max(d//2, 3)).apply(
        lambda v: rankdata(v)[-1] / len(v), raw=True)

def ts_delta(x, d):
    return x - x.shift(d)

def ts_ret(x, d):
    shifted = x.shift(d)
    return (x - shifted) / shifted.clip(lower=1e-8)

def ts_max(x, d):
    return x.rolling(d, min_periods=max(d//2, 3)).max()

def ts_min(x, d):
    return x.rolling(d, min_periods=max(d//2, 3)).min()

def ts_maxpos(x, d):
    """max值在窗口中的位置 (0=最早, 1=最新)"""
    return x.rolling(d, min_periods=max(d//2, 3)).apply(
        lambda v: np.argmax(v) / max(len(v)-1, 1), raw=True)

def ts_minpos(x, d):
    return x.rolling(d, min_periods=max(d//2, 3)).apply(
        lambda v: np.argmin(v) / max(len(v)-1, 1), raw=True)

def ts_skew(x, d):
    return x.rolling(d, min_periods=max(d//2, 5)).skew()

def ts_kurt(x, d):
    return x.rolling(d, min_periods=max(d//2, 5)).kurt()

def ts_zscore(x, d):
    m = x.rolling(d, min_periods=max(d//2, 3)).mean()
    s = x.rolling(d, min_periods=max(d//2, 3)).std().clip(lower=1e-8)
    return (x - m) / s

def ts_decay_linear(x, d):
    """线性衰减加权均值"""
    weights = np.arange(1, d+1, dtype=float)
    weights /= weights.sum()
    return x.rolling(d, min_periods=max(d//2, 3)).apply(
        lambda v: np.dot(v[-len(weights):], weights[-len(v):]) if len(v) >= d//2 else np.nan, raw=True)

# 二元时序算子
def ts_corr(x, y, d):
    return x.rolling(d, min_periods=max(d//2, 5)).corr(y)

def ts_cov(x, y, d):
    return x.rolling(d, min_periods=max(d//2, 5)).cov(y)

# 元素算子
def op_abs(x):
    return x.abs()

def op_neg(x):
    return -x

def op_sign(x):
    return np.sign(x)

def op_log(x):
    return np.log(x.clip(lower=1e-8))

def op_inv(x):
    return 1.0 / x.clip(lower=1e-8)


# ============================================================
# 2. 因子生成器
# ============================================================

# 一元时序算子
UNARY_TS_OPS = {
    'ts_mean': ts_mean,
    'ts_std': ts_std,
    'ts_rank': ts_rank,
    'ts_delta': ts_delta,
    'ts_ret': ts_ret,
    'ts_zscore': ts_zscore,
    'ts_decay': ts_decay_linear,
    'ts_skew': ts_skew,
    'ts_maxpos': ts_maxpos,
    'ts_minpos': ts_minpos,
}

# 二元时序算子
BINARY_TS_OPS = {
    'ts_corr': ts_corr,
    'ts_cov': ts_cov,
}

# 元素算子 (用于depth=2组合)
ELEM_OPS = {
    'abs': op_abs,
    'neg': op_neg,
    'sign': op_sign,
}

WINDOWS = [5, 10, 20, 60]


def generate_operands(df_stock):
    """从单只股票的OHLCV数据生成操作数"""
    close = pd.Series(df_stock['close'].values, dtype=float)
    opn = pd.Series(df_stock['open'].values, dtype=float)
    high = pd.Series(df_stock['high'].values, dtype=float)
    low = pd.Series(df_stock['low'].values, dtype=float)
    volume = pd.Series(df_stock['volume'].values, dtype=float)
    pct = pd.Series(df_stock['price_change_pct'].values, dtype=float)
    turnover = pd.Series(df_stock.get('turnover_rate', pd.Series(np.zeros(len(close)))).values, dtype=float)

    hl_range = high - low
    body = close - opn
    hl_safe = hl_range.clip(lower=1e-8)
    position = (close - low) / hl_safe  # K线位置

    return {
        'close': close,
        'open': opn,
        'high': high,
        'low': low,
        'volume': volume,
        'ret': pct,
        'turnover': turnover,
        'range': hl_range,
        'body': body,
        'position': position,
        'vwap_approx': (high + low + close) / 3,
    }


def generate_depth1_factors():
    """生成depth=1因子候选列表: op(operand, window)"""
    candidates = []
    operand_names = ['close', 'open', 'high', 'low', 'volume', 'ret',
                     'turnover', 'range', 'body', 'position', 'vwap_approx']

    # 一元时序
    for op_name, op_func in UNARY_TS_OPS.items():
        for operand in operand_names:
            for w in WINDOWS:
                name = f'{op_name}({operand},{w})'
                candidates.append({
                    'name': name,
                    'type': 'unary_ts',
                    'op': op_name,
                    'operand': operand,
                    'window': w,
                })

    # 二元时序 (operand pairs)
    pair_operands = ['close', 'volume', 'ret', 'turnover', 'high', 'low']
    for op_name, op_func in BINARY_TS_OPS.items():
        for a, b in combinations(pair_operands, 2):
            for w in WINDOWS:
                name = f'{op_name}({a},{b},{w})'
                candidates.append({
                    'name': name,
                    'type': 'binary_ts',
                    'op': op_name,
                    'operand_a': a,
                    'operand_b': b,
                    'window': w,
                })

    return candidates


def generate_depth2_factors():
    """生成depth=2因子: elem_op(ts_op(operand, window))"""
    candidates = []
    operand_names = ['close', 'ret', 'volume', 'turnover', 'position']
    # 限制组合数: 精选算子对
    ts_ops_d2 = ['ts_mean', 'ts_std', 'ts_ret', 'ts_rank']
    elem_ops_d2 = ['abs', 'neg', 'sign']

    for elem_name in elem_ops_d2:
        for ts_name in ts_ops_d2:
            for operand in operand_names:
                for w in WINDOWS:
                    name = f'{elem_name}({ts_name}({operand},{w}))'
                    candidates.append({
                        'name': name,
                        'type': 'depth2',
                        'elem_op': elem_name,
                        'ts_op': ts_name,
                        'operand': operand,
                        'window': w,
                    })

    # ts_op1(ts_op2(operand, w2), w1) — 精选
    for ts1 in ['ts_rank', 'ts_zscore']:
        for ts2 in ['ts_ret', 'ts_std', 'ts_mean']:
            for operand in ['close', 'ret', 'volume']:
                for w1 in [10, 20]:
                    for w2 in [5, 20]:
                        if w1 != w2:
                            name = f'{ts1}({ts2}({operand},{w2}),{w1})'
                            candidates.append({
                                'name': name,
                                'type': 'depth2_ts',
                                'ts_op1': ts1,
                                'ts_op2': ts2,
                                'operand': operand,
                                'window1': w1,
                                'window2': w2,
                            })

    return candidates


# ============================================================
# 3. 因子计算引擎
# ============================================================

def compute_factor(candidate, operands):
    """计算单个因子值 (返回pd.Series)"""
    try:
        ftype = candidate['type']

        if ftype == 'unary_ts':
            x = operands[candidate['operand']]
            func = UNARY_TS_OPS[candidate['op']]
            return func(x, candidate['window'])

        elif ftype == 'binary_ts':
            a = operands[candidate['operand_a']]
            b = operands[candidate['operand_b']]
            func = BINARY_TS_OPS[candidate['op']]
            return func(a, b, candidate['window'])

        elif ftype == 'depth2':
            x = operands[candidate['operand']]
            ts_func = UNARY_TS_OPS[candidate['ts_op']]
            elem_func = ELEM_OPS[candidate['elem_op']]
            intermediate = ts_func(x, candidate['window'])
            return elem_func(intermediate)

        elif ftype == 'depth2_ts':
            x = operands[candidate['operand']]
            ts_func2 = UNARY_TS_OPS[candidate['ts_op2']]
            ts_func1 = UNARY_TS_OPS[candidate['ts_op1']]
            intermediate = ts_func2(x, candidate['window2'])
            return ts_func1(intermediate, candidate['window1'])

    except Exception:
        return None
    return None


def compute_all_factors_for_stock(candidates, df_stock):
    """计算单只股票所有候选因子"""
    operands = generate_operands(df_stock)
    results = {}
    for c in candidates:
        val = compute_factor(c, operands)
        if val is not None:
            results[c['name']] = val.values
    return results


# ============================================================
# 4. IC筛选引擎
# ============================================================

def compute_factor_ic(factor_df, label_col='label_10d', min_stocks=100):
    """计算因子的日截面IC序列

    Args:
        factor_df: DataFrame with columns [trade_date, code, factor_value, label_10d]
        label_col: label column name
        min_stocks: 每天最少股票数

    Returns:
        dict: {ic_mean, ic_std, icir, ic_positive_pct, top_quintile_ic, n_days}
    """
    ics = []
    top_q_ics = []

    for date, grp in factor_df.groupby('trade_date'):
        g = grp.dropna(subset=['factor_value', label_col])
        if len(g) < min_stocks:
            continue

        fv = g['factor_value'].values
        rv = g[label_col].values

        # 全局IC
        try:
            ic, _ = spearmanr(fv, rv)
            if np.isfinite(ic):
                ics.append(ic)
        except Exception:
            continue

        # Top-Quintile IC (只在top 20%因子值中计算)
        try:
            threshold = np.percentile(fv, 80)
            mask = fv >= threshold
            if mask.sum() >= 20:
                tq_ic, _ = spearmanr(fv[mask], rv[mask])
                if np.isfinite(tq_ic):
                    top_q_ics.append(tq_ic)
        except Exception:
            pass

    if len(ics) < 30:
        return None

    ics = np.array(ics)
    ic_mean = float(np.mean(ics))
    ic_std = float(np.std(ics))
    icir = ic_mean / max(ic_std, 1e-8)

    result = {
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        'ic_positive_pct': float(np.mean(ics > 0)),
        'n_days': len(ics),
    }

    if top_q_ics:
        tq = np.array(top_q_ics)
        result['top_quintile_ic'] = float(np.mean(tq))
        result['top_quintile_icir'] = float(np.mean(tq) / max(np.std(tq), 1e-8))
    else:
        result['top_quintile_ic'] = 0.0
        result['top_quintile_icir'] = 0.0

    return result


# ============================================================
# 5. 去冗余 (相关性过滤)
# ============================================================

def remove_redundant_factors(factor_values_dict, max_corr=0.7):
    """移除高相关性因子, 保留IC更高的"""
    names = list(factor_values_dict.keys())
    if len(names) <= 1:
        return names

    # 构建因子值矩阵 (sample rows)
    all_vals = []
    valid_names = []
    for name in names:
        v = factor_values_dict[name]
        if len(v) > 0:
            all_vals.append(v)
            valid_names.append(name)

    if len(valid_names) <= 1:
        return valid_names

    # 采样计算相关性 (全量太慢)
    # 因子长度不一致时取最小长度防止索引越界 (不同因子 NaN 数量不同)
    n = min(len(v) for v in all_vals)
    sample_idx = np.random.choice(n, min(n, 50000), replace=False)
    mat = np.column_stack([v[:n][sample_idx] for v in all_vals])

    # 处理NaN
    mat = np.nan_to_num(mat, nan=0.0)
    corr = np.corrcoef(mat.T)

    # 贪心去冗余: 按|IC|降序, 已选因子与候选相关>0.7则跳过
    selected = []
    for i in range(len(valid_names)):
        redundant = False
        for j in selected:
            if abs(corr[i, j]) > max_corr:
                redundant = True
                break
        if not redundant:
            selected.append(i)

    return [valid_names[i] for i in selected]


# ============================================================
# 6. 主Pipeline
# ============================================================

def load_data(start_date='2023-01-01', end_date='2026-02-13'):
    """加载OHLCV + labels"""
    conn = sqlite3.connect(DB_PATH)

    print(f"  加载OHLCV数据 ({start_date} → {end_date})...", flush=True)
    from datetime import datetime as dt_cls, timedelta as td_cls
    ext_start = (dt_cls.strptime(start_date, '%Y-%m-%d') - td_cls(days=70)).strftime('%Y-%m-%d')

    df = pd.read_sql("""
        SELECT s.code, q.trade_date, q.open, q.high, q.low, q.close,
               q.volume, q.price_change_pct
        FROM daily_quotes q
        JOIN securities s ON q.security_id = s.id
        WHERE s.type = 'A股' AND q.trade_date >= ? AND q.trade_date <= ?
        ORDER BY s.code, q.trade_date
    """, conn, params=[ext_start, end_date])

    # turnover_rate
    df_turn = pd.read_sql("""
        SELECT s.code, db.trade_date, db.turnover_rate
        FROM daily_basic db
        JOIN securities s ON db.security_id = s.id
        WHERE db.trade_date >= ? AND db.trade_date <= ?
    """, conn, params=[ext_start, end_date])
    if not df_turn.empty:
        df = df.merge(df_turn, on=['code', 'trade_date'], how='left')
        df['turnover_rate'] = df['turnover_rate'].fillna(0.0)

    print(f"  OHLCV: {len(df):,} rows, {df['code'].nunique()} stocks", flush=True)

    # labels (10d forward return)
    print(f"  计算10d forward labels...", flush=True)
    df = df.sort_values(['code', 'trade_date']).reset_index(drop=True)
    df['label_10d'] = df.groupby('code')['close'].transform(
        lambda x: x.shift(-10) / x - 1)

    # 过滤到目标日期范围
    df_eval = df[df['trade_date'] >= start_date].copy()
    print(f"  评估数据: {len(df_eval):,} rows ({df_eval['trade_date'].min()} → {df_eval['trade_date'].max()})", flush=True)

    conn.close()
    return df, df_eval


def run_pipeline(start_date='2023-01-01', end_date='2026-02-13',
                 depth=1, max_corr=0.7, min_icir=0.3, min_ic=0.02,
                 n_sample_stocks=500):
    """运行完整pipeline"""
    t0 = time.time()

    # 1. 生成候选
    print("\n=== Phase 1: 生成候选因子 ===", flush=True)
    candidates = generate_depth1_factors()
    if depth >= 2:
        candidates += generate_depth2_factors()
    print(f"  候选因子: {len(candidates)}", flush=True)

    # 2. 加载数据
    print("\n=== Phase 2: 加载数据 ===", flush=True)
    df_full, df_eval = load_data(start_date, end_date)

    # 3. 采样股票计算因子 (全A股太慢)
    print(f"\n=== Phase 3: 计算因子 (采样{n_sample_stocks}只股票) ===", flush=True)
    all_codes = df_full['code'].unique()
    if len(all_codes) > n_sample_stocks:
        sample_codes = np.random.choice(all_codes, n_sample_stocks, replace=False)
    else:
        sample_codes = all_codes

    # 逐股票计算所有因子
    factor_data = {c['name']: [] for c in candidates}
    meta_data = []  # (code, trade_date, label_10d)

    processed = 0
    for code in sample_codes:
        stock_df = df_full[df_full['code'] == code].sort_values('trade_date')
        if len(stock_df) < 60:
            continue

        # 计算因子
        stock_factors = compute_all_factors_for_stock(candidates, stock_df)

        # 只保留评估日期范围
        mask = stock_df['trade_date'].values >= start_date
        for fname, vals in stock_factors.items():
            factor_data[fname].append(pd.DataFrame({
                'code': code,
                'trade_date': stock_df['trade_date'].values[mask],
                'factor_value': vals[mask],
                'label_10d': stock_df['label_10d'].values[mask],
            }))

        processed += 1
        if processed % 100 == 0:
            print(f"  进度: {processed}/{len(sample_codes)} stocks", flush=True)

    print(f"  完成: {processed} stocks", flush=True)

    # 4. IC筛选
    print(f"\n=== Phase 4: IC筛选 (|IC|>{min_ic}, |ICIR|>{min_icir}) ===", flush=True)
    results = []
    factor_values_for_dedup = {}

    for fname, parts in factor_data.items():
        if not parts:
            continue
        df_factor = pd.concat(parts, ignore_index=True)
        df_factor = df_factor.dropna(subset=['factor_value', 'label_10d'])

        if len(df_factor) < 1000:
            continue

        metrics = compute_factor_ic(df_factor, 'label_10d')
        if metrics is None:
            continue

        # 筛选条件
        if abs(metrics['ic_mean']) >= min_ic and abs(metrics['icir']) >= min_icir:
            # 合并 candidate 结构字段 (type/op/operand/window) 以便下游 reconstruct factor
            candidate = next((c for c in candidates if c['name'] == fname), {})
            results.append({
                **candidate,  # type, op, operand[_a/_b], window, elem_op, ts_op 等
                'name': fname,
                **metrics,
            })
            # 保存因子值用于去冗余
            factor_values_for_dedup[fname] = df_factor['factor_value'].values

    print(f"  通过IC筛选: {len(results)}/{len(candidates)}", flush=True)

    if not results:
        print("\n没有因子通过筛选!", flush=True)
        return []

    # 按|ICIR|排序
    results.sort(key=lambda x: abs(x['icir']), reverse=True)

    # 5. 去冗余
    print(f"\n=== Phase 5: 去冗余 (max_corr={max_corr}) ===", flush=True)
    # 按ICIR排序后去冗余
    sorted_names = [r['name'] for r in results]
    sorted_values = {name: factor_values_for_dedup[name]
                     for name in sorted_names if name in factor_values_for_dedup}
    kept_names = remove_redundant_factors(sorted_values, max_corr)
    final_results = [r for r in results if r['name'] in kept_names]

    print(f"  去冗余后: {len(final_results)}/{len(results)}", flush=True)

    # 6. 输出
    elapsed = time.time() - t0
    print(f"\n{'='*70}", flush=True)
    print(f"因子挖掘完成: {elapsed/60:.1f}分钟", flush=True)
    print(f"候选: {len(candidates)} → IC筛选: {len(results)} → 去冗余: {len(final_results)}", flush=True)
    print(f"{'='*70}", flush=True)

    print(f"\n{'排名':<4} {'因子':<45} {'IC':>8} {'ICIR':>8} {'IC>0%':>7} {'TopQ_IC':>8} {'TopQ_ICIR':>9}", flush=True)
    print('-' * 95, flush=True)
    for i, r in enumerate(final_results[:50]):
        print(f"{i+1:<4} {r['name']:<45} {r['ic_mean']:>+8.4f} {r['icir']:>+8.3f} "
              f"{r['ic_positive_pct']*100:>6.1f}% {r.get('top_quintile_ic',0):>+8.4f} "
              f"{r.get('top_quintile_icir',0):>+9.3f}", flush=True)

    # 保存结果
    output_path = PROJECT_ROOT / 'scripts' / 'mined_factors_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'start_date': start_date,
                'end_date': end_date,
                'depth': depth,
                'n_sample_stocks': n_sample_stocks,
                'min_ic': min_ic,
                'min_icir': min_icir,
                'max_corr': max_corr,
            },
            'summary': {
                'candidates': len(candidates),
                'passed_ic': len(results),
                'after_dedup': len(final_results),
                'elapsed_minutes': round(elapsed/60, 1),
            },
            'factors': final_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_path}", flush=True)

    return final_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='自动因子挖掘Pipeline')
    parser.add_argument('--start-date', default='2023-01-01')
    parser.add_argument('--end-date', default='2026-02-13')
    parser.add_argument('--depth', type=int, default=1, choices=[1, 2],
                        help='因子公式深度 (1=单算子, 2=双算子组合)')
    parser.add_argument('--n-stocks', type=int, default=500,
                        help='采样股票数 (默认500)')
    parser.add_argument('--min-icir', type=float, default=0.3,
                        help='最小|ICIR|阈值')
    parser.add_argument('--min-ic', type=float, default=0.02,
                        help='最小|IC|阈值')
    parser.add_argument('--max-corr', type=float, default=0.7,
                        help='去冗余最大相关系数')
    args = parser.parse_args()

    run_pipeline(
        start_date=args.start_date,
        end_date=args.end_date,
        depth=args.depth,
        n_sample_stocks=args.n_stocks,
        min_icir=args.min_icir,
        min_ic=args.min_ic,
        max_corr=args.max_corr,
    )
