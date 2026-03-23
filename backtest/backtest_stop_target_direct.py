#!/usr/bin/env python3
"""
高效止盈止损回测 — 直接评分模式 (跳过报告生成)

核心优化:
  1. 批量预加载特征缓存 → 内存评分 (无文件I/O)
  2. 一次性预加载全时段日线数据 (open/high/low/close)
  3. 向量化止盈止损模拟
  4. ~2秒/天评分 + ~0.1秒/天模拟 = 500天 < 20分钟

用法:
    # V4.7.3 从2024-01-01回测 (默认top-20, 10天持仓)
    python3 backtest/backtest_stop_target_direct.py \
        --version v4.7.3 --start-date 2024-01-01

    # V4.7.5 自定义参数
    python3 backtest/backtest_stop_target_direct.py \
        --version v4.7.5 --start-date 2024-01-01 --top-n 10 --hold-days 10

    # 只看买入+强烈买入
    python3 backtest/backtest_stop_target_direct.py \
        --version v4.7.3 --start-date 2024-01-01 --filter-rec "强烈买入,买入"
"""

import sys
import os
import json
import time
import sqlite3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')

# ============================================================
# 复用 batch_generate_v395_reports 的基础设施
# ============================================================
from backtest.batch_generate_v395_reports import (
    get_trading_dates,
    fast_preload_feature_cache,
    load_securities_info,
    preload_daily_basic_bulk,
    score_all_stocks_from_preloaded,
)


def load_scorer(version: str):
    """加载ML评分器"""
    if version == 'v4.8.1':
        from ml_models.v39.v481_production_scorer import V481ProductionScorer
        return V481ProductionScorer()
    elif version == 'v4.7.9':
        from ml_models.v39.v479_production_scorer import V479ProductionScorer
        return V479ProductionScorer()
    elif version == 'v4.7.5':
        from ml_models.v39.v475_production_scorer import V475ProductionScorer
        return V475ProductionScorer()
    elif version == 'v4.7.3':
        from ml_models.v39.v473_production_scorer import V473ProductionScorer
        return V473ProductionScorer()
    elif version == 'v4.6':
        from ml_models.v39.v46_production_scorer import V46ProductionScorer
        return V46ProductionScorer()
    elif version == 'v4.4':
        from ml_models.v39.v44_production_scorer import V44ProductionScorer
        return V44ProductionScorer()
    elif version == 'v4.3':
        from ml_models.v39.v43_production_scorer import V43ProductionScorer
        return V43ProductionScorer()
    elif version == 'v3.9':
        from ml_models.v39.v390_production_scorer import V390ProductionScorer
        return V390ProductionScorer()
    else:
        raise ValueError(f"不支持的版本: {version}")


def preload_all_quotes(start_date: str, end_date: str) -> pd.DataFrame:
    """
    一次性预加载全时段日线数据 (open/high/low/close)

    Returns:
        DataFrame with MultiIndex (code, trade_date), columns: [open, high, low, close]
        已排序, 支持 .loc[code] 快速切片
    """
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT s.code, dq.trade_date, dq.open, dq.high, dq.low, dq.close
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.type = 'A股'
          AND dq.trade_date >= ? AND dq.trade_date <= ?
        ORDER BY s.code, dq.trade_date
    """
    df = pd.read_sql_query(query, conn, params=(start_date, end_date))
    conn.close()
    n_dates = df['trade_date'].nunique()
    n_codes = df['code'].nunique()
    print(f"  预加载日线: {len(df):,} 条 ({n_dates} 天, {n_codes} 只股票)", flush=True)

    # 设置MultiIndex便于快速 .loc[code] 切片
    df = df.set_index(['code', 'trade_date']).sort_index()
    return df


def get_recommendation(pred_3d, pred_5d, pred_10d, pred_15d, scorer=None):
    """生成投资建议 (复用scorer的阈值)"""
    # 尝试使用scorer的方法
    if scorer and hasattr(scorer, '_recommendation_from_composite'):
        return scorer._recommendation_from_composite(pred_3d, pred_5d, pred_10d, pred_15d)

    # fallback: composite绝对值
    composite = pred_3d * 0.1 + pred_5d * 0.2 + pred_10d * 0.4 + pred_15d * 0.3
    if composite >= 0.015:
        return '强烈买入'
    elif composite >= 0.008:
        return '买入'
    elif composite >= 0.003:
        return '谨慎买入'
    elif composite >= -0.002:
        return '观望'
    return '回避'


def get_risk_level(pred_3d, pred_5d, pred_10d, pred_15d, scorer=None):
    """生成风险等级"""
    if scorer and hasattr(scorer, '_risk_level_from_composite'):
        return scorer._risk_level_from_composite(pred_3d, pred_5d, pred_10d, pred_15d)

    composite = pred_3d * 0.1 + pred_5d * 0.2 + pred_10d * 0.4 + pred_15d * 0.3
    if composite >= 0.008:
        return 'low'
    elif composite >= 0.0:
        return 'medium'
    return 'high'


def compute_stop_target_buy(
    stock_code: str,
    close_price: float,
    pred_3d: float,
    pred_5d: float,
    pred_10d: float,
    pred_15d: float,
    risk_level: str,
    recommendation: str,
) -> dict:
    """
    计算止损/目标/买入价和仓位 (Autoresearch优化版 2026-03-23)

    优化结果 (V4.8.1, 536天回测):
    - 买入价: 收盘价-2.5% (vs旧-1%, 更优入场点)
    - 止损: 主板-10% / 创科-15% (vs旧-7%/-9%, 宽止损=少触发)
    - 目标: 主板+8% / 创科+12% (vs旧+3%/+4%, 高目标=大收益)
    - 仓位: 强信号集中 (强烈买入15%, 买入8%, 谨慎3%, 观望1%)

    Returns:
        {buy_price, stop_loss, target, position_pct}
    """
    if close_price <= 0:
        return {'buy_price': 0, 'stop_loss': 0, 'target': 0, 'position_pct': 0}

    primary_pred = pred_10d if pred_10d != 0 else pred_5d

    is_wide_limit = stock_code.startswith('30') or stock_code.startswith('688')
    daily_limit = 0.20 if is_wide_limit else 0.10

    # === 买入价: 收盘价下方2.5% ===
    buy_price = round(close_price * 0.975, 2)

    # === 止损价: 主板-10%, 创/科-15% ===
    base_stop_pct = 0.15 if is_wide_limit else 0.10
    enhanced_stop = close_price * (1 - base_stop_pct)

    # 约束
    min_stop = close_price * (1 - daily_limit)
    enhanced_stop = max(enhanced_stop, min_stop)
    enhanced_stop = max(enhanced_stop, buy_price * 0.85)
    enhanced_stop = min(enhanced_stop, buy_price * 0.96)

    # === 目标价: 主板+8%, 创/科+12% ===
    max_target_pct = 0.12 if is_wide_limit else 0.10
    min_target_pct = 0.12 if is_wide_limit else 0.08
    ml_target_pct = max(min(primary_pred * 0.8, max_target_pct), min_target_pct)
    tech_target_pct = ml_target_pct

    confidence = {'low': 0.85, 'medium': 0.6, 'high': 0.3}.get(risk_level, 0.5)
    if confidence >= 0.7 and primary_pred > 0.01:
        ml_w, tech_w = 0.65, 0.35
    elif confidence >= 0.4:
        ml_w, tech_w = 0.45, 0.55
    else:
        ml_w, tech_w = 0.25, 0.75

    blended_pct = ml_target_pct * ml_w + tech_target_pct * tech_w
    blended_target = close_price * (1 + blended_pct)

    # R:R上限约束
    final_risk = buy_price - enhanced_stop
    final_reward = blended_target - buy_price
    if final_risk > 0 and final_reward / final_risk > 3.0:
        blended_target = buy_price + final_risk * 2.5

    # === 仓位: 强信号集中 ===
    if recommendation in ('回避', '卖出'):
        position_pct = 0
    else:
        base = {'强烈买入': 15, '买入': 8, '谨慎买入': 3, '观望': 1}.get(recommendation, 1)
        risk_adj = {'low': 0, 'medium': -2, 'high': -4}.get(risk_level, -2)
        position_pct = max(base + risk_adj, 0)

    return {
        'buy_price': round(buy_price, 2),
        'stop_loss': round(enhanced_stop, 2),
        'target': round(blended_target, 2),
        'position_pct': position_pct,
    }


def simulate_trade(
    code: str,
    analysis_date: str,
    buy_price: float,
    stop_loss: float,
    target: float,
    all_quotes: pd.DataFrame,
    hold_days: int = 10,
) -> Optional[dict]:
    """
    模拟单笔交易: 限价买入 → 止盈/止损/到期退出

    使用预加载的MultiIndex日线数据, 零SQL查询
    """
    try:
        code_quotes = all_quotes.loc[code]
    except KeyError:
        return None

    # 找到 analysis_date 之后的交易日
    future = code_quotes[code_quotes.index > analysis_date].head(hold_days + 1)
    if future.empty:
        return None

    if buy_price <= 0 or stop_loss <= 0 or target <= 0:
        return None

    # Step 1: 限价买入检查
    entry_day = None
    actual_entry = None
    entry_date = None

    for idx in range(len(future)):
        row = future.iloc[idx]
        if row['low'] <= buy_price:
            entry_day = idx + 1
            actual_entry = min(row['open'], buy_price)
            entry_date = future.index[idx]
            break
        elif row['open'] <= buy_price * 1.005:
            entry_day = idx + 1
            actual_entry = row['open']
            entry_date = future.index[idx]
            break

    if entry_day is None:
        return {
            'outcome': 'no_fill',
            'actual_entry': 0,
            'exit_price': 0,
            'trade_return': 0,
            'hold_return': 0,
            'entry_day': 0,
            'target_hit_day': None,
            'stop_hit_day': None,
        }

    # Step 2: 止盈/止损检测 (从买入日之后)
    remaining = future.iloc[entry_day:]
    target_hit_day = None
    stop_hit_day = None

    for idx in range(len(remaining)):
        row = remaining.iloc[idx]
        day_after_entry = idx + 1
        if target_hit_day is None and row['high'] >= target:
            target_hit_day = day_after_entry
        if stop_hit_day is None and row['low'] <= stop_loss:
            stop_hit_day = day_after_entry

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

    return {
        'outcome': outcome,
        'actual_entry': actual_entry,
        'exit_price': exit_price,
        'trade_return': trade_return,
        'hold_return': hold_return,
        'entry_day': entry_day,
        'entry_date': entry_date,
        'target_hit_day': target_hit_day,
        'stop_hit_day': stop_hit_day,
    }


def print_group_stats(df: pd.DataFrame, label: str):
    """打印一组股票的统计"""
    total = len(df)
    if total == 0:
        print(f"  (无数据)")
        return

    filled = df[df['outcome'] != 'no_fill']
    no_fill = df[df['outcome'] == 'no_fill']
    n_filled = len(filled)

    print(f"  样本: {total}, 成交: {n_filled}, 未成交: {len(no_fill)}")
    if n_filled == 0:
        return

    wins = filled['outcome'].isin(['target_first', 'target_only']).sum()
    losses = filled['outcome'].isin(['stop_first', 'stop_only']).sum()
    neither = filled['outcome'].eq('neither').sum()
    wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    print(f"  止盈触发: {wins} ({wins/n_filled*100:.0f}%)  "
          f"止损触发: {losses} ({losses/n_filled*100:.0f}%)  "
          f"持有到期: {neither} ({neither/n_filled*100:.0f}%)")
    print(f"  胜率: {wr:.1f}%")

    avg_trade = filled['trade_return'].mean() * 100
    med_trade = filled['trade_return'].median() * 100
    print(f"  策略收益: 均值{avg_trade:+.2f}%, 中位{med_trade:+.2f}%")

    avg_hold = filled['hold_return'].mean() * 100
    print(f"  持有到期收益: 均值{avg_hold:+.2f}%")

    if filled['position_pct'].sum() > 0:
        weights = filled['position_pct'] / 100
        weighted_ret = (filled['trade_return'] * weights).sum() / weights.sum() * 100
        print(f"  仓位加权收益: {weighted_ret:+.2f}%")


def print_buy_group_detail(df: pd.DataFrame):
    """买入+强烈买入的详细分析"""
    buy_df = df[df['recommendation'].isin(['强烈买入', '买入'])]
    if len(buy_df) == 0:
        return

    buy_filled = buy_df[buy_df['outcome'] != 'no_fill']
    print(f"\n{'='*70}")
    print(f"🎯 重点: 买入+强烈买入 综合表现")
    print(f"{'='*70}")
    print_group_stats(buy_df, "买入类")

    if len(buy_filled) > 0:
        wins = buy_filled['outcome'].isin(['target_first', 'target_only']).sum()
        losses = buy_filled['outcome'].isin(['stop_first', 'stop_only']).sum()

        if wins > 0:
            avg_win = buy_filled.loc[
                buy_filled['outcome'].isin(['target_first', 'target_only']),
                'trade_return'
            ].mean()
            print(f"\n  平均盈利: {avg_win*100:+.2f}% ({wins}次)")
        if losses > 0:
            avg_loss = buy_filled.loc[
                buy_filled['outcome'].isin(['stop_first', 'stop_only']),
                'trade_return'
            ].mean()
            print(f"  平均亏损: {avg_loss*100:+.2f}% ({losses}次)")

        n_decided = wins + losses
        if n_decided > 0 and wins > 0 and losses > 0:
            p_win = wins / n_decided
            expected = p_win * avg_win + (1 - p_win) * avg_loss
            print(f"  期望收益(每笔): {expected*100:+.2f}%")


def print_monthly_breakdown(df: pd.DataFrame):
    """按月统计表现"""
    filled = df[df['outcome'] != 'no_fill'].copy()
    if len(filled) == 0:
        return

    filled['month'] = filled['analysis_date'].str[:7]
    monthly = filled.groupby('month').agg(
        trades=('trade_return', 'count'),
        avg_return=('trade_return', 'mean'),
        med_return=('trade_return', 'median'),
        win_rate=('outcome', lambda x: x.isin(['target_first', 'target_only']).mean()),
        stop_rate=('outcome', lambda x: x.isin(['stop_first', 'stop_only']).mean()),
    )

    print(f"\n{'='*70}")
    print(f"📅 月度分解")
    print(f"{'='*70}")
    print(f"{'月份':>8} | {'交易数':>5} | {'均值收益':>8} | {'中位收益':>8} | {'胜率':>6} | {'止损率':>6}")
    print(f"{'-'*8}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}")

    for month, row in monthly.iterrows():
        print(f"{month:>8} | {row['trades']:>5.0f} | "
              f"{row['avg_return']*100:>+7.2f}% | {row['med_return']*100:>+7.2f}% | "
              f"{row['win_rate']*100:>5.1f}% | {row['stop_rate']*100:>5.1f}%")

    # 汇总
    print(f"{'-'*8}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}")
    print(f"{'合计':>8} | {len(filled):>5} | "
          f"{filled['trade_return'].mean()*100:>+7.2f}% | "
          f"{filled['trade_return'].median()*100:>+7.2f}% | "
          f"{filled['outcome'].isin(['target_first','target_only']).mean()*100:>5.1f}% | "
          f"{filled['outcome'].isin(['stop_first','stop_only']).mean()*100:>5.1f}%")


def print_cumulative_equity(df: pd.DataFrame):
    """累计净值曲线摘要"""
    filled = df[df['outcome'] != 'no_fill'].copy()
    if len(filled) == 0:
        return

    # 按日期分组, 计算每日组合收益 (等权)
    daily = filled.groupby('analysis_date')['trade_return'].mean()
    daily = daily.sort_index()

    # 考虑交易成本 (双边0.3%)
    cost_per_trade = 0.003
    daily_net = daily - cost_per_trade

    cumulative = (1 + daily_net).cumprod()
    total_return = cumulative.iloc[-1] - 1
    n_days = len(cumulative)
    n_years = n_days / 252  # 近似交易日

    ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # 最大回撤
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()

    # Sharpe (假设无风险利率2%)
    daily_excess = daily_net - 0.02 / 252
    sharpe = daily_excess.mean() / daily_excess.std() * np.sqrt(252) if daily_excess.std() > 0 else 0

    print(f"\n{'='*70}")
    print(f"📈 累计净值 (等权, 扣除0.3%交易成本)")
    print(f"{'='*70}")
    print(f"  交易天数: {n_days} (约{n_years:.1f}年)")
    print(f"  累计收益: {total_return*100:+.1f}%")
    print(f"  年化收益: {ann_return*100:+.1f}%")
    print(f"  最大回撤: {max_dd*100:.1f}%")
    print(f"  Sharpe比: {sharpe:.3f}")
    print(f"  月度胜率: {(daily_net > 0).mean()*100:.1f}%")


def run_backtest(
    version: str,
    start_date: str,
    end_date: str,
    top_n: int = 20,
    hold_days: int = 10,
    filter_rec: str = None,
    rank_field: str = 'pred_10d',
):
    """主回测流程"""
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"ML-Enhanced 止盈止损直接回测")
    print(f"{'='*70}")
    print(f"版本: {version}, 区间: {start_date} → {end_date}", flush=True)
    print(f"Top-N: {top_n}, 持仓天数: {hold_days}, 排名字段: {rank_field}", flush=True)

    # Step 1: 加载评分器
    print(f"\n[1/5] 加载评分器...", flush=True)
    scorer = load_scorer(version)
    print(f"  评分器加载完成", flush=True)

    # Step 2: 获取交易日 & 预加载特征
    print(f"\n[2/5] 预加载特征缓存...", flush=True)
    dates = get_trading_dates(start_date, end_date, version)
    print(f"  交易日: {len(dates)} 天 ({dates[0]} → {dates[-1]})", flush=True)

    features_cache = fast_preload_feature_cache(dates)
    daily_basic_cache = preload_daily_basic_bulk(dates)
    securities_info = load_securities_info()
    t1 = time.time()
    print(f"  特征加载完成 ({t1-t0:.1f}秒)", flush=True)

    # Step 3: 预加载全时段日线数据
    print(f"\n[3/5] 预加载日线数据...", flush=True)
    # 需要包含end_date之后hold_days天的数据 (多加30天余量)
    from datetime import timedelta
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    quotes_end = (end_dt + timedelta(days=hold_days + 30)).strftime('%Y-%m-%d')
    all_quotes = preload_all_quotes(start_date, quotes_end)
    # 构建当日收盘价快速查询: {date: {code: close}}
    close_by_date = {}
    raw_quotes = all_quotes.reset_index()
    for date_val, grp in raw_quotes.groupby('trade_date'):
        close_by_date[date_val] = dict(zip(grp['code'], grp['close']))
    t2 = time.time()
    print(f"  日线加载完成 ({t2-t1:.1f}秒)", flush=True)

    # Step 4: 逐日评分 + 模拟交易
    print(f"\n[4/5] 逐日评分 + 交易模拟...", flush=True)
    all_results = []
    scored_days = 0

    for i, date in enumerate(dates):
        features_df = features_cache.get(date)
        daily_basic_df = daily_basic_cache.get(date)

        if features_df is None or len(features_df) == 0:
            continue

        # 评分
        scored = score_all_stocks_from_preloaded(
            scorer, features_df, date, daily_basic_df, version
        )

        if not scored:
            continue

        # 获取当日收盘价
        close_prices = close_by_date.get(date, {})

        # 为每只股票计算建议 & 止损/目标
        stocks_with_prices = []
        for code, data in scored.items():
            close = close_prices.get(code, 0)
            if close <= 0:
                continue

            pred_3d = data.get('pred_3d', 0)
            pred_5d = data.get('pred_5d', 0)
            pred_10d = data.get('pred_10d', 0)
            pred_15d = data.get('pred_15d', 0)

            rec = get_recommendation(pred_3d, pred_5d, pred_10d, pred_15d, scorer)
            risk = get_risk_level(pred_3d, pred_5d, pred_10d, pred_15d, scorer)

            prices = compute_stop_target_buy(
                code, close, pred_3d, pred_5d, pred_10d, pred_15d, risk, rec
            )

            # 排名值 (用于选top-N)
            if rank_field == 'composite':
                rank_val = pred_3d * 0.1 + pred_5d * 0.2 + pred_10d * 0.4 + pred_15d * 0.3
            elif rank_field == 'pred_10d':
                rank_val = pred_10d
            elif rank_field == 'score':
                rank_val = data.get('score', 0)
            else:
                rank_val = data.get(rank_field, pred_10d)

            stock_entry = {
                'code': code,
                'name': securities_info.get(code, {}).get('name', ''),
                'rank_val': rank_val,
                'close': close,
                'recommendation': rec,
                'risk_level': risk,
                'pred_3d': pred_3d,
                'pred_5d': pred_5d,
                'pred_10d': pred_10d,
                'pred_15d': pred_15d,
                **prices,
            }
            # V4.7.9: pass through market_confidence for dynamic top-N
            if 'market_confidence' in data:
                stock_entry['market_confidence'] = data['market_confidence']
            stocks_with_prices.append(stock_entry)

        # 按排名取 top-N (V4.7.9: dynamic top-N based on market confidence)
        stocks_with_prices.sort(key=lambda x: x['rank_val'], reverse=True)
        effective_top_n = top_n
        if version == 'v4.7.9' and stocks_with_prices:
            mc = stocks_with_prices[0].get('market_confidence', 1.0)
            from ml_models.v39.v479_production_scorer import V479ProductionScorer
            effective_top_n = V479ProductionScorer.compute_effective_top_n(top_n, mc)
        top_stocks = stocks_with_prices[:effective_top_n]

        # 模拟每只股票的交易
        for stock in top_stocks:
            sim = simulate_trade(
                stock['code'], date,
                stock['buy_price'], stock['stop_loss'], stock['target'],
                all_quotes, hold_days
            )
            if sim is None:
                continue

            all_results.append({
                'analysis_date': date,
                'stock_code': stock['code'],
                'stock_name': stock['name'],
                'recommendation': stock['recommendation'],
                'risk_level': stock['risk_level'],
                'position_pct': stock['position_pct'],
                'close_price': stock['close'],
                'buy_price': stock['buy_price'],
                'stop_loss': stock['stop_loss'],
                'target': stock['target'],
                'pred_10d': stock['pred_10d'],
                'rank_val': stock['rank_val'],
                **sim,
            })

        scored_days += 1
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t2
            speed = scored_days / elapsed if elapsed > 0 else 0
            print(f"  进度: {i+1}/{len(dates)} ({scored_days} 有效, "
                  f"{speed:.1f}天/秒, 累计{len(all_results)}笔)", flush=True)

    t3 = time.time()
    print(f"  评分+模拟完成: {scored_days}天, {len(all_results)}笔交易 ({t3-t2:.1f}秒)")

    if not all_results:
        print("没有有效回测结果")
        return

    # Step 5: 统计与报告
    print(f"\n[5/5] 生成统计报告...")
    df = pd.DataFrame(all_results)

    # 可选过滤
    if filter_rec:
        recs = [r.strip() for r in filter_rec.split(',')]
        df = df[df['recommendation'].isin(recs)]

    total = len(df)
    filled = df[df['outcome'] != 'no_fill']

    print(f"\n{'='*70}")
    print(f"ML-Enhanced 止盈止损回测报告 (直接评分模式)")
    print(f"{'='*70}")
    print(f"版本: {version}, 区间: {start_date} → {end_date}")
    print(f"有效交易日: {scored_days}, 排名: {rank_field}, Top-N: {top_n}")
    print(f"总样本: {total}, 限价成交: {len(filled)}, 未成交: {total-len(filled)}")
    print(f"持仓天数: {hold_days}天")
    print(f"总耗时: {time.time()-t0:.1f}秒")

    if len(filled) > 0:
        print(f"\n--- 总体表现 ---")
        print_group_stats(df, "全部")

    # 按投资建议分组
    rec_order = ['强烈买入', '买入', '谨慎买入', '观望', '回避']
    recs_in_data = [r for r in rec_order if r in df['recommendation'].values]

    if len(recs_in_data) > 1:
        print(f"\n{'='*70}")
        print(f"按投资建议分组")
        print(f"{'='*70}")
        for rec in recs_in_data:
            group = df[df['recommendation'] == rec]
            if len(group) == 0:
                continue
            print(f"\n📌 [{rec}]")
            print_group_stats(group, rec)

    # 买入+强烈买入详细分析
    print_buy_group_detail(df)

    # 月度分解
    print_monthly_breakdown(df)

    # 累计净值
    print_cumulative_equity(df)

    # 保存CSV
    output_dir = Path(PROJECT_ROOT) / "reports" / "target_price_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_dir / f"direct_backtest_{version}_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='ML-Enhanced 止盈止损直接回测')
    parser.add_argument('--version', type=str, default='v4.7.3',
                        help='ML版本 (v4.7.5/v4.7.3/v4.6/v4.4/v4.3/v3.9)')
    parser.add_argument('--start-date', type=str, default='2024-01-01',
                        help='回测起始日期 (default: 2024-01-01)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='回测结束日期 (default: 最新数据)')
    parser.add_argument('--top-n', type=int, default=20,
                        help='每日选股数量 (default: 20)')
    parser.add_argument('--hold-days', type=int, default=10,
                        help='持仓天数 (default: 10)')
    parser.add_argument('--filter-rec', type=str, default=None,
                        help='过滤投资建议 (如: "强烈买入,买入")')
    parser.add_argument('--rank-field', type=str, default='pred_10d',
                        help='排名字段 (pred_10d/composite/score, default: pred_10d)')

    args = parser.parse_args()

    if args.end_date is None:
        # 获取最新数据日期
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT MAX(trade_date) FROM v39_feature_cache"
        ).fetchone()
        conn.close()
        args.end_date = row[0] if row else '2026-03-15'

    run_backtest(
        version=args.version,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=args.top_n,
        hold_days=args.hold_days,
        filter_rec=args.filter_rec,
        rank_field=args.rank_field,
    )


if __name__ == '__main__':
    main()
