#!/usr/bin/env python3
"""
对比回测：策略筛选 vs 全市场评分

对V4.7.3和V4.7.5模型，比较：
1. 全市场评分 (无策略筛选，现有报告)
2. 策略筛选后评分 (仅保留8大策略选中的股票)

方法：
- 批量加载历史行情+技术指标
- 对每个交易日运行8大策略，获取候选股票列表
- 过滤现有全市场报告，仅保留策略候选
- 分别回测两种模式并对比
"""

import sys
import os
import json
import time
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')


def load_all_stock_data_from_db(start_date: str, end_date: str) -> dict:
    """
    从DB批量加载所有A股的OHLCV + 技术指标数据
    分两步加载避免巨大JOIN操作

    Returns:
        {code: DataFrame} with columns: date, open, high, low, close, volume,
        BBI, K, D, J, DIF, ZXDQ, ZXDKX, MA60
    """
    conn = sqlite3.connect(DB_PATH)

    # 需要额外200天历史用于策略计算 (最远策略MA60需60天+指标计算需120天=200天)
    lookback_date = (pd.Timestamp(start_date) - pd.Timedelta(days=500)).strftime('%Y-%m-%d')

    # Step 1: 加载security_id -> code映射
    print(f"  加载证券映射...")
    sec_map = {}
    rows = conn.execute("SELECT id, code FROM securities WHERE type = 'A股'").fetchall()
    for sid, code in rows:
        sec_map[sid] = code
    sec_ids = list(sec_map.keys())
    print(f"  A股: {len(sec_ids)} 只")

    # Step 2: 分块加载daily_quotes
    print(f"  加载行情数据 ({lookback_date} ~ {end_date})...")
    t0 = time.time()

    CHUNK = 500
    dq_frames = []
    for i in range(0, len(sec_ids), CHUNK):
        chunk_ids = sec_ids[i:i+CHUNK]
        placeholders = ','.join(['?' for _ in chunk_ids])
        q = f"""SELECT security_id, trade_date, open, high, low, close, volume
                FROM daily_quotes
                WHERE security_id IN ({placeholders})
                  AND trade_date >= ? AND trade_date <= ?"""
        df = pd.read_sql_query(q, conn, params=chunk_ids + [lookback_date, end_date])
        dq_frames.append(df)
        if (i // CHUNK + 1) % 4 == 0:
            print(f"    行情: {i+CHUNK}/{len(sec_ids)}...")

    dq_all = pd.concat(dq_frames, ignore_index=True)
    dq_all['code'] = dq_all['security_id'].map(sec_map)
    print(f"  行情加载完成: {len(dq_all)} 行 ({time.time()-t0:.1f}秒)")

    # Step 3: 分块加载technical_indicators (只取需要的列)
    print(f"  加载技术指标...")
    t1 = time.time()

    ti_frames = []
    for i in range(0, len(sec_ids), CHUNK):
        chunk_ids = sec_ids[i:i+CHUNK]
        placeholders = ','.join(['?' for _ in chunk_ids])
        q = f"""SELECT security_id, trade_date,
                       bbi, kdj_k, kdj_d, kdj_j, macd_dif,
                       zhixing_short_trend, zhixing_multi_kong
                FROM technical_indicators
                WHERE security_id IN ({placeholders})
                  AND trade_date >= ? AND trade_date <= ?"""
        df = pd.read_sql_query(q, conn, params=chunk_ids + [lookback_date, end_date])
        ti_frames.append(df)
        if (i // CHUNK + 1) % 4 == 0:
            print(f"    指标: {i+CHUNK}/{len(sec_ids)}...")

    conn.close()

    ti_all = pd.concat(ti_frames, ignore_index=True)
    print(f"  指标加载完成: {len(ti_all)} 行 ({time.time()-t1:.1f}秒)")

    # Step 4: 在pandas中合并
    print(f"  合并数据...")
    t2 = time.time()

    merged = dq_all.merge(
        ti_all,
        on=['security_id', 'trade_date'],
        how='left'
    )
    merged.rename(columns={
        'trade_date': 'date',
        'bbi': 'BBI', 'kdj_k': 'K', 'kdj_d': 'D', 'kdj_j': 'J',
        'macd_dif': 'DIF',
        'zhixing_short_trend': 'ZXDQ', 'zhixing_multi_kong': 'ZXDKX'
    }, inplace=True)

    del dq_all, ti_all, dq_frames, ti_frames

    print(f"  合并完成: {len(merged)} 行 ({time.time()-t2:.1f}秒)")

    # Step 5: 按股票分组
    print(f"  构建股票数据字典 + 计算MA60...")
    t3 = time.time()

    stock_data = {}
    for code, group in merged.groupby('code'):
        sdf = group[['date', 'open', 'high', 'low', 'close', 'volume',
                      'BBI', 'K', 'D', 'J', 'DIF', 'ZXDQ', 'ZXDKX']].copy()
        sdf['date'] = pd.to_datetime(sdf['date'])
        sdf = sdf.sort_values('date').reset_index(drop=True)
        sdf['MA60'] = sdf['close'].rolling(60, min_periods=60).mean()

        for col in ['BBI', 'K', 'D', 'J', 'DIF', 'ZXDQ', 'ZXDKX']:
            sdf[col] = pd.to_numeric(sdf[col], errors='coerce')

        stock_data[code] = sdf

    del merged
    print(f"  构建完成: {len(stock_data)} 只股票 ({time.time()-t3:.1f}秒)")
    print(f"  总数据加载时间: {time.time()-t0:.1f}秒")
    return stock_data


def run_strategies_for_date(stock_data: dict, target_date: pd.Timestamp) -> set:
    """
    对指定日期运行8大策略，返回被选中的股票集合

    使用DB预计算的指标值，直接检查策略条件，
    而非通过Selector.py (避免重复计算指标)
    """
    selected = set()
    target_str = target_date.strftime('%Y-%m-%d')

    for code, df in stock_data.items():
        # 截取到目标日期
        hist = df[df['date'] <= target_date]
        if len(hist) < 20:
            continue

        last = hist.iloc[-1]

        # 快速排除: 无效数据
        if pd.isna(last['close']) or last['close'] <= 0:
            continue
        if pd.isna(last['volume']) or last['volume'] <= 0:
            continue

        # 检查每个策略 (简化版本，使用DB预计算指标)
        if _check_bbikdj(hist, last):
            selected.add(code)
            continue
        if _check_superb1(hist, last):
            selected.add(code)
            continue
        if _check_bbi_short_long(hist, last):
            selected.add(code)
            continue
        if _check_breakout_volume_kdj(hist, last):
            selected.add(code)
            continue
        if _check_peak_kdj(hist, last):
            selected.add(code)
            continue
        if _check_zhixing(hist, last):
            selected.add(code)
            continue
        if _check_ma60_cross(hist, last):
            selected.add(code)
            continue
        if _check_big_bullish(hist, last):
            selected.add(code)
            continue

    return selected


def _check_bbikdj(hist, last):
    """少负战法: J极低 + DIF>0 + BBI上升趋势"""
    if pd.isna(last.get('J')) or pd.isna(last.get('DIF')) or pd.isna(last.get('BBI')):
        return False

    J = last['J']
    DIF = last['DIF']

    # J < -5 或 J在近60天的5%分位以下
    tail = hist.tail(60)
    j_values = tail['J'].dropna()
    if len(j_values) < 10:
        return False

    j_threshold = j_values.quantile(0.05)
    if J > -5 and J > j_threshold:
        return False

    if DIF <= 0:
        return False

    # 价格幅度检查
    close_tail = tail['close'].dropna()
    if len(close_tail) < 2:
        return False
    price_range = close_tail.max() / close_tail.min() if close_tail.min() > 0 else 999
    if price_range > 1.4:
        return False

    # BBI上升趋势: 近20天BBI差分90%为正
    bbi_values = tail['BBI'].dropna()
    if len(bbi_values) < 20:
        return False
    bbi_diffs = bbi_values.diff().dropna()
    if len(bbi_diffs) < 10:
        return False
    positive_ratio = (bbi_diffs >= 0).sum() / len(bbi_diffs)
    if positive_ratio < 0.9:
        # 尝试更短的窗口
        short_bbi = bbi_values.tail(20)
        short_diffs = short_bbi.diff().dropna()
        if len(short_diffs) >= 10:
            short_ratio = (short_diffs >= 0).sum() / len(short_diffs)
            if short_ratio < 0.9:
                return False
        else:
            return False

    return True


def _check_superb1(hist, last):
    """SuperB1战法: 历史BBIKDJ匹配 + 窄幅盘整 + 当日下跌"""
    if len(hist) < 30:
        return False
    if pd.isna(last.get('J')):
        return False

    # 当日下跌>2%
    if len(hist) < 2:
        return False
    prev = hist.iloc[-2]
    if pd.isna(prev['close']) or prev['close'] <= 0:
        return False
    drop = (prev['close'] - last['close']) / prev['close']
    if drop < 0.02:
        return False

    # J < 10
    J = last['J']
    tail15 = hist.tail(15)
    j_vals = tail15['J'].dropna()
    j_q = j_vals.quantile(0.1) if len(j_vals) >= 5 else 10
    if J >= 10 and J > j_q:
        return False

    # 近15天内有BBIKDJ匹配 + 之后窄幅盘整
    lookback = hist.iloc[-16:-1]  # 排除今天
    for i in range(len(lookback)):
        row = lookback.iloc[i]
        if pd.isna(row.get('J')) or pd.isna(row.get('DIF')) or pd.isna(row.get('BBI')):
            continue
        if row['J'] <= -5 and row['DIF'] > 0:
            # 检查从该点到昨天的窄幅盘整
            consolidation = hist.iloc[-(16-i):-1]
            if len(consolidation) < 2:
                continue
            closes = consolidation['close'].dropna()
            if len(closes) < 2:
                continue
            if closes.min() > 0:
                vol_pct = closes.max() / closes.min()
                if vol_pct <= 1.02:
                    return True

    return False


def _check_bbi_short_long(hist, last):
    """补票战法: BBI上升 + 长期RSV>80% + 短期RSV振荡"""
    if pd.isna(last.get('DIF')) or pd.isna(last.get('BBI')):
        return False
    if last['DIF'] <= 0:
        return False

    if len(hist) < 25:
        return False

    # BBI上升趋势 (宽松)
    bbi_tail = hist.tail(30)['BBI'].dropna()
    if len(bbi_tail) < 10:
        return False
    bbi_diffs = bbi_tail.diff().dropna()
    if len(bbi_diffs) < 5:
        return False
    positive_ratio = (bbi_diffs >= 0).sum() / len(bbi_diffs)
    if positive_ratio < 0.8:
        return False

    # 计算RSV (短期n=3, 长期n=21)
    tail_3d = hist.tail(3 + 3)  # 多取几天
    if len(tail_3d) < 6:
        return False

    # 长期RSV (21天)
    for day_offset in range(3):
        idx = -(day_offset + 1)
        if abs(idx) > len(hist):
            return False
        row = hist.iloc[idx]
        lookback_21 = hist.iloc[max(0, len(hist)+idx-21):len(hist)+idx+1]
        if len(lookback_21) < 5:
            return False
        highest = lookback_21['high'].max()
        lowest = lookback_21['low'].min()
        if highest == lowest:
            return False
        rsv_long = (row['close'] - lowest) / (highest - lowest) * 100
        if rsv_long < 80:
            return False

    # 短期RSV (3天) - day1和day3 >= 80, 中间有<20
    short_rsvs = []
    for day_offset in range(3):
        idx = -(day_offset + 1)
        row = hist.iloc[idx]
        lookback_3 = hist.iloc[max(0, len(hist)+idx-3):len(hist)+idx+1]
        if len(lookback_3) < 2:
            return False
        highest = lookback_3['high'].max()
        lowest = lookback_3['low'].min()
        if highest == lowest:
            short_rsvs.append(50)
        else:
            short_rsvs.append((row['close'] - lowest) / (highest - lowest) * 100)

    # short_rsvs[0]=today, [1]=yesterday, [2]=2 days ago
    if short_rsvs[0] < 80 or short_rsvs[2] < 80:
        return False
    if short_rsvs[1] >= 20:  # 中间没有跌到20以下
        return False

    return True


def _check_breakout_volume_kdj(hist, last):
    """TePu战法: 放量突破 + J保持高位"""
    if pd.isna(last.get('J')) or pd.isna(last.get('DIF')):
        return False

    J = last['J']
    DIF = last['DIF']

    if DIF <= 0:
        return False

    # J > 1 或在近60天10%分位以上
    tail60 = hist.tail(60)
    j_vals = tail60['J'].dropna()
    if len(j_vals) < 5:
        return False
    j_q = j_vals.quantile(0.1)
    if J <= 1 and J <= j_q:
        return False

    # 搜索近15天内的突破日T
    if len(hist) < 20:
        return False

    recent = hist.tail(16)  # 包括今天
    for t_idx in range(1, min(16, len(recent))):
        t_row = recent.iloc[-(t_idx+1)] if t_idx < len(recent) else None
        if t_row is None:
            continue

        # T日涨幅>3%
        if t_idx + 2 > len(recent):
            continue
        prev_row = recent.iloc[-(t_idx+2)]
        if pd.isna(prev_row['close']) or prev_row['close'] <= 0:
            continue
        gain = (t_row['close'] - prev_row['close']) / prev_row['close'] * 100
        if gain < 3.0:
            continue

        # T日成交量 > 其他所有日的1.5倍
        t_vol = t_row['volume']
        if pd.isna(t_vol) or t_vol <= 0:
            continue

        other_vols = recent.drop(recent.index[-(t_idx+1)])['volume'].dropna()
        if len(other_vols) < 3:
            continue
        if (other_vols > t_vol * 0.6667).any():  # 其他日成交量不超过T日的2/3
            continue

        # T日创新高
        before_t = hist.iloc[:-(t_idx+1)]
        if len(before_t) > 0:
            prev_max = before_t['close'].max()
            if t_row['close'] <= prev_max:
                continue

        # T到今天J保持高位
        j_slice = recent.iloc[-(t_idx+1):]['J'].dropna()
        if len(j_slice) < 2:
            continue
        if (j_slice < last['J'] - 10).any():
            continue

        return True

    return False


def _check_peak_kdj(hist, last):
    """填坑战法: 双峰形态 + 回踩"""
    if pd.isna(last.get('J')):
        return False

    J = last['J']
    tail = hist.tail(100)
    j_vals = tail['J'].dropna()
    if len(j_vals) < 5:
        return False
    j_q = j_vals.quantile(0.1)
    if J >= 10 and J > j_q:
        return False

    # 寻找峰值 (简化版: 使用高点)
    if len(tail) < 20:
        return False

    peak_vals = np.maximum(tail['high'].values, tail['close'].values)

    # 简化峰值检测
    peaks = []
    for i in range(3, len(peak_vals) - 1):
        if (peak_vals[i] > peak_vals[i-1] and peak_vals[i] > peak_vals[i+1] and
            peak_vals[i] > peak_vals[max(0,i-3):i].mean()):
            # 检查与上一个峰距离
            if not peaks or i - peaks[-1] >= 6:
                peaks.append(i)

    if len(peaks) < 2:
        return False

    # 最新两个峰
    peak_t = peaks[-1]
    peak_tn = peaks[-2]

    # 最新峰 > 前一个峰
    if peak_vals[peak_t] <= peak_vals[peak_tn]:
        return False

    # 前一个峰到最新峰之间有显著下跌
    valley = tail['close'].iloc[peak_tn:peak_t].min()
    if valley <= 0:
        return False
    gap = (peak_vals[peak_tn] - valley) / valley
    if gap < 0.2:
        return False

    # 今天的收盘价接近前一个峰
    today_close = last['close']
    peak_tn_close = tail['close'].iloc[peak_tn]
    if peak_tn_close <= 0:
        return False
    diff_pct = abs(today_close - peak_tn_close) / peak_tn_close
    if diff_pct > 0.03:
        return False

    return True


def _check_zhixing(hist, last):
    """知行战法: J极低 + 小日振幅 + 在多空线上方"""
    if pd.isna(last.get('J')) or pd.isna(last.get('ZXDQ')) or pd.isna(last.get('ZXDKX')):
        return False

    J = last['J']
    if J >= 5:
        return False

    # 日涨跌幅在 -1% ~ 1%
    if len(hist) < 2:
        return False
    prev_close = hist.iloc[-2]['close']
    if pd.isna(prev_close) or prev_close <= 0:
        return False
    change_pct = (last['close'] - prev_close) / prev_close * 100
    if change_pct < -1.0 or change_pct > 1.0:
        return False

    # 振幅 < 4%
    amplitude = (last['high'] - last['low']) / prev_close * 100
    if amplitude >= 4.0:
        return False

    # 短趋线 > 多空线
    if last['ZXDQ'] <= last['ZXDKX']:
        return False

    # 收盘 > 多空线
    if last['close'] <= last['ZXDKX']:
        return False

    return True


def _check_ma60_cross(hist, last):
    """上穿60放量战法: MA60金叉 + 放量 + MA60上升"""
    if pd.isna(last.get('J')) or pd.isna(last.get('ZXDQ')) or pd.isna(last.get('ZXDKX')):
        return False
    if pd.isna(last.get('MA60')):
        return False

    J = last['J']
    tail60 = hist.tail(60)
    j_vals = tail60['J'].dropna()
    j_q = j_vals.quantile(0.05) if len(j_vals) >= 5 else 5
    if J >= 5 and J > j_q:
        return False

    # 搜索近20天的MA60金叉
    if len(hist) < 25:
        return False

    found_cross = False
    cross_idx = None
    for t in range(1, min(21, len(hist) - 1)):
        idx = -(t + 1)
        prev_idx = idx - 1
        if abs(prev_idx) >= len(hist):
            break

        curr = hist.iloc[idx]
        prev = hist.iloc[prev_idx]

        if (pd.notna(curr.get('MA60')) and pd.notna(prev.get('MA60')) and
            pd.notna(prev['close']) and pd.notna(curr['close'])):
            if prev['close'] <= prev['MA60'] and curr['close'] > curr['MA60']:
                found_cross = True
                cross_idx = len(hist) + idx
                break

    if not found_cross:
        return False

    # 放量检查
    wave = hist.iloc[cross_idx:]
    pre_wave = hist.iloc[max(0, cross_idx - len(wave)):cross_idx]

    if len(wave) < 2 or len(pre_wave) < 2:
        return False

    wave_avg_vol = wave['volume'].mean()
    pre_wave_avg_vol = pre_wave['volume'].mean()
    if pre_wave_avg_vol <= 0:
        return False
    if wave_avg_vol < 2.2 * pre_wave_avg_vol:
        return False

    # MA60上升
    ma60_recent = hist.tail(5)['MA60'].dropna()
    if len(ma60_recent) < 3:
        return False
    slope = np.polyfit(range(len(ma60_recent)), ma60_recent.values, 1)[0]
    if slope <= 0:
        return False

    # 知行约束
    if last['close'] <= last['ZXDKX'] or last['ZXDQ'] <= last['ZXDKX']:
        return False

    return True


def _check_big_bullish(hist, last):
    """暴力K战法: 大阳线 + 放量 + 贴近短趋线"""
    if pd.isna(last.get('ZXDQ')):
        return False

    # 阳线
    if last['close'] < last['open']:
        return False

    # 涨幅>4%
    if len(hist) < 2:
        return False
    prev_close = hist.iloc[-2]['close']
    if pd.isna(prev_close) or prev_close <= 0:
        return False
    gain = (last['close'] - prev_close) / prev_close
    if gain < 0.04:
        return False

    # 上影线控制 (上影线 < 50% × max(open, close))
    upper_wick = last['high'] - max(last['open'], last['close'])
    body_top = max(last['open'], last['close'])
    if body_top > 0 and upper_wick / body_top > 0.5:
        return False

    # 放量 (今日>1.5x近20天均量)
    vol_tail = hist.tail(21).iloc[:-1]['volume'].dropna()
    vol_tail = vol_tail[vol_tail > 0]
    if len(vol_tail) < 5:
        return False
    avg_vol = vol_tail.mean()
    if avg_vol <= 0:
        return False
    if last['volume'] < 1.5 * avg_vol:
        return False

    # 收盘 < 短趋线
    if last['close'] >= last['ZXDQ']:
        return False

    return True


def filter_report(report_path: Path, selected_codes: set, output_path: Path):
    """过滤已有的全市场报告，只保留策略选中的股票"""
    with open(report_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_stocks = data.get('all_stocks_with_scores', [])
    filtered = [s for s in all_stocks if s['stock_code'] in selected_codes]

    # 更新策略信息
    for s in filtered:
        s['strategies'] = ['Strategy_Filtered']
        s['selected_by_strategies'] = 1

    data['all_stocks_with_scores'] = filtered
    data['total_scored_stocks'] = len(filtered)
    data['generation_mode'] = 'strategy_filtered'

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    return len(filtered)


def run_backtest(report_dir: str, label: str, rank_field: str = 'pred_10d',
                 top_n: int = 10, focus_days: int = 10) -> dict:
    """运行北极星回测"""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'backtest'))
    import backtest.backtest_report_based as brb
    import backtest.north_star_metrics as nsm

    nsm.DB_PATH = DB_PATH
    brb.DB_PATH = DB_PATH

    # 加载报告
    daily_picks = brb.load_reports(report_dir, rank_field=rank_field)
    if not daily_picks:
        print(f"  ⚠️ {label}: 无报告数据")
        return {}

    print(f"  加载 {len(daily_picks)} 天报告")

    # 运行回测
    result = brb.run_single_backtest(
        daily_picks, label,
        top_n=top_n,
        focus_days=focus_days,
        benchmark_code='000905.SH',
        vol_target=0.0,
        cppi_floor=0.0,
        cppi_multiplier=3.0,
    )

    if not result:
        print(f"  ⚠️ {label}: 回测失败")
        return {}

    # 从 summary[focus_days] 提取关键指标
    s = result.get('summary', {}).get(focus_days, {})

    # 计算V2评分
    total_score = 0
    max_score = 0
    layer_scores = {1: 0, 2: 0, 3: 0, 4: 0}
    layer_max = {1: 0, 2: 0, 3: 0, 4: 0}

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

    for metric_key, target_info in nsm.NORTH_STAR_TARGETS_V2.items():
        current = metric_value_map.get(metric_key)
        if current is None:
            continue
        score, _ = nsm.score_metric_v2(current, target_info)
        layer_id = target_info['layer']
        total_score += score
        max_score += 5
        layer_scores[layer_id] = layer_scores.get(layer_id, 0) + score
        layer_max[layer_id] = layer_max.get(layer_id, 0) + 5

    grade = nsm.compute_v2_grade(total_score, max_score) if max_score > 0 else 'N/A'

    return {
        'label': label,
        'total_score': total_score,
        'max_score': max_score,
        'grade': grade,
        'annual_return': s.get('annual_return', 0),
        'net_annual_return': s.get('net_annual_return', 0),
        'max_drawdown': s.get('max_drawdown', 0),
        'sharpe': s.get('sharpe_ratio', 0),
        'ic_mean': s.get('ic_mean', 0),
        'icir': s.get('icir', 0),
        'ic_positive_ratio': s.get('ic_positive_pct', 0),
        'avg_turnover': s.get('annual_turnover', 0),
        'monthly_win_rate': s.get('monthly_win_rate', 0),
        'n_trades': s.get('n_trades', 0),
        'n_days': len(daily_picks),
        'l1': layer_scores.get(1, 0),
        'l2': layer_scores.get(2, 0),
        'l3': layer_scores.get(3, 0),
        'l4': layer_scores.get(4, 0),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='对比：策略筛选 vs 全市场评分')
    parser.add_argument('--versions', nargs='+', default=['v4.7.3', 'v4.7.5'],
                        help='要对比的版本')
    parser.add_argument('--rank-field', default='pred_10d',
                        help='排序字段')
    parser.add_argument('--top-n', type=int, default=10)
    parser.add_argument('--focus-days', type=int, default=10)
    parser.add_argument('--skip-strategy-gen', action='store_true',
                        help='跳过策略运行，仅回测已有的过滤报告')
    args = parser.parse_args()

    print("=" * 70)
    print("策略筛选 vs 全市场评分 对比回测")
    print("=" * 70)

    results_all = []

    for version in args.versions:
        print(f"\n{'='*70}")
        print(f"模型: {version}")
        print(f"{'='*70}")

        # 确定报告目录
        all_stocks_dir = Path(f'reports/daily_selection_{version}_merged_extended')
        if not all_stocks_dir.exists():
            all_stocks_dir = Path(f'reports/daily_selection_{version}')

        filtered_dir = Path(f'reports/daily_selection_{version}_strategy_filtered')

        # 获取可用的报告日期
        report_files = sorted(all_stocks_dir.glob('analysis_data_*.json'))
        if not report_files:
            print(f"  ⚠️ 未找到 {all_stocks_dir} 的报告")
            continue

        dates = []
        for f in report_files:
            date_str = f.stem.replace('analysis_data_', '')
            if len(date_str) == 8:
                dates.append(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")

        print(f"  报告目录: {all_stocks_dir}")
        print(f"  报告日期范围: {dates[0]} ~ {dates[-1]} ({len(dates)} 天)")

        if not args.skip_strategy_gen:
            # ========== 运行策略筛选 ==========
            filtered_dir.mkdir(parents=True, exist_ok=True)

            # 检查已有的过滤报告
            existing_filtered = set()
            for f in filtered_dir.glob('analysis_data_*.json'):
                date_str = f.stem.replace('analysis_data_', '')
                if len(date_str) == 8:
                    existing_filtered.add(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")

            dates_to_process = [d for d in dates if d not in existing_filtered]

            if dates_to_process:
                print(f"\n  [1] 加载历史数据用于策略筛选...")
                # 使用完整日期范围的起点确保足够的历史lookback
                stock_data = load_all_stock_data_from_db(dates[0], dates[-1])

                print(f"\n  [2] 对 {len(dates_to_process)} 天运行8大策略筛选...")
                t_start = time.time()

                total_selected = 0
                for i, date in enumerate(dates_to_process):
                    target_date = pd.Timestamp(date)

                    # 运行策略
                    selected = run_strategies_for_date(stock_data, target_date)

                    # 过滤报告
                    date_str = date.replace('-', '')
                    src_file = all_stocks_dir / f'analysis_data_{date_str}.json'
                    dst_file = filtered_dir / f'analysis_data_{date_str}.json'

                    if src_file.exists():
                        n_filtered = filter_report(src_file, selected, dst_file)
                        total_selected += n_filtered

                    elapsed = time.time() - t_start
                    rate = (i + 1) / elapsed if elapsed > 0 else 1
                    eta = (len(dates_to_process) - i - 1) / rate if rate > 0 else 0

                    if (i + 1) % 20 == 0 or i == len(dates_to_process) - 1:
                        avg_sel = total_selected / (i + 1) if i > 0 else n_filtered
                        print(f"    [{i+1}/{len(dates_to_process)}] {date}: "
                              f"策略选中 {len(selected)} → 评分后 {n_filtered}, "
                              f"ETA: {eta:.0f}秒")

                elapsed = time.time() - t_start
                avg = total_selected / len(dates_to_process) if dates_to_process else 0
                print(f"\n  策略筛选完成: {elapsed:.1f}秒, "
                      f"平均每日选中 {avg:.0f} 只股票")
            else:
                print(f"  已有 {len(existing_filtered)} 天过滤报告，跳过策略运行")

        # ========== 回测对比 ==========
        print(f"\n  [3] 回测对比...")

        # 全市场回测
        print(f"\n  --- 全市场评分 (无策略筛选) ---")
        result_all = run_backtest(
            str(all_stocks_dir), f"{version}_AllStocks",
            rank_field=args.rank_field, top_n=args.top_n, focus_days=args.focus_days
        )
        if result_all:
            results_all.append(result_all)

        # 策略筛选回测
        if filtered_dir.exists() and list(filtered_dir.glob('analysis_data_*.json')):
            print(f"\n  --- 策略筛选后评分 ---")
            result_filtered = run_backtest(
                str(filtered_dir), f"{version}_StrategyFiltered",
                rank_field=args.rank_field, top_n=args.top_n, focus_days=args.focus_days
            )
            if result_filtered:
                results_all.append(result_filtered)

    # ========== 汇总对比 ==========
    if results_all:
        print(f"\n\n{'='*90}")
        print(f"{'对比汇总':^90}")
        print(f"{'='*90}")

        header = (f"{'模式':<30} {'北极星':>6} {'等级':>4} "
                  f"{'年化(毛)':>8} {'年化(净)':>8} {'MaxDD':>7} {'Sharpe':>7} "
                  f"{'ICIR':>6} {'IC>0':>5} {'换手':>5} "
                  f"{'L1':>4} {'L2':>4} {'L3':>4} {'L4':>4}")
        print(header)
        print("-" * 100)

        for r in results_all:
            # ic_positive_ratio is already 0-100 from summary, avg_turnover is multiplier
            ic_pos = r['ic_positive_ratio']
            if ic_pos <= 1:  # fraction form
                ic_pos *= 100
            line = (f"{r['label']:<30} "
                    f"{r['total_score']:>3}/{r['max_score']:<3} "
                    f"{r['grade']:>4} "
                    f"{r['annual_return']*100:>7.1f}% "
                    f"{r['net_annual_return']*100:>7.1f}% "
                    f"{r['max_drawdown']*100:>6.1f}% "
                    f"{r['sharpe']:>7.3f} "
                    f"{r['icir']:>6.3f} "
                    f"{ic_pos:>4.0f}% "
                    f"{r['avg_turnover']:>4.0f}x "
                    f"{r['l1']:>4} {r['l2']:>4} {r['l3']:>4} {r['l4']:>4}")
            print(line)

        # 保存结果
        report_path = Path('reports/backtest/strategy_filter_comparison.md')
        report_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# 策略筛选 vs 全市场评分 对比回测",
            f"",
            f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"排序字段: {args.rank_field}, Top-N: {args.top_n}, 持仓天数: {args.focus_days}",
            f"",
            f"## 结果汇总",
            f"",
            f"| 模式 | 北极星 | 等级 | 年化(净) | MaxDD | Sharpe | ICIR | IC>0 | 换手率 | L1 | L2 | L3 | L4 |",
            f"|:-----|:------:|:----:|:--------:|:-----:|:------:|:----:|:----:|:------:|:--:|:--:|:--:|:--:|",
        ]

        for r in results_all:
            lines.append(
                f"| {r['label']} "
                f"| {r['total_score']}/{r['max_score']} "
                f"| {r['grade']} "
                f"| {r['net_annual_return']*100:.1f}% "
                f"| {r['max_drawdown']*100:.1f}% "
                f"| {r['sharpe']:.3f} "
                f"| {r['icir']:.3f} "
                f"| {r['ic_positive_ratio']*100:.0f}% "
                f"| {r['avg_turnover']*100:.1f}% "
                f"| {r['l1']} | {r['l2']} | {r['l3']} | {r['l4']} |"
            )

        lines.extend([
            f"",
            f"## 分析",
            f"",
            f"- **全市场评分**: ML模型对全A股~5400只股票评分，选top-{args.top_n}",
            f"- **策略筛选**: 先用8大量化策略(少负/SuperB1/补票/TePu/填坑/知行/MA60/暴力K)筛选候选，再用ML评分选top-{args.top_n}",
            f"",
        ])

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"\n报告已保存: {report_path}")


if __name__ == '__main__':
    main()
