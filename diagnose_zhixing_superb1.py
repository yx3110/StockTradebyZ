#!/usr/bin/env python3
"""
诊断知行战法和SuperB1战法的过滤条件通过率
"""
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from data_adapter.stock_data_loader import StockDataLoader
from stock_selctor.Selector import ZhiXingSelector, SuperB1Selector, compute_kdj, compute_bbi

def diagnose_zhixing_selector(stock_data, target_date='2025-10-10'):
    """诊断知行战法各个过滤条件的通过率"""
    print("\n" + "="*60)
    print("诊断知行战法 (ZhiXingSelector)")
    print("="*60)

    # 当前参数
    params = {
        "j_threshold": 13.0,
        "min_change_pct": -2.0,
        "max_change_pct": 1.8,
        "max_amplitude_pct": 7.0,
        "close_threshold_pct": 97.0,
        "max_window": 120
    }

    print(f"\n当前参数:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    target_date = pd.Timestamp(target_date)

    # 统计数据
    stats = {
        'total': 0,
        'enough_data': 0,
        'j_pass': 0,
        'change_pass': 0,
        'amplitude_pass': 0,
        'close_threshold_pass': 0,
        'trend_line_pass': 0,  # 新增：短期趋势线 > 多空线
        'all_pass': 0
    }

    # 采样测试（1000只股票）
    stock_codes = list(stock_data.keys())[:1000]

    for code in stock_codes:
        stats['total'] += 1
        df = stock_data[code]

        if df.empty:
            continue

        # 过滤到目标日期
        df = df[df['date'] <= target_date].copy()
        if len(df) < params['max_window']:
            continue

        stats['enough_data'] += 1

        # 计算KDJ指标
        df = compute_kdj(df)

        # 获取最新数据
        latest = df.iloc[-1]

        # 条件1: J值检查
        j_value = latest.get('J', None)
        if j_value is None or pd.isna(j_value):
            continue

        j_pass = j_value < params['j_threshold']
        if j_pass:
            stats['j_pass'] += 1
        else:
            continue

        # 条件2: 涨跌幅检查
        if len(df) < 2:
            continue
        prev = df.iloc[-2]
        # 计算涨幅比率
        price_change_ratio = latest['close'] / prev['close']
        # 转换为百分比
        change_pct = (price_change_ratio - 1) * 100

        change_pass = params['min_change_pct'] <= change_pct <= params['max_change_pct']
        if change_pass:
            stats['change_pass'] += 1
        else:
            continue

        # 条件3: 振幅检查
        high = latest.get('high', 0)
        low = latest.get('low', 0)
        close = latest.get('close', 0)
        if close > 0:
            amplitude = ((high - low) / close) * 100
            amplitude_pass = amplitude <= params['max_amplitude_pct']
            if amplitude_pass:
                stats['amplitude_pass'] += 1
            else:
                continue
        else:
            continue

        # 条件4: 收盘价位置检查
        if high > 0:
            close_position = (close / high) * 100
            close_threshold_pass = close_position >= params['close_threshold_pct']
            if close_threshold_pass:
                stats['close_threshold_pass'] += 1
            else:
                continue
        else:
            continue

        # 条件5: 知行趋势线检查（短期趋势线 > 多空线）
        # 计算短期趋势线：EMA(EMA(CLOSE, 10), 10)
        ema10 = df['close'].ewm(span=10, adjust=False).mean()
        zhixing_short_trend = ema10.ewm(span=10, adjust=False).mean()

        # 计算多空线：(MA14 + MA28 + MA57 + MA114) / 4
        ma14 = df['close'].rolling(window=14, min_periods=1).mean()
        ma28 = df['close'].rolling(window=28, min_periods=1).mean()
        ma57 = df['close'].rolling(window=57, min_periods=1).mean()
        ma114 = df['close'].rolling(window=114, min_periods=1).mean()
        zhixing_multi_kong = (ma14 + ma28 + ma57 + ma114) / 4

        short_trend_today = zhixing_short_trend.iloc[-1]
        multi_kong_today = zhixing_multi_kong.iloc[-1]

        if pd.isna(short_trend_today) or pd.isna(multi_kong_today):
            continue

        trend_line_pass = short_trend_today > multi_kong_today
        if trend_line_pass:
            stats['trend_line_pass'] += 1
            stats['all_pass'] += 1

    # 打印统计结果
    print(f"\n统计结果（样本: {stats['total']}只股票）:")
    print(f"  数据充足: {stats['enough_data']} ({stats['enough_data']/stats['total']*100:.1f}%)")

    if stats['enough_data'] > 0:
        print(f"\n各条件通过率（基于数据充足的股票）:")
        print(f"  1. J < {params['j_threshold']}: {stats['j_pass']}/{stats['enough_data']} ({stats['j_pass']/stats['enough_data']*100:.1f}%)")

        if stats['j_pass'] > 0:
            print(f"  2. 涨跌幅 [{params['min_change_pct']}, {params['max_change_pct']}]: {stats['change_pass']}/{stats['j_pass']} ({stats['change_pass']/stats['j_pass']*100:.1f}%)")

            if stats['change_pass'] > 0:
                print(f"  3. 振幅 <= {params['max_amplitude_pct']}%: {stats['amplitude_pass']}/{stats['change_pass']} ({stats['amplitude_pass']/stats['change_pass']*100:.1f}%)")

                if stats['amplitude_pass'] > 0:
                    print(f"  4. 收盘位置 >= {params['close_threshold_pct']}%: {stats['close_threshold_pass']}/{stats['amplitude_pass']} ({stats['close_threshold_pass']/stats['amplitude_pass']*100:.1f}%)")

                    if stats['close_threshold_pass'] > 0:
                        print(f"  5. 短期趋势线 > 多空线: {stats['trend_line_pass']}/{stats['close_threshold_pass']} ({stats['trend_line_pass']/stats['close_threshold_pass']*100:.1f}%)")

        print(f"\n最终通过率: {stats['all_pass']}/{stats['enough_data']} ({stats['all_pass']/stats['enough_data']*100:.2f}%)")

    return stats


def diagnose_superb1_selector(stock_data, target_date='2025-10-10'):
    """诊断SuperB1战法各个过滤条件的通过率"""
    print("\n" + "="*60)
    print("诊断SuperB1战法 (SuperB1Selector)")
    print("="*60)

    # 当前参数
    params = {
        "lookback_n": 10,
        "close_vol_pct": 0.02,
        "price_drop_pct": 0.02,
        "j_threshold": 10,
        "j_q_threshold": 0.10,
        "B1_params": {
            "j_threshold": 10,
            "bbi_min_window": 20,
            "max_window": 60,
            "price_range_pct": 1,
            "bbi_q_threshold": 0.3,
            "j_q_threshold": 0.10
        }
    }

    print(f"\n当前参数:")
    print(f"  lookback_n: {params['lookback_n']}")
    print(f"  close_vol_pct: {params['close_vol_pct']}")
    print(f"  price_drop_pct: {params['price_drop_pct']}")
    print(f"  j_threshold: {params['j_threshold']}")
    print(f"  j_q_threshold: {params['j_q_threshold']}")
    print(f"  B1_params: {params['B1_params']}")

    target_date = pd.Timestamp(target_date)

    # 统计数据
    stats = {
        'total': 0,
        'enough_data': 0,
        'j_pass': 0,
        'volume_spike_found': 0,
        'price_drop_found': 0,
        'b1_condition_pass': 0,
        'all_pass': 0
    }

    # 采样测试（1000只股票）
    stock_codes = list(stock_data.keys())[:1000]

    for code in stock_codes:
        stats['total'] += 1
        df = stock_data[code]

        if df.empty:
            continue

        # 过滤到目标日期
        df = df[df['date'] <= target_date].copy()
        if len(df) < params['B1_params']['max_window']:
            continue

        stats['enough_data'] += 1

        # 计算KDJ指标
        df = compute_kdj(df)

        # 获取最新数据
        latest = df.iloc[-1]

        # 条件1: J值检查
        j_value = latest.get('J', None)
        if j_value is None or pd.isna(j_value):
            continue

        # 计算J的分位数
        j_window = df['J'].iloc[-params['B1_params']['max_window']:].dropna()
        if len(j_window) == 0:
            continue
        j_quantile = (j_window < j_value).sum() / len(j_window)

        j_pass = (j_value < params['j_threshold']) or (j_quantile <= params['j_q_threshold'])
        if j_pass:
            stats['j_pass'] += 1
        else:
            continue

        # 条件2: 查找最近lookback_n天内的量价异常
        lookback_data = df.iloc[-params['lookback_n']-1:-1] if len(df) > params['lookback_n'] else df.iloc[:-1]
        if len(lookback_data) == 0:
            continue

        # 找放量日
        volume_spike_found = False
        for i in range(len(lookback_data)):
            row = lookback_data.iloc[i]
            if i > 0:
                prev_vol = lookback_data.iloc[i-1]['volume']
                if prev_vol > 0 and row['volume'] / prev_vol > (1 + params['close_vol_pct']):
                    volume_spike_found = True
                    break

        if volume_spike_found:
            stats['volume_spike_found'] += 1
        else:
            continue

        # 条件3: 检查价格下跌（手动计算pct_change）
        price_drop_found = False
        for i in range(1, len(lookback_data)):  # 从第2行开始，需要前一行数据
            curr_close = lookback_data.iloc[i]['close']
            prev_close = lookback_data.iloc[i-1]['close']
            if prev_close > 0:
                pct_change = (curr_close - prev_close) / prev_close
                if pct_change < -params['price_drop_pct']:
                    price_drop_found = True
                    break

        if price_drop_found:
            stats['price_drop_found'] += 1
        else:
            continue

        # 条件4: B1选股器条件（简化版检查BBI）
        # 计算BBI
        df['BBI'] = compute_bbi(df)
        latest_with_bbi = df.iloc[-1]

        if not pd.isna(latest_with_bbi['BBI']):
            close = latest_with_bbi['close']
            bbi = latest_with_bbi['BBI']

            # 简化检查：价格在BBI附近
            if abs(close - bbi) / bbi < params['B1_params']['price_range_pct'] / 100:
                stats['b1_condition_pass'] += 1
                stats['all_pass'] += 1

    # 打印统计结果
    print(f"\n统计结果（样本: {stats['total']}只股票）:")
    print(f"  数据充足: {stats['enough_data']} ({stats['enough_data']/stats['total']*100:.1f}%)")

    if stats['enough_data'] > 0:
        print(f"\n各条件通过率（基于数据充足的股票）:")
        print(f"  1. J < {params['j_threshold']} 或 J分位 <= {params['j_q_threshold']}: {stats['j_pass']}/{stats['enough_data']} ({stats['j_pass']/stats['enough_data']*100:.1f}%)")

        if stats['j_pass'] > 0:
            print(f"  2. 最近{params['lookback_n']}天内放量 >{params['close_vol_pct']*100}%: {stats['volume_spike_found']}/{stats['j_pass']} ({stats['volume_spike_found']/stats['j_pass']*100:.1f}%)")

            if stats['volume_spike_found'] > 0:
                print(f"  3. 最近{params['lookback_n']}天内下跌 >{params['price_drop_pct']*100}%: {stats['price_drop_found']}/{stats['volume_spike_found']} ({stats['price_drop_found']/stats['volume_spike_found']*100:.1f}%)")

                if stats['price_drop_found'] > 0:
                    print(f"  4. B1条件(价格在BBI附近): {stats['b1_condition_pass']}/{stats['price_drop_found']} ({stats['b1_condition_pass']/stats['price_drop_found']*100:.1f}%)")

        print(f"\n最终通过率: {stats['all_pass']}/{stats['enough_data']} ({stats['all_pass']/stats['enough_data']*100:.2f}%)")

    return stats


def main():
    print("开始诊断知行战法和SuperB1战法...")

    # 加载数据
    print("加载股票数据...")
    loader = StockDataLoader()
    stock_data = loader.load_all_stock_data(
        days=250,
        security_types=['A股'],
        target_date='2025-10-10'
    )
    print(f"成功加载 {len(stock_data)} 只股票数据")

    # 诊断知行战法
    zhixing_stats = diagnose_zhixing_selector(stock_data, '2025-10-10')

    # 诊断SuperB1战法
    superb1_stats = diagnose_superb1_selector(stock_data, '2025-10-10')

    # 总结建议
    print("\n" + "="*60)
    print("诊断总结与建议")
    print("="*60)

    print("\n知行战法:")
    if zhixing_stats['enough_data'] > 0:
        pass_rate = zhixing_stats['all_pass'] / zhixing_stats['enough_data'] * 100
        if pass_rate < 0.5:
            print(f"  ⚠️  通过率过低 ({pass_rate:.2f}%)，建议放宽参数:")
            print(f"     - 增大 j_threshold (当前13.0 -> 建议15-20)")
            print(f"     - 增大涨跌幅范围 (当前[-2.0, 1.8] -> 建议[-3.0, 3.0])")
            print(f"     - 增大振幅上限 (当前7.0% -> 建议8-10%)")
            print(f"     - 降低收盘位置要求 (当前97% -> 建议95%)")
        elif pass_rate < 1.0:
            print(f"  ⚠️  通过率较低 ({pass_rate:.2f}%)，建议适度放宽参数")
        else:
            print(f"  ✓ 通过率正常 ({pass_rate:.2f}%)")

    print("\nSuperB1战法:")
    if superb1_stats['enough_data'] > 0:
        pass_rate = superb1_stats['all_pass'] / superb1_stats['enough_data'] * 100
        if pass_rate < 0.5:
            print(f"  ⚠️  通过率过低 ({pass_rate:.2f}%)，建议放宽参数:")
            print(f"     - 增大 lookback_n (当前10 -> 建议15-20)")
            print(f"     - 降低放量阈值 close_vol_pct (当前2% -> 建议1.5%)")
            print(f"     - 降低价格下跌阈值 price_drop_pct (当前2% -> 建议1.5%)")
        elif pass_rate < 1.0:
            print(f"  ⚠️  通过率较低 ({pass_rate:.2f}%)，建议适度放宽参数")
        else:
            print(f"  ✓ 通过率正常 ({pass_rate:.2f}%)")


if __name__ == '__main__':
    main()
