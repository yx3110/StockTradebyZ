#!/usr/bin/env python3
"""
V4901 推荐阈值校准脚本

基于 analysis_data_*.json 报告中的 pred_10d/pred_15d 预测值,
结合 daily_quotes 中的实际未来收益, 校准推荐阈值。

目标:
- strong_buy: 每日 3-10 只 (中位数 ~5), 正超额收益
- buy: 每日 10-30 只
- cautious: 每日 30-100 只
- hold: 其余有正预测的股票
"""

import json
import glob
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# ── 配置 ──────────────────────────────────────────
REPORT_DIR = Path('/Users/yangxu/StockTradebyZ/reports/daily_selection_v4901')
DB_PATH = '/Users/yangxu/StockTradebyZ/data_adapter/stock_data.db'
OUTPUT_PATH = Path('/Users/yangxu/StockTradebyZ/ml_models/trained_models/v4901/recommendation_thresholds.json')
BENCHMARK_CODE = '000905.SH'  # 中证500
FORWARD_DAYS = 10  # 10日前瞻收益

# composite = 0.6 * pred_10d + 0.4 * pred_15d
W_10D, W_15D = 0.6, 0.4


def load_all_reports():
    """加载所有 analysis_data JSON 报告, 提取每日股票预测"""
    files = sorted(glob.glob(str(REPORT_DIR / 'analysis_data_*.json')))
    print(f"找到 {len(files)} 个报告文件")

    daily_data = {}  # date -> list of dicts
    for f in files:
        basename = os.path.basename(f)
        date_str = basename.replace('analysis_data_', '').replace('.json', '')

        with open(f) as fh:
            data = json.load(fh)

        stocks = data.get('all_stocks_with_scores', [])
        records = []
        for s in stocks:
            p10 = s.get('pred_10d', 0) or 0
            p15 = s.get('pred_15d', 0) or 0
            if p10 == 0 and p15 == 0:
                continue
            code = s.get('stock_code', '')
            if not code:
                continue
            # Add exchange suffix if missing
            if '.' not in code:
                market = s.get('market', '')
                if market:
                    code = f"{code}.{market}"
                else:
                    # Infer from code prefix
                    if code.startswith(('6', '5')):
                        code = f"{code}.SH"
                    elif code.startswith(('0', '3')):
                        code = f"{code}.SZ"
                    elif code.startswith(('4', '8', '9')):
                        code = f"{code}.BJ"
                    else:
                        continue
            composite = W_10D * p10 + W_15D * p15
            records.append({
                'stock_code': code,
                'composite': composite,
                'pred_10d': p10,
                'pred_15d': p15,
            })

        if records:
            # Convert date to DB format: 20240102 -> 2024-01-02
            if len(date_str) == 8 and '-' not in date_str:
                db_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            else:
                db_date = date_str
            daily_data[db_date] = records

    print(f"有效报告日数: {len(daily_data)}")
    return daily_data


def load_close_prices(conn, codes, dates):
    """批量加载收盘价"""
    cur = conn.cursor()

    # code -> security_id
    cur.execute("SELECT id, code FROM securities")
    sec_map = {row[1]: row[0] for row in cur.fetchall()}

    code_to_id = {}
    needed_ids = set()
    for code in codes:
        if code in sec_map:
            code_to_id[code] = sec_map[code]
            needed_ids.add(sec_map[code])
        else:
            # Try bare code (without exchange suffix)
            bare = code.split('.')[0]
            if bare in sec_map:
                code_to_id[code] = sec_map[bare]
                needed_ids.add(sec_map[bare])

    if not needed_ids:
        return {}

    all_dates = sorted(dates)
    min_date = all_dates[0]

    # 分批查询
    price_data = defaultdict(dict)
    id_list_all = list(needed_ids)
    batch_size = 500

    for i in range(0, len(id_list_all), batch_size):
        batch_ids = id_list_all[i:i + batch_size]
        id_str = ','.join(str(x) for x in batch_ids)
        cur.execute(f"""
            SELECT security_id, trade_date, close
            FROM daily_quotes
            WHERE security_id IN ({id_str})
              AND trade_date >= ?
            ORDER BY security_id, trade_date
        """, (min_date,))
        for sid, td, close in cur.fetchall():
            price_data[sid][td] = close

    id_to_code = {v: k for k, v in code_to_id.items()}
    result = {}
    for sid, date_prices in price_data.items():
        code = id_to_code.get(sid)
        if code:
            result[code] = date_prices

    return result


def compute_forward_returns(daily_data, conn):
    """计算每只股票的实际未来N日收益率"""
    print(f"\n计算 {FORWARD_DAYS} 日前瞻收益...")

    all_codes = set()
    all_dates = set()
    for date_str, records in daily_data.items():
        all_dates.add(date_str)
        for r in records:
            all_codes.add(r['stock_code'])
    all_codes.add(BENCHMARK_CODE)

    print(f"  需查询 {len(all_codes)} 只股票, {len(all_dates)} 个日期")

    price_data = load_close_prices(conn, all_codes, all_dates)
    print(f"  成功加载 {len(price_data)} 只股票的价格数据")

    # 交易日历
    benchmark_prices = price_data.get(BENCHMARK_CODE, {})
    if not benchmark_prices:
        for code in price_data:
            if '000905' in code:
                benchmark_prices = price_data[code]
                break
    trading_days = sorted(benchmark_prices.keys())
    td_index = {d: i for i, d in enumerate(trading_days)}
    print(f"  交易日历: {trading_days[0]} ~ {trading_days[-1]} ({len(trading_days)} 日)")

    results = []
    skipped_no_price = 0
    skipped_no_future = 0

    for date_str, records in daily_data.items():
        if date_str not in td_index:
            continue
        idx = td_index[date_str]
        if idx + FORWARD_DAYS >= len(trading_days):
            continue

        future_date = trading_days[idx + FORWARD_DAYS]

        bm_t0 = benchmark_prices.get(date_str)
        bm_tf = benchmark_prices.get(future_date)
        if bm_t0 is None or bm_tf is None:
            continue
        bm_return = (bm_tf - bm_t0) / bm_t0

        for r in records:
            code = r['stock_code']
            stock_prices = price_data.get(code, {})
            if not stock_prices:
                skipped_no_price += 1
                continue

            close_t0 = stock_prices.get(date_str)
            close_tf = stock_prices.get(future_date)
            if close_t0 is None:
                skipped_no_price += 1
                continue
            if close_tf is None:
                skipped_no_future += 1
                continue

            actual_return = (close_tf - close_t0) / close_t0
            excess_return = actual_return - bm_return

            results.append({
                'date': date_str,
                'stock_code': code,
                'composite': r['composite'],
                'pred_10d': r['pred_10d'],
                'pred_15d': r['pred_15d'],
                'actual_10d_return': actual_return,
                'benchmark_return': bm_return,
                'excess_return': excess_return,
            })

    print(f"  有效样本: {len(results)} (跳过无价格: {skipped_no_price}, 无未来: {skipped_no_future})")
    return pd.DataFrame(results)


def calibrate_thresholds(df, new_model_composites):
    """
    百分位校准: 用历史数据找最优百分位, 再映射到新模型分布.

    核心思想: 不同模型的 composite 绝对值不同, 但「前 X% 的股票」
    这个相对概念是稳定的. 所以我们:
    1. 在历史数据上扫描百分位, 找到每个档位的最优百分位
    2. 用新模型的分布, 把百分位映射回绝对阈值
    """
    print("\n" + "=" * 75)
    print("  V4901 推荐阈值校准 (百分位法)")
    print("=" * 75)

    # ── Step 1: 每日分布统计 ──
    print("\n[1] 每日 composite 分布统计")
    daily_stats = df.groupby('date')['composite'].agg(['count', 'mean', 'std'])
    print(f"  日均股票数: {daily_stats['count'].mean():.0f}")
    print(f"  composite均值: {daily_stats['mean'].mean():.6f}")
    print(f"  composite标准差: {daily_stats['std'].mean():.6f}")

    # ── Step 2: 百分位扫描 ──
    print(f"\n[2] 百分位扫描 (用每日top-N%选股, 计算实际超额收益)")
    header = f"  {'百分位':>8} {'日均数':>8} {'实际10d%':>10} {'基准10d%':>10} {'超额%':>10} {'胜率%':>8}"
    print(header)
    print("  " + "-" * 63)

    sweep_pcts = np.arange(90.0, 99.95, 0.1)
    sweep_results = []

    grouped = dict(list(df.groupby('date')))

    for pct in sweep_pcts:
        daily_counts = []
        daily_returns = []
        daily_benchmark = []
        daily_excess = []
        daily_win = []

        for date_str, group in grouped.items():
            composites = group['composite'].values
            if len(composites) < 100:
                continue
            cutoff = np.percentile(composites, pct)
            above = group[group['composite'] >= cutoff]
            daily_counts.append(len(above))
            if len(above) > 0:
                avg_ret = above['actual_10d_return'].mean()
                avg_bm = above['benchmark_return'].mean()
                avg_excess = above['excess_return'].mean()
                daily_returns.append(avg_ret)
                daily_benchmark.append(avg_bm)
                daily_excess.append(avg_excess)
                daily_win.append(1 if avg_excess > 0 else 0)

        if len(daily_returns) < 10:
            continue

        mean_count = np.mean(daily_counts)
        mean_return = np.mean(daily_returns) * 100
        mean_bm = np.mean(daily_benchmark) * 100
        mean_excess = np.mean(daily_excess) * 100
        win_rate = np.mean(daily_win) * 100

        sweep_results.append({
            'percentile': pct,
            'mean_count': mean_count,
            'mean_return': mean_return,
            'mean_benchmark': mean_bm,
            'mean_excess': mean_excess,
            'win_rate': win_rate,
        })

        # Print at key points
        if pct % 1.0 < 0.05 or pct >= 99.0:
            print(f"  P{pct:5.1f} {mean_count:8.1f} {mean_return:10.2f} {mean_bm:10.2f} {mean_excess:10.2f} {win_rate:8.1f}")

    sweep_df = pd.DataFrame(sweep_results)

    # ── Step 3: 选择最优百分位 ──
    print(f"\n[3] 百分位选择")

    # strong_buy: target 3-10 stocks/day, prefer ~5, maximize excess * win_rate
    sb_cands = sweep_df[
        (sweep_df['mean_count'] >= 2) &
        (sweep_df['mean_count'] <= 15)
    ].copy()

    if len(sb_cands) > 0:
        sb_cands['score'] = (
            -abs(sb_cands['mean_count'] - 5) * 0.3
            + sb_cands['mean_excess'] * 5
            + sb_cands['win_rate'] * 0.05
        )
        best_sb = sb_cands.sort_values('score', ascending=False).iloc[0]
        sb_pct = best_sb['percentile']
        print(f"  strong_buy: P{sb_pct:.1f}"
              f"  (日均 {best_sb['mean_count']:.1f}, 超额 {best_sb['mean_excess']:.2f}%, 胜率 {best_sb['win_rate']:.1f}%)")
    else:
        sb_pct = 99.9
        print(f"  strong_buy: P{sb_pct:.1f} (fallback)")

    # buy: target 10-30
    buy_cands = sweep_df[
        (sweep_df['mean_count'] >= 8) &
        (sweep_df['mean_count'] <= 40) &
        (sweep_df['percentile'] < sb_pct)
    ].copy()

    if len(buy_cands) > 0:
        buy_cands['score'] = (
            -abs(buy_cands['mean_count'] - 20) * 0.2
            + buy_cands['mean_excess'] * 5
        )
        best_buy = buy_cands.sort_values('score', ascending=False).iloc[0]
        buy_pct = best_buy['percentile']
        print(f"  buy:        P{buy_pct:.1f}"
              f"  (日均 {best_buy['mean_count']:.1f}, 超额 {best_buy['mean_excess']:.2f}%)")
    else:
        buy_pct = sb_pct - 2.0
        print(f"  buy:        P{buy_pct:.1f} (fallback)")

    # cautious: target 30-100
    caut_cands = sweep_df[
        (sweep_df['mean_count'] >= 25) &
        (sweep_df['mean_count'] <= 150) &
        (sweep_df['percentile'] < buy_pct)
    ].copy()

    if len(caut_cands) > 0:
        caut_cands['score'] = (
            -abs(caut_cands['mean_count'] - 60) * 0.1
            + caut_cands['mean_excess'] * 3
        )
        best_caut = caut_cands.sort_values('score', ascending=False).iloc[0]
        caut_pct = best_caut['percentile']
        print(f"  cautious:   P{caut_pct:.1f}"
              f"  (日均 {best_caut['mean_count']:.1f}, 超额 {best_caut['mean_excess']:.2f}%)")
    else:
        caut_pct = buy_pct - 3.0
        print(f"  cautious:   P{caut_pct:.1f} (fallback)")

    # hold: target 100-400
    hold_cands = sweep_df[
        (sweep_df['mean_count'] >= 80) &
        (sweep_df['mean_count'] <= 600) &
        (sweep_df['percentile'] < caut_pct)
    ].copy()

    if len(hold_cands) > 0:
        hold_cands['score'] = (
            -abs(hold_cands['mean_count'] - 200) * 0.05
            + hold_cands['mean_excess'] * 2
        )
        best_hold = hold_cands.sort_values('score', ascending=False).iloc[0]
        hold_pct = best_hold['percentile']
        print(f"  hold:       P{hold_pct:.1f}"
              f"  (日均 {best_hold['mean_count']:.1f})")
    else:
        hold_pct = caut_pct - 5.0
        print(f"  hold:       P{hold_pct:.1f} (fallback)")

    percentiles = {
        'strong_buy': sb_pct,
        'buy': buy_pct,
        'cautious': caut_pct,
        'hold': hold_pct,
    }

    # ── Step 4: 映射到新模型的绝对阈值 ──
    print(f"\n[4] 映射到新模型绝对阈值")
    print(f"  新模型有 {len(new_model_composites)} 只股票的 composite 值")

    if len(new_model_composites) > 0:
        nm_arr = np.array(new_model_composites)
        thresholds = {}
        for name, pct in percentiles.items():
            thresh = np.percentile(nm_arr, pct)
            thresholds[name] = round(float(thresh), 6)
            count = int((nm_arr >= thresh).sum())
            print(f"  {name:12s}: P{pct:.1f} -> {thresh:.6f} ({count} 只)")
    else:
        # Fallback: use historical median of daily percentile thresholds
        print("  WARNING: 无新模型数据, 使用历史中位数")
        thresholds = {}
        for name, pct in percentiles.items():
            daily_thresholds = []
            for date_str, group in grouped.items():
                composites = group['composite'].values
                if len(composites) >= 100:
                    daily_thresholds.append(np.percentile(composites, pct))
            if daily_thresholds:
                thresholds[name] = round(float(np.median(daily_thresholds)), 6)
            else:
                thresholds[name] = 0.0

    # ── Step 5: 验证历史回测表现 ──
    print(f"\n[5] 验证: 用新阈值回测历史数据")
    header = f"  {'级别':>12} {'日均数':>8} {'实际10d%':>10} {'基准10d%':>10} {'超额%':>10} {'胜率%':>8}"
    print(header)
    print("  " + "-" * 63)

    for name in ['strong_buy', 'buy', 'cautious', 'hold']:
        pct = percentiles[name]
        daily_counts = []
        daily_excess = []
        daily_win = []
        daily_ret = []
        daily_bm = []

        for date_str, group in grouped.items():
            composites = group['composite'].values
            if len(composites) < 100:
                continue
            cutoff = np.percentile(composites, pct)
            above = group[group['composite'] >= cutoff]
            daily_counts.append(len(above))
            if len(above) > 0:
                daily_ret.append(above['actual_10d_return'].mean())
                daily_bm.append(above['benchmark_return'].mean())
                daily_excess.append(above['excess_return'].mean())
                daily_win.append(1 if above['excess_return'].mean() > 0 else 0)

        if daily_ret:
            print(f"  {name:>12} {np.mean(daily_counts):8.1f}"
                  f" {np.mean(daily_ret)*100:10.2f}"
                  f" {np.mean(daily_bm)*100:10.2f}"
                  f" {np.mean(daily_excess)*100:10.2f}"
                  f" {np.mean(daily_win)*100:8.1f}")

    return thresholds, percentiles, sweep_df


def main():
    print("=" * 75)
    print("V4901 推荐阈值校准 (百分位法)")
    print(f"报告目录: {REPORT_DIR}")
    print(f"数据库:   {DB_PATH}")
    print(f"基准:     {BENCHMARK_CODE}")
    print(f"前瞻:     {FORWARD_DAYS} 日")
    print("=" * 75)

    # 1. 加载报告
    daily_data = load_all_reports()

    # 2. 提取新模型(最新日期)的composite分布
    # 用最新报告文件直接读取完整composite列表(包括无future return的)
    latest_date = max(daily_data.keys())
    print(f"\n最新报告日期: {latest_date}")
    new_model_composites = [r['composite'] for r in daily_data[latest_date]]
    print(f"新模型 composite: {len(new_model_composites)} 只,"
          f" range [{min(new_model_composites):.6f}, {max(new_model_composites):.6f}]")

    # 3. 计算前瞻收益
    conn = sqlite3.connect(DB_PATH, timeout=30)
    df = compute_forward_returns(daily_data, conn)
    conn.close()

    if len(df) == 0:
        print("ERROR: 没有有效数据, 无法校准")
        sys.exit(1)

    # 4. 校准阈值
    thresholds, percentiles, sweep_df = calibrate_thresholds(df, new_model_composites)

    # 5. 新模型预期数量
    nm_arr = np.array(new_model_composites)
    new_model_counts = {}
    for name, thresh in thresholds.items():
        new_model_counts[name] = int((nm_arr >= thresh).sum())

    # 6. 保存
    print(f"\n[6] 保存结果")
    output = {
        'strong_buy': thresholds['strong_buy'],
        'buy': thresholds['buy'],
        'cautious': thresholds['cautious'],
        'hold': thresholds['hold'],
        '_calibration': {
            'method': 'percentile_based_calibration',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'report_days': len(set(df['date'])),
            'total_samples': len(df),
            'forward_days': FORWARD_DAYS,
            'benchmark': BENCHMARK_CODE,
            'composite_formula': f'{W_10D}*pred_10d + {W_15D}*pred_15d',
            'optimal_percentiles': {k: round(v, 1) for k, v in percentiles.items()},
            'new_model_expected_counts': new_model_counts,
            'new_model_date': latest_date,
            'new_model_stocks': len(new_model_composites),
            'note': '百分位法: 从历史542天报告找最优百分位, 再映射到新模型分布',
        }
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  已保存: {OUTPUT_PATH}")

    # 7. 总结
    print("\n" + "=" * 75)
    print("  校准结果总结")
    print("=" * 75)
    print(f"  {'级别':>12} {'百分位':>8} {'阈值':>12} {'新模型数':>10}")
    print(f"  {'---':>12} {'---':>8} {'---':>12} {'---':>10}")
    for name in ['strong_buy', 'buy', 'cautious', 'hold']:
        c = new_model_counts.get(name, '?')
        p = percentiles.get(name, 0)
        print(f"  {name:>12} P{p:5.1f} {thresholds[name]:12.6f} {str(c):>10}")

    print()
    print("  对比:")
    print(f"    旧 strong_buy = 0.00980  (日均 ~146 只 - 太多)")
    print(f"    模型内嵌      = 0.02265  (日均 ~0 只 - 太少)")
    print(f"    新 strong_buy = {thresholds['strong_buy']:.6f}  (P{percentiles['strong_buy']:.1f}, {new_model_counts.get('strong_buy', '?')} 只)")
    print()

    # Also save log
    log_path = OUTPUT_PATH.parent / 'optimizer_params_calibrated.log.json'
    log_data = {
        'thresholds': thresholds,
        'percentiles': percentiles,
        'new_model_counts': new_model_counts,
        'calibration_date': datetime.now().isoformat(),
    }
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    print(f"  校准日志: {log_path}")


if __name__ == '__main__':
    main()
