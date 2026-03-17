#!/usr/bin/env python3
"""
ML-Enhanced 止盈止损目标价回测验证

验证报告中的 buy_price / stop_loss / target_price 在实际市场中的表现：
- 买入触发率: 未来N天min(low) <= buy_price (限价单成交)
- 目标命中率: 买入后max(high) >= target
- 止损命中率: 买入后min(low) <= stop_loss
- 胜率: 目标先于止损被触达
- 按投资建议分组统计 (强烈买入/买入/谨慎买入/观望/回避)

Usage:
    python3 backtest/validate_target_prices.py \
        --report-dir reports/daily_selection_v4.7.3 \
        --top-n 20 --hold-days 10
"""
import argparse
import re
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data_adapter" / "stock_data.db"


def parse_report(report_path: Path) -> dict:
    """从报告markdown中解析日期和股票数据"""
    content = report_path.read_text(encoding='utf-8')

    date_match = re.search(r'分析日期.*?(\d{4}-\d{2}-\d{2})', content)
    if not date_match:
        return {'date': None, 'stocks': []}
    analysis_date = date_match.group(1)

    # 解析汇总排名表 (优先，包含所有新字段)
    stocks = _parse_table(content)

    # 补充: 从详细分析区解析 (fallback)
    detailed = _parse_detail_blocks(content)
    table_codes = {s['stock_code'] for s in stocks}
    for ds in detailed:
        if ds['stock_code'] not in table_codes:
            stocks.append(ds)

    return {'date': analysis_date, 'stocks': stocks}


def _parse_table(content: str) -> list:
    """解析汇总排名表 — 自动识别列位置"""
    stocks = []
    lines = content.split('\n')
    in_table = False
    col_map = {}

    for line in lines:
        # 寻找含有"股票代码"的表头行
        if '股票代码' in line and '|' in line and not in_table:
            cols = [c.strip() for c in line.split('|')]
            for idx, h in enumerate(cols):
                if '股票代码' in h: col_map['code'] = idx
                elif '股票名称' in h: col_map['name'] = idx
                elif '投资建议' in h: col_map['rec'] = idx
                elif '收盘价' in h: col_map['close'] = idx
                elif '买入价' in h: col_map['buy'] = idx
                elif '止损价' in h: col_map['stop'] = idx
                elif '目标价' in h: col_map['target'] = idx
                elif '仓位' in h: col_map['pos'] = idx
            # 必须有止损和目标列才算新格式表
            if 'stop' in col_map and 'target' in col_map:
                in_table = True
            continue

        if in_table and line.strip().startswith('|---'):
            continue

        if in_table and '|' in line and line.strip().startswith('|'):
            cols = [c.strip() for c in line.split('|')]
            try:
                stock_code = cols[col_map['code']].strip()
                if not re.match(r'^\d{6}$', stock_code):
                    continue

                def _float(idx_key):
                    if idx_key not in col_map:
                        return 0.0
                    val = cols[col_map[idx_key]].strip()
                    try:
                        return float(val)
                    except ValueError:
                        return 0.0

                rec = cols[col_map.get('rec', 0)].strip() if 'rec' in col_map else ''
                pos_str = cols[col_map.get('pos', 0)].strip() if 'pos' in col_map else '0'
                pos_val = int(pos_str.replace('%', '')) if pos_str.replace('%', '').isdigit() else 0

                stocks.append({
                    'stock_code': stock_code,
                    'stock_name': cols[col_map.get('name', 0)].strip() if 'name' in col_map else '',
                    'recommendation': rec,
                    'close_price': _float('close'),
                    'buy_price': _float('buy'),
                    'stop_loss': _float('stop'),
                    'target': _float('target'),
                    'position_pct': pos_val,
                })
            except (IndexError, ValueError):
                continue
        elif in_table and not line.strip().startswith('|'):
            in_table = False

    return stocks


def _parse_detail_blocks(content: str) -> list:
    """解析详细分析区域的止损/止盈价 (fallback)"""
    stocks = []
    blocks = re.split(r'####\s+\d+\.', content)
    for block in blocks[1:]:
        code_match = re.search(r'(\d{6})\s*-\s*(.+?)(?:\n|$)', block)
        if not code_match:
            continue
        buy_match = re.search(r'建议买入价.*?(\d+\.?\d*)元', block)
        stop_match = re.search(r'建议止损价.*?(\d+\.?\d*)元', block)
        target_match = re.search(r'建议止盈价.*?(\d+\.?\d*)元', block)
        close_match = re.search(r'收盘价.*?(\d+\.?\d*)元', block)
        if buy_match and stop_match and target_match:
            stocks.append({
                'stock_code': code_match.group(1),
                'stock_name': code_match.group(2).strip(),
                'recommendation': '',
                'close_price': float(close_match.group(1)) if close_match else 0,
                'buy_price': float(buy_match.group(1)),
                'stop_loss': float(stop_match.group(1)),
                'target': float(target_match.group(1)),
                'position_pct': 0,
            })
    return stocks


def get_future_prices(conn: sqlite3.Connection, stock_code: str,
                      start_date: str, hold_days: int) -> pd.DataFrame:
    """获取指定股票从start_date起的未来hold_days个交易日价格"""
    query = """
    SELECT dq.trade_date, dq.open, dq.high, dq.low, dq.close
    FROM daily_quotes dq
    JOIN securities s ON dq.security_id = s.id
    WHERE s.code = ? AND dq.trade_date > ?
    ORDER BY dq.trade_date ASC
    LIMIT ?
    """
    return pd.read_sql_query(query, conn, params=(stock_code, start_date, hold_days))


def validate_single_stock(conn: sqlite3.Connection, stock: dict,
                          analysis_date: str, hold_days: int) -> dict:
    """验证单只股票: 限价买入 → 持仓期内止盈/止损触发情况"""
    future = get_future_prices(conn, stock['stock_code'], analysis_date, hold_days)
    if future.empty:
        return None

    buy_price = stock['buy_price']
    stop_loss = stock['stop_loss']
    target = stock['target']
    close_price = stock['close_price']

    if buy_price <= 0:
        buy_price = close_price
    if stop_loss <= 0 or target <= 0 or buy_price <= 0:
        return None

    # Step 1: 检查是否能以buy_price买入 (限价单: 当日low <= buy_price)
    entry_day = None
    actual_entry = None
    for idx in range(len(future)):
        row = future.iloc[idx]
        if row['low'] <= buy_price:
            entry_day = idx + 1
            # 限价单成交价 = min(open, buy_price)
            actual_entry = min(row['open'], buy_price)
            break
        elif row['open'] <= buy_price * 1.005:
            # 开盘价接近买入价(0.5%内), 也算成交
            entry_day = idx + 1
            actual_entry = row['open']
            break

    if entry_day is None:
        # 未成交
        return {
            'stock_code': stock['stock_code'],
            'stock_name': stock['stock_name'],
            'recommendation': stock.get('recommendation', ''),
            'position_pct': stock.get('position_pct', 0),
            'analysis_date': analysis_date,
            'close_price': close_price,
            'buy_price': buy_price,
            'stop_loss': stop_loss,
            'target': target,
            'actual_entry': 0,
            'exit_price': 0,
            'entry_day': 0,
            'target_hit_day': None,
            'stop_hit_day': None,
            'outcome': 'no_fill',
            'hold_return': 0,
            'trade_return': 0,
            'preset_risk_pct': (buy_price - stop_loss) / buy_price * 100,
            'preset_reward_pct': (target - buy_price) / buy_price * 100,
            'days_available': len(future),
        }

    # Step 2: 从买入日起检查止盈/止损
    remaining = future.iloc[entry_day:]  # 买入日之后
    target_hit_day = None
    stop_hit_day = None

    for idx in range(len(remaining)):
        row = remaining.iloc[idx]
        day_after_entry = idx + 1
        if target_hit_day is None and row['high'] >= target:
            target_hit_day = day_after_entry
        if stop_hit_day is None and row['low'] <= stop_loss:
            stop_hit_day = day_after_entry

    # 持仓期末
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

    # 按止盈/止损执行的收益
    if outcome in ('target_first', 'target_only'):
        trade_return = (target - actual_entry) / actual_entry
    elif outcome in ('stop_first', 'stop_only'):
        trade_return = (stop_loss - actual_entry) / actual_entry
    else:
        trade_return = hold_return  # 未触发，持有到期

    return {
        'stock_code': stock['stock_code'],
        'stock_name': stock['stock_name'],
        'recommendation': stock.get('recommendation', ''),
        'position_pct': stock.get('position_pct', 0),
        'analysis_date': analysis_date,
        'close_price': close_price,
        'buy_price': buy_price,
        'stop_loss': stop_loss,
        'target': target,
        'actual_entry': actual_entry,
        'exit_price': exit_price,
        'entry_day': entry_day,
        'target_hit_day': target_hit_day,
        'stop_hit_day': stop_hit_day,
        'outcome': outcome,
        'hold_return': hold_return,
        'trade_return': trade_return,
        'preset_risk_pct': (buy_price - stop_loss) / buy_price * 100,
        'preset_reward_pct': (target - buy_price) / buy_price * 100,
        'days_available': len(future),
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

    # 按执行策略的收益 (止盈=target收益, 止损=stop收益, 持有=到期收益)
    avg_trade = filled['trade_return'].mean() * 100
    med_trade = filled['trade_return'].median() * 100
    print(f"  策略收益: 均值{avg_trade:+.2f}%, 中位{med_trade:+.2f}%")

    # 持有到期收益
    avg_hold = filled['hold_return'].mean() * 100
    print(f"  持有到期: 均值{avg_hold:+.2f}%")

    # 加权收益 (按仓位)
    if filled['position_pct'].sum() > 0:
        weights = filled['position_pct'] / 100
        weighted_ret = (filled['trade_return'] * weights).sum() / weights.sum() * 100
        print(f"  仓位加权收益: {weighted_ret:+.2f}%")


def run_validation(report_dir: str, top_n: int = 20, hold_days: int = 10,
                   filter_rec: str = None):
    """运行完整的止盈止损验证"""
    report_path = Path(report_dir)
    if not report_path.exists():
        print(f"报告目录不存在: {report_dir}")
        return

    conn = sqlite3.connect(str(DB_PATH))

    report_files = sorted(report_path.glob("选股分析报告_*.md"))
    print(f"找到 {len(report_files)} 份报告")

    all_results = []
    skipped = 0

    for rf in report_files:
        report_data = parse_report(rf)
        if not report_data['date'] or not report_data['stocks']:
            skipped += 1
            continue

        stocks = report_data['stocks'][:top_n]

        for stock in stocks:
            result = validate_single_stock(conn, stock, report_data['date'], hold_days)
            if result:
                all_results.append(result)

    conn.close()

    if not all_results:
        print("没有有效的验证结果")
        return

    df = pd.DataFrame(all_results)

    # 可选过滤
    if filter_rec:
        recs = [r.strip() for r in filter_rec.split(',')]
        df = df[df['recommendation'].isin(recs)]
        if df.empty:
            print(f"过滤后无数据 (filter: {filter_rec})")
            return

    # ========== 总体统计 ==========
    total = len(df)
    filled = df[df['outcome'] != 'no_fill']

    print(f"\n{'='*70}")
    print(f"ML-Enhanced 止盈止损回测报告")
    print(f"{'='*70}")
    print(f"报告目录: {report_dir}")
    print(f"报告数量: {len(report_files)} (有效: {len(report_files)-skipped})")
    print(f"总样本: {total}, 限价成交: {len(filled)}, 未成交: {total-len(filled)}")
    print(f"持仓天数: {hold_days}天, Top-N: {top_n}")

    if len(filled) > 0:
        print(f"\n--- 总体表现 ---")
        print_group_stats(df, "全部")

    # ========== 按投资建议分组 ==========
    rec_order = ['强烈买入', '买入', '谨慎买入', '观望', '回避']
    recs_in_data = [r for r in rec_order if r in df['recommendation'].values]
    # 加上不在预定义列表中的
    other_recs = [r for r in df['recommendation'].unique() if r and r not in rec_order]
    recs_in_data += other_recs

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

    # ========== 只看买入+强烈买入 (重点) ==========
    buy_df = df[df['recommendation'].isin(['强烈买入', '买入'])]
    if len(buy_df) > 0:
        buy_filled = buy_df[buy_df['outcome'] != 'no_fill']
        print(f"\n{'='*70}")
        print(f"🎯 重点: 买入+强烈买入 综合表现")
        print(f"{'='*70}")
        print_group_stats(buy_df, "买入类")

        if len(buy_filled) > 0:
            # 模拟组合收益
            wins = buy_filled['outcome'].isin(['target_first', 'target_only']).sum()
            losses = buy_filled['outcome'].isin(['stop_first', 'stop_only']).sum()
            avg_win = buy_filled.loc[buy_filled['outcome'].isin(['target_first', 'target_only']), 'trade_return'].mean() if wins > 0 else 0
            avg_loss = buy_filled.loc[buy_filled['outcome'].isin(['stop_first', 'stop_only']), 'trade_return'].mean() if losses > 0 else 0

            print(f"\n  平均盈利: {avg_win*100:+.2f}% ({wins}次)")
            print(f"  平均亏损: {avg_loss*100:+.2f}% ({losses}次)")

            # 期望收益 E = P(win)*avg_win + P(loss)*avg_loss
            n_decided = wins + losses
            if n_decided > 0:
                p_win = wins / n_decided
                expected = p_win * avg_win + (1 - p_win) * avg_loss
                print(f"  期望收益(每笔): {expected*100:+.2f}%")

    # ========== 保存 ==========
    output_dir = PROJECT_ROOT / "reports" / "target_price_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_dir / f"validation_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='验证ML增强止盈止损目标价')
    parser.add_argument('--report-dir', required=True, help='报告目录路径')
    parser.add_argument('--top-n', type=int, default=20, help='每份报告取前N只股票 (default: 20)')
    parser.add_argument('--hold-days', type=int, default=10, help='持仓天数 (default: 10)')
    parser.add_argument('--filter-rec', type=str, default=None,
                        help='过滤投资建议, 逗号分隔 (如: "强烈买入,买入")')

    args = parser.parse_args()
    run_validation(args.report_dir, args.top_n, args.hold_days, args.filter_rec)


if __name__ == '__main__':
    main()
