#!/usr/bin/env python3
"""
ML-Enhanced 止盈止损目标价回测验证

验证报告中的 stop_loss_price / take_profit_price 在实际市场中的表现：
- 目标命中率: 未来N天max(high) >= target_price
- 止损命中率: 未来N天min(low) <= stop_loss_price
- 胜率: 目标先于止损被触达
- 实际R:R vs 预设R:R

Usage:
    python3 backtest/validate_target_prices.py \
        --report-dir reports/daily_selection_v4.7.5 \
        --top-n 10 --hold-days 10
"""
import argparse
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data_adapter" / "stock_data.db"


def load_report_json(report_path: Path) -> dict:
    """从报告markdown中提取JSON格式的股票数据"""
    # 报告是markdown格式, 需要解析表格或找到JSON嵌入
    # 实际上报告中的数据存储在 all_stocks_with_scores 中
    # 我们需要从markdown表格中解析
    content = report_path.read_text(encoding='utf-8')

    stocks = []
    # 解析日期
    date_match = re.search(r'分析日期.*?(\d{4}-\d{2}-\d{2})', content)
    if not date_match:
        return {'date': None, 'stocks': []}
    analysis_date = date_match.group(1)

    # 解析详细分析区域中的止损/止盈价
    # 格式: #### N. XXXXXX - 股票名称
    #        建议买入价: XX.XX元
    #        建议止损价: XX.XX元
    #        建议止盈价: XX.XX元
    stock_blocks = re.split(r'####\s+\d+\.', content)

    for block in stock_blocks[1:]:  # 跳过第一个空块
        code_match = re.search(r'(\d{6})\s*-\s*(.+?)(?:\n|$)', block)
        if not code_match:
            continue

        stock_code = code_match.group(1)
        stock_name = code_match.group(2).strip()

        buy_match = re.search(r'建议买入价.*?(\d+\.?\d*)元', block)
        stop_match = re.search(r'建议止损价.*?(\d+\.?\d*)元', block)
        target_match = re.search(r'建议止盈价.*?(\d+\.?\d*)元', block)
        close_match = re.search(r'收盘价.*?(\d+\.?\d*)元', block)

        if buy_match and stop_match and target_match:
            stocks.append({
                'stock_code': stock_code,
                'stock_name': stock_name,
                'close_price': float(close_match.group(1)) if close_match else 0,
                'buy_price': float(buy_match.group(1)),
                'stop_loss': float(stop_match.group(1)),
                'target': float(target_match.group(1)),
            })

    # 也从汇总表中解析（如果有止损/目标列）
    # 格式: | 排名 | 股票代码 | ... | 止损价 | 目标价 |
    table_stocks = parse_summary_table(content)

    # 合并: 详细区优先(有更多字段), 汇总表补充
    detailed_codes = {s['stock_code'] for s in stocks}
    for ts in table_stocks:
        if ts['stock_code'] not in detailed_codes:
            stocks.append(ts)

    return {'date': analysis_date, 'stocks': stocks}


def parse_summary_table(content: str) -> list:
    """解析汇总排名表中的止损/目标价"""
    stocks = []

    # 找到包含"止损价"的表头
    lines = content.split('\n')
    in_table = False
    header_cols = []

    for line in lines:
        if '止损价' in line and '目标价' in line and '|' in line:
            # 找到表头
            header_cols = [c.strip() for c in line.split('|')]
            in_table = True
            continue

        if in_table and line.strip().startswith('|---'):
            continue  # 跳过分隔线

        if in_table and '|' in line and line.strip().startswith('|'):
            cols = [c.strip() for c in line.split('|')]
            if len(cols) >= len(header_cols) - 1:
                try:
                    # 找列索引
                    code_idx = next(i for i, h in enumerate(header_cols) if '股票代码' in h)
                    name_idx = next(i for i, h in enumerate(header_cols) if '股票名称' in h)
                    stop_idx = next(i for i, h in enumerate(header_cols) if '止损价' in h)
                    target_idx = next(i for i, h in enumerate(header_cols) if '目标价' in h)

                    stock_code = cols[code_idx].strip()
                    if not re.match(r'^\d{6}$', stock_code):
                        continue

                    stocks.append({
                        'stock_code': stock_code,
                        'stock_name': cols[name_idx].strip(),
                        'close_price': 0,  # 汇总表可能没有收盘价
                        'buy_price': 0,
                        'stop_loss': float(cols[stop_idx]),
                        'target': float(cols[target_idx]),
                    })
                except (StopIteration, ValueError, IndexError):
                    continue
        elif in_table and not line.strip().startswith('|'):
            in_table = False  # 表格结束

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
    df = pd.read_sql_query(query, conn, params=(stock_code, start_date, hold_days))
    return df


def validate_single_stock(conn: sqlite3.Connection, stock: dict,
                          analysis_date: str, hold_days: int) -> dict:
    """验证单只股票的止损/目标价表现"""
    future = get_future_prices(conn, stock['stock_code'], analysis_date, hold_days)

    if future.empty:
        return None

    stop_loss = stock['stop_loss']
    target = stock['target']
    buy_price = stock['buy_price'] if stock['buy_price'] > 0 else stock['close_price']

    if stop_loss <= 0 or target <= 0 or buy_price <= 0:
        return None

    # 逐日检查: 目标和止损哪个先被触达
    target_hit_day = None
    stop_hit_day = None

    for idx, row in future.iterrows():
        day_num = idx + 1
        # 假设次日以open买入
        if day_num == 1:
            actual_entry = row['open']  # 实际买入价

        if target_hit_day is None and row['high'] >= target:
            target_hit_day = day_num
        if stop_hit_day is None and row['low'] <= stop_loss:
            stop_hit_day = day_num

    # 持仓期末收盘价
    exit_price = future.iloc[-1]['close']
    max_high = future['high'].max()
    min_low = future['low'].min()

    # 判断结果
    if target_hit_day and stop_hit_day:
        if target_hit_day <= stop_hit_day:
            outcome = 'target_first'
        else:
            outcome = 'stop_first'
    elif target_hit_day:
        outcome = 'target_only'
    elif stop_hit_day:
        outcome = 'stop_only'
    else:
        outcome = 'neither'

    # 实际收益 (持有到期末)
    hold_return = (exit_price - buy_price) / buy_price if buy_price > 0 else 0

    # 预设R:R
    preset_risk = (buy_price - stop_loss) / buy_price if buy_price > 0 else 0
    preset_reward = (target - buy_price) / buy_price if buy_price > 0 else 0
    preset_rr = preset_reward / preset_risk if preset_risk > 0 else 0

    return {
        'stock_code': stock['stock_code'],
        'stock_name': stock['stock_name'],
        'analysis_date': analysis_date,
        'buy_price': buy_price,
        'stop_loss': stop_loss,
        'target': target,
        'actual_entry': actual_entry if len(future) > 0 else buy_price,
        'exit_price': exit_price,
        'max_high': max_high,
        'min_low': min_low,
        'target_hit_day': target_hit_day,
        'stop_hit_day': stop_hit_day,
        'outcome': outcome,
        'hold_return': hold_return,
        'preset_risk_pct': preset_risk * 100,
        'preset_reward_pct': preset_reward * 100,
        'preset_rr': preset_rr,
        'days_available': len(future),
    }


def run_validation(report_dir: str, top_n: int = 10, hold_days: int = 10):
    """运行完整的止盈止损验证"""
    report_path = Path(report_dir)
    if not report_path.exists():
        print(f"报告目录不存在: {report_dir}")
        return

    # 连接数据库
    conn = sqlite3.connect(str(DB_PATH))

    # 扫描所有报告
    report_files = sorted(report_path.glob("选股分析报告_*.md"))
    print(f"找到 {len(report_files)} 份报告")

    all_results = []
    skipped = 0

    for rf in report_files:
        report_data = load_report_json(rf)
        if not report_data['date'] or not report_data['stocks']:
            skipped += 1
            continue

        # 只取top_n只股票
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

    # ========== 统计分析 ==========
    total = len(df)
    print(f"\n{'='*70}")
    print(f"ML-Enhanced 止盈止损验证报告")
    print(f"{'='*70}")
    print(f"报告目录: {report_dir}")
    print(f"报告数量: {len(report_files)} (有效: {len(report_files)-skipped}, 跳过: {skipped})")
    print(f"验证样本: {total} 只股票")
    print(f"持仓天数: {hold_days}天")
    print(f"Top-N: {top_n}")

    # 1. 命中率
    target_hit = df['target_hit_day'].notna().sum()
    stop_hit = df['stop_hit_day'].notna().sum()
    print(f"\n--- 命中率 ---")
    print(f"目标命中率: {target_hit}/{total} = {target_hit/total*100:.1f}%")
    print(f"止损命中率: {stop_hit}/{total} = {stop_hit/total*100:.1f}%")

    # 2. 结果分布
    outcomes = df['outcome'].value_counts()
    print(f"\n--- 结果分布 ---")
    for outcome, count in outcomes.items():
        label = {
            'target_first': '目标先触达 (胜)',
            'target_only': '仅触达目标 (胜)',
            'stop_first': '止损先触达 (负)',
            'stop_only': '仅触达止损 (负)',
            'neither': '均未触达 (持有到期)',
        }.get(outcome, outcome)
        print(f"  {label}: {count} ({count/total*100:.1f}%)")

    # 3. 胜率
    wins = ((df['outcome'] == 'target_first') | (df['outcome'] == 'target_only')).sum()
    losses = ((df['outcome'] == 'stop_first') | (df['outcome'] == 'stop_only')).sum()
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    print(f"\n--- 胜率 ---")
    print(f"胜: {wins}, 负: {losses}, 未决: {total - wins - losses}")
    print(f"胜率: {win_rate:.1f}%")

    # 4. 收益统计
    print(f"\n--- 持有到期收益 ---")
    print(f"平均收益: {df['hold_return'].mean()*100:+.2f}%")
    print(f"中位收益: {df['hold_return'].median()*100:+.2f}%")
    print(f"胜率(正收益): {(df['hold_return']>0).mean()*100:.1f}%")
    print(f"最大盈利: {df['hold_return'].max()*100:+.2f}%")
    print(f"最大亏损: {df['hold_return'].min()*100:+.2f}%")

    # 5. R:R分析
    print(f"\n--- 风险收益比 ---")
    print(f"预设平均R:R: 1:{df['preset_rr'].mean():.2f}")
    print(f"预设中位R:R: 1:{df['preset_rr'].median():.2f}")
    print(f"预设平均风险: -{df['preset_risk_pct'].mean():.2f}%")
    print(f"预设平均收益: +{df['preset_reward_pct'].mean():.2f}%")

    # 6. 目标命中天数分布
    hit_days = df.loc[df['target_hit_day'].notna(), 'target_hit_day']
    if len(hit_days) > 0:
        print(f"\n--- 目标触达天数 ---")
        print(f"平均: {hit_days.mean():.1f}天")
        print(f"中位: {hit_days.median():.1f}天")
        print(f"1-3天: {(hit_days<=3).sum()} ({(hit_days<=3).mean()*100:.0f}%)")
        print(f"4-7天: {((hit_days>3)&(hit_days<=7)).sum()} ({((hit_days>3)&(hit_days<=7)).mean()*100:.0f}%)")
        print(f"8-10天: {(hit_days>7).sum()} ({(hit_days>7).mean()*100:.0f}%)")

    # 保存详细结果
    output_dir = PROJECT_ROOT / "reports" / "target_price_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_dir / f"validation_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='验证ML增强止盈止损目标价')
    parser.add_argument('--report-dir', required=True, help='报告目录路径')
    parser.add_argument('--top-n', type=int, default=10, help='每份报告取前N只股票 (default: 10)')
    parser.add_argument('--hold-days', type=int, default=10, help='持仓天数 (default: 10)')

    args = parser.parse_args()
    run_validation(args.report_dir, args.top_n, args.hold_days)


if __name__ == '__main__':
    main()
