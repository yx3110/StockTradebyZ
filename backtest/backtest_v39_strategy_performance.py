#!/usr/bin/env python3
"""
V3.9策略回测分析
分析各策略在1d, 3d, 5d, 10d, 15d的表现，按月汇总
"""

import os
import sys
import re
import glob
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import warnings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore')

# 策略名称映射
STRATEGY_MAP = {
    '少负战法': 'shaofu',
    '少负': 'shaofu',
    'SuperB1战法': 'superb1',
    'SuperB1': 'superb1',
    '补票战法': 'bupiao',
    '补票': 'bupiao',
    'TePu战法': 'tepu',
    'TePu': 'tepu',
    '填坑战法': 'tiankeng',
    '填坑': 'tiankeng',
    '知行战法': 'zhixing',
    '知行': 'zhixing',
    '上穿60放量战法': 'ma60',
    '上穿60': 'ma60',
    '暴力K战法': 'baoli',
    '暴力K战': 'baoli',
    '暴力K': 'baoli',
}

STRATEGY_CHINESE = {
    'shaofu': '少负战法',
    'superb1': 'SuperB1战法',
    'bupiao': '补票战法',
    'tepu': 'TePu战法',
    'tiankeng': '填坑战法',
    'zhixing': '知行战法',
    'ma60': '上穿60放量战法',
    'baoli': '暴力K战法',
}


class V39StrategyBacktester:
    """V3.9策略回测器"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(PROJECT_ROOT / "data_adapter" / "stock_data.db")
        self.db_path = db_path
        self.reports_dir = str(PROJECT_ROOT / "reports" / "daily_selection_v3.9")
        self.holding_periods = [1, 3, 5, 10, 15]

    def get_db_connection(self):
        """获取数据库连接"""
        return sqlite3.connect(self.db_path)

    def parse_report(self, report_path: str) -> dict:
        """解析单个报告文件，提取各策略选中的股票"""
        result = {
            'report_date': None,
            'buy_date': None,
            'strategies': defaultdict(list)
        }

        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取分析日期和买入日期
        date_match = re.search(r'\*\*分析日期\*\*:\s*(\d{4}-\d{2}-\d{2})', content)
        buy_match = re.search(r'\*\*推荐买入日期\*\*:\s*(\d{4}-\d{2}-\d{2})', content)

        if date_match:
            result['report_date'] = date_match.group(1)
        if buy_match:
            result['buy_date'] = buy_match.group(1)

        # 解析股票评分表格
        table_pattern = r'\|\s*\d+\s*\|\s*(\d{6})\s*\|\s*[^\|]+\|\s*([^\|]+)\|'
        matches = re.findall(table_pattern, content)

        for code, strategies_str in matches:
            # 解析策略字符串（可能有多个策略，用逗号分隔）
            strategies = [s.strip() for s in strategies_str.split(',')]
            for strategy in strategies:
                # 标准化策略名称
                std_strategy = None
                for key, value in STRATEGY_MAP.items():
                    if key in strategy:
                        std_strategy = value
                        break

                if std_strategy:
                    result['strategies'][std_strategy].append(code)

        return result

    def get_future_returns(self, code: str, buy_date: str) -> dict:
        """获取股票在买入日期后的各期收益"""
        returns = {}

        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()

            # 日期格式已经是 YYYY-MM-DD，不需要转换
            buy_date_db = buy_date

            # 获取可能的代码格式
            if code.startswith('6'):
                code_with_suffix = f"{code}.SH"
            elif code.startswith(('0', '3')):
                code_with_suffix = f"{code}.SZ"
            elif code.startswith('8'):
                code_with_suffix = f"{code}.BJ"
            else:
                code_with_suffix = code

            # 获取买入日期的收盘价
            query = """
            SELECT dq.close
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE (s.code = ? OR s.code = ?) AND dq.trade_date = ?
            """
            cursor.execute(query, (code, code_with_suffix, buy_date_db))
            result = cursor.fetchone()

            if not result:
                conn.close()
                return returns

            buy_price = result[0]

            # 获取买入日期后的所有交易日收盘价
            query = """
            SELECT dq.trade_date, dq.close
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE (s.code = ? OR s.code = ?) AND dq.trade_date > ?
            ORDER BY dq.trade_date
            LIMIT 20
            """
            cursor.execute(query, (code, code_with_suffix, buy_date_db))
            future_prices = cursor.fetchall()

            conn.close()

            if not future_prices:
                return returns

            # 计算各期收益
            for period in self.holding_periods:
                if len(future_prices) >= period:
                    future_price = future_prices[period - 1][1]
                    returns[f'{period}d'] = (future_price - buy_price) / buy_price * 100

        except Exception as e:
            print(f"Error getting returns for {code} on {buy_date}: {e}")

        return returns

    def run_backtest(self) -> pd.DataFrame:
        """运行回测，分析所有报告"""
        # 获取所有报告文件
        report_files = glob.glob(f"{self.reports_dir}/选股分析报告_*.md")
        print(f"找到 {len(report_files)} 份报告")

        # 收集所有选股记录
        all_records = []

        for i, report_path in enumerate(sorted(report_files)):
            if (i + 1) % 10 == 0:
                print(f"处理进度: {i + 1}/{len(report_files)}")

            report_data = self.parse_report(report_path)

            if not report_data['buy_date']:
                continue

            buy_date = report_data['buy_date']
            report_date = report_data['report_date']

            # 对每个策略的每只股票计算收益
            for strategy, codes in report_data['strategies'].items():
                for code in codes:
                    returns = self.get_future_returns(code, buy_date)

                    if returns:
                        record = {
                            'report_date': report_date,
                            'buy_date': buy_date,
                            'month': buy_date[:7],  # YYYY-MM
                            'code': code,
                            'strategy': strategy,
                        }
                        record.update(returns)
                        all_records.append(record)

        df = pd.DataFrame(all_records)
        print(f"共收集 {len(df)} 条有效记录")
        return df

    def generate_monthly_summary(self, df: pd.DataFrame) -> dict:
        """生成按月汇总的策略表现"""
        if df.empty:
            return {}

        periods = ['1d', '3d', '5d', '10d', '15d']

        # 按策略和月份汇总
        summary = {}

        for strategy in df['strategy'].unique():
            strategy_df = df[df['strategy'] == strategy]
            summary[strategy] = {}

            for month in sorted(strategy_df['month'].unique()):
                month_df = strategy_df[strategy_df['month'] == month]

                stats = {
                    'count': len(month_df),
                }

                for period in periods:
                    if period in month_df.columns:
                        valid_data = month_df[period].dropna()
                        if len(valid_data) > 0:
                            stats[f'{period}_mean'] = valid_data.mean()
                            stats[f'{period}_median'] = valid_data.median()
                            stats[f'{period}_win_rate'] = (valid_data > 0).mean() * 100
                            stats[f'{period}_max'] = valid_data.max()
                            stats[f'{period}_min'] = valid_data.min()

                summary[strategy][month] = stats

        return summary

    def generate_report(self, df: pd.DataFrame, summary: dict):
        """生成回测报告"""
        report_lines = []
        report_lines.append("# V3.9策略月度回测分析报告\n")
        report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_lines.append(f"**数据来源**: reports/daily_selection_v3.9\n")
        report_lines.append(f"**总记录数**: {len(df)}\n\n")

        periods = ['1d', '3d', '5d', '10d', '15d']

        # 总体策略表现
        report_lines.append("## 1. 各策略总体表现\n")
        report_lines.append("| 策略 | 选股数 | 1d均收益 | 3d均收益 | 5d均收益 | 10d均收益 | 15d均收益 | 1d胜率 | 5d胜率 | 10d胜率 |\n")
        report_lines.append("|------|--------|----------|----------|----------|-----------|-----------|--------|--------|--------|\n")

        for strategy in sorted(summary.keys()):
            strategy_df = df[df['strategy'] == strategy]
            count = len(strategy_df)

            row = [STRATEGY_CHINESE.get(strategy, strategy), str(count)]

            for period in periods:
                if period in strategy_df.columns:
                    mean_val = strategy_df[period].dropna().mean()
                    row.append(f"{mean_val:+.2f}%")
                else:
                    row.append("N/A")

            for period in ['1d', '5d', '10d']:
                if period in strategy_df.columns:
                    win_rate = (strategy_df[period].dropna() > 0).mean() * 100
                    row.append(f"{win_rate:.1f}%")
                else:
                    row.append("N/A")

            report_lines.append("| " + " | ".join(row) + " |\n")

        # 按月份详细分析
        report_lines.append("\n## 2. 各策略月度详细表现\n")

        for strategy in sorted(summary.keys()):
            strategy_name = STRATEGY_CHINESE.get(strategy, strategy)
            report_lines.append(f"\n### {strategy_name}\n")

            if not summary[strategy]:
                report_lines.append("无数据\n")
                continue

            report_lines.append("| 月份 | 选股数 | 1d均收益 | 1d胜率 | 3d均收益 | 3d胜率 | 5d均收益 | 5d胜率 | 10d均收益 | 10d胜率 | 15d均收益 | 15d胜率 |\n")
            report_lines.append("|------|--------|----------|--------|----------|--------|----------|--------|-----------|---------|-----------|--------|\n")

            for month in sorted(summary[strategy].keys()):
                stats = summary[strategy][month]
                row = [month, str(stats['count'])]

                for period in periods:
                    mean_key = f'{period}_mean'
                    wr_key = f'{period}_win_rate'

                    if mean_key in stats:
                        row.append(f"{stats[mean_key]:+.2f}%")
                        row.append(f"{stats[wr_key]:.1f}%")
                    else:
                        row.extend(["N/A", "N/A"])

                report_lines.append("| " + " | ".join(row) + " |\n")

        # 月度汇总（所有策略）
        report_lines.append("\n## 3. 月度汇总（所有策略合计）\n")
        report_lines.append("| 月份 | 总选股数 | 1d均收益 | 1d胜率 | 5d均收益 | 5d胜率 | 10d均收益 | 10d胜率 | 15d均收益 | 15d胜率 |\n")
        report_lines.append("|------|----------|----------|--------|----------|--------|-----------|---------|-----------|--------|\n")

        for month in sorted(df['month'].unique()):
            month_df = df[df['month'] == month]
            row = [month, str(len(month_df))]

            for period in ['1d', '5d', '10d', '15d']:
                if period in month_df.columns:
                    valid_data = month_df[period].dropna()
                    if len(valid_data) > 0:
                        row.append(f"{valid_data.mean():+.2f}%")
                        row.append(f"{(valid_data > 0).mean() * 100:.1f}%")
                    else:
                        row.extend(["N/A", "N/A"])
                else:
                    row.extend(["N/A", "N/A"])

            report_lines.append("| " + " | ".join(row) + " |\n")

        # 最佳和最差表现
        report_lines.append("\n## 4. 策略表现排名\n")

        for period in periods:
            if period not in df.columns:
                continue

            report_lines.append(f"\n### {period}收益排名\n")

            strategy_performance = []
            for strategy in df['strategy'].unique():
                strategy_df = df[df['strategy'] == strategy]
                valid_data = strategy_df[period].dropna()
                if len(valid_data) >= 10:  # 至少10条记录
                    strategy_performance.append({
                        'strategy': STRATEGY_CHINESE.get(strategy, strategy),
                        'mean': valid_data.mean(),
                        'median': valid_data.median(),
                        'win_rate': (valid_data > 0).mean() * 100,
                        'count': len(valid_data)
                    })

            # 按均收益排序
            strategy_performance.sort(key=lambda x: x['mean'], reverse=True)

            report_lines.append("| 排名 | 策略 | 均收益 | 中位数 | 胜率 | 样本数 |\n")
            report_lines.append("|------|------|--------|--------|------|--------|\n")

            for i, perf in enumerate(strategy_performance, 1):
                report_lines.append(f"| {i} | {perf['strategy']} | {perf['mean']:+.2f}% | {perf['median']:+.2f}% | {perf['win_rate']:.1f}% | {perf['count']} |\n")

        return ''.join(report_lines)


def main():
    """主函数"""
    print("=" * 60)
    print("V3.9策略月度回测分析")
    print("=" * 60)

    backtester = V39StrategyBacktester()

    # 运行回测
    print("\n[1/3] 解析报告并获取收益数据...")
    df = backtester.run_backtest()

    if df.empty:
        print("未找到有效数据，请检查报告目录和数据库")
        return

    # 保存原始数据
    output_dir = PROJECT_ROOT / "reports" / "backtest"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"v39_strategy_backtest_data_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"原始数据已保存: {csv_path}")

    # 生成月度汇总
    print("\n[2/3] 生成月度汇总...")
    summary = backtester.generate_monthly_summary(df)

    # 生成报告
    print("\n[3/3] 生成回测报告...")
    report = backtester.generate_report(df, summary)

    report_path = output_dir / f"v39_strategy_monthly_backtest_{datetime.now().strftime('%Y%m%d')}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"回测报告已保存: {report_path}")

    # 打印概要
    print("\n" + "=" * 60)
    print("回测概要")
    print("=" * 60)

    periods = ['1d', '3d', '5d', '10d', '15d']

    print("\n各策略表现概览:")
    print("-" * 80)
    print(f"{'策略':<15} {'选股数':>8} {'1d收益':>10} {'5d收益':>10} {'10d收益':>10} {'5d胜率':>10}")
    print("-" * 80)

    for strategy in sorted(df['strategy'].unique()):
        strategy_df = df[df['strategy'] == strategy]
        count = len(strategy_df)

        d1 = strategy_df['1d'].dropna().mean() if '1d' in strategy_df.columns else float('nan')
        d5 = strategy_df['5d'].dropna().mean() if '5d' in strategy_df.columns else float('nan')
        d10 = strategy_df['10d'].dropna().mean() if '10d' in strategy_df.columns else float('nan')
        wr5 = (strategy_df['5d'].dropna() > 0).mean() * 100 if '5d' in strategy_df.columns else float('nan')

        strategy_name = STRATEGY_CHINESE.get(strategy, strategy)
        print(f"{strategy_name:<15} {count:>8} {d1:>+10.2f}% {d5:>+10.2f}% {d10:>+10.2f}% {wr5:>10.1f}%")

    print("-" * 80)
    print(f"\n详细报告请查看: {report_path}")


if __name__ == "__main__":
    main()
