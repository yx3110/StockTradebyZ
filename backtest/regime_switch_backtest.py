"""
双模型牛熊切换回测

根据0AMV牛熊体制信号，牛市用ng1.0.1，熊市用ng1.0.4-3seed。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import shutil
import argparse
from backtest.backtest_report_based import load_reports, run_single_backtest


DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data_adapter', 'stock_data.db'
)


def load_regime(db_path=None, version: str = 'v1'):
    """加载每日 regime，返回 {date_str: regime_int}.

    version='v1': use market_amv.amv_regime (legacy V11 0AMV)
    version='v2': use market_regime_signals.regime_v2 (ng2.0a multi-beta vote)
    """
    if db_path is None:
        db_path = DB_PATH
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute('PRAGMA busy_timeout=30000')
    try:
        if version == 'v1':
            cur = conn.execute(
                'SELECT trade_date, amv_regime FROM market_amv ORDER BY trade_date'
            )
        elif version == 'v2':
            cur = conn.execute(
                'SELECT trade_date, regime_v2 FROM market_regime_signals '
                'WHERE regime_v2 IS NOT NULL ORDER BY trade_date'
            )
        else:
            raise ValueError(f'unknown regime version: {version!r}')
        regime = {}
        for date_str, r in cur.fetchall():
            # Keep YYYY-MM-DD format to match report keys from load_reports()
            regime[date_str] = int(r)
        return regime
    finally:
        conn.close()


def merge_reports_by_regime(bull_reports, bear_reports, regime):
    """按regime合并两个模型的报告"""
    merged = {}
    bull_count = 0
    bear_count = 0
    skip_count = 0

    all_dates = sorted(set(bull_reports.keys()) & set(bear_reports.keys()))

    for date in all_dates:
        r = regime.get(date)
        if r is None:
            skip_count += 1
            continue
        if r == 1:
            merged[date] = bull_reports[date]
            bull_count += 1
        else:
            merged[date] = bear_reports[date]
            bear_count += 1

    print(f'  合并报告: {len(merged)}天 (牛市用101: {bull_count}天, '
          f'熊市用104: {bear_count}天, 无regime跳过: {skip_count}天)')
    return merged


def generate_merged_report_dir(bull_dir, bear_dir, regime, out_dir):
    """将合并报告复制到临时目录，供北极星评估使用"""
    os.makedirs(out_dir, exist_ok=True)
    # 清空旧文件
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))

    bull_dates = set()
    bear_dates = set()

    def _regime_for_filename(date_compact):
        """date_compact is YYYYMMDD; regime keys are YYYY-MM-DD"""
        date_dashed = f'{date_compact[:4]}-{date_compact[4:6]}-{date_compact[6:]}'
        return regime.get(date_dashed)

    for f in os.listdir(bull_dir):
        if f.startswith('analysis_data_') and f.endswith('.json'):
            date = f.replace('analysis_data_', '').replace('.json', '')
            r = _regime_for_filename(date)
            if r == 1:
                shutil.copy2(os.path.join(bull_dir, f), os.path.join(out_dir, f))
                bull_dates.add(date)
            elif r == -1:
                bear_f = os.path.join(bear_dir, f)
                if os.path.exists(bear_f):
                    shutil.copy2(bear_f, os.path.join(out_dir, f))
                    bear_dates.add(date)

    # 补充bear_dir中有但bull_dir中没有的日期
    for f in os.listdir(bear_dir):
        if f.startswith('analysis_data_') and f.endswith('.json'):
            date = f.replace('analysis_data_', '').replace('.json', '')
            if date not in bull_dates and date not in bear_dates:
                r = _regime_for_filename(date)
                if r == -1:
                    shutil.copy2(os.path.join(bear_dir, f), os.path.join(out_dir, f))
                    bear_dates.add(date)

    total = len(bull_dates) + len(bear_dates)
    print(f'  生成合并报告目录: {out_dir} ({total}份, 牛{len(bull_dates)}+熊{len(bear_dates)})')
    return total


def run_comparison(
    bull_dir='reports/daily_selection_ng101',
    bear_dir='reports/daily_selection_ng104_ensemble_3seed',
    top_n=10,
    focus_days=10,
    rank_field='score',
    regime_version='v1',
    out_dir=None,
):
    """运行三方对比回测"""
    if out_dir is None:
        out_dir = f'reports/daily_selection_regime_switch_{regime_version}'
    print('=' * 70)
    print(f'  0AMV牛熊切换 双模型回测 (regime={regime_version})')
    print('=' * 70)

    regime = load_regime(version=regime_version)
    if not regime:
        print('ERROR: market_amv表为空，先运行 indicators/market_amv.py')
        return

    bull_days = sum(1 for v in regime.values() if v == 1)
    bear_days = sum(1 for v in regime.values() if v == -1)
    print(f'\n体制信号: {len(regime)}天, 牛市{bull_days}天, 熊市{bear_days}天')

    # 加载报告
    print(f'\n加载牛市模型(ng101): {bull_dir}')
    bull_reports = load_reports(bull_dir, rank_field=rank_field)
    print(f'  {len(bull_reports)}天')

    print(f'加载熊市模型(ng104-3s): {bear_dir}')
    bear_reports = load_reports(bear_dir, rank_field=rank_field)
    print(f'  {len(bear_reports)}天')

    # 合并
    print(f'\n按regime合并...')
    merged_reports = merge_reports_by_regime(bull_reports, bear_reports, regime)
    if not merged_reports:
        print('ERROR: 合并后无报告')
        return

    # 生成合并报告目录（供后续北极星评估）
    merged_dir = out_dir
    generate_merged_report_dir(bull_dir, bear_dir, regime, merged_dir)

    # 三方回测
    configs = [
        ('NG101-纯牛模型', bull_reports),
        ('NG104-纯熊模型', bear_reports),
        ('AMV切换(101+104)', merged_reports),
    ]

    results = []
    for label, reports in configs:
        print(f'\n{"=" * 50}')
        print(f'  回测: {label} (Top-{top_n}, {focus_days}日持仓)')
        print(f'{"=" * 50}')
        result = run_single_backtest(reports, label, top_n=top_n, focus_days=focus_days)
        results.append((label, result))

    # 对比摘要
    print(f'\n{"=" * 70}')
    print(f'  三方对比摘要 (Top-{top_n}, {focus_days}日持仓, 无CPPI)')
    print(f'{"=" * 70}')
    header = f'{"指标":<20}'
    for label, _ in results:
        header += f' {label:<20}'
    print(header)
    print('-' * 80)

    metrics = [
        ('年化收益(毛)', 'annual_return'),
        ('Sharpe', 'sharpe'),
        ('最大回撤', 'max_drawdown'),
        ('月度胜率', 'monthly_win_rate'),
    ]

    for display_name, key in metrics:
        row = f'{display_name:<20}'
        for label, result in results:
            s = result.get('summary', {}).get(focus_days, {})
            v = s.get(key, 0)
            if v is None:
                v = 0
            if key in ('annual_return', 'max_drawdown', 'monthly_win_rate'):
                if abs(v) < 50:
                    row += f' {v*100:>7.1f}%{"":>12}'
                else:
                    row += f' {v:>7.1f}%{"":>12}'
            else:
                row += f' {v:>7.3f}{"":>13}'
        print(row)

    print(f'\n合并报告已保存到: {merged_dir}')
    print(f'可用北极星V5.2评估:')
    print(f'  python3 backtest/run_north_star_eval.py --backtest \\')
    print(f'    --report-dir {merged_dir} \\')
    print(f'    --label "AMV切换" --top-n {top_n} --focus-days {focus_days} \\')
    print(f'    --rank-field {rank_field} --start-date 2020-01-01 --score-version v52')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='0AMV牛熊切换双模型回测')
    parser.add_argument('--bull-dir', default='reports/daily_selection_ng101')
    parser.add_argument('--bear-dir', default='reports/daily_selection_ng104_ensemble_3seed')
    parser.add_argument('--top-n', type=int, default=10)
    parser.add_argument('--focus-days', type=int, default=10)
    parser.add_argument('--rank-field', default='score')
    parser.add_argument('--regime-version', choices=['v1', 'v2'], default='v1',
                        help='regime来源: v1=market_amv (V11 0AMV), v2=market_regime_signals (ng2.0a多beta投票)')
    parser.add_argument('--out-dir', default=None,
                        help='合并报告输出目录 (默认: reports/daily_selection_regime_switch_{regime-version})')
    args = parser.parse_args()
    run_comparison(
        bull_dir=args.bull_dir, bear_dir=args.bear_dir,
        top_n=args.top_n, focus_days=args.focus_days,
        rank_field=args.rank_field,
        regime_version=args.regime_version,
        out_dir=args.out_dir,
    )
