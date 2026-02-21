#!/usr/bin/env python3
"""
采样回测 - 使用实时评分器
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sqlite3
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_future_returns(db_path: str, date: str, days: int = 5) -> Dict[str, float]:
    """获取未来收益"""
    conn = sqlite3.connect(db_path)

    # 获取未来日期
    cur = conn.execute(f"""
    SELECT DISTINCT trade_date FROM daily_quotes
    WHERE trade_date > '{date}'
    ORDER BY trade_date LIMIT {days}
    """)
    future_dates = [r[0] for r in cur.fetchall()]

    if len(future_dates) < days:
        conn.close()
        return {}

    target_date = future_dates[-1]

    # 获取收益
    query = f"""
    SELECT s.code,
           (q2.close - q1.close) / q1.close as future_return
    FROM securities s
    JOIN daily_quotes q1 ON s.id = q1.security_id AND q1.trade_date = '{date}'
    JOIN daily_quotes q2 ON s.id = q2.security_id AND q2.trade_date = '{target_date}'
    WHERE s.type = 'A股'
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return dict(zip(df['code'], df['future_return']))


def backtest_single_date(scorer, date: str, db_path: str, top_n: int = 10, holding_days: int = 5):
    """回测单个日期"""
    # 获取股票列表
    conn = sqlite3.connect(db_path)
    stocks = pd.read_sql_query("""
    SELECT code FROM securities
    WHERE type = 'A股' AND code NOT LIKE '688%' AND code NOT LIKE '8%'
    """, conn)['code'].tolist()
    conn.close()

    # 评分
    print(f"  评分中...", end=' ', flush=True)
    scores = scorer(stocks[:500], date)  # 限制500只以加速
    print(f"完成({len(scores)}只)")

    if not scores:
        return None

    # 选择top N
    sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    selected = [code for code, _ in sorted_stocks]

    # 获取收益
    returns = get_future_returns(db_path, date, holding_days)

    valid_returns = [returns.get(c, 0) for c in selected if c in returns]
    if not valid_returns:
        return None

    period_return = np.mean(valid_returns)

    return {
        'date': date,
        'return': period_return,
        'top5': selected[:5],
        'top5_scores': [scores[c] for c in selected[:5]]
    }


def main():
    db_path = str(Path(__file__).parent / 'data_adapter' / 'stock_data.db')

    # 采样日期
    sample_dates = ['2025-10-09', '2025-10-23', '2025-11-06', '2025-11-20']

    results = {}

    # V3.9回测
    print("\n" + "="*60)
    print("V3.9 回测")
    print("="*60)
    try:
        from ml_models.v39.v390_production_scorer import V390ProductionScorer
        scorer_v39 = V390ProductionScorer()

        def score_v39(stocks, date):
            preds = scorer_v39.predict_scores(stocks, date)
            return {k: v.get('score', 50) for k, v in preds.items()}

        v39_results = []
        for date in sample_dates:
            print(f"\n{date}:", end='')
            r = backtest_single_date(score_v39, date, db_path)
            if r:
                v39_results.append(r)
                print(f"  收益: {r['return']:+.2%}, Top3: {r['top5'][:3]}")

        if v39_results:
            avg_ret = np.mean([r['return'] for r in v39_results])
            win_rate = sum(1 for r in v39_results if r['return'] > 0) / len(v39_results)
            results['V3.9'] = {'avg_return': avg_ret, 'win_rate': win_rate, 'details': v39_results}
            print(f"\nV3.9汇总: 平均收益={avg_ret:+.2%}, 胜率={win_rate:.0%}")
    except Exception as e:
        print(f"V3.9失败: {e}")

    # V3.94回测
    print("\n" + "="*60)
    print("V3.94 回测")
    print("="*60)
    try:
        from ml_models.v39.v394_production_scorer import V394ProductionScorer
        scorer_v394 = V394ProductionScorer()

        def score_v394(stocks, date):
            preds = scorer_v394.predict_scores_with_ranking(stocks, date)
            return {k: v.get('score', 50) for k, v in preds.items()}

        v394_results = []
        for date in sample_dates:
            print(f"\n{date}:", end='')
            r = backtest_single_date(score_v394, date, db_path)
            if r:
                v394_results.append(r)
                print(f"  收益: {r['return']:+.2%}, Top3: {r['top5'][:3]}")

        if v394_results:
            avg_ret = np.mean([r['return'] for r in v394_results])
            win_rate = sum(1 for r in v394_results if r['return'] > 0) / len(v394_results)
            results['V3.94'] = {'avg_return': avg_ret, 'win_rate': win_rate, 'details': v394_results}
            print(f"\nV3.94汇总: 平均收益={avg_ret:+.2%}, 胜率={win_rate:.0%}")
    except Exception as e:
        print(f"V3.94失败: {e}")
        import traceback
        traceback.print_exc()

    # V3.95回测 (使用缓存)
    print("\n" + "="*60)
    print("V3.95 回测 (使用滚动模型)")
    print("="*60)
    try:
        from ml_models.v39.v395_production_scorer import V395ProductionScorer
        scorer_v395 = V395ProductionScorer(model_type='rolling')

        def score_v395(stocks, date):
            preds = scorer_v395.predict_scores(stocks, date)
            return {k: v.get('score', 50) for k, v in preds.items()}

        v395_results = []
        for date in sample_dates:
            print(f"\n{date}:", end='')
            r = backtest_single_date(score_v395, date, db_path)
            if r:
                v395_results.append(r)
                print(f"  收益: {r['return']:+.2%}, Top3: {r['top5'][:3]}")

        if v395_results:
            avg_ret = np.mean([r['return'] for r in v395_results])
            win_rate = sum(1 for r in v395_results if r['return'] > 0) / len(v395_results)
            results['V3.95'] = {'avg_return': avg_ret, 'win_rate': win_rate, 'details': v395_results}
            print(f"\nV3.95汇总: 平均收益={avg_ret:+.2%}, 胜率={win_rate:.0%}")
    except Exception as e:
        print(f"V3.95失败: {e}")
        import traceback
        traceback.print_exc()

    # 对比结果
    print("\n" + "="*70)
    print("采样回测对比结果")
    print("="*70)
    print(f"{'模型':<10} {'平均收益':<12} {'胜率':<10}")
    print("-"*70)
    for model, data in results.items():
        print(f"{model:<10} {data['avg_return']:+.2%}{'':>6} {data['win_rate']:.0%}")


if __name__ == '__main__':
    main()
