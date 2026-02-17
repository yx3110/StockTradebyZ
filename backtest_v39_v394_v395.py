#!/usr/bin/env python3
"""
V3.9 / V3.94 / V3.95 模型对比回测
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from data_adapter.database_manager import DatabaseManager


class SimpleBacktester:
    """简化回测器"""

    def __init__(self,
                 initial_capital: float = 1000000,
                 max_positions: int = 10,
                 commission_rate: float = 0.0003,
                 stamp_tax: float = 0.001):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.commission_rate = commission_rate
        self.stamp_tax = stamp_tax
        self.db_path = Path(__file__).parent / 'data_adapter' / 'stock_data.db'

    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日列表"""
        conn = sqlite3.connect(self.db_path)
        query = f"""
        SELECT DISTINCT trade_date
        FROM daily_quotes
        WHERE trade_date >= '{start_date}'
          AND trade_date <= '{end_date}'
        ORDER BY trade_date
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df['trade_date'].tolist()

    def get_stock_returns(self, stock_codes: List[str], date: str, days: int = 5) -> Dict[str, float]:
        """获取股票未来N天收益率"""
        conn = sqlite3.connect(self.db_path)

        # 获取未来日期
        query = f"""
        SELECT DISTINCT trade_date
        FROM daily_quotes
        WHERE trade_date > '{date}'
        ORDER BY trade_date
        LIMIT {days}
        """
        future_dates = pd.read_sql_query(query, conn)['trade_date'].tolist()

        if len(future_dates) < days:
            conn.close()
            return {}

        target_date = future_dates[-1]

        # 获取当日和目标日的收盘价
        codes_str = ','.join([f"'{c}'" for c in stock_codes])
        query = f"""
        SELECT s.code,
               q1.close as price_start,
               q2.close as price_end
        FROM securities s
        JOIN daily_quotes q1 ON s.id = q1.security_id AND q1.trade_date = '{date}'
        JOIN daily_quotes q2 ON s.id = q2.security_id AND q2.trade_date = '{target_date}'
        WHERE s.code IN ({codes_str})
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        returns = {}
        for _, row in df.iterrows():
            if row['price_start'] > 0:
                ret = (row['price_end'] - row['price_start']) / row['price_start']
                returns[row['code']] = ret

        return returns

    def backtest_model(self, model_name: str, scorer_func,
                       start_date: str, end_date: str,
                       top_n: int = 10,
                       holding_days: int = 5) -> Dict:
        """
        回测单个模型

        Args:
            model_name: 模型名称
            scorer_func: 评分函数 (stock_list, date) -> Dict[code, score]
            start_date: 开始日期
            end_date: 结束日期
            top_n: 每次选择top N只股票
            holding_days: 持仓天数
        """
        print(f"\n{'='*60}")
        print(f"回测模型: {model_name}")
        print(f"回测期间: {start_date} ~ {end_date}")
        print(f"策略: Top{top_n}股票, 持仓{holding_days}天")
        print('='*60)

        trading_dates = self.get_trading_dates(start_date, end_date)
        print(f"交易日数量: {len(trading_dates)}")

        # 获取所有A股股票
        conn = sqlite3.connect(self.db_path)
        query = """
        SELECT code FROM securities
        WHERE type = 'A股'
          AND code NOT LIKE '688%'
          AND code NOT LIKE '8%'
          AND code NOT LIKE '4%'
        """
        all_stocks = pd.read_sql_query(query, conn)['code'].tolist()
        conn.close()
        print(f"股票池: {len(all_stocks)}只")

        # 回测
        results = []
        total_return = 0
        win_count = 0
        trade_count = 0

        # 每隔holding_days进行一次换仓
        for i in range(0, len(trading_dates) - holding_days, holding_days):
            date = trading_dates[i]

            # 获取评分
            try:
                scores = scorer_func(all_stocks, date)
            except Exception as e:
                print(f"  {date}: 评分失败 - {e}")
                continue

            if not scores:
                continue

            # 选择top N
            sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
            selected_codes = [code for code, score in sorted_stocks]

            # 获取未来收益
            returns = self.get_stock_returns(selected_codes, date, holding_days)

            if not returns:
                continue

            # 计算组合收益（等权重）
            valid_returns = [returns[c] for c in selected_codes if c in returns]
            if valid_returns:
                period_return = np.mean(valid_returns)
                total_return += period_return
                trade_count += 1

                if period_return > 0:
                    win_count += 1

                results.append({
                    'date': date,
                    'return': period_return,
                    'selected': selected_codes[:5],  # 只记录前5只
                    'avg_score': np.mean([s for _, s in sorted_stocks])
                })

                print(f"  {date}: 收益={period_return:+.2%}, "
                      f"Top5: {selected_codes[:5]}")

        # 汇总结果
        if trade_count > 0:
            win_rate = win_count / trade_count
            avg_return = total_return / trade_count
            cumulative = (1 + avg_return) ** trade_count - 1
        else:
            win_rate = 0
            avg_return = 0
            cumulative = 0

        summary = {
            'model': model_name,
            'start_date': start_date,
            'end_date': end_date,
            'trade_count': trade_count,
            'win_count': win_count,
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_return': avg_return,
            'cumulative_return': cumulative,
            'details': results
        }

        print(f"\n{model_name} 回测结果:")
        print(f"  交易次数: {trade_count}")
        print(f"  胜率: {win_rate:.1%}")
        print(f"  平均单次收益: {avg_return:.2%}")
        print(f"  累计收益: {cumulative:.2%}")

        return summary


def create_v39_scorer():
    """创建V3.9评分器"""
    from ml_models.v39.v390_production_scorer import V390ProductionScorer
    scorer = V390ProductionScorer()

    def score_func(stock_list, date):
        predictions = scorer.predict_scores(stock_list, date)
        return {code: data.get('score', 50) for code, data in predictions.items()}

    return score_func


def create_v394_scorer():
    """创建V3.94评分器"""
    from ml_models.v39.v394_production_scorer import V394ProductionScorer
    scorer = V394ProductionScorer()

    def score_func(stock_list, date):
        predictions = scorer.predict_scores_with_ranking(stock_list, date)
        return {code: data.get('score', 50) for code, data in predictions.items()}

    return score_func


def create_v395_scorer():
    """创建V3.95评分器"""
    from ml_models.v39.v395_production_scorer import V395ProductionScorer
    scorer = V395ProductionScorer(model_type='rolling')

    def score_func(stock_list, date):
        predictions = scorer.predict_scores(stock_list, date)
        return {code: data.get('score', 50) for code, data in predictions.items()}

    return score_func


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='V3.9/V3.94/V3.95模型对比回测')
    parser.add_argument('--start-date', type=str, default='2025-10-01', help='开始日期')
    parser.add_argument('--end-date', type=str, default='2025-12-20', help='结束日期')
    parser.add_argument('--top-n', type=int, default=10, help='每次选择股票数')
    parser.add_argument('--holding-days', type=int, default=5, help='持仓天数')
    parser.add_argument('--models', type=str, default='v39,v394,v395', help='要回测的模型')

    args = parser.parse_args()

    backtester = SimpleBacktester()

    # 模型评分器
    scorers = {}
    models_to_test = args.models.split(',')

    if 'v39' in models_to_test:
        try:
            scorers['V3.9'] = create_v39_scorer()
            print("✅ V3.9 评分器加载成功")
        except Exception as e:
            print(f"❌ V3.9 评分器加载失败: {e}")

    if 'v394' in models_to_test:
        try:
            scorers['V3.94'] = create_v394_scorer()
            print("✅ V3.94 评分器加载成功")
        except Exception as e:
            print(f"❌ V3.94 评分器加载失败: {e}")

    if 'v395' in models_to_test:
        try:
            scorers['V3.95'] = create_v395_scorer()
            print("✅ V3.95 评分器加载成功")
        except Exception as e:
            print(f"❌ V3.95 评分器加载失败: {e}")

    if not scorers:
        print("没有可用的评分器，退出")
        return

    # 运行回测
    all_results = []
    for model_name, scorer_func in scorers.items():
        try:
            result = backtester.backtest_model(
                model_name=model_name,
                scorer_func=scorer_func,
                start_date=args.start_date,
                end_date=args.end_date,
                top_n=args.top_n,
                holding_days=args.holding_days
            )
            all_results.append(result)
        except Exception as e:
            print(f"❌ {model_name} 回测失败: {e}")
            import traceback
            traceback.print_exc()

    # 对比结果
    print("\n" + "=" * 80)
    print("模型对比结果")
    print("=" * 80)
    print(f"{'模型':<10} {'交易次数':<10} {'胜率':<10} {'平均收益':<12} {'累计收益':<12}")
    print("-" * 80)

    for result in all_results:
        print(f"{result['model']:<10} "
              f"{result['trade_count']:<10} "
              f"{result['win_rate']:.1%}{'':>5} "
              f"{result['avg_return']:+.2%}{'':>6} "
              f"{result['cumulative_return']:+.2%}")

    # 保存结果
    output_dir = Path(__file__).parent / 'reports' / 'backtest'
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f'v39_v394_v395_comparison_{timestamp}.json'

    # 转换为可JSON序列化的格式
    save_results = []
    for r in all_results:
        save_r = r.copy()
        save_r['details'] = save_r['details'][:10]  # 只保存前10条详情
        save_results.append(save_r)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n结果已保存: {output_path}")


if __name__ == '__main__':
    main()
