#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.8预测准确性回测验证

验证V3.8评分是否能有效预测未来股票收益
这是最关键的性能验证测试

Created: 2025-09-16
Author: Claude Code
Version: 3.8.0
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# 添加项目路径
sys.path.append('/Users/yangxu/StockTradebyZ')

from data_adapter.database_manager import DatabaseManager

def test_predictive_accuracy():
    """V3.8预测准确性回测"""
    print("🎯 V3.8预测准确性回测验证")
    print("="*60)

    # 配置日志
    logging.basicConfig(level=logging.WARNING)

    try:
        from adaptive_scoring.v38_selector_adapter import V38SelectorAdapter

        v38_adapter = V38SelectorAdapter()
        db_manager = DatabaseManager("data_adapter/stock_data.db")

        # 回测配置
        test_stocks = ["000001", "000002", "600036", "600000", "000858", "002415", "300059", "002142"]
        eval_dates = ["2025-09-10", "2025-09-11", "2025-09-12"]  # 评估日期
        prediction_periods = [1, 3, 5]  # 预测1天、3天、5天后收益

        print(f"📊 回测配置:")
        print(f"  测试股票: {len(test_stocks)}只")
        print(f"  评估日期: {len(eval_dates)}个")
        print(f"  预测期间: {prediction_periods}天后收益")

        all_predictions = []

        # 第一步：收集评分和实际收益
        for eval_date in eval_dates:
            print(f"\n📅 处理评估日期: {eval_date}")

            # 获取V3.8评分
            results = v38_adapter.evaluate_stocks(test_stocks, eval_date)

            if 'stocks' not in results:
                print(f"    ⚠️ 未获取到评分数据")
                continue

            # 获取后续收益数据
            with db_manager.get_connection() as conn:
                eval_datetime = datetime.strptime(eval_date, '%Y-%m-%d')

                for period in prediction_periods:
                    future_date = eval_datetime + timedelta(days=period + 2)  # +2因为可能包含周末
                    future_date_str = future_date.strftime('%Y-%m-%d')

                    for stock_data in results['stocks']:
                        code = stock_data.get('code')
                        score = stock_data.get('final_score', 0)

                        # 获取评估日和预测日的收盘价
                        sql = """
                            SELECT
                                s.code,
                                dq1.close as eval_close,
                                dq2.close as future_close,
                                dq1.trade_date as eval_date,
                                dq2.trade_date as future_date
                            FROM securities s
                            LEFT JOIN daily_quotes dq1 ON s.id = dq1.security_id AND dq1.trade_date <= ?
                            LEFT JOIN daily_quotes dq2 ON s.id = dq2.security_id AND dq2.trade_date <= ?
                            WHERE s.code = ?
                            ORDER BY dq1.trade_date DESC, dq2.trade_date DESC
                            LIMIT 1
                        """

                        result = pd.read_sql(sql, conn, params=[eval_date, future_date_str, code])

                        if not result.empty and result.iloc[0]['eval_close'] and result.iloc[0]['future_close']:
                            eval_close = result.iloc[0]['eval_close']
                            future_close = result.iloc[0]['future_close']

                            # 计算实际收益率
                            actual_return = (future_close - eval_close) / eval_close

                            all_predictions.append({
                                'eval_date': eval_date,
                                'code': code,
                                'prediction_days': period,
                                'v38_score': score,
                                'eval_close': eval_close,
                                'future_close': future_close,
                                'actual_return': actual_return
                            })

                            print(f"    {code}: 评分={score:.3f}, {period}天收益={actual_return:.3%}")

        # 第二步：分析预测准确性
        if all_predictions:
            df = pd.DataFrame(all_predictions)

            print(f"\n📊 预测准确性分析:")
            print(f"  总预测样本: {len(df)}个")

            for period in prediction_periods:
                period_data = df[df['prediction_days'] == period]
                if len(period_data) == 0:
                    continue

                # 计算相关性
                correlation = period_data['v38_score'].corr(period_data['actual_return'])

                # 计算分档收益分析
                period_data_sorted = period_data.sort_values('v38_score')
                n_quantiles = 3
                quantile_size = len(period_data_sorted) // n_quantiles

                quantile_returns = []
                for i in range(n_quantiles):
                    start_idx = i * quantile_size
                    end_idx = (i + 1) * quantile_size if i < n_quantiles - 1 else len(period_data_sorted)
                    quantile_data = period_data_sorted.iloc[start_idx:end_idx]
                    avg_return = quantile_data['actual_return'].mean()
                    quantile_returns.append(avg_return)

                print(f"\n  📈 {period}天预测分析:")
                print(f"    相关系数: {correlation:.4f}")
                print(f"    分档收益分析:")
                for i, ret in enumerate(quantile_returns):
                    level = ['低分档', '中分档', '高分档'][i]
                    print(f"      {level}: {ret:.3%}")

                # 多空效果
                if len(quantile_returns) >= 2:
                    long_short = quantile_returns[-1] - quantile_returns[0]
                    print(f"    多空效果: {long_short:.3%}")

            # 综合评估
            overall_corr = df.groupby('prediction_days')['v38_score'].corr(df.groupby('prediction_days')['actual_return']).mean()

            print(f"\n🎯 综合评估:")
            print(f"  平均相关系数: {overall_corr:.4f}")

            if abs(overall_corr) >= 0.15:
                print(f"  ✅ 预测效果优秀 (|相关系数| ≥ 0.15)")
                return True
            elif abs(overall_corr) >= 0.08:
                print(f"  ⚡ 预测效果良好 (|相关系数| ≥ 0.08)")
                return True
            else:
                print(f"  ❌ 预测效果需要改进 (|相关系数| < 0.08)")
                return False
        else:
            print(f"\n❌ 未获取到足够的预测数据")
            return False

    except Exception as e:
        print(f"\n💥 回测异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_predictive_accuracy()
    if success:
        print(f"\n🎉 V3.8预测准确性验证通过！")
    else:
        print(f"\n🔧 V3.8预测准确性需要改进")