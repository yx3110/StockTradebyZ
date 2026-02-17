#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.8数据变化调试

调试为什么连续交易日的评分完全相同
检查底层数据是否真的有变化
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目路径
sys.path.append('/Users/yangxu/StockTradebyZ')

from data_adapter.database_manager import DatabaseManager

def debug_data_changes():
    """调试数据变化"""
    print("🔍 V3.8数据变化调试")
    print("="*50)

    db_manager = DatabaseManager("data_adapter/stock_data.db")

    # 测试股票
    test_stock = "000001"
    test_dates = ["2025-09-12", "2025-09-13", "2025-09-16"]

    with db_manager.get_connection() as conn:
        # 查询技术指标数据
        sql = """
            SELECT
                dq.trade_date,
                dq.close, dq.high, dq.low,
                dq.ma5, dq.ma10, dq.ma20,
                ti.rsi6, ti.rsi12, ti.kdj_k, ti.kdj_d,
                ti.macd_dif, ti.boll_upper, ti.boll_lower
            FROM daily_quotes dq
            LEFT JOIN technical_indicators ti ON dq.security_id = ti.security_id AND dq.trade_date = ti.trade_date
            JOIN securities s ON dq.security_id = s.id
            WHERE s.code = ?
                AND dq.trade_date >= '2025-09-10'
                AND dq.trade_date <= '2025-09-16'
            ORDER BY dq.trade_date ASC
        """

        data = pd.read_sql(sql, conn, params=[test_stock])

        print(f"📊 {test_stock} 技术指标数据 (最近7天):")
        print("-" * 100)

        if not data.empty:
            # 显示关键指标
            columns_to_show = ['trade_date', 'close', 'rsi6', 'kdj_k', 'macd_dif']
            display_data = data[columns_to_show].copy()

            for col in columns_to_show:
                if col != 'trade_date':
                    display_data[col] = display_data[col].round(4)

            print(display_data.to_string(index=False))

            # 分析连续日期间的变化
            print(f"\n📈 连续日期间的变化分析:")
            print("-" * 60)

            key_indicators = ['close', 'rsi6', 'kdj_k', 'macd_dif']

            for i in range(1, len(data)):
                current_date = data.iloc[i]['trade_date']
                prev_date = data.iloc[i-1]['trade_date']

                print(f"\n{prev_date} → {current_date}:")

                has_change = False
                for indicator in key_indicators:
                    current_val = data.iloc[i][indicator]
                    prev_val = data.iloc[i-1][indicator]

                    if pd.isna(current_val) or pd.isna(prev_val):
                        change_pct = "N/A"
                    elif prev_val != 0:
                        change_pct = f"{((current_val - prev_val) / prev_val * 100):+.2f}%"
                        if abs((current_val - prev_val) / prev_val) > 0.001:  # 0.1%以上变化
                            has_change = True
                    else:
                        change_pct = "N/A"

                    print(f"  {indicator}: {prev_val:.4f} → {current_val:.4f} ({change_pct})")

                if has_change:
                    print(f"  ✅ 有显著数据变化")
                else:
                    print(f"  ⚠️ 数据变化很小")

            # 检查数据窗口的影响
            print(f"\n🪟 数据窗口分析 (120天窗口):")
            print("-" * 60)

            for target_date in test_dates:
                # 模拟V3.8的数据窗口获取
                window_sql = """
                    SELECT
                        COUNT(*) as data_points,
                        AVG(ti.rsi6) as avg_rsi,
                        STDDEV(ti.rsi6) as std_rsi,
                        MIN(dq.trade_date) as start_date,
                        MAX(dq.trade_date) as end_date
                    FROM daily_quotes dq
                    LEFT JOIN technical_indicators ti ON dq.security_id = ti.security_id AND dq.trade_date = ti.trade_date
                    JOIN securities s ON dq.security_id = s.id
                    WHERE s.code = ?
                        AND dq.trade_date <= ?
                    ORDER BY dq.trade_date DESC
                    LIMIT 120
                """

                window_data = pd.read_sql(window_sql, conn, params=[test_stock, target_date])

                if not window_data.empty:
                    row = window_data.iloc[0]
                    print(f"  {target_date}: {row['data_points']}天 ({row['start_date']} to {row['end_date']})")
                    print(f"    平均RSI: {row['avg_rsi']:.4f}, 标准差: {row['std_rsi']:.4f}")

        else:
            print(f"❌ 未找到 {test_stock} 的数据")

if __name__ == "__main__":
    debug_data_changes()