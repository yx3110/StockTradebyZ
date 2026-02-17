#!/usr/bin/env python3
"""
识别需要重新抓取的有问题日期
"""

import sqlite3
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def identify_problematic_dates():
    """识别price_change_pct数据异常的日期"""
    db_path = "data_adapter/stock_data.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # 查找2025年4月后每个日期的price_change_pct统计
        # 正常的A股市场应该有很多股票涨跌幅超过3%
        cursor.execute("""
            SELECT trade_date,
                   COUNT(*) as total_records,
                   COUNT(CASE WHEN ABS(price_change_pct) > 3 THEN 1 END) as big_changes,
                   COUNT(CASE WHEN ABS(price_change_pct) > 10 THEN 1 END) as huge_changes,
                   MAX(ABS(price_change_pct)) as max_abs_change,
                   AVG(ABS(price_change_pct)) as avg_abs_change
            FROM daily_quotes dq
            JOIN securities s ON s.id = dq.security_id
            WHERE trade_date >= '2025-04-01'
            AND s.type = 'A股'
            AND price_change_pct IS NOT NULL
            GROUP BY trade_date
            ORDER BY trade_date
        """)
        
        results = cursor.fetchall()
        
        problematic_dates = []
        normal_dates = []
        
        logger.info("分析2025年4月以后的数据质量:")
        logger.info("日期\t\t总记录\t>3%变动\t>10%变动\t最大变动\t平均变动")
        logger.info("-" * 80)
        
        for row in results:
            date, total, big, huge, max_change, avg_change = row
            
            # 判断标准：
            # 1. 如果>3%变动的股票少于总数的5%，可能有问题
            # 2. 如果最大变动小于8%，很可能有问题（A股经常有涨停股票）
            # 3. 如果平均绝对变动小于1%，可能有问题
            
            big_pct = (big / total * 100) if total > 0 else 0
            status = "正常"
            
            if big_pct < 5 or max_change < 8 or (avg_change and avg_change < 1):
                status = "异常"
                problematic_dates.append(date)
            else:
                normal_dates.append(date)
            
            logger.info(f"{date}\t{total:,}\t{big}\t{huge}\t{max_change:.2f}%\t{avg_change:.2f}%\t{status}")
        
        logger.info("-" * 80)
        logger.info(f"需要重新抓取的异常日期数: {len(problematic_dates)}")
        logger.info(f"正常日期数: {len(normal_dates)}")
        
        if problematic_dates:
            logger.info("\n需要重新抓取的日期:")
            for date in problematic_dates:
                logger.info(f"  {date}")
        
        return problematic_dates
        
    finally:
        conn.close()

if __name__ == "__main__":
    problematic_dates = identify_problematic_dates()
    
    # 保存到文件供后续使用
    with open('temp_scripts/problematic_dates.txt', 'w') as f:
        for date in problematic_dates:
            f.write(f"{date}\n")
    
    print(f"\n异常日期已保存到 temp_scripts/problematic_dates.txt")