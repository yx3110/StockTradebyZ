#!/usr/bin/env python3
"""
数据库数据质量检查工具
"""

import pandas as pd
import logging
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_adapter.database_manager import DatabaseManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# 初始化数据库管理器
db_manager = DatabaseManager("data_adapter/stock_data.db")

def check_data_completeness(days_back: int = 30):
    """检查数据完整性"""
    logger.info(f"检查最近{days_back}天的数据完整性...")
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    with db_manager.get_connection() as conn:
        # 检查每个证券的数据完整性
        query = """
        SELECT 
            s.code,
            s.name,
            s.type,
            COUNT(DISTINCT q.trade_date) as data_days,
            MIN(q.trade_date) as first_date,
            MAX(q.trade_date) as last_date
        FROM securities s
        LEFT JOIN daily_quotes q ON s.id = q.security_id
            AND q.trade_date >= ?
            AND q.trade_date <= ?
        WHERE s.is_active = 1
        GROUP BY s.id, s.code, s.name, s.type
        HAVING data_days < ?
        ORDER BY s.type, data_days
        """
        
        # 假设最少应该有20个交易日的数据
        min_trading_days = 20
        
        df = pd.read_sql_query(query, conn, params=(start_date, end_date, min_trading_days))
        
        if not df.empty:
            logger.warning(f"发现 {len(df)} 只证券数据不完整")
            for _, row in df.head(10).iterrows():
                logger.warning(f"{row['code']} {row['name']} ({row['type']}): "
                             f"只有 {row['data_days']} 天数据, "
                             f"最新: {row['last_date']}")
        else:
            logger.info("所有活跃证券数据完整")
        
        return df

def check_data_gaps():
    """检查数据间隙"""
    logger.info("检查数据间隙...")
    
    with db_manager.get_connection() as conn:
        # 查找有数据间隙的证券
        query = """
        WITH date_diffs AS (
            SELECT 
                s.code,
                s.name,
                q.trade_date,
                LAG(q.trade_date) OVER (PARTITION BY s.id ORDER BY q.trade_date) as prev_date,
                julianday(q.trade_date) - julianday(LAG(q.trade_date) OVER (PARTITION BY s.id ORDER BY q.trade_date)) as day_diff
            FROM securities s
            JOIN daily_quotes q ON s.id = q.security_id
            WHERE s.is_active = 1
                AND q.trade_date >= date('now', '-60 days')
        )
        SELECT 
            code,
            name,
            prev_date,
            trade_date,
            day_diff
        FROM date_diffs
        WHERE day_diff > 5  -- 超过5天的间隙（考虑周末和假期）
        ORDER BY code, trade_date
        LIMIT 20
        """
        
        df = pd.read_sql_query(query, conn)
        
        if not df.empty:
            logger.warning(f"发现 {len(df)} 处数据间隙")
            for _, row in df.iterrows():
                logger.warning(f"{row['code']} {row['name']}: "
                             f"{row['prev_date']} 到 {row['trade_date']} "
                             f"间隙 {int(row['day_diff'])} 天")
        else:
            logger.info("未发现异常数据间隙")
        
        return df

def check_data_anomalies():
    """检查数据异常"""
    logger.info("检查数据异常...")
    
    anomalies = []
    
    with db_manager.get_connection() as conn:
        # 检查价格异常（开高低收关系）
        query1 = """
        SELECT 
            s.code,
            s.name,
            q.trade_date,
            q.open,
            q.high,
            q.low,
            q.close
        FROM securities s
        JOIN daily_quotes q ON s.id = q.security_id
        WHERE s.is_active = 1
            AND (
                q.high < q.low OR
                q.high < q.open OR
                q.high < q.close OR
                q.low > q.open OR
                q.low > q.close OR
                q.open <= 0 OR
                q.close <= 0
            )
        ORDER BY q.trade_date DESC
        LIMIT 20
        """
        
        df1 = pd.read_sql_query(query1, conn)
        if not df1.empty:
            logger.warning(f"发现 {len(df1)} 条价格异常记录")
            anomalies.extend(df1.to_dict('records'))
        
        # 检查成交量异常
        query2 = """
        SELECT 
            s.code,
            s.name,
            q.trade_date,
            q.volume,
            q.close
        FROM securities s
        JOIN daily_quotes q ON s.id = q.security_id
        WHERE s.is_active = 1
            AND q.volume < 0
        ORDER BY q.trade_date DESC
        LIMIT 20
        """
        
        df2 = pd.read_sql_query(query2, conn)
        if not df2.empty:
            logger.warning(f"发现 {len(df2)} 条成交量异常记录")
            anomalies.extend(df2.to_dict('records'))
        
        # 检查涨跌幅异常
        query3 = """
        SELECT 
            s.code,
            s.name,
            q.trade_date,
            q.price_change_pct
        FROM securities s
        JOIN daily_quotes q ON s.id = q.security_id
        WHERE s.is_active = 1
            AND (q.price_change_pct > 0.20 OR q.price_change_pct < -0.20)  -- 超过20%涨跌幅
            AND s.type = 'A股'  -- A股有涨跌停限制
        ORDER BY q.trade_date DESC
        LIMIT 20
        """
        
        df3 = pd.read_sql_query(query3, conn)
        if not df3.empty:
            logger.warning(f"发现 {len(df3)} 条涨跌幅异常记录")
            anomalies.extend(df3.to_dict('records'))
    
    return anomalies

def generate_data_report():
    """生成数据质量报告"""
    logger.info("生成数据质量报告...")
    
    stats = db_manager.get_database_stats()
    
    report = f"""
数据库数据质量报告
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

1. 数据库概况:
   - 证券总数: {stats['total_securities']}
   - A股数量: {stats['securities_by_type'].get('A股', 0)}
   - ETF/基金数量: {stats['securities_by_type'].get('ETF_基金', 0)}
   - 数据记录总数: {stats['total_quotes']:,}
   - 数据日期范围: {stats['date_range']['start']} 至 {stats['date_range']['end']}
   - 数据库大小: {stats['db_size_mb']:.2f} MB

2. 数据完整性检查:
"""
    
    # 检查数据完整性
    incomplete_df = check_data_completeness()
    if not incomplete_df.empty:
        report += f"   - 发现 {len(incomplete_df)} 只证券数据不完整\n"
        report += "   - 前10只不完整证券:\n"
        for _, row in incomplete_df.head(10).iterrows():
            report += f"     * {row['code']} {row['name']}: {row['data_days']} 天数据\n"
    else:
        report += "   - 所有证券数据完整\n"
    
    report += "\n3. 数据间隙检查:\n"
    gaps_df = check_data_gaps()
    if not gaps_df.empty:
        report += f"   - 发现 {len(gaps_df)} 处数据间隙\n"
    else:
        report += "   - 未发现异常数据间隙\n"
    
    report += "\n4. 数据异常检查:\n"
    anomalies = check_data_anomalies()
    if anomalies:
        report += f"   - 发现 {len(anomalies)} 条异常记录\n"
    else:
        report += "   - 未发现数据异常\n"
    
    report += f"\n{'='*60}\n"
    
    # 保存报告
    report_file = f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"数据质量报告已保存至: {report_file}")
    print(report)
    
    return report

if __name__ == "__main__":
    generate_data_report()