#!/usr/bin/env python3
"""
清理非活跃证券
"""

import pandas as pd
import logging
import json
import sys
import os
from datetime import datetime, timedelta
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

def identify_inactive_securities():
    """识别非活跃证券"""
    logger.info("识别非活跃证券...")
    
    with db_manager.get_connection() as conn:
        # 查找长期无数据的证券（6个月以上无数据或总数据少于30天）
        query = """
        SELECT 
            s.id,
            s.code,
            s.name,
            s.type,
            s.exchange,
            COUNT(q.trade_date) as total_data_days,
            MIN(q.trade_date) as first_date,
            MAX(q.trade_date) as last_date,
            julianday('now') - julianday(MAX(q.trade_date)) as days_since_last_data
        FROM securities s
        LEFT JOIN daily_quotes q ON s.id = q.security_id
        WHERE s.is_active = 1
        GROUP BY s.id, s.code, s.name, s.type, s.exchange
        HAVING 
            total_data_days = 0 OR  -- 完全无数据
            (days_since_last_data > 180 AND total_data_days < 100) OR  -- 6个月无数据且历史数据少
            (total_data_days < 30 AND days_since_last_data > 30)  -- 数据太少且最近无数据
        ORDER BY total_data_days, days_since_last_data DESC
        """
        
        df = pd.read_sql_query(query, conn)
    
    logger.info(f"发现 {len(df)} 只可能非活跃的证券")
    return df

def analyze_delisted_securities():
    """分析退市证券"""
    logger.info("分析可能的退市证券...")
    
    with db_manager.get_connection() as conn:
        # 查找长期无数据且符合退市特征的证券
        query = """
        SELECT 
            s.code,
            s.name,
            s.type,
            COUNT(q.trade_date) as total_data_days,
            MAX(q.trade_date) as last_date,
            julianday('now') - julianday(MAX(q.trade_date)) as days_since_last_data
        FROM securities s
        LEFT JOIN daily_quotes q ON s.id = q.security_id
        WHERE s.is_active = 1
        GROUP BY s.id, s.code, s.name, s.type
        HAVING 
            total_data_days = 0 OR  -- 完全无数据
            days_since_last_data > 365  -- 一年以上无数据
        ORDER BY days_since_last_data DESC NULLS LAST
        """
        
        df = pd.read_sql_query(query, conn)
    
    return df

def mark_securities_inactive(security_ids: list, reason: str = "长期无数据"):
    """标记证券为非活跃"""
    if not security_ids:
        return 0
    
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        
        # 更新为非活跃状态
        placeholders = ','.join(['?' for _ in security_ids])
        query = f"""
        UPDATE securities 
        SET is_active = 0, updated_at = CURRENT_TIMESTAMP 
        WHERE id IN ({placeholders})
        """
        
        cursor.execute(query, security_ids)
        updated_count = cursor.rowcount
        conn.commit()
        
        logger.info(f"标记 {updated_count} 只证券为非活跃状态，原因：{reason}")
        return updated_count

def cleanup_inactive_securities(dry_run: bool = True):
    """清理非活跃证券"""
    logger.info(f"开始清理非活跃证券 (干运行: {dry_run})...")
    
    # 1. 识别非活跃证券
    inactive_securities = identify_inactive_securities()
    
    if inactive_securities.empty:
        logger.info("未发现需要清理的非活跃证券")
        return
    
    # 2. 分类处理
    total_updated = 0
    
    # 完全无数据的证券
    no_data_securities = inactive_securities[
        inactive_securities['total_data_days'] == 0
    ]
    
    # 长期无数据的证券（6个月以上）
    long_inactive_securities = inactive_securities[
        (inactive_securities['total_data_days'] > 0) & 
        (inactive_securities['days_since_last_data'] > 180)
    ]
    
    # 数据太少的新证券（可能是无效代码）
    insufficient_data_securities = inactive_securities[
        (inactive_securities['total_data_days'] < 30) & 
        (inactive_securities['days_since_last_data'] > 30) &
        (inactive_securities['total_data_days'] > 0)
    ]
    
    # 报告分析结果
    logger.info("="*60)
    logger.info("非活跃证券分析结果:")
    logger.info(f"完全无数据证券: {len(no_data_securities)} 只")
    logger.info(f"长期无数据证券: {len(long_inactive_securities)} 只")
    logger.info(f"数据不足证券: {len(insufficient_data_securities)} 只")
    logger.info("="*60)
    
    # 显示样本
    if not no_data_securities.empty:
        logger.info("完全无数据证券样本:")
        for _, row in no_data_securities.head(10).iterrows():
            logger.info(f"  {row['code']} {row['name']} ({row['type']})")
    
    if not long_inactive_securities.empty:
        logger.info("长期无数据证券样本:")
        for _, row in long_inactive_securities.head(5).iterrows():
            logger.info(f"  {row['code']} {row['name']} - 最后数据: {row['last_date']} ({int(row['days_since_last_data'])}天前)")
    
    if not dry_run:
        # 实际执行清理
        logger.info("开始标记非活跃证券...")
        
        if not no_data_securities.empty:
            count = mark_securities_inactive(
                no_data_securities['id'].tolist(), 
                "完全无数据"
            )
            total_updated += count
        
        if not long_inactive_securities.empty:
            count = mark_securities_inactive(
                long_inactive_securities['id'].tolist(), 
                "长期无数据(6个月+)"
            )
            total_updated += count
        
        if not insufficient_data_securities.empty:
            count = mark_securities_inactive(
                insufficient_data_securities['id'].tolist(), 
                "数据不足且无近期数据"
            )
            total_updated += count
        
        logger.info(f"总计标记 {total_updated} 只证券为非活跃状态")
    else:
        logger.info("这是干运行，未实际更新数据库")
        logger.info(f"如需实际执行，请运行: python3 {__file__} --execute")

def restore_recently_active_securities():
    """恢复最近重新活跃的证券"""
    logger.info("检查最近重新活跃的证券...")
    
    with db_manager.get_connection() as conn:
        # 查找被标记为非活跃但最近有数据的证券
        query = """
        SELECT DISTINCT s.id, s.code, s.name, s.type
        FROM securities s
        JOIN daily_quotes q ON s.id = q.security_id
        WHERE s.is_active = 0 
            AND q.trade_date >= date('now', '-30 days')
        """
        
        cursor = conn.cursor()
        cursor.execute(query)
        reactivate_securities = cursor.fetchall()
        
        if reactivate_securities:
            security_ids = [row['id'] for row in reactivate_securities]
            
            # 重新激活这些证券
            placeholders = ','.join(['?' for _ in security_ids])
            update_query = f"""
            UPDATE securities 
            SET is_active = 1, updated_at = CURRENT_TIMESTAMP 
            WHERE id IN ({placeholders})
            """
            
            cursor.execute(update_query, security_ids)
            updated_count = cursor.rowcount
            conn.commit()
            
            logger.info(f"重新激活 {updated_count} 只证券")
            for row in reactivate_securities:
                logger.info(f"  {row['code']} {row['name']} ({row['type']})")
        else:
            logger.info("未发现需要重新激活的证券")

def generate_cleanup_report():
    """生成清理报告"""
    logger.info("生成清理报告...")
    
    with db_manager.get_connection() as conn:
        # 统计活跃和非活跃证券
        stats_query = """
        SELECT 
            type,
            is_active,
            COUNT(*) as count
        FROM securities
        GROUP BY type, is_active
        ORDER BY type, is_active
        """
        
        stats_df = pd.read_sql_query(stats_query, conn)
    
    # 生成报告
    report = f"""
证券活跃状态报告
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

"""
    
    for security_type in stats_df['type'].unique():
        type_stats = stats_df[stats_df['type'] == security_type]
        active_count = type_stats[type_stats['is_active'] == 1]['count'].sum()
        inactive_count = type_stats[type_stats['is_active'] == 0]['count'].sum()
        total_count = active_count + inactive_count
        
        report += f"{security_type}:\n"
        report += f"  活跃: {active_count} 只 ({active_count/total_count*100:.1f}%)\n"
        report += f"  非活跃: {inactive_count} 只 ({inactive_count/total_count*100:.1f}%)\n"
        report += f"  总计: {total_count} 只\n\n"
    
    report += f"{'='*60}\n"
    
    # 保存报告
    report_file = f"cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"清理报告已保存至: {report_file}")
    print(report)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="清理非活跃证券")
    parser.add_argument("--execute", action="store_true", help="实际执行清理（默认为干运行）")
    parser.add_argument("--restore", action="store_true", help="恢复最近重新活跃的证券")
    
    args = parser.parse_args()
    
    logger.info("开始证券清理任务...")
    
    # 恢复最近重新活跃的证券
    if args.restore:
        restore_recently_active_securities()
    
    # 清理非活跃证券
    cleanup_inactive_securities(dry_run=not args.execute)
    
    # 生成报告
    generate_cleanup_report()
    
    logger.info("证券清理任务完成!")

if __name__ == "__main__":
    main()