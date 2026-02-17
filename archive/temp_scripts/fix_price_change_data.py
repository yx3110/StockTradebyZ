#!/usr/bin/env python3
"""
修复数据库中错误的price_change_pct数据
将所有被错误除以100的数据恢复正常
"""

import sqlite3
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_price_change_data():
    """修复price_change_pct数据"""
    db_path = "data_adapter/stock_data.db"
    
    if not Path(db_path).exists():
        logger.error("数据库文件不存在")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # 先检查数据范围
        cursor.execute("""
            SELECT COUNT(*) as total,
                   MIN(price_change_pct) as min_pct,
                   MAX(price_change_pct) as max_pct,
                   AVG(price_change_pct) as avg_pct
            FROM daily_quotes 
            WHERE price_change_pct IS NOT NULL
        """)
        
        stats = cursor.fetchone()
        logger.info(f"修复前统计:")
        logger.info(f"  总记录数: {stats[0]:,}")
        logger.info(f"  最小涨跌幅: {stats[1]:.6f}")
        logger.info(f"  最大涨跌幅: {stats[2]:.6f}")
        logger.info(f"  平均涨跌幅: {stats[3]:.6f}")
        
        # 检查多少数据需要修复（绝对值小于1的，很可能是被错误除以100的）
        cursor.execute("""
            SELECT COUNT(*) FROM daily_quotes 
            WHERE price_change_pct IS NOT NULL 
            AND ABS(price_change_pct) < 1
            AND ABS(price_change_pct) > 0.001
        """)
        
        need_fix = cursor.fetchone()[0]
        logger.info(f"需要修复的记录数: {need_fix:,}")
        
        if need_fix == 0:
            logger.info("没有需要修复的数据")
            return
        
        # 自动确认修复
        logger.info(f"将修复 {need_fix:,} 条记录")
        
        # 开始修复：将小于1的price_change_pct乘以100
        # 但要排除真正的小幅变动（比如绝对值小于0.001的）
        logger.info("开始修复数据...")
        
        cursor.execute("""
            UPDATE daily_quotes 
            SET price_change_pct = price_change_pct * 100
            WHERE price_change_pct IS NOT NULL 
            AND ABS(price_change_pct) < 1
            AND ABS(price_change_pct) > 0.001
        """)
        
        modified = cursor.rowcount
        logger.info(f"已修复 {modified:,} 条记录")
        
        # 再次检查数据范围
        cursor.execute("""
            SELECT COUNT(*) as total,
                   MIN(price_change_pct) as min_pct,
                   MAX(price_change_pct) as max_pct,
                   AVG(price_change_pct) as avg_pct
            FROM daily_quotes 
            WHERE price_change_pct IS NOT NULL
        """)
        
        stats = cursor.fetchone()
        logger.info(f"修复后统计:")
        logger.info(f"  总记录数: {stats[0]:,}")
        logger.info(f"  最小涨跌幅: {stats[1]:.2f}%")
        logger.info(f"  最大涨跌幅: {stats[2]:.2f}%")
        logger.info(f"  平均涨跌幅: {stats[3]:.2f}%")
        
        # 提交更改
        conn.commit()
        logger.info("✅ 数据修复完成并已保存")
        
        # 显示一些样本数据验证
        logger.info("\n验证修复效果 - 查看5月21日涨幅前10:")
        cursor.execute("""
            SELECT s.code, s.name, dq.price_change_pct 
            FROM daily_quotes dq 
            JOIN securities s ON s.id = dq.security_id 
            WHERE dq.trade_date = '2025-05-21' 
            AND dq.price_change_pct > 5 
            ORDER BY dq.price_change_pct DESC 
            LIMIT 10
        """)
        
        samples = cursor.fetchall()
        for code, name, pct in samples:
            logger.info(f"  {code} {name}: {pct:.2f}%")
        
    except Exception as e:
        logger.error(f"修复过程出错: {e}")
        conn.rollback()
    
    finally:
        conn.close()

if __name__ == "__main__":
    fix_price_change_data()