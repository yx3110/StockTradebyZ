#!/usr/bin/env python3
"""
更新数据库中的证券信息，从 securities_list.csv 文件导入行业和地区信息
"""

import pandas as pd
import logging
from pathlib import Path
from data_adapter.database_manager import DatabaseManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_securities_info():
    """从 CSV 文件更新数据库中的证券信息"""
    
    # 读取 CSV 文件
    csv_file = Path("full_securities_data/securities_list.csv")
    if not csv_file.exists():
        logger.error(f"证券列表文件不存在: {csv_file}")
        return False
    
    logger.info(f"读取证券信息文件: {csv_file}")
    df = pd.read_csv(csv_file, dtype={'code': str})
    
    # 初始化数据库管理器
    db_manager = DatabaseManager()
    
    # 检查并添加新列（如果不存在）
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        
        # 检查列是否存在
        cursor.execute("PRAGMA table_info(securities)")
        columns = [col[1] for col in cursor.fetchall()]
        
        # 添加缺失的列
        if 'industry' not in columns:
            logger.info("添加 industry 列...")
            cursor.execute("ALTER TABLE securities ADD COLUMN industry VARCHAR(50)")
            
        if 'area' not in columns:
            logger.info("添加 area 列...")
            cursor.execute("ALTER TABLE securities ADD COLUMN area VARCHAR(20)")
        
        conn.commit()
    
    # 更新证券信息
    updated_count = 0
    not_found_count = 0
    
    logger.info(f"开始更新 {len(df)} 只证券的信息...")
    
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        
        for _, row in df.iterrows():
            code = row['code'].zfill(6)  # 确保6位格式
            
            # 检查证券是否存在于数据库
            cursor.execute("SELECT id FROM securities WHERE code = ?", (code,))
            result = cursor.fetchone()
            
            if result:
                # 更新证券信息
                update_query = """
                UPDATE securities 
                SET industry = ?, area = ?, name = ?, type = ?, exchange = ?, list_date = ?
                WHERE code = ?
                """
                cursor.execute(update_query, (
                    row.get('industry', '未知'),
                    row.get('area', '未知'),
                    row.get('name', ''),
                    row.get('type', ''),
                    row.get('market', ''),
                    row.get('list_date', ''),
                    code
                ))
                updated_count += 1
            else:
                # 如果数据库中没有，插入新记录
                insert_query = """
                INSERT OR IGNORE INTO securities (code, name, type, exchange, industry, area, list_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """
                cursor.execute(insert_query, (
                    code,
                    row.get('name', ''),
                    row.get('type', ''),
                    row.get('market', ''),
                    row.get('industry', '未知'),
                    row.get('area', '未知'),
                    row.get('list_date', '')
                ))
                not_found_count += 1
            
            if updated_count % 1000 == 0:
                conn.commit()
                logger.info(f"已更新 {updated_count} 只证券...")
        
        conn.commit()
    
    logger.info(f"更新完成!")
    logger.info(f"- 更新现有证券: {updated_count} 只")
    logger.info(f"- 新增证券: {not_found_count} 只")
    
    return True


if __name__ == "__main__":
    success = update_securities_info()
    if success:
        print("✅ 证券信息更新成功！现在行业和地区信息应该可以正确显示了。")
    else:
        print("❌ 证券信息更新失败！")