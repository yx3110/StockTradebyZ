#!/usr/bin/env python3
"""
数据库迁移脚本 - 升级到v3.2版本
为technical_indicators表添加挤压动量相关字段
"""

import sqlite3
import logging
import sys
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@contextmanager
def get_db_connection(db_path: str):
    """获取数据库连接"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def check_column_exists(conn, table_name: str, column_name: str) -> bool:
    """检查列是否存在"""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return column_name in columns

def add_squeeze_momentum_columns(db_path: str):
    """添加挤压动量相关字段到technical_indicators表"""
    
    logger.info("🔄 开始数据库迁移到v3.2版本...")
    
    # 定义要添加的列
    new_columns = [
        # 肯特纳通道
        ("kc_upper", "DECIMAL(10,3)"),
        ("kc_middle", "DECIMAL(10,3)"),
        ("kc_lower", "DECIMAL(10,3)"),
        ("kc_width", "DECIMAL(10,3)"),
        
        # 挤压状态指标
        ("squeeze_state", "BOOLEAN DEFAULT 0"),
        ("squeeze_release", "BOOLEAN DEFAULT 0"),
        ("squeeze_intensity", "DECIMAL(10,4)"),
        ("squeeze_days", "INTEGER DEFAULT 0"),
        ("recent_releases", "INTEGER DEFAULT 0"),
        
        # 动量相关指标
        ("squeeze_momentum", "DECIMAL(10,4)"),
        ("momentum_direction", "INTEGER DEFAULT 0"),
        ("momentum_strength", "DECIMAL(10,4)"),
        ("momentum_acceleration", "DECIMAL(10,4)"),
        ("momentum_consistency", "DECIMAL(10,4)")
    ]
    
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # 检查表是否存在
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='technical_indicators'
        """)
        if not cursor.fetchone():
            logger.error("❌ technical_indicators表不存在！")
            return False
        
        # 逐个添加字段
        added_columns = 0
        skipped_columns = 0
        
        for column_name, column_type in new_columns:
            try:
                # 检查列是否已存在
                if check_column_exists(conn, 'technical_indicators', column_name):
                    logger.info(f"⏭️  字段 {column_name} 已存在，跳过")
                    skipped_columns += 1
                    continue
                
                # 添加字段
                sql = f"ALTER TABLE technical_indicators ADD COLUMN {column_name} {column_type}"
                cursor.execute(sql)
                logger.info(f"✅ 已添加字段: {column_name} {column_type}")
                added_columns += 1
                
            except sqlite3.Error as e:
                logger.error(f"❌ 添加字段 {column_name} 失败: {e}")
                return False
        
        # 提交事务
        conn.commit()
        
        logger.info(f"🎉 数据库迁移完成！")
        logger.info(f"📊 新增字段: {added_columns} 个")
        logger.info(f"⏭️  跳过字段: {skipped_columns} 个")
        logger.info(f"📅 升级时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 记录迁移版本
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version VARCHAR(10) PRIMARY KEY,
                    upgraded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            """)
            
            cursor.execute("""
                INSERT OR REPLACE INTO schema_version (version, description)
                VALUES ('v3.2', '添加挤压动量指标支持')
            """)
            
            conn.commit()
            logger.info("📝 已记录迁移版本信息")
            
        except sqlite3.Error as e:
            logger.warning(f"⚠️ 记录版本信息失败: {e}")
    
    return True

def verify_migration(db_path: str):
    """验证迁移结果"""
    logger.info("🔍 验证迁移结果...")
    
    required_columns = [
        'kc_upper', 'kc_middle', 'kc_lower', 'kc_width',
        'squeeze_state', 'squeeze_release', 'squeeze_intensity', 
        'squeeze_days', 'recent_releases',
        'squeeze_momentum', 'momentum_direction', 'momentum_strength',
        'momentum_acceleration', 'momentum_consistency'
    ]
    
    with get_db_connection(db_path) as conn:
        missing_columns = []
        
        for column in required_columns:
            if not check_column_exists(conn, 'technical_indicators', column):
                missing_columns.append(column)
        
        if missing_columns:
            logger.error(f"❌ 验证失败！缺少字段: {missing_columns}")
            return False
        
        # 检查数据表状态
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM technical_indicators")
        record_count = cursor.fetchone()[0]
        
        logger.info(f"✅ 验证通过！")
        logger.info(f"📊 technical_indicators表共有 {record_count:,} 条记录")
        logger.info(f"🆕 新增 {len(required_columns)} 个挤压动量相关字段")
    
    return True

def main():
    """主函数"""
    db_path = "data_adapter/stock_data.db"
    
    # 检查数据库文件是否存在
    if not Path(db_path).exists():
        logger.error(f"❌ 数据库文件不存在: {db_path}")
        sys.exit(1)
    
    # 备份提醒
    logger.info("⚠️  建议在执行迁移前备份数据库文件")
    
    # 执行迁移
    success = add_squeeze_momentum_columns(db_path)
    if not success:
        logger.error("❌ 数据库迁移失败！")
        sys.exit(1)
    
    # 验证迁移
    if not verify_migration(db_path):
        logger.error("❌ 迁移验证失败！")
        sys.exit(1)
    
    logger.info("🚀 数据库已成功升级到v3.2版本！")
    logger.info("💡 现在可以开始计算和使用挤压动量指标了")

if __name__ == "__main__":
    main()