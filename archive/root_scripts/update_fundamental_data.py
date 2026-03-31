#!/usr/bin/env python3
"""
基础数据更新脚本
集成到每日更新流程中，维护financial_indicator、index_daily、stock_basic_info等表
"""

import subprocess
import sys
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def update_fundamental_data():
    """更新基础面数据"""
    logger.info("开始基础数据更新流程...")
    
    try:
        # 使用已有的完整数据库更新脚本
        result = subprocess.run([
            sys.executable, 
            'temp_scripts/complete_database_update.py', 
            '--securities'
        ], capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            logger.info("基础数据更新成功")
            logger.info(result.stdout)
        else:
            logger.error(f"基础数据更新失败: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"执行基础数据更新失败: {e}")
        return False
    
    return True

def main():
    """主函数"""
    success = update_fundamental_data()
    
    if success:
        logger.info("基础数据更新流程完成")
        sys.exit(0)
    else:
        logger.error("基础数据更新流程失败")
        sys.exit(1)

if __name__ == "__main__":
    main()