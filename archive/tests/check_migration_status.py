#!/usr/bin/env python3
"""
检查数据迁移状态
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from data_adapter.database_manager import DatabaseManager
from data_adapter.data_access import StockDataDAO

def check_migration_status():
    """检查迁移状态"""
    try:
        db = DatabaseManager()
        dao = StockDataDAO(db)
        
        # 获取数据库统计信息
        stats = db.get_database_stats()
        
        print("=== 数据库迁移状态 ===")
        print(f"数据库文件大小: {stats['db_size_mb']:.2f} MB")
        print(f"证券总数: {stats['total_securities']}")
        print(f"行情记录总数: {stats['total_records']:,}")
        
        if 'securities_by_type' in stats:
            print("\n证券类型分布:")
            for sec_type, count in stats['securities_by_type'].items():
                print(f"  {sec_type}: {count}")
        
        if 'date_range' in stats and stats['date_range']['start']:
            print(f"\n数据日期范围: {stats['date_range']['start']} 至 {stats['date_range']['end']}")
        
        # 检查CSV文件总数
        csv_dir = Path("full_securities_data")
        if csv_dir.exists():
            csv_files = list(csv_dir.glob("*.csv"))
            csv_files = [f for f in csv_files if not f.name.startswith('securities_list')]
            print(f"\nCSV文件总数: {len(csv_files)}")
            
            if stats['total_securities'] > 0:
                completion_rate = (stats['total_securities'] / len(csv_files)) * 100
                print(f"迁移完成度: {completion_rate:.1f}%")
            
        print("\n=== 数据库状态检查完成 ===")
        
        # 如果迁移未完成，提供继续迁移的建议
        if stats['total_securities'] < len(csv_files) * 0.9:  # 如果完成度低于90%
            print("\n建议继续完成数据迁移:")
            print("python3 continue_migration.py")
            
    except Exception as e:
        print(f"检查失败: {e}")

if __name__ == "__main__":
    check_migration_status()