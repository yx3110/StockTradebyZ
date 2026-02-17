#!/usr/bin/env python3
"""
迁移进度监控脚本
"""

import sqlite3
import os
import time
import sys

def monitor_migration():
    """监控迁移进度"""
    db_path = 'data_adapter/stock_data.db'
    csv_total = 7590
    
    print("🚀 数据迁移进度监控中...")
    print("按 Ctrl+C 停止监控\n")
    
    try:
        while True:
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                cursor.execute('SELECT COUNT(*) FROM securities WHERE is_active = 1')
                securities_count = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM daily_quotes')
                quotes_count = cursor.fetchone()[0]
                
                db_size = os.path.getsize(db_path) / 1024 / 1024
                completion = (securities_count / csv_total) * 100
                
                conn.close()
                
                # 清屏并显示当前状态
                os.system('clear')
                print("📊 数据库迁移实时状态")
                print("=" * 50)
                print(f"📈 进度: {completion:.1f}% ({securities_count:,}/{csv_total:,})")
                print(f"💾 数据库大小: {db_size:.2f} MB")
                print(f"📋 行情记录: {quotes_count:,} 条")
                
                # 进度条
                bar_length = 40
                filled_length = int(bar_length * completion / 100)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"🔄 [{bar}] {completion:.1f}%")
                
                # 估算剩余时间（粗略）
                if completion > 0:
                    estimated_total_time = (time.time() - start_time) / (completion / 100)
                    remaining_time = estimated_total_time - (time.time() - start_time)
                    if remaining_time > 0:
                        mins = int(remaining_time / 60)
                        secs = int(remaining_time % 60)
                        print(f"⏱️  预计剩余时间: {mins}分{secs}秒")
                
                print(f"\n⏰ 最后更新: {time.strftime('%H:%M:%S')}")
                print("按 Ctrl+C 停止监控")
                
                # 检查是否完成
                if completion >= 99.5:
                    print("\n🎉 迁移基本完成！")
                    break
                    
            else:
                print("❌ 数据库文件不存在")
                
            time.sleep(10)  # 每10秒更新一次
            
    except KeyboardInterrupt:
        print("\n👋 监控已停止")
    except Exception as e:
        print(f"\n❌ 监控出错: {e}")

if __name__ == "__main__":
    start_time = time.time()
    monitor_migration()