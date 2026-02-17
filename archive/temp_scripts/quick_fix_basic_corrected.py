#!/usr/bin/env python3

import sqlite3
import tushare as ts
import json
import time

# 读取配置
with open('config.json', 'r') as f:
    config = json.load(f)
    ts_token = config['tushare']['token']

# 初始化Tushare
ts.set_token(ts_token)
pro = ts.pro_api()

# 数据库路径
db_path = 'data_adapter/stock_data.db'

print("获取股票基础信息...")
time.sleep(1)
df = pro.stock_basic(
    exchange='',
    list_status='L',
    fields='ts_code,name,area,industry,market,fullname,enname,employees,main_business'
)

print(f"获取到 {len(df)} 只股票")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

saved_count = 0
error_count = 0

for _, row in df.iterrows():
    try:
        ts_code = row['ts_code']
        code = ts_code.split('.')[0]
        
        # 查找security_id
        cursor.execute("SELECT id FROM securities WHERE code = ?", (code,))
        result = cursor.fetchone()
        
        if result:
            security_id = result[0]
            
            # 根据实际表结构插入数据
            cursor.execute("""
                INSERT OR REPLACE INTO stock_basic_info 
                (security_id, ts_code, market, fullname, enname, employees, main_business)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                security_id,
                ts_code,
                row.get('market'),
                row.get('fullname'),
                row.get('enname'),
                row.get('employees'),
                row.get('main_business')
            ))
            saved_count += 1
            
            if saved_count % 1000 == 0:
                print(f"已保存 {saved_count} 条记录...")
                conn.commit()
        else:
            error_count += 1
            
    except Exception as e:
        print(f"处理{ts_code}失败: {e}")
        error_count += 1

conn.commit()
conn.close()

print(f"保存了 {saved_count} 条记录，失败 {error_count} 条")

# 验证
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM stock_basic_info")
count = cursor.fetchone()[0]
print(f"当前stock_basic_info总数: {count}")
conn.close()