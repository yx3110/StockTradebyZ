#!/usr/bin/env python3

import sqlite3
import tushare as ts
import json

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
df = pro.stock_basic(
    exchange='',
    list_status='L',
    fields='ts_code,name,area,industry,market,list_date,main_business,employees'
)

print(f"获取到 {len(df)} 只股票")

# 测试前几只股票
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

saved_count = 0
for i, (_, row) in enumerate(df.head(10).iterrows()):
    ts_code = row['ts_code']
    code = ts_code.split('.')[0]
    
    print(f"处理股票: {code} ({ts_code})")
    
    # 查找security_id
    cursor.execute("SELECT id FROM securities WHERE code = ?", (code,))
    result = cursor.fetchone()
    
    if result:
        security_id = result[0]
        print(f"找到security_id: {security_id}")
        
        # 插入数据
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO stock_basic_info 
                (security_id, market, list_date, main_business, employees)
                VALUES (?, ?, ?, ?, ?)
            """, (
                security_id,
                row.get('market'),
                row.get('list_date'),
                row.get('main_business'),
                row.get('employees')
            ))
            saved_count += 1
            print(f"保存成功: {ts_code}")
        except Exception as e:
            print(f"保存失败: {e}")
    else:
        print(f"未找到stock: {code}")

conn.commit()
conn.close()

print(f"保存了 {saved_count} 条记录")

# 验证
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM stock_basic_info")
count = cursor.fetchone()[0]
print(f"当前stock_basic_info总数: {count}")
conn.close()