#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9.1特征缓存高效构建脚本
- 只计算v3.9.1配置 (52特征: Phase1增强 + 基础特征，无Phase2)
- 目标期间: 2024-01-01 ~ 2025-11-24
- 优化策略: 大批量 + 检查点 + 错误恢复
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from ml_models.v39.v390_enhanced_feature_ml_system import V390EnhancedFeatureMLSystem

print("="*80)
print("V3.9.1特征缓存高效构建")
print("="*80)
print("配置: 52特征 (Phase1增强 + 基础特征，无Phase2)")
print("期间: 2024-01-01 ~ 2025-11-24")
print("="*80)

# 1. 连接数据库
conn = sqlite3.connect('data_adapter/stock_data.db')

# 2. 创建缓存表 (如果不存在)
print("\n[1/5] 初始化缓存表...")
conn.execute("""
CREATE TABLE IF NOT EXISTS v39_feature_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    features TEXT NOT NULL,
    label REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(code, trade_date, feature_version)
)
""")
conn.execute("CREATE INDEX IF NOT EXISTS idx_v39_cache_code_date ON v39_feature_cache(code, trade_date)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_v39_cache_version ON v39_feature_cache(feature_version)")
conn.commit()
print("  ✅ 缓存表已就绪")

# 3. 获取需要计算的(code, date)对
print("\n[2/5] 查询需要计算的样本...")
query = """
SELECT DISTINCT s.code, dq.trade_date
FROM securities s
JOIN daily_quotes dq ON s.id = dq.security_id
WHERE s.type = 'A股'
AND dq.trade_date >= '2024-01-01'
AND dq.trade_date <= '2025-11-24'
AND NOT EXISTS (
    SELECT 1 FROM v39_feature_cache c
    WHERE c.code = s.code
    AND c.trade_date = dq.trade_date
    AND c.feature_version = 'v3.9.1'
)
ORDER BY dq.trade_date, s.code
"""

df_tasks = pd.read_sql_query(query, conn)
print(f"  需要计算: {len(df_tasks):,} 个样本")

if len(df_tasks) == 0:
    print("\n✅ 所有样本已缓存，无需计算")
    conn.close()
    sys.exit(0)

print(f"  日期范围: {df_tasks['trade_date'].min()} ~ {df_tasks['trade_date'].max()}")
print(f"  涉及股票: {df_tasks['code'].nunique()} 只")

# 4. 初始化v3.9.1系统
print("\n[3/5] 初始化v3.9.1特征提取器...")
system = V390EnhancedFeatureMLSystem(config={
    'use_enhanced_features': True,   # Phase 1增强特征 (10个)
    'use_phase2_features': False     # 不使用Phase 2 (避免性能下降)
})
print("  ✅ v3.9.1系统已初始化 (52特征)")

# 5. 批量计算特征
print("\n[4/5] 批量计算特征...")
batch_size = 200  # 每200个样本保存一次
total_saved = 0
failed = 0
batch_data = []

from tqdm import tqdm
for idx, row in tqdm(df_tasks.iterrows(), total=len(df_tasks), desc="计算进度"):
    code = row['code']
    date = row['trade_date']

    try:
        # 提取特征
        features = system.extract_features(code, date)
        if features is None:
            failed += 1
            continue

        # 计算标签
        label = system.calculate_label(code, date)
        if label is None:
            failed += 1
            continue

        # 转换为JSON字符串
        import json
        feature_dict = features.iloc[0].to_dict()
        feature_json = json.dumps(feature_dict)

        batch_data.append({
            'code': code,
            'trade_date': date,
            'feature_version': 'v3.9.1',
            'features': feature_json,
            'label': label
        })

        # 达到批量大小，保存到数据库
        if len(batch_data) >= batch_size:
            df_batch = pd.DataFrame(batch_data)
            df_batch.to_sql('v39_feature_cache', conn, if_exists='append', index=False)
            conn.commit()
            total_saved += len(batch_data)
            batch_data = []

    except Exception as e:
        failed += 1
        continue

# 保存剩余数据
if len(batch_data) > 0:
    df_batch = pd.DataFrame(batch_data)
    df_batch.to_sql('v39_feature_cache', conn, if_exists='append', index=False)
    conn.commit()
    total_saved += len(batch_data)

print(f"\n  成功: {total_saved:,} 个样本")
print(f"  失败: {failed:,} 个样本")

# 6. 验证缓存
print("\n[5/5] 验证缓存...")
df_verify = pd.read_sql_query("""
SELECT
    feature_version,
    COUNT(*) as sample_count,
    COUNT(DISTINCT code) as stock_count,
    MIN(trade_date) as start_date,
    MAX(trade_date) as end_date
FROM v39_feature_cache
WHERE feature_version = 'v3.9.1'
GROUP BY feature_version
""", conn)

print("\n缓存统计:")
print(df_verify.to_string(index=False))

# 测试读取
sample_data = pd.read_sql_query("""
SELECT code, trade_date, features, label
FROM v39_feature_cache
WHERE feature_version = 'v3.9.1'
LIMIT 1
""", conn)

if len(sample_data) > 0:
    import json
    sample_features = json.loads(sample_data['features'].iloc[0])
    print(f"\n样本验证:")
    print(f"  股票: {sample_data['code'].iloc[0]}")
    print(f"  日期: {sample_data['trade_date'].iloc[0]}")
    print(f"  特征数: {len(sample_features)}")
    print(f"  标签: {sample_data['label'].iloc[0]:.4f}")

    # 验证特征数量
    if len(sample_features) == 52:
        print(f"\n✅ 特征数量正确 (52个)")
    else:
        print(f"\n⚠️  特征数量异常: {len(sample_features)} (预期52)")

conn.close()

print("\n" + "="*80)
print("V3.9.1特征缓存构建完成!")
print("="*80)
print(f"已保存: {total_saved:,} 个样本")
print(f"失败: {failed:,} 个样本")
print(f"成功率: {total_saved/(total_saved+failed)*100:.2f}%")
print("\n下一步: 使用缓存训练v3.9.1完整版本")
print("  命令: python3 train_v391_from_cache.py")
print("="*80)
