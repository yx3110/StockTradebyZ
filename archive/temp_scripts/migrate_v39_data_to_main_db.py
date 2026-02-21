#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全地将v3.9数据从根目录数据库迁移到主数据库

策略：
1. 基于stock code进行JOIN（而不是security_id）
2. 使用INSERT OR REPLACE避免重复
3. 保留主数据库的existing数据
4. 迁移前自动备份
"""

import sqlite3
import os
from datetime import datetime

def backup_database(db_path):
    """备份数据库"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{db_path}.backup_{timestamp}"

    print(f"📦 备份数据库: {db_path}")
    print(f"   → {backup_path}")

    os.system(f"cp '{db_path}' '{backup_path}'")
    return backup_path

def migrate_financial_indicator():
    """迁移financial_indicator数据"""

    print("\n" + "="*80)
    print("开始迁移v3.9财务指标数据")
    print("="*80)

    # 连接两个数据库
    src_db = sqlite3.connect('stock_data.db')
    dst_db = sqlite3.connect('data_adapter/stock_data.db')

    src_cursor = src_db.cursor()
    dst_cursor = dst_db.cursor()

    # 1. 统计源数据库数据
    src_cursor.execute("SELECT COUNT(*) FROM financial_indicator")
    src_count = src_cursor.fetchone()[0]
    print(f"\n📊 源数据库(根目录): {src_count:,} 条记录")

    dst_cursor.execute("SELECT COUNT(*) FROM financial_indicator")
    dst_count_before = dst_cursor.fetchone()[0]
    print(f"📊 目标数据库(主库)迁移前: {dst_count_before:,} 条记录")

    # 2. 获取所有需要迁移的财务数据（基于stock code）
    print(f"\n🔄 读取源数据...")
    src_cursor.execute("""
        SELECT
            s.code,
            fi.ann_date,
            fi.end_date,
            fi.eps, fi.dt_eps, fi.roe, fi.roe_waa, fi.roe_dt, fi.roa,
            fi.grossprofit_margin, fi.netprofit_margin, fi.profit_to_gr,
            fi.ocf_to_profit, fi.debt_to_assets, fi.current_ratio, fi.quick_ratio,
            fi.ar_turn, fi.ca_turn, fi.fa_turn, fi.assets_turn,
            fi.netprofit_yoy, fi.or_yoy
        FROM financial_indicator fi
        JOIN securities s ON fi.security_id = s.id
        WHERE s.type = 'A股'
    """)

    rows = src_cursor.fetchall()
    print(f"✅ 读取到 {len(rows):,} 条A股财务数据")

    # 3. 迁移数据（基于code映射到目标数据库的security_id）
    print(f"\n🔄 开始迁移...")

    success_count = 0
    skip_count = 0
    error_count = 0

    for i, row in enumerate(rows):
        code = row[0]
        ann_date = row[1]
        end_date = row[2]
        fields = row[3:]  # 所有财务字段

        try:
            # 规范化股票代码（去掉.SZ/.SH后缀）
            normalized_code = code.split('.')[0] if '.' in code else code

            # 在目标数据库中查找对应的security_id
            dst_cursor.execute("""
                SELECT id FROM securities WHERE code = ? AND type = 'A股'
            """, (normalized_code,))

            result = dst_cursor.fetchone()
            if not result:
                skip_count += 1
                if skip_count <= 5:
                    print(f"⚠️  跳过: {code} (在主数据库中不存在)")
                continue

            target_security_id = result[0]

            # 插入数据（使用INSERT OR REPLACE）
            dst_cursor.execute("""
                INSERT OR REPLACE INTO financial_indicator
                (security_id, ann_date, end_date,
                 eps, dt_eps, roe, roe_waa, roe_dt, roa,
                 grossprofit_margin, netprofit_margin, profit_to_gr,
                 ocf_to_profit, debt_to_assets, current_ratio, quick_ratio,
                 ar_turn, ca_turn, fa_turn, assets_turn,
                 netprofit_yoy, or_yoy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (target_security_id, ann_date, end_date, *fields))

            success_count += 1

            # 进度显示
            if (i + 1) % 5000 == 0:
                print(f"   进度: {i+1:,}/{len(rows):,} ({(i+1)/len(rows)*100:.1f}%)")

        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"❌ 错误: {code} - {e}")

    # 4. 提交事务
    dst_db.commit()

    # 5. 统计结果
    dst_cursor.execute("SELECT COUNT(*) FROM financial_indicator")
    dst_count_after = dst_cursor.fetchone()[0]

    print("\n" + "="*80)
    print("📊 迁移结果统计")
    print("="*80)
    print(f"✅ 成功迁移: {success_count:,} 条")
    print(f"⚠️  跳过记录: {skip_count:,} 条 (股票在主库不存在)")
    print(f"❌ 错误记录: {error_count:,} 条")
    print(f"\n📊 主数据库记录数:")
    print(f"   迁移前: {dst_count_before:,} 条")
    print(f"   迁移后: {dst_count_after:,} 条")
    print(f"   新增: {dst_count_after - dst_count_before:,} 条")

    # 关闭连接
    src_db.close()
    dst_db.close()

    return success_count, skip_count, error_count

def verify_migration():
    """验证迁移结果"""
    print("\n" + "="*80)
    print("🔍 验证迁移结果")
    print("="*80)

    db = sqlite3.connect('data_adapter/stock_data.db')
    cursor = db.cursor()

    # 检查记录数
    cursor.execute("SELECT COUNT(*) FROM financial_indicator")
    count = cursor.fetchone()[0]
    print(f"✅ 总记录数: {count:,}")

    # 检查日期范围
    cursor.execute("SELECT MIN(ann_date), MAX(ann_date) FROM financial_indicator")
    min_date, max_date = cursor.fetchone()
    print(f"✅ 日期范围: {min_date} ~ {max_date}")

    # 检查字段完整性（v3.9新增字段）
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(roe) as roe_count,
            COUNT(netprofit_margin) as npm_count,
            COUNT(debt_to_assets) as dta_count,
            COUNT(ar_turn) as art_count
        FROM financial_indicator
    """)
    stats = cursor.fetchone()
    print(f"✅ 关键字段覆盖率:")
    print(f"   ROE: {stats[1]/stats[0]*100:.1f}%")
    print(f"   净利率: {stats[2]/stats[0]*100:.1f}%")
    print(f"   资产负债率: {stats[3]/stats[0]*100:.1f}%")
    print(f"   应收账款周转率: {stats[4]/stats[0]*100:.1f}%")

    # 样本数据
    cursor.execute("""
        SELECT s.code, fi.end_date, fi.roe, fi.netprofit_margin
        FROM financial_indicator fi
        JOIN securities s ON fi.security_id = s.id
        ORDER BY fi.end_date DESC
        LIMIT 5
    """)

    print(f"\n✅ 最新5条记录样本:")
    for row in cursor.fetchall():
        roe_str = f"{row[2]:.2f}%" if row[2] is not None else "N/A"
        npm_str = f"{row[3]:.2f}%" if row[3] is not None else "N/A"
        print(f"   {row[0]}: {row[1]}, ROE={roe_str}, 净利率={npm_str}")

    db.close()

def main():
    print("\n" + "="*80)
    print("🚀 V3.9数据迁移工具")
    print("="*80)
    print("从根目录 stock_data.db 迁移财务数据到 data_adapter/stock_data.db")
    print()

    # 确认操作
    response = input("⚠️  此操作将修改主数据库，是否继续？(yes/no): ")
    if response.lower() != 'yes':
        print("❌ 操作已取消")
        return

    # 备份主数据库
    backup_path = backup_database('data_adapter/stock_data.db')

    # 执行迁移
    success, skip, error = migrate_financial_indicator()

    # 验证结果
    verify_migration()

    print("\n" + "="*80)
    print("✅ 迁移完成！")
    print("="*80)
    print(f"备份文件: {backup_path}")
    print()

if __name__ == "__main__":
    main()
