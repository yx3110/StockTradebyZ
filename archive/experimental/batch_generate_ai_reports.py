#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量生成缺失的AI报告
"""

import os
import subprocess
import time
from datetime import datetime

def generate_ai_report(date: str) -> bool:
    """生成单个AI报告"""
    print(f"\n🤖 生成AI报告: {date}")
    
    # 1. 先确保有选股报告
    date_str = date.replace('-', '')
    selection_report = f"reports/daily_selection/选股分析报告_{date_str}.md"
    if not os.path.exists(selection_report):
        print(f"   📊 先运行选股程序...")
        try:
            cmd = f"python3 tomorrow_stock_selector.py {date}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                print(f"   ❌ 选股程序失败: {result.stderr[:200]}")
                return False
            print(f"   ✅ 选股程序完成")
        except Exception as e:
            print(f"   ❌ 选股程序异常: {e}")
            return False
    else:
        print(f"   ✅ 选股报告已存在")
    
    # 2. 生成AI增强报告
    ai_report = f"reports/ai_enhanced/AI增强选股报告_{date}.md"
    if os.path.exists(ai_report):
        print(f"   ✅ AI报告已存在")
        return True
        
    try:
        cmd = f"python3 ai_enhanced_daily_report.py --date {date}"
        print(f"   🤖 正在生成AI报告...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0 and os.path.exists(ai_report):
            print(f"   ✅ AI报告生成成功")
            return True
        else:
            print(f"   ❌ AI报告生成失败: {result.stderr[:200] if result.stderr else 'Unknown error'}")
            return False
    except subprocess.TimeoutExpired:
        print(f"   ⏰ AI报告生成超时")
        return False
    except Exception as e:
        print(f"   ❌ AI报告生成异常: {e}")
        return False

def main():
    """主函数"""
    missing_dates = [
        '2025-07-07', '2025-07-09', '2025-07-10', '2025-07-11', '2025-07-14',
        '2025-07-15', '2025-07-16', '2025-07-17', '2025-07-18', '2025-07-21',
        '2025-07-22', '2025-07-23', '2025-07-24', '2025-07-25', '2025-07-28',
        '2025-07-29', '2025-07-30', '2025-07-31', '2025-08-04'
    ]
    
    print(f"🚀 开始批量生成 {len(missing_dates)} 个缺失的AI报告")
    print("="*70)
    
    success_count = 0
    failed_dates = []
    
    for i, date in enumerate(missing_dates, 1):
        print(f"\n📅 进度: {i}/{len(missing_dates)} - {date}")
        
        if generate_ai_report(date):
            success_count += 1
        else:
            failed_dates.append(date)
            
        # 避免API调用过快
        if i < len(missing_dates):
            print("   ⏳ 等待30秒避免API限制...")
            time.sleep(30)
    
    print("\n" + "="*70)
    print(f"📊 生成完成统计:")
    print(f"   ✅ 成功: {success_count}个")
    print(f"   ❌ 失败: {len(failed_dates)}个")
    
    if failed_dates:
        print(f"   失败日期: {', '.join(failed_dates)}")
    
    print(f"\n🎯 现在可以运行完整的AI驱动交易回测!")

if __name__ == "__main__":
    main()