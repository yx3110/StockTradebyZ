#!/usr/bin/env python3
"""
强制重新生成V3.95评分异常的报告
"""

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime

REPORT_DIR = Path('reports/daily_selection_v3.95')

def find_reports_to_regenerate():
    """找出需要重新生成的报告（评分范围异常的）"""
    dates_to_regenerate = []

    for f in sorted(REPORT_DIR.glob('analysis_data_2025*.json')):
        date_str = f.stem.replace('analysis_data_', '')
        date_formatted = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}'

        # 只检查2025-09到2025-12
        if date_formatted < '2025-09-01' or date_formatted > '2025-12-31':
            continue

        with open(f, 'r', encoding='utf-8') as file:
            data = json.load(file)

        stocks = data.get('all_stocks_with_scores', [])
        if not stocks:
            dates_to_regenerate.append((date_formatted, f))
            continue

        scores = [s.get('score', 0) for s in stocks if isinstance(s, dict)]
        max_score = max(scores) if scores else 0

        # 如果最高分只有65或以下，说明评分异常
        if max_score <= 65:
            dates_to_regenerate.append((date_formatted, f))

    return dates_to_regenerate

def regenerate_report(date_str, old_file):
    """重新生成单个日期的V3.95报告"""
    print(f"重新生成 {date_str} 的V3.95报告...")

    # 删除旧报告文件
    date_formatted = date_str.replace('-', '')
    old_files = list(REPORT_DIR.glob(f'*{date_formatted}*'))
    for f in old_files:
        f.unlink()
        print(f"  删除旧文件: {f.name}")

    # 生成新报告
    try:
        result = subprocess.run(
            ['python3', 'tomorrow_stock_selector.py', date_str, '--scoring-version', 'v3.95'],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            # 验证新报告
            new_file = REPORT_DIR / f'analysis_data_{date_formatted}.json'
            if new_file.exists():
                with open(new_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                stocks = data.get('all_stocks_with_scores', [])
                if stocks:
                    scores = [s.get('score', 0) for s in stocks if isinstance(s, dict)]
                    high_87 = sum(1 for s in scores if s >= 87)
                    high_90 = sum(1 for s in scores if s >= 90)
                    print(f"  ✅ 成功: 股票数={len(stocks)}, 评分范围={min(scores):.1f}-{max(scores):.1f}, >=87分={high_87}, >=90分={high_90}")
                    return True, high_90
            print(f"  ⚠️ 报告生成但验证失败")
            return False, 0
        else:
            print(f"  ❌ 失败: {result.stderr[:100]}")
            return False, 0
    except subprocess.TimeoutExpired:
        print(f"  ❌ 超时")
        return False, 0
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        return False, 0

def main():
    print("="*60)
    print("V3.95 报告重新生成工具")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 找出需要重新生成的报告
    reports_to_regenerate = find_reports_to_regenerate()
    print(f"\n需要重新生成的报告: {len(reports_to_regenerate)} 个")

    if not reports_to_regenerate:
        print("没有需要重新生成的报告")
        return

    # 重新生成报告
    success_count = 0
    total_90_scores = 0

    for i, (date_str, old_file) in enumerate(reports_to_regenerate):
        print(f"\n[{i+1}/{len(reports_to_regenerate)}] ", end="")
        success, count_90 = regenerate_report(date_str, old_file)
        if success:
            success_count += 1
            total_90_scores += count_90

    print(f"\n{'='*60}")
    print(f"重新生成完成!")
    print(f"成功: {success_count}/{len(reports_to_regenerate)}")
    print(f"90分以上股票总数: {total_90_scores}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
