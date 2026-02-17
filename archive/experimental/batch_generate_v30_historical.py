#!/usr/bin/env python3
"""
批量生成v3.0历史选股报告脚本
用于生成过去8个月的v3.0版本选股数据，作为与v3.41版本回测对比的对照组
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
import argparse
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("v30_batch_generation.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_single_day_report(date_str):
    """为单个日期生成v3.0选股报告"""
    try:
        import subprocess
        import time
        
        logger.info(f"开始生成 {date_str} 的v3.0选股报告")
        
        # 运行tomorrow_stock_selector.py生成v3.0报告
        cmd = [
            sys.executable, 
            "tomorrow_stock_selector.py", 
            date_str, 
            "--scoring-version", "v3"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            logger.info(f"✅ {date_str} v3.0报告生成成功")
            return {"date": date_str, "status": "success", "message": "报告生成成功"}
        else:
            error_msg = result.stderr[:200] if result.stderr else "Unknown error"
            logger.warning(f"❌ {date_str} v3.0报告生成失败: {error_msg}")
            return {"date": date_str, "status": "failed", "message": error_msg}
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ {date_str} v3.0报告生成超时")
        return {"date": date_str, "status": "timeout", "message": "生成超时"}
    except Exception as e:
        logger.error(f"❌ {date_str} v3.0报告生成异常: {e}")
        return {"date": date_str, "status": "error", "message": str(e)}

def generate_trading_days(start_date, end_date):
    """生成交易日列表（排除周末）"""
    trading_days = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    while current <= end:
        # 排除周末（Monday=0, Sunday=6）
        if current.weekday() < 5:  # 0-4 是周一到周五
            trading_days.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    return trading_days

def batch_generate_v30_reports(start_date="2025-01-01", end_date="2025-09-02", max_workers=2):
    """批量生成v3.0历史选股报告"""
    
    logger.info("🚀 开始批量生成v3.0历史选股报告")
    logger.info(f"📅 日期范围: {start_date} 到 {end_date}")
    logger.info(f"🔧 并行进程数: {max_workers}")
    logger.info("=" * 70)
    
    # 生成交易日列表
    trading_days = generate_trading_days(start_date, end_date)
    logger.info(f"📊 总交易日数: {len(trading_days)} 天")
    
    # 确保报告目录存在
    report_dir = "reports/daily_selection_v3"
    os.makedirs(report_dir, exist_ok=True)
    
    # 检查已存在的报告
    existing_reports = []
    if os.path.exists(report_dir):
        existing_files = os.listdir(report_dir)
        for file in existing_files:
            if file.endswith('.md') and '选股分析报告' in file:
                # 从文件名中提取日期
                try:
                    date_part = file.split('_')[-1].replace('.md', '')
                    if len(date_part) == 8:  # YYYYMMDD格式
                        existing_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                        existing_reports.append(existing_date)
                except:
                    pass
    
    # 过滤出需要生成的日期
    pending_days = [day for day in trading_days if day not in existing_reports]
    logger.info(f"📋 已存在报告: {len(existing_reports)} 个")
    logger.info(f"⏳ 待生成报告: {len(pending_days)} 个")
    
    if not pending_days:
        logger.info("✅ 所有v3.0报告都已存在，无需重新生成")
        return
    
    # 并行生成报告
    results = []
    success_count = 0
    failed_count = 0
    
    logger.info("🔄 开始并行生成报告...")
    
    if max_workers == 1:
        # 单进程模式
        for date_str in pending_days:
            result = generate_single_day_report(date_str)
            results.append(result)
            if result["status"] == "success":
                success_count += 1
            else:
                failed_count += 1
            
            # 进度报告
            progress = (len(results) / len(pending_days)) * 100
            logger.info(f"进度: {len(results)}/{len(pending_days)} ({progress:.1f}%)")
    else:
        # 多进程模式
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(generate_single_day_report, date): date for date in pending_days}
            
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=800)  # 单个任务最长13分钟
                    results.append(result)
                    
                    if result["status"] == "success":
                        success_count += 1
                    else:
                        failed_count += 1
                        
                    # 进度报告
                    progress = ((i + 1) / len(futures)) * 100
                    logger.info(f"进度: {i + 1}/{len(futures)} ({progress:.1f}%) - {result['status']}")
                    
                except Exception as e:
                    date = futures[future]
                    logger.error(f"❌ {date} 处理异常: {e}")
                    results.append({"date": date, "status": "exception", "message": str(e)})
                    failed_count += 1
    
    # 生成汇总报告
    logger.info("=" * 70)
    logger.info("📊 v3.0批量生成结果汇总:")
    logger.info(f"✅ 成功生成: {success_count} 个报告")
    logger.info(f"❌ 生成失败: {failed_count} 个报告")
    logger.info(f"📁 报告保存位置: {report_dir}")
    
    # 保存结果到CSV
    results_df = pd.DataFrame(results)
    results_file = f"v30_batch_generation_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
    logger.info(f"💾 详细结果已保存: {results_file}")
    
    # 显示失败的日期
    if failed_count > 0:
        failed_dates = [r["date"] for r in results if r["status"] != "success"]
        logger.warning(f"⚠️ 失败的日期: {failed_dates[:10]}{'...' if len(failed_dates) > 10 else ''}")
    
    logger.info("🎉 v3.0历史选股报告批量生成完成!")
    return results

def main():
    parser = argparse.ArgumentParser(description='批量生成v3.0历史选股报告')
    parser.add_argument('--start-date', default='2025-01-01', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2025-09-02', help='结束日期 (YYYY-MM-DD)')  
    parser.add_argument('--max-workers', type=int, default=2, help='并行进程数 (1-4)')
    
    args = parser.parse_args()
    
    # 验证日期格式
    try:
        datetime.strptime(args.start_date, "%Y-%m-%d")
        datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError:
        logger.error("❌ 日期格式错误，请使用YYYY-MM-DD格式")
        return
    
    # 限制并行进程数
    if args.max_workers < 1 or args.max_workers > 4:
        logger.warning("⚠️ 并行进程数限制在1-4之间")
        args.max_workers = max(1, min(4, args.max_workers))
    
    # 执行批量生成
    batch_generate_v30_reports(
        start_date=args.start_date,
        end_date=args.end_date, 
        max_workers=args.max_workers
    )

if __name__ == "__main__":
    main()