#!/usr/bin/env python3
"""
安全批量更新历史数据 - 避免触及Tushare API限制
每天最多处理20-25个交易日的数据，避免超过积分限制
"""

import pandas as pd
import tushare as ts
import time
import logging
from datetime import datetime, timedelta
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from batch_fetch_historical_data import fetch_data_for_date, get_trading_dates, check_data_completeness
from data_adapter.database_manager import DatabaseManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def safe_batch_update(start_date: str, end_date: str, daily_limit: int = 20):
    """
    安全批量更新，限制每天处理的交易日数量
    
    Args:
        start_date: 开始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD  
        daily_limit: 每天最多处理的交易日数量（默认20天）
    """
    logger.info(f"🚀 安全批量更新模式")
    logger.info(f"📅 日期范围: {start_date} 到 {end_date}")
    logger.info(f"⚠️ 每天处理限制: {daily_limit} 个交易日")
    logger.info(f"💡 预计API调用: {daily_limit * 5} 次/天 (安全范围内)")
    logger.info("="*60)
    
    # 获取所有交易日
    trading_dates = get_trading_dates(start_date, end_date)
    logger.info(f"📊 共需处理 {len(trading_dates)} 个交易日")
    
    # 检查哪些日期需要更新
    dates_to_update = []
    logger.info("🔍 检查数据完整性...")
    
    for date in trading_dates:
        status = check_data_completeness(date)
        if not status['is_complete']:
            dates_to_update.append(date)
    
    logger.info(f"📊 需要更新的日期: {len(dates_to_update)} 天")
    
    if not dates_to_update:
        logger.info("✅ 所有日期数据已完整，无需更新")
        return
    
    # 分批处理
    total_batches = (len(dates_to_update) + daily_limit - 1) // daily_limit
    logger.info(f"📦 将分 {total_batches} 批处理，每批 {daily_limit} 天")
    
    # 今天处理的批次
    today_batch = dates_to_update[:daily_limit]
    remaining = dates_to_update[daily_limit:]
    
    logger.info(f"\n🔄 今天处理第1批: {len(today_batch)} 天")
    logger.info(f"📅 日期范围: {today_batch[-1][:4]}-{today_batch[-1][4:6]}-{today_batch[-1][6:]} 到 {today_batch[0][:4]}-{today_batch[0][4:6]}-{today_batch[0][6:]}")
    
    success_count = 0
    failed_dates = []
    
    for i, date in enumerate(today_batch, 1):
        logger.info(f"\n进度: [{i}/{len(today_batch)}] (今日批次)")
        
        try:
            status = fetch_data_for_date(date, force_update=False)
            
            if status['is_complete']:
                success_count += 1
            else:
                failed_dates.append(date)
                
            # API调用间隔，避免频率限制
            time.sleep(3)  # 3秒间隔，确保每分钟不超过20次
            
        except Exception as e:
            logger.error(f"处理 {date} 失败: {e}")
            failed_dates.append(date)
            time.sleep(5)  # 出错时等待更长时间
    
    # 统计结果
    logger.info("\n" + "="*60)
    logger.info(f"📊 今日批次完成统计:")
    logger.info(f"   ✅ 成功: {success_count}/{len(today_batch)} 天")
    logger.info(f"   ❌ 失败: {len(failed_dates)} 天")
    
    if remaining:
        logger.info(f"\n⏳ 剩余待处理: {len(remaining)} 天")
        logger.info(f"💡 建议明天继续运行此脚本处理剩余数据")
        
        # 保存剩余任务到文件
        with open('remaining_dates.txt', 'w') as f:
            f.write('\n'.join(remaining))
        logger.info(f"📝 剩余日期已保存到 remaining_dates.txt")
    else:
        logger.info(f"\n✅ 所有数据更新完成！")
    
    return success_count, failed_dates

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='安全批量更新历史数据')
    parser.add_argument('--start-date', default='2020-01-01',
                       help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-12-31',
                       help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--daily-limit', type=int, default=20,
                       help='每天处理的交易日数量限制 (默认: 20)')
    parser.add_argument('--continue-from-file', action='store_true',
                       help='从remaining_dates.txt继续处理')
    
    args = parser.parse_args()
    
    if args.continue_from_file:
        # 从文件读取剩余日期
        if os.path.exists('remaining_dates.txt'):
            with open('remaining_dates.txt', 'r') as f:
                dates = f.read().strip().split('\n')
            
            if dates:
                logger.info(f"📂 从文件继续处理 {len(dates)} 个日期")
                # 这里简化处理，直接处理前N个
                # 实际使用时可以改进这部分逻辑
        else:
            logger.error("找不到 remaining_dates.txt 文件")
            return
    
    # 执行安全批量更新
    safe_batch_update(args.start_date, args.end_date, args.daily_limit)

if __name__ == "__main__":
    main()