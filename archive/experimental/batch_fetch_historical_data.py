#!/usr/bin/env python3
"""
批量获取历史数据 - 补全指定时间段内所有缺失的数据
支持获取daily_quotes, daily_basic, technical_indicators等数据
"""

import pandas as pd
import tushare as ts
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_adapter.database_manager import DatabaseManager
from fetch_data.quick_daily_update import (
    batch_update_stocks, 
    batch_update_funds,
    update_market_indices,
    update_daily_basic,
    update_financial_indicators,
    calculate_technical_indicators
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# 读取配置
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)
    ts_token = config['tushare']['token']

ts.set_token(ts_token)
pro = ts.pro_api()

# 初始化数据库管理器
db_manager = DatabaseManager()

def get_trading_dates(start_date: str, end_date: str) -> list:
    """获取指定时间段内的所有交易日"""
    try:
        # 使用Tushare获取交易日历
        df = pro.trade_cal(
            exchange='SSE',
            start_date=start_date.replace('-', ''),
            end_date=end_date.replace('-', ''),
            is_open='1'
        )
        
        if df.empty:
            logger.warning(f"未获取到交易日历数据")
            return []
        
        # 转换日期格式
        dates = df['cal_date'].tolist()
        return dates
        
    except Exception as e:
        logger.error(f"获取交易日历失败: {e}")
        # 如果API失败，生成工作日列表作为备用
        dates = []
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        current = start
        while current <= end:
            if current.weekday() < 5:  # 周一到周五
                dates.append(current.strftime('%Y%m%d'))
            current += timedelta(days=1)
        
        return dates

def check_data_completeness(date: str, skip_technical: bool = False) -> dict:
    """检查指定日期的数据完整性"""
    date_dash = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    
    # 检查各类数据的数量
    result = {
        'date': date,
        'daily_quotes': 0,
        'daily_basic': 0,
        'technical_indicators': 0,
        'is_complete': False
    }
    
    try:
        # 检查daily_quotes
        query = "SELECT COUNT(DISTINCT security_id) FROM daily_quotes WHERE trade_date = ?"
        rows = db_manager.execute_query(query, (date_dash,))
        result['daily_quotes'] = rows[0][0] if rows else 0
        
        # 检查daily_basic
        query = "SELECT COUNT(DISTINCT security_id) FROM daily_basic WHERE trade_date = ?"
        rows = db_manager.execute_query(query, (date_dash,))
        result['daily_basic'] = rows[0][0] if rows else 0
        
        # 检查technical_indicators
        query = "SELECT COUNT(DISTINCT security_id) FROM technical_indicators WHERE trade_date = ?"
        rows = db_manager.execute_query(query, (date_dash,))
        result['technical_indicators'] = rows[0][0] if rows else 0
        
        # 判断数据是否完整（至少有3000只股票的数据）
        if skip_technical:
            result['is_complete'] = (
                result['daily_quotes'] >= 3000 and
                result['daily_basic'] >= 3000
            )
        else:
            result['is_complete'] = (
                result['daily_quotes'] >= 3000 and
                result['daily_basic'] >= 3000 and
                result['technical_indicators'] >= 3000
            )
        
    except Exception as e:
        logger.error(f"检查数据完整性失败 {date}: {e}")
    
    return result

def fetch_data_for_date(date: str, force_update: bool = False, skip_technical: bool = False) -> dict:
    """获取指定日期的所有数据"""
    logger.info(f"\n{'='*60}")
    logger.info(f"处理日期: {date}")
    
    # 检查数据完整性
    if not force_update:
        status = check_data_completeness(date, skip_technical)
        if status['is_complete']:
            logger.info(f"✅ {date} 数据已完整，跳过")
            return status
        else:
            logger.info(f"📊 {date} 数据不完整:")
            logger.info(f"   - daily_quotes: {status['daily_quotes']}")
            logger.info(f"   - daily_basic: {status['daily_basic']}")
            logger.info(f"   - technical_indicators: {status['technical_indicators']}")
    
    stats = {
        'date': date,
        'daily_quotes': 0,
        'daily_basic': 0,
        'technical_indicators': 0,
        'is_complete': False
    }
    
    try:
        # 1. 获取市场行情数据（A股 + ETF）
        if stats['daily_quotes'] < 3000:
            logger.info(f"🔄 获取市场行情数据...")
            count = batch_update_stocks(date)
            stats['daily_quotes'] += count
            time.sleep(1)
            
            count = batch_update_funds(date)
            stats['daily_quotes'] += count
            time.sleep(1)
            
            # 获取大盘指数
            update_market_indices(date)
            time.sleep(1)
        
        # 2. 获取基本面数据
        if stats['daily_basic'] < 3000:
            logger.info(f"🔄 获取基本面数据...")
            count = update_daily_basic(date)
            stats['daily_basic'] = count
            time.sleep(1)
        
        # 3. 计算技术指标
        if not skip_technical and stats['technical_indicators'] < 3000:
            logger.info(f"🔄 计算技术指标...")
            count = calculate_technical_indicators(date)
            stats['technical_indicators'] = count
        elif skip_technical:
            logger.info(f"⏭️ 跳过技术指标计算")
            stats['technical_indicators'] = -1  # 标记为跳过
        
        # 重新检查完整性
        final_status = check_data_completeness(date, skip_technical)
        stats.update(final_status)
        
        if stats['is_complete']:
            logger.info(f"✅ {date} 数据获取成功!")
        else:
            logger.warning(f"⚠️ {date} 数据可能不完整")
        
    except Exception as e:
        logger.error(f"❌ {date} 数据获取失败: {e}")
    
    return stats

def batch_fetch_historical_data(start_date: str, end_date: str, 
                               max_workers: int = 1, 
                               force_update: bool = False,
                               skip_technical: bool = False):
    """批量获取历史数据
    
    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        max_workers: 并行进程数（默认1，串行处理）
        force_update: 是否强制更新已有数据
        skip_technical: 是否跳过技术指标计算
    """
    logger.info(f"🚀 开始批量获取历史数据")
    logger.info(f"📅 日期范围: {start_date} 到 {end_date}")
    logger.info(f"🔧 并行进程数: {max_workers}")
    logger.info(f"💾 强制更新: {'是' if force_update else '否'}")
    logger.info(f"⚡ 跳过技术指标: {'是' if skip_technical else '否'}")
    logger.info("="*60)
    
    # 获取所有交易日
    trading_dates = get_trading_dates(start_date, end_date)
    logger.info(f"📊 共需处理 {len(trading_dates)} 个交易日")
    
    # 首先检查哪些日期需要更新
    dates_to_update = []
    if not force_update:
        logger.info("🔍 检查数据完整性...")
        for date in trading_dates:
            status = check_data_completeness(date, skip_technical)
            if not status['is_complete']:
                dates_to_update.append(date)
        
        logger.info(f"📊 需要更新的日期: {len(dates_to_update)} 天")
        
        if not dates_to_update:
            logger.info("✅ 所有日期数据已完整，无需更新")
            return
    else:
        dates_to_update = trading_dates
    
    # 统计
    success_count = 0
    failed_dates = []
    incomplete_dates = []
    
    if max_workers == 1:
        # 串行处理（推荐，避免API限制）
        for i, date in enumerate(dates_to_update, 1):
            logger.info(f"\n进度: [{i}/{len(dates_to_update)}]")
            
            status = fetch_data_for_date(date, force_update, skip_technical)
            
            if status['is_complete']:
                success_count += 1
            elif status['daily_quotes'] > 0:
                incomplete_dates.append(date)
            else:
                failed_dates.append(date)
            
            # 避免API限制
            if i < len(dates_to_update):
                time.sleep(2)
    else:
        # 并行处理（注意API限制）
        logger.warning("⚠️ 并行处理可能触发API限制，建议使用串行模式")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_date = {
                executor.submit(fetch_data_for_date, date, force_update, skip_technical): date 
                for date in dates_to_update
            }
            
            completed = 0
            for future in as_completed(future_to_date):
                date = future_to_date[future]
                completed += 1
                
                try:
                    status = future.result()
                    
                    if status['is_complete']:
                        success_count += 1
                        logger.info(f"✅ [{completed}/{len(dates_to_update)}] {date} - 完整")
                    elif status['daily_quotes'] > 0:
                        incomplete_dates.append(date)
                        logger.warning(f"⚠️ [{completed}/{len(dates_to_update)}] {date} - 不完整")
                    else:
                        failed_dates.append(date)
                        logger.error(f"❌ [{completed}/{len(dates_to_update)}] {date} - 失败")
                        
                except Exception as e:
                    failed_dates.append(date)
                    logger.error(f"❌ [{completed}/{len(dates_to_update)}] {date} - 异常: {e}")
    
    # 输出统计结果
    logger.info("\n" + "="*60)
    logger.info(f"📊 批量获取完成统计:")
    logger.info(f"   ✅ 数据完整: {success_count} 天")
    logger.info(f"   ⚠️ 数据不完整: {len(incomplete_dates)} 天")
    logger.info(f"   ❌ 获取失败: {len(failed_dates)} 天")
    logger.info(f"   📈 成功率: {success_count/len(dates_to_update)*100:.1f}%")
    
    if incomplete_dates:
        logger.warning(f"不完整的日期: {', '.join(sorted(incomplete_dates[:10]))}")
        if len(incomplete_dates) > 10:
            logger.warning(f"... 等共 {len(incomplete_dates)} 天")
    
    if failed_dates:
        logger.error(f"失败的日期: {', '.join(sorted(failed_dates[:10]))}")
        if len(failed_dates) > 10:
            logger.error(f"... 等共 {len(failed_dates)} 天")
    
    logger.info(f"\n🎯 历史数据批量获取完成!")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='批量获取历史数据')
    parser.add_argument('--start-date', required=True, 
                       help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True,
                       help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--max-workers', type=int, default=1,
                       help='并行进程数 (默认: 1，推荐使用串行)')
    parser.add_argument('--force-update', action='store_true',
                       help='强制更新已有数据')
    parser.add_argument('--skip-technical', action='store_true',
                       help='跳过技术指标计算，仅获取API数据')
    
    args = parser.parse_args()
    
    # 验证日期格式
    try:
        datetime.strptime(args.start_date, '%Y-%m-%d')
        datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError:
        logger.error("日期格式错误，请使用 YYYY-MM-DD 格式")
        sys.exit(1)
    
    # 执行批量获取
    batch_fetch_historical_data(
        args.start_date,
        args.end_date,
        args.max_workers,
        args.force_update,
        args.skip_technical
    )

if __name__ == "__main__":
    main()