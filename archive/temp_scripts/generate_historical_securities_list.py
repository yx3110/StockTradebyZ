#!/usr/bin/env python3
"""
生成历史数据所需的证券清单
为下载2018-2025年历史数据做准备
"""

import pandas as pd
import tushare as ts
import time
import logging
from pathlib import Path
from datetime import datetime

# 确保logs目录存在
import os
os.makedirs("logs", exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/generate_historical_list.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# Tushare配置
ts_token = "a9109bae0b20e885ba526e5d2fc42ea80137b979da2b6a5b469fa31e"
ts.set_token(ts_token)

def get_all_securities():
    """获取所有证券代码（A股+ETF/基金）"""
    pro = ts.pro_api()
    all_securities = []
    
    try:
        logger.info("获取A股证券列表...")
        
        # 获取A股证券
        stock_basic = pro.stock_basic(
            exchange='',
            list_status='L',  # 仅上市证券
            fields='ts_code,symbol,name,area,industry,list_date,market'
        )
        
        if not stock_basic.empty:
            stock_basic['type'] = 'A股'
            stock_basic['code'] = stock_basic['symbol']
            all_securities.append(stock_basic[['ts_code', 'code', 'name', 'type', 'list_date']])
            logger.info(f"获取到 {len(stock_basic)} 只A股")
        
        time.sleep(0.5)  # API限制
        
        # 获取ETF基金
        logger.info("获取ETF/基金列表...")
        fund_basic = pro.fund_basic(
            market='E',  # ETF
            fields='ts_code,name,fund_type,list_date'
        )
        
        if not fund_basic.empty:
            fund_basic['type'] = 'ETF/基金'
            fund_basic['code'] = fund_basic['ts_code'].str[:6]
            all_securities.append(fund_basic[['ts_code', 'code', 'name', 'type', 'list_date']])
            logger.info(f"获取到 {len(fund_basic)} 只ETF/基金")
        
    except Exception as e:
        logger.error(f"获取证券列表失败: {e}")
        return pd.DataFrame()
    
    if not all_securities:
        return pd.DataFrame()
    
    # 合并所有证券
    combined_df = pd.concat(all_securities, ignore_index=True)
    logger.info(f"总共获取到 {len(combined_df)} 只证券")
    
    return combined_df

def check_data_gaps(securities_df, start_date="20180101"):
    """检查哪些证券需要下载历史数据"""
    data_dir = Path("full_securities_data")
    missing_securities = []
    
    logger.info(f"检查从 {start_date} 开始的历史数据缺失情况...")
    
    for _, row in securities_df.iterrows():
        ts_code = row['ts_code']
        code = row['code']
        sec_type = row['type']
        list_date = row.get('list_date', start_date)
        
        # 确保使用6位代码格式
        formatted_code = str(code).zfill(6)
        safe_type = sec_type.replace('/', '_').replace('\\', '_')
        csv_path = data_dir / f"{formatted_code}_{safe_type}.csv"
        
        needs_download = False
        reason = ""
        
        if not csv_path.exists():
            needs_download = True
            reason = "文件不存在"
        else:
            try:
                # 检查数据日期范围
                df = pd.read_csv(csv_path, parse_dates=['date'])
                if df.empty:
                    needs_download = True
                    reason = "文件为空"
                else:
                    earliest_date = df['date'].min()
                    latest_date = df['date'].max()
                    
                    # 检查起始日期（考虑上市日期）
                    target_start = max(pd.to_datetime(start_date), pd.to_datetime(str(list_date)))
                    current_date = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
                    
                    if earliest_date > target_start:
                        needs_download = True
                        reason = f"缺少早期数据(有:{earliest_date.strftime('%Y-%m-%d')}, 需要:{target_start.strftime('%Y-%m-%d')})"
                    elif latest_date < current_date - pd.Timedelta(days=7):
                        needs_download = True
                        reason = f"数据过时(最新:{latest_date.strftime('%Y-%m-%d')})"
                        
            except Exception as e:
                needs_download = True
                reason = f"读取文件错误: {e}"
        
        if needs_download:
            missing_securities.append({
                'ts_code': ts_code,
                'code': formatted_code,
                'name': row['name'],
                'type': sec_type,
                'list_date': list_date,
                'reason': reason
            })
    
    logger.info(f"发现 {len(missing_securities)} 只证券需要下载/更新历史数据")
    return pd.DataFrame(missing_securities)

def save_missing_list(missing_df, filename="missing_securities.csv"):
    """保存缺失证券列表"""
    missing_df.to_csv(filename, index=False, encoding='utf-8')
    logger.info(f"缺失证券列表已保存到: {filename}")
    
    # 按类型统计
    type_counts = missing_df['type'].value_counts()
    logger.info("缺失证券统计:")
    for sec_type, count in type_counts.items():
        logger.info(f"  {sec_type}: {count} 只")

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("开始生成历史数据证券清单")
    logger.info("=" * 60)
    
    # 获取所有证券
    logger.info("1. 获取全部证券列表...")
    securities_df = get_all_securities()
    
    if securities_df.empty:
        logger.error("获取证券列表失败，程序退出")
        return
    
    # 检查数据缺失情况
    logger.info("2. 检查历史数据缺失情况...")
    missing_df = check_data_gaps(securities_df, start_date="20180101")
    
    if missing_df.empty:
        logger.info("所有证券的历史数据都已完整，无需下载")
        return
    
    # 保存缺失列表
    logger.info("3. 保存缺失证券列表...")
    save_missing_list(missing_df)
    
    logger.info("=" * 60)
    logger.info("历史数据证券清单生成完成")
    logger.info(f"下一步可以运行: python3 download_missing_data.py --start 20180101")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()