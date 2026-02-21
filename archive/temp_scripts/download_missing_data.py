#!/usr/bin/env python3
"""
下载缺失的股票数据 - 处理API限制
"""

import pandas as pd
import tushare as ts
import time
import random
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 确保logs目录存在
import os
os.makedirs("../logs", exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("../logs/download_missing.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# Tushare配置
ts_token = "a9109bae0b20e885ba526e5d2fc42ea80137b979da2b6a5b469fa31e"
ts.set_token(ts_token)

def get_kline_data_with_retry(ts_code: str, sec_type: str, start: str, end: str, max_retries: int = 3) -> pd.DataFrame:
    """获取K线数据，带重试机制"""
    pro = ts.pro_api()
    
    for attempt in range(1, max_retries + 1):
        try:
            # 轻微延迟，符合API限制
            time.sleep(random.uniform(0.06, 0.08))
            
            if sec_type == 'A股':
                df = pro.daily(
                    ts_code=ts_code,
                    start_date=start,
                    end_date=end
                )
            else:  # ETF/基金
                df = pro.fund_daily(
                    ts_code=ts_code,
                    start_date=start,
                    end_date=end
                )
            
            if df is None or df.empty:
                logger.warning(f"{ts_code} ({sec_type}) 返回空数据")
                return pd.DataFrame()
            
            # 标准化列名
            df = df.rename(columns={
                "trade_date": "date",
                "vol": "volume"
            })
            
            df["date"] = pd.to_datetime(df["date"])
            
            # 确保有必要的列
            required_cols = ["date", "open", "close", "high", "low", "volume"]
            available_cols = [col for col in required_cols if col in df.columns]
            
            if "date" not in available_cols:
                return pd.DataFrame()
            
            df = df[available_cols]
            return df.sort_values("date").reset_index(drop=True)
            
        except Exception as e:
            error_msg = str(e)
            if "每分钟最多访问该接口" in error_msg:
                # API限制错误，等待更长时间
                wait_time = 70 + random.uniform(10, 20)  # 70-90秒
                logger.warning(f"API限制 {ts_code} ({sec_type}) 第{attempt}次，等待{wait_time:.1f}秒")
                time.sleep(wait_time)
            else:
                logger.warning(f"获取 {ts_code} ({sec_type}) 失败({attempt}/{max_retries}): {e}")
                time.sleep(random.uniform(2, 5) * attempt)
    
    logger.error(f"{ts_code} ({sec_type}) {max_retries}次尝试均失败")
    return pd.DataFrame()

def download_single_missing(ts_code: str, code: str, sec_type: str, 
                          start: str, end: str, out_dir: Path) -> bool:
    """下载单个缺失的证券数据"""
    
    # 安全的文件名，保持6位前导0格式
    safe_type = sec_type.replace('/', '_').replace('\\', '_')
    formatted_code = str(code).zfill(6)  # 确保6位前导0
    csv_path = out_dir / f"{formatted_code}_{safe_type}.csv"
    
    try:
        # 获取数据
        df = get_kline_data_with_retry(ts_code, sec_type, start, end)
        if df.empty:
            return False
        
        # 检查是否需要与现有数据合并
        if csv_path.exists():
            try:
                existing_df = pd.read_csv(csv_path, parse_dates=["date"])
                # 合并数据并去重
                df = pd.concat([existing_df, df], ignore_index=True)
                df = df.drop_duplicates(subset="date").sort_values("date")
                logger.info(f"合并现有数据: {csv_path}")
            except Exception as e:
                logger.warning(f"合并现有数据失败 {csv_path}: {e}")
        
        # 保存数据
        df.to_csv(csv_path, index=False)
        logger.info(f"成功下载: {formatted_code} ({ts_code}) - {len(df)}天数据")
        return True
        
    except Exception as e:
        logger.error(f"下载 {ts_code} ({sec_type}) 失败: {e}")
        return False

def download_missing_securities(batch_size: int = 50, workers: int = 3, start_date: str = "20130101"):
    """批量下载缺失的证券数据"""
    
    # 读取缺失证券列表
    missing_file = Path("../missing_securities.csv")
    if not missing_file.exists():
        logger.error("缺失证券列表文件不存在，请先运行 analyze_missing_data.py")
        return
    
    missing_df = pd.read_csv(missing_file)
    logger.info(f"发现 {len(missing_df)} 个缺失证券")
    
    # 数据参数 - 支持更长历史数据
    start = start_date  # 使用参数指定的起始日期
    from datetime import datetime
    end = datetime.now().strftime("%Y%m%d")  # 使用今天的日期
    out_dir = Path("../full_securities_data")
    
    # 按类型分批处理，避免混合不同API调用
    a_stocks = missing_df[missing_df['type'] == 'A股']
    etf_funds = missing_df[missing_df['type'] == 'ETF/基金']
    
    total_success = 0
    total_failed = 0
    
    for sec_type, securities in [('A股', a_stocks), ('ETF/基金', etf_funds)]:
        if securities.empty:
            continue
            
        logger.info(f"开始下载 {sec_type} 数据: {len(securities)} 只")
        
        # 分批处理，每批之间有更长的休息时间
        for i in range(0, len(securities), batch_size):
            batch = securities.iloc[i:i+batch_size]
            logger.info(f"处理第 {i//batch_size + 1} 批 {sec_type} ({len(batch)} 只)")
            
            batch_success = 0
            batch_failed = 0
            
            # 降低并发数，避免API限制
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = []
                
                for _, row in batch.iterrows():
                    future = executor.submit(
                        download_single_missing,
                        row['ts_code'], row['code'], row['type'],
                        start, end, out_dir
                    )
                    futures.append((future, row['code'], row['ts_code']))
                
                # 处理结果
                with tqdm(total=len(futures), desc=f"下载{sec_type}") as pbar:
                    for future, code, ts_code in futures:
                        try:
                            success = future.result(timeout=300)  # 5分钟超时
                            if success:
                                batch_success += 1
                            else:
                                batch_failed += 1
                        except Exception as e:
                            logger.error(f"处理 {code} ({ts_code}) 超时或出错: {e}")
                            batch_failed += 1
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            '成功': batch_success,
                            '失败': batch_failed
                        })
            
            total_success += batch_success
            total_failed += batch_failed
            
            # 批次间休息，避免API限制
            if i + batch_size < len(securities):
                rest_time = random.uniform(2, 4)
                logger.info(f"批次完成，休息 {rest_time:.1f} 秒...")
                time.sleep(rest_time)
    
    logger.info(f"下载完成！总成功: {total_success}, 总失败: {total_failed}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="下载缺失的证券历史数据")
    parser.add_argument("--start", default="20130101", help="起始日期 YYYYMMDD (默认: 20130101)")
    parser.add_argument("--batch-size", type=int, default=100, help="批次大小 (默认: 100)")
    parser.add_argument("--workers", type=int, default=8, help="并发线程数 (默认: 8)")
    
    args = parser.parse_args()
    
    print(f"开始下载历史数据：起始日期 {args.start}")
    download_missing_securities(
        batch_size=args.batch_size, 
        workers=args.workers, 
        start_date=args.start
    )