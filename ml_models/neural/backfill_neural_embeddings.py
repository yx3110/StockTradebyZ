#!/usr/bin/env python3
"""
回填 GRU Neural Embeddings 到缓存

从训练好的 GRU 模型提取所有 (code, date) 组合的 16 维嵌入向量,
存储到 neural_embedding_cache 表中.

用法:
  # 回填 2024-01-01 ~ 2026-02-13 (约 260 万条, ~15 分钟)
  python3 ml_models/neural/backfill_neural_embeddings.py \
      --start-date 2024-01-01 --end-date 2026-02-13

  # 只回填最近 30 天 (日常增量)
  python3 ml_models/neural/backfill_neural_embeddings.py --recent 30
"""

import sys
import numpy as np
import pandas as pd
import sqlite3
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def backfill_embeddings(start_date: str, end_date: str,
                        model_path: str = None,
                        batch_size: int = 4096,
                        db_path: str = DB_PATH):
    """回填 GRU embeddings"""
    if not HAS_TORCH:
        raise ImportError("PyTorch required. Install: pip install torch")

    from ml_models.neural.gru_encoder import (
        load_gru_model, load_ohlcv_data, StockSequenceDataset, get_device,
        SEQUENCE_LENGTH, prebuild_all_sequences
    )
    from ml_models.neural.embedding_cache_manager import EmbeddingCacheManager

    device = get_device()
    logger.info(f"Device: {device}")

    # 1. Load model
    model, config = load_gru_model(model_path, device)
    logger.info(f"GRU 模型加载完成 (embedding_dim={config['embedding_dim']})")

    # 2. Load OHLCV data (need 60 days before start_date)
    ohlcv_start = pd.to_datetime(start_date) - pd.Timedelta(days=120)
    ohlcv_data = load_ohlcv_data(db_path, ohlcv_start.strftime('%Y-%m-%d'), end_date)

    # 3. Get all (code, date) pairs to process
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT DISTINCT v.code, v.trade_date
    FROM v39_feature_cache v
    WHERE v.trade_date >= '{start_date}'
      AND v.trade_date <= '{end_date}'
    ORDER BY v.trade_date, v.code
    """
    codes_dates = pd.read_sql(query, conn)
    conn.close()

    logger.info(f"待处理: {len(codes_dates):,} 个 (code, date) 组合")
    logger.info(f"日期范围: {start_date} ~ {end_date}")

    # 4. Check what's already cached
    cache_mgr = EmbeddingCacheManager(db_path)
    stats = cache_mgr.get_stats()
    if 'gru_v1' in stats:
        existing = stats['gru_v1']
        logger.info(f"已有缓存: {existing['count']:,} 条 ({existing['date_range']})")

    # 5. Pre-build sequences and create dataset
    logger.info("预构建序列...")
    import time as _time
    t0 = _time.time()
    sequences = prebuild_all_sequences(codes_dates[['code', 'trade_date']], ohlcv_data)
    logger.info(f"  序列构建完成: {sequences.shape}, 耗时 {_time.time()-t0:.1f}秒")
    dataset = StockSequenceDataset(sequences)

    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_embeddings = []
    idx = 0

    with torch.no_grad():
        for batch_x in tqdm(loader, desc="提取 embeddings"):
            if isinstance(batch_x, (list, tuple)):
                batch_x = batch_x[0]
            batch_x = batch_x.to(device)
            embeddings = model.get_embedding(batch_x)  # (batch, 16)
            embeddings_np = embeddings.cpu().numpy()

            for i in range(len(embeddings_np)):
                if idx + i < len(codes_dates):
                    row = codes_dates.iloc[idx + i]
                    all_embeddings.append(
                        (row['code'], row['trade_date'], embeddings_np[i])
                    )

            idx += len(embeddings_np)

    # 6. Store to cache
    logger.info(f"存储 {len(all_embeddings):,} 个 embeddings...")

    # Store in chunks to avoid memory issues
    chunk_size = 50000
    for i in range(0, len(all_embeddings), chunk_size):
        chunk = all_embeddings[i:i + chunk_size]
        cache_mgr.batch_store(chunk, model_version='gru_v1')
        logger.info(f"  已存储: {min(i + chunk_size, len(all_embeddings)):,}/{len(all_embeddings):,}")

    # 7. Verify
    stats = cache_mgr.get_stats()
    if 'gru_v1' in stats:
        logger.info(f"\n回填完成: {stats['gru_v1']}")


def daily_update_embeddings(date: str, model_path: str = None,
                            db_path: str = DB_PATH):
    """日常更新: 计算指定日期的 embeddings (<5秒)"""
    if not HAS_TORCH:
        logger.warning("PyTorch not installed, skipping neural embeddings")
        return

    from ml_models.neural.gru_encoder import (
        load_gru_model, load_ohlcv_data, StockSequenceDataset, get_device,
        prebuild_all_sequences
    )
    from ml_models.neural.embedding_cache_manager import EmbeddingCacheManager

    device = get_device()

    # Load model
    try:
        model, config = load_gru_model(model_path, device)
    except FileNotFoundError:
        logger.warning("GRU model not found, skipping neural embeddings")
        return

    # Get codes for this date
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT DISTINCT code FROM v39_feature_cache
    WHERE trade_date = '{date}'
    """
    codes = pd.read_sql(query, conn)['code'].tolist()
    conn.close()

    if not codes:
        return

    # Load OHLCV for these codes (last 120 days)
    ohlcv_start = pd.to_datetime(date) - pd.Timedelta(days=120)
    ohlcv_data = load_ohlcv_data(db_path, ohlcv_start.strftime('%Y-%m-%d'), date)

    # Build dataset
    codes_dates = pd.DataFrame({'code': codes, 'trade_date': [date] * len(codes)})
    sequences = prebuild_all_sequences(codes_dates, ohlcv_data)
    dataset = StockSequenceDataset(sequences)

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=0)

    # Extract embeddings
    all_embeddings = []
    idx = 0
    with torch.no_grad():
        for batch_x in loader:
            if isinstance(batch_x, (list, tuple)):
                batch_x = batch_x[0]
            batch_x = batch_x.to(device)
            embeddings = model.get_embedding(batch_x)
            embeddings_np = embeddings.cpu().numpy()

            for i in range(len(embeddings_np)):
                if idx + i < len(codes):
                    all_embeddings.append(
                        (codes[idx + i], date, embeddings_np[i])
                    )
            idx += len(embeddings_np)

    # Store
    cache_mgr = EmbeddingCacheManager(db_path)
    cache_mgr.batch_store(all_embeddings, model_version='gru_v1')
    logger.info(f"Neural embeddings 更新: {date}, {len(all_embeddings)} 只股票")


def main():
    parser = argparse.ArgumentParser(description='Backfill GRU Neural Embeddings')
    parser.add_argument('--start-date', type=str, default='2024-01-01')
    parser.add_argument('--end-date', type=str, default=None)
    parser.add_argument('--recent', type=int, default=None,
                       help='只回填最近 N 天 (覆盖 start/end)')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=4096)
    args = parser.parse_args()

    if args.recent:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=args.recent)).strftime('%Y-%m-%d')
    else:
        start_date = args.start_date
        end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')

    backfill_embeddings(
        start_date=start_date,
        end_date=end_date,
        model_path=args.model_path,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
