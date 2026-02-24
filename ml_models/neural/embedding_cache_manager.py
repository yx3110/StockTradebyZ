#!/usr/bin/env python3
"""
Neural Embedding Cache Manager

SQLite-backed storage for pre-computed GRU embeddings.
Uses BLOB storage (16 x float32 = 64 bytes per embedding) for efficiency.
"""

import sqlite3
import struct
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
EMBEDDING_DIM = 16


class EmbeddingCacheManager:
    """Neural embedding 缓存管理器"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._ensure_table()

    def _ensure_table(self):
        """创建缓存表 (如果不存在)"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS neural_embedding_cache (
            code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            model_version TEXT NOT NULL DEFAULT 'gru_v1',
            embedding BLOB NOT NULL,
            UNIQUE(code, trade_date, model_version)
        )
        """)
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_neural_emb_date
            ON neural_embedding_cache(trade_date)
        """)
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_neural_emb_code_date
            ON neural_embedding_cache(code, trade_date)
        """)
        conn.commit()
        conn.close()

    @staticmethod
    def _encode_embedding(embedding: np.ndarray) -> bytes:
        """将 float32 数组编码为 BLOB"""
        return struct.pack(f'{len(embedding)}f', *embedding.astype(np.float32))

    @staticmethod
    def _decode_embedding(blob: bytes, dim: int = EMBEDDING_DIM) -> np.ndarray:
        """将 BLOB 解码为 float32 数组"""
        return np.array(struct.unpack(f'{dim}f', blob), dtype=np.float32)

    def batch_store(self, embeddings: List[Tuple[str, str, np.ndarray]],
                    model_version: str = 'gru_v1'):
        """
        批量存储 embeddings

        Args:
            embeddings: [(code, trade_date, embedding_array), ...]
            model_version: 模型版本标识
        """
        conn = sqlite3.connect(self.db_path)
        data = []
        for code, date, emb in embeddings:
            blob = self._encode_embedding(emb)
            data.append((code, date, model_version, blob))

        conn.executemany("""
        INSERT OR REPLACE INTO neural_embedding_cache
            (code, trade_date, model_version, embedding)
        VALUES (?, ?, ?, ?)
        """, data)
        conn.commit()
        conn.close()

    def batch_load(self, codes: List[str], date: str,
                   model_version: str = 'gru_v1') -> Dict[str, np.ndarray]:
        """
        加载指定日期指定股票的 embeddings

        Returns:
            {code: embedding_array}
        """
        conn = sqlite3.connect(self.db_path)
        placeholders = ','.join(['?' for _ in codes])
        query = f"""
        SELECT code, embedding FROM neural_embedding_cache
        WHERE code IN ({placeholders})
          AND trade_date = ?
          AND model_version = ?
        """
        cursor = conn.execute(query, codes + [date, model_version])
        result = {}
        for code, blob in cursor:
            result[code] = self._decode_embedding(blob)
        conn.close()
        return result

    def load_date_range(self, start_date: str, end_date: str,
                        model_version: str = 'gru_v1') -> Optional[pd.DataFrame]:
        """
        加载日期范围内所有 embeddings, 返回 DataFrame

        Returns:
            DataFrame with columns: code, trade_date, gru_emb_0, ..., gru_emb_15
        """
        conn = sqlite3.connect(self.db_path)
        query = """
        SELECT code, trade_date, embedding FROM neural_embedding_cache
        WHERE trade_date >= ? AND trade_date <= ?
          AND model_version = ?
        ORDER BY trade_date, code
        """
        cursor = conn.execute(query, (start_date, end_date, model_version))

        records = []
        for code, date, blob in cursor:
            emb = self._decode_embedding(blob)
            record = {'code': code, 'trade_date': date}
            for i, val in enumerate(emb):
                record[f'gru_emb_{i}'] = float(val)
            records.append(record)

        conn.close()

        if not records:
            return None

        return pd.DataFrame(records)

    def load_dates(self, dates: List[str],
                   model_version: str = 'gru_v1') -> Dict[str, pd.DataFrame]:
        """
        按日期加载 embeddings (用于批量评分)

        Returns:
            {date: DataFrame(code, gru_emb_0, ..., gru_emb_15)}
        """
        conn = sqlite3.connect(self.db_path)
        placeholders = ','.join(['?' for _ in dates])
        query = f"""
        SELECT code, trade_date, embedding FROM neural_embedding_cache
        WHERE trade_date IN ({placeholders})
          AND model_version = ?
        ORDER BY trade_date, code
        """
        cursor = conn.execute(query, dates + [model_version])

        date_records = {}
        for code, date, blob in cursor:
            emb = self._decode_embedding(blob)
            if date not in date_records:
                date_records[date] = []
            record = {'code': code}
            for i, val in enumerate(emb):
                record[f'gru_emb_{i}'] = float(val)
            date_records[date].append(record)

        conn.close()

        result = {}
        for date, records in date_records.items():
            result[date] = pd.DataFrame(records)

        return result

    def get_stats(self) -> dict:
        """获取缓存统计"""
        conn = sqlite3.connect(self.db_path)
        stats = {}

        cursor = conn.execute("""
        SELECT model_version, COUNT(*), MIN(trade_date), MAX(trade_date),
               COUNT(DISTINCT trade_date), COUNT(DISTINCT code)
        FROM neural_embedding_cache
        GROUP BY model_version
        """)

        for version, count, min_date, max_date, n_dates, n_codes in cursor:
            stats[version] = {
                'count': count,
                'date_range': f'{min_date} ~ {max_date}',
                'n_dates': n_dates,
                'n_codes': n_codes,
            }

        conn.close()
        return stats

    def delete_version(self, model_version: str):
        """删除指定版本的所有 embeddings"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM neural_embedding_cache WHERE model_version = ?",
                    (model_version,))
        conn.commit()
        conn.close()
        logger.info(f"已删除 model_version={model_version} 的所有缓存")
