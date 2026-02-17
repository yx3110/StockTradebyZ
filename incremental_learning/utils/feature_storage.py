#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.8特征存储结构优化
负责特征版本管理、存储优化和数据压缩
"""

import os
import sys
import pickle
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import hashlib

# 添加项目路径
sys.path.append('/Users/yangxu/StockTradebyZ')
from data_adapter.database_manager import DatabaseManager

class FeatureStorageManager:
    """
    特征存储管理器

    核心功能：
    1. 特征版本管理
    2. 增量特征存储
    3. 特征数据压缩
    4. 特征索引优化
    """

    def __init__(self, base_dir: str = "models/v380/features", db_manager: DatabaseManager = None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.db_manager = db_manager
        self.logger = logging.getLogger('FeatureStorageManager')

        # 特征版本数据库
        self.feature_db_path = self.base_dir / "feature_versions.db"
        self._init_feature_database()

        # 内存缓存
        self.feature_cache = {}
        self.cache_max_size = 1000  # 最大缓存特征数

    def _init_feature_database(self):
        """初始化特征版本数据库"""
        conn = sqlite3.connect(self.feature_db_path)
        cursor = conn.cursor()

        # 创建特征版本表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS feature_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version_id TEXT UNIQUE NOT NULL,
            feature_set_name TEXT NOT NULL,
            feature_count INTEGER NOT NULL,
            feature_names TEXT NOT NULL,  -- JSON格式
            data_hash TEXT NOT NULL,
            storage_path TEXT NOT NULL,
            compression_type TEXT DEFAULT 'pickle',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT  -- JSON格式
        )
        """)

        # 创建特征索引表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS feature_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version_id TEXT NOT NULL,
            code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            feature_hash TEXT NOT NULL,
            file_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (version_id) REFERENCES feature_versions (version_id)
        )
        """)

        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feature_version ON feature_index (version_id, code, trade_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feature_hash ON feature_index (feature_hash)")

        conn.commit()
        conn.close()

        self.logger.info(f"📊 初始化特征存储数据库: {self.feature_db_path}")

    def save_feature_set(self,
                         features_data: Union[pd.DataFrame, Dict],
                         feature_set_name: str,
                         version_id: str = None,
                         metadata: Dict = None) -> str:
        """
        保存特征集合

        Args:
            features_data: 特征数据
            feature_set_name: 特征集名称
            version_id: 版本ID，如果为None则自动生成
            metadata: 元数据

        Returns:
            str: 版本ID
        """
        if version_id is None:
            version_id = f"v380_{feature_set_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            # 处理特征数据
            if isinstance(features_data, dict):
                # 如果是字典格式，转换为DataFrame
                features_df = pd.DataFrame([features_data])
            else:
                features_df = features_data.copy()

            # 计算特征哈希
            feature_names = list(features_df.columns)
            data_hash = self._compute_data_hash(features_df)

            # 确保存储目录存在
            version_dir = self.base_dir / version_id
            version_dir.mkdir(exist_ok=True)

            # 保存特征数据
            storage_path = version_dir / f"{feature_set_name}.pkl"

            # 使用pickle保存并压缩
            with open(storage_path, 'wb') as f:
                pickle.dump(features_df, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 记录到数据库
            conn = sqlite3.connect(self.feature_db_path)
            cursor = conn.cursor()

            cursor.execute("""
            INSERT OR REPLACE INTO feature_versions
            (version_id, feature_set_name, feature_count, feature_names, data_hash, storage_path, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                version_id,
                feature_set_name,
                len(feature_names),
                json.dumps(feature_names),
                data_hash,
                str(storage_path),
                json.dumps(metadata or {})
            ))

            conn.commit()
            conn.close()

            self.logger.info(f"💾 保存特征集 {feature_set_name} (版本: {version_id}): {len(feature_names)}个特征")
            return version_id

        except Exception as e:
            self.logger.error(f"❌ 保存特征集失败: {e}")
            return None

    def load_feature_set(self, version_id: str, feature_set_name: str) -> Optional[pd.DataFrame]:
        """
        加载特征集合

        Args:
            version_id: 版本ID
            feature_set_name: 特征集名称

        Returns:
            pd.DataFrame: 特征数据
        """
        cache_key = f"{version_id}_{feature_set_name}"

        # 检查内存缓存
        if cache_key in self.feature_cache:
            self.logger.info(f"📊 从缓存加载特征集: {cache_key}")
            return self.feature_cache[cache_key]

        try:
            # 从数据库获取存储路径
            conn = sqlite3.connect(self.feature_db_path)
            cursor = conn.cursor()

            cursor.execute("""
            SELECT storage_path FROM feature_versions
            WHERE version_id = ? AND feature_set_name = ?
            """, (version_id, feature_set_name))

            result = cursor.fetchone()
            conn.close()

            if not result:
                self.logger.warning(f"⚠️ 未找到特征集: {version_id}/{feature_set_name}")
                return None

            storage_path = Path(result[0])

            if not storage_path.exists():
                self.logger.error(f"❌ 特征文件不存在: {storage_path}")
                return None

            # 加载特征数据
            with open(storage_path, 'rb') as f:
                features_df = pickle.load(f)

            # 添加到缓存
            self._add_to_cache(cache_key, features_df)

            self.logger.info(f"📊 加载特征集 {feature_set_name} (版本: {version_id}): {len(features_df.columns)}个特征")
            return features_df

        except Exception as e:
            self.logger.error(f"❌ 加载特征集失败: {e}")
            return None

    def save_incremental_features(self,
                                code: str,
                                trade_date: str,
                                features: Dict[str, float],
                                version_id: str) -> bool:
        """
        保存增量特征数据

        Args:
            code: 股票代码
            trade_date: 交易日期
            features: 特征字典
            version_id: 版本ID

        Returns:
            bool: 是否保存成功
        """
        try:
            # 计算特征哈希
            feature_hash = self._compute_feature_hash(features)

            # 确保增量存储目录存在
            incremental_dir = self.base_dir / version_id / "incremental"
            incremental_dir.mkdir(parents=True, exist_ok=True)

            # 生成文件路径
            file_name = f"{code}_{trade_date}.pkl"
            file_path = incremental_dir / file_name

            # 保存特征数据
            with open(file_path, 'wb') as f:
                pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 记录到索引表
            conn = sqlite3.connect(self.feature_db_path)
            cursor = conn.cursor()

            cursor.execute("""
            INSERT OR REPLACE INTO feature_index
            (version_id, code, trade_date, feature_hash, file_path)
            VALUES (?, ?, ?, ?, ?)
            """, (version_id, code, trade_date, feature_hash, str(file_path)))

            conn.commit()
            conn.close()

            self.logger.debug(f"💾 保存增量特征 {code} {trade_date}: {len(features)}个特征")
            return True

        except Exception as e:
            self.logger.error(f"❌ 保存增量特征失败: {e}")
            return False

    def load_incremental_features(self,
                                code: str,
                                trade_date: str,
                                version_id: str) -> Optional[Dict[str, float]]:
        """
        加载增量特征数据

        Args:
            code: 股票代码
            trade_date: 交易日期
            version_id: 版本ID

        Returns:
            Dict: 特征字典
        """
        try:
            # 从索引表查找文件路径
            conn = sqlite3.connect(self.feature_db_path)
            cursor = conn.cursor()

            cursor.execute("""
            SELECT file_path FROM feature_index
            WHERE version_id = ? AND code = ? AND trade_date = ?
            """, (version_id, code, trade_date))

            result = cursor.fetchone()
            conn.close()

            if not result:
                return None

            file_path = Path(result[0])

            if not file_path.exists():
                self.logger.warning(f"⚠️ 增量特征文件不存在: {file_path}")
                return None

            # 加载特征数据
            with open(file_path, 'rb') as f:
                features = pickle.load(f)

            return features

        except Exception as e:
            self.logger.error(f"❌ 加载增量特征失败: {e}")
            return None

    def get_feature_versions(self, feature_set_name: str = None) -> List[Dict]:
        """获取特征版本列表"""
        try:
            conn = sqlite3.connect(self.feature_db_path)
            cursor = conn.cursor()

            if feature_set_name:
                cursor.execute("""
                SELECT version_id, feature_set_name, feature_count, created_at, metadata
                FROM feature_versions
                WHERE feature_set_name = ?
                ORDER BY created_at DESC
                """, (feature_set_name,))
            else:
                cursor.execute("""
                SELECT version_id, feature_set_name, feature_count, created_at, metadata
                FROM feature_versions
                ORDER BY created_at DESC
                """)

            results = cursor.fetchall()
            conn.close()

            versions = []
            for row in results:
                versions.append({
                    'version_id': row[0],
                    'feature_set_name': row[1],
                    'feature_count': row[2],
                    'created_at': row[3],
                    'metadata': json.loads(row[4]) if row[4] else {}
                })

            return versions

        except Exception as e:
            self.logger.error(f"❌ 获取特征版本失败: {e}")
            return []

    def cleanup_old_versions(self, keep_versions: int = 5):
        """清理旧版本特征数据"""
        try:
            versions = self.get_feature_versions()

            if len(versions) <= keep_versions:
                self.logger.info(f"📊 特征版本数量({len(versions)})未超过保留数量({keep_versions})")
                return

            # 获取需要删除的版本
            versions_to_delete = versions[keep_versions:]

            conn = sqlite3.connect(self.feature_db_path)
            cursor = conn.cursor()

            deleted_count = 0
            for version in versions_to_delete:
                version_id = version['version_id']

                # 删除文件
                version_dir = self.base_dir / version_id
                if version_dir.exists():
                    import shutil
                    shutil.rmtree(version_dir)

                # 删除数据库记录
                cursor.execute("DELETE FROM feature_index WHERE version_id = ?", (version_id,))
                cursor.execute("DELETE FROM feature_versions WHERE version_id = ?", (version_id,))

                deleted_count += 1

            conn.commit()
            conn.close()

            self.logger.info(f"🗑️ 清理旧特征版本: {deleted_count}个")

        except Exception as e:
            self.logger.error(f"❌ 清理旧版本失败: {e}")

    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        try:
            stats = {
                'total_versions': 0,
                'total_feature_sets': 0,
                'total_incremental_records': 0,
                'storage_size_mb': 0,
                'cache_size': len(self.feature_cache)
            }

            # 数据库统计
            conn = sqlite3.connect(self.feature_db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM feature_versions")
            stats['total_versions'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM feature_index")
            stats['total_incremental_records'] = cursor.fetchone()[0]

            conn.close()

            # 存储大小统计
            total_size = 0
            if self.base_dir.exists():
                for file_path in self.base_dir.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size

            stats['storage_size_mb'] = total_size / 1024 / 1024

            return stats

        except Exception as e:
            self.logger.error(f"❌ 获取存储统计失败: {e}")
            return {}

    def _compute_data_hash(self, data: pd.DataFrame) -> str:
        """计算数据哈希"""
        return hashlib.md5(pd.util.hash_pandas_object(data, index=True).values).hexdigest()

    def _compute_feature_hash(self, features: Dict[str, float]) -> str:
        """计算特征哈希"""
        features_str = json.dumps(features, sort_keys=True)
        return hashlib.md5(features_str.encode()).hexdigest()

    def _add_to_cache(self, cache_key: str, data: pd.DataFrame):
        """添加到缓存"""
        # 如果缓存超过最大大小，删除最老的条目
        if len(self.feature_cache) >= self.cache_max_size:
            # 删除第一个条目（最老的）
            oldest_key = next(iter(self.feature_cache))
            del self.feature_cache[oldest_key]

        self.feature_cache[cache_key] = data

    def clear_cache(self):
        """清理内存缓存"""
        cache_count = len(self.feature_cache)
        self.feature_cache.clear()
        self.logger.info(f"🗑️ 清理特征缓存: {cache_count}条记录")