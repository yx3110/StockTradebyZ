#!/usr/bin/env python3
"""
Persistent cache for North Star evaluation pipeline.

Three cache types with distinct invalidation:
- Static: invalidated when DB changes (daily_quotes max_date + row_count)
- Report: invalidated when report directory contents change (file list hash)
- Result: invalidated when backtest params change (param hash)
"""
import os
import sys
import hashlib
import sqlite3
import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
DEFAULT_CACHE_DIR = PROJECT_ROOT / 'backtest' / '.eval_cache'


class EvalCache:
    """Persistent cache manager for North Star evaluation data."""

    def __init__(self, cache_dir: Path = None, db_path: str = None):
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.db_path = db_path or DEFAULT_DB_PATH
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._db_version = None

    # -- Version Keys -------------------------------------------

    def _get_db_version(self) -> str:
        """DB version = max_trade_date + row_count from daily_quotes."""
        if self._db_version is not None:
            return self._db_version
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA busy_timeout = 30000")
        try:
            row = conn.execute(
                "SELECT MAX(trade_date), COUNT(*) FROM daily_quotes"
            ).fetchone()
            self._db_version = f"{row[0]}_{row[1]}"
        finally:
            conn.close()
        return self._db_version

    @staticmethod
    def _get_report_dir_version(report_dir: str) -> str:
        """Hash of sorted (filename, mtime) pairs in report directory."""
        report_path = Path(report_dir)
        if not report_path.exists():
            return "empty"
        entries = []
        for f in sorted(report_path.glob('analysis_data_*.json')):
            entries.append(f"{f.name}:{f.stat().st_mtime_ns}")
        if not entries:
            return "empty"
        h = hashlib.md5('\n'.join(entries).encode()).hexdigest()[:12]
        return h

    @staticmethod
    def _get_param_hash(**params) -> str:
        """Hash of sorted backtest parameters."""
        import json
        key_str = json.dumps(params, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    # -- Generic Cache Operations -------------------------------

    def _cache_path(self, name: str) -> Path:
        return self.cache_dir / name

    def load(self, name: str, version: str) -> Optional[Any]:
        """Load cached data if version matches."""
        meta_path = self._cache_path(f"{name}.meta")
        data_path = self._cache_path(f"{name}.pkl")
        if not meta_path.exists() or not data_path.exists():
            return None
        with open(meta_path, 'r') as f:
            cached_version = f.read().strip()
        if cached_version != version:
            return None
        with open(data_path, 'rb') as f:
            return pickle.load(f)

    def save(self, name: str, version: str, data: Any):
        """Save data with version marker."""
        meta_path = self._cache_path(f"{name}.meta")
        data_path = self._cache_path(f"{name}.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(meta_path, 'w') as f:
            f.write(version)

    def load_npz(self, name: str, version: str) -> Optional[Dict]:
        """Load numpy arrays if version matches."""
        meta_path = self._cache_path(f"{name}.meta")
        data_path = self._cache_path(f"{name}.npz")
        if not meta_path.exists() or not data_path.exists():
            return None
        with open(meta_path, 'r') as f:
            cached_version = f.read().strip()
        if cached_version != version:
            return None
        return dict(np.load(data_path, allow_pickle=True))

    def save_npz(self, name: str, version: str, **arrays):
        """Save numpy arrays with version marker."""
        meta_path = self._cache_path(f"{name}.meta")
        data_path = self._cache_path(f"{name}.npz")
        np.savez_compressed(data_path, **arrays)
        with open(meta_path, 'w') as f:
            f.write(version)

    # -- High-Level Cache API -----------------------------------

    def get_future_returns(self, report_dates: list, loader_fn) -> dict:
        """Load future returns from cache or compute via loader_fn."""
        db_ver = self._get_db_version()
        dates_hash = hashlib.md5(','.join(sorted(report_dates)).encode()).hexdigest()[:8]
        name = f"future_returns_{dates_hash}"
        version = db_ver

        cached = self.load(name, version)
        if cached is not None:
            return cached

        data = loader_fn(report_dates)
        self.save(name, version, data)
        return data

    def get_parsed_reports(self, report_dir: str, rank_field: str, loader_fn) -> dict:
        """Load parsed reports from cache or compute via loader_fn."""
        dir_ver = self._get_report_dir_version(report_dir)
        name = f"reports_{hashlib.md5(report_dir.encode()).hexdigest()[:8]}_{rank_field}"
        version = dir_ver

        cached = self.load(name, version)
        if cached is not None:
            return cached

        data = loader_fn(report_dir, rank_field)
        self.save(name, version, data)
        return data

    def get_metric_data(self, buy_dates: list, loader_fn) -> tuple:
        """Load metric data (market_cap, limit_up, median_cap) from cache."""
        db_ver = self._get_db_version()
        dates_hash = hashlib.md5(','.join(sorted(buy_dates)).encode()).hexdigest()[:8]
        name = f"metric_data_{dates_hash}"
        version = db_ver

        cached = self.load(name, version)
        if cached is not None:
            return cached

        data = loader_fn(buy_dates)
        self.save(name, version, data)
        return data

    def get_trading_dates_map(self, report_dates: list, loader_fn) -> dict:
        """Load next-trading-date map from cache."""
        db_ver = self._get_db_version()
        dates_hash = hashlib.md5(','.join(sorted(report_dates)).encode()).hexdigest()[:8]
        name = f"trading_dates_{dates_hash}"
        version = db_ver

        cached = self.load(name, version)
        if cached is not None:
            return cached

        data = loader_fn(report_dates)
        self.save(name, version, data)
        return data

    def clear(self):
        """Remove all cached files."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Cache cleared: {self.cache_dir}")

    def _list_entries(self) -> list:
        """按缓存条目 (stem) 分组列出文件: [{'stem', 'paths', 'bytes', 'mtime'}], 旧的在前."""
        if not self.cache_dir.exists():
            return []
        groups: Dict[str, dict] = {}
        for f in self.cache_dir.iterdir():
            if not f.is_file():
                continue
            stem = f.name
            for suffix in ('.pkl', '.npz', '.meta'):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            st = f.stat()
            g = groups.setdefault(stem, {'stem': stem, 'paths': [], 'bytes': 0, 'mtime': 0.0})
            g['paths'].append(f)
            g['bytes'] += st.st_size
            g['mtime'] = max(g['mtime'], st.st_mtime)
        return sorted(groups.values(), key=lambda g: g['mtime'])

    def prune(self, max_age_days: float = None, max_total_gb: float = None,
              dry_run: bool = True) -> dict:
        """LRU/age 淘汰机制: DB 每日更新后 db_version 变化, 旧版本 pkl 永不再命中,
        无淘汰会无限膨胀 (实测 88GB)。

        Args:
            max_age_days: 淘汰 mtime 距今超过 N 天的条目
            max_total_gb: 总大小预算 (GB), 超出部分从最旧条目 (LRU) 开始淘汰
            dry_run: True (默认) 只打印将删除的文件与释放空间, 不实际删除

        Returns:
            {'n_entries', 'n_prune', 'free_mb', 'total_mb', 'dry_run'}
        """
        entries = self._list_entries()  # 已按 mtime 从旧到新排序
        total_bytes = sum(e['bytes'] for e in entries)
        now = time.time()

        to_prune, prune_stems = [], set()
        # 1) age 淘汰
        if max_age_days is not None:
            cutoff = now - max_age_days * 86400
            for e in entries:
                if e['mtime'] < cutoff:
                    to_prune.append(e)
                    prune_stems.add(e['stem'])
        # 2) 容量预算 LRU 淘汰 (在 age 淘汰之后仍超预算时, 继续删最旧的)
        if max_total_gb is not None:
            budget = max_total_gb * 1024 ** 3
            remaining = total_bytes - sum(e['bytes'] for e in to_prune)
            for e in entries:
                if remaining <= budget:
                    break
                if e['stem'] in prune_stems:
                    continue
                to_prune.append(e)
                prune_stems.add(e['stem'])
                remaining -= e['bytes']

        free_bytes = sum(e['bytes'] for e in to_prune)
        mode = 'DRY-RUN (不删除)' if dry_run else 'APPLY (实际删除)'
        print(f"  [{mode}] 缓存 {len(entries)} 条目 / {total_bytes / 1024**2:.0f} MB, "
              f"计划淘汰 {len(to_prune)} 条目 / 释放 {free_bytes / 1024**2:.0f} MB")
        for e in sorted(to_prune, key=lambda x: -x['bytes']):
            age_days = (now - e['mtime']) / 86400
            print(f"    - {e['stem']:<50} {e['bytes'] / 1024**2:>9.1f} MB  {age_days:>6.1f} 天前")
            if not dry_run:
                for p in e['paths']:
                    try:
                        p.unlink()
                    except OSError as err:
                        print(f"      ⚠️ 删除失败 {p.name}: {err}")

        return {
            'n_entries': len(entries),
            'n_prune': len(to_prune),
            'free_mb': round(free_bytes / 1024 / 1024, 1),
            'total_mb': round(total_bytes / 1024 / 1024, 1),
            'dry_run': dry_run,
        }

    def stats(self) -> dict:
        """Return cache statistics."""
        if not self.cache_dir.exists():
            return {'n_files': 0, 'total_size_mb': 0}
        files = list(self.cache_dir.iterdir())
        total_bytes = sum(f.stat().st_size for f in files if f.is_file())
        return {
            'n_files': len(files),
            'total_size_mb': round(total_bytes / 1024 / 1024, 1),
        }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='EvalCache 维护工具 — 默认 dry-run, 必须显式加 --apply 才实际删除')
    parser.add_argument('--cache-dir', default=None, help='缓存目录 (默认 backtest/.eval_cache)')
    parser.add_argument('--stats', action='store_true', help='打印缓存统计')
    parser.add_argument('--prune-days', type=float, default=None,
                        help='淘汰 N 天前的缓存条目 (mtime)')
    parser.add_argument('--max-gb', type=float, default=None,
                        help='总大小预算 GB, 超出部分按 LRU (最旧优先) 淘汰')
    parser.add_argument('--apply', action='store_true',
                        help='实际执行删除 (默认 dry-run 只打印)')
    cli_args = parser.parse_args()

    ec = EvalCache(cache_dir=cli_args.cache_dir)
    if cli_args.stats or (cli_args.prune_days is None and cli_args.max_gb is None):
        s = ec.stats()
        print(f"  缓存目录: {ec.cache_dir}")
        print(f"  文件数: {s['n_files']}, 总大小: {s['total_size_mb']} MB")
    if cli_args.prune_days is not None or cli_args.max_gb is not None:
        ec.prune(max_age_days=cli_args.prune_days, max_total_gb=cli_args.max_gb,
                 dry_run=not cli_args.apply)
