#!/usr/bin/env python3
"""训练数据 joblib 缓存清理 (ml_models/trained_models/cache/data_*.joblib)

缓存 key 掺整库 mtime, 每次 DB 写入即失效 → 旧文件永远不会再被命中, 只占磁盘
(2026-07 实测 73GB/46 个文件, 绝大部分是 4 月实验残留)。重建成本 = 一次冷加载 ~2-3min。

默认 dry-run 只列清单; --apply 才真删。保留策略: mtime 距今 <= --keep-days (默认 14) 的一律保留。
"""
import argparse
import time
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / 'ml_models' / 'trained_models' / 'cache'


def main():
    parser = argparse.ArgumentParser(description='清理过期的训练数据 joblib 缓存')
    parser.add_argument('--keep-days', type=int, default=14,
                        help='保留 mtime 距今 N 天以内的缓存 (默认 14)')
    parser.add_argument('--apply', action='store_true',
                        help='真删 (默认 dry-run 只打印清单)')
    args = parser.parse_args()

    if not CACHE_DIR.exists():
        print(f'缓存目录不存在: {CACHE_DIR}')
        return

    now = time.time()
    cutoff = now - args.keep_days * 86400
    files = sorted(CACHE_DIR.glob('data_*.joblib'), key=lambda p: p.stat().st_mtime)

    keep, drop = [], []
    for f in files:
        (keep if f.stat().st_mtime > cutoff else drop).append(f)

    def fmt(f):
        st = f.stat()
        age_d = (now - st.st_mtime) / 86400
        return f'  {f.name}  {st.st_size / 1e9:.1f}GB  {age_d:.0f} 天前'

    drop_gb = sum(f.stat().st_size for f in drop) / 1e9
    keep_gb = sum(f.stat().st_size for f in keep) / 1e9
    print(f'缓存目录: {CACHE_DIR}')
    print(f'保留 (mtime <= {args.keep_days} 天, {len(keep)} 个, {keep_gb:.1f}GB):')
    for f in keep:
        print(fmt(f))
    print(f'待删 ({len(drop)} 个, {drop_gb:.1f}GB):')
    for f in drop:
        print(fmt(f))

    if not drop:
        print('无可清理文件')
        return
    if args.apply:
        for f in drop:
            f.unlink()
        print(f'\n✅ 已删除 {len(drop)} 个文件, 释放 {drop_gb:.1f}GB')
    else:
        print(f'\n(dry-run — 加 --apply 才真删, 预计释放 {drop_gb:.1f}GB)')


if __name__ == '__main__':
    main()
