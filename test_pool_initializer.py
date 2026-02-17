#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Pool initializer机制
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from multiprocessing import Pool
import time

# 全局变量
_worker_id = None

def init_worker(worker_id):
    """Worker初始化函数"""
    global _worker_id
    _worker_id = worker_id
    print(f"Worker {worker_id} initialized", flush=True)

def process_task(task_id):
    """处理单个任务"""
    global _worker_id
    result = f"Task {task_id} processed by worker {_worker_id}"
    print(result, flush=True)
    return result

def main():
    print("=" * 50)
    print("测试Pool initializer机制")
    print("=" * 50)

    tasks = list(range(20))
    results = []

    print(f"\n启动4个worker进程...")
    with Pool(processes=4, initializer=init_worker, initargs=("worker",)) as pool:
        print("Pool创建成功，开始处理任务...")

        # 使用imap_unordered
        for i, result in enumerate(pool.imap_unordered(process_task, tasks, chunksize=5)):
            results.append(result)
            if (i + 1) % 5 == 0:
                print(f"进度: {i+1}/{len(tasks)}", flush=True)

    print(f"\n✅ 完成! 处理了 {len(results)} 个任务")
    print("前5个结果:")
    for r in results[:5]:
        print(f"  - {r}")

if __name__ == "__main__":
    main()
