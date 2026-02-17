#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试V390系统在Pool中的运行
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from multiprocessing import Pool
from ml_models.v39 import V390EnhancedFeatureMLSystem

# 全局变量
_worker_system = None

def init_worker(lookback_days, lookahead_days):
    """Worker初始化函数"""
    global _worker_system
    print(f"[Worker] 开始初始化 V390系统 (lookback={lookback_days}, lookahead={lookahead_days})", flush=True)
    _worker_system = V390EnhancedFeatureMLSystem(
        lookback_days=lookback_days,
        lookahead_days=lookahead_days
    )
    print(f"[Worker] V390系统初始化完成", flush=True)

def process_sample(args):
    """处理单个样本"""
    code, date = args
    global _worker_system

    try:
        print(f"[Worker] 处理 {code} @ {date}", flush=True)
        features = _worker_system.extract_features(code, date)
        if features is None or features.empty:
            return None

        label = _worker_system.calculate_label(code, date)
        if label is None:
            return None

        return (features.iloc[0].to_dict(), label, {'code': code, 'date': date})
    except Exception as e:
        print(f"[Worker] 错误 {code} @ {date}: {e}", flush=True)
        return None

def main():
    print("=" * 50)
    print("测试V390系统在Pool中的运行")
    print("=" * 50)

    # 准备少量测试任务（使用有足够未来数据的日期）
    tasks = [
        ('000001', '2025-10-20'),
        ('600000', '2025-10-20'),
        ('000002', '2025-10-20'),
        ('600036', '2025-10-20'),
        ('000858', '2025-10-20'),
    ]

    print(f"\n启动2个worker进程...")
    results = []

    with Pool(processes=2, initializer=init_worker, initargs=(10, 5)) as pool:
        print("Pool创建成功，开始处理任务...", flush=True)

        for i, result in enumerate(pool.imap_unordered(process_sample, tasks, chunksize=2)):
            if result is not None:
                results.append(result)
            print(f"进度: {i+1}/{len(tasks)}", flush=True)

    print(f"\n✅ 完成! 处理了 {len(results)} 个有效样本 / {len(tasks)} 个任务")

if __name__ == "__main__":
    main()
