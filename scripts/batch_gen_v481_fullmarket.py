#!/usr/bin/env python3
"""批量生成 V4.8.1 全市场报告 — 用ThreadPool并行调subprocess"""
import subprocess, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

dates = [
    "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09",
    "2026-01-12", "2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16",
    "2026-01-19", "2026-01-20", "2026-01-21", "2026-01-22", "2026-01-23",
    "2026-01-26", "2026-01-27", "2026-01-28", "2026-01-29", "2026-01-30",
    "2026-02-02", "2026-02-03", "2026-02-04", "2026-02-05", "2026-02-06",
    "2026-02-09", "2026-02-10", "2026-02-11", "2026-02-12", "2026-02-13",
]

existing = set()
report_dir = 'reports/daily_selection_v4.8.1_fullmarket'
for f in os.listdir(report_dir):
    if f.startswith('选股分析报告_') and f.endswith('.md'):
        d = f.replace('选股分析报告_', '').replace('.md', '')
        existing.add(f'{d[:4]}-{d[4:6]}-{d[6:8]}')

todo = [d for d in dates if d not in existing]
print(f'已有: {len(existing)}, 待生成: {len(todo)}')
if not todo:
    print('全部完成!')
    sys.exit(0)

def gen(date):
    r = subprocess.run(
        ['python3', 'tomorrow_stock_selector.py', date,
         '--scoring-version', 'v4.8.1', '--full-market'],
        capture_output=True, text=True, timeout=300
    )
    ok = '选股分析完成' in r.stderr or '选股分析完成' in r.stdout
    return date, ok

t0 = time.time()
done = 0
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = {pool.submit(gen, d): d for d in todo}
    for fut in as_completed(futures):
        date, ok = fut.result()
        done += 1
        elapsed = time.time() - t0
        eta = elapsed / done * (len(todo) - done)
        print(f'  [{done}/{len(todo)}] {date} {"✓" if ok else "✗"} ({elapsed:.0f}s, ETA {eta:.0f}s)', flush=True)

print(f'\n完成! {done}/{len(todo)} 天, {time.time()-t0:.0f}s')
