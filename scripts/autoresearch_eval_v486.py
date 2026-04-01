#!/usr/bin/env python3
"""V4.8.6 autoresearch 评估脚本 — 输出单个数字(北极星V4加权分)到stdout"""
import sys, subprocess, re

result = subprocess.run(
    [sys.executable, 'backtest/run_north_star_eval.py',
     '--backtest',
     '--report-dir', 'reports/daily_selection_v4.8.6',
     '--label', 'V4.8.6',
     '--top-n', '10',
     '--focus-days', '10',
     '--rank-field', 'composite'],
    capture_output=True, text=True, timeout=600
)

output = result.stdout + result.stderr

# 提取V4加权评分
match = re.search(r'北极星评分卡 V4.*?加权评分:\s*([\d.]+)%', output, re.DOTALL)
if match:
    print(match.group(1))
else:
    # fallback: 找最后一个加权评分
    matches = re.findall(r'加权评分:\s*([\d.]+)%', output)
    if matches:
        print(matches[-1])
    else:
        print("0")
        sys.exit(1)
