#!/usr/bin/env python3
"""
历史数据下载进度摘要
快速查看下载状态和预估完成时间
"""

import re
import os
from datetime import datetime, timedelta

def get_download_progress():
    """获取下载进度信息"""
    
    # 检查进程状态
    pid_file = "historical_download.pid"
    log_file = "../logs/historical_download.log"
    
    if not os.path.exists(pid_file):
        return "未找到下载进程PID文件"
    
    with open(pid_file, 'r') as f:
        pid = f.read().strip()
    
    # 检查进程是否运行
    import subprocess
    try:
        subprocess.check_output(['ps', '-p', pid])
        process_running = True
    except subprocess.CalledProcessError:
        process_running = False
    
    if not process_running:
        return f"下载进程 {pid} 已停止"
    
    # 分析日志文件
    if not os.path.exists(log_file):
        return "未找到下载日志文件"
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 提取关键信息
    total_securities = None
    completed_batches = 0
    current_batch = None
    success_count = 0
    failed_count = 0
    start_time = None
    latest_progress = None
    
    for line in lines:
        # 总数量
        if "个缺失证券" in line:
            match = re.search(r'(\d+) 个缺失证券', line)
            if match:
                total_securities = int(match.group(1))
        
        # 开始时间
        if "开始下载历史数据" in line and start_time is None:
            match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if match:
                start_time = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
        
        # 当前批次
        if "处理第" in line and "批" in line:
            match = re.search(r'处理第 (\d+) 批', line)
            if match:
                current_batch = int(match.group(1))
        
        # 成功下载
        if "成功下载:" in line:
            success_count += 1
        
        # 批次完成
        if "批次完成" in line:
            completed_batches += 1
        
        # 最新进度
        if "下载A股:" in line and "%" in line:
            latest_progress = line.strip()
    
    # 计算进度和预估时间
    current_time = datetime.now()
    
    result = f"""
📊 历史数据下载进度报告

🔄 进程状态: 运行中 (PID: {pid})
📈 总证券数量: {total_securities:,} 只
✅ 已下载: {success_count:,} 只
❌ 失败: {failed_count} 只

⏱️ 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S') if start_time else '未知'}
⏰ 当前时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')}
"""

    if start_time:
        elapsed = current_time - start_time
        if success_count > 0:
            rate = success_count / elapsed.total_seconds() * 3600  # 每小时下载数量
            remaining = total_securities - success_count
            estimated_hours = remaining / rate if rate > 0 else float('inf')
            estimated_completion = current_time + timedelta(hours=estimated_hours)
            
            result += f"""
⚡ 下载速度: {rate:.1f} 只/小时
📊 完成度: {success_count/total_securities*100:.1f}%
⏳ 预估剩余时间: {estimated_hours:.1f} 小时
🎯 预估完成时间: {estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}
"""

    if latest_progress:
        result += f"\n🔍 最新进度: {latest_progress.split(']')[-1].strip() if ']' in latest_progress else latest_progress}"
    
    return result

if __name__ == "__main__":
    print(get_download_progress())