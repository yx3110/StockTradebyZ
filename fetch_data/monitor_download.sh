#!/bin/bash

LOG_FILE="../logs/historical_download.log"

if [ -f historical_download.pid ]; then
    PID=$(cat historical_download.pid)
    if ps -p $PID > /dev/null; then
        echo "下载进程 $PID 正在运行"
        echo "最近进度:"
        tail -10 $LOG_FILE | grep -E "(成功下载|批次完成|总成功)"
    else
        echo "下载进程 $PID 已结束"
        echo "最终状态:"
        tail -5 $LOG_FILE
    fi
else
    echo "未找到下载进程PID文件"
fi
