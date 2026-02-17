#!/bin/bash

# 后台下载历史数据脚本
# 从2018年开始下载所有证券的历史数据

echo "开始后台下载历史数据..."
echo "下载日志将保存到 ../logs/historical_download.log"

# 创建日志文件
LOG_FILE="../logs/historical_download.log"
touch $LOG_FILE

# 记录开始时间
echo "$(date): 开始下载历史数据" >> $LOG_FILE

# 运行下载命令，参数优化
nohup python3 download_missing_data.py \
    --start 20180101 \
    --batch-size 30 \
    --workers 2 \
    >> $LOG_FILE 2>&1 &

# 获取进程ID
PID=$!
echo "下载进程ID: $PID"
echo "$(date): 下载进程启动，PID: $PID" >> $LOG_FILE

# 保存PID到文件
echo $PID > historical_download.pid

echo "历史数据下载已在后台启动"
echo "监控进度: tail -f $LOG_FILE"
echo "查看进程: ps aux | grep download_missing_data"
echo "停止下载: kill $PID"

# 创建监控脚本
cat > monitor_download.sh << 'EOF'
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
EOF

chmod +x monitor_download.sh
echo "监控脚本已创建: ./monitor_download.sh"