#!/bin/bash
# v3.9完整数据抓取脚本
# 时间范围: 2024-01-01 ~ 现在
# 预计耗时: 1-2小时

echo "🚀 启动v3.9完整数据抓取"
echo "时间范围: 2024-01-01 ~ $(date +%Y%m%d)"
echo "预计耗时: 1-2小时"
echo ""

# 后台运行，输出到日志
nohup python3 fetch_data/v39_data_initializer.py \
  --start-date 20240101 \
  --end-date $(date +%Y%m%d) \
  --steps 2,3 \
  > logs/v39_full_fetch_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!
echo "✅ 后台进程已启动: PID=$PID"
echo "📋 进度监控命令:"
echo "   tail -f logs/v39_full_fetch_*.log | grep -E '(进度|完成|成功|失败)'"
echo ""
echo "📊 数据库检查命令:"
echo "   sqlite3 stock_data.db 'SELECT COUNT(*) FROM daily_basic'"
echo "   sqlite3 stock_data.db 'SELECT COUNT(*) FROM financial_indicator'"
echo ""
echo "🛑 停止命令 (如果需要):"
echo "   kill $PID"
echo ""
echo "$PID" > logs/v39_fetch.pid
echo "✅ PID已保存到 logs/v39_fetch.pid"
