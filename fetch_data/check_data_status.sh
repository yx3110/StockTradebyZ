#!/bin/bash
# 快速检查数据状态脚本

echo "=================================================="
echo "           📊 股票数据状态检查"
echo "=================================================="
echo ""

# 检查数据追踪器是否存在
if [ -f "data_update_tracker.py" ]; then
    echo "🔍 正在检查数据状态..."
    python3 data_update_tracker.py --check
    echo ""
else
    echo "❌ 数据追踪器未找到"
    exit 1
fi

# 检查更新标记文件
MARKER_FILE="../full_securities_data/LAST_UPDATE.txt"
if [ -f "$MARKER_FILE" ]; then
    echo "📄 更新标记文件信息:"
    echo "$(head -2 "$MARKER_FILE")"
    echo ""
else
    echo "⚠️  未找到更新标记文件"
    echo ""
fi

# 检查最近几个数据文件的日期
echo "📈 抽样检查数据文件最新日期:"
echo "--------------------------------------------"
for file in ../full_securities_data/00000{1,2,8}_A股.csv; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        last_date=$(tail -1 "$file" | cut -d',' -f1)
        echo "  $filename: $last_date"
    fi
done
echo ""

# 快速统计
echo "📊 数据文件统计:"
echo "--------------------------------------------"
csv_count=$(ls ../full_securities_data/*.csv 2>/dev/null | wc -l | tr -d ' ')
echo "  CSV文件总数: $csv_count"

if [ -d "../full_securities_data" ]; then
    data_size=$(du -sh ../full_securities_data | cut -f1)
    echo "  数据目录大小: $data_size"
fi

echo ""
echo "💡 常用命令提示:"
echo "  查看详细报告: python3 data_update_tracker.py --report"
echo "  查看更新历史: python3 data_update_tracker.py --history"
echo "  更新数据: ./run_daily_update.sh -m update"
echo ""
echo "=================================================="