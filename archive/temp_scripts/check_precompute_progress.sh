#!/bin/bash
# 预计算进度查看脚本

echo "========================================"
echo "V3.9 特征预计算进度监控"
echo "========================================"
echo ""

# 检查进程状态
if ps aux | grep "precompute_v39_features.py" | grep -v grep > /dev/null; then
    echo "✅ 预计算进程正在运行"
else
    echo "⚠️  预计算进程未运行"
fi

echo ""
echo "📊 最新进度："
tail -3 /tmp/precompute_1000.log | grep "进度:"

echo ""
echo "💾 数据库统计："
sqlite3 data_adapter/stock_data.db "SELECT COUNT(*) || ' 个样本' as total, COUNT(DISTINCT code) || ' 只股票' as stocks FROM v39_feature_cache;"

echo ""
echo "📁 完整日志: /tmp/precompute_1000.log"
echo "========================================"
