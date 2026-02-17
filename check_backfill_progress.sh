#!/bin/bash
# 数据补全进度监控脚本

echo "=== V3.9 数据补全进度 ==="
echo ""

# 检查进程
if pgrep -f "backfill_v39_parallel.py" > /dev/null; then
    echo "✅ 任务运行中"
else
    echo "❌ 任务未运行"
fi

echo ""

# 检查最新进度
echo "📊 最新进度:"
grep "进度:" /tmp/backfill_2024_2025.log 2>/dev/null | tail -3

echo ""
echo "📈 数据库统计:"
sqlite3 data_adapter/stock_data.db "
SELECT '2024年: ' || COUNT(*) || ' 条'
FROM v39_feature_cache
WHERE trade_date BETWEEN '2024-01-01' AND '2024-12-31'
UNION ALL
SELECT '2025年1-5月: ' || COUNT(*) || ' 条'
FROM v39_feature_cache
WHERE trade_date BETWEEN '2025-01-01' AND '2025-05-31'
UNION ALL
SELECT '总计: ' || COUNT(*) || ' 条'
FROM v39_feature_cache
"

echo ""
echo "运行 'tail -f /tmp/backfill_2024_2025.log' 查看实时日志"
