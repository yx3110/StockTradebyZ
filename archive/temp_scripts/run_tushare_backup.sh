#!/bin/bash

# Tushare备份数据更新脚本
# 在Tushare API维护期间使用此脚本更新数据

echo "=========================================="
echo "   Tushare备份数据更新工具"
echo "=========================================="
echo ""

# 激活虚拟环境（如果存在）
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 检查参数
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    python3 temp_scripts/tushare_backup_updater.py --help
    exit 0
fi

# 如果提供了日期参数，使用它；否则使用最新可用日期
if [ -n "$1" ]; then
    echo "📅 使用指定日期: $1"
    python3 temp_scripts/tushare_backup_updater.py --date $1 --type all
else
    echo "📅 使用最新可用日期"
    python3 temp_scripts/tushare_backup_updater.py --type all
fi

# 检查更新是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 数据更新成功！"
    echo ""
    echo "下一步操作建议："
    echo "1. 运行选股: python3 tomorrow_stock_selector.py"
    echo "2. 生成AI报告: python3 ai_enhanced_daily_report.py"
    echo "3. 获取交易建议: python3 trading_advisor.py"
else
    echo ""
    echo "❌ 数据更新失败，请检查日志"
fi