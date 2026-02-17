#!/bin/bash

# 监控回测进度脚本

echo "======================================================================================================"
echo "📊 ML模型回测进度监控"
echo "======================================================================================================"
echo ""
echo "⏰ 当前时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "🔍 监控回测进度..."
echo ""

# 检查最新的回测报告
REPORT_DIR="reports/backtest"
echo "📁 检查报告目录: $REPORT_DIR"

if [ -d "$REPORT_DIR" ]; then
    LATEST_JSON=$(ls -t $REPORT_DIR/ml_versions_comparison_*.json 2>/dev/null | head -1)
    LATEST_MD=$(ls -t $REPORT_DIR/ml_versions_comparison_*.md 2>/dev/null | head -1)

    if [ -n "$LATEST_JSON" ]; then
        echo "✅ 找到最新JSON报告: $(basename $LATEST_JSON)"
        echo ""
        echo "📊 回测结果快速预览:"
        echo "------------------------------------------------------------------------------------------------------"
        python3 << EOF
import json
try:
    with open('$LATEST_JSON', 'r') as f:
        data = json.load(f)

    print(f"测试期间: {data.get('test_period', 'N/A')}")
    print(f"测试版本: {', '.join(data.get('versions_tested', []))}")
    print("")

    results = data.get('individual_results', {})
    for version, result in results.items():
        if 'error' not in result:
            print(f"{version}:")
            print(f"  总收益率: {result.get('total_return', 0)*100:.2f}%")
            print(f"  年化收益: {result.get('annual_return', 0)*100:.2f}%")
            print(f"  夏普比率: {result.get('sharpe_ratio', 0):.2f}")
            print(f"  最大回撤: {result.get('max_drawdown', 0)*100:.2f}%")
            print(f"  交易次数: {result.get('total_trades', 0)}")
            print("")
        else:
            print(f"{version}: ❌ 失败 - {result.get('error', 'Unknown error')}")
            print("")

    analysis = data.get('comparison_analysis', {})
    if 'best_performance' in analysis and analysis['best_performance'].get('version'):
        print("🏆 最佳表现:")
        best = analysis['best_performance']
        print(f"  版本: {best['version']}")
        print(f"  收益率: {best['return']*100:.2f}%")

except Exception as e:
    print(f"❌ 读取报告失败: {e}")
EOF
        echo "------------------------------------------------------------------------------------------------------"
    else
        echo "⏳ 回测仍在进行中，暂无报告生成..."
    fi

    if [ -n "$LATEST_MD" ]; then
        echo ""
        echo "📄 Markdown报告: $(basename $LATEST_MD)"
        echo "   查看详情: cat $LATEST_MD"
    fi
else
    echo "❌ 报告目录不存在"
fi

echo ""
echo "======================================================================================================"
