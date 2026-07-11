#!/bin/bash
# AI增强每日选股系统 - 自动化运行脚本
# 集成量化选股、AI分析、情绪分析和交易建议生成

set -e  # 遇到错误立即退出
set -o pipefail  # 管道任一环节失败即整体失败 (否则 `python3 .. | tee` 只取 tee 的退出码, 掩盖 python 失败)

# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
DATE=$(date +%Y-%m-%d)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 创建日志目录
mkdir -p "$LOG_DIR"

# 日志文件
MAIN_LOG="$LOG_DIR/ai_enhanced_daily_$TIMESTAMP.log"
ERROR_LOG="$LOG_DIR/ai_enhanced_daily_error_$TIMESTAMP.log"

# 函数：记录日志
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

error_log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$ERROR_LOG" >&2
}

# 函数：检查命令是否成功
check_command() {
    if [ $? -eq 0 ]; then
        log "✅ $1 - 成功"
    else
        error_log "❌ $1 - 失败"
        return 1
    fi
}

# 函数：显示帮助信息
show_help() {
    cat << EOF
AI增强每日选股系统 - 使用说明

用法:
    $0 [选项]

选项:
    -h, --help          显示帮助信息
    -d, --date DATE     指定分析日期 (YYYY-MM-DD，默认今天)
    -m, --mode MODE     运行模式:
                        full    - 完整流程 (量化选股 + AI分析，默认)
                        ai-only - 仅AI分析 (需要已有量化结果)
                        update  - 仅数据更新
    --scoring-version   评分版本 (v3 或 v3.1，默认v3)
    -s, --skip-update   跳过数据更新
    -v, --verbose       详细输出

示例:
    $0                              # 运行完整流程
    $0 -d 2025-08-04                # 分析指定日期
    $0 --scoring-version v3.1       # 使用V3.1优化权重版
    $0 -m ai-only                   # 仅运行AI分析
    $0 -s                           # 跳过数据更新

生成的报告将保存在 reports/ 目录下。
EOF
}

# 默认参数
ANALYSIS_DATE="$DATE"
MODE="full"
SCORING_VERSION="v3"
SKIP_UPDATE=false
VERBOSE=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--date)
            ANALYSIS_DATE="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        --scoring-version)
            SCORING_VERSION="$2"
            shift 2
            ;;
        -s|--skip-update)
            SKIP_UPDATE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            error_log "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查Python环境
check_python() {
    log "🔍 检查Python环境..."
    
    if ! command -v python3 &> /dev/null; then
        error_log "Python3 未安装"
        return 1
    fi
    
    # 检查必要的Python包
    python3 -c "import pandas, numpy, requests, bs4" 2>/dev/null
    check_command "Python依赖包检查"
}

# 数据更新
update_data() {
    if [ "$SKIP_UPDATE" = true ]; then
        log "⏭️  跳过数据更新"
        return 0
    fi
    
    log "📊 开始数据更新..."
    
    # 运行快速数据更新
    if [ "$VERBOSE" = true ]; then
        python3 fetch_data/quick_daily_update.py --date "${ANALYSIS_DATE//-/}" | tee -a "$MAIN_LOG"
    else
        python3 fetch_data/quick_daily_update.py --date "${ANALYSIS_DATE//-/}" >> "$MAIN_LOG" 2>&1
    fi
    
    check_command "数据更新"
}

# 量化选股分析
run_quantitative_analysis() {
    log "🧮 开始量化选股分析 (版本: $SCORING_VERSION)..."
    
    if [ "$VERBOSE" = true ]; then
        python3 tomorrow_stock_selector.py "$ANALYSIS_DATE" --scoring-version "$SCORING_VERSION" | tee -a "$MAIN_LOG"
    else
        python3 tomorrow_stock_selector.py "$ANALYSIS_DATE" --scoring-version "$SCORING_VERSION" >> "$MAIN_LOG" 2>&1
    fi
    
    check_command "量化选股分析"
}

# AI增强分析
run_ai_enhanced_analysis() {
    log "🤖 开始AI增强分析 (并行版，版本: $SCORING_VERSION)..."
    
    # 构建AI分析命令
    AI_CMD="python3 ai_enhanced_daily_report.py --date $ANALYSIS_DATE --scoring-version $SCORING_VERSION"
    
    # 默认分析所有股票
    
    # 添加详细日志参数
    if [ "$VERBOSE" = true ]; then
        AI_CMD="$AI_CMD --verbose"
        eval "$AI_CMD" | tee -a "$MAIN_LOG"
    else
        eval "$AI_CMD" >> "$MAIN_LOG" 2>&1
    fi
    
    check_command "AI增强分析 (并行处理)"
}

# 生成交易建议
generate_trading_advice() {
    log "💡 生成交易建议..."
    
    # 如果存在交易建议生成器，则运行
    if [ -f "trading_advisor.py" ]; then
        if [ "$VERBOSE" = true ]; then
            python3 trading_advisor.py | tee -a "$MAIN_LOG"
        else
            python3 trading_advisor.py >> "$MAIN_LOG" 2>&1
        fi
        check_command "交易建议生成"
    else
        log "⚠️  交易建议生成器不存在，跳过"
    fi
}

# 报告汇总
summarize_reports() {
    log "📋 汇总分析报告..."
    
    REPORTS_DIR="$SCRIPT_DIR/reports"
    DATE_STR="${ANALYSIS_DATE//-/}"
    
    # 检查生成的报告文件
    if [ "$SCORING_VERSION" = "v3.1" ]; then
        QUANTITATIVE_REPORT="$REPORTS_DIR/daily_selection_v3.1/选股分析报告_$DATE_STR.md"
        AI_ENHANCED_REPORT="$REPORTS_DIR/ai_enhanced/AI增强选股报告_${DATE_STR}_V31.md"
    else
        QUANTITATIVE_REPORT="$REPORTS_DIR/daily_selection/选股分析报告_$DATE_STR.md"
        AI_ENHANCED_REPORT="$REPORTS_DIR/ai_enhanced/AI增强选股报告_$DATE_STR.md"
    fi
    TRADING_ADVICE_REPORT="$REPORTS_DIR/trading_advice/交易建议报告_$DATE_STR.md"
    
    log "📁 生成的报告文件:"
    
    if [ -f "$QUANTITATIVE_REPORT" ]; then
        log "✅ 量化选股报告: $QUANTITATIVE_REPORT"
        QUANT_SIZE=$(wc -c < "$QUANTITATIVE_REPORT" 2>/dev/null || echo "0")
        log "   文件大小: ${QUANT_SIZE} 字节"
    else
        log "❌ 量化选股报告: 未生成"
    fi
    
    if [ -f "$AI_ENHANCED_REPORT" ]; then
        log "✅ AI增强分析报告: $AI_ENHANCED_REPORT"
        AI_SIZE=$(wc -c < "$AI_ENHANCED_REPORT" 2>/dev/null || echo "0")
        log "   文件大小: ${AI_SIZE} 字节"
    else
        log "❌ AI增强分析报告: 未生成"
    fi
    
    if [ -f "$TRADING_ADVICE_REPORT" ]; then
        log "✅ 交易建议报告: $TRADING_ADVICE_REPORT"
        ADVICE_SIZE=$(wc -c < "$TRADING_ADVICE_REPORT" 2>/dev/null || echo "0")
        log "   文件大小: ${ADVICE_SIZE} 字节"
    else
        log "⚠️  交易建议报告: 未生成"
    fi
}

# 主函数
main() {
    log "🚀 开始AI增强每日选股系统"
    log "📅 分析日期: $ANALYSIS_DATE"
    log "📊 评分版本: $SCORING_VERSION"
    log "🎯 分析股票数: 全部 (并行处理)"
    log "⚙️  运行模式: $MODE"
    log "🚀 优化特性: 支持并行、重试机制、API频率控制"
    
    # 进入脚本目录
    cd "$SCRIPT_DIR"
    
    # 检查环境
    check_python || exit 1
    
    # 根据模式执行不同流程
    case $MODE in
        full)
            log "📊 运行完整流程..."
            update_data || exit 1
            run_quantitative_analysis || exit 1
            run_ai_enhanced_analysis || exit 1
            generate_trading_advice
            ;;
        ai-only)
            log "🤖 仅运行AI分析..."
            run_ai_enhanced_analysis || exit 1
            generate_trading_advice
            ;;
        update)
            log "📊 仅更新数据..."
            update_data || exit 1
            ;;
        *)
            error_log "未知运行模式: $MODE"
            exit 1
            ;;
    esac
    
    # 汇总报告
    summarize_reports
    
    # 完成
    log "🎉 AI增强每日选股系统运行完成！"
    log "📊 详细日志: $MAIN_LOG"
    
    if [ -s "$ERROR_LOG" ]; then
        log "⚠️  错误日志: $ERROR_LOG"
    fi
    
    # 显示关键统计信息
    if [ -f "$MAIN_LOG" ]; then
        TOTAL_TIME=$(grep -o "AI增强每日选股系统运行完成" "$MAIN_LOG" | wc -l)
        if [ "$TOTAL_TIME" -gt 0 ]; then
            log "✨ 系统已成功运行"
        fi
    fi
}

# 信号处理（Ctrl+C时的清理）
cleanup() {
    log "⚠️  收到中断信号，正在清理..."
    exit 1
}

trap cleanup SIGINT SIGTERM

# 运行主函数
main "$@"