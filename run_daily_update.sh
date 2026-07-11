#!/bin/bash
# 每日股票数据更新和选股分析启动脚本

# 配置变量
# API Token 从环境变量读取，不在脚本中硬编码
# 设置方法: export TUSHARE_TOKEN="your_token" 或写入 ~/.bashrc / ~/.zshrc
WORKERS=5
MODE="both"  # 可选: update, report, both, check, ai-enhance
USE_DATABASE=true  # 使用数据库模式
ENABLE_AI=true  # 启用AI增强报告
AI_TOP_N=10  # AI增强分析的股票数量

# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CMD="python3"
LOG_FILE="$SCRIPT_DIR/logs/daily_update.log"

# 颜色输出函数
print_info() {
    echo -e "\033[32m[INFO]\033[0m $1"
}

print_warning() {
    echo -e "\033[33m[WARNING]\033[0m $1"
}

print_error() {
    echo -e "\033[31m[ERROR]\033[0m $1"
}

# 环境检查
check_environment() {
    print_info "检查运行环境..."
    
    # 检查Python
    if ! command -v $PYTHON_CMD &> /dev/null; then
        print_error "Python3 未找到，请安装Python 3.7+"
        exit 1
    fi
    
    # 检查必要的Python包
    $PYTHON_CMD -c "import tushare, pandas, numpy" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_error "缺少必要的Python包，请运行: pip install tushare pandas numpy scipy"
        exit 1
    fi
    
    # 检查Token配置 (从环境变量或config.json读取)
    if [ -z "$TUSHARE_TOKEN" ]; then
        # 尝试从config.json读取
        TUSHARE_TOKEN=$($PYTHON_CMD -c "
import json, pathlib
p = pathlib.Path('$SCRIPT_DIR/config.json')
if p.exists():
    cfg = json.loads(p.read_text())
    print(cfg.get('tushare', {}).get('token', ''))
" 2>/dev/null)
        if [ -z "$TUSHARE_TOKEN" ]; then
            print_error "未找到 Tushare Token。请设置环境变量: export TUSHARE_TOKEN='your_token'"
            exit 1
        fi
    fi
    export TUSHARE_TOKEN  # 确保子进程可用
    
    print_info "环境检查通过"
}

# 创建必要目录
setup_directories() {
    mkdir -p "$SCRIPT_DIR/$DATA_DIR"
    mkdir -p "$SCRIPT_DIR/logs"
    mkdir -p "$SCRIPT_DIR/backups"
    
    # 确保logs目录存在（用于日志文件）
    if [ ! -f "$LOG_FILE" ]; then
        touch "$LOG_FILE"
    fi
}

# 备份旧报告
backup_reports() {
    local backup_dir="$SCRIPT_DIR/backups/$(date +%Y%m)"
    mkdir -p "$backup_dir"
    
    # 备份旧的选股报告
    if [ -f "$SCRIPT_DIR/明日选股分析报告.md" ]; then
        cp "$SCRIPT_DIR/明日选股分析报告.md" "$backup_dir/选股分析报告_$(date +%Y%m%d_%H%M%S).md"
    fi
    
    # 清理超过30天的备份
    find "$SCRIPT_DIR/backups" -name "*.md" -mtime +30 -delete 2>/dev/null
}

# P0.2 (2026-04-27): forward OOS scan + 90d rolling dashboard
# 每日尾声把生产报告 D-7..D 增量 scan 进 forward_samples.csv, 重算 dashboard.
# 任一步失败不应阻塞主流程 (forward 收益要等 N 个交易日才能算, 早期日为空很正常).
run_forward_oos_tracking() {
    print_info "Forward OOS tracking..."
    $PYTHON_CMD $SCRIPT_DIR/scripts/forward_test_tracker.py scan \
        --scoring-version ng1.0.6 \
        2>&1 | tail -3 || print_info "(forward scan skipped — non-fatal)"
    $PYTHON_CMD $SCRIPT_DIR/scripts/forward_test_dashboard.py \
        --scoring-version ng1.0.6 --window-days 90 --horizon 10d \
        2>&1 | tail -3 || print_info "(forward dashboard skipped — non-fatal)"
    if [ -f "$SCRIPT_DIR/reports/forward_test/dashboard.md" ]; then
        print_info "已更新 forward OOS dashboard: reports/forward_test/dashboard.md"
    fi
}

# 运行主程序
run_main_program() {
    print_info "开始执行每日更新任务..."
    print_info "模式: $MODE"
    print_info "数据目录: $DATA_DIR"
    print_info "并发数: $WORKERS"
    print_info "数据库模式: $USE_DATABASE"
    print_info "AI增强功能: $ENABLE_AI"
    if [ "$ENABLE_AI" = "true" ]; then
        print_info "AI分析股票数: $AI_TOP_N"
    fi
    
    if [ "$USE_DATABASE" = "true" ]; then
        # 数据库模式：使用快速更新和数据库报告生成
        print_info "使用数据库模式..."
        
        if [ "$MODE" = "update" ]; then
            # 仅数据更新（包含市场行情、基本面、财务和技术指标）
            CMD="$PYTHON_CMD $SCRIPT_DIR/fetch_data/quick_daily_update.py"
        elif [ "$MODE" = "report" ]; then
            # 仅报告生成
            CMD="$PYTHON_CMD $SCRIPT_DIR/tomorrow_stock_selector.py"
        elif [ "$MODE" = "ai-enhance" ]; then
            # 仅AI增强报告生成
            CMD="$PYTHON_CMD $SCRIPT_DIR/ai_enhanced_daily_report.py"
        else
            # 完整流程：数据更新 + 报告生成
            print_info "执行完整流程: 数据更新 → 选股报告 → AI增强分析"
            
            # 步骤1: 完整数据更新（市场行情、基本面、财务、技术指标）
            print_info "步骤1: 完整数据更新（市场行情、基本面、财务、技术指标）"
            $PYTHON_CMD $SCRIPT_DIR/fetch_data/quick_daily_update.py
            
            if [ $? -ne 0 ]; then
                print_error "数据更新失败"
                exit 1
            fi
            
            # 步骤1.5 (已移除 2026-07-11): 原每周一调用 update_fundamental_data.py 更新行业/地区信息,
            # 但该脚本已归档 (archive/root_scripts/, 且其依赖的 temp_scripts/complete_database_update.py 亦已归档),
            # 每周一实际只报 "can't open file" 假失败。行业分类现由 quick_daily_update.py 步骤5
            # check_sw_industry (月度检查) 覆盖, 股票名称/ST 状态由其步骤17 refresh_stock_names 覆盖。

            # 步骤2: 生成选股报告
            print_info "步骤2: 生成量化选股报告"
            $PYTHON_CMD $SCRIPT_DIR/tomorrow_stock_selector.py
            
            if [ $? -ne 0 ]; then
                print_error "选股报告生成失败"
                exit 1
            fi

            # 步骤 2.5: Signal Trust 增量 + 贴标签
            print_info "步骤2.5: 更新信号可信度"
            $PYTHON_CMD $SCRIPT_DIR/scripts/update_signal_trust_daily.py
            if [ $? -ne 0 ]; then
                print_warning "Signal Trust 更新失败(非阻塞)"
            fi
            # 查找今日最新的选股 JSON 并贴标签
            LATEST_JSON=$(ls -t $SCRIPT_DIR/reports/daily_selection_*/analysis_data_$(date +%Y%m%d).json 2>/dev/null | head -1)
            if [ -n "$LATEST_JSON" ]; then
                $PYTHON_CMD -c "from signal_trust.report_appender import append_trust_tags; n = append_trust_tags('$LATEST_JSON', 'data_adapter/stock_data.db'); print(f'已为 {n} 只股票贴可信度标签')"
            fi

            # 步骤3: AI增强分析（可选）
            if [ "$ENABLE_AI" = "true" ]; then
                print_info "步骤3: 生成AI增强分析报告"
                $PYTHON_CMD $SCRIPT_DIR/ai_enhanced_daily_report.py
                
                if [ $? -ne 0 ]; then
                    print_warning "AI增强分析失败，但基础报告已生成"
                fi
            fi
            
            # P0.2 Forward OOS 监控 (2026-07-11 修复: 原块在函数尾部, both 模式提前 return 从未执行)
            run_forward_oos_tracking

            echo "$(date): 完整流程(both)执行完成" >> "$LOG_FILE"

            print_info "✅ 每日更新流程完成"
            return 0
        fi
    else
        # 传统CSV模式 (deprecated)
        print_info "使用传统CSV模式..."
        CMD="$PYTHON_CMD $SCRIPT_DIR/daily_update_system.py"
        # Token通过环境变量TUSHARE_TOKEN传递，不再通过命令行参数
        CMD="$CMD --data-dir $DATA_DIR"
        CMD="$CMD --workers $WORKERS"
        CMD="$CMD --mode $MODE"
        if [ "$ENABLE_AI" = "false" ]; then
            CMD="$CMD --disable-ai"
        else
            CMD="$CMD --ai-top-n $AI_TOP_N"
        fi
    fi
    
    # 执行命令
    echo "$(date): 开始执行命令: $CMD" >> "$LOG_FILE"
    
    $CMD
    local exit_code=$?
    
    echo "$(date): 命令执行完成，退出码: $exit_code" >> "$LOG_FILE"
    
    if [ $exit_code -eq 0 ]; then
        print_info "任务执行成功！"
        
        # 显示结果摘要
        if [ -f "$SCRIPT_DIR/reports/daily_selection/选股分析报告_最新.md" ]; then
            print_info "已生成最新选股报告: reports/daily_selection/选股分析报告_最新.md"
        fi
        
        if [ -f "$SCRIPT_DIR/reports/ai_enhanced/AI增强选股报告_最新.md" ]; then
            print_info "已生成AI增强选股报告: reports/ai_enhanced/AI增强选股报告_最新.md"
        fi

        # P0.2 Forward OOS 监控 (逻辑抽到 run_forward_oos_tracking, 与 both 模式共用)
        run_forward_oos_tracking

        # 显示数据库统计信息（如果使用数据库模式）
        if [ "$USE_DATABASE" = "true" ]; then
            print_info "数据库统计信息:"
            $PYTHON_CMD -c "
from data_adapter.database_manager import DatabaseManager
db = DatabaseManager()
stats = db.get_database_stats()
print(f'  证券总数: {stats[\"total_securities\"]}')
print(f'  数据记录数: {stats[\"total_quotes\"]:,}')
print(f'  数据库大小: {stats[\"db_size_mb\"]:.2f} MB')
            "
        fi
        
    else
        print_error "任务执行失败，请检查日志文件: $LOG_FILE"
        exit $exit_code
    fi
}

# 显示使用帮助
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -m, --mode MODE     运行模式 (update|report|both|check|ai-enhance) [默认: both]"
    echo "  -w, --workers NUM   并发线程数 [默认: 5]"
    echo "  -t, --token TOKEN   Tushare API Token (推荐使用环境变量 TUSHARE_TOKEN)"
    echo "  --database          启用数据库模式 (默认)"
    echo "  --enable-ai         启用AI增强报告 [默认启用]"
    echo "  --disable-ai        禁用AI增强报告"
    echo "  --ai-top-n NUM      AI增强分析的股票数量 [默认: 10]"
    echo "  -h, --help          显示此帮助信息"
    echo ""
    echo "数据库模式优势:"
    echo "  • 快速数据更新：18秒 vs 60分钟+"
    echo "  • 高效查询和分析"
    echo "  • 自动数据质量检查"
    echo "  • 支持历史数据分析"
    echo ""
    echo "AI增强功能:"
    echo "  • 自动生成AI增强选股报告"
    echo "  • Claude多智能体分析（技术、基本面、情绪、新闻）"
    echo "  • Bull/Bear辩论机制"
    echo "  • AI买入/卖出决策建议"
    echo "  • 需要设置ANTHROPIC_API_KEY环境变量"
    echo ""
    echo "示例:"
    echo "  $0                          # 使用默认配置（数据库模式+AI增强）"
    echo "  $0 -m update                # 仅快速更新数据"
    echo "  $0 -m report                # 生成基础报告+AI增强报告"
    echo "  $0 -m ai-enhance            # 仅生成AI增强报告"
    echo "  $0 --disable-ai             # 禁用AI增强功能"
    echo "  $0 --ai-top-n 5             # AI分析前5只股票"
    echo "  $0 --csv -m both            # 使用CSV模式运行"
    echo "  TUSHARE_TOKEN=xxx $0        # 通过环境变量指定token"
}

# 解析命令行参数
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--mode)
                MODE="$2"
                shift 2
                ;;
            -w|--workers)
                WORKERS="$2"
                shift 2
                ;;
            -t|--token)
                TUSHARE_TOKEN="$2"
                shift 2
                ;;
            -d|--data-dir)
                DATA_DIR="$2"
                shift 2
                ;;
            --database)
                USE_DATABASE=true
                shift
                ;;
            --csv)
                USE_DATABASE=false
                shift
                ;;
            --enable-ai)
                ENABLE_AI=true
                shift
                ;;
            --disable-ai)
                ENABLE_AI=false
                shift
                ;;
            --ai-top-n)
                AI_TOP_N="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                print_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 主函数
main() {
    echo "=================================================="
    echo "     每日股票数据更新和选股分析系统"
    echo "=================================================="
    echo ""
    
    # 解析参数
    parse_arguments "$@"
    
    # 环境检查
    check_environment
    
    # 设置目录
    setup_directories
    
    # 备份报告
    backup_reports
    
    # 运行主程序
    run_main_program
    
    echo ""
    echo "=================================================="
    echo "任务完成时间: $(date)"
    echo "=================================================="
}

# 运行主函数
main "$@"