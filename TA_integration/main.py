#!/usr/bin/env python3
"""
TradingAgents集成主脚本
用TradingAgents分析每日选股报告中的股票，替代原有评分系统
"""

import os
import sys
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))

# 添加adapters、core、utils到路径
sys.path.append(str(current_dir / "adapters"))
sys.path.append(str(current_dir / "core"))
sys.path.append(str(current_dir / "utils"))

# 导入所需模块
from china_stock_analyzer import ChinaStockAnalyzer
from claude_config import ClaudeConfig, create_claude_trading_config
from report_parser import ReportParser
from logger import setup_logger

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="TradingAgents集成分析工具")
    parser.add_argument("--date", type=str, 
                       default=datetime.now().strftime("%Y-%m-%d"),
                       help="分析日期 (YYYY-MM-DD)")
    parser.add_argument("--top-n", type=int, default=-1,
                       help="分析前N只股票，设置为-1表示分析全部股票")
    parser.add_argument("--mode", type=str, default="enhance",
                       choices=["enhance", "replace", "compare"],
                       help="运行模式: enhance(增强), replace(替代), compare(对比)")
    parser.add_argument("--output-dir", type=str, default="../reports",
                       help="输出目录")
    parser.add_argument("--config", type=str, 
                       choices=["claude_4", "claude_high_quality", "claude_balanced", "claude_fast", "claude_premium", "custom"],
                       default="claude_4",
                       help="配置类型: claude_4 (最新，默认), claude_high_quality, claude_balanced, claude_fast, claude_premium, custom")
    parser.add_argument("--custom-config", type=str, default="TA_integration/config/config.json",
                       help="自定义配置文件路径")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logger("TA_Integration", verbose=args.verbose)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取配置
    if args.config == "custom":
        # 使用自定义配置文件
        if os.path.exists(args.custom_config):
            with open(args.custom_config, 'r', encoding='utf-8') as f:
                ta_config = json.load(f)
        else:
            logger.warning(f"自定义配置文件不存在: {args.custom_config}，使用默认配置")
            ta_config = create_claude_trading_config("balanced")
    else:
        # 使用预设Claude配置
        config_type = args.config.replace("claude_", "")
        ta_config = create_claude_trading_config(config_type)
    
    # 验证Claude API密钥
    if ta_config["llm_provider"].lower() in ["anthropic", "claude"]:
        if not ClaudeConfig.validate_api_key():
            logger.error("Claude API密钥验证失败，请设置ANTHROPIC_API_KEY环境变量")
            sys.exit(1)
        logger.info(f"使用Claude配置: {args.config}")
        logger.info(f"深度思考模型: {ta_config['deep_think_llm']}")
        logger.info(f"快速响应模型: {ta_config['quick_think_llm']}")
    
    logger.info(f"开始TradingAgents集成分析")
    logger.info(f"分析日期: {args.date}")
    logger.info(f"分析模式: {args.mode}")
    logger.info(f"分析股票数: {args.top_n}")
    
    try:
        # 初始化分析器
        analyzer = ChinaStockAnalyzer(config=ta_config)
        
        if args.mode == "enhance":
            result = run_enhance_mode(analyzer, args, logger)
        elif args.mode == "replace":
            result = run_replace_mode(analyzer, args, logger)
        elif args.mode == "compare":
            result = run_compare_mode(analyzer, args, logger)
        
        # 保存结果
        save_results(result, args, logger)
        
        logger.info("分析完成！")
        
    except Exception as e:
        logger.error(f"分析过程中出错: {e}")
        sys.exit(1)

def run_enhance_mode(analyzer, args, logger):
    """增强模式 - 在原有评分基础上增加AI分析"""
    logger.info("运行增强模式")
    
    # 分析每日选股报告
    result = analyzer.analyze_daily_report(args.date, args.top_n)
    
    if "error" in result:
        raise Exception(result["error"])
    
    logger.info(f"成功分析 {result['summary']['total_analyzed']} 只股票")
    logger.info(f"AI推荐买入: {result['summary']['buy_recommendations']} 只")
    logger.info(f"AI建议谨慎: {result['summary']['sell_warnings']} 只")
    
    return result

def run_replace_mode(analyzer, args, logger):
    """替代模式 - 用AI分析完全替代原有评分"""
    logger.info("运行替代模式")
    
    # 解析选股报告获取股票列表
    parser = ReportParser()
    # 修复报告路径 - 确保从项目根目录正确访问报告
    date_str = args.date.replace('-', '')
    report_path = f"reports/daily_selection/选股分析报告_{date_str}.md"
    if not os.path.exists(report_path):
        # 尝试从TA_integration目录的相对路径
        report_path = f"../reports/daily_selection/选股分析报告_{date_str}.md"
    
    if not os.path.exists(report_path):
        raise Exception(f"选股报告不存在: {report_path}")
    
    parsed_data = parser.parse_report(report_path)
    if not parsed_data:
        raise Exception("解析选股报告失败")
    
    # 获取所有股票并用AI重新评分
    if args.top_n == -1:
        all_stocks = parsed_data['stocks']  # 使用全部股票
        print(f"替代模式：将分析全部 {len(all_stocks)} 只股票")
    else:
        all_stocks = parsed_data['stocks'][:args.top_n]
    stock_list = [{"code": s.code, "name": s.name} for s in all_stocks]
    
    # 运行AI分析
    ta_results = analyzer.china_ta.batch_analyze_stocks(stock_list, args.date)
    
    # 根据AI结果重新排序
    ai_ranked_stocks = []
    for stock_code, ta_result in ta_results.items():
        if "error" in ta_result:
            continue
        
        original_stock = next((s for s in all_stocks if s.code == stock_code), None)
        if original_stock:
            # 创建AI评分版本
            ai_stock = {
                **original_stock.__dict__,
                "ai_decision": ta_result["decision"],
                "ai_confidence": ta_result["confidence"],
                "ai_score": calculate_ai_score(ta_result),
                "original_score": original_stock.comprehensive_score
            }
            ai_ranked_stocks.append(ai_stock)
    
    # 按AI评分排序
    ai_ranked_stocks.sort(key=lambda x: x["ai_score"], reverse=True)
    
    # 生成替代报告
    replace_report = generate_replace_report(ai_ranked_stocks, args.date)
    
    return {
        "mode": "replace",
        "ai_ranked_stocks": ai_ranked_stocks,
        "replace_report": replace_report,
        "summary": {
            "total_analyzed": len(ai_ranked_stocks),
            "top_ai_picks": ai_ranked_stocks[:5]
        }
    }

def run_compare_mode(analyzer, args, logger):
    """对比模式 - 对比量化评分和AI评分"""
    logger.info("运行对比模式")
    
    # 获取增强模式结果
    enhance_result = run_enhance_mode(analyzer, args, logger)
    
    # 获取替代模式结果
    replace_result = run_replace_mode(analyzer, args, logger)
    
    # 生成对比分析
    comparison = generate_comparison_analysis(enhance_result, replace_result)
    
    return {
        "mode": "compare",
        "enhance_result": enhance_result,
        "replace_result": replace_result,
        "comparison": comparison
    }

def calculate_ai_score(ta_result):
    """计算AI评分"""
    base_score = 50
    
    decision = ta_result.get("decision", "HOLD").upper()
    confidence = ta_result.get("confidence", 0.5)
    
    if "BUY" in decision:
        decision_score = 30
    elif "SELL" in decision:
        decision_score = -30
    else:
        decision_score = 0
    
    confidence_score = confidence * 20
    
    return base_score + decision_score + confidence_score

def generate_replace_report(ai_ranked_stocks, analysis_date):
    """生成替代模式报告"""
    lines = [
        "# 🤖 AI驱动选股分析报告\n",
        f"## 📊 分析概览",
        f"- **分析日期**: {analysis_date}",
        f"- **评分系统**: TradingAgents AI评分",
        f"- **分析股票数**: {len(ai_ranked_stocks)}",
        f"- **推荐标准**: AI置信度 + 多智能体一致性\n",
        "## 🏆 AI推荐股票排行\n"
    ]
    
    for i, stock in enumerate(ai_ranked_stocks[:10], 1):
        lines.extend([
            f"### {i}. {stock['code']} - {stock['name']}",
            f"**AI评分**: {stock['ai_score']:.1f}分 (原评分: {stock['original_score']:.1f}分)",
            f"**AI决策**: {stock['ai_decision']}",
            f"**AI置信度**: {stock['ai_confidence']:.1%}",
            f"**价格**: {stock['close_price']}元",
            f"**策略支持**: {', '.join(stock['strategies'])}",
            ""
        ])
    
    return "\n".join(lines)

def generate_comparison_analysis(enhance_result, replace_result):
    """生成对比分析"""
    comparison = {
        "ranking_differences": [],
        "score_correlations": [],
        "decision_consistency": [],
        "summary": {}
    }
    
    # 分析排名差异
    enhance_stocks = enhance_result["original_data"]["stocks"]
    replace_stocks = replace_result["ai_ranked_stocks"]
    
    for i, enhance_stock in enumerate(enhance_stocks[:10]):
        replace_rank = next((j for j, r in enumerate(replace_stocks) 
                           if r["code"] == enhance_stock.code), -1)
        
        if replace_rank >= 0:
            rank_diff = (i + 1) - (replace_rank + 1)
            comparison["ranking_differences"].append({
                "stock_code": enhance_stock.code,
                "quant_rank": i + 1,
                "ai_rank": replace_rank + 1,
                "difference": rank_diff
            })
    
    return comparison

def save_results(result, args, logger):
    """保存分析结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 按照CLAUDE.md规范创建子目录
    if args.mode == "enhance":
        json_subdir = os.path.join(args.output_dir, "ai_enhanced")
        report_subdir = os.path.join(args.output_dir, "ai_enhanced")
    elif args.mode == "replace":
        json_subdir = os.path.join(args.output_dir, "ai_portfolio") 
        report_subdir = os.path.join(args.output_dir, "ai_portfolio")
    else:  # compare
        json_subdir = os.path.join(args.output_dir, "performance")
        report_subdir = os.path.join(args.output_dir, "performance")
        
    # 创建子目录
    os.makedirs(json_subdir, exist_ok=True)
    os.makedirs(report_subdir, exist_ok=True)
    
    # 保存JSON结果
    json_file = os.path.join(json_subdir, f"analysis_result_{args.mode}_{timestamp}.json")
    
    # 处理不能序列化的对象
    serializable_result = make_serializable(result)
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"结果已保存到: {json_file}")
    
    # 保存报告文件
    if "enhanced_report" in result:
        report_file = os.path.join(report_subdir, f"AI增强选股报告_{timestamp[:-7]}.md")  # 去除时分秒，只保留日期
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(result["enhanced_report"])
        logger.info(f"增强报告已保存到: {report_file}")
    
    if "replace_report" in result:
        report_file = os.path.join(report_subdir, f"AI集中投资组合_{timestamp[:-7]}.md")  # 去除时分秒，只保留日期
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(result["replace_report"])
        logger.info(f"AI驱动报告已保存到: {report_file}")

def make_serializable(obj):
    """将对象转换为可序列化格式"""
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    else:
        return obj

if __name__ == "__main__":
    main()