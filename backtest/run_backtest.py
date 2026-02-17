#!/usr/bin/env python3
"""
运行股票选股策略回测的主脚本
使用生成的选股报告和历史价格数据进行完整回测分析
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from backtest_engine import StockBacktester

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("backtest/logs/run_backtest.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("run_backtest")

def main():
    """主函数：执行完整的回测流程"""
    logger.info("="*60)
    logger.info("🚀 开始执行股票选股策略回测")
    logger.info("="*60)
    
    try:
        # 1. 初始化回测器
        logger.info("1️⃣ 初始化回测器...")
        initial_capital = 1000000  # 100万初始资金
        backtester = StockBacktester(initial_capital=initial_capital)
        
        # 2. 设置回测参数
        start_date = "2025-01-01"
        end_date = "2025-07-28"
        holding_days = 5  # 持股天数
        
        # 数据路径配置
        stock_data_dir = str(project_root / "full_securities_data")
        reports_dir = str(project_root / "daily_result")
        
        logger.info(f"回测参数:")
        logger.info(f"  - 初始资金: {initial_capital:,.0f}元")
        logger.info(f"  - 回测期间: {start_date} 至 {end_date}")
        logger.info(f"  - 持股天数: {holding_days}天")
        logger.info(f"  - 股票数据: {stock_data_dir}")
        logger.info(f"  - 选股报告: {reports_dir}")
        
        # 3. 加载股票价格数据
        logger.info("2️⃣ 加载股票价格数据...")
        stock_data = backtester.load_stock_data(
            data_dir=stock_data_dir,
            start_date=start_date,
            end_date=end_date
        )
        
        # 4. 加载选股信号数据
        logger.info("3️⃣ 加载选股报告数据...")
        signals_data = backtester.load_selection_reports(reports_dir=reports_dir)
        
        # 5. 执行回测
        logger.info("4️⃣ 执行回测策略...")
        results = backtester.execute_backtest(
            stock_data=stock_data,
            signals_data=signals_data,
            holding_days=holding_days
        )
        
        # 6. 保存回测结果
        logger.info("5️⃣ 保存回测结果...")
        output_dir = backtester.save_results(results)
        
        # 7. 打印关键指标
        logger.info("6️⃣ 回测结果摘要:")
        logger.info("="*40)
        logger.info(f"📊 累计收益率: {results['total_return']:.2%}")
        logger.info(f"📈 年化收益率: {results['annual_return']:.2%}")
        logger.info(f"📉 最大回撤: {results['max_drawdown']:.2%}")
        logger.info(f"🎯 交易胜率: {results['win_rate']:.2%}")
        logger.info(f"⚖️ 夏普比率: {results['sharpe_ratio']:.2f}")
        logger.info(f"🔄 总交易次数: {results['total_trades']}次")
        logger.info(f"💰 最终资金: {results['final_value']:,.0f}元")
        logger.info("="*40)
        
        # 8. 生成报告预览
        logger.info("7️⃣ 生成专业回测报告...")
        report = backtester.generate_report(results)
        print("\n" + "="*60)
        print(report[:1000])  # 显示报告前1000字符
        print("...")
        print("="*60)
        
        logger.info(f"✅ 回测完成！详细结果保存在: {output_dir}")
        logger.info(f"📄 查看完整报告: {output_dir}/backtest_report_*.md")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 回测执行失败: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return None

def quick_analysis():
    """快速分析模式：仅显示关键统计信息"""
    logger.info("🔍 快速分析模式")
    
    # 检查必要文件
    project_root = Path(__file__).parent.parent
    reports_dir = project_root / "daily_result"
    data_dir = project_root / "full_securities_data"
    
    # 统计选股报告数量
    report_files = list(reports_dir.glob("选股分析报告_*.md"))
    valid_reports = 0
    
    for report_file in report_files:
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if "不是交易日" not in content:
                    valid_reports += 1
        except:
            continue
    
    # 统计股票数据文件
    data_files = list(data_dir.glob("*.csv")) if data_dir.exists() else []
    
    logger.info(f"📈 发现 {len(report_files)} 个选股报告文件")
    logger.info(f"✅ 其中 {valid_reports} 个有效交易日报告")
    logger.info(f"💾 发现 {len(data_files)} 个股票数据文件")
    
    if valid_reports > 0 and len(data_files) > 0:
        logger.info("🎯 数据充足，可以执行完整回测")
        return True
    else:
        logger.warning("⚠️ 数据不足，请先生成选股报告或下载股票数据")
        return False

if __name__ == "__main__":
    # 创建必要目录
    Path("backtest/logs").mkdir(parents=True, exist_ok=True)
    Path("backtest/results").mkdir(parents=True, exist_ok=True)
    
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # 快速分析模式
        if quick_analysis():
            print("\n是否继续执行完整回测? (y/n): ", end="")
            if input().lower().startswith('y'):
                main()
    else:
        # 执行完整回测
        main()