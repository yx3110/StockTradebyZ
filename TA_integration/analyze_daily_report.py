#!/usr/bin/env python3
"""
分析今日选股报告中的所有股票
使用AI增强情绪分析生成加强版报告
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import threading

# 添加项目路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def extract_stocks_from_report():
    """从选股报告中提取所有股票信息"""
    # 获取最新的选股报告
    today = datetime.now().strftime('%Y%m%d')
    report_paths = [
        f"/Users/yangxu/StockTradebyZ/reports/daily_selection/选股分析报告_{today}.md",
        f"/Users/yangxu/StockTradebyZ/daily_result/选股分析报告_{today}.md"  # 兼容旧路径
    ]
    
    report_path = None
    for path in report_paths:
        if Path(path).exists():
            report_path = path
            break
    
    if not report_path:
        print(f"❌ 未找到今日选股报告")
        return []
    
    # 从报告中提取股票信息
    import re
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    stocks = []
    # 查找股票信息的正则模式
    stock_pattern = r'### \d+\. (\d{6}) - (.+?)\n'
    matches = re.findall(stock_pattern, content)
    
    for i, (code, name) in enumerate(matches, 1):
        stocks.append({
            'code': code,
            'name': name.strip(),
            'rank': i,
            'score': 100 - i  # 简单的评分
        })
    
    print(f"📋 从报告中提取到 {len(stocks)} 只股票")
    return stocks

def analyze_single_stock(stock: Dict) -> Dict:
    """分析单只股票的情绪数据（用于多进程）"""
    try:
        # 在每个进程中重新导入模块
        sys.path.append('/Users/yangxu/StockTradebyZ/TA_integration')
        from data_sources.sentiment_integrator import ChineseSentimentIntegrator
        
        stock_code = stock['code']
        stock_name = stock['name']
        
        # 跳过ETF和未知股票
        if 'ETF' in stock_name or '未知' in stock_name or stock_code in ['515600', '560700']:
            return {
                'stock_code': stock_code,
                'error': 'ETF/基金类产品，跳过情绪分析',
                'stock_info': stock
            }
        
        # 获取综合情绪分析
        integrator = ChineseSentimentIntegrator()
        sentiment_result = integrator.get_comprehensive_sentiment(stock_code)
        
        # 添加股票基本信息
        sentiment_result['stock_info'] = stock
        sentiment_result['stock_code'] = stock_code
        
        return sentiment_result
        
    except Exception as e:
        return {
            'stock_code': stock_code,
            'error': str(e),
            'stock_info': stock
        }

def analyze_stocks_batch(stocks: List[Dict], batch_size: int = 5, max_workers: int = 4):
    """批量并行分析股票（分批进行）"""
    print(f"🚀 开始并行分析今日选股报告中的{len(stocks)}只股票")
    print(f"📊 批量设置: 每批{batch_size}只, 最大进程数{max_workers}")
    print("=" * 80)
    
    all_results = {}
    total_batches = (len(stocks) + batch_size - 1) // batch_size
    
    # 分批处理
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(stocks))
        batch_stocks = stocks[start_idx:end_idx]
        
        print(f"\n📦 处理第 {batch_num + 1}/{total_batches} 批 (股票 {start_idx + 1}-{end_idx})")
        print("-" * 60)
        
        # 显示当前批次的股票
        for i, stock in enumerate(batch_stocks):
            print(f"  {start_idx + i + 1:2d}. {stock['code']} - {stock['name']}")
        
        # 并行分析当前批次
        batch_results = analyze_batch_parallel(batch_stocks, max_workers)
        
        # 合并结果
        all_results.update(batch_results)
        
        # 显示批次结果
        successful = len([r for r in batch_results.values() if 'error' not in r])
        failed = len(batch_results) - successful
        print(f"✅ 批次完成: 成功{successful}只, 失败{failed}只")
        
        # 批次间休息
        if batch_num < total_batches - 1:
            print("😴 批次间休息3秒...")
            time.sleep(3)
    
    # 最终统计
    successful_total = len([r for r in all_results.values() if 'error' not in r])
    failed_total = len(all_results) - successful_total
    advanced_analysis_count = len([r for r in all_results.values() if r.get('advanced_analysis')])
    
    print(f"\n📊 全部分析完成!")
    print(f"✅ 成功分析: {successful_total}只")
    print(f"❌ 分析失败: {failed_total}只")
    print(f"🤖 AI增强分析: {advanced_analysis_count}只")
    
    # 显示失败股票
    failed_stocks = [(code, result) for code, result in all_results.items() if 'error' in result]
    if failed_stocks:
        print("\n失败股票列表:")
        for code, result in failed_stocks:
            stock_info = result.get('stock_info', {})
            stock_name = stock_info.get('name', '未知')
            error_msg = result.get('error', '未知错误')
            print(f"  - {code} ({stock_name}): {error_msg}")
    
    return all_results

class ProgressReporter:
    """进度汇报器"""
    def __init__(self, total_stocks: int):
        self.total_stocks = total_stocks
        self.completed = 0
        self.successful = 0
        self.failed = 0
        self.current_batch = 0
        self.total_batches = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        
    def set_batch_info(self, current_batch: int, total_batches: int):
        """设置批次信息"""
        with self.lock:
            self.current_batch = current_batch
            self.total_batches = total_batches
    
    def update_progress(self, success: bool = True):
        """更新进度"""
        with self.lock:
            self.completed += 1
            if success:
                self.successful += 1
            else:
                self.failed += 1
            
            # 计算进度
            progress_pct = (self.completed / self.total_stocks) * 100
            elapsed_time = time.time() - self.start_time
            
            if self.completed > 0:
                avg_time_per_stock = elapsed_time / self.completed
                remaining_stocks = self.total_stocks - self.completed
                eta_seconds = remaining_stocks * avg_time_per_stock
                eta_minutes = int(eta_seconds // 60)
                eta_seconds = int(eta_seconds % 60)
                eta_str = f"{eta_minutes}m{eta_seconds}s"
            else:
                eta_str = "计算中..."
            
            # 显示进度条
            bar_length = 30
            filled_length = int(bar_length * progress_pct // 100)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            print(f"\r📊 总进度: [{bar}] {progress_pct:.1f}% ({self.completed}/{self.total_stocks}) | "
                  f"✅{self.successful} ❌{self.failed} | 批次{self.current_batch}/{self.total_batches} | "
                  f"预计剩余:{eta_str}", end='', flush=True)
    
    def batch_complete(self, batch_successful: int, batch_failed: int):
        """批次完成汇报"""
        print(f"\n✅ 第{self.current_batch}批完成: 成功{batch_successful}只, 失败{batch_failed}只")
        
        # 显示阶段性统计
        elapsed_time = time.time() - self.start_time
        elapsed_minutes = int(elapsed_time // 60)
        elapsed_seconds = int(elapsed_time % 60)
        
        if self.completed > 0:
            success_rate = (self.successful / self.completed) * 100
            print(f"📈 阶段统计: 总成功率{success_rate:.1f}%, 已用时{elapsed_minutes}m{elapsed_seconds}s")

def analyze_batch_parallel(batch_stocks: List[Dict], max_workers: int, progress_reporter=None) -> Dict:
    """并行分析一批股票"""
    results = {}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_stock = {
            executor.submit(analyze_single_stock, stock): stock 
            for stock in batch_stocks
        }
        
        # 收集结果
        for future in as_completed(future_to_stock):
            stock = future_to_stock[future]
            stock_code = stock['code']
            stock_name = stock['name']
            
            try:
                result = future.result(timeout=45)  # 45秒超时
                results[stock_code] = result
                
                # 更新进度
                success = 'error' not in result
                if progress_reporter:
                    progress_reporter.update_progress(success)
                
                # 显示详细结果（可选）
                if 'error' in result:
                    # print(f"❌ {stock_code}-{stock_name}: {result['error']}")
                    pass
                else:
                    total_posts = result.get('total_posts', 0)
                    avg_sentiment = result.get('avg_sentiment', 0)
                    sentiment_label = result.get('sentiment_label', '未知')
                    # print(f"✅ {stock_code}-{stock_name}: {total_posts}条讨论, {sentiment_label}({avg_sentiment:.3f})")
                    
                    if result.get('advanced_analysis'):
                        filtered = result.get('filtered_posts', 0)
                        filter_rate = result.get('filter_rate', 0)
                        # print(f"   🤖 AI增强: 过滤后{filtered}条, 过滤率{filter_rate*100:.1f}%")
                
            except Exception as e:
                # print(f"❌ {stock_code}-{stock_name}: 进程执行失败 - {e}")
                results[stock_code] = {
                    'stock_code': stock_code,
                    'error': f'进程执行失败: {e}',
                    'stock_info': stock
                }
                if progress_reporter:
                    progress_reporter.update_progress(False)
    
    return results

def analyze_all_stocks(stocks: List[Dict]):
    """分析所有股票的情绪数据（兼容性包装）"""
    # 根据CPU核心数和股票数量确定最优参数
    cpu_count = multiprocessing.cpu_count()
    max_workers = min(cpu_count, 6)  # 最多6个进程，避免过度并发
    batch_size = max(3, len(stocks) // 6)  # 动态批次大小
    
    print(f"💻 系统信息: {cpu_count}个CPU核心")
    print(f"⚙️  并行配置: 最大{max_workers}进程, 每批{batch_size}只股票")
    
    return analyze_stocks_batch(stocks, batch_size, max_workers)

def generate_enhanced_report(analysis_results: Dict):
    """生成AI增强版选股报告"""
    print(f"\n📝 生成AI增强版选股报告...")
    
    # 统计分析结果
    total_stocks = len(analysis_results)
    successful_analysis = len([r for r in analysis_results.values() if 'error' not in r])
    advanced_analysis_count = len([r for r in analysis_results.values() if r.get('advanced_analysis')])
    
    # 按情绪分数排序（成功分析的股票）
    analyzed_stocks = []
    for code, result in analysis_results.items():
        if 'error' not in result:
            analyzed_stocks.append((code, result))
    
    # 按综合情绪分数排序
    analyzed_stocks.sort(key=lambda x: x[1].get('avg_sentiment', 0), reverse=True)
    
    # 生成报告
    report_lines = [
        "# 🤖 AI增强版量化选股分析报告",
        "",
        "## 📊 分析概览",
        f"- **分析日期**: {datetime.now().strftime('%Y-%m-%d')}",
        f"- **原始推荐股票**: {total_stocks}只",
        f"- **成功分析股票**: {successful_analysis}只", 
        f"- **AI增强分析**: {advanced_analysis_count}只",
        f"- **分析引擎**: TradingAgents + 中国市场情绪分析师",
        f"- **数据源**: 雪球 + 东方财富股吧 (真实爬取)",
        "",
        "## 🏆 AI情绪排行榜",
        "",
        "*按AI情绪分析结果重新排序的推荐股票*",
        ""
    ]
    
    # 添加成功分析的股票详情
    for rank, (stock_code, result) in enumerate(analyzed_stocks, 1):
        stock_info = result.get('stock_info', {})
        stock_name = stock_info.get('name', '未知')
        quant_score = stock_info.get('score', 0)
        strategies = stock_info.get('strategies', [])
        
        # 情绪分析结果
        total_posts = result.get('total_posts', 0)
        avg_sentiment = result.get('avg_sentiment', 0)
        sentiment_label = result.get('sentiment_label', '未知')
        confidence = result.get('confidence_level', '未知')
        
        report_lines.extend([
            f"### {rank}. {stock_code} - {stock_name}",
            "",
            "**量化分析结果**",
            f"- **综合评分**: {quant_score}分",
            f"- **通过策略**: {', '.join(strategies)}",
            "",
            "**AI情绪分析结果**",
            f"- **讨论总数**: {total_posts}条",
            f"- **综合情绪**: {sentiment_label} ({avg_sentiment:.3f})",
            f"- **数据置信度**: {confidence}",
        ])
        
        # 如果是AI增强分析，添加更多详情
        if result.get('advanced_analysis'):
            filtered_posts = result.get('filtered_posts', 0)
            water_army = result.get('water_army_detected', 0)
            reverse_talk = result.get('reverse_talk_detected', 0)
            filter_rate = result.get('filter_rate', 0)
            enhanced_summary = result.get('enhanced_summary', '')
            
            report_lines.extend([
                f"- **🤖 AI增强分析**: 过滤后{filtered_posts}条高质量讨论",
                f"- **水军检测**: {water_army}条",
                f"- **反话检测**: {reverse_talk}条",
                f"- **内容过滤率**: {filter_rate*100:.1f}%",
                "",
                "**AI分析摘要**:",
                enhanced_summary[:300] + "..." if len(enhanced_summary) > 300 else enhanced_summary,
            ])
        
        # 添加热门话题
        hot_topics = result.get('hot_topics', [])
        if hot_topics:
            report_lines.extend([
                "",
                "**热门讨论话题**:",
            ])
            for i, topic in enumerate(hot_topics[:3], 1):
                report_lines.append(f"{i}. {topic}")
        
        # 添加平台分析
        platform_analysis = result.get('platform_analysis', {})
        if platform_analysis:
            report_lines.extend([
                "",
                "**各平台情绪**:",
            ])
            for platform, analysis in platform_analysis.items():
                report_lines.append(f"- {analysis}")
        
        report_lines.extend([
            "",
            "---",
            ""
        ])
    
    # 添加失败分析的股票
    failed_stocks = [(code, result) for code, result in analysis_results.items() if 'error' in result]
    if failed_stocks:
        report_lines.extend([
            "## ⚠️ 未完成情绪分析的股票",
            "",
            "*以下股票因各种原因未能完成情绪分析，仍可参考量化评分*",
            ""
        ])
        
        for stock_code, result in failed_stocks:
            stock_info = result.get('stock_info', {})
            stock_name = stock_info.get('name', '未知')
            quant_score = stock_info.get('score', 0)
            error_msg = result.get('error', '未知错误')
            
            report_lines.extend([
                f"### {stock_code} - {stock_name}",
                f"- **量化评分**: {quant_score}分",
                f"- **情绪分析**: {error_msg}",
                ""
            ])
    
    # 添加总结和建议
    report_lines.extend([
        "## 📈 AI增强投资建议",
        "",
        "### 🎯 重点关注股票",
        "*AI情绪分析表现优异的股票*",
        ""
    ])
    
    # 推荐情绪最好的前5只股票
    top_sentiment_stocks = analyzed_stocks[:5]
    for i, (stock_code, result) in enumerate(top_sentiment_stocks, 1):
        stock_info = result.get('stock_info', {})
        stock_name = stock_info.get('name', '未知')
        avg_sentiment = result.get('avg_sentiment', 0)
        quant_score = stock_info.get('score', 0)
        
        report_lines.append(f"{i}. **{stock_code} - {stock_name}**: 情绪{avg_sentiment:.3f} + 量化{quant_score}分")
    
    report_lines.extend([
        "",
        "### 📊 分析统计",
        f"- **积极情绪股票**: {len([s for _, s in analyzed_stocks if s.get('avg_sentiment', 0) > 0.1])}只",
        f"- **中性情绪股票**: {len([s for _, s in analyzed_stocks if -0.1 <= s.get('avg_sentiment', 0) <= 0.1])}只",
        f"- **消极情绪股票**: {len([s for _, s in analyzed_stocks if s.get('avg_sentiment', 0) < -0.1])}只",
        f"- **高置信度分析**: {len([s for _, s in analyzed_stocks if s.get('confidence_level') in ['high', '高']])}只",
        "",
        "### 🤖 AI分析优势",
        "- **真实数据**: 直接爬取雪球、东方财富股吧真实讨论",
        "- **水军过滤**: 智能识别和过滤水军、广告帖",
        "- **反话识别**: 检测讽刺、反话等隐含情绪表达",
        "- **可信度评估**: 基于多维度指标评估讨论质量",
        "- **综合决策**: 结合量化评分和AI情绪分析",
        "",
        "## ⚠️ 风险提示",
        "",
        "### 情绪分析局限性",
        "- 社交媒体情绪可能存在滞后性或超前性",
        "- 部分股票讨论数量较少，情绪分析置信度有限",
        "- 需要结合基本面分析和技术分析综合判断",
        "",
        "### 投资建议",
        "- 建议优先关注AI情绪和量化评分双高的股票",
        "- 情绪分析仅作为参考因素之一，不应单独依赖",
        "- 保持理性投资，控制仓位风险",
        "",
        f"---",
        f"",
        f"🤖 **报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"📊 **数据来源**: 量化选股系统 + 雪球 + 东方财富股吧",
        f"🔬 **分析引擎**: Claude Code + 中国市场情绪分析师",
        f"",
        f"Generated with [Claude Code](https://claude.ai/code)"
    ])
    
    return "\n".join(report_lines)

def main():
    """主函数"""
    print("🚀 AI增强版选股分析系统")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 1. 提取股票列表
    stocks = extract_stocks_from_report()
    print(f"📋 从今日选股报告中提取到 {len(stocks)} 只推荐股票")
    
    # 2. 批量分析所有股票
    analysis_results = analyze_all_stocks(stocks)
    
    # 3. 生成增强版报告
    enhanced_report = generate_enhanced_report(analysis_results)
    
    # 4. 保存报告
    # 确保报告目录存在
    os.makedirs("/Users/yangxu/StockTradebyZ/reports/enhanced", exist_ok=True)
    output_file = f"/Users/yangxu/StockTradebyZ/reports/enhanced/AI增强选股报告_{datetime.now().strftime('%Y%m%d')}.md"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(enhanced_report)
    
    print(f"\n✅ AI增强版选股报告已生成!")
    print(f"📄 报告路径: {output_file}")
    print(f"📊 报告长度: {len(enhanced_report)} 字符")
    
    # 5. 保存JSON结果（便于后续分析）
    # 确保JSON目录存在
    os.makedirs("/Users/yangxu/StockTradebyZ/reports/sentiment_analysis", exist_ok=True)
    json_file = f"/Users/yangxu/StockTradebyZ/reports/sentiment_analysis/分析结果_{datetime.now().strftime('%Y%m%d')}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 详细分析数据已保存: {json_file}")
    
    print(f"\n🎉 分析完成! 耗时: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()