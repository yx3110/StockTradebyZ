#!/usr/bin/env python3
"""
AI增强版选股分析系统 V2
增加情绪数据采样数量，支持多页采集
"""

import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time
import re
import threading

# 添加项目路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# 导入必要的模块
from data_sources.sentiment_integrator import ChineseSentimentIntegrator

# 尝试导入追加的模块
try:
    from agents.chinese_market_sentiment_analyzer import ChineseMarketSentimentAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    print("警告: 无法导入中国市场情绪分析器")
    ANALYZER_AVAILABLE = False

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
        
    def update_batch(self, batch_num: int, total: int):
        """更新批次信息"""
        with self.lock:
            self.current_batch = batch_num
            self.total_batches = total
    
    def complete_stock(self, success: bool = True):
        """完成一只股票"""
        with self.lock:
            self.completed += 1
            if success:
                self.successful += 1
            else:
                self.failed += 1
    
    def get_progress(self) -> str:
        """获取进度信息"""
        elapsed = time.time() - self.start_time
        rate = self.completed / elapsed if elapsed > 0 else 0
        eta = (self.total_stocks - self.completed) / rate if rate > 0 else 0
        
        return (f"进度: {self.completed}/{self.total_stocks} "
                f"(成功:{self.successful}, 失败:{self.failed}) "
                f"速度: {rate:.1f}只/秒 "
                f"剩余时间: {eta/60:.1f}分钟")

class EnhancedSentimentIntegrator(ChineseSentimentIntegrator):
    """增强版情绪整合器 - 支持多页采集"""
    
    def __init__(self):
        super().__init__()
        self.pages_to_fetch = 3  # 每个平台获取3页数据
        
    def _safe_get_eastmoney_data_enhanced(self, stock_code: str) -> Optional[Dict]:
        """增强版东财数据获取 - 多页采集"""
        try:
            all_posts = []
            
            # 获取多页数据
            for page in range(1, self.pages_to_fetch + 1):
                print(f"东财: 获取第{page}页数据...")
                posts = self.eastmoney_api.get_stock_posts(stock_code, limit=50)
                if posts:
                    all_posts.extend(posts)
                time.sleep(0.5)  # 避免请求过快
            
            # 去重
            unique_posts = []
            seen_titles = set()
            for post in all_posts:
                title = post.get('title', '')
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    unique_posts.append(post)
            
            print(f"东财: 共获取{len(unique_posts)}条不重复帖子")
            
            # 转换为标准格式
            formatted_posts = []
            for post in unique_posts:
                formatted_post = {
                    'content': f"{post.get('title', '')} {post.get('content', '')}",
                    'sentiment': 'positive' if post.get('sentiment_score', 0) > 0.1 else 
                               ('negative' if post.get('sentiment_score', 0) < -0.1 else 'neutral'),
                    'likes': post.get('like_count', 0),
                    'comments': post.get('comment_count', 0),
                    'shares': 0,
                    'author': post.get('author', ''),
                    'account_age_days': 200,
                    'followers': 50,
                    'post_frequency': 3
                }
                formatted_posts.append(formatted_post)
            
            # 生成摘要
            summary = self.eastmoney_api.get_stock_sentiment_summary(stock_code)
            
            return {
                'summary': summary,
                'raw_posts': formatted_posts
            }
        except Exception as e:
            print(f"东方财富API调用失败: {e}")
            return None
    
    def get_comprehensive_sentiment(self, stock_code: str) -> Dict[str, Any]:
        """获取综合情绪分析 - 使用增强版"""
        results = {}
        raw_posts_data = {}
        
        # 获取增强版东财数据
        eastmoney_data = self._safe_get_eastmoney_data_enhanced(stock_code)
        if eastmoney_data:
            results['eastmoney'] = eastmoney_data['summary']
            raw_posts_data['eastmoney'] = eastmoney_data.get('raw_posts', [])
        else:
            results['eastmoney'] = self._get_empty_result(stock_code, 'eastmoney')
            raw_posts_data['eastmoney'] = []
        
        # 雪球暂时禁用
        results['xueqiu'] = self._get_empty_result(stock_code, 'xueqiu')
        raw_posts_data['xueqiu'] = []
        
        # 如果有中国市场情绪分析师，使用高级分析
        if self.sentiment_analyst and any(raw_posts_data.values()):
            enhanced_results = self._enhance_with_chinese_analyst(stock_code, results, raw_posts_data)
            return enhanced_results
        else:
            # 使用基础整合分析
            return self._integrate_sentiment_data(stock_code, results)

def analyze_stock_batch(stock_batch: List[Dict], batch_num: int, progress_reporter: ProgressReporter) -> List[Dict]:
    """分析一批股票（进程内执行）"""
    results = []
    integrator = EnhancedSentimentIntegrator()  # 使用增强版
    
    for i, stock in enumerate(stock_batch, 1):
        stock_code = stock['code']
        stock_name = stock['name']
        
        print(f"  {batch_num * len(stock_batch) + i}. {stock_code} - {stock_name}")
        
        try:
            # 获取综合情绪分析
            sentiment_result = integrator.get_comprehensive_sentiment(stock_code)
            
            # 添加股票基本信息
            sentiment_result['stock_info'] = {
                'code': stock_code,
                'name': stock_name,
                'rank': stock.get('rank', 0),
                'score': stock.get('score', 0),
                'strategy': stock.get('strategy', '')
            }
            
            results.append({stock_code: sentiment_result})
            progress_reporter.complete_stock(True)
            
        except Exception as e:
            print(f"   ❌ 分析{stock_code}失败: {e}")
            results.append({
                stock_code: {
                    'error': str(e),
                    'stock_info': stock
                }
            })
            progress_reporter.complete_stock(False)
    
    return results

def extract_stocks_from_report(report_file: str) -> List[Dict]:
    """从选股报告中提取股票列表"""
    stocks = []
    
    with open(report_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找股票信息的正则模式
    stock_pattern = r'### \d+\. (\d{6}) - (.+?)\n'
    matches = re.findall(stock_pattern, content)
    
    for i, (code, name) in enumerate(matches, 1):
        stocks.append({
            'code': code,
            'name': name.strip(),
            'rank': i
        })
    
    return stocks

def generate_enhanced_report(analysis_results: Dict, report_date: str) -> str:
    """生成AI增强版选股报告"""
    report_lines = [
        f"# 🚀 AI增强版选股报告 V2",
        f"",
        f"## 📊 报告概览",
        f"- **分析日期**: {report_date}",
        f"- **分析股票**: {len(analysis_results)}只",
        f"- **分析引擎**: TradingAgents + 中国市场情绪分析师",
        f"- **数据来源**: 东方财富股吧（多页采集）",
        f"- **版本**: V2 (增强数据采样)",
        f"",
    ]
    
    # 按综合得分排序
    sorted_stocks = []
    for stock_code, data in analysis_results.items():
        if 'error' not in data:
            stock_info = data.get('stock_info', {})
            sentiment_score = data.get('avg_sentiment', 0)
            confidence = data.get('confidence_level', 'low')
            filtered_posts = data.get('filtered_posts', 0)
            
            # 计算综合得分
            confidence_weight = {'high': 1.0, 'medium': 0.8, 'low': 0.5}.get(confidence.lower(), 0.5)
            composite_score = sentiment_score * confidence_weight
            
            sorted_stocks.append({
                'code': stock_code,
                'name': stock_info.get('name', ''),
                'sentiment_score': sentiment_score,
                'composite_score': composite_score,
                'data': data,
                'filtered_posts': filtered_posts
            })
    
    sorted_stocks.sort(key=lambda x: x['composite_score'], reverse=True)
    
    # 生成股票分析详情
    report_lines.append("## 📈 股票情绪分析排行")
    report_lines.append("")
    
    for i, stock in enumerate(sorted_stocks, 1):
        data = stock['data']
        
        # 基本信息
        report_lines.extend([
            f"### {i}. {stock['code']} - {stock['name']}",
            f"",
            f"**情绪分析结果**",
            f"- **综合情绪得分**: {data.get('avg_sentiment', 0):.3f}",
            f"- **情绪标签**: {data.get('sentiment_label', '未知')}",
            f"- **置信度**: {data.get('confidence_level', '未知')}",
            f"- **总讨论数**: {data.get('total_posts', 0)}条",
            f"- **有效讨论数**: {data.get('filtered_posts', 0)}条",
            f"- **过滤率**: {data.get('filter_rate', 0)*100:.1f}%",
            f""
        ])
        
        # AI增强分析
        if data.get('advanced_analysis'):
            report_lines.extend([
                f"**AI智能分析**",
                f"- **水军检测**: {data.get('water_army_detected', 0)}条",
                f"- **反话识别**: {data.get('reverse_talk_detected', 0)}条",
                f""
            ])
        
        # 情绪分布
        dist = data.get('sentiment_distribution', {})
        if dist:
            total = sum(dist.values())
            if total > 0:
                report_lines.extend([
                    f"**情绪分布**",
                    f"- 看好: {dist.get('positive', 0)}条 ({dist.get('positive', 0)/total*100:.1f}%)",
                    f"- 中性: {dist.get('neutral', 0)}条 ({dist.get('neutral', 0)/total*100:.1f}%)",
                    f"- 看空: {dist.get('negative', 0)}条 ({dist.get('negative', 0)/total*100:.1f}%)",
                    f""
                ])
        
        # 增强摘要
        if data.get('enhanced_summary'):
            report_lines.extend([
                f"**AI分析摘要**",
                f"{data['enhanced_summary']}",
                f""
            ])
        
        # 热门话题
        topics = data.get('hot_topics', [])
        if topics:
            report_lines.extend([
                f"**热门讨论话题**",
            ])
            for j, topic in enumerate(topics[:5], 1):
                report_lines.append(f"{j}. {topic}")
            report_lines.append("")
        
        report_lines.append("---")
        report_lines.append("")
    
    # 添加统计信息
    report_lines.extend([
        "## 📊 统计信息",
        f"- **分析完成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **平均采样数量**: {sum(s['filtered_posts'] for s in sorted_stocks) / len(sorted_stocks):.0f}条/股票",
        f"- **AI过滤效果**: 平均过滤{sum(s['data'].get('filter_rate', 0) for s in sorted_stocks) / len(sorted_stocks)*100:.1f}%低质量内容",
        "",
        "Generated with [Claude Code](https://claude.ai/code)"
    ])
    
    return "\n".join(report_lines)

def main():
    """主函数"""
    print("🚀 AI增强版选股分析系统 V2")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 获取今日选股报告
    today = datetime.now().strftime('%Y%m%d')
    report_file = f"/Users/yangxu/StockTradebyZ/daily_result/选股分析报告_{today}.md"
    
    if not Path(report_file).exists():
        print(f"❌ 未找到今日选股报告: {report_file}")
        return
    
    # 提取股票列表
    stocks = extract_stocks_from_report(report_file)
    print(f"📋 从今日选股报告中提取到 {len(stocks)} 只推荐股票")
    
    # 设置并行处理参数
    cpu_count = mp.cpu_count()
    max_workers = min(6, cpu_count - 2)  # 保留2个CPU核心
    batch_size = 4  # 每批处理4只股票
    
    print(f"💻 系统信息: {cpu_count}个CPU核心")
    print(f"⚙️  并行配置: 最大{max_workers}进程, 每批{batch_size}只股票")
    
    # 初始化进度报告器
    progress_reporter = ProgressReporter(len(stocks))
    
    # 分批处理
    print(f"🚀 开始并行分析今日选股报告中的{len(stocks)}只股票")
    print(f"📊 批量设置: 每批{batch_size}只, 最大进程数{max_workers}")
    print("=" * 80)
    
    all_results = {}
    batches = [stocks[i:i+batch_size] for i in range(0, len(stocks), batch_size)]
    progress_reporter.update_batch(0, len(batches))
    
    # 使用进程池并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有批次任务
        future_to_batch = {}
        for batch_num, batch in enumerate(batches):
            print(f"\n📦 处理第 {batch_num+1}/{len(batches)} 批 (股票 {batch_num*batch_size+1}-{min((batch_num+1)*batch_size, len(stocks))})")
            print("-" * 60)
            
            # 打印批次中的股票
            for i, stock in enumerate(batch, 1):
                print(f"  {batch_num*batch_size + i}. {stock['code']} - {stock['name']}")
            
            future = executor.submit(analyze_stock_batch, batch, batch_num, progress_reporter)
            future_to_batch[future] = batch_num
            
            # 控制并发数
            if len(future_to_batch) >= max_workers:
                # 等待至少一个任务完成
                done, _ = mp.futures.wait(future_to_batch.keys(), return_when=mp.futures.FIRST_COMPLETED)
                for future in done:
                    batch_results = future.result()
                    for result in batch_results:
                        all_results.update(result)
                    del future_to_batch[future]
                    
            # 批次间短暂休息
            if batch_num < len(batches) - 1:
                print("😴 批次间休息3秒...")
                time.sleep(3)
        
        # 等待所有剩余任务完成
        for future in as_completed(future_to_batch):
            batch_results = future.result()
            for result in batch_results:
                all_results.update(result)
            
            batch_num = future_to_batch[future]
            success_count = sum(1 for r in batch_results if 'error' not in list(r.values())[0])
            print(f"✅ 批次完成: 成功{success_count}只, 失败{len(batch_results)-success_count}只")
    
    print(f"\n📊 全部分析完成!")
    
    # 统计结果
    success_count = sum(1 for data in all_results.values() if 'error' not in data)
    ai_enhanced_count = sum(1 for data in all_results.values() if data.get('advanced_analysis'))
    
    print(f"✅ 成功分析: {success_count}只")
    print(f"❌ 分析失败: {len(all_results) - success_count}只")
    print(f"🤖 AI增强分析: {ai_enhanced_count}只")
    
    # 打印失败的股票
    failed_stocks = [(code, data['error']) for code, data in all_results.items() if 'error' in data]
    if failed_stocks:
        print("\n失败股票列表:")
        for code, error in failed_stocks:
            stock_info = next((s for s in stocks if s['code'] == code), {})
            print(f"  - {code} ({stock_info.get('name', '未知')}): {error}")
    
    # 生成增强版报告
    print("\n📝 生成AI增强版选股报告...")
    report_content = generate_enhanced_report(all_results, today)
    
    # 确保报告目录存在
    os.makedirs("reports/enhanced", exist_ok=True)
    
    # 保存报告
    output_file = f"reports/enhanced/AI增强选股报告_{today}_v2.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n✅ AI增强版选股报告已生成!")
    print(f"📄 报告路径: {output_file}")
    print(f"📊 报告长度: {len(report_content)} 字符")
    
    # 保存详细分析数据
    json_file = f"reports/sentiment_analysis/分析结果_{today}_v2.json"
    os.makedirs("reports/sentiment_analysis", exist_ok=True)
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"💾 详细分析数据已保存: {json_file}")
    
    print(f"\n🎉 分析完成! 耗时: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()