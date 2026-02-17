#!/usr/bin/env python3
"""
测试市场综合分析引擎
"""

import sys
import os
import logging

# 添加项目根路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_comprehensive_analyzer import MarketComprehensiveAnalyzer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_comprehensive_analysis():
    """测试综合市场分析"""
    print("🚀 开始测试市场综合分析引擎")
    
    try:
        # 创建分析引擎
        analyzer = MarketComprehensiveAnalyzer()
        
        # 进行综合分析
        print("🔄 开始市场综合分析...")
        result = analyzer.analyze_comprehensive_market(days=3)
        
        if 'error' in result:
            print(f"❌ 分析失败: {result['error']}")
            return
        
        # 显示核心结果
        market_rating = result.get('market_rating', {})
        print(f"\n🎯 分析结果:")
        print(f"  市场评级: {market_rating.get('rating', 'N/A')}")
        print(f"  综合评分: {market_rating.get('score', 'N/A')}")
        print(f"  风险等级: {market_rating.get('risk_level', 'N/A')}")
        print(f"  投资建议: {market_rating.get('investment_advice', 'N/A')}")
        
        # 显示各维度评分
        print(f"\n📊 各维度评分:")
        dimensions = ['technical_analysis', 'fundamental_analysis', 'sentiment_analysis', 'news_analysis']
        names = ['技术面', '基本面', '市场情绪', '消息面']
        
        for dim, name in zip(dimensions, names):
            analysis = result.get(dim, {})
            score = analysis.get('score', 'N/A')
            level = analysis.get('level', 'N/A')
            print(f"  {name}: {score}分 ({level})")
        
        # 显示数据质量
        data_quality = result.get('data_quality', {})
        print(f"\n📈 数据质量:")
        print(f"  质量评级: {data_quality.get('quality_level', 'N/A')}")
        print(f"  数据完整性: {data_quality.get('completeness', 'N/A')}")
        
        # 显示交易指导
        trading_guidance = result.get('trading_guidance', {})
        if trading_guidance:
            print(f"\n🎯 交易指导:")
            print(f"  仓位建议: {trading_guidance.get('position_suggestion', 'N/A')}")
            print(f"  市场时机: {trading_guidance.get('market_timing', 'N/A')}")
            print(f"  下日展望: {trading_guidance.get('next_trading_day_outlook', 'N/A')}")
        
        # 测试报告保存
        print(f"\n💾 测试报告保存...")
        report_path = analyzer.save_analysis_report(result)
        print(f"报告已保存到: {report_path}")
        
        print(f"\n✅ 市场综合分析测试完成!")
        return result
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_individual_components():
    """测试各个分析组件"""
    print("\n🔧 测试各个分析组件...")
    
    try:
        analyzer = MarketComprehensiveAnalyzer()
        
        # 获取原始市场数据
        market_data = analyzer.data_fetcher.get_comprehensive_market_data(days=2)
        
        if 'error' in market_data:
            print(f"❌ 市场数据获取失败: {market_data['error']}")
            return
        
        print(f"✅ 市场数据获取成功:")
        summary = market_data.get('summary', {})
        print(f"  指数: {summary.get('indices_count', 0)}个")
        print(f"  行业: {summary.get('sectors_count', 0)}个")
        print(f"  新闻: {summary.get('news_count', 0)}条")
        
        # 测试各维度分析
        print(f"\n📊 测试各维度分析...")
        
        technical = analyzer._analyze_technical_aspect(market_data)
        print(f"  技术面: {technical['score']}分 ({technical['level']})")
        
        fundamental = analyzer._analyze_fundamental_aspect(market_data)
        print(f"  基本面: {fundamental['score']}分 ({fundamental['level']})")
        
        sentiment = analyzer._analyze_sentiment_aspect(market_data)
        print(f"  市场情绪: {sentiment['score']}分 ({sentiment['level']})")
        
        news = analyzer._analyze_news_aspect(market_data)
        print(f"  消息面: {news['score']}分 ({news['level']})")
        
        print(f"\n✅ 各组件测试完成!")
        
    except Exception as e:
        print(f"❌ 组件测试失败: {e}")

def main():
    """主测试函数"""
    print("🧪 开始测试市场综合分析引擎")
    
    # 测试综合分析
    result = test_comprehensive_analysis()
    
    # 测试各个组件
    test_individual_components()
    
    print(f"\n🎉 所有测试完成!")
    
    if result:
        print(f"\n📋 本次分析摘要:")
        print(f"分析日期: {result.get('analysis_date', 'N/A')}")
        print(f"市场评级: {result.get('market_rating', {}).get('rating', 'N/A')}")
        print(f"综合评分: {result.get('market_rating', {}).get('score', 'N/A')}")

if __name__ == "__main__":
    main()