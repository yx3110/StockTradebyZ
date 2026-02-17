#!/usr/bin/env python3
"""
测试包含大盘指数的每日数据更新流程
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# 添加项目根路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_indices_update():
    """测试大盘指数更新功能"""
    print("🧪 测试大盘指数更新功能")
    
    try:
        # 导入更新函数
        sys.path.append('fetch_data')
        from fetch_data.quick_daily_update import update_market_indices
        
        # 获取昨天的日期（避免当天数据可能不完整）
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
        
        print(f"📅 更新日期: {yesterday}")
        print(f"🔄 开始更新大盘指数...")
        
        # 执行更新
        result = update_market_indices(yesterday)
        
        print(f"✅ 大盘指数更新完成: 成功更新{result}个指数")
        
        return result > 0
        
    except Exception as e:
        print(f"❌ 大盘指数更新测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_query():
    """测试数据库中的指数数据查询"""
    print(f"\n📊 测试数据库指数数据查询")
    
    try:
        from data_adapter.database_manager import DatabaseManager
        
        db = DatabaseManager()
        
        # 查询指数类型的证券
        cursor = db.conn.cursor()
        cursor.execute("SELECT code, name, type FROM securities WHERE type = '指数'")
        indices = cursor.fetchall()
        
        print(f"✅ 数据库中的指数数据:")
        for code, name, type_name in indices:
            print(f"  {code}: {name} ({type_name})")
        
        # 查询最新的指数价格数据
        if indices:
            sample_code = indices[0][0]
            cursor.execute("""
                SELECT s.name, dq.trade_date, dq.close, dq.price_change_pct
                FROM securities s
                JOIN daily_quotes dq ON s.id = dq.security_id
                WHERE s.code = ? AND s.type = '指数'
                ORDER BY dq.trade_date DESC
                LIMIT 5
            """, (sample_code,))
            
            recent_data = cursor.fetchall()
            print(f"\n📈 {sample_code} 最近5日数据:")
            for name, date, close, change_pct in recent_data:
                print(f"  {date}: {close:.2f} ({change_pct*100:+.2f}%)")
        
        return len(indices) > 0
        
    except Exception as e:
        print(f"❌ 数据库查询测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_market_analysis():
    """测试市场综合分析功能"""
    print(f"\n🌍 测试市场综合分析功能")
    
    try:
        from market_comprehensive_analyzer import MarketComprehensiveAnalyzer
        
        analyzer = MarketComprehensiveAnalyzer()
        
        print(f"🔄 开始市场综合分析...")
        result = analyzer.analyze_comprehensive_market(days=2)
        
        if 'error' in result:
            print(f"❌ 市场分析失败: {result['error']}")
            return False
        
        # 显示分析结果
        market_rating = result.get('market_rating', {})
        print(f"✅ 市场分析完成:")
        print(f"  市场评级: {market_rating.get('rating', 'N/A')}")
        print(f"  综合评分: {market_rating.get('score', 'N/A')}")
        print(f"  风险等级: {market_rating.get('risk_level', 'N/A')}")
        
        # 检查数据质量
        data_quality = result.get('data_quality', {})
        print(f"  数据质量: {data_quality.get('quality_level', 'N/A')}")
        print(f"  数据完整性: {data_quality.get('completeness', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 市场分析测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ai_report_generation():
    """测试AI增强报告生成（包含市场分析）"""
    print(f"\n🤖 测试AI增强报告生成")
    
    try:
        from ai_enhanced_daily_report import AIEnhancedDailyReport
        
        # 创建报告生成器
        generator = AIEnhancedDailyReport(max_workers=1)  # 限制并发避免API限制
        
        # 获取昨天的日期
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"📅 生成日期: {yesterday}")
        print(f"🔄 开始生成AI增强报告...")
        
        # 生成报告（只处理前3只股票避免API限制）
        result = generator.generate_daily_report(yesterday)
        
        if result.get("success"):
            print(f"✅ AI增强报告生成成功!")
            print(f"  分析股票: {result.get('total_stocks', 0)}只")
            print(f"  详细分析: {result.get('detailed_analysis_count', 0)}只")
            print(f"  市场分析: {'包含' if result.get('market_analysis') else '未包含'}")
            return True
        else:
            print(f"❌ AI报告生成失败: {result.get('error', '未知错误')}")
            return False
        
    except Exception as e:
        print(f"❌ AI报告生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试完整的大盘分析集成功能")
    print("="*60)
    
    test_results = []
    
    # 1. 测试大盘指数更新
    test_results.append(("大盘指数更新", test_indices_update()))
    
    # 2. 测试数据库查询
    test_results.append(("数据库查询", test_database_query()))
    
    # 3. 测试市场综合分析
    test_results.append(("市场综合分析", test_market_analysis()))
    
    # 4. 测试AI报告生成（可选，因为可能受API限制）
    # test_results.append(("AI报告生成", test_ai_report_generation()))
    
    # 显示测试结果
    print("\n" + "="*60)
    print("📋 测试结果汇总:")
    
    success_count = 0
    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if success:
            success_count += 1
    
    print(f"\n🎯 总体结果: {success_count}/{len(test_results)} 测试通过")
    
    if success_count == len(test_results):
        print("🎉 所有测试通过！大盘分析功能集成成功！")
    else:
        print("⚠️  部分测试失败，请检查相关功能")
    
    return success_count == len(test_results)

if __name__ == "__main__":
    main()