#!/usr/bin/env python3
"""
为今日推荐股票获取财务指标数据
专门为量化选股推荐的股票补充财务指标数据
"""

import sys
sys.path.append("data_adapter")

from financial_indicator_fetcher_simple import SimpleFinancialIndicatorFetcher
import sqlite3
import json
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def get_recommended_stocks():
    """从最新的选股报告获取推荐股票列表"""
    try:
        # 读取今日推荐股票代码
        with open('reports/daily_selection/选股分析报告_20250806.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 简单解析推荐股票代码（从报告中提取6位数字代码）
        import re
        stock_codes = re.findall(r'(?:^|\s)(\d{6})(?:\s|$|-)', content)
        
        # 去重并返回
        unique_codes = list(set(stock_codes))
        logger.info(f"从选股报告中提取到{len(unique_codes)}只推荐股票")
        
        return unique_codes
        
    except Exception as e:
        logger.error(f"获取推荐股票列表失败: {e}")
        return []

def get_stock_database_info(stock_codes):
    """从数据库获取股票的详细信息"""
    try:
        with sqlite3.connect("data_adapter/stock_data.db") as conn:
            cursor = conn.cursor()
            
            # 构建查询条件
            placeholders = ','.join(['?' for _ in stock_codes])
            query = f"""
                SELECT id, code, name, type
                FROM securities 
                WHERE code IN ({placeholders})
                ORDER BY code
            """
            
            cursor.execute(query, stock_codes)
            return cursor.fetchall()
            
    except Exception as e:
        logger.error(f"从数据库获取股票信息失败: {e}")
        return []

def main():
    """主函数"""
    logger.info("🎯 开始为推荐股票获取财务指标数据")
    
    # 获取推荐股票列表
    recommended_codes = get_recommended_stocks()
    if not recommended_codes:
        logger.error("❌ 未获取到推荐股票列表")
        return
    
    logger.info(f"📋 推荐股票代码: {', '.join(recommended_codes)}")
    
    # 从数据库获取股票详细信息
    stock_info = get_stock_database_info(recommended_codes)
    if not stock_info:
        logger.error("❌ 未从数据库获取到股票信息")
        return
    
    logger.info(f"📊 数据库中找到{len(stock_info)}只匹配的股票")
    
    # 创建财务指标获取器
    fetcher = SimpleFinancialIndicatorFetcher()
    
    success_count = 0
    fail_count = 0
    total_records = 0
    
    # 逐一处理推荐股票
    for i, (security_id, code, name, stock_type) in enumerate(stock_info, 1):
        logger.info(f"[{i}/{len(stock_info)}] 处理推荐股票 {code} - {name}")
        
        # 转换股票代码
        ts_code = fetcher.convert_stock_code(code)
        
        # 获取财务指标数据
        df = fetcher.fetch_financial_indicator(ts_code)
        
        if df is not None:
            # 保存数据
            saved_count = fetcher.save_financial_indicator(df, security_id)
            if saved_count > 0:
                logger.info(f"  ✅ 新增{saved_count}条财务指标记录")
                total_records += saved_count
            else:
                logger.info(f"  ✅ 数据已存在，无需更新")
            success_count += 1
        else:
            logger.warning(f"  ❌ 获取{code}财务数据失败")
            fail_count += 1
    
    # 最终统计
    logger.info(f"🎉 推荐股票财务指标数据获取完成！")
    logger.info(f"📊 处理股票: {len(stock_info)}只")
    logger.info(f"✅ 成功: {success_count}只")
    logger.info(f"❌ 失败: {fail_count}只")
    logger.info(f"📝 新增记录: {total_records}条")
    
    # 验证数据
    logger.info("🔍 验证推荐股票财务数据...")
    with sqlite3.connect("data_adapter/stock_data.db") as conn:
        cursor = conn.cursor()
        
        for code in recommended_codes[:5]:  # 检查前5只
            cursor.execute("""
                SELECT s.code, s.name, fi.roe, fi.roa, fi.gross_margin 
                FROM securities s 
                LEFT JOIN financial_indicator fi ON s.id = fi.security_id 
                WHERE s.code = ? 
                ORDER BY fi.end_date DESC 
                LIMIT 1
            """, (code,))
            
            result = cursor.fetchone()
            if result:
                code, name, roe, roa, gross_margin = result
                logger.info(f"  {code} {name}: ROE={roe}%, ROA={roa}%, 毛利率={gross_margin}%")
    
    logger.info("✅ 推荐股票财务指标数据获取任务完成！")

if __name__ == "__main__":
    main()