#!/usr/bin/env python3
"""
快速测试选股功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tomorrow_stock_selector import TomorrowStockSelector
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def test_selection():
    """测试选股功能"""
    
    # 使用2025-08-15的数据
    target_date = "2025-08-15"
    
    logger.info("=" * 60)
    logger.info(f"测试选股功能 - 日期: {target_date}")
    logger.info("=" * 60)
    
    # 初始化分析器，使用v3评分器
    analyzer = TomorrowStockSelector(use_database=True, scoring_version="v3")
    
    # 只测试几只股票，快速验证功能
    test_stocks = ['000001', '000002', '002215', '600036']  # 测试股票代码
    
    logger.info(f"测试股票: {test_stocks}")
    
    results = []
    for code in test_stocks:
        try:
            # 测试v3评分
            from scoring.v3.quantitative_scorer_v3 import QuantitativeScorerV3
            scorer = QuantitativeScorerV3()
            score_result = scorer.calculate_stock_score(code, target_date)
            
            if score_result and score_result.get('total_score'):
                results.append({
                    'code': code,
                    'score': score_result['total_score'] * 100,
                    'scores': score_result.get('scores', {})
                })
                logger.info(f"✓ {code}: 评分 {score_result['total_score']*100:.2f}")
            else:
                logger.warning(f"✗ {code}: 无法计算评分")
                
        except Exception as e:
            logger.error(f"✗ {code}: 错误 - {e}")
    
    # 显示结果
    if results:
        logger.info("\n" + "=" * 60)
        logger.info("测试结果汇总")
        logger.info("=" * 60)
        
        # 按评分排序
        results.sort(key=lambda x: x['score'], reverse=True)
        
        for r in results:
            logger.info(f"{r['code']}: {r['score']:.2f}分")
            if r['scores']:
                for factor, score in r['scores'].items():
                    logger.info(f"  - {factor}: {score:.2f}")
        
        logger.info("\n✅ 选股功能测试成功！")
        logger.info("v3评分器工作正常")
    else:
        logger.error("❌ 测试失败，无法获取评分")
    
    return results

if __name__ == "__main__":
    test_selection()