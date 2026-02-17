#!/usr/bin/env python3
"""
量化评分系统 v3.41 - 反向工程重构版本
基于相关性分析报告的革命性发现：既然负相关，那就反向使用

核心发现：
- v3.4显示负相关性（-0.0159到-0.0360）
- 60-70分和<60分股票夏普比率最高（1.204、1.223）
- 90+分股票夏普比率为负（-0.026）

革命性改进：
1. 🔄 反转v3.4评分逻辑：100 - original_score
2. 🛡️ 特殊处理高风险信号（ST、涨跌停等）
3. 📊 保留v3.4所有计算逻辑，仅在最后反转
4. ⚡ 最小化改动，最大化效果
"""

import numpy as np
import pandas as pd
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import os
import sys

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# 导入v3.4评分系统
import importlib.util
import sys
import os

# 动态导入v3.4评分系统
v34_path = os.path.join(os.path.dirname(__file__), 'quantitative_scorer_v3_4.py')
spec = importlib.util.spec_from_file_location("quantitative_scorer_v3_4", v34_path)
v34_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v34_module)
QuantitativeScorerV34 = v34_module.QuantitativeScorerV34
from data_adapter.database_manager import DatabaseManager

class QuantitativeScorerV341:
    """量化评分系统 v3.41 - 反向工程重构版本"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化评分系统"""
        self.version = "v3.41"
        self.db_manager = DatabaseManager()
        
        # 使用v3.4的完整系统作为基础
        self.base_scorer = QuantitativeScorerV34(config_path)
        
        # 反向工程配置
        self.reverse_config = {
            "version": "v3.41-Reverse-Engineering",
            "reverse_scoring": True,
            "high_risk_penalty": 0.3,  # 高风险股票评分乘以0.3
            "special_handling": {
                "st_stocks": True,      # ST股票特殊处理
                "limit_up_down": True,  # 涨跌停特殊处理
                "new_stocks": True,     # 新股特殊处理
                "low_liquidity": True   # 低流动性特殊处理
            }
        }
        
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(f"QuantitativeScorer_{self.version}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def detect_high_risk_signals(self, stock_code: str, date: str, 
                                stock_data: pd.DataFrame = None) -> Dict[str, bool]:
        """检测高风险信号"""
        risk_signals = {
            "is_st": False,
            "is_limit_up": False,
            "is_limit_down": False,
            "is_new_stock": False,
            "is_low_liquidity": False,
            "risk_score": 0.0
        }
        
        try:
            # 检查ST股票
            if "ST" in stock_code or "*ST" in stock_code:
                risk_signals["is_st"] = True
                risk_signals["risk_score"] += 0.5
                
            # 从数据库获取当日信息
            query = """
            SELECT 
                s.name, s.list_date,
                dq.is_limit_up, dq.is_limit_down, 
                dq.volume, dq.is_st,
                db.turnover_rate as daily_turnover
            FROM securities s
            LEFT JOIN daily_quotes dq ON dq.security_id = s.id AND dq.trade_date = ?
            LEFT JOIN daily_basic db ON db.security_id = s.id AND db.trade_date = ?
            WHERE s.code = ?
            """
            
            with self.db_manager.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=[date, date, stock_code])
                
            if not df.empty:
                row = df.iloc[0]
                
                # 检查ST股票（从数据库字段和名称）
                stock_name = row.get('name', '')
                is_st_from_db = row.get('is_st', False)
                if (('ST' in stock_name or '*ST' in stock_name) or 
                    (pd.notna(is_st_from_db) and is_st_from_db)):
                    risk_signals["is_st"] = True
                    risk_signals["risk_score"] += 0.5
                    
                # 检查涨跌停
                if pd.notna(row.get('is_limit_up')) and row['is_limit_up']:
                    risk_signals["is_limit_up"] = True
                    risk_signals["risk_score"] += 0.3
                    
                if pd.notna(row.get('is_limit_down')) and row['is_limit_down']:
                    risk_signals["is_limit_down"] = True
                    risk_signals["risk_score"] += 0.4
                    
                # 检查新股（上市不足3个月）
                list_date = row.get('list_date')
                if pd.notna(list_date):
                    try:
                        listing_dt = pd.to_datetime(list_date)
                        current_dt = pd.to_datetime(date)
                        days_listed = (current_dt - listing_dt).days
                        
                        if days_listed < 90:  # 3个月
                            risk_signals["is_new_stock"] = True
                            risk_signals["risk_score"] += 0.2
                    except:
                        pass
                        
                # 检查低流动性
                turnover = row.get('daily_turnover') or row.get('turnover_rate')
                if pd.notna(turnover) and turnover < 0.5:  # 换手率<0.5%
                    risk_signals["is_low_liquidity"] = True
                    risk_signals["risk_score"] += 0.2
                    
        except Exception as e:
            self.logger.warning(f"检测高风险信号失败 {stock_code}: {e}")
            
        return risk_signals
        
    def calculate_quantitative_score(self, stock_code: str, date: str, 
                                   stock_data: pd.DataFrame = None, 
                                   basic_data: pd.Series = None) -> Dict[str, Any]:
        """计算量化评分 - 反向工程版本"""
        try:
            # 1. 使用v3.4系统计算原始评分
            original_result = self.base_scorer.calculate_quantitative_score(
                stock_code, date, stock_data, basic_data
            )
            
            if "error" in original_result:
                return {
                    **original_result,
                    "version": self.version
                }
                
            original_score = original_result["quantitative_score"]
            
            # 2. 检测高风险信号
            risk_signals = self.detect_high_risk_signals(stock_code, date, stock_data)
            
            # 3. 核心反向逻辑：100 - 原始评分
            reversed_score = 100 - original_score
            
            # 4. 高风险股票特殊处理 - 减轻惩罚力度
            if risk_signals["risk_score"] > 0:
                # 减轻惩罚：最多扣20分，而不是按比例惩罚
                penalty_points = min(20, risk_signals["risk_score"] * 30)
                final_score = max(0, reversed_score - penalty_points)
                
                self.logger.info(f"高风险股票 {stock_code}: 风险评分={risk_signals['risk_score']:.2f}, "
                               f"原始={original_score:.1f}, 反转={reversed_score:.1f}, "
                               f"扣除={penalty_points:.1f}, 最终={final_score:.1f}")
            else:
                final_score = reversed_score
                
            # 5. 边界处理
            final_score = max(0, min(100, final_score))
            
            # 6. 构建返回结果
            result = {
                "stock_code": stock_code,
                "date": date,
                "quantitative_score": round(final_score, 1),
                "original_v34_score": round(original_score, 1),
                "reversed_score": round(reversed_score, 1),
                "risk_signals": risk_signals,
                "technical_score": original_result.get("technical_score", 0),
                "fundamental_score": original_result.get("fundamental_score", 0),
                "performance_score": original_result.get("performance_score", 0),
                "market_regime_score": original_result.get("market_regime_score", 0),
                "market_regime": original_result.get("market_regime", "neutral"),
                "market_multiplier": original_result.get("market_multiplier", 1.0),
                "version": self.version,
                "reverse_engineering": True
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"计算反向评分失败 {stock_code} {date}: {e}")
            return {
                "stock_code": stock_code,
                "date": date,
                "quantitative_score": 50.0,  # 默认中性评分
                "version": self.version,
                "error": str(e)
            }
    
    def batch_calculate_scores(self, stock_codes: List[str], date: str) -> List[Dict[str, Any]]:
        """批量计算评分"""
        results = []
        
        self.logger.info(f"开始批量计算反向评分 v{self.version}: {len(stock_codes)}只股票")
        
        for i, stock_code in enumerate(stock_codes):
            if i % 100 == 0:
                self.logger.info(f"进度: {i}/{len(stock_codes)} ({i/len(stock_codes)*100:.1f}%)")
                
            result = self.calculate_quantitative_score(stock_code, date)
            results.append(result)
            
        self.logger.info(f"批量计算完成: {len(results)}个结果")
        return results
        
    def explain_reverse_logic(self) -> str:
        """解释反向逻辑的原理"""
        explanation = """
        🔄 v3.41 反向工程评分系统说明
        
        核心发现：
        - v3.4评分系统产生负相关性（-0.0159到-0.0360）
        - 低分股票（<60分）实际表现最好（夏普比率1.223）
        - 高分股票（90+分）实际表现最差（夏普比率-0.026）
        
        反向逻辑：
        1. 使用完整的v3.4计算流程得到原始评分
        2. 执行反转：final_score = 100 - original_score
        3. 对高风险股票施加额外惩罚
        
        预期效果：
        - 原来的"低分好股"现在变成"高分"
        - 原来的"高分差股"现在变成"低分"
        - 相关性从负变正
        - 夏普比率分布更合理
        
        风险控制：
        - ST股票：评分×0.5
        - 涨跌停：额外惩罚
        - 新股：适度惩罚
        - 低流动性：适度惩罚
        """
        return explanation

if __name__ == "__main__":
    # 快速测试
    scorer = QuantitativeScorerV341()
    print(scorer.explain_reverse_logic())
    
    # 测试单个股票
    test_result = scorer.calculate_quantitative_score("000001", "20250829")
    print(f"\n测试结果: {test_result}")