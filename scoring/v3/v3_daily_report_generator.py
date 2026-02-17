#!/usr/bin/env python3
"""
v3版本专用日报生成器
确保与v2版本完全分开，不覆盖任何现有文件
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from quantitative_scorer_v3 import QuantitativeScorerV3
from scoring_system_manager import ScoringSystemManager
from data_adapter.database_manager import DatabaseManager

class V3DailyReportGenerator:
    """v3版本专用日报生成器"""
    
    def __init__(self, config_path: str = None):
        self.db_manager = DatabaseManager()
        self.scorer = QuantitativeScorerV3(config_path)
        self.version = "v3.0"
        self.logger = self._setup_logging()
        
        # v3专用报告目录
        self.report_base_dir = "reports/v3_quantitative_scoring"
        os.makedirs(self.report_base_dir, exist_ok=True)
        
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(f"V3DailyReportGenerator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def get_active_stocks(self, date: str, limit: int = 500) -> List[str]:
        """获取活跃股票池"""
        query = """
        SELECT s.code, db.turnover_rate, db.total_mv as market_cap, dq.close
        FROM securities s
        JOIN daily_basic db ON s.id = db.security_id
        JOIN daily_quotes dq ON s.id = dq.security_id AND db.trade_date = dq.trade_date
        WHERE s.type = 'A股' 
        AND db.trade_date = ?
        AND db.turnover_rate > 0.5
        AND db.total_mv > 1000000  -- 10亿以上市值（万元）
        AND dq.close > 3.0  -- 股价大于3元
        ORDER BY db.total_mv DESC
        LIMIT ?
        """
        
        try:
            with self.db_manager.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=[date, limit])
                return df['code'].tolist()
        except Exception as e:
            self.logger.error(f"获取活跃股票池失败: {e}")
            return []
            
    def generate_daily_report(self, date: str, top_n: int = 50) -> str:
        """生成v3版本日报"""
        self.logger.info(f"开始生成v3版本日报: {date}")
        
        # 获取股票池
        stock_codes = self.get_active_stocks(date, limit=500)
        if not stock_codes:
            self.logger.error("无法获取股票池")
            return ""
            
        self.logger.info(f"股票池大小: {len(stock_codes)}")
        
        # 批量计算得分
        results = self.scorer.batch_score_stocks(stock_codes, date)
        
        # 过滤有效结果并排序
        valid_results = [r for r in results if r.get('total_score', 0) > 0]
        valid_results.sort(key=lambda x: x['total_score'], reverse=True)
        
        # 取前N只股票
        top_results = valid_results[:top_n]
        
        # 生成报告
        report_content = self._generate_report_content(date, top_results, len(stock_codes))
        
        # 保存报告
        report_path = self._save_report(date, report_content, top_results)
        
        self.logger.info(f"v3版本日报生成完成: {report_path}")
        return report_path
        
    def _generate_report_content(self, date: str, results: List[Dict[str, Any]], 
                               total_stocks: int) -> str:
        """生成报告内容"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 统计信息
        if results:
            avg_score = sum(r['total_score'] for r in results) / len(results)
            score_distribution = self._analyze_score_distribution(results)
        else:
            avg_score = 0
            score_distribution = {}
            
        # 市场环境
        market_regime = results[0].get('market_regime', {}) if results else {}
        
        content = f"""# v3量化评分日报 - {date}

生成时间: {timestamp}  
版本: {self.version}  
评分算法: 动态权重智能评分系统

## 📊 统计概览

- 总股票池大小: {total_stocks}
- 有效评分股票: {len(results)}
- 平均得分: {avg_score:.4f}
- 市场环境: {market_regime.get('regime', '未知')} | 波动性: {market_regime.get('volatility', '未知')}

## 🏆 TOP50 股票评分

| 排名 | 股票代码 | 综合得分 | 技术得分 | 基本面得分 | 表现得分 | 市场环境得分 | 收盘价 | PE | PB | 市值(亿) |
|------|----------|----------|----------|------------|----------|------------|---------|----|----|----------|
"""
        
        for i, result in enumerate(results[:50], 1):
            details = result.get('details', {})
            scores = result.get('scores', {})
            
            market_cap_yi = details.get('market_cap', 0) / 10000 if details.get('market_cap', 0) > 0 else 0
            
            content += f"| {i:2d} | {result['code']} | {result['total_score']:.4f} | "
            content += f"{scores.get('technical', 0):.4f} | "
            content += f"{scores.get('fundamental', 0):.4f} | "
            content += f"{scores.get('performance', 0):.4f} | "
            content += f"{scores.get('market_regime', 0):.4f} | "
            content += f"{details.get('close', 0):.2f} | "
            content += f"{details.get('pe_ttm', 'N/A')} | "
            content += f"{details.get('pb', 'N/A')} | "
            content += f"{market_cap_yi:.1f} |\\n"
            
        # 得分分布分析
        content += f"""

## 📈 得分分布分析

"""
        
        if score_distribution:
            for range_key, count in score_distribution.items():
                content += f"- {range_key}: {count}只股票\\n"
                
        # 权重配置
        weights = self.scorer.get_weight_summary()
        content += f"""

## ⚖️ v3版本权重配置

### 技术指标权重 ({sum(v for k, v in weights.items() if k.startswith('technical_')):.3f})
"""
        
        for key, value in weights.items():
            if key.startswith('technical_'):
                content += f"- {key.replace('technical_', '')}: {value:.3f}\\n"
                
        content += f"""

### 基本面权重 ({sum(v for k, v in weights.items() if k.startswith('fundamental_')):.3f})
"""
        
        for key, value in weights.items():
            if key.startswith('fundamental_'):
                content += f"- {key.replace('fundamental_', '')}: {value:.3f}\\n"
                
        content += f"""

### 市场表现权重 ({sum(v for k, v in weights.items() if k.startswith('performance_')):.3f})
"""
        
        for key, value in weights.items():
            if key.startswith('performance_'):
                content += f"- {key.replace('performance_', '')}: {value:.3f}\\n"
                
        content += f"""

### 市场环境权重 ({sum(v for k, v in weights.items() if k.startswith('market_regime_')):.3f})
"""
        
        for key, value in weights.items():
            if key.startswith('market_regime_'):
                content += f"- {key.replace('market_regime_', '')}: {value:.3f}\\n"
                
        content += f"""

## 💡 v3版本改进要点

1. **动态权重调整**: 根据市场环境自动调整各指标权重
2. **多时间窗口**: 综合5、10、20、30日多个时间窗口的技术指标
3. **成交量异动**: 新增成交量异动检测，识别资金关注度
4. **波动率风险**: 增加波动率风险评估，控制选股风险
5. **市场环境适应**: 自动识别牛熊市环境，调整选股策略

## 📝 使用说明

本报告基于v3智能评分算法生成，与v2版本完全独立。
- 评分范围: 0-1.0
- 推荐关注: 得分>0.6的股票
- 风险提示: 仅供参考，不构成投资建议

---
*v3量化评分系统 - 智能权重优化版*
"""
        
        return content
        
    def _analyze_score_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """分析得分分布"""
        distribution = {
            "0.8-1.0 (优秀)": 0,
            "0.6-0.8 (良好)": 0,
            "0.4-0.6 (一般)": 0,
            "0.2-0.4 (偏弱)": 0,
            "0-0.2 (较弱)": 0
        }
        
        for result in results:
            score = result['total_score']
            if score >= 0.8:
                distribution["0.8-1.0 (优秀)"] += 1
            elif score >= 0.6:
                distribution["0.6-0.8 (良好)"] += 1
            elif score >= 0.4:
                distribution["0.4-0.6 (一般)"] += 1
            elif score >= 0.2:
                distribution["0.2-0.4 (偏弱)"] += 1
            else:
                distribution["0-0.2 (较弱)"] += 1
                
        return distribution
        
    def _save_report(self, date: str, content: str, results: List[Dict[str, Any]]) -> str:
        """保存报告"""
        # Markdown报告
        date_str = date.replace('-', '')
        timestamp = datetime.now().strftime("%H%M")
        
        md_filename = f"v3量化评分日报_{date_str}_{timestamp}.md"
        md_path = os.path.join(self.report_base_dir, md_filename)
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        # JSON数据
        json_filename = f"v3量化评分数据_{date_str}_{timestamp}.json"
        json_path = os.path.join(self.report_base_dir, json_filename)
        
        json_data = {
            "version": self.version,
            "date": date,
            "timestamp": datetime.now().isoformat(),
            "total_stocks": len(results),
            "results": results
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
            
        return md_path
        

def main():
    """主程序"""
    import argparse
    
    parser = argparse.ArgumentParser(description='v3版本量化评分日报生成器')
    parser.add_argument('--date', default='2025-08-12', help='评分日期')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--top-n', type=int, default=50, help='报告中显示的股票数量')
    
    args = parser.parse_args()
    
    # 创建报告生成器
    generator = V3DailyReportGenerator(config_path=args.config)
    
    print(f"🚀 开始生成v3版本量化评分日报...")
    print(f"评分日期: {args.date}")
    print(f"版本: {generator.version}")
    
    # 生成报告
    report_path = generator.generate_daily_report(args.date, top_n=args.top_n)
    
    if report_path:
        print(f"✅ v3日报生成成功!")
        print(f"报告路径: {report_path}")
        print(f"报告目录: {generator.report_base_dir}")
    else:
        print("❌ 报告生成失败")


if __name__ == "__main__":
    main()