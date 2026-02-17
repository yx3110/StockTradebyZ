#!/usr/bin/env python3
"""
V4日报生成器 - V3格式版本
参考V3报告格式，生成V4挤压动量增强版选股报告
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

# 导入因子管理框架组件
from factor_manager import FactorManager
from v4_factor_extractor import V4FactorExtractor

class V4DailyReportV3Style:
    """V4日报生成器 - V3格式版本"""
    
    def __init__(self, db_path: str = "../data_adapter/stock_data.db"):
        self.db_path = db_path
        self.logger = self._setup_logger()
        
        # 初始化V4组件（不初始化V2，避免意外调用）
        self.factor_manager = FactorManager(db_path)
        self.v4_extractor = V4FactorExtractor(db_path)
        
        self.logger.info("📊 V4日报生成器（V3格式）已初始化")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("V4DailyReportV3Style")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def get_latest_trade_date(self) -> str:
        """获取最新交易日期"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT MAX(trade_date) FROM daily_quotes")
                result = cursor.fetchone()
                latest_date = result[0] if result and result[0] else datetime.now().strftime('%Y-%m-%d')
                self.logger.info(f"最新交易日期: {latest_date}")
                return latest_date
        except Exception as e:
            self.logger.error(f"获取最新交易日期失败: {e}")
            return datetime.now().strftime('%Y-%m-%d')
    
    def get_strategy_selected_stocks(self, trade_date: str) -> list:
        """🆕 使用5个策略获取候选股票池（替代全市场扫描）"""
        try:
            # 导入5个选股策略
            import sys
            import os
            sys.path.append('..')
            
            from stock_selctor.Selector import (
                BBIKDJSelector, BBIShortLongSelector, 
                BreakoutVolumeKDJSelector, PeakKDJSelector, SuperB1Selector
            )
            from data_adapter.stock_data_loader import StockDataLoader
            
            self.logger.info("🎯 开始使用5个策略筛选候选股票...")
            
            # 加载数据
            data_loader = StockDataLoader(db_path="../data_adapter/stock_data.db")
            all_data = data_loader.load_all_stock_data(days=120)
            
            if not all_data:
                self.logger.error("无法加载股票数据")
                return []
            
            # 转换日期格式
            target_date = pd.Timestamp(trade_date)
            
            # 配置5个选股策略
            strategies = {
                "少负战法": BBIKDJSelector(
                    j_threshold=10, bbi_min_window=20, max_window=60,
                    price_range_pct=1, bbi_q_threshold=0.3, j_q_threshold=0.10
                ),
                "SuperB1战法": SuperB1Selector(
                    lookback_n=10, close_vol_pct=0.02, price_drop_pct=0.02,
                    j_threshold=10, j_q_threshold=0.10,
                    B1_params={
                        "j_threshold": 10, "bbi_min_window": 20, "max_window": 60,
                        "price_range_pct": 1, "bbi_q_threshold": 0.3, "j_q_threshold": 0.10
                    }
                ),
                "补票战法": BBIShortLongSelector(
                    n_short=3, n_long=21, m=3, 
                    bbi_min_window=2, max_window=60, bbi_q_threshold=0.2
                ),
                "TePu战法": BreakoutVolumeKDJSelector(
                    j_threshold=1, j_q_threshold=0.10, up_threshold=3.0,
                    volume_threshold=0.6667, offset=15, max_window=60, price_range_pct=1
                ),
                "填坑战法": PeakKDJSelector(
                    j_threshold=10, max_window=100, fluc_threshold=0.03,
                    j_q_threshold=0.10, gap_threshold=0.2
                )
            }
            
            # 运行所有策略，收集选中的股票
            all_selected_stocks = set()
            strategy_results = {}
            
            for strategy_name, selector in strategies.items():
                try:
                    picks = selector.select(target_date, all_data)
                    strategy_results[strategy_name] = len(picks)
                    all_selected_stocks.update(picks)
                    self.logger.info(f"{strategy_name} 选出 {len(picks)} 只股票")
                except Exception as e:
                    self.logger.error(f"运行 {strategy_name} 失败: {e}")
                    strategy_results[strategy_name] = 0
            
            candidate_stocks = list(all_selected_stocks)
            total_selected = len(candidate_stocks)
            
            self.logger.info(f"✅ 5个策略总共选出 {total_selected} 只候选股票")
            self.logger.info(f"📊 策略结果: {strategy_results}")
            
            return candidate_stocks
                
        except Exception as e:
            self.logger.error(f"使用策略获取候选股票失败: {e}")
            # 如果策略失败，回退到活跃股票池（但限制数量）
            return self.get_fallback_stock_pool()
    
    def get_fallback_stock_pool(self) -> list:
        """策略失败时的回退股票池（限制数量避免过度计算）"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT DISTINCT s.code, s.name
                    FROM securities s
                    JOIN daily_quotes dq ON s.id = dq.security_id
                    JOIN daily_basic db ON s.id = db.security_id AND dq.trade_date = db.trade_date
                    WHERE s.type = 'A股'
                    AND dq.trade_date >= date('now', '-5 days')
                    AND dq.volume > 0
                    AND db.turnover_rate > 1.0  -- 提高换手率要求
                    AND dq.close > 5.0  -- 提高股价要求
                    ORDER BY dq.volume DESC
                    LIMIT 500  -- 🆕 限制为500只最活跃股票
                """
                
                cursor = conn.execute(query)
                stocks = cursor.fetchall()
                
                stock_codes = [stock[0] for stock in stocks]
                self.logger.info(f"获取到回退股票池 {len(stock_codes)} 只（限制活跃股票）")
                return stock_codes
                
        except Exception as e:
            self.logger.error(f"获取回退股票池失败: {e}")
            return []
    
    def run_v4_selection(self, stock_pool: list, trade_date: str) -> dict:
        """运行V4选股分析"""
        self.logger.info(f"🔥 开始V4挤压动量选股分析，股票池: {len(stock_pool)} 只")
        
        v4_results = []
        processed_count = 0
        error_count = 0
        
        for i, stock_code in enumerate(stock_pool):
            try:
                # 获取V4因子数据
                v4_data = self.v4_extractor.extract_factors_for_stock(
                    stock_code, trade_date, trade_date
                )
                
                if not v4_data.empty and 'v4_comprehensive_score' in v4_data.columns:
                    latest_data = v4_data.iloc[-1]
                    v4_score = latest_data['v4_comprehensive_score']
                    
                    if pd.notna(v4_score) and v4_score > 60:  # V4评分阈值
                        # 获取股票基本信息
                        basic_info = self._get_stock_basic_info(stock_code, trade_date)
                        
                        # 获取因子分解得分
                        factor_scores = self._get_factor_breakdown(latest_data)
                        
                        result = {
                            'stock_code': stock_code,
                            'stock_name': basic_info.get('name', stock_code),
                            'v4_score': v4_score,
                            'close': basic_info.get('close'),
                            'pe_ttm': basic_info.get('pe_ttm'),
                            'pb': basic_info.get('pb'),
                            'market_cap': basic_info.get('market_cap'),
                            'turnover_rate': basic_info.get('turnover_rate'),
                            'strategy': 'V4挤压动量',
                            **factor_scores  # 展开因子得分
                        }
                        
                        v4_results.append(result)
                
                processed_count += 1
                if processed_count % 100 == 0:
                    self.logger.info(f"已处理 {processed_count}/{len(stock_pool)} 只股票")
                    
            except Exception as e:
                error_count += 1
                if error_count <= 10:  # 只记录前10个错误
                    self.logger.error(f"处理股票 {stock_code} 失败: {e}")
                continue
        
        # 按V4评分排序
        v4_results.sort(key=lambda x: x['v4_score'], reverse=True)
        
        self.logger.info(f"✅ V4选股完成，有效股票: {len(v4_results)} 只")
        
        return {
            'trade_date': trade_date,
            'total_pool': len(stock_pool),
            'processed_count': processed_count,
            'error_count': error_count,
            'selected_stocks': v4_results,
            'selection_count': len(v4_results)
        }
    
    def _get_stock_basic_info(self, stock_code: str, trade_date: str) -> dict:
        """获取股票基本信息"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT 
                        s.name,
                        dq.close,
                        db.pe_ttm,
                        db.pb,
                        db.circ_mv,
                        db.turnover_rate
                    FROM securities s
                    JOIN daily_quotes dq ON s.id = dq.security_id
                    LEFT JOIN daily_basic db ON s.id = db.security_id AND dq.trade_date = db.trade_date
                    WHERE s.code = ? AND dq.trade_date = ?
                    LIMIT 1
                """
                
                cursor = conn.execute(query, [stock_code, trade_date])
                result = cursor.fetchone()
                
                if result:
                    return {
                        'name': result[0],
                        'close': result[1],
                        'pe_ttm': result[2],
                        'pb': result[3],
                        'market_cap': result[4] / 100000000 if result[4] else None,  # 转换为亿元
                        'turnover_rate': result[5]
                    }
                
                # 如果没有当日数据，获取最近数据
                query_recent = """
                    SELECT 
                        s.name,
                        dq.close,
                        db.pe_ttm,
                        db.pb,
                        db.circ_mv,
                        db.turnover_rate
                    FROM securities s
                    JOIN daily_quotes dq ON s.id = dq.security_id
                    LEFT JOIN daily_basic db ON s.id = db.security_id AND dq.trade_date = db.trade_date
                    WHERE s.code = ? AND dq.trade_date <= ?
                    ORDER BY dq.trade_date DESC
                    LIMIT 1
                """
                
                cursor = conn.execute(query_recent, [stock_code, trade_date])
                result = cursor.fetchone()
                
                if result:
                    return {
                        'name': result[0],
                        'close': result[1],
                        'pe_ttm': result[2],
                        'pb': result[3],
                        'market_cap': result[4] / 100000000 if result[4] else None,
                        'turnover_rate': result[5]
                    }
                
        except Exception as e:
            self.logger.error(f"获取股票 {stock_code} 基本信息失败: {e}")
        
        return {}
    
    def _get_factor_breakdown(self, latest_data: pd.Series) -> dict:
        """获取因子分解得分"""
        # V4因子映射到V3格式的因子类别
        factor_mapping = {
            'technical': ['v4_kdj_strength', 'v4_rsi_momentum', 'v4_bbi_trend', 'v4_volume_surge'],
            'squeeze': ['v4_squeeze_state', 'v4_squeeze_release', 'v4_momentum_direction', 'v4_momentum_acceleration'],
            'fundamental': ['v4_pe_valuation', 'v4_pb_valuation', 'v4_market_cap_factor', 'v4_turnover_activity'],
            'performance': ['v4_price_momentum', 'v4_relative_strength', 'v4_volatility_risk'],
            'market': ['v4_market_beta']
        }
        
        # 计算各类因子得分
        scores = {}
        
        # 技术指标得分 (对应V3的动量) - 修复：V4因子已经是0-100分数，不需要再乘100
        technical_factors = factor_mapping['technical']
        technical_scores = [latest_data.get(factor, 0) for factor in technical_factors if factor in latest_data]
        scores['momentum'] = int(np.mean(technical_scores)) if technical_scores else 50
        
        # 挤压动量得分 (对应V3的回归) - 修复：V4因子已经是0-100分数，不需要再乘100
        squeeze_factors = factor_mapping['squeeze']
        squeeze_scores = [latest_data.get(factor, 0) for factor in squeeze_factors if factor in latest_data]
        scores['reversion'] = int(np.mean(squeeze_scores)) if squeeze_scores else 50
        
        # 市场表现得分 (对应V3的突破) - 修复：V4因子已经是0-100分数，不需要再乘100
        performance_factors = factor_mapping['performance']  
        performance_scores = [latest_data.get(factor, 0) for factor in performance_factors if factor in latest_data]
        scores['breakout'] = int(np.mean(performance_scores)) if performance_scores else 50
        
        # 相对强度得分 (对应V3的相对) - 修复：V4因子已经是0-100分数，不需要再乘100
        scores['relative'] = int(latest_data.get('v4_relative_strength', 50))
        
        # 基本面稳定性得分 (对应V3的稳定) - 修复：V4因子已经是0-100分数，不需要再乘100
        fundamental_factors = factor_mapping['fundamental']
        fundamental_scores = [latest_data.get(factor, 0) for factor in fundamental_factors if factor in latest_data]
        scores['stability'] = int(np.mean(fundamental_scores)) if fundamental_scores else 50
        
        return scores
    
    def _get_investment_advice(self, v4_score: float) -> str:
        """根据V4评分给出投资建议"""
        if v4_score >= 85:
            return "买入"
        elif v4_score >= 75:
            return "谨慎买入"
        elif v4_score >= 65:
            return "关注"
        else:
            return "回避"
    
    def generate_v3_style_report(self, trade_date: str = None) -> str:
        """生成V3格式的V4选股报告"""
        if trade_date is None:
            trade_date = self.get_latest_trade_date()
        
        self.logger.info(f"🚀 开始生成 {trade_date} V4选股报告（V3格式）...")
        
        # 🆕 使用5个策略获取候选股票池（替代全市场扫描）
        stock_pool = self.get_strategy_selected_stocks(trade_date)
        if not stock_pool:
            self.logger.error("无法获取候选股票池")
            return ""
        
        self.logger.info(f"📊 策略筛选得到 {len(stock_pool)} 只候选股票，开始V4评分...")
        
        v4_result = self.run_v4_selection(stock_pool, trade_date)
        
        if not v4_result['selected_stocks']:
            self.logger.warning("V4选股无结果")
            return ""
        
        # 生成报告内容
        report_content = self._format_v3_style_report(trade_date, v4_result)
        
        # 保存报告到指定的v4文件夹
        analysis_date = datetime.strptime(trade_date, '%Y-%m-%d').strftime('%Y%m%d')
        report_filename = f"V4选股分析报告_{analysis_date}.md"
        report_path = Path(f"../reports/daily_selection_v4/{report_filename}")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"📋 V4选股报告已生成: {report_path}")
        return str(report_path)
    
    def _format_v3_style_report(self, trade_date: str, v4_result: dict) -> str:
        """格式化V3风格的报告"""
        
        selected_stocks = v4_result['selected_stocks']
        tomorrow = (datetime.strptime(trade_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # 计算统计信息
        total_selected = len(selected_stocks)
        high_score_count = len([s for s in selected_stocks if s['v4_score'] >= 80])
        medium_score_count = len([s for s in selected_stocks if 70 <= s['v4_score'] < 80])
        
        # 生成报告头部
        report = f"""# 📈 量化选股分析报告 (v4.0 挤压动量增强版)

## 📊 分析概览
- **分析日期**: {trade_date}
- **推荐买入日期**: {tomorrow}
- **分析模式**: 🆕 策略预筛选 + V4评分优化模式
- **预筛选策略**: 少负战法、SuperB1战法、补票战法、TePu战法、填坑战法
- **候选股票池**: {v4_result['total_pool']}只（策略预筛选）
- **V4评分处理**: {v4_result['processed_count']}只
- **V4筛选通过**: {total_selected}只（评分≥60分）
- **高分股票**(≥80分): {high_score_count}只
- **中等股票**(70-80分): {medium_score_count}只
- **详细分析股票数**: {min(50, total_selected)}只
- **推荐股票总数**: {total_selected}只

## 🎯 V4策略优化筛选结果
- **策略预筛选**: 5个经典策略筛选候选股票
- **V4挤压动量评分**: {total_selected}只股票通过V4评分（≥60分）

## 🔥 V4挤压动量核心特色

### ⚡ TTM挤压动量原理
**🔥 核心创新**: 基于John Carter的TTM Squeeze指标，专门识别从低波动到高波动的市场转换点

**⚡ 挤压动量逻辑**:
1. **挤压状态识别**: 布林带收窄在肯特纳通道内，市场蓄势待发
2. **挤压释放捕捉**: 布林带突破肯特纳通道边界，波动率开始扩张  
3. **动量方向确认**: 线性回归斜率判断突破方向
4. **加速度验证**: 确保趋势的持续性和强度

### ⚖️ V4权重配置
- **技术指标 (50%)**: KDJ强度15% + RSI动量14% + BBI趋势10% + 成交量异动11%
- **🆕挤压动量 (20%)**: 挤压状态5% + 挤压释放6% + 动量方向5% + 动量加速度4%  
- **基本面 (8%)**: PE估值2% + PB估值2% + 市值因子2% + 换手率活跃度2%
- **市场表现 (18%)**: 价格动量13% + 相对强度3% + 波动率风险2%
- **市场环境 (4%)**: 市场贝塔1% + 板块轮动1.5% + 流动性1.5%

## 📊 所有筛选股票评分排名

*以下显示V4挤压动量系统筛选的 {total_selected} 只股票的量化评分和因子分解：*

| 排名 | 股票代码 | 股票名称 | 选中策略 | 量化评分 | 投资建议 | 技术 | 挤压 | 表现 | 相对 | 基面 | 收盘价 | PE | PB | 市值(亿) |
|------|----------|----------|----------|----------|----------|------|------|------|------|------|---------|----|----|----------|"""

        # 添加股票数据行
        for i, stock in enumerate(selected_stocks[:50]):  # 显示前50只
            investment_advice = self._get_investment_advice(stock['v4_score'])
            
            # 格式化数据
            close = f"{stock['close']:.2f}" if stock['close'] else "-"
            pe = f"{stock['pe_ttm']:.1f}" if stock['pe_ttm'] else "-"
            pb = f"{stock['pb']:.2f}" if stock['pb'] else "-"
            market_cap = f"{stock['market_cap']:.1f}" if stock['market_cap'] else "-"
            
            report += f"""
| {i+1} | {stock['stock_code']} | {stock['stock_name']} | V4挤压动量 | {stock['v4_score']:.1f} | {investment_advice} | {stock.get('momentum', 50)} | {stock.get('reversion', 50)} | {stock.get('breakout', 50)} | {stock.get('relative', 50)} | {stock.get('stability', 50)} | {close} | {pe} | {pb} | {market_cap} |"""
        
        # 添加统计和说明部分
        report += f"""

## 📈 评分分布统计

### V4挤压动量评分分布
- **85-100分 (强买入)**: {len([s for s in selected_stocks if s['v4_score'] >= 85])}只
- **75-85分 (谨慎买入)**: {len([s for s in selected_stocks if 75 <= s['v4_score'] < 85])}只  
- **65-75分 (关注)**: {len([s for s in selected_stocks if 65 <= s['v4_score'] < 75])}只
- **60-65分 (观望)**: {len([s for s in selected_stocks if 60 <= s['v4_score'] < 65])}只

### 因子得分说明
- **技术**: 技术指标综合得分（KDJ、RSI、BBI、成交量）
- **挤压**: 🆕挤压动量得分（挤压状态、释放信号、方向、加速度）
- **表现**: 市场表现得分（价格动量、波动风险等）
- **相对**: 相对强度得分（相对市场表现）
- **基面**: 基本面稳定性得分（估值、市值、活跃度等）

## 💡 V4选股系统优势

### 🚀 双重筛选机制
1. **策略预筛选**: 使用5个经典量化策略先筛选出技术形态良好的股票
2. **V4挤压动量评分**: 对候选股票进行深度挤压动量分析
3. **效率大幅提升**: 从全市场7000+只股票缩减到策略选中的候选股票
4. **质量显著优化**: 双重筛选确保股票具备技术面和挤压动量双重优势

### 🔥 核心突破点
1. **挤压识别精准**: 准确捕捉市场从平静到激烈的转换时机
2. **假突破过滤**: 通过挤压状态验证，有效降低假突破概率
3. **方向确认机制**: 线性回归动量确保突破方向的可靠性
4. **多维度验证**: 结合技术、基本面、市场环境的综合评估

### ⚡ 适用场景
- **横盘整理后**: 最适合在横盘整理末期识别突破方向
- **震荡市场**: 在震荡环境中筛选真正的趋势股
- **低波动转高波动**: 专门捕捉波动率扩张的投资机会

## ⚠️ 风险提示

1. **挤压释放风险**: 挤压释放不一定都能形成持续趋势，需要量价配合确认
2. **市场环境影响**: 在极端单边市场中，挤压动量信号可能失效
3. **止损设置**: 建议设置8-10%的止损位，挤压失败时及时止损
4. **仓位控制**: 单只股票建议不超过5%仓位，分散投资降低风险
5. **持续跟踪**: V4系统需要持续监控挤压状态变化，及时调整

## 📞 使用建议

1. **优先关注**: 重点关注85分以上的强买入信号股票
2. **分批建仓**: 建议分2-3次建仓，避免单点买入风险
3. **量价确认**: 结合成交量变化确认挤压释放的有效性
4. **趋势跟踪**: 建仓后持续跟踪挤压动量变化，趋势确认后可适度加仓
5. **及时止损**: 如果挤压释放失败或出现反向挤压，应及时减仓

---

*本报告由V4挤压动量增强系统生成，采用John Carter TTM Squeeze核心算法*  
*数据来源: A股全市场量化数据库 | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*  
*🚀 V4系统专注挤压动量，适合捕捉横盘突破和波动率扩张机会*  
*⚠️ 本报告仅供参考，不构成投资建议，投资有风险，入市需谨慎*
"""
        
        return report


def main():
    """生成V4选股报告（V3格式）"""
    
    # 初始化报告生成器
    generator = V4DailyReportV3Style()
    
    # 生成报告
    report_path = generator.generate_v3_style_report()
    
    print(f"\n🎉 V4选股报告已生成!")
    print(f"📋 报告路径: {report_path}")
    print(f"🔥 系统版本: V4挤压动量增强系统")
    print(f"📊 报告格式: V3经典格式 + V4核心特色")
    print(f"⚡ 核心亮点: TTM Squeeze挤压动量算法")


if __name__ == "__main__":
    main()