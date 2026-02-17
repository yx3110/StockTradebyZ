#!/usr/bin/env python3
"""
增强量化评分系统主程序
Enhanced Quantitative Scoring System Main Application

集成所有改进模块，提供完整的评分和监控功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sqlite3
from datetime import datetime, timedelta
import argparse
import json
from pathlib import Path

from .core_framework import ScoringEngine, ScoringConfig, create_default_config
from .fundamental_factors import CompositeFundamentalFactor
from .technical_factors import CompositeTechnicalFactor
from .capital_market_factors import CompositeCapitalFactor, CompositeMarketFactor
from .backtest_framework import ScoringBacktester, BacktestConfig

class EnhancedScoringSystem:
    """增强量化评分系统主类"""
    
    def __init__(self, config_path: Optional[str] = None, db_path: str = "data_adapter/stock_data.db"):
        """
        初始化增强评分系统
        
        Args:
            config_path: 配置文件路径
            db_path: 数据库路径
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
        # 加载配置
        if config_path and Path(config_path).exists():
            self.config = self._load_config(config_path)
        else:
            self.config = create_default_config()
        
        # 初始化评分引擎
        self.scoring_engine = ScoringEngine(self.config, db_path)
        
        # 注册所有因子
        self._register_all_factors()
        
        print("✅ 增强量化评分系统初始化完成")
        print(f"📊 系统配置: 技术面{self.config.technical_weight:.0%} | "
              f"基本面{self.config.fundamental_weight:.0%} | "
              f"资金面{self.config.capital_weight:.0%} | "
              f"市场面{self.config.market_weight:.0%}")
    
    def _load_config(self, config_path: str) -> ScoringConfig:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            return ScoringConfig(**config_data)
        except Exception as e:
            print(f"⚠️ 加载配置文件失败，使用默认配置: {e}")
            return create_default_config()
    
    def _register_all_factors(self):
        """注册所有评分因子"""
        # 基本面因子
        fundamental_factor = CompositeFundamentalFactor(self.db_path)
        self.scoring_engine.register_factor(fundamental_factor, 'fundamental')
        
        # 技术面因子
        technical_factor = CompositeTechnicalFactor(self.db_path)
        self.scoring_engine.register_factor(technical_factor, 'technical')
        
        # 资金面因子
        capital_factor = CompositeCapitalFactor(self.db_path)
        self.scoring_engine.register_factor(capital_factor, 'capital')
        
        # 市场面因子
        market_factor = CompositeMarketFactor(self.db_path)
        self.scoring_engine.register_factor(market_factor, 'market')
    
    def get_available_stocks(self, trade_date: str, min_market_cap: float = 500000000) -> List[str]:
        """获取可用股票列表"""
        try:
            query = """
                SELECT DISTINCT s.code, s.name, db.total_mv as market_cap
                FROM securities s
                LEFT JOIN daily_basic db ON s.id = db.security_id AND db.trade_date = ?
                JOIN daily_quotes dq ON s.id = dq.security_id AND dq.trade_date = ?
                WHERE s.type = 'A股'
                AND s.is_active = 1
                AND dq.is_suspend = 0
                AND dq.volume > 0
                AND s.code NOT LIKE 'ST%'
                AND s.code NOT LIKE '*ST%'
                AND (db.total_mv IS NULL OR db.total_mv >= ?)
                ORDER BY db.total_mv DESC
            """
            
            df = pd.read_sql_query(query, self.conn, params=[trade_date, trade_date, min_market_cap])
            return df['code'].tolist()
            
        except Exception as e:
            print(f"❌ 获取可用股票列表失败: {e}")
            return []
    
    def run_daily_scoring(self, trade_date: str = None, top_n: int = 50, 
                         min_score: float = 70.0, output_dir: str = "scoring_improvements") -> List[Dict]:
        """
        运行每日评分
        
        Args:
            trade_date: 交易日期，默认为最新交易日
            top_n: 返回前N只股票
            min_score: 最低评分阈值
            output_dir: 输出目录
            
        Returns:
            评分结果列表
        """
        if not trade_date:
            trade_date = self._get_latest_trading_date()
        
        print(f"🚀 开始执行 {trade_date} 的股票评分...")
        
        # 获取可用股票
        available_stocks = self.get_available_stocks(trade_date)
        
        if not available_stocks:
            print("❌ 未找到可用股票")
            return []
        
        print(f"📊 待评分股票数量: {len(available_stocks)}")
        
        # 分批评分（避免内存问题）
        batch_size = 100
        all_results = []
        
        for i in range(0, len(available_stocks), batch_size):
            batch_stocks = available_stocks[i:i + batch_size]
            print(f"   处理批次 {i//batch_size + 1}/{(len(available_stocks)-1)//batch_size + 1}")
            
            batch_results = self.scoring_engine.batch_score(batch_stocks, trade_date)
            all_results.extend(batch_results)
        
        # 过滤和排序
        high_score_results = [r for r in all_results if r.total_score >= min_score]
        high_score_results.sort(key=lambda x: x.total_score, reverse=True)
        
        top_results = high_score_results[:top_n]
        
        print(f"✅ 评分完成，共 {len(high_score_results)} 只股票评分 >= {min_score}")
        print(f"📈 TOP {len(top_results)} 股票已选出")
        
        # 保存结果到数据库
        self.scoring_engine.save_results_to_db(all_results, "enhanced_stock_scores")
        
        # 生成报告
        self._generate_daily_report(top_results, trade_date, output_dir)
        
        return [self._scoring_result_to_dict(r) for r in top_results]
    
    def _get_latest_trading_date(self) -> str:
        """获取最新交易日期"""
        try:
            query = """
                SELECT MAX(trade_date) as latest_date
                FROM daily_quotes
            """
            df = pd.read_sql_query(query, self.conn)
            return df.iloc[0]['latest_date']
        except Exception as e:
            print(f"❌ 获取最新交易日失败: {e}")
            return datetime.now().strftime('%Y-%m-%d')
    
    def _scoring_result_to_dict(self, result) -> Dict:
        """将评分结果转换为字典"""
        return {
            'stock_code': result.stock_code,
            'stock_name': result.stock_name,
            'trade_date': result.trade_date,
            'total_score': round(result.total_score, 2),
            'technical_score': round(result.technical_score, 2),
            'fundamental_score': round(result.fundamental_score, 2),
            'capital_score': round(result.capital_score, 2),
            'market_score': round(result.market_score, 2),
            'weights': {k: round(v, 4) for k, v in result.weights.items()},
            'market_state': result.market_state,
            'risk_level': result.risk_level,
            'recommendation': result.recommendation
        }
    
    def _generate_daily_report(self, results, trade_date: str, output_dir: str):
        """生成每日评分报告"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        report_path = output_path / f"enhanced_scoring_report_{trade_date}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"""# 增强量化评分系统日报

**日期**: {trade_date}  
**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 系统配置

- **技术面权重**: {self.config.technical_weight:.1%}
- **基本面权重**: {self.config.fundamental_weight:.1%}
- **资金面权重**: {self.config.capital_weight:.1%}
- **市场面权重**: {self.config.market_weight:.1%}

## 🏆 TOP {len(results)} 推荐股票

| 排名 | 股票代码 | 股票名称 | 综合评分 | 技术面 | 基本面 | 资金面 | 市场面 | 投资建议 | 风险等级 |
|------|----------|----------|----------|---------|---------|---------|---------|----------|----------|
""")
            
            for i, result_dict in enumerate([self._scoring_result_to_dict(r) for r in results], 1):
                f.write(f"| {i} | {result_dict['stock_code']} | {result_dict['stock_name']} | "
                       f"{result_dict['total_score']:.1f} | {result_dict['technical_score']:.1f} | "
                       f"{result_dict['fundamental_score']:.1f} | {result_dict['capital_score']:.1f} | "
                       f"{result_dict['market_score']:.1f} | {result_dict['recommendation']} | "
                       f"{result_dict['risk_level']} |\n")
            
            f.write(f"""

## 📈 市场状态分析

**当前市场状态**: {results[0].market_state if results else 'Unknown'}

### 评分分布

- **平均综合评分**: {np.mean([r.total_score for r in results]):.1f}
- **评分标准差**: {np.std([r.total_score for r in results]):.1f}
- **最高评分**: {max([r.total_score for r in results]):.1f}
- **最低评分**: {min([r.total_score for r in results]):.1f}

### 各维度平均得分

- **技术面平均**: {np.mean([r.technical_score for r in results]):.1f}
- **基本面平均**: {np.mean([r.fundamental_score for r in results]):.1f}
- **资金面平均**: {np.mean([r.capital_score for r in results]):.1f}
- **市场面平均**: {np.mean([r.market_score for r in results]):.1f}

## 💡 投资建议

### 强烈推荐 (评分 >= 85)
""")
            
            strong_buy = [r for r in results if r.total_score >= 85]
            if strong_buy:
                for stock in strong_buy[:5]:
                    f.write(f"- **{stock.stock_code} {stock.stock_name}**: {stock.total_score:.1f}分\n")
            else:
                f.write("- 无\n")
            
            f.write(f"""
### 买入 (评分 75-84)
""")
            
            buy = [r for r in results if 75 <= r.total_score < 85]
            if buy:
                for stock in buy[:5]:
                    f.write(f"- **{stock.stock_code} {stock.stock_name}**: {stock.total_score:.1f}分\n")
            else:
                f.write("- 无\n")
            
            f.write(f"""
### 谨慎买入 (评分 70-74)
""")
            
            cautious = [r for r in results if 70 <= r.total_score < 75]
            if cautious:
                for stock in cautious[:5]:
                    f.write(f"- **{stock.stock_code} {stock.stock_name}**: {stock.total_score:.1f}分\n")
            else:
                f.write("- 无\n")
            
            f.write(f"""
## ⚠️ 风险提示

1. **模型风险**: 本评分系统基于历史数据和量化模型，不能保证未来表现
2. **市场风险**: 股市有风险，投资需谨慎，建议分散投资
3. **流动性风险**: 注意股票的流动性，避免集中投资小盘股
4. **操作建议**: 
   - 建议分批买入，控制单只股票仓位
   - 设置合理止损，一般建议8-10%
   - 关注市场整体趋势变化

---
🤖 Generated by Enhanced Quantitative Scoring System v2.0
""")
        
        print(f"📊 评分报告已保存: {report_path}")
    
    def run_backtest(self, start_date: str, end_date: str, **kwargs) -> Dict:
        """运行系统回测"""
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )
        
        backtester = ScoringBacktester(config, self.db_path)
        result = backtester.run_backtest()
        backtester.save_results(result, "scoring_improvements")
        
        return {
            'total_return': result.total_return,
            'annual_return': result.annual_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'score_correlation': result.score_correlation
        }
    
    def monitor_performance(self, days_back: int = 30) -> Dict:
        """监控最近评分表现"""
        try:
            # 获取最近的评分记录
            end_date = self._get_latest_trading_date()
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            query = """
                SELECT ess.stock_code, ess.stock_name, ess.trade_date, ess.total_score,
                       dq1.close as entry_price, dq2.close as exit_price
                FROM enhanced_stock_scores ess
                JOIN daily_quotes dq1 ON ess.trade_date = dq1.trade_date
                JOIN securities s1 ON ess.stock_code = s1.code AND s1.id = dq1.security_id
                LEFT JOIN daily_quotes dq2 ON s1.id = dq2.security_id 
                WHERE ess.trade_date >= ? AND ess.trade_date <= ?
                AND ess.total_score >= 70
                AND dq2.trade_date = (
                    SELECT MAX(trade_date) 
                    FROM daily_quotes 
                    WHERE security_id = s1.id AND trade_date > ess.trade_date
                    LIMIT 1
                )
                ORDER BY ess.trade_date DESC, ess.total_score DESC
            """
            
            df = pd.read_sql_query(query, self.conn, params=[start_date, end_date])
            
            if df.empty:
                return {'error': '没有找到足够的历史数据'}
            
            # 计算收益率
            df['return_pct'] = (df['exit_price'] - df['entry_price']) / df['entry_price']
            
            # 按评分分组分析
            high_score = df[df['total_score'] >= 80]['return_pct']
            medium_score = df[(df['total_score'] >= 70) & (df['total_score'] < 80)]['return_pct']
            
            performance = {
                'period': f"{start_date} to {end_date}",
                'total_stocks': len(df),
                'avg_return': df['return_pct'].mean(),
                'win_rate': (df['return_pct'] > 0).sum() / len(df),
                'high_score_return': high_score.mean() if len(high_score) > 0 else 0,
                'medium_score_return': medium_score.mean() if len(medium_score) > 0 else 0,
                'score_correlation': df['total_score'].corr(df['return_pct']),
                'best_pick': {
                    'code': df.loc[df['return_pct'].idxmax(), 'stock_code'],
                    'name': df.loc[df['return_pct'].idxmax(), 'stock_name'],
                    'return': df.loc[df['return_pct'].idxmax(), 'return_pct'],
                    'score': df.loc[df['return_pct'].idxmax(), 'total_score']
                } if len(df) > 0 else None,
                'worst_pick': {
                    'code': df.loc[df['return_pct'].idxmin(), 'stock_code'],
                    'name': df.loc[df['return_pct'].idxmin(), 'stock_name'],
                    'return': df.loc[df['return_pct'].idxmin(), 'return_pct'],
                    'score': df.loc[df['return_pct'].idxmin(), 'total_score']
                } if len(df) > 0 else None
            }
            
            return performance
            
        except Exception as e:
            return {'error': f'监控性能分析失败: {e}'}
    
    def export_config(self, output_path: str = "scoring_improvements/current_config.json"):
        """导出当前配置"""
        config_dict = {
            'technical_weight': self.config.technical_weight,
            'fundamental_weight': self.config.fundamental_weight,
            'capital_weight': self.config.capital_weight,
            'market_weight': self.config.market_weight,
            'volatility_threshold': self.config.volatility_threshold,
            'volume_threshold': self.config.volume_threshold,
            'bull_market_threshold': self.config.bull_market_threshold,
            'bear_market_threshold': self.config.bear_market_threshold,
            'min_score': self.config.min_score,
            'max_score': self.config.max_score
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 配置已导出至: {output_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="增强量化评分系统")
    parser.add_argument('--mode', choices=['score', 'backtest', 'monitor'], 
                       default='score', help='运行模式')
    parser.add_argument('--date', help='指定交易日期 (YYYY-MM-DD)')
    parser.add_argument('--top-n', type=int, default=50, help='返回前N只股票')
    parser.add_argument('--min-score', type=float, default=70.0, help='最低评分阈值')
    parser.add_argument('--start-date', help='回测开始日期')
    parser.add_argument('--end-date', help='回测结束日期')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--output-dir', default='scoring_improvements', help='输出目录')
    
    args = parser.parse_args()
    
    # 初始化系统
    system = EnhancedScoringSystem(args.config)
    
    if args.mode == 'score':
        # 运行评分
        results = system.run_daily_scoring(
            trade_date=args.date,
            top_n=args.top_n,
            min_score=args.min_score,
            output_dir=args.output_dir
        )
        
        print(f"\n🏆 评分结果 TOP {min(5, len(results))}:")
        for i, result in enumerate(results[:5], 1):
            print(f"{i}. {result['stock_code']} {result['stock_name']}: "
                  f"{result['total_score']:.1f}分 - {result['recommendation']}")
    
    elif args.mode == 'backtest':
        # 运行回测
        if not args.start_date or not args.end_date:
            print("❌ 回测模式需要指定 --start-date 和 --end-date")
            return
        
        result = system.run_backtest(args.start_date, args.end_date)
        
        print(f"\n📊 回测结果:")
        print(f"总收益率: {result['total_return']:.2%}")
        print(f"年化收益率: {result['annual_return']:.2%}")
        print(f"夏普比率: {result['sharpe_ratio']:.3f}")
        print(f"最大回撤: {result['max_drawdown']:.2%}")
        print(f"评分相关性: {result['score_correlation']:.4f}")
    
    elif args.mode == 'monitor':
        # 监控表现
        performance = system.monitor_performance()
        
        if 'error' in performance:
            print(f"❌ {performance['error']}")
        else:
            print(f"\n📈 最近30天表现监控:")
            print(f"评估股票数: {performance['total_stocks']}")
            print(f"平均收益率: {performance['avg_return']:.2%}")
            print(f"胜率: {performance['win_rate']:.1%}")
            print(f"高分组收益: {performance['high_score_return']:.2%}")
            print(f"评分相关性: {performance['score_correlation']:.4f}")
            
            if performance['best_pick']:
                bp = performance['best_pick']
                print(f"最佳选择: {bp['code']} {bp['name']} ({bp['return']:.2%}, 评分{bp['score']:.1f})")

if __name__ == "__main__":
    main()