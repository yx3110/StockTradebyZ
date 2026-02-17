#!/usr/bin/env python3
"""
投资组合性能跟踪器
基于ChatGPT Micro-Cap Experiment的思路，实现夏普比率、收益率等性能指标计算
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os
import logging
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceTracker:
    """投资组合性能跟踪器"""
    
    def __init__(self, 
                 portfolio_csv: str = "TA_integration/data/ai_portfolio_history.csv",
                 benchmark_files: Dict[str, str] = None):
        """
        初始化性能跟踪器
        
        Args:
            portfolio_csv: 投资组合历史数据CSV
            benchmark_files: 基准指数文件路径字典
        """
        self.portfolio_csv = portfolio_csv
        self.benchmark_files = benchmark_files or {
            "沪深300": "full_securities_data/000300_指数.csv",
            "创业板指": "full_securities_data/399006_指数.csv",
            "科创50": "full_securities_data/000688_指数.csv"
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(portfolio_csv), exist_ok=True)
    
    def load_portfolio_history(self) -> pd.DataFrame:
        """加载投资组合历史数据"""
        if os.path.exists(self.portfolio_csv):
            df = pd.read_csv(self.portfolio_csv)
            df['date'] = pd.to_datetime(df['date'])
            return df.sort_values('date')
        else:
            logger.warning(f"投资组合历史文件不存在: {self.portfolio_csv}")
            return pd.DataFrame(columns=['date', 'total_equity', 'cash', 'stock_value', 'daily_return'])
    
    def load_benchmark_data(self, benchmark_name: str, start_date: str = None) -> pd.DataFrame:
        """加载基准指数数据"""
        if benchmark_name not in self.benchmark_files:
            logger.error(f"未找到基准指数: {benchmark_name}")
            return pd.DataFrame()
        
        file_path = self.benchmark_files[benchmark_name]
        if not os.path.exists(file_path):
            logger.error(f"基准指数文件不存在: {file_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            if start_date:
                start_date = pd.to_datetime(start_date)
                df = df[df['date'] >= start_date]
            
            # 计算基准指数的日收益率
            df['daily_return'] = df['close'].pct_change()
            
            return df
        except Exception as e:
            logger.error(f"加载基准指数数据失败: {e}")
            return pd.DataFrame()
    
    def calculate_returns(self, portfolio_history: pd.DataFrame) -> Dict:
        """计算收益率指标"""
        if portfolio_history.empty or len(portfolio_history) < 2:
            return {}
        
        equity_series = portfolio_history['total_equity'].dropna()
        
        # 总收益率
        total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0]
        
        # 年化收益率
        days = len(equity_series)
        annual_return = (1 + total_return) ** (252 / days) - 1
        
        # 日收益率
        if 'daily_return' not in portfolio_history.columns:
            portfolio_history['daily_return'] = equity_series.pct_change()
        
        daily_returns = portfolio_history['daily_return'].dropna()
        
        # 平均日收益率
        avg_daily_return = daily_returns.mean()
        
        # 年化波动率
        annual_volatility = daily_returns.std() * np.sqrt(252)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'avg_daily_return': avg_daily_return,
            'annual_volatility': annual_volatility,
            'trading_days': days
        }
    
    def calculate_risk_metrics(self, portfolio_history: pd.DataFrame, risk_free_rate: float = 0.03) -> Dict:
        """计算风险指标"""
        if portfolio_history.empty or len(portfolio_history) < 2:
            return {}
        
        daily_returns = portfolio_history['daily_return'].dropna()
        
        if daily_returns.empty:
            return {}
        
        # 夏普比率
        excess_returns = daily_returns - risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / daily_returns.std() * np.sqrt(252)
        
        # 索提诺比率 (只考虑下行风险)
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 0:
            downside_std = negative_returns.std()
            sortino_ratio = excess_returns.mean() / downside_std * np.sqrt(252)
        else:
            sortino_ratio = np.inf
        
        # 最大回撤
        equity_series = portfolio_history['total_equity'].dropna()
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # VaR (5%分位数)
        var_5 = daily_returns.quantile(0.05)
        
        # 胜率
        win_rate = (daily_returns > 0).sum() / len(daily_returns)
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'var_5': var_5,
            'win_rate': win_rate
        }
    
    def calculate_benchmark_comparison(self, portfolio_history: pd.DataFrame, 
                                     benchmark_name: str = "沪深300") -> Dict:
        """与基准指数比较"""
        # 获取投资组合起始日期
        if portfolio_history.empty:
            return {}
        
        start_date = portfolio_history['date'].min().strftime('%Y-%m-%d')
        
        # 加载基准数据
        benchmark_data = self.load_benchmark_data(benchmark_name, start_date)
        
        if benchmark_data.empty:
            return {}
        
        # 对齐日期
        portfolio_aligned = portfolio_history.set_index('date')['daily_return'].dropna()
        benchmark_aligned = benchmark_data.set_index('date')['daily_return'].dropna()
        
        # 找到共同交易日
        common_dates = portfolio_aligned.index.intersection(benchmark_aligned.index)
        
        if len(common_dates) == 0:
            return {}
        
        portfolio_returns = portfolio_aligned.loc[common_dates]
        benchmark_returns = benchmark_aligned.loc[common_dates]
        
        # 计算超额收益
        excess_returns = portfolio_returns - benchmark_returns
        
        # 跟踪误差
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        # 信息比率
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Beta
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = benchmark_returns.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
        
        # Alpha (CAPM)
        risk_free_daily = 0.03 / 252  # 假设3%无风险利率
        alpha = portfolio_returns.mean() - risk_free_daily - beta * (benchmark_returns.mean() - risk_free_daily)
        alpha_annual = alpha * 252
        
        # 累计收益对比
        portfolio_cumulative = (1 + portfolio_returns).cumprod().iloc[-1] - 1
        benchmark_cumulative = (1 + benchmark_returns).cumprod().iloc[-1] - 1
        
        return {
            'benchmark_name': benchmark_name,
            'portfolio_cumulative_return': portfolio_cumulative,
            'benchmark_cumulative_return': benchmark_cumulative,
            'excess_return': portfolio_cumulative - benchmark_cumulative,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'beta': beta,
            'alpha_annual': alpha_annual,
            'correlation': portfolio_returns.corr(benchmark_returns)
        }
    
    def generate_performance_report(self, portfolio_history: pd.DataFrame) -> str:
        """生成性能分析报告"""
        if portfolio_history.empty:
            return "# 投资组合性能报告\n\n无历史数据可分析。"
        
        # 计算各项指标
        returns = self.calculate_returns(portfolio_history)
        risks = self.calculate_risk_metrics(portfolio_history)
        
        # 与多个基准比较
        benchmark_comparisons = {}
        for benchmark in ["沪深300", "创业板指", "科创50"]:
            comparison = self.calculate_benchmark_comparison(portfolio_history, benchmark)
            if comparison:
                benchmark_comparisons[benchmark] = comparison
        
        # 生成报告
        report = f"""# 📊 投资组合性能分析报告

## 📅 分析期间
- **开始日期**: {portfolio_history['date'].min().strftime('%Y-%m-%d')}
- **结束日期**: {portfolio_history['date'].max().strftime('%Y-%m-%d')}
- **交易天数**: {returns.get('trading_days', 0)}天

## 💰 收益率分析
- **总收益率**: {returns.get('total_return', 0):.2%}
- **年化收益率**: {returns.get('annual_return', 0):.2%}
- **平均日收益率**: {returns.get('avg_daily_return', 0):.4%}
- **年化波动率**: {returns.get('annual_volatility', 0):.2%}

## ⚠️ 风险指标
- **夏普比率**: {risks.get('sharpe_ratio', 0):.3f}
- **索提诺比率**: {risks.get('sortino_ratio', 0):.3f}
- **最大回撤**: {risks.get('max_drawdown', 0):.2%}
- **VaR(5%)**: {risks.get('var_5', 0):.2%}
- **胜率**: {risks.get('win_rate', 0):.1%}
"""

        # 添加基准比较
        if benchmark_comparisons:
            report += "\n## 📈 基准比较\n"
            
            for benchmark_name, comparison in benchmark_comparisons.items():
                report += f"""
### {benchmark_name}
- **组合累计收益**: {comparison.get('portfolio_cumulative_return', 0):.2%}
- **基准累计收益**: {comparison.get('benchmark_cumulative_return', 0):.2%}
- **超额收益**: {comparison.get('excess_return', 0):.2%}
- **信息比率**: {comparison.get('information_ratio', 0):.3f}
- **Beta**: {comparison.get('beta', 0):.3f}
- **Alpha(年化)**: {comparison.get('alpha_annual', 0):.2%}
- **相关性**: {comparison.get('correlation', 0):.3f}
"""

        # 添加性能评价
        report += self._generate_performance_evaluation(returns, risks, benchmark_comparisons)
        
        report += f"""
## 💡 改进建议

基于性能分析结果，建议关注以下方面：

1. **收益率优化**: {'当前收益率良好' if returns.get('annual_return', 0) > 0.1 else '考虑优化选股策略提高收益率'}
2. **风险控制**: {'风险控制适当' if risks.get('max_drawdown', 0) > -0.2 else '需要加强风险管理，控制回撤'}
3. **基准比较**: {'相对基准表现优异' if any(comp.get('excess_return', 0) > 0 for comp in benchmark_comparisons.values()) else '需要改进相对基准的表现'}

---
🤖 性能分析系统 | 灵感来源: ChatGPT Micro-Cap Experiment
"""
        return report
    
    def _generate_performance_evaluation(self, returns: Dict, risks: Dict, 
                                       benchmark_comparisons: Dict) -> str:
        """生成性能评价"""
        evaluation = "\n## 🎯 性能评价\n"
        
        # 收益率评价
        annual_return = returns.get('annual_return', 0)
        if annual_return > 0.2:
            evaluation += "- **收益表现**: 🟢 优秀 (年化收益率>20%)\n"
        elif annual_return > 0.1:
            evaluation += "- **收益表现**: 🟡 良好 (年化收益率10-20%)\n"
        elif annual_return > 0:
            evaluation += "- **收益表现**: 🟠 一般 (年化收益率0-10%)\n"
        else:
            evaluation += "- **收益表现**: 🔴 亏损 (负收益率)\n"
        
        # 风险评价
        sharpe = risks.get('sharpe_ratio', 0)
        if sharpe > 1.5:
            evaluation += "- **风险调整收益**: 🟢 优秀 (夏普比率>1.5)\n"
        elif sharpe > 1:
            evaluation += "- **风险调整收益**: 🟡 良好 (夏普比率1-1.5)\n"
        elif sharpe > 0.5:
            evaluation += "- **风险调整收益**: 🟠 一般 (夏普比率0.5-1)\n"
        else:
            evaluation += "- **风险调整收益**: 🔴 较差 (夏普比率<0.5)\n"
        
        # 回撤评价
        max_dd = risks.get('max_drawdown', 0)
        if max_dd > -0.1:
            evaluation += "- **回撤控制**: 🟢 优秀 (最大回撤<10%)\n"
        elif max_dd > -0.2:
            evaluation += "- **回撤控制**: 🟡 良好 (最大回撤10-20%)\n"
        elif max_dd > -0.3:
            evaluation += "- **回撤控制**: 🟠 一般 (最大回撤20-30%)\n"
        else:
            evaluation += "- **回撤控制**: 🔴 较差 (最大回撤>30%)\n"
        
        return evaluation
    
    def save_daily_performance(self, date: str, total_equity: float, 
                             cash: float, stock_value: float, 
                             daily_return: float = None):
        """保存每日性能数据"""
        # 读取现有数据
        if os.path.exists(self.portfolio_csv):
            df = pd.read_csv(self.portfolio_csv)
            df['date'] = pd.to_datetime(df['date'])
        else:
            df = pd.DataFrame(columns=['date', 'total_equity', 'cash', 'stock_value', 'daily_return'])
        
        # 计算日收益率
        if daily_return is None and not df.empty:
            prev_equity = df[df['date'] < pd.to_datetime(date)]['total_equity'].iloc[-1] if len(df) > 0 else total_equity
            daily_return = (total_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
        
        # 添加新记录
        new_record = {
            'date': pd.to_datetime(date),
            'total_equity': total_equity,
            'cash': cash,
            'stock_value': stock_value,
            'daily_return': daily_return or 0
        }
        
        # 检查是否已存在该日期的记录
        if not df.empty and pd.to_datetime(date) in df['date'].values:
            # 更新现有记录
            df.loc[df['date'] == pd.to_datetime(date), :] = list(new_record.values())
        else:
            # 添加新记录
            df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
        
        # 排序并保存
        df = df.sort_values('date').reset_index(drop=True)
        df.to_csv(self.portfolio_csv, index=False)
        
        logger.info(f"保存性能数据: {date}, 权益: ¥{total_equity:,.2f}, 收益率: {daily_return:.2%}")


def main():
    """测试性能跟踪器"""
    tracker = PerformanceTracker()
    
    # 模拟保存一些历史数据
    base_equity = 100000
    for i in range(30):
        date = (datetime.now() - timedelta(days=30-i)).strftime('%Y-%m-%d')
        # 模拟随机波动
        daily_return = np.random.normal(0.001, 0.02)  # 日均0.1%收益，2%波动
        base_equity *= (1 + daily_return)
        cash = base_equity * 0.1
        stock_value = base_equity * 0.9
        
        tracker.save_daily_performance(date, base_equity, cash, stock_value)
    
    # 生成性能报告
    history = tracker.load_portfolio_history()
    report = tracker.generate_performance_report(history)
    
    # 保存报告
    report_dir = "reports/performance"
    os.makedirs(report_dir, exist_ok=True)
    report_file = os.path.join(report_dir, f"performance_report_{datetime.now().strftime('%Y%m%d')}.md")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"性能报告已生成: {report_file}")


if __name__ == "__main__":
    main()