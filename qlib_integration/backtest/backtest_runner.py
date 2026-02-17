"""
回测运行器 - 整合StockTradebyZ策略与Qlib回测框架的主要入口

主要功能：
1. 配置和初始化Qlib回测环境
2. 集成StockTradebyZ的数据和策略
3. 运行回测并生成专业报告
4. 支持多策略对比和参数优化
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from pathlib import Path

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# 导入Qlib组件
try:
    import qlib
    from qlib import init
    from qlib.backtest import backtest_loop
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.backtest.account import Account
    try:
        from qlib.portfolio import Portfolio
        from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
    except ImportError:
        # 这些模块可能在某些版本中不存在，先跳过
        Portfolio = None
        SignalRecord = None
        PortAnaRecord = None
    from qlib.utils import get_or_create_path
except ImportError as e:
    raise ImportError(f"请先安装qlib: pip install qlib\n{e}")

# 导入本地模块
from .data_adapter import StockTradebyzDataAdapter
from .stocktrader_strategy import StockTraderStrategy
from .chinese_exchange import ChineseAShareExchange
from data_adapter.database_manager import DatabaseManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestRunner:
    """
    StockTradebyZ回测运行器
    
    整合数据、策略、交易所等组件，提供完整的回测解决方案
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 qlib_provider_uri: Optional[str] = None):
        """
        初始化回测运行器
        
        Args:
            config_path: 配置文件路径
            qlib_provider_uri: Qlib数据提供者URI
        """
        self.config = self._load_config(config_path)
        self.db_manager = DatabaseManager()
        self.data_adapter = StockTradebyzDataAdapter()
        
        # 初始化Qlib环境
        self._init_qlib_environment(qlib_provider_uri)
        
        # 回测结果存储
        self.backtest_results = {}
        self.performance_metrics = {}
        
        logger.info("StockTradebyZ回测运行器初始化完成")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """加载配置文件"""
        if config_path is None:
            config_path = os.path.join(current_dir, "config", "a_share_config.yaml")
        
        default_config = {
            'backtest': {
                'start_time': '2023-01-01',
                'end_time': '2024-12-31', 
                'benchmark': '000300.SH',  # 沪深300
                'account': 1000000,        # 100万初始资金
                'freq': 'day'
            },
            'strategy': {
                'strategies': ['bbikdj', 'bbilongshort', 'breakout', 'peak'],
                'max_positions': 10,
                'position_size': 0.1,
                'stop_loss': 0.08,
                'take_profit': 0.15,
                'min_score': 70.0,
                'rebalance_freq': 5
            },
            'exchange': {
                'limit_threshold': 0.095,
                'deal_price': 'close',
                'trade_unit': 100,
                'open_cost': 0.0003,
                'close_cost': 0.0013,  # 包含印花税
                'min_cost': 5
            },
            'universe': {
                'min_trading_days': 100,
                'exclude_st': True,
                'exclude_new': True,
                'max_stocks': 2000
            }
        }
        
        try:
            if os.path.exists(config_path):
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                    # 递归更新配置
                    self._update_dict(default_config, user_config)
        except Exception as e:
            logger.warning(f"加载配置文件失败，使用默认配置: {e}")
        
        return default_config
    
    def _update_dict(self, target: Dict, source: Dict):
        """递归更新字典"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_dict(target[key], value)
            else:
                target[key] = value
    
    def _init_qlib_environment(self, provider_uri: Optional[str] = None):
        """初始化Qlib环境"""
        try:
            if provider_uri is None:
                # 使用本地数据适配器，不需要外部数据源
                provider_uri = "sqlite:///" + os.path.join(project_root, "stock_data.db")
            
            # 初始化qlib（简化配置，主要使用自定义数据适配器）
            qlib.init(
                provider_uri=provider_uri,
                region='cn',
                auto_mount=False  # 不自动挂载，使用自定义数据适配器
            )
            
            logger.info(f"Qlib环境初始化完成，数据源: {provider_uri}")
            
        except Exception as e:
            logger.warning(f"Qlib环境初始化失败: {e}")
            # 如果Qlib初始化失败，仍可以使用自定义组件
    
    def run_backtest(self,
                    strategy_config: Optional[Dict] = None,
                    backtest_config: Optional[Dict] = None,
                    save_results: bool = True) -> Dict[str, Any]:
        """
        运行单个回测
        
        Args:
            strategy_config: 策略配置
            backtest_config: 回测配置
            save_results: 是否保存结果
            
        Returns:
            回测结果字典
        """
        try:
            # 使用传入配置或默认配置
            strategy_config = strategy_config or self.config['strategy']
            backtest_config = backtest_config or self.config['backtest']
            
            logger.info(f"开始回测: {backtest_config['start_time']} - {backtest_config['end_time']}")
            
            # 1. 准备股票池
            stock_universe = self._prepare_stock_universe(
                backtest_config['start_time'],
                backtest_config['end_time']
            )
            
            if not stock_universe:
                raise ValueError("股票池为空，无法进行回测")
            
            # 2. 创建策略实例
            strategy = StockTraderStrategy(
                strategies=strategy_config['strategies'],
                max_positions=strategy_config['max_positions'],
                position_size=strategy_config['position_size'],
                stop_loss=strategy_config['stop_loss'],
                take_profit=strategy_config['take_profit'],
                min_score=strategy_config['min_score'],
                rebalance_freq=strategy_config['rebalance_freq']
            )
            
            # 3. 创建交易所
            # 过滤出交易所支持的参数
            exchange_config = self.config['exchange'].copy()
            
            # 移除不被Exchange基类支持的自定义参数
            custom_params = ['t_plus_1', 'min_trade_value', 'limit_orders', 'market_rules', 'trade_unit']
            for param in custom_params:
                if param in exchange_config:
                    del exchange_config[param]
            
            exchange = ChineseAShareExchange(
                start_time=backtest_config['start_time'],
                end_time=backtest_config['end_time'],
                freq=backtest_config['freq'],
                codes=stock_universe,
                **exchange_config
            )
            
            # 4. 创建执行器
            executor = SimulatorExecutor(
                time_per_step=backtest_config['freq'],
                start_time=backtest_config['start_time'],
                end_time=backtest_config['end_time'],
                trade_exchange=exchange,
                generate_portfolio_metrics=True
            )
            
            # 5. 创建账户
            account = Account(
                init_cash=backtest_config['account'],
                trade_exchange=exchange
            )
            
            # 6. 运行回测
            logger.info("执行回测循环...")
            portfolio_dict, indicator_dict = backtest_loop(
                start_time=backtest_config['start_time'],
                end_time=backtest_config['end_time'],
                trade_strategy=strategy,
                trade_executor=executor
            )
            
            # 7. 分析结果
            results = self._analyze_backtest_results(
                portfolio_dict, 
                indicator_dict, 
                backtest_config
            )
            
            # 8. 保存结果
            if save_results:
                self._save_results(results, strategy_config, backtest_config)
            
            logger.info(f"回测完成，年化收益率: {results.get('annual_return', 0):.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"回测执行失败: {e}")
            raise
    
    def _prepare_stock_universe(self, start_date: str, end_date: str) -> List[str]:
        """准备股票池"""
        try:
            universe_config = self.config['universe']
            
            # 获取基础股票池
            stock_list = self.data_adapter.get_stock_list(
                start_date=start_date,
                end_date=end_date,
                min_trading_days=universe_config['min_trading_days']
            )
            
            # 过滤条件
            if universe_config['exclude_st']:
                stock_list = [s for s in stock_list if 'ST' not in s]
            
            if universe_config['exclude_new']:
                # 排除上市不足1年的新股（简化处理）
                cutoff_date = pd.to_datetime(start_date) - timedelta(days=365)
                # 这里应该查询上市日期，简化处理暂跳过
            
            # 限制股票数量
            if universe_config['max_stocks'] and len(stock_list) > universe_config['max_stocks']:
                # 按市值或活跃度排序取前N只（简化为随机采样）
                import random
                stock_list = random.sample(stock_list, universe_config['max_stocks'])
            
            logger.info(f"股票池准备完成: {len(stock_list)}只股票")
            
            return stock_list
            
        except Exception as e:
            logger.error(f"准备股票池失败: {e}")
            return []
    
    def _analyze_backtest_results(self, 
                                portfolio_dict: Dict, 
                                indicator_dict: Dict,
                                backtest_config: Dict) -> Dict[str, Any]:
        """分析回测结果"""
        try:
            results = {
                'config': backtest_config,
                'portfolio': portfolio_dict,
                'indicators': indicator_dict,
                'performance': {}
            }
            
            # 提取主要性能指标
            if portfolio_dict:
                for freq, (portfolio_df, portfolio_info) in portfolio_dict.items():
                    if portfolio_df is not None and not portfolio_df.empty:
                        performance = self._calculate_performance_metrics(portfolio_df)
                        results['performance'][freq] = performance
            
            # 提取交易指标
            if indicator_dict:
                results['trade_indicators'] = {}
                for freq, (indicator_df, indicator_obj) in indicator_dict.items():
                    if indicator_df is not None and not indicator_df.empty:
                        results['trade_indicators'][freq] = {
                            'dataframe': indicator_df,
                            'summary': self._summarize_trade_indicators(indicator_df)
                        }
            
            # 计算主要的性能总结
            main_performance = results['performance'].get('day', {})
            results.update({
                'annual_return': main_performance.get('annual_return', 0),
                'total_return': main_performance.get('total_return', 0),
                'sharpe_ratio': main_performance.get('sharpe_ratio', 0),
                'max_drawdown': main_performance.get('max_drawdown', 0),
                'win_rate': main_performance.get('win_rate', 0),
                'profit_loss_ratio': main_performance.get('profit_loss_ratio', 0)
            })
            
            return results
            
        except Exception as e:
            logger.error(f"分析回测结果失败: {e}")
            return {'error': str(e)}
    
    def _calculate_performance_metrics(self, portfolio_df: pd.DataFrame) -> Dict[str, float]:
        """计算性能指标"""
        try:
            if 'return' not in portfolio_df.columns:
                return {}
            
            returns = portfolio_df['return'].dropna()
            
            if len(returns) == 0:
                return {}
            
            # 累计收益率
            cumulative_return = (1 + returns).prod() - 1
            
            # 年化收益率
            trading_days = len(returns)
            years = trading_days / 252  # 假设252个交易日/年
            annual_return = (1 + cumulative_return) ** (1 / years) - 1 if years > 0 else 0
            
            # 夏普比率
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # 最大回撤
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # 胜率
            win_rate = (returns > 0).mean()
            
            # 盈亏比
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            profit_loss_ratio = 0
            if len(negative_returns) > 0 and len(positive_returns) > 0:
                avg_gain = positive_returns.mean()
                avg_loss = abs(negative_returns.mean())
                profit_loss_ratio = avg_gain / avg_loss
            
            return {
                'total_return': cumulative_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_loss_ratio': profit_loss_ratio,
                'volatility': returns.std() * np.sqrt(252),
                'trading_days': trading_days
            }
            
        except Exception as e:
            logger.error(f"计算性能指标失败: {e}")
            return {}
    
    def _summarize_trade_indicators(self, indicator_df: pd.DataFrame) -> Dict[str, Any]:
        """总结交易指标"""
        try:
            summary = {}
            
            if 'ffr' in indicator_df.columns:  # 成交率
                summary['fulfill_rate'] = indicator_df['ffr'].mean()
            
            if 'pa' in indicator_df.columns:   # 价格优势
                summary['price_advantage'] = indicator_df['pa'].mean()
            
            if 'pos' in indicator_df.columns:  # 正收益率
                summary['positive_rate'] = indicator_df['pos'].mean()
            
            return summary
            
        except Exception as e:
            logger.debug(f"总结交易指标失败: {e}")
            return {}
    
    def _save_results(self, 
                     results: Dict[str, Any],
                     strategy_config: Dict,
                     backtest_config: Dict):
        """保存回测结果"""
        try:
            # 创建报告目录
            reports_dir = os.path.join(project_root, "reports", "qlib_backtest")
            os.makedirs(reports_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            strategies_str = '_'.join(strategy_config['strategies'])
            filename = f"backtest_{strategies_str}_{timestamp}"
            
            # 保存JSON结果
            json_path = os.path.join(reports_dir, f"{filename}.json")
            
            # 简化results以便JSON序列化
            simple_results = {
                'config': {
                    'strategy': strategy_config,
                    'backtest': backtest_config
                },
                'performance': results.get('performance', {}),
                'summary': {
                    'annual_return': results.get('annual_return', 0),
                    'total_return': results.get('total_return', 0),
                    'sharpe_ratio': results.get('sharpe_ratio', 0),
                    'max_drawdown': results.get('max_drawdown', 0),
                    'win_rate': results.get('win_rate', 0)
                }
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(simple_results, f, ensure_ascii=False, indent=2, default=str)
            
            # 生成Markdown报告
            md_path = os.path.join(reports_dir, f"{filename}.md")
            self._generate_markdown_report(results, strategy_config, backtest_config, md_path)
            
            logger.info(f"回测结果已保存:")
            logger.info(f"  JSON: {json_path}")
            logger.info(f"  报告: {md_path}")
            
        except Exception as e:
            logger.error(f"保存回测结果失败: {e}")
    
    def _generate_markdown_report(self, 
                                 results: Dict[str, Any],
                                 strategy_config: Dict,
                                 backtest_config: Dict,
                                 output_path: str):
        """生成Markdown格式的回测报告"""
        try:
            report_content = f"""# StockTradebyZ Qlib回测报告

## 📊 回测概述

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**回测期间**: {backtest_config['start_time']} 至 {backtest_config['end_time']}

**初始资金**: ¥{backtest_config['account']:,.0f}

**基准指数**: {backtest_config.get('benchmark', '沪深300')}

## 🎯 策略配置

**使用策略**: {', '.join(strategy_config['strategies'])}

**最大持仓**: {strategy_config['max_positions']}只股票

**单仓占比**: {strategy_config['position_size']:.1%}

**止损线**: {strategy_config['stop_loss']:.1%}

**止盈线**: {strategy_config['take_profit']:.1%}

**最低评分**: {strategy_config['min_score']}分

**调仓频率**: {strategy_config['rebalance_freq']}交易日

## 📈 核心业绩指标

| 指标 | 数值 |
|------|------|
| 总收益率 | {results.get('total_return', 0):.2%} |
| 年化收益率 | {results.get('annual_return', 0):.2%} |
| 夏普比率 | {results.get('sharpe_ratio', 0):.2f} |
| 最大回撤 | {results.get('max_drawdown', 0):.2%} |
| 胜率 | {results.get('win_rate', 0):.2%} |
| 盈亏比 | {results.get('profit_loss_ratio', 0):.2f} |

## 🔍 详细分析

### 策略表现
"""
            
            # 添加各频率的详细指标
            performance = results.get('performance', {})
            for freq, metrics in performance.items():
                report_content += f"""
#### {freq.upper()}频率指标

- **交易天数**: {metrics.get('trading_days', 0)}天
- **年化波动率**: {metrics.get('volatility', 0):.2%}
- **累计收益率**: {metrics.get('total_return', 0):.2%}
"""
            
            # 添加交易指标分析
            trade_indicators = results.get('trade_indicators', {})
            if trade_indicators:
                report_content += "\n### 交易执行质量\n"
                for freq, indicators in trade_indicators.items():
                    summary = indicators.get('summary', {})
                    if summary:
                        report_content += f"""
#### {freq.upper()}频率交易指标

- **成交率**: {summary.get('fulfill_rate', 0):.2%}
- **价格优势**: {summary.get('price_advantage', 0):.4f}
- **正收益率**: {summary.get('positive_rate', 0):.2%}
"""
            
            report_content += f"""

## 🎭 风险提示

1. 该回测基于历史数据，不能保证未来表现
2. 实际交易中可能面临滑点、冲击成本等额外费用
3. 市场环境变化可能影响策略有效性
4. 建议结合多种分析方法制定投资决策

## 🛠️ 技术说明

- **回测引擎**: Qlib Professional Backtesting Framework
- **数据源**: StockTradebyZ SQLite Database
- **策略实现**: 基于V3.0量化评分系统
- **市场规则**: 中国A股T+1交易制度，涨跌停限制
- **交易成本**: 佣金0.03% + 印花税0.1%(卖出) + 过户费

---

🤖 *本报告由StockTradebyZ x Qlib自动生成*

📧 *如有问题请查看系统日志或联系开发团队*
"""
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
                
        except Exception as e:
            logger.error(f"生成Markdown报告失败: {e}")
    
    def run_multi_strategy_comparison(self, 
                                    strategy_configs: List[Dict],
                                    backtest_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        运行多策略对比回测
        
        Args:
            strategy_configs: 多个策略配置列表
            backtest_config: 回测配置
            
        Returns:
            对比结果字典
        """
        logger.info(f"开始多策略对比回测，共{len(strategy_configs)}个策略")
        
        comparison_results = {
            'strategies': [],
            'summary': {},
            'comparison_table': []
        }
        
        for i, strategy_config in enumerate(strategy_configs):
            try:
                strategy_name = f"策略{i+1}({'+'.join(strategy_config['strategies'])})"
                logger.info(f"执行{strategy_name}")
                
                result = self.run_backtest(
                    strategy_config=strategy_config,
                    backtest_config=backtest_config,
                    save_results=False
                )
                
                comparison_results['strategies'].append({
                    'name': strategy_name,
                    'config': strategy_config,
                    'result': result
                })
                
                # 添加到对比表
                comparison_results['comparison_table'].append({
                    'strategy': strategy_name,
                    'annual_return': result.get('annual_return', 0),
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'max_drawdown': result.get('max_drawdown', 0),
                    'win_rate': result.get('win_rate', 0)
                })
                
            except Exception as e:
                logger.error(f"策略{i+1}执行失败: {e}")
        
        # 生成对比报告
        self._save_comparison_results(comparison_results, backtest_config)
        
        return comparison_results
    
    def _save_comparison_results(self, 
                               comparison_results: Dict[str, Any],
                               backtest_config: Optional[Dict]):
        """保存策略对比结果"""
        try:
            reports_dir = os.path.join(project_root, "reports", "qlib_backtest")
            os.makedirs(reports_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"strategy_comparison_{timestamp}.md"
            
            report_path = os.path.join(reports_dir, filename)
            
            # 生成对比报告
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# 多策略对比回测报告\n\n")
                f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                if backtest_config:
                    f.write(f"**回测期间**: {backtest_config['start_time']} - {backtest_config['end_time']}\n\n")
                
                f.write("## 策略对比表\n\n")
                f.write("| 策略 | 年化收益 | 夏普比率 | 最大回撤 | 胜率 |\n")
                f.write("|------|----------|----------|----------|------|\n")
                
                for row in comparison_results['comparison_table']:
                    f.write(f"| {row['strategy']} | "
                           f"{row['annual_return']:.2%} | "
                           f"{row['sharpe_ratio']:.2f} | "
                           f"{row['max_drawdown']:.2%} | "
                           f"{row['win_rate']:.2%} |\n")
                
                f.write("\n## 策略详情\n\n")
                
                for strategy_info in comparison_results['strategies']:
                    name = strategy_info['name']
                    config = strategy_info['config']
                    result = strategy_info['result']
                    
                    f.write(f"### {name}\n\n")
                    f.write(f"**策略配置**: {config}\n\n")
                    f.write(f"**年化收益率**: {result.get('annual_return', 0):.2%}\n\n")
                    f.write(f"**夏普比率**: {result.get('sharpe_ratio', 0):.2f}\n\n")
                    f.write("---\n\n")
            
            logger.info(f"策略对比报告已保存: {report_path}")
            
        except Exception as e:
            logger.error(f"保存策略对比结果失败: {e}")


# 便捷函数
def run_simple_backtest(strategies: List[str] = None,
                       start_date: str = '2023-01-01',
                       end_date: str = '2024-12-31',
                       initial_cash: float = 1000000) -> Dict[str, Any]:
    """
    运行简单回测的便捷函数
    
    Args:
        strategies: 策略列表
        start_date: 开始日期
        end_date: 结束日期
        initial_cash: 初始资金
        
    Returns:
        回测结果
    """
    if strategies is None:
        strategies = ['bbikdj', 'breakout']
    
    runner = BacktestRunner()
    
    strategy_config = {
        'strategies': strategies,
        'max_positions': 10,
        'position_size': 0.1,
        'min_score': 70.0
    }
    
    backtest_config = {
        'start_time': start_date,
        'end_time': end_date,
        'account': initial_cash,
        'freq': 'day'
    }
    
    return runner.run_backtest(strategy_config, backtest_config)


if __name__ == "__main__":
    # 测试运行
    try:
        logger.info("开始测试回测系统")
        result = run_simple_backtest(
            strategies=['bbikdj'],
            start_date='2024-01-01',
            end_date='2024-06-30'
        )
        
        print(f"回测完成，年化收益率: {result.get('annual_return', 0):.2%}")
        
    except Exception as e:
        logger.error(f"测试回测失败: {e}")