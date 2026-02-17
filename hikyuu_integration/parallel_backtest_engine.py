#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ParallelBacktestEngine - 并行回测引擎

支持多进程并行回测，提升回测速度
"""

import logging
from typing import List, Dict, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import time

from .backtest_engine import HikyuuStyleBacktestEngine, BacktestResult
from .signal_base import SignalBase
from .money_manager import MoneyManagerBase
from .stop_loss import StopLossBase

logger = logging.getLogger(__name__)


def _run_single_stock_backtest(args):
    """
    单只股票回测（用于并行执行）

    这个函数会在单独的进程中执行

    参数:
        args: (stock_code, config) 元组
            stock_code: 股票代码
            config: 回测配置字典

    返回:
        (stock_code, result_dict) 元组
    """
    stock_code, config = args

    try:
        # 在子进程中重新创建数据适配器和引擎
        from data_adapter.database_manager import DatabaseManager
        from .data_adapter import HikyuuStyleDataAdapter
        import importlib

        # 重新创建数据库连接（每个进程独立连接）
        db = DatabaseManager(db_path=config['db_path'])
        adapter = HikyuuStyleDataAdapter(db_manager=db)

        # 动态导入并创建Signal对象
        signal_module_name, signal_class_name = config['signal_class_name'].rsplit('.', 1)
        signal_module = importlib.import_module(signal_module_name)
        SignalClass = getattr(signal_module, signal_class_name)
        signal = SignalClass(**config['signal_params'])

        # 动态导入并创建MM对象
        money_manager = None
        if config.get('mm_class_name'):
            mm_module_name, mm_class_name = config['mm_class_name'].rsplit('.', 1)
            mm_module = importlib.import_module(mm_module_name)
            MMClass = getattr(mm_module, mm_class_name)
            money_manager = MMClass(**config['mm_params'])

        # 动态导入并创建SL对象
        stop_loss = None
        if config.get('sl_class_name'):
            sl_module_name, sl_class_name = config['sl_class_name'].rsplit('.', 1)
            sl_module = importlib.import_module(sl_module_name)
            SLClass = getattr(sl_module, sl_class_name)
            stop_loss = SLClass(**config['sl_params'])

        # 创建回测引擎（单只股票）
        engine = HikyuuStyleBacktestEngine(
            data_adapter=adapter,
            signal=signal,
            money_manager=money_manager,
            stop_loss=stop_loss,
            initial_cash=config['initial_cash'],
            max_positions=1  # 单股票回测，最大持仓=1
        )

        # 运行回测
        result = engine.run(
            stock_list=[stock_code],
            start_date=config['start_date'],
            end_date=config['end_date']
        )

        # 返回结果字典（用于进程间传输）
        return (stock_code, {
            'portfolio': result.portfolio,
            'trades': result.portfolio.trades,
            'total_return': result.total_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown_pct,
            'win_rate': result.win_rate,
            'start_date': config['start_date'],
            'end_date': config['end_date']
        })

    except Exception as e:
        logger.error(f"回测 {stock_code} 失败: {e}")
        return (stock_code, None)


class ParallelBacktestEngine:
    """
    并行回测引擎

    支持多只股票并行回测，充分利用多核CPU

    特点:
    - 股票级并行：每只股票独立回测
    - 自动进程池管理
    - 结果合并和统计
    - 进度追踪

    示例:
        engine = ParallelBacktestEngine(
            data_adapter=adapter,
            signal=BBISignal(),
            max_workers=4
        )

        result = engine.run(
            stock_list=['000001', '000002', ...],
            start_date='2025-01-01',
            end_date='2025-09-30'
        )
    """

    def __init__(self,
                 data_adapter,
                 signal: SignalBase,
                 money_manager: Optional[MoneyManagerBase] = None,
                 stop_loss: Optional[StopLossBase] = None,
                 initial_cash: float = 100000,
                 max_positions: int = 10,
                 max_workers: Optional[int] = None):
        """
        初始化并行回测引擎

        参数:
            data_adapter: 数据适配器
            signal: 信号对象
            money_manager: 资金管理对象（可选）
            stop_loss: 止损策略对象（可选）
            initial_cash: 初始资金
            max_positions: 最大持仓数
            max_workers: 最大工作进程数（默认=CPU核心数）
        """
        self.data_adapter = data_adapter
        self.signal = signal
        self.money_manager = money_manager
        self.stop_loss = stop_loss
        self.initial_cash = initial_cash
        self.max_positions = max_positions

        # 并行参数
        self.max_workers = max_workers or cpu_count()

        logger.info(f"✅ ParallelBacktestEngine initialized (max_workers={self.max_workers})")

    def _extract_init_params(self, obj) -> dict:
        """
        提取对象的初始化参数

        从MoneyManager或StopLoss对象中提取初始化参数

        参数:
            obj: MoneyManager或StopLoss对象

        返回:
            参数字典
        """
        import inspect

        # 获取__init__方法的签名
        init_signature = inspect.signature(obj.__class__.__init__)
        param_names = [p.name for p in init_signature.parameters.values() if p.name != 'self']

        # 只提取__init__方法接受的参数
        params = {}
        for key, value in obj.__dict__.items():
            if key in param_names and isinstance(value, (int, float, str, bool, list, dict, tuple, type(None))):
                params[key] = value

        return params

    def run(self,
            stock_list: List[str],
            start_date: str,
            end_date: str,
            on_stock_complete: Optional[Callable] = None) -> Dict:
        """
        运行并行回测

        参数:
            stock_list: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            on_stock_complete: 单只股票完成时的回调函数（可选）

        返回:
            回测结果字典
        """
        logger.info(f"🚀 开始并行回测: {len(stock_list)}只股票, "
                   f"{start_date} → {end_date}, "
                   f"工作进程={self.max_workers}")

        start_time = time.time()

        # 构建配置字典（用于传递给子进程）
        # 注意：由于multiprocessing的限制，我们传递类名字符串而不是类对象
        config = {
            'db_path': self.data_adapter.db.db_path,
            'signal_class_name': f"{self.signal.__class__.__module__}.{self.signal.__class__.__name__}",
            'signal_params': getattr(self.signal, 'params', {}),
            'mm_class_name': f"{self.money_manager.__class__.__module__}.{self.money_manager.__class__.__name__}" if self.money_manager else None,
            'mm_params': self._extract_init_params(self.money_manager) if self.money_manager else {},
            'sl_class_name': f"{self.stop_loss.__class__.__module__}.{self.stop_loss.__class__.__name__}" if self.stop_loss else None,
            'sl_params': self._extract_init_params(self.stop_loss) if self.stop_loss else {},
            'initial_cash': self.initial_cash,
            'start_date': start_date,
            'end_date': end_date
        }

        # 创建任务列表
        tasks = [(code, config) for code in stock_list]

        # 并行执行回测
        results = {}
        completed = 0
        failed = 0

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_code = {
                executor.submit(_run_single_stock_backtest, task): task[0]
                for task in tasks
            }

            # 收集结果
            for future in as_completed(future_to_code):
                stock_code = future_to_code[future]

                try:
                    code, result = future.result()

                    if result is not None:
                        results[code] = result
                        completed += 1

                        if on_stock_complete:
                            on_stock_complete(code, result)

                        logger.info(f"✅ {code} 回测完成 ({completed}/{len(stock_list)})")
                    else:
                        failed += 1
                        logger.warning(f"❌ {code} 回测失败")

                except Exception as e:
                    failed += 1
                    logger.error(f"❌ {stock_code} 回测异常: {e}")

        elapsed_time = time.time() - start_time

        # 合并结果
        merged_result = self._merge_results(results, start_date, end_date)

        logger.info(f"\n{'='*80}")
        logger.info(f"🎉 并行回测完成!")
        logger.info(f"{'='*80}")
        logger.info(f"总股票数: {len(stock_list)}")
        logger.info(f"成功: {completed}, 失败: {failed}")
        logger.info(f"耗时: {elapsed_time:.2f}秒")
        logger.info(f"平均速度: {elapsed_time/len(stock_list):.3f}秒/股票")
        logger.info(f"{'='*80}\n")

        return merged_result

    def _merge_results(self, results: Dict, start_date: str, end_date: str) -> Dict:
        """
        合并多只股票的回测结果

        参数:
            results: {stock_code: result_dict} 字典
            start_date: 开始日期
            end_date: 结束日期

        返回:
            合并后的回测结果字典
        """
        if not results:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'total_pnl': 0.0,
                'by_stock': {},
                'start_date': start_date,
                'end_date': end_date,
                'num_stocks': 0
            }

        # 统计各股票结果
        by_stock = {}
        total_trades = 0
        total_pnl = 0.0
        winning_trades = 0

        for code, result in results.items():
            by_stock[code] = {
                'total_return': result['total_return'],
                'sharpe_ratio': result['sharpe_ratio'],
                'max_drawdown': result['max_drawdown'],
                'win_rate': result['win_rate'],
                'trades': len(result['trades'])
            }

            total_trades += len(result['trades'])

            for trade in result['trades']:
                pnl = trade.pnl
                total_pnl += pnl
                if pnl > 0:
                    winning_trades += 1

        # 计算总体指标
        avg_return = sum(r['total_return'] for r in results.values()) / len(results)
        avg_sharpe = sum(r['sharpe_ratio'] for r in results.values()) / len(results)
        avg_drawdown = sum(r['max_drawdown'] for r in results.values()) / len(results)
        overall_win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

        return {
            'total_return': avg_return,
            'sharpe_ratio': avg_sharpe,
            'max_drawdown': avg_drawdown,
            'win_rate': overall_win_rate,
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'by_stock': by_stock,
            'start_date': start_date,
            'end_date': end_date,
            'num_stocks': len(results)
        }


# 使用示例
if __name__ == '__main__':
    from data_adapter.database_manager import DatabaseManager
    from .data_adapter import HikyuuStyleDataAdapter
    from .signal_base import BBISignal
    from .money_manager import MM_FixedPercent
    from .stop_loss import ST_FixedPercent

    print("🚀 并行回测引擎测试")

    # 创建数据适配器
    db = DatabaseManager(db_path='data_adapter/stock_data.db')
    adapter = HikyuuStyleDataAdapter(db_manager=db)

    # 获取测试股票
    stocks = adapter.get_all_stocks('A股')[:20]  # 测试20只股票
    print(f"测试股票: {stocks[:5]}... (共{len(stocks)}只)")

    # 创建并行回测引擎
    engine = ParallelBacktestEngine(
        data_adapter=adapter,
        signal=BBISignal(),
        money_manager=MM_FixedPercent(0.2),
        stop_loss=ST_FixedPercent(0.08),
        initial_cash=100000,
        max_workers=4
    )

    # 运行并行回测
    result = engine.run(
        stock_list=stocks,
        start_date='2025-07-01',
        end_date='2025-09-30'
    )

    # 打印结果
    print("\n📊 回测结果:")
    print(f"总收益率: {result['total_return']:.2f}%")
    print(f"夏普比率: {result['sharpe_ratio']:.2f}")
    print(f"最大回撤: {result['max_drawdown']:.2f}%")
    print(f"胜率: {result['win_rate']:.2f}%")
    print(f"总交易次数: {result['total_trades']}")
