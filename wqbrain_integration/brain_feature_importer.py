#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BRAIN Alpha 特征导入器

功能:
1. 从 BRAIN 导入已验证的 alpha 表达式
2. 将 BRAIN 表达式编译为高效的 pandas 计算函数
3. 批量计算新特征并写入 v39_feature_cache / v40_feature_cache
4. 与现有训练管线无缝衔接

用法:
    importer = BrainFeatureImporter()

    # 导入单个 alpha
    importer.add_alpha('brain_momentum_decay',
                       'ts_decay_linear(returns, 10)',
                       description='10日线性衰减动量')

    # 从文件批量导入
    importer.load_from_json('brain_alphas.json')

    # 计算并写入缓存
    importer.compute_and_cache('2024-01-01', '2025-12-31')
"""

import os
import sys
import json
import sqlite3
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from core.config import get_db_path
    _DB_PATH = str(get_db_path())
except ImportError:
    _DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')

from .alpha_translator import AlphaTranslator

logger = logging.getLogger(__name__)


class BrainAlphaCompiler:
    """将 BRAIN FASTEXPR 编译为可执行的 pandas 函数"""

    def __init__(self):
        self.translator = AlphaTranslator()

    def compile(self, brain_expr: str) -> Callable:
        """
        将 BRAIN 表达式编译为接受 DataFrame 的计算函数

        Args:
            brain_expr: BRAIN FASTEXPR 表达式

        Returns:
            func(df) -> pd.Series
        """
        pandas_code = self.translator.parse_brain_expr(brain_expr)

        # 构建可执行函数
        def compute(df: pd.DataFrame) -> pd.Series:
            """动态执行翻译后的 pandas 代码"""
            safe_globals = {'np': np, 'pd': pd, '__builtins__': {}}
            local_ns = {'df': df}
            try:
                result = eval(pandas_code, safe_globals, local_ns)
                if isinstance(result, np.ndarray):
                    result = pd.Series(result, index=df.index)
                elif isinstance(result, (int, float)):
                    result = pd.Series(result, index=df.index)
                return result
            except Exception as e:
                logger.warning(f"Alpha 计算失败: {brain_expr} → {pandas_code}: {e}")
                return pd.Series(np.nan, index=df.index)

        compute.__doc__ = f'BRAIN: {brain_expr}\nPandas: {pandas_code}'
        compute._brain_expr = brain_expr
        compute._pandas_code = pandas_code
        return compute


class BrainFeatureImporter:
    """BRAIN Alpha 特征导入器"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or _DB_PATH
        self.compiler = BrainAlphaCompiler()

        # 已注册的 BRAIN alpha
        # {name: {'expr': str, 'func': Callable, 'description': str, 'brain_metrics': dict}}
        self.alphas: Dict[str, Dict] = {}

        # 缓存表名 (brain alpha 单独存表, 不污染 v39 缓存)
        self.cache_table = 'brain_alpha_cache'

    # ----------------------------------------------------------
    # Alpha 注册
    # ----------------------------------------------------------

    def add_alpha(self, name: str, brain_expr: str,
                  description: str = '',
                  brain_metrics: Dict = None) -> None:
        """
        注册一个 BRAIN alpha

        Args:
            name: 特征名 (会加 'brain_' 前缀如果没有)
            brain_expr: BRAIN FASTEXPR 表达式
            description: 说明
            brain_metrics: BRAIN 回测指标 {'sharpe': 2.1, 'ic': 0.05, ...}
        """
        if not name.startswith('brain_'):
            name = f'brain_{name}'

        func = self.compiler.compile(brain_expr)
        self.alphas[name] = {
            'expr': brain_expr,
            'func': func,
            'description': description,
            'brain_metrics': brain_metrics or {},
            'pandas_code': func._pandas_code,
        }
        logger.info(f"注册 BRAIN alpha: {name} = {brain_expr}")

    def add_alpha_from_result(self, name: str, alpha_result) -> None:
        """从 BrainAPIClient 的 AlphaResult 导入"""
        self.add_alpha(
            name=name,
            brain_expr=alpha_result.expression,
            brain_metrics={
                'sharpe': alpha_result.sharpe,
                'fitness': alpha_result.fitness,
                'turnover': alpha_result.turnover,
                'ic': alpha_result.ic,
            }
        )

    def load_from_json(self, json_path: str) -> int:
        """
        从 JSON 文件批量导入 alpha

        JSON 格式:
        [
            {
                "name": "momentum_decay_10",
                "expression": "ts_decay_linear(returns, 10)",
                "description": "10日线性衰减动量",
                "brain_metrics": {"sharpe": 2.1, "ic": 0.05}
            },
            ...
        ]
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            alphas = json.load(f)

        count = 0
        for alpha in alphas:
            name = alpha.get('name', f'alpha_{count}')
            expr = alpha.get('expression', alpha.get('brain_expr', ''))
            if not expr:
                continue
            self.add_alpha(
                name=name,
                brain_expr=expr,
                description=alpha.get('description', ''),
                brain_metrics=alpha.get('brain_metrics', {}),
            )
            count += 1
        logger.info(f"从 {json_path} 导入 {count} 个 alpha")
        return count

    def save_registry(self, output_path: str = None) -> str:
        """保存当前注册的 alpha 到 JSON"""
        if output_path is None:
            output_path = str(Path(__file__).parent / 'registered_alphas.json')

        data = []
        for name, info in self.alphas.items():
            data.append({
                'name': name,
                'expression': info['expr'],
                'description': info['description'],
                'brain_metrics': info['brain_metrics'],
                'pandas_code': info['pandas_code'],
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"保存 {len(data)} 个 alpha 到 {output_path}")
        return output_path

    # ----------------------------------------------------------
    # 特征计算
    # ----------------------------------------------------------

    def compute_for_date(self, date: str, stock_data: Dict[str, pd.DataFrame] = None
                         ) -> pd.DataFrame:
        """
        计算所有注册 alpha 在指定日期的值

        Args:
            date: 交易日期 (YYYY-MM-DD)
            stock_data: {code: DataFrame with OHLCV columns}
                       如果 None, 从数据库加载

        Returns:
            DataFrame with columns [code, trade_date, alpha1, alpha2, ...]
        """
        if not self.alphas:
            logger.warning("没有注册任何 alpha")
            return pd.DataFrame()

        if stock_data is None:
            stock_data = self._load_stock_data(date, lookback=60)

        rows = []
        for code, df in stock_data.items():
            if df.empty or len(df) < 5:
                continue

            row = {'code': code, 'trade_date': date}
            for alpha_name, alpha_info in self.alphas.items():
                try:
                    values = alpha_info['func'](df)
                    # 取最后一个值 (当日)
                    row[alpha_name] = float(values.iloc[-1]) if not pd.isna(values.iloc[-1]) else 0.0
                except Exception:
                    row[alpha_name] = 0.0
            rows.append(row)

        result = pd.DataFrame(rows)
        logger.info(f"计算完成: {date}, {len(result)} 只股票, {len(self.alphas)} 个 alpha")
        return result

    def compute_and_cache(self, start_date: str, end_date: str,
                          batch_size: int = 5) -> int:
        """
        批量计算 BRAIN alpha 并写入数据库缓存

        Args:
            start_date: 开始日期
            end_date: 结束日期
            batch_size: 每批处理天数

        Returns:
            总计算记录数
        """
        if not self.alphas:
            logger.warning("没有注册任何 alpha")
            return 0

        # 确保缓存表存在
        self._ensure_cache_table()

        # 获取交易日历
        trade_dates = self._get_trade_dates(start_date, end_date)
        logger.info(f"BRAIN alpha 缓存: {len(trade_dates)} 个交易日, "
                    f"{len(self.alphas)} 个 alpha")

        total_records = 0
        conn = sqlite3.connect(self.db_path)

        for i in range(0, len(trade_dates), batch_size):
            batch_dates = trade_dates[i:i + batch_size]

            for date in batch_dates:
                # 检查是否已缓存
                cursor = conn.execute(
                    f"SELECT COUNT(*) FROM {self.cache_table} WHERE trade_date = ?",
                    (date,)
                )
                if cursor.fetchone()[0] > 0:
                    continue

                # 加载数据并计算
                stock_data = self._load_stock_data(date, lookback=60)
                result_df = self.compute_for_date(date, stock_data)

                if result_df.empty:
                    continue

                # 写入缓存
                alpha_names = list(self.alphas.keys())
                for _, row in result_df.iterrows():
                    features_json = json.dumps({
                        name: float(row.get(name, 0)) for name in alpha_names
                    })
                    conn.execute(
                        f"""INSERT OR REPLACE INTO {self.cache_table}
                            (code, trade_date, features_json)
                            VALUES (?, ?, ?)""",
                        (row['code'], date, features_json)
                    )

                total_records += len(result_df)

            conn.commit()
            logger.info(f"  进度: {min(i + batch_size, len(trade_dates))}/{len(trade_dates)}, "
                        f"累计 {total_records} 条")

        conn.close()
        logger.info(f"缓存完成: {total_records} 条记录")
        return total_records

    # ----------------------------------------------------------
    # 与训练管线对接
    # ----------------------------------------------------------

    def get_features_for_training(self, start_date: str, end_date: str
                                   ) -> Optional[pd.DataFrame]:
        """
        从缓存加载 BRAIN alpha 特征, 格式与 v39_feature_cache 兼容

        Returns:
            DataFrame [code, trade_date, brain_alpha_1, brain_alpha_2, ...]
            可直接 merge 到训练数据
        """
        conn = sqlite3.connect(self.db_path)

        try:
            query = f"""
                SELECT code, trade_date, features_json
                FROM {self.cache_table}
                WHERE trade_date >= ? AND trade_date <= ?
                ORDER BY trade_date, code
            """
            df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        except Exception as e:
            logger.error(f"加载缓存失败: {e}")
            conn.close()
            return None

        conn.close()

        if df.empty:
            return None

        # 展开 features_json
        try:
            import orjson
            _loads = orjson.loads
        except ImportError:
            _loads = json.loads

        features = pd.json_normalize(df['features_json'].apply(_loads))
        result = pd.concat([df[['code', 'trade_date']], features], axis=1)
        result.drop(columns=['features_json'], errors='ignore', inplace=True)

        logger.info(f"加载 BRAIN alpha 特征: {len(result)} 条, "
                    f"{len(features.columns)} 个特征")
        return result

    def merge_with_v39_cache(self, v39_df: pd.DataFrame,
                              start_date: str = None,
                              end_date: str = None) -> pd.DataFrame:
        """
        将 BRAIN alpha 特征合并到 v39 训练数据中

        Args:
            v39_df: 从 v39_feature_cache 加载的训练数据
            start_date: 过滤开始日期
            end_date: 过滤结束日期

        Returns:
            合并后的 DataFrame (额外列: brain_xxx)
        """
        if start_date is None:
            start_date = v39_df['trade_date'].min()
        if end_date is None:
            end_date = v39_df['trade_date'].max()

        brain_df = self.get_features_for_training(start_date, end_date)
        if brain_df is None or brain_df.empty:
            logger.warning("无 BRAIN alpha 缓存数据, 返回原始 v39 数据")
            return v39_df

        merged = v39_df.merge(brain_df, on=['code', 'trade_date'], how='left')

        # NaN 填 0 (缓存可能不完整)
        brain_cols = [c for c in brain_df.columns if c not in ('code', 'trade_date')]
        merged[brain_cols] = merged[brain_cols].fillna(0.0)

        logger.info(f"合并完成: {len(v39_df)} → {len(merged)} 条, "
                    f"新增 {len(brain_cols)} 个 BRAIN 特征")
        return merged

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    def _ensure_cache_table(self):
        """创建 BRAIN alpha 缓存表"""
        conn = sqlite3.connect(self.db_path)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.cache_table} (
                code TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                features_json TEXT,
                PRIMARY KEY (code, trade_date)
            )
        """)
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.cache_table}_date
            ON {self.cache_table}(trade_date)
        """)
        conn.commit()
        conn.close()

    def _get_trade_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取日期范围内的交易日"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT DISTINCT trade_date FROM daily_quotes
            WHERE trade_date >= ? AND trade_date <= ?
            ORDER BY trade_date
        """, (start_date, end_date))
        dates = [row[0] for row in cursor.fetchall()]
        conn.close()
        return dates

    def _load_stock_data(self, date: str, lookback: int = 60) -> Dict[str, pd.DataFrame]:
        """加载股票数据"""
        end_date = datetime.strptime(date, '%Y-%m-%d')
        start_date = end_date - timedelta(days=lookback + 30)

        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT s.code, q.trade_date, q.open, q.high, q.low, q.close,
                   q.volume, q.price_change_pct
            FROM daily_quotes q
            JOIN securities s ON q.security_id = s.id
            WHERE s.type = 'A股'
            AND q.trade_date >= ? AND q.trade_date <= ?
            ORDER BY s.code, q.trade_date
        """
        df = pd.read_sql_query(query, conn,
                               params=(start_date.strftime('%Y-%m-%d'), date))

        # 加载估值数据 (pe, pb, ps, turnover, market_cap)
        basic_query = """
            SELECT s.code, db.trade_date, db.pe_ttm, db.pb, db.ps_ttm,
                   db.turnover_rate, db.total_mv
            FROM daily_basic db
            JOIN securities s ON db.security_id = s.id
            WHERE db.trade_date >= ? AND db.trade_date <= ?
        """
        basic_df = pd.read_sql_query(basic_query, conn,
                                      params=(start_date.strftime('%Y-%m-%d'), date))
        conn.close()

        # 合并
        if not basic_df.empty:
            df = df.merge(basic_df, on=['code', 'trade_date'], how='left')

        # 按股票分组
        result = {}
        for code, group in df.groupby('code'):
            result[code] = group.reset_index(drop=True)

        return result

    # ----------------------------------------------------------
    # 工具方法
    # ----------------------------------------------------------

    def list_alphas(self) -> List[Dict]:
        """列出所有注册的 alpha"""
        return [
            {
                'name': name,
                'expression': info['expr'],
                'description': info['description'],
                'brain_metrics': info['brain_metrics'],
            }
            for name, info in self.alphas.items()
        ]

    def summary(self) -> str:
        """摘要信息"""
        lines = [f'BRAIN Alpha 导入器: {len(self.alphas)} 个已注册']
        for name, info in self.alphas.items():
            metrics = info['brain_metrics']
            m_str = ''
            if metrics:
                m_str = f" (Sharpe={metrics.get('sharpe', 'N/A')}, IC={metrics.get('ic', 'N/A')})"
            lines.append(f"  {name}: {info['expr']}{m_str}")
        return '\n'.join(lines)


# ============================================================
# 预置的高质量 BRAIN Alpha (从 WorldQuant 101 和社区精选)
# ============================================================

def load_wq101_alphas() -> BrainFeatureImporter:
    """
    加载 WorldQuant 101 Alphas 中适合A股的经典因子

    这些因子已在全球市场验证, 可直接用于特征增强
    """
    importer = BrainFeatureImporter()

    # --- 动量类 ---
    importer.add_alpha(
        'wq101_momentum_decay5',
        'ts_decay_linear(returns, 5)',
        description='WQ101: 5日线性衰减动量',
    )
    importer.add_alpha(
        'wq101_momentum_decay10',
        'ts_decay_linear(returns, 10)',
        description='WQ101: 10日线性衰减动量',
    )
    importer.add_alpha(
        'wq101_volume_surprise',
        'rank(volume / ts_mean(volume, 20))',
        description='WQ101: 成交量突变排名',
    )

    # --- 反转类 ---
    importer.add_alpha(
        'wq101_short_reversal',
        '-ts_delta(close, 3) / ts_delay(close, 3)',
        description='WQ101: 3日短期反转',
    )
    importer.add_alpha(
        'wq101_volume_reversal',
        '-ts_corr(rank(volume), rank(close), 5)',
        description='WQ101: 5日量价反转相关性',
    )

    # --- 波动率类 ---
    importer.add_alpha(
        'wq101_realized_vol_ratio',
        'ts_std_dev(returns, 5) / (ts_std_dev(returns, 20) + 0.00000001)',
        description='WQ101: 短期/长期已实现波动率比',
    )
    importer.add_alpha(
        'wq101_vol_of_vol',
        'ts_std_dev(ts_std_dev(returns, 5), 20)',
        description='WQ101: 波动率的波动率 (Vol-of-Vol)',
    )

    # --- 量价结构 ---
    importer.add_alpha(
        'wq101_intraday_intensity',
        '(2 * close - high - low) / (high - low + 0.00000001)',
        description='WQ101: 日内强度 (收盘偏向)',
    )
    importer.add_alpha(
        'wq101_high_low_ratio',
        'ts_mean(high / low - 1, 10)',
        description='WQ101: 10日平均日内振幅',
    )
    importer.add_alpha(
        'wq101_close_to_high',
        'ts_mean((high - close) / (high - low + 0.00000001), 10)',
        description='WQ101: 10日收盘偏离最高价程度',
    )

    # --- 流动性类 ---
    importer.add_alpha(
        'wq101_turnover_momentum',
        'ts_delta(volume / ts_mean(volume, 20), 5)',
        description='WQ101: 换手率动量 (5日量比变化)',
    )

    # --- A股特色 ---
    importer.add_alpha(
        'wq101_limit_up_pressure',
        'ts_sum(if_else(returns > 0.09, 1, 0), 20) / 20',
        description='A股: 20日涨停频率 (主板)',
    )
    importer.add_alpha(
        'wq101_volume_price_divergence',
        'ts_corr(close, volume, 20) - ts_corr(close, volume, 5)',
        description='A股: 量价背离 (长期相关-短期相关)',
    )

    logger.info(f"加载 {len(importer.alphas)} 个 WQ101 预置 alpha")
    return importer


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='BRAIN Alpha 特征导入器')
    sub = parser.add_subparsers(dest='cmd')

    # load-wq101
    wq_p = sub.add_parser('load-wq101', help='加载 WQ101 预置因子')
    wq_p.add_argument('--compute', action='store_true', help='计算并缓存')
    wq_p.add_argument('--start-date', default='2024-01-01')
    wq_p.add_argument('--end-date', default='2025-12-31')

    # import
    imp_p = sub.add_parser('import', help='从 JSON 导入')
    imp_p.add_argument('file', type=str, help='JSON 文件')
    imp_p.add_argument('--compute', action='store_true')
    imp_p.add_argument('--start-date', default='2024-01-01')
    imp_p.add_argument('--end-date', default='2025-12-31')

    # test
    test_p = sub.add_parser('test', help='测试单个 alpha 计算')
    test_p.add_argument('expression', type=str, help='BRAIN 表达式')
    test_p.add_argument('--date', default='2025-12-31')
    test_p.add_argument('--top-n', type=int, default=10)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    if args.cmd == 'load-wq101':
        importer = load_wq101_alphas()
        print(importer.summary())
        if args.compute:
            n = importer.compute_and_cache(args.start_date, args.end_date)
            print(f'\n缓存 {n} 条记录')

    elif args.cmd == 'import':
        importer = BrainFeatureImporter()
        count = importer.load_from_json(args.file)
        print(f'导入 {count} 个 alpha')
        print(importer.summary())
        if args.compute:
            n = importer.compute_and_cache(args.start_date, args.end_date)
            print(f'\n缓存 {n} 条记录')

    elif args.cmd == 'test':
        importer = BrainFeatureImporter()
        importer.add_alpha('test', args.expression, description='测试')
        result = importer.compute_for_date(args.date)
        if not result.empty:
            result = result.sort_values('brain_test', ascending=False)
            print(f'\n=== {args.expression} @ {args.date} ===')
            print(f'计算 {len(result)} 只股票')
            print(f'\nTop {args.top_n}:')
            for _, row in result.head(args.top_n).iterrows():
                print(f"  {row['code']}  {row['brain_test']:+.6f}")
            print(f'\nBottom {args.top_n}:')
            for _, row in result.tail(args.top_n).iterrows():
                print(f"  {row['code']}  {row['brain_test']:+.6f}")

    else:
        parser.print_help()
