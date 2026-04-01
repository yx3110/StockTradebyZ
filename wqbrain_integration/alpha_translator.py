#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpha 双向翻译器

功能:
1. StockTradebyZ 特征 → BRAIN FASTEXPR 表达式
2. BRAIN FASTEXPR 表达式 → pandas 计算代码
3. 批量导出/导入

BRAIN 表达式语法参考:
- 截面算子: rank(x), zscore(x), scale(x), group_rank(x, group)
- 时序算子: ts_mean(x,d), ts_std_dev(x,d), ts_delta(x,d), ts_delay(x,d),
            ts_min(x,d), ts_max(x,d), ts_rank(x,d), ts_corr(x,y,d),
            ts_sum(x,d), ts_decay_linear(x,d), ts_arg_max(x,d), ts_arg_min(x,d)
- 数据字段: close, open, high, low, volume, returns, cap (market_cap),
            pe, pb, ps, turnover, cashflow_op, debt_lt, assets, ...
"""

import re
import json
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class AlphaDefinition:
    """Alpha 定义"""
    name: str                        # 特征名
    brain_expr: str                  # BRAIN FASTEXPR 表达式
    pandas_expr: str                 # pandas 等价代码
    category: str = 'stock'          # stock / macro / valuation / industry
    description: str = ''            # 中文说明
    brain_settings: Dict = field(default_factory=dict)  # BRAIN 提交参数


# ============================================================
# 我们系统的全部特征 → BRAIN 表达式映射
# ============================================================

FEATURE_TO_BRAIN: Dict[str, AlphaDefinition] = {}


def _register(name, brain_expr, pandas_expr, category='stock', description='',
              brain_settings=None):
    """注册一个特征翻译"""
    FEATURE_TO_BRAIN[name] = AlphaDefinition(
        name=name,
        brain_expr=brain_expr,
        pandas_expr=pandas_expr,
        category=category,
        description=description,
        brain_settings=brain_settings or {},
    )


# ---------- 收益类 ----------
_register(
    'return_5d',
    brain_expr='ts_delta(close, 5) / ts_delay(close, 5)',
    pandas_expr='df["close"] / df["close"].shift(5) - 1',
    description='5日收益率',
)
_register(
    'return_10d',
    brain_expr='ts_delta(close, 10) / ts_delay(close, 10)',
    pandas_expr='df["close"] / df["close"].shift(10) - 1',
    description='10日收益率',
)
_register(
    'return_20d',
    brain_expr='ts_delta(close, 20) / ts_delay(close, 20)',
    pandas_expr='df["close"] / df["close"].shift(20) - 1',
    description='20日收益率',
)
_register(
    'return_1d',
    brain_expr='returns',
    pandas_expr='df["close"] / df["close"].shift(1) - 1',
    description='1日收益率',
)
_register(
    'return_3d',
    brain_expr='ts_delta(close, 3) / ts_delay(close, 3)',
    pandas_expr='df["close"] / df["close"].shift(3) - 1',
    description='3日收益率',
)
_register(
    'avg_pct_change_5d',
    brain_expr='ts_mean(returns, 5)',
    pandas_expr='(df["close"] / df["close"].shift(1) - 1).rolling(5).mean()',
    description='5日平均涨跌幅',
)
_register(
    'max_pct_change_5d',
    brain_expr='ts_max(returns, 5)',
    pandas_expr='(df["close"] / df["close"].shift(1) - 1).rolling(5).max()',
    description='5日最大涨幅',
)
_register(
    'min_pct_change_5d',
    brain_expr='ts_min(returns, 5)',
    pandas_expr='(df["close"] / df["close"].shift(1) - 1).rolling(5).min()',
    description='5日最大跌幅',
)

# ---------- 波动率 ----------
_register(
    'volatility_10d',
    brain_expr='ts_std_dev(log(close / ts_delay(close, 1)), 10) * 15.875',
    pandas_expr='np.log(df["close"] / df["close"].shift(1)).rolling(10).std() * np.sqrt(252)',
    description='10日年化波动率 (sqrt(252)≈15.875)',
)
_register(
    'volatility_20d',
    brain_expr='ts_std_dev(log(close / ts_delay(close, 1)), 20) * 15.875',
    pandas_expr='np.log(df["close"] / df["close"].shift(1)).rolling(20).std() * np.sqrt(252)',
    description='20日年化波动率',
)

# ---------- 量价 ----------
_register(
    'volume_ratio',
    brain_expr='volume / ts_mean(volume, 20)',
    pandas_expr='df["volume"] / df["volume"].rolling(20).mean()',
    description='量比 (当日成交量 / 20日均量)',
)
_register(
    'volume_trend',
    brain_expr='ts_mean(volume, 5) / ts_mean(volume, 20) - 1',
    pandas_expr='df["volume"].rolling(5).mean() / df["volume"].rolling(20).mean() - 1',
    description='量能趋势 (5日均量/20日均量 - 1)',
)

# ---------- 价格位置 ----------
_register(
    'price_position_20d',
    brain_expr='(close - ts_min(close, 20)) / (ts_max(close, 20) - ts_min(close, 20) + 0.00000001)',
    pandas_expr='(df["close"] - df["close"].rolling(20).min()) / (df["close"].rolling(20).max() - df["close"].rolling(20).min() + 0.00000001)',
    description='20日价格位置 (0=最低, 1=最高)',
)

# ---------- 均线 ----------
_register(
    'ma5_ratio',
    brain_expr='close / ts_mean(close, 5) - 1',
    pandas_expr='df["close"] / df["close"].rolling(5).mean() - 1',
    description='价格/MA5偏离度',
)
_register(
    'ma10_ratio',
    brain_expr='close / ts_mean(close, 10) - 1',
    pandas_expr='df["close"] / df["close"].rolling(10).mean() - 1',
    description='价格/MA10偏离度',
)
_register(
    'ma20_ratio',
    brain_expr='close / ts_mean(close, 20) - 1',
    pandas_expr='df["close"] / df["close"].rolling(20).mean() - 1',
    description='价格/MA20偏离度',
)
_register(
    'ma_cross',
    brain_expr=(
        'if_else(ts_mean(close,5) > ts_mean(close,10) and ts_mean(close,10) > ts_mean(close,20), 1, '
        'if_else(ts_mean(close,5) < ts_mean(close,10) and ts_mean(close,10) < ts_mean(close,20), -1, 0))'
    ),
    pandas_expr=textwrap.dedent("""\
        ma5 = df["close"].rolling(5).mean()
        ma10 = df["close"].rolling(10).mean()
        ma20 = df["close"].rolling(20).mean()
        np.where(ma5 > ma10 > ma20, 1, np.where(ma5 < ma10 < ma20, -1, 0))"""),
    description='均线排列 (+1多头, -1空头, 0混合)',
)

# ---------- RSI ----------
_register(
    'rsi_14',
    brain_expr=(
        '100 - 100 / (1 + ts_mean(max(ts_delta(close,1), 0), 14) '
        '/ (ts_mean(max(-ts_delta(close,1), 0), 14) + 0.00000001))'
    ),
    pandas_expr=textwrap.dedent("""\
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta).clip(lower=0).rolling(14).mean()
        100 - 100 / (1 + gain / (loss + 0.00000001))"""),
    description='14日RSI相对强弱指标',
)

# ---------- 估值 (行业内排名) ----------
_register(
    'pe_industry_rank',
    brain_expr='group_rank(pe, subindustry)',
    pandas_expr='df.groupby("industry")["pe_ttm"].rank(pct=True)',
    category='valuation',
    description='行业内PE百分位排名',
)
_register(
    'pb_industry_rank',
    brain_expr='group_rank(pb, subindustry)',
    pandas_expr='df.groupby("industry")["pb"].rank(pct=True)',
    category='valuation',
    description='行业内PB百分位排名',
)
_register(
    'ps_industry_rank',
    brain_expr='group_rank(ps, subindustry)',
    pandas_expr='df.groupby("industry")["ps_ttm"].rank(pct=True)',
    category='valuation',
    description='行业内PS百分位排名',
)

# ---------- 估值原始值 (from daily_basic) ----------
_register(
    'pe_ttm',
    brain_expr='pe',
    pandas_expr='df["pe_ttm"]',
    category='valuation',
    description='市盈率TTM',
)
_register(
    'pb',
    brain_expr='pb',
    pandas_expr='df["pb"]',
    category='valuation',
    description='市净率',
)
_register(
    'ps_ttm',
    brain_expr='ps',
    pandas_expr='df["ps_ttm"]',
    category='valuation',
    description='市销率TTM',
)
_register(
    'turnover_rate',
    brain_expr='turnover',
    pandas_expr='df["turnover_rate"]',
    category='valuation',
    description='换手率',
)
_register(
    'log_market_cap',
    brain_expr='log(cap)',
    pandas_expr='np.log(df["total_mv"])',
    category='valuation',
    description='对数市值',
)

# ---------- 市场宏观特征 ----------
_register(
    'market_return_5d',
    brain_expr='ts_delta(close, 5) / ts_delay(close, 5)',
    pandas_expr='idx["close"] / idx["close"].shift(5) - 1',
    category='macro',
    description='沪深300 5日收益',
    brain_settings={'note': '需要在BRAIN中用指数数据单独计算'},
)
_register(
    'market_return_10d',
    brain_expr='ts_delta(close, 10) / ts_delay(close, 10)',
    pandas_expr='idx["close"] / idx["close"].shift(10) - 1',
    category='macro',
    description='沪深300 10日收益',
)
_register(
    'market_return_20d',
    brain_expr='ts_delta(close, 20) / ts_delay(close, 20)',
    pandas_expr='idx["close"] / idx["close"].shift(20) - 1',
    category='macro',
    description='沪深300 20日收益',
)
_register(
    'market_volatility_20d',
    brain_expr='ts_std_dev(log(close / ts_delay(close,1)), 20) * 15.875',
    pandas_expr='np.log(idx["close"] / idx["close"].shift(1)).rolling(20).std() * np.sqrt(252)',
    category='macro',
    description='沪深300 20日年化波动率',
)
_register(
    'market_volatility_10d',
    brain_expr='ts_std_dev(log(close / ts_delay(close,1)), 10) * 15.875',
    pandas_expr='np.log(idx["close"] / idx["close"].shift(1)).rolling(10).std() * np.sqrt(252)',
    category='macro',
    description='沪深300 10日年化波动率',
)

# ---------- V4.7.1 增强特征 ----------
_register(
    'amihud_illiquidity',
    brain_expr='ts_mean(abs(returns) / (volume * close / 1e8 + 0.00000001), 20)',
    pandas_expr='(df["close"].pct_change().abs() / (df["volume"] * df["close"] / 1e8 + 0.00000001)).rolling(20).mean()',
    description='Amihud非流动性 (20日均值)',
)
_register(
    'volume_price_corr_10d',
    brain_expr='ts_corr(volume, close, 10)',
    pandas_expr='df["volume"].rolling(10).corr(df["close"])',
    description='10日量价相关性',
)
_register(
    'max_drawdown_20d',
    brain_expr='ts_min(close / ts_max(close, 20) - 1, 20)',
    pandas_expr='(df["close"] / df["close"].rolling(20).max() - 1).rolling(20).min()',
    description='20日最大回撤',
)
_register(
    'updown_volume_asymmetry',
    brain_expr=(
        '(ts_sum(if_else(returns > 0, volume, 0), 20) - '
        'ts_sum(if_else(returns < 0, volume, 0), 20)) / '
        '(ts_sum(volume, 20) + 0.00000001)'
    ),
    pandas_expr=textwrap.dedent("""\
        ret = df["close"].pct_change()
        up_vol = (df["volume"] * (ret > 0)).rolling(20).sum()
        dn_vol = (df["volume"] * (ret < 0)).rolling(20).sum()
        (up_vol - dn_vol) / (df["volume"].rolling(20).sum() + 0.00000001)"""),
    description='20日上涨/下跌成交量不对称性',
)
_register(
    'idio_volatility_20d',
    brain_expr='ts_std_dev(vector_neut(returns, close), 20)',
    pandas_expr=textwrap.dedent("""\
        # 简化: 用残差波动率近似 (回归市场后)
        from sklearn.linear_model import LinearRegression
        # 实际需要市场收益做回归, 此处用总波动率近似"""),
    description='20日特质波动率 (市场中性残差)',
)

# ---------- 行业特征 (BRAIN 中需要 group 算子) ----------
_register(
    'industry_relative_strength',
    brain_expr='returns - group_mean(returns, subindustry)',
    pandas_expr='df["return_5d"] - df.groupby("industry")["return_5d"].transform("mean")',
    category='industry',
    description='个股相对行业超额收益',
)
_register(
    'industry_return_5d',
    brain_expr='group_mean(ts_delta(close,5)/ts_delay(close,5), subindustry)',
    pandas_expr='df.groupby("industry")["return_5d"].transform("mean")',
    category='industry',
    description='行业5日平均收益',
)
_register(
    'industry_return_20d',
    brain_expr='group_mean(ts_delta(close,20)/ts_delay(close,20), subindustry)',
    pandas_expr='df.groupby("industry")["return_20d"].transform("mean")',
    category='industry',
    description='行业20日平均收益',
)


class AlphaTranslator:
    """Alpha 双向翻译器"""

    def __init__(self):
        self.registry = FEATURE_TO_BRAIN

    # ----------------------------------------------------------
    # 方向1: 我们的特征 → BRAIN 表达式
    # ----------------------------------------------------------

    def to_brain_expr(self, feature_name: str) -> Optional[str]:
        """将一个特征名翻译为 BRAIN FASTEXPR"""
        alpha = self.registry.get(feature_name)
        return alpha.brain_expr if alpha else None

    def to_brain_alpha(self, feature_name: str,
                       region: str = 'CHINA',
                       universe: str = 'TOP3000',
                       decay: int = 5,
                       neutralization: str = 'subindustry',
                       truncation: float = 0.08,
                       delay: int = 1) -> Optional[Dict]:
        """
        生成可直接提交 BRAIN 的 alpha 配置

        Returns:
            {
                'expression': 'rank(-ts_delta(close, 5) / ts_delay(close, 5))',
                'settings': { region, universe, decay, ... }
            }
        """
        alpha = self.registry.get(feature_name)
        if alpha is None:
            return None

        # BRAIN alpha 通常需要取 rank 或 -rank 来生成信号
        # 正向因子(越大越好): 直接 rank
        # 反向因子(越小越好): -rank
        expr = alpha.brain_expr

        return {
            'expression': expr,
            'settings': {
                'instrumentType': 'EQUITY',
                'region': region,
                'universe': universe,
                'delay': delay,
                'decay': decay,
                'neutralization': neutralization,
                'truncation': truncation,
                'nanHandling': 'ON',
                'pasteurization': 'ON',
                'unitHandling': 'VERIFY',
                'language': 'FASTEXPR',
                **alpha.brain_settings,
            }
        }

    def batch_to_brain(self, feature_names: List[str] = None,
                       categories: List[str] = None,
                       **brain_kwargs) -> List[Dict]:
        """
        批量导出为 BRAIN alpha 配置

        Args:
            feature_names: 指定特征列表, None=全部
            categories: 过滤类别 ('stock', 'valuation', 'macro', 'industry')
        """
        results = []
        for name, alpha in self.registry.items():
            if feature_names and name not in feature_names:
                continue
            if categories and alpha.category not in categories:
                continue
            config = self.to_brain_alpha(name, **brain_kwargs)
            if config:
                config['name'] = name
                config['description'] = alpha.description
                config['category'] = alpha.category
                results.append(config)
        return results

    def export_brain_alphas_json(self, output_path: str,
                                 categories: List[str] = None,
                                 **brain_kwargs) -> int:
        """导出为 JSON 文件, 可直接用于 BRAIN API 提交"""
        alphas = self.batch_to_brain(categories=categories, **brain_kwargs)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(alphas, f, ensure_ascii=False, indent=2)
        return len(alphas)

    # ----------------------------------------------------------
    # 方向2: BRAIN 表达式 → pandas 代码
    # ----------------------------------------------------------

    # BRAIN 算子 → pandas/numpy 映射
    BRAIN_OP_MAP = {
        'ts_mean':          'rolling({d}).mean()',
        'ts_sum':           'rolling({d}).sum()',
        'ts_std_dev':       'rolling({d}).std()',
        'ts_stddev':        'rolling({d}).std()',
        'ts_min':           'rolling({d}).min()',
        'ts_max':           'rolling({d}).max()',
        'ts_rank':          'rolling({d}).rank(pct=True)',
        'ts_arg_max':       'rolling({d}).apply(np.argmax)',
        'ts_arg_min':       'rolling({d}).apply(np.argmin)',
        'ts_delta':         'diff({d})',
        'ts_delay':         'shift({d})',
        'ts_corr':          'rolling({d}).corr({y})',
        'ts_decay_linear':  'rolling({d}).apply(lambda x: np.dot(x, np.arange(1,{d}+1)) / np.sum(np.arange(1,{d}+1)))',
        'ts_zscore':        'pipe(lambda s: (s - s.rolling({d}).mean()) / (s.rolling({d}).std() + 0.00000001))',
        'rank':             'rank(pct=True)',
        'zscore':           'pipe(lambda s: (s - s.mean()) / (s.std() + 0.00000001))',
        'scale':            'pipe(lambda s: (s - s.min()) / (s.max() - s.min() + 0.00000001))',
        'log':              'pipe(np.log)',
        'abs':              'abs()',
        'sign':             'pipe(np.sign)',
        'max':              'clip(lower={y})',
    }

    # BRAIN 数据字段 → 我们的列名
    BRAIN_FIELD_MAP = {
        'close':    'close',
        'open':     'open',
        'high':     'high',
        'low':      'low',
        'volume':   'volume',
        'returns':  'price_change_pct',
        'cap':      'total_mv',
        'pe':       'pe_ttm',
        'pb':       'pb',
        'ps':       'ps_ttm',
        'turnover': 'turnover_rate',
    }

    def parse_brain_expr(self, expr: str) -> str:
        """
        将 BRAIN FASTEXPR 表达式解析为 pandas 可执行代码

        示例:
            'ts_mean(close, 20)' → 'df["close"].rolling(20).mean()'
            'rank(volume / ts_mean(volume, 20))'
                → '(df["volume"] / df["volume"].rolling(20).mean()).rank(pct=True)'

        注意: 这是一个启发式解析器, 覆盖常见模式但不保证100%正确
        """
        result = self._translate_recursive(expr.strip())
        return result

    def _translate_recursive(self, expr: str) -> str:
        """递归翻译 BRAIN 表达式"""
        expr = expr.strip()

        # 基本情况: 纯数字
        if re.match(r'^-?\d+\.?\d*$', expr):
            return expr

        # 基本情况: BRAIN 数据字段
        if expr in self.BRAIN_FIELD_MAP:
            return f'df["{self.BRAIN_FIELD_MAP[expr]}"]'

        # 先检查二元运算 (优先级低于函数调用)
        # 按优先级从低到高: and/or → 比较 → +/- → */
        for ops in [
            [' and ', ' or '],
            [' > ', ' < ', ' >= ', ' <= ', ' == ', ' != '],
            [' + ', ' - '],
            [' * ', ' / '],
        ]:
            for op in ops:
                parts = self._split_binary(expr, op)
                if parts:
                    left = self._translate_recursive(parts[0])
                    right = self._translate_recursive(parts[1])
                    py_op = op
                    if op == ' and ':
                        py_op = ' & '
                    elif op == ' or ':
                        py_op = ' | '
                    return f'({left}{py_op}{right})'

        # 一元负号
        if expr.startswith('-') and not expr[1:].strip().startswith('('):
            inner = self._translate_recursive(expr[1:].strip())
            return f'-({inner})'

        # 括号
        if expr.startswith('(') and expr.endswith(')'):
            # 确认这对括号是匹配的整体 (不是 (a)+(b) 的情况)
            depth = 0
            is_matching = True
            for i, ch in enumerate(expr):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                if depth == 0 and i < len(expr) - 1:
                    is_matching = False
                    break
            if is_matching:
                return f'({self._translate_recursive(expr[1:-1])})'

        # 函数调用: func(arg1, arg2, ...)
        match = re.match(r'^(\w+)\((.+)\)$', expr)
        if match:
            func_name = match.group(1)
            args_str = match.group(2)
            args = self._split_args(args_str)

            # 特殊处理 if_else (wrap in pd.Series for chaining)
            if func_name == 'if_else' and len(args) == 3:
                cond = self._translate_recursive(args[0])
                true_val = self._translate_recursive(args[1])
                false_val = self._translate_recursive(args[2])
                return f'pd.Series(np.where({cond}, {true_val}, {false_val}), index=df.index)'

            # 特殊处理 log
            if func_name == 'log' and len(args) == 1:
                inner = self._translate_recursive(args[0])
                return f'np.log({inner})'

            # 特殊处理 abs
            if func_name in ('abs', 'sign') and len(args) == 1:
                inner = self._translate_recursive(args[0])
                return f'np.{func_name}({inner})'

            # 特殊处理 max/min (二元)
            if func_name in ('max', 'min') and len(args) == 2:
                a = self._translate_recursive(args[0])
                b = self._translate_recursive(args[1])
                return f'np.maximum({a}, {b})' if func_name == 'max' else f'np.minimum({a}, {b})'

            # 时序函数 ts_xxx(x, d)
            if func_name.startswith('ts_') and len(args) >= 2:
                series = self._translate_recursive(args[0])
                d = args[1].strip()

                if func_name == 'ts_delta':
                    return f'{series}.diff({d})'
                elif func_name == 'ts_delay':
                    return f'{series}.shift({d})'
                elif func_name == 'ts_corr' and len(args) >= 3:
                    y = self._translate_recursive(args[1])
                    d = args[2].strip()
                    return f'{series}.rolling({d}).corr({y})'
                elif func_name in ('ts_mean', 'ts_sum', 'ts_std_dev', 'ts_stddev',
                                   'ts_min', 'ts_max'):
                    op_map = {
                        'ts_mean': 'mean', 'ts_sum': 'sum',
                        'ts_std_dev': 'std', 'ts_stddev': 'std',
                        'ts_min': 'min', 'ts_max': 'max',
                    }
                    return f'{series}.rolling({d}).{op_map[func_name]}()'
                elif func_name == 'ts_rank':
                    return f'{series}.rolling({d}).rank(pct=True)'
                elif func_name == 'ts_arg_max':
                    return f'{series}.rolling({d}).apply(np.argmax)'
                elif func_name == 'ts_arg_min':
                    return f'{series}.rolling({d}).apply(np.argmin)'
                elif func_name == 'ts_decay_linear':
                    return (f'{series}.rolling({d}).apply('
                            f'lambda x: np.dot(x, np.arange(1,{d}+1)) / np.sum(np.arange(1,{d}+1)))')
                elif func_name == 'ts_zscore':
                    return (f'({series} - {series}.rolling({d}).mean()) '
                            f'/ ({series}.rolling({d}).std() + 0.00000001)')

            # 截面函数 rank(x), zscore(x)
            if func_name == 'rank' and len(args) == 1:
                inner = self._translate_recursive(args[0])
                return f'({inner}).rank(pct=True)'
            if func_name == 'zscore' and len(args) == 1:
                inner = self._translate_recursive(args[0])
                return f'(lambda s: (s - s.mean()) / (s.std() + 0.00000001))({inner})'

            # group 函数
            if func_name in ('group_rank', 'group_mean') and len(args) >= 2:
                val = self._translate_recursive(args[0])
                group = args[1].strip()
                if func_name == 'group_rank':
                    return f'{val}.groupby(df["industry"]).rank(pct=True)'
                else:
                    return f'{val}.groupby(df["industry"]).transform("mean")'

            # vector_neut (因子中性化)
            if func_name == 'vector_neut' and len(args) >= 2:
                a = self._translate_recursive(args[0])
                b = self._translate_recursive(args[1])
                return f'# vector_neut: 对 {a} 关于 {b} 做回归取残差'

        # 未识别 → 原样返回
        return expr

    def _split_args(self, args_str: str) -> List[str]:
        """分割函数参数, 注意括号嵌套"""
        args = []
        depth = 0
        current = []
        for ch in args_str:
            if ch == '(':
                depth += 1
                current.append(ch)
            elif ch == ')':
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0:
                args.append(''.join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            args.append(''.join(current).strip())
        return args

    def _split_binary(self, expr: str, op: str) -> Optional[Tuple[str, str]]:
        """在最外层分割二元运算 (考虑括号嵌套, 取最右匹配保证左结合)"""
        depth = 0
        last_pos = -1
        i = 0
        while i < len(expr):
            if expr[i] == '(':
                depth += 1
            elif expr[i] == ')':
                depth -= 1
            elif depth == 0 and expr[i:i + len(op)] == op:
                last_pos = i
            i += 1
        if last_pos >= 0:
            return (expr[:last_pos].strip(), expr[last_pos + len(op):].strip())
        return None

    # ----------------------------------------------------------
    # 组合 Alpha: 将多个特征组合成一个 BRAIN alpha
    # ----------------------------------------------------------

    def compose_brain_alpha(self, feature_weights: Dict[str, float],
                            use_rank: bool = True) -> str:
        """
        将多个特征按权重组合成一个 BRAIN alpha 表达式

        Args:
            feature_weights: {'return_5d': -0.3, 'volume_ratio': 0.2, ...}
                            负权重 = 反向因子 (值越小越好)
            use_rank: 是否对每个因子先取 rank 再加权

        Returns:
            BRAIN FASTEXPR 组合表达式
        """
        parts = []
        for name, weight in feature_weights.items():
            alpha = self.registry.get(name)
            if alpha is None:
                continue
            expr = alpha.brain_expr
            if use_rank:
                expr = f'rank({expr})'
            if weight != 1.0:
                expr = f'{weight} * {expr}'
            parts.append(expr)

        return ' + '.join(parts) if parts else ''

    # ----------------------------------------------------------
    # 信息查询
    # ----------------------------------------------------------

    def list_features(self, category: str = None) -> List[Dict]:
        """列出所有已注册的特征翻译"""
        results = []
        for name, alpha in self.registry.items():
            if category and alpha.category != category:
                continue
            results.append({
                'name': name,
                'category': alpha.category,
                'description': alpha.description,
                'brain_expr': alpha.brain_expr,
            })
        return results

    def summary(self) -> str:
        """输出翻译器的摘要统计"""
        cats = {}
        for alpha in self.registry.values():
            cats[alpha.category] = cats.get(alpha.category, 0) + 1
        lines = [f'Alpha翻译器: 共 {len(self.registry)} 个特征']
        for cat, count in sorted(cats.items()):
            lines.append(f'  {cat}: {count}个')
        return '\n'.join(lines)


# ============================================================
# CLI 入口
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Alpha 双向翻译器')
    parser.add_argument('--export', type=str, help='导出 BRAIN alpha 到 JSON 文件')
    parser.add_argument('--category', type=str, nargs='+',
                        choices=['stock', 'valuation', 'macro', 'industry'],
                        help='过滤类别')
    parser.add_argument('--parse', type=str, help='解析 BRAIN 表达式为 pandas 代码')
    parser.add_argument('--list', action='store_true', help='列出所有特征')
    parser.add_argument('--region', default='CHINA', help='BRAIN region')
    parser.add_argument('--universe', default='TOP3000', help='BRAIN universe')

    args = parser.parse_args()
    translator = AlphaTranslator()

    if args.list:
        print(translator.summary())
        print()
        for feat in translator.list_features(category=args.category[0] if args.category else None):
            print(f"  {feat['name']:30s} [{feat['category']:10s}] {feat['description']}")
            print(f"    BRAIN: {feat['brain_expr']}")
            print()

    elif args.export:
        n = translator.export_brain_alphas_json(
            args.export,
            categories=args.category,
            region=args.region,
            universe=args.universe,
        )
        print(f'导出 {n} 个 alpha 到 {args.export}')

    elif args.parse:
        pandas_code = translator.parse_brain_expr(args.parse)
        print(f'BRAIN:  {args.parse}')
        print(f'Pandas: {pandas_code}')

    else:
        print(translator.summary())
