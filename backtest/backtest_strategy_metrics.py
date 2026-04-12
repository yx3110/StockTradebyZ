#!/usr/bin/env python3
"""
策略独立回测指标分析
====================
单独评估8个量化选股策略（不使用ML打分），计算每个策略的
IC、ICIR、Sharpe、Hit Rate等专业量化指标，以及3d/5d/10d前瞻回报。

核心架构：复用 batch_generate_reports.py 的"加载一次、预计算一次、按日循环"模式。

Usage:
    python3 backtest/backtest_strategy_metrics.py \\
        --start-date 2025-01-01 --end-date 2026-02-13 \\
        --holding-periods 3 5 10 --sample-every 1
"""

import sys
import os
import argparse
import logging
import time
import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pointbiserialr

# ── 路径设置 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_adapter.stock_data_loader import StockDataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  8个策略定义（与 tomorrow_stock_selector.py:677-773 完全一致）
# ═══════════════════════════════════════════════════════════════

STRATEGIES = {
    "少负战法": {
        "class": "BBIKDJSelector",
        "params": {
            "j_threshold": -5,
            "bbi_min_window": 20,
            "max_window": 60,
            "price_range_pct": 0.4,
            "bbi_q_threshold": 0.10,
            "j_q_threshold": 0.05,
        },
    },
    "SuperB1战法": {
        "class": "SuperB1Selector",
        "params": {
            "lookback_n": 15,
            "close_vol_pct": 0.02,
            "price_drop_pct": 0.02,
            "j_threshold": 10,
            "j_q_threshold": 0.10,
            "B1_params": {
                "j_threshold": 10,
                "bbi_min_window": 20,
                "max_window": 60,
                "price_range_pct": 2.0,
                "bbi_q_threshold": 0.3,
                "j_q_threshold": 0.10,
            },
        },
    },
    "补票战法": {
        "class": "BBIShortLongSelector",
        "params": {
            "n_short": 3,
            "n_long": 21,
            "m": 3,
            "bbi_min_window": 2,
            "max_window": 60,
            "bbi_q_threshold": 0.2,
        },
    },
    "TePu战法": {
        "class": "BreakoutVolumeKDJSelector",
        "params": {
            "j_threshold": 1,
            "j_q_threshold": 0.10,
            "up_threshold": 3.0,
            "volume_threshold": 0.6667,
            "offset": 15,
            "max_window": 60,
            "price_range_pct": 1,
        },
    },
    "填坑战法": {
        "class": "PeakKDJSelector",
        "params": {
            "j_threshold": 10,
            "max_window": 100,
            "fluc_threshold": 0.03,
            "j_q_threshold": 0.10,
            "gap_threshold": 0.2,
        },
    },
    "知行战法": {
        "class": "ZhiXingSelector",
        "params": {
            "j_threshold": 5.0,
            "min_change_pct": -1.0,
            "max_change_pct": 1.0,
            "max_amplitude_pct": 4.0,
            "close_threshold_pct": 100.0,
            "max_window": 120,
        },
    },
    "上穿60放量战法": {
        "class": "MA60CrossVolumeWaveSelector",
        "params": {
            "lookback_n": 20,
            "vol_multiple": 2.2,
            "j_threshold": 5,
            "j_q_threshold": 0.05,
            "ma60_slope_days": 5,
            "max_window": 120,
        },
    },
    "暴力K战法": {
        "class": "BigBullishVolumeSelector",
        "params": {
            "up_pct_threshold": 0.04,
            "upper_wick_pct_max": 0.5,
            "vol_lookback_n": 20,
            "vol_multiple": 1.5,
            "require_bullish_close": True,
            "ignore_zero_volume": True,
            "close_lt_zxdq_mult": 1.0,
        },
    },
}

STRATEGY_NAMES = list(STRATEGIES.keys())


# ═══════════════════════════════════════════════════════════════
#  导入选股器
# ═══════════════════════════════════════════════════════════════

def load_selector_module():
    """动态导入 Selector.py（与 tomorrow_stock_selector.py 同方式）"""
    selector_path = PROJECT_ROOT / "stock_selctor" / "Selector.py"
    spec = importlib.util.spec_from_file_location("Selector", selector_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ═══════════════════════════════════════════════════════════════
#  前瞻收益计算
# ═══════════════════════════════════════════════════════════════

def build_forward_return_table(
    data: Dict[str, pd.DataFrame],
    holding_periods: List[int],
) -> pd.DataFrame:
    """
    构建 (code, date) → forward_return 查找表。

    收益定义：buy_price = open[T+1], sell_price = close[T+N]
    return = (sell_price - buy_price) / buy_price

    Returns:
        DataFrame with columns: code, date, ret_3d, ret_5d, ret_10d, ...
    """
    max_period = max(holding_periods)
    records = []

    for code, df in data.items():
        if len(df) < 30:
            continue
        dates = df["date"].values
        opens = df["open"].values
        closes = df["close"].values

        for i in range(len(df) - 1):
            buy_price = opens[i + 1]  # T+1 open
            if buy_price <= 0 or np.isnan(buy_price):
                continue

            row = {"code": code, "date": dates[i]}
            for hp in holding_periods:
                sell_idx = i + 1 + hp  # T+1+hp
                if sell_idx < len(df):
                    sell_price = closes[sell_idx]
                    if sell_price > 0 and not np.isnan(sell_price):
                        row[f"ret_{hp}d"] = (sell_price - buy_price) / buy_price
                    else:
                        row[f"ret_{hp}d"] = np.nan
                else:
                    row[f"ret_{hp}d"] = np.nan
            records.append(row)

    if not records:
        cols = ["code", "date"] + [f"ret_{hp}d" for hp in holding_periods]
        return pd.DataFrame(columns=cols)

    ret_df = pd.DataFrame(records)
    ret_df["date"] = pd.to_datetime(ret_df["date"])
    return ret_df


# ═══════════════════════════════════════════════════════════════
#  日级别选股执行
# ═══════════════════════════════════════════════════════════════

def run_strategies_one_day(
    selectors: Dict[str, object],
    data: Dict[str, pd.DataFrame],
    target_date: pd.Timestamp,
) -> Dict[str, List[str]]:
    """
    在 target_date 上执行 8 个策略，返回 {strategy_name: [codes]}。

    预截断数据一次，避免 8 个策略各自重复截断 5000+ 只股票。
    """
    # 一次性截断（而不是让每个 selector.select 各做一遍）
    truncated = {}
    for code, df in data.items():
        hist = df[df["date"] <= target_date]
        if len(hist) >= 20:
            truncated[code] = hist

    if not truncated:
        return {name: [] for name in selectors}

    # 并行执行 8 个策略
    results = {}

    def _run_one(name, selector):
        return name, selector.select(target_date, truncated)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(_run_one, n, s): n for n, s in selectors.items()
        }
        for fut in as_completed(futures):
            try:
                name, picks = fut.result()
                results[name] = picks
            except Exception as e:
                results[futures[fut]] = []
                logger.warning(f"{futures[fut]} 异常: {e}")
    return results


# ═══════════════════════════════════════════════════════════════
#  指标计算
# ═══════════════════════════════════════════════════════════════

def compute_strategy_metrics(
    signals_df: pd.DataFrame,
    ret_df: pd.DataFrame,
    holding_periods: List[int],
    strategy_name: str,
) -> dict:
    """
    计算单个策略的完整量化指标。

    Args:
        signals_df: DataFrame(date, code, selected=1)
        ret_df: 前瞻收益表(code, date, ret_3d, ...)
        holding_periods: [3, 5, 10]
        strategy_name: 策略名

    Returns:
        {hp: {metric: value}} 嵌套字典
    """
    if signals_df.empty:
        return {hp: _empty_metrics() for hp in holding_periods}

    # 合并信号与收益
    merged = signals_df.merge(ret_df, on=["code", "date"], how="left")

    metrics = {}
    for hp in holding_periods:
        ret_col = f"ret_{hp}d"
        if ret_col not in merged.columns:
            metrics[hp] = _empty_metrics()
            continue

        valid = merged[merged[ret_col].notna()].copy()
        if valid.empty:
            metrics[hp] = _empty_metrics()
            continue

        returns = valid[ret_col].values
        n_signals = len(returns)

        # ── 收益指标 ──
        avg_ret = np.mean(returns)
        med_ret = np.median(returns)

        # ── 胜率 ──
        n_win = np.sum(returns > 0)
        hit_rate = n_win / n_signals if n_signals > 0 else 0

        # ── 盈亏比 ──
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = np.mean(np.abs(losses)) if len(losses) > 0 else 0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else np.inf

        # Profit Factor
        total_win = np.sum(wins) if len(wins) > 0 else 0
        total_loss = np.sum(np.abs(losses)) if len(losses) > 0 else 0
        profit_factor = total_win / total_loss if total_loss > 0 else np.inf

        # ── 风险调整收益 ──
        ann_factor = 252 / hp
        std_ret = np.std(returns, ddof=1) if n_signals > 1 else 0
        sharpe = (avg_ret * ann_factor) / (std_ret * np.sqrt(ann_factor)) if std_ret > 0 else 0

        downside = returns[returns < 0]
        downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 0
        sortino = (avg_ret * ann_factor) / (downside_std * np.sqrt(ann_factor)) if downside_std > 0 else 0

        # 最大回撤（逐信号累计净值）
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (running_max - cum_returns) / running_max
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

        # Calmar
        ann_ret = avg_ret * ann_factor
        calmar = ann_ret / max_dd if max_dd > 0 else 0

        # ── Point-Biserial IC（二值信号 vs 收益）──
        # 需要全市场数据做对比，这里用选中=1的收益
        # 由于是纯选中信号，IC在此为 selected=1 子集无法计算
        # 后续通过 compute_cross_sectional_ic 补充

        # ── 逐日胜率统计 ──
        daily_returns = valid.groupby("date")[ret_col].apply(
            lambda x: (x > 0).mean()
        )
        monthly_win_rate = np.nan
        if len(daily_returns) > 0:
            valid_dates = valid.copy()
            valid_dates["month"] = valid_dates["date"].dt.to_period("M")
            monthly_rets = valid_dates.groupby("month")[ret_col].mean()
            monthly_win_rate = (monthly_rets > 0).mean() if len(monthly_rets) > 0 else np.nan

        # ── 月度收益一致性 ──
        if len(daily_returns) > 0:
            valid_dates2 = valid.copy()
            valid_dates2["month"] = valid_dates2["date"].dt.to_period("M")
            monthly_means = valid_dates2.groupby("month")[ret_col].mean()
            monthly_consistency = monthly_means.std() if len(monthly_means) > 1 else np.nan
        else:
            monthly_consistency = np.nan

        # ── 日均选股数 ──
        daily_count = merged.groupby("date").size()
        avg_daily_picks = daily_count.mean()

        metrics[hp] = {
            "n_signals": n_signals,
            "avg_return": avg_ret,
            "med_return": med_ret,
            "hit_rate": hit_rate,
            "win_loss_ratio": win_loss_ratio,
            "profit_factor": profit_factor,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_dd,
            "monthly_win_rate": monthly_win_rate,
            "monthly_consistency": monthly_consistency,
            "avg_daily_picks": avg_daily_picks,
        }

    return metrics


def compute_cross_sectional_ic(
    all_signals: pd.DataFrame,
    ret_df: pd.DataFrame,
    holding_periods: List[int],
    strategy_name: str,
) -> dict:
    """
    计算截面IC：每个交易日，用 selected (0/1) 与该日全市场收益做 point-biserial 相关。

    Args:
        all_signals: DataFrame(date, code, selected) — 全市场 0/1 标记
        ret_df: 前瞻收益表
        holding_periods: [3, 5, 10]

    Returns:
        {hp: {ic_mean, ic_std, icir, ic_positive_pct, n_ic_days}}
    """
    merged = all_signals.merge(ret_df, on=["code", "date"], how="left")
    ic_metrics = {}

    for hp in holding_periods:
        ret_col = f"ret_{hp}d"
        if ret_col not in merged.columns:
            ic_metrics[hp] = {"ic_mean": 0, "ic_std": 0, "icir": 0,
                              "ic_positive_pct": 0, "n_ic_days": 0}
            continue

        ic_records = []
        for dt, group in merged.groupby("date"):
            valid = group[group[ret_col].notna()]
            if len(valid) < 20:  # 需要足够样本
                continue
            if valid["selected"].nunique() < 2:  # 需要有0和1
                continue
            try:
                corr, pval = pointbiserialr(valid["selected"].values, valid[ret_col].values)
                if not np.isnan(corr):
                    ic_records.append({"date": dt, "ic": corr, "p_val": pval})
            except Exception:
                continue

        if ic_records:
            ic_df = pd.DataFrame(ic_records)
            ic_mean = ic_df["ic"].mean()
            ic_std = ic_df["ic"].std()
            icir = ic_mean / ic_std if ic_std > 0 else 0
            ic_pos = (ic_df["ic"] > 0).mean() * 100
            n_days = len(ic_df)
        else:
            ic_mean, ic_std, icir, ic_pos, n_days = 0, 0, 0, 0, 0

        ic_metrics[hp] = {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "icir": icir,
            "ic_positive_pct": ic_pos,
            "n_ic_days": n_days,
        }

    return ic_metrics


def compute_count_factor_ic(
    count_signals: pd.DataFrame,
    ret_df: pd.DataFrame,
    holding_periods: List[int],
) -> dict:
    """
    多策略计数因子分析：策略选中数量(0-8) vs 收益的Spearman Rank IC。

    Args:
        count_signals: DataFrame(date, code, count) — 被多少个策略选中
        ret_df: 前瞻收益表
    """
    merged = count_signals.merge(ret_df, on=["code", "date"], how="left")
    result = {}

    for hp in holding_periods:
        ret_col = f"ret_{hp}d"
        if ret_col not in merged.columns:
            result[hp] = {"ic_mean": 0, "icir": 0, "ic_positive_pct": 0,
                          "n_ic_days": 0, "layer_returns": {}}
            continue

        ic_records = []
        for dt, group in merged.groupby("date"):
            valid = group[group[ret_col].notna()]
            if len(valid) < 20:
                continue
            if valid["count"].nunique() < 2:
                continue
            try:
                corr, pval = spearmanr(valid["count"].values, valid[ret_col].values)
                if not np.isnan(corr):
                    ic_records.append({"date": dt, "ic": corr})
            except Exception:
                continue

        if ic_records:
            ic_df = pd.DataFrame(ic_records)
            ic_mean = ic_df["ic"].mean()
            ic_std = ic_df["ic"].std()
            icir = ic_mean / ic_std if ic_std > 0 else 0
            ic_pos = (ic_df["ic"] > 0).mean() * 100
            n_days = len(ic_df)
        else:
            ic_mean, ic_std, icir, ic_pos, n_days = 0, 0, 0, 0, 0

        # 分层收益：按 count 分组
        valid_merged = merged[merged[ret_col].notna()]
        layer_returns = {}
        for cnt, grp in valid_merged.groupby("count"):
            layer_returns[int(cnt)] = {
                "avg_return": grp[ret_col].mean(),
                "n_signals": len(grp),
                "hit_rate": (grp[ret_col] > 0).mean(),
            }

        result[hp] = {
            "ic_mean": ic_mean,
            "icir": icir,
            "ic_positive_pct": ic_pos,
            "n_ic_days": n_days,
            "layer_returns": layer_returns,
        }

    return result


def _empty_metrics():
    return {
        "n_signals": 0, "avg_return": 0, "med_return": 0,
        "hit_rate": 0, "win_loss_ratio": 0, "profit_factor": 0,
        "sharpe": 0, "sortino": 0, "calmar": 0, "max_drawdown": 0,
        "monthly_win_rate": np.nan, "monthly_consistency": np.nan,
        "avg_daily_picks": 0,
    }


# ═══════════════════════════════════════════════════════════════
#  策略重叠度（Jaccard）
# ═══════════════════════════════════════════════════════════════

def compute_overlap_matrix(daily_picks: Dict[str, Dict[str, set]]) -> pd.DataFrame:
    """
    计算策略间 Jaccard 相似度矩阵。

    Args:
        daily_picks: {date: {strategy: set(codes)}}
    """
    names = STRATEGY_NAMES
    n = len(names)
    matrix = np.zeros((n, n))

    for date_picks in daily_picks.values():
        for i, si in enumerate(names):
            for j, sj in enumerate(names):
                if j <= i:
                    continue
                a = date_picks.get(si, set())
                b = date_picks.get(sj, set())
                if len(a) == 0 and len(b) == 0:
                    continue
                jaccard = len(a & b) / len(a | b) if len(a | b) > 0 else 0
                matrix[i, j] += jaccard
                matrix[j, i] += jaccard

    n_days = len(daily_picks)
    if n_days > 0:
        matrix /= n_days

    return pd.DataFrame(matrix, index=names, columns=names)


# ═══════════════════════════════════════════════════════════════
#  报告生成
# ═══════════════════════════════════════════════════════════════

def generate_report(
    strategy_metrics: Dict[str, dict],
    strategy_ic: Dict[str, dict],
    count_factor: dict,
    overlap_matrix: pd.DataFrame,
    holding_periods: List[int],
    start_date: str,
    end_date: str,
    n_trading_days: int,
    output_dir: str,
) -> str:
    """生成 Markdown 报告"""

    today = datetime.now().strftime("%Y%m%d")
    report_path = Path(output_dir) / f"strategy_metrics_{today}.md"

    lines = []
    lines.append(f"# 策略独立回测指标分析报告")
    lines.append(f"")
    lines.append(f"- **回测区间**: {start_date} ~ {end_date}")
    lines.append(f"- **交易日数**: {n_trading_days}")
    lines.append(f"- **持仓周期**: {', '.join(str(hp)+'d' for hp in holding_periods)}")
    lines.append(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"")

    # ── 1. 策略综合概览表 ──
    lines.append("## 1. 策略综合概览")
    lines.append("")

    for hp in holding_periods:
        lines.append(f"### {hp}日持仓")
        lines.append("")
        lines.append("| 策略 | 信号数 | 日均 | 平均收益 | 中位收益 | 胜率 | 盈亏比 | Sharpe | Sortino | 最大回撤 | IC均值 | ICIR | IC>0% |")
        lines.append("|------|--------|------|----------|----------|------|--------|--------|---------|----------|--------|------|-------|")

        for name in STRATEGY_NAMES:
            m = strategy_metrics[name].get(hp, _empty_metrics())
            ic = strategy_ic[name].get(hp, {})
            lines.append(
                f"| {name} "
                f"| {m['n_signals']} "
                f"| {m['avg_daily_picks']:.1f} "
                f"| {m['avg_return']*100:+.3f}% "
                f"| {m['med_return']*100:+.3f}% "
                f"| {m['hit_rate']*100:.1f}% "
                f"| {m['win_loss_ratio']:.2f} "
                f"| {m['sharpe']:.3f} "
                f"| {m['sortino']:.3f} "
                f"| {m['max_drawdown']*100:.1f}% "
                f"| {ic.get('ic_mean', 0):.4f} "
                f"| {ic.get('icir', 0):.3f} "
                f"| {ic.get('ic_positive_pct', 0):.1f}% |"
            )
        lines.append("")

    # ── 2. 各策略详细指标 ──
    lines.append("## 2. 各策略详细指标")
    lines.append("")

    for name in STRATEGY_NAMES:
        lines.append(f"### {name}")
        lines.append("")
        for hp in holding_periods:
            m = strategy_metrics[name].get(hp, _empty_metrics())
            ic = strategy_ic[name].get(hp, {})
            lines.append(f"**{hp}日持仓** ({m['n_signals']}个信号)")
            lines.append("")
            lines.append(f"- 平均收益: {m['avg_return']*100:+.3f}%, 中位收益: {m['med_return']*100:+.3f}%")
            lines.append(f"- 胜率: {m['hit_rate']*100:.1f}%, 盈亏比: {m['win_loss_ratio']:.2f}, Profit Factor: {m['profit_factor']:.2f}")
            lines.append(f"- Sharpe: {m['sharpe']:.3f}, Sortino: {m['sortino']:.3f}, Calmar: {m['calmar']:.3f}")
            lines.append(f"- 最大回撤: {m['max_drawdown']*100:.1f}%")
            mw = m.get('monthly_win_rate', np.nan)
            mc = m.get('monthly_consistency', np.nan)
            lines.append(f"- 月度胜率: {mw*100:.1f}%" if not np.isnan(mw) else "- 月度胜率: N/A")
            lines.append(f"- 月度收益一致性(std): {mc*100:.3f}%" if not np.isnan(mc) else "- 月度收益一致性: N/A")
            lines.append(f"- IC均值: {ic.get('ic_mean',0):.4f}, ICIR: {ic.get('icir',0):.3f}, IC>0%: {ic.get('ic_positive_pct',0):.1f}% ({ic.get('n_ic_days',0)}天)")
            lines.append("")
        lines.append("---")
        lines.append("")

    # ── 3. 策略排名（按 ICIR）──
    lines.append("## 3. 策略排名（按ICIR）")
    lines.append("")

    for hp in holding_periods:
        lines.append(f"### {hp}日持仓 ICIR排名")
        lines.append("")
        ranked = []
        for name in STRATEGY_NAMES:
            ic = strategy_ic[name].get(hp, {})
            ranked.append((name, ic.get("icir", 0), ic.get("ic_mean", 0)))
        ranked.sort(key=lambda x: x[1], reverse=True)

        lines.append("| 排名 | 策略 | ICIR | IC均值 |")
        lines.append("|------|------|------|--------|")
        for i, (name, icir, ic_mean) in enumerate(ranked, 1):
            lines.append(f"| {i} | {name} | {icir:.4f} | {ic_mean:.4f} |")
        lines.append("")

    # ── 4. 多策略计数因子 ──
    lines.append("## 4. 多策略计数因子IC分析")
    lines.append("")
    lines.append("被多少个策略同时选中（0-8）与未来收益的 Spearman Rank IC。")
    lines.append("")

    for hp in holding_periods:
        cf = count_factor.get(hp, {})
        lines.append(f"### {hp}日持仓")
        lines.append(f"- IC均值: {cf.get('ic_mean',0):.4f}")
        lines.append(f"- ICIR: {cf.get('icir',0):.3f}")
        lines.append(f"- IC>0%: {cf.get('ic_positive_pct',0):.1f}% ({cf.get('n_ic_days',0)}天)")
        lines.append("")

        # 分层收益
        layer = cf.get("layer_returns", {})
        if layer:
            lines.append("**分层收益:**")
            lines.append("")
            lines.append("| 策略数 | 平均收益 | 信号数 | 胜率 |")
            lines.append("|--------|----------|--------|------|")
            for cnt in sorted(layer.keys()):
                lr = layer[cnt]
                lines.append(
                    f"| {cnt} "
                    f"| {lr['avg_return']*100:+.3f}% "
                    f"| {lr['n_signals']} "
                    f"| {lr['hit_rate']*100:.1f}% |"
                )
            lines.append("")

    # ── 5. 策略重叠度矩阵 ──
    lines.append("## 5. 策略重叠度矩阵（Jaccard相似度）")
    lines.append("")
    lines.append("日均 Jaccard 相似度（选股交集/并集）:")
    lines.append("")

    # 短名
    short = {n: n[:4] for n in STRATEGY_NAMES}
    header = "| | " + " | ".join(short[n] for n in STRATEGY_NAMES) + " |"
    lines.append(header)
    sep = "|---|" + "|".join(["---"] * len(STRATEGY_NAMES)) + "|"
    lines.append(sep)

    for i, ni in enumerate(STRATEGY_NAMES):
        row = f"| {short[ni]} |"
        for j, nj in enumerate(STRATEGY_NAMES):
            if i == j:
                row += " - |"
            else:
                val = overlap_matrix.iloc[i, j]
                row += f" {val:.3f} |"
        lines.append(row)
    lines.append("")

    report_text = "\n".join(lines)

    # 写入文件
    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return str(report_path)


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="策略独立回测指标分析")
    parser.add_argument("--start-date", default="2025-01-01", help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", default="2026-02-13", help="结束日期 YYYY-MM-DD")
    parser.add_argument("--holding-periods", nargs="+", type=int, default=[3, 5, 10],
                        help="持仓周期（天数）")
    parser.add_argument("--sample-every", type=int, default=1,
                        help="每N个交易日采样一次（加速）")
    parser.add_argument("--output-dir", default="reports/backtest",
                        help="报告输出目录")
    args = parser.parse_args()

    t_start = time.time()

    # ── 1. 加载选股器模块 ──
    logger.info("加载选股器模块...")
    sel_mod = load_selector_module()
    selector_classes = {
        "BBIKDJSelector": sel_mod.BBIKDJSelector,
        "SuperB1Selector": sel_mod.SuperB1Selector,
        "BBIShortLongSelector": sel_mod.BBIShortLongSelector,
        "BreakoutVolumeKDJSelector": sel_mod.BreakoutVolumeKDJSelector,
        "PeakKDJSelector": sel_mod.PeakKDJSelector,
        "ZhiXingSelector": sel_mod.ZhiXingSelector,
        "MA60CrossVolumeWaveSelector": sel_mod.MA60CrossVolumeWaveSelector,
        "BigBullishVolumeSelector": sel_mod.BigBullishVolumeSelector,
    }
    precompute_fn = sel_mod.precompute_indicators

    # 预实例化 8 个选股器（避免每天重复创建）
    selectors = {}
    for name, cfg in STRATEGIES.items():
        cls = selector_classes[cfg["class"]]
        selectors[name] = cls(**cfg["params"])
    logger.info(f"成功加载 {len(selectors)} 个选股器")

    # ── 2. 加载数据（一次SQL）──
    logger.info(f"加载数据: {args.start_date} ~ {args.end_date} (lookback=200天)...")
    loader = StockDataLoader()
    data = loader.load_all_stock_data_wide(
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_days=200,
        security_types=["A股"],
    )
    logger.info(f"加载完成: {len(data)} 只股票")

    # ── 3. 预计算技术指标（一次性）──
    logger.info("预计算技术指标（BBI/KDJ/DIF/ZX/MA60）...")
    t_pre = time.time()
    precompute_fn(data, pd.Timestamp(args.end_date))
    logger.info(f"指标预计算完成, 耗时 {time.time()-t_pre:.1f}秒")

    # ── 4. 构建前瞻收益表 ──
    t1 = time.time()
    logger.info("构建前瞻收益表...")
    ret_df = build_forward_return_table(data, args.holding_periods)
    logger.info(f"前瞻收益表: {len(ret_df)} 行, 耗时 {time.time()-t1:.1f}秒")

    # ── 5. 获取交易日列表 + 构建 date→codes 索引 ──
    start_ts = pd.Timestamp(args.start_date)
    end_ts = pd.Timestamp(args.end_date)

    # 一次性构建 date→available_codes 映射（用于截面IC）
    logger.info("构建交易日索引...")
    date_to_codes: Dict[pd.Timestamp, set] = {}
    for code, df in data.items():
        for dt in df["date"][(df["date"] >= start_ts) & (df["date"] <= end_ts)]:
            if dt not in date_to_codes:
                date_to_codes[dt] = set()
            date_to_codes[dt].add(code)

    trading_days = sorted(date_to_codes.keys())

    # 采样
    if args.sample_every > 1:
        trading_days = trading_days[:: args.sample_every]

    n_trading_days = len(trading_days)
    logger.info(f"交易日: {n_trading_days} 天 (sample_every={args.sample_every})")

    # 过滤 ret_df 到只含采样交易日（大幅加速后续 merge）
    if not ret_df.empty:
        trading_days_set = set(trading_days)
        ret_df = ret_df[ret_df["date"].isin(trading_days_set)].reset_index(drop=True)
        logger.info(f"过滤后前瞻收益表: {len(ret_df)} 行")

    # ── 6. 按日循环执行选股 ──
    t2 = time.time()
    strategy_signals = {name: [] for name in STRATEGY_NAMES}
    daily_picks_for_overlap = {}

    for i, tday in enumerate(trading_days):
        t_day_start = time.time()
        if i % 5 == 0:
            print(f"进度: {i+1}/{n_trading_days} ({tday.strftime('%Y-%m-%d')})", flush=True)

        results = run_strategies_one_day(selectors, data, tday)

        date_picks = {}
        for name, picks in results.items():
            for code in picks:
                strategy_signals[name].append({"date": tday, "code": code})
            date_picks[name] = set(picks)

        daily_picks_for_overlap[tday] = date_picks

        if i < 3:
            print(f"  Day {i+1} took {time.time()-t_day_start:.1f}s", flush=True)

    elapsed = time.time()-t2
    print(f"选股完成，耗时 {elapsed:.1f}秒 ({elapsed/n_trading_days:.1f}秒/天)", flush=True)
    for name in STRATEGY_NAMES:
        logger.info(f"  {name}: {len(strategy_signals[name])} 个信号")

    # ── 7. 计算策略指标 ──
    logger.info("计算策略指标...")
    all_strategy_metrics = {}
    all_strategy_ic = {}

    for name in STRATEGY_NAMES:
        sig_df = pd.DataFrame(strategy_signals[name])
        if not sig_df.empty:
            sig_df["date"] = pd.to_datetime(sig_df["date"])
            sig_df["selected"] = 1
        else:
            sig_df = pd.DataFrame(columns=["date", "code", "selected"])

        # 基本指标
        all_strategy_metrics[name] = compute_strategy_metrics(
            sig_df, ret_df, args.holding_periods, name
        )

        # 截面IC（使用预建的 date→codes 索引，避免逐行 O(n) 查找）
        if not sig_df.empty:
            signal_dates = sig_df["date"].unique()
            # 向量化构造: 每个日期 × 该日可用股票
            cross_frames = []
            for dt in signal_dates:
                available = date_to_codes.get(dt, set())
                if len(available) < 20:
                    continue
                selected_codes = set(sig_df[sig_df["date"] == dt]["code"])
                codes_list = list(available)
                frame = pd.DataFrame({
                    "date": dt,
                    "code": codes_list,
                    "selected": [1 if c in selected_codes else 0 for c in codes_list],
                })
                cross_frames.append(frame)

            if cross_frames:
                cross_df = pd.concat(cross_frames, ignore_index=True)
                all_strategy_ic[name] = compute_cross_sectional_ic(
                    cross_df, ret_df, args.holding_periods, name
                )
            else:
                all_strategy_ic[name] = {
                    hp: {"ic_mean": 0, "ic_std": 0, "icir": 0,
                         "ic_positive_pct": 0, "n_ic_days": 0}
                    for hp in args.holding_periods
                }
        else:
            all_strategy_ic[name] = {
                hp: {"ic_mean": 0, "ic_std": 0, "icir": 0,
                     "ic_positive_pct": 0, "n_ic_days": 0}
                for hp in args.holding_periods
            }
        logger.info(f"  {name} 完成")

    # ── 8. 多策略计数因子 ──
    logger.info("计算多策略计数因子...")
    # 向量化构造 count 信号
    count_frames = []
    for tday in trading_days:
        dp = daily_picks_for_overlap.get(tday, {})
        available = date_to_codes.get(tday, set())
        if not available:
            continue
        codes_list = list(available)
        counts = [
            sum(1 for sn in STRATEGY_NAMES if c in dp.get(sn, set()))
            for c in codes_list
        ]
        count_frames.append(pd.DataFrame({
            "date": tday, "code": codes_list, "count": counts,
        }))

    if count_frames:
        count_df = pd.concat(count_frames, ignore_index=True)
    else:
        count_df = pd.DataFrame(columns=["date", "code", "count"])

    count_factor = compute_count_factor_ic(count_df, ret_df, args.holding_periods)

    # ── 9. 策略重叠度 ──
    logger.info("计算策略重叠度...")
    overlap_matrix = compute_overlap_matrix(daily_picks_for_overlap)

    # ── 10. 打印概览 ──
    print("\n" + "=" * 80)
    print("  策略独立回测指标分析")
    print(f"  回测区间: {args.start_date} ~ {args.end_date} ({n_trading_days}天)")
    print("=" * 80)

    for hp in args.holding_periods:
        print(f"\n{'─'*60}")
        print(f"  {hp}日持仓")
        print(f"{'─'*60}")
        print(f"{'策略':>10s} | {'信号':>5s} | {'平均收益':>8s} | {'胜率':>6s} | {'盈亏比':>6s} | {'Sharpe':>7s} | {'IC均值':>7s} | {'ICIR':>6s}")
        print("-" * 75)
        for name in STRATEGY_NAMES:
            m = all_strategy_metrics[name].get(hp, _empty_metrics())
            ic = all_strategy_ic[name].get(hp, {})
            print(
                f"{name:>10s} | {m['n_signals']:>5d} "
                f"| {m['avg_return']*100:>+7.3f}% "
                f"| {m['hit_rate']*100:>5.1f}% "
                f"| {m['win_loss_ratio']:>6.2f} "
                f"| {m['sharpe']:>7.3f} "
                f"| {ic.get('ic_mean',0):>7.4f} "
                f"| {ic.get('icir',0):>6.3f}"
            )

    # 计数因子
    print(f"\n{'─'*60}")
    print("  多策略计数因子")
    print(f"{'─'*60}")
    for hp in args.holding_periods:
        cf = count_factor.get(hp, {})
        print(f"  {hp}日: IC={cf.get('ic_mean',0):.4f}, ICIR={cf.get('icir',0):.3f}, IC>0%={cf.get('ic_positive_pct',0):.1f}%")

    # ── 11. 生成报告 ──
    output_dir = str(PROJECT_ROOT / args.output_dir)
    report_path = generate_report(
        all_strategy_metrics, all_strategy_ic, count_factor,
        overlap_matrix, args.holding_periods,
        args.start_date, args.end_date, n_trading_days,
        output_dir,
    )

    elapsed = time.time() - t_start
    logger.info(f"完成! 耗时 {elapsed:.1f}秒")
    logger.info(f"报告已保存: {report_path}")

    # ── 12. 导出CSV (向量化merge, 避免O(n²)) ──
    csv_path = Path(output_dir) / f"strategy_signals_{datetime.now().strftime('%Y%m%d')}.csv"
    frames = []
    for name in STRATEGY_NAMES:
        if not strategy_signals[name]:
            continue
        df_sig = pd.DataFrame(strategy_signals[name])
        df_sig["strategy"] = name
        df_sig["date"] = pd.to_datetime(df_sig["date"])
        frames.append(df_sig)

    if frames:
        sig_csv = pd.concat(frames, ignore_index=True)
        sig_csv = sig_csv.merge(ret_df, on=["code", "date"], how="left")
        sig_csv = sig_csv[["strategy", "date", "code"] + [f"ret_{hp}d" for hp in args.holding_periods]]
        sig_csv.to_csv(csv_path, index=False)
        logger.info(f"信号CSV已保存: {csv_path} ({len(sig_csv)} rows)")

    print(f"\n报告: {report_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
