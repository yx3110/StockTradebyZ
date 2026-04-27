"""Scoring version router (P3.1 渐进式拆分).

把 tomorrow_stock_selector.main() 里的 ng106 / ng2.0a / ng2.1 regime 路由逻辑
抽出来作为独立可测模块.

输入: scoring_version (含 +overlay/+alt 后缀), target_date, db_path
输出: RouteResult dataclass — 含解析后的 scoring_version + 各种 mode flag.

设计原则:
- 不改变路由语义, 仅作物理拆分 (tests 保护)
- 每个 ng 模型路径一个函数 (route_ng106 / route_ng200a / route_ng21)
- 顶层 dispatch 函数 route_scoring_version() 自动选合适路径
"""
from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RouteResult:
    """Routing 结果, 跟原内联实现的局部变量 1:1 对应."""
    scoring_version: str            # 实际用的子模型版本 (如 ng1.0.1, ng1.0.4, ng2.1-bull)
    version_tag: str = ''           # 用户传入的原始 tag (如 ng1.0.6+overlay)
    bull_model: str = ''
    bear_model: str = ''
    # ng106 专属
    ng106_mode: bool = False
    ng106_overlay_mode: bool = False
    ng106_alt_mode: bool = False
    ng106_overlay_regime: str = 'bull'   # 'bull' | 'bear'
    # ng2.0a 专属
    ng200a_mode: bool = False
    ng200a_regime_table: str = 'market_regime_signals'
    # ng2.1 专属
    ng21_mode: bool = False
    ng21_regime: str = 'bull'
    ng21_regime_table: str = 'market_regime_signals'


# ────────────────────────────────────────────────────────────
# DB regime 读取
# ────────────────────────────────────────────────────────────

def _read_amv_regime(db_path: str, target_date: Optional[str]) -> tuple[Optional[int], str]:
    """读 0AMV regime (ng1.0.6 路径). 返回 (regime_value, label)."""
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        if target_date:
            row = conn.execute(
                'SELECT amv_regime FROM market_amv WHERE trade_date <= ? '
                'ORDER BY trade_date DESC LIMIT 1', (target_date,)
            ).fetchone()
            label = f"{target_date}"
        else:
            row = conn.execute(
                'SELECT amv_regime FROM market_amv ORDER BY trade_date DESC LIMIT 1'
            ).fetchone()
            label = "最新"
    finally:
        conn.close()
    return (row[0] if row else None), label


def _read_v2_regime(db_path: str, regime_table: str, target_date: Optional[str]) -> tuple[Optional[int], str]:
    """读 v2 multi-beta regime (ng2.0a / ng2.1 路径)."""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute('PRAGMA busy_timeout=30000')
    try:
        if target_date:
            row = conn.execute(
                f'SELECT regime_v2 FROM {regime_table} '
                f'WHERE regime_v2 IS NOT NULL AND trade_date <= ? '
                f'ORDER BY trade_date DESC LIMIT 1', (target_date,)
            ).fetchone()
            label = f"{target_date}"
        else:
            row = conn.execute(
                f'SELECT regime_v2 FROM {regime_table} '
                f'WHERE regime_v2 IS NOT NULL ORDER BY trade_date DESC LIMIT 1'
            ).fetchone()
            label = "最新"
    finally:
        conn.close()
    return (row[0] if row else None), label


# ────────────────────────────────────────────────────────────
# 三个独立路由函数
# ────────────────────────────────────────────────────────────

def route_ng106(scoring_version: str, target_date: Optional[str], db_path: str) -> RouteResult:
    """ng1.0.6 / ng1.0.62 路由: 0AMV → bull (ng1.0.1/0.7/1.7) / bear (ng1.0.4).

    后缀:
      +overlay: P1.1 L1-L5 风控
      +alt:    P4.1 bull 子模型 → ng1.7.0 (alt-data)
    """
    res = RouteResult(scoring_version='', version_tag=scoring_version)
    res.ng106_mode = True

    # 解析后缀
    base_version = scoring_version
    for suf in ("+overlay", "+alt"):
        if suf in base_version:
            base_version = base_version.replace(suf, "")
    # P0.1 (2026-04-27): default-on overlay for ng1.0.6 / ng1.0.62 base versions.
    # Old behavior required '+overlay' suffix; we keep the suffix as a no-op to
    # remain backwards compatible with existing scripts.
    res.ng106_overlay_mode = (
        "+overlay" in scoring_version
        or scoring_version in ("ng1.0.6", "ng1.0.62")
    )
    res.ng106_alt_mode = "+alt" in scoring_version
    if base_version not in ("ng1.0.6", "ng1.0.62"):
        base_version = "ng1.0.6"

    # bull 子模型选择
    if res.ng106_alt_mode:
        res.bull_model = "ng1.7.0"
    elif base_version == "ng1.0.62":
        res.bull_model = "ng1.0.7"
    else:
        res.bull_model = "ng1.0.1"
    res.bear_model = "ng1.0.4"

    try:
        regime, label = _read_amv_regime(db_path, target_date)
    except Exception as e:
        res.scoring_version = res.bull_model
        res.ng106_overlay_regime = "bull"
        print(f"⚠️ {scoring_version}: 读取 AMV regime 失败({e}), 默认使用 {res.bull_model}")
        return res

    if regime == 1:
        res.scoring_version = res.bull_model
        res.ng106_overlay_regime = "bull"
        overlay_tag = " + L1-L5 牛市风控" if res.ng106_overlay_mode else ""
        print(f"🐂 {scoring_version}: 0AMV判定 {label}【牛市】→ 使用 {res.bull_model}{overlay_tag}")
    else:
        res.scoring_version = res.bear_model
        res.ng106_overlay_regime = "bear"
        overlay_tag = " + L1-L5 熊市风控" if res.ng106_overlay_mode else ""
        print(f"🐻 {scoring_version}: 0AMV判定 {label}【熊市】→ 使用 {res.bear_model}{overlay_tag}")
    return res


def route_ng200a(scoring_version: str, target_date: Optional[str], db_path: str) -> RouteResult:
    """ng2.0a multi-beta vote regime → ng1.0.1 bull / ng1.0.4 bear (baseline calibration)."""
    res = RouteResult(scoring_version='', version_tag='ng2.0a')
    res.ng200a_mode = True
    res.bull_model = "ng1.0.1"
    res.bear_model = "ng1.0.4"
    res.ng200a_regime_table = "market_regime_signals"

    try:
        regime, label = _read_v2_regime(db_path, res.ng200a_regime_table, target_date)
    except Exception as e:
        res.scoring_version = res.bull_model
        print(f"⚠️ ng2.0a: 读取 v2 regime 失败({e}), 默认使用 {res.bull_model}")
        return res

    if regime == 1:
        res.scoring_version = res.bull_model
        print(f"🐂 ng2.0a: v2 regime判定 {label}【牛市】→ 使用 {res.bull_model}")
    else:
        res.scoring_version = res.bear_model
        print(f"🐻 ng2.0a: v2 regime判定 {label}【熊市】→ 使用 {res.bear_model}")
    return res


def route_ng21(scoring_version: str, target_date: Optional[str], db_path: str) -> RouteResult:
    """ng2.1: V11 router → ng2.1-bull/ng2.1-bear + L1-L5 overlay 强制开启."""
    res = RouteResult(scoring_version='', version_tag='ng2.1')
    res.ng21_mode = True
    res.bull_model = "ng2.1-bull"
    res.bear_model = "ng2.1-bear"
    res.ng21_regime_table = "market_regime_signals"

    try:
        regime, label = _read_v2_regime(db_path, res.ng21_regime_table, target_date)
    except Exception as e:
        res.scoring_version = res.bull_model
        res.ng21_regime = "bull"
        print(f"⚠️ ng2.1: 读取 V11 regime 失败({e}), 默认使用 {res.bull_model}")
        return res

    if regime == 1:
        res.scoring_version = res.bull_model
        res.ng21_regime = "bull"
        print(f"🐂 ng2.1: V11 regime判定 {label}【牛市】→ {res.bull_model} + L1-L5 牛市风控")
    else:
        res.scoring_version = res.bear_model
        res.ng21_regime = "bear"
        print(f"🐻 ng2.1: V11 regime判定 {label}【熊市】→ {res.bear_model} + L1-L5 熊市风控")
    return res


# ────────────────────────────────────────────────────────────
# 顶层 dispatch
# ────────────────────────────────────────────────────────────

def route_scoring_version(scoring_version: str, target_date: Optional[str],
                           db_path: str) -> RouteResult:
    """根据 scoring_version 自动选合适路由函数. 非 MOE 版本透传."""
    if scoring_version.startswith("ng1.0.6") or scoring_version.startswith("ng1.0.62"):
        return route_ng106(scoring_version, target_date, db_path)
    if scoring_version == "ng2.0a":
        return route_ng200a(scoring_version, target_date, db_path)
    if scoring_version == "ng2.1":
        return route_ng21(scoring_version, target_date, db_path)
    # 非 MOE: 透传
    return RouteResult(scoring_version=scoring_version, version_tag=scoring_version)
