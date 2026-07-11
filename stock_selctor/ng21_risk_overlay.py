"""ng2.1 risk overlay (L1-L5) — applied at selection time, NOT inside the model.

Design rationale (see docs/superpowers/plans/2026-04-26-ng21-bull-bear-specialist.md):
  - All regime info enters via training-data filtering and this overlay.
  - Nothing regime-aware lives inside the model (avoids ng1.5.0 β-explosion).
  - Bull → "let winners run": low turnover, high VT, no crisis stop.
  - Bear → "exit fast": high turnover, low VT, crisis hard-stop on RV spikes.

L1: score floor + ST/退市 exclusion (always on, regime-agnostic)
L2: regime-aware retention bonus, EMA smoothing, rebalance freq, industry cap
L3: vol-target overlay (pos sizing) + cash ceiling
L4: crisis hard-stop (bear-only): RV percentile + tape spike → halve top_n
L5: per-stock SL (bull -8% / bear -4%) + bear trailing -6% from high

L4/L5 emit ADVICE rather than mutating positions in this skill — the daily
selection skill should consume `risk_decisions` to produce position sizes
and per-stock stops in the trading plan.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _norm_date(d) -> str:
    """归一化日期为 'YYYY-MM-DD' 字符串.

    sqlite3 不支持直接绑定 pd.Timestamp (Error binding parameter: type
    'Timestamp' is not supported) — 2026-04-28~07-11 L4 熔断即因此静默失效.
    """
    if hasattr(d, 'strftime'):
        return d.strftime('%Y-%m-%d')
    return str(d)[:10]


# ---------------------------------------------------------------------------
# Default parameters (chosen pre Stage-4b sweep; sweep refines bear params)
# ---------------------------------------------------------------------------

BULL_PARAMS = {
    'retention_bonus': 0.20,
    'ema_alpha': 0.7,
    'rebalance_freq_days': 15,
    'industry_cap': 3,
    'vol_target_annual': 0.25,
    'cash_ceiling': 0.20,
    'stop_loss': -0.08,
    'trailing_stop': None,
}

BEAR_PARAMS = {
    'retention_bonus': 0.0,
    'ema_alpha': 0.5,
    'rebalance_freq_days': 5,
    'industry_cap': 2,
    'vol_target_annual': 0.15,
    'cash_ceiling': 0.50,
    'stop_loss': -0.04,
    'trailing_stop': -0.06,
    # L4 crisis trigger
    'crisis_rv_pct': 0.90,        # B2 60d RV percentile threshold
    'crisis_index_drop': -0.03,   # 大盘当日跌幅 < -3% 触发
    'crisis_top_n_factor': 0.5,   # top_n→top_n*0.5
    'crisis_pos_cap': 0.05,       # 单票仓位上限
    'crisis_cash_floor': 0.70,    # 现金下限
}

# L1 universal — score floor in 0-100 scale (legacy V3 composite). NG models
# emit rank_score as predicted return ∈ [-0.05, +0.02]; for those we percentile
# floor instead. _resolve_floor() picks per-call based on observed scale.
L1_SCORE_FLOOR = 30
L1_DEFAULT_TOP_N = 10
# When all scores are < 1, treat them as predicted-return scale and drop bottom
# N% (rather than absolute floor). Default 10% bottom-cut.
L1_PERCENTILE_FLOOR_PCT = 0.10


# ---------------------------------------------------------------------------
# Decision record (consumed by skill / report generator)
# ---------------------------------------------------------------------------

@dataclass
class RiskDecision:
    regime: str  # 'bull' | 'bear'
    top_n: int
    industry_cap: int
    vol_target_annual: float
    cash_ceiling: float
    cash_floor: float
    stop_loss: float
    trailing_stop: Optional[float]
    crisis_active: bool
    pos_cap_per_stock: float
    rebalance_freq_days: int
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            'regime': self.regime,
            'top_n': self.top_n,
            'industry_cap': self.industry_cap,
            'vol_target_annual': self.vol_target_annual,
            'cash_ceiling': self.cash_ceiling,
            'cash_floor': self.cash_floor,
            'stop_loss_pct': self.stop_loss,
            'trailing_stop_pct': self.trailing_stop,
            'crisis_active': self.crisis_active,
            'pos_cap_per_stock': self.pos_cap_per_stock,
            'rebalance_freq_days': self.rebalance_freq_days,
            'notes': self.notes,
        }


# ---------------------------------------------------------------------------
# L4 crisis detector
# ---------------------------------------------------------------------------

def _check_l4_crisis(
    db_path: str,
    target_date: str,
    regime_table: str = 'market_regime_signals',
) -> tuple[bool, list[str]]:
    """Return (crisis_active, reasons)."""
    notes = []
    target_date = _norm_date(target_date)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute('PRAGMA busy_timeout=30000')
    try:
        # B2 RV percentile from regime table
        row = conn.execute(
            f'SELECT b2_rv_percentile_252 FROM {regime_table} '
            f'WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT 1',
            (target_date,),
        ).fetchone()
        rv_pct = float(row[0]) if row and row[0] is not None else 0.0

        # 沪深300 当日 returns (proxy via daily_quotes if available)
        idx_row = conn.execute(
            "SELECT (close - prev_close) / prev_close FROM ("
            "  SELECT close, LAG(close) OVER (ORDER BY trade_date) AS prev_close, trade_date "
            "  FROM daily_quotes dq JOIN securities s ON s.id = dq.security_id "
            "  WHERE s.code = '000300.SH'"
            ") WHERE trade_date = ?",
            (target_date,),
        ).fetchone()
        idx_chg = float(idx_row[0]) if idx_row and idx_row[0] is not None else 0.0
    except sqlite3.Error as e:
        # 风控层失败必须高可见: L4 曾因 Timestamp 绑定错误在 warning 级静默失效 2.5 个月
        logger.error(f"L4 crisis check FAILED — 风控降级为 fail-open (DB error: {e})")
        return False, [f"⚠️ L4 check DEGRADED (fail-open): {e}"]
    finally:
        conn.close()

    rv_trigger = rv_pct >= BEAR_PARAMS['crisis_rv_pct']
    drop_trigger = idx_chg <= BEAR_PARAMS['crisis_index_drop']
    crisis = rv_trigger and drop_trigger

    notes.append(
        f"B2_RV_pct={rv_pct:.2f} ({'≥0.90 ✓' if rv_trigger else '<0.90 ✗'}); "
        f"hs300_chg={idx_chg:+.2%} ({'≤-3% ✓' if drop_trigger else '>-3% ✗'})"
    )
    if crisis:
        notes.append("L4 CRISIS ACTIVE: top_n→×0.5, pos_cap=5%, cash_floor=70%")
    return crisis, notes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_risk_decision(
    regime: str,
    target_date: str,
    db_path: str,
    base_top_n: int = L1_DEFAULT_TOP_N,
    regime_table: str = 'market_regime_signals',
) -> RiskDecision:
    """Compute L1-L5 risk decision for given regime + date.

    Args:
        regime: 'bull' or 'bear' (determined upstream by V11 router)
        target_date: trading date YYYY-MM-DD
        db_path: path to stock_data.db
        base_top_n: pre-overlay top_n (default 10)
        regime_table: which regime table to read (baseline / unanimous etc.)
    """
    if regime not in ('bull', 'bear'):
        raise ValueError(f"regime must be bull/bear, got {regime!r}")

    target_date = _norm_date(target_date)
    params = BULL_PARAMS if regime == 'bull' else BEAR_PARAMS
    notes = [
        f"regime={regime}; date={target_date}; base_top_n={base_top_n}",
        f"L1: score_floor={L1_SCORE_FLOOR}, ST/退市 excluded by pre-filter",
        (
            f"L2: retention=+{int(params['retention_bonus']*100)}%, ema_α={params['ema_alpha']}, "
            f"rebalance={params['rebalance_freq_days']}d, industry_cap={params['industry_cap']}"
        ),
        f"L3: VT={params['vol_target_annual']:.0%} annual, cash_ceiling={params['cash_ceiling']:.0%}",
        (
            f"L5: stop_loss={params['stop_loss']:.0%}"
            + (f", trailing={params['trailing_stop']:.0%}" if params['trailing_stop'] else "")
        ),
    ]

    crisis_active = False
    cash_floor = 0.0
    pos_cap_per_stock = 0.10  # default 10% / stock
    top_n = base_top_n

    if regime == 'bear':
        crisis_active, l4_notes = _check_l4_crisis(db_path, target_date, regime_table)
        notes.extend([f"L4: {n}" for n in l4_notes])
        if crisis_active:
            top_n = max(int(round(base_top_n * params['crisis_top_n_factor'])), 3)
            pos_cap_per_stock = params['crisis_pos_cap']
            cash_floor = params['crisis_cash_floor']
    else:
        notes.append("L4: bull regime — crisis stop disabled")

    return RiskDecision(
        regime=regime,
        top_n=top_n,
        industry_cap=params['industry_cap'],
        vol_target_annual=params['vol_target_annual'],
        cash_ceiling=params['cash_ceiling'],
        cash_floor=cash_floor,
        stop_loss=params['stop_loss'],
        trailing_stop=params['trailing_stop'],
        crisis_active=crisis_active,
        pos_cap_per_stock=pos_cap_per_stock,
        rebalance_freq_days=params['rebalance_freq_days'],
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Selector-side post-filter wrapper
# ---------------------------------------------------------------------------

def apply_overlay_to_picks(
    picks: List[Dict],
    decision: RiskDecision,
) -> tuple[List[Dict], List[Dict]]:
    """Apply L1 (score floor) + L2 (industry cap) cuts and tag survivors with risk metadata.

    L3/L4/L5 are advisory and attached as metadata; they do NOT mutate the
    picks here (downstream sizing logic owns that).

    Returns:
        (kept, dropped) — both list[dict]
    """
    kept: list[dict] = []
    dropped: list[dict] = []
    industry_count: dict[str, int] = {}

    # Resolve effective floor based on score scale to handle both legacy V3
    # 0-100 composite and NG predicted-return rank_score (typically in
    # [-0.05, +0.02]). If max score < 1, switch to percentile floor.
    raw_scores = [float(s.get('rank_score') or s.get('composite') or 0) for s in picks]
    if raw_scores and max(raw_scores) < 1.0:
        sorted_scores = sorted(raw_scores)
        cutoff_idx = int(len(sorted_scores) * L1_PERCENTILE_FLOOR_PCT)
        effective_floor = sorted_scores[cutoff_idx] if cutoff_idx < len(sorted_scores) else sorted_scores[-1]
        floor_label = f'L1 score<p{int(L1_PERCENTILE_FLOOR_PCT*100)} ({effective_floor:+.4f})'
    else:
        effective_floor = L1_SCORE_FLOOR
        floor_label = f'L1 score<{L1_SCORE_FLOOR}'

    for s in picks:
        # 涨停/停牌股 T+1 不可买入, 不许占用 top_n 席位
        if s.get('exec_warning'):
            d = dict(s); d['_drop_reason'] = f"L1 exec_warning({s['exec_warning']}) 不可T+1买入"
            dropped.append(d)
            continue

        score = float(s.get('rank_score') or s.get('composite') or 0)
        if score < effective_floor:
            d = dict(s); d['_drop_reason'] = floor_label
            dropped.append(d)
            continue

        ind = s.get('industry') or 'UNKNOWN'
        if industry_count.get(ind, 0) >= decision.industry_cap:
            d = dict(s); d['_drop_reason'] = f'L2 industry_cap={decision.industry_cap} ({ind})'
            dropped.append(d)
            continue
        industry_count[ind] = industry_count.get(ind, 0) + 1

        # Attach L3/L5 sizing/SL guidance
        s2 = dict(s)
        s2['_ng21_pos_cap'] = decision.pos_cap_per_stock
        s2['_ng21_stop_loss_pct'] = decision.stop_loss
        if decision.trailing_stop is not None:
            s2['_ng21_trailing_stop_pct'] = decision.trailing_stop
        s2['_ng21_regime'] = decision.regime
        kept.append(s2)

        if len(kept) >= decision.top_n:
            break

    return kept, dropped


# ---------------------------------------------------------------------------
# P0.1: L3 vol-target sizing + L5 SL persistence
# ---------------------------------------------------------------------------

def compute_position_size(
    picks: List[Dict],
    decision: RiskDecision,
    est_portfolio_vol: float = 0.20,
) -> List[Dict]:
    """L3 vol-target sizing.

    Sizing rule (equal-weight within survivors, monotone caps):
      gross_budget = min(1 - cash_floor, vol_target / max(est_vol, eps))
      raw_weight   = gross_budget / n_picks
      final_weight = min(raw_weight, pos_cap_per_stock)

    Each pick gets ``position_size`` (float in [0, pos_cap]) plus L5 stops
    (stop_loss_pct, trailing_stop_pct) and the regime tag.
    """
    if not picks:
        return list(picks)
    n = len(picks)
    cash_budget = max(0.0, 1.0 - float(decision.cash_floor))
    vt_scale = min(1.0, float(decision.vol_target_annual) / max(float(est_portfolio_vol), 1e-6))
    gross = min(cash_budget, vt_scale)
    raw_w = gross / n
    capped_w = min(raw_w, float(decision.pos_cap_per_stock))
    out: list[dict] = []
    for s in picks:
        s2 = dict(s)
        s2['position_size'] = round(capped_w, 6)
        s2['stop_loss_pct'] = float(decision.stop_loss)
        if decision.trailing_stop is not None:
            s2['trailing_stop_pct'] = float(decision.trailing_stop)
        s2['regime'] = decision.regime
        s2['crisis_active'] = bool(decision.crisis_active)
        out.append(s2)
    return out


def estimate_portfolio_vol(
    picks: List[Dict],
    db_path: str,
    target_date: str,
    lookback_days: int = 60,
) -> float:
    """Avg of constituent 60d realized vol (annualized). Conservative under equal weights."""

    def _fallback(reason: str) -> float:
        # est_vol fallback 意味着 L3 vol-target 退化为常数, 仓位与真实波动脱钩
        # (2026-04-28~07-11 曾因 'code'/'stock_code' 键错配恒走此分支) — 必须高可见
        logger.error(f"[P0.1] est_vol fallback→0.25, L3 vol-target 降级 ({reason})")
        return 0.25

    if not picks:
        return 0.25
    target_date = _norm_date(target_date)
    # selector 的 stock_info 用 'stock_code' 键; 兼容裸 'code'
    codes = [s.get('stock_code') or s.get('code') for s in picks]
    codes = [c for c in codes if c]
    if not codes:
        return _fallback('picks 中无 stock_code/code 键')
    placeholders = ','.join('?' * len(codes))
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        conn.execute('PRAGMA busy_timeout=30000')
        try:
            rows = conn.execute(
                f"""
                SELECT s.code, dq.close
                  FROM daily_quotes dq JOIN securities s ON s.id = dq.security_id
                 WHERE s.code IN ({placeholders})
                   AND dq.trade_date <= ?
                   AND dq.trade_date >= date(?, '-{int(lookback_days * 2)} days')
                 ORDER BY s.code, dq.trade_date
                """,
                list(codes) + [target_date, target_date],
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as e:
        return _fallback(f'DB error: {e}')
    if not rows:
        return _fallback(f'{len(codes)} 只票在 {target_date} 前无行情数据')
    import collections
    import math
    series: dict[str, list[float]] = collections.defaultdict(list)
    for code, close in rows:
        if close is not None:
            series[code].append(float(close))
    vols: list[float] = []
    for prices in series.values():
        if len(prices) < 20:
            continue
        log_rets = [math.log(prices[i] / prices[i - 1])
                    for i in range(1, len(prices))
                    if prices[i - 1] > 0 and prices[i] > 0]
        if len(log_rets) < 10:
            continue
        mean = sum(log_rets) / len(log_rets)
        var = sum((r - mean) ** 2 for r in log_rets) / max(len(log_rets) - 1, 1)
        vol_annual = (var ** 0.5) * (252 ** 0.5)
        if 0.05 < vol_annual < 2.0:
            vols.append(vol_annual)
    if not vols:
        return _fallback('所有成分股波动率计算无效 (数据不足或超界)')
    return float(sum(vols) / len(vols))


def estimate_portfolio_vol_forward(
    picks: List[Dict],
    db_path: str,
    target_date: str,
) -> Optional[float]:
    """前瞻组合波动率: P1.6 vol_10d 风险头对成分股的预测均值 (年化).

    2026-07-11 影子模式: 与 estimate_portfolio_vol (60d 后视统计) 并排产出,
    只记录不参与 sizing — VT 阈值 (0.25/0.15) 需先在该度量空间重 sweep
    才能切换消费。协议修复后 3 折 WF 均值 IC=+0.60 (risk_head pkl 内有 fold 明细)。
    任何失败返回 None (影子信号缺失不降级生产)。
    """
    try:
        import json as _json
        import joblib
        from pathlib import Path
        pkl = (Path(db_path).resolve().parent.parent / 'ml_models' / 'trained_models'
               / 'ng' / 'risk_head_vol_10d_seed42.pkl')
        if not pkl.exists():
            return None
        bundle = joblib.load(pkl)
        model, feat_cols = bundle['model'], bundle['feat_cols']
        codes = [(s.get('stock_code') or s.get('code') or '').split('.')[0] for s in picks]
        codes = [c for c in codes if c]
        if not codes:
            return None
        target_date = _norm_date(target_date)
        conn = sqlite3.connect(db_path, timeout=30)
        conn.execute('PRAGMA busy_timeout=30000')
        try:
            ph = ','.join('?' * len(codes))
            rows = conn.execute(
                f"SELECT code, features_json FROM ng101_feature_cache "
                f"WHERE trade_date = ? AND code IN ({ph})",
                [target_date] + codes,
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            return None
        feats = [{**_json.loads(fj)} for _, fj in rows]
        import numpy as _np
        X = _np.array([[float(f.get(c, 0.0) or 0.0) for c in feat_cols] for f in feats])
        preds = model.predict(X)
        preds = preds[(preds > 0.05) & (preds < 2.0)]
        if len(preds) == 0:
            return None
        return float(preds.mean())
    except Exception as e:  # 影子信号, 任何异常不干扰生产
        logger.warning(f"forward vol shadow 计算失败: {e}")
        return None
