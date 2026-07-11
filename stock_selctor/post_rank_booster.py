"""P2.8: post-rank booster — 8-strategy hits & signal-trust weighting.

CRITICAL: This is a POST-rank booster, not a pre-filter. Per wiki/known-pitfalls:
using 8-strategy as ML pre-filter dropped v4.7.5 from A+ to B (alpha leaked).
Booster runs *after* ML produced the rank list and only nudges scores within
ML's already-selected universe.

Behavior:
  1. For each pick already on the ML list, look up which (if any) of the 8
     quantitative strategies hit it today, and which signal-trust tag the
     stock currently carries.
  2. Apply additive bonus by regime (bull/bear specialists differ) and a
     multiplicative trust weight (🔴 picks get scaled down by 40%).
  3. Re-sort by `rank_score_boosted`, keep the same top-N count as before.

Wiki references:
  - docs/wiki/evaluation/quant-strategies-2018-2026.md (regime affinity)
  - docs/wiki/architecture/signal-trust.md (tag definitions)
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from stock_selctor.post_filters import sign_aware_scale

# Bonus points added to rank_score when a pick is also flagged by a quant
# strategy whose long-horizon backtest favors the current regime.
# Source: 8-strategy long-horizon backtest 2018-2026 (alpha by regime).
STRATEGY_BONUS_BY_REGIME: Dict[str, Dict[str, float]] = {
    'bull': {
        '少负战法':       8.0,   # bull-Sharpe 1.75
        'SuperB1战法':    5.0,   # cross-regime stable
        '补票战法':       5.0,
        '暴力K战法':      3.0,   # weak in bull but still positive alpha
    },
    'bear': {
        '暴力K战法':      8.0,   # bear-Sharpe 2.05 (top)
        'SuperB1战法':    5.0,
        '补票战法':       5.0,
    },
}

# Strategies with no long-horizon alpha — never grant bonus.
DEAD_STRATEGIES = frozenset({'知行战法', 'TePu战法', '填坑战法', '上穿60放量战法'})

# Signal-trust multiplier (history of "predicted vs realized" credibility).
# Production tag format from signal_trust_scores is the emoji optionally
# followed by a Chinese descriptor (e.g. '🟢可信', '🔴高风险'). We match by
# leading emoji only via _trust_lookup below.
TRUST_MULT: Dict[str, float] = {
    '🟢': 1.00,   # high credibility
    '🟡': 0.85,   # mild penalty (some doubt)
    '🔴': 0.60,   # heavy penalty (historical false signals)
    '⚪': 1.00,   # no data → neutral
}


def _trust_lookup(tag: Optional[str]) -> float:
    """Find multiplier by leading emoji, robust to '🟡存疑' / '🟢可信' suffixes."""
    if not tag:
        return 1.0
    for emoji, mult in TRUST_MULT.items():
        if tag.startswith(emoji):
            return mult
    return 1.0


def _strategy_bonus(strategies: Iterable[str], regime: str) -> float:
    table = STRATEGY_BONUS_BY_REGIME.get(regime, {})
    return sum(
        table.get(s, 0.0) for s in strategies if s not in DEAD_STRATEGIES
    )


def apply_post_rank_booster(
    picks: List[Dict],
    regime: str,
    *,
    strategy_field: str = 'strategy_hits',
    trust_field: str = 'trust_tag',
    score_field: str = 'rank_score',
    top_n: Optional[int] = None,
    skip_when_score_zero: bool = True,
) -> List[Dict]:
    """Add `rank_score_boosted` and re-sort.

    Args:
      picks: list of dicts (already ranked by ML composite). Each pick may
             carry ``strategy_hits`` (iterable of zh names) and ``trust_tag``
             (one of 🟢🟡🔴⚪).
      regime: 'bull' or 'bear'. Wrong values fall through to no bonus.
      strategy_field / trust_field / score_field: column names if non-default.
      top_n: trim to top N after re-sort (default: keep all).
      skip_when_score_zero: if True (default), picks with score == 0 (e.g.
        strategy candidates the ML scoring step did not score) are NOT
        promoted by strategy bonus — they keep score = 0 and fall to the bottom.
        This prevents the regression where strategy hits without ML scores
        outrank legitimately scored picks.

    Bonus magnitude is auto-scaled when score field is in NG predicted-return
    scale (max < 1.0): 8 pts becomes 8 × max(score) / 100, so the bonus is
    proportional to typical score magnitude rather than overwhelming it.

    Returns:
      A new list (does not mutate the input items beyond adding boost fields).
    """
    raw_scores = [float(p.get(score_field, 0.0) or 0.0) for p in picks]
    pos_max = max((s for s in raw_scores if s > 0), default=0.0)
    # If scores live in [0, 100] (legacy V3 composite), keep bonus pts as-is.
    # If in NG predicted-return scale (max < 1), scale bonus to 1% of typical
    # positive score per point so 8 pts ≈ 8% lift relative to top-tier alpha.
    if 0 < pos_max < 1.0:
        bonus_scale = pos_max / 100.0
    else:
        bonus_scale = 1.0

    out = []
    for s in picks:
        bonus_raw = _strategy_bonus(s.get(strategy_field) or (), regime)
        mult = _trust_lookup(s.get(trust_field))
        base = float(s.get(score_field, 0.0) or 0.0)
        # Skip bonus when ML did not score this pick (avoids strategy-only
        # candidates outranking real ML picks).
        bonus = 0.0 if (skip_when_score_zero and base == 0.0) else bonus_raw * bonus_scale
        combined = base + bonus
        # NG rank_score 可为负: trust mult(<1) 的惩罚按符号选方向, 保证永远降低排名
        boosted = sign_aware_scale(combined, mult)
        s2 = dict(s)
        s2['rank_score_boosted'] = round(boosted, 6)
        s2['_booster_strategy_bonus'] = round(bonus, 6)
        s2['_booster_strategy_bonus_raw'] = round(bonus_raw, 6)
        s2['_booster_bonus_scale'] = round(bonus_scale, 6)
        s2['_booster_trust_mult'] = mult
        out.append(s2)
    out.sort(key=lambda x: -float(x.get('rank_score_boosted', 0.0)))
    if top_n is not None and top_n > 0:
        out = out[:top_n]
    return out
