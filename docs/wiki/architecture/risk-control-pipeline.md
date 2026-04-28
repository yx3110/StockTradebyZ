# 风控管线 (Risk Control Pipeline)

> 选股报告生成时的 4 层风控数据流, 自 2026-04-27 P0/P1/P2 系列改造后定型.

## 整体数据流 (ng1.0.6 v1 production)

```
[ML 全市场评分: 7000+ 票]
        ↓
[post_filters] ─── trust 🔴 drop / 🟡 penalize / industry cap
        ↓
[P2.8 booster*] ── strategy bonus + trust mult (--enable-booster, default OFF)
        ↓
[ng2.1 overlay] ── L1 score floor + L2 industry cap → top_n 截断
        ↓
[P0.1 sizing] ──── L3 vol target + L5 stop-loss → position_size 字段
        ↓
[JSON 报告输出]
```

每只 pick 在最终 JSON 中带的字段:

```json
{
  "stock_code": "002371",
  "rank_score": 0.0028,             // ML 预测的 10d 收益 (NG scale)
  "rank_score_boosted": 0.0028,     // booster 重排分 (--enable-booster on)
  "_booster_strategy_bonus": 0.0,   // 来自 8 策略 regime-conditional bonus
  "_booster_trust_mult": 1.0,       // signal_trust 乘子 (post_filters 已处理)
  "position_size": 0.06,            // L3 VT sizing 后的实际仓位 (6%)
  "stop_loss_pct": -0.04,           // L5 单票止损
  "trailing_stop_pct": -0.06,       // L5 追踪止损 (仅熊市)
  "regime": "bear",                 // bull/bear from V11 0AMV classifier
  "crisis_active": false            // L4 crisis hard-stop 状态
}
```

## 层间约定 (commit `b337934a` 起)

| 层 | 模块 | 输入 | 输出 | 副作用 |
|---|---|---|---|---|
| post_filters | `stock_selctor/post_filters.py` | rank_score / composite | 同列, 部分 muter | 删 🔴, 改 🟡 score |
| booster | `stock_selctor/post_rank_booster.py` | rank_score, strategies, trust_tag | + rank_score_boosted | 不破坏原 rank_score |
| ng21 overlay | `stock_selctor/ng21_risk_overlay.py` | rank_score (or composite) | top_n 截断 + meta | _ng21_pos_cap, _ng21_stop_loss_pct |
| P0.1 sizing | `stock_selctor/ng21_risk_overlay.compute_position_size` | RiskDecision + est_vol | position_size, stop_loss_pct | 把 advisory 转实际仓位 |

## Score-scale 自适应 (重要约定)

NG 模型 `rank_score` 是预测收益小数 ∈ [-0.05, +0.02]; 旧 V3 composite ∈ [0, 100].
两层都做 auto-detect:

- **overlay** (`apply_overlay_to_picks`): `max(scores) < 1.0` → percentile floor (底 10%); `≥ 1.0` → absolute floor=30
- **booster** (`apply_post_rank_booster`): `max(scores) < 1.0` → bonus_scale = pos_max/100; `≥ 1.0` → bonus_scale=1.0

历史教训详见 [known-pitfalls](../lessons/known-pitfalls.md#score-scale-量纲混淆).

## 监控

- **Forward IC dashboard** (`scripts/forward_test_dashboard.py`): 90d 滚动 IC, ALERT Δ<-0.02
- **A/B booster** (`scripts/booster_ab_compare.py`): trust filter +33bp/10d 历史验证
- **0-picks 防御**: overlay/booster auto-scale 后, 报告应至少 ≥ 1 票 (空报告 = production alarm)

## 控制点

```bash
# 默认开启 overlay + sizing (生产)
python3 tomorrow_stock_selector.py YYYY-MM-DD --scoring-version ng1.0.6

# 加 booster (灰度, 默认 off)
python3 tomorrow_stock_selector.py YYYY-MM-DD --scoring-version ng1.0.6 --enable-booster

# 关 trust filter (调试)
python3 tomorrow_stock_selector.py YYYY-MM-DD --scoring-version ng1.0.6 --no-trust-filter --industry-cap 0
```
