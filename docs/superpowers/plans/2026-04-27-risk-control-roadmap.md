# 风控糅合进模型 — P0/P1/P2 紧凑执行计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 把当前活在 backtest layer / advisory metadata 里的风控逻辑迁到生产 selector + 模型训练目标，分三档落地。

**Architecture:**
- **P0**: 生产化已有风控 advisory（L3 vol-target → 真 position_size, L5 SL → JSON 字段），forward OOS 闭环监测。零模型变动。
- **P1**: 引入 risk-aware label (Calmar) + 独立 maxDD/vol 预测头，为 ng2.2 meta-learner 铺路。需要重训。
- **P2**: regime soft probability 替换 hard switch，8策略 + signal trust 进 ranking weight。

**Tech Stack:** Python 3, LightGBM/XGBoost/CatBoost, SQLite, pandas/numpy, scipy, joblib, pyarrow.

**铁律（来自 wiki/known-pitfalls.md + 三次失败实证）：**
- ❌ 不在特征里加 regime state (ng1.5.0 β=+1.42 死)
- ❌ 不做 conditional label by regime (ng1.0.7 Pre-2020 -36% 死)
- ❌ 8策略不当 ML pre-filter (v4.7.5 A+→B 死)
- ✅ 只走 label-side risk adjustment + auxiliary head 路径

---

## P0 — 生产化已有风控（0-2 周，确定性高）

### Task P0.1: ng1.0.6 v1 默认启用 L1-L5 overlay + JSON 写入 position_size/stop_loss_pct

**背景**：`ng21_risk_overlay.apply_overlay_to_picks` 已经把 `_ng21_pos_cap` / `_ng21_stop_loss_pct` / `_ng21_trailing_stop_pct` 当 advisory metadata 挂到每只票上 (`stock_selctor/ng21_risk_overlay.py:256-261`)，但 (a) 默认 ng1.0.6 v1 不触发 (`tomorrow_stock_selector.py:5868` 只在 `+overlay` 后缀时启用)；(b) 这些字段从未写进生产 JSON 报告。

**Files:**
- Modify: `stock_selctor/scoring_router.py:112` — 让 `ng1.0.6` (无 +overlay) 也启用 overlay
- Modify: `tomorrow_stock_selector.py` — 报告 JSON 写入逻辑里加 position_size + stop_loss_pct
- Modify: `stock_selctor/ng21_risk_overlay.py` — 加 `compute_position_size(picks, decision)` 把 vol-target 转成具体仓位
- Test: `stock_selctor/test/test_ng21_position_size.py`

- [ ] **Step 1: 写 compute_position_size 失败测试**

```python
# stock_selctor/test/test_ng21_position_size.py
from stock_selctor.ng21_risk_overlay import RiskDecision, compute_position_size

def test_bull_equal_weight_under_vt():
    """牛市 VT=25%, 假设组合实际 vol < target → 满仓 Top-10 等权 = 0.10/票."""
    picks = [{'code': f'00000{i}.SZ', '_ng21_pos_cap': 0.10} for i in range(10)]
    decision = RiskDecision(regime='bull', top_n=10, industry_cap=3,
                            vol_target_annual=0.25, cash_ceiling=0.20, cash_floor=0.0,
                            stop_loss=-0.08, trailing_stop=None, crisis_active=False,
                            pos_cap_per_stock=0.10, rebalance_freq_days=15)
    sized = compute_position_size(picks, decision, est_portfolio_vol=0.20)
    weights = [s['position_size'] for s in sized]
    assert all(abs(w - 0.10) < 1e-9 for w in weights)
    assert abs(sum(weights) - 1.0) < 1e-9

def test_bear_crisis_pos_cap():
    """熊 + crisis: 单票 pos_cap=5%, top_n=5 → 仓位 ≤ 25% 总, 现金底 70%."""
    picks = [{'code': f'00000{i}.SZ'} for i in range(5)]
    decision = RiskDecision(regime='bear', top_n=5, industry_cap=2,
                            vol_target_annual=0.15, cash_ceiling=0.50, cash_floor=0.70,
                            stop_loss=-0.04, trailing_stop=-0.06, crisis_active=True,
                            pos_cap_per_stock=0.05, rebalance_freq_days=5)
    sized = compute_position_size(picks, decision, est_portfolio_vol=0.30)
    weights = [s['position_size'] for s in sized]
    assert all(w <= 0.05 + 1e-9 for w in weights)
    assert sum(weights) <= 1.0 - decision.cash_floor + 1e-9

def test_vt_scale_down_high_vol():
    """估计组合 vol 30% > target 15% → 缩仓到 50%."""
    picks = [{'code': f'00000{i}.SZ'} for i in range(10)]
    decision = RiskDecision(regime='bear', top_n=10, industry_cap=2,
                            vol_target_annual=0.15, cash_ceiling=0.50, cash_floor=0.0,
                            stop_loss=-0.04, trailing_stop=-0.06, crisis_active=False,
                            pos_cap_per_stock=0.10, rebalance_freq_days=5)
    sized = compute_position_size(picks, decision, est_portfolio_vol=0.30)
    total = sum(s['position_size'] for s in sized)
    assert 0.45 < total < 0.55  # 0.15/0.30 = 0.5
```

- [ ] **Step 2: 跑失败**

```bash
cd /Users/yangxu/StockTradebyZ
python3 -m pytest stock_selctor/test/test_ng21_position_size.py -v
```
Expected: FAIL `ImportError: cannot import name 'compute_position_size'`

- [ ] **Step 3: 实现 compute_position_size**

加到 `stock_selctor/ng21_risk_overlay.py` 末尾：

```python
def compute_position_size(
    picks: List[Dict],
    decision: RiskDecision,
    est_portfolio_vol: float = 0.20,
) -> List[Dict]:
    """L3 vol-target sizing: scale = min(1, target_vol / est_vol).
    Equal-weight within survivors, capped by pos_cap_per_stock,
    floored by (1 - cash_floor) total budget.
    """
    if not picks:
        return picks
    n = len(picks)
    cash_budget = max(0.0, 1.0 - decision.cash_floor)
    vt_scale = min(1.0, decision.vol_target_annual / max(est_portfolio_vol, 1e-6))
    gross = min(cash_budget, vt_scale)
    raw_w = gross / n
    capped_w = min(raw_w, decision.pos_cap_per_stock)
    out = []
    for s in picks:
        s2 = dict(s)
        s2['position_size'] = round(capped_w, 6)
        s2['stop_loss_pct'] = decision.stop_loss
        if decision.trailing_stop is not None:
            s2['trailing_stop_pct'] = decision.trailing_stop
        s2['regime'] = decision.regime
        s2['crisis_active'] = decision.crisis_active
        out.append(s2)
    return out
```

- [ ] **Step 4: 跑测试**

```bash
python3 -m pytest stock_selctor/test/test_ng21_position_size.py -v
```
Expected: 3 PASS

- [ ] **Step 5: 估算 portfolio vol 的工具函数**

加到同文件：

```python
def estimate_portfolio_vol(
    picks: List[Dict],
    db_path: str,
    target_date: str,
    lookback_days: int = 60,
) -> float:
    """简化估计: avg(stock 60d realized vol). 不算协方差 (太贵), 等权下偏保守."""
    codes = [s.get('code') for s in picks if s.get('code')]
    if not codes:
        return 0.25  # default conservative
    placeholders = ','.join('?' * len(codes))
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute('PRAGMA busy_timeout=30000')
    try:
        rows = conn.execute(
            f"""
            SELECT s.code, dq.close
              FROM daily_quotes dq JOIN securities s ON s.id = dq.security_id
             WHERE s.code IN ({placeholders})
               AND dq.trade_date <= ?
               AND dq.trade_date >= date(?, '-{lookback_days * 2} days')
             ORDER BY s.code, dq.trade_date
            """,
            codes + [target_date, target_date],
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return 0.25
    import collections
    series = collections.defaultdict(list)
    for code, close in rows:
        if close is not None:
            series[code].append(float(close))
    vols = []
    for code, prices in series.items():
        if len(prices) >= 20:
            import numpy as np
            rets = np.diff(np.log(prices))
            vol_annual = float(np.std(rets) * np.sqrt(252))
            if 0.05 < vol_annual < 2.0:
                vols.append(vol_annual)
    if not vols:
        return 0.25
    return float(sum(vols) / len(vols))
```

- [ ] **Step 6: 在 selector 调用 compute_position_size**

`tomorrow_stock_selector.py:3884-3888` 改成（替换原 stock_with_scores = kept 行）：

```python
                kept, dropped_by_overlay = apply_overlay_to_picks(stock_with_scores, _decision)
                if dropped_by_overlay:
                    logger.info(
                        f"[ng2.1] L1+L2 dropped {len(dropped_by_overlay)} picks "
                        f"(reasons in _drop_reason)"
                    )
                # P0.1: L3 vol-target sizing — compute position_size + persist stop_loss_pct
                from stock_selctor.ng21_risk_overlay import compute_position_size, estimate_portfolio_vol
                _db = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'data_adapter', 'stock_data.db',
                )
                est_vol = estimate_portfolio_vol(kept, _db, _td)
                kept = compute_position_size(kept, _decision, est_portfolio_vol=est_vol)
                logger.info(
                    f"[P0.1 sizing] est_portfolio_vol={est_vol:.2%}, "
                    f"VT={_decision.vol_target_annual:.0%}, "
                    f"avg_pos={sum(s.get('position_size',0) for s in kept)/max(len(kept),1):.2%}, "
                    f"total={sum(s.get('position_size',0) for s in kept):.2%}"
                )
                self._ng21_decision = _decision
                self._ng21_dropped_by_overlay = dropped_by_overlay
                stock_with_scores = kept
```

- [ ] **Step 7: 默认 ng1.0.6 v1 启用 overlay (保守牛市参数)**

`stock_selctor/scoring_router.py:112` 附近改：当版本是 `ng1.0.6` (不带 +overlay 后缀) 也设 `ng106_overlay_mode = True`，这样进 selector 的 ng106_overlay_mode 分支自动启用 `_ng21_mode`。

具体: 找到 `res.ng106_overlay_mode = "+overlay" in scoring_version` 改为：

```python
res.ng106_overlay_mode = (
    "+overlay" in scoring_version or scoring_version in ("ng1.0.6", "ng1.0.62")
)
```

- [ ] **Step 8: 跑生产模拟 (今日报告)**

```bash
python3 tomorrow_stock_selector.py 2026-04-27 --scoring-version ng1.0.6 2>&1 | tee /tmp/p01_smoke.log
grep -E "P0.1 sizing|risk overlay" /tmp/p01_smoke.log
ls reports/daily_selection_ng106/analysis_data_*.json | tail -1 | xargs -I {} python3 -c "
import json
d = json.load(open('{}'))
sample = (d.get('all_stocks_with_scores') or [])[:3]
for s in sample:
    print(s.get('code'), s.get('rank_score'), 'pos=', s.get('position_size'), 'sl=', s.get('stop_loss_pct'))
"
```

Expected: log 出现 `[P0.1 sizing] ... avg_pos=10.00%, total=80.00%`，JSON 前 3 票每个有 `position_size` + `stop_loss_pct=-0.08`。

- [ ] **Step 9: Commit**

```bash
git add stock_selctor/ng21_risk_overlay.py stock_selctor/scoring_router.py \
        tomorrow_stock_selector.py stock_selctor/test/test_ng21_position_size.py
git commit -m "feat(P0.1): 生产化 L3 vol-target + L5 SL — ng1.0.6 默认启用 overlay, JSON 写入 position_size/stop_loss_pct"
```

---

### Task P0.2: forward_test_tracker daily wire + 90-day rolling dashboard

**背景**：`scripts/forward_test_tracker.py` 已经把 panel schema、scan/report/gate 子命令写完，但 (a) 没接入 daily 流程；(b) 没 wiki dashboard 展现。

**Files:**
- Verify: `scripts/forward_test_tracker.py` (已存在, 测能跑)
- Modify: `run_daily_update.sh` 末尾加 forward scan 步骤
- Create: `scripts/forward_test_dashboard.py` — 90 日滚动 IC/Sharpe markdown
- Create: `reports/forward_test/dashboard.md` (output)

- [ ] **Step 1: smoke test forward_test_tracker scan**

```bash
python3 scripts/forward_test_tracker.py scan \
  --report-dir reports/daily_selection_ng106 \
  --scoring-version ng1.0.6 \
  --since 2026-03-01 --until 2026-04-26 2>&1 | tail -20
ls reports/forward_test/forward_samples.csv
wc -l reports/forward_test/forward_samples.csv
```
Expected: 创建 csv，行数 > 0。如失败需修。

- [ ] **Step 2: smoke test report 子命令**

```bash
python3 scripts/forward_test_tracker.py report \
  --scoring-version ng1.0.6 --window-days 90 2>&1 | tail -30
```
Expected: 输出 forward IC / Top-10 实际 / win rate。如脚本接口和我假设不一致，读源码核对。

- [ ] **Step 3: 建 dashboard 脚本**

```python
# scripts/forward_test_dashboard.py
"""90 日滚动 forward OOS dashboard, 输出到 wiki-readable markdown."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_PATH = ROOT / "reports" / "forward_test" / "forward_samples.csv"
OUT_PATH = ROOT / "reports" / "forward_test" / "dashboard.md"


def load_panel() -> pd.DataFrame:
    if not SAMPLES_PATH.exists():
        raise SystemExit(f"missing {SAMPLES_PATH} — run forward_test_tracker scan first")
    return pd.read_csv(SAMPLES_PATH)


def rolling_ic(df: pd.DataFrame, score_col: str, ret_col: str, window_days: int) -> list[dict]:
    out = []
    df = df.dropna(subset=[score_col, ret_col]).copy()
    df['report_date'] = pd.to_datetime(df['report_date'])
    by_date = df.groupby('report_date')
    daily = []
    for d, g in by_date:
        if len(g) >= 10:
            ic, _ = spearmanr(g[score_col], g[ret_col])
            daily.append((d, ic, len(g)))
    daily.sort()
    for i in range(len(daily)):
        end = daily[i][0]
        start = end - pd.Timedelta(days=window_days)
        win = [(d, ic, n) for d, ic, n in daily if start <= d <= end and not pd.isna(ic)]
        if len(win) >= 5:
            ics = np.array([x[1] for x in win])
            out.append({
                'date': end.strftime('%Y-%m-%d'),
                'forward_ic_mean': float(ics.mean()),
                'forward_icir': float(ics.mean() / (ics.std() + 1e-9)),
                'n_days': len(win),
                'avg_breadth': int(np.mean([x[2] for x in win])),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scoring-version', default='ng1.0.6')
    ap.add_argument('--window-days', type=int, default=90)
    args = ap.parse_args()
    df = load_panel()
    df = df[df['scoring_version'] == args.scoring_version]
    if df.empty:
        raise SystemExit(f"no rows for {args.scoring_version}")
    rows = rolling_ic(df, 'rank_score', 'forward_ret_10d', args.window_days)
    if not rows:
        raise SystemExit("not enough overlapping samples")
    last = rows[-1]
    lines = [
        f"# Forward OOS Dashboard — {args.scoring_version}",
        "",
        f"**Updated**: {last['date']}  ",
        f"**Window**: {args.window_days} 日滚动  ",
        f"**最新 forward IC** (90d): {last['forward_ic_mean']:.4f}  ",
        f"**最新 forward ICIR** (90d): {last['forward_icir']:.4f}  ",
        f"**样本天数**: {last['n_days']}  ",
        "",
        "## 历史趋势 (最近 30 个观测)",
        "",
        "| date | forward_ic_90d | forward_icir_90d | n_days |",
        "|---|---:|---:|---:|",
    ]
    for r in rows[-30:]:
        lines.append(f"| {r['date']} | {r['forward_ic_mean']:+.4f} | {r['forward_icir']:+.4f} | {r['n_days']} |")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑 dashboard**

```bash
python3 scripts/forward_test_dashboard.py --scoring-version ng1.0.6 --window-days 90
cat reports/forward_test/dashboard.md
```
Expected: dashboard.md 写出，最后 30 行表格。

- [ ] **Step 5: 接入 daily update**

`run_daily_update.sh` 末尾追加（注意 set -e 时单步失败不要拖死整个脚本）：

```bash
# Forward OOS tracking (P0.2)
echo "[forward-OOS] scan + dashboard"
python3 scripts/forward_test_tracker.py scan \
  --report-dir reports/daily_selection_ng106 \
  --scoring-version ng1.0.6 \
  --since "$(date -v-7d +%Y-%m-%d)" \
  --until "$(date +%Y-%m-%d)" \
  || echo "[forward-OOS] scan skipped (non-fatal)"
python3 scripts/forward_test_dashboard.py --scoring-version ng1.0.6 --window-days 90 \
  || echo "[forward-OOS] dashboard skipped (non-fatal)"
```

- [ ] **Step 6: Commit**

```bash
git add scripts/forward_test_dashboard.py run_daily_update.sh
git commit -m "feat(P0.2): forward OOS 90 日滚动 dashboard + daily wire"
```

---

## P1 — 风控融合骨架（2-6 周）

### Task P1.3: Risk-adjusted Calmar label spike (ng1.6.2)

**Files:**
- Modify: `ml_models/ng/ng_trainer.py` — 加 `--label-mode {industry_excess,calmar}` 选项
- Modify: `ml_models/ng/ng_schema.py` — 新增版本 `ng1.6.2`
- Test: smoke train 2020-2023 一窗口 + WF-OOS

- [ ] **Step 1: 加 label-mode 参数到 ng_trainer**

在 `ng_trainer.py` argparse 处加：
```python
parser.add_argument('--label-mode', choices=['industry_excess', 'calmar', 'sortino'],
                    default='industry_excess')
parser.add_argument('--calmar-floor', type=float, default=0.05,
                    help='floor for |maxDD| denominator to avoid blow-up')
```

- [ ] **Step 2: 实现 calmar 标签计算**

在 trainer 标签生成函数中分支：

```python
if label_mode == 'calmar':
    # future_ret_Nd / max(|future_maxdd_Nd|, floor)
    # 需要从 daily_quotes 里算 future N-day 的 max drawdown
    future_dd = compute_future_maxdd(df, n_days=10)
    df['label_10d'] = df['label_10d'] / np.maximum(np.abs(future_dd), calmar_floor)
    # winsorize ±3σ
    q_lo, q_hi = df['label_10d'].quantile([0.005, 0.995])
    df['label_10d'] = df['label_10d'].clip(q_lo, q_hi)
```

- [ ] **Step 3: smoke train 2020-2023 单窗口 fast-check**

```bash
python3 ml_models/ng/ng_trainer.py \
  --start-date 2020-01-01 --end-date 2023-06-30 \
  --label-mode calmar --version ng1.6.2 \
  --fast-check --purge-days 15 --seed 42 2>&1 | tee logs/ng162_fastcheck.log
```
ABORT line: 10d ICIR < 0.6 OR β_UMD > 0.8

- [ ] **Step 4: 全量训练 + WF-OOS**

```bash
caffeinate -i python3 ml_models/ng/ng_trainer.py \
  --start-date 2020-01-01 --label-mode calmar --version ng1.6.2 \
  --purge-days 15 --target-parallel 4 --seed 42 2>&1 | tee logs/ng162_full.log
```

- [ ] **Step 5: 北极星对比**

```bash
python3 backtest/batch_generate_v395_reports.py --version ng1.6.2 --start 2020-01-01 --end 2026-04-26
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_ng162 \
    --label NG162-WFOOS --top-n 10 --focus-days 10 --rank-field composite
```

**接受标准**: V5.2 ≥ 70% AND MaxDD 比 ng1.0.6 (-21.4%) 改善 ≥ 3pp。

- [ ] **Step 6: Commit (无论 PASS/REJECT)**

```bash
git add ml_models/ng/ng_trainer.py ml_models/ng/ng_schema.py logs/ng162_*.log
git commit -m "experiment(P1.3): ng1.6.2 Calmar label spike — V5.2=XX%, MaxDD=XX%"
```

---

### Task P1.4: Pre-2020 风格诊断

**Files:**
- Create: `scripts/diagnose_pre2020_factor_decay.py`
- Output: `reports/diagnostics/pre2020_factor_decay.md`

- [ ] **Step 1: 因子分段 IC 比较脚本**

```python
# scripts/diagnose_pre2020_factor_decay.py
"""每个 ng1.0.1 因子在 2018-2019 vs 2020-2026 的 IC 对比, 找符号翻转 / |IC| 崩塌."""
from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data_adapter" / "stock_data.db"
OUT = ROOT / "reports" / "diagnostics" / "pre2020_factor_decay.md"

def load_panel(start, end):
    conn = sqlite3.connect(DB, timeout=30)
    try:
        df = pd.read_sql_query(
            "SELECT trade_date, code, features_json FROM ng101_feature_cache "
            "WHERE trade_date BETWEEN ? AND ?",
            conn, params=[start, end],
        )
    finally:
        conn.close()
    if df.empty:
        return df
    import json
    feats = df['features_json'].apply(json.loads).apply(pd.Series)
    return pd.concat([df[['trade_date','code']], feats], axis=1)

def fwd_ret(panel, n=10):
    conn = sqlite3.connect(DB, timeout=30)
    try:
        ret = pd.read_sql_query(
            "SELECT s.code, dq.trade_date, dq.close FROM daily_quotes dq "
            "JOIN securities s ON s.id = dq.security_id "
            "WHERE dq.trade_date BETWEEN ? AND ?",
            conn, params=[panel['trade_date'].min(), panel['trade_date'].max()],
        )
    finally:
        conn.close()
    ret = ret.sort_values(['code','trade_date'])
    ret['fwd_ret'] = ret.groupby('code')['close'].pct_change(periods=n).shift(-n)
    return ret[['code','trade_date','fwd_ret']]

def daily_ic(panel_with_ret, factor):
    out = []
    for d, g in panel_with_ret.groupby('trade_date'):
        g2 = g[[factor, 'fwd_ret']].dropna()
        if len(g2) >= 30:
            ic, _ = spearmanr(g2[factor], g2['fwd_ret'])
            if not np.isnan(ic):
                out.append(ic)
    return np.array(out)

def main():
    pre = load_panel('2018-01-01', '2019-12-31')
    post = load_panel('2020-01-01', '2026-04-26')
    if pre.empty or post.empty:
        raise SystemExit("missing ng101 cache rows for one of the windows")
    pre_ret = fwd_ret(pre)
    post_ret = fwd_ret(post)
    pre_join = pre.merge(pre_ret, on=['code','trade_date'])
    post_join = post.merge(post_ret, on=['code','trade_date'])

    factors = [c for c in pre.columns if c not in ('trade_date','code')]
    rows = []
    for f in factors:
        ic_pre = daily_ic(pre_join, f)
        ic_post = daily_ic(post_join, f)
        if len(ic_pre) < 50 or len(ic_post) < 50:
            continue
        rows.append({
            'factor': f,
            'ic_pre_mean': ic_pre.mean(),
            'ic_post_mean': ic_post.mean(),
            'icir_pre': ic_pre.mean() / (ic_pre.std()+1e-9),
            'icir_post': ic_post.mean() / (ic_post.std()+1e-9),
            'sign_flip': np.sign(ic_pre.mean()) != np.sign(ic_post.mean()),
            'abs_decay': abs(ic_pre.mean()) - abs(ic_post.mean()),
        })
    df = pd.DataFrame(rows).sort_values('abs_decay', ascending=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Pre-2020 因子风格衰减诊断", "",
             f"对比窗口: 2018-01 ~ 2019-12 (pre) vs 2020-01 ~ 2026-04 (post)", "",
             "## 符号翻转因子", ""]
    flip = df[df['sign_flip']]
    if not flip.empty:
        lines.append(flip.to_markdown(index=False, floatfmt=".4f"))
    lines += ["", "## |IC| 衰减 Top 15", ""]
    lines.append(df.head(15).to_markdown(index=False, floatfmt=".4f"))
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑诊断**

```bash
python3 scripts/diagnose_pre2020_factor_decay.py
cat reports/diagnostics/pre2020_factor_decay.md | head -50
```
Expected: 符号翻转因子列表 + Top 15 衰减榜。

- [ ] **Step 3: Commit**

```bash
git add scripts/diagnose_pre2020_factor_decay.py reports/diagnostics/pre2020_factor_decay.md
git commit -m "feat(P1.4): Pre-2020 因子风格衰减诊断脚本 + 报告"
```

---

### Task P1.5: 真实容量回测 (ADV 5%)

**Files:**
- Modify: `backtest/backtest_report_based.py` — 加 `--capital` + `--adv-cap` 参数

- [ ] **Step 1: 加 ADV cap 逻辑**

在 backtest 主循环里 (定位 buy 阶段)，每只票 fill 量计算成 `min(target_size, adv_20d × adv_cap × capital)`。如果 fill < 90% target，记 `slippage_warning`。

- [ ] **Step 2: 跑 4 档容量**

```bash
for cap in 50_000_000 100_000_000 300_000_000 1_000_000_000; do
  python3 backtest/backtest_report_based.py \
    --report-dir reports/daily_selection_ng106 \
    --capital $cap --adv-cap 0.05 \
    --start 2020-01-01 --end 2026-04-26 \
    --output reports/capacity/ng106_cap_${cap}.json
done
python3 -c "
import json
for cap in [50000000, 100000000, 300000000, 1000000000]:
    d = json.load(open(f'reports/capacity/ng106_cap_{cap}.json'))
    print(f'{cap/1e8:.1f}亿: ann={d[\"annualized_return\"]:.2%}, sharpe={d[\"sharpe_ratio\"]:.2f}, fill_rate={d.get(\"avg_fill_rate\",1.0):.2%}')
"
```

- [ ] **Step 3: Commit**

```bash
git add backtest/backtest_report_based.py reports/capacity/
git commit -m "feat(P1.5): 真实容量回测 (ADV 5% cap, 4 档资金规模)"
```

---

### Task P1.6: Multi-task auxiliary head (Layer 1)

**Files:**
- Create: `ml_models/ng/ng_risk_head.py` — 独立训 maxDD / vol 头
- Test: `ml_models/ng/test_risk_head.py`

- [ ] **Step 1: 标签生成**

```python
# ml_models/ng/ng_risk_head.py
"""Layer 1 of risk-aware MOE: independent GBDTs predicting future maxDD/vol.

Output goes to ng2.2 meta-learner Layer 2 — see plan P1.6/P2 (TBD ng2.2 spec).
"""
from __future__ import annotations
import argparse, os, sqlite3
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
import joblib

ROOT = Path(__file__).resolve().parent.parent.parent
DB = ROOT / "data_adapter" / "stock_data.db"

def compute_future_maxdd(close_panel: pd.DataFrame, n: int = 60) -> pd.Series:
    """前向 N 日相对于持仓首日的最大回撤. close_panel: pivot (date × code)."""
    fwd_min = close_panel.rolling(n, min_periods=10).min().shift(-n)
    return (fwd_min / close_panel - 1.0).clip(lower=-1.0, upper=0.0).stack()

def compute_future_vol(close_panel: pd.DataFrame, n: int = 10) -> pd.Series:
    log_ret = np.log(close_panel).diff()
    fwd_vol = log_ret.rolling(n).std().shift(-n) * np.sqrt(252)
    return fwd_vol.stack()

def load_features(start: str, end: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB, timeout=30)
    try:
        df = pd.read_sql_query(
            "SELECT trade_date, code, features_json FROM ng101_feature_cache "
            "WHERE trade_date BETWEEN ? AND ?", conn, params=[start, end])
    finally:
        conn.close()
    import json
    feats = df['features_json'].apply(json.loads).apply(pd.Series)
    return pd.concat([df[['trade_date','code']], feats], axis=1)

def train_head(X, y, valid_mask, params=None):
    p = dict(objective='regression', metric='rmse', learning_rate=0.05,
             num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=5, verbose=-1)
    if params: p.update(params)
    train = lgb.Dataset(X[~valid_mask], y[~valid_mask])
    valid = lgb.Dataset(X[valid_mask], y[valid_mask], reference=train)
    model = lgb.train(p, train, num_boost_round=2000, valid_sets=[valid],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', choices=['maxdd_60d','vol_10d'], required=True)
    ap.add_argument('--start', default='2020-01-01')
    ap.add_argument('--end', default='2024-12-31')
    ap.add_argument('--purge-days', type=int, default=15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    df = load_features(args.start, args.end)
    # 拉 close 算标签
    conn = sqlite3.connect(DB)
    try:
        q = pd.read_sql_query(
            "SELECT s.code, dq.trade_date, dq.close FROM daily_quotes dq "
            "JOIN securities s ON s.id = dq.security_id "
            "WHERE dq.trade_date BETWEEN ? AND ?",
            conn, params=[args.start, args.end])
    finally:
        conn.close()
    close_p = q.pivot(index='trade_date', columns='code', values='close').sort_index()
    if args.target == 'maxdd_60d':
        y = compute_future_maxdd(close_p, 60).rename('y').reset_index()
    else:
        y = compute_future_vol(close_p, 10).rename('y').reset_index()
    y.columns = ['trade_date','code','y']
    j = df.merge(y, on=['trade_date','code']).dropna(subset=['y'])
    feat_cols = [c for c in j.columns if c not in ('trade_date','code','y')]
    X = j[feat_cols].astype(float).fillna(0).values
    yv = j['y'].astype(float).values
    cutoff = pd.Timestamp(args.end) - pd.Timedelta(days=180)
    valid_mask = pd.to_datetime(j['trade_date']) >= cutoff

    np.random.seed(args.seed)
    model = train_head(X, yv, valid_mask.values)
    out = args.out or f"ml_models/trained_models/ng/risk_head_{args.target}.pkl"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'feat_cols': feat_cols, 'target': args.target}, out)
    print(f"saved {out}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: smoke train both heads**

```bash
caffeinate -i python3 ml_models/ng/ng_risk_head.py --target maxdd_60d \
  --start 2020-01-01 --end 2024-12-31 --seed 42 2>&1 | tee logs/risk_head_maxdd.log
caffeinate -i python3 ml_models/ng/ng_risk_head.py --target vol_10d \
  --start 2020-01-01 --end 2024-12-31 --seed 42 2>&1 | tee logs/risk_head_vol.log
```

- [ ] **Step 3: WF-OOS IC 验证**

写个简单 IC 评估脚本 (复用 P1.4 的 daily_ic) 在 2025 数据上验证 pred_maxdd_60d 和 pred_vol_10d 的 IC。
**接受标准**: 两个头的 OOS IC > 0.10 (signal 真实)。

- [ ] **Step 4: Commit**

```bash
git add ml_models/ng/ng_risk_head.py logs/risk_head_*.log
git commit -m "feat(P1.6): ng risk auxiliary heads (maxdd_60d / vol_10d) Layer 1"
```

> **Layer 2 (utility-aware meta-learner) 留 ng2.2 spec 单独写, 这里只完成 Layer 1**.

---

## P2 — 长期实验（1-3 月）

### Task P2.7: regime soft-MOE (V11 → P_bull probability)

**Files:**
- Modify: `indicators/regime_classifier.py` — 加 `compute_regime_proba()`
- Modify: `tomorrow_stock_selector.py` — ng106_mode 分支用 P 加权

- [ ] **Step 1: P_bull logistic**

```python
# indicators/regime_classifier.py 末尾追加:
def compute_regime_proba(var1: float, ma60: float, macd: float, macd_prev: float,
                         streak: int) -> float:
    """V11 hard switch 的 logistic 软化版."""
    pos = (var1 - ma60) / max(abs(ma60), 1e-6)        # 位置 (vs MA60 % 距)
    above = 1.0 if macd > 0 else 0.0
    rising = 1.0 if macd > macd_prev else 0.0
    streak_n = max(min(streak, 5), -5) / 5.0          # [-1, 1]
    z = 5.0 * pos + 1.0 * above + 1.0 * rising + 1.0 * streak_n
    return 1.0 / (1.0 + np.exp(-z))                    # ∈ (0, 1)
```

- [ ] **Step 2: 推理用 P 加权**

selector 的 ng106 分支里:
```python
p_bull = compute_regime_proba(var1, ma60, macd, macd_prev, streak)
score = p_bull * bull_score + (1 - p_bull) * bear_score
```

- [ ] **Step 3: WF-OOS 对比 hard switch**

```bash
python3 backtest/batch_generate_v395_reports.py --version ng1.0.6-soft --start 2020-01-01
python3 backtest/run_north_star_eval.py --report-dir reports/daily_selection_ng106soft --label SOFT-MOE
```

**接受标准**: V5.2 ≥ ng1.0.6 v1 (78.9%) AND 切换日 (regime change) 当周换手率 < hard switch 的 80%。

- [ ] **Step 4: Commit**

```bash
git add indicators/regime_classifier.py tomorrow_stock_selector.py
git commit -m "feat(P2.7): regime soft-MOE — logistic P_bull 替代 V11 hard switch"
```

---

### Task P2.8: 8-strategy + Signal Trust ranking weight

**Files:**
- Modify: `tomorrow_stock_selector.py` — ML Top-30 后加 booster

- [ ] **Step 1: post-rank booster**

定位 ML 排序后、L1-L5 overlay 前的位置，加：

```python
# P2.8: 8-strategy & Signal Trust ranking booster (NOT pre-filter, post-rank only)
STRATEGY_BONUS_BY_REGIME = {
    'bull': {'少负': 8, 'SuperB1': 5, '补票': 5, '暴力K': 3},
    'bear': {'暴力K': 8, 'SuperB1': 5, '补票': 5},
}
TRUST_MULT = {'🟢': 1.0, '🟡': 0.85, '🔴': 0.6, '⚪': 1.0}

def apply_post_rank_booster(picks, regime, trust_lookup):
    bonus_table = STRATEGY_BONUS_BY_REGIME.get(regime, {})
    for s in picks:
        rs = float(s.get('rank_score', 0))
        bonus = sum(bonus_table.get(strat, 0) for strat in s.get('strategy_hits', []))
        trust = trust_lookup.get(s.get('code'), '⚪')
        mult = TRUST_MULT.get(trust, 1.0)
        s['rank_score_boosted'] = (rs + bonus) * mult
    return sorted(picks, key=lambda x: -x['rank_score_boosted'])[:30]
```

- [ ] **Step 2: 接 strategy hits 数据**

`stock_signals` 表已有当日各策略命中股；用 `SELECT code, strategy FROM stock_signals WHERE date=?` 喂给 booster。

- [ ] **Step 3: A/B 对比**

跑两份报告 (booster on/off), 北极星打分差异。

**接受标准**: V5.2 不退化 (Δ ≥ -1pp) AND 命中策略股的实际 Top-10 收益不低于无 booster (无害 baseline)。

- [ ] **Step 4: Commit**

```bash
git add tomorrow_stock_selector.py
git commit -m "feat(P2.8): post-rank booster (8策略 regime-conditional + signal trust 加权)"
```

---

## Self-Review

**Spec coverage**:
- ✅ P0.1 = §二 P0.1 productionize L3+L5
- ✅ P0.2 = §二 P0.2 forward OOS tracker
- ✅ P1.3-P1.6 = §三 path A (Calmar) + path B Layer 1 (risk heads) + 配套诊断/容量
- ✅ P2.7-P2.8 = §二 P2.6 soft MOE + P2.7 booster

**Gaps acknowledged**:
- P1.6 Layer 2 (utility-aware meta-learner) **不在本 plan**, 留给后续 ng2.2 专项 plan
- P2 path C (Differentiable Sharpe) **不在本 plan**, ng3.x 长期实验

**Type consistency**: `RiskDecision` / `apply_overlay_to_picks` / `compute_position_size` 在 P0.1 三步统一签名一致。

**Risk gates 写死**:
- P1.3 (Calmar): V5.2 ≥ 70% AND MaxDD-3pp 改善
- P1.6 (heads): OOS IC > 0.10
- P2.7 (soft-MOE): V5.2 ≥ 78.9% AND 切换日换手 < 80% hard switch
- P2.8 (booster): V5.2 退化 ≤ 1pp
