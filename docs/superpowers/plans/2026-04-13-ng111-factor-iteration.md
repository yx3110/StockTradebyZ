# ng1.1.1 因子迭代 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 ng1.0.1 bugfix (66 特征) 基础上，用 EMT 四关验证框架做 Phase 1 清理 + Phase 2 筛选 14 个候选新因子 + Phase 3 重训，产出 ng1.1.1 模型；硬目标 MaxDD<-10% AND Sharpe>2.5。

**Architecture:** 两阶段 funnel：Phase 1 用 EMT Gate 1+3 审计 ng1.0.1 bugfix 的 60 个现役特征（删除 3 bug 冗余 + 3 P0 近零）→ Phase 2 用 EMT Gate 1-4 筛选 14 个零成本候选因子 → Phase 3 用筛选后的 feature list 新建 `ng111_feature_cache` 并训练 `ng1.1.1` 模型，三层评估（WF / Pre-2020 / 生产）。

**Tech Stack:** Python 3.13, LightGBM/XGBoost/CatBoost ensemble, SQLite, EMT 四关验证框架 (`~/EastMoneyTrader/analysis/feature_validator.py`), NG trainer 继承体系。

**Spec:** `docs/superpowers/specs/2026-04-13-ng111-factor-iteration-design.md`

---

## 前置：文件结构

**新增文件**
- `scripts/ng111/phase1_audit.py` — Phase 1 现役特征 IC + 冗余审计
- `scripts/ng111/phase2_candidate_factors.py` — 14 个候选因子的 SQL+pandas 计算逻辑
- `scripts/ng111/phase2_run_validator.py` — 14 候选批量调用 EMT validator
- `scripts/ng111/__init__.py`
- `reports/ng111/` — 全部审计产物（gitignore 外，会 commit）

**修改文件**
- `ml_models/ng/ng_schema.py` — 加 `ng1.1.1` 映射
- `ml_models/ng/ng_trainer.py` — 加 `NG111_*` 特征常量 + 版本分支
- `ml_models/ng/ng_cache_updater.py` — 支持 `ng1.1.1` 缓存表
- `tomorrow_stock_selector.py` — SCORER_REGISTRY 注册 `ng1.1.1`
- `docs/wiki/models/ng-series.md` — 加 ng1.1.1 章节
- `docs/wiki/models/ng-factor-quality.md` — 加 Phase 1/2 结论
- `.claude/projects/-Users-yangxu-StockTradebyZ/memory/MEMORY.md` — 一行索引

---

## Task 0: 环境校验 & Sanity Check

**目的：** 跑通一个已知高置信度的 candidate（F3 `close_to_ma60_pct`，wiki 记录 ng1.0.4 贡献 2.5%），验证"candidate CSV → EMT validator → 结果落盘"链路可用。如果 sanity check 都 REJECT 说明工具链或数据有问题。

**Files:**
- Create: `scripts/ng111/__init__.py`
- Create: `scripts/ng111/phase2_candidate_factors.py`（仅含 `compute_close_to_ma60_pct`，后续扩）
- Create: `reports/ng111/sanity/` （目录）

- [ ] **Step 0.1: 创建工作目录结构**

```bash
mkdir -p /Users/yangxu/StockTradebyZ/scripts/ng111
mkdir -p /Users/yangxu/StockTradebyZ/reports/ng111/sanity
mkdir -p /Users/yangxu/StockTradebyZ/reports/ng111/phase1
mkdir -p /Users/yangxu/StockTradebyZ/reports/ng111/phase2
touch /Users/yangxu/StockTradebyZ/scripts/ng111/__init__.py
```

- [ ] **Step 0.2: 写一个最小 candidate compute helper**

Create `scripts/ng111/phase2_candidate_factors.py`:

```python
"""ng1.1.1 候选因子计算 — 输出 (date, stock_code, value) CSV 供 EMT validator 消费."""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data_adapter" / "stock_data.db"
OUT_DIR = PROJECT_ROOT / "reports" / "ng111" / "phase2" / "candidates_csv"


def _load_daily_quotes(start: str, end: str) -> pd.DataFrame:
    """Load code, trade_date, close, volume, amount, high, low, open for date range."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        q = """
            SELECT s.code, dq.trade_date, dq.close, dq.volume, dq.amount,
                   dq.high, dq.low, dq.open, dq.price_change_pct
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.type = 'A股'
              AND dq.trade_date BETWEEN ? AND ?
            ORDER BY s.code, dq.trade_date
        """
        df = pd.read_sql(q, conn, params=[start, end])
    finally:
        conn.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.rename(columns={'code': 'stock_code', 'trade_date': 'date'})
    return df


def compute_close_to_ma60_pct(start: str, end: str) -> pd.DataFrame:
    """F3: close / MA60 - 1. Computed from daily_quotes (not technical_indicators 避免 trust 已有 col)."""
    df = _load_daily_quotes(start, end)
    df = df.sort_values(['stock_code', 'date'])
    df['ma60'] = df.groupby('stock_code')['close'].transform(
        lambda s: s.rolling(window=60, min_periods=40).mean()
    )
    df['value'] = df['close'] / df['ma60'] - 1
    # shift(1) — 用 T-1 日特征预测 T+1 日收益 (EMT 铁律)
    df['value'] = df.groupby('stock_code')['value'].shift(1)
    return df[['date', 'stock_code', 'value']].dropna()


def export_candidate(name: str, start: str, end: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fn = {
        'close_to_ma60_pct': compute_close_to_ma60_pct,
    }[name]
    out = OUT_DIR / f"{name}.csv"
    fn(start, end).to_csv(out, index=False)
    return out


if __name__ == '__main__':
    import sys
    name = sys.argv[1]
    start = sys.argv[2] if len(sys.argv) > 2 else '2023-01-01'
    end = sys.argv[3] if len(sys.argv) > 3 else '2026-04-10'
    path = export_candidate(name, start, end)
    print(f"✅ Exported {name} → {path}")
```

- [ ] **Step 0.3: 导出 sanity CSV**

Run:
```bash
cd /Users/yangxu/StockTradebyZ
python3 scripts/ng111/phase2_candidate_factors.py close_to_ma60_pct 2023-01-01 2026-04-10
```

Expected: `✅ Exported close_to_ma60_pct → /Users/yangxu/StockTradebyZ/reports/ng111/phase2/candidates_csv/close_to_ma60_pct.csv`
Verify: `wc -l <output>` 应该有 > 1M 行（4000+ 股票 × 700+ 交易日）

- [ ] **Step 0.4: 跑 EMT validator 验证链路**

Run:
```bash
cd /Users/yangxu/EastMoneyTrader
python3 scripts/validate_feature.py \
    --candidate /Users/yangxu/StockTradebyZ/reports/ng111/phase2/candidates_csv/close_to_ma60_pct.csv \
    --name close_to_ma60_pct \
    --target 10d \
    --save-result /Users/yangxu/StockTradebyZ/reports/ng111/sanity/close_to_ma60_pct_10d.txt
```

Expected: 输出 4 关结果；decision ∈ {ACCEPT, MARGINAL}（REJECT 说明 sanity fail，要调查）。
存储：`reports/ng111/sanity/close_to_ma60_pct_10d.txt`

- [ ] **Step 0.5: 检查 sanity 结果**

```bash
cat /Users/yangxu/StockTradebyZ/reports/ng111/sanity/close_to_ma60_pct_10d.txt
```

判定：
- 如果 decision = ACCEPT 或 MARGINAL → 工具链可用，下一步
- 如果 decision = REJECT AND Gate1 ic_mean<0.005 → 工具链异常，检查 shift(1)、数据日期、股票过滤，不要继续
- 如果 Gate4 baseline_ic 非常低（<0.03） → EMT 的采样策略可能 nan 过多，`--sample` 调到 10000 再试

- [ ] **Step 0.6: Commit sanity**

```bash
cd /Users/yangxu/StockTradebyZ
git add scripts/ng111/__init__.py scripts/ng111/phase2_candidate_factors.py
git add reports/ng111/sanity/
git commit -m "feat(ng111): sanity check — EMT validator链路打通 (F3 close_to_ma60_pct)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 1: Phase 1 — 现役特征审计（Gate 1+3）

**目的：** 对 ng1.0.1 bugfix 模型的 60 现役特征（66 − 3 bug 冗余 − 3 P0 近零，候选淘汰池）跑 EMT Gate 1 IC + Gate 3 冗余，产出 KEEP/DROP 清单。

**Files:**
- Create: `scripts/ng111/phase1_audit.py`
- Create: `reports/ng111/phase1/audit.csv`
- Create: `reports/ng111/phase1/audit.md`

- [ ] **Step 1.1: 写 Phase 1 审计脚本**

Create `scripts/ng111/phase1_audit.py`:

```python
"""ng1.1.1 Phase 1 — 现役特征 Gate 1+3 审计.

对 ng1.0.1 bugfix 模型的 feature_names 减去已知必删的 6 个 (3 bug redundant + 3 P0 近零),
对剩余每个特征跑 EMT Gate 1 (单因子 IC) + Gate 3 (冗余/正交).
市场特征跳过 Gate 1 (截面常数), 只查 Gate 3 + gain+shap 排名.
输出 CSV + Markdown.
"""
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

EMT_PATH = Path("/Users/yangxu/EastMoneyTrader")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(EMT_PATH))
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.data_loader import DataLoader
from analysis.feature_audit import (
    load_feature_matrix, load_ng_model,
    compute_gain_importance, compute_shap_importance,
    compute_univariate_ic_batched, compute_correlation_matrix,
    find_redundant_pairs,
    MARKET_FEATURE_COLS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ng111_phase1")

# ng1.0.1 bugfix 模型里已知必删的 6 个 (不再审计)
PRE_DROPPED = frozenset({
    # 3 bug redundant (ng110 已删)
    'volume_contraction', 'sw_index_return_5d', 'industry_relative_strength',
    # 3 P0 near-zero (ng110 已删)
    'roe_change', 'n_sectors_strong', 'days_since_breakout',
})

# 放宽阈值 (Phase 1 审计用, 避免误伤 market/core)
GATE1_ICIR_MIN = 0.3
GATE1_IC_MEAN_MIN = 0.01   # 默认 0.02, 放宽到 0.01
GATE1_HIT_RATE_MIN = 52.0   # 默认 55, 放宽到 52
GATE3_CORR_MAX = 0.85       # 默认 0.8, 放宽到 0.85

# Market 特征跳过 Gate 1, 用 gain/shap 阈值淘汰
MARKET_GAIN_MIN_PCT = 0.005  # gain_importance < 0.5% 判弱
MARKET_SHAP_RANK_WORST_PCT = 0.30  # SHAP rank 后 30% 判弱


def audit(start: str = '2023-01-01', end: str = '2026-04-10',
          target: str = '10d', sample_n: int = 50000,
          out_dir: Path = PROJECT_ROOT / 'reports' / 'ng111' / 'phase1'):
    out_dir.mkdir(parents=True, exist_ok=True)

    data_loader = DataLoader(config_path=str(EMT_PATH / "config/settings.json"))
    model_path = data_loader.stz_path / "ml_models/trained_models/ng/ng101_seed42_multi_target_20260412_233749.pkl"
    model_data = load_ng_model(model_path)
    all_features = list(model_data['feature_names'])
    audit_features = [f for f in all_features if f not in PRE_DROPPED]
    logger.info(f"Pre-dropped: {len(PRE_DROPPED)}, 待审计: {len(audit_features)}")

    label_col = f"label_{target}"
    df = load_feature_matrix(data_loader, start, end, sample_n=sample_n)
    if label_col not in df.columns:
        raise RuntimeError(f"{label_col} not in feature matrix")

    # Gate 1 IC (stock features 才算, market 特征在 audit_df 中会是 nan)
    logger.info("计算 Gate 1 单因子 IC (batched)...")
    ic_stats = compute_univariate_ic_batched(df, audit_features, label_col)

    # Gain + SHAP (for market feature weak detection)
    logger.info("Gain + SHAP importance...")
    gain = compute_gain_importance(model_data, target)
    X = df[all_features].copy()  # 原 69 列按模型 feature_names 对齐
    shap = compute_shap_importance(model_data, X, target)

    # Gate 3 Redundancy pairs
    logger.info("Gate 3 correlation matrix...")
    corr = compute_correlation_matrix(df, audit_features)
    redundant = find_redundant_pairs(corr, threshold=GATE3_CORR_MAX)

    # Build per-feature decision
    rows = []
    for f in audit_features:
        is_market = f in MARKET_FEATURE_COLS
        ic_m = ic_stats.loc[f, 'ic_mean'] if f in ic_stats.index else np.nan
        ic_ir = ic_stats.loc[f, 'ic_ir'] if f in ic_stats.index else np.nan
        hit = ic_stats.loc[f, 'pct_positive'] if f in ic_stats.index else np.nan
        gain_val = float(gain.get(f, 0.0))
        shap_val = float(shap.get(f, 0.0))

        # Gate 1 check (只对 stock 特征)
        if is_market:
            g1_pass = True  # skip
            g1_reason = "market feature — skipped"
        else:
            g1_pass = (
                (abs(ic_m) >= GATE1_IC_MEAN_MIN) and
                (abs(ic_ir) >= GATE1_ICIR_MIN) and
                (hit >= GATE1_HIT_RATE_MIN) and
                not pd.isna(ic_m)
            )
            g1_reason = (
                f"|ic|={abs(ic_m):.4f}(thr {GATE1_IC_MEAN_MIN}), "
                f"|icir|={abs(ic_ir):.4f}(thr {GATE1_ICIR_MIN}), "
                f"hit={hit:.1f}%(thr {GATE1_HIT_RATE_MIN}%)"
            )

        # Gate 3 check: find strongest same-group redundant
        rdf = redundant[(redundant['feature_a'] == f) | (redundant['feature_b'] == f)]
        if len(rdf):
            partners = [
                (r['feature_b'] if r['feature_a'] == f else r['feature_a'], r['correlation'])
                for _, r in rdf.iterrows()
            ]
            partners.sort(key=lambda x: abs(x[1]), reverse=True)
            g3_partner, g3_corr = partners[0]
            # 保留 IC 更强的一个
            my_ic_abs = abs(ic_m) if not pd.isna(ic_m) else 0
            partner_ic_abs = (
                abs(ic_stats.loc[g3_partner, 'ic_mean']) if g3_partner in ic_stats.index else 0
            )
            g3_drop = my_ic_abs < partner_ic_abs
            g3_reason = f"max_corr={g3_corr:.3f} with {g3_partner} (partner_ic={partner_ic_abs:.4f})"
        else:
            g3_drop = False
            g3_reason = ""

        # Market weak check
        market_weak = False
        market_reason = ""
        if is_market:
            shap_rank = shap.rank(pct=True).get(f, np.nan)
            if gain_val < MARKET_GAIN_MIN_PCT and shap_rank < (1 - MARKET_SHAP_RANK_WORST_PCT):
                market_weak = True
                market_reason = f"market: gain={gain_val:.4f}<{MARKET_GAIN_MIN_PCT}, shap_rank={shap_rank:.2f}"

        drop = (not g1_pass) or g3_drop or market_weak
        reasons = []
        if not g1_pass: reasons.append(f"G1: {g1_reason}")
        if g3_drop: reasons.append(f"G3: {g3_reason}")
        if market_weak: reasons.append(market_reason)

        rows.append({
            'feature': f,
            'is_market': is_market,
            'ic_mean': ic_m,
            'ic_ir': ic_ir,
            'hit_rate': hit,
            'gain_importance': gain_val,
            'shap_importance': shap_val,
            'g1_pass': g1_pass,
            'g3_drop': g3_drop,
            'market_weak': market_weak,
            'decision': 'DROP' if drop else 'KEEP',
            'reason': '; '.join(reasons) if reasons else 'pass',
        })

    out_df = pd.DataFrame(rows).sort_values(['decision', 'feature'])
    csv_path = out_dir / 'audit.csv'
    out_df.to_csv(csv_path, index=False)
    logger.info(f"CSV → {csv_path}")

    # Markdown summary
    keep = out_df[out_df['decision'] == 'KEEP']
    drop = out_df[out_df['decision'] == 'DROP']
    md = [
        "# ng1.1.1 Phase 1 — 现役特征审计",
        "",
        f"**Base**: ng1.0.1 bugfix (66 feat) − {len(PRE_DROPPED)} 预删 = {len(audit_features)} 审计特征",
        f"**Target**: label_{target}, Sample {sample_n}, 日期 {start} ~ {end}",
        f"**阈值**: Gate1 |IC|≥{GATE1_IC_MEAN_MIN}, |ICIR|≥{GATE1_ICIR_MIN}, hit≥{GATE1_HIT_RATE_MIN}%; Gate3 max_corr≥{GATE3_CORR_MAX}",
        "",
        f"## 结果: KEEP {len(keep)}, DROP {len(drop)}",
        "",
        "### DROP 列表 (待从 ng1.1.1 移除)",
        "",
        drop[['feature', 'ic_mean', 'ic_ir', 'hit_rate', 'reason']].to_markdown(index=False, floatfmt=".4f"),
        "",
        "### KEEP 列表",
        "",
        keep[['feature', 'is_market', 'ic_mean', 'ic_ir', 'gain_importance']].to_markdown(index=False, floatfmt=".4f"),
        "",
        "### 预删 (ng1.0.1 bugfix 里保留, ng1.1.0 已证明无用, 我们继承删除)",
        "",
        *[f"- `{f}`" for f in sorted(PRE_DROPPED)],
    ]
    md_path = out_dir / 'audit.md'
    md_path.write_text("\n".join(md))
    logger.info(f"MD → {md_path}")

    return out_df


if __name__ == '__main__':
    audit()
```

- [ ] **Step 1.2: 跑 Phase 1 审计**

Run:
```bash
cd /Users/yangxu/StockTradebyZ
python3 scripts/ng111/phase1_audit.py 2>&1 | tee reports/ng111/phase1/run.log
```

Expected: 3-8 分钟。最终输出 `reports/ng111/phase1/audit.csv` + `audit.md`

- [ ] **Step 1.3: 检查结果**

```bash
cat reports/ng111/phase1/audit.md | head -80
```

判定：
- 如果 DROP 数 ≥ 3 → 清理有价值，记下 DROP 的 feature 列表供 Task 9 使用
- 如果 DROP 数 < 3 → 清理阶段价值低，仍继续；Task 9 只用预删 6 个
- 如果 DROP 超过 15 → 阈值太严，人工复核不要误伤 `roe_ttm` 等核心因子

- [ ] **Step 1.4: Commit Phase 1**

```bash
git add scripts/ng111/phase1_audit.py reports/ng111/phase1/
git commit -m "feat(ng111): Phase 1 — 现役特征 Gate 1+3 审计 (X drop / Y keep)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Phase 2 候选因子计算模块 (扩 F1-F14)

**目的：** 把所有 14 个候选因子的计算函数写到 `phase2_candidate_factors.py`，统一接口 `compute_<name>(start, end) → pd.DataFrame(columns=[date, stock_code, value])`。每个函数强制 `shift(1)` 防未来函数。

**Files:**
- Modify: `scripts/ng111/phase2_candidate_factors.py`

- [ ] **Step 2.1: 补全全部 14 个候选的计算函数**

Edit `scripts/ng111/phase2_candidate_factors.py` — 在文件末尾（`if __name__` 之前）追加：

```python
# ---------------------------------------------------------------------------
# F1: amount_acceleration = d(log(amount_ma5))/Δt (日差)
# ---------------------------------------------------------------------------
def compute_amount_acceleration(start: str, end: str) -> pd.DataFrame:
    df = _load_daily_quotes(start, end).sort_values(['stock_code', 'date'])
    df['amount_ma5'] = df.groupby('stock_code')['amount'].transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )
    df['log_amt_ma5'] = np.log(df['amount_ma5'].clip(lower=1))
    df['value'] = df.groupby('stock_code')['log_amt_ma5'].diff()
    df['value'] = df.groupby('stock_code')['value'].shift(1)
    return df[['date', 'stock_code', 'value']].dropna()


# ---------------------------------------------------------------------------
# F2: turnover_volatility_20d = std(turnover_rate)/mean(turnover_rate) 20d
# ---------------------------------------------------------------------------
def _load_daily_basic(start: str, end: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        q = """
            SELECT s.code, db.trade_date, db.turnover_rate, db.pe_ttm, db.pb,
                   db.total_share, db.float_share, db.free_share, db.total_mv, db.circ_mv
            FROM daily_basic db
            JOIN securities s ON db.security_id = s.id
            WHERE s.type = 'A股'
              AND db.trade_date BETWEEN ? AND ?
            ORDER BY s.code, db.trade_date
        """
        df = pd.read_sql(q, conn, params=[start, end])
    finally:
        conn.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.rename(columns={'code': 'stock_code', 'trade_date': 'date'})
    return df


def compute_turnover_volatility_20d(start: str, end: str) -> pd.DataFrame:
    df = _load_daily_basic(start, end).sort_values(['stock_code', 'date'])
    g = df.groupby('stock_code')['turnover_rate']
    df['to_mean'] = g.transform(lambda s: s.rolling(20, min_periods=15).mean())
    df['to_std'] = g.transform(lambda s: s.rolling(20, min_periods=15).std())
    df['value'] = df['to_std'] / (df['to_mean'].abs() + 1e-8)
    df['value'] = df.groupby('stock_code')['value'].shift(1)
    return df[['date', 'stock_code', 'value']].dropna()


# F3 close_to_ma60_pct 已在 Step 0.2 实现 — 保持不变


# ---------------------------------------------------------------------------
# F4: up_days_ratio_20d = sum(return > 0) / 20
# ---------------------------------------------------------------------------
def compute_up_days_ratio_20d(start: str, end: str) -> pd.DataFrame:
    df = _load_daily_quotes(start, end).sort_values(['stock_code', 'date'])
    df['ret'] = df.groupby('stock_code')['close'].pct_change()
    df['up'] = (df['ret'] > 0).astype(float)
    df['value'] = df.groupby('stock_code')['up'].transform(
        lambda s: s.rolling(20, min_periods=15).mean()
    )
    df['value'] = df.groupby('stock_code')['value'].shift(1)
    return df[['date', 'stock_code', 'value']].dropna()


# ---------------------------------------------------------------------------
# F5: moneyflow_net_5d_z = (sum(net_mf_amount, 5d) - mean_60d) / std_60d
# ---------------------------------------------------------------------------
def compute_moneyflow_net_5d_z(start: str, end: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        q = """
            SELECT code AS stock_code, trade_date AS date, net_mf_amount
            FROM moneyflow_daily
            WHERE trade_date BETWEEN ? AND ?
            ORDER BY code, trade_date
        """
        df = pd.read_sql(q, conn, params=[start, end])
    finally:
        conn.close()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date'])
    df['net_mf_5d'] = df.groupby('stock_code')['net_mf_amount'].transform(
        lambda s: s.rolling(5, min_periods=5).sum()
    )
    df['mu_60'] = df.groupby('stock_code')['net_mf_5d'].transform(
        lambda s: s.rolling(60, min_periods=40).mean()
    )
    df['sd_60'] = df.groupby('stock_code')['net_mf_5d'].transform(
        lambda s: s.rolling(60, min_periods=40).std()
    )
    df['value'] = (df['net_mf_5d'] - df['mu_60']) / (df['sd_60'] + 1e-8)
    df['value'] = df.groupby('stock_code')['value'].shift(1)
    return df[['date', 'stock_code', 'value']].dropna()


# ---------------------------------------------------------------------------
# F6: signal_trust_score_60d = 最近 60d 发出的 signal 的 trust color 加权分
#     (🟢=1, 🟡=0.5, 🔴=-0.5, ⚪=0)
# ---------------------------------------------------------------------------
def compute_signal_trust_score_60d(start: str, end: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        q = """
            SELECT code AS stock_code, signal_date AS date, trust_color
            FROM signal_trust_records
            WHERE signal_date BETWEEN ? AND ?
        """
        try:
            df = pd.read_sql(q, conn, params=[start, end])
        except Exception:
            # signal_trust table 可能不是这个名字, 探测
            q2 = "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%signal_trust%'"
            tables = [r[0] for r in conn.execute(q2).fetchall()]
            if not tables:
                raise RuntimeError(f"signal_trust 表未找到 (tried signal_trust_records). 检查 scripts/rebuild_signal_trust.py 是否已建库.")
            # use first signal_trust table
            table = tables[0]
            q3 = f"SELECT * FROM {table} LIMIT 1"
            cols = [d[0] for d in conn.execute(q3).description]
            logger.info(f"signal_trust 表 = {table}, cols = {cols}")
            raise RuntimeError(f"请修正 signal_trust schema, 实际表/列: {table} {cols}")
    finally:
        conn.close()

    color_map = {'🟢': 1.0, '🟡': 0.5, '🔴': -0.5, '⚪': 0.0}
    df['w'] = df['trust_color'].map(color_map).fillna(0.0)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date'])
    df['value'] = df.groupby('stock_code')['w'].transform(
        lambda s: s.rolling(60, min_periods=10).mean()
    )
    df['value'] = df.groupby('stock_code')['value'].shift(1)
    return df[['date', 'stock_code', 'value']].dropna()


# ---------------------------------------------------------------------------
# F7: return_skew_20d
# ---------------------------------------------------------------------------
def compute_return_skew_20d(start: str, end: str) -> pd.DataFrame:
    df = _load_daily_quotes(start, end).sort_values(['stock_code', 'date'])
    df['ret'] = df.groupby('stock_code')['close'].pct_change()
    df['value'] = df.groupby('stock_code')['ret'].transform(
        lambda s: s.rolling(20, min_periods=15).skew()
    )
    df['value'] = df.groupby('stock_code')['value'].shift(1)
    return df[['date', 'stock_code', 'value']].dropna()


# ---------------------------------------------------------------------------
# F8: illiq_amihud_20d = mean(|return|/amount) * 1e8 (scaling for numerical stability)
# ---------------------------------------------------------------------------
def compute_illiq_amihud_20d(start: str, end: str) -> pd.DataFrame:
    df = _load_daily_quotes(start, end).sort_values(['stock_code', 'date'])
    df['ret'] = df.groupby('stock_code')['close'].pct_change()
    df['illiq'] = df['ret'].abs() / (df['amount'].clip(lower=1) / 1e8)
    df['value'] = df.groupby('stock_code')['illiq'].transform(
        lambda s: s.rolling(20, min_periods=15).mean()
    )
    df['value'] = df.groupby('stock_code')['value'].shift(1)
    return df[['date', 'stock_code', 'value']].dropna()


# ---------------------------------------------------------------------------
# F9: log_close_vs_vwap_20d = log(close / vwap_20d), vwap = sum(amount)/sum(volume)
# ---------------------------------------------------------------------------
def compute_log_close_vs_vwap_20d(start: str, end: str) -> pd.DataFrame:
    df = _load_daily_quotes(start, end).sort_values(['stock_code', 'date'])
    df['amt20'] = df.groupby('stock_code')['amount'].transform(
        lambda s: s.rolling(20, min_periods=15).sum()
    )
    df['vol20'] = df.groupby('stock_code')['volume'].transform(
        lambda s: s.rolling(20, min_periods=15).sum()
    )
    df['vwap20'] = df['amt20'] / (df['vol20'].abs() + 1e-8)
    df['value'] = np.log((df['close'] + 1e-8) / (df['vwap20'] + 1e-8))
    df['value'] = df.groupby('stock_code')['value'].shift(1)
    return df[['date', 'stock_code', 'value']].dropna()


# ---------------------------------------------------------------------------
# F10: max_drawdown_60d = (running_max - close) / running_max within 60d
# ---------------------------------------------------------------------------
def compute_max_drawdown_60d(start: str, end: str) -> pd.DataFrame:
    df = _load_daily_quotes(start, end).sort_values(['stock_code', 'date'])
    df['max60'] = df.groupby('stock_code')['close'].transform(
        lambda s: s.rolling(60, min_periods=40).max()
    )
    df['value'] = (df['max60'] - df['close']) / (df['max60'].abs() + 1e-8)
    df['value'] = df.groupby('stock_code')['value'].shift(1)
    return df[['date', 'stock_code', 'value']].dropna()


# ---------------------------------------------------------------------------
# F11: beta_to_market_60d = cov(r_stock, r_mkt)/var(r_mkt), 60d
#      使用上证指数 000001.SH 作为市场代理
# ---------------------------------------------------------------------------
def compute_beta_to_market_60d(start: str, end: str) -> pd.DataFrame:
    stock_df = _load_daily_quotes(start, end)
    stock_df['ret'] = stock_df.groupby('stock_code')['close'].pct_change()

    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        q = """
            SELECT dq.trade_date, dq.close
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.code = '000001.SH' AND s.type = '指数'
              AND dq.trade_date BETWEEN ? AND ?
            ORDER BY dq.trade_date
        """
        mkt = pd.read_sql(q, conn, params=[start, end])
    finally:
        conn.close()
    mkt['trade_date'] = pd.to_datetime(mkt['trade_date'])
    mkt['mkt_ret'] = mkt['close'].pct_change()
    mkt = mkt.rename(columns={'trade_date': 'date'})[['date', 'mkt_ret']]

    df = stock_df.merge(mkt, on='date', how='left').sort_values(['stock_code', 'date'])

    def _rolling_beta(g: pd.DataFrame) -> pd.Series:
        cov = g['ret'].rolling(60, min_periods=40).cov(g['mkt_ret'])
        var = g['mkt_ret'].rolling(60, min_periods=40).var()
        return cov / (var + 1e-8)

    df['value'] = df.groupby('stock_code', group_keys=False).apply(_rolling_beta)
    df['value'] = df.groupby('stock_code')['value'].shift(1)
    return df[['date', 'stock_code', 'value']].dropna()


# ---------------------------------------------------------------------------
# F12: accrual_ratio = (net_profit - ocf) / total_assets
#      从 financial_indicator 按季度取, forward-fill 到日频
# ---------------------------------------------------------------------------
def compute_accrual_ratio(start: str, end: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        # financial_indicator 通常有 end_date (报告期) 而非每日
        q = """
            SELECT s.code AS stock_code, fi.end_date AS report_date,
                   fi.n_income AS net_profit,
                   fi.ocf AS ocf,
                   fi.total_assets AS total_assets
            FROM financial_indicator fi
            JOIN securities s ON fi.security_id = s.id
            WHERE s.type = 'A股'
              AND fi.end_date BETWEEN ? AND ?
        """
        # 财务报告至少要提前查 1 年才能 forward-fill
        fi_start = (pd.to_datetime(start) - pd.Timedelta(days=400)).strftime('%Y-%m-%d')
        df_fi = pd.read_sql(q, conn, params=[fi_start, end])
    finally:
        conn.close()
    if df_fi.empty:
        raise RuntimeError("financial_indicator 查无数据, 检查表名/列名是否正确")
    df_fi['report_date'] = pd.to_datetime(df_fi['report_date'])
    df_fi['accrual'] = (df_fi['net_profit'] - df_fi['ocf']) / (df_fi['total_assets'].abs() + 1e-8)
    df_fi = df_fi.sort_values(['stock_code', 'report_date'])

    # 合并到日频: 对每支股票, 每日取最近一期 report_date 的 accrual
    stock_df = _load_daily_quotes(start, end)[['date', 'stock_code']]
    merged = pd.merge_asof(
        stock_df.sort_values('date'),
        df_fi[['stock_code', 'report_date', 'accrual']].sort_values('report_date'),
        left_on='date', right_on='report_date',
        by='stock_code', direction='backward',
    )
    merged = merged.rename(columns={'accrual': 'value'})
    # shift(1): 财报披露会有滞后, 已隐含在 report_date <= date; 仍显式 shift(1) 保守
    merged = merged.sort_values(['stock_code', 'date'])
    merged['value'] = merged.groupby('stock_code')['value'].shift(1)
    return merged[['date', 'stock_code', 'value']].dropna()


# ---------------------------------------------------------------------------
# F13: industry_alpha_20d = cum_return_stock_20d - cum_return_industry_20d
# ---------------------------------------------------------------------------
def compute_industry_alpha_20d(start: str, end: str) -> pd.DataFrame:
    stock_df = _load_daily_quotes(start, end).sort_values(['stock_code', 'date'])
    stock_df['ret'] = stock_df.groupby('stock_code')['close'].pct_change()

    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        ind = pd.read_sql(
            """SELECT code AS stock_code, industry
               FROM stock_basic_info WHERE industry IS NOT NULL""",
            conn,
        )
    finally:
        conn.close()

    df = stock_df.merge(ind, on='stock_code', how='left').dropna(subset=['industry'])
    df['stock_cum20'] = df.groupby('stock_code')['ret'].transform(
        lambda s: (1 + s).rolling(20, min_periods=15).apply(np.prod, raw=True) - 1
    )
    ind_ret = df.groupby(['date', 'industry'])['ret'].mean().reset_index().rename(columns={'ret': 'ind_ret'})
    ind_cum = ind_ret.sort_values(['industry', 'date'])
    ind_cum['ind_cum20'] = ind_cum.groupby('industry')['ind_ret'].transform(
        lambda s: (1 + s).rolling(20, min_periods=15).apply(np.prod, raw=True) - 1
    )
    df = df.merge(ind_cum[['date', 'industry', 'ind_cum20']], on=['date', 'industry'], how='left')
    df['value'] = df['stock_cum20'] - df['ind_cum20']
    df['value'] = df.groupby('stock_code')['value'].shift(1)
    return df[['date', 'stock_code', 'value']].dropna()


# ---------------------------------------------------------------------------
# F14: overnight_return_20d = mean((open_t - close_{t-1})/close_{t-1}, 20d)
# ---------------------------------------------------------------------------
def compute_overnight_return_20d(start: str, end: str) -> pd.DataFrame:
    df = _load_daily_quotes(start, end).sort_values(['stock_code', 'date'])
    df['prev_close'] = df.groupby('stock_code')['close'].shift(1)
    df['overnight'] = (df['open'] - df['prev_close']) / (df['prev_close'].abs() + 1e-8)
    df['value'] = df.groupby('stock_code')['overnight'].transform(
        lambda s: s.rolling(20, min_periods=15).mean()
    )
    df['value'] = df.groupby('stock_code')['value'].shift(1)
    return df[['date', 'stock_code', 'value']].dropna()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
CANDIDATE_REGISTRY = {
    'amount_acceleration': compute_amount_acceleration,           # F1
    'turnover_volatility_20d': compute_turnover_volatility_20d,   # F2
    'close_to_ma60_pct': compute_close_to_ma60_pct,               # F3
    'up_days_ratio_20d': compute_up_days_ratio_20d,               # F4
    'moneyflow_net_5d_z': compute_moneyflow_net_5d_z,             # F5
    'signal_trust_score_60d': compute_signal_trust_score_60d,     # F6
    'return_skew_20d': compute_return_skew_20d,                   # F7
    'illiq_amihud_20d': compute_illiq_amihud_20d,                 # F8
    'log_close_vs_vwap_20d': compute_log_close_vs_vwap_20d,       # F9
    'max_drawdown_60d': compute_max_drawdown_60d,                 # F10
    'beta_to_market_60d': compute_beta_to_market_60d,             # F11
    'accrual_ratio': compute_accrual_ratio,                       # F12
    'industry_alpha_20d': compute_industry_alpha_20d,             # F13
    'overnight_return_20d': compute_overnight_return_20d,         # F14
}


def export_all_candidates(start: str = '2023-01-01', end: str = '2026-04-10'):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, fn in CANDIDATE_REGISTRY.items():
        out = OUT_DIR / f"{name}.csv"
        if out.exists():
            print(f"⏭  {name} 已存在, 跳过")
            results[name] = out
            continue
        try:
            df = fn(start, end)
            df.to_csv(out, index=False)
            print(f"✅ {name} → {len(df)} 行 → {out}")
            results[name] = out
        except Exception as e:
            print(f"❌ {name} 失败: {e}")
            results[name] = None
    return results
```

更新 `if __name__` 部分支持 `--all`:

```python
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        start = sys.argv[2] if len(sys.argv) > 2 else '2023-01-01'
        end = sys.argv[3] if len(sys.argv) > 3 else '2026-04-10'
        export_all_candidates(start, end)
    else:
        name = sys.argv[1]
        start = sys.argv[2] if len(sys.argv) > 2 else '2023-01-01'
        end = sys.argv[3] if len(sys.argv) > 3 else '2026-04-10'
        path = export_candidate(name, start, end) if name == 'close_to_ma60_pct' else \
               CANDIDATE_REGISTRY[name](start, end).pipe(
                   lambda d: (OUT_DIR.mkdir(parents=True, exist_ok=True),
                              d.to_csv(OUT_DIR / f"{name}.csv", index=False),
                              OUT_DIR / f"{name}.csv")[-1])
        print(f"✅ Exported {name} → {path}")
```

- [ ] **Step 2.2: 验证 F5 / F6 / F12 的数据表存在**

Run:
```bash
cd /Users/yangxu/StockTradebyZ
python3 -c "
import sqlite3
conn = sqlite3.connect('data_adapter/stock_data.db', timeout=30)
for name in ['moneyflow_daily', 'financial_indicator', 'stock_basic_info']:
    cur = conn.execute(f\"SELECT name FROM sqlite_master WHERE type='table' AND name='{name}'\")
    print(f'{name}: {\"✅ exists\" if cur.fetchone() else \"❌ MISSING\"}')
# signal_trust 表名探测
cur = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%signal_trust%'\")
print('signal_trust tables:', [r[0] for r in cur.fetchall()])
# financial_indicator 列名
cur = conn.execute('PRAGMA table_info(financial_indicator)')
print('financial_indicator cols:', [r[1] for r in cur.fetchall()][:20])
"
```

Expected: `moneyflow_daily ✅`, `financial_indicator ✅`, `stock_basic_info ✅`, signal_trust 有至少 1 个表。如果 F12 的 `ocf` / `n_income` / `total_assets` 列不存在，需要查实际列名并改 `compute_accrual_ratio`。

- [ ] **Step 2.3: 对 F5 (moneyflow) 跑一个 sanity export**

```bash
python3 scripts/ng111/phase2_candidate_factors.py moneyflow_net_5d_z 2023-01-01 2026-04-10
```

Expected: `✅ Exported moneyflow_net_5d_z → ...`, CSV 行数 > 500k

- [ ] **Step 2.4: Commit 因子计算模块**

```bash
git add scripts/ng111/phase2_candidate_factors.py
git add reports/ng111/phase2/candidates_csv/moneyflow_net_5d_z.csv
git commit -m "feat(ng111): 14 候选因子计算模块 + F5 sanity

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Phase 2 候选批量验证

**目的：** 对 14 候选逐个跑 EMT validator（target=10d），结果归档到 `reports/ng111/phase2/<name>_10d.txt`，产出 summary markdown。

**Files:**
- Create: `scripts/ng111/phase2_run_validator.py`
- Create: `reports/ng111/phase2/summary.md`
- Create: `reports/ng111/phase2/accepted_features.json`

- [ ] **Step 3.1: 导出全部 14 个候选 CSV**

```bash
cd /Users/yangxu/StockTradebyZ
python3 scripts/ng111/phase2_candidate_factors.py --all 2023-01-01 2026-04-10 2>&1 | tee reports/ng111/phase2/export.log
```

Expected: 14 个 `✅` 或若干 `❌`。对每个 `❌` 查 stderr，修 compute 函数后重跑（用 `rm reports/ng111/phase2/candidates_csv/<name>.csv` 清掉再跑）。

- [ ] **Step 3.2: 写批量验证脚本**

Create `scripts/ng111/phase2_run_validator.py`:

```python
"""Phase 2 批量调用 EMT validator for 14 candidates."""
import subprocess
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EMT_PATH = Path("/Users/yangxu/EastMoneyTrader")
CSV_DIR = PROJECT_ROOT / "reports" / "ng111" / "phase2" / "candidates_csv"
OUT_DIR = PROJECT_ROOT / "reports" / "ng111" / "phase2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = [
    'amount_acceleration', 'turnover_volatility_20d', 'close_to_ma60_pct',
    'up_days_ratio_20d', 'moneyflow_net_5d_z', 'signal_trust_score_60d',
    'return_skew_20d', 'illiq_amihud_20d', 'log_close_vs_vwap_20d',
    'max_drawdown_60d', 'beta_to_market_60d', 'accrual_ratio',
    'industry_alpha_20d', 'overnight_return_20d',
]

def run_validator(name: str, target: str = '10d', sample: int = 30000) -> dict:
    csv = CSV_DIR / f"{name}.csv"
    out_txt = OUT_DIR / f"{name}_{target}.txt"
    if out_txt.exists() and out_txt.stat().st_size > 500:
        print(f"⏭  {name} 已验证, 跳过")
        return {'name': name, 'target': target, 'skipped': True, 'txt': str(out_txt)}
    if not csv.exists():
        return {'name': name, 'target': target, 'error': f'missing CSV: {csv}'}
    cmd = [
        'python3', 'scripts/validate_feature.py',
        '--candidate', str(csv),
        '--name', name,
        '--target', target,
        '--sample', str(sample),
        '--save-result', str(out_txt),
    ]
    print(f"▶️  {name}({target})")
    r = subprocess.run(cmd, cwd=EMT_PATH, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        return {'name': name, 'target': target, 'error': r.stderr[-500:]}
    return parse_result(out_txt, name, target)


def parse_result(txt: Path, name: str, target: str) -> dict:
    if not txt.exists():
        return {'name': name, 'target': target, 'error': 'no output file'}
    content = txt.read_text()
    # 决策行: "决策: ACCEPT  (4/4 关通过)"
    m = re.search(r'决策:\s*(\w+)\s*\((\d+)/(\d+)', content)
    if not m:
        return {'name': name, 'target': target, 'error': 'cannot parse decision'}
    decision, passed, total = m.group(1), int(m.group(2)), int(m.group(3))
    # 解析 Gate 1 ic_mean (简单 regex)
    ic_m = re.search(r'ic_mean:\s*([\-\d.]+)', content)
    icir_m = re.search(r'ic_ir:\s*([\-\d.]+)', content)
    delta_m = re.search(r'delta_ic:\s*([\-\d.]+)', content)
    return {
        'name': name, 'target': target,
        'decision': decision,
        'passed': passed, 'total': total,
        'ic_mean': float(ic_m.group(1)) if ic_m else None,
        'ic_ir': float(icir_m.group(1)) if icir_m else None,
        'delta_ic': float(delta_m.group(1)) if delta_m else None,
        'txt': str(txt),
    }


def main():
    results = []
    for name in CANDIDATES:
        r = run_validator(name, target='10d')
        results.append(r)
        print(f"   → {r.get('decision', r.get('error', 'unknown'))}")

    # 对每个 10d ACCEPT 的候选, 额外跑 5d / 15d 确认多 target 稳定
    extra = []
    for r in results:
        if r.get('decision') == 'ACCEPT':
            for t in ['5d', '15d']:
                extra.append(run_validator(r['name'], target=t))
    results.extend(extra)

    # 汇总
    summary_path = OUT_DIR / 'summary.md'
    accepted_path = OUT_DIR / 'accepted_features.json'

    # Build markdown
    header = "| Feature | Target | Decision | ICmean | ICIR | ΔIC |\n|---|---|---|---|---|---|"
    rows = []
    for r in sorted(results, key=lambda x: (x['name'], x['target'])):
        if 'error' in r:
            rows.append(f"| {r['name']} | {r['target']} | ERROR: {r['error'][:40]} | - | - | - |")
        else:
            rows.append(
                f"| {r['name']} | {r['target']} | **{r['decision']}** "
                f"({r['passed']}/{r['total']}) | "
                f"{r.get('ic_mean', 'n/a'):.4f} | "
                f"{r.get('ic_ir', 'n/a'):.4f} | "
                f"{r.get('delta_ic', 'n/a') if r.get('delta_ic') is not None else 'n/a'} |"
            )
    summary_path.write_text(
        "# ng1.1.1 Phase 2 — 14 候选因子四关验证\n\n"
        "严格模式: 只采纳 ACCEPT, MARGINAL 也 REJECT\n\n"
        + header + "\n"
        + "\n".join(rows)
    )

    # Build accepted list (10d ACCEPT, 去冗余 — 只保留每个高相关组的 IC 最强)
    accepted_10d = [r for r in results
                    if r['target'] == '10d' and r.get('decision') == 'ACCEPT']
    accepted_10d.sort(key=lambda x: abs(x.get('ic_ir', 0) or 0), reverse=True)
    accepted_path.write_text(json.dumps(
        [{'name': r['name'], 'ic_mean': r['ic_mean'], 'ic_ir': r['ic_ir'],
          'delta_ic': r['delta_ic']} for r in accepted_10d],
        ensure_ascii=False, indent=2,
    ))

    print(f"\n✅ Summary → {summary_path}")
    print(f"✅ Accepted JSON → {accepted_path}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 3.3: 跑批量验证（耗时 1-3 h）**

```bash
cd /Users/yangxu/StockTradebyZ
python3 scripts/ng111/phase2_run_validator.py 2>&1 | tee reports/ng111/phase2/run.log
```

Expected: 对每个 candidate 输出 `▶️ <name>(10d)` + decision。14 × ~5-15min（Gate 4 LGB 训练主耗时）。ACCEPT 候选额外跑 5d/15d。

- [ ] **Step 3.4: Review summary**

```bash
cat reports/ng111/phase2/summary.md
cat reports/ng111/phase2/accepted_features.json
```

判定分支：
- **accepted ≥ 1 并且 IC_IR > 0.3 稳定**：正常流程 → Task 4
- **accepted = 0**：spec 止损生效 → ng1.1.1 = 纯清理版 → 直接 Task 4，跳过加新因子
- **accepted ≥ 3**：人工复查 CSV 的 date 列最小值，确认 shift(1) 正确，无未来函数

- [ ] **Step 3.5: Commit Phase 2**

```bash
# CSV 太大, 只 commit 验证结果 txt + summary + accepted json
git add scripts/ng111/phase2_run_validator.py
git add reports/ng111/phase2/*.md reports/ng111/phase2/*.json reports/ng111/phase2/*.log
git add reports/ng111/phase2/*.txt
git commit -m "feat(ng111): Phase 2 — 14 候选四关验证结果 (N个ACCEPT)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

并将 candidates_csv 加到 .gitignore：
```bash
echo "reports/ng111/phase2/candidates_csv/" >> .gitignore
git add .gitignore && git commit -m "chore: gitignore ng111 candidates_csv (too large)"
```

---

## Task 4: ng1.1.1 Schema 注册

**目的：** 给 `ng_schema.py` 加 ng1.1.1 映射，新建独立的 `ng111_feature_cache` 表。

**Files:**
- Modify: `ml_models/ng/ng_schema.py`

- [ ] **Step 4.1: 加 version 映射**

Edit `ml_models/ng/ng_schema.py` at lines 19-27 (the `VERSION_TABLE_MAP` dict):

```python
VERSION_TABLE_MAP = {
    'ng1.0.0': 'ng_feature_cache',
    'ng1.0.1': 'ng101_feature_cache',
    'ng1.0.2': 'ng102_feature_cache',
    'ng1.0.3': 'ng103_feature_cache',
    'ng1.0.4': 'ng104_feature_cache',
    'ng1.0.7': 'ng107_feature_cache',
    'ng1.1.0': 'ng101_feature_cache',  # 基于ng1.0.1(69feat)精简, 复用ng101缓存
    'ng1.1.1': 'ng111_feature_cache',  # 基于ng1.0.1 + Phase1清理 + Phase2新因子, 独立表
}
```

并在 `SCHEMA_VERSION_MAP` 加一行（ng1.1.1 和 ng1.0.1 共享列结构，features_json 语义不同）：

```python
SCHEMA_VERSION_MAP = {
    'ng1.1.0': 'ng1.0.1',
    'ng1.1.1': 'ng1.0.1',  # same column schema, different features_json content
}
```

- [ ] **Step 4.2: 创建 ng111_feature_cache 表**

```bash
cd /Users/yangxu/StockTradebyZ
python3 ml_models/ng/ng_schema.py ng1.1.1
```

Expected: `ng111_feature_cache table ready: .../stock_data.db`

验证:
```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('data_adapter/stock_data.db', timeout=30)
cur = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='ng111_feature_cache'\")
print('✅' if cur.fetchone() else '❌')
"
```

- [ ] **Step 4.3: Commit schema**

```bash
git add ml_models/ng/ng_schema.py
git commit -m "feat(ng111): schema — ng111_feature_cache 表 (复用 ng101 列结构)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: ng1.1.1 Feature List 定义

**目的：** 根据 Phase 1 KEEP + Phase 2 accepted 构造最终 feature list，加到 `ng_trainer.py`。

**Files:**
- Modify: `ml_models/ng/ng_trainer.py`（追加 `NG111_*` 常量 + 版本分支）

- [ ] **Step 5.1: 从 Phase 1/2 产物读 feature list**

```bash
cd /Users/yangxu/StockTradebyZ
python3 -c "
import pandas as pd, json
keep = pd.read_csv('reports/ng111/phase1/audit.csv').query('decision == \"KEEP\"')
print('PHASE1_KEEP_STOCK =', [f for f in keep.query('is_market == False').feature])
print()
print('PHASE1_KEEP_MARKET =', [f for f in keep.query('is_market == True').feature])
print()
accepted = json.load(open('reports/ng111/phase2/accepted_features.json'))
print('PHASE2_ACCEPTED =', [a['name'] for a in accepted])
" | tee /tmp/ng111_feature_list.txt
```

- [ ] **Step 5.2: 把 feature list 写入 ng_trainer.py**

Edit `ml_models/ng/ng_trainer.py`—在 `NG110_VERSION = 'ng1.1.0'` 之后追加：

```python
# ---------------------------------------------------------------------------
# ng1.1.1: Phase1 清理 + Phase2 新因子
# 来源: reports/ng111/phase1/audit.csv (decision=KEEP) + reports/ng111/phase2/accepted_features.json
# 保持 market features 独立 list 方便 cache 的 extra cols 对齐
# ---------------------------------------------------------------------------

# 从 /tmp/ng111_feature_list.txt 复制进来 (Phase 1/2 完成后更新)
NG111_STOCK_FEATURES: List[str] = [
    # <-- Phase 1 KEEP (stock features, ~ 40-50 个) + Phase 2 ACCEPTED (0-3 个)
    # 占位, 根据 /tmp/ng111_feature_list.txt 在 Step 5.2 手填
]
NG111_MARKET_FEATURES: List[str] = list(MARKET_FEATURE_NAMES)
# 若 Phase 1 淘汰了 market feature, 在此覆盖; 否则保持 MARKET_FEATURE_NAMES 不变
NG111_ALL_FEATURES: List[str] = NG111_STOCK_FEATURES + NG111_MARKET_FEATURES
NG111_VERSION = 'ng1.1.1'
```

并在 `version_feature_table`（line 230+）增加 ng1.1.1 分支——改为：

```python
        version_feature_table = [
            ('ng1.1.1', NG111_ALL_FEATURES, NG111_STOCK_FEATURES, NG111_MARKET_FEATURES, []),
            ('ng1.1.0', NG110_ALL_FEATURES, NG110_STOCK_FEATURES, MARKET_FEATURE_NAMES, []),
            ('ng1.0.7', NG107_ALL_FEATURES, STOCK_FEATURE_NAMES,  NG107_MARKET_FEATURES, CONDITIONAL_IX_FEATURE_NAMES),
            ('ng1.0.4', NG104_ALL_FEATURES, NG104_STOCK_FEATURES, MARKET_FEATURE_NAMES,  []),
            ('ng0.0.0', ALL_FEATURE_NAMES,  STOCK_FEATURE_NAMES,  MARKET_FEATURE_NAMES,  []),
        ]
```

注意：`version_ge` 自左向右扫，ng1.1.1 必须在 ng1.1.0 之前（版本号更大的在上面）。

- [ ] **Step 5.3: 手填 NG111_STOCK_FEATURES 占位符**

打开 `/tmp/ng111_feature_list.txt`，把 `PHASE1_KEEP_STOCK + PHASE2_ACCEPTED` 的 list 粘到 `NG111_STOCK_FEATURES = []` 里。

例如（假设 Phase 1 drop 了 5 个，Phase 2 accept 了 1 个 `close_to_ma60_pct`）：
```python
NG111_STOCK_FEATURES: List[str] = [
    'trend_strength_20d', 'adx_proxy', 'pullback_to_ma10', 'pullback_to_ma20',
    'rsi_14', 'kdj_j_value', 'lower_shadow_ratio', 'intraday_recovery',
    # ... (其它 KEEP 的 stock feature)
    'close_to_ma60_pct',  # Phase 2 ACCEPT
]
```

- [ ] **Step 5.4: Smoke test trainer 加载**

```bash
python3 -c "
from ml_models.ng.ng_trainer import NGTrainer, NG111_ALL_FEATURES, NG111_VERSION
print('NG111_VERSION:', NG111_VERSION)
print('NG111_ALL_FEATURES:', len(NG111_ALL_FEATURES))
t = NGTrainer(version='ng1.1.1')
print('trainer feature_names:', len(t.feature_names))
print('cache_table:', t.cache_table)
"
```

Expected: feature_names 数量等于 `NG111_ALL_FEATURES`，cache_table = `ng111_feature_cache`

- [ ] **Step 5.5: Commit feature list**

```bash
git add ml_models/ng/ng_trainer.py
git commit -m "feat(ng111): feature list — phase1清理 + phase2新因子 (共X个特征)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: ng111_feature_cache 回填

**目的：** 把 Phase 2 ACCEPT 的新因子写入特征计算链路，然后回填 2020-2026 的 ng111_feature_cache。

**Files:**
- Modify: `ml_models/ng/ng_feature_calculator.py`（仅在 Phase 2 有 ACCEPT 时）
- Modify: `ml_models/ng/ng_cache_updater.py`（加 ng1.1.1 分支）

### 分支 A：Phase 2 有 ACCEPT 候选

- [ ] **Step 6.A.1：在 ng_feature_calculator.py 加 accepted 新因子的计算**

对每个 ACCEPT 因子，找到 `ng_feature_calculator.py` 合适的 group，添加新的 `def compute_<name>(...)` 函数，返回 `Dict[str, float]`。参考现有模式（如 `compute_trend_state` line 130+）。

新因子的实现逻辑：**和 `scripts/ng111/phase2_candidate_factors.py` 里的计算一致**，但接口变为 numpy-based（接受 `highs/lows/closes/volumes/amounts` numpy 数组，返回 dict）。

具体模板（以 F3 close_to_ma60_pct 为例，需根据最终 accepted 候选复制/改写）：

```python
def compute_close_to_ma60_pct(closes: np.ndarray) -> Dict[str, float]:
    """F3: close/MA60 - 1. 需要至少 60 根 K 线."""
    if len(closes) < 60:
        return {'close_to_ma60_pct': np.nan}
    ma60 = closes[-60:].mean()
    if ma60 < 1e-8:
        return {'close_to_ma60_pct': np.nan}
    return {'close_to_ma60_pct': float(closes[-1] / ma60 - 1)}
```

同时找到 `compute_all_features` 的主流程（search `def compute_all_features`），把新因子的返回 merge 进去。

- [ ] **Step 6.A.2：同步给 ng_cache_updater.py 加 ng1.1.1 分支**

Search `ng_cache_updater.py` 中对 ng1.1.0 的处理，复制模式到 ng1.1.1：
```bash
grep -n "ng1.1.0\|NG110" /Users/yangxu/StockTradebyZ/ml_models/ng/ng_cache_updater.py
```

编辑文件，对每个 `if version == 'ng1.1.0'` / `elif version_ge(version, 'ng1.1.0')` 分支后加对应 ng1.1.1 分支。需要保证 ng1.1.1 用 `NG111_STOCK_FEATURES + NG111_MARKET_FEATURES`。

### 分支 B：Phase 2 零 ACCEPT（纯清理版）

- [ ] **Step 6.B.1：ng_feature_calculator.py 无需改动**
- [ ] **Step 6.B.2：ng_cache_updater.py 只需加 ng1.1.1 分支用 ng1.0.1 同样的 feature 计算**

### 通用 step

- [ ] **Step 6.3：回填 ng111_feature_cache（全量 2020-2026）**

如果是分支 B（纯清理），ng111_feature_cache 的 features_json 是 ng101 的严格子集 — 可以用 SQL 从 ng101_feature_cache 快速派生，不需重新计算。把下面脚本存为 `scripts/ng111/derive_cache_from_ng101.py` 再运行：

```python
#!/usr/bin/env python3
"""纯清理版 (Phase 2 零 ACCEPT): 从 ng101_feature_cache 派生 ng111_feature_cache.

只过滤 features_json 的 keys, 不重新计算; 几分钟完成.
"""
import sqlite3, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml_models.ng.ng_trainer import NG111_STOCK_FEATURES

INSERT_SQL = """
INSERT OR REPLACE INTO ng111_feature_cache
(code, trade_date, features_json, label_3d, label_5d, label_10d, label_15d,
 market_return_5d, market_return_20d, market_volatility_20d, market_breadth,
 market_new_high_ratio, northbound_flow_5d, market_volume_ratio,
 market_drawdown, vix_proxy, market_momentum_diff)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

def main():
    conn = sqlite3.connect('data_adapter/stock_data.db', timeout=30)
    src_rows = conn.execute("SELECT COUNT(*) FROM ng101_feature_cache").fetchone()
    print(f'ng101 source rows: {src_rows[0]}')

    stock_cols = set(NG111_STOCK_FEATURES)
    conn.execute("DELETE FROM ng111_feature_cache")

    batch, total = [], 0
    cursor = conn.execute("""
        SELECT code, trade_date, features_json, label_3d, label_5d, label_10d, label_15d,
               market_return_5d, market_return_20d, market_volatility_20d, market_breadth,
               market_new_high_ratio, northbound_flow_5d, market_volume_ratio,
               market_drawdown, vix_proxy, market_momentum_diff
        FROM ng101_feature_cache
    """)
    for row in cursor:
        code, td, fj, *rest = row
        try:
            feat = json.loads(fj)
        except Exception:
            continue
        new_feat = {k: feat.get(k) for k in stock_cols}
        batch.append((code, td, json.dumps(new_feat, separators=(',', ':')), *rest))
        if len(batch) >= 5000:
            conn.executemany(INSERT_SQL, batch)
            total += len(batch); batch = []
            if total % 100000 == 0:
                print(f'  inserted {total}...')
    if batch:
        conn.executemany(INSERT_SQL, batch)
        total += len(batch)
    conn.commit()
    print(f'✅ done. Total rows inserted: {total}')

if __name__ == '__main__':
    main()
```

Run:
```bash
python3 scripts/ng111/derive_cache_from_ng101.py
```

（如果是分支 A 有新因子，改用 `ml_models/ng/ng_cache_updater.py --version ng1.1.1 --start-date 2020-01-01 --end-date 2026-04-13` 全量重算，耗时 0.5-2 h。）

- [ ] **Step 6.4：验证 cache**

```bash
python3 -c "
import sqlite3, json
conn = sqlite3.connect('data_adapter/stock_data.db', timeout=30)
n, mn, mx = conn.execute('SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM ng111_feature_cache').fetchone()
print(f'rows={n}, date range {mn} ~ {mx}')
one = conn.execute('SELECT features_json FROM ng111_feature_cache LIMIT 1').fetchone()[0]
feat = json.loads(one)
print(f'features_json keys: {len(feat)}')
print(f'sample: {list(feat.keys())[:5]}')
"
```

Expected: rows > 3M, date range 约 2020-01-02 ~ 2026-04-13, features_json keys 数 = len(NG111_STOCK_FEATURES)

- [ ] **Step 6.5：Commit cache 回填**

```bash
git add ml_models/ng/ng_cache_updater.py ml_models/ng/ng_feature_calculator.py
git commit -m "feat(ng111): ng111_feature_cache 回填 (X M rows, Y features/stock)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: 训练 ng1.1.1 模型

**目的：** 用 ng_trainer.py 训练 ng1.1.1，继承 ng1.1.0 的 P1 权重 shrinkage。

**Files:**
- Run only: `ml_models/ng/ng_trainer.py`

- [ ] **Step 7.1: 启动训练**

```bash
cd /Users/yangxu/StockTradebyZ
python3 ml_models/ng/ng_trainer.py \
    --version ng1.1.1 \
    --start-date 2020-01-01 \
    --purge-days 15 \
    2>&1 | tee logs/ng111_training_$(date +%Y%m%d_%H%M%S).log
```

Expected: 3-6 h。日志末尾：
```
✅ WF summary saved to reports/daily_selection_ng111_wf_oos/.../
✅ Model saved to ml_models/trained_models/ng/ng111_seed42_multi_target_<timestamp>.pkl
```

- [ ] **Step 7.2: 读 WF summary**

```bash
ls -la ml_models/trained_models/ng/ng111_*.pkl
cat reports/daily_selection_ng111_wf_oos/wf_summary.json 2>/dev/null || \
    find reports -name "wf_summary*.json" -newer logs/ng111_training_*.log -exec cat {} \;
```

Expected: 3 窗口 ICIR for 3d/5d/10d/15d，10d ≥ 0.93 是目标。

- [ ] **Step 7.3: Commit model + log 元数据（.pkl 本身不提交）**

确认 `.gitignore` 有 `*.pkl` / `ml_models/trained_models/`，然后：
```bash
git add logs/ng111_training_*.log reports/daily_selection_ng111_wf_oos/
git commit -m "train(ng111): WF OOS — 10d ICIR=X.XXX, 15d ICIR=Y.YYY

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Pre-2020 跨 regime 评估

**目的：** 用训练完的 ng1.1.1 模型反推 2018-2019 OOS，验证跨 regime 泛化。

**Files:**
- Run: `ml_models/ng/ng_cache_updater.py` (如需回填 2018-2019 cache)
- Run: `backtest/batch_generate_v395_reports.py` 或类似

- [ ] **Step 8.1: 回填 2018-2019 ng111 cache**

```bash
cd /Users/yangxu/StockTradebyZ
python3 fetch_data/v39_feature_cache_updater.py --start-date 2018-01-01 --end-date 2019-12-31
# 如果 ng111 需要独立回填 (分支 A 新因子), 还要跑:
# python3 ml_models/ng/ng_cache_updater.py --version ng1.1.1 --start-date 2018-01-01 --end-date 2019-12-31
```

- [ ] **Step 8.2: 生成 Pre-2020 报告**

```bash
python3 backtest/batch_generate_v395_reports.py \
    --version ng1.1.1 \
    --start-date 2018-01-01 --end-date 2019-12-31 \
    --out-dir reports/daily_selection_ng111_pre2020
```

Expected: 约 500 天的日选股报告，每天一个 json。

- [ ] **Step 8.3: 跑 Pre-2020 北极星评估**

```bash
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_ng111_pre2020 \
    --label PRE-2020 --top-n 10 --focus-days 10 --rank-field composite \
    2>&1 | tee reports/ng111/pre2020_eval.log
```

- [ ] **Step 8.4: 检查指标**

```bash
tail -80 reports/ng111/pre2020_eval.log
```

判定：V5.2 ≥ 60% A 级 = PASS。<60% 记录但不阻断。

- [ ] **Step 8.5: Commit Pre-2020 评估**

```bash
git add reports/ng111/pre2020_eval.log reports/daily_selection_ng111_pre2020/summary.json 2>/dev/null || true
git commit -m "eval(ng111): Pre-2020 OOS — V5.2=XX% <grade>

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: 生产级评估（主要判据）

**目的：** 跑 `run_north_star_eval.py --production` 看 MaxDD/Sharpe 硬目标达成。

**Files:**
- Run: `backtest/run_north_star_eval.py`
- Modify: `production_config.json`（临时切到 ng1.1.1 评估）

- [ ] **Step 9.1: 把 ng1.1.1 注册到 SCORER_REGISTRY**

Edit `tomorrow_stock_selector.py` line 63-64 (the `'ng1.1.0'` entry), 在它后面加：
```python
    'ng1.1.1': ('ml_models.ng.ng_production_scorer', 'NGProductionScorer',
                'scoring_engine_v44', 'v44_batch_cache', {'version': 'ng1.1.1'}, False),
```

并在 line 6132 左右的 `choices=[...]` 列表末尾加 `'ng1.1.1'`：
```bash
grep -n "ng1\.1\.0" /Users/yangxu/StockTradebyZ/tomorrow_stock_selector.py
# 在每个匹配处加 ng1.1.1 同级
```

- [ ] **Step 9.2: 生成 ng1.1.1 的生产日期区间报告**

```bash
cd /Users/yangxu/StockTradebyZ
python3 backtest/batch_generate_v395_reports.py \
    --version ng1.1.1 \
    --start-date 2020-01-01 --end-date 2026-04-13 \
    --out-dir reports/daily_selection_ng111
```

Expected: 报告目录 `reports/daily_selection_ng111/` 内含 1900+ 个日报告。

- [ ] **Step 9.3: 生产回测（硬目标判据）**

```bash
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_ng111 \
    --label NG111-PROD --top-n 10 --focus-days 10 --rank-field composite \
    2>&1 | tee reports/ng111/production_eval.log

# 对比 baseline: ng1.0.1 bugfix (Sharpe=2.753, MaxDD=-11.7%)
```

- [ ] **Step 9.4: 解读 MaxDD / Sharpe**

```bash
grep -E "Sharpe|MaxDD|年化|V5\.2" reports/ng111/production_eval.log | tail -20
```

对照硬目标表格（spec 5.3 决策矩阵）：
- MaxDD<-10% AND Sharpe>2.5 → PASS
- 其一达标 → 候选保留
- 双指标均退步 → 放弃

- [ ] **Step 9.5: Commit 生产评估**

```bash
git add reports/ng111/production_eval.log tomorrow_stock_selector.py
git commit -m "eval(ng111): 生产评估 — Sharpe=X.XX, MaxDD=-Y.Y%, V5.2=ZZ% <grade>

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: 决策 + Wiki/MEMORY 更新

**目的：** 根据 spec 5.3 决策矩阵做决策；无论结果如何更新 wiki 和 MEMORY。

**Files:**
- Modify: `docs/wiki/models/ng-series.md`
- Modify: `docs/wiki/models/ng-factor-quality.md`
- Modify: `docs/wiki/log.md`
- Modify: `.claude/projects/-Users-yangxu-StockTradebyZ/memory/MEMORY.md`
- Create: `.claude/projects/-Users-yangxu-StockTradebyZ/memory/ng111_factor_iteration.md`

- [ ] **Step 10.1: 写 ng1.1.1 wiki 章节**

在 `docs/wiki/models/ng-series.md` 里，定位 `## NG 1.1.0 — 已废弃` 章节，在它前面插入 ng1.1.1 新章节。模板：

```markdown
## NG 1.1.1 — EMT 四关验证 + 清理+新增 (2026-04-13)

**升级动机**: ng1.1.0 生产切换后, 发现 ng1.0.1 bugfix 反而 Sharpe=2.753 更强. 用 EMT 四关框架严格重审 ng1.0.1 base 的每个特征 + 新加候选.

**核心改动**:
- Phase 1 现役审计: DROP X 个 (详见 `reports/ng111/phase1/audit.md`)
- Phase 2 候选筛选: 14 候选中 Y 个 ACCEPT (详见 `reports/ng111/phase2/summary.md`)
- 继承 ng1.1.0 P1 权重 shrinkage (70% ICIR + 30% 等权)

**性能** (10d持仓, WF-OOS):
- V5.2: ...
- 年化(净): ...
- Sharpe: ...
- MaxDD: ...
- Pre-2020 V5.2: ...

**模型**: `ng111_seed42_multi_target_<ts>.pkl`
**缓存表**: `ng111_feature_cache`

**训练命令**:
```bash
python3 ml_models/ng/ng_trainer.py --version ng1.1.1 --start-date 2020-01-01 --purge-days 15
```

**选股命令**:
```bash
python3 tomorrow_stock_selector.py <date> --scoring-version ng1.1.1
```

**决策**: <Merge 为生产默认 | 候选保留 | 放弃>
```

- [ ] **Step 10.2: 在 ng-factor-quality.md 加 Phase 1/2 结论**

在 `ng-factor-quality.md` 开头"EMT 独立审计结果 (2026-04-12)"章节后加一节：

```markdown
### ng1.1.1 Phase 1/2 审计结论 (2026-04-13)

**Phase 1** 用放宽阈值（IC≥0.01, ICIR≥0.3, corr≥0.85）扫 ng1.0.1 bugfix 60 特征 (已减去 6 个预删)：
- DROP: X 个 (列出)
- KEEP: Y 个

**Phase 2** 14 候选因子四关严格（ACCEPT-only）：
- ACCEPT: <列出>
- REJECT: <列出> + 主要原因（IC 过低 / 冗余 / ΔIC 不足）
- 淘汰率: Z% (行业基准 95%)

候选结果全表见 `reports/ng111/phase2/summary.md`。
```

- [ ] **Step 10.3: 更新 log.md**

在 `docs/wiki/log.md` 追加：

```markdown
- **2026-04-13**: ng1.1.1 EMT 四关验证迭代 — Phase 1 DROP X / KEEP Y, Phase 2 ACCEPT Z / REJECT (14-Z). 结果: Sharpe=..., MaxDD=..., 决策 <XXX>.
```

- [ ] **Step 10.4: 写 memory 详情文件**

Create `.claude/projects/-Users-yangxu-StockTradebyZ/memory/ng111_factor_iteration.md`:

```markdown
---
name: ng1.1.1 因子迭代 (EMT 四关验证)
description: ng101 base + Phase1清理 + Phase2候选筛选, 硬目标 MaxDD<-10% & Sharpe>2.5
type: project
---

## 核心结果 (2026-04-13)
- Phase 1 DROP: [...]
- Phase 2 ACCEPT: [...]
- 最终特征数: N (vs ng1.0.1 bugfix 66, vs ng1.1.0 68)
- WF 10d ICIR: ...
- Pre-2020 V5.2: ...
- 生产 Sharpe: ..., MaxDD: -X.X%, V5.2: ...%

## 决策
<MERGE 生产 / 候选保留 / 放弃>
理由: ...

## 关键文件
- Spec: docs/superpowers/specs/2026-04-13-ng111-factor-iteration-design.md
- Plan: docs/superpowers/plans/2026-04-13-ng111-factor-iteration.md
- Phase 1: reports/ng111/phase1/audit.{csv,md}
- Phase 2: reports/ng111/phase2/{summary.md,accepted_features.json}
- 训练: ml_models/trained_models/ng/ng111_seed42_multi_target_*.pkl
- 生产评估: reports/ng111/production_eval.log

**Why:** 为了确定 EMT 四关框架能否产出超过 ng1.0.1 bugfix 的 base model
**How to apply:** 下次 NG 迭代前先跑 Phase 1 audit 校验上一版 feature list
```

- [ ] **Step 10.5: 更新 MEMORY.md 索引**

Edit `.claude/projects/-Users-yangxu-StockTradebyZ/memory/MEMORY.md`, 在最上面（"# StockTradebyZ 项目记忆"下方）加一行：

```markdown
## 🏆 ng1.1.1 EMT 四关迭代 (2026-04-13) → 详见 `ng111_factor_iteration.md`
- Phase1清理+Phase2候选筛选: X dropped/Y accepted/Z final特征; Sharpe=..., MaxDD=-X.X%, 决策<...>
```

- [ ] **Step 10.6: Commit 文档更新**

```bash
git add docs/wiki/models/ng-series.md docs/wiki/models/ng-factor-quality.md docs/wiki/log.md
git add .claude/projects/-Users-yangxu-StockTradebyZ/memory/
git commit -m "docs(ng111): wiki + memory — Phase1/2 结论 + 决策 <XXX>

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 10.7 (仅当决策=MERGE): 切生产默认**

Edit `ml_models/ng/ng_schema.py` 第 31 行：
```python
PRODUCTION_VERSION = 'ng1.1.1'
```

并确认 `fetch_data/quick_daily_update.py` 的 `update_v39_feature_cache()` / `update_ng_feature_cache()` 会覆盖 ng1.1.1 cache（可能需要加一行 version）。

Commit:
```bash
git add ml_models/ng/ng_schema.py fetch_data/quick_daily_update.py
git commit -m "prod(ng111): 切生产默认 ng1.1.0→ng1.1.1 (Sharpe=X.X, MaxDD=-Y.Y%)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## 完成标准

- [ ] `reports/ng111/phase1/audit.md` — 现役特征审计报告（60 特征全覆盖）
- [ ] `reports/ng111/phase2/summary.md` — 14 候选四关结果表
- [ ] `reports/ng111/phase2/accepted_features.json` — ACCEPT 列表
- [ ] `ml_models/trained_models/ng/ng111_seed42_multi_target_*.pkl` — ng1.1.1 模型
- [ ] `reports/daily_selection_ng111/` — 生产日期区间报告
- [ ] `reports/daily_selection_ng111_pre2020/` — Pre-2020 报告
- [ ] `reports/ng111/production_eval.log` — 生产评估 log
- [ ] `reports/ng111/pre2020_eval.log` — Pre-2020 评估 log
- [ ] `docs/wiki/models/ng-series.md` 加 ng1.1.1 章节
- [ ] MEMORY.md 索引更新
- [ ] 若达成硬目标 → `PRODUCTION_VERSION = 'ng1.1.1'`

---

## 总结时间估算

| Task | 估时 |
|------|------|
| Task 0 Sanity | 0.5 h |
| Task 1 Phase 1 audit | 1 h |
| Task 2 候选因子计算 | 2 h |
| Task 3 Phase 2 批量验证 | 2-4 h（Gate 4 LGB 主要耗时）|
| Task 4-5 Schema + feature list | 0.5 h |
| Task 6 cache 回填 | 0.5-2 h |
| Task 7 训练 | 3-6 h |
| Task 8 Pre-2020 评估 | 1 h |
| Task 9 生产评估 | 1 h |
| Task 10 文档 | 0.5 h |
| **总计** | **12-18 h** (= 2-3 工作日) |
