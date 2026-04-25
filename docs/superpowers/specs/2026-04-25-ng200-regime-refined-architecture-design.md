# NG v2.0 Design Spec: Multi-Beta Regime Classifier + Per-Regime Alpha Sub-Models

**Date**: 2026-04-25
**Status**: Draft → Pending User Review
**Base**: ng1.0.6 v2 (V11 0AMV regime + ng1.0.7 bull / ng104-3s bear, V5.2=79% on 2024-2026)
**Goal (ng2.0a)**: regime classifier 升级到 multi-beta 投票, 端到端 V5.2 ≥ 78%, MaxDD ≤ -22.9%, Top-10 与 ng106 v2 重合度 ≥ 50%
**Goal (ng2.0b)**: per-regime sub-model 重训, V5.2 ≥ 基线 +1pp, MaxDD ≤ -18% (改善 ≥ 4pp)
**Codename**: "MBRA" (Multi-Beta Regime + Alpha)

---

## 0. 核心洞察 (设计前提)

ng1.0.6 v2 已实证两件事, ng2.0 在此基础上做精细化:

### 已锁定的事实

**F1. 外部 regime switch 是真 alpha 来源 (memory ng106_amv_regime)**
ng1.0.6 v1 (V3 strict + ng101+ng104-3s) WF-OOS V5.2=78.9%, β_UMD=+0.005 (ng1.0.1 的 1/76), 是全 NG 系列**唯一** Pre-2020 年化正 (+0.7%) 的版本。机制: 牛市用 ng101 (动量+), 熊市用 ng104-3s (动量-), 净 β 几乎为零。

**F2. ng1.0.6 v2 sub-expert swap 是当前最强 (memory ng106_v2_moe_experiments)**
- 冠军组合 = V11 + ng1.0.7 bull + ng104-3s bear
- V5.2 = 79% on 2024-2026 (vs v1 78%)
- 单模型路线已彻底闭合, MOE 子专家是唯一留下的提升通道

**F3. Regime 内化进 ML 必失败 (memory ng14x / ng150)**
- ng1.4.0 (base + 4 downside + 3 AMV 特征): Stage 4a V5.2=68%, 比 ng101 76% 差
- ng1.5.0 (Tier B regime-refined 5 特征): V5.2=63% A, β_UMD=+1.42 爆炸, Sharpe 0.89 腰斩
- ng1.0.7 conditional label + AMV: Pre-2020 V5.2=34.7% C
- 共同点: regime 信号一旦进 ML 模型 (特征 / label / loss), 模型就在历史 regime 上过拟合, 跨 regime 全败

**F4. ng1.0.6 短板 = MaxDD (-22.9%, ng1.0.1 两倍)**
ng1.0.6 v1 唯一劣势是 MaxDD, 需叠加 ng1.0.5 三层风控或重新设计 sub-model. ng2.0b 选择后者 (训练阶段就让 bear model 学控回撤)。

**F5. V11 0AMV 单信号已击败 V3 strict +5pp (commit b2a93580)**
但 V11 仍是单信号, **位置 + 动量是同一维度**, 缺 "市场内部健康度" + "风险偏好" 维度。

### ng2.0 的设计前提 (从 F1-F5 推出)

- **不内化 regime 进 ML** (F3) → regime 必须是外部 switch, sub-model 严格只用 alpha 因子
- **多维信号合成** (F5) → 加 B1 breadth + B2 realized vol, 互补 V11 的位置/动量维度
- **bear model 训练阶段控 MDD** (F4) → ng2.0b sample-weighted bear 训练, bear regime 样本 ×2

---

## 1. 方向候选对比

| 方向 | 描述 | 违反的洞察 | 风险 | 潜在收益 |
|---|---|---|---|---|
| A. 单一信号继续优化 V11 | 调 V11 streak / threshold | F5 (维度不足) | 低 | 低 |
| B. 加 ML 训练 regime classifier | 用 logistic / 小 lightgbm 学 regime label | F3 | 高 (regime 过拟合) | 中 |
| C. 3-state regime (牛/震荡/熊) | ng1.0.6 v2 已测试 -8pp (memory) | F1 二元已对 | 高 | 负 |
| D. 加 alt-data 信号 (北向/转债) | B3/B4 候选 | — | 中 (数据稀疏/衰减) | 中 |
| **E. Multi-beta hard-vote (V11 + B1 breadth + B2 vol)** | 三信号投票 + 系统级 streak | 最少 | **低** | **高** |
| F. 全栈 ng2.0 一次重写 | regime + bull + bear 同时换 | — | 极高 (变量太多) | 高 |

### 选中: 方向 E (ng2.0a) + sample-weighted sub-model retrain (ng2.0b)

**为什么选 E**:
- B1 (内部健康度) + B2 (风险偏好) 是和 V11 (位置/动量) **互补维度**, 不是同一维度叠加
- 硬投票 = 完全可解释, 无超参训练, 不踩 F3 红线
- 数据全在 DB (daily_quotes), 零 fetcher 工作量
- 与 ng106 v2 head-to-head 公平: 只动 regime classifier 一个变量, 提升来源 100% 可定位

**为什么不选 B/C/D/F**: 见上表.

---

## 2. 架构

```
┌────────────────────────────────────────────────────────────┐
│ Layer 1: Multi-beta Regime Classifier (ng2.0a)             │
│                                                            │
│   V11 (0AMV 位置+MACD, 3d streak)  ──┐                    │
│   B1  (% stocks > MA20/MA60, 3d)    ├─→ majority vote ──┐ │
│   B2  (60d 沪深300 RV state, 3d)    ──┘  (2-of-3)       │ │
│                                                          ↓ │
│                                          system 3d streak  │
│                                                          ↓ │
│                                              regime_v2     │
│                                              (bull/bear)   │
└────────────────────────────────────────────────────────────┘
                              ↓ regime label
┌────────────────────────────────────────────────────────────┐
│ Layer 2: Per-regime Alpha Sub-models                       │
│                                                            │
│  ng2.0a Step B (now):                                      │
│    bull → ml_models/ng/ng_production_scorer (ng1.0.7)      │
│    bear → ml_models/ng/ng_production_scorer (ng104-3s)     │
│                                                            │
│  ng2.0b (后续, sample-weighted retrain):                   │
│    bull → 新训 ng2.0b-bull (sample weight 2x bull)         │
│    bear → 新训 ng2.0b-bear (sample weight 2x bear)         │
└────────────────────────────────────────────────────────────┘
                              ↓ daily scores
┌────────────────────────────────────────────────────────────┐
│ Layer 3: Daily Routing & Selection                         │
│   backtest/regime_switch_backtest.py (extend)              │
│   tomorrow_stock_selector.py (--scoring-version ng2.0)     │
└────────────────────────────────────────────────────────────┘
```

### 严格分工

- **Layer 1 (regime)**: 只用市场级 beta 信号, 严禁个股 alpha 因子混入
- **Layer 2 (alpha)**: 只用个股 alpha 因子 (ng101_feature_cache 现有 66+ 特征), 严禁 regime / 市场广度 / 波动率特征混入
- **Layer 3 (routing)**: 纯路由, 不做信号合成

---

## 3. 组件清单

### 3.1 新增文件

| 文件 | 行数估计 | 作用 |
|---|---|---|
| `indicators/breadth.py` | ~150 | B1 信号: 计算每日 % stocks > MA20 / % > MA60 / advance-decline ratio, 三者加权后阈值化, 含 3d streak |
| `indicators/realized_vol.py` | ~120 | B2 信号: 沪深300 60d realized vol, vol percentile (滚动 252d), 高 vol percentile (>70%) → bear, 低 vol (<30%) → bull, 中间 → 沿用前值 (sticky), 含 3d streak |
| `scripts/build_regime_v2_history.py` | ~80 | 一次性脚本, 回算 2018-2026 全段 regime_v2 写 DB |
| `scripts/compare_regime_v1_v2.py` | ~150 | Step A 验证: flip count + 状态分布 + 重合矩阵 + 2018-2019 sanity check |

### 3.2 扩展文件

| 文件 | 改动 |
|---|---|
| `indicators/regime_classifier.py` | 新增 `compute_regime_v2(v11_series, b1_series, b2_series, system_streak=3) → regime_v2 (-1=bear, +1=bull)` |
| `backtest/regime_switch_backtest.py` | `--regime-version {v1, v2}` (default=v1 暂不切默认), v2 走新 routing |
| `ml_models/ng/ng_trainer.py` | (ng2.0b only) `--sample-weight-mode regime --regime-weight-multiplier {1.5, 2.0, 3.0}`, 读 market_regime_signals 表给训练样本打权重 |
| `ml_models/ng/ng_schema.py` | 新增 `'ng2.0a'` 版本号 (复用 ng101 feature cache, 仅 regime layer 不同) + `'ng2.0b'` (新 pkl, 同 schema) |
| `tomorrow_stock_selector.py` | 支持 `--scoring-version ng2.0` (实际走 ng106 v2 sub-model + ng2.0a regime) |
| `fetch_data/quick_daily_update.py` | 新增步 7c: 调 indicators 计算 B1/B2 + regime_v2, 写 market_regime_signals 表 |

### 3.3 数据库 schema

新增表 `market_regime_signals`:

```sql
CREATE TABLE IF NOT EXISTS market_regime_signals (
    trade_date TEXT PRIMARY KEY,
    -- V11 子信号
    v11_var1 REAL,           -- 0AMV var1
    v11_ma60 REAL,           -- 0AMV ma60
    v11_macd REAL,           -- 0AMV MACD
    v11_bull INTEGER,        -- 0/1
    v11_streak INTEGER,
    -- B1 子信号
    b1_pct_above_ma20 REAL,
    b1_pct_above_ma60 REAL,
    b1_adv_dec_ratio REAL,
    b1_score REAL,           -- 加权合成 [0,1]
    b1_bull INTEGER,
    b1_streak INTEGER,
    -- B2 子信号
    b2_rv_60d REAL,
    b2_rv_percentile_252 REAL,
    b2_bull INTEGER,         -- 低 vol → bull
    b2_streak INTEGER,
    -- 系统层
    vote_count INTEGER,      -- bull 投票数 (0-3)
    regime_v2_raw INTEGER,   -- 投票直出 (-1/+1)
    regime_v2_streak INTEGER,
    regime_v2 INTEGER,       -- 含 3d streak 后的最终值 (-1 bear / +1 bull)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_mrs_regime_v2 ON market_regime_signals(regime_v2);
```

无破坏性 schema 变更, 新表独立。ng101_feature_cache / ng_feature_cache 等不动。

---

## 4. 数据流

### 4.1 每日推断 (production)

```
quick_daily_update.py 已有步骤
    └─ 步 7c (NEW)
        ├─ load_market_amount + load_market_circ_mv  (现有)
        ├─ compute V11 (现有 indicators/market_amv.py)
        ├─ compute B1 (NEW indicators/breadth.py)
        │     - 取最近 60 个交易日 daily_quotes
        │     - 每日算 % stocks 收盘价 > 自身 MA20, % > MA60
        │     - 取沪深 A 股全样本 (排除 ETF / 指数)
        │     - 加权 score = 0.5 × pct_ma20 + 0.5 × pct_ma60
        │     - threshold: score > 0.55 → b1_bull=1, < 0.45 → b1_bull=0
        │     - 0.45-0.55 hysteresis: 沿用前值
        │     - 3d streak: 连续 3 日 b1_bull=1 才确认 bull
        ├─ compute B2 (NEW indicators/realized_vol.py)
        │     - 取沪深 300 (000300.SH) 最近 312 个交易日 daily_quotes
        │     - 60d realized vol = std(log_returns, 60d)
        │     - 252d percentile rank of 60d RV
        │     - vol_pct < 30% → b2_bull=1 (低波时期)
        │     - vol_pct > 70% → b2_bull=0 (高波 = 风险)
        │     - 30-70% hysteresis: 沿用前值
        │     - 3d streak
        ├─ vote = v11_bull + b1_bull + b2_bull
        ├─ regime_v2_raw = +1 if vote ≥ 2 else -1
        ├─ system 3d streak on regime_v2_raw → regime_v2
        └─ INSERT INTO market_regime_signals
```

### 4.2 选股 (production)

```
tomorrow_stock_selector.py --scoring-version ng2.0 --date 2026-04-25
    ├─ SELECT regime_v2 FROM market_regime_signals WHERE trade_date='2026-04-25'
    ├─ if regime_v2=+1: scorer = NGProductionScorer(version='ng1.0.7')
    │  else:            scorer = NGProductionScorer(version='ng1.0.4-3s')
    └─ scorer.predict_scores(...)
```

### 4.3 回测

```
backtest/regime_switch_backtest.py --regime-version v2 --start 2020-01-01 --end 2026-04-25
    ├─ 每日读 regime_v2 (从 market_regime_signals 表)
    ├─ 路由到对应 sub-model 的 daily report
    └─ 走原有 backtest_report_based 链路
```

---

## 5. ng2.0b sub-model 训练设计 (后续阶段)

ng2.0a Step B 通过后才启动 ng2.0b。

### 5.1 训练数据

- 训练区间: 2020-01-01 ~ 2026-04-25, expanding WF, purge=15 天
- Pre-2020 OOS: 2018-01-01 ~ 2019-12-31 (向后泛化 gate, 沿用现有方法学)
- 不抓 2015-2017 数据 (outlier regime, fetcher 工作量)

### 5.2 sample weight 机制

```python
# 在 ng_trainer.py 内
def compute_sample_weights(df, target_regime: str, multiplier: float):
    """
    df: 训练样本, 含 trade_date 列
    target_regime: 'bull' | 'bear'
    multiplier: 目标 regime 日的权重倍数 (e.g. 2.0)
    """
    regime_lookup = read_market_regime_signals()  # date → regime_v2 (+1/-1)
    target_val = +1 if target_regime == 'bull' else -1
    weights = np.where(
        df['trade_date'].map(regime_lookup) == target_val,
        multiplier,  # 目标 regime
        1.0          # 非目标 regime
    )
    return weights
```

LightGBM / CatBoost / XGBoost / RF 全部支持 sample_weight 参数, 直接传入。

### 5.3 训练命令

```bash
# bull model (ng2.0b-bull)
python3 ml_models/ng/ng_trainer.py \
    --version ng2.0b-bull \
    --start-date 2020-01-01 \
    --end-date 2026-04-25 \
    --purge-days 15 \
    --sample-weight-mode regime \
    --regime-weight-multiplier 2.0 \
    --regime-target bull \
    --target-parallel 4 \
    --seed 42

# bear model (ng2.0b-bear)
python3 ml_models/ng/ng_trainer.py \
    --version ng2.0b-bear \
    --start-date 2020-01-01 \
    --end-date 2026-04-25 \
    --purge-days 15 \
    --sample-weight-mode regime \
    --regime-weight-multiplier 2.0 \
    --regime-target bear \
    --target-parallel 4 \
    --seed 42
```

### 5.4 ablation matrix

multiplier ∈ {1.5, 2.0, 3.0} × {bull, bear} = 6 组训练。
取 per-regime Sharpe 最优组合作为 ng2.0b 生产配置。

---

## 6. 验证 / 接受准则

### 6.1 ng2.0a Step A: 仅 regime classifier (无 sub-model 端到端)

**验证脚本**: `scripts/compare_regime_v1_v2.py`

| 指标 | 基线 (V11) | PASS 门槛 | ABORT 门槛 |
|---|---|---|---|
| flip count (2020-2026) | V11 实测值 (待 measure) | ≤ V11 ×1.0 | > V11 ×1.5 |
| 状态分布 (% bull / % bear) | V11 实测 | 偏离 V11 < 10pp | 偏离 > 25pp |
| 2018-2019 sanity check | V3 strict 实测 | 在 2018Q4 + 2019Q1 能识别熊→牛切换 | 完全无切换 / 反向切换 |
| 重合矩阵 (V11 vs ng2.0a) | — | 至少 70% 日子一致 | < 50% 一致 |

**Step A 早杀**: paper-trade 第一周 (2026-04-25 ~ 2026-05-02) flip > 2 次 → kill, 回头加 hysteresis。

### 6.2 ng2.0a Step B: 端到端 (用现成 ng1.0.7 / ng104-3s sub-model)

**验证**: `python3 backtest/regime_switch_backtest.py --regime-version v2` + `python3 backtest/run_north_star_eval.py`

基线 = ng1.0.6 v2 (V11 + ng1.0.7 + ng104-3s) WF-OOS 2020-2026 V5.2=79%, MaxDD=-22.9%, Sharpe=2.808.

| 指标 | 基线 | PASS 门槛 | ABORT 门槛 |
|---|---|---|---|
| V5.2 (10d composite) | 79% A+ | ≥ 78% (持平) | < 75% |
| 10d Sharpe | 2.808 | ≥ 2.7 | < 2.4 |
| MaxDD | -22.9% | ≤ -22.9% (持平或更小) | > -28% |
| Pre-2020 净年化 | ng106 v2 待实测 | ≥ 0% (同向 alpha) | < -5% |
| Pre-2020 V5.2 | 待实测 | ≥ 40% (B 级 + 同向) | < 35% |
| **Top-10 与 ng106 v2 重合度** | 100% (相同 regime 时) | 全段平均 ≥ 50% | < 30% (列表飘了) |

### 6.3 ng2.0b sub-model retrain gates

基线 = ng2.0a Step B 实测值。

| 指标 | 基线 | PASS 门槛 | ABORT 门槛 |
|---|---|---|---|
| V5.2 (10d) | ng2.0a Step B | ≥ 基线 +1pp | < 基线 -2pp |
| MaxDD | ng2.0a Step B | ≤ -18% (改善 ≥ 4pp) | > 基线 |
| bear regime 内 Sharpe (在 bear 日上单独算) | ng104-3s 实测 | ≥ ×1.1 | < ×0.9 |
| bull regime 内 Sharpe | ng1.0.7 实测 | ≥ ×1.0 | < ×0.9 |
| β_UMD (整体) | ng106 v1 +0.005 | \|β\| ≤ 0.5, t-stat \|t\| < 3 | \|β\| > 1.5 |

**ng2.0b 早杀**: 第 1 个 WF 窗口的 bear model 10d ICIR < 0.2 → kill (memory ng124/ng150 教训, 不补救).

---

## 7. Pre-flight Checklist 映射 (CLAUDE.md Check 1-10)

| Check | 应用到 ng2.0 |
|---|---|
| 1. Schema 一致性 | ng2.0a 只加 `market_regime_signals` 表, 不动 feature schema. ng2.0b 复用 ng101 schema (无新特征). 验证: `PRAGMA table_info(market_regime_signals)` 列与 indicators 输出一致 |
| 2. Backfill 逻辑 | `scripts/build_regime_v2_history.py` 跑 2018-2026 全段, 验证非空且 v11/b1/b2 各列均有值 |
| 3. 高效路径 | regime 是规则计算, 秒级; ng2.0b 训练沿用 `--target-parallel 4` |
| 4. 接受准则 | 第 6 节写死, 含 ABORT 早杀线 |
| 5. Baseline 公平 | 钉死 ng106 v2 (V11+ng107+ng104-3s); ng2.0b 训练对齐 ng106 v2 的 expanding WF + purge=15 + seed=42 |
| 6. 保命 | Step A 秒级无需 caffeinate; ng2.0b 训练用 `caffeinate -i` + `tee logs/train_ng20b_*_$(date).log` |
| 7. 数据泄露 | B1 用 t 日收盘 vs t-N..t-1 MA, 隔日生效 (next-day open 可执行); B2 同; regime_v2 用 t 日产出, t+1 才生效 (避免 same-day look-ahead). purge=15 cover 10d label |
| 8. 资源预算 | regime 计算 < 1GB 内存. ng2.0b 训练 ~20GB RAM (沿用 ng1.0.x). 训练前 `df -h` ≥ 20GB |
| 9. 元数据 | ng2.0b pkl 写入 git_commit_hash + schema_version + seed + sample_weight_mode + multiplier + regime_target |
| 10. /simplify | 新写的 `breadth.py` / `realized_vol.py` / `compute_regime_v2()` / sample_weight 改动后, 跑 3 轮 /simplify 再 kickoff |

---

## 8. 阶段计划 (Plan-level, 实际 plan 由 writing-plans skill 产出)

### ng2.0a (本期, 1-2 周)

1. **Step A** (1-3 天): 实现 B1/B2 信号 + 投票合成 + 历史回算 + Step A 验证脚本 + flip count / 状态分布对比
2. **Step B** (1 周): 接 backtest_report_based 端到端跑 2020-2026, 跑 Pre-2020 OOS, 对比 ng106 v2

### ng2.0b (后续, 仅当 ng2.0a Step B PASS, 2-3 周)

1. trainer 加 sample_weight_mode (1 天 + /simplify)
2. fast-check (2 min) 验证 sample_weight 路径正确, IC 方向正常
3. ablation matrix 训练: 6 组 (multiplier × regime), per-regime Sharpe 选最优 (~3 天 × 2 ≈ 1 周)
4. WF-OOS + Pre-2020 OOS 全量评估
5. 通过 ng2.0b gates 后切生产; 否则维持 ng106 v2

### ng2.0c (留坑, 不在本 spec 范围)

非对称 loss 实验 (LinEx for bear, LambdaRank top-heavy for bull). spec 后写。

---

## 9. YAGNI 砍除 (用户已确认)

- ❌ 抓 2015-2017 数据 (outlier regime + fetcher 工作量, ng2.0b bear 不够再启)
- ❌ asymmetric loss (LinEx / LambdaRank) → 留 ng2.0c
- ❌ 显式 factor residualization vs size/value/momentum → industry excess label 已做大部分 de-beta
- ❌ B3 北向 / B4 信用利差 / B5 dispersion / B6 量价背离 → ng2.0a-bis 后续迭代
- ❌ 3-state regime (牛/震荡/熊) → memory ng106_v2 已实证 -8pp
- ❌ ML 训练 regime classifier → F3 红线
- ❌ 全栈 ng2.0 一次重写 → 变量太多, 不可定位

---

## 10. 风险登记

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| B1/B2 与 V11 高相关, 投票退化为 V11 单一票 | 中 | 中 | Step A 实测三信号相关性矩阵, > 0.7 重新选 B1/B2 阈值 |
| B2 vol percentile 在 2020 疫情极端段失效 | 低 | 中 | 252d rolling percentile 自适应, 不会卡死 |
| Pre-2020 OOS 退化 (失去 ng106 v1 的 +0.7%) | 中 | 高 | gate 写死 net annual ≥ 0%, 否则 reject |
| ng2.0b bear 样本不足过拟合 | 中 | 高 | sample_weight 而非 hard filter; ablation 含 multiplier=1.5 保守组; ICIR 早杀 |
| Top-10 列表与 ng106 v2 重合 < 30% | 中 | 高 | 这是新 gate, 防"数字好看列表飘了"; 不达标即 reject |
| `market_regime_signals` 表与 daily_update 链路冲突 | 低 | 低 | 新表独立, 失败不影响其他表 |

---

## 11. Open questions (写入 plan 阶段再填)

- B1 加权系数: 0.5 × pct_ma20 + 0.5 × pct_ma60 是经验起点, 可在 Step A ablation 微调到 (0.4, 0.6) 或 (0.6, 0.4)
- B2 percentile 阈值 (30/70): 经验起点, Step A 看 V11 disagreement 段表现再调
- ng2.0b 取 ensemble 方式: bull/bear 各自 4-model ensemble (LGB+XGB+CB+RF) 沿用 ng_trainer 现状, 不引入新模型类
- regime_v2 边界日 (regime 切换当日) 的处理: 用 t 日产出 t+1 应用 (next-day open), 切换日按前一 regime 持仓不动, 切换次日才换 sub-model

---

## 12. 命名 / 版本号

- `ng2.0a` = multi-beta regime + 现成 sub-model
- `ng2.0b` = ng2.0a + sample-weighted retrain bull/bear
- `ng2.0c` (后续) = ng2.0b + asymmetric loss
- DB 表: `market_regime_signals` (regime 层独立, 不绑版本号)
- pkl 文件: `ng20b_bull_seed42_*.pkl` / `ng20b_bear_seed42_*.pkl`

切生产命令最终形态:
```bash
python3 tomorrow_stock_selector.py 2026-04-25 --scoring-version ng2.0a   # Step B 通过后
python3 tomorrow_stock_selector.py 2026-04-25 --scoring-version ng2.0b   # ng2.0b 通过后
```

---

## 13. 终止条件 (本 spec 失效)

任一条触发后, 立即停止 ng2.0 路线, 回到 ng106 v2 维稳:

1. ng2.0a Step A 在两次 hysteresis 调参后 flip 仍 > V11 ×1.5
2. ng2.0a Step B V5.2 < 75% (3pp 降级 = 实质回退)
3. ng2.0a Step B Pre-2020 净年化 < -5% (失去 ng106 v1 的核心优势)
4. ng2.0b ablation 全部 6 组 V5.2 均不达 +1pp 增量

---

**End of spec.**
