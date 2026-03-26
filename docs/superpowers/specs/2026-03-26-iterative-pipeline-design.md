# 分级验证迭代管线设计

**日期**: 2026-03-26
**状态**: Draft
**目标**: 将模型迭代反馈循环从 5-6 小时压缩到 5-7 分钟（快筛+快评），同时保留全量验证能力

## 问题

当前模型迭代流程：改参数 → 训练(5-6h) → 批量报告(4-8min) → 北极星评估(1-2min)，端到端 5-6 小时。一天最多跑 1-2 个变体，试错成本太高。

## 解决方案：四级验证管线

### 总体架构

```
              iterative_pipeline.py (统一入口)

  输入: params.json (训练参数/特征列表/模型配置)
  模式: --level L1|L2|L3|L4  或  --auto-gate (逐级升级)

  ┌──────┐   gate   ┌──────┐   gate   ┌──────┐   gate   ┌──────┐
  │  L1  │ ──────→  │  L2  │ ──────→  │  L3  │ ──────→  │  L4  │
  │快筛  │  IC>阈值  │快评  │ miniNS>  │确认  │  NS>阈值  │生产  │
  │3-5min│         │+2min │  阈值    │1-2h  │         │5-6h  │
  └──────┘         └──────┘         └──────┘         └──────┘
       │               │               │               │
       ▼               ▼               ▼               ▼
   ic_summary     mini_ns_score    ns_score_200d   ns_score_full

  ┌─────────────────────────────────────────────────────────┐
  │  iteration_comparison.tsv — 所有变体横向对比              │
  └─────────────────────────────────────────────────────────┘
```

**设计原则：**

1. **单一入口** — `scripts/iterative_pipeline.py` 统管所有级别
2. **参数驱动** — 每个变体是一个 params.json，方便批量生成和对比
3. **自动门控** — `--auto-gate` 模式下，不达标自动停止，达标自动升级
4. **结果累积** — 所有变体的结果追加到同一个对比表
5. **零重复** — L1 模型直接给 L2 复用；复用现有训练/评分/评估代码，不造新轮子

---

## L1 快筛（3-5 分钟）

**目标**：用最小训练成本判断一组参数/特征是否值得深入

### 训练配置裁剪

| 参数 | Full 训练 | L1 快筛 | 加速比 |
|------|----------|---------|--------|
| 数据窗口 | 5年 (2020-2026) | 2年 (2024-2026) | ~2.5x |
| 模型数 | 5 (LGB+XGB+CB+RF+HGB) | 1 (LGB only) | ~5x |
| boost rounds | 500 | 150 | ~3x |
| 目标数 | 3-4 (3d/5d/10d/15d) | 2 (5d/10d) | ~1.5x |
| Walk-Forward | 4-6折 | 单折 (70/15/15) | ~5x |
| **综合** | **5-6h** | **3-5 min** | **~80x** |

### 单折数据切分

```
2024-01 ──────────────── 2025-08 ── 2025-11 ── 2026-02
         train (70%)        val(15%)   test(15%)
                            ← 10d purge gap →
```

- 训练集：~400 天 x ~5000 股 = ~200 万样本
- LGB 150 轮在 200 万样本上约 30-60 秒
- 加上特征加载和归一化，总计 3-5 分钟

### L1 输出

```python
l1_result = {
    "variant_name": "v488_add_brain_factors",
    "level": "L1",
    "duration_sec": 210,
    "metrics": {
        "test_ic_5d": 0.067,
        "test_ic_10d": 0.072,
        "test_icir_5d": 0.81,
        "test_icir_10d": 0.89,
        "val_ic_10d": 0.075,
        "train_val_gap": 0.008,  # train IC - val IC
        "n_features": 52,
        "top10_feature_importance": ["..."],
    },
    "gate_pass": True,
    "model_path": "/tmp/l1_v488_lgb.pkl"
}
```

### L1 门控条件

```python
L1_GATE = {
    "test_ic_10d": ">= 0.04",      # 最低信号要求
    "test_icir_10d": ">= 0.40",    # 最低稳定性
    "train_val_gap": "<= 0.05",    # 过拟合检查
}
# 三条全过才进 L2
```

阈值偏宽松——L1 的目的是快速淘汰明显差的变体，不是精确评估。

---

## L2 快评（+2 分钟）

**目标**：用 L1 模型生成迷你报告 + 迷你北极星，拿到 NS 预估分

### 流程

```
L1 模型 (LGB only, /tmp/l1_xxx.pkl)
    │
    ▼
batch_generate (复用现有核心函数)
    │  最近 60 个交易日的报告 (~2s/天 = 120s)
    │  输出到临时目录 /tmp/l2_reports_xxx/
    ▼
run_north_star_eval (复用现有核心函数)
    │  对 60 天报告跑 V2/V3 评分 (~25s)
    ▼
mini_ns_score + 校准修正
```

### 为什么 60 天

| 报告天数 | 报告耗时 | NS评估耗时 | 统计可靠性 |
|---------|---------|-----------|-----------|
| 30 天 | 60s | 15s | 太少，月度胜率不稳定 |
| **60 天** | **120s** | **25s** | **够算 IC/ICIR/月度胜率(3个月)** |
| 100 天 | 200s | 40s | 更好，但多花 2 分钟 |

### 校准修正

迷你 NS 和全量 NS 之间有系统偏差（短窗口波动大、回测长度折扣等）。

```python
# 首次使用：直接用 mini_ns 排序（变体间相对排名有效）
# 累积 5+ 组 (mini_ns, full_ns) 配对后：拟合线性校准
calibrated_ns = slope * mini_ns + intercept

# 校准数据自动保存到 scripts/l2_calibration.json
{
    "pairs": [
        {"variant": "v475", "mini_ns": 62, "full_ns": 77},
        {"variant": "v473", "mini_ns": 58, "full_ns": 75}
    ],
    "slope": 1.12,
    "intercept": -5.3,
    "r_squared": 0.87
}
```

### L2 输出

```python
l2_result = {
    "variant_name": "v488_add_brain_factors",
    "level": "L2",
    "duration_sec": 145,
    "mini_ns_raw": 58,
    "mini_ns_calibrated": 62,    # 有校准数据时
    "mini_ns_grade": "A",
    "metrics": {
        "ic_10d": 0.078,
        "icir_10d": 0.92,
        "annual_return_gross": 0.45,
        "max_drawdown": -0.18,
        "sharpe": 1.05,
        "monthly_win_rate": 0.67,
    },
    "gate_pass": True,
    "reports_dir": "/tmp/l2_reports_v488/"
}
```

### L2 门控条件

```python
L2_GATE = {
    "mini_ns_raw": ">= 40",           # 至少 B 级水平
    "ic_10d": ">= 0.05",              # 信号质量下限
    "max_drawdown": ">= -0.30",       # 回撤不能太离谱
}
```

### L1+L2 合计

```
L1 训练:    3-5 min
L2 报告:    ~2 min
L2 评估:    ~0.5 min
─────────────────
合计:       5-7 min 拿到 NS 预估分
```

一个 session 能跑 5-8 个变体，全部有 NS 预估分横向对比。

---

## L3 确认（1-2 小时）

**目标**：用接近生产的配置验证最优变体，拿到可信 NS 分数

### 配置

| 参数 | L4 生产 | L3 确认 | 节省 |
|------|--------|---------|------|
| Walk-Forward | 4-6 折 | 3 折 | ~40% |
| 模型数 | 5 | 3 (LGB+XGB+CB) | ~40% |
| boost rounds | 500 | 300 | ~40% |
| 数据窗口 | 5年 | 3年 | ~40% |
| 报告天数 | 500+ | 200 天 | ~60% |

### 耗时

```
L3 训练:     ~60-90 min
L3 报告:     ~7 min (200天)
L3 北极星:   ~30s
合计:        1-2h
```

### L3 门控条件

```python
L3_GATE = {
    "ns_score": ">= 60",  # 至少 A 级才值得跑全量
}
```

---

## L4 生产（5-6 小时）

和现有全量训练完全一致，不做改动。只有 L3 通过的最优变体才跑 L4。

---

## 自动化与 CLI

### 命令行接口

```bash
# 单级别
python3 scripts/iterative_pipeline.py --level L1 --params params_v488.json

# 自动升级（最常用）
python3 scripts/iterative_pipeline.py --auto-gate --params params_v488.json

# 批量对比 + 自动选优
python3 scripts/iterative_pipeline.py --batch \
    --params params_v488a.json params_v488b.json params_v488c.json \
    --promote-top 2   # L2 后取最好的 2 个进 L3
```

### 参数文件格式

`scripts/params/params_v488a.json`：

```json
{
    "variant_name": "v488a_brain_factors",
    "base_version": "v395",
    "features": {
        "add": ["brain_roll_spread", "brain_momentum"],
        "remove": ["old_factor_x"]
    },
    "training": {
        "start_date": "2020-01-01",
        "num_boost_round": 500,
        "num_leaves": 31,
        "min_data_in_leaf": 200,
        "sharpe_blend": 0.3,
        "purge_days": 15
    },
    "scoring": {
        "rank_field": "pred_10d",
        "top_n": 10,
        "focus_days": 10
    }
}
```

### 对比表

所有结果自动追加到 `scripts/iteration_comparison.tsv`：

```
variant          level  duration  ic_10d  icir_10d  mini_ns  ns_200d  ns_full  grade
v488a_brain      L2     6m12s     0.078   0.92      62       -        -        A(est)
v488b_no_brain   L2     5m48s     0.051   0.61      44       -        -        B(est)
v488c_pruned     L3     1h22m     0.081   0.95      65       71       -        A
v488a_brain      L4     5h30m     0.083   0.98      62       73       77       A+
```

---

## 文件结构

```
scripts/
├── iterative_pipeline.py          # 统一入口（新建）
├── iteration_comparison.tsv       # 对比表（自动生成）
├── l2_calibration.json           # L2 校准数据（自动积累）
└── params/                       # 参数变体目录
    ├── params_v488a.json
    └── ...
```

### 代码复用（不造新轮子）

`iterative_pipeline.py` 只做编排，调用已有模块：

| 功能 | 复用来源 |
|------|---------|
| 训练 | `ml_models/training/train_v395_multi_target.py` 核心函数 |
| 批量报告 | `backtest/batch_generate_v395_reports.py` 的 `score_all_stocks_from_preloaded()` |
| 北极星评估 | `backtest/north_star_metrics.py` + `backtest/backtest_report_based.py` |

需要的改造：
- 训练脚本需暴露可被 import 调用的函数（目前可能是 `__main__` 一把梭）
- 批量报告生成需支持传入已加载的 scorer 对象（避免重复加载模型文件）
- 评估脚本需支持函数式调用并返回结构化结果（目前可能是 print 输出）

---

## 不做的事

- 不改现有训练逻辑/模型结构
- 不引入 GPU 加速（ROI 不高，当前单机 CPU 够用）
- 不建 NS 代理回归模型（用 L2 迷你北极星代替，更可信）
- L4 不做任何裁剪，保持和现有生产流程一致
