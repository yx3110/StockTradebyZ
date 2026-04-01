# StockTradebyZ 全面代码审查 & 重构/性能提升计划

> 审查时间: 2026-03-31  
> 审查工具: OpenAI Codex (gpt-5.3-codex) + Claude Opus 4.6  
> 审查范围: 全仓库 217+ 模块

---

## 一、审查发现汇总

### 🔴 P0 - 严重问题（影响数据正确性/安全性）

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 1 | **明文 API 密钥落盘** | `config.json:9,14` | tushare/anthropic 密钥直接在仓库中 |
| 2 | **推理 fallback 未来数据泄漏** | `v390_production_scorer.py:652,1089` | 缓存 miss 时 SQL 未限制 `trade_date <= 预测日`，回测结果被污染 |
| 3 | **V4.4 T+1 数据过滤** | `v44_production_scorer.py:297,644,987` | `_apply_executability_filters` 读取下一交易日真实涨跌幅改分，属于未来函数 |
| 4 | **训练阶段全量统计泄漏** | `train_v500_unified.py:277,395`; `train_v400_cross_sectional.py:1220` | winsorize/IC筛选/特征选择在 split 前全量执行，验证集指标虚高 |
| 5 | **SQL 注入面** | `data_access.py:98,100,238` | 动态字段和 LIMIT 直接字符串拼接 |
| 6 | **Web 配置弱安全** | `webapp/config.py:106,107,153` | 默认 DEBUG=True、开发密钥回退、硬编码绝对路径 |

### 🟠 P1 - 高危问题（影响模型/系统可靠性）

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 7 | **训练-推理特征偏移 (train-serve skew)** | `v490_production_scorer.py:220,264`; `train_market_gate_v2.py:198,257` | 门控特征推理端常量化，与训练端 rolling 统计不一致 |
| 8 | **特征命名不一致** | `v390_enhanced_feature_ml_system.py:240` | 缺失特征用 `fund_feature_i` 占位，与训练真实列名不符 |
| 9 | **时序任务用随机切分** | `train_v390_simple.py:80`; `train_v394_ensemble.py:469` 等 | `shuffle=True` + 默认 KFold，时序泄漏 |
| 10 | **模型加载靠 mtime 选最新** | `v487_production_scorer.py:58,130`; `v44/v46_production_scorer.py` | 部署不可复现，schema 不匹配时无 fail-fast |
| 11 | **data_adapter 导入会失败** | `data_access.py:13`; `squeeze_momentum_updater.py:22` 等 | `import data_adapter.data_access` 触发 `ModuleNotFoundError` |
| 12 | **外键约束未持续启用** | `database_manager.py:48,63` | SQLite pragma 是连接级，新连接不会自动启用 |
| 13 | **三套 DB 管理器并存** | `database_manager.py` / `db_manager.py` / `webapp/core/database.py` | 事务语义、WAL、timeout 策略不一致 |

### 🟡 P2 - 中危问题（影响维护效率/性能）

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 14 | **主文件 6171 行** | `tomorrow_stock_selector.py` | 职责混杂：模型装配/策略/报告/交易日判断 |
| 15 | **Scorer 版本间大量重复** | `v44_production_scorer.py` 多处 copy-paste | 修一处漏一处 |
| 16 | **异常吞噬** | `tomorrow_stock_selector.py:1306,1315`; `webapp/core/database.py:140,345` | bare `except` 掩盖故障 |
| 17 | **日志初始化分散** | 4+ 模块各自 `basicConfig` | 全局日志行为不可预测 |
| 18 | **配置管理未统一** | 多处绕过 `core/config.py` 直接读 `config.json` | 修改配置逻辑需改多处 |
| 19 | **根目录文件堆积** | `Selector.py`(旧副本) + 5个 `batch_*.py` + 20+ 脚本 | 难以区分哪个是权威版本 |
| 20 | **无连接池化** | `database_manager.py:63` | 每次新建连接，无 WAL/timeout 统一策略 |
| 21 | **pd.concat() 在循环中** | `backtest_report_based.py:1592` | 二次复杂度内存分配 |
| 22 | **8+ 版本缓存并存** | `tomorrow_stock_selector.py:__init__` | 缓存一致性风险，内存浪费 |
| 23 | **目录名拼写错误** | `stock_selctor/` (少了 'e') | 长期技术债 |
| 24 | **训练可复现性不足** | 多脚本 seed 不统一，DB 连接无 context manager | 结果不稳定 |
| 25 | **`__init__.py` 导出不完整** | `ml_models/v39/__init__.py` | v488/v490/v491/v492 scorer 未注册 |

---

## 二、重构计划

### Phase 1: 安全加固 (预计 1 天)

**目标**: 消除所有安全隐患

| Task | 具体操作 | 文件 |
|------|----------|------|
| 1.1 密钥迁移 | 从 `config.json` 移除明文密钥，改用 `.env` + `python-dotenv`，仅保留 `.env.example` | `config.json`, `core/config.py`, `.env.example` |
| 1.2 SQL 注入修复 | `data_access.py` 的 fields 做白名单映射，top_n 强制 `int()` | `data_adapter/data_access.py` |
| 1.3 Web 安全加固 | 生产环境强制 SECRET_KEY、禁用 DEBUG、移除硬编码路径 | `webapp/config.py` |
| 1.4 Pickle 反序列化 | 模型加载前增加 checksum 校验 | 所有 `*_production_scorer.py` |

### Phase 2: 数据泄漏修复 (预计 2-3 天)

**目标**: 消除训练/推理中的未来信息泄漏

| Task | 具体操作 | 文件 |
|------|----------|------|
| 2.1 Fallback 时间限制 | `_fallback_score` 所有 SQL 加 `WHERE trade_date <= ?` | `v390_production_scorer.py` |
| 2.2 T+1 路径拆分 | 拆为 `online_score(T)` + `offline_eval(T+1)`，线上路径禁止未来数据 | `v44_production_scorer.py` |
| 2.3 训练管线修复 | split 先于 winsorize/IC筛选，变换器只在 train 上 fit | `train_v500_unified.py`, `train_v400_cross_sectional.py` |
| 2.4 时序切分统一 | 全部改用 `TimeSeriesSplit` / PurgedGroupTimeSeriesSplit | `train_v390_simple.py`, `train_v394_ensemble.py` 等 |
| 2.5 泄漏回归测试 | 新增自动化测试：验证训练/推理不使用未来日期数据 | `tests/test_no_lookahead.py` (新建) |

### Phase 3: 架构重构 (预计 3-5 天)

**目标**: 解决代码重复、职责混杂、可维护性差

#### 3.1 拆分 `tomorrow_stock_selector.py` (6171行 → 4个模块)
```
tomorrow_stock_selector.py (6171行)
  ├── scoring/orchestrator.py        # 评分调度 (~1500行)
  ├── scoring/model_adapter.py       # 模型版本适配 (~1000行)
  ├── reporting/report_writer.py     # 报告生成 (~1500行)
  ├── utils/trading_calendar.py      # 交易日判断 (~200行)
  └── tomorrow_stock_selector.py     # 主入口/CLI (~800行)
```

#### 3.2 统一数据库管理器 (3套 → 1套)
```
data_adapter/
  ├── db_core.py           # 统一连接层 (WAL + timeout + 连接池 + 事务)
  ├── database_manager.py  # 高级接口 (继承 db_core)
  └── data_access.py       # 查询层 (白名单字段 + 参数化)

删除/废弃:
  - db_manager.py (根目录)
  - webapp/core/database.py → 改为 import db_core
```

#### 3.3 Scorer 版本管理重构
```
ml_models/v39/
  ├── base_scorer.py              # 抽象基类 (模板方法模式)
  │   ├── load_model()           # 统一加载 + checksum + schema校验
  │   ├── compute_features()     # 训练/推理共用特征管线
  │   └── predict()              # 预测 + 后处理
  ├── scorer_registry.py          # 版本工厂 + manifest
  ├── v390_scorer.py              # 仅差异化逻辑
  ├── v44_scorer.py
  ├── v490_scorer.py
  └── v492_scorer.py
```

#### 3.4 导入与包结构修复
- `data_adapter/` 内全部改为相对导入 (`from .database_manager import ...`)
- `ml_models/v39/__init__.py` 注册所有活跃版本
- 删除根目录旧 `Selector.py`，权威实现在 `stock_selctor/`

### Phase 4: 性能优化 (预计 2-3 天)

**目标**: 消除性能热点，提升日常运行效率

| Task | 具体操作 | 预期提升 |
|------|----------|----------|
| 4.1 DB 连接池化 | 实现线程本地连接复用 + WAL + busy_timeout=30s | 减少连接开销 30%+ |
| 4.2 消除循环 concat | `backtest_report_based.py:1592` 改为 list append + 单次 concat | 大回测加速 2-5x |
| 4.3 向量化 lambda | `tomorrow_stock_selector.py` 中 `.apply(lambda)` 改为向量化操作 | 选股加速 20-50% |
| 4.4 缓存精简 | 仅保留生产版本缓存 (v492)，其余按需加载 | 内存减少 50%+ |
| 4.5 批量 DB 查询 | 合并多次单股查询为单次批量查询 | I/O 减少 80%+ |
| 4.6 特征计算缓存 | 训练/推理共用 canonical 特征管线，结果缓存到 `v39_feature_cache` | 消除 train-serve skew + 加速推理 |

### Phase 5: 工程治理 (预计 1-2 天)

**目标**: 提升长期可维护性

| Task | 具体操作 |
|------|----------|
| 5.1 日志统一 | 仅入口脚本 `basicConfig`，库模块只用 `getLogger(__name__)` |
| 5.2 配置统一 | 所有模块通过 `core.config` 读配置，支持环境变量覆盖 |
| 5.3 异常处理 | 消除所有 bare `except`，至少 `except Exception as e` + 日志 |
| 5.4 根目录清理 | 旧脚本移入 `archive/` 或 `scripts/`，根目录只保留入口 |
| 5.5 目录重命名 | `stock_selctor/` → `stock_selector/` (修正拼写) + 全局替换导入 |
| 5.6 模型 manifest | 每个模型文件配套 `.manifest.json` (版本/checksum/schema/训练日期) |
| 5.7 seed 统一 | 所有训练脚本统一 `GLOBAL_SEED=42`，DB 连接全部用 context manager |

---

## 三、优先级路线图

```
Week 1:  Phase 1 (安全) + Phase 2 (泄漏修复)
         ↓ 验证: 回测指标是否因泄漏修复而下降 (预期会降，但更真实)
         
Week 2:  Phase 3.1 (拆分主文件) + Phase 3.2 (统一DB)
         ↓ 验证: 所有现有功能不回归 (python3 tomorrow_stock_selector.py 正常)

Week 3:  Phase 3.3 (Scorer重构) + Phase 3.4 (导入修复)
         ↓ 验证: 所有评分版本输出一致

Week 4:  Phase 4 (性能) + Phase 5 (工程治理)
         ↓ 验证: 日常流程端到端 benchmark
```

---

## 四、关键指标

| 指标 | 当前 | 目标 |
|------|------|------|
| `tomorrow_stock_selector.py` 行数 | 6,171 | < 1,000 (主入口) |
| DB 管理器数量 | 3 套 | 1 套 |
| Scorer 重复代码率 | ~60% | < 15% |
| bare `except` 数量 | 10+ | 0 |
| 明文密钥 | 2 处 | 0 |
| 未来数据泄漏点 | 4 处 | 0 |
| 训练可复现性 (相同 seed 结果一致) | ❌ | ✅ |
| 根目录 .py 文件数 | 20+ | < 5 |

---

## 五、风险与注意事项

1. **Phase 2 泄漏修复后回测指标会下降** — 这是正确的，说明之前的指标被虚高了
2. **Phase 3 重构需要全面的回归测试** — 建议先建立 baseline 输出快照
3. **`stock_selctor` 重命名影响面大** — 需要全局搜索替换，建议单独一个 commit
4. **生产环境切换窗口** — 重构期间保留旧路径兼容，确认无误后再清理
