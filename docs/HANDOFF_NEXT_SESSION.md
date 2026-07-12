# 下 Session 入口 (Handoff Doc)

> 状态: 2026-07-11/12 EOS — 全仓审计 (12 agent, 25 旧项 + 93 新发现) + 集中重构, 14 commits。
> 上一版 handoff (4-28 风控路线图) 已被本次取代; 历史见 git。

---

## 🎯 本次 session 做了什么 (一句话版)

修复 20+ 潜伏 2.5 个月的生产实弹 bug, 财报数据修复 + ng101 缓存全量重建, 模型加载改固定注册表, 磁盘净释放 ~211GB, 全部测试转绿。

## ✅ 已落地的关键变化 (影响你接下来怎么干活)

1. **生产模型固定加载**: `ng_schema.PINNED_PRODUCTION_MODELS` 是唯一入口, scorer 不再 mtime-glob。
   **切生产模型 = 训完验收 → 把新 pkl 文件名写进注册表 → commit**。别的方式都不生效。
2. **生产 MOE v3** (并行 session 落地): `PRODUCTION_MOE_EXPERTS['ng1.0.6'] = bull/bear 均 ng1.0.1` = 单模 + regime 风控 overlay。回滚 = bear 改回 'ng1.0.4'。
3. **评估口径变更 (⚠️ 最重要 gotcha)**: 成本双扣 + 稀疏年化已修, **7-11 之前的北极星数字与新跑数字不可混比**。新基线: ng101 单模 81.3% S (见 `reports/system_evaluation/新口径基线重跑_20260711.md`)。
4. **财报数据修复**: financial_indicator 日期统一 TEXT (81% 不可见→100%), 补数到 2026-06-18, 每日自动 catch-up (`--include-financial` 已进 run_daily_update.sh)。备份表 `financial_indicator_backup_20260711` 确认无误后可 DROP。
5. **ng101_feature_cache 已全量重建** (2068 天 / 386 万行, 基于完整财报): roe_ttm/revenue_growth 等 99-100% 非空。**当前生产 pkl (4-12 训) 是在残缺财报缓存上训的 — 重训评估是最高优先级**。
6. **选股产物结构变化**: 全市场 JSON 恢复 ~5400 只 (曾被截到 10); overlay 最终持仓在 `analysis['risk_overlay']` (含 position_size/stop_loss); 淘汰股带 `_post_filter_drop`/`_drop_reason` 标记; 报告新增 "风控 Overlay 最终持仓" 表。下游消费按此对接。
7. **风控层失败高可见**: 任何层异常 → ERROR + 报告头 "风控降级" 戳 + `analysis['risk_layer_degraded']`。看到即排查, 不许无视。
8. **密钥**: 已迁 `.env`, config.json 无明文。**待用户: Tushare/Anthropic 控制台轮换旧 key 后更新 .env**。
9. **测试基线**: `pytest tests/` = 316 绿; `pytest stock_selctor/test/` = 115 绿; ng tests = 223 绿。
   ⚠️ 两套**不能合跑** (双 Selector 副本 conftest 冲突, 见下面裁决项)。
10. **磁盘**: DB 241→172GB (5 张 REJECT 缓存表已 DROP + VACUUM); eval_cache 91.8→1.8GB (prune CLI: `python3 backtest/eval_cache.py --prune-days N --max-gb N --apply`)。

## 🚀 下一步优先级

### ~~P0: ng1.0.1 重训评估~~ ✅ 完成 (2026-07-12)

双 gate 通过 (V5.2 83.8% S ≥ 81.3% / Pre-2020 +19.4% ≥ 0%), 新 pkl `ng101_seed42_multi_target_20260712_213343.pkl`
已写入 PINNED_PRODUCTION_MODELS。遗留: paper trade ≥20 交易日盯 10d 持仓口径 (前向 OOS 唯一弱格)。
详见 `reports/system_evaluation/ng101重训评估_20260712.md`。原任务描述如下 (存档):

### P0 (已完成): ng1.0.1 重训评估 (新缓存就绪, 3-5h)
- 按 CLAUDE.md 十项 pre-flight 走; 先 `--fast-check` 2min 判方向
- 命令基座: `python3 ml_models/ng/ng_trainer.py --start-date 2020-01-01 --purge-days 15 --seed 42`
- **接受准则 (写死)**: 新口径 V5.2 ≥ 81.3% AND Pre-2020 净年化 ≥ 0% (完整财报最可能改善 Pre-2020 泛化 — 旧模型 Pre-2020 = 45.5% B / 年化 -19%)
- ABORT 线: 第 1 个 WF 窗口 10d ICIR < 0.6
- 通过 → 新 pkl 写进 `PINNED_PRODUCTION_MODELS['ng1.0.1']`; 不过 → 保持 4-12 pkl, 结论留档

### P1: Selector.py 双副本合并裁决 (调查已完成, 见 refactor plan 附录)
- 两副本是**语义不同的活策略变体** (根副本带知行闸门被 tests+configs.json 锁定; stock_selctor 版被生产锁定), 机械合并会改一方行为 — shim 模拟实测 296 passed / 20 failed
- **推荐选项 1**: 根副本类改名 Strict 版并入 stock_selctor, 根变 shim 保留旧名映射 — 零行为变化, 两套测试可合跑
- compute_kdj 无论如何以 ss 版 (ewm, 与 DB 口径一致) 为准

### P2: 剩余结构重构 (各需专门 session)
- tomorrow_stock_selector.py 拆分 (6400 行, ~1200 行死分支可先删)
- 三套 DB 管理器统一; v39/v40/ng 三代 cache updater 去重
- webapp 安全 (debug=True/0.0.0.0/CORS 全开/无鉴权) + 功能腐化
- 完整清单: `docs/code_review_and_refactor_plan.md` 第六节

## 🛠️ Sanity checks (session 开场跑)

```bash
# 1. 生产选股 (期望: 全市场 ~5400 只 + risk_overlay 10 只 + 无 风控降级 戳)
python3 tomorrow_stock_selector.py <最近交易日> --scoring-version ng1.0.6
# 2. 模型加载 (期望: "使用固定生产模型 ng101_seed42_multi_target_20260412_233749.pkl")
python3 -c "from ml_models.ng.ng_production_scorer import NGProductionScorer; NGProductionScorer(version='ng1.0.1')"
# 3. 测试 (分开跑, 期望 316 + 115 全绿)
python3 -m pytest tests/ -q; python3 -m pytest stock_selctor/test/ -q
# 4. 财报新鲜度 (期望 max ann_date 随中报季推进)
python3 -c "import sqlite3; c=sqlite3.connect('data_adapter/stock_data.db'); print(c.execute('SELECT MAX(ann_date), COUNT(*) FROM financial_indicator').fetchone())"
```

## 📁 产物索引

| 文件 | 内容 |
|---|---|
| `docs/code_review_and_refactor_plan.md` 第六节 | 审计结论 + 8 commit 记录 + 遗留 roadmap + Selector 裁决选项 |
| `docs/wiki/lessons/known-pitfalls.md` (末尾 4 条) | Timestamp 绑定 / 截断类 bug / mtime-glob / SQLite 类型排序 |
| `reports/system_evaluation/新口径基线重跑_20260711.md` | 新口径基线 (ng101 81.3% S) + 生产切 v3 依据 |
| `scripts/migrate_financial_indicator_date_format.py` | 日期迁移脚本 (已执行, 留作参考) |
| `logs/ng101_cache_rebuild_20260711.log` | 缓存重建全日志 (2068 天) |
| memory `refactor_audit_2026_07_11.md` | 本次 session 沉淀 |

## 💬 给下 session 的开场提示词

```
项目: StockTradebyZ。上 session (7-11) 完成全仓审计+重构 (14 commits): 修 20+ 生产 bug、
财报数据修复、ng101 缓存已基于完整财报全量重建、模型改 PINNED 注册表加载、生产已切
ng1.0.1 单模 v3。状态详见 docs/HANDOFF_NEXT_SESSION.md。

请先跑 sanity checks, 然后做 P0: ng1.0.1 重训评估 (fast-check → 全量, 接受准则
V5.2 ≥ 81.3% + Pre-2020 净年化 ≥ 0%, ABORT 线首窗 ICIR < 0.6)。通过就把新 pkl
写进 PINNED_PRODUCTION_MODELS 完成切换。
```
