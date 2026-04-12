# Signal Trust — 选股信号可信度系统

**建立日期**: 2026-04-12
**状态**: 生产中

## 目的

基于每只股票的历史 "预测 vs 实际" 统计, 给每日选股贴可信度标签, 识别庄家释放的假信号; 周度输出市值/行业/流动性分组的模型失效诊断.

## 标签语义

- 🟢 可信: 方向命中≥55% 且 系统偏差≥-2% 且 兑现率≥40%
- 🔴 高风险: 任一严重指标(方向<45% 或 偏差<-3% 或 兑现率<20%)
- 🟡 存疑: 介于两者间
- ⚪ 数据不足: 全历史样本 <10 次

## 数据流

1. **样本池**: `pred_10d > 0.01` (温和看多以上) 的历史记录
2. **跨版本去重**: 同 (code, date) 在多个版本报告出现时, 按 `VERSION_PRIORITY` 取最新
3. **实际收益**: T+10 交易日的 close / T 日 close - 1
4. **可信度分数**: 每日 `update_signal_trust_daily.py` 刷新, SQL 过滤 `sample_end_date < today` 防泄露

## 已知边界

- 停牌 ≥10 日的样本 `actual_10d=NULL` 永不回填, 对应股票样本偏少
- 全历史累计, 不做时间衰减 (早期市场环境不同会有噪音)
- 今日新入库样本要到 T+10 日才参与可信度计算 (正确行为, 非 bug)

## 命令速查

```bash
# 首次建库(一次性)
python3 scripts/rebuild_signal_trust.py

# 每日增量(集成在 run_daily_update.sh)
python3 scripts/update_signal_trust_daily.py

# 周度全局统计(建议周日晚跑)
python3 scripts/weekly_signal_trust_stats.py

# 健康检查
python3 scripts/validate_signal_trust.py
```

## 关键文件

- `signal_trust/` — Python 包
- `signal_trust_samples` / `signal_trust_scores` 两张 SQLite 表
- Design spec: `docs/superpowers/specs/2026-04-12-signal-trust-design.md`
- Implementation plan: `docs/superpowers/plans/2026-04-12-signal-trust.md`
