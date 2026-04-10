# 已知陷阱

项目历史中踩过的坑，避免重复犯错。按类别组织，每个陷阱记录：现象、根因、解决方案。

---

## 数据类

### 股票代码必须带交易所后缀
- **现象**: 查询/匹配失败，数据关联错误
- **根因**: A股代码不唯一，需要 `.SZ`/`.SH` 区分
- **解决**: 全系统统一使用 `000001.SZ`、`600519.SH` 格式

### price_change_pct 单位不一致
- **现象**: 波动率计算异常，旧数据和新数据结果差异巨大
- **根因**: 旧数据(2025年前)是百分比格式(2.16=2.16%)，新数据(2026年)是小数格式(0.012=1.2%)
- **解决**: 波动率从 close 价格直接计算，避免依赖 price_change_pct 字段

### 行业数据来源
- **现象**: 用 `stock_basic_info.industry` 查不到行业或数据不全
- **根因**: 行业分类在 `securities` 表的 `industry` 字段，不在 `stock_basic_info`
- **解决**: 统一从 `securities` 表读取行业信息

### Tushare API 分页查询
- **现象**: 指数成分股查询返回不全或超时
- **根因**: 部分 API 有单次返回行数限制
- **解决**: 使用 `pro.index_member_all()` 做分页查询，不要用 per-industry 调用

---

## 数据库类

### SQLite 并发死锁
- **现象**: 多进程同时写数据库时报 `database is locked`
- **根因**: SQLite 默认 busy_timeout 太短
- **解决**: 设置 `busy_timeout=30000`（至少30秒）

### 多进程 Pool pipe 死锁
- **现象**: `multiprocessing.Pool` 传递大数据时卡死不返回
- **根因**: Pool 通过 OS pipe 传递 `computer_data`（5700个 DataFrame），超出 pipe buffer 64KB
- **解决**: `backfill_v39_complete.py` 添加 `--num-workers 0` 顺序模式；推荐使用 `fetch_data/v39_feature_cache_updater.py` 代替（每天1.8秒 vs 数分钟）

### 数据库恢复流程
- **现象**: auto-claude 自动任务做 git merge/checkout 时删除未 commit 的文件
- **根因**: 数据库文件太大无法 git commit，被 .gitignore 忽略
- **解决**: 所有核心代码已 commit；数据库需从备份恢复
- **备份路径**: `~/StockTradebyZ_backup_20260216/data_adapter/stock_data.db.backup_20251103_231057`

---

## 模型训练类

### 数据泄露 — --production 回测
- **现象**: --production 回测显示 S 级评分，但实际 OOS 表现远低于预期
- **根因**: 训练数据和回测数据有重叠，评分虚高
- **解决**: 必须使用 WF-OOS + Pre-2020 双向无泄露评估；`--production` 仅供内部参考
- **参考**: [回测方法论](../evaluation/backtesting.md)

### 四分位小盘加权扭曲学习
- **现象**: V4.6 重训练后 WF IC 近随机（W3 10d=0.002），年化仅 39.5%
- **根因**: bottom 25% 市值股票样本权重 ×2.5，过度加权噪声样本，损害预测能力
- **解决**: 保留旧模型 `v46_multi_target_20260301_050112.pkl` 为生产版；移除极端小盘加权

### Composite 排名不总是最优
- **现象**: 多周期 composite 评分有时反而低于单周期 pred_10d
- **根因**: 特征裁剪可能提升某个预测周期但降低其他周期，混合后稀释最优信号
- **解决**: 不能假设 composite 总是好的，需要实测对比
  - V4.7.3: composite=75 > pred_10d=63（composite 有益）
  - V4.7.5: pred_10d=77 > composite=69（composite 有害）

### Meta-Learner + Combined Isotonic 压缩头部区分度
- **现象**: V4.7.4/V4.8 评分退化
- **根因**: Meta-Learner + Combined Isotonic 校准把头部预测值压缩到窄区间，top-10 严重同分
- **解决**: V4.7.3 去掉这两层，直接用裸信号；V4.7.5 进一步裁剪特征

### 全局百分位离散化导致同分
- **现象**: `_to_global_score()` 输出 0-100 整数，平均 67 只股票同分竞争 top-10
- **根因**: 离散化丢失细粒度排名信息
- **解决**: 使用 `rank_field=composite` 或 `rank_field=pred_10d` 连续值排名

### 隐性动量暴露
- **现象**: V4.9.0.1 WF-OOS 仅 B 级 54.1%，MaxDD=-21.5%
- **根因**: β_UMD=3.029，模型学到了动量因子但没有显式控制
- **解决**: V5.0 诚实重建，使用因子残差标签 + Rank-Transform 消除动量暴露

---

## 回测类

### 涨停检测方法
- **现象**: `is_limit_up` 字段不可靠
- **根因**: 数据源的 `is_limit_up` 标记不准确
- **解决**: 用 `price_change_pct` 判断：主板≥9.5%、创/科≥19.5%、北交≥29.5%

### 交易成本低估
- **现象**: 回测收益显著高于实盘
- **根因**: 初始只算了 0.15% 单边手续费
- **解决**: 总成本 0.302%（含双边滑点 0.1% + 过户费）；CPPI 调仓 exposure 变化 >1% 时扣减

---

## 工程类

### .pkl/.joblib 模型文件不能 git add
- **现象**: git repo 膨胀，push 失败
- **根因**: 训练好的模型文件动辄几十 MB~几百 MB
- **解决**: 确保在 `.gitignore` 中排除

### 性能优化前先 profile
- **现象**: 凭直觉优化错误的瓶颈
- **根因**: 数据管线的瓶颈往往不在直觉判断的位置
- **解决**: `python -m cProfile` 先定位，再优化

### Fast-check 再完整训练
- **现象**: 花 5-19 小时完整训练后发现方向错误
- **根因**: 没有快速验证就投入全量训练
- **解决**: `--fast-check` 模式（2个WF窗口，~2min）先验证 IC/ICIR 方向，通过后再完整训练

### 条件化标签/市场regime特征过拟合 (2026-04-10, ng1.0.7)
- **现象**: ng1.0.7训练期内信号质量L1=91.3%(全系列最高), 但Pre-2020=C级(年化-36%)
- **根因**: 条件化标签(熊市用rank_pct blend)和8个扩展市场特征(0AMV连续值等)过拟合了2020+的市场regime模式。2018-2019的regime分布与2020+显著不同，模型学到的"熊市偏好"在旧数据上不适用
- **教训**: 
  1. WF-OOS好(A+)不代表模型泛化好——Pre-2020双向评估是检验过拟合的金标准
  2. 市场regime特征本质上是非平稳的，GBDT容易记住训练期regime模式
  3. 交叉特征(stock×market)全被IC筛选淘汰，说明GBDT自身已能学到交叉效应
  4. ng1.0.1(纯行业超额标签+69特征)反而是跨regime泛化最强的——简单模型胜过复杂模型
- **解决**: 不使用regime-conditional标签；如需regime感知，用后置风控(ng1.0.5)而非改变模型标签

### RF权重失衡导致行业集中 (2026-04-08, ng1.0.4)
- **现象**: Top-10几乎全是银行股
- **根因**: ICIR优化后RandomForest权重94-95%，RF天然偏好低波动→银行垄断
- **解决**: 限制单模型在ensemble中最大权重；或加行业分散约束
