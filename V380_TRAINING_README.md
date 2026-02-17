# V3.80模型训练与自动化完整指南

**文档版本**: 1.0
**创建时间**: 2025-09-30
**作者**: Claude Code
**适用范围**: V3.80机器学习模型训练与维护

---

## 📋 目录

1. [快速开始](#快速开始)
2. [手动训练](#手动训练)
3. [自动化训练](#自动化训练)
4. [模型管理](#模型管理)
5. [故障排查](#故障排查)
6. [监控与维护](#监控与维护)
7. [高级配置](#高级配置)

---

## 🚀 快速开始

### 一键训练（推荐新手）
```bash
cd /Users/yangxu/StockTradebyZ

# 使用最近6个月数据训练
python3 train_v380_parameterized.py --auto

# 查看训练日志
tail -f logs/v380_retrain_*.log
```

### 一键配置自动化
```bash
# 配置每周日凌晨2:00自动训练
bash scripts/setup_cron.sh
```

---

## 📚 手动训练

### 基本用法

#### 1. 自动模式（推荐）
使用最近N个月的数据自动训练：
```bash
# 最近6个月（默认）
python3 train_v380_parameterized.py --auto

# 最近3个月
python3 train_v380_parameterized.py --auto --months 3

# 最近12个月
python3 train_v380_parameterized.py --auto --months 12
```

#### 2. 手动指定日期
精确控制训练数据范围：
```bash
python3 train_v380_parameterized.py \
  --start-date 2025-04-01 \
  --end-date 2025-09-30
```

#### 3. 自定义输出目录
```bash
python3 train_v380_parameterized.py \
  --auto \
  --output-dir models/v380/experimental
```

### 训练流程说明

训练脚本会执行以下4个步骤：

```
步骤1: 特征提取
├─ 加载股票列表（1200-1500只A股）
├─ 从数据库提取历史行情数据
└─ 计算48维高级特征

步骤2: 准备训练数据
├─ 计算多期收益标签（1日、3日、5日、10日）
├─ 数据清洗与缺失值处理
└─ 特征分组（技术面、基本面、量价）

步骤3: 三层Ensemble模型训练
├─ Level 1: 专家模型（LightGBM、XGBoost、CatBoost）
├─ Level 2: 集成投票（加权平均）
└─ Level 3: Meta学习器（最终预测）

步骤4: 模型保存与验证
├─ 保存模型文件（.pkl格式）
├─ 输出训练统计
└─ 性能指标验证
```

### 预期输出

**训练时间**: 2-4小时（取决于数据量）
**训练样本**: 100,000-200,000条
**模型大小**: 5-8GB
**输出文件**: `models/v380/v380_models_YYYYMMDD_HHMMSS_retrained.pkl`

---

## 🤖 自动化训练

### 周度自动训练配置

#### 安装步骤

**方法1: 使用安装脚本（推荐）**
```bash
cd /Users/yangxu/StockTradebyZ
bash scripts/setup_cron.sh
```

按提示输入 `y` 确认安装。

**方法2: 手动配置**
```bash
# 编辑cron任务
crontab -e

# 添加以下内容
0 2 * * 0 cd /Users/yangxu/StockTradebyZ && /bin/bash scripts/weekly_retrain_v380.sh >> logs/cron_weekly_retrain.log 2>&1
```

#### 验证cron任务
```bash
# 查看已安装的cron任务
crontab -l

# 应该看到包含 "weekly_retrain_v380.sh" 的行
```

### 自动训练触发条件

自动训练脚本会在执行前检查以下条件：

| 条件 | 要求 | 说明 |
|------|------|------|
| **新增数据** | >= 5个交易日 | 相比上次训练日期 |
| **数据完整性** | 缺失率 < 1% | 确保数据质量 |
| **磁盘空间** | 可用 >= 20GB | 训练和备份需要 |

**如果条件不满足**：脚本会跳过本周训练，记录日志。

### 自动训练流程

```
周日凌晨2:00触发
    ↓
检查环境与数据
    ↓
备份当前模型（6.8GB）
    ↓
执行训练（2-4小时）
    ↓
验证新模型
    ↓
清理旧备份（保留3个）
    ↓
记录日志
```

### 监控自动训练

```bash
# 查看最新训练日志
tail -f logs/weekly_retrain_*.log

# 查看cron执行日志
tail -f logs/cron_weekly_retrain.log

# 检查最新模型
ls -lht models/v380/*.pkl | head -5
```

---

## 💾 模型管理

### 目录结构

```
models/v380/
├── v380_models_20250921_163748_2025_trained.pkl    # 原始训练模型
├── v380_models_20250930_143052_retrained.pkl       # 重训练模型
├── backup_v380_models_20250921_BACKUP_20250930.pkl # 手动备份
└── backups/                                         # 自动备份目录
    ├── backup_v380_models_..._20250930_020001.pkl
    ├── backup_v380_models_..._20251006_020001.pkl
    └── backup_v380_models_..._20251013_020001.pkl   # 保留最近3个
```

### 模型命名规则

**原始训练模型**:
```
v380_models_YYYYMMDD_HHMMSS_2025_trained.pkl
```

**重训练模型**:
```
v380_models_YYYYMMDD_HHMMSS_retrained.pkl
```

**备份模型**:
```
backup_v380_models_[原模型名]_[备份时间].pkl
```

### 模型切换

#### 使用新训练的模型
新模型会自动被系统加载（按修改时间排序，优先加载最新的）。

#### 回退到旧模型
```bash
# 1. 查看可用备份
ls -lht models/v380/backups/

# 2. 复制备份到主目录
cp models/v380/backups/backup_XXX.pkl \
   models/v380/v380_models_$(date +%Y%m%d_%H%M%S)_restored.pkl

# 3. 验证
python3 -c "
from ml_models.v38 import V380AdvancedIncrementalMLSystem
system = V380AdvancedIncrementalMLSystem()
print(f'当前加载模型: {system.model_path}')
"
```

### 清理策略

**自动备份**: 保留最近3个版本
**手动备份**: 永久保留（需手动删除）
**旧训练模型**: 建议每月归档一次

```bash
# 归档旧模型（每月执行）
mkdir -p archive/models/v380/$(date +%Y%m)
mv models/v380/v380_models_2025*.pkl archive/models/v380/$(date +%Y%m)/
```

---

## 🔧 故障排查

### 常见问题

#### 1. 训练失败：特征提取错误

**症状**: 日志显示 "❌ 未能提取任何特征数据"

**原因**:
- 数据库中指定日期范围无数据
- 股票代码格式不正确
- 数据库连接问题

**解决方案**:
```bash
# 检查数据库数据范围
python3 -c "
from data_adapter.database_manager import DatabaseManager
import pandas as pd
db = DatabaseManager()
with db.get_connection() as conn:
    query = 'SELECT MIN(trade_date), MAX(trade_date), COUNT(*) FROM daily_quotes'
    result = pd.read_sql(query, conn)
    print(result)
"

# 运行数据质量检查
python3 fetch_data/data_quality_check_db.py
```

#### 2. 训练失败：内存不足

**症状**: 训练中途崩溃，日志显示 "MemoryError" 或 "Killed"

**解决方案**:
```bash
# 减少训练数据量（使用3个月而非6个月）
python3 train_v380_parameterized.py --auto --months 3

# 或减少股票数量（修改train_v380_parameterized.py中的LIMIT）
```

#### 3. 自动训练未执行

**症状**: 到了周日凌晨2:00，但训练没有运行

**排查步骤**:
```bash
# 1. 检查cron任务是否安装
crontab -l | grep v380

# 2. 检查cron服务是否运行（macOS）
sudo launchctl list | grep cron

# 3. 查看系统日志
grep CRON /var/log/system.log

# 4. 手动测试脚本
bash scripts/weekly_retrain_v380.sh
```

#### 4. 模型文件损坏

**症状**: 加载模型时报错 "EOFError" 或 "UnpicklingError"

**解决方案**:
```bash
# 使用备份恢复
ls -lht models/v380/backups/ | head -5

# 复制最近的有效备份
cp models/v380/backups/backup_XXX.pkl \
   models/v380/v380_models_restored.pkl
```

---

## 📊 监控与维护

### 关键监控指标

| 指标 | 正常范围 | 告警阈值 |
|------|----------|----------|
| **训练样本数** | 100,000-200,000 | < 50,000 |
| **训练时长** | 2-4小时 | > 6小时 |
| **模型文件大小** | 5-8GB | < 1GB或 > 15GB |
| **Meta模型性能** | 0.60-0.75 | < 0.50 |
| **特征提取成功率** | > 95% | < 85% |
| **质量评分std** | 0.15-0.25 | < 0.10 |

### 定期维护任务

#### 每周（自动）
- [x] 模型重训练
- [x] 自动备份
- [x] 旧备份清理

#### 每月（手动）
```bash
# 1. 检查模型性能趋势
python3 scripts/analyze_model_performance.py --months 3

# 2. 归档旧模型
bash scripts/archive_old_models.sh

# 3. 数据库优化
python3 data_adapter/optimize_database.py

# 4. 审查训练日志
grep "ERROR\|WARN" logs/weekly_retrain_*.log | tail -50
```

#### 每季度（手动）
- [ ] 重新评估训练参数
- [ ] 更新股票列表（新上市/退市）
- [ ] 审计模型预测准确性
- [ ] 优化特征工程

### 日志文件管理

```bash
# 日志位置
logs/
├── weekly_retrain_YYYYMMDD_HHMMSS.log  # 训练日志
├── cron_weekly_retrain.log             # Cron执行日志
└── v380_retrain_YYYYMMDD.log           # 手动训练日志

# 日志清理（保留3个月）
find logs/ -name "*.log" -mtime +90 -delete
```

---

## 🔬 高级配置

### 自定义训练参数

编辑 `train_v380_parameterized.py`：

```python
# 修改股票数量限制
LIMIT 1500  # 改为你想要的数量

# 修改目标收益期
target_days=[1, 3, 5, 10]  # 可以添加20、30日

# 修改特征计算窗口
window_size = 60  # 默认60天
```

### 自定义自动训练频率

```bash
# 编辑cron任务
crontab -e

# 每周日凌晨2:00（默认）
0 2 * * 0 /path/to/weekly_retrain_v380.sh

# 改为每周三和周日
0 2 * * 0,3 /path/to/weekly_retrain_v380.sh

# 改为每天凌晨3:00
0 3 * * * /path/to/weekly_retrain_v380.sh
```

### 通知配置

在 `scripts/weekly_retrain_v380.sh` 末尾添加：

```bash
# 邮件通知
send_notification() {
    echo "$1" | mail -s "V3.80训练通知" your@email.com
}

# 在训练成功/失败时调用
send_notification "训练成功！查看: $LOG_FILE"
```

### 性能优化

**加速训练**:
```python
# 减少交叉验证折数
cv_folds = 3  # 默认5

# 使用更快的模型
models = ['lgb']  # 只用LightGBM，不用XGBoost和CatBoost
```

**降低内存占用**:
```python
# 减少特征维度
use_pca = True
n_components = 30  # 从48维降到30维
```

---

## 📞 获取帮助

### 快速参考

```bash
# 查看训练脚本帮助
python3 train_v380_parameterized.py --help

# 测试训练流程（不实际训练）
python3 train_v380_parameterized.py --dry-run  # TODO: 待实现

# 查看系统状态
python3 scripts/check_training_status.py  # TODO: 待实现
```

### 文档位置

- **主计划文档**: `V380_RETRAIN_PLAN.md`
- **训练指南**: `V380_TRAINING_README.md` (本文档)
- **系统架构**: `CLAUDE.md`

### 联系与反馈

- **Issues**: https://github.com/anthropics/claude-code/issues
- **项目目录**: `/Users/yangxu/StockTradebyZ`

---

**最后更新**: 2025-09-30
**文档版本**: 1.0
**维护者**: Claude Code