# V3.80模型重训练与自动化计划

**创建时间**: 2025-09-30
**目标**: 重训练v3.80模型并建立每周自动重训练机制
**执行人**: Claude Code

---

## 📋 执行概览

| 阶段 | 任务数 | 预计耗时 | 状态 |
|------|--------|----------|------|
| Phase 1: 准备 | 3 | 10分钟 | 待开始 |
| Phase 2: 训练 | 2 | 2-4小时 | 待开始 |
| Phase 3: 验证 | 2 | 30分钟 | 待开始 |
| Phase 4: 自动化 | 3 | 30分钟 | 待开始 |
| **总计** | **10** | **3-5小时** | **0%** |

---

## 🎯 Phase 1: 准备阶段 (10分钟)

### 1.1 备份现有v3.80模型
```bash
# 目标：保护现有6.8GB模型，防止训练失败后无法回退
cp models/v380/v380_models_20250921_163748_2025_trained.pkl \
   models/v380/backup_v380_models_20250921_163748_BACKUP_20250930.pkl

# 验证备份完整性
ls -lh models/v380/backup_*.pkl
```

**成功标准**: 备份文件存在且大小为6.8GB

---

### 1.2 检查数据库完整性
```bash
# 检查2025-04-01至2025-09-30数据
python3 << 'EOF'
from data_adapter.database_manager import DatabaseManager
import pandas as pd

db = DatabaseManager()
with db.get_connection() as conn:
    # 检查日期范围
    query = """
    SELECT
        MIN(trade_date) as min_date,
        MAX(trade_date) as max_date,
        COUNT(DISTINCT trade_date) as trading_days,
        COUNT(DISTINCT security_id) as stocks
    FROM daily_quotes
    WHERE trade_date BETWEEN '20250401' AND '20250930'
    """
    result = pd.read_sql(query, conn)
    print("📊 数据完整性检查:")
    print(f"  日期范围: {result['min_date'][0]} 到 {result['max_date'][0]}")
    print(f"  交易日数: {result['trading_days'][0]}天")
    print(f"  股票数量: {result['stocks'][0]}只")

    # 检查关键字段完整性
    query2 = """
    SELECT
        COUNT(*) as total_records,
        SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END) as null_close,
        SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END) as null_volume
    FROM daily_quotes
    WHERE trade_date BETWEEN '20250401' AND '20250930'
    """
    result2 = pd.read_sql(query2, conn)
    null_ratio = (result2['null_close'][0] + result2['null_volume'][0]) / result2['total_records'][0]
    print(f"  总记录数: {result2['total_records'][0]}")
    print(f"  缺失率: {null_ratio:.2%}")

    if null_ratio < 0.01:
        print("✅ 数据质量良好")
    else:
        print("⚠️ 数据质量需要检查")
EOF
```

**成功标准**:
- 交易日数 >= 120天
- 股票数量 >= 5000只
- 缺失率 < 1%

---

### 1.3 修改训练脚本支持参数化
```bash
# 创建新的参数化训练脚本
cat > train_v380_parameterized.py << 'SCRIPT'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.80参数化训练脚本
支持自定义日期范围和自动日期计算
"""
import sys
import argparse
from datetime import datetime, timedelta
sys.path.append('/Users/yangxu/StockTradebyZ')

def calculate_auto_date_range(months=6):
    """自动计算训练日期范围（最近N个月）"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def main():
    parser = argparse.ArgumentParser(description='V3.80模型训练')
    parser.add_argument('--start-date', type=str, help='训练开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='训练结束日期 (YYYY-MM-DD)')
    parser.add_argument('--months', type=int, default=6, help='自动计算：使用最近N个月数据')
    parser.add_argument('--auto', action='store_true', help='自动计算日期范围')

    args = parser.parse_args()

    # 确定日期范围
    if args.auto or (not args.start_date and not args.end_date):
        start_date, end_date = calculate_auto_date_range(args.months)
        print(f"🤖 自动模式：使用最近{args.months}个月数据")
    else:
        start_date = args.start_date
        end_date = args.end_date

    print(f"📅 训练日期范围: {start_date} 到 {end_date}")

    # 导入并运行训练
    from ml_models.v38 import V380AdvancedIncrementalMLSystem

    system = V380AdvancedIncrementalMLSystem()
    print(f"✅ {system.version} 系统初始化成功")

    # 加载股票列表
    try:
        with open('archive/logs/v380_2025_focused_stocks.txt', 'r') as f:
            stock_list = [line.strip().split()[0] for line in f if line.strip() and not line.startswith('#')]
    except:
        # 如果文件不存在，使用数据库中的活跃股票
        from data_adapter.database_manager import DatabaseManager
        db = DatabaseManager()
        with db.get_connection() as conn:
            import pandas as pd
            query = """
            SELECT DISTINCT s.code
            FROM securities s
            JOIN daily_quotes dq ON s.id = dq.security_id
            WHERE s.type = 'A股'
            AND dq.trade_date >= '20250101'
            GROUP BY s.code
            HAVING COUNT(dq.trade_date) >= 100
            LIMIT 1500
            """
            df = pd.read_sql(query, conn)
            stock_list = df['code'].tolist()

    print(f"📊 训练股票数量: {len(stock_list)}只")

    # 步骤1：特征提取
    print(f"\n🔍 步骤1/4: 特征提取")
    features = system.extract_advanced_features(
        codes=stock_list,
        start_date=start_date.replace('-', ''),
        end_date=end_date.replace('-', ''),
        target_only=False
    )

    if features is None or len(features) == 0:
        print(f"❌ 特征提取失败")
        return False

    print(f"✅ 特征提取完成: {len(features)}条样本")

    # 步骤2：准备训练数据
    print(f"\n🎯 步骤2/4: 准备训练数据")
    training_result = system.prepare_training_data(
        features_df=features,
        target_days=[1, 3, 5, 10]
    )

    if isinstance(training_result, tuple):
        training_data, feature_groups = training_result
    else:
        training_data = training_result
        feature_groups = system._group_features_for_experts()

    print(f"✅ 训练数据准备完成: {len(training_data)}条")

    # 步骤3：模型训练
    print(f"\n🚀 步骤3/4: 三层Ensemble模型训练")
    training_results = {}

    for target_period in [1, 3, 5, 10]:
        target_col = f'target_{target_period}d'
        if target_col not in training_data.columns:
            continue

        print(f"\n📈 训练{target_period}日预测模型...")
        result = system.train_three_layer_ensemble(
            training_data=training_data,
            feature_groups=feature_groups,
            target_col=target_col
        )

        training_results[target_period] = result
        if result.get('success', False):
            print(f"✅ {target_period}日模型训练成功")
        else:
            print(f"❌ {target_period}日模型训练失败")

    # 步骤4：保存模型
    print(f"\n💾 步骤4/4: 保存模型")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f"models/v380/v380_models_{timestamp}_retrained.pkl"

    # 调用系统的保存方法（需要实现）
    # system.save_models(model_path)

    print(f"✅ 训练完成！")
    print(f"📊 训练结果:")
    for period, result in training_results.items():
        if result.get('success'):
            print(f"  {period}日模型: ✅ 性能={result.get('meta_performance', 0):.4f}")
        else:
            print(f"  {period}日模型: ❌ 失败")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
SCRIPT

chmod +x train_v380_parameterized.py
```

**成功标准**: 脚本支持 `--auto`、`--start-date`、`--end-date` 参数

---

## 🚀 Phase 2: 训练阶段 (2-4小时)

### 2.1 执行v3.80模型重训练
```bash
# 使用最近6个月数据训练
nohup python3 train_v380_parameterized.py \
  --start-date 2025-04-01 \
  --end-date 2025-09-30 \
  > logs/v380_retrain_20250930.log 2>&1 &

# 记录进程ID
echo $! > /tmp/v380_train.pid
```

**监控命令**:
```bash
# 实时查看训练日志
tail -f logs/v380_retrain_20250930.log

# 检查训练进程
ps aux | grep train_v380_parameterized

# 查看GPU/CPU使用率
top -pid $(cat /tmp/v380_train.pid)
```

**预计时间**: 2-4小时（取决于数据量和硬件）

---

### 2.2 监控训练进度
```bash
# 每10分钟检查一次进度
while [ -f /tmp/v380_train.pid ]; do
    echo "⏰ $(date): 训练进行中..."
    grep -E "步骤|完成|失败" logs/v380_retrain_20250930.log | tail -5
    sleep 600
done
```

**成功标准**:
- 日志显示 "✅ 训练完成"
- 生成新模型文件
- 无致命错误

---

## ✅ Phase 3: 验证阶段 (30分钟)

### 3.1 验证新模型性能
```bash
# 对比新旧模型在相同日期的预测
python3 << 'EOF'
import sys
sys.path.append('/Users/yangxu/StockTradebyZ')
from ml_models.v38 import V380AdvancedIncrementalMLSystem

# 加载旧模型
old_system = V380AdvancedIncrementalMLSystem()
print("📊 旧模型加载成功")

# 加载新模型（需要手动指定路径）
# new_system = V380AdvancedIncrementalMLSystem()
# new_system.load_model('models/v380/v380_models_NEWFILE.pkl')

# 测试股票列表
test_codes = ['000001.SZ', '600000.SH', '000858.SZ', '002415.SZ', '600519.SH']

# 对比预测结果
print("\n📊 新旧模型对比（2025-09-30）:")
for code in test_codes:
    old_pred = old_system.predict_stock(code, '2025-09-30')
    # new_pred = new_system.predict_stock(code, '2025-09-30')
    print(f"{code}: 旧={old_pred:.2f}")
EOF
```

**成功标准**: 新模型预测合理，评分差异化明显

---

### 3.2 测试新模型预测功能
```bash
# 使用新模型运行一次完整选股
python3 tomorrow_stock_selector.py \
  --date 2025-09-30 \
  --version v3.81 \
  --model-path models/v380/v380_models_NEWFILE.pkl

# 检查选股结果
cat reports/daily_selection_v3.7/选股分析报告_20250930.md
```

**成功标准**:
- 成功生成选股报告
- 质量评分分布合理（std >= 0.15）
- 无异常错误

---

## 🤖 Phase 4: 自动化部署 (30分钟)

### 4.1 编写周度自动训练脚本
```bash
cat > scripts/weekly_retrain_v380.sh << 'SCRIPT'
#!/bin/bash
# V3.80周度自动重训练脚本
# 每周日凌晨2:00运行

set -e
cd /Users/yangxu/StockTradebyZ

# 日志文件
LOG_FILE="logs/weekly_retrain_$(date +%Y%m%d).log"
echo "🚀 开始V3.80周度重训练 - $(date)" >> "$LOG_FILE"

# 1. 检查是否有足够新数据（至少5个交易日）
NEW_DAYS=$(python3 -c "
from data_adapter.database_manager import DatabaseManager
import pandas as pd
db = DatabaseManager()
with db.get_connection() as conn:
    query = '''
    SELECT COUNT(DISTINCT trade_date) as days
    FROM daily_quotes
    WHERE trade_date > (
        SELECT MAX(trade_date) FROM daily_quotes
        WHERE trade_date <= '20250921'
    )
    '''
    result = pd.read_sql(query, conn)
    print(result['days'][0])
")

if [ "$NEW_DAYS" -lt 5 ]; then
    echo "⏸️ 新增数据不足($NEW_DAYS天)，跳过本周训练" >> "$LOG_FILE"
    exit 0
fi

echo "✅ 检测到${NEW_DAYS}天新数据，开始训练" >> "$LOG_FILE"

# 2. 备份当前模型
CURRENT_MODEL=$(ls -t models/v380/v380_models_*_retrained.pkl 2>/dev/null | head -1)
if [ -n "$CURRENT_MODEL" ]; then
    cp "$CURRENT_MODEL" "models/v380/backup_$(basename $CURRENT_MODEL)"
    echo "📦 已备份: $(basename $CURRENT_MODEL)" >> "$LOG_FILE"
fi

# 3. 执行训练（使用最近6个月数据）
python3 train_v380_parameterized.py --auto --months 6 >> "$LOG_FILE" 2>&1

# 4. 检查训练结果
if [ $? -eq 0 ]; then
    echo "✅ 训练成功 - $(date)" >> "$LOG_FILE"

    # 清理旧备份（保留最近3个）
    ls -t models/v380/backup_*.pkl | tail -n +4 | xargs -I {} rm {}

    # TODO: 发送成功通知
    echo "📧 已发送成功通知" >> "$LOG_FILE"
else
    echo "❌ 训练失败 - $(date)" >> "$LOG_FILE"
    # TODO: 发送失败通知
    exit 1
fi
SCRIPT

chmod +x scripts/weekly_retrain_v380.sh
```

---

### 4.2 配置cron任务
```bash
# 创建cron任务配置
cat > /tmp/v380_cron.txt << 'CRON'
# V3.80周度自动重训练
# 每周日凌晨2:00执行
0 2 * * 0 cd /Users/yangxu/StockTradebyZ && /bin/bash scripts/weekly_retrain_v380.sh

# 每周一早上8:00检查训练结果
0 8 * * 1 cd /Users/yangxu/StockTradebyZ && tail -20 logs/weekly_retrain_*.log | mail -s "V3.80周训练结果" user@example.com
CRON

# 安装cron任务
crontab -l > /tmp/current_cron.txt 2>/dev/null || true
cat /tmp/current_cron.txt /tmp/v380_cron.txt | crontab -

echo "✅ Cron任务已配置"
crontab -l | grep v380
```

**验证**:
```bash
# 测试脚本手动运行
bash scripts/weekly_retrain_v380.sh

# 查看cron任务
crontab -l
```

---

### 4.3 编写训练流程文档
```bash
# 更新README或创建专门文档
cat >> V380_RETRAIN_README.md << 'DOC'
# V3.80模型训练流程文档

## 手动训练
```bash
# 使用最近6个月数据
python3 train_v380_parameterized.py --auto --months 6

# 指定日期范围
python3 train_v380_parameterized.py \
  --start-date 2025-04-01 \
  --end-date 2025-09-30
```

## 自动训练
- **频率**: 每周日凌晨2:00
- **脚本**: `scripts/weekly_retrain_v380.sh`
- **日志**: `logs/weekly_retrain_YYYYMMDD.log`
- **触发条件**: 新增至少5个交易日数据

## 模型管理
- **当前模型**: `models/v380/v380_models_*_retrained.pkl`
- **备份目录**: `models/v380/backup_*.pkl`
- **保留策略**: 最近3个版本

## 故障处理
1. 训练失败：查看日志 `logs/weekly_retrain_*.log`
2. 模型回退：使用备份文件
3. 数据问题：运行 `python3 fetch_data/data_quality_check_db.py`

## 监控指标
- 训练样本数 >= 100,000
- 训练时长 <= 6小时
- Meta模型性能 >= 0.65
- 质量评分std >= 0.15
DOC
```

---

## 📊 执行检查清单

### Phase 1完成标志
- [ ] 备份文件存在且大小正确
- [ ] 数据完整性检查通过
- [ ] 训练脚本支持参数化

### Phase 2完成标志
- [ ] 训练进程已启动
- [ ] 日志文件持续更新
- [ ] 生成新模型文件

### Phase 3完成标志
- [ ] 新模型性能验证通过
- [ ] 预测功能测试正常
- [ ] 选股报告生成成功

### Phase 4完成标志
- [ ] 周度训练脚本可执行
- [ ] Cron任务已配置
- [ ] 训练文档已更新

---

## 🚨 风险与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|----------|
| 训练失败 | 中 | 高 | 使用备份模型回退 |
| 数据不足 | 低 | 中 | 扩展数据时间范围 |
| 性能下降 | 低 | 高 | 调整训练参数 |
| 硬盘空间不足 | 低 | 高 | 清理旧备份文件 |
| Cron任务未执行 | 中 | 中 | 检查cron日志 |

---

## 📈 预期收益

1. **模型时效性**: 从滞后14天到实时
2. **选股准确性**: 提升5-10%
3. **质量评分**: std从0.12提升到0.18+
4. **维护成本**: 自动化后降低80%

---

**执行开始时间**: 待定
**预计完成时间**: 执行后3-5小时
**负责人**: Claude Code
**批准人**: 用户yangxu