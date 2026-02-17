#!/bin/bash
# 监控训练进度并自动验证

echo "🔍 监控v3.9训练进度..."
echo "训练配置: 3000股票 × 97天 = ~250,000样本"
echo "预计时间: 40-70分钟"
echo ""

# 等待训练完成（最多2小时）
max_wait=7200
elapsed=0
interval=60

while [ $elapsed -lt $max_wait ]; do
    # 检查训练进程是否还在运行
    if pgrep -f "train_v390.py.*2025-06-12" > /dev/null; then
        minutes=$((elapsed / 60))
        echo "[$minutes min] 训练进行中..."
        sleep $interval
        elapsed=$((elapsed + interval))
    else
        echo ""
        echo "✅ 训练进程已结束"
        break
    fi
done

if [ $elapsed -ge $max_wait ]; then
    echo "⚠️  训练超时（2小时）"
    exit 1
fi

# 检查最新的模型文件
echo ""
echo "📦 检查模型文件..."
latest_model=$(ls -t models/v39/v390_model_*.pkl 2>/dev/null | head -1)

if [ -z "$latest_model" ]; then
    echo "❌ 未找到模型文件"
    exit 1
fi

echo "✅ 找到模型: $latest_model"
model_size=$(du -h "$latest_model" | cut -f1)
echo "   文件大小: $model_size"

# 运行验证脚本
echo ""
echo "🔍 开始验证模型..."
echo "=" * 80

python3 quick_validate_v390.py

validation_result=$?

echo ""
echo "=" * 80
if [ $validation_result -eq 0 ]; then
    echo "🎉 训练和验证全部成功！"
    echo "下一步: 集成到选股器并生成报告"
else
    echo "⚠️  验证失败，请检查日志"
fi

exit $validation_result
