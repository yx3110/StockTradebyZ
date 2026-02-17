# 🚀 快速开始指南

## 立即启动

### 方法1: 一键启动（推荐）

```bash
cd /Users/yangxu/StockTradebyZ/webapp
./start.sh
```

### 方法2: 手动启动

```bash
cd /Users/yangxu/StockTradebyZ/webapp

# 安装依赖
pip3 install -r requirements.txt

# 启动应用
python3 app.py
```

### 访问应用

在浏览器打开: **http://127.0.0.1:5000**

---

## 📱 功能导航

### 1. 首页 (/)
- 查看系统概览
- 系统统计信息（股票总数、最新数据日期、数据库大小）
- 快速操作入口

### 2. 日常任务 (/daily-tasks)
**数据更新:**
- 点击"开始更新"按钮
- 实时查看更新进度（市场行情、基本面、技术指标、指数数据）
- 等待更新完成（约30-45秒）

**运行选股:**
- 选择ML版本（v3.9推荐）
- 点击"运行选股"按钮
- 查看选股结果

### 3. 模型训练 (/model-training)
**查看模型:**
- 查看所有模型版本（v3.7/v3.8/v3.81/v3.9）
- 查看模型文件和训练状态

**训练模型:**
- 选择模型版本
- 配置训练参数（日期范围）
- 点击"开始训练"（约30-60分钟）

### 4. 回测 (/backtest)
**运行回测:**
- 选择ML版本
- 设置日期范围
- 点击"开始回测"

**查看结果:**
- 查看历史回测报告
- 分析性能指标

---

## 🔧 常用API端点

### 健康检查
```bash
curl http://127.0.0.1:5000/api/health
```

### 获取系统状态
```bash
curl http://127.0.0.1:5000/api/daily/status
```

### 获取模型列表
```bash
curl http://127.0.0.1:5000/api/models/
```

### 获取可用策略
```bash
curl http://127.0.0.1:5000/api/backtest/strategies
```

---

## 💡 使用技巧

### 1. 每日工作流程
```
早上 9:00
1. 运行数据更新
2. 等待更新完成
3. 运行选股（v3.9版本）
4. 查看选股结果
5. 根据结果制定交易计划
```

### 2. 周末分析
```
周末
1. 回顾本周选股效果
2. 运行回测分析不同策略
3. 必要时重新训练模型
4. 优化参数配置
```

### 3. 模型更新
```
每月或季度
1. 收集新数据
2. 重新训练模型
3. 对比新旧模型性能
4. 选择最优模型投入使用
```

---

## 🐛 故障排查

### 问题1: 无法启动应用
**解决方案:**
```bash
# 检查Python版本
python3 --version  # 需要3.8+

# 重新安装依赖
pip3 install -r requirements.txt

# 检查端口占用
lsof -i :5000
```

### 问题2: 数据库连接失败
**解决方案:**
```bash
# 检查数据库文件是否存在
ls -lh /Users/yangxu/StockTradebyZ/stock_data.db

# 检查文件权限
chmod 644 /Users/yangxu/StockTradebyZ/stock_data.db
```

### 问题3: 任务执行失败
**解决方案:**
1. 查看任务错误信息
2. 检查对应Python脚本是否存在
3. 查看webapp/logs/webapp.log日志文件
4. 确保项目根目录脚本可执行

---

## 📞 获取帮助

- 查看详细文档: `webapp/README.md`
- 查看设计文档: `webapp/DESIGN_PLAN.md`
- 检查日志: `webapp/logs/webapp.log`

---

**祝您使用愉快！📈**
