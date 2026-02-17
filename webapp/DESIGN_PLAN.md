# StockTradebyZ Web可视化系统 - 详细设计计划书

## 📋 项目概述

### 目标
创建一个本地Web应用，提供实时可视化界面来监控和管理整个量化交易系统，包括日常任务、模型训练和回测功能。

### 核心价值
- **实时监控**：直观展示系统运行状态和进度
- **数据可视化**：图表化展示选股结果、模型指标、回测报告
- **操作便捷**：通过Web界面触发任务，无需命令行
- **历史追溯**：查看历史选股、训练、回测记录

---

## 🏗️ 系统架构设计

### 整体架构
```
┌─────────────────────────────────────────────────────────┐
│                    浏览器 (Browser)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │日常任务  │  │模型训练  │  │  回测    │              │
│  └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────┘
                        ↕ HTTP/WebSocket
┌─────────────────────────────────────────────────────────┐
│              Flask Web Server (后端)                     │
│  ┌────────────────────────────────────────────────────┐ │
│  │ REST API (JSON)        SSE (实时进度推送)          │ │
│  ├────────────────────────────────────────────────────┤ │
│  │ 任务管理器 (TaskManager)                           │ │
│  │  - 后台线程池                                      │ │
│  │  - 任务队列                                        │ │
│  │  - 进度追踪                                        │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                        ↕
┌─────────────────────────────────────────────────────────┐
│                 数据层 (Data Layer)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ SQLite   │  │Markdown  │  │  Python  │              │
│  │  数据库   │  │  报告    │  │  脚本    │              │
│  └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────┘
```

### 技术栈选型

#### 后端技术
| 技术 | 选择 | 理由 |
|------|------|------|
| **Web框架** | Flask 3.0+ | • 轻量级，易于集成现有Python代码<br>• 丰富的扩展生态<br>• 适合中小型应用<br>• 学习曲线平缓 |
| **任务执行** | Threading + Queue | • 无需额外依赖<br>• 适合本地单机部署<br>• 进程内通信简单 |
| **实时通信** | Server-Sent Events | • 单向推送，适合进度监控<br>• HTTP协议，无需WebSocket<br>• 浏览器原生支持 |
| **数据库** | 现有SQLite | • 已有数据，直接查询<br>• 无需迁移<br>• 零配置 |

#### 前端技术
| 技术 | 选择 | 理由 |
|------|------|------|
| **UI框架** | Bootstrap 5.3 | • 响应式布局<br>• 丰富的组件库<br>• 无需构建工具 |
| **图表库** | ApexCharts 3.x | • 现代化交互式图表<br>• 支持实时更新<br>• 中文文档 |
| **表格库** | DataTables 1.13 | • 强大的数据表格<br>• 排序、搜索、分页<br>• Excel导出 |
| **HTTP客户端** | Fetch API | • 浏览器原生<br>• Promise支持<br>• 现代化 |

---

## 📦 模块详细设计

### 模块1：日常任务 (Daily Tasks)

#### 功能需求
1. **数据更新监控**
   - 显示最后更新时间
   - 展示更新状态（成功/失败/进行中）
   - 实时进度条（市场行情、基本面、技术指标、指数数据）
   - 错误日志展示
   - 手动触发更新按钮

2. **选股结果展示**
   - 按日期查看历史选股
   - 支持所有版本（v3.0/v3.7/v3.8/v3.81/v3.9）
   - 表格展示选中股票及评分
   - 技术指标小图表（K线缩略图）
   - 导出功能（CSV/Excel）

3. **AI增强分析**
   - 显示AI分析报告
   - 市场分析概要
   - 风险评估结果
   - 查看完整Markdown报告

#### 数据源
```python
数据更新状态:
  - logs/daily_update.log (解析日志文件)
  - 实时监控: fetch_data/quick_daily_update.py 执行状态

选股结果:
  - reports/daily_selection_v3/选股报告_YYYYMMDD.md
  - reports/daily_selection_v3.7/选股报告_YYYYMMDD.md
  - reports/daily_selection_v3.8/选股报告_YYYYMMDD.md
  - reports/daily_selection_v3.81/选股报告_YYYYMMDD.md
  - reports/daily_selection_v3.9/选股报告_YYYYMMDD.md
  - SQLite: stock_signals 表

AI分析:
  - reports/ai_enhanced/AI分析报告_YYYYMMDD.md
```

#### API端点设计
```
GET  /api/daily/status              # 获取数据更新状态
POST /api/daily/update              # 触发数据更新
GET  /api/daily/update/stream       # SSE实时进度
GET  /api/daily/selections          # 获取选股列表（按日期）
GET  /api/daily/selection/:date     # 获取指定日期选股详情
POST /api/daily/select              # 触发选股任务
GET  /api/daily/select/stream       # SSE选股进度
GET  /api/daily/ai-analysis/:date   # 获取AI分析报告
```

#### 前端页面设计
```
日常任务页面 (daily_tasks.html)
├─ 顶部卡片: 系统状态概览
│  ├─ 最后更新时间
│  ├─ 数据库统计（股票数、最新数据日期）
│  └─ 快捷操作按钮
├─ 数据更新区域
│  ├─ 进度条组（4个子进度）
│  ├─ 日志输出窗口
│  └─ 更新按钮
├─ 选股结果区域
│  ├─ 日期选择器 + 版本选择器
│  ├─ 数据表格（股票代码、名称、评分、涨跌幅）
│  └─ 操作列（查看详情、导出）
└─ AI分析区域
   ├─ 市场概况卡片
   ├─ 风险评估仪表盘
   └─ 完整报告查看器
```

---

### 模块2：模型训练 (Model Training)

#### 功能需求
1. **模型版本管理**
   - 列表展示所有模型版本（v3.7/v3.8/v3.81/v3.9）
   - 每个版本的训练参数展示
   - 模型文件信息（大小、创建时间）
   - 版本切换和比较

2. **训练参数配置**
   - Web表单配置训练参数
   - 参数预设（默认/快速/全面）
   - 参数验证和提示
   - 保存参数配置

3. **训练进度监控**
   - 实时进度条（epoch级别）
   - 训练日志流式输出
   - Loss/Accuracy曲线实时更新
   - 预估剩余时间

4. **模型性能指标**
   - 核心指标仪表盘（准确率、AUC、F1-Score）
   - 混淆矩阵热力图
   - ROC曲线
   - 特征重要性排名
   - 模型间性能对比表

5. **训练历史记录**
   - 历史训练任务列表
   - 训练参数对比
   - 性能趋势图

#### 数据源
```python
模型文件:
  - models/v37/*.pkl
  - models/v38/*.pkl
  - models/v381/*.pkl
  - models/v39/*.pkl

训练参数:
  - 从 train_v380_parameterized.py 提取默认参数
  - 用户配置保存在 webapp/configs/training_configs.json

训练记录:
  - logs/training_*.log
  - 新增: webapp/data/training_history.db (SQLite)
    表结构:
    - training_jobs (id, version, start_time, end_time, status, params)
    - training_metrics (job_id, epoch, loss, accuracy, val_loss, val_accuracy)
    - model_performance (job_id, metric_name, metric_value)

实时进度:
  - 需要修改训练脚本，添加进度回调
  - 通过文件或内存队列传递进度
```

#### API端点设计
```
GET  /api/models                    # 获取所有模型版本列表
GET  /api/models/:version           # 获取指定版本详情
GET  /api/models/:version/metrics   # 获取模型性能指标
POST /api/models/train              # 启动训练任务
GET  /api/models/train/stream       # SSE训练进度
DELETE /api/models/train/:job_id    # 取消训练任务
GET  /api/models/compare            # 模型性能对比
GET  /api/models/history            # 训练历史记录
```

#### 前端页面设计
```
模型训练页面 (model_training.html)
├─ 模型版本切换卡片
│  ├─ 版本标签页（v3.7/v3.8/v3.81/v3.9）
│  ├─ 当前版本状态
│  └─ 模型文件信息
├─ 训练配置区域
│  ├─ 参数表单（日期范围、窗口大小等）
│  ├─ 预设按钮（默认/快速/全面）
│  └─ 开始训练按钮
├─ 训练进度区域（训练时显示）
│  ├─ 总体进度条
│  ├─ 当前Epoch进度
│  ├─ 实时Loss/Accuracy曲线
│  ├─ 日志输出窗口
│  └─ 取消按钮
├─ 性能指标仪表盘
│  ├─ 核心指标卡片（大数字展示）
│  ├─ 混淆矩阵热力图
│  ├─ ROC曲线图
│  └─ 特征重要性柱状图
└─ 模型对比区域
   ├─ 版本选择器（多选）
   └─ 对比表格 + 雷达图
```

---

### 模块3：回测 (Backtesting)

#### 功能需求
1. **回测配置**
   - 策略选择（4个量化策略）
   - ML版本选择（v3.0/v3.7/v3.8/v3.81/v3.9）
   - 日期范围选择
   - 初始资金、手续费等参数配置

2. **回测进度监控**
   - 实时进度条（按交易日）
   - 当前处理的股票
   - 已完成的交易数
   - 预估剩余时间

3. **回测结果可视化**
   - **收益曲线**: 净值曲线、累计收益曲线
   - **风险指标**: 最大回撤、夏普比率、索提诺比率
   - **交易统计**: 总交易次数、胜率、盈亏比
   - **持仓分析**: 持仓分布、行业分布
   - **月度/年度收益**: 热力图展示

4. **交易明细表**
   - 所有交易记录（买入/卖出）
   - 每笔交易的盈亏
   - 持仓时间
   - 搜索和过滤功能

5. **回测报告管理**
   - 历史回测列表
   - 报告对比
   - 导出功能（PDF/Excel）

#### 数据源
```python
回测结果:
  - reports/backtest/回测结果_YYYYMMDD.md (解析Markdown)
  - SQLite: backtest_results 表
  - SQLite: backtest_trades 表

实时进度:
  - 需要修改 extensible_backtest_engine.py
  - 添加进度回调函数
  - 通过文件/队列传递进度

配置保存:
  - webapp/configs/backtest_configs.json
```

#### API端点设计
```
GET  /api/backtest/strategies        # 获取可用策略列表
POST /api/backtest/run               # 启动回测任务
GET  /api/backtest/run/stream        # SSE回测进度
GET  /api/backtest/results           # 获取历史回测列表
GET  /api/backtest/result/:id        # 获取指定回测详情
GET  /api/backtest/result/:id/trades # 获取交易明细
DELETE /api/backtest/result/:id      # 删除回测结果
GET  /api/backtest/compare           # 对比多个回测结果
```

#### 前端页面设计
```
回测页面 (backtest.html)
├─ 回测配置区域
│  ├─ 策略选择器（多选）
│  ├─ ML版本选择
│  ├─ 日期范围选择器
│  ├─ 高级参数折叠面板
│  └─ 开始回测按钮
├─ 回测进度区域（回测时显示）
│  ├─ 总体进度条
│  ├─ 当前处理股票信息
│  ├─ 交易统计（实时更新）
│  └─ 取消按钮
├─ 结果概览卡片
│  ├─ 总收益率（大数字）
│  ├─ 最大回撤
│  ├─ 夏普比率
│  └─ 胜率
├─ 收益曲线图
│  ├─ 净值曲线
│  ├─ 基准对比（沪深300）
│  └─ 回撤曲线
├─ 风险分析区域
│  ├─ 月度收益热力图
│  ├─ 收益分布直方图
│  └─ 风险指标雷达图
├─ 交易明细表格
│  ├─ DataTables表格
│  ├─ 搜索/过滤
│  └─ 导出按钮
└─ 历史回测列表
   ├─ 卡片式布局
   └─ 快速对比功能
```

---

## 🔧 技术实现细节

### 任务执行框架

#### TaskManager设计
```python
class TaskManager:
    """统一的任务管理器"""

    def __init__(self):
        self.tasks = {}  # {task_id: TaskInfo}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)

    def submit_task(self, task_type, func, *args, **kwargs):
        """提交新任务"""
        task_id = str(uuid.uuid4())
        task_info = TaskInfo(
            id=task_id,
            type=task_type,
            status='pending',
            progress=0,
            result=None,
            error=None,
            created_at=datetime.now()
        )

        with self.lock:
            self.tasks[task_id] = task_info

        future = self.executor.submit(
            self._run_task, task_id, func, *args, **kwargs
        )

        return task_id

    def get_task_status(self, task_id):
        """获取任务状态"""
        with self.lock:
            return self.tasks.get(task_id)

    def update_progress(self, task_id, progress, message=None):
        """更新任务进度"""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].progress = progress
                self.tasks[task_id].message = message
```

#### 进度回调机制
```python
# 修改现有脚本，支持进度回调
def data_update_with_progress(progress_callback=None):
    """带进度回调的数据更新"""

    if progress_callback:
        progress_callback(0, "开始更新市场行情...")

    # 市场行情更新
    update_daily_quotes()

    if progress_callback:
        progress_callback(25, "市场行情更新完成，开始更新基本面...")

    # 基本面更新
    update_daily_basic()

    if progress_callback:
        progress_callback(50, "基本面更新完成，开始更新技术指标...")

    # ... 依此类推
```

### SSE实现
```python
@app.route('/api/daily/update/stream')
def update_stream():
    """SSE端点：实时推送更新进度"""

    def generate():
        task_id = request.args.get('task_id')

        while True:
            task_info = task_manager.get_task_status(task_id)

            if task_info:
                data = {
                    'progress': task_info.progress,
                    'message': task_info.message,
                    'status': task_info.status
                }
                yield f"data: {json.dumps(data)}\n\n"

                if task_info.status in ['completed', 'failed']:
                    break

            time.sleep(0.5)

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )
```

### Markdown报告解析
```python
class ReportParser:
    """统一的报告解析器"""

    @staticmethod
    def parse_selection_report(filepath):
        """解析选股报告"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取日期
        date_match = re.search(r'日期[：:]\s*(\d{4}-\d{2}-\d{2})', content)
        date = date_match.group(1) if date_match else None

        # 提取股票列表（假设有表格）
        stocks = []
        table_pattern = r'\|\s*(\d{6})\s*\|\s*([^\|]+)\s*\|'
        for match in re.finditer(table_pattern, content):
            stocks.append({
                'code': match.group(1),
                'name': match.group(2).strip()
            })

        return {
            'date': date,
            'stocks': stocks,
            'raw_content': content
        }
```

---

## 📊 数据库扩展设计

### 新增表结构
```sql
-- 训练任务记录表
CREATE TABLE training_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT UNIQUE NOT NULL,
    model_version TEXT NOT NULL,
    start_time DATETIME NOT NULL,
    end_time DATETIME,
    status TEXT NOT NULL,  -- pending, running, completed, failed
    params TEXT,  -- JSON格式的训练参数
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 训练指标记录表
CREATE TABLE training_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    loss REAL,
    accuracy REAL,
    val_loss REAL,
    val_accuracy REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES training_jobs(job_id)
);

-- 模型性能指标表
CREATE TABLE model_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    FOREIGN KEY (job_id) REFERENCES training_jobs(job_id)
);

-- 回测任务记录表（扩展现有backtest_results）
ALTER TABLE backtest_results ADD COLUMN job_id TEXT;
ALTER TABLE backtest_results ADD COLUMN ml_version TEXT;
ALTER TABLE backtest_results ADD COLUMN params TEXT;
```

---

## 🚀 实施计划

### 阶段1：基础框架搭建（2-3小时）
**目标**: 建立可运行的Web应用骨架

**交付物**:
- Flask应用主文件 (app.py)
- 基础路由和模板
- Bootstrap前端框架集成
- 导航栏和主页面
- TaskManager基础类
- 配置文件加载

**验收标准**:
- ✅ 可以通过浏览器访问 http://localhost:5000
- ✅ 三个主要页面可以正常导航
- ✅ 页面布局美观，响应式设计

### 阶段2：日常任务模块（4-5小时）
**目标**: 实现数据更新和选股结果展示

**交付物**:
- 数据更新状态API
- 选股结果API（解析Markdown报告）
- 前端页面完整实现
- SSE实时进度推送
- 手动触发数据更新功能

**验收标准**:
- ✅ 可以查看历史选股结果
- ✅ 可以触发数据更新并看到实时进度
- ✅ 可以切换不同ML版本查看结果
- ✅ 可以导出选股结果为CSV

### 阶段3：模型训练模块（5-6小时）
**目标**: 实现模型训练监控和性能展示

**交付物**:
- 模型信息API
- 训练任务API
- 修改训练脚本支持进度回调
- 数据库表创建和迁移
- 前端训练配置表单
- 实时训练进度监控
- 性能指标可视化（图表）

**验收标准**:
- ✅ 可以查看所有模型版本信息
- ✅ 可以通过Web界面配置并启动训练
- ✅ 可以实时看到训练进度和Loss曲线
- ✅ 训练完成后可以查看性能指标
- ✅ 可以对比不同模型版本的性能

### 阶段4：回测模块（5-6小时）
**目标**: 实现回测配置、监控和结果展示

**交付物**:
- 回测配置API
- 回测任务API
- 修改回测引擎支持进度回调
- 回测结果解析和存储
- 前端回测配置表单
- 实时回测进度监控
- 收益曲线和风险指标可视化
- 交易明细表格

**验收标准**:
- ✅ 可以通过Web界面配置并启动回测
- ✅ 可以实时看到回测进度
- ✅ 回测完成后可以查看详细报告
- ✅ 可以查看交易明细表
- ✅ 可以对比多个回测结果

### 阶段5：优化和完善（3-4小时）
**目标**: 性能优化、错误处理、用户体验提升

**交付物**:
- 缓存机制（Redis可选）
- 错误处理和日志记录
- 用户友好的错误提示
- 加载动画和占位符
- 导出功能完善（PDF/Excel）
- 响应式设计优化
- 使用文档和README

**验收标准**:
- ✅ 页面加载速度快（<2秒）
- ✅ 所有错误都有友好提示
- ✅ 移动端访问体验良好
- ✅ 所有导出功能正常工作
- ✅ 有完整的使用文档

---

## 🔒 安全和性能考虑

### 安全
- **本地部署**: 仅监听localhost，不对外暴露
- **可选认证**: 可以添加简单的密码认证
- **输入验证**: 所有用户输入进行验证和清理
- **SQL注入防护**: 使用参数化查询

### 性能
- **缓存策略**:
  - 静态数据（模型列表）缓存5分钟
  - 报告内容缓存到内存
  - 使用LRU缓存策略

- **数据库优化**:
  - 适当的索引
  - 连接池管理
  - 查询结果限制（分页）

- **并发控制**:
  - 任务队列防止重复提交
  - 线程池限制并发数
  - 长时间任务超时机制

---

## 📂 项目文件结构

```
webapp/
├── app.py                          # Flask主应用
├── config.py                       # 配置文件
├── requirements.txt                # Python依赖
├── DESIGN_PLAN.md                  # 本设计文档
├── README.md                       # 使用文档
├── start.sh                        # 启动脚本
│
├── api/                            # API模块
│   ├── __init__.py
│   ├── daily_tasks.py             # 日常任务API
│   ├── model_training.py          # 模型训练API
│   └── backtest.py                # 回测API
│
├── core/                           # 核心功能
│   ├── __init__.py
│   ├── task_manager.py            # 任务管理器
│   ├── report_parser.py           # 报告解析器
│   ├── database.py                # 数据库操作
│   └── utils.py                   # 工具函数
│
├── static/                         # 静态资源
│   ├── css/
│   │   ├── main.css               # 主样式
│   │   └── components.css         # 组件样式
│   ├── js/
│   │   ├── main.js                # 主逻辑
│   │   ├── daily_tasks.js         # 日常任务前端
│   │   ├── model_training.js      # 模型训练前端
│   │   ├── backtest.js            # 回测前端
│   │   └── utils.js               # 工具函数
│   └── images/
│       └── logo.png
│
├── templates/                      # HTML模板
│   ├── base.html                  # 基础模板
│   ├── index.html                 # 首页/仪表板
│   ├── daily_tasks.html           # 日常任务页面
│   ├── model_training.html        # 模型训练页面
│   └── backtest.html              # 回测页面
│
├── configs/                        # 配置文件
│   ├── training_configs.json      # 训练配置
│   └── backtest_configs.json      # 回测配置
│
├── data/                           # 运行时数据
│   ├── webapp.db                  # Web应用数据库
│   └── cache/                     # 缓存目录
│
└── logs/                           # 日志文件
    └── webapp.log
```

---

## 📝 关键技术决策说明

### 为什么选择Flask而不是FastAPI？
- **集成简单**: Flask与现有Python代码集成更容易
- **学习曲线**: 更平缓，适合快速开发
- **同步模型**: 与现有脚本的执行模型一致
- **生态成熟**: 丰富的扩展和文档

### 为什么选择SSE而不是WebSocket？
- **单向推送**: 进度监控只需服务器向客户端推送
- **简单实现**: HTTP协议，无需额外握手
- **浏览器支持**: 原生EventSource API
- **自动重连**: 浏览器自动处理断线重连

### 为什么不使用Celery？
- **部署复杂**: 需要Redis/RabbitMQ
- **过度设计**: 本地单机部署不需要分布式任务队列
- **依赖增加**: 增加系统复杂度
- **Threading足够**: 对于本项目的并发需求足够

### 为什么使用Bootstrap而不是React/Vue？
- **无需构建**: 开发和部署更简单
- **快速开发**: 现成的组件库
- **服务端渲染**: 更好的首屏加载
- **项目定位**: 内部工具，不需要复杂SPA

---

## 🎯 成功指标

### 功能完整性
- ✅ 所有三个主模块功能完整
- ✅ 可以通过Web界面完成所有日常操作
- ✅ 实时进度监控正常工作

### 性能指标
- ✅ 页面加载时间 < 2秒
- ✅ API响应时间 < 500ms
- ✅ 实时进度更新延迟 < 1秒

### 用户体验
- ✅ 界面美观，符合现代Web设计规范
- ✅ 响应式设计，支持不同屏幕尺寸
- ✅ 错误提示友好，操作流程清晰
- ✅ 有完整的使用文档

---

## 🔄 后续扩展方向

### 短期（1-2周）
- 添加用户认证系统
- 邮件/微信通知功能
- 移动端App（React Native）
- 更多图表类型（K线图、瀑布图）

### 中期（1-2月）
- 多用户支持
- 权限管理系统
- 策略回测对比工具
- 实时监控告警

### 长期（3-6月）
- 云端部署版本
- 分布式回测支持
- 策略市场（分享和下载策略）
- AI自动优化参数

---

## 📞 技术支持

### 开发联系人
- 开发者: Claude & User
- 项目仓库: /Users/yangxu/StockTradebyZ

### 依赖库版本
```txt
Flask==3.0.0
Flask-CORS==4.0.0
python-dotenv==1.0.0
pandas==2.1.0
numpy==1.24.0
plotly==5.17.0
markdown==3.5.0
```

---

## 🎨 UI/UX设计原则

### 设计理念
1. **简洁优先**: 避免信息过载，突出核心功能
2. **实时反馈**: 所有操作都有即时反馈
3. **数据驱动**: 用图表说话，减少文字
4. **容错设计**: 友好的错误提示和恢复机制

### 色彩方案
- **主色调**: 深蓝色 (#1e3a8a) - 专业、稳重
- **辅色调**: 绿色 (#10b981) - 盈利、增长
- **强调色**: 橙色 (#f59e0b) - 警告、关注
- **背景色**: 浅灰 (#f9fafb) - 舒适阅读

### 图表设计
- **统一风格**: 使用ApexCharts统一图表风格
- **颜色一致**: 涨用红色，跌用绿色（A股习惯）
- **交互友好**: 支持悬停提示、缩放、导出

---

**计划书版本**: v1.0
**最后更新**: 2025-11-24
**预计总工时**: 20-25小时
**建议实施周期**: 3-5天（每天4-6小时）
