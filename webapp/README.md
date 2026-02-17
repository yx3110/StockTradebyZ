# StockTradebyZ Web 管理平台

本地 Web 管理平台，用于可视化和管理 StockTradebyZ 量化交易系统。

## 功能特性

### 1. 系统仪表板 (首页)
- **核心指标卡片**：股票/ETF总数、最新数据日期、ML模型数、回测报告数
- **快速操作入口**：数据更新、今日选股、模型管理、策略回测
- **三大模块概览**：每日任务、模型训练、回测分析预览
- **数据库统计**：大小、日线数据条数、技术指标条数、时间跨度
- **系统信息**：版本、ML评分版本、量化策略列表

### 2. 每日任务模块
- **数据更新**：一键更新市场行情、基本面、技术指标数据
- **选股执行**：支持多个ML版本（v3.0-v3.9）的选股策略
- **报告查看**：浏览历史选股报告，支持导出CSV
- **AI分析**：查看AI增强分析报告
- **实时进度**：通过SSE推送任务执行进度

### 3. 模型训练模块
- **模型概览**：显示所有ML版本状态（v3.6-v3.9）
- **特征重要性**：可视化各版本Top特征（ApexCharts横向柱状图）
- **训练报告**：查看训练参数、性能指标、训练历史
- **模型训练**：配置参数启动训练任务
- **模型文件**：浏览模型目录下的文件列表

### 4. 回测分析模块
- **回测汇总**：各版本平均收益、夏普比率对比图表
- **版本表现图**：柱状图对比不同ML版本表现
- **运行回测**：配置策略、时间范围、资金参数
- **历史报告**：浏览JSON/MD格式回测报告
- **详情查看**：Modal展示完整回测结果和对比图表

## 快速开始

### 环境要求
- Python 3.8+
- Flask 3.0+
- SQLite3
- 项目根目录存在 `stock_data.db` 数据库

### 安装依赖
```bash
cd webapp
pip install -r requirements.txt
```

### 启动服务

#### 方式1: 使用启动脚本（推荐）
```bash
cd webapp
./start.sh
```

#### 方式2: 手动启动
```bash
cd webapp

# 开发模式
python3 app.py

# 生产模式
export FLASK_ENV=production
python3 app.py
```

服务启动后访问: http://127.0.0.1:5000

## 项目结构

```
webapp/
├── app.py                  # Flask 应用入口
├── config.py               # 配置文件
├── requirements.txt        # Python 依赖
├── start.sh               # 启动脚本
├── README.md              # 本文档
│
├── api/                    # API 蓝图
│   ├── __init__.py
│   ├── daily_tasks.py     # 每日任务 API
│   ├── model_training.py  # 模型训练 API
│   └── backtest.py        # 回测分析 API
│
├── core/                   # 核心模块
│   ├── __init__.py
│   ├── database.py        # 数据库管理器
│   ├── task_manager.py    # 任务管理器
│   └── report_parser.py   # 报告解析器
│
├── templates/              # Jinja2 模板
│   ├── base.html          # 基础模板（导航栏、布局）
│   ├── index.html         # 首页仪表板
│   ├── daily_tasks.html   # 每日任务页
│   ├── model_training.html # 模型训练页
│   └── backtest.html      # 回测分析页
│
├── static/                 # 静态资源
│   ├── css/
│   │   └── main.css       # 自定义样式
│   └── js/
│       └── main.js        # 自定义脚本
│
├── data/                   # Web应用数据
│   └── webapp.db          # Web应用数据库
│
└── logs/                   # 日志目录
    └── webapp.log
```

## API 文档

### 每日任务 API (`/api/daily/`)

| 端点 | 方法 | 说明 | 参数 |
|------|------|------|------|
| `/status` | GET | 获取系统状态和数据库统计 | - |
| `/update` | POST | 触发数据更新任务 | `date` (可选) |
| `/update/stream` | GET | SSE实时更新进度 | `task_id` |
| `/selections` | GET | 获取选股日期列表 | `version`, `limit` |
| `/selection/<date>` | GET | 获取指定日期选股详情 | `version` |
| `/selection/<date>/export` | GET | 导出选股结果CSV | `version` |
| `/select` | POST | 触发选股任务 | `date`, `version` |
| `/select/stream` | GET | SSE实时选股进度 | `task_id` |
| `/reports` | GET | 获取选股报告列表 | `version`, `limit` |
| `/ai-analysis/<date>` | GET | 获取AI分析报告 | - |

### 模型训练 API (`/api/models/`)

| 端点 | 方法 | 说明 | 参数 |
|------|------|------|------|
| `/` | GET | 获取所有模型版本列表 | - |
| `/summary` | GET | 获取模型汇总统计 | - |
| `/<version>` | GET | 获取指定版本详情 | - |
| `/<version>/features` | GET | 获取特征重要性 | - |
| `/<version>/report` | GET | 获取训练报告 | - |
| `/<version>/metrics` | GET | 获取模型性能指标 | - |
| `/train` | POST | 启动模型训练 | `version`, `start_date`, `end_date`, `lookback_days`, `lookahead_days` |
| `/train/stream` | GET | SSE实时训练进度 | `task_id` |
| `/history` | GET | 获取训练历史记录 | `version`, `limit` |

### 回测分析 API (`/api/backtest/`)

| 端点 | 方法 | 说明 | 参数 |
|------|------|------|------|
| `/strategies` | GET | 获取可用策略列表 | - |
| `/run` | POST | 启动回测任务 | `strategies`, `ml_version`, `start_date`, `end_date`, `initial_capital`, `commission` |
| `/run/stream` | GET | SSE实时回测进度 | `task_id` |
| `/results` | GET | 获取历史回测列表 | `limit` |
| `/result/<id>` | GET | 获取回测详情 | - |
| `/summary` | GET | 获取回测汇总统计 | - |

### 通用 API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查 |

## 配置说明

### config.py 主要配置项

```python
# Flask配置
SECRET_KEY = 'dev-secret-key-change-in-production'
DEBUG = True

# 数据库路径
STOCK_DB_PATH = BASE_DIR / 'stock_data.db'
WEBAPP_DB_PATH = WEBAPP_DIR / 'data' / 'webapp.db'

# 报告目录配置
REPORTS_DIR = BASE_DIR / 'reports'
DAILY_SELECTION_DIRS = {
    'v3.0': REPORTS_DIR / 'daily_selection_v3',
    'v3.7': REPORTS_DIR / 'daily_selection_v3.7',
    'v3.8': REPORTS_DIR / 'daily_selection_v3.8',
    'v3.81': REPORTS_DIR / 'daily_selection_v3.81',
    'v3.9': REPORTS_DIR / 'daily_selection_v3.9',
}
AI_ENHANCED_DIR = REPORTS_DIR / 'ai_enhanced'
BACKTEST_DIR = REPORTS_DIR / 'backtest'

# 模型目录配置
MODELS_DIR = BASE_DIR / 'models'
MODEL_DIRS = {
    'v3.6': MODELS_DIR / 'v360',
    'v3.7': MODELS_DIR / 'v370',
    'v3.8': MODELS_DIR / 'v380',
    'v3.81': MODELS_DIR / 'v380',  # v3.81 uses v380 directory
    'v3.9': MODELS_DIR / 'v39',
}

# 脚本路径配置
QUICK_DAILY_UPDATE_SCRIPT = SCRIPTS_DIR / 'fetch_data' / 'quick_daily_update.py'
STOCK_SELECTOR_SCRIPT = SCRIPTS_DIR / 'tomorrow_stock_selector.py'
TRAIN_SCRIPT = SCRIPTS_DIR / 'train_v380_parameterized.py'
BACKTEST_SCRIPT = SCRIPTS_DIR / 'extensible_backtest_engine.py'

# 任务配置
MAX_WORKERS = 4          # 最大并发任务数
TASK_TIMEOUT = 3600      # 任务超时时间(秒)
CACHE_TTL = 300          # 缓存时间(秒)

# Python解释器
PYTHON_EXECUTABLE = 'python3'
```

## 技术栈

### 后端
- **Flask 3.0**: Web框架
- **SQLite**: 数据库存储
- **Server-Sent Events (SSE)**: 实时进度推送
- **ThreadPoolExecutor**: 后台任务执行
- **Blueprint**: API模块化组织

### 前端
- **Bootstrap 5.3**: UI框架
- **jQuery 3.7**: DOM操作和AJAX
- **ApexCharts**: 数据可视化图表
- **DataTables 1.13**: 交互式表格
- **Bootstrap Icons**: 图标库

## 使用说明

### 数据更新
1. 访问"每日任务"页面
2. 点击"数据更新"区域的"开始更新"按钮
3. 实时查看进度条和状态
4. 等待更新完成

### 运行选股
1. 在"每日任务"页面选择ML版本（v3.0-v3.9）
2. 设置选股日期
3. 点击"运行选股"按钮
4. 等待选股完成，查看推荐股票

### 模型训练
1. 访问"模型训练"页面
2. 选择要训练的模型版本
3. 配置参数：日期范围、Lookback天数、Lookahead天数
4. 点击"开始训练"按钮
5. 等待训练完成（约30-80分钟）

### 策略回测
1. 访问"回测分析"页面
2. 选择ML版本和策略
3. 设置日期范围、初始资金、手续费率
4. 点击"开始回测"按钮
5. 查看回测结果和版本对比

## 故障排除

### 常见问题

1. **数据库连接失败**
   - 检查 `stock_data.db` 路径是否正确
   - 确保数据库文件存在且有读取权限

2. **任务执行失败**
   - 检查 Python 解释器路径配置
   - 查看 `logs/webapp.log` 日志
   - 确保相关脚本存在且可执行

3. **页面加载缓慢**
   - 检查数据库大小和查询效率
   - 考虑添加数据库索引

4. **SSE连接中断**
   - 检查网络连接
   - 确保浏览器支持 EventSource
   - 检查防火墙设置

5. **模型文件未找到**
   - 确认 `models/` 目录结构正确
   - 检查 MODEL_DIRS 配置与实际目录名称匹配

## 开发指南

### 添加新的API端点

```python
# 在 api/ 目录对应模块添加
@blueprint.route('/new-endpoint', methods=['GET'])
def new_endpoint():
    try:
        # 业务逻辑
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        logger.error(f'错误: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500
```

### 添加新页面

1. 在 `templates/` 添加HTML模板（继承base.html）
2. 在 `app.py` 添加页面路由
3. 在 `base.html` 导航栏添加链接

### 后台任务

使用 `task_manager` 提交长时间运行的任务：

```python
from core.task_manager import task_manager, TaskType

task_id = task_manager.submit_task(
    task_type=TaskType.YOUR_TYPE,
    func=your_task_function,
    metadata={'param': 'value'}
)
```

## 安全说明

- 默认仅监听 `127.0.0.1`，不对外暴露
- 生产环境请修改 `SECRET_KEY`
- 建议在生产环境添加认证机制
- 数据库文件权限应限制为仅所有者可读写

## 更新日志

### v1.1.0 (2025-11-25)
- 增强首页仪表板，整合三大模块概览
- 添加数据库详细统计（日线条数、技术指标条数、时间跨度）
- 模型训练页面添加特征重要性图表
- 回测页面添加版本表现对比图
- 新增报告列表API
- 优化加载动画和错误提示

### v1.0.0 (2025-11-24)
- 初始版本发布
- 实现三大核心模块：每日任务、模型训练、回测分析
- SSE实时进度推送
- 数据可视化图表

## 许可证

MIT License

---

**版本**: 1.1.0
**最后更新**: 2025-11-25
