# 持仓管理系统设计计划书

## 1. 系统概述

### 1.1 功能需求
1. **持仓管理**: 添加、减少当前持仓
2. **智能建议**: 根据ML模型评估持仓，每日生成操作建议
3. **交易记录**: 记录所有持仓变化和操作历史
4. **事后回顾**: 对历史操作进行评估和打分

### 1.2 技术架构
- **后端**: Flask Blueprint + SQLite
- **前端**: Bootstrap 5 + jQuery + ApexCharts
- **数据源**: 集成stock_data.db行情数据 + ML模型评分

---

## 2. 数据库设计

### 2.1 持仓表 (positions)
```sql
CREATE TABLE positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL,              -- 股票代码
    name TEXT,                       -- 股票名称
    quantity INTEGER NOT NULL,       -- 持仓数量
    avg_cost REAL NOT NULL,          -- 平均成本价
    current_price REAL,              -- 当前价格
    market_value REAL,               -- 市值
    profit_loss REAL,                -- 盈亏金额
    profit_loss_pct REAL,            -- 盈亏百分比
    first_buy_date DATE,             -- 首次买入日期
    last_update DATETIME,            -- 最后更新时间
    status TEXT DEFAULT 'holding',   -- 状态: holding/closed
    notes TEXT,                      -- 备注
    UNIQUE(code, status)
);
```

### 2.2 交易记录表 (trades)
```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date DATE NOT NULL,        -- 交易日期
    trade_time DATETIME,             -- 交易时间
    code TEXT NOT NULL,              -- 股票代码
    name TEXT,                       -- 股票名称
    action TEXT NOT NULL,            -- 操作: buy/sell/add/reduce
    quantity INTEGER NOT NULL,       -- 交易数量
    price REAL NOT NULL,             -- 交易价格
    amount REAL,                     -- 交易金额
    commission REAL DEFAULT 0,       -- 手续费
    reason TEXT,                     -- 交易理由
    strategy TEXT,                   -- 对应策略
    signal_source TEXT,              -- 信号来源
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### 2.3 操作建议表 (recommendations)
```sql
CREATE TABLE recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,              -- 建议日期
    code TEXT NOT NULL,              -- 股票代码
    name TEXT,                       -- 股票名称
    action TEXT NOT NULL,            -- 建议操作: hold/sell/add/reduce
    urgency TEXT DEFAULT 'normal',   -- 紧急程度: low/normal/high/critical
    reason TEXT,                     -- 建议理由
    ml_score REAL,                   -- ML评分
    technical_signal TEXT,           -- 技术信号
    stop_loss_price REAL,            -- 止损价
    take_profit_price REAL,          -- 止盈价
    confidence REAL,                 -- 置信度 (0-1)
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_executed INTEGER DEFAULT 0,   -- 是否已执行
    executed_at DATETIME,            -- 执行时间
    UNIQUE(date, code)
);
```

### 2.4 操作评估表 (trade_evaluations)
```sql
CREATE TABLE trade_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id INTEGER NOT NULL,       -- 关联交易ID
    eval_date DATE NOT NULL,         -- 评估日期
    days_after INTEGER,              -- 交易后天数
    price_after REAL,                -- N天后价格
    return_pct REAL,                 -- 收益率
    score REAL,                      -- 评分 (0-100)
    grade TEXT,                      -- 等级: A/B/C/D/F
    comments TEXT,                   -- 评价
    max_profit_pct REAL,             -- 期间最大盈利
    max_loss_pct REAL,               -- 期间最大亏损
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (trade_id) REFERENCES trades(id)
);
```

### 2.5 持仓快照表 (position_snapshots)
```sql
CREATE TABLE position_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date DATE NOT NULL,
    total_market_value REAL,         -- 总市值
    total_cost REAL,                 -- 总成本
    total_profit_loss REAL,          -- 总盈亏
    total_profit_loss_pct REAL,      -- 总收益率
    position_count INTEGER,          -- 持仓数量
    details TEXT,                    -- JSON格式详情
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(snapshot_date)
);
```

---

## 3. API设计

### 3.1 持仓管理 API

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | /api/portfolio/positions | 获取所有当前持仓 |
| POST | /api/portfolio/positions | 添加新持仓 |
| PUT | /api/portfolio/positions/{id} | 修改持仓 |
| DELETE | /api/portfolio/positions/{id} | 删除持仓 |
| POST | /api/portfolio/positions/{id}/add | 加仓 |
| POST | /api/portfolio/positions/{id}/reduce | 减仓 |
| GET | /api/portfolio/summary | 获取持仓汇总 |

### 3.2 交易记录 API

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | /api/portfolio/trades | 获取交易记录列表 |
| POST | /api/portfolio/trades | 添加交易记录 |
| PUT | /api/portfolio/trades/{id} | 修改交易记录 |
| DELETE | /api/portfolio/trades/{id} | 删除交易记录 |
| GET | /api/portfolio/trades/{id} | 获取单条交易详情 |

### 3.3 操作建议 API

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | /api/portfolio/recommendations | 获取今日建议 |
| POST | /api/portfolio/recommendations/generate | 生成最新建议 |
| POST | /api/portfolio/recommendations/{id}/execute | 标记建议已执行 |
| GET | /api/portfolio/recommendations/history | 获取历史建议 |

### 3.4 操作评估 API

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | /api/portfolio/evaluations | 获取评估列表 |
| POST | /api/portfolio/evaluations/run | 执行评估计算 |
| GET | /api/portfolio/evaluations/stats | 获取评估统计 |
| GET | /api/portfolio/evaluations/trade/{trade_id} | 获取单笔交易评估 |

---

## 4. 前端页面设计

### 4.1 页面布局
```
持仓管理页面
├── 顶部统计卡片
│   ├── 总市值
│   ├── 总盈亏
│   ├── 今日盈亏
│   └── 持仓数量
├── 标签页导航
│   ├── 当前持仓
│   ├── 操作建议
│   ├── 交易记录
│   └── 事后回顾
└── 内容区域
    ├── 当前持仓表格 + 操作按钮
    ├── 今日建议列表
    ├── 交易记录表格
    └── 评估报告
```

### 4.2 功能模块

#### 4.2.1 当前持仓
- 持仓列表表格（代码、名称、数量、成本、现价、盈亏、操作）
- 添加新持仓按钮 + 模态框
- 每行操作按钮：加仓、减仓、清仓
- 实时价格更新（从stock_data.db获取）

#### 4.2.2 操作建议
- 今日建议卡片列表
- 建议来源说明（ML评分、技术信号等）
- 一键执行按钮
- 手动生成建议按钮

#### 4.2.3 交易记录
- 交易历史表格（支持筛选和分页）
- 编辑/删除功能
- 导出CSV功能

#### 4.2.4 事后回顾
- 评估统计图表（胜率、平均收益等）
- 交易评分列表
- 按时间段筛选
- 详细评价内容

---

## 5. 核心算法

### 5.1 持仓评估算法
```python
def evaluate_position(position):
    """评估单个持仓"""
    score = 0
    signals = []

    # 1. 获取ML评分 (40%)
    ml_score = get_ml_score(position.code)
    score += ml_score * 0.4

    # 2. 技术指标分析 (30%)
    tech_score = analyze_technical(position.code)
    score += tech_score * 0.3

    # 3. 盈亏状态 (20%)
    pl_score = evaluate_profit_loss(position)
    score += pl_score * 0.2

    # 4. 持仓时间 (10%)
    time_score = evaluate_holding_time(position)
    score += time_score * 0.1

    return {
        'score': score,
        'action': determine_action(score, position),
        'signals': signals
    }
```

### 5.2 建议生成算法
```python
def generate_recommendations():
    """生成每日操作建议"""
    recommendations = []

    for position in get_all_positions():
        eval_result = evaluate_position(position)

        if eval_result['action'] != 'hold':
            recommendations.append({
                'code': position.code,
                'action': eval_result['action'],
                'urgency': calculate_urgency(eval_result),
                'reason': format_reason(eval_result),
                'confidence': eval_result['score'] / 100
            })

    return sorted(recommendations, key=lambda x: x['urgency'], reverse=True)
```

### 5.3 交易评分算法
```python
def evaluate_trade(trade, days_after=5):
    """评估交易操作"""
    # 获取交易后N天的价格
    price_after = get_price_after_days(trade.code, trade.trade_date, days_after)

    if trade.action in ['buy', 'add']:
        return_pct = (price_after - trade.price) / trade.price * 100
    else:  # sell, reduce
        return_pct = (trade.price - price_after) / trade.price * 100

    # 计算评分
    if return_pct >= 10: grade, score = 'A', 95
    elif return_pct >= 5: grade, score = 'B', 80
    elif return_pct >= 0: grade, score = 'C', 65
    elif return_pct >= -5: grade, score = 'D', 45
    else: grade, score = 'F', 25

    return {'grade': grade, 'score': score, 'return_pct': return_pct}
```

---

## 6. 实现步骤

### 第一阶段：数据库和基础API
1. 在database.py中添加持仓相关表的初始化
2. 创建api/portfolio.py实现基础CRUD API
3. 在app.py中注册新蓝图

### 第二阶段：前端页面
1. 创建templates/portfolio.html
2. 实现持仓列表和操作功能
3. 实现交易记录管理

### 第三阶段：智能建议
1. 集成ML模型评分
2. 实现建议生成算法
3. 添加建议显示和执行功能

### 第四阶段：事后回顾
1. 实现交易评估算法
2. 添加评估统计和图表
3. 完成评价展示功能

---

## 7. 文件清单

### 后端
- `webapp/core/database.py` - 添加新表初始化
- `webapp/core/portfolio_manager.py` - 持仓管理核心逻辑
- `webapp/api/portfolio.py` - 持仓管理API
- `webapp/app.py` - 注册路由

### 前端
- `webapp/templates/portfolio.html` - 持仓管理页面
- `webapp/static/js/portfolio.js` - 页面交互逻辑
- `webapp/templates/base.html` - 添加导航链接

---

创建时间: 2025-11-28
预计完成: 当日
