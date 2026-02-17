-- 股票交易数据库设计
-- 使用SQLite作为本地数据库，支持高性能查询和数据分析

-- 1. 证券基本信息表
CREATE TABLE IF NOT EXISTS securities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code VARCHAR(10) NOT NULL UNIQUE,
    name VARCHAR(50) NOT NULL,
    type VARCHAR(20) NOT NULL, -- A股, ETF_基金, 等
    exchange VARCHAR(10), -- SH, SZ
    industry VARCHAR(50), -- 所属行业
    area VARCHAR(20), -- 注册地区
    list_date DATE,
    delist_date DATE,
    is_active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_securities_code ON securities(code);
CREATE INDEX idx_securities_type ON securities(type);
CREATE INDEX idx_securities_active ON securities(is_active);

-- 2. 日线行情数据表（主要数据表）
CREATE TABLE IF NOT EXISTS daily_quotes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    security_id INTEGER NOT NULL,
    trade_date DATE NOT NULL,
    open DECIMAL(10,3) NOT NULL,
    high DECIMAL(10,3) NOT NULL,
    low DECIMAL(10,3) NOT NULL,
    close DECIMAL(10,3) NOT NULL,
    volume BIGINT NOT NULL,
    amount DECIMAL(20,2), -- 成交额
    
    -- 复权价格
    adj_open DECIMAL(10,3),
    adj_high DECIMAL(10,3),
    adj_low DECIMAL(10,3),
    adj_close DECIMAL(10,3),
    adj_factor DECIMAL(10,6) DEFAULT 1.0,
    
    -- A股特有指标
    price_change DECIMAL(10,3), -- 价格变动
    price_change_pct DECIMAL(10,4), -- 涨跌幅
    is_limit_up BOOLEAN DEFAULT 0, -- 涨停标记
    is_limit_down BOOLEAN DEFAULT 0, -- 跌停标记
    is_st BOOLEAN DEFAULT 0, -- ST标记
    is_suspend BOOLEAN DEFAULT 0, -- 停牌标记
    
    -- 技术指标（可选，用于加速查询）
    ma5 DECIMAL(10,3),
    ma10 DECIMAL(10,3),
    ma20 DECIMAL(10,3),
    ma60 DECIMAL(10,3),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (security_id) REFERENCES securities(id),
    UNIQUE(security_id, trade_date)
);

-- 创建复合索引优化查询性能
CREATE INDEX idx_daily_quotes_security_date ON daily_quotes(security_id, trade_date);
CREATE INDEX idx_daily_quotes_date ON daily_quotes(trade_date);
CREATE INDEX idx_daily_quotes_limit ON daily_quotes(is_limit_up, is_limit_down);

-- 3. 技术指标表（扩展指标）
CREATE TABLE IF NOT EXISTS technical_indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    security_id INTEGER NOT NULL,
    trade_date DATE NOT NULL,
    
    -- KDJ指标
    kdj_k DECIMAL(10,3),
    kdj_d DECIMAL(10,3),
    kdj_j DECIMAL(10,3),
    
    -- MACD指标
    macd_dif DECIMAL(10,3),
    macd_dea DECIMAL(10,3),
    macd_macd DECIMAL(10,3),
    
    -- RSI指标
    rsi6 DECIMAL(10,3),
    rsi12 DECIMAL(10,3),
    rsi24 DECIMAL(10,3),
    
    -- 布林带
    boll_upper DECIMAL(10,3),
    boll_middle DECIMAL(10,3),
    boll_lower DECIMAL(10,3),
    
    -- BBI指标
    bbi DECIMAL(10,3),
    
    -- 成交量指标
    volume_ma5 BIGINT,
    volume_ma10 BIGINT,
    volume_ratio DECIMAL(10,3), -- 量比
    
    -- 🆕 挤压动量指标 (v3.2新增)
    -- 肯特纳通道
    kc_upper DECIMAL(10,3),
    kc_middle DECIMAL(10,3), 
    kc_lower DECIMAL(10,3),
    kc_width DECIMAL(10,3),
    
    -- 挤压状态指标
    squeeze_state BOOLEAN DEFAULT 0, -- 是否处于挤压状态
    squeeze_release BOOLEAN DEFAULT 0, -- 是否挤压释放
    squeeze_intensity DECIMAL(10,4), -- 挤压强度 (布林带宽度/肯特纳通道宽度)
    squeeze_days INTEGER DEFAULT 0, -- 最近30天挤压天数
    recent_releases INTEGER DEFAULT 0, -- 最近10天释放次数
    
    -- 动量相关指标
    squeeze_momentum DECIMAL(10,4), -- 挤压动量值
    momentum_direction INTEGER DEFAULT 0, -- 动量方向 (1上升, -1下降, 0中性)
    momentum_strength DECIMAL(10,4), -- 动量强度
    momentum_acceleration DECIMAL(10,4), -- 动量加速度
    momentum_consistency DECIMAL(10,4), -- 动量一致性 (0-1)
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (security_id) REFERENCES securities(id),
    UNIQUE(security_id, trade_date)
);

CREATE INDEX idx_tech_indicators_security_date ON technical_indicators(security_id, trade_date);

-- 4. 选股信号表
CREATE TABLE IF NOT EXISTS stock_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_date DATE NOT NULL,
    security_id INTEGER NOT NULL,
    strategy_name VARCHAR(50) NOT NULL, -- 策略名称
    signal_type VARCHAR(20) NOT NULL, -- BUY, SELL, HOLD
    
    -- 评分和建议
    comprehensive_score DECIMAL(5,2),
    suggested_buy_price DECIMAL(10,3),
    stop_loss_price DECIMAL(10,3),
    take_profit_price DECIMAL(10,3),
    risk_reward_ratio DECIMAL(5,2),
    
    -- 策略参数
    strategy_params TEXT, -- JSON格式存储策略参数
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (security_id) REFERENCES securities(id)
);

CREATE INDEX idx_signals_date ON stock_signals(signal_date);
CREATE INDEX idx_signals_security ON stock_signals(security_id);
CREATE INDEX idx_signals_strategy ON stock_signals(strategy_name);

-- 5. 回测交易记录表
CREATE TABLE IF NOT EXISTS backtest_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_id VARCHAR(50) NOT NULL, -- 回测批次ID
    trade_date DATE NOT NULL,
    security_id INTEGER NOT NULL,
    
    -- 交易信息
    action VARCHAR(10) NOT NULL, -- BUY, SELL
    shares INTEGER NOT NULL,
    price DECIMAL(10,3) NOT NULL,
    amount DECIMAL(20,2) NOT NULL,
    
    -- 成本费用
    commission DECIMAL(10,2),
    stamp_tax DECIMAL(10,2),
    transfer_fee DECIMAL(10,2),
    
    -- 盈亏信息
    profit DECIMAL(20,2),
    profit_pct DECIMAL(10,4),
    
    -- 交易原因
    reason VARCHAR(200),
    strategy_name VARCHAR(50),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (security_id) REFERENCES securities(id)
);

CREATE INDEX idx_trades_backtest ON backtest_trades(backtest_id);
CREATE INDEX idx_trades_date ON backtest_trades(trade_date);

-- 6. 回测结果汇总表
CREATE TABLE IF NOT EXISTS backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_id VARCHAR(50) NOT NULL UNIQUE,
    strategy_name VARCHAR(50) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    
    -- 资金情况
    initial_capital DECIMAL(20,2) NOT NULL,
    final_capital DECIMAL(20,2) NOT NULL,
    
    -- 收益指标
    total_return DECIMAL(10,4),
    annual_return DECIMAL(10,4),
    max_drawdown DECIMAL(10,4),
    
    -- 风险指标
    sharpe_ratio DECIMAL(10,3),
    sortino_ratio DECIMAL(10,3),
    volatility DECIMAL(10,4),
    
    -- 交易统计
    total_trades INTEGER,
    win_rate DECIMAL(10,4),
    profit_loss_ratio DECIMAL(10,3),
    
    -- 详细结果（JSON格式）
    detailed_metrics TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 7. 数据更新日志表
CREATE TABLE IF NOT EXISTS data_update_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    update_date DATE NOT NULL,
    update_type VARCHAR(20) NOT NULL, -- DAILY, HISTORICAL, MANUAL
    securities_updated INTEGER,
    records_added INTEGER,
    records_updated INTEGER,
    status VARCHAR(20) NOT NULL, -- SUCCESS, FAILED, PARTIAL
    error_message TEXT,
    duration_seconds INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建视图：最新价格视图
CREATE VIEW IF NOT EXISTS latest_quotes AS
SELECT 
    s.code,
    s.name,
    s.type,
    q.trade_date,
    q.close,
    q.price_change_pct,
    q.volume,
    q.is_limit_up,
    q.is_limit_down
FROM securities s
JOIN daily_quotes q ON s.id = q.security_id
WHERE q.trade_date = (
    SELECT MAX(trade_date) 
    FROM daily_quotes 
    WHERE security_id = s.id
)
AND s.is_active = 1;

-- 创建视图：技术指标综合视图
CREATE VIEW IF NOT EXISTS technical_overview AS
SELECT 
    s.code,
    s.name,
    q.trade_date,
    q.close,
    q.ma5,
    q.ma20,
    t.kdj_k,
    t.kdj_d,
    t.kdj_j,
    t.macd_macd,
    t.rsi12,
    t.bbi
FROM securities s
JOIN daily_quotes q ON s.id = q.security_id
LEFT JOIN technical_indicators t ON s.id = t.security_id AND q.trade_date = t.trade_date
WHERE s.is_active = 1;