-- =====================================================
-- 因子数据库扩展架构设计
-- 支持预计算因子存储、版本管理和高效查询
-- =====================================================

-- 1. 因子定义表（元数据）
CREATE TABLE IF NOT EXISTS factor_definitions (
    factor_id INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_name VARCHAR(50) NOT NULL UNIQUE,
    factor_category VARCHAR(30) NOT NULL, -- technical, fundamental, market, sentiment, squeeze, custom
    factor_type VARCHAR(20) NOT NULL, -- numeric, boolean, categorical
    description TEXT,
    formula TEXT, -- 计算公式或方法描述
    dependencies TEXT, -- 依赖的其他因子或原始数据，JSON格式
    version VARCHAR(10) NOT NULL DEFAULT '1.0',
    created_date DATE DEFAULT CURRENT_DATE,
    update_frequency VARCHAR(20), -- daily, weekly, monthly, quarterly
    is_active BOOLEAN DEFAULT 1,
    UNIQUE(factor_name, version)
);

-- 2. 技术因子表（每日预计算）
CREATE TABLE IF NOT EXISTS technical_factors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    security_id INTEGER NOT NULL,
    trade_date DATE NOT NULL,
    
    -- 动量类因子
    momentum_5d REAL,      -- 5日动量
    momentum_10d REAL,     -- 10日动量
    momentum_20d REAL,     -- 20日动量
    momentum_60d REAL,     -- 60日动量
    momentum_acceleration REAL, -- 动量加速度
    
    -- 均值回归因子
    mean_reversion_score REAL,  -- 均值回归得分
    price_to_ma5 REAL,         -- 价格/MA5比率
    price_to_ma20 REAL,        -- 价格/MA20比率
    price_to_ma60 REAL,        -- 价格/MA60比率
    
    -- 波动率因子
    volatility_5d REAL,        -- 5日波动率
    volatility_20d REAL,       -- 20日波动率
    volatility_60d REAL,       -- 60日波动率
    volatility_ratio REAL,     -- 短期/长期波动率比
    
    -- 成交量因子
    volume_ratio_5d REAL,      -- 5日成交量比率
    volume_ratio_20d REAL,     -- 20日成交量比率
    volume_momentum REAL,      -- 成交量动量
    volume_volatility REAL,   -- 成交量波动率
    
    -- 价格形态因子
    support_level REAL,        -- 支撑位
    resistance_level REAL,     -- 阻力位
    price_position REAL,       -- 价格位置（0-1，最低到最高）
    breakout_strength REAL,    -- 突破强度
    
    -- 技术指标衍生因子
    rsi_divergence REAL,       -- RSI背离信号
    macd_histogram_slope REAL, -- MACD柱状图斜率
    kdj_golden_cross BOOLEAN,  -- KDJ金叉
    kdj_death_cross BOOLEAN,   -- KDJ死叉
    bbi_trend_strength REAL,   -- BBI趋势强度
    
    -- 挤压动量因子（V4新增）
    squeeze_state INTEGER,      -- -1:释放, 0:正常, 1:挤压
    squeeze_duration INTEGER,   -- 挤压持续天数
    squeeze_momentum REAL,      -- 挤压动量值
    squeeze_momentum_change REAL, -- 动量变化率
    squeeze_release_signal BOOLEAN, -- 挤压释放信号
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (security_id) REFERENCES securities(id),
    UNIQUE(security_id, trade_date)
);

-- 3. 基本面因子表（季度/年度更新）
CREATE TABLE IF NOT EXISTS fundamental_factors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    security_id INTEGER NOT NULL,
    report_date DATE NOT NULL,     -- 财报日期
    
    -- 估值因子
    pe_ttm_percentile REAL,        -- PE历史百分位
    pb_percentile REAL,            -- PB历史百分位
    ps_ttm_percentile REAL,        -- PS历史百分位
    peg_ratio REAL,                -- PEG比率
    ev_to_ebitda REAL,            -- EV/EBITDA
    
    -- 盈利能力因子
    roe_trend REAL,                -- ROE趋势
    roa_trend REAL,                -- ROA趋势
    gross_margin_trend REAL,       -- 毛利率趋势
    net_margin_trend REAL,         -- 净利率趋势
    
    -- 成长因子
    revenue_growth_3y REAL,        -- 3年营收增长率
    profit_growth_3y REAL,         -- 3年利润增长率
    revenue_acceleration REAL,     -- 营收增长加速度
    earnings_surprise REAL,        -- 盈利超预期程度
    
    -- 质量因子
    debt_to_equity_change REAL,    -- 负债率变化
    current_ratio_trend REAL,      -- 流动比率趋势
    inventory_turnover_trend REAL, -- 存货周转率趋势
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (security_id) REFERENCES securities(id),
    UNIQUE(security_id, report_date)
);

-- 4. 市场因子表（每日计算）
CREATE TABLE IF NOT EXISTS market_factors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    security_id INTEGER NOT NULL,
    trade_date DATE NOT NULL,
    
    -- 相对强度因子
    relative_strength_index REAL,  -- 相对强度指数
    alpha_60d REAL,                -- 60日阿尔法
    beta_60d REAL,                 -- 60日贝塔
    sharpe_ratio_60d REAL,         -- 60日夏普比率
    
    -- 行业/板块因子
    industry_rank REAL,            -- 行业内排名(0-1)
    sector_momentum REAL,          -- 板块动量
    sector_relative_strength REAL, -- 板块相对强度
    
    -- 市场情绪因子
    correlation_with_market REAL,  -- 与大盘相关性
    idiosyncratic_volatility REAL, -- 特质波动率
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (security_id) REFERENCES securities(id),
    UNIQUE(security_id, trade_date)
);

-- 5. 综合因子评分表（神经网络输出）
CREATE TABLE IF NOT EXISTS factor_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    security_id INTEGER NOT NULL,
    trade_date DATE NOT NULL,
    
    -- 传统加权评分
    weighted_score_v3 REAL,        -- V3版本评分
    weighted_score_v4 REAL,        -- V4版本评分
    
    -- 神经网络评分
    nn_score_v1 REAL,              -- 神经网络V1评分
    nn_confidence REAL,            -- 预测置信度
    
    -- 分类评分
    technical_score REAL,          -- 技术面评分
    fundamental_score REAL,        -- 基本面评分
    market_score REAL,             -- 市场面评分
    sentiment_score REAL,          -- 情绪面评分
    
    -- 综合评分
    final_score REAL,              -- 最终综合评分
    score_percentile REAL,         -- 评分百分位
    
    -- 预测目标
    predicted_return_1d REAL,      -- 预测1日收益
    predicted_return_5d REAL,      -- 预测5日收益
    predicted_return_20d REAL,     -- 预测20日收益
    
    model_version VARCHAR(20),     -- 使用的模型版本
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (security_id) REFERENCES securities(id),
    UNIQUE(security_id, trade_date, model_version)
);

-- 6. 因子计算日志表
CREATE TABLE IF NOT EXISTS factor_calculation_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    calculation_date DATE NOT NULL,
    factor_category VARCHAR(30),
    factor_count INTEGER,
    securities_processed INTEGER,
    calculation_time_seconds REAL,
    error_count INTEGER DEFAULT 0,
    error_details TEXT,
    status VARCHAR(20), -- pending, running, completed, failed
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 7. 因子回测表（存储因子有效性分析结果）
CREATE TABLE IF NOT EXISTS factor_backtest (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_name VARCHAR(50) NOT NULL,
    test_period_start DATE,
    test_period_end DATE,
    
    -- 相关性指标
    correlation_1d REAL,           -- 与1日收益相关性
    correlation_5d REAL,           -- 与5日收益相关性
    correlation_20d REAL,          -- 与20日收益相关性
    
    -- IC指标
    ic_mean REAL,                  -- 平均IC值
    ic_std REAL,                   -- IC标准差
    ic_ir REAL,                    -- IC信息比率
    
    -- 分组测试
    top_quintile_return REAL,      -- 最高五分位收益
    bottom_quintile_return REAL,   -- 最低五分位收益
    long_short_return REAL,        -- 多空组合收益
    
    test_date DATE DEFAULT CURRENT_DATE,
    FOREIGN KEY (factor_name) REFERENCES factor_definitions(factor_name)
);

-- 创建索引以提高查询性能
CREATE INDEX IF NOT EXISTS idx_tech_factors_date ON technical_factors(trade_date);
CREATE INDEX IF NOT EXISTS idx_tech_factors_security ON technical_factors(security_id);
CREATE INDEX IF NOT EXISTS idx_tech_factors_security_date ON technical_factors(security_id, trade_date);

CREATE INDEX IF NOT EXISTS idx_fundamental_factors_date ON fundamental_factors(report_date);
CREATE INDEX IF NOT EXISTS idx_fundamental_factors_security ON fundamental_factors(security_id);

CREATE INDEX IF NOT EXISTS idx_market_factors_date ON market_factors(trade_date);
CREATE INDEX IF NOT EXISTS idx_market_factors_security ON market_factors(security_id);
CREATE INDEX IF NOT EXISTS idx_market_factors_security_date ON market_factors(security_id, trade_date);

CREATE INDEX IF NOT EXISTS idx_factor_scores_date ON factor_scores(trade_date);
CREATE INDEX IF NOT EXISTS idx_factor_scores_security ON factor_scores(security_id);
CREATE INDEX IF NOT EXISTS idx_factor_scores_final ON factor_scores(trade_date, final_score DESC);

-- 创建视图：最新因子数据
CREATE VIEW IF NOT EXISTS latest_factors AS
SELECT 
    s.code,
    s.name,
    tf.*,
    mf.*,
    fs.final_score,
    fs.score_percentile
FROM securities s
LEFT JOIN technical_factors tf ON s.id = tf.security_id
LEFT JOIN market_factors mf ON s.id = mf.security_id AND mf.trade_date = tf.trade_date
LEFT JOIN factor_scores fs ON s.id = fs.security_id AND fs.trade_date = tf.trade_date
WHERE tf.trade_date = (SELECT MAX(trade_date) FROM technical_factors);

-- 创建视图：因子时间序列
CREATE VIEW IF NOT EXISTS factor_timeseries AS
SELECT 
    s.code,
    tf.trade_date,
    tf.momentum_5d,
    tf.momentum_20d,
    tf.volatility_20d,
    tf.volume_ratio_20d,
    tf.squeeze_state,
    mf.relative_strength_index,
    mf.beta_60d,
    fs.final_score
FROM securities s
JOIN technical_factors tf ON s.id = tf.security_id
LEFT JOIN market_factors mf ON s.id = mf.security_id AND mf.trade_date = tf.trade_date
LEFT JOIN factor_scores fs ON s.id = fs.security_id AND fs.trade_date = tf.trade_date
ORDER BY s.code, tf.trade_date DESC;