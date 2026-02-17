-- V3.9特征缓存表设计
-- 目的：预计算所有特征，避免训练时重复计算

CREATE TABLE IF NOT EXISTS v39_feature_cache (
    -- 主键
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- 标识字段
    code TEXT NOT NULL,
    trade_date TEXT NOT NULL,

    -- 24个技术特征 (technical_features)
    rsi_6 REAL,
    rsi_12 REAL,
    rsi_24 REAL,
    macd_dif REAL,
    macd_dea REAL,
    macd_hist REAL,
    kdj_k REAL,
    kdj_d REAL,
    kdj_j REAL,
    ma_5 REAL,
    ma_10 REAL,
    ma_20 REAL,
    ma_60 REAL,
    ema_12 REAL,
    ema_26 REAL,
    boll_upper REAL,
    boll_middle REAL,
    boll_lower REAL,
    bbi REAL,
    atr_14 REAL,
    adx_14 REAL,
    cci_14 REAL,
    willr_14 REAL,
    obv REAL,

    -- 10个基本面特征 (fundamental_features)
    pe_ttm REAL,
    pb REAL,
    ps_ttm REAL,
    dv_ttm REAL,
    total_mv REAL,
    circ_mv REAL,
    turnover_rate_f REAL,
    volume_ratio REAL,
    eps REAL,
    roe REAL,

    -- 8个市场特征 (market_features)
    market_return_5d REAL,
    market_return_10d REAL,
    market_return_20d REAL,
    relative_strength_5d REAL,
    relative_strength_10d REAL,
    relative_strength_20d REAL,
    volume_5d_avg REAL,
    volume_10d_avg REAL,

    -- 标签 (用于训练)
    label_5d REAL,  -- 5日收益率

    -- 元信息
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

    -- 联合唯一索引
    UNIQUE(code, trade_date)
);

-- 创建索引加速查询
CREATE INDEX IF NOT EXISTS idx_v39_code ON v39_feature_cache(code);
CREATE INDEX IF NOT EXISTS idx_v39_date ON v39_feature_cache(trade_date);
CREATE INDEX IF NOT EXISTS idx_v39_code_date ON v39_feature_cache(code, trade_date);
CREATE INDEX IF NOT EXISTS idx_v39_label ON v39_feature_cache(label_5d);

-- 统计信息表
CREATE TABLE IF NOT EXISTS v39_feature_cache_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    total_records INTEGER,
    date_range_start TEXT,
    date_range_end TEXT,
    stock_count INTEGER,
    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
);
