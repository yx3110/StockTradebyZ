-- V3.9特征缓存表设计 (v2 - JSON存储方案)
-- 优势：灵活，无需提前定义所有字段

CREATE TABLE IF NOT EXISTS v39_feature_cache (
    -- 主键
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- 标识字段
    code TEXT NOT NULL,
    trade_date TEXT NOT NULL,

    -- 特征JSON (存储所有42个特征)
    features_json TEXT NOT NULL,

    -- 标签
    label_5d REAL,

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
