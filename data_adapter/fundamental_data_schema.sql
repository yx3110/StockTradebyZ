-- 基本面数据库表结构设计
-- 用于存储从Tushare获取的基本面和指数数据

-- 1. 股票基本信息扩展表（已有securities表的补充）
CREATE TABLE IF NOT EXISTS stock_basic_info (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    security_id INTEGER NOT NULL,
    ts_code VARCHAR(12) NOT NULL,  -- Tushare股票代码 (如000001.SZ)
    market VARCHAR(10),             -- 市场(主板/创业板/科创板等)
    list_status VARCHAR(1),         -- 上市状态 L上市 D退市 P暂停上市
    fullname VARCHAR(100),          -- 股票全称
    enname VARCHAR(200),            -- 英文全称
    setup_date DATE,                -- 成立日期
    employees INTEGER,              -- 员工数
    main_business TEXT,             -- 主要业务
    business_scope TEXT,            -- 经营范围
    website VARCHAR(200),           -- 公司主页
    email VARCHAR(100),             -- 电邮
    office VARCHAR(200),            -- 办公室地址
    ann_date DATE,                  -- 公告日期
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (security_id) REFERENCES securities(id),
    UNIQUE(security_id)
);

-- 2. 每日基本面数据表
CREATE TABLE IF NOT EXISTS daily_basic (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    security_id INTEGER NOT NULL,
    trade_date DATE NOT NULL,
    close DECIMAL(10,3),           -- 当日收盘价
    turnover_rate DECIMAL(10,4),   -- 换手率（%）
    turnover_rate_f DECIMAL(10,4), -- 换手率（自由流通股）
    volume_ratio DECIMAL(10,4),    -- 量比
    pe DECIMAL(10,4),              -- 市盈率（总市值/净利润TTM）
    pe_ttm DECIMAL(10,4),          -- 市盈率（TTM）
    pb DECIMAL(10,4),              -- 市净率（总市值/净资产）
    ps DECIMAL(10,4),              -- 市销率
    ps_ttm DECIMAL(10,4),          -- 市销率（TTM）
    dv_ratio DECIMAL(10,4),        -- 股息率（%）
    dv_ttm DECIMAL(10,4),          -- 股息率（TTM）（%）
    total_share DECIMAL(20,4),     -- 总股本（万股）
    float_share DECIMAL(20,4),     -- 流通股本（万股）
    free_share DECIMAL(20,4),      -- 自由流通股本（万）
    total_mv DECIMAL(20,4),        -- 总市值（万元）
    circ_mv DECIMAL(20,4),         -- 流通市值（万元）
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (security_id) REFERENCES securities(id),
    UNIQUE(security_id, trade_date)
);

-- 3. 财务指标数据表（季度）
CREATE TABLE IF NOT EXISTS financial_indicator (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    security_id INTEGER NOT NULL,
    ann_date DATE,                 -- 公告日期
    end_date DATE NOT NULL,        -- 报告期
    eps DECIMAL(10,4),             -- 基本每股收益
    dt_eps DECIMAL(10,4),          -- 稀释每股收益
    total_revenue_ps DECIMAL(10,4), -- 每股营业总收入
    revenue_ps DECIMAL(10,4),      -- 每股营业收入
    capital_rese_ps DECIMAL(10,4), -- 每股资本公积
    surplus_rese_ps DECIMAL(10,4), -- 每股盈余公积
    undist_profit_ps DECIMAL(10,4), -- 每股未分配利润
    extra_item DECIMAL(20,2),      -- 非经常性损益
    profit_dedt DECIMAL(20,2),     -- 扣除非经常性损益后的净利润
    gross_margin DECIMAL(10,4),    -- 毛利率
    current_ratio DECIMAL(10,4),   -- 流动比率
    quick_ratio DECIMAL(10,4),     -- 速动比率
    cash_ratio DECIMAL(10,4),      -- 保守速动比率
    invturn_days DECIMAL(10,2),    -- 存货周转天数
    arturn_days DECIMAL(10,2),     -- 应收账款周转天数
    inv_turn DECIMAL(10,4),        -- 存货周转率
    ar_turn DECIMAL(10,4),         -- 应收账款周转率
    ca_turn DECIMAL(10,4),         -- 流动资产周转率
    fa_turn DECIMAL(10,4),         -- 固定资产周转率
    assets_turn DECIMAL(10,4),     -- 总资产周转率
    op_income DECIMAL(20,2),       -- 经营活动净收益
    valuechange_income DECIMAL(20,2), -- 价值变动净收益
    interst_income DECIMAL(20,2),  -- 利息费用
    daa DECIMAL(20,2),             -- 总资产
    ebit DECIMAL(20,2),            -- 息税前利润
    ebitda DECIMAL(20,2),          -- 息税折旧摊销前利润
    fcff DECIMAL(20,2),            -- 企业自由现金流量
    fcfe DECIMAL(20,2),            -- 股权自由现金流量
    current_exint DECIMAL(20,2),   -- 无息流动负债
    noncurrent_exint DECIMAL(20,2), -- 无息非流动负债
    interestdebt DECIMAL(20,2),    -- 带息债务
    netdebt DECIMAL(20,2),         -- 净债务
    tangible_asset DECIMAL(20,2),  -- 有形资产
    working_capital DECIMAL(20,2), -- 营运资金
    networking_capital DECIMAL(20,2), -- 营运流动资本
    invest_capital DECIMAL(20,2),  -- 全部投入资本
    retained_earnings DECIMAL(20,2), -- 留存收益
    diluted2_eps DECIMAL(10,4),    -- 期末摊薄每股收益
    bps DECIMAL(10,4),             -- 每股净资产
    ocfps DECIMAL(10,4),           -- 每股经营活动产生的现金流量净额
    retainedps DECIMAL(10,4),      -- 每股留存收益
    cfps DECIMAL(10,4),            -- 每股现金流量净额
    ebit_ps DECIMAL(10,4),         -- 每股息税前利润
    fcff_ps DECIMAL(10,4),         -- 每股企业自由现金流量
    fcfe_ps DECIMAL(10,4),         -- 每股股东自由现金流量
    netprofit_margin DECIMAL(10,4), -- 销售净利率
    grossprofit_margin DECIMAL(10,4), -- 销售毛利率
    cogs_of_sales DECIMAL(10,4),   -- 销售成本率
    expense_of_sales DECIMAL(10,4), -- 销售期间费用率
    profit_to_gr DECIMAL(10,4),    -- 净利润/营业总收入
    saleexp_to_gr DECIMAL(10,4),   -- 销售费用/营业总收入
    adminexp_of_gr DECIMAL(10,4),  -- 管理费用/营业总收入
    finaexp_of_gr DECIMAL(10,4),   -- 财务费用/营业总收入
    impai_ttm DECIMAL(20,2),       -- 资产减值损失/营业总收入
    gc_of_gr DECIMAL(10,4),        -- 营业总成本/营业总收入
    op_of_gr DECIMAL(10,4),        -- 营业利润/营业总收入
    ebit_of_gr DECIMAL(10,4),      -- 息税前利润/营业总收入
    roe DECIMAL(10,4),             -- 净资产收益率
    roe_waa DECIMAL(10,4),         -- 加权平均净资产收益率
    roe_dt DECIMAL(10,4),          -- 净资产收益率(扣除非经常损益)
    roa DECIMAL(10,4),             -- 总资产报酬率
    npta DECIMAL(10,4),            -- 总资产净利润
    roic DECIMAL(10,4),            -- 投入资本回报率
    roe_yearly DECIMAL(10,4),      -- 年化净资产收益率
    roa2_yearly DECIMAL(10,4),     -- 年化总资产报酬率
    roe_avg DECIMAL(10,4),         -- 平均净资产收益率(增发条件)
    opincome_of_ebt DECIMAL(10,4), -- 经营活动净收益/利润总额
    investincome_of_ebt DECIMAL(10,4), -- 价值变动净收益/利润总额
    n_op_profit_of_ebt DECIMAL(10,4), -- 营业外收支净额/利润总额
    tax_to_ebt DECIMAL(10,4),      -- 所得税/利润总额
    dtprofit_to_profit DECIMAL(10,4), -- 扣除非经常损益后的净利润/净利润
    salescash_to_or DECIMAL(10,4), -- 销售商品提供劳务收到的现金/营业收入
    ocf_to_or DECIMAL(10,4),       -- 经营活动产生的现金流量净额/营业收入
    ocf_to_opincome DECIMAL(10,4), -- 经营活动产生的现金流量净额/经营活动净收益
    capitalized_to_da DECIMAL(10,4), -- 资本化支出/折旧和摊销
    debt_to_assets DECIMAL(10,4),  -- 资产负债率
    assets_to_eqt DECIMAL(10,4),   -- 权益乘数
    dp_assets_to_eqt DECIMAL(10,4), -- 权益乘数(杜邦分析)
    ca_to_assets DECIMAL(10,4),    -- 流动资产/总资产
    nca_to_assets DECIMAL(10,4),   -- 非流动资产/总资产
    tbassets_to_totalassets DECIMAL(10,4), -- 有形资产/总资产
    int_to_talcap DECIMAL(10,4),   -- 带息债务/全部投入资本
    eqt_to_talcapital DECIMAL(10,4), -- 归属于母公司的股东权益/全部投入资本
    currentdebt_to_debt DECIMAL(10,4), -- 流动负债/负债合计
    longdeb_to_debt DECIMAL(10,4), -- 非流动负债/负债合计
    ocf_to_shortdebt DECIMAL(10,4), -- 经营活动产生的现金流量净额/流动负债
    debt_to_eqt DECIMAL(10,4),     -- 产权比率
    eqt_to_debt DECIMAL(10,4),     -- 归属于母公司的股东权益/负债合计
    eqt_to_interestdebt DECIMAL(10,4), -- 归属于母公司的股东权益/带息债务
    tangibleasset_to_debt DECIMAL(10,4), -- 有形资产/负债合计
    tangasset_to_intdebt DECIMAL(10,4), -- 有形资产/带息债务
    tangibleasset_to_netdebt DECIMAL(10,4), -- 有形资产/净债务
    ocf_to_debt DECIMAL(10,4),     -- 经营活动产生的现金流量净额/负债合计
    ocf_to_interestdebt DECIMAL(10,4), -- 经营活动产生的现金流量净额/带息债务
    ocf_to_netdebt DECIMAL(10,4),  -- 经营活动产生的现金流量净额/净债务
    ebit_to_interest DECIMAL(10,4), -- 已获利息倍数(EBIT/利息费用)
    longdebt_to_workingcapital DECIMAL(10,4), -- 长期债务与营运资金比率
    ebitda_to_debt DECIMAL(10,4),  -- 息税折旧摊销前利润/负债合计
    turn_days DECIMAL(10,2),       -- 营业周期
    roa_yearly DECIMAL(10,4),      -- 年化总资产净利率
    roa_dp DECIMAL(10,4),          -- 总资产净利率(杜邦分析)
    fixed_assets DECIMAL(20,2),    -- 固定资产合计
    profit_prefin_exp DECIMAL(20,2), -- 扣除财务费用前营业利润
    non_op_profit DECIMAL(20,2),   -- 非营业利润
    op_to_ebt DECIMAL(10,4),       -- 营业利润／利润总额
    nop_to_ebt DECIMAL(10,4),      -- 非营业利润／利润总额
    ocf_to_profit DECIMAL(10,4),   -- 经营活动产生的现金流量净额／营业利润
    cash_to_liqdebt DECIMAL(10,4), -- 货币资金／流动负债
    cash_to_liqdebt_withinterest DECIMAL(10,4), -- 货币资金／带息流动负债
    op_to_liqdebt DECIMAL(10,4),   -- 营业利润／流动负债
    op_to_debt DECIMAL(10,4),      -- 营业利润／负债合计
    roic_yearly DECIMAL(10,4),     -- 年化投入资本回报率
    total_fa_trun DECIMAL(10,4),   -- 固定资产合计周转率
    profit_to_op DECIMAL(10,4),    -- 利润总额／营业收入
    q_opincome DECIMAL(20,2),      -- 经营活动单季度净收益
    q_investincome DECIMAL(20,2),  -- 价值变动单季度净收益
    q_dtprofit DECIMAL(20,2),      -- 扣除非经常损益后的单季度净利润
    q_eps DECIMAL(10,4),           -- 每股收益(单季度)
    q_netprofit_margin DECIMAL(10,4), -- 销售净利率(单季度)
    q_gsprofit_margin DECIMAL(10,4), -- 销售毛利率(单季度)
    q_exp_to_sales DECIMAL(10,4),  -- 销售期间费用率(单季度)
    q_profit_to_gr DECIMAL(10,4),  -- 净利润／营业总收入(单季度)
    q_saleexp_to_gr DECIMAL(10,4), -- 销售费用／营业总收入 (单季度)
    q_adminexp_to_gr DECIMAL(10,4), -- 管理费用／营业总收入 (单季度)
    q_finaexp_to_gr DECIMAL(10,4), -- 财务费用／营业总收入 (单季度)
    q_gc_to_gr DECIMAL(10,4),      -- 营业总成本／营业总收入 (单季度)
    q_op_to_gr DECIMAL(10,4),      -- 营业利润／营业总收入(单季度)
    q_roe DECIMAL(10,4),           -- 净资产收益率(单季度)
    q_dt_roe DECIMAL(10,4),        -- 净资产单季度收益率(扣除非经常损益)
    q_npta DECIMAL(10,4),          -- 总资产净利润(单季度)
    q_opincome_to_ebt DECIMAL(10,4), -- 经营活动净收益／利润总额(单季度)
    q_investincome_to_ebt DECIMAL(10,4), -- 价值变动净收益／利润总额(单季度)
    q_dtprofit_to_profit DECIMAL(10,4), -- 扣除非经常损益后的净利润／净利润(单季度)
    q_salescash_to_or DECIMAL(10,4), -- 销售商品提供劳务收到的现金／营业收入(单季度)
    q_ocf_to_sales DECIMAL(10,4),  -- 经营活动产生的现金流量净额／营业收入(单季度)
    q_ocf_to_or DECIMAL(10,4),     -- 经营活动产生的现金流量净额／经营活动净收益(单季度)
    basic_eps_yoy DECIMAL(10,4),   -- 基本每股收益同比增长率(%)
    dt_eps_yoy DECIMAL(10,4),      -- 稀释每股收益同比增长率(%)
    cfps_yoy DECIMAL(10,4),        -- 每股经营活动产生的现金流量净额同比增长率(%)
    op_yoy DECIMAL(10,4),          -- 营业利润同比增长率(%)
    ebt_yoy DECIMAL(10,4),         -- 利润总额同比增长率(%)
    netprofit_yoy DECIMAL(10,4),   -- 归属母公司股东的净利润同比增长率(%)
    dt_netprofit_yoy DECIMAL(10,4), -- 归属母公司股东的净利润-扣除非经常损益同比增长率(%)
    ocf_yoy DECIMAL(10,4),         -- 经营活动产生的现金流量净额同比增长率(%)
    roe_yoy DECIMAL(10,4),         -- 净资产收益率(摊薄)同比增长率(%)
    bps_yoy DECIMAL(10,4),         -- 每股净资产相对年初增长率(%)
    assets_yoy DECIMAL(10,4),      -- 资产总计相对年初增长率(%)
    eqt_yoy DECIMAL(10,4),         -- 归属母公司的股东权益相对年初增长率(%)
    tr_yoy DECIMAL(10,4),          -- 营业总收入同比增长率(%)
    or_yoy DECIMAL(10,4),          -- 营业收入同比增长率(%)
    q_gr_yoy DECIMAL(10,4),        -- 营业总收入同比增长率(%)(单季度)
    q_gr_qoq DECIMAL(10,4),        -- 营业总收入环比增长率(%)(单季度)
    q_sales_yoy DECIMAL(10,4),     -- 营业收入同比增长率(%)(单季度)
    q_sales_qoq DECIMAL(10,4),     -- 营业收入环比增长率(%)(单季度)
    q_op_yoy DECIMAL(10,4),        -- 营业利润同比增长率(%)(单季度)
    q_op_qoq DECIMAL(10,4),        -- 营业利润环比增长率(%)(单季度)
    q_profit_yoy DECIMAL(10,4),    -- 净利润同比增长率(%)(单季度)
    q_profit_qoq DECIMAL(10,4),    -- 净利润环比增长率(%)(单季度)
    q_netprofit_yoy DECIMAL(10,4), -- 归属母公司股东的净利润同比增长率(%)(单季度)
    q_netprofit_qoq DECIMAL(10,4), -- 归属母公司股东的净利润环比增长率(%)(单季度)
    equity_yoy DECIMAL(10,4),      -- 净资产同比增长率
    rd_exp DECIMAL(20,2),          -- 研发费用
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (security_id) REFERENCES securities(id),
    UNIQUE(security_id, end_date)
);

-- 4. 大盘指数数据表
CREATE TABLE IF NOT EXISTS market_indices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_code VARCHAR(12) NOT NULL,  -- 指数代码
    name VARCHAR(50) NOT NULL,     -- 指数名称
    fullname VARCHAR(100),         -- 指数全称
    market VARCHAR(10),            -- 市场
    publisher VARCHAR(50),         -- 发布方
    index_type VARCHAR(20),        -- 指数类型
    category VARCHAR(20),          -- 指数分类
    base_date DATE,                -- 基期
    base_point DECIMAL(10,4),      -- 基点
    list_date DATE,                -- 发布日期
    weight_rule VARCHAR(100),      -- 加权方式
    desc_detail TEXT,              -- 描述
    exp_date DATE,                 -- 终止日期
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code)
);

-- 5. 指数日线数据表
CREATE TABLE IF NOT EXISTS index_daily (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    index_id INTEGER NOT NULL,
    trade_date DATE NOT NULL,
    close DECIMAL(10,4),           -- 收盘点位
    open DECIMAL(10,4),            -- 开盘点位
    high DECIMAL(10,4),            -- 最高点位
    low DECIMAL(10,4),             -- 最低点位
    pre_close DECIMAL(10,4),       -- 昨收盘点位
    change DECIMAL(10,4),          -- 涨跌点
    pct_chg DECIMAL(10,4),         -- 涨跌幅
    vol DECIMAL(20,2),             -- 成交量（手）
    amount DECIMAL(20,2),          -- 成交额（千元）
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (index_id) REFERENCES market_indices(id),
    UNIQUE(index_id, trade_date)
);

-- 6. 行业分类数据表
CREATE TABLE IF NOT EXISTS industry_classify (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    security_id INTEGER NOT NULL,
    ts_code VARCHAR(12) NOT NULL,
    industry_name VARCHAR(50),     -- 行业名称
    industry_code VARCHAR(20),     -- 行业代码
    level VARCHAR(10),             -- 行业级别
    industry_parent VARCHAR(50),   -- 上级行业
    src VARCHAR(20),               -- 数据源
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (security_id) REFERENCES securities(id)
);

-- 7. 创建索引优化查询性能
CREATE INDEX IF NOT EXISTS idx_daily_basic_date ON daily_basic(trade_date);
CREATE INDEX IF NOT EXISTS idx_daily_basic_security ON daily_basic(security_id);
CREATE INDEX IF NOT EXISTS idx_financial_indicator_date ON financial_indicator(end_date);
CREATE INDEX IF NOT EXISTS idx_financial_indicator_security ON financial_indicator(security_id);
CREATE INDEX IF NOT EXISTS idx_index_daily_date ON index_daily(trade_date);
CREATE INDEX IF NOT EXISTS idx_index_daily_index ON index_daily(index_id);
CREATE INDEX IF NOT EXISTS idx_stock_basic_info_security ON stock_basic_info(security_id);