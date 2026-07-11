# 市场行情三页面 (Market Pulse)

> 2026-07-11 上线。对标 stock.maidrem.top (A股全景) 重建的 板块排行 / 资金流向 / 全A行情 三个 webapp 页面。

## 页面与功能

| 页面 | 路由 | 功能 |
|---|---|---|
| 板块排行 | `/market-rotation` | 轮动日历 (每日涨幅TOP25板块, 板块颜色恒定) + 创新高日历 (每日创20日新高个股数排名) |
| 资金流向 | `/market-fundflow` | 每日主力净流入/净流出 TOP20 板块日历, 单位亿元 (主力=大单+超大单) |
| 全A行情 | `/market-sentiment` | 涨停/跌停/炸板家数、涨跌家数、沪深300 & 中证2000 成分股20日新高数 (联动折线图) |

三个 taxonomy: `sw_l1` (申万一级31个) / `sw_l2` (申万二级~124个) / `concept` (东财概念~497个)。

## 数据链路

```
Tushare (index_member_all / dc_index / dc_member / index_weight / limit_list_d)
    └→ fetch_data/market_board_fetcher.py  (--backfill 首次 / --daily 日更)
        └→ 表: sw_industry(+L2/L3列) · dc_index_daily · dc_member ·
               index_weight_snapshot · limit_list_daily
本地 daily_quotes + moneyflow_daily + 上述表
    └→ scripts/build_market_pulse.py  (按月分块, 增量)
        └→ 表: sector_daily_stats (taxonomy×日×板块: 等权涨幅/主力净流入亿/新高数/涨跌家数)
               market_sentiment_daily (日: 涨停/跌停/炸板/涨跌家数/两指数新高数)
webapp/api/market.py  (蓝图 /api/market/rotation|fundflow|sentiment|status)
    └→ templates/market_*.html + static/js/market_common.js (固定板块配色+日历渲染)
```

日更接入: `quick_daily_update.py` 步骤18 (在 moneyflow 更新之后)。

## 关键决策与陷阱

- **板块涨幅口径**: 申万 L1/L2 用成分股等权平均 (点时成分, in_date/out_date 过滤);
  概念优先用 `dc_index_daily.pct_change` 官方值, 缺失回退等权平均。与东财官方排名验证高度一致。
- **主力净流入口径 = 东财 `moneyflow_dc`** (表 `moneyflow_dc_daily.net_amount`, 万元→亿,
  2023-09-11 起)。2026-07-11 实证: 按成分聚合可 99.5%+ 复现东财官方板块资金流
  (`moneyflow_ind_dc`), 数字可与东财 App 直接对照。**曾用 Tushare 自算 `moneyflow_daily`
  (lg+elg 口径), 因无处交叉验证已切换** — 该表仅供 NG 因子, 两口径同股票差 30-60%, 勿混用。
- **`moneyflow_daily.code_6` 曾全 NULL** — ng_cache_updater 写入端漏填 (已修 + 回填 285k 行)。
- **`daily_quotes.is_limit_up` 从未回填 (全为0)** — 涨停数改用 Tushare `limit_list_d`
  (U涨停/D跌停/Z炸板, 数据起点 2024-01-02), 存 `limit_list_daily`。
- **`moneyflow_ind_dc` 历史只有行业无概念** (~86行/天), 不能做概念资金流历史 — 故全部走本地聚合。
- **dc_member 是当前快照** — 概念成分应用于历史日期存在漂移 (行业惯例可接受); 申万成分是点时准确的。
- **并发写锁**: 与其他回填任务并发时 busy_timeout 不够, fetcher/builder 都带 `_write_retry`
  (指数退避重试)。dc_member 刷新按板块粒度断点续跑。
- **webapp 前端陷阱**: `window.api` 定义在 base.html 的 `{% block extra_js %}` **之后**,
  新页面初始加载必须 `$(load)` 包一层等 DOM ready, 否则 `api is not defined`。
- 数据起点: dc_index 2024-12-20, limit_list 2024-01-02, sector_daily_stats 默认从 2024-01-02。
