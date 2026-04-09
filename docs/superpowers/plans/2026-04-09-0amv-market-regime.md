# 0AMV 全市场活跃市值 + 牛熊体制切换 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 复刻指南针0AMV（活筹指数）全市场指标，基于其构建牛熊体制信号，回测ng1.0.1(牛市)/ng1.0.4(熊市)双模型切换策略。

**Architecture:** 三层实现——(1) 数据层：修复指数amount入库 + 历史回填；(2) 计算层：`indicators/market_amv.py` 实现SMA/DMA/MACD + 牛熊状态机，结果存入`market_amv`表；(3) 回测层：`backtest/regime_switch_backtest.py` 按regime合并两个模型报告并评估。

**Tech Stack:** Python 3, SQLite, tushare, pandas, numpy

**File Structure:**

| File | Action | Responsibility |
|------|--------|----------------|
| `data_adapter/database_manager.py` | Modify | `insert_daily_quotes` 增加 amount 字段 |
| `fetch_data/quick_daily_update.py` | Modify | `update_market_indices` 传递 amount；新增 `update_market_amv()` 调用 |
| `fetch_data/backfill_index_amount.py` | Create | 一次性脚本：回填2018-2026指数amount历史数据 |
| `indicators/__init__.py` | Create | 空文件 |
| `indicators/market_amv.py` | Create | 0AMV计算引擎 + 牛熊状态机 + DB读写 |
| `backtest/regime_switch_backtest.py` | Create | 双模型切换回测脚本 |

---

### Task 1: 修复指数amount入库

`insert_daily_quotes()` 的INSERT语句不含amount列，导致tushare已抓到的amount被丢弃。

**Files:**
- Modify: `data_adapter/database_manager.py:167-192`

- [ ] **Step 1: 修改 insert_daily_quotes 增加 amount 字段**

在 `data_adapter/database_manager.py` 的 `insert_daily_quotes` 方法中，将 amount 加入 INSERT 语句：

```python
    def insert_daily_quotes(self, data: List[Dict[str, Any]]) -> int:
        """批量插入日线行情数据"""
        query = """
        INSERT OR REPLACE INTO daily_quotes (
            security_id, trade_date, open, high, low, close, volume, amount,
            price_change_pct, is_limit_up, is_limit_down, is_suspend
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        rows_data = []
        for item in data:
            rows_data.append((
                item['security_id'],
                item['trade_date'],
                item['open'],
                item['high'],
                item['low'],
                item['close'],
                item['volume'],
                item.get('amount'),
                item.get('price_change_pct', 0),
                item.get('is_limit_up', False),
                item.get('is_limit_down', False),
                item.get('is_suspend', False)
            ))
        
        return self.execute_many(query, rows_data)
```

- [ ] **Step 2: 修改 update_market_indices 传递 amount**

在 `fetch_data/quick_daily_update.py:275-286`，在 `all_data.append` 字典中增加 amount 字段：

```python
                        all_data.append({
                            'security_id': security_id,
                            'trade_date': trade_date,
                            'open': row['open'],
                            'close': row['close'],
                            'high': row['high'],
                            'low': row['low'],
                            'volume': row.get('vol', 0),
                            'amount': row.get('amount'),
                            'price_change_pct': pct_val / 100 if pd.notna(pct_val) else 0,
                            'is_limit_up': False,
                            'is_limit_down': False
                        })
```

- [ ] **Step 3: 验证修改**

```bash
python3 -c "
from data_adapter.database_manager import DatabaseManager
db = DatabaseManager()
# 测试insert_daily_quotes接受amount
db.insert_daily_quotes([{
    'security_id': 1, 'trade_date': '1999-01-01',
    'open': 1, 'high': 1, 'low': 1, 'close': 1,
    'volume': 100, 'amount': 999.99
}])
import sqlite3
conn = sqlite3.connect('data_adapter/stock_data.db', timeout=30)
r = conn.execute('SELECT amount FROM daily_quotes WHERE trade_date=\"1999-01-01\"').fetchone()
print(f'amount stored: {r[0]}')
conn.execute('DELETE FROM daily_quotes WHERE trade_date=\"1999-01-01\"')
conn.commit(); conn.close()
print('OK')
"
```

Expected: `amount stored: 999.99` + `OK`

- [ ] **Step 4: Commit**

```bash
git add data_adapter/database_manager.py fetch_data/quick_daily_update.py
git commit -m "fix: insert_daily_quotes增加amount字段，指数成交额正确入库"
```

---

### Task 2: 回填历史指数amount数据

2018-01至今的上证+深证指数amount需要从tushare补入。

**Files:**
- Create: `fetch_data/backfill_index_amount.py`

- [ ] **Step 1: 创建回填脚本**

创建 `fetch_data/backfill_index_amount.py`：

```python
"""回填上证指数+深证成指的历史amount数据到daily_quotes表"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import sqlite3
import tushare as ts
import pandas as pd
import time


def backfill_index_amount(start_date='20180101', end_date=None):
    config = json.load(open('config.json'))
    pro = ts.pro_api(config['tushare']['token'])
    conn = sqlite3.connect('data_adapter/stock_data.db', timeout=30)

    if end_date is None:
        end_date = pd.Timestamp.now().strftime('%Y%m%d')

    indices = {
        '000001.SH': '上证指数',
        '399001.SZ': '深证成指',
    }

    # 获取security_id映射
    sec_map = {}
    for code in indices:
        row = conn.execute(
            'SELECT id FROM securities WHERE code = ?', (code,)
        ).fetchone()
        if row:
            sec_map[code] = row[0]
        else:
            print(f'WARNING: {code} not in securities table, skip')

    for ts_code, name in indices.items():
        if ts_code not in sec_map:
            continue
        sid = sec_map[ts_code]
        print(f'\n回填 {name} ({ts_code}) amount...')

        df = pro.index_daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields='ts_code,trade_date,amount'
        )
        if df.empty:
            print(f'  无数据')
            continue

        print(f'  获取 {len(df)} 天数据')

        updated = 0
        for _, row in df.iterrows():
            trade_date = pd.to_datetime(
                row['trade_date'], format='%Y%m%d'
            ).strftime('%Y-%m-%d')
            amount = row.get('amount')
            if pd.isna(amount):
                continue
            conn.execute(
                'UPDATE daily_quotes SET amount = ? '
                'WHERE security_id = ? AND trade_date = ?',
                (amount, sid, trade_date)
            )
            updated += 1

        conn.commit()
        print(f'  更新 {updated} 条')
        time.sleep(0.5)

    # 验证
    print('\n=== 验证 ===')
    for ts_code, name in indices.items():
        if ts_code not in sec_map:
            continue
        r = conn.execute('''
            SELECT MIN(trade_date), MAX(trade_date), COUNT(*)
            FROM daily_quotes
            WHERE security_id = ? AND amount IS NOT NULL AND amount > 0
        ''', (sec_map[ts_code],)).fetchone()
        print(f'{name}: {r[0]} ~ {r[1]}, {r[2]}条')

    conn.close()
    print('\n完成')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='回填指数amount历史数据')
    parser.add_argument('--start-date', default='20180101')
    parser.add_argument('--end-date', default=None)
    args = parser.parse_args()
    backfill_index_amount(args.start_date, args.end_date)
```

- [ ] **Step 2: 运行回填**

```bash
python3 fetch_data/backfill_index_amount.py
```

Expected: 上证指数和深证成指各约2000条amount数据被更新。

- [ ] **Step 3: Commit**

```bash
git add fetch_data/backfill_index_amount.py
git commit -m "feat: 回填上证+深证指数历史amount数据"
```

---

### Task 3: 0AMV计算引擎

核心模块，实现中国式SMA、DMA、标准MACD、牛熊状态机。

**Files:**
- Create: `indicators/__init__.py`
- Create: `indicators/market_amv.py`

- [ ] **Step 1: 创建 indicators 包**

```bash
mkdir -p indicators
touch indicators/__init__.py
```

- [ ] **Step 2: 创建 market_amv.py**

创建 `indicators/market_amv.py`：

```python
"""
0AMV 全市场活跃市值指标

复刻指南针(Compass)活筹指数，改造为全市场版本。
数据源: 上证指数+深证成指 成交额之和
"""
import numpy as np
import pandas as pd
import sqlite3
import os
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')

# === 通达信函数复刻 ===

def tdx_sma(series: np.ndarray, n: int, m: int = 1) -> np.ndarray:
    """中国式SMA: Y = (M * X + (N - M) * Y_prev) / N
    等价于 EWM alpha = M/N
    """
    alpha = m / n
    result = np.zeros(len(series))
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


def tdx_dma(series: np.ndarray, a: np.ndarray) -> np.ndarray:
    """动态移动平均: Y = A * X + (1 - A) * Y_prev
    A是动态系数数组，需clamp到(0, 1)
    """
    a_clamped = np.clip(a, 0.001, 1.0)
    result = np.zeros(len(series))
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = a_clamped[i] * series[i] + (1 - a_clamped[i]) * result[i - 1]
    return result


def ema(series: np.ndarray, n: int) -> np.ndarray:
    """标准EMA"""
    s = pd.Series(series)
    return s.ewm(span=n, adjust=False).mean().values


def ma(series: np.ndarray, n: int) -> np.ndarray:
    """简单移动平均，前n-1个值用累积均值填充"""
    s = pd.Series(series)
    result = s.rolling(n, min_periods=1).mean().values
    return result


# === 数据加载 ===

def load_market_amount(conn: sqlite3.Connection) -> pd.DataFrame:
    """加载上证+深证每日成交额，返回 DataFrame(trade_date, market_amount)"""
    query = """
        SELECT dq.trade_date,
               SUM(dq.amount) as market_amount
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.code IN ('000001.SH', '399001.SZ')
          AND dq.amount IS NOT NULL AND dq.amount > 0
        GROUP BY dq.trade_date
        HAVING COUNT(*) = 2
        ORDER BY dq.trade_date
    """
    df = pd.read_sql(query, conn)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df


def load_market_circ_mv(conn: sqlite3.Connection) -> pd.DataFrame:
    """加载每日全A股流通市值合计，返回 DataFrame(trade_date, market_circ_mv)"""
    query = """
        SELECT db.trade_date,
               SUM(db.circ_mv) as market_circ_mv
        FROM daily_basic db
        JOIN securities s ON db.security_id = s.id
        WHERE s.type = 'A股'
          AND db.circ_mv IS NOT NULL AND db.circ_mv > 0
        GROUP BY db.trade_date
        ORDER BY db.trade_date
    """
    df = pd.read_sql(query, conn)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df


# === 0AMV核心计算 ===

def compute_amv(market_amount: np.ndarray,
                market_turnover: np.ndarray) -> dict:
    """计算完整0AMV指标

    Args:
        market_amount: 每日全市场成交额序列
        market_turnover: 每日全市场换手率序列 (amount / circ_mv)

    Returns:
        dict of numpy arrays: var1, c5, c13, c34, inf, ma60, dif, dea, macd
    """
    # Var1: SMA(amount, 10, 1) / 1e7
    var1 = tdx_sma(market_amount, 10, 1) / 1e7

    # SMA中间值
    sma_var1_3 = tdx_sma(var1, 3, 1)
    sma_var1_8 = tdx_sma(var1, 8, 1)

    # DMA平滑因子 = turnover / 基准换手率
    a_c5 = market_turnover / 0.02
    a_c13 = market_turnover / 0.10
    a_c34 = market_turnover / 0.18
    a_inf = market_turnover / 1.10

    # 四条线
    c5 = tdx_dma(sma_var1_3, a_c5)
    c13 = tdx_dma(sma_var1_3, a_c13)
    c34 = tdx_dma(sma_var1_8, a_c34)
    inf_line = tdx_dma(var1, a_inf)

    # MA60
    ma60 = ma(var1, 60)

    # MACD (12/26/9)
    ema12 = ema(var1, 12)
    ema26 = ema(var1, 26)
    dif = ema12 - ema26
    dea = ema(dif, 9)
    macd_hist = (dif - dea) * 2

    return {
        'var1': var1,
        'amv_c5': c5,
        'amv_c13': c13,
        'amv_c34': c34,
        'amv_inf': inf_line,
        'amv_ma60': ma60,
        'amv_dif': dif,
        'amv_dea': dea,
        'amv_macd': macd_hist,
    }


# === 牛熊状态机 ===

def compute_regime(var1: np.ndarray, ma60: np.ndarray,
                   macd: np.ndarray) -> np.ndarray:
    """牛熊体制判断

    转牛 (三者同时):
      1. var1 单日涨幅 >= +4.3%
      2. var1 > ma60
      3. macd 从 <0 转为 >0

    转熊 (三者同时):
      1. var1 单日跌幅 <= -2.3%
      2. var1 < ma60
      3. macd 从 >0 转为 <0

    Returns:
        numpy array of 1 (bull) / -1 (bear)
    """
    n = len(var1)
    regime = np.zeros(n, dtype=int)

    # 日涨跌幅
    pct_change = np.zeros(n)
    pct_change[1:] = (var1[1:] - var1[:-1]) / (var1[:-1] + 1e-15)

    # 初始状态: 根据var1与ma60的关系
    regime[0] = 1 if var1[0] > ma60[0] else -1

    for i in range(1, n):
        prev_regime = regime[i - 1]
        prev_macd = macd[i - 1]
        curr_macd = macd[i]

        # 检查转牛
        if prev_regime == -1:
            bull_signal = (
                pct_change[i] >= 0.043
                and var1[i] > ma60[i]
                and prev_macd < 0 and curr_macd > 0
            )
            regime[i] = 1 if bull_signal else -1

        # 检查转熊
        elif prev_regime == 1:
            bear_signal = (
                pct_change[i] <= -0.023
                and var1[i] < ma60[i]
                and prev_macd > 0 and curr_macd < 0
            )
            regime[i] = -1 if bear_signal else 1

    return regime


# === DB读写 ===

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS market_amv (
    trade_date DATE PRIMARY KEY,
    market_amount REAL,
    market_circ_mv REAL,
    market_turnover REAL,
    var1 REAL,
    amv_c5 REAL,
    amv_c13 REAL,
    amv_c34 REAL,
    amv_inf REAL,
    amv_ma60 REAL,
    amv_dif REAL,
    amv_dea REAL,
    amv_macd REAL,
    amv_regime INTEGER
)
"""


def save_to_db(conn: sqlite3.Connection, df: pd.DataFrame):
    """将计算结果写入market_amv表"""
    conn.execute(CREATE_TABLE_SQL)
    conn.execute('DELETE FROM market_amv')

    rows = []
    for _, r in df.iterrows():
        rows.append((
            r['trade_date'].strftime('%Y-%m-%d') if hasattr(r['trade_date'], 'strftime')
            else str(r['trade_date']),
            r['market_amount'], r['market_circ_mv'], r['market_turnover'],
            r['var1'], r['amv_c5'], r['amv_c13'], r['amv_c34'], r['amv_inf'],
            r['amv_ma60'], r['amv_dif'], r['amv_dea'], r['amv_macd'],
            int(r['amv_regime']),
        ))
    conn.executemany(
        'INSERT OR REPLACE INTO market_amv VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
        rows
    )
    conn.commit()
    logger.info(f'Saved {len(rows)} rows to market_amv')


# === 主入口 ===

def compute_and_save(db_path: str = None):
    """完整计算流程: 加载数据 → 计算0AMV → 牛熊判断 → 写入DB"""
    if db_path is None:
        db_path = DB_PATH
    conn = sqlite3.connect(db_path, timeout=30)

    # 1. 加载数据
    amount_df = load_market_amount(conn)
    circ_mv_df = load_market_circ_mv(conn)

    if amount_df.empty:
        logger.error('No market amount data found. Run backfill_index_amount.py first.')
        conn.close()
        return None

    # 2. 合并 (inner join on trade_date)
    df = amount_df.merge(circ_mv_df, on='trade_date', how='inner')
    df = df.sort_values('trade_date').reset_index(drop=True)

    # 3. 计算换手率 (amount单位: 千元 from tushare, circ_mv单位: 万元)
    # tushare index_daily amount 单位是千元, circ_mv 单位是万元
    # 统一为元: amount * 1000, circ_mv * 10000
    df['market_turnover'] = (df['market_amount'] * 1000) / (df['market_circ_mv'] * 10000 + 1e-15)

    logger.info(f'Data: {len(df)} trading days, '
                f'{df["trade_date"].min()} ~ {df["trade_date"].max()}')

    # 4. 计算0AMV
    # amount传入原始值(千元)，var1内部会/1e7
    amv = compute_amv(df['market_amount'].values, df['market_turnover'].values)
    for key, arr in amv.items():
        df[key] = arr

    # 5. 牛熊判断
    df['amv_regime'] = compute_regime(
        amv['var1'], amv['amv_ma60'], amv['amv_macd']
    )

    # 6. 写入DB
    save_to_db(conn, df)
    conn.close()

    # 7. 打印摘要
    bull_days = (df['amv_regime'] == 1).sum()
    bear_days = (df['amv_regime'] == -1).sum()
    switches = (df['amv_regime'].diff().abs() > 0).sum()
    print(f'\n0AMV计算完成: {len(df)}天')
    print(f'  牛市: {bull_days}天 ({100*bull_days/len(df):.1f}%)')
    print(f'  熊市: {bear_days}天 ({100*bear_days/len(df):.1f}%)')
    print(f'  切换次数: {switches}')
    print(f'  最新regime: {"牛市" if df.iloc[-1]["amv_regime"]==1 else "熊市"}')
    print(f'  最新var1: {df.iloc[-1]["var1"]:.2f}')
    print(f'  最新ma60: {df.iloc[-1]["amv_ma60"]:.2f}')

    return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    compute_and_save()
```

- [ ] **Step 3: 运行并验证**

```bash
python3 indicators/market_amv.py
```

Expected: 打印出牛市/熊市天数、切换次数等摘要信息。

- [ ] **Step 4: 快速目视验证牛熊切换时点**

```bash
python3 -c "
import sqlite3, pandas as pd
conn = sqlite3.connect('data_adapter/stock_data.db', timeout=30)
df = pd.read_sql('SELECT trade_date, var1, amv_ma60, amv_macd, amv_regime FROM market_amv ORDER BY trade_date', conn)
conn.close()
# 打印所有切换点
switches = df[df['amv_regime'].diff().abs() > 0]
for _, r in switches.iterrows():
    label = '→牛' if r['amv_regime'] == 1 else '→熊'
    print(f'{r[\"trade_date\"]} {label}  var1={r[\"var1\"]:.1f} ma60={r[\"amv_ma60\"]:.1f} macd={r[\"amv_macd\"]:.2f}')
"
```

Expected: 切换时点应大致对应A股历史牛熊转折（如2020-07牛, 2021-12熊, 2024-09牛等）。

- [ ] **Step 5: Commit**

```bash
git add indicators/__init__.py indicators/market_amv.py
git commit -m "feat: 0AMV全市场活跃市值指标 + 牛熊体制判断"
```

---

### Task 4: 集成到每日更新流程

**Files:**
- Modify: `fetch_data/quick_daily_update.py`

- [ ] **Step 1: 在 quick_daily_update.py 末尾的主流程中添加0AMV更新**

找到 `quick_daily_update.py` 的主入口函数（通常是 `main()` 或 `if __name__` 块），在所有数据更新完成后添加：

```python
    # 更新0AMV市场活跃市值指标
    try:
        from indicators.market_amv import compute_and_save
        logger.info("更新0AMV活跃市值指标...")
        compute_and_save()
        logger.info("0AMV更新完成")
    except Exception as e:
        logger.warning(f"0AMV更新失败(非关键): {e}")
```

- [ ] **Step 2: 验证每日更新流程**

```bash
python3 fetch_data/quick_daily_update.py --date 20260408 2>&1 | grep -i amv
```

Expected: 日志中出现 "更新0AMV活跃市值指标..." 和 "0AMV更新完成"。

- [ ] **Step 3: Commit**

```bash
git add fetch_data/quick_daily_update.py
git commit -m "feat: 每日更新流程集成0AMV计算"
```

---

### Task 5: 双模型切换回测

**Files:**
- Create: `backtest/regime_switch_backtest.py`

- [ ] **Step 1: 创建回测脚本**

创建 `backtest/regime_switch_backtest.py`：

```python
"""
双模型牛熊切换回测

根据0AMV牛熊体制信号，牛市用ng1.0.1，熊市用ng1.0.4-3seed。
对比三种策略: 纯ng101 / 纯ng104-3s / 切换策略
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import pandas as pd
import argparse
from backtest.backtest_report_based import load_reports, run_single_backtest, compute_ns_scores


DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data_adapter', 'stock_data.db'
)


def load_regime(db_path: str = None) -> dict:
    """加载每日amv_regime，返回 {date_str: regime_int}"""
    if db_path is None:
        db_path = DB_PATH
    conn = sqlite3.connect(db_path, timeout=30)
    rows = conn.execute(
        'SELECT trade_date, amv_regime FROM market_amv ORDER BY trade_date'
    ).fetchall()
    conn.close()
    # date format: YYYY-MM-DD → YYYYMMDD (报告文件名格式)
    regime = {}
    for date_str, r in rows:
        key = date_str.replace('-', '')
        regime[key] = r
    return regime


def merge_reports_by_regime(
    bull_reports: dict,
    bear_reports: dict,
    regime: dict,
) -> dict:
    """按regime合并两个模型的报告

    Args:
        bull_reports: ng1.0.1报告 {date: [stocks]}
        bear_reports: ng1.0.4-3s报告 {date: [stocks]}
        regime: {date: 1/-1}

    Returns:
        merged reports dict
    """
    merged = {}
    bull_count = 0
    bear_count = 0
    skip_count = 0

    all_dates = sorted(set(bull_reports.keys()) & set(bear_reports.keys()))

    for date in all_dates:
        r = regime.get(date)
        if r is None:
            skip_count += 1
            continue
        if r == 1:
            merged[date] = bull_reports[date]
            bull_count += 1
        else:
            merged[date] = bear_reports[date]
            bear_count += 1

    print(f'  合并报告: {len(merged)}天 (牛市用101: {bull_count}天, '
          f'熊市用104: {bear_count}天, 无regime跳过: {skip_count}天)')
    return merged


def run_comparison(
    bull_dir: str = 'reports/daily_selection_ng101',
    bear_dir: str = 'reports/daily_selection_ng104_ensemble_3seed',
    top_n: int = 10,
    focus_days: int = 10,
    rank_field: str = 'score',
):
    """运行三方对比回测"""
    print('=' * 70)
    print('  0AMV牛熊切换 双模型回测')
    print('=' * 70)

    # 1. 加载regime
    regime = load_regime()
    if not regime:
        print('ERROR: market_amv表为空，先运行 indicators/market_amv.py')
        return

    bull_days = sum(1 for v in regime.values() if v == 1)
    bear_days = sum(1 for v in regime.values() if v == -1)
    print(f'\n体制信号: {len(regime)}天, 牛市{bull_days}天, 熊市{bear_days}天')

    # 2. 加载两个模型的报告
    print(f'\n加载牛市模型报告: {bull_dir}')
    bull_reports = load_reports(bull_dir, rank_field=rank_field)
    print(f'  {len(bull_reports)} 天')

    print(f'加载熊市模型报告: {bear_dir}')
    bear_reports = load_reports(bear_dir, rank_field=rank_field)
    print(f'  {len(bear_reports)} 天')

    # 3. 合并
    print(f'\n按regime合并报告...')
    merged_reports = merge_reports_by_regime(bull_reports, bear_reports, regime)

    if not merged_reports:
        print('ERROR: 合并后无报告')
        return

    # 4. 三方回测
    configs = [
        ('NG101-纯牛模型', bull_reports),
        ('NG104-纯熊模型', bear_reports),
        ('切换策略(101+104)', merged_reports),
    ]

    results = []
    for label, reports in configs:
        print(f'\n{"=" * 50}')
        print(f'  回测: {label}')
        print(f'{"=" * 50}')
        result = run_single_backtest(
            reports, label, top_n=top_n, focus_days=focus_days,
        )
        results.append((label, result))

    # 5. 打印对比摘要
    print(f'\n{"=" * 70}')
    print(f'  三方对比摘要 (Top-{top_n}, {focus_days}日持仓, 无CPPI)')
    print(f'{"=" * 70}')
    print(f'{"指标":<20} {"NG101":<18} {"NG104-3s":<18} {"切换策略":<18}')
    print('-' * 74)

    for metric_name, metric_key in [
        ('年化收益(毛)', 'annual_return'),
        ('年化收益(净)', 'annual_return_net'),
        ('Sharpe', 'sharpe'),
        ('最大回撤', 'max_drawdown'),
        ('月度胜率', 'monthly_win_rate'),
        ('超额年化', 'excess_annual'),
    ]:
        vals = []
        for label, result in results:
            s = result['summary'].get(focus_days, {})
            v = s.get(metric_key, 0)
            if metric_key in ('annual_return', 'annual_return_net',
                              'max_drawdown', 'monthly_win_rate',
                              'excess_annual'):
                vals.append(f'{v*100:.1f}%' if isinstance(v, float) and abs(v) < 10 else f'{v:.1f}%')
            else:
                vals.append(f'{v:.3f}')
        print(f'{metric_name:<20} {vals[0]:<18} {vals[1]:<18} {vals[2]:<18}')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='0AMV牛熊切换双模型回测')
    parser.add_argument('--bull-dir', default='reports/daily_selection_ng101',
                        help='牛市模型报告目录')
    parser.add_argument('--bear-dir', default='reports/daily_selection_ng104_ensemble_3seed',
                        help='熊市模型报告目录')
    parser.add_argument('--top-n', type=int, default=10)
    parser.add_argument('--focus-days', type=int, default=10)
    parser.add_argument('--rank-field', default='score')
    args = parser.parse_args()

    run_comparison(
        bull_dir=args.bull_dir,
        bear_dir=args.bear_dir,
        top_n=args.top_n,
        focus_days=args.focus_days,
        rank_field=args.rank_field,
    )
```

- [ ] **Step 2: 运行回测**

```bash
python3 backtest/regime_switch_backtest.py
```

Expected: 三方对比表，显示纯ng101、纯ng104-3s、切换策略的年化收益/Sharpe/MaxDD对比。

- [ ] **Step 3: 用北极星V5.2评估切换策略**

如果切换策略表现有意义，用标准评估流程深入分析：

```bash
# 先生成合并报告到临时目录
python3 -c "
import sys; sys.path.insert(0, '.')
from backtest.regime_switch_backtest import load_regime, merge_reports_by_regime
from backtest.backtest_report_based import load_reports
import json, os

regime = load_regime()
bull = load_reports('reports/daily_selection_ng101', rank_field='score')
bear = load_reports('reports/daily_selection_ng104_ensemble_3seed', rank_field='score')
merged = merge_reports_by_regime(bull, bear, regime)

out_dir = 'reports/daily_selection_regime_switch'
os.makedirs(out_dir, exist_ok=True)

# 需要从原始报告目录复制对应日期的JSON文件
import shutil
for date in sorted(merged.keys()):
    r = regime.get(date, -1)
    src_dir = 'reports/daily_selection_ng101' if r == 1 else 'reports/daily_selection_ng104_ensemble_3seed'
    src = os.path.join(src_dir, f'analysis_data_{date}.json')
    dst = os.path.join(out_dir, f'analysis_data_{date}.json')
    if os.path.exists(src):
        shutil.copy2(src, dst)

print(f'Copied {len(os.listdir(out_dir))} reports to {out_dir}')
"

# 北极星评估
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_regime_switch \
    --label "AMV切换" --top-n 10 --focus-days 10 --rank-field score \
    --start-date 2020-01-01 --score-version v52
```

- [ ] **Step 4: Commit**

```bash
git add backtest/regime_switch_backtest.py
git commit -m "feat: 0AMV牛熊切换双模型回测 — ng101(牛)+ng104(熊)"
```

---

### Task 6: 验证与汇总

- [ ] **Step 1: 检查牛熊切换时点合理性**

打印所有切换点，人工对照A股大盘走势验证：

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('data_adapter/stock_data.db', timeout=30)
rows = conn.execute('''
    SELECT a.trade_date, a.amv_regime, a.var1, a.amv_ma60, a.amv_macd,
           LAG(a.amv_regime) OVER (ORDER BY a.trade_date) as prev_regime
    FROM market_amv a
    ORDER BY a.trade_date
''').fetchall()
conn.close()
print(f'{\"日期\":<12} {\"变化\":<6} {\"var1\":>8} {\"ma60\":>8} {\"macd\":>8}')
print('-' * 50)
for r in rows:
    date, regime, var1, ma60, macd, prev = r
    if prev is not None and regime != prev:
        label = '→牛市' if regime == 1 else '→熊市'
        print(f'{date:<12} {label:<6} {var1:>8.1f} {ma60:>8.1f} {macd:>8.2f}')
"
```

- [ ] **Step 2: 汇总对比结果**

记录三个策略的关键指标：
- 年化收益、Sharpe、MaxDD
- 切换策略是否实现了"牛市高收益 + 熊市低回撤"的目标
- 牛熊切换次数是否合理（预期5-10次/6年）

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat(0amv): 全市场活跃市值指标+牛熊切换回测完成"
```
