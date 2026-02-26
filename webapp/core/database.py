"""
数据库管理器 - 统一管理数据库操作
"""
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    数据库管理器

    管理主数据库（stock_data.db）和Web应用数据库（webapp.db）的操作
    """

    def __init__(self, stock_db_path: Path, webapp_db_path: Path):
        """
        初始化数据库管理器

        Args:
            stock_db_path: 主数据库路径
            webapp_db_path: Web应用数据库路径
        """
        self.stock_db_path = stock_db_path
        self.webapp_db_path = webapp_db_path

        # 初始化Web应用数据库
        self._init_webapp_db()

    @contextmanager
    def get_stock_db_connection(self):
        """获取主数据库连接（上下文管理器）"""
        conn = sqlite3.connect(self.stock_db_path)
        conn.row_factory = sqlite3.Row  # 使查询结果可以通过列名访问
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def get_webapp_db_connection(self):
        """获取Web应用数据库连接（上下文管理器）"""
        conn = sqlite3.connect(self.webapp_db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_webapp_db(self):
        """初始化Web应用数据库表"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()

            # 训练任务记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT UNIQUE NOT NULL,
                    model_version TEXT NOT NULL,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME,
                    status TEXT NOT NULL,
                    params TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 训练指标记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    loss REAL,
                    accuracy REAL,
                    val_loss REAL,
                    val_accuracy REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES training_jobs(job_id)
                )
            ''')

            # 模型性能指标表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    FOREIGN KEY (job_id) REFERENCES training_jobs(job_id)
                )
            ''')

            # ==================== 持仓管理相关表 ====================

            # 持仓分组表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS position_groups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    color TEXT DEFAULT '#6c757d',
                    description TEXT,
                    sort_order INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 持仓表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT NOT NULL,
                    name TEXT,
                    quantity INTEGER NOT NULL,
                    avg_cost REAL NOT NULL,
                    current_price REAL,
                    market_value REAL,
                    profit_loss REAL,
                    profit_loss_pct REAL,
                    first_buy_date DATE,
                    last_update DATETIME,
                    status TEXT DEFAULT 'holding',
                    notes TEXT,
                    group_id INTEGER,
                    UNIQUE(code, status),
                    FOREIGN KEY (group_id) REFERENCES position_groups(id)
                )
            ''')

            # 为已有的positions表添加group_id列（如果不存在）
            try:
                cursor.execute('ALTER TABLE positions ADD COLUMN group_id INTEGER')
            except:
                pass  # 列已存在

            # 交易记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_date DATE NOT NULL,
                    trade_time DATETIME,
                    code TEXT NOT NULL,
                    name TEXT,
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    amount REAL,
                    commission REAL DEFAULT 0,
                    reason TEXT,
                    strategy TEXT,
                    signal_source TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 操作建议表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    code TEXT NOT NULL,
                    name TEXT,
                    action TEXT NOT NULL,
                    urgency TEXT DEFAULT 'normal',
                    reason TEXT,
                    ml_score REAL,
                    technical_signal TEXT,
                    stop_loss_price REAL,
                    take_profit_price REAL,
                    confidence REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    is_executed INTEGER DEFAULT 0,
                    executed_at DATETIME,
                    UNIQUE(date, code)
                )
            ''')

            # 为已有的recommendations表添加新列（如果不存在）
            for col, coltype in [('kelly_position', 'REAL'), ('predicted_return_5d', 'REAL')]:
                try:
                    cursor.execute(f'ALTER TABLE recommendations ADD COLUMN {col} {coltype}')
                except:
                    pass  # 列已存在

            # 交易评估表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id INTEGER NOT NULL,
                    eval_date DATE NOT NULL,
                    days_after INTEGER,
                    price_after REAL,
                    return_pct REAL,
                    score REAL,
                    grade TEXT,
                    comments TEXT,
                    max_profit_pct REAL,
                    max_loss_pct REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (trade_id) REFERENCES trades(id)
                )
            ''')

            # 持仓快照表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS position_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_date DATE NOT NULL,
                    total_market_value REAL,
                    total_cost REAL,
                    total_profit_loss REAL,
                    total_profit_loss_pct REAL,
                    position_count INTEGER,
                    details TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(snapshot_date)
                )
            ''')

            # 组合评分历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    score_date DATE NOT NULL,
                    total_score INTEGER,
                    total_max INTEGER,
                    total_pct REAL,
                    grade TEXT,
                    grade_label TEXT,
                    layer1_score INTEGER,
                    layer1_pct REAL,
                    layer2_score INTEGER,
                    layer2_pct REAL,
                    layer3_score INTEGER,
                    layer3_pct REAL,
                    layer4_score INTEGER,
                    layer4_pct REAL,
                    position_count INTEGER,
                    total_market_value REAL,
                    details TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(score_date)
                )
            ''')

            conn.commit()
            logger.info('Web应用数据库初始化完成（含持仓管理表+评分表）')

    # ==================== 主数据库查询方法 ====================

    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        with self.get_stock_db_connection() as conn:
            cursor = conn.cursor()

            # 股票总数
            cursor.execute("SELECT COUNT(*) FROM securities WHERE type='A股'")
            total_stocks = cursor.fetchone()[0]

            # ETF数量
            cursor.execute("SELECT COUNT(*) FROM securities WHERE type LIKE '%ETF%'")
            total_etfs = cursor.fetchone()[0]

            # 最新数据日期
            cursor.execute("""
                SELECT MAX(trade_date) FROM daily_quotes
            """)
            latest_date = cursor.fetchone()[0]

            # 最早数据日期
            cursor.execute("""
                SELECT MIN(trade_date) FROM daily_quotes
            """)
            earliest_date = cursor.fetchone()[0]

            # 日线数据条数
            cursor.execute("SELECT COUNT(*) FROM daily_quotes")
            total_quotes = cursor.fetchone()[0]

            # 技术指标条数
            total_indicators = 0
            try:
                cursor.execute("SELECT COUNT(*) FROM technical_indicators")
                total_indicators = cursor.fetchone()[0]
            except:
                pass

            # 数据库大小
            db_size_mb = self.stock_db_path.stat().st_size / (1024 * 1024)

            # 计算数据时间跨度
            date_range = ''
            if earliest_date and latest_date:
                try:
                    from datetime import datetime
                    early = datetime.strptime(earliest_date, '%Y-%m-%d')
                    late = datetime.strptime(latest_date, '%Y-%m-%d')
                    days = (late - early).days
                    years = days // 365
                    months = (days % 365) // 30
                    if years > 0:
                        date_range = f'{years}年{months}月'
                    else:
                        date_range = f'{months}月'
                except:
                    date_range = f'{earliest_date} - {latest_date}'

            return {
                'total_stocks': total_stocks,
                'total_etfs': total_etfs,
                'latest_date': latest_date,
                'earliest_date': earliest_date,
                'date_range': date_range,
                'total_quotes': total_quotes,
                'total_indicators': total_indicators,
                'db_size_mb': round(db_size_mb, 2)
            }

    def get_latest_selections(
        self,
        date: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        获取最新选股结果

        Args:
            date: 日期（YYYY-MM-DD），None表示最新
            limit: 返回数量限制

        Returns:
            选股列表
        """
        with self.get_stock_db_connection() as conn:
            cursor = conn.cursor()

            if date:
                query = """
                    SELECT
                        ss.code,
                        ss.date,
                        ss.signal_type,
                        ss.strength,
                        ss.strategy,
                        s.name,
                        dq.close,
                        dq.price_change_pct
                    FROM stock_signals ss
                    JOIN securities s ON ss.code = s.code
                    LEFT JOIN daily_quotes dq ON ss.code = dq.security_id
                        AND ss.date = dq.trade_date
                    WHERE ss.date = ?
                    ORDER BY ss.strength DESC
                    LIMIT ?
                """
                cursor.execute(query, (date, limit))
            else:
                query = """
                    SELECT
                        ss.code,
                        ss.date,
                        ss.signal_type,
                        ss.strength,
                        ss.strategy,
                        s.name,
                        dq.close,
                        dq.price_change_pct
                    FROM stock_signals ss
                    JOIN securities s ON ss.code = s.code
                    LEFT JOIN daily_quotes dq ON ss.code = dq.security_id
                        AND ss.date = dq.trade_date
                    WHERE ss.date = (SELECT MAX(date) FROM stock_signals)
                    ORDER BY ss.strength DESC
                    LIMIT ?
                """
                cursor.execute(query, (limit,))

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_stock_recent_data(
        self,
        code: str,
        days: int = 30
    ) -> List[Dict]:
        """
        获取股票最近N天的数据

        Args:
            code: 股票代码
            days: 天数

        Returns:
            日线数据列表
        """
        with self.get_stock_db_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT
                    trade_date,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    price_change_pct
                FROM daily_quotes
                WHERE security_id = (SELECT id FROM securities WHERE code = ?)
                ORDER BY trade_date DESC
                LIMIT ?
            """
            cursor.execute(query, (code, days))

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    # ==================== 大盘指数方法 ====================

    def get_market_indices(self) -> List[Dict]:
        """获取主要大盘指数最新数据（含涨跌幅）"""
        with self.get_stock_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.code, s.name,
                       dq.trade_date, dq.open, dq.high, dq.low, dq.close, dq.volume,
                       dq.price_change_pct
                FROM securities s
                JOIN daily_quotes dq ON dq.security_id = s.id
                WHERE s.type = '指数'
                AND dq.trade_date = (
                    SELECT MAX(dq2.trade_date) FROM daily_quotes dq2
                    JOIN securities s2 ON dq2.security_id = s2.id
                    WHERE s2.code = '000001.SH'
                )
                ORDER BY s.code
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_index_history(self, code: str, days: int = 30) -> List[Dict]:
        """获取指数历史数据"""
        with self.get_stock_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT dq.trade_date, dq.close
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.code = ?
                ORDER BY dq.trade_date DESC
                LIMIT ?
            """, (code, days))
            rows = cursor.fetchall()
            return [dict(row) for row in reversed(rows)]

    # ==================== 个股数据方法 ====================

    def search_stocks(self, query: str, limit: int = 20) -> List[Dict]:
        """搜索股票（代码/名称模糊匹配）"""
        with self.get_stock_db_connection() as conn:
            cursor = conn.cursor()
            like_query = f'%{query}%'
            cursor.execute("""
                SELECT code, name, type FROM securities
                WHERE (code LIKE ? OR name LIKE ?)
                AND type IN ('A股', 'ETF_基金')
                LIMIT ?
            """, (like_query, like_query, limit))
            return [dict(row) for row in cursor.fetchall()]

    def get_stock_kline(self, code: str, days: int = 120) -> List[Dict]:
        """获取K线数据（OHLCV + MA）"""
        code = self._normalize_code(code)
        # 获取更多数据用于计算MA
        extra_days = days + 60
        with self.get_stock_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT dq.trade_date, dq.open, dq.high, dq.low, dq.close, dq.volume,
                       dq.price_change_pct
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.code = ?
                ORDER BY dq.trade_date DESC
                LIMIT ?
            """, (code, extra_days))
            rows = cursor.fetchall()
            all_data = [dict(row) for row in reversed(rows)]

        # 计算MA
        closes = [d['close'] for d in all_data]
        for i, d in enumerate(all_data):
            d['ma5'] = round(sum(closes[max(0,i-4):i+1]) / min(5, i+1), 2) if i >= 4 else None
            d['ma10'] = round(sum(closes[max(0,i-9):i+1]) / min(10, i+1), 2) if i >= 9 else None
            d['ma20'] = round(sum(closes[max(0,i-19):i+1]) / min(20, i+1), 2) if i >= 19 else None
            d['ma60'] = round(sum(closes[max(0,i-59):i+1]) / min(60, i+1), 2) if i >= 59 else None

        # 只返回请求的天数
        return all_data[-days:] if len(all_data) > days else all_data

    def get_stock_technical(self, code: str, days: int = 120) -> List[Dict]:
        """获取技术指标（KDJ/MACD/RSI/BOLL）"""
        code = self._normalize_code(code)
        with self.get_stock_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT ti.trade_date,
                       ti.kdj_k, ti.kdj_d, ti.kdj_j,
                       ti.macd_dif, ti.macd_dea,
                       ti.macd_macd as macd_hist,
                       ti.rsi6 as rsi_6, ti.rsi12 as rsi_12, ti.rsi24 as rsi_24,
                       ti.boll_upper, ti.boll_middle as boll_mid, ti.boll_lower
                FROM technical_indicators ti
                JOIN securities s ON ti.security_id = s.id
                WHERE s.code = ?
                ORDER BY ti.trade_date DESC
                LIMIT ?
            """, (code, days))
            rows = cursor.fetchall()
            return [dict(row) for row in reversed(rows)]

    def get_stock_fundamental(self, code: str) -> Optional[Dict]:
        """获取最新基本面数据（PE/PB/市值/换手率）"""
        code = self._normalize_code(code)
        with self.get_stock_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT db.trade_date, db.pe_ttm, db.pb, db.ps_ttm,
                       db.total_mv, db.circ_mv, db.turnover_rate
                FROM daily_basic db
                JOIN securities s ON db.security_id = s.id
                WHERE s.code = ?
                ORDER BY db.trade_date DESC
                LIMIT 1
            """, (code,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_stock_info(self, code: str) -> Optional[Dict]:
        """获取公司信息（合并securities和stock_basic_info）"""
        code = self._normalize_code(code)
        with self.get_stock_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.code, s.name, s.type, s.exchange,
                       s.industry, s.area, s.list_date,
                       sbi.fullname, sbi.market, sbi.main_business
                FROM securities s
                LEFT JOIN stock_basic_info sbi ON sbi.security_id = s.id
                WHERE s.code = ?
            """, (code,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # ==================== Web应用数据库方法 ====================

    def save_training_job(
        self,
        job_id: str,
        model_version: str,
        params: Dict,
        status: str = 'running'
    ):
        """保存训练任务记录"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()

            import json
            cursor.execute('''
                INSERT INTO training_jobs
                (job_id, model_version, start_time, status, params)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                job_id,
                model_version,
                datetime.now().isoformat(),
                status,
                json.dumps(params, ensure_ascii=False)
            ))

            conn.commit()

    def update_training_job_status(
        self,
        job_id: str,
        status: str,
        end_time: Optional[datetime] = None
    ):
        """更新训练任务状态"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()

            if end_time:
                cursor.execute('''
                    UPDATE training_jobs
                    SET status = ?, end_time = ?
                    WHERE job_id = ?
                ''', (status, end_time.isoformat(), job_id))
            else:
                cursor.execute('''
                    UPDATE training_jobs
                    SET status = ?
                    WHERE job_id = ?
                ''', (status, job_id))

            conn.commit()

    def save_training_metrics(
        self,
        job_id: str,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """保存训练指标"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO training_metrics
                (job_id, epoch, loss, accuracy, val_loss, val_accuracy)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                job_id,
                epoch,
                metrics.get('loss'),
                metrics.get('accuracy'),
                metrics.get('val_loss'),
                metrics.get('val_accuracy')
            ))

            conn.commit()

    def get_training_history(
        self,
        model_version: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """获取训练历史记录"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()

            if model_version:
                query = '''
                    SELECT * FROM training_jobs
                    WHERE model_version = ?
                    ORDER BY start_time DESC
                    LIMIT ?
                '''
                cursor.execute(query, (model_version, limit))
            else:
                query = '''
                    SELECT * FROM training_jobs
                    ORDER BY start_time DESC
                    LIMIT ?
                '''
                cursor.execute(query, (limit,))

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    # ==================== 持仓分组管理方法 ====================

    def get_all_groups(self) -> List[Dict]:
        """获取所有分组"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM position_groups
                ORDER BY sort_order, name
            ''')
            return [dict(row) for row in cursor.fetchall()]

    def get_group(self, group_id: int) -> Optional[Dict]:
        """获取单个分组"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM position_groups WHERE id = ?', (group_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def add_group(self, data: Dict) -> int:
        """添加分组"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO position_groups (name, color, description, sort_order)
                VALUES (?, ?, ?, ?)
            ''', (
                data['name'],
                data.get('color', '#6c757d'),
                data.get('description', ''),
                data.get('sort_order', 0)
            ))
            conn.commit()
            return cursor.lastrowid

    def update_group(self, group_id: int, data: Dict) -> bool:
        """更新分组"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()
            updates = []
            values = []
            for key in ['name', 'color', 'description', 'sort_order']:
                if key in data:
                    updates.append(f'{key} = ?')
                    values.append(data[key])
            if not updates:
                return False
            values.append(group_id)
            cursor.execute(f'''
                UPDATE position_groups SET {', '.join(updates)} WHERE id = ?
            ''', values)
            conn.commit()
            return cursor.rowcount > 0

    def delete_group(self, group_id: int) -> bool:
        """删除分组（持仓的group_id会变为NULL）"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()
            # 先将该分组的持仓移到未分组
            cursor.execute('UPDATE positions SET group_id = NULL WHERE group_id = ?', (group_id,))
            cursor.execute('DELETE FROM position_groups WHERE id = ?', (group_id,))
            conn.commit()
            return cursor.rowcount > 0

    def set_position_group(self, position_id: int, group_id: Optional[int]) -> bool:
        """设置持仓的分组"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE positions SET group_id = ? WHERE id = ?', (group_id, position_id))
            conn.commit()
            return cursor.rowcount > 0

    # ==================== 持仓管理方法 ====================

    def get_all_positions(self, status: str = 'holding', group_id: Optional[int] = None) -> List[Dict]:
        """获取所有持仓"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()
            if group_id is not None:
                cursor.execute('''
                    SELECT p.*, g.name as group_name, g.color as group_color
                    FROM positions p
                    LEFT JOIN position_groups g ON p.group_id = g.id
                    WHERE p.status = ? AND p.group_id = ?
                    ORDER BY p.market_value DESC
                ''', (status, group_id))
            else:
                cursor.execute('''
                    SELECT p.*, g.name as group_name, g.color as group_color
                    FROM positions p
                    LEFT JOIN position_groups g ON p.group_id = g.id
                    WHERE p.status = ?
                    ORDER BY g.sort_order, g.name, p.market_value DESC
                ''', (status,))
            return [dict(row) for row in cursor.fetchall()]

    def get_position_by_code(self, code: str, status: str = 'holding') -> Optional[Dict]:
        """根据代码获取持仓"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM positions
                WHERE code = ? AND status = ?
            ''', (code, status))
            row = cursor.fetchone()
            return dict(row) if row else None

    def add_position(self, data: Dict) -> int:
        """添加新持仓"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO positions
                (code, name, quantity, avg_cost, current_price, market_value,
                 profit_loss, profit_loss_pct, first_buy_date, last_update, status, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['code'],
                data.get('name', ''),
                data['quantity'],
                data['avg_cost'],
                data.get('current_price', data['avg_cost']),
                data['quantity'] * data.get('current_price', data['avg_cost']),
                0,
                0,
                data.get('first_buy_date', datetime.now().strftime('%Y-%m-%d')),
                datetime.now().isoformat(),
                'holding',
                data.get('notes', '')
            ))
            conn.commit()
            return cursor.lastrowid

    def update_position(self, position_id: int, data: Dict) -> bool:
        """更新持仓"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()

            # 构建更新语句
            fields = []
            values = []
            for key in ['quantity', 'avg_cost', 'current_price', 'notes', 'status']:
                if key in data:
                    fields.append(f'{key} = ?')
                    values.append(data[key])

            if not fields:
                return False

            # 更新计算字段
            fields.append('last_update = ?')
            values.append(datetime.now().isoformat())

            values.append(position_id)

            cursor.execute(f'''
                UPDATE positions
                SET {', '.join(fields)}
                WHERE id = ?
            ''', values)
            conn.commit()

            # 重新计算市值和盈亏
            self._recalculate_position(position_id)
            return cursor.rowcount > 0

    def _recalculate_position(self, position_id: int):
        """重新计算持仓的市值和盈亏"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM positions WHERE id = ?', (position_id,))
            row = cursor.fetchone()
            if row:
                pos = dict(row)
                quantity = pos['quantity']
                avg_cost = pos['avg_cost']
                current_price = pos['current_price'] or avg_cost

                market_value = quantity * current_price
                cost = quantity * avg_cost
                profit_loss = market_value - cost
                profit_loss_pct = (profit_loss / cost * 100) if cost > 0 else 0

                cursor.execute('''
                    UPDATE positions
                    SET market_value = ?, profit_loss = ?, profit_loss_pct = ?
                    WHERE id = ?
                ''', (market_value, profit_loss, profit_loss_pct, position_id))
                conn.commit()

    def delete_position(self, position_id: int) -> bool:
        """删除持仓"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM positions WHERE id = ?', (position_id,))
            conn.commit()
            return cursor.rowcount > 0

    def update_position_prices(self):
        """更新所有持仓的当前价格"""
        positions = self.get_all_positions()
        updated_count = 0
        for pos in positions:
            code = pos['code']
            # 标准化代码
            normalized_code = self._normalize_code(code)
            # 从主数据库获取最新价格
            with self.get_stock_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT close FROM daily_quotes
                    WHERE security_id = (SELECT id FROM securities WHERE code = ?)
                    ORDER BY trade_date DESC LIMIT 1
                ''', (normalized_code,))
                row = cursor.fetchone()
                if row:
                    self.update_position(pos['id'], {'current_price': row['close']})
                    updated_count += 1
        return updated_count

    # ==================== 交易记录方法 ====================

    def get_all_trades(self, code: str = None, limit: int = 100) -> List[Dict]:
        """获取交易记录"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()
            if code:
                cursor.execute('''
                    SELECT * FROM trades
                    WHERE code = ?
                    ORDER BY trade_date DESC, trade_time DESC
                    LIMIT ?
                ''', (code, limit))
            else:
                cursor.execute('''
                    SELECT * FROM trades
                    ORDER BY trade_date DESC, trade_time DESC
                    LIMIT ?
                ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def add_trade(self, data: Dict) -> int:
        """添加交易记录"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()
            amount = data['quantity'] * data['price']
            cursor.execute('''
                INSERT INTO trades
                (trade_date, trade_time, code, name, action, quantity, price, amount,
                 commission, reason, strategy, signal_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['trade_date'],
                data.get('trade_time', datetime.now().isoformat()),
                data['code'],
                data.get('name', ''),
                data['action'],
                data['quantity'],
                data['price'],
                amount,
                data.get('commission', 0),
                data.get('reason', ''),
                data.get('strategy', ''),
                data.get('signal_source', '')
            ))
            conn.commit()
            return cursor.lastrowid

    def update_trade(self, trade_id: int, data: Dict) -> bool:
        """更新交易记录"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()
            fields = []
            values = []
            for key in ['trade_date', 'code', 'name', 'action', 'quantity', 'price', 'reason', 'strategy']:
                if key in data:
                    fields.append(f'{key} = ?')
                    values.append(data[key])

            if 'quantity' in data and 'price' in data:
                fields.append('amount = ?')
                values.append(data['quantity'] * data['price'])

            if not fields:
                return False

            values.append(trade_id)
            cursor.execute(f'''
                UPDATE trades SET {', '.join(fields)} WHERE id = ?
            ''', values)
            conn.commit()
            return cursor.rowcount > 0

    def delete_trade(self, trade_id: int) -> bool:
        """删除交易记录"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM trades WHERE id = ?', (trade_id,))
            conn.commit()
            return cursor.rowcount > 0

    # ==================== 操作建议方法 ====================

    def get_recommendations(self, date: str = None) -> List[Dict]:
        """获取操作建议（只返回当前持仓的建议）"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()
            # 获取当前持仓的代码列表
            cursor.execute("SELECT code FROM positions WHERE status = 'holding'")
            holding_codes = [row['code'] for row in cursor.fetchall()]

            if not holding_codes:
                return []

            placeholders = ','.join(['?' for _ in holding_codes])

            if date:
                cursor.execute(f'''
                    SELECT * FROM recommendations
                    WHERE date = ? AND code IN ({placeholders})
                    ORDER BY urgency DESC, confidence DESC
                ''', [date] + holding_codes)
            else:
                # 获取最近一天的建议（只针对当前持仓）
                cursor.execute(f'''
                    SELECT * FROM recommendations
                    WHERE date = (SELECT MAX(date) FROM recommendations WHERE code IN ({placeholders}))
                    AND code IN ({placeholders})
                    ORDER BY urgency DESC, confidence DESC
                ''', holding_codes + holding_codes)
            return [dict(row) for row in cursor.fetchall()]

    def add_recommendation(self, data: Dict) -> int:
        """添加操作建议"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO recommendations
                (date, code, name, action, urgency, reason, ml_score,
                 technical_signal, stop_loss_price, take_profit_price, confidence,
                 kelly_position, predicted_return_5d)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['date'],
                data['code'],
                data.get('name', ''),
                data['action'],
                data.get('urgency', 'normal'),
                data.get('reason', ''),
                data.get('ml_score'),
                data.get('technical_signal', ''),
                data.get('stop_loss_price'),
                data.get('take_profit_price'),
                data.get('confidence', 0.5),
                data.get('kelly_position'),
                data.get('predicted_return_5d')
            ))
            conn.commit()
            return cursor.lastrowid

    def mark_recommendation_executed(self, rec_id: int) -> bool:
        """标记建议已执行"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE recommendations
                SET is_executed = 1, executed_at = ?
                WHERE id = ?
            ''', (datetime.now().isoformat(), rec_id))
            conn.commit()
            return cursor.rowcount > 0

    # ==================== 交易评估方法 ====================

    def get_trade_evaluations(self, trade_id: int = None) -> List[Dict]:
        """获取交易评估"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()
            if trade_id:
                cursor.execute('''
                    SELECT e.*, t.code, t.name, t.action, t.trade_date, t.price as trade_price
                    FROM trade_evaluations e
                    JOIN trades t ON e.trade_id = t.id
                    WHERE e.trade_id = ?
                    ORDER BY e.days_after
                ''', (trade_id,))
            else:
                cursor.execute('''
                    SELECT e.*, t.code, t.name, t.action, t.trade_date, t.price as trade_price
                    FROM trade_evaluations e
                    JOIN trades t ON e.trade_id = t.id
                    ORDER BY e.created_at DESC
                    LIMIT 100
                ''')
            return [dict(row) for row in cursor.fetchall()]

    def add_trade_evaluation(self, data: Dict) -> int:
        """添加交易评估"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trade_evaluations
                (trade_id, eval_date, days_after, price_after, return_pct,
                 score, grade, comments, max_profit_pct, max_loss_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['trade_id'],
                data['eval_date'],
                data['days_after'],
                data['price_after'],
                data['return_pct'],
                data['score'],
                data['grade'],
                data.get('comments', ''),
                data.get('max_profit_pct', 0),
                data.get('max_loss_pct', 0)
            ))
            conn.commit()
            return cursor.lastrowid

    def get_evaluation_stats(self) -> Dict:
        """获取评估统计"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()

            # 总体统计
            cursor.execute('''
                SELECT
                    COUNT(*) as total_evals,
                    AVG(score) as avg_score,
                    AVG(return_pct) as avg_return,
                    SUM(CASE WHEN return_pct > 0 THEN 1 ELSE 0 END) as win_count,
                    SUM(CASE WHEN return_pct <= 0 THEN 1 ELSE 0 END) as loss_count
                FROM trade_evaluations
            ''')
            row = cursor.fetchone()
            stats = dict(row) if row else {}

            # 按等级统计
            cursor.execute('''
                SELECT grade, COUNT(*) as count
                FROM trade_evaluations
                GROUP BY grade
                ORDER BY grade
            ''')
            stats['grade_distribution'] = {row['grade']: row['count'] for row in cursor.fetchall()}

            # 计算胜率
            if stats.get('total_evals', 0) > 0:
                stats['win_rate'] = stats['win_count'] / stats['total_evals'] * 100
            else:
                stats['win_rate'] = 0

            return stats

    # ==================== 持仓快照方法 ====================

    def save_position_snapshot(self):
        """保存当日持仓快照"""
        import json

        positions = self.get_all_positions()
        if not positions:
            return None

        total_mv = sum(p['market_value'] or 0 for p in positions)
        total_cost = sum((p['quantity'] * p['avg_cost']) for p in positions)
        total_pl = total_mv - total_cost
        total_pl_pct = (total_pl / total_cost * 100) if total_cost > 0 else 0

        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO position_snapshots
                (snapshot_date, total_market_value, total_cost, total_profit_loss,
                 total_profit_loss_pct, position_count, details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d'),
                total_mv,
                total_cost,
                total_pl,
                total_pl_pct,
                len(positions),
                json.dumps(positions, ensure_ascii=False, default=str)
            ))
            conn.commit()
            return cursor.lastrowid

    def get_position_snapshots(self, days: int = 30) -> List[Dict]:
        """获取持仓快照历史"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM position_snapshots
                ORDER BY snapshot_date DESC
                LIMIT ?
            ''', (days,))
            return [dict(row) for row in cursor.fetchall()]

    # ==================== 组合评分方法 ====================

    def save_portfolio_score(self, score_result: Dict) -> Optional[int]:
        """保存组合评分结果"""
        import json

        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()
            layers = score_result.get('layers', {})
            cursor.execute('''
                INSERT OR REPLACE INTO portfolio_scores
                (score_date, total_score, total_max, total_pct,
                 grade, grade_label,
                 layer1_score, layer1_pct,
                 layer2_score, layer2_pct,
                 layer3_score, layer3_pct,
                 layer4_score, layer4_pct,
                 position_count, total_market_value, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d'),
                score_result.get('total_score'),
                score_result.get('total_max'),
                score_result.get('total_pct'),
                score_result.get('grade'),
                score_result.get('grade_label'),
                layers.get(1, {}).get('score'),
                layers.get(1, {}).get('pct'),
                layers.get(2, {}).get('score'),
                layers.get(2, {}).get('pct'),
                layers.get(3, {}).get('score'),
                layers.get(3, {}).get('pct'),
                layers.get(4, {}).get('score'),
                layers.get(4, {}).get('pct'),
                score_result.get('position_count'),
                score_result.get('total_market_value'),
                json.dumps(score_result, ensure_ascii=False, default=str)
            ))
            conn.commit()
            return cursor.lastrowid

    def get_portfolio_scores(self, days: int = 30) -> List[Dict]:
        """获取组合评分历史"""
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, score_date, total_score, total_max, total_pct,
                       grade, grade_label,
                       layer1_score, layer1_pct,
                       layer2_score, layer2_pct,
                       layer3_score, layer3_pct,
                       layer4_score, layer4_pct,
                       position_count, total_market_value
                FROM portfolio_scores
                ORDER BY score_date DESC
                LIMIT ?
            ''', (days,))
            return [dict(row) for row in cursor.fetchall()]

    def get_portfolio_score_detail(self, score_id: int) -> Optional[Dict]:
        """获取评分详情 (含完整JSON)"""
        import json
        with self.get_webapp_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT details FROM portfolio_scores WHERE id = ?',
                (score_id,))
            row = cursor.fetchone()
            if row and row['details']:
                return json.loads(row['details'])
            return None

    def _normalize_code(self, code: str) -> str:
        """标准化股票代码（移除后缀，因为data_adapter数据库不使用后缀）"""
        if '.' in code:
            # 移除后缀 (.SZ, .SH, .BJ)
            return code.split('.')[0]
        return code

    def get_stock_name(self, code: str) -> str:
        """从主数据库获取股票名称"""
        code = self._normalize_code(code)
        with self.get_stock_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT name FROM securities WHERE code = ?', (code,))
            row = cursor.fetchone()
            return row['name'] if row else ''

    def get_stock_latest_price(self, code: str) -> Optional[float]:
        """获取股票最新价格"""
        code = self._normalize_code(code)
        with self.get_stock_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT close FROM daily_quotes
                WHERE security_id = (SELECT id FROM securities WHERE code = ?)
                ORDER BY trade_date DESC LIMIT 1
            ''', (code,))
            row = cursor.fetchone()
            return row['close'] if row else None

    def get_stock_price_history(self, code: str, start_date: str, end_date: str) -> List[Dict]:
        """获取股票价格历史"""
        code = self._normalize_code(code)
        with self.get_stock_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT trade_date, open, high, low, close, volume
                FROM daily_quotes
                WHERE security_id = (SELECT id FROM securities WHERE code = ?)
                AND trade_date >= ? AND trade_date <= ?
                ORDER BY trade_date
            ''', (code, start_date, end_date))
            return [dict(row) for row in cursor.fetchall()]
