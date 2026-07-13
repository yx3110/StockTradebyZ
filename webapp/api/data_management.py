"""
数据管理API - 数据完整度检查、Backfill任务管理
"""
import subprocess
import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from flask import Blueprint, jsonify, request, current_app
from datetime import datetime
from typing import Dict, List, Any

from core.task_manager import task_manager, TaskType
from api._helpers import api_error_handler, task_progress_sse

logger = logging.getLogger(__name__)

data_management_bp = Blueprint('data_management', __name__)


_BACKFILL_SCRIPTS = [
    {'name': 'v39_features', 'description': 'V3.9特征缓存预计算',
     'script': 'precompute_v39_features.py', 'estimated_time': '约30分钟/月数据'},
    {'name': 'daily_basic', 'description': '日线基本面数据 (PE/PB/市值)',
     'script': 'fetch_data/v39_data_backfill.py', 'estimated_time': '约10分钟/月数据'},
    {'name': 'daily_quotes', 'description': '日线行情数据',
     'script': 'fetch_data/quick_daily_update.py', 'estimated_time': '约5分钟/日数据'},
    {'name': 'technical_indicators', 'description': '技术指标计算',
     'script': 'fetch_data/technical_indicator_calculator.py', 'estimated_time': '约15分钟/月数据'},
    {'name': 'active_mv', 'description': '活跃市值特征回填',
     'script': 'backfill_active_mv_for_v39.py', 'estimated_time': '约20分钟/月数据'},
]


# ==================== 数据完整度检查 API ====================

@data_management_bp.route('/completeness', methods=['GET'])
@api_error_handler
def get_data_completeness():
    """获取数据完整度概览"""
    start_date = request.args.get('start_date', '2020-01-01')
    end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    completeness = _check_data_completeness(
        current_app.config['STOCK_DB_PATH'], start_date, end_date)
    return jsonify({
        'success': True, 'start_date': start_date, 'end_date': end_date,
        'completeness': completeness,
    })


@data_management_bp.route('/completeness/daily', methods=['GET'])
@api_error_handler
def get_daily_completeness():
    """获取每日数据完整度详情"""
    table = request.args.get('table', 'daily_quotes')
    start_date = request.args.get('start_date', '2024-01-01')
    end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    limit = int(request.args.get('limit', 365))
    daily_stats = _get_daily_stats(
        current_app.config['STOCK_DB_PATH'], table, start_date, end_date, limit)
    return jsonify({'success': True, 'table': table, 'daily_stats': daily_stats})


@data_management_bp.route('/completeness/missing', methods=['GET'])
@api_error_handler
def get_missing_dates():
    """获取缺失数据的日期列表"""
    table = request.args.get('table', 'v39_feature_cache')
    start_date = request.args.get('start_date', '2024-01-01')
    end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    threshold = float(request.args.get('threshold', 80))
    missing_dates = _get_missing_dates(
        current_app.config['STOCK_DB_PATH'], table, start_date, end_date, threshold)
    return jsonify({
        'success': True, 'table': table, 'threshold': threshold,
        'missing_dates': missing_dates, 'missing_count': len(missing_dates),
    })


@data_management_bp.route('/stats', methods=['GET'])
@api_error_handler
def get_database_stats():
    """获取数据库整体统计"""
    return jsonify({
        'success': True,
        'stats': _get_database_stats(current_app.config['STOCK_DB_PATH']),
    })


# ==================== Backfill 任务管理 API ====================

@data_management_bp.route('/backfill/start', methods=['POST'])
@api_error_handler
def start_backfill():
    """启动 Backfill 任务"""
    data = request.get_json() or {}
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    if not start_date or not end_date:
        return jsonify({'success': False, 'error': '请提供start_date和end_date'}), 400

    task_type = data.get('task_type', 'v39_features')
    task_id = task_manager.submit_task(
        task_type=TaskType.DATA_UPDATE,
        func=_run_backfill_task,
        metadata={
            'task_type': task_type,
            'start_date': start_date, 'end_date': end_date,
            'workers': data.get('workers', 4),
            'batch_size': data.get('batch_size', 50),
            '_python_exec': current_app.config['PYTHON_EXECUTABLE'],
            '_base_dir': str(current_app.config['BASE_DIR']),
            '_task_timeout': current_app.config.get('TASK_TIMEOUT', 7200),
        }
    )
    return jsonify({
        'success': True, 'task_id': task_id,
        'message': f'Backfill任务已启动 ({task_type}: {start_date} ~ {end_date})',
    })


@data_management_bp.route('/backfill/stream', methods=['GET'])
def backfill_stream():
    """SSE: Backfill 进度流"""
    return task_progress_sse(request.args.get('task_id'))


@data_management_bp.route('/backfill/history', methods=['GET'])
@api_error_handler
def get_backfill_history():
    """获取 Backfill 任务历史"""
    limit = int(request.args.get('limit', 20))
    return jsonify({
        'success': True,
        'history': _get_backfill_history(current_app.config['STOCK_DB_PATH'], limit),
    })


@data_management_bp.route('/backfill/scripts', methods=['GET'])
@api_error_handler
def get_available_scripts():
    """获取可用的 Backfill 脚本列表"""
    return jsonify({'success': True, 'scripts': _BACKFILL_SCRIPTS})


# ==================== 辅助函数 ====================

def _check_data_completeness(db_path: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """检查各表的数据完整度"""
    tables_config = {
        'daily_quotes': {
            'date_column': 'trade_date',
            'count_column': 'security_id',
            'expected_daily': 7300  # 预期每日证券数（A股+ETF约7300只）
        },
        'daily_basic': {
            'date_column': 'trade_date',
            'count_column': 'security_id',
            'expected_daily': 5600  # 仅A股有基本面数据
        },
        'technical_indicators': {
            'date_column': 'trade_date',
            'count_column': 'security_id',
            'expected_daily': 7300
        },
        'v39_feature_cache': {
            'date_column': 'trade_date',
            'count_column': 'code',
            'expected_daily': 5400  # v39特征涵盖的A股数
        }
    }

    result = {}

    with closing(sqlite3.connect(db_path, timeout=30.0)) as conn:
        conn.execute("PRAGMA busy_timeout=30000")
        for table, config in tables_config.items():
            try:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table,)
                )
                if not cursor.fetchone():
                    result[table] = {
                        'exists': False,
                        'total_records': 0,
                        'date_range': None,
                        'completeness_pct': 0
                    }
                    continue

                query = f"""
                    SELECT
                        COUNT(*) as total,
                        MIN({config['date_column']}) as min_date,
                        MAX({config['date_column']}) as max_date,
                        COUNT(DISTINCT {config['date_column']}) as trading_days
                    FROM {table}
                    WHERE {config['date_column']} >= ? AND {config['date_column']} <= ?
                """
                total, min_date, max_date, trading_days = conn.execute(
                    query, (start_date, end_date)).fetchone()

                expected_total = trading_days * config['expected_daily'] if trading_days else 0
                completeness = (total / expected_total * 100) if expected_total > 0 else 0

                threshold = int(config['expected_daily'] * 0.8)
                complete_days_query = f"""
                    SELECT COUNT(*) FROM (
                        SELECT {config['date_column']}, COUNT(*) as cnt
                        FROM {table}
                        WHERE {config['date_column']} >= ? AND {config['date_column']} <= ?
                        GROUP BY {config['date_column']}
                        HAVING cnt >= ?
                    )
                """
                complete_days = conn.execute(
                    complete_days_query, (start_date, end_date, threshold)).fetchone()[0]

                expected_days = conn.execute("""
                    SELECT COUNT(DISTINCT trade_date) FROM daily_quotes
                    WHERE trade_date >= ? AND trade_date <= ?
                """, (start_date, end_date)).fetchone()[0]

                day_coverage = (trading_days / expected_days * 100) if expected_days > 0 else 0

                result[table] = {
                    'exists': True,
                    'total_records': total,
                    'date_range': {'start': min_date, 'end': max_date},
                    'trading_days': trading_days,
                    'expected_days': expected_days,
                    'complete_days': complete_days,
                    'expected_daily': config['expected_daily'],
                    'completeness_pct': round(completeness, 2),
                    'day_coverage_pct': round(day_coverage, 2)
                }

            except Exception as e:
                logger.error("检查 %s 完整度失败: %s", table, e)
                result[table] = {'exists': False, 'error': str(e)}

    return result


def _get_daily_stats(db_path: str, table: str, start_date: str, end_date: str, limit: int) -> List[Dict]:
    """获取每日数据统计"""
    # 确定日期列和计数列 (table 白名单 — table 来自 request.args, 必须校验以防 SQL 注入 / 任意表读取)
    table_config = {
        'v39_feature_cache': {'date_col': 'trade_date', 'count_col': 'code', 'expected': 5400},
        'daily_quotes': {'date_col': 'trade_date', 'count_col': 'security_id', 'expected': 7300},
        'daily_basic': {'date_col': 'trade_date', 'count_col': 'security_id', 'expected': 5600},
        'technical_indicators': {'date_col': 'trade_date', 'count_col': 'security_id', 'expected': 7300},
    }
    config = table_config.get(table)
    if config is None:
        logger.warning('_get_daily_stats: 非法 table 参数被拒绝: %r', table)
        return []
    date_col = config['date_col']
    count_col = config['count_col']
    expected = config['expected']

    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA busy_timeout=30000")

    try:
        query = f"""
            SELECT
                {date_col} as date,
                COUNT(DISTINCT {count_col}) as count
            FROM {table}
            WHERE {date_col} >= ? AND {date_col} <= ?
            GROUP BY {date_col}
            ORDER BY {date_col} DESC
            LIMIT ?
        """
        cursor = conn.execute(query, (start_date, end_date, limit))
        rows = cursor.fetchall()

        result = []
        for row in rows:
            date, count = row
            completeness = (count / expected * 100) if expected > 0 else 0
            result.append({
                'date': date,
                'count': count,
                'expected': expected,
                'completeness': round(completeness, 2)
            })

        return result

    except Exception as e:
        logger.error(f'获取{table}每日统计失败: {e}')
        return []
    finally:
        conn.close()


def _get_missing_dates(db_path: str, table: str, start_date: str, end_date: str, threshold: float) -> List[str]:
    """获取缺失或不完整的日期。

    以 daily_quotes 的交易日为期望全集: 目标表中某交易日
    (a) 完整度低于阈值, 或 (b) 完全没有数据, 都算缺失。
    修复前只看 _get_daily_stats 返回的日期 (GROUP BY 只含有数据的日期),
    完全缺失的交易日被静默漏掉, 且 LIMIT 1000 会截断长区间。
    """
    with closing(sqlite3.connect(db_path, timeout=30.0)) as conn:
        conn.execute("PRAGMA busy_timeout=30000")
        trading_dates = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT trade_date FROM daily_quotes "
                "WHERE trade_date >= ? AND trade_date <= ? ORDER BY trade_date",
                (start_date, end_date)
            ).fetchall()
        ]

    if not trading_dates:
        return []

    # limit 按交易日数量给足, 避免硬编码 1000 截断
    daily_stats = _get_daily_stats(db_path, table, start_date, end_date, len(trading_dates) + 10)
    ok_dates = {s['date'] for s in daily_stats if s['completeness'] >= threshold}

    missing = [d for d in trading_dates if d not in ok_dates]
    return sorted(missing)


def _get_database_stats(db_path: str) -> Dict[str, Any]:
    """获取数据库整体统计"""
    db_path_str = str(db_path)
    stats = {
        'db_path': db_path_str,
        'db_size_mb': 0,
        'tables': {}
    }

    db_file = Path(db_path)
    if db_file.exists():
        stats['db_size_mb'] = round(db_file.stat().st_size / (1024 * 1024), 2)

    try:
        with closing(sqlite3.connect(db_path_str, timeout=30.0)) as conn:
            conn.execute("PRAGMA busy_timeout=30000")

            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]

            for table in tables:
                try:
                    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    stats['tables'][table] = {'count': count}
                except Exception as e:
                    logger.warning("db stats: skip table %s: %s", table, e)

            try:
                stats['active_securities'] = conn.execute(
                    "SELECT COUNT(*) FROM securities WHERE is_active = 1").fetchone()[0]
            except Exception as e:
                logger.warning("active securities query failed: %s", e)
                stats['active_securities'] = 0

            try:
                row = conn.execute(
                    "SELECT MIN(trade_date), MAX(trade_date) FROM daily_quotes").fetchone()
                stats['date_range'] = {'start': row[0], 'end': row[1]}
            except Exception as e:
                logger.warning("date range query failed: %s", e)
                stats['date_range'] = None
    except Exception as e:
        logger.error("获取数据库统计失败: %s", e)

    return stats


def _get_backfill_history(db_path: str, limit: int) -> List[Dict]:
    """获取backfill历史记录"""
    history = []
    try:
        with closing(sqlite3.connect(db_path, timeout=30.0)) as conn:
            conn.execute("PRAGMA busy_timeout=30000")
            rows = conn.execute("""
                SELECT
                    id, update_date, update_type,
                    securities_updated, records_added, records_updated,
                    status, error_message, duration_seconds, created_at
                FROM data_update_log
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
        for row in rows:
            history.append({
                'id': row[0],
                'update_date': row[1],
                'update_type': row[2],
                'securities_updated': row[3],
                'records_added': row[4],
                'records_updated': row[5],
                'status': row[6],
                'error_message': row[7],
                'duration_seconds': row[8],
                'created_at': row[9]
            })
    except Exception as e:
        logger.error("获取backfill历史失败: %s", e)
    return history


def _run_backfill_task(progress_callback, **params):
    """
    执行Backfill任务

    Args:
        progress_callback: 进度回调函数
        **params: 任务参数
    """
    try:
        progress_callback(0, '开始Backfill任务...')

        task_type = params.get('task_type', 'v39_features')
        start_date = params.get('start_date')
        end_date = params.get('end_date')
        workers = params.get('workers', 4)
        batch_size = params.get('batch_size', 50)

        python_exec = params.get('_python_exec', 'python3')
        base_dir = params.get('_base_dir')
        task_timeout = params.get('_task_timeout', 7200)

        # 根据任务类型选择脚本
        script_map = {
            'v39_features': 'precompute_v39_features.py',
            'daily_basic': 'fetch_data/v39_data_backfill.py',
            'daily_quotes': 'fetch_data/quick_daily_update.py',
            'technical_indicators': 'fetch_data/technical_indicator_calculator.py',
            'active_mv': 'backfill_active_mv_for_v39.py'
        }

        script = script_map.get(task_type)
        if not script:
            raise ValueError(f'未知的任务类型: {task_type}')

        progress_callback(10, f'执行脚本: {script}')

        # 构建命令
        cmd = [python_exec, script]

        # 添加参数
        if task_type == 'v39_features':
            cmd.extend([
                '--start-date', start_date,
                '--end-date', end_date,
                '--num-workers', str(workers),
                '--batch-size', str(batch_size)
            ])
        elif task_type in ['daily_basic', 'technical_indicators']:
            cmd.extend([
                '--start-date', start_date,
                '--end-date', end_date
            ])
        elif task_type == 'daily_quotes':
            # quick_daily_update只处理单日
            cmd.extend(['--date', end_date])
        elif task_type == 'active_mv':
            cmd.extend([
                '--start-date', start_date,
                '--end-date', end_date
            ])

        progress_callback(20, f'运行命令: {" ".join(cmd)}')

        # 执行脚本
        result = subprocess.run(
            cmd,
            cwd=base_dir,
            capture_output=True,
            text=True,
            timeout=task_timeout
        )

        if result.returncode == 0:
            progress_callback(100, 'Backfill任务完成')
            return {
                'success': True,
                'output': result.stdout[-2000:] if result.stdout else ''  # 限制输出长度
            }
        else:
            error_msg = result.stderr[:500] if result.stderr else '未知错误'
            raise Exception(f'脚本执行失败: {error_msg}')

    except subprocess.TimeoutExpired:
        raise Exception(f'任务超时（超过{task_timeout/3600:.1f}小时）')
    except Exception as e:
        logger.error(f'Backfill任务失败: {e}')
        raise
