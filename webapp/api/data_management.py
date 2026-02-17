"""
数据管理API - 数据完整度检查、Backfill任务管理
"""
import subprocess
import logging
import json
import sqlite3
from pathlib import Path
from flask import Blueprint, jsonify, request, Response, current_app
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

from core.task_manager import task_manager, TaskType
from core.database import DatabaseManager

logger = logging.getLogger(__name__)

data_management_bp = Blueprint('data_management', __name__)


# ==================== 数据完整度检查 API ====================

@data_management_bp.route('/completeness', methods=['GET'])
def get_data_completeness():
    """
    获取数据完整度概览

    Query Parameters:
        start_date: 开始日期 (默认: 2020-01-01)
        end_date: 结束日期 (默认: 今天)

    Returns:
        {
            "success": True,
            "completeness": {
                "daily_quotes": {...},
                "daily_basic": {...},
                "technical_indicators": {...},
                "v39_feature_cache": {...}
            }
        }
    """
    try:
        start_date = request.args.get('start_date', '2020-01-01')
        end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))

        db_path = current_app.config['STOCK_DB_PATH']
        completeness = _check_data_completeness(db_path, start_date, end_date)

        return jsonify({
            'success': True,
            'start_date': start_date,
            'end_date': end_date,
            'completeness': completeness
        })

    except Exception as e:
        logger.error(f'获取数据完整度失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@data_management_bp.route('/completeness/daily', methods=['GET'])
def get_daily_completeness():
    """
    获取每日数据完整度详情

    Query Parameters:
        table: 表名 (daily_quotes, daily_basic, technical_indicators, v39_feature_cache)
        start_date: 开始日期
        end_date: 结束日期
        limit: 返回数量限制

    Returns:
        {
            "success": True,
            "daily_stats": [
                {"date": "2025-01-01", "count": 5000, "expected": 5400, "completeness": 92.6},
                ...
            ]
        }
    """
    try:
        table = request.args.get('table', 'daily_quotes')
        start_date = request.args.get('start_date', '2024-01-01')
        end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        limit = int(request.args.get('limit', 365))

        db_path = current_app.config['STOCK_DB_PATH']
        daily_stats = _get_daily_stats(db_path, table, start_date, end_date, limit)

        return jsonify({
            'success': True,
            'table': table,
            'daily_stats': daily_stats
        })

    except Exception as e:
        logger.error(f'获取每日完整度失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@data_management_bp.route('/completeness/missing', methods=['GET'])
def get_missing_dates():
    """
    获取缺失数据的日期列表

    Query Parameters:
        table: 表名
        start_date: 开始日期
        end_date: 结束日期
        threshold: 完整度阈值 (低于此值视为缺失，默认80)

    Returns:
        {
            "success": True,
            "missing_dates": ["2025-01-02", "2025-01-03", ...]
        }
    """
    try:
        table = request.args.get('table', 'v39_feature_cache')
        start_date = request.args.get('start_date', '2024-01-01')
        end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        threshold = float(request.args.get('threshold', 80))

        db_path = current_app.config['STOCK_DB_PATH']
        missing_dates = _get_missing_dates(db_path, table, start_date, end_date, threshold)

        return jsonify({
            'success': True,
            'table': table,
            'threshold': threshold,
            'missing_dates': missing_dates,
            'missing_count': len(missing_dates)
        })

    except Exception as e:
        logger.error(f'获取缺失日期失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@data_management_bp.route('/stats', methods=['GET'])
def get_database_stats():
    """
    获取数据库统计信息

    Returns:
        {
            "success": True,
            "stats": {
                "securities_count": 5400,
                "total_quotes": 10000000,
                "date_range": {"start": "2018-01-01", "end": "2025-12-14"},
                "tables": {...}
            }
        }
    """
    try:
        db_path = current_app.config['STOCK_DB_PATH']
        stats = _get_database_stats(db_path)

        return jsonify({
            'success': True,
            'stats': stats
        })

    except Exception as e:
        logger.error(f'获取数据库统计失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== Backfill 任务管理 API ====================

@data_management_bp.route('/backfill/start', methods=['POST'])
def start_backfill():
    """
    启动Backfill任务

    Request Body:
        {
            "task_type": "v39_features",  // v39_features, daily_basic, daily_quotes
            "start_date": "2024-01-01",
            "end_date": "2024-06-30",
            "workers": 4,
            "batch_size": 50
        }

    Returns:
        {
            "success": True,
            "task_id": "uuid"
        }
    """
    try:
        data = request.get_json() or {}
        task_type = data.get('task_type', 'v39_features')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        workers = data.get('workers', 4)
        batch_size = data.get('batch_size', 50)

        if not start_date or not end_date:
            return jsonify({
                'success': False,
                'error': '请提供start_date和end_date'
            }), 400

        # 获取配置
        python_exec = current_app.config['PYTHON_EXECUTABLE']
        base_dir = str(current_app.config['BASE_DIR'])
        task_timeout = current_app.config.get('TASK_TIMEOUT', 7200)  # 2小时

        # 提交任务
        task_id = task_manager.submit_task(
            task_type=TaskType.DATA_UPDATE,
            func=_run_backfill_task,
            metadata={
                'task_type': task_type,
                'start_date': start_date,
                'end_date': end_date,
                'workers': workers,
                'batch_size': batch_size,
                '_python_exec': python_exec,
                '_base_dir': base_dir,
                '_task_timeout': task_timeout
            }
        )

        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': f'Backfill任务已启动 ({task_type}: {start_date} ~ {end_date})'
        })

    except Exception as e:
        logger.error(f'启动Backfill任务失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@data_management_bp.route('/backfill/stream', methods=['GET'])
def backfill_stream():
    """
    SSE端点：实时推送Backfill进度

    Query Parameters:
        task_id: 任务ID

    Returns:
        Server-Sent Events stream
    """
    task_id = request.args.get('task_id')

    if not task_id:
        return jsonify({'error': 'Missing task_id'}), 400

    def generate():
        """生成SSE数据流"""
        for progress_data in task_manager.get_task_progress_stream(task_id):
            yield f"data: {json.dumps(progress_data, ensure_ascii=False)}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@data_management_bp.route('/backfill/history', methods=['GET'])
def get_backfill_history():
    """
    获取Backfill任务历史

    Returns:
        {
            "success": True,
            "history": [...]
        }
    """
    try:
        limit = int(request.args.get('limit', 20))

        db_manager = DatabaseManager(
            current_app.config['STOCK_DB_PATH'],
            current_app.config['WEBAPP_DB_PATH']
        )

        # 从data_update_log表获取历史
        history = _get_backfill_history(current_app.config['STOCK_DB_PATH'], limit)

        return jsonify({
            'success': True,
            'history': history
        })

    except Exception as e:
        logger.error(f'获取Backfill历史失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@data_management_bp.route('/backfill/scripts', methods=['GET'])
def get_available_scripts():
    """
    获取可用的Backfill脚本列表

    Returns:
        {
            "success": True,
            "scripts": [
                {"name": "v39_features", "description": "..."},
                ...
            ]
        }
    """
    scripts = [
        {
            'name': 'v39_features',
            'description': 'V3.9特征缓存预计算',
            'script': 'precompute_v39_features.py',
            'estimated_time': '约30分钟/月数据'
        },
        {
            'name': 'daily_basic',
            'description': '日线基本面数据 (PE/PB/市值)',
            'script': 'fetch_data/v39_data_initializer.py',
            'estimated_time': '约10分钟/月数据'
        },
        {
            'name': 'daily_quotes',
            'description': '日线行情数据',
            'script': 'fetch_data/quick_daily_update.py',
            'estimated_time': '约5分钟/日数据'
        },
        {
            'name': 'technical_indicators',
            'description': '技术指标计算',
            'script': 'fetch_data/technical_indicator_calculator.py',
            'estimated_time': '约15分钟/月数据'
        },
        {
            'name': 'active_mv',
            'description': '活跃市值特征回填',
            'script': 'backfill_active_mv_for_v39.py',
            'estimated_time': '约20分钟/月数据'
        }
    ]

    return jsonify({
        'success': True,
        'scripts': scripts
    })


# ==================== 辅助函数 ====================

def _check_data_completeness(db_path: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """检查各表的数据完整度"""
    conn = sqlite3.connect(db_path)

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

    for table, config in tables_config.items():
        try:
            # 检查表是否存在
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

            # 获取统计信息
            query = f"""
                SELECT
                    COUNT(*) as total,
                    MIN({config['date_column']}) as min_date,
                    MAX({config['date_column']}) as max_date,
                    COUNT(DISTINCT {config['date_column']}) as trading_days
                FROM {table}
                WHERE {config['date_column']} >= ? AND {config['date_column']} <= ?
            """
            cursor = conn.execute(query, (start_date, end_date))
            row = cursor.fetchone()

            total, min_date, max_date, trading_days = row

            # 计算完整度（按记录数）
            expected_total = trading_days * config['expected_daily'] if trading_days else 0
            completeness = (total / expected_total * 100) if expected_total > 0 else 0

            # 计算完整交易日数（每日记录数 >= 预期的80%）
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
            cursor = conn.execute(complete_days_query, (start_date, end_date, threshold))
            complete_days = cursor.fetchone()[0]

            # 查询日期范围内应有的交易日数（从daily_quotes获取）
            expected_days_query = """
                SELECT COUNT(DISTINCT trade_date) FROM daily_quotes
                WHERE trade_date >= ? AND trade_date <= ?
            """
            cursor = conn.execute(expected_days_query, (start_date, end_date))
            expected_days = cursor.fetchone()[0]

            # 交易日覆盖率
            day_coverage = (trading_days / expected_days * 100) if expected_days > 0 else 0

            result[table] = {
                'exists': True,
                'total_records': total,
                'date_range': {
                    'start': min_date,
                    'end': max_date
                },
                'trading_days': trading_days,
                'expected_days': expected_days,
                'complete_days': complete_days,
                'expected_daily': config['expected_daily'],
                'completeness_pct': round(completeness, 2),
                'day_coverage_pct': round(day_coverage, 2)
            }

        except Exception as e:
            logger.error(f'检查{table}完整度失败: {e}')
            result[table] = {
                'exists': False,
                'error': str(e)
            }

    conn.close()
    return result


def _get_daily_stats(db_path: str, table: str, start_date: str, end_date: str, limit: int) -> List[Dict]:
    """获取每日数据统计"""
    conn = sqlite3.connect(db_path)

    # 确定日期列和计数列
    table_config = {
        'v39_feature_cache': {'date_col': 'trade_date', 'count_col': 'code', 'expected': 5400},
        'daily_quotes': {'date_col': 'trade_date', 'count_col': 'security_id', 'expected': 7300},
        'daily_basic': {'date_col': 'trade_date', 'count_col': 'security_id', 'expected': 5600},
        'technical_indicators': {'date_col': 'trade_date', 'count_col': 'security_id', 'expected': 7300},
    }
    config = table_config.get(table, {'date_col': 'trade_date', 'count_col': 'security_id', 'expected': 7300})
    date_col = config['date_col']
    count_col = config['count_col']
    expected = config['expected']

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
    """获取缺失或不完整的日期"""
    daily_stats = _get_daily_stats(db_path, table, start_date, end_date, 1000)

    # 找出低于阈值的日期
    missing = [
        stat['date'] for stat in daily_stats
        if stat['completeness'] < threshold
    ]

    return sorted(missing)


def _get_database_stats(db_path: str) -> Dict[str, Any]:
    """获取数据库整体统计"""
    # 确保db_path是字符串
    db_path_str = str(db_path)
    conn = sqlite3.connect(db_path_str)

    stats = {
        'db_path': db_path_str,
        'db_size_mb': 0,
        'tables': {}
    }

    try:
        # 数据库文件大小
        db_file = Path(db_path)
        if db_file.exists():
            stats['db_size_mb'] = round(db_file.stat().st_size / (1024 * 1024), 2)

        # 获取所有表的统计
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            try:
                count_cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                count = count_cursor.fetchone()[0]
                stats['tables'][table] = {'count': count}
            except:
                pass

        # 证券数量
        try:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM securities WHERE is_active = 1"
            )
            stats['active_securities'] = cursor.fetchone()[0]
        except:
            stats['active_securities'] = 0

        # 日期范围
        try:
            cursor = conn.execute(
                "SELECT MIN(trade_date), MAX(trade_date) FROM daily_quotes"
            )
            row = cursor.fetchone()
            stats['date_range'] = {
                'start': row[0],
                'end': row[1]
            }
        except:
            stats['date_range'] = None

    except Exception as e:
        logger.error(f'获取数据库统计失败: {e}')

    conn.close()
    return stats


def _get_backfill_history(db_path: str, limit: int) -> List[Dict]:
    """获取backfill历史记录"""
    conn = sqlite3.connect(db_path)
    history = []

    try:
        cursor = conn.execute("""
            SELECT
                id, update_date, update_type,
                securities_updated, records_added, records_updated,
                status, error_message, duration_seconds, created_at
            FROM data_update_log
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))

        for row in cursor.fetchall():
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
        logger.error(f'获取backfill历史失败: {e}')

    conn.close()
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
            'daily_basic': 'fetch_data/v39_data_initializer.py',
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
