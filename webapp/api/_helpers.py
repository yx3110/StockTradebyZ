"""
api blueprint shared helpers
"""
import json
import logging
from functools import wraps
from flask import jsonify, Response, current_app, request, abort
from werkzeug.exceptions import HTTPException

from core.task_manager import task_manager
from core.database import DatabaseManager

# 按 (stock_db, webapp_db) 路径缓存 DatabaseManager 实例。
# DatabaseManager 每次方法调用都新开连接 (无共享连接状态), 因此实例可跨线程复用;
# 否则每个请求都会重新执行 _init_webapp_db() 的一堆 CREATE TABLE / ALTER + commit。
_DB_MANAGER_CACHE = {}


def get_db_manager() -> DatabaseManager:
    """获取 (缓存的) DatabaseManager 实例 (使用当前 app config 的 DB 路径)。"""
    key = (str(current_app.config['STOCK_DB_PATH']),
           str(current_app.config['WEBAPP_DB_PATH']))
    mgr = _DB_MANAGER_CACHE.get(key)
    if mgr is None:
        mgr = DatabaseManager(
            current_app.config['STOCK_DB_PATH'],
            current_app.config['WEBAPP_DB_PATH'],
        )
        _DB_MANAGER_CACHE[key] = mgr
    return mgr


def parse_int_arg(name: str, default):
    """从 request.args 解析整数参数, 非法输入返回 400 (而非 500)。"""
    raw = request.args.get(name, default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        abort(400, description=f'参数 {name} 必须是整数, 收到: {raw!r}')


def api_error_handler(view_func):
    """Wrap a Flask view in try/except → JSON error response.

    Logs with the view's own module logger so the log line still attributes
    to api/portfolio.py etc. instead of api/_helpers.py.

    Use after the route decorator:

        @bp.route('/foo')
        @api_error_handler
        def foo():
            ...
            return jsonify({'success': True, ...})
    """
    logger = logging.getLogger(view_func.__module__)

    @wraps(view_func)
    def wrapper(*args, **kwargs):
        try:
            return view_func(*args, **kwargs)
        except HTTPException as e:
            # abort(400/404/...) 应保留原状态码 (返回 JSON), 而不是被下面统一改写成 500
            return jsonify({'success': False, 'error': e.description}), e.code
        except Exception as e:
            logger.error("%s failed: %s", view_func.__name__, e, exc_info=True)
            return jsonify({'success': False, 'error': str(e)}), 500

    return wrapper


def task_progress_sse(task_id):
    """Return an SSE Response streaming progress events for `task_id`.

    Mirrors the boilerplate previously copy-pasted across daily_tasks /
    backtest / data_management / model_training stream endpoints.
    Returns 400 JSON if task_id is missing.
    """
    if not task_id:
        return jsonify({'error': 'Missing task_id'}), 400

    def generate():
        for event in task_manager.get_task_progress_stream(task_id):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )
