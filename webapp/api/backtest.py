"""
回测API
"""
import os
import re
import time
import subprocess
import logging
import json
from flask import Blueprint, jsonify, request, current_app
from datetime import datetime

from core.task_manager import task_manager, TaskType
from core.report_parser import ReportParser
from api._helpers import api_error_handler, task_progress_sse


logger = logging.getLogger(__name__)

backtest_bp = Blueprint('backtest', __name__)


_QUANT_STRATEGIES = [
    {'id': 'bbi_kdj', 'name': '少负战法', 'description': 'BBI + KDJ组合策略'},
    {'id': 'bbi_short_long', 'name': '补票战法', 'description': 'BBI短长期RSV策略'},
    {'id': 'breakout_volume_kdj', 'name': 'TePu战法', 'description': '成交量突破 + KDJ策略'},
    {'id': 'peak_kdj', 'name': '填坑战法', 'description': '峰值检测 + KDJ策略'},
]


def _get_ml_versions() -> list:
    """动态获取可用的 ML 版本列表"""
    from config import ML_VERSION_DISPLAY_NAMES
    ml_versions = [{'id': 'v3.0', 'name': ML_VERSION_DISPLAY_NAMES['v3.0']}]

    for version in sorted(current_app.config.get('MODEL_DIRS', {}).keys()):
        if version == 'v3.0':
            continue
        name = ML_VERSION_DISPLAY_NAMES.get(version, f'{version.upper()} 模型')
        ml_versions.append({'id': version, 'name': name})

    return ml_versions


@backtest_bp.route('/strategies', methods=['GET'])
@api_error_handler
def get_strategies():
    """获取可用策略列表 + ML 版本"""
    return jsonify({
        'success': True,
        'strategies': _QUANT_STRATEGIES,
        'ml_versions': _get_ml_versions(),
    })


@backtest_bp.route('/run', methods=['POST'])
@api_error_handler
def run_backtest():
    """启动回测任务"""
    data = request.get_json() or {}
    task_id = task_manager.submit_task(
        task_type=TaskType.BACKTEST,
        func=_run_backtest_task,
        metadata={
            'strategies': data.get('strategies', ['bbi_kdj']),
            'ml_version': data.get('ml_version', 'v3.9'),
            'start_date': data.get('start_date', '2024-01-01'),
            'end_date': data.get('end_date', datetime.now().strftime('%Y-%m-%d')),
            'initial_capital': data.get('initial_capital', 1000000),
            'commission': data.get('commission', 0.0003),
            '_backtest_script': str(current_app.config['BACKTEST_SCRIPT']),
            '_python_exec': current_app.config['PYTHON_EXECUTABLE'],
            '_base_dir': str(current_app.config['BASE_DIR']),
            '_task_timeout': current_app.config['TASK_TIMEOUT'],
        }
    )
    return jsonify({'success': True, 'task_id': task_id, 'message': '回测任务已启动'})


@backtest_bp.route('/run/stream', methods=['GET'])
def backtest_stream():
    """SSE: 回测进度流"""
    return task_progress_sse(request.args.get('task_id'))


@backtest_bp.route('/results', methods=['GET'])
@api_error_handler
def get_backtest_results():
    """获取历史回测列表 (JSON 优先 + MD 补充)"""
    limit = int(request.args.get('limit', 50))
    backtest_dir = current_app.config['BACKTEST_DIR']
    if not backtest_dir.exists():
        return jsonify({'success': True, 'results': []})

    results = []
    json_files = sorted(backtest_dir.glob('extensible_backtest_*.json'),
                        key=lambda f: f.stat().st_mtime, reverse=True)
    for jf in json_files[:limit]:
        stat = jf.stat()
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            comparison = data.get('comparison_analysis', {})
            best = comparison.get('best_performance', {})
            results.append({
                'id': jf.stem, 'name': jf.name, 'path': str(jf), 'type': 'json',
                'modified_time': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
                'size': stat.st_size,
                'test_period': data.get('test_period', ''),
                'versions_tested': data.get('versions_tested', []),
                'best_version': best.get('version', ''),
                'best_return': best.get('return', 0),
            })
        except Exception as e:
            logger.warning("解析JSON失败 %s: %s", jf, e)

    md_files = sorted(backtest_dir.glob('*.md'),
                      key=lambda f: f.stat().st_mtime, reverse=True)
    for mf in md_files[:limit - len(results)]:
        stat = mf.stat()
        results.append({
            'id': mf.stem, 'name': mf.name, 'path': str(mf), 'type': 'md',
            'modified_time': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
            'size': stat.st_size,
        })

    return jsonify({'success': True, 'results': results, 'total': len(results)})


@backtest_bp.route('/result/<result_id>', methods=['GET'])
@api_error_handler
def get_backtest_result_detail(result_id: str):
    """获取指定回测结果详情"""
    backtest_dir = current_app.config['BACKTEST_DIR']

    if result_id.endswith('.json'):
        report_file = backtest_dir / result_id
    elif result_id.startswith('extensible_backtest_'):
        report_file = backtest_dir / f'{result_id}.json'
    else:
        possible_json = backtest_dir / f'extensible_backtest_{result_id}.json'
        possible_md = backtest_dir / f'{result_id}.md'
        report_file = possible_json if possible_json.exists() else (
            possible_md if possible_md.exists() else None)

    if not report_file or not report_file.exists():
        return jsonify({'success': False, 'error': f'未找到回测结果: {result_id}'}), 404

    if report_file.suffix == '.json':
        with open(report_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        result_data['file_type'] = 'json'
    else:
        result_data = ReportParser.parse_backtest_report(report_file)
        result_data['file_type'] = 'md'
    result_data['file_path'] = str(report_file)

    return jsonify({'success': True, 'result': result_data})


@backtest_bp.route('/summary', methods=['GET'])
@api_error_handler
def get_backtest_summary():
    """获取回测汇总统计"""
    backtest_dir = current_app.config['BACKTEST_DIR']
    if not backtest_dir.exists():
        return jsonify({
            'success': True,
            'summary': {'total_backtests': 0, 'json_reports': 0, 'md_reports': 0},
        })

    json_files = list(backtest_dir.glob('extensible_backtest_*.json'))
    md_files = list(backtest_dir.glob('*.md'))

    version_stats = {}
    for jf in json_files:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for version, result in data.get('individual_results', {}).items():
                slot = version_stats.setdefault(version, {
                    'count': 0, 'total_return': 0, 'avg_sharpe': 0, 'sharpe_sum': 0
                })
                slot['count'] += 1
                slot['total_return'] += result.get('total_return', 0)
                slot['sharpe_sum'] += result.get('sharpe_ratio', 0)
        except Exception as e:
            logger.warning("aggregate stats: skip corrupt report %s: %s", jf, e)

    for v in version_stats.values():
        if v['count'] > 0:
            v['avg_return'] = round(v['total_return'] / v['count'] * 100, 2)
            v['avg_sharpe'] = round(v['sharpe_sum'] / v['count'], 3)

    latest_report = latest_time = None
    all_files = json_files + md_files
    if all_files:
        latest = max(all_files, key=lambda f: f.stat().st_mtime)
        latest_report = latest.name
        latest_time = datetime.fromtimestamp(latest.stat().st_mtime).strftime('%Y-%m-%d %H:%M')

    return jsonify({
        'success': True,
        'summary': {
            'total_backtests': len(json_files) + len(md_files),
            'json_reports': len(json_files),
            'md_reports': len(md_files),
            'latest_report': latest_report,
            'latest_time': latest_time,
            'version_stats': version_stats,
        }
    })


# ==================== 内部任务执行函数 ====================

# 回测脚本输出关键字 → 进度百分比 映射
_BACKTEST_PHASE_PROGRESS = {
    '依赖加载': 2, '初始化': 5, '注册模型': 8,
    '批量加载': 15, '批量加载完成': 20, '加载完成': 20,
    '量化策略预过滤': 30, '预过滤完成': 35, '评分系统初始化': 40,
    '开始回测': 45, '模拟交易': 50, '计算日收益': 60,
    '交易日': 65, '持仓': 70, '清仓': 75,
    '回测进度': 80, '回测结果': 90, '回测完成': 95,
    '报告已保存': 98, '✅': 20,
}


def _run_backtest_task(progress_callback, **params):
    """执行回测任务 - 实时进度更新版"""
    try:
        progress_callback(0, '准备回测环境...')

        ml_version = params.get('ml_version', 'v3.9')
        ml_version_formatted = ml_version.upper() if ml_version.startswith('v') else ml_version
        start_date = params.get('start_date', '2024-01-01')
        end_date = params.get('end_date', '2025-11-24')

        cmd = [
            params.get('_python_exec', 'python3'),
            params['_backtest_script'],
            '--versions', ml_version_formatted,
            '--start-date', start_date,
            '--end-date', end_date,
            '--save-report',
        ]
        if params.get('initial_capital'):
            cmd.extend(['--capital', str(params['initial_capital'])])

        progress_callback(1, f'启动回测: {ml_version} ({start_date} ~ {end_date})')

        env = os.environ.copy()
        env['PYTHONPATH'] = params['_base_dir']
        env['PYTHONUNBUFFERED'] = '1'

        process = subprocess.Popen(
            cmd, cwd=params['_base_dir'], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
        )

        output_lines = []
        current_progress = 1.0
        start_time = time.time()

        try:
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if not line:
                    continue
                line = line.strip()
                output_lines.append(line)

                new_progress = current_progress
                message = line[:60] + '...' if len(line) > 60 else line

                for keyword, prog in _BACKTEST_PHASE_PROGRESS.items():
                    if keyword in line:
                        new_progress = max(current_progress, prog)
                        break

                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', line)
                if date_match and '回测' in line:
                    new_progress = min(85, current_progress + 0.5)
                if '买入' in line or '卖出' in line:
                    new_progress = min(85, current_progress + 0.2)

                if new_progress > current_progress:
                    current_progress = round(new_progress, 1)
                    progress_callback(current_progress, message)
                elif time.time() - start_time > 5:
                    progress_callback(current_progress, message)

            return_code = process.wait(timeout=60)
            if return_code == 0:
                progress_callback(100, '回测完成')
                return {'success': True, 'output': '\n'.join(output_lines)}
            error_output = '\n'.join(output_lines[-20:])
            raise Exception(f'回测失败 (退出码 {return_code}): {error_output[:500]}')

        except subprocess.TimeoutExpired:
            process.kill()
            raise Exception('回测超时')

    except Exception as e:
        logger.error("回测任务失败: %s", e)
        raise
