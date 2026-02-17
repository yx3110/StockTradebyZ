"""
回测API
"""
import os
import subprocess
import logging
import json
from pathlib import Path
from flask import Blueprint, jsonify, request, Response, current_app
from datetime import datetime

from core.task_manager import task_manager, TaskType
from core.report_parser import ReportParser


logger = logging.getLogger(__name__)

backtest_bp = Blueprint('backtest', __name__)


@backtest_bp.route('/strategies', methods=['GET'])
def get_strategies():
    """
    获取可用策略列表

    Returns:
        {
            "success": True,
            "strategies": [...]
        }
    """
    strategies = [
        {
            'id': 'bbi_kdj',
            'name': '少负战法',
            'description': 'BBI + KDJ组合策略'
        },
        {
            'id': 'bbi_short_long',
            'name': '补票战法',
            'description': 'BBI短长期RSV策略'
        },
        {
            'id': 'breakout_volume_kdj',
            'name': 'TePu战法',
            'description': '成交量突破 + KDJ策略'
        },
        {
            'id': 'peak_kdj',
            'name': '填坑战法',
            'description': '峰值检测 + KDJ策略'
        }
    ]

    # 动态获取ML版本列表
    ml_versions = _get_ml_versions()

    return jsonify({
        'success': True,
        'strategies': strategies,
        'ml_versions': ml_versions
    })


def _get_ml_versions() -> list:
    """
    动态获取可用的ML版本列表

    Returns:
        [{'id': 'v3.9', 'name': 'V3.9 增强特征系统'}, ...]
    """
    # 已知版本的名称映射
    version_names = {
        'v3.0': 'V3.0 量化评分',
        'v3.6': 'V3.6 基础评分系统',
        'v3.7': 'V3.7 三层Ensemble',
        'v3.8': 'V3.8 增量学习',
        'v3.81': 'V3.81 质量元学习器',
        'v3.9': 'V3.9 增强特征',
        'v3.91': 'V3.91 优化特征',
        'v3.92': 'V3.92 高级特征',
        'v3.93': 'V3.93 实验特征',
        'v3.94': 'V3.94 活跃市值优化',
        'v3.95': 'V3.95 多目标滚动预测',
    }

    ml_versions = []

    # 始终添加V3.0基础版本
    ml_versions.append({'id': 'v3.0', 'name': 'V3.0 量化评分'})

    # 从config动态获取已扫描的模型版本
    model_dirs = current_app.config.get('MODEL_DIRS', {})

    for version in sorted(model_dirs.keys()):
        if version == 'v3.0':
            continue  # 已经添加
        name = version_names.get(version, f'{version.upper()} 模型')
        ml_versions.append({'id': version, 'name': name})

    return ml_versions


@backtest_bp.route('/run', methods=['POST'])
def run_backtest():
    """
    启动回测任务

    Request Body:
        {
            "strategies": ["bbi_kdj", "bbi_short_long"],
            "ml_version": "v3.9",
            "start_date": "2024-01-01",
            "end_date": "2025-11-24",
            "initial_capital": 1000000,
            "commission": 0.0003
        }

    Returns:
        {
            "success": True,
            "task_id": "uuid"
        }
    """
    try:
        data = request.get_json() or {}
        strategies = data.get('strategies', ['bbi_kdj'])
        ml_version = data.get('ml_version', 'v3.9')
        start_date = data.get('start_date', '2024-01-01')
        end_date = data.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        initial_capital = data.get('initial_capital', 1000000)
        commission = data.get('commission', 0.0003)

        # 在主线程中获取配置（线程池中无法访问current_app）
        backtest_script = str(current_app.config['BACKTEST_SCRIPT'])
        python_exec = current_app.config['PYTHON_EXECUTABLE']
        base_dir = str(current_app.config['BASE_DIR'])
        task_timeout = current_app.config['TASK_TIMEOUT']

        # 提交回测任务
        task_id = task_manager.submit_task(
            task_type=TaskType.BACKTEST,
            func=_run_backtest_task,
            metadata={
                'strategies': strategies,
                'ml_version': ml_version,
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital,
                'commission': commission,
                # 传入配置（线程中使用）
                '_backtest_script': backtest_script,
                '_python_exec': python_exec,
                '_base_dir': base_dir,
                '_task_timeout': task_timeout
            }
        )

        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': '回测任务已启动'
        })

    except Exception as e:
        logger.error(f'启动回测失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@backtest_bp.route('/run/stream', methods=['GET'])
def backtest_stream():
    """
    SSE端点：实时推送回测进度

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
        # 发送初始retry设置，告诉浏览器如果连接断开应在3秒后重试
        yield "retry: 3000\n\n"
        for progress_data in task_manager.get_task_progress_stream(task_id):
            yield f"data: {json.dumps(progress_data, ensure_ascii=False)}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',  # 保持连接活跃
        }
    )


@backtest_bp.route('/results', methods=['GET'])
def get_backtest_results():
    """
    获取历史回测列表

    Query Parameters:
        limit: 返回数量限制（默认50）

    Returns:
        {
            "success": True,
            "results": [...]
        }
    """
    try:
        limit = int(request.args.get('limit', 50))

        backtest_dir = current_app.config['BACKTEST_DIR']
        if not backtest_dir.exists():
            return jsonify({
                'success': True,
                'results': []
            })

        # 优先获取JSON报告
        results = []

        # 扫描JSON文件
        json_files = list(backtest_dir.glob('extensible_backtest_*.json'))
        json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        for jf in json_files[:limit]:
            stat = jf.stat()
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 提取摘要信息
                summary = {
                    'id': jf.stem,
                    'name': jf.name,
                    'path': str(jf),
                    'type': 'json',
                    'modified_time': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
                    'size': stat.st_size,
                    'test_period': data.get('test_period', ''),
                    'versions_tested': data.get('versions_tested', []),
                }

                # 提取最佳性能
                comparison = data.get('comparison_analysis', {})
                best = comparison.get('best_performance', {})
                summary['best_version'] = best.get('version', '')
                summary['best_return'] = best.get('return', 0)

                results.append(summary)
            except Exception as e:
                logger.warning(f'解析JSON失败: {jf}, {e}')

        # 补充MD报告
        md_files = list(backtest_dir.glob('*.md'))
        md_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        for mf in md_files[:limit - len(results)]:
            stat = mf.stat()
            results.append({
                'id': mf.stem,
                'name': mf.name,
                'path': str(mf),
                'type': 'md',
                'modified_time': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
                'size': stat.st_size,
            })

        return jsonify({
            'success': True,
            'results': results,
            'total': len(results)
        })

    except Exception as e:
        logger.error(f'获取回测结果失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@backtest_bp.route('/result/<result_id>', methods=['GET'])
def get_backtest_result_detail(result_id: str):
    """
    获取指定回测结果详情

    Path Parameters:
        result_id: 回测结果ID（文件名或日期）

    Returns:
        {
            "success": True,
            "result": {...}
        }
    """
    try:
        backtest_dir = current_app.config['BACKTEST_DIR']
        report_file = None

        # 尝试查找JSON文件
        if result_id.endswith('.json'):
            report_file = backtest_dir / result_id
        elif result_id.startswith('extensible_backtest_'):
            report_file = backtest_dir / f'{result_id}.json'
        else:
            # 尝试模糊匹配
            possible_json = backtest_dir / f'extensible_backtest_{result_id}.json'
            possible_md = backtest_dir / f'{result_id}.md'

            if possible_json.exists():
                report_file = possible_json
            elif possible_md.exists():
                report_file = possible_md

        if not report_file or not report_file.exists():
            return jsonify({
                'success': False,
                'error': f'未找到回测结果: {result_id}'
            }), 404

        # 根据文件类型解析
        if report_file.suffix == '.json':
            with open(report_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            result_data['file_type'] = 'json'
        else:
            result_data = ReportParser.parse_backtest_report(report_file)
            result_data['file_type'] = 'md'

        result_data['file_path'] = str(report_file)

        return jsonify({
            'success': True,
            'result': result_data
        })

    except Exception as e:
        logger.error(f'获取回测结果详情失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@backtest_bp.route('/summary', methods=['GET'])
def get_backtest_summary():
    """
    获取回测汇总统计

    Returns:
        {
            "success": True,
            "summary": {...}
        }
    """
    try:
        backtest_dir = current_app.config['BACKTEST_DIR']
        if not backtest_dir.exists():
            return jsonify({
                'success': True,
                'summary': {
                    'total_backtests': 0,
                    'json_reports': 0,
                    'md_reports': 0
                }
            })

        json_files = list(backtest_dir.glob('extensible_backtest_*.json'))
        md_files = list(backtest_dir.glob('*.md'))

        # 统计最佳版本表现
        version_stats = {}
        for jf in json_files:
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for version, result in data.get('individual_results', {}).items():
                    if version not in version_stats:
                        version_stats[version] = {
                            'count': 0,
                            'total_return': 0,
                            'avg_sharpe': 0,
                            'sharpe_sum': 0
                        }
                    version_stats[version]['count'] += 1
                    version_stats[version]['total_return'] += result.get('total_return', 0)
                    version_stats[version]['sharpe_sum'] += result.get('sharpe_ratio', 0)
            except:
                pass

        # 计算平均值
        for v in version_stats.values():
            if v['count'] > 0:
                v['avg_return'] = round(v['total_return'] / v['count'] * 100, 2)
                v['avg_sharpe'] = round(v['sharpe_sum'] / v['count'], 3)

        # 找出最新的回测
        latest_report = None
        latest_time = None
        all_files = json_files + md_files
        if all_files:
            latest = max(all_files, key=lambda f: f.stat().st_mtime)
            latest_report = latest.name
            latest_time = datetime.fromtimestamp(latest.stat().st_mtime).strftime('%Y-%m-%d %H:%M')

        summary = {
            'total_backtests': len(json_files) + len(md_files),
            'json_reports': len(json_files),
            'md_reports': len(md_files),
            'latest_report': latest_report,
            'latest_time': latest_time,
            'version_stats': version_stats
        }

        return jsonify({
            'success': True,
            'summary': summary
        })

    except Exception as e:
        logger.error(f'获取回测汇总失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== 内部任务执行函数 ====================

def _run_backtest_task(progress_callback, **params):
    """
    执行回测任务 - 实时进度更新版

    Args:
        progress_callback: 进度回调函数
        **params: 回测参数（包含配置信息）
    """
    import re
    import time
    import select

    try:
        progress_callback(0, '准备回测环境...')

        # 从参数获取配置
        script_path = params.get('_backtest_script')
        python_exec = params.get('_python_exec', 'python3')
        base_dir = params.get('_base_dir')
        task_timeout = params.get('_task_timeout', 3600)

        # 构建命令
        ml_version = params.get('ml_version', 'v3.9')
        ml_version_formatted = ml_version.upper() if ml_version.startswith('v') else ml_version
        start_date = params.get('start_date', '2024-01-01')
        end_date = params.get('end_date', '2025-11-24')

        cmd = [
            python_exec,
            script_path,
            '--versions', ml_version_formatted,
            '--start-date', start_date,
            '--end-date', end_date,
            '--save-report'
        ]

        initial_capital = params.get('initial_capital')
        if initial_capital:
            cmd.extend(['--capital', str(initial_capital)])

        progress_callback(1, f'启动回测: {ml_version} ({start_date} ~ {end_date})')

        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONPATH'] = base_dir
        env['PYTHONUNBUFFERED'] = '1'  # 禁用Python输出缓冲

        # 使用Popen实时读取输出
        process = subprocess.Popen(
            cmd,
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )

        output_lines = []
        current_progress = 1.0
        start_time = time.time()

        # 进度阶段定义
        phase_progress = {
            '依赖加载': 2,
            '初始化': 5,
            '注册模型': 8,
            '批量加载': 15,
            '批量加载完成': 20,
            '加载完成': 20,
            '量化策略预过滤': 30,
            '预过滤完成': 35,
            '评分系统初始化': 40,
            '开始回测': 45,
            '模拟交易': 50,
            '计算日收益': 60,
            '交易日': 65,
            '持仓': 70,
            '清仓': 75,
            '回测进度': 80,
            '回测结果': 90,
            '回测完成': 95,
            '报告已保存': 98,
            '✅': 20,  # emoji标记通常表示阶段完成
        }

        try:
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break

                if line:
                    line = line.strip()
                    output_lines.append(line)

                    # 根据输出内容更新进度
                    new_progress = current_progress
                    message = line[:60] + '...' if len(line) > 60 else line

                    for keyword, prog in phase_progress.items():
                        if keyword in line:
                            new_progress = max(current_progress, prog)
                            break

                    # 检测日期处理进度 (如果脚本输出日期信息)
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', line)
                    if date_match and '回测' in line:
                        # 粗略估计进度
                        new_progress = min(85, current_progress + 0.5)

                    # 检测交易信号
                    if '买入' in line or '卖出' in line:
                        new_progress = min(85, current_progress + 0.2)

                    # 更新进度（确保单调递增，且有变化时才更新）
                    if new_progress > current_progress:
                        current_progress = round(new_progress, 1)
                        progress_callback(current_progress, message)
                    elif time.time() - start_time > 5:  # 每5秒至少更新一次消息
                        # 保持进度但更新消息
                        progress_callback(current_progress, message)

            # 等待进程结束
            return_code = process.wait(timeout=60)

            if return_code == 0:
                progress_callback(100, '回测完成')
                return {'success': True, 'output': '\n'.join(output_lines)}
            else:
                error_output = '\n'.join(output_lines[-20:])  # 最后20行
                raise Exception(f'回测失败 (退出码 {return_code}): {error_output[:500]}')

        except subprocess.TimeoutExpired:
            process.kill()
            raise Exception('回测超时')

    except Exception as e:
        logger.error(f'回测任务失败: {e}')
        raise
