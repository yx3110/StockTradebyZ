"""
模型训练API - 模型管理、训练和性能指标展示
"""
import subprocess
import logging
import json
import csv
import re
import pickle
from pathlib import Path
from flask import Blueprint, jsonify, request, Response, current_app
from datetime import datetime
from typing import Dict, List, Any, Optional

from core.task_manager import task_manager, TaskType
from core.database import DatabaseManager


logger = logging.getLogger(__name__)

model_training_bp = Blueprint('model_training', __name__)


@model_training_bp.route('/', methods=['GET'])
def get_models():
    """
    获取所有模型版本列表

    Returns:
        {
            "success": True,
            "models": [
                {
                    "version": "v3.7",
                    "name": "三层Ensemble系统",
                    "status": "trained|not_trained",
                    "model_files": [...],
                    ...
                }
            ]
        }
    """
    try:
        models = []
        model_dirs = current_app.config['MODEL_DIRS']

        for version, model_dir in model_dirs.items():
            model_info = {
                'version': version,
                'name': _get_model_name(version),
                'status': 'trained' if model_dir.exists() and list(model_dir.glob('*.pkl')) else 'not_trained',
                'model_dir': str(model_dir),
                'model_files': []
            }

            # 列出模型文件
            if model_dir.exists():
                model_files = list(model_dir.glob('*.pkl'))
                for model_file in model_files:
                    stat = model_file.stat()
                    model_info['model_files'].append({
                        'name': model_file.name,
                        'size': stat.st_size,
                        'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })

            models.append(model_info)

        return jsonify({
            'success': True,
            'models': models
        })

    except Exception as e:
        logger.error(f'获取模型列表失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@model_training_bp.route('/<version>', methods=['GET'])
def get_model_detail(version: str):
    """
    获取指定版本模型详情

    Returns:
        {
            "success": True,
            "model": {...}
        }
    """
    try:
        model_dir = current_app.config['MODEL_DIRS'].get(version)

        if not model_dir:
            return jsonify({
                'success': False,
                'error': f'未知的模型版本: {version}'
            }), 404

        model_info = {
            'version': version,
            'name': _get_model_name(version),
            'description': _get_model_description(version),
            'status': 'trained' if model_dir.exists() and list(model_dir.glob('*.pkl')) else 'not_trained',
            'model_dir': str(model_dir),
            'model_files': []
        }

        # 列出模型文件
        if model_dir.exists():
            model_files = list(model_dir.glob('*.pkl'))
            for model_file in model_files:
                stat = model_file.stat()
                model_info['model_files'].append({
                    'name': model_file.name,
                    'size': stat.st_size,
                    'size_mb': round(stat.st_size / (1024 * 1024), 2),
                    'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

        return jsonify({
            'success': True,
            'model': model_info
        })

    except Exception as e:
        logger.error(f'获取模型详情失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@model_training_bp.route('/train', methods=['POST'])
def train_model():
    """
    启动模型训练任务

    Request Body:
        {
            "version": "v3.9",
            "start_date": "2020-01-01",  // 可选
            "end_date": "2025-11-24",    // 可选
            "months": 6,                 // 可选，自动模式使用
            "auto": true                 // 可选，自动计算日期
        }

    Returns:
        {
            "success": True,
            "task_id": "uuid"
        }
    """
    try:
        data = request.get_json() or {}
        version = data.get('version', 'v3.9')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        months = data.get('months', 6)
        auto_mode = data.get('auto', True)  # 默认使用自动模式

        # 在主线程中获取配置（线程池中无法访问current_app）
        train_script = str(current_app.config['TRAIN_SCRIPT'])
        python_exec = current_app.config['PYTHON_EXECUTABLE']
        base_dir = str(current_app.config['BASE_DIR'])
        task_timeout = current_app.config['TASK_TIMEOUT']

        # 提交训练任务
        task_id = task_manager.submit_task(
            task_type=TaskType.MODEL_TRAINING,
            func=_run_model_training,
            metadata={
                'version': version,
                'start_date': start_date,
                'end_date': end_date,
                'months': months,
                'auto': auto_mode,
                # 传入配置（线程中使用）
                '_train_script': train_script,
                '_python_exec': python_exec,
                '_base_dir': base_dir,
                '_task_timeout': task_timeout
            }
        )

        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': f'模型训练任务已启动（版本: {version}）'
        })

    except Exception as e:
        logger.error(f'启动模型训练失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@model_training_bp.route('/train/stream', methods=['GET'])
def train_stream():
    """
    SSE端点：实时推送训练进度

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


@model_training_bp.route('/history', methods=['GET'])
def get_training_history():
    """
    获取训练历史记录

    Query Parameters:
        version: 模型版本（可选）
        limit: 返回数量限制（默认50）

    Returns:
        {
            "success": True,
            "history": [...]
        }
    """
    try:
        version = request.args.get('version')
        limit = int(request.args.get('limit', 50))

        db_manager = DatabaseManager(
            current_app.config['STOCK_DB_PATH'],
            current_app.config['WEBAPP_DB_PATH']
        )

        history = db_manager.get_training_history(version, limit)

        return jsonify({
            'success': True,
            'history': history
        })

    except Exception as e:
        logger.error(f'获取训练历史失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@model_training_bp.route('/<version>/features', methods=['GET'])
def get_feature_importance(version: str):
    """
    获取模型特征重要性

    Returns:
        {
            "success": True,
            "features": [{"name": "...", "importance": ...}, ...]
        }
    """
    try:
        features = _load_feature_importance(version)

        return jsonify({
            'success': True,
            'version': version,
            'features': features
        })

    except Exception as e:
        logger.error(f'获取特征重要性失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@model_training_bp.route('/<version>/report', methods=['GET'])
def get_training_report(version: str):
    """
    获取训练报告

    Returns:
        {
            "success": True,
            "report": {...}
        }
    """
    try:
        report = _load_training_report(version)

        return jsonify({
            'success': True,
            'version': version,
            'report': report
        })

    except Exception as e:
        logger.error(f'获取训练报告失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@model_training_bp.route('/<version>/metrics', methods=['GET'])
def get_model_metrics(version: str):
    """
    获取模型性能指标

    Returns:
        {
            "success": True,
            "metrics": {...}
        }
    """
    try:
        metrics = _load_model_metrics(version)

        return jsonify({
            'success': True,
            'version': version,
            'metrics': metrics
        })

    except Exception as e:
        logger.error(f'获取模型指标失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@model_training_bp.route('/<version>/training_curves', methods=['GET'])
def get_training_curves(version: str):
    """
    获取模型训练曲线数据 (loss曲线、学习曲线等)

    Returns:
        {
            "success": True,
            "version": "v3.9",
            "training_curves": {
                "models": {
                    "lightgbm": {
                        "train_losses": [...],
                        "val_losses": [...],
                        "iterations": [...]
                    },
                    ...
                },
                "meta_model": {...},
                "summary": {...}
            }
        }
    """
    try:
        curves = _load_training_curves(version)

        return jsonify({
            'success': True,
            'version': version,
            'training_curves': curves
        })

    except Exception as e:
        logger.error(f'获取训练曲线失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@model_training_bp.route('/<version>/training_history', methods=['GET'])
def get_training_history_list(version: str):
    """
    获取指定版本的所有训练历史记录

    Returns:
        {
            "success": True,
            "version": "v3.9",
            "histories": [...]
        }
    """
    try:
        histories = _list_training_histories(version)

        return jsonify({
            'success': True,
            'version': version,
            'histories': histories
        })

    except Exception as e:
        logger.error(f'获取训练历史列表失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@model_training_bp.route('/summary', methods=['GET'])
def get_models_summary():
    """
    获取所有模型的汇总信息

    Returns:
        {
            "success": True,
            "summary": {
                "total_models": 4,
                "trained_models": 3,
                "total_size_mb": 150.5,
                "latest_training": "2025-11-24",
                "versions": [...]
            }
        }
    """
    try:
        model_dirs = current_app.config['MODEL_DIRS']
        models_dir = current_app.config['MODELS_DIR']

        summary = {
            'total_models': len(model_dirs),
            'trained_models': 0,
            'total_size_mb': 0,
            'latest_training': None,
            'versions': []
        }

        latest_time = None

        for version, model_dir in model_dirs.items():
            version_info = {
                'version': version,
                'name': _get_model_name(version),
                'status': 'not_trained',
                'files_count': 0,
                'size_mb': 0,
                'latest_modified': None
            }

            if model_dir.exists():
                pkl_files = list(model_dir.glob('*.pkl'))
                if pkl_files:
                    version_info['status'] = 'trained'
                    version_info['files_count'] = len(pkl_files)
                    summary['trained_models'] += 1

                    for f in pkl_files:
                        stat = f.stat()
                        version_info['size_mb'] += stat.st_size / (1024 * 1024)

                        if latest_time is None or stat.st_mtime > latest_time:
                            latest_time = stat.st_mtime

                    version_info['size_mb'] = round(version_info['size_mb'], 2)
                    summary['total_size_mb'] += version_info['size_mb']

                    # 获取最新修改时间
                    newest = max(pkl_files, key=lambda f: f.stat().st_mtime)
                    version_info['latest_modified'] = datetime.fromtimestamp(
                        newest.stat().st_mtime
                    ).strftime('%Y-%m-%d %H:%M')

            summary['versions'].append(version_info)

        # 检查根目录下的模型文件
        if models_dir.exists():
            for f in models_dir.glob('*.pkl'):
                stat = f.stat()
                summary['total_size_mb'] += stat.st_size / (1024 * 1024)
                if latest_time is None or stat.st_mtime > latest_time:
                    latest_time = stat.st_mtime

        summary['total_size_mb'] = round(summary['total_size_mb'], 2)
        if latest_time:
            summary['latest_training'] = datetime.fromtimestamp(latest_time).strftime('%Y-%m-%d')

        return jsonify({
            'success': True,
            'summary': summary
        })

    except Exception as e:
        logger.error(f'获取模型汇总失败: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== 辅助函数 ====================

def _get_model_name(version: str) -> str:
    """获取模型名称"""
    # 已知版本的特定名称
    names = {
        'v3.6': 'V3.6 基础评分系统',
        'v3.7': 'V3.7 三层Ensemble系统',
        'v3.8': 'V3.8 增量学习系统',
        'v3.81': 'V3.81 质量元学习器',
        'v3.9': 'V3.9 增强特征系统',
        'v3.91': 'V3.91 优化特征系统',
        'v3.92': 'V3.92 高级特征系统',
        'v3.93': 'V3.93 实验特征系统',
        'v3.94': 'V3.94 优化Ensemble系统',
        'v3.95': 'V3.95 多目标滚动预测',
    }
    # 对于未知版本，生成默认名称
    if version in names:
        return names[version]
    # 生成动态名称：v3.91 -> V3.91 模型
    return f'{version.upper()} 模型'


def _get_model_description(version: str) -> str:
    """获取模型描述"""
    descriptions = {
        'v3.6': '基础量化评分系统',
        'v3.7': '三层Ensemble架构，49个特征，5个基础模型（LightGBM, XGBoost, CatBoost, RandomForest, MLP）',
        'v3.8': '增量学习引擎，实时特征计算，自适应评分标准化，模型漂移检测',
        'v3.81': 'V3.8基础 + Level 4质量元学习器，解决质量评分聚集问题',
        'v3.9': '42个增强特征，包含17个扩展财务指标',
        'v3.91': '基于V3.9优化的特征组合',
        'v3.92': '高级特征工程与模型优化',
        'v3.93': '实验性特征与算法测试',
        'v3.94': '优化Ensemble架构：移除MLP，加权Ensemble替代Ridge回归，71个特征，5个基础模型',
        'v3.95': '多目标预测(3d/5d/10d)，滚动训练，77个特征(65基础+12市场状态)，5个基础模型，IC=+0.16',
    }
    return descriptions.get(version, f'{version} 机器学习评分模型')


def _load_feature_importance(version: str) -> List[Dict[str, Any]]:
    """加载特征重要性数据"""
    features = []
    models_dir = current_app.config['MODELS_DIR']

    # 尝试从CSV文件加载 (v3.6格式)
    version_dir = _get_version_dir(version)
    csv_file = version_dir / 'feature_importance_target_1d.csv' if version_dir else None

    if csv_file and csv_file.exists():
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                features.append({
                    'name': row.get('feature', ''),
                    'lgb_importance': float(row.get('lgb_importance', 0)),
                    'xgb_importance': float(row.get('xgb_importance', 0)),
                    'avg_importance': float(row.get('avg_importance', 0))
                })
        # 按平均重要性排序
        features.sort(key=lambda x: x['avg_importance'], reverse=True)
        return features

    # 尝试从JSON元数据加载 (level4格式)
    if version in ['v3.81']:
        metadata_file = models_dir / 'level4_model_metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            importance = metadata.get('feature_importance', {})
            for name, value in importance.items():
                features.append({
                    'name': name,
                    'importance': value
                })
            features.sort(key=lambda x: x['importance'], reverse=True)
            return features

    # 尝试从训练报告获取特征列表 (v3.9格式)
    if version in ['v3.9']:
        version_dir = models_dir / 'v39'
        if version_dir.exists():
            report_files = list(version_dir.glob('training_report_*.md'))
            if report_files:
                report = _parse_training_report_md(max(report_files, key=lambda f: f.stat().st_mtime))
                feature_list = report.get('features', [])
                for i, name in enumerate(feature_list):
                    features.append({
                        'name': name,
                        'importance': len(feature_list) - i  # 假设按重要性排序
                    })
                return features

    return features


def _load_training_report(version: str) -> Dict[str, Any]:
    """加载训练报告"""
    models_dir = current_app.config['MODELS_DIR']

    # v3.9训练报告
    if version == 'v3.9':
        version_dir = models_dir / 'v39'
        if version_dir.exists():
            report_files = list(version_dir.glob('training_report_*.md'))
            if report_files:
                return _parse_training_report_md(max(report_files, key=lambda f: f.stat().st_mtime))

    # Level4元数据 (v3.81)
    if version == 'v3.81':
        metadata_file = models_dir / 'level4_model_metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            return {
                'training_history': metadata.get('training_history', {}),
                'best_params': metadata.get('best_params'),
                'random_state': metadata.get('random_state')
            }

    # 通用模型文件信息
    version_dir = _get_version_dir(version)
    if version_dir and version_dir.exists():
        pkl_files = list(version_dir.glob('*.pkl'))
        if pkl_files:
            newest = max(pkl_files, key=lambda f: f.stat().st_mtime)
            return {
                'latest_model': newest.name,
                'modified_time': datetime.fromtimestamp(newest.stat().st_mtime).isoformat(),
                'model_count': len(pkl_files)
            }

    return {}


def _parse_training_report_md(filepath: Path) -> Dict[str, Any]:
    """解析Markdown格式训练报告"""
    report = {
        'filepath': str(filepath),
        'features': [],
        'params': {}
    }

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取训练时间
        time_match = re.search(r'\*\*训练时间\*\*:\s*(.+)', content)
        if time_match:
            report['training_time'] = time_match.group(1).strip()

        # 提取训练参数
        param_patterns = {
            'time_range': r'时间范围:\s*(.+)',
            'lookback_days': r'回望天数:\s*(\d+)',
            'lookahead_days': r'前瞻天数:\s*(\d+)',
            'training_samples': r'训练样本数:\s*([\d,]+)',
            'feature_count': r'特征数量:\s*(\d+)',
            'model_version': r'模型版本:\s*(.+)'
        }

        for key, pattern in param_patterns.items():
            match = re.search(pattern, content)
            if match:
                value = match.group(1).strip().replace(',', '')
                report['params'][key] = int(value) if value.isdigit() else value

        # 提取特征列表
        feature_section = re.search(r'## 特征列表\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
        if feature_section:
            features = re.findall(r'\d+\.\s*(\w+)', feature_section.group(1))
            report['features'] = features

        # 提取模型文件路径
        path_match = re.search(r'保存路径:\s*(.+\.pkl)', content)
        if path_match:
            report['model_path'] = path_match.group(1).strip()

    except Exception as e:
        logger.error(f'解析训练报告失败: {e}')

    return report


def _load_model_metrics(version: str) -> Dict[str, Any]:
    """加载模型性能指标"""
    metrics = {
        'rmse': None,
        'mae': None,
        'r2': None,
        'best_iteration': None,
        'train_score': None,
        'val_score': None
    }

    models_dir = current_app.config['MODELS_DIR']

    # Level4元数据 (v3.81)
    if version == 'v3.81':
        metadata_file = models_dir / 'level4_model_metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            history = metadata.get('training_history', {})
            best_score = history.get('best_score', {})

            metrics['best_iteration'] = history.get('best_iteration')
            metrics['n_features'] = history.get('n_features')
            metrics['train_score'] = best_score.get('train', {}).get('rmse')
            metrics['val_score'] = best_score.get('val', {}).get('rmse')

            if metrics['train_score']:
                metrics['rmse'] = metrics['val_score']

    # v3.9从训练报告获取
    elif version == 'v3.9':
        version_dir = models_dir / 'v39'
        if version_dir.exists():
            report_files = list(version_dir.glob('training_report_*.md'))
            if report_files:
                report = _parse_training_report_md(max(report_files, key=lambda f: f.stat().st_mtime))
                params = report.get('params', {})
                metrics['training_samples'] = params.get('training_samples')
                metrics['feature_count'] = params.get('feature_count')
                metrics['training_time'] = report.get('training_time')

    return metrics


def _get_version_dir(version: str) -> Optional[Path]:
    """获取版本对应的目录"""
    model_dirs = current_app.config.get('MODEL_DIRS', {})
    return model_dirs.get(version)


def _load_training_curves(version: str) -> Dict[str, Any]:
    """
    加载训练曲线数据

    从 training_history_latest.json 文件加载训练过程中的loss曲线数据
    """
    models_dir = current_app.config['MODELS_DIR']
    version_clean = version.replace('.', '').replace('v', 'v')

    # 尝试从 training_history_latest.json 加载
    version_dir = _get_version_dir(version)
    if version_dir and version_dir.exists():
        history_file = version_dir / 'training_history_latest.json'
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                return {
                    'models': history.get('models', {}),
                    'meta_model': history.get('meta_model'),
                    'summary': history.get('summary', {}),
                    'start_time': history.get('start_time'),
                    'end_time': history.get('end_time'),
                    'duration_seconds': history.get('duration_seconds'),
                    'status': history.get('status')
                }
            except Exception as e:
                logger.error(f'加载训练历史失败: {e}')

    # 尝试从level4_model_metadata.json获取 (v3.81)
    if version == 'v3.81':
        metadata_file = models_dir / 'level4_model_metadata.json'
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                # 模拟训练曲线数据
                history = metadata.get('training_history', {})
                best_score = history.get('best_score', {})
                return {
                    'models': {
                        'level4_meta': {
                            'metric_name': 'rmse',
                            'best_iteration': history.get('best_iteration'),
                            'final_train_loss': best_score.get('train', {}).get('rmse'),
                            'final_val_loss': best_score.get('val', {}).get('rmse'),
                            'n_features': history.get('n_features')
                        }
                    },
                    'summary': {
                        'feature_count': history.get('n_features'),
                        'final_metrics': {
                            'train_rmse': best_score.get('train', {}).get('rmse'),
                            'val_rmse': best_score.get('val', {}).get('rmse')
                        }
                    }
                }
            except Exception as e:
                logger.error(f'加载level4元数据失败: {e}')

    return {'models': {}, 'meta_model': None, 'summary': {}}


def _list_training_histories(version: str) -> List[Dict[str, Any]]:
    """
    列出指定版本的所有训练历史记录

    返回按时间倒序排列的训练历史列表
    """
    histories = []
    version_dir = _get_version_dir(version)

    if version_dir and version_dir.exists():
        # 查找所有训练历史文件
        for filepath in sorted(version_dir.glob('training_history_*.json'), reverse=True):
            if 'latest' not in filepath.name:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        history = json.load(f)
                    # 提取摘要信息
                    summary = {
                        'filename': filepath.name,
                        'version': history.get('version'),
                        'status': history.get('status'),
                        'start_time': history.get('start_time'),
                        'end_time': history.get('end_time'),
                        'duration_seconds': history.get('duration_seconds'),
                        'models_count': len(history.get('models', {})),
                        'summary': history.get('summary', {})
                    }
                    histories.append(summary)
                except Exception as e:
                    logger.error(f'加载 {filepath} 失败: {e}')

    return histories


def _run_model_training(progress_callback, **params):
    """
    执行模型训练任务

    Args:
        progress_callback: 进度回调函数
        **params: 训练参数（包含配置信息）
    """
    try:
        progress_callback(0, '开始模型训练...')

        # 从参数获取配置（在主线程中已提前获取）
        script_path = params.get('_train_script')
        python_exec = params.get('_python_exec', 'python3')
        base_dir = params.get('_base_dir')
        task_timeout = params.get('_task_timeout', 3600)

        # 构建命令行参数
        cmd = [python_exec, script_path]

        # 如果使用自动模式
        if params.get('auto', True):
            cmd.append('--auto')
            if params.get('months'):
                cmd.extend(['--months', str(params.get('months', 6))])
        else:
            # 手动指定日期范围
            if params.get('start_date'):
                cmd.extend(['--start-date', params.get('start_date')])
            if params.get('end_date'):
                cmd.extend(['--end-date', params.get('end_date')])

        version = params.get('version', 'v3.9')
        progress_callback(10, f"执行训练脚本: {version}...")

        # 执行脚本
        result = subprocess.run(
            cmd,
            cwd=base_dir,
            capture_output=True,
            text=True,
            timeout=task_timeout
        )

        if result.returncode == 0:
            progress_callback(100, '模型训练完成')
            return {'success': True, 'output': result.stdout}
        else:
            error_msg = result.stderr[:500] if result.stderr else '未知错误'
            raise Exception(f'模型训练失败: {error_msg}')

    except subprocess.TimeoutExpired:
        raise Exception('模型训练超时（超过1小时）')
    except Exception as e:
        logger.error(f'模型训练任务失败: {e}')
        raise
