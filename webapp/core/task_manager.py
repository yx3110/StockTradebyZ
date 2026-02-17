"""
任务管理器 - 统一管理所有后台任务
"""
import uuid
import threading
import time
import queue
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any
from enum import Enum


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


class TaskType(Enum):
    """任务类型枚举"""
    DATA_UPDATE = 'data_update'
    STOCK_SELECTION = 'stock_selection'
    AI_ANALYSIS = 'ai_analysis'
    MODEL_TRAINING = 'model_training'
    BACKTEST = 'backtest'


@dataclass
class TaskInfo:
    """任务信息"""
    id: str
    type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0  # 0-100
    message: str = ''
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        """转换为字典"""
        return {
            'id': self.id,
            'type': self.type.value,
            'status': self.status.value,
            'progress': self.progress,
            'message': self.message,
            'result': self.result,
            'error': self.error,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'metadata': self.metadata,
            'duration': self._calculate_duration(),
        }

    def _calculate_duration(self):
        """计算任务持续时间（秒）"""
        if self.started_at is None:
            return 0
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()


class TaskManager:
    """
    任务管理器 - 管理所有后台任务的执行、进度追踪和状态管理

    特性:
    - 线程池执行长时间任务
    - 实时进度追踪
    - 任务状态管理
    - 支持任务取消
    - 任务历史记录
    """

    def __init__(self, max_workers: int = 4):
        """
        初始化任务管理器

        Args:
            max_workers: 最大并发任务数
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tasks: Dict[str, TaskInfo] = {}
        self.futures: Dict[str, Future] = {}
        self.lock = threading.Lock()

        # 任务进度队列（用于实时进度推送）
        self.progress_queues: Dict[str, queue.Queue] = {}

    def submit_task(
        self,
        task_type: TaskType,
        func: Callable,
        *args,
        metadata: Optional[Dict] = None,
        **kwargs
    ) -> str:
        """
        提交新任务

        Args:
            task_type: 任务类型
            func: 要执行的函数
            *args: 位置参数
            metadata: 任务元数据
            **kwargs: 关键字参数

        Returns:
            task_id: 任务ID
        """
        task_id = str(uuid.uuid4())

        # 创建任务信息
        task_info = TaskInfo(
            id=task_id,
            type=task_type,
            status=TaskStatus.PENDING,
            metadata=metadata or {}
        )

        with self.lock:
            self.tasks[task_id] = task_info
            self.progress_queues[task_id] = queue.Queue()

        # 创建进度回调函数
        def progress_callback(progress: float, message: str = ''):
            """进度回调"""
            self.update_progress(task_id, progress, message)

        # 包装函数，添加进度追踪
        # 将metadata作为kwargs传递给函数
        combined_kwargs = {**(metadata or {}), **kwargs}

        def wrapped_func():
            return self._run_task(task_id, func, progress_callback, *args, **combined_kwargs)

        # 提交到线程池
        future = self.executor.submit(wrapped_func)
        with self.lock:
            self.futures[task_id] = future

        return task_id

    def _run_task(
        self,
        task_id: str,
        func: Callable,
        progress_callback: Callable,
        *args,
        **kwargs
    ):
        """
        运行任务（内部方法）

        Args:
            task_id: 任务ID
            func: 要执行的函数
            progress_callback: 进度回调函数
            *args: 位置参数
            **kwargs: 关键字参数
        """
        try:
            # 更新任务状态为运行中
            with self.lock:
                if task_id in self.tasks:
                    self.tasks[task_id].status = TaskStatus.RUNNING
                    self.tasks[task_id].started_at = datetime.now()

            # 执行任务（传入进度回调）
            result = func(progress_callback, *args, **kwargs)

            # 更新任务状态为完成
            with self.lock:
                if task_id in self.tasks:
                    self.tasks[task_id].status = TaskStatus.COMPLETED
                    self.tasks[task_id].progress = 100.0
                    self.tasks[task_id].result = result
                    self.tasks[task_id].completed_at = datetime.now()
                    self.tasks[task_id].message = '任务完成'

            return result

        except Exception as e:
            # 更新任务状态为失败
            with self.lock:
                if task_id in self.tasks:
                    self.tasks[task_id].status = TaskStatus.FAILED
                    self.tasks[task_id].error = str(e)
                    self.tasks[task_id].completed_at = datetime.now()
                    self.tasks[task_id].message = f'任务失败: {str(e)}'
            raise

    def update_progress(self, task_id: str, progress: float, message: str = ''):
        """
        更新任务进度

        Args:
            task_id: 任务ID
            progress: 进度（0-100）
            message: 进度消息
        """
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].progress = min(100.0, max(0.0, progress))
                self.tasks[task_id].message = message

                # 推送进度到队列
                if task_id in self.progress_queues:
                    try:
                        self.progress_queues[task_id].put_nowait({
                            'progress': self.tasks[task_id].progress,
                            'message': message,
                            'status': self.tasks[task_id].status.value,
                            'timestamp': datetime.now().isoformat()
                        })
                    except queue.Full:
                        pass  # 队列满了，跳过

    def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        """
        获取任务状态

        Args:
            task_id: 任务ID

        Returns:
            任务信息，如果不存在返回None
        """
        with self.lock:
            return self.tasks.get(task_id)

    def get_task_progress_stream(self, task_id: str):
        """
        获取任务进度流（用于SSE）

        Args:
            task_id: 任务ID

        Yields:
            进度信息字典
        """
        if task_id not in self.progress_queues:
            return

        progress_queue = self.progress_queues[task_id]

        while True:
            try:
                # 等待进度更新（超时1秒）
                progress_data = progress_queue.get(timeout=1.0)
                yield progress_data

                # 如果任务已完成或失败，结束流
                task_info = self.get_task_status(task_id)
                if task_info and task_info.status in [
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED
                ]:
                    break

            except queue.Empty:
                # 队列为空，检查任务状态
                task_info = self.get_task_status(task_id)
                if task_info and task_info.status in [
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED
                ]:
                    # 发送最终状态
                    yield {
                        'progress': task_info.progress,
                        'message': task_info.message,
                        'status': task_info.status.value,
                        'timestamp': datetime.now().isoformat()
                    }
                    break

                # 发送心跳
                yield {
                    'heartbeat': True,
                    'timestamp': datetime.now().isoformat()
                }

    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务

        Args:
            task_id: 任务ID

        Returns:
            是否取消成功
        """
        with self.lock:
            if task_id in self.futures:
                future = self.futures[task_id]
                cancelled = future.cancel()

                if cancelled and task_id in self.tasks:
                    self.tasks[task_id].status = TaskStatus.CANCELLED
                    self.tasks[task_id].completed_at = datetime.now()
                    self.tasks[task_id].message = '任务已取消'

                return cancelled

        return False

    def get_all_tasks(self, limit: int = 100) -> list:
        """
        获取所有任务列表

        Args:
            limit: 返回任务数量限制

        Returns:
            任务列表（按创建时间倒序）
        """
        with self.lock:
            tasks = sorted(
                self.tasks.values(),
                key=lambda t: t.created_at,
                reverse=True
            )
            return [task.to_dict() for task in tasks[:limit]]

    def get_tasks_by_type(self, task_type: TaskType, limit: int = 50) -> list:
        """
        获取指定类型的任务列表

        Args:
            task_type: 任务类型
            limit: 返回任务数量限制

        Returns:
            任务列表
        """
        with self.lock:
            tasks = [
                task for task in self.tasks.values()
                if task.type == task_type
            ]
            tasks = sorted(tasks, key=lambda t: t.created_at, reverse=True)
            return [task.to_dict() for task in tasks[:limit]]

    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """
        清理旧任务

        Args:
            max_age_hours: 最大保留时间（小时）
        """
        now = datetime.now()
        with self.lock:
            old_task_ids = [
                task_id for task_id, task in self.tasks.items()
                if task.completed_at and
                (now - task.completed_at).total_seconds() > max_age_hours * 3600
            ]

            for task_id in old_task_ids:
                del self.tasks[task_id]
                if task_id in self.futures:
                    del self.futures[task_id]
                if task_id in self.progress_queues:
                    del self.progress_queues[task_id]

    def shutdown(self, wait: bool = True):
        """
        关闭任务管理器

        Args:
            wait: 是否等待所有任务完成
        """
        self.executor.shutdown(wait=wait)


# 全局任务管理器实例
task_manager = TaskManager(max_workers=4)
