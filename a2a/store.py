"""A2A 任务表：内存态 + TTL 淘汰。

为什么要 TTL：A2A 的 Task 会被外部 client 用 ``tasks/get`` 反复查询，
但如果只增不删，长跑的服务进程会一直涨。这里给一个默认 1 小时的存活期
（够一次答疑闭环 + 客户端取产物），过期即回收；进程重启后清空属于可接受行为
（任务本身是一次问答，不是需要持久化的业务数据）。
"""
from __future__ import annotations

import threading
import time

from a2a.models import Task


class A2ATaskStore:
    """线程安全的任务表。"""

    def __init__(self, *, ttl_seconds: float = 3600.0, max_tasks: int = 2000) -> None:
        self.ttl_seconds = float(ttl_seconds)
        self.max_tasks = int(max_tasks)
        self._tasks: dict[str, Task] = {}
        self._created: dict[str, float] = {}
        self._lock = threading.Lock()

    def put(self, task: Task) -> Task:
        with self._lock:
            self._purge_locked()
            self._tasks[task.id] = task
            self._created[task.id] = time.time()
        return task

    def get(self, task_id: str | None) -> Task | None:
        if not task_id:
            return None
        with self._lock:
            self._purge_locked()
            return self._tasks.get(task_id)

    def remove(self, task_id: str) -> None:
        with self._lock:
            self._tasks.pop(task_id, None)
            self._created.pop(task_id, None)

    def list_ids(self) -> list[str]:
        with self._lock:
            self._purge_locked()
            return list(self._tasks)

    # ---------------------------------------------------------------- 内部
    def _purge_locked(self) -> None:
        """回收过期任务；若仍超上限，再按创建时间淘汰最旧的一批。"""
        now = time.time()
        expired = [
            tid for tid, born in self._created.items()
            if now - born > self.ttl_seconds
        ]
        for tid in expired:
            self._tasks.pop(tid, None)
            self._created.pop(tid, None)
        overflow = len(self._tasks) - self.max_tasks
        if overflow > 0:
            oldest = sorted(self._created, key=self._created.get)[:overflow]  # type: ignore[arg-type]
            for tid in oldest:
                self._tasks.pop(tid, None)
                self._created.pop(tid, None)
