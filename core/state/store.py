"""会话仓库：内存态管理多个 SessionState（单机可直接用，分布式可换 Redis）。"""
from __future__ import annotations

import threading
import uuid

from core.state.session import SessionState


class SessionStore:
    """线程安全的会话表。"""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def create(self, student_id: str, session_id: str | None = None) -> SessionState:
        sid = session_id or uuid.uuid4().hex[:12]
        with self._lock:
            state = SessionState(session_id=sid, student_id=student_id)
            self._sessions[sid] = state
        return state

    def get(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str, student_id: str = "anonymous") -> SessionState:
        state = self.get(session_id)
        if state is None:
            state = self.create(student_id, session_id)
        return state

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def list_ids(self) -> list[str]:
        return list(self._sessions.keys())


# 进程内共享单例（与 server 同生命周期）
session_store = SessionStore()
