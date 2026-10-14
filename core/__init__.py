"""core 包：会话编排核心（路由、图执行、麦控、校验、弹性、装配）。"""
from core.graph import Node, NodeSpec, TaskGraph
from core.router import IntentRouter
from core.state.session import SessionState
from core.state.store import SessionStore, session_store
from core.validator import ResponseValidator

__all__ = [
    "Node",
    "NodeSpec",
    "TaskGraph",
    "IntentRouter",
    "SessionState",
    "SessionStore",
    "session_store",
    "Supervisor",
    "ResponseValidator",
]


def __getattr__(name: str):
    """惰性再导出 Supervisor（定义在 ``agents/supervisor``，即总调度 agent）。

    保留 ``from core import Supervisor`` 这条历史写法可用：本模块只做转发，
    真正的定义与依赖都在 ``agents`` 包内，惰性求值避免 core 包导入期被拖重。
    """
    if name == "Supervisor":
        from agents.supervisor import Supervisor

        return Supervisor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
