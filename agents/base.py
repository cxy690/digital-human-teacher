"""Agent 基类：统一智能体的生命周期与日志。"""
from __future__ import annotations

from abc import ABC

from config.logging import get_logger
from core.state.session import SessionState


class BaseAgent(ABC):
    """所有智能体的基类（总控 / 学生端代理 / AI 教师 / AI 讲课）。

    四个角色都在本包内：总调度 ``agents/supervisor/``、学生端 ``agents/student/``、
    教师 ``agents/teacher/``、讲课 ``agents/lecturer/``。

    总控虽然依赖一批 ``core.*`` 能力模块，但那批模块对它的引用只在
    ``TYPE_CHECKING`` 下（``core/turn_graph.py`` / ``core/langgraph_flow.py``），
    运行期没有反向导入，因此不存在循环依赖。
    """

    role: str = "agent"

    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.role
        self.log = get_logger(f"agents.{self.name}")

    async def on_session_start(self, session: SessionState) -> None:  # noqa: D401 - 钩子
        """会话建立钩子（默认空实现）。"""

    async def on_session_end(self, session: SessionState) -> None:
        """会话结束钩子（默认空实现）。"""
