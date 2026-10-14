"""Harness：项目装配根（composition root）。

把模型、检索、防逆向守卫等「进程级单例」组装好，并为每条连接按需
创建会话态对象（``SessionState`` / ``Supervisor`` / ``SessionLoop``）。
server 层只依赖 Harness，不直接 new 具体实现，便于替换与测试。
"""
from __future__ import annotations

from typing import Any, Callable

from agents.supervisor import Supervisor
from agents.lecturer.agent import LecturerAgent
from agents.student.agent import StudentAgent
from agents.teacher.agent import TeacherAgent
from config.logging import get_logger
from config.settings import settings
from core.anti_rev import AntiRevGuard
from core.checkpoint import Checkpointer
from core.loop import SessionLoop
from core.math_solver import warn_if_unavailable
from core.router import IntentRouter
from core.state.session import SessionState
from core.state.store import session_store
from db import runtime_sync

log = get_logger("core.harness")

EmitFn = Callable[[dict[str, Any]], None]


def _build_retriever() -> Any | None:
    """尝试构建 RAG 检索器；依赖缺失时返回 None（答疑退化为纯 LLM）。"""
    try:
        from rag.retriever import build_retriever  # noqa: PLC0415

        return build_retriever()
    except Exception as err:  # noqa: BLE001 - 可选组件
        log.warning("retriever unavailable ({}), run without RAG", err)
        return None


class EducationHarness:
    """全局装配与工厂。"""

    def __init__(self, retriever: Any | None = None) -> None:
        self.retriever = retriever if retriever is not None else _build_retriever()
        self.checkpointer = Checkpointer()
        self.anti_rev = AntiRevGuard(settings.anti_rev_secret, settings.rate_limit_per_min)
        # 关键能力的「降级告警」放在启动时喊：sympy 缺失不会崩，只会静默失去
        # 方程验算（AI 老师讲错答案不再被纠正）。曾因依赖漏声明而在云端长期失效，
        # 因此这里必须让它在启动日志里刺出来，而不是等到讲课出错才发现。
        math_verify = warn_if_unavailable()
        # rag 打「后端名而不是 True/False」：排查时一眼看出当前走向量检索还是词法兜底
        rag_info = "off" if self.retriever is None else getattr(self.retriever, "backend_name", "on")
        log.info(
            "harness ready: env={} rag={} math_verify={}",
            settings.env,
            rag_info,
            math_verify,
        )

    # ---------------------------------------------------------------- 会话
    def create_session(
        self,
        student_id: str,
        *,
        session_id: str | None = None,
        emit: EmitFn | None = None,
        with_loop: bool = True,
    ) -> tuple[SessionState, Supervisor, SessionLoop | None]:
        """创建一条会话及其 Supervisor（可选实时循环）。"""
        state = session_store.get(session_id) if session_id else None
        if state is not None and state.student_id != student_id:
            state = None
            session_id = None
        if state is None:
            state = session_store.create(student_id, session_id)
        state.emit = emit
        supervisor = Supervisor(
            state,
            student=StudentAgent(),
            teacher=TeacherAgent(),          # 答疑链路
            lecturer=LecturerAgent(),        # 讲课链路（课件逐页讲解 + 页面标注）
            router=IntentRouter(),
            retriever=self.retriever,
        )
        loop = SessionLoop(state, supervisor) if with_loop else None
        # 限流校验已上移到「连接入口」（server/websocket 的握手处）统一判定，
        # 这里只装配会话，避免同一次连接被重复计数（旧代码此处调用后丢弃返回值，形同虚设）。
        return state, supervisor, loop

    def checkpoint(self, session: SessionState) -> None:
        """保存会话检查点（断线续聊），并镜像写入数据库（Navicat 实时可见）。"""
        try:
            self.checkpointer.save(session)
        except Exception as err:  # noqa: BLE001
            log.warning("checkpoint failed: {}", err)
        # 运行时写库：与 JSON 落盘同一个钩子（本机 MySQL 场景默认启用，见
        # settings.runtime_db_write）；内部全包异常，失败仅告警一次、不影响答疑。
        runtime_sync.maybe_sync(session)
