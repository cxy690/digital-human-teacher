"""AI 讲课 Agent（``LecturerAgent``）负责的流程节点。

链路「总控 → 讲课 Agent → 白板」只有一个入口：**讲某一页课件**。它的节点与顺序：

| 图 | 节点（按连接顺序） | 说明 |
|---|---|---|
| ``lecturer.slide`` | ``request`` → ``explain`` → ``fallback`` → ``degrade`` | 登记请求 → 讲课链路讲这一页 → 失败回退答疑链路的等价实现 → 两条都失败返回兜底话术 |

节点即 ``docs/supervisor_orchestration.md`` §6「失败与回退」的三级降级链，
每一级都在 :class:`~agents.protocol.AgentBus` 留下可追溯记录
（``request`` / ``fallback`` / ``error``），并把「谁在第几页请求了谁」写进会话追踪缓冲。

设计边界（与 ``TeacherAgent`` 的分工）：

* 本图的 ``explain`` 节点只调用 :meth:`LecturerAgent.explain_slide`，产出**话术 + 页面标注**；
* 换页、白板下发、配速与播报都在总控（见 ``agents/supervisor/nodes.py`` 的 ``supervisor.slide`` 图），
  讲课 Agent 自身不改会话状态。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agents.protocol import (
    KIND_ERROR,
    KIND_FALLBACK,
    KIND_REQUEST,
    ROLE_LECTURER,
    ROLE_SUPERVISOR,
    ROLE_TEACHER,
    AgentBus,
)
from config.logging import get_logger
from core.graph import NodeSpec, TaskGraph
from core.state.models import AIResponse

if TYPE_CHECKING:  # 仅类型标注：handler 挂在总控实例上，运行期无需导入
    from agents.supervisor import Supervisor

log = get_logger("agents.lecturer.nodes")

#: 单页讲解图名
SLIDE_GRAPH = "lecturer.slide"

#: 兜底话术（两条链路都失败时使用；不因为一页异常打断整节课）
DEGRADED_SPEECH = "这一页的内容我们待会儿再看，先接着往下讲。"


SLIDE_SPECS: tuple[NodeSpec, ...] = (
    NodeSpec(
        name="request",
        summary="向协作总线登记「总控请求讲课，第 N 页」",
        impl="agents.protocol.AgentBus.record(KIND_REQUEST)",
        inputs=("deck", "index"),
        outputs=(),
        deps=(),
        requires=("agents.protocol.AgentBus",),
    ),
    NodeSpec(
        name="explain",
        summary="讲课链路：生成本页话术 + 页面圈画/高亮标注（LLM 失败内部降级为要点口播）",
        impl="LecturerAgent.explain_slide",
        inputs=("deck", "index"),
        outputs=("response",),
        deps=("request",),
        requires=("models.llm.get_llm", "core.courseware.loader（课件页元素）"),
    ),
    NodeSpec(
        name="fallback",
        summary="讲课链路失败时回退 TeacherAgent.explain_slide（等价实现），并记 fallback",
        impl="TeacherAgent.explain_slide",
        inputs=("response", "deck", "index"),
        outputs=("response",),
        deps=("explain",),
        requires=("TeacherAgent.explain_slide",),
    ),
    NodeSpec(
        name="degrade",
        summary="两条链路都失败：记 error 并返回兜底话术（绝不哑巴翻页）",
        impl="AIResponse 兜底话术",
        inputs=("response",),
        outputs=("response",),
        deps=("fallback",),
    ),
)


def build_slide_graph(supervisor: "Supervisor") -> TaskGraph:
    """构建「讲某一页课件」的节点图（由 ``Supervisor._explain_slide`` 调用）。"""

    def _page(ctx: dict[str, Any]) -> dict[str, Any]:
        deck = ctx.get("deck")
        return {
            "page": int(ctx.get("index") or 0),
            "deck_id": str(getattr(deck, "deck_id", "") or ""),
        }

    def _page_no(ctx: dict[str, Any]) -> int:
        return int(ctx.get("index") or 0)

    def request(ctx: dict[str, Any]) -> dict[str, Any]:
        AgentBus.record(
            supervisor.session,
            sender=ROLE_SUPERVISOR,
            recipient=ROLE_LECTURER,
            kind=KIND_REQUEST,
            detail=f"讲解课件第 {_page_no(ctx) + 1} 页",
            **_page(ctx),
        )
        return {}

    def _budget(ctx: dict[str, Any]) -> dict[str, Any]:
        """单页时长预算（秒）→ 调用 kwargs。

        ``0`` 表示「不限额」，此时**不传这个参数**：沿用实现自己的默认值，
        也让只暴露基础签名（``session, deck, index``）的实现照样能被调用。
        连讲（``_play_slides``）会逐页算出预算，前端单页预览则不带。
        """
        budget = float(ctx.get("budget_sec") or 0.0)
        return {"budget_sec": budget} if budget > 0 else {}

    async def explain(ctx: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await supervisor.lecturer.explain_slide(
                supervisor.session, ctx["deck"], _page_no(ctx), **_budget(ctx)
            )
            return {"response": response}
        except Exception as err:  # noqa: BLE001 - 讲课失败不应中断整节课
            log.warning("讲课链路失败，回退 teacher.explain_slide：{}", err)
            return {"response": None, "error": str(err)}

    async def fallback(ctx: dict[str, Any]) -> dict[str, Any]:
        if ctx.get("response") is not None:
            return {"__skip__": True}
        AgentBus.record(
            supervisor.session,
            sender=ROLE_SUPERVISOR,
            recipient=ROLE_TEACHER,
            kind=KIND_FALLBACK,
            ok=False,
            detail=f"讲课失败，回退答疑链路的等价实现：{ctx.get('error') or ''}",
            **_page(ctx),
        )
        try:
            response = await supervisor.teacher.explain_slide(
                supervisor.session, ctx["deck"], _page_no(ctx)
            )
            return {"response": response}
        except Exception as err:  # noqa: BLE001 - 回退也失败：交给 degrade 兜底
            log.warning("课件讲解两条链路均失败：{}", err)
            return {"response": None, "error": str(err)}

    def degrade(ctx: dict[str, Any]) -> dict[str, Any]:
        if ctx.get("response") is not None:
            return {"__skip__": True}
        AgentBus.record(
            supervisor.session,
            sender=ROLE_SUPERVISOR,
            recipient=ROLE_LECTURER,
            kind=KIND_ERROR,
            ok=False,
            detail=f"课件讲解失败：{ctx.get('error') or ''}",
            **_page(ctx),
        )
        return {
            "response": AIResponse(
                speech=DEGRADED_SPEECH,
                meta={"slide": {"index": _page_no(ctx)}, "mood": "speaking", "degraded": True},
            )
        }

    return TaskGraph.from_specs(
        SLIDE_GRAPH,
        SLIDE_SPECS,
        {"request": request, "explain": explain, "fallback": fallback, "degrade": degrade},
        session=getattr(supervisor, "session", None),   # 活动日志按会话归组（见 core/activity_log.py）
    )


#: 本 agent 的图名 → 节点声明（供 ``agents.nodes`` 注册表汇总）
GRAPH_SPECS: dict[str, tuple[NodeSpec, ...]] = {SLIDE_GRAPH: SLIDE_SPECS}
