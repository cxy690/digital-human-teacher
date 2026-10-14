"""回合主图：把一次学生输入的编排（意图路由 → 答疑 / 闲聊 / 控制）图化到 LangGraph。

与 ``Supervisor._handle_turn_builtin`` 行为保持一致：节点直接复用 Supervisor 的同一批
方法，避免两套实现分叉。逐帧链路（麦控）不在此图内。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from config.constants import IntentType
from config.logging import get_logger
from core.state.models import AIResponse, ChatMessage, StudentTurn

if TYPE_CHECKING:
    from agents.supervisor import Supervisor

log = get_logger("core.turn_graph")

try:
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover - 取决于环境
    LANGGRAPH_AVAILABLE = False
    END = START = "__end__"
    StateGraph = None  # type: ignore[assignment]


class TurnState(TypedDict, total=False):
    """回合状态（运行时经闭包访问 Supervisor 与 SessionState）。"""

    turn: StudentTurn
    intent: IntentType
    response: AIResponse | None
    # 多意图执行计划（IntentSegment 列表）：route 节点算好，multi_intent 节点消费
    multi_plan: list[Any]


def build_turn_graph(supervisor: "Supervisor") -> Any:
    """构建并编译回合主图；依赖缺失时抛 ``RuntimeError``。"""
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError("langgraph is not installed")

    async def route(state: TurnState) -> dict[str, Any]:
        turn = state["turn"]
        # 0) 课件续讲/重讲指令最优先（与内建 route 同一判定，见 Supervisor.lecture_command）：
        #    「继续」「接着讲」→ 断点续讲，「再讲一遍」→ 重讲当前页。
        if supervisor.lecture_command(turn.text):  # noqa: SLF001
            turn.intent = IntentType.CONTROL
            supervisor.session.add_message(
                ChatMessage(role="student", text=turn.text, image_id=turn.image_id)
            )
            return {"intent": IntentType.CONTROL}
        # 正等待作答且输入像作答：不重新路由，直接标为作答路径
        if supervisor._should_treat_as_answer(turn):  # noqa: SLF001
            turn.intent = IntentType.TEXT_QUESTION
            supervisor.session.add_message(
                ChatMessage(role="student", text=turn.text, image_id=turn.image_id)
            )
            return {"intent": IntentType.TEXT_QUESTION}
        # 意图识别 / 分流：与 Supervisor._handle_turn_builtin 走同一个入口 Agent，
        # 复杂 / 多意图同样在此标记并随回合带下去（两套编排不分叉）。
        decision = await supervisor.router.analyze_async(turn)
        intent = decision.intent
        turn.intent = intent
        turn.meta["intent_decision"] = decision.to_meta()
        turn.meta["intent_notes"] = decision.delegation_notes()
        supervisor.session.add_message(
            ChatMessage(role="student", text=turn.text, image_id=turn.image_id)
        )
        # 多意图且含 2 个以上提问：走逐条作答（与内建实现同一分发逻辑）。
        # ⚠️ 计划必须放进**返回值**：LangGraph 只认节点 return 的更新，
        #    直接改 state 字典会被丢掉（route_after 里就取不到，多意图会退化成单次答疑）。
        plan = supervisor.router.plan(decision)
        if len([seg for seg in plan if seg.intent is not IntentType.CONTROL]) >= 2:
            return {"intent": intent, "multi_plan": plan}
        # 多意图里夹带的控制指令：简单意图，入口层当场执行，其余转交下游
        if decision.delegate and decision.control_clauses and intent is not IntentType.CONTROL:
            for clause in decision.control_clauses:
                supervisor._handle_control(clause)  # noqa: SLF001 - 复用同一实现
        return {"intent": intent}

    async def control(state: TurnState) -> dict[str, Any]:
        supervisor._handle_control(state["turn"].text)  # noqa: SLF001 - 复用同一实现
        return {"response": None}

    async def chitchat(state: TurnState) -> dict[str, Any]:
        return {"response": await supervisor._turn_chitchat()}  # noqa: SLF001

    async def teach(state: TurnState) -> dict[str, Any]:
        return {"response": await supervisor._turn_teach(state["turn"], state["intent"])}  # noqa: SLF001

    async def answer_check(state: TurnState) -> dict[str, Any]:
        return {"response": await supervisor._turn_answer_check(state["turn"])}  # noqa: SLF001

    async def resume_lecture(state: TurnState) -> dict[str, Any]:
        # 「继续」= 断点续讲；「再讲一遍」= 重讲当前页（模式由 lecture_command 内部再判）
        await supervisor.resume_lecture(state["turn"].text)  # noqa: SLF001
        return {"response": None}

    async def multi_intent(state: TurnState) -> dict[str, Any]:
        return {
            "response": await supervisor._turn_multi_intent(  # noqa: SLF001
                state["turn"], state.get("multi_plan") or []
            )
        }

    def route_after(state: TurnState) -> str:
        """优先课件续讲指令；再判题（待答追问且输入像作答 -> answer_check）；否则按意图分流。"""
        # 课件续讲/重讲指令最优先（与内建 route 同一判定函数，两套编排不分叉）
        if supervisor.lecture_command(state["turn"].text):  # noqa: SLF001
            return "resume_lecture"
        if supervisor._should_treat_as_answer(state["turn"]):  # noqa: SLF001
            return "answer_check"
        plan = state.get("multi_plan") or []
        if len([seg for seg in plan if seg.intent is not IntentType.CONTROL]) >= 2:
            return "multi_intent"
        intent = state["intent"]
        if intent is IntentType.CONTROL:
            return "control"
        if intent is IntentType.CHITCHAT:
            return "chitchat"
        return "teach"

    graph: Any = StateGraph(TurnState)
    graph.add_node("route", route)
    graph.add_node("control", control)
    graph.add_node("chitchat", chitchat)
    graph.add_node("teach", teach)
    graph.add_node("answer_check", answer_check)
    graph.add_node("multi_intent", multi_intent)
    graph.add_node("resume_lecture", resume_lecture)
    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        route_after,
        {
            "answer_check": "answer_check",
            "control": "control",
            "chitchat": "chitchat",
            "teach": "teach",
            "multi_intent": "multi_intent",
            "resume_lecture": "resume_lecture",
        },
    )
    graph.add_edge("answer_check", END)
    graph.add_edge("control", END)
    graph.add_edge("chitchat", END)
    graph.add_edge("teach", END)
    graph.add_edge("multi_intent", END)
    graph.add_edge("resume_lecture", END)
    return graph.compile()


async def run_turn_graph(supervisor: "Supervisor", turn: StudentTurn) -> AIResponse | None:
    """执行回合主图（编译结果按 Supervisor 缓存）；返回 AI 产物（控制回合为 None）。"""
    compiled = getattr(supervisor, "_turn_graph", None)
    if compiled is None:
        compiled = build_turn_graph(supervisor)
        supervisor._turn_graph = compiled
    final = await compiled.ainvoke({"turn": turn})
    return final.get("response")
