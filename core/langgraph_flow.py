"""答疑子图的 LangGraph 实现（可选）。

把 Supervisor 的答疑链路 ``retrieve -> generate -> validate`` 迁移为 LangGraph
``StateGraph``，通过配置 ``ORCHESTRATOR=langgraph`` 启用。未安装 langgraph 时本模块
仅暴露 ``LANGGRAPH_AVAILABLE=False``，上层自动回退自研 ``TaskGraph``（core/graph.py）。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from config.constants import IntentType
from config.logging import get_logger
from core import math_solver
from core.hint_homework import HintLevel
from core.state.models import AIResponse, StudentTurn
from core.validator import ValidationResult, drop_invalid_board_ops

if TYPE_CHECKING:
    from agents.supervisor import Supervisor

log = get_logger("core.langgraph_flow")

try:
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover - 取决于环境
    LANGGRAPH_AVAILABLE = False
    END = START = "__end__"
    StateGraph = None  # type: ignore[assignment]


class AnswerState(TypedDict, total=False):
    """答疑子图状态（每个字段单写者，无需 reducer）。"""

    turn: StudentTurn
    intent: IntentType
    hint_level: HintLevel | None
    ocr_text: str
    notes: list[str]
    response: AIResponse
    validation: ValidationResult


def build_answer_graph(supervisor: "Supervisor") -> Any:
    """构建并编译答疑子图；依赖缺失时抛 ``RuntimeError``。"""
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError("langgraph is not installed")

    async def retrieve(state: AnswerState) -> dict[str, Any]:
        turn = state["turn"]
        # 保留入口意图 Agent 写入的分流提示（多意图子问题清单），再叠加 RAG 检索结果
        notes = list(state.get("notes") or [])
        if supervisor.retriever is None:
            return {"notes": notes}
        notes += await supervisor.retriever.retrieve(turn.text)
        return {"notes": notes}

    async def generate(state: AnswerState) -> dict[str, Any]:
        turn = state["turn"]
        hint_level = state.get("hint_level")
        if turn.intent is IntentType.IMAGE_QUESTION:
            response = await supervisor.teacher.answer_image(
                turn, supervisor.session, ocr_text=state.get("ocr_text", ""), hint_level=hint_level
            )
        else:
            response = await supervisor.teacher.answer(
                turn, supervisor.session, hint_level=hint_level, context_notes=state.get("notes")
            )
        return {"response": response}

    async def validate(state: AnswerState) -> dict[str, Any]:
        response: AIResponse = state["response"]
        result = supervisor.validator.validate(response, hint_level=state.get("hint_level"))
        return {"validation": result}

    async def degrade(state: AnswerState) -> dict[str, Any]:
        # 校验不通过：只丢掉有问题的白板指令（如坐标越界的那几行），保留其余。
        # 不要整块清空 —— 一处越界就把白板清成空白，学生看到的是「讲了半天板上一片白」。
        response: AIResponse = state["response"]
        result = state["validation"]
        if not result.ok:
            log.warning("response validation issues: {}", result.reasons)
            dropped = drop_invalid_board_ops(response)
            if dropped:
                log.warning("dropped {} invalid board op(s), kept {}",
                            dropped, len(response.board_ops))
        return {"response": response}

    async def solve(state: AnswerState) -> dict[str, Any]:
        # 独立求解/验算：用 sympy 校验模型答案（仅含等号的等式会生效）；
        # require_answer 避免把概念公式（如 ax+b=0）误判为「解不唯一」而污染话术
        response: AIResponse = state["response"]
        math_solver.apply_verification(response, supervisor.teacher.board, require_answer=True)
        return {"response": response}

    def route_after_validate(state: AnswerState) -> str:
        return "degrade" if not state["validation"].ok else "solve"

    graph: Any = StateGraph(AnswerState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("validate", validate)
    graph.add_node("degrade", degrade)
    graph.add_node("solve", solve)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges("validate", route_after_validate, {"degrade": "degrade", "solve": "solve"})
    graph.add_edge("degrade", "solve")
    graph.add_edge("solve", END)
    return graph.compile()


async def run_answer_graph(supervisor: "Supervisor", state: AnswerState) -> AIResponse:
    """执行答疑子图（编译结果按 Supervisor 缓存），返回 AI 产物。"""
    compiled = getattr(supervisor, "_lg_graph", None)
    if compiled is None:
        compiled = build_answer_graph(supervisor)
        supervisor._lg_graph = compiled  # noqa: SLF001 - 会话内缓存，避免每回合重编译
    final = await compiled.ainvoke(state)
    response: AIResponse | None = final.get("response")
    if response is None:
        return AIResponse(speech="这道题我们再想想看。")
    return response
