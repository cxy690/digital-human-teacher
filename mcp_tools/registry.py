"""MCP 工具注册中心：把可复用能力（OCR、白板、检索）暴露为可调用工具。

Agent / 上层编排可通过统一入口按名称调用工具，工具自带 JSON Schema，
便于后续接入真正的 MCP Server 或函数调用（Function Calling）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from config.constants import BoardOp
from config.logging import get_logger
from core import math_solver

log = get_logger("mcp_tools.registry")

ToolHandler = Callable[..., Awaitable[Any] | Any]


@dataclass
class ToolSpec:
    """工具定义。"""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    handler: ToolHandler | None = None


class ToolRegistry:
    """工具注册表。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def specs(self) -> list[dict[str, Any]]:
        """以 OpenAI function-calling 风格返回工具清单。"""
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in self._tools.values()
        ]

    async def call(self, name: str, **kwargs: Any) -> Any:
        spec = self._tools.get(name)
        if spec is None or spec.handler is None:
            raise KeyError(f"tool not found: {name}")
        result = spec.handler(**kwargs)
        if hasattr(result, "__await__"):
            result = await result  # type: ignore[misc]
        return result


def build_default_registry(
    *,
    parser: Any | None = None,
    retriever: Any | None = None,
    board: Any | None = None,
) -> ToolRegistry:
    """装配内置工具集。"""
    registry = ToolRegistry()

    async def _ocr(image: bytes) -> dict[str, Any]:
        if parser is None:
            return {"error": "parser_unavailable"}
        parsed = await parser.parse(image)
        return {"text": parsed.raw_text, "subject": parsed.subject, "type": parsed.question_type}

    async def _search(query: str, top_k: int = 4) -> list[str]:
        if retriever is None:
            return []
        return await retriever.retrieve(query, top_k=top_k)

    def _write_text(text: str, x: float, y: float) -> dict[str, Any]:
        if board is None:
            return {"error": "board_unavailable"}
        return board.write_text(text, x, y).model_dump()

    registry.register(
        ToolSpec(
            name="ocr_parse",
            description="识别题目图片，返回题干文本、学科与题型。",
            parameters={
                "type": "object",
                "properties": {"image": {"type": "string", "description": "图片二进制"}},
                "required": ["image"],
            },
            handler=_ocr,
        )
    )
    registry.register(
        ToolSpec(
            name="knowledge_search",
            description="在教材/题库中检索与问题相关的知识点片段。",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
                "required": ["query"],
            },
            handler=_search,
        )
    )
    registry.register(
        ToolSpec(
            name="whiteboard_write",
            description="在白板指定位置写入文字（坐标 0~1 归一化）。",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                },
                "required": ["text", "x", "y"],
            },
            handler=_write_text,
        )
    )
    registry.register(
        ToolSpec(
            name="whiteboard_clear",
            description="清空共享白板。",
            parameters={"type": "object", "properties": {}},
            handler=lambda: {"op": BoardOp.CLEAR.value},
        )
    )
    registry.register(
        ToolSpec(
            name="math_solve",
            description="用 sympy 求解含未知数的等式，返回未知数的解（如 3x+5=20 -> x=5）。",
            parameters={
                "type": "object",
                "properties": {"equation": {"type": "string", "description": "如 3x+5=20"}},
                "required": ["equation"],
            },
            handler=lambda equation: math_solver.solve_equation(equation),
        )
    )
    registry.register(
        ToolSpec(
            name="math_verify",
            description="用 sympy 校验等式是否成立，或校验答案是否满足方程。",
            parameters={
                "type": "object",
                "properties": {
                    "equation": {"type": "string"},
                    "answer": {"type": "string", "description": "如 x=5，可留空"},
                },
                "required": ["equation"],
            },
            handler=lambda equation, answer="": (
                math_solver.verify_answer(equation, answer)
                if answer
                else math_solver.check_equation(equation)
            ),
        )
    )
    return registry
