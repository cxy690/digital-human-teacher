"""会话级工具包：把可调用能力包装成模型可自主调用的工具（function calling）。

为什么需要它
------------
加装之前，「先检索 -> 再生成 -> 最后验算」的顺序被写死在答疑流水线上
（``agents/teacher/nodes.py`` 的 ``retrieve → generate → validate → degrade → solve``）：
模型没有机会决定「这题要不要查教材」「算出来的答案要不要复核」。
本模块把同一批能力包成**工具**交给模型，由它在 ReAct 闭环
（``agents/teacher/react.py``）里自己决定调哪个、调几次、什么时候收尾。

工具从哪来（复用而非另起一套）
------------------------------
* 复用 ``mcp_tools.registry`` 已有注册表 —— MCP 对外暴露的就是这批能力，
  内部链路用同一份实现，不存在「对外一套、对内另一套」的漂移；
* 注册表里需要二进制 / 坐标的接口（``ocr_parse`` 的 image bytes、
  ``whiteboard_write`` 的归一化坐标）换成 Agent 真正拿得到的东西：前者改成按
  ``image_id`` 找会话里已解析好的题干，后者**刻意不暴露**（见下）。

两条刻意的边界
--------------
1. **不暴露坐标级白板工具**：板书版式由确定性布局器 ``Whiteboard`` 负责
   （行距、防重叠、归一化坐标），让模型直接下发坐标只会把版面写乱；
   板上「内容」仍由模型的 JSON 契约产出，工具侧只允许追加补充行（``whiteboard_note``）。
2. **工具失败绝不外抛**：任何异常都变成一段「观察」文本回给模型，由模型自己决定
   换个方式还是直接作答 —— 工具是增强，不是新的故障点。

除原子工具外，本模块还登记 **Agent 技能**（``core/skills/agent_skills``）：
技能是带判断的业务能力（何时表扬、要不要翻旧账、本科怎么讲），由模型在同一套
function-calling 接口里自主调用，战绩按回合裁决回写（见 ``docs/agent_skills.md``）。
"""
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from agents.protocol import KIND_ERROR, ROLE_TEACHER, AgentBus
from config.constants import BoardOp
from config.logging import get_logger
from config.settings import settings
from mcp_tools.registry import ToolRegistry, build_default_registry

log = get_logger("agents.shared.toolkit")

ToolHandler = Callable[[dict[str, Any]], Awaitable[str] | str]

#: 观察文本的最大长度：回给模型的工具结果要短，太长会挤占讲解 token 预算
MAX_OBSERVATION_CHARS = 600

#: 读取白板时最多回给模型几行（板书可能很长，全量回灌没有意义）
MAX_BOARD_LINES = 12


@dataclass
class ToolCallRecord:
    """一次工具调用的审计记录（进 ``turn.meta`` 与调试面板）。"""

    name: str
    arguments: dict[str, Any]
    observation: str
    ok: bool
    elapsed_ms: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": dict(self.arguments),
            "observation": self.observation[:400],
            "ok": self.ok,
            "elapsed_ms": self.elapsed_ms,
            "reason": self.reason,
        }


def _spec(description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    """构造工具的参数 schema（JSON Schema 子集，够 function calling 用）。"""
    return {
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(required or []),
        },
    }


class AgentToolkit:
    """一段会话内可用的工具集（由 ``Supervisor`` 持有，``TeacherAgent`` 取用）。

    ``records`` / ``consume_board_notes`` 是**回合级**的：ReAct 一轮结束后由调用方
    取走并写进 ``turn.meta``，然后 :meth:`reset_turn` 清空，避免跨回合串数据。
    """

    def __init__(
        self,
        session: Any = None,
        *,
        parser: Any | None = None,
        retriever: Any | None = None,
        board: Any | None = None,
        max_calls: int | None = None,
    ) -> None:
        self.session = session
        self.board = board
        self.max_calls = int(
            max_calls if max_calls is not None else settings.agent_tool_max_calls
        )
        self._registry: ToolRegistry = build_default_registry(
            parser=parser, retriever=retriever, board=board
        )
        self._retriever = retriever
        self._records: list[ToolCallRecord] = []
        self._board_notes: list[str] = []
        #: 本回合已经附给模型看过的「实战补充」的教师 key（同一回合多个技能不必重复贴）
        self._learned_shown: set[str] = set()
        self._handlers: dict[str, tuple[dict[str, Any], ToolHandler]] = {}
        self._register_tools()
        self._register_skills()

    # ---------------------------------------------------------------- 属性
    @property
    def enabled(self) -> bool:
        """工具是否可用（开关关闭时为 False，调用方据此走「无工具」的老链路）。"""
        return bool(settings.agent_tools_enable and self._handlers)

    @property
    def records(self) -> list[ToolCallRecord]:
        return list(self._records)

    @property
    def call_count(self) -> int:
        return len(self._records)

    def tool_names(self) -> list[str]:
        return list(self._handlers)

    def schemas(self) -> list[dict[str, Any]]:
        """OpenAI function-calling 格式的工具清单。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": spec["description"],
                    "parameters": spec["parameters"],
                },
            }
            for name, (spec, _handler) in self._handlers.items()
        ]

    def trace(self) -> list[dict[str, Any]]:
        """本回合全部工具调用的可序列化轨迹。"""
        return [record.to_dict() for record in self._records]

    # ---------------------------------------------------------------- 回合
    def reset_turn(self) -> None:
        """回合开始前清空（调用方在每轮答疑入口调用）。"""
        self._records.clear()
        self._board_notes.clear()
        self._learned_shown.clear()

    def consume_board_notes(self) -> list[str]:
        """取走工具要求补充上板的行（取走即清空，避免重复上板）。"""
        notes = list(self._board_notes)
        self._board_notes.clear()
        return notes

    # ---------------------------------------------------------------- 调用
    async def call(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """执行一次工具调用，返回给模型看的「观察」文本。**任何情况下都不抛异常。**"""
        started = time.perf_counter()
        args = dict(arguments or {})
        if "__raw__" in args:
            return self._finish(
                name, args,
                f"工具参数不是合法 JSON，无法执行：{str(args['__raw__'])[:120]}",
                ok=False, started=started, reason="bad_arguments",
            )
        entry = self._handlers.get(name)
        if entry is None:
            return self._finish(
                name, args,
                f"没有名为 {name} 的工具。可用工具：{'、'.join(self._handlers)}",
                ok=False, started=started, reason="unknown_tool",
            )
        if len(self._records) >= self.max_calls:
            return self._finish(
                name, args,
                f"本回合工具调用次数已达上限（{self.max_calls} 次），请直接用已有信息作答。",
                ok=False, started=started, reason="limit",
            )
        try:
            # 工具 handler 可同步可异步（读白板这类纯本地读取没必要写成协程）
            outcome = entry[1](args)
            if inspect.isawaitable(outcome):
                outcome = await outcome
            return self._finish(name, args, str(outcome), ok=True, started=started)
        except Exception as err:  # noqa: BLE001 - 工具异常必须降级成「观察」，不能打断答疑
            log.warning("tool {} failed: {}", name, err)
            return self._finish(
                name, args,
                f"工具执行失败（{err}）。可以换一种方式，或直接用已有知识作答。",
                ok=False, started=started, reason="exception",
            )

    # ---------------------------------------------------------------- 工具实现
    def _register_tools(self) -> None:
        def add(name: str, spec: dict[str, Any], handler: ToolHandler) -> None:
            self._handlers[name] = (spec, handler)

        async def search(args: dict[str, Any]) -> str:
            query = str(args.get("query") or "").strip()
            if not query:
                return "缺少 query 参数。"
            if self._retriever is None:
                return "当前没有可用的教材知识库，请直接用你自己的知识作答。"
            top_k = _as_int(args.get("top_k"), default=settings.rag_top_k, low=1, high=8)
            hits = await self._registry.call("knowledge_search", query=query, top_k=top_k)
            if not hits:
                return f"教材里没有检索到与「{query}」直接相关的内容。"
            return "教材/题库检索结果：\n" + "\n".join(f"- {h}" for h in hits)

        async def solve(args: dict[str, Any]) -> str:
            equation = str(args.get("equation") or "").strip()
            if not equation:
                return "缺少 equation 参数（例如 3x+5=20）。"
            if not _math_available():
                return "当前环境没有 sympy，无法独立求解，请自行推理并说明依据。"
            solutions = await self._registry.call("math_solve", equation=equation)
            if not solutions:
                return f"没能解出「{equation}」，请检查式子是否写全（含未知数与等号）。"
            return f"{equation} 的解：" + "，".join(f"{k}={v}" for k, v in solutions.items())

        async def verify(args: dict[str, Any]) -> str:
            equation = str(args.get("equation") or "").strip()
            if not equation:
                return "缺少 equation 参数。"
            if not _math_available():
                return "当前环境没有 sympy，无法独立验算。"
            answer = str(args.get("answer") or "").strip()
            verdict = await self._registry.call(
                "math_verify", equation=equation, answer=answer
            )
            if verdict is None:
                return "无法判定（式子可能不含未知数或格式不完整）。"
            if answer:
                return f"结论：{answer} {'满足' if verdict else '不满足'} {equation}。"
            return f"结论：等式 {equation} {'成立' if verdict else '不成立'}。"

        def read_question(args: dict[str, Any]) -> str:
            """按 image_id 取会话里已 OCR 好的题干（不要求模型传二进制）。"""
            images = getattr(self.session, "board_images", None) or {}
            if not images:
                return "当前没有已上传的题目图片。"
            image_id = str(args.get("image_id") or "").strip()
            if not image_id:
                image_id = next(iter(reversed(list(images))), "")
            info = images.get(image_id)
            if not isinstance(info, dict):
                return f"没有找到图片 {image_id} 的识别结果。"
            text = str(info.get("ocr") or "").strip()
            subject = str(info.get("subject") or "").strip()
            if not text:
                return f"图片 {image_id} 没有识别到文字，请让学生重拍或改用文字描述。"
            head = f"图片 {image_id}" + (f"（{subject}）" if subject else "")
            return f"{head} 的识别题干：\n{text}"

        def read_board(_args: dict[str, Any]) -> str:
            ops = list(getattr(self.session, "board_ops", None) or [])
            lines = [
                str(op.text).strip()
                for op in ops
                if op.op is BoardOp.TEXT and op.text and str(op.text).strip()
            ]
            if not lines:
                return "白板目前是空的。"
            shown = lines[-MAX_BOARD_LINES:]
            return "白板当前内容（最后几行）：\n" + "\n".join(f"- {line}" for line in shown)

        def board_note(args: dict[str, Any]) -> str:
            text = str(args.get("text") or "").strip()
            if not text:
                return "缺少 text 参数。"
            if text not in self._board_notes:
                self._board_notes.append(text[:120])
            return f"已记下，将补写到白板：{text[:60]}"

        add(
            "knowledge_search",
            _spec(
                "在教材 / 题库中检索与问题相关的知识点片段。不确定概念、公式或教材原文时先查这里。",
                {
                    "query": {"type": "string", "description": "检索关键词或问题"},
                    "top_k": {"type": "integer", "description": "返回条数，默认 4"},
                },
                ["query"],
            ),
            search,
        )
        add(
            "math_solve",
            _spec(
                "用 sympy 独立求解含未知数的等式，返回未知数的解（如 3x+5=20 -> x=5）。"
                "题目涉及方程时先求解、再写讲解。",
                {"equation": {"type": "string", "description": "含未知数与等号的式子，如 3x+5=20"}},
                ["equation"],
            ),
            solve,
        )
        add(
            "math_verify",
            _spec(
                "用 sympy 复核等式是否成立、或答案是否满足方程。给出最终答案前应当复核一次。",
                {
                    "equation": {"type": "string", "description": "如 3x+5=20"},
                    "answer": {"type": "string", "description": "要复核的答案，如 x=5；留空则只校验等式"},
                },
                ["equation"],
            ),
            verify,
        )
        add(
            "ocr_parse",
            _spec(
                "取回某张已上传题目图片的识别题干（按 image_id）。图片题看不清条件时先读一次。",
                {"image_id": {"type": "string", "description": "图片 id，留空取最近一张"}},
            ),
            read_question,
        )
        add(
            "read_board",
            _spec("读取共享白板当前内容。学生说「上面那一步」「刚才写的」时用它确认。", {}),
            read_board,
        )
        add(
            "whiteboard_note",
            _spec(
                "在白板上追加一行补充说明（只写文字，版式由系统负责）。"
                "只用于「讲解中途想随手记一句」，完整的解题步骤仍写在最终 JSON 的 steps 里。",
                {"text": {"type": "string", "description": "要补写到白板上的一行文字"}},
                ["text"],
            ),
            board_note,
        )

    # ---------------------------------------------------------------- 技能
    def _register_skills(self) -> None:
        """把 Agent 技能（``core/skills/agent_skills``）登记成模型可调用的 function。

        技能与原子工具对模型是**同一个命名空间**（都在 ``_handlers`` 里），因此：

        * 调用次数同样受 ``AGENT_TOOL_MAX_CALLS`` 约束（技能内部可能再调 LLM，
          不设限会让一个回合的工具预算全耗在技能上）；
        * 失败同样降级成「观察」文本（走 :meth:`call` 的同一套收尾逻辑）；
        * 战绩记账由 :func:`core.skills.agent_skills.feedback.mark_skill_used` 负责，
          回合末由 ``Supervisor.handle_turn`` 按裁决回写。

        关掉 ``AGENT_SKILLS_ENABLE`` 时这里什么都不登记 —— 模型看到的工具清单
        与加装技能前完全一致（技能是可关的增强，不是新的故障点）。
        """
        if not settings.agent_skills_enable:
            return
        try:
            from core.skills.agent_skills import get_skill_registry  # noqa: PLC0415

            registry = get_skill_registry()
            for skill in registry.enabled():
                spec = skill.spec
                self._handlers[spec.name] = (
                    _spec(spec.description, dict(spec.inputs), list(spec.required)),
                    self._make_skill_handler(skill, registry),
                )
        except Exception as err:  # noqa: BLE001 - 技能装配失败绝不能影响答疑链路
            log.warning("agent skills 装配失败（忽略，回到只有工具的清单）：{}", err)

    def _make_skill_handler(self, skill: Any, registry: Any) -> ToolHandler:
        """把一项技能包成工具 handler（同步签名由 :meth:`call` 统一 await）。"""

        async def handler(args: dict[str, Any]) -> str:
            context = registry.context_for(skill, self.session)
            outcome = await skill.run(args, context)
            for note in outcome.board_notes:
                if note not in self._board_notes:
                    self._board_notes.append(note[:120])
            for event_type, payload in outcome.events:
                try:
                    if self.session is not None:
                        self.session.push_event(event_type, payload)
                except Exception:  # noqa: BLE001 - 客户端未接 emit 时静默
                    pass
            try:
                from core.skills.agent_skills import mark_skill_used  # noqa: PLC0415

                mark_skill_used(self.session, skill.spec.name)
            except Exception:  # noqa: BLE001 - 记账失败不影响本次调用
                pass
            # 学习增强必须**真的到模型眼前**：注册表把经验表里选出的「实战补充」装进了
            # 上下文，这里负责把它附在观察末尾 —— 否则技能照旧执行、模型照旧看不到，
            # 「自我更新」就只剩战绩那一半。同一回合多个技能只贴一次（按教师 key 去重）。
            return self._with_learned(outcome.text, context)

        return handler

    def _with_learned(self, text: str, context: Any) -> str:
        """把学到的「实战补充」附在观察文本末尾（空则原样返回）。"""
        notes = str(getattr(context, "learned_notes", "") or "").strip()
        teacher = str(getattr(context, "teacher_key", "") or "").strip()
        if not notes:
            return text
        if teacher and teacher in self._learned_shown:
            return text
        if teacher:
            self._learned_shown.add(teacher)
        return f"{text}\n\n{notes}" if text.strip() else notes

    # ---------------------------------------------------------------- 内部
    def _finish(
        self,
        name: str,
        arguments: dict[str, Any],
        observation: str,
        *,
        ok: bool,
        started: float,
        reason: str = "",
    ) -> str:
        """落记录 + 推事件 + 返回观察文本。"""
        text = (observation or "").strip()[:MAX_OBSERVATION_CHARS]
        record = ToolCallRecord(
            name=name,
            arguments=arguments,
            observation=text,
            ok=ok,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            reason=reason,
        )
        self._records.append(record)
        if not ok:
            # 失败路径进协作总线（与 AgentBus「只记转交与异常路径」的约定一致）；
            # 正常调用只走 turn.meta 与前端调试事件，避免把 80 条环形缓冲刷满
            AgentBus.record(
                self.session,
                sender=ROLE_TEACHER,
                recipient=name,
                kind=KIND_ERROR,
                ok=False,
                detail=f"工具调用失败（{reason or 'unknown'}）",
                arguments=arguments,
            )
        try:
            if self.session is not None:
                self.session.push_event(
                    "agent_tool",
                    {"name": name, "arguments": arguments, "ok": ok, "reason": reason},
                )
        except Exception:  # noqa: BLE001 - 客户端未接 emit 时静默
            pass
        return text


def _as_int(value: Any, *, default: int, low: int, high: int) -> int:
    """把模型给的数值参数夹到合法区间（模型经常给字符串或离谱数字）。"""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _math_available() -> bool:
    """sympy 是否可用（缺失时工具给出可读提示，而不是让模型拿到空结果）。"""
    try:
        from core import math_solver  # noqa: PLC0415 - 避免导入期绑定

        return bool(math_solver.available())
    except Exception:  # noqa: BLE001
        return False


@dataclass
class ToolkitPool:
    """按会话缓存工具包（``Supervisor`` 持有：一次会话只装配一次工具集）。"""

    parser: Any | None = None
    retriever: Any | None = None
    board: Any | None = None
    _cache: dict[str, AgentToolkit] = field(default_factory=dict)

    def get(self, session: Any) -> AgentToolkit:
        key = str(getattr(session, "session_id", "") or id(session))
        toolkit = self._cache.get(key)
        if toolkit is None:
            toolkit = AgentToolkit(
                session, parser=self.parser, retriever=self.retriever, board=self.board
            )
            self._cache[key] = toolkit
        return toolkit
