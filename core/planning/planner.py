"""回合规划器：模型在作答前先制定「怎么做这道题」的计划。

为什么需要它
------------
加装之前，「复杂题要先拆条件、多意图要逐条作答、图片题要先看 OCR」这些**策略**是
写死在代码里的（``_is_complex`` 的关键词表、``delegation_notes()`` 的固定话术）。
结果是：凡是规则没覆盖到的题，老师就只会「从头讲到尾」——它没有机会先想一下
「这题条件是不是不够，要不要先问学生」。

本模块把这一步交回模型：先产出一份结构化计划，再把计划作为上下文注入答疑链路。

三条硬约束
----------
1. **计划不能拖慢上课**：超时（默认 6s）或模型不可用一律回退**规则计划**，
   规则计划由意图判定结果直接生成，零额外开销；
2. **计划不能变成硬指令**：它只写进 ``context_notes``（提示词上下文），
   真正调不调工具、按什么顺序讲，仍由模型在 ReAct 闭环里决定；
3. **计划必须可回溯**：落进 ``turn.meta["turn_plan"]``，可序列化，排障时能看清
   「老师当时是怎么想的」。
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from config.logging import get_logger
from config.settings import settings
from models.base import BaseLLMClient
from models.llm import get_llm

if TYPE_CHECKING:  # 仅类型标注，避免 core 内部循环导入
    from core.intent.analyzer import IntentDecision
    from core.state.models import StudentTurn

log = get_logger("core.planning.planner")

#: 计划动作词表（也用于校验模型输出：不认识的动作降级成 explain）
ACTIONS: tuple[str, ...] = ("clarify", "retrieve", "explain", "verify", "ask_followup")

#: 建议工具名（必须是工具包里真实存在的，见 agents/shared/toolkit.py）
TOOLS: tuple[str, ...] = (
    "knowledge_search", "math_solve", "math_verify", "ocr_parse", "read_board",
)

#: 「像方程」的轻量判据：含等号**且**含字母未知数（与 core/math_solver 的适用面一致）。
#: 刻意写得宽松（不匹配具体形态）：3x+5=20 这种带运算的式子也要算进来，
#: 宁可多建议一次求解（模型可以不调），也不要漏掉一道该验算的题。
_LETTER_RE = re.compile(r"[A-Za-z]")

_PLAN_PROMPT = """你是教学规划器。老师开口之前，请你先为这一回合制定一份简短、可执行的计划。
只输出 JSON：
{{"goal": "这一回合要解决什么（不超过 20 字）",
 "steps": ["按顺序要做的 2~4 件事"],
 "tools": ["建议先调用的工具名，可选：{tools}"],
 "ask_followup": true/false,
 "reason": "为什么这么安排（不超过 30 字）"}}

判定要求：
1. 学生只是打招呼 / 闲聊 -> steps 只写一条「回应学生」，tools 为空数组，ask_followup=false；
2. 问概念、定义、教材原文 -> 建议 knowledge_search；
3. 题目含方程 / 等式 / 需要计算 -> 建议 math_solve 与 math_verify；
4. 题目条件不足、学生表述含糊 -> 第一步写「先问清缺少的条件」，ask_followup=true；
5. 复杂题（多条件、要分步推导）-> steps 要体现「先拆条件，再逐层推导」；
6. 不要复述题目，不要直接写出答案。

学生这次说：{text}
""".replace("{tools}", " / ".join(TOOLS))


@dataclass(frozen=True)
class PlanStep:
    """计划里的一步。"""

    action: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"action": self.action, "detail": self.detail}

    def text(self) -> str:
        return f"{self.action}：{self.detail}" if self.detail else self.action


@dataclass(frozen=True)
class TurnPlan:
    """一回合的执行计划。"""

    goal: str = ""
    steps: tuple[PlanStep, ...] = ()
    tools: tuple[str, ...] = ()
    ask_followup: bool = True
    source: str = "rule"          # llm / rule
    reason: str = ""

    @property
    def uses_tools(self) -> bool:
        return bool(self.tools)

    def to_meta(self) -> dict[str, Any]:
        """写进 ``turn.meta["turn_plan"]`` 的可序列化快照。"""
        return {
            "goal": self.goal,
            "steps": [step.to_dict() for step in self.steps],
            "tools": list(self.tools),
            "ask_followup": self.ask_followup,
            "source": self.source,
            "reason": self.reason,
        }

    def notes(self) -> list[str]:
        """注入 ``context_notes`` 的文本（空计划返回空列表）。"""
        if not self.goal and not self.steps and not self.tools:
            return []
        lines = ["【执行计划】（这是你自己刚才定下的安排，按它来组织这一轮讲解）"]
        if self.goal:
            lines.append(f"目标：{self.goal}")
        for index, step in enumerate(self.steps, 1):
            lines.append(f"{index}. {step.text()}")
        if self.tools:
            lines.append("建议先调用的工具：" + "、".join(self.tools))
        if not self.ask_followup:
            lines.append("本轮不需要向学生追问。")
        return ["\n".join(lines)]


class TurnPlanner:
    """回合规划器（模型优先，规则兜底）。"""

    def __init__(
        self,
        llm: BaseLLMClient | None = None,
        *,
        timeout: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._llm = llm
        self.timeout = float(
            timeout if timeout is not None else settings.agent_planner_timeout_seconds
        )
        self.max_tokens = int(
            max_tokens if max_tokens is not None else settings.agent_planner_max_tokens
        )

    @property
    def llm(self) -> BaseLLMClient:
        """懒取 LLM：规划器可能只用于规则兜底，不必提前建立客户端。"""
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    # ---------------------------------------------------------------- 对外
    async def plan(
        self,
        turn: "StudentTurn",
        decision: "IntentDecision | None" = None,
    ) -> TurnPlan:
        """产出本回合计划：模型可用则用模型，超时 / 失败 / 关闭时回退规则计划。"""
        rule_plan = self.rule_plan(turn, decision)
        if not settings.agent_planner_enable:
            return rule_plan
        if self._llm is None and not settings.llm_api_key:
            # 无密钥 = MockLLM，它不会产出 JSON 计划，省掉这一次无用的调用
            return rule_plan
        text = (turn.text or "").strip()
        if not text:
            return rule_plan
        try:
            raw = await asyncio.wait_for(
                self.llm.chat_json(
                    [{"role": "user", "content": _PLAN_PROMPT.format(text=text[:500])}],
                    temperature=0.2,
                    max_tokens=self.max_tokens,
                ),
                timeout=self.timeout,
            )
        except Exception as err:  # noqa: BLE001 - 规划失败绝不能阻断答疑
            log.warning("回合规划失败（回退规则计划）：{}", err)
            return rule_plan
        plan = self._from_llm(raw, fallback=rule_plan)
        if plan is None:
            return rule_plan
        log.info("回合计划（模型）：{} | 步骤 {} | 工具 {}",
                 plan.goal, len(plan.steps), "、".join(plan.tools) or "无")
        return plan

    # ---------------------------------------------------------------- 规则兜底
    def rule_plan(self, turn: "StudentTurn", decision: Any = None) -> TurnPlan:
        """规则计划：由意图判定结果直接派生，零额外开销（规划超时 / 关闭时的兜底）。

        ``decision`` 兼容两种形态：``IntentDecision`` 对象，或它写进 ``turn.meta``
        的字典快照（``to_meta()``）—— 两套编排留下的都是后者，不必为了规划再造一个对象。
        """
        text = (turn.text or "").strip()
        steps: list[PlanStep] = []
        tools: list[str] = []
        multi = bool(_field(decision, "multi", False))
        complex_ = bool(_field(decision, "complex", False))

        if multi:
            clauses = [str(c) for c in (_field(decision, "clauses", []) or [])]
            controls = {str(c) for c in (_field(decision, "control_clauses", []) or [])}
            tasks = [c for c in clauses if c not in controls]
            steps = [PlanStep("explain", q) for q in tasks[:4]] or [
                PlanStep("explain", "逐个回答学生的诉求")
            ]
            goal = f"逐条回答学生的 {len(tasks) or 1} 个诉求"
        elif complex_:
            steps = [
                PlanStep("clarify", "先拆解题目给出的条件与要求"),
                PlanStep("explain", "再逐层推导，写清每一步的依据"),
            ]
            goal = "拆解条件后分步推导"
        else:
            steps = [PlanStep("explain", "讲清这道题")]
            goal = "回答学生这一问题"

        if turn.image_id:
            tools.append("ocr_parse")
        if _looks_like_equation(text):
            tools.extend(["math_solve", "math_verify"])
        if not turn.image_id and text and len(text) <= 60 and not tools and not multi:
            # 短问题多半是概念 / 定义类提问：建议先查教材原文再讲
            tools.append("knowledge_search")

        return TurnPlan(
            goal=goal,
            steps=tuple(steps),
            tools=tuple(dict.fromkeys(tools)),
            ask_followup=bool(settings.interaction_enable),
            source="rule",
            reason="规则派生（模型规划未启用或不可用）",
        )

    # ---------------------------------------------------------------- 内部
    def _from_llm(self, raw: Any, *, fallback: TurnPlan) -> TurnPlan | None:
        """校验并归一化模型输出；不可用时返回 ``None``（调用方用规则计划）。"""
        if not isinstance(raw, dict) or not raw:
            return None
        goal = str(raw.get("goal") or "").strip()[:40]
        steps: list[PlanStep] = []
        for item in (raw.get("steps") or [])[:5]:
            if isinstance(item, dict):
                action = _clean_action(str(item.get("action") or ""))
                detail = str(item.get("detail") or "").strip()[:80]
            else:
                # 模型可能把步骤写成纯字符串（「先拆条件」），此时归类为 explain
                action, detail = "explain", str(item).strip()[:80]
            if action and (detail or action != "explain"):
                steps.append(PlanStep(action, detail))
        tools = [
            str(t).strip()
            for t in (raw.get("tools") or [])
            if str(t).strip() in TOOLS
        ]
        if not goal and not steps and not tools:
            return None
        return TurnPlan(
            goal=goal or fallback.goal,
            steps=tuple(steps) or fallback.steps,
            tools=tuple(dict.fromkeys(tools)),
            ask_followup=bool(raw.get("ask_followup", fallback.ask_followup)),
            source="llm",
            reason=str(raw.get("reason") or "").strip()[:60],
        )


def _looks_like_equation(text: str) -> bool:
    """文本里是否含「等号 + 字母未知数」（含等号的算式才值得建议 sympy 求解 / 验算）。"""
    if "=" not in text and "＝" not in text:
        return False
    return bool(_LETTER_RE.search(text))


def _field(source: Any, name: str, default: Any = None) -> Any:
    """从 ``IntentDecision`` 对象或它的字典快照里取字段（两种形态都能读）。"""
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _clean_action(value: str) -> str:
    """把模型给的动作名归一到词表内（不认识的一律当作 ``explain``）。"""
    name = (value or "").strip().lower()
    if name in ACTIONS:
        return name
    for action in ACTIONS:
        if action in name:
            return action
    return "explain"


@lru_cache(maxsize=1)
def get_turn_planner() -> TurnPlanner:
    """进程级默认规划器（LLM 客户端与答疑共用同一单例）。"""
    return TurnPlanner()
