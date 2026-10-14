"""意图识别 / 分流决策层：先判「简单意图」，复杂或多意图转交下游执行者。

它是会话入口的**第一道闸门**：学生每说一句，先由它回答两个问题——

1. 这句话是什么意图？（控制指令 / 闲聊 / 图片题 / 语音提问 / 文字提问）
2. 能不能一句话搞定？还是得交给下面的智能体来干活。

分流规则：
* **简单意图**：控制指令、闲聊、图片题、单一明确提问 —— 用**轻量规则**直接判定
  （零 LLM 开销、微秒级），按既有单意图链路（控制 / 闲聊 / 答疑）执行；
* **复杂意图**：长文本 / 多条件、需要分步推理；
* **多意图**：一句话里包含多个诉求（如「这道题怎么做，另外帮我打开麦克风」）。

后两类会标记为「转交下游」（``IntentDecision.delegate``），并附上子问题清单，
由 Supervisor + TeacherAgent 的答疑链路逐个实现 —— 本层只做**识别与分流**，
不做具体讲解。

**两级判定（规则 + 小模型）**：
* 一级 = 轻量规则（微秒级、零成本），覆盖控制指令 / 闲聊 / 单一提问等最常见输入；
* 二级 = 小模型复核（``core/intent/classifier.py``），**只在规则判不准（UNKNOWN）
  或判为复杂 / 多意图时**才调用 —— 常见输入不付模型开销，首响不受影响。

**为什么放在 ``core/`` 而不是 ``agents/``**：本类不持有会话状态、不调用其他执行者、
不参与 ``AgentBus``，也不需要任何生命周期钩子 —— 它只是总控下面的一层决策件，
不该占 ``agents/`` 的位置（``agents/`` 只放真正干活的执行者）。

与 ``core.router.IntentRouter`` 的关系：本类**内部持有一个 IntentRouter**（规则引擎），
并把 ``route`` / ``route_async`` / ``_looks_like_question`` 原样透出，
因此可以直接替换 Supervisor 里的 router —— 既有调用方无需改动。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from config.constants import IntentType
from config.logging import get_logger
from config.settings import settings
from core.intent.classifier import IntentLlmClassifier
from core.router import IntentRouter
from core.state.models import StudentTurn

# 「提问类」意图集合（多意图统计时用来数「问了几个问题」）
_QUESTION_INTENTS: tuple[IntentType, ...] = (
    IntentType.TEXT_QUESTION,
    IntentType.VOICE_QUESTION,
    IntentType.IMAGE_QUESTION,
)

# 多意图切分：句末标点，或「另外 / 还有 / 然后 …」这类并列连接词。
# 长连接词排前面，避免「再帮我」被「再」之类的短词先切走。
_CONNECTORS: tuple[str, ...] = (
    "另外", "还有", "以及", "顺便", "然后", "其次", "再帮我", "同时", "并且",
)
_CONNECTOR_ALT = "|".join(map(re.escape, sorted(_CONNECTORS, key=len, reverse=True)))
_CLAUSE_SPLIT_RE = re.compile(
    r"[。；;！!？?\n]+"
    r"|[,，](?:" + _CONNECTOR_ALT + r")"
    r"|(?:" + _CONNECTOR_ALT + r")"
)

# 复杂意图阈值：超长文本，或同时命中多个「需要分步推理」的信号词。
_COMPLEX_MIN_CHARS = 60
_COMPLEX_HINTS: tuple[str, ...] = (
    "证明", "推导", "一步一步", "分步", "详细讲讲", "结合", "比较", "区别", "分别", "为什么",
)


@dataclass(frozen=True)
class IntentSegment:
    """一个子诉求：原文片段 + 它自己的意图。

    与 ``clauses``（纯文本清单）的区别：这里**保留了每条子句对应的意图**，
    下游才能按条分发（控制当场办 / 提问逐条答疑），而不是把整句话丢给一个链路。
    """

    text: str
    intent: IntentType


@dataclass(frozen=True)
class IntentDecision:
    """一次输入的意图判定结果（简单 / 复杂 / 多意图 + 转交下游标记）。"""

    intent: IntentType
    simple: bool = True
    multi: bool = False
    complex: bool = False
    uncertain: bool = False     # 规则没识别出任何线索（关键词/疑问特征都没命中）
    sub_intents: list[IntentType] = field(default_factory=list)
    clauses: list[str] = field(default_factory=list)
    control_clauses: list[str] = field(default_factory=list)
    segments: list[IntentSegment] = field(default_factory=list)
    reason: str = ""

    @property
    def delegate(self) -> bool:
        """是否需要转交下游智能体实现（复杂 / 多意图）。"""
        return self.multi or self.complex

    def question_segments(self) -> list[IntentSegment]:
        """需要「逐条答疑」的子诉求（提问类）。"""
        return [s for s in self.segments if s.intent in _QUESTION_INTENTS]

    def control_segments(self) -> list[IntentSegment]:
        """需要「当场执行」的子诉求（设备/界面指令）。"""
        return [s for s in self.segments if s.intent is IntentType.CONTROL]

    def to_meta(self) -> dict[str, Any]:
        """可序列化快照（写进 ``turn.meta``，供日志与前端排查）。"""
        return {
            "intent": self.intent.value,
            "simple": self.simple,
            "multi": self.multi,
            "complex": self.complex,
            "uncertain": self.uncertain,
            "delegate": self.delegate,
            "sub_intents": [i.value for i in self.sub_intents],
            "clauses": list(self.clauses),
            "control_clauses": list(self.control_clauses),
            "segments": [{"text": s.text, "intent": s.intent.value} for s in self.segments],
            "reason": self.reason,
        }

    def delegation_notes(self) -> list[str]:
        """转交下游时附上的上下文：多意图给子问题清单，复杂题要求分步拆解。

        夹带的控制指令（如「顺带打开麦克风」）属于简单意图，由入口层当场执行，
        因此不进清单 —— 否则老师会去「讲解」一条设备指令。
        """
        if not self.delegate:
            return []
        tasks = [c for c in self.clauses if c not in self.control_clauses]
        if self.multi and tasks:
            lines = ["【意图分流】学生这次输入包含多个诉求，请按下面的顺序逐个作答，不要遗漏："]
            lines += [f"{i}. {c}" for i, c in enumerate(tasks, 1)]
            return lines
        if self.complex:
            return ["【意图分流】这是一个需要分步推理的复杂问题，请先拆解题目条件，再逐层推导给出结论。"]
        return []


class IntentAnalyzer:
    """意图识别 / 分流决策器（会话入口闸门；非 Agent）。

    不持有会话状态、不调用其他执行者、不参与 ``AgentBus`` —— 只把一次学生输入归一成
    :class:`IntentDecision`；「谁去干活」由 ``agents.supervisor.Supervisor`` 决定。
    """

    role = "router"          # 协议角色标识（见 agents/protocol.py 的角色表）

    def __init__(
        self,
        router: IntentRouter | None = None,
        *,
        llm_classifier: Callable[[str], Awaitable[IntentType]] | None = None,
        classifier: Any | None = None,
        name: str | None = None,
    ) -> None:
        # 拆掉 BaseAgent 外壳后自行持有名字与日志：本类无状态、无生命周期钩子，
        # 唯一需要的只是一个日志命名空间（原为 agents.router）。
        self.name = name or "intent"
        self.log = get_logger("core.intent.analyzer")
        self.router = router or IntentRouter(llm_classifier)
        # 二级判定（小模型）。``classifier=None`` 表示按配置自动装配；
        # 传入对象则直接用它（测试可注入桩），配置关闭时整体不装配。
        if classifier is not None:
            self.classifier = classifier
        else:
            self.classifier = IntentLlmClassifier() if settings.intent_llm_enable else None

    # ---------------------------------------------------------------- 兼容层
    # 直接替换 Supervisor 的 router 时，既有调用（route_async / route /
    # _looks_like_question）保持原样可用。
    def route(self, turn: StudentTurn) -> IntentType:
        return self.router.route(turn)

    async def route_async(self, turn: StudentTurn) -> IntentType:
        return await self.router.route_async(turn)

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        return IntentRouter._looks_like_question(text)  # noqa: SLF001 - 复用同一判定规则

    # ---------------------------------------------------------------- 识别/分流
    def analyze(self, turn: StudentTurn, *, base_intent: IntentType | None = None) -> IntentDecision:
        """同步判定：意图 + 是否简单 + 是否复杂 / 多意图（纯规则，零 LLM）。"""
        text = (turn.text or "").strip()
        base = base_intent if base_intent is not None else self.route(turn)
        if not text:
            return IntentDecision(intent=base, simple=True, reason="无文本，按既有意图直判")

        pairs = self._clauses_with_intent(text, from_voice=bool(turn.meta.get("from_voice")))
        clauses = [c for c, _ in pairs]
        segments = [IntentSegment(text=c, intent=k) for c, k in pairs]
        question_count = sum(1 for _, i in pairs if i in _QUESTION_INTENTS)
        distinct = {i for _, i in pairs}
        multi = question_count >= 2 or len(distinct) >= 2
        is_complex = self._is_complex(text)
        # 「规则没学到任何东西」= 有文本（不限图片题）却一个线索都没命中。
        # ``route()`` 对任意非空文本都兜底成提问，所以这里必须单独标出来，
        # 否则「判不出来」会被误当成「判准了」而永远不请小模型复核。
        uncertain = not pairs and turn.image_id is None

        if multi:
            # 多意图：主意图取「非控制类」的第一个子意图，保证答疑/闲聊链路不被控制指令带跑；
            # 若整句都是控制指令（如「打开麦克风，然后关闭摄像头」），仍按控制处理。
            primary = next((i for i in (i for _, i in pairs) if i is not IntentType.CONTROL), base)
            return IntentDecision(
                intent=primary,
                simple=False,
                multi=True,
                complex=is_complex,
                sub_intents=list(distinct),
                clauses=clauses,
                control_clauses=[c for c, i in pairs if i is IntentType.CONTROL],
                segments=segments,
                reason=f"识别到 {len(pairs)} 个诉求",
            )
        if is_complex:
            return IntentDecision(
                intent=base,
                simple=False,
                complex=True,
                uncertain=uncertain,
                sub_intents=list(distinct),
                clauses=clauses,
                control_clauses=[c for c, i in pairs if i is IntentType.CONTROL],
                segments=segments,
                reason="长文本 / 多条件，需要分步推理",
            )
        return IntentDecision(
            intent=base,
            simple=True,
            uncertain=uncertain,
            sub_intents=list(distinct),
            clauses=clauses,
            segments=segments,
            reason="规则未命中任何线索，交小模型复核" if uncertain else "单意图，轻量规则直判",
        )

    async def analyze_async(self, turn: StudentTurn) -> IntentDecision:
        """异步判定：规则直判优先，只有判不准 / 复杂·多意图才请小模型复核。

        常见输入（控制指令 / 闲聊 / 单一提问）**不调模型**，零额外首响开销；
        复杂与多意图本来就要走秒级 LLM 答疑，多一次分类调用不影响观感。
        模型超时 / 报错 / 输出不合法 -> 原样返回规则结果，绝不阻塞答疑。
        """
        decision = self.analyze(turn)
        if not self._needs_model(turn, decision):
            return decision
        try:
            data = await self.classifier.classify(  # type: ignore[union-attr]
                turn.text, base=decision.intent
            )
        except Exception as err:  # noqa: BLE001 - 注入的分类器也可能抛错，绝不能影响答疑
            self.log.warning("意图小模型复核失败（回退规则判定）：{}", err)
            return decision
        if not data:
            return decision
        merged = self._merge(decision, data, turn)
        self.log.info(
            "意图小模型复核：{} -> intent={} multi={} complex={}",
            (turn.text or "")[:40], merged.intent.value, merged.multi, merged.complex,
        )
        return merged

    def _needs_model(self, turn: StudentTurn, decision: IntentDecision) -> bool:
        """是否值得为这次输入付一次模型调用（只有这三种情况值）。

        触发条件刻意收得很紧 —— 每次调用都要多一次模型往返，而会话首响目标是 1.2s：
        * ``uncertain``：规则一个线索都没命中（判不准）；
        * ``delegate``：复杂 / 多意图，需要拆子问题、定主意图；
        * ``intent is UNKNOWN``：兜底（正常不会走到，``route()`` 对非空文本有默认值）。
        """
        if self.classifier is None or not (turn.text or "").strip():
            return False
        return (
            decision.uncertain
            or decision.delegate
            or decision.intent is IntentType.UNKNOWN
        )

    def _merge(
        self, decision: IntentDecision, data: dict[str, Any], turn: StudentTurn
    ) -> IntentDecision:
        """把模型判定合并进规则判定：模型给的信息优先，它没说的仍沿用规则。

        两个原则：
        * **控制指令仍以规则关键词为准**（设备指令是确定性的，不该交给模型猜），
          因此子问题清单里的控制项一律用规则的 ``_clause_kind`` 重新判定；
        * 模型只把 ``multi`` 往"是"里改、不往"否"里改 —— 规则已识别出的多意图
          不该被一次模型抖动抹掉（宁可多拆一个子问题，也不要漏答学生的诉求）。
        """
        # 注入的 classifier 可能是自定义实现，意图字段按类型校验（``to_meta`` 依赖 .value）
        raw_intent = data.get("intent")
        intent = raw_intent if isinstance(raw_intent, IntentType) else decision.intent
        subs = [str(s).strip() for s in (data.get("sub_questions") or []) if str(s).strip()]
        clauses = subs or list(decision.clauses)
        from_voice = bool(turn.meta.get("from_voice"))
        pairs = [(c, self._clause_kind(c, from_voice=from_voice)) for c in clauses]
        control_clauses = [c for c, kind in pairs if kind is IntentType.CONTROL]
        multi = bool(data.get("multi")) or decision.multi
        is_complex = bool(data.get("complex")) or decision.complex
        # 模型把控制指令塞进了同一句诉求里时，按「控制 + 其它」也算多意图
        if control_clauses and len(clauses) > len(control_clauses):
            multi = True
        sub_intents = list(
            dict.fromkeys(kind for _, kind in pairs if kind is not IntentType.UNKNOWN)
        )
        return IntentDecision(
            intent=intent,
            simple=not (multi or is_complex),
            multi=multi,
            complex=is_complex,
            sub_intents=sub_intents or list(decision.sub_intents),
            clauses=clauses,
            control_clauses=control_clauses,
            segments=[IntentSegment(text=c, intent=k) for c, k in pairs],
            reason=f"{decision.reason}（小模型复核）",
        )

    # ---------------------------------------------------------------- 执行计划
    def plan(self, decision: IntentDecision) -> list[IntentSegment]:
        """把判定结果整理成**按顺序执行**的子意图计划。

        顺序：控制指令 -> 提问（保持学生原话顺序）。

        为什么把控制指令排最前：它们即时、静默（开关麦克风/清板），先办掉不会打断讲解，
        反而能保证后面逐条答疑时的设备状态是对的。

        为什么闲聊不进计划：多意图里的寒暄（如「你好，这道题怎么做」）通常是衬词，
        为它单独播一句「我在呢」很啰嗦；**单意图闲聊**仍走既有 chitchat 链路不受影响。
        """
        return decision.control_segments() + decision.question_segments()

    # ---------------------------------------------------------------- 内部
    def _clauses_with_intent(
        self, text: str, *, from_voice: bool
    ) -> list[tuple[str, IntentType]]:
        """把一句话切成子句，并给出每个子句的意图（判不出的一律忽略）。"""
        pairs: list[tuple[str, IntentType]] = []
        for raw in _CLAUSE_SPLIT_RE.split(text):
            clause = raw.strip().strip("，,。.、;；:：!！?？ \t")
            if not clause:
                continue
            kind = self._clause_kind(clause, from_voice=from_voice)
            if kind is IntentType.UNKNOWN:
                continue
            pairs.append((clause, kind))
        return pairs

    def _clause_kind(self, clause: str, *, from_voice: bool) -> IntentType:
        """单个子句的轻量意图判定（关键词 + 疑问特征，不依赖「非空即提问」的默认值）。"""
        if any(kw in clause for kw in self.router.control_keywords):
            return IntentType.CONTROL
        if self._looks_like_question(clause):
            return IntentType.VOICE_QUESTION if from_voice else IntentType.TEXT_QUESTION
        if any(kw in clause for kw in self.router.chitchat_hints):
            return IntentType.CHITCHAT
        return IntentType.UNKNOWN

    @staticmethod
    def _is_complex(text: str) -> bool:
        """复杂判定：超长文本，或同时命中两个以上「要分步推理」的信号词。"""
        if len(text) >= _COMPLEX_MIN_CHARS:
            return True
        hits = sum(1 for hint in _COMPLEX_HINTS if hint in text)
        return hits >= 2
