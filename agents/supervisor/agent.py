"""AI 教师总控（Supervisor）：把路由、答疑、麦控串成完整会话闭环。

一次会话一个 Supervisor 实例，持有本会话的：
* 学生端代理（麦控 + 设备状态）；
* AI 教师（讲解 + 白板）；
* 意图路由 / 回复校验 / 题目解析；
* 可选作业模式与知识检索。

对外只暴露 ``handle_turn`` / ``on_vad`` 等事件入口。

**位置**：本模块是 ``agents/supervisor`` 包的实现（``agents/supervisor/agent.py``，
``role = supervisor``），与它调度/对接的执行者同属 ``agents`` 包——**包名 = 角色名**
（``student`` / ``teacher`` / ``lecturer`` / ``supervisor``）：

* ``agents/supervisor/``（本文件）—— 总调度：编排、抢占、回退；
* ``agents/teacher/`` · ``agents/lecturer/`` · ``agents/student/`` —— 被调度的执行者。

它虽然依赖 ``core.*`` 的一批能力模块，但那些模块对总控的引用只在
``TYPE_CHECKING`` 下（``core/turn_graph.py`` / ``core/langgraph_flow.py``），
运行期不存在反向导入，因此不会形成循环依赖。

装配入口：``core/harness.py``（工厂）与 ``core/loop.py``（事件分发），
外部沿用 ``from core import Supervisor`` 亦可（``core/__init__.py`` 惰性再导出）。
"""
from __future__ import annotations

import asyncio
import base64
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from agents.base import BaseAgent
from agents.lecturer.agent import LecturerAgent
from agents.lecturer import nodes as lecturer_nodes
from agents.protocol import (
    KIND_HANDOFF,
    ROLE_LECTURER,
    ROLE_SUPERVISOR,
    ROLE_TEACHER,
    AgentBus,
)
from agents.student import nodes as student_nodes
from agents.student.agent import StudentAgent
from agents.shared.emphasis import strip_emphasis
from agents.shared.toolkit import AgentToolkit, ToolkitPool
from agents.supervisor import nodes as supervisor_nodes
from agents.teacher import nodes as teacher_nodes
from agents.teacher.agent import TeacherAgent
from config.constants import ClassPhase, IntentType, is_valid_mood
from config.logging import get_logger
from config.settings import settings
from config.subjects import SUBJECT_BY_NAME, resolve_subject
from config.teachers import TeacherProfile, teacher_for_subject
from core import course_session
from core.audio_utils import append_tail_silence
from core.avatar_stream import save_avatar_video
from core.graph import TaskGraph
from core.guardrail import GuardrailVerdict, TeachingGuardrail
from core.hint_homework import HintLevel
from core.intent import IntentAnalyzer, IntentSegment
from core.langgraph_flow import LANGGRAPH_AVAILABLE, run_answer_graph
from core.memory import MemoryWriter
from core.planning import TurnPlan, get_turn_planner
from core.postprocess import postprocess_response
from core.router import IntentRouter
from core.safety_audit import safety_audit
from core.safety_gate import SafetyGate
from core.skills.agent_skills.feedback import record_skill_outcome
from core.skills.experience import SkillReflector
from core.skills.experience.feedback import record_turn_outcome
from core.state.models import (
    AIResponse,
    ChatMessage,
    StudentTurn,
)
from core.state.session import SessionState
from core.trace import traced
from core.turn_graph import run_turn_graph
from core.validator import ResponseValidator
from core.verdict import record_turn_verdict
from core.viseme import build_viseme_timeline
from document_parsing.pipeline import QuestionParser
from models.asr import get_asr
from models.avatar import get_avatar
from models.tts import get_tts, split_sentences

log = get_logger("agents.supervisor.agent")

# 问候词：待机问候命中这些词时，数字人用“打招呼片段”作底片对口型
_GREET_WORDS = ("你好", "您好", "哈喽", "哈啰", "嗨", "大家好", "同学们好", "早上好", "中午好", "晚上好")

# 开关指向 langgraph 但依赖缺失时，仅告警一次并回退自研图
_langgraph_warned = False


def _warn_langgraph_missing() -> None:
    global _langgraph_warned
    if not _langgraph_warned:
        log.warning("ORCHESTRATOR=langgraph but langgraph is not installed; fallback to builtin TaskGraph")
        _langgraph_warned = True


async def _aiter_known_last(sentences: list[str]) -> AsyncIterator[tuple[str, bool | None]]:
    """完整话术切出的句子表 -> ``(句子, 是否整段最后一句)`` 流。

    末句位置**事先算得出**，所以每句都能立刻判定「要不要补尾部静音」，不必等下一句。
    """
    last = len(sentences) - 1
    for index, sentence in enumerate(sentences):
        yield sentence, index >= last


async def _aiter_unknown_last(
    sentences: AsyncIterator[str],
) -> AsyncIterator[tuple[str, bool | None]]:
    """流式句子源（LLM 逐句产出）-> ``(句子, None)`` 流：末句位置要等下一句到达才知道。"""
    async for sentence in sentences:
        yield sentence, None


class Supervisor(BaseAgent):
    """单会话 AI 教师总控（Agent 角色 ``supervisor``：唯一持有会话状态的一方）。

    加装 :class:`~agents.base.BaseAgent` 只为了统一角色名、日志与生命周期钩子
    （``on_session_start`` / ``on_session_end``）—— 编排职责与状态归属不变：
    白板 / 历史 / 麦控 / 课件页序都在 ``SessionState`` 里，其它 agent 只通过传参拿它。
    """

    role = ROLE_SUPERVISOR

    def __init__(
        self,
        session: SessionState,
        *,
        student: StudentAgent | None = None,
        teacher: TeacherAgent | None = None,
        lecturer: LecturerAgent | None = None,
        router: IntentRouter | IntentAnalyzer | None = None,
        validator: ResponseValidator | None = None,
        parser: QuestionParser | None = None,
        retriever: Any | None = None,
    ) -> None:
        # 加装 Agent 身份：只提供 name / log 与生命周期钩子，会话状态仍由本类独占。
        super().__init__()
        self.session = session
        self.student = student or StudentAgent()
        self.teacher = teacher or TeacherAgent()
        # 讲课 Agent：与 teacher 同职责边界外的另一条链路（课件讲解 + 页面标注）。
        # 它不写正文板书、不答疑；学生讲课时提问由本层转交 teacher（见 _handle_turn_builtin）。
        self.lecturer = lecturer or LecturerAgent()
        # 意图识别 / 分流：会话入口闸门的决策层（core.intent.IntentAnalyzer，不是 Agent）。
        # 简单意图（控制/闲聊/单一提问）直接直判；复杂 / 多意图标注「转交下游」，
        # 再由下面这条答疑链路逐项实现。
        # 它同时透出 IntentRouter 的 route 接口，因此 ``self.router`` 既可当决策器
        # （analyze / analyze_async / plan），也兼容既有调用（route / _looks_like_question）。
        self.router = router if isinstance(router, IntentAnalyzer) else IntentAnalyzer(router)
        self.validator = validator or ResponseValidator()
        self.parser = parser or QuestionParser()
        self.guardrail = TeachingGuardrail()    # 教学护栏：学科范围 + 内容安全
        # 安全网关（见 core/safety_gate.py）：**挂载点**，而不是「记得调用一下的工具」。
        # 接线后每个学生输入与每段下行内容（话术 / 板书 / 分步解析 / 追问）都必经此处，
        # 没有「选择性调用」的余地 —— 护栏一旦能被绕过，它就不是护栏。
        self.safety = SafetyGate(reporter=self._report_safety)
        # 安全事件审计（共享单例；测试可替换为临时路径，避免污染 data/raw）。
        self.safety_audit = safety_audit
        self.retriever = retriever
        # ---- Agent 自主性（见 docs/agent_autonomy.md）----
        # 工具包：本会话可被模型自主调用的工具（检索 / 解方程 / 验算 / OCR / 读白板）
        # 白板取教师 agent 那一份（同一实例）；教师被替换成桩实现时取不到就给 None ——
        # 工具侧只用到它的版式方法，取不到也只是少一类工具，不该让装配期报错
        self.toolkits = ToolkitPool(
            parser=self.parser,
            retriever=self.retriever,
            board=getattr(self.teacher, "board", None),
        )
        # 规划器：回合开始前由模型产出「目标 + 步骤 + 建议工具」，超时回退规则计划
        self.planner = get_turn_planner()
        # 学情记忆写入方：判题判错零成本记错题，可选模型蒸馏薄弱点
        self.memory = MemoryWriter()
        self._lg_graph: Any | None = None       # LangGraph 答疑子图编译结果（懒加载缓存）
        self._turn_graph: Any | None = None     # LangGraph 回合主图编译结果
        self._idle_nudges = 0                   # 空闲问候已触发次数
        # 教师 Skill 自我更新（经验沉淀）：复盘器懒建 + 后台任务引用（防被 GC 提前回收）
        self.skill_reflector: SkillReflector | None = None
        self._bg_tasks: set[asyncio.Task[Any]] = set()
        self._slide_task: asyncio.Task[Any] | None = None
        self.asr = get_asr()
        self.tts = get_tts()
        self.avatar = get_avatar()

    # ================================================================ 生命周期
    async def on_session_start(self, session: SessionState) -> None:
        """会话建立钩子（BaseAgent 契约）：初始化学生端并置 AI 运行态。"""
        await self.student.on_session_start(session)
        session.push_status(ai_running=True)

    async def on_session_end(self, session: SessionState) -> None:
        """会话结束钩子（BaseAgent 契约）：停掉本会话后台任务，不留悬挂协程。"""
        session.auto_playing = False
        session.ai_speaking = False
        tasks = [task for task in self._bg_tasks if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            # 取消后必须 await 回收：否则 Skill 复盘等后台协程会继续跑（可能继续写经验文件），
            # 并在事件循环关闭时留下「Task was destroyed but it is pending」告警。
            # gather(return_exceptions=True)：取回异常，又不吞掉自身的取消信号。
            await asyncio.gather(*tasks, return_exceptions=True)
        self._bg_tasks.clear()

    async def on_start(self) -> None:
        """兼容入口（历史调用方：``SessionLoop`` / 单测 / ``run_demo``）。

        等价于 ``on_session_start(self.session)``；新代码优先用生命周期钩子。
        """
        await self.on_session_start(self.session)

    @traced(desc="空闲问候")
    async def on_idle(self) -> AIResponse | None:
        """长时间无交互时，AI 主动打招呼并**围绕刚才的问题**询问（温柔、不打扰）。"""
        if (
            not settings.idle_nudge_enable
            or self.session.course_active
            or self.session.auto_playing
            or self.session.scratch.get("suppress_idle_until_input")
            or self.session.ai_speaking
            or self.session.teaching
            or self._idle_nudges >= settings.idle_nudge_max
        ):
            return None
        self._idle_nudges += 1
        speech = await self._idle_speech_async()
        # 命中“你好”类问候词 -> 下发给数字人用“打招呼片段”对口型（其余时候用讲解底片）
        meta = {"gesture": "greet"} if any(w in speech for w in _GREET_WORDS) else {}
        return await self._speak_async(AIResponse(speech=speech, meta=meta))

    def _idle_speech(self) -> str:
        """生成主动问候话术：优先“对应上 AI 刚才问的问题”。"""
        # 1) 正在等学生作答的小测验/追问 -> 直接追问那题
        pq = self.session.pending_question or {}
        q = str(pq.get("question") or "").strip()
        if q:
            return f"同学，你还在屏幕前吗？刚才那道题——{q} 想到答案了吗？慢慢来，想好了告诉我就好。"
        # 2) 最近一条 AI 提问 -> 围绕它追问
        last_q = self._last_ai_question()
        if last_q:
            return f"同学，你还在屏幕前吗？关于「{last_q}」，有想法了吗？也可以直接说「继续」哦。"
        # 3) 无上下文 -> 温柔问候
        prompts = [
            "你好同学，你还在屏幕前吗？有不懂的随时叫我哦。",
            "我在呢，需要我继续讲哪道题吗？",
            "同学，休息一下也没关系，想继续时说一声就好～",
        ]
        return prompts[min(self._idle_nudges - 1, len(prompts) - 1)]

    def _idle_context(self) -> str:
        """空闲问候可围绕的“上一问题”上下文（优先待作答的小测验）。"""
        pq = self.session.pending_question or {}
        q = str(pq.get("question") or "").strip()
        return q or self._last_ai_question()

    async def _idle_speech_async(self) -> str:
        """空闲问候话术：优先用大模型生成（可能含“你好”->触发打招呼动作），失败回退模板。"""
        ctx = self._idle_context()
        try:
            from models.llm import get_llm  # 局部导入避免循环依赖

            prompt = (
                "你是耐心温柔的小学 AI 老师。学生已经静默了一小会儿。"
                "请用一句不超过30字的中文，自然地询问学生是否还在，可适度使用‘你好’等问候语；"
                "只输出这一句话，不要加引号或解释。"
            )
            if ctx:
                prompt += f" 可在句中顺带追问：{ctx}。"
            raw = await get_llm().chat([{"role": "user", "content": prompt}], temperature=0.8, max_tokens=60)
            text = (raw or "").strip().strip("“”\"' \n")
            if text and len(text) <= 60:
                return text
        except Exception as err:  # noqa: BLE001 - 失败回退模板
            log.warning("idle llm greet failed: {}", err)
        return self._idle_speech()

    def _last_ai_question(self) -> str:
        """取最近一条“像问题”的 AI 消息（用于空闲追问问对题）。"""
        for msg in reversed(self.session.history):
            if getattr(msg, "role", "") != "ai" or not getattr(msg, "text", ""):
                continue
            t = msg.text.strip()
            if t.endswith(("？", "?")) or ("吗" in t and len(t) <= 40):
                return t[:40]
        return ""

    # ================================================================ 语音
    def on_vad(self, has_voice: bool, ts_ms: int) -> None:
        """VAD 帧驱动麦控状态机（学生说话 -> AI 静默）。"""
        self.student.on_vad(has_voice, ts_ms, self.session)

    def voice_barge_in_due(self) -> bool:
        """「学生开口抢话、老师该让出话语权」判定（vad 帧驱动，见 ``SessionLoop._dispatch``）。

        与文字/语音提问走**同一条**打断路径（``SessionLoop._barge_in``）：
        学生按住说话期间后端就能立即闭嘴停讲，不必等松手后 ASR 转写回来。

        * 学生持续开口 ≥ ``VAD_SPEECH_MS``（麦控进入 ``student_speaking``）才开始算；
        * 仅当老师正在播报（``ai_speaking``）或连讲课件（``auto_playing``）时才需让位；
        * 同一轮说话只打断一次（``scratch["voice_barged"]`` 去重，vad 帧每 100ms 一帧），
          学生松手（vad 静默）后标记自动复位。
        """
        if not self.student.mic.student_speaking:
            self.session.scratch.pop("voice_barged", None)
            return False
        if self.session.scratch.get("voice_barged"):
            return False
        if not (self.session.ai_speaking or self.session.auto_playing):
            return False
        self.session.scratch["voice_barged"] = True
        return True

    async def on_voice(self, audio: bytes) -> AIResponse | None:
        """收到一段学生语音：转写 -> 若已静默则触发答疑。

        节点与连接顺序见 ``agents/student/nodes.py`` 的 ``student.voice``：
        关麦校验 -> ASR 转写 -> 空转写反馈 -> 构造回合 -> 交回 :meth:`handle_turn`。
        """
        run = await student_nodes.build_voice_graph(self).run({"audio": audio})
        return run.context.get("response")

    # ================================================================ 主导入
    @traced(root=True)
    async def handle_turn(self, turn: StudentTurn) -> AIResponse | None:
        """处理一次学生输入（按 ORCHESTRATOR 分派到 LangGraph 主图或内建实现）。

        这里是**调用链追踪的回合根**：返回时 ``core.trace`` 会打一条「经过了哪些函数」的日志。
        它同时是**裁决记录的唯一收口点**（见 :mod:`core.verdict`）：所有分支都在这里返回，
        因此成败信号只在这一处采集就够 —— 放到任何一个更靠内的分支都会漏掉别的分支。
        """
        self._idle_nudges = 0  # 学生一有输入，空闲问候计数归零
        self.session.scratch.pop("suppress_idle_until_input", None)
        started = time.perf_counter()
        response: AIResponse | None = None
        error: BaseException | None = None
        try:
            # 客户端送来 text=None 时下游 ChatMessage(text=None) 会抛 pydantic 校验错，
            # **整条 WebSocket 会话就断了**。这里先兜一次，保证脏输入进不了业务层。
            if not isinstance(turn.text, str):
                turn.text = "" if turn.text is None else str(turn.text)
            # 安全网关·输入侧：危机信号优先于一切教学逻辑。
            # 必须放在这个总入口（而不是各分支里），才能让「内建图 / LangGraph / 判题作答」
            # 所有路径都必经此处 —— 护栏一旦能被绕过，它就不是护栏。
            crisis = self.safety.screen_student_text(turn.text or "")
            if crisis.crisis:
                response = await self._turn_crisis(turn, crisis)
                return response
            if settings.orchestrator == "langgraph" and LANGGRAPH_AVAILABLE:
                response = await run_turn_graph(self, turn)
                return response
            if settings.orchestrator == "langgraph":
                _warn_langgraph_missing()
            response = await self._handle_turn_builtin(turn)
            return response
        except BaseException as err:  # 先记下异常再原样抛出，交给更外层处理
            error = err
            raise
        finally:
            # 观测层：把本回合的成败信号落盘。内部吞掉一切异常 —— 记录失败绝不能
            # 把课堂搞挂；也不能因为这里而改变 handle_turn 的返回值/异常语义。
            verdict = record_turn_verdict(
                session=self.session,
                turn=turn,
                response=response,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                error=error,
            )
            # 进化决策层（L2）：拿刚算出来的裁决，回写「本回合注入的经验到底有没有帮上忙」。
            # 必须紧挨着裁决：奖励信号就是裁决本身，隔远了会丢（下一轮会覆盖注入记账）。
            # 同样吞掉一切异常 —— 学习链路绝不能影响课堂。
            record_turn_outcome(self.session, verdict)
            # 技能战绩层：同一份裁决，回写「本回合调用过的技能有没有帮上忙」。
            # 记账对象来自 AgentToolkit 调用时写进 scratch 的技能名（见 agent_skills.feedback）。
            # 不记的话，技能就只能靠直觉维护 —— 那正是这套机制要替代的东西。
            record_skill_outcome(self.session, verdict)

    @traced(desc="回合分流")
    async def _handle_turn_builtin(self, turn: StudentTurn) -> AIResponse | None:
        """内建（TaskGraph）回合处理，与 LangGraph 主图行为保持一致。

        流程已图化：节点与连接顺序见 ``agents/supervisor/nodes.py`` 的 ``TURN_SPECS``
        —— ``route`` 节点做「待答追问优先 + 意图分流」，再按分流结果只执行
        ``answer_check`` / ``multi_intent`` / ``control`` / ``chitchat`` / ``teach``
        中命中的那一个（其余节点返回 ``__skip__`` 跳过）。
        与 ``core/turn_graph.py`` 的 LangGraph 主图同名同序，两套编排行为由
        ``tests/test_graph_parity.py`` 强绑定一致。
        """
        run = await supervisor_nodes.build_turn_graph(self).run({"turn": turn})
        return run.context.get("response")

    async def _turn_multi_intent(
        self, turn: StudentTurn, plan: list[IntentSegment]
    ) -> AIResponse:
        """多意图回合：控制指令当场办，提问**逐条**独立作答。

        与「把 N 个子问题塞进一次答疑」的区别：

        * 每条子问题独立走完整答疑链路，所以**学科护栏、RAG 检索、白板、追问**都是按条生效的
          —— 「数学题…另外帮我看看英语作文」里的英语那条会被单独拦下，而不是混在一段讲解里；
        * 白板按条刷新：学生看到的是「当前这一题」的板书，不是两道题混成一坨
          （前端会先清板再渲染，推全量天然正确）；语音也按条依次播报；
        * 追问只在**最后一条**之后提出 —— 否则前一题刚问完就被下一题覆盖，
          ``session.pending_question`` 也只会留下最后一题，学生答了也判不上。
        """
        questions = [seg for seg in plan if seg.intent is not IntentType.CONTROL]
        log.info(
            "多意图逐条作答：共 {} 条诉求（{} 条提问）-> {}",
            len(plan), len(questions), " | ".join(s.text[:20] for s in questions),
        )

        # 1) 控制指令：即时执行、不出声，先把设备状态摆正，避免打断后面的讲解
        for seg in plan:
            if seg.intent is IntentType.CONTROL:
                self._handle_control(seg.text)

        # 2) 提问：逐条独立答疑（每条自带白板与播报）
        last: AIResponse | None = None
        for index, seg in enumerate(questions):
            sub_turn = StudentTurn(
                intent=seg.intent,
                text=seg.text,
                # 只带「是不是语音说的」：子问题各自答疑，不该继承上一题的理解/子问题清单；
                # sub_turn 标记让规划层跳过「每子问题各请一次模型规划」（见 _resolve_plan）
                meta={
                    "from_voice": bool(turn.meta.get("from_voice")),
                    "sub_turn": True,
                },
            )
            last = await self._turn_teach(
                sub_turn, seg.intent, ask_followup=index == len(questions) - 1
            )
        if last is None:
            return AIResponse(speech="这几个问题我们一个一个来看。")
        last.meta["multi_intent_count"] = len(questions)
        return last

    @traced(desc="闲聊")
    async def _turn_chitchat(self) -> AIResponse:
        """闲聊回合：温柔回应，不进入答疑链路。"""
        return await self._speak_async(AIResponse(speech="我在呢，我们把不会的题发给我，我来帮你讲～"))

    async def _turn_crisis(self, turn: StudentTurn, verdict: GuardrailVerdict) -> AIResponse:
        """危机信号专线：**不答疑、不追问、不评判**，只做关怀 + 求助引导 + 上报。

        为什么必须独立成一条链路，而不是「过滤一下接着讲」：学生表达自伤意念时，
        继续讲一道数学题是最糟糕的回应；而把这句话打码后当普通问题处理，等于什么都没发生。
        """
        self.session.add_message(
            ChatMessage(role="student", text=turn.text, image_id=turn.image_id)
        )
        self.session.pending_question = None    # 危机优先：丢弃待答追问，不再索要作答
        self.session.followup_asked = 0
        self.session.push_event("crisis_alert", {
            "hits": list(verdict.crisis),
            "text": turn.text or "",
            "session_id": self.session.session_id,
        })
        return await self._speak_async(
            AIResponse(speech=verdict.speech or self.safety.CRISIS_SPEECH)
        )

    def _report_safety(self, payload: dict[str, Any]) -> None:
        """安全事件上报（由 :class:`~core.safety_gate.SafetyGate` 回调）。

        两条去向，缺一不可：

        * **推给前端**（``safety_alert``）—— 在场的人立刻能看到；
        * **落盘审计**（``data/raw/safety_events.jsonl``）—— ``push_event`` 只到
          当前连接，没人在线就等于没上报、刷新即丢、事后不可查；
          危机干预必须可追溯。

        后续无论接教师端监控台还是人工工单，都在这里挂，不必再改业务链路。
        """
        self.session.push_event("safety_alert", payload)
        self.safety_audit.record({
            **payload,
            "session_id": self.session.session_id,
            "student_id": self.session.student_id,
        })

    def _push_board(self, board_ops: list[Any]) -> None:
        """白板下发的**唯一出口**：先过安全网关，再推。

        为什么不让各处直接 ``session.push_board``：流式答疑与课件连讲会绕过 ``_speak``
        直接上板，板书就成了不经过滤的通道 —— 而板书恰恰是学生一定会看到的内容。
        """
        if not board_ops:
            return
        self.safety.screen_board_ops(board_ops)
        if board_ops:
            self.session.push_board(board_ops)

    async def _screen_sentence_stream(
        self, source: AsyncIterator[tuple[str, Any]]
    ) -> AsyncIterator[tuple[str, Any]]:
        """给流式句子流套一层安全过滤（保持 ``(文本, 是否末句)`` 结构不变）。

        与白板同理：流式话术不经过 ``_speak``，是一条约等于「学生耳朵」的直连通道。
        过滤后为空的句子直接丢弃（不当用语整句被遮蔽成 ``*`` 时仍保留，语义不丢）。
        """
        async for text, is_last in source:
            safe = self.safety.screen_text(text) if text else text
            if safe:
                yield safe, is_last

    def _record_lecture_handoff(self) -> None:
        """讲课中学生提问 = 一次**转交**：LecturerAgent 让出话语权，TeacherAgent 接手。

        判定用「有没有打开课件」而不是 ``auto_playing``：连讲被打断时 ``auto_playing``
        已经归零，但「学生是在课件课堂上提问」这个事实没有变。

        去重：同一轮学生输入只记一次 —— 多意图逐条作答会对每条子问题各调一次
        :meth:`_turn_teach`，而 ``speech_epoch`` 在每次学生输入入队时自增，
        正好可以当「轮次」标识。
        """
        deck_id = self.session.deck_id
        if not deck_id:
            return
        epoch = self.session.speech_epoch
        if self.session.scratch.get("_lecture_handoff_epoch") == epoch:
            return
        self.session.scratch["_lecture_handoff_epoch"] = epoch
        AgentBus.record(
            self.session,
            sender=ROLE_LECTURER,
            recipient=ROLE_TEACHER,
            kind=KIND_HANDOFF,
            detail="讲课中学生提问，课件讲解让出话语权，转答疑",
            page=self.session.page_index,
            deck_id=deck_id,
        )

    @traced(desc="答疑")
    async def _turn_teach(
        self, turn: StudentTurn, intent: IntentType, *, ocr_text: str = "", ask_followup: bool = True
    ) -> AIResponse:
        """答疑回合：标记教学中 -> 跑答疑链路 -> 播报 -> 视情况追问。

        ``ask_followup=False``：本回合不追问（多意图逐条作答时，只有最后一条才追问）。

        **转交记录点**：这是两条编排（内建 ``_handle_turn_builtin`` 与 LangGraph 主图）
        唯一的答疑入口，因此「讲课中学生提问」的转交记录放在这里，两套编排都不会漏。
        """
        self._record_lecture_handoff()
        # 学科范围护栏：问题明确指向其它学科时，礼貌引导回本学科，不进答疑链路
        # （作答回合不拦：学生回答「A」这类选项不含学科词，也无需范围判断）
        verdict = self.guardrail.screen_subject(turn.text or "", self.session.subject)
        if not verdict.in_scope:
            self.session.add_message(ChatMessage(role="student", text=turn.text, image_id=turn.image_id))
            return await self._speak_async(AIResponse(speech=verdict.redirect_speech))
        self.session.pending_question = None      # 新答疑：丢弃上一题待答追问
        self.session.followup_asked = 0
        self.session.teaching = True
        self.session.ai_speaking = True
        self.session.push_status(ai_running=True)
        try:
            # ReAct 闭环开启时**不走流式话术路**：那条路的「话术」与结构化答疑并行，
            # 模型还在调工具复核时话术就已经生成并播出去了，两个版本会在学生耳朵里打架。
            # 取舍（写在 docs/agent_autonomy.md）：要自主性（先复核再开口）就让出流式省下的
            # 那几秒；要极限首响，把 AGENT_REACT_ENABLE 关掉即回到原来的并行流式链路。
            react_active = settings.agent_react_enable and settings.agent_tools_enable
            use_streaming = (
                settings.digital_human_enable
                and self.avatar.produces_video
                and intent is not IntentType.IMAGE_QUESTION
                and not react_active
            )
            hint_level = self._resolve_hint_level(intent)
            board_first_mode = (settings.board_first_mode or "off").strip().lower()
            if board_first_mode != "off":
                # 白板先行：先写「题意理解」，再结合它给出最终解答（与流式链路互斥）
                # ★ 工具/技能接线：single_call 走的是「流式单次调用」（teacher.stream_board_first），
                # 那条路没有工具集，模型既调不了工具也调不了技能（Agent 技能会被静默跳过）。
                # 因此 ReAct 开启时改走 two_call —— 它内部走 teacher.answer，工具集齐全，
                # 且同样保留「先讲理解再给解答」的形态。取舍与上面 use_streaming 一致：
                # 要自主性（先查证再开口）就让出单次调用省下的那次往返。
                if react_active and board_first_mode == "single_call":
                    board_first_mode = "two_call"
                response = await self._turn_teach_board_first(
                    turn, intent, hint_level=hint_level, ocr_text=ocr_text,
                    mode=board_first_mode)
            elif use_streaming:
                response = await self._turn_teach_streaming(
                    turn, intent, hint_level=hint_level, ocr_text=ocr_text)
            else:
                response = await self._run_answer_pipeline(
                    turn, intent, ocr_text=ocr_text, hint_level=hint_level)
                await self._speak_async(response)
            # 记录本题知识点，供辅导课「随机出题」使用
            self.session.last_topic = str(response.meta.get("title") or turn.text or "").strip()[:40]
            if self.session.course_active and settings.course_quiz_enable:
                # 辅导课模式：出题节奏由 COURSE_QUIZ_INTERVAL_MINUTES 控制（默认讲满 30 分钟后一道）。
                # 到点就出（讲解刚结束正是最自然的出题时机，不必再等空闲巡检）；
                # 没到点就只讲解、不提问（既不模拟题也不追问），避免「讲一题问一题」打断讲课。
                if self.quiz_due():
                    await self.on_quiz_request(self.session.last_topic)
            elif ask_followup:
                self._ask_followup(response)
            # ★ Skill 自我更新：本回合答疑结束，异步复盘沉淀经验（不阻塞讲课）
            self._schedule_skill_reflection(turn, response)
            # ★ 学情记忆：同一次异步任务里顺带蒸馏这位学生的薄弱点（默认关闭，见 MEMORY_DISTILL_ENABLE）
            self._schedule_memory_distill(turn, response)
            return response
        finally:
            self.session.teaching = False
            self.session.ai_speaking = False
            self.session.push_status(ai_running=False)

    # ================================================================ 理解先行（先说理解 -> 再上解答）
    async def _turn_teach_board_first(
        self, turn: StudentTurn, intent: IntentType, *,
        hint_level: HintLevel | None, ocr_text: str = "", mode: str = "single_call",
    ) -> AIResponse:
        """理解先行分派：``single_call``（方案 B，单次调用两段）/ ``two_call``（方案 A，两次调用）。"""
        if mode == "two_call":
            return await self._board_first_two_call(
                turn, intent, hint_level=hint_level, ocr_text=ocr_text)
        return await self._board_first_single_call(
            turn, intent, hint_level=hint_level, ocr_text=ocr_text)

    @staticmethod
    def _understanding_text(resp: AIResponse) -> str:
        """把「理解」产物还原成文本，供阶段二作为上下文回流给老师。"""
        parts: list[str] = []
        title = str(resp.meta.get("title") or "").strip()
        if title:
            parts.append(title)
        parts.extend(f"· {p}" for p in (resp.meta.get("understanding") or []))
        steps = resp.meta.get("understand_steps") or []
        parts.extend(f"{i}. {s}" for i, s in enumerate(steps, 1))
        return "\n".join(parts)

    def _remember_understanding(self, resp: AIResponse) -> str:
        """记下本回合的「理解」文本（便于追问/复盘复用），并打一条日志。"""
        text = self._understanding_text(resp)
        self.session.scratch["understanding_text"] = text
        log.info("理解先行：已用语音讲出理解：{}", text.replace("\n", " | ")[:120])
        return text

    async def _board_first_single_call(
        self, turn: StudentTurn, intent: IntentType, *,
        hint_level: HintLevel | None, ocr_text: str = "",
    ) -> AIResponse:
        """方案 B：单次 LLM 产出「理解 + 解答」两段。

        理解段**只用语音讲**（不写白板），白板只下发结合后的解答；
        相比两次调用少一次 LLM 往返，且理解与解答同源不跑偏。
        """
        notes = await self._board_notes_async(turn, intent, ocr_text)
        spoken_understanding = False
        understand_task: asyncio.Task[AIResponse] | None = None
        final: AIResponse | None = None
        async for stage, resp in self.teacher.stream_board_first(
            turn, self.session, hint_level=hint_level, context_notes=notes,
        ):
            if stage == "understanding":
                self._remember_understanding(resp)
                spoken_understanding = True
                # 理解一出来就让老师先讲（后台任务），与「解答仍在流式生成」并行，首响更早
                understand_task = asyncio.create_task(self._speak_async(resp))
            elif stage == "board":
                # 边流边上板：解答段的标题/步骤一生成就写到白板上（累计快照，可能推多次）。
                # 这是「问了半天白板才出来」的主要修复点：不再等整段 JSON 生成完。
                if resp.board_ops:
                    self._push_board(resp.board_ops)
            else:
                final = resp

        if final is None:
            final = AIResponse(speech="这道题我们再想想看。")
        image_region = turn.image_region
        if intent is IntentType.IMAGE_QUESTION and turn.image_id and image_region:
            # 图片题：圈画**真实**题目区域（白板上只叠解答，圈画保留）
            x1, y1, x2, y2 = image_region
            final.board_ops = [
                self.teacher.board.annotate_region(
                    x1, y1, x2 - x1, y2 - y1, image_id=turn.image_id, label="题目区域")
            ] + list(final.board_ops)
        self._finalize_streamed_response(final, hint_level)

        # ★ 白板先上：LLM 一出结果就推给学生看。
        # 语音（克隆 TTS + 数字人逐句渲染）要数秒到十几秒，绝不能占着首屏时间——
        # 否则观感就是「问了半天白板才出来」。
        if final.board_ops:
            self._push_board(final.board_ops)

        # 先等「理解」讲完，再讲解答，保证两段语音不重叠
        if understand_task is not None:
            try:
                await understand_task
            except Exception as err:  # noqa: BLE001 - 理解播报失败不应阻断解答
                log.warning("understanding speech failed: {}", err)
        final.meta["understand_spoken"] = spoken_understanding
        final.meta["board_first"] = "single_call"
        await self._speak_async(final, push_board=False)   # 白板已下发，只补文本与语音
        return final

    async def _board_first_two_call(
        self, turn: StudentTurn, intent: IntentType, *,
        hint_level: HintLevel | None, ocr_text: str = "",
    ) -> AIResponse:
        """方案 A：两次 LLM 调用 —— 先用语音讲「理解」，再把理解回流生成结合后的解答上板。"""
        # ---- 阶段一：老师先用语音讲出「理解」（不上板）----
        draft = await self.teacher.draft_understanding(turn, self.session)
        understand_text = self._remember_understanding(draft)
        await self._speak_async(draft)

        # ---- 阶段二：把「理解」作为上下文回流给老师，生成结合后的解答 ----
        notes = await self._board_notes_async(turn, intent, ocr_text)
        combined_note = (
            "【你刚在语音里讲过的题意理解】\n"
            + (understand_text or "（无）")
            + "\n请在此基础上继续推导，给出完整、正确的解答与最终答案；"
            "不要重复照抄上面的理解文字，直接给出结论与关键步骤。"
        )
        if intent is IntentType.IMAGE_QUESTION:
            response = await self.teacher.answer_image(
                turn, self.session, ocr_text=ocr_text, hint_level=hint_level,
                extra_notes=[combined_note],
                toolkit=self.toolkit_for(self.session),
            )
        else:
            response = await self.teacher.answer(
                turn, self.session, hint_level=hint_level,
                context_notes=notes + [combined_note],
                toolkit=self.toolkit_for(self.session),
            )
        self._finalize_streamed_response(response, hint_level)

        # ---- 阶段三：白板只下发「解答」----
        response.meta["understand_spoken"] = True
        response.meta["board_first"] = "two_call"
        await self._speak_async(response)
        return response

    async def _board_notes_async(
        self, turn: StudentTurn, intent: IntentType, ocr_text: str
    ) -> list[str]:
        """异步取答疑前上下文：意图分流提示 + 回合计划 + RAG 检索（失败忽略）+ 图片题 OCR 说明。"""
        notes: list[str] = self._intent_notes(turn) + await self._plan_notes(turn)
        if self.retriever is not None and turn.text:
            try:
                # 检索结果**追加**在意图/计划说明之后：意图分流与执行计划是「怎么答」的
                # 指令，不该因为检索成功而丢失（历史行为是覆盖，等于把指令吃掉）
                notes += list(await self.retriever.retrieve(turn.text) or [])
            except Exception as err:  # noqa: BLE001 - 检索失败不影响答疑
                log.warning("retriever failed: {}", err)
        if intent is IntentType.IMAGE_QUESTION and ocr_text:
            notes = [f"题目OCR识别内容：{ocr_text}"] + notes
        return notes

    @staticmethod
    def _intent_notes(turn: StudentTurn) -> list[str]:
        """意图 Agent 给出的「转交上下文」：多意图的子问题清单 / 复杂题的拆解要求。

        由入口处的 :meth:`IntentAnalyzer.analyze` 写入 ``turn.meta``，
        下游答疑链路（TeacherAgent）据此逐个作答，而不是只答第一个诉求。
        """
        return [str(n) for n in (turn.meta.get("intent_notes") or []) if str(n).strip()]

    # ================================================================ Agent 自主性
    def toolkit_for(self, session: SessionState | None = None) -> AgentToolkit:
        """本会话的工具包（一次会话装配一次，供 TeacherAgent 在 ReAct 闭环里调用）。

        工具带着**这节课的真实上下文**：同一个检索器、同一个白板、同一批已 OCR 的图片，
        因此模型查到 / 算出来的结果与链路其余部分天然一致，不需要另建一套数据源。
        """
        return self.toolkits.get(session or self.session)

    async def _resolve_plan(self, turn: StudentTurn) -> TurnPlan:
        """取本回合的执行计划（**同一回合只规划一次**，结果缓存在 ``turn.meta``）。

        缓存的意义：流式答疑会并行跑「话术路」与「结构路」，两条路都要用同一份计划。

        多意图子回合（``meta["sub_turn"]``）**不请模型规划**：它们的目标已经由意图层
        拆好了（子问题清单就在上下文里），再规划一次既重复、又要为一次学生输入多付
        两三次模型往返 —— 规则计划足够，这也正是「计划不能拖慢上课」的具体落实。
        """
        cached = turn.meta.get("_turn_plan")
        if isinstance(cached, TurnPlan):
            return cached
        if turn.meta.get("sub_turn"):
            plan = self.planner.rule_plan(turn, turn.meta.get("intent_decision"))
        else:
            try:
                plan = await self.planner.plan(turn, turn.meta.get("intent_decision"))
            except Exception as err:  # noqa: BLE001 - 规划异常退化为「不规划」
                log.warning("回合规划异常，按无计划继续：{}", err)
                plan = TurnPlan(source="rule", reason="规划异常")
        turn.meta["_turn_plan"] = plan          # 供同回合复用（私有键，下划线前缀）
        turn.meta["turn_plan"] = plan.to_meta()  # 对外快照（可序列化，前端 / 排障可读）
        return plan

    async def _plan_notes(self, turn: StudentTurn) -> list[str]:
        """计划注入答疑上下文的文本（计划为空时返回空列表）。"""
        plan = await self._resolve_plan(turn)
        return plan.notes()

    # ================================================================ 互动追问
    def _ask_followup(self, response: AIResponse) -> None:
        """把 AI 产物里的 followup 设为待答问题，并推 ai_ask 让前端渲染作答卡片。"""
        followup = response.meta.get("followup")
        if not settings.interaction_enable or not isinstance(followup, dict):
            return
        question = str(followup.get("question") or "").strip()
        if not question or self.session.followup_asked >= settings.interaction_max_followups:
            return
        options = [str(o).strip() for o in (followup.get("options") or []) if str(o).strip()]
        # 只拦截明显缺少比较范围的选择题，保留“哪个数更大，2还是3”等完整题干。
        missing_choices = ("哪个数是有理数", "下列哪个", "下列哪一", "以下哪个", "以下哪一")
        if not options and any(word in question for word in missing_choices) and not any(
            mark in question for mark in ("：", ":", "还是", "A.", "A、", "①")
        ):
            log.warning("丢弃缺少候选项的选择型追问：{}", question)
            return
        self.session.pending_question = {
            "question": question,
            "expected": str(followup.get("answer") or "").strip(),
            "hint": str(followup.get("hint") or "").strip(),
            "options": options,
            "attempts": 0,
            "source_title": str(response.meta.get("title") or ""),
        }
        self.session.followup_asked += 1
        self.session.push_event(
            "ai_ask",
            {"question": question, "options": options, "index": self.session.followup_asked},
        )

    def _should_treat_as_answer(self, turn: StudentTurn) -> bool:
        """有待答追问、且本次输入不像“新问题”时，视为对追问的作答。"""
        if self.session.pending_question is None:
            return False
        text = (turn.text or "").strip()
        if not text or turn.image_id:
            return False
        if self.router.route(turn) is IntentType.CONTROL:
            return False
        # 模拟题/追问的选项作答：与选项原文精确一致即判定为作答
        # （选项可能含标点且较长，需豁免下面的长度与标点规则）
        pq = self.session.pending_question or {}
        options = [str(o).strip() for o in (pq.get("options") or [])]
        if options and text in options:
            return True
        # 像新问题（含疑问特征）则不当作答，视为新的答疑
        if self.router._looks_like_question(text):  # noqa: SLF001 - 复用同一判定规则
            return False
        # 过长或含句读的，多半是新的提问而非一个答案，按新答疑处理
        if len(text) > 30 or any(ch in text for ch in "，,。.；;、"):
            return False
        return True

    @traced(desc="判题作答")
    async def _turn_answer_check(self, turn: StudentTurn) -> AIResponse:
        """判定学生对追问的作答，给出反馈与提示（错误可重试，超限公布答案）。

        判题流水线已图化：节点与连接顺序见 ``agents/teacher/nodes.py`` 的
        ``teacher.judge``（``evaluate`` → ``decide`` → ``announce``）。
        会话态的置位/复位（``teaching`` / ``ai_speaking``）留在本方法：图本身没有
        ``finally``，异常时不能让「老师正在讲」的状态卡在那里。
        """
        self.session.teaching = True
        self.session.ai_speaking = True
        self.session.push_status(ai_running=True)
        try:
            run = await teacher_nodes.build_judge_graph(self).run({
                "turn": turn,
                "pending_question": self.session.pending_question or {},
            })
            response: AIResponse | None = run.context.get("response")
            return response or AIResponse(speech="再想想看～")
        finally:
            self.session.teaching = False
            self.session.ai_speaking = False
            self.session.push_status(ai_running=False)

    # ================================================================ 1v1 辅导课
    async def on_course_start(
        self, subject: str | None = None, teacher: str | None = None
    ) -> AIResponse | None:
        """进入辅导页：记录学科与所选老师，教师主动打招呼（语言 + 挥手动作）。

        ``teacher`` 为同科多老师时学生在选择界面挑中的档案 key；为空则用学科默认老师。
        阶段推进由前端计时器驱动，后端只用 ``course_session`` 做时间映射，
        不引入长驻定时器，避免与会话循环的 tick 二次调度冲突。
        """
        if subject:
            self.session.subject = resolve_subject(subject)
            # 同步写入 scratch：teacher 的提示词从 scratch 读学科（历史上无人写入，
            # 导致提示词永远显示「综合」，讲解/出题被 LLM 默认成数学）
            self.session.scratch["subject"] = self.session.subject
            # 学科英文 key：教师人格技能注册表（core/skills/teacher_personality）
            # 按 subject_id 取人格，这里写一次，后续 get_for_session() 即可命中。
            _meta = SUBJECT_BY_NAME.get(self.session.subject)
            self.session.scratch["subject_id"] = str((_meta or {}).get("key") or "")
        tid = (teacher or "").strip()
        if tid:
            self.session.teacher = tid
        self.session.scratch["teacher"] = self.session.teacher
        self.session.course_active = True
        self.session.scratch["suppress_idle_until_input"] = True
        self.session.course_started_at = time.time()
        self.session.course_deadline = self.session.course_started_at + course_session.duration_seconds()
        self.session.course_phase = ClassPhase.OPENING.value
        self.session.quiz_count = 0
        # 出题节奏重新起算：第一道题在开课满 COURSE_QUIZ_INTERVAL_MINUTES（默认 30）分钟后出
        self.session.scratch.pop("last_quiz_at", None)

        self.session.ai_speaking = True
        self.session.push_status(ai_running=True)
        try:
            # 同科多老师：用所选老师的专属开场白（体现风格差异）
            response = self.teacher.compose_course_greeting(
                self.session.subject, self.session.teacher
            )
            await self._speak_async(response)
        finally:
            self.session.ai_speaking = False
            self.session.push_status(ai_running=False)
        self.session.push_event("course_state", self._course_state())

        # 默认**不自动开课件**：打完招呼先进入自由问答，学生自己去点「课件」
        # 选了哪一份，才切到「按页讲解 + 标注 + 连讲」模式（见 handle_courseware）。
        # 想恢复「一上课就讲课件」，把 .env 的 COURSEWARE_AUTO_START 设为 true。
        if settings.courseware_auto_start:
            await self._auto_open_courseware()
        else:
            # 只发文字、不发声：否则每次开课都要多念一句，反而啰嗦。
            # 但必须给个入口提示，不然学生会不知道为什么"没有课件讲解"。
            self.session.add_message(ChatMessage(
                role="ai",
                text="（先随便问我问题吧～想让我照着课件讲，点工具栏的「课件」选一份就行）",
            ))
        return response

    async def _auto_open_courseware(self) -> None:
        """开课后自动挑一份课件讲第一页（学科匹配优先，找不到就不开）。"""
        from core.courseware.loader import get_store

        store = get_store()
        # 磁盘活必须进线程：以前这里直接调，冷启动时会**卡住事件循环**（200+ 份
        # pptx 全量解析要几十秒），期间所有学生端的 WebSocket 心跳都会超时断连。
        decks = await asyncio.to_thread(store.list_decks)
        if not decks:
            return

        subject = (self.session.subject or "").strip()
        picked: dict[str, Any] | None = None
        if subject:
            picked = next((d for d in decks if d.get("subject") == subject), None)
            if picked is None:
                # 未标学科的课件按「通用」处理，但只在唯一一份时才敢用，
                # 否则数学课可能自动开出一份语文 PPT。
                generic = [d for d in decks if not d.get("subject")]
                if len(generic) == 1:
                    picked = generic[0]
        if picked is None and len(decks) == 1:
            picked = decks[0]
        if picked is None:
            log.info("开课自动讲解：{} 未找到匹配课件（现有 {} 份），按普通答疑进行",
                     subject or "综合", len(decks))
            return

        deck = await asyncio.to_thread(store.get_deck, str(picked.get("deck_id") or ""))
        if deck is None or not deck.slides:
            return
        log.info("开课自动连讲：{} -> 课件「{}」共 {} 页",
                 subject or "综合", deck.title, deck.total)
        await self._play_slides(deck, 0)

    def _course_state(self) -> dict[str, Any]:
        """当前课程状态（前端据此更新学科 / 阶段 / 计时显示）。"""
        return {
            "subject": self.session.subject,
            "teacher": self._teacher_id(),
            "active": self.session.course_active,
            "phase": self.session.course_phase,
            "phase_label": course_session.phase_label(self.session.course_phase),
            "duration_seconds": course_session.duration_seconds(),
            "started_at": self.session.course_started_at,
            "deadline": self.session.course_deadline,
            "quiz_count": self.session.quiz_count,
            # 出题节奏（秒）：前端后续做「下次提问倒计时」时直接用它（40 分钟课堂计时后期再实现）
            "quiz_interval_seconds": int(self.quiz_interval_seconds()),
        }

    async def on_course_phase(self, phase: str | None) -> AIResponse | None:
        """前端计时器到点上报的阶段切换：播报过渡话术。

        正在讲解/思考时不插话（避免打断），仅更新阶段状态。
        """
        target = course_session.to_phase(phase)
        if target is None:
            return None
        self.session.course_phase = target.value
        if target is ClassPhase.ENDED:
            self.session.course_active = False
        self.session.push_event("course_state", self._course_state())
        if self.session.ai_speaking or self.session.teaching:
            return None
        response = self.teacher.compose_phase_intro(target, self.session.subject)
        if not response.speech:
            return None
        await self._speak_async(response)
        return response

    # ---------------------------------------------------------------- 出题节奏（每 N 分钟一道）
    def quiz_interval_seconds(self) -> float:
        """出题间隔（秒）：由 ``COURSE_QUIZ_INTERVAL_MINUTES`` 换算；<=0 表示关闭节流。"""
        try:
            minutes = float(settings.course_quiz_interval_minutes or 0)
        except (TypeError, ValueError):
            minutes = 0.0
        return max(0.0, minutes) * 60.0

    def quiz_due(self) -> bool:
        """是否已到「该出下一道模拟题」的时间点（节流关闭时恒为 True = 旧行为）。"""
        interval = self.quiz_interval_seconds()
        if interval <= 0:
            return True
        base = self.session.scratch.get("last_quiz_at") or self.session.course_started_at
        if not base:
            return False
        return (time.time() - float(base)) >= interval

    def _mark_quiz_done(self) -> None:
        """记下本次出题时间：下一次要再等满一个间隔（默认 30 分钟）。"""
        self.session.scratch["last_quiz_at"] = time.time()

    async def maybe_course_quiz(self) -> AIResponse | None:
        """会话空闲巡检：到点且老师空闲时主动出一道模拟题（默认讲满 30 分钟后一道）。

        与 :meth:`_turn_teach` 收尾处的判断共用同一套节流（谁先到点谁出题），
        所以「每 N 分钟提问一次」由 ``last_quiz_at`` 单点推进，不会重复出题。
        老师正在讲解 / 连讲课件 / 上一题还没作答时让位，等下一个巡检点再补。
        """
        if not settings.course_quiz_enable or not self.session.course_active:
            return None
        if self.session.ai_speaking or self.session.teaching or self.session.auto_playing:
            return None
        if self.session.pending_question:       # 上一题还没作答：不叠题
            return None
        if not self.quiz_due():
            return None
        return await self.on_quiz_request(self.session.last_topic)

    @staticmethod
    def _lead_readable(text: str, *, limit: int = 16) -> bool:
        """过渡语里能不能安全带上一段文字：要短、且不含数学符号。

        TTS 念「x²+6x=-4」这类公式既别扭又容易读错，含符号的页标题一律不带，
        退回通用过渡语（学生听到的效果依然是「老师讲到这里要做题了」）。
        """
        t = str(text or "").strip()
        if not t or len(t) > limit:
            return False
        return re.search(r"[=+×÷√≈≠≤≥²³π°]|(?<![A-Za-z])x(?![A-Za-z])", t) is None

    def _quiz_lead(self, quiz: Any) -> str:
        """出题前的过渡语：先说清「围绕什么、为什么现在做题」，避免突然出题显得莫名其妙。

        优先用**当前课件页标题**（学生眼前刚看完的内容），其次上一题知识点；
        两者都不适合念（含公式/太长）则退到通用说法。
        """
        page_title = str(self.session.scratch.get("slide_page_title") or "").strip()
        if self._lead_readable(page_title):
            return f"我们刚讲了「{page_title}」，来做一道题检验一下"
        topic = str(getattr(quiz, "topic", "") or "").strip()
        if self._lead_readable(topic, limit=12) and topic != str(self.session.subject or ""):
            return f"关于「{topic}」，我来出一道题检验一下"
        return "我们来做一道题检验一下刚才的内容"

    async def on_quiz_request(self, topic: str = "") -> AIResponse | None:
        """生成一道模拟题并进入「等待作答」状态。

        复用既有追问判题链路：题目写入 ``pending_question``，学生的作答会经
        ``_should_treat_as_answer`` -> ``_turn_answer_check`` 自动判题与反馈。
        """
        if not settings.course_quiz_enable:
            return None
        self.session.teaching = True
        self.session.ai_speaking = True
        self.session.push_status(ai_running=True)
        try:
            # 出题流水线（节点与顺序见 agents/teacher/nodes.py 的 teacher.quiz）：
            # 生成题目 -> 写入待答状态 + 推 ai_quiz -> 播报
            # grade 供真题库按学生年级选题（登录用户 users.grade 由 server 写入 scratch，
            # 见 server/websocket.py；缺失时题库内部按「不限年级」处理）。
            run = await teacher_nodes.build_quiz_graph(self).run({
                "subject": self.session.subject,
                "topic": topic or self.session.last_topic,
                "grade": self.session.scratch.get("grade", ""),
            })
            return run.context.get("response")
        finally:
            self.session.teaching = False
            self.session.ai_speaking = False
            self.session.push_status(ai_running=False)

    # ================================================================ 图片
    @traced(desc="图片题")
    async def handle_image(self, image_id: str, image_bytes: bytes, text: str = "") -> AIResponse | None:
        """上传题目图片：OCR 解析 + 图片讲解。"""
        parsed = await self.parser.parse(image_bytes)
        self.session.board_images[image_id] = {"ocr": parsed.raw_text, "subject": parsed.subject}
        self.session.scratch["last_parsed"] = parsed
        turn = self.student.build_turn_from_image(image_id, text=text or parsed.content)

        # 题目区域：用 OCR 行框算出来随回合带下去，白板据此圈画（见 document_parsing/region.py）。
        # 定位不出来就不写 —— 下游宁可不画框，也不罩住整页误导学生。
        region = parsed.question_region()
        if region:
            turn.meta["image_region"] = list(region)
        else:
            log.info("题目区域未定位出来（{} 个 OCR 行框），本回合不画圈画框", len(parsed.boxes))

        return await self._turn_teach(turn, IntentType.IMAGE_QUESTION, ocr_text=parsed.raw_text)

    # ================================================================ 课件（PPT）
    async def _current_deck(self) -> Any | None:
        """取当前打开的课件（未打开 / 课件已删除时返回 ``None``）。

        冷启动时取课件会触发**同步全量扫描**（231 份约 40~60s），因此放线程里跑，
        避免把事件循环卡死（卡死期间 WebSocket 心跳会超时、连接直接断掉）。
        """
        from core.courseware.loader import get_store

        if not self.session.deck_id:
            return None
        return await asyncio.to_thread(get_store().get_deck, self.session.deck_id)

    async def handle_courseware(self, deck_id: str, index: int = 0, *, speak: bool = True) -> AIResponse | None:
        """打开一份课件，并从指定页开始**连续讲下去**（前端「选课件」触发）。"""
        from core.courseware.loader import get_store

        deck = await asyncio.to_thread(get_store().get_deck, str(deck_id or "").strip())
        if deck is None or not deck.slides:
            return await self._speak_async(AIResponse(speech="这份课件还没有准备好哦，换一份试试。"))
        if not speak:
            return await self._present_slide(deck, max(0, int(index)), speak=False)
        await self._schedule_slide_playback(deck, max(0, int(index)))
        return None

    async def handle_slide_goto(self, index: int, *, speak: bool = True) -> AIResponse | None:
        """跳到某一页，并从那里接着**连讲**（前端上一页 / 下一页 / 页码跳转）。"""
        deck = await self._current_deck()
        if deck is None:
            return await self._speak_async(AIResponse(speech="我们还没有打开课件哦，先选一份课件吧。"))
        if index < 0:
            return await self._speak_async(AIResponse(speech="这已经是第一页啦。"))
        if index >= deck.total:
            return await self._speak_async(AIResponse(speech="这已经是最后一页啦。"))
        if not speak:
            return await self._present_slide(deck, index, speak=False)
        await self._schedule_slide_playback(deck, index)
        return None

    async def _schedule_slide_playback(
        self,
        deck: Any,
        start: int,
        resume_events: list[tuple[str, dict[str, Any]]] | None = None,
    ) -> None:
        """单会话只保留一个连讲任务，旧任务回收后再启动新任务。

        ``resume_events`` 非空 = 断点续讲：本页不重讲，直接补发这些剩余语音事件。
        """
        if self._slide_task is not None and not self._slide_task.done():
            await self._reap_task(self._slide_task)
        task = asyncio.create_task(
            self._play_slides(deck, start, resume_events=resume_events))
        self._slide_task = task
        self._bg_tasks.add(task)

        def _finish(done: asyncio.Task[Any]) -> None:
            self._bg_tasks.discard(done)
            if done.cancelled():
                return
            try:
                done.result()
            except Exception as err:  # noqa: BLE001 - 后台连讲失败不能影响后续课堂操作
                log.exception("课件连讲任务异常：{}", err)

        task.add_done_callback(_finish)

    async def handle_slide_step(self, step: int) -> AIResponse | None:
        """相对翻页（+1 下一页 / -1 上一页），同样接着连讲。"""
        return await self.handle_slide_goto(self.session.page_index + int(step))

    async def handle_slide_play(self, index: int = -1) -> None:
        """从指定页（默认当前页）开始连讲（前端「继续讲」按钮）。

        ``index < 0``（＝「继续讲」）时优先**断点续讲**：把本页中断前剩余的语音事件
        直接补发（不重新生成话术、不重新渲染），学生听到的正好接着上次停下的地方 ——
        旧行为是整页重讲（「暂停一下再继续，老师从头念」），体验很差。
        """
        deck = await self._current_deck()
        if deck is None:
            await self._speak_async(AIResponse(speech="我们还没有打开课件哦，先选一份课件吧。"))
            return
        start = self.session.page_index if int(index) < 0 else int(index)
        resume_events: list[tuple[str, dict[str, Any]]] | None = None
        if int(index) < 0:
            remaining = self._slide_resume_events(start)
            if remaining:                       # 有断点：本页从停下的地方接着讲
                resume_events = remaining
            elif remaining is not None and start + 1 < deck.total:
                start += 1                      # 本页恰好已讲完：继续 = 讲下一页
        await self._schedule_slide_playback(deck, start, resume_events=resume_events)

    async def handle_slide_stop(self) -> None:
        """停止连讲（前端「暂停」按钮）。

        除了置位 ``auto_playing=False``，还要自增 ``speech_epoch``：
        正在渲染的那一句语音会在下一个句边界立刻中止，老师马上闭嘴。
        """
        self.session.auto_playing = False
        self.session.speech_epoch += 1
        self.session.push_event("avatar_interrupt", {})
        self.session.push_event("slide_play_state", {"playing": False, "finished": False})

    def abort_auto_play(self) -> None:
        """学生插话/要求静音：立即中止课件连讲并同步前端按钮状态（幂等）。

        由 ``SessionLoop._barge_in`` 统一调用（文字/语音/图片/翻页等一切打断场景）。
        只置标志 + 推状态，**不 cancel 连讲任务**：在途播报由 ``speech_epoch`` 作废，
        连讲任务在下一个检查点自行收工（见 :meth:`_play_slides`），
        这样「打断方自己要讲的话」不受影响。

        翻页/重讲场景由 ``_schedule_slide_playback`` 直接取消旧任务、新任务推
        ``playing=True``，不走这里；所以这里的 ``playing=False`` 不会和新连讲打架。
        """
        if not self.session.auto_playing:
            return
        self.session.auto_playing = False
        self.session.push_event("slide_play_state", {"playing": False, "finished": False})

    # ---------------------------------------------------- 课件断点续讲（暂停/插话后接着讲）
    def _open_slide_event_log(
        self,
        deck_id: str,
        index: int,
        events: list[tuple[str, dict[str, Any]]] | None = None,
    ) -> None:
        """开始记录第 ``index`` 页的语音事件，供「继续讲」从断点续发。

        * ``events`` 非空 = 预取路径（整页事件已备货，直接当裁剪基准）；
        * ``events`` 为空 = 单页/流式路径：由 :meth:`_record_slide_event` 在逐句
          下发时累积（``_push_audio(record=True)`` 只对页讲解开，答疑播报不会混进来）。

        收到前端 ``slide_pos`` 上报只记进度；真正的裁剪发生在
        :meth:`_slide_resume_events`（「继续讲」那一刻，只裁一次 = 幂等）。
        """
        self.session.scratch["slide_event_log"] = {
            "deck_id": str(deck_id or ""),
            "index": int(index),
            "base": list(events) if events else [],
            "pos": 0,
        }

    def _record_slide_event(self, event: str, payload: dict[str, Any]) -> None:
        """页讲解的语音事件逐条入账（只有带 ``record=True`` 的讲解链路会调）。"""
        log = self.session.scratch.get("slide_event_log")
        if log is not None:
            log["base"].append((event, payload))

    def on_slide_pos(self, index: int, pos: int) -> None:
        """前端上报「本轮已播语音事件数」（打断瞬间发一次）；续讲时据此裁剪剩余。"""
        log = self.session.scratch.get("slide_event_log")
        if not log or int(log.get("index", -1)) != int(index):
            return
        log["pos"] = max(0, int(pos))

    def _slide_resume_events(self, start: int) -> list[tuple[str, dict[str, Any]]] | None:
        """第 ``start`` 页的续讲剩余事件；不可续讲返回 ``None``（调用方重讲本页）。

        裁剪 ``base[pos:]`` 只在这里做一次，并把基准滚动到剩余 —— 前端重复上报
        同一 pos（``bargeIn`` 被本地与 ``avatar_interrupt`` 各触发一次）也不会重复裁剪。
        返回空列表 = 本页已讲完（由调用方决定是翻下一页还是重讲）。
        """
        log = self.session.scratch.get("slide_event_log")
        if not log or log.get("deck_id") != self.session.deck_id:
            return None
        if int(log.get("index", -1)) != int(start):
            return None
        base = list(log.get("base") or [])
        pos = max(0, min(int(log.get("pos", 0)), len(base)))
        remaining = base[pos:]
        log["base"] = list(remaining)
        log["pos"] = 0
        return remaining

    async def _play_slides(
        self,
        deck: Any,
        start: int = 0,
        resume_events: list[tuple[str, dict[str, Any]]] | None = None,
    ) -> None:
        """**连续讲课件**：一页讲完自动翻下一页，直到讲完、或被学生打断。

        ``resume_events`` 非空 = 断点续讲（见 :meth:`handle_slide_play`）：
        第 ``start`` 页不再重新生成/渲染，直接补发这些剩余语音事件。

        要同时保证两件事，学生才不会觉得「老师在念 PPT、而且跟画面不同步」：

        1. **预生成**：讲第 i 页时，后台已经在写第 i+1 页的话术、并把语音渲染好
           （:meth:`_prepare_slide`）——翻页时不必再等「LLM 生成 + TTS + 数字人渲染」；
        2. **页音同到**：翻页那一刻把「页图 + 标注 + 语音」一次性下发
           （:meth:`_present_prepared`），学生听到的第一句就对得上眼前的画面。

        打断机制：学生一开口，``SessionLoop._barge_in`` 就把 ``speech_epoch`` 加一，
        这里在每页开始前比对，一旦变了立刻收工——所以「一直讲下去」不等于
        「停不下来」，学生随时能插话把老师喊停（在途的预取任务也会被取消）。
        """
        total = deck.total
        if total <= 0:
            return
        start = max(0, min(int(start), total - 1))
        epoch = self.session.speech_epoch
        self.session.scratch["slide_ready_index"] = start - 1
        self.session.auto_playing = True
        self.session.push_event("slide_play_state",
                                {"playing": True, "finished": False, "index": start, "total": total})
        finished = False
        prepared: Any = None
        prepare_task: "asyncio.Task[Any] | None" = None
        try:
            for index in range(start, total):
                if not self.session.auto_playing or self.session.speech_epoch != epoch:
                    log.info("课件连讲中止（讲到第 {}/{} 页）", index + 1, total)
                    break
                if resume_events is not None and index == start:
                    # 断点续讲：页面已经亮着（不推页图、不清标注），直接补发剩余语音事件
                    log.info("课件续讲：第 {} 页从断点补发 {} 条语音事件",
                             index + 1, len(resume_events))
                    for event, payload in resume_events:
                        self.session.push_event(event, payload)
                    resume_events = None
                elif prepared is None:
                    # 首屏（学生刚选完课件）或上一页预取失败：这一页现做现讲——
                    # 先把页面亮出来别让画面空等，再补上讲解。
                    await self._present_slide(deck, index, speak=True)
                else:
                    await self._present_prepared(deck, index, prepared)
                    prepared = None
                # 学生正在听这一页，后台把下一页备好（话术 + TTS + 数字人渲染并行进行）。
                # 这一步是「同步」的关键：翻页时声音已经就绪，不会出现
                # 「画面已经翻到下一页、老师愣几秒才开口」的空窗。
                if index + 1 < total:
                    prepare_task = asyncio.create_task(
                        # 整页语音先渲染好攥在手里，翻页那一刻再连页图一起下发（页音同到）
                        self._prepare_slide(deck, index + 1, speak=True)
                    )
                # 配速：等前端把这一页的音频播完再翻下一页。
                # 带上本轮 epoch：本页讲解（LLM/TTS/渲染，实测首屏 20s+）期间学生若已点了
                # 「下一页 / 暂停」，那次打断必须能在等待**开始那一刻**就生效，
                # 否则要一路等到前端播完或 5 分钟超时（见 _await_page_played 的说明）。
                await self._await_page_played(index, epoch=epoch)
                if prepare_task is not None:
                    task, prepare_task = prepare_task, None
                    if self.session.auto_playing and self.session.speech_epoch == epoch:
                        try:
                            prepared = await task
                        except Exception as err:  # noqa: BLE001 - 预取失败退回「现做现讲」
                            log.warning("课件预取下一页失败，改为现做现讲：{}", err)
                            prepared = None
                    else:
                        await self._reap_task(task)   # 学生插话/暂停：作废预取
            else:
                finished = True
        finally:
            if prepare_task is not None:
                await self._reap_task(prepare_task)
            self.session.auto_playing = False
        if finished:
            log.info("课件已讲完（共 {} 页）", total)
            self.session.push_event("slide_play_state", {"playing": False, "finished": True})
        elif not self.session.auto_playing:
            # 被学生插话/暂停中止（epoch 过期或 auto_playing 归零）：同步前端按钮状态，
            # 否则「暂停讲解」按钮会一直停在讲解态（实际已停）。
            # 翻页/重讲场景旧任务被 _schedule_slide_playback 取消（CancelledError 直接传播，
            # 不会走到这里），新任务自会推 playing=True，两边不打架。重复推送幂等无害。
            log.info("课件连讲中止（讲到第 {}/{} 页）", self.session.page_index + 1, total)
            self.session.push_event("slide_play_state", {"playing": False, "finished": False})

    @staticmethod
    async def _reap_task(task: "asyncio.Task[Any]") -> None:
        """回收被作废的后台预取任务（取消 + 取回异常），避免「Task exception never retrieved」告警。

        用 ``gather(return_exceptions=True)`` 而不是 ``try/except CancelledError``：
        后者分不清「子任务被取消」和「**当前协程自己**被取消」（如学生断连时的
        ``runner.cancel()``），会把取消信号一并吞掉，让本回合在已取消的状态下
        继续走完剩余分支、甚至继续下发事件。
        """
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    # ---------------------------------------------------- 连讲配速（前端播完再翻页）
    def _slide_ready_index(self) -> int:
        """前端已回报「播完」的最大页序（尚无回报返回 -1）。

        ⚠️ 不能写成 ``int(raw or -1)``：页序 **0** 是合法值，`0 or -1` 会变成 -1，
        回报就永远匹配不上，连讲只能干等超时（表现为「每页卡一个超时时长」）。
        """
        raw = self.session.scratch.get("slide_ready_index")
        try:
            return int(raw)
        except (TypeError, ValueError):
            return -1

    def mark_slide_ready(self, index: int) -> None:
        """前端回报「某一页的音频已播完」（连讲配速用，见 ``_await_page_played``）。"""
        self.session.scratch["slide_ready_index"] = max(self._slide_ready_index(), int(index))

    async def _await_page_played(self, index: int, *, epoch: int | None = None) -> None:
        """连讲配速：通知前端「本页音频已全部下发」，等它播完再翻下一页。

        等待期间若学生插话 / 暂停 / 翻页（``speech_epoch`` 变化或 ``auto_playing`` 归零）
        立即返回；``COURSEWARE_PAGE_WAIT_SECONDS`` 为上限，超时也继续，
        保证连讲不会因为前端异常而卡死。

        ``epoch`` 由 :meth:`_play_slides` 传入**开始连讲时**的轮次号，必须用它而不是
        在入口重取 ``self.session.speech_epoch``：一页话术要经过 LLM + TTS + 数字人渲染
        （实测首屏 20 秒以上），学生完全可以在这段时间里点「下一页」——
        那时 ``speech_epoch`` 已经自增过了，入口重取的「当前值」等于新值，比对不出差异，
        等待就会一路耗到超时（默认 5 分钟）。传进来的旧值才比得出来。
        """
        self.session.push_event("slide_audio_end", {"index": index})
        limit = float(getattr(settings, "courseware_page_wait_seconds", 90.0) or 0.0)
        if limit <= 0:
            return
        if self._slide_ready_index() >= index:
            return
        epoch = self.session.speech_epoch if epoch is None else epoch
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            if self._slide_ready_index() >= index:
                return
            if self.session.speech_epoch != epoch or not self.session.auto_playing:
                return                      # 学生插话 / 暂停：立即作废等待
            await asyncio.sleep(0.3)
        log.info("连讲等待前端播完第 {} 页超时（{}s），继续翻页", index + 1, int(limit))

    @staticmethod
    def _page_budget(slide: Any) -> float:
        """单页讲解时长预算（秒）：按**这一页的内容量**给，不按「剩余时间 ÷ 剩余页数」。

        为什么按内容量而不是按剩余时间反推：后者在页数多的课件上会把每页压到
        几十个字，学生会看到「赶着翻页」；而页面本身的信息量才是决定该讲多久的依据。
        连讲整份课件时逐页调用本方法，前端单页预览不传（0 = 不限额）。
        """
        try:
            from core.courseware.models import LIGHT_WEIGHT, RICH_WEIGHT, slide_weight  # noqa: PLC0415

            weight = slide_weight(slide)
        except Exception:  # noqa: BLE001 - 取不到权重就按「普通内容页」处理
            return 60.0
        if weight < LIGHT_WEIGHT:
            return 8.0          # 导航页（章节封面/目录/谢谢页）：一两句话就翻
        if weight >= RICH_WEIGHT:
            return 150.0        # 重点页：留足字数讲透
        return 60.0

    async def _explain_slide(self, deck: Any, index: int, *, budget_sec: float = 0.0) -> AIResponse:
        """讲课件某一页的唯一入口：**讲课链路**（Supervisor → LecturerAgent → 白板）。

        协作与回退策略已图化，节点与连接顺序见 ``agents/lecturer/nodes.py`` 的
        ``lecturer.slide``：

        1. ``request``：记一条 ``request`` 到协作链 —— 谁在第几页请求了哪个 agent 可追溯；
        2. ``explain``：交给 :attr:`lecturer`（它有独立的课件契约，不借答疑的 JSON 字段）；
        3. ``fallback``：它抛异常 -> 回退到 :meth:`TeacherAgent.explain_slide`
           （等价实现，共用同一 llm 与白板工厂），并记一条 ``fallback``；
        4. ``degrade``：回退也失败 -> 记 ``error`` 并返回一句兜底话术。

        为什么第 4 步不直接抛：本方法被连讲流程调用（``_play_slides``），
        异常会打断**整节课**。宁可这一页少讲一句，也不能让学生对着白板干等。
        """
        run = await lecturer_nodes.build_slide_graph(self).run(
            {"deck": deck, "index": index, "budget_sec": budget_sec}
        )
        response: AIResponse | None = run.context.get("response")
        if response is None:      # degrade 节点已兜底；这里只是不返回 None 的保险
            return AIResponse(
                speech=lecturer_nodes.DEGRADED_SPEECH,
                meta={"slide": {"index": index}, "mood": "speaking", "degraded": True},
            )
        return response

    async def _prepare_slide(
        self, deck: Any, index: int, *, speak: bool = True
    ) -> tuple[Any, AIResponse, list[tuple[str, dict[str, Any]]]] | None:
        """**备好**某一页的讲解（话术 + 标注 + 语音事件），但先不下发。

        老流程是「先换页 → 再等 LLM 写话术（几秒）→ 再等 TTS/数字人渲染（几秒）」，
        学生看到新的一页之后要干等好几秒才听到声音，而且开口说的还是上一页的收尾
        —— 这就是「讲解和 PPT 不同步」的根源。

        这里把整页语音**渲染好先攥在手里**，交给 :meth:`_present_prepared` 在翻页
        那一刻和页图一起下发；连讲时它是在**上一页还在播**的时候跑的（见
        :meth:`_play_slides`），所以那几秒的准备时间学生完全感知不到。

        副作用：本方法**不改会话状态**（页序 / deck_id 都在下发那一步设置）——
        这样预取失败或学生插话时直接丢掉即可，不会留下「页序跳了却没讲」的脏状态。
        """
        slide = deck.page(index)
        if slide is None:
            return None
        response = AIResponse(speech="")
        events: list[tuple[str, dict[str, Any]]] = []
        if speak:
            # 逐页预算：按这一页的内容量给讲解长度（导航页一两句 / 重点页讲透）
            response = await self._explain_slide(deck, index, budget_sec=self._page_budget(slide))
            events = await self._prepare_speech_events(response)
        return slide, response, events

    def _present_slide_page(self, deck: Any, index: int, slide: Any) -> None:
        """换页：把这一页的页图推上白板，并记下当前页序。

        前端据此换背景图并清掉上一页的标注；``slide_state`` 驱动页码栏。
        """
        from core.courseware.loader import get_store

        image_url = get_store().page_image_url(deck.deck_id, slide.image)
        self.session.deck_id = deck.deck_id
        self.session.page_index = index
        # 当前页标题（供出题过渡语「我们刚讲了…」引用，见 _quiz_lead）
        self.session.scratch["slide_page_title"] = str(getattr(slide, "title", "") or "")
        self._push_board([
            self.teacher.board.slide_page(
                deck.deck_id, index,
                image_url=image_url, title=slide.title, total=deck.total,
                text=slide.text,
            )
        ])
        self.session.push_event("slide_state", {
            "deck_id": deck.deck_id, "title": deck.title,
            "index": index, "total": deck.total,
            "page_title": slide.title,
        })

    async def _present_prepared(
        self,
        deck: Any,
        index: int,
        prepared: tuple[Any, AIResponse, list[tuple[str, dict[str, Any]]]],
    ) -> AIResponse:
        """把备好的一页**一次性**呈现：换页 → 标注 → 文本与语音。

        顺序很重要：**先上页、后上标注、最后出声**。学生先看到页面，再看到老师圈哪里，
        然后声音正好接上——页图和语音几乎同时到，听到的第一句就对得上眼前的画面。
        连讲翻页走这条路径（语音已在上一页播放期间渲染好）。
        """
        # 节点顺序见 agents/supervisor/nodes.py 的 supervisor.slide_prepared：
        # present_page → annotate → speak → flush_events（页图、标注、字幕、语音备货一起到）
        await supervisor_nodes.build_presented_graph(self).run({
            "deck": deck, "index": index, "prepared": prepared,
        })
        # 断点续讲：整页事件已在下发前备好，直接整份入账（续讲时从断点补发剩余）
        self._open_slide_event_log(deck.deck_id, index, events=prepared[2])
        return prepared[1]

    async def _present_slide(
        self, deck: Any, index: int, *, speak: bool = True
    ) -> AIResponse | None:
        """呈现某一页：**先亮页面，再讲解**（首屏 / 手动翻页 / 预览走这里）。

        与「学生自己点下一页」的预期一致：点一下立刻看到页面，老师随即开口；
        语音边合成边下发，不必等整页渲染完——反正这一页没有上一页可用来
        提前准备（首屏是新开课件，手动翻页是学生临时打断）。

        ``speak=False``：只上页面不讲解（预览某页）。

        节点与连接顺序见 ``agents/supervisor/nodes.py`` 的 ``supervisor.slide``：
        ``present_page`` → ``explain`` → ``annotate`` → ``speak`` → ``push_audio``
        （``with_speech=False`` 时在 ``explain`` 节点收工）。
        """
        if speak:
            # 断点续讲：流式路径边讲边记（push_audio 节点带 record=True），
            # 学生暂停/插话后再点「继续讲」就能从断点补发；不讲解（预览）则不记。
            self._open_slide_event_log(deck.deck_id, index)
        run = await supervisor_nodes.build_slide_graph(self).run({
            "deck": deck, "index": index, "with_speech": speak,
        })
        return run.context.get("response")

    # ================================================================ 内部
    def _build_answer_graph(self) -> TaskGraph:
        """答疑流水线图：节点与连接顺序见 ``agents/teacher/nodes.py`` 的 ``ANSWER_SPECS``
        （``retrieve`` → ``generate`` → ``validate`` → ``degrade`` → ``solve``）。

        与 ``core/langgraph_flow.py`` 的 LangGraph 版答疑子图节点同名同序，
        行为由 ``tests/test_graph_parity.py`` 强绑定一致。
        """
        return teacher_nodes.build_answer_graph(self)

    async def _run_answer_pipeline(
        self,
        turn: StudentTurn,
        intent: IntentType,
        *,
        ocr_text: str = "",
        hint_level: HintLevel | None = None,
    ) -> AIResponse:
        # hint_level 已由调用方解析时直接复用（避免作业提示层级被重复升级）
        if hint_level is None:
            hint_level = self._resolve_hint_level(intent)
        ctx: dict[str, Any] = {
            "turn": turn,
            "intent": intent,
            "hint_level": hint_level,
            "ocr_text": ocr_text,
            # 上下文 = 意图转交说明（闸门给的）+ 本回合执行计划（模型自己定的安排）
            "notes": self._intent_notes(turn) + await self._plan_notes(turn),
        }
        if settings.orchestrator == "langgraph":
            if LANGGRAPH_AVAILABLE:
                return await run_answer_graph(self, ctx)
            _warn_langgraph_missing()
        await self._build_answer_graph().run(ctx)
        response: AIResponse = ctx.get("response") or AIResponse(speech="这道题我们再想想看。")
        return response

    # ---------------------------------------------------------------- 低延迟流式答疑
    @traced(desc="流式答疑")
    async def _turn_teach_streaming(
        self, turn: StudentTurn, intent: IntentType, *,
        hint_level: HintLevel | None, ocr_text: str = "",
    ) -> AIResponse:
        """流式答疑：LLM 话术边生成边逐句 TTS/渲染，白板等结构化内容并行生成。

        双路并行（均含同一份 RAG 上下文）：
        * 话术路经 ``sentence_queue`` 逐句到达 -> 立刻送 TTS/数字人（首句延迟 ≈ 首句生成时间）；
        * 结构路 ``teacher.answer`` 完成后再下发白板与聊天文本，并做校验/验算收尾。
        """
        notes: list[str] = self._intent_notes(turn) + await self._plan_notes(turn)
        if self.retriever is not None and turn.text:
            try:
                notes += await self.retriever.retrieve(turn.text)
            except Exception as err:  # noqa: BLE001 - 检索失败不影响答疑
                log.warning("retriever failed: {}", err)

        queue: asyncio.Queue[str | None] = asyncio.Queue()
        # 结构路：完整答疑管线（RAG -> JSON -> 校验降级 -> 验算）原样保留
        response_task = asyncio.create_task(self._run_answer_pipeline(
            turn, intent, ocr_text=ocr_text, hint_level=hint_level,
        ))
        # 话术路：LLM 流式生成纯讲解正文，逐句入队
        speech_task = asyncio.create_task(self.teacher.speech_stream(
            turn, self.session, hint_level=hint_level, context_notes=notes, sentence_queue=queue,
        ))

        async def _board_ready() -> AIResponse:
            """白板先行：结构路一生成完立刻上白板，不等逐句渲染完成。

            之前白板要等全部句子渲染下发后才推送，导致「老师已经在讲、
            白板却慢好几秒」；现在两路并行，白板与讲解音视频几乎同步出现。
            """
            resp = await response_task
            streamed_text = await speech_task     # 话术已全部生成（渲染仍在后台进行）
            if streamed_text:
                resp.speech = streamed_text
            self._finalize_streamed_response(resp, hint_level)
            if resp.board_ops:
                self._push_board(resp.board_ops)
            return resp

        board_task = asyncio.create_task(_board_ready())

        async def _sentences() -> AsyncIterator[str]:
            while True:
                item = await queue.get()
                if item is None:
                    return
                yield item

        # 过渡句：结构路（白板）要完整跑完 RAG -> 结构化 JSON -> 校验 -> 验算，
        # 通常比话术路的首句晚几秒。先用一句简短开场白顶上，把「提问后的静默」
        # 变成「老师先应一声再开讲」，正式讲解与白板随后自然跟上。
        # 该过渡句的 TTS + 渲染放到后台任务，与「正式讲解首句生成」并行，削减首响延迟：
        # 原本要等 opener 整段渲染完（数秒）才开始消费句子流，现在这段等待与 LLM 首句生成重叠。
        opener_task: asyncio.Task[dict[str, Any] | None] | None = None
        if settings.digital_human_enable and self.avatar.produces_video:
            opener_task = asyncio.create_task(
                self._render_opener_sentence(self._opener_sentence())
            )

        sentences_iter = _sentences()
        if opener_task is not None:
            try:
                opener_result = await opener_task
                if opener_result is not None:
                    self.session.push_event("ai_avatar", opener_result)
            except Exception as err:  # noqa: BLE001 - 过渡句失败不影响正式答疑
                log.warning("opener sentence skipped: {}", err)

        speak_ok = await self._push_avatar_video(sentences_iter)
        response = await board_task
        if speak_ok:
            # 白板已在 _board_ready 先行下发；这里只补聊天文本与教师状态
            self._speak(response, push_board=False)
        else:
            # 渲染不可用：退回「纯语音 + 静态待机画面」。
            # 绝不能什么都不发——否则学生会等半天一点声音都没有（比慢更糟）。
            self._speak(response, push_board=False)
            log.warning("avatar video unavailable; 退回纯语音播报")
            await self._push_audio(response)
            self.session.push_event(
                "chat_message",
                {"role": "ai", "text": "（数字人引擎正在加载，先用语音讲解～）"},
            )
        return response

    _OPENERS: tuple[str, ...] = (
        "好，这道题我们一起来看一下。",
        "嗯，这个问题问得好，我们一步步来分析。",
        "来，先别急，我们把题目读一遍，找出已知条件。",
        "好，我来帮你理一理这道题的思路。",
    )

    def _opener_sentence(self) -> str:
        """过渡开场白：答疑正式开始前先应一声，掩盖结构化输出的生成延迟。"""
        import random

        return random.choice(self._OPENERS)

    async def _render_opener_sentence(self, opener: str) -> dict[str, Any] | None:
        """合成并渲染过渡句；无音频或渲染失败返回 None（静默跳过，不影响正式答疑）。"""
        try:
            opener_audio = await self.tts.synthesize(opener, teacher_id=self._teacher_id())
        except Exception as err:  # noqa: BLE001 - TTS 失败仅跳过过渡句
            log.warning("opener tts failed: {}", err)
            return None
        if not opener_audio:
            return None
        return await self._render_avatar_sentence(opener_audio, opener, None)

    def _finalize_streamed_response(self, response: AIResponse, hint_level: HintLevel | None) -> None:
        """流式答疑收尾：与答疑管线保持一致的校验降级 + sympy 验算。

        收尾逻辑**只有一份实现**（``core/postprocess.py``）：校验 -> 只丢非法白板指令 -> 独立验算。
        这样「改一处漏两处」不会发生 —— 该方法与答疑管线的 validate/solve 节点共用同一函数。
        """
        postprocess_response(
            response, hint_level=hint_level, board=self.teacher.board, validator=self.validator
        )

    # ================================================================ Skill 自我更新
    def _schedule_skill_reflection(self, turn: StudentTurn, response: AIResponse) -> None:
        """答疑结束后**异步**复盘，把可复用的教学经验沉淀进该老师的 skill。

        为什么必须异步（fire-and-forget）：复盘要再调一次 LLM（几百毫秒到数秒），
        串在答疑尾部会让学生等下一句话时莫名变慢。它只写经验文件、不影响本回合输出，
        失败也只记日志 —— **绝不能因为「学习」把「上课」拖慢或搞挂**。
        """
        if not settings.skill_self_update_enable:
            return
        teacher_key = self._teacher_id()
        if not teacher_key or not str(response.speech or "").strip():
            return
        interval = float(settings.skill_reflect_min_interval_seconds or 0.0)
        if interval > 0:
            now = time.monotonic()
            last = float(self.session.scratch.get("last_reflect_at") or 0.0)
            if now - last < interval:
                return
            self.session.scratch["last_reflect_at"] = now
        try:
            task = asyncio.create_task(self._reflect_skill(teacher_key, turn, response))
        except RuntimeError:
            return                      # 没有运行中的事件循环（同步直调场景）
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _reflect_skill(
        self, teacher_key: str, turn: StudentTurn, response: AIResponse
    ) -> None:
        """复盘本体：提炼经验并落盘（异常一律吞掉，只留日志）。"""
        try:
            items = await self._skill_reflector().reflect(
                teacher_key=teacher_key,
                subject=self.session.subject,
                question=turn.text or "",
                answer=response.speech or "",
                points=list(response.knowledge_points or []),
                pitfalls=list(response.pitfalls or []),
            )
            if items:
                log.info(
                    "教师 Skill 自我更新：{} 沉淀 {} 条经验（{}）",
                    teacher_key, len(items), "；".join(i.text[:16] for i in items),
                )
        except Exception as err:  # noqa: BLE001 - 复盘失败不影响课堂
            log.warning("教师经验复盘失败（忽略）：{}", err)

    def _skill_reflector(self) -> SkillReflector:
        if self.skill_reflector is None:
            self.skill_reflector = SkillReflector()
        return self.skill_reflector

    # ================================================================ 学情记忆
    def _schedule_memory_distill(self, turn: StudentTurn, response: AIResponse) -> None:
        """答疑结束后**异步**蒸馏这位学生的薄弱点（默认关闭，``MEMORY_DISTILL_ENABLE``）。

        与教师经验复盘同源的理由：这是一次额外的 LLM 调用，串在答疑尾部只会让学生
        等下一句话时莫名变慢。它只写记忆文件、不影响本回合输出，失败也只记日志。
        """
        if not settings.memory_enable or not settings.memory_distill_enable:
            return
        if not str(response.speech or "").strip() or not (turn.text or "").strip():
            return
        try:
            task = asyncio.create_task(
                self.memory.distill(
                    self.session, question=turn.text, answer=response.speech
                )
            )
        except RuntimeError:
            return                      # 没有运行中的事件循环（同步直调场景）
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    def _record_mistake_memory(self, question: str, expected: str, student_answer: str) -> None:
        """判题判错：把这道题记进学情记忆（零 LLM 成本，写入失败不影响判题反馈）。"""
        if not settings.memory_enable:
            return
        try:
            self.memory.record_mistake(
                self.session,
                question=question,
                expected=expected,
                student_answer=student_answer,
            )
        except Exception as err:  # noqa: BLE001 - 记忆写入是旁路，绝不打断判题
            log.warning("写入学情记忆失败（忽略）：{}", err)

    def _resolve_hint_level(self, intent: IntentType) -> HintLevel | None:
        """作业模式下按题目请求下一层提示。"""
        hw = self.session.scratch.get("homework")
        if hw is None:
            return None
        level, _ = hw.request_hint(escalate=True)
        return level

    def _handle_control(self, text: str) -> None:
        """设备 / 白板控制指令：节点与命中顺序见 ``agents/student/nodes.py`` 的
        ``student.control``（``camera_off`` / ``camera_on`` / ``mic_off`` / ``mic_on`` /
        ``clear_board``，先匹配先执行，命中即收工）。

        保持**同步**：它会被同步调用栈直接调用（LangGraph 主图的节点、
        ``_turn_multi_intent``），因此跑同步图 :meth:`core.graph.TaskGraph.run_sync`。
        """
        student_nodes.build_control_graph(self).run_sync({"text": text})

    # ---------------------------------------------------- 课件续讲/重讲（学生说「继续」）
    #: 「接着讲」类指令关键词（≥3 字的明确说法）
    _LECTURE_RESUME_HINTS = ("接着讲", "往下讲", "继续讲", "讲下去", "继续吧")
    #: 「重讲」类指令关键词：重讲当前页
    _LECTURE_REPLAY_HINTS = ("重新讲", "再讲一遍", "重讲", "再讲一次")

    def lecture_command(self, text: str) -> str:
        """识别「继续讲 / 重讲」类课堂指令：返回 ``"resume"`` / ``"replay"`` / ``""``。

        * 仅当**课件已打开**时生效（``deck_id`` 非空）——其余场景这些词回落常规链路；
        * 只认**短指令**（清理标点后 ≤14 字；纯「继续」要求 ≤5 字且是句子主干），
          长句不劫持（如「继续讲一下这道题为什么选 B……」应是提问，不是指令）；
        * 「继续做题」这类不以「继续」开头/结尾的短句也不劫持。

        判定在 ``route`` 节点最优先执行（见 ``agents/supervisor/nodes.py`` 与
        ``core/turn_graph.py`` 两套编排同名同序），避免「请继续」被判成寒暄吞掉。
        """
        if not self.session.deck_id:
            return ""
        t = re.sub(r"[\s，。！？!?~～、,.;；:：\"'“”‘’]", "", str(text or ""))
        if not t or len(t) > 14:
            return ""
        if any(h in t for h in self._LECTURE_REPLAY_HINTS):
            return "replay"
        if any(h in t for h in self._LECTURE_RESUME_HINTS):
            return "resume"
        # 最短指令：「继续」/「请继续」/「那我们继续」……——把语气词/衬词去掉后
        # **只剩「继续」**才算，避免「继续做题」这类被误判成继续讲课。
        core = t
        for filler in ("请", "那", "我们", "咱们", "就", "可以", "来",
                       "啊", "吧", "呀", "哦", "呢", "嘛", "了"):
            core = core.replace(filler, "")
        if len(t) <= 8 and core == "继续":
            return "resume"
        return ""

    async def resume_lecture(self, text: str) -> bool:
        """执行课件续讲/重讲（``route`` 判定为 ``resume_lecture`` 分支后调用）。

        * ``resume`` → :meth:`handle_slide_play` 的断点续讲（接着停下的地方讲）；
        * ``replay`` → 重讲当前页（学生说「再讲一遍」）。
        返回 False = 不是指令（调用方按原链路处理，正常不会走到）。
        """
        mode = self.lecture_command(text)
        if not mode:
            return False
        if mode == "replay":
            await self.handle_slide_play(self.session.page_index)
        else:
            await self.handle_slide_play(-1)
        return True

    def _speak(self, response: AIResponse, *, push_board: bool = True) -> AIResponse:
        """把 AI 产物同步到白板、消息与客户端（语音由客户端 TTS 播放）。

        ``push_board=False``：白板已在别处（如流式答疑的 _board_ready）先行下发，
        这里只补聊天文本与教师状态，避免白板指令重复推送。

        **安全网关·输出侧在这里收口**：话术 / 板书文字 / 分步解析 / 追问四类内容
        全部经过滤后才可能到达学生的眼睛与耳朵（TTS 的合成源就是这里的 speech）。
        """
        self.safety.screen_response(response)
        if push_board and response.board_ops:
            self._push_board(response.board_ops)
        # 聊天记录与字幕要纯文字：重点标记只属于白板（白板指令里的标记原样保留）。
        response.speech = strip_emphasis(response.speech)
        if response.speech:
            self.session.add_message(
                ChatMessage(role="ai", text=response.speech, board_ops=response.board_ops)
            )
            payload: dict[str, Any] = {"text": response.speech}
            mood = response.meta.get("mood")
            if is_valid_mood(mood):
                # 教师形象状态（打招呼/讲解/思考/表扬/等待）；非法值不下发，前端保持原状态
                payload["mood"] = mood
            elif mood:
                log.debug("ignored unknown teacher mood: {}", mood)
            self.session.push_event("ai_speech", payload)
        return response

    @traced(desc="播报")
    async def _speak_async(self, response: AIResponse, *, push_board: bool = True) -> AIResponse:
        """异步播报：先同步白板/文本，再合成真实 TTS 音频下发（数字人口型用）。

        ``push_board=False``：白板已提前单独下发过（避免重复推送与清屏重绘）。
        """
        self._speak(response, push_board=push_board)
        await self._push_audio(response)
        return response

    @staticmethod
    def _spoken_text(response: AIResponse) -> str:
        """播报文本 = 主话术 + 白板分步解析（让学生听到白板上的内容）。

        每段都先过 :func:`models.tts.sanitize_for_tts` 与
        :func:`agents.shared.emphasis.strip_emphasis`：**只影响「念什么」**——
        填空题的连续下划线、错误标记 ``❌``、重点标记 ``**`` 都不朗读，
        画面上的文字保持不变（白板指令里的 ``**`` 原样保留，供前端标红）。
        """
        from models.tts import sanitize_for_tts  # noqa: PLC0415 - 局部导入避免循环依赖

        def _clean(value: str) -> str:
            # 先摘重点标记再清 TTS 噪音：两步都只改「念出来的文字」。
            # 顺序无所谓，但都要做 —— 漏掉前者学生就会听到「星号星号」。
            text = strip_emphasis(sanitize_for_tts(str(value or "")))
            return text.strip().rstrip("。！？!?；;、")

        segments: list[str] = []
        if _clean(response.speech):
            segments.append(_clean(response.speech))
        for i, step in enumerate(response.meta.get("steps") or [], 1):
            text = _clean(str(step))
            if text:
                segments.append(f"第{i}步，{text}")
        followup = response.meta.get("followup") or {}
        question = _clean(str(followup.get("question") or ""))
        if question:
            segments.append(f"我来考考你：{question}")
        return ("。".join(segments) + "。") if segments else ""

    async def _push_audio(
        self, response: AIResponse, *, allow_avatar: bool = True, record: bool = False
    ) -> None:
        """合成 TTS 音频并下发 ``ai_audio``；不可用时下发 ``ai_speak_text`` 让前端回退朗读。

        ``allow_avatar=False``：只走纯语音（数字人降级后不再尝试），见下方递归保护。
        ``record=True``：本段属于课件页讲解 —— 逐条语音事件记入断点日志
        （继续讲时从断点补发）；答疑等其它播报不传，避免污染断点。
        """
        speech = self._spoken_text(response)
        if not speech:
            return
        # 一段播报开始：让 TTS 重新锁定音色（保证整段用同一种声音）
        begin_utterance = getattr(self.tts, "begin_utterance", None)
        if callable(begin_utterance):
            begin_utterance(self._teacher_id())
        if not settings.digital_human_enable:
            self.session.push_event("ai_speak_text", {"text": speech})
            if record:
                self._record_slide_event("ai_speak_text", {"text": speech})
            return
        gesture = response.meta.get("gesture")        # 手势基元名（可为 None = 纯讲解）
        if allow_avatar and self.avatar.produces_video:
            if await self._push_avatar_video(speech, gesture, record=record):
                return
            # 渲染不可用：退回「纯语音 + 静态待机画面」，绝不能静音。
            # ⚠️ 必须带 allow_avatar=False 降级：否则本方法会再进数字人分支、再失败，
            #    无限递归（会话卡死，日志每几秒刷一条「avatar video unavailable」）。
            log.warning("avatar video unavailable; 退回纯语音播报")
            self.session.push_event(
                "chat_message",
                {"role": "ai", "text": "（数字人引擎正在加载，先用语音讲解～）"},
            )
            await self._push_audio(response, allow_avatar=False, record=record)
            return
        try:
            audio = await self.tts.synthesize(speech, teacher_id=self._teacher_id())
        except Exception as err:  # noqa: BLE001 - TTS 失败不影响文本播报
            log.warning("tts synthesize failed: {}", err)
            audio = b""
        if not audio:
            # MockTTS / 空音频：让前端退回浏览器朗读
            self.session.push_event("ai_speak_text", {"text": speech})
            if record:
                self._record_slide_event("ai_speak_text", {"text": speech})
            return
        payload: dict[str, Any] = {
            "audio": base64.b64encode(audio).decode("ascii"),
            "mime": "audio/mpeg",
            "text": speech,
        }
        if settings.digital_human_viseme_enable:
            visemes = build_viseme_timeline(speech)
            if visemes:
                payload["visemes"] = visemes
        self.session.push_event("ai_audio", payload)
        if record:
            self._record_slide_event("ai_audio", dict(payload))

    async def _prepare_speech_events(
        self, response: AIResponse
    ) -> list[tuple[str, dict[str, Any]]]:
        """合成这一段的音视频，但**先不下发**，只返回待下发的事件列表。

        与 :meth:`_push_audio` 的区别：这里只「备货」，让调用方能在翻页那一刻
        把语音和页图一起下发，保证画面与声音同时到（见 :meth:`_prepare_slide`）。
        返回空列表 = 这段没有可播报内容。

        降级链与 :meth:`_push_audio` 完全一致（数字人 → 纯语音 → 前端朗读），
        只是每一步都产出「事件」而不是直接推送，所以不会静音。
        """
        speech = self._spoken_text(response)
        if not speech:
            return []
        if not settings.digital_human_enable:
            return [("ai_speak_text", {"text": speech})]
        gesture = response.meta.get("gesture")
        if self.avatar.produces_video:
            sink: list[tuple[str, dict[str, Any]]] = []
            if await self._push_avatar_video(speech, gesture, sink=sink):
                return sink
            log.warning("avatar video unavailable; 退回纯语音播报")
            events: list[tuple[str, dict[str, Any]]] = [
                ("chat_message", {"role": "ai", "text": "（数字人引擎正在加载，先用语音讲解～）"})
            ]
            events.extend(await self._prepare_audio_only(speech))
            return events
        return await self._prepare_audio_only(speech)

    async def _prepare_audio_only(self, speech: str) -> list[tuple[str, dict[str, Any]]]:
        """只合成 TTS 语音（照片木偶链路 / 数字人不可用时的降级），返回待下发事件。"""
        try:
            audio = await self.tts.synthesize(speech, teacher_id=self._teacher_id())
        except Exception as err:  # noqa: BLE001 - TTS 失败不影响文本播报
            log.warning("tts synthesize failed: {}", err)
            audio = b""
        if not audio:
            # MockTTS / 空音频：让前端退回浏览器朗读
            return [("ai_speak_text", {"text": speech})]
        payload: dict[str, Any] = {
            "audio": base64.b64encode(audio).decode("ascii"),
            "mime": "audio/mpeg",
            "text": speech,
        }
        if settings.digital_human_viseme_enable:
            visemes = build_viseme_timeline(speech)
            if visemes:
                payload["visemes"] = visemes
        return [("ai_audio", payload)]

    def _teacher_profile(self) -> "TeacherProfile | None":
        """当前会话生效的老师档案：优先学生选中的（同科多老师），否则学科默认。"""
        return teacher_for_subject(self.session.subject, getattr(self.session, "teacher", ""))

    def _teacher_id(self) -> str:
        """当前会话的教师标识（= 教师档案 key）；该学科没有独立档案时返回空串。

        同科多老师时由 ``session.teacher`` 决定（学生在选择界面挑的那位），
        为空则回退学科默认老师——旧的「一科一师」部署行为完全不变。
        空串让渲染引擎沿用默认底片（向后兼容旧的单老师部署）。
        """
        profile = self._teacher_profile()
        return profile.key if profile else ""

    async def _render_avatar_sentence(
        self, audio_chunk: bytes, sentence: str, gesture: str | None
    ) -> dict[str, Any] | None:
        """渲染单句数字人视频；失败时短暂重试，避免引擎重启/加载中被误判为不可用。

        返回引擎结果（``mode == "video"``）；重试仍失败返回 ``None``。
        """
        teacher_id = self._teacher_id()
        if not teacher_id:
            # 还没确定学科（如刚进页面的打招呼）时引擎没有对应底片，渲染必然 404；
            # 直接跳过，避免「重试 3 次 + 每次等 1.5s」白白拖慢首响。
            log.info("跳过数字人渲染：本回合尚未确定学科（teacher_id 为空）")
            return None

        attempts = max(1, settings.avatar_render_attempts)
        last = ""
        for i in range(attempts):
            try:
                result = await self.avatar.render(
                    audio_chunk, text=sentence, gesture=gesture,
                    teacher_id=teacher_id,
                )
                if result.get("mode") == "video":
                    return result
                last = f"mode={result.get('mode')}"
            except Exception as err:  # noqa: BLE001 - 网络/引擎异常均可重试
                last = str(err)
                if "404" in last:
                    # 底片缺失这类确定性错误：重试无意义，立刻放弃（省掉数秒重试等待）
                    log.warning("avatar render 404（底片缺失），放弃本句渲染：{}", last)
                    return None
            log.warning("avatar render attempt {}/{} failed: {}", i + 1, attempts, last)
            if i + 1 < attempts:
                await asyncio.sleep(1.5)
        return None

    async def _render_avatar_sentence_padded(
        self, audio_chunk: bytes, sentence: str, gesture: str | None, *, tail: bool
    ) -> dict[str, Any] | None:
        """渲染单句：按位置补尾部静音 + 视频落盘为 URL（避免大 base64 卡顿）。

        补静音分两种（口型衔接的关键）：
        * ``tail=True``（整段话的**最后一句**）：补 ``avatar_tail_silence_seconds``，
          让引擎以「闭嘴稳定帧」收尾，讲完切回待机才能平滑交叉淡化；
        * ``tail=False``（句间）：只补 ``avatar_sentence_gap_seconds``（默认 0 = 不补），
          上一句的末帧嘴型直接接下一句的首帧 —— 不会「每句说完都闭嘴、下句再重新张嘴」，
          一段话讲完能自然衔接到下一段话的讲解口型。

        补静音与落盘都是阻塞 IO，放到线程里做，避免卡住事件循环（也与会话循环并行）。
        """
        pad = (
            settings.avatar_tail_silence_seconds if tail
            else settings.avatar_sentence_gap_seconds
        )
        padded = audio_chunk
        if pad and pad > 0:
            padded = await asyncio.to_thread(append_tail_silence, audio_chunk, pad)
        result = await self._render_avatar_sentence(padded, sentence, gesture)
        # 视频落盘 -> stream_url：前端 <video> 按 URL 流式加载，
        # 省掉 1~3MB base64 经 WebSocket 传输后的 JSON.parse/解码卡顿。
        if (
            result is not None
            and settings.avatar_stream_url_enable
            and result.get("mode") == "video"
            and result.get("video")
        ):
            seq = int(self.session.scratch.get("avatar_seq", 0)) + 1
            self.session.scratch["avatar_seq"] = seq
            url = await asyncio.to_thread(
                save_avatar_video,
                str(result["video"]),
                f"{self.session.session_id}_{seq}",
            )
            if url:
                result.pop("video", None)
                result["stream_url"] = url
        return result

    async def _tts_sentence_stream(
        self, sentences: AsyncIterator[tuple[str, bool | None]]
    ) -> AsyncIterator[tuple[bytes, str, bool | None]]:
        """把「(句子, 是否末句) 流」转换为 TTS 的 ``(音频, 句子, 是否末句)`` 流（每句独立合成）。"""
        async for sentence, is_last in sentences:
            async for audio_chunk, text in self.tts.synthesize_stream(
                sentence, teacher_id=self._teacher_id()
            ):
                yield audio_chunk, text, is_last

    @traced(desc="数字人渲染")
    async def _push_avatar_video(
        self,
        speech: str | AsyncIterator[str],
        gesture: str | None = None,
        *,
        sink: list[tuple[str, dict[str, Any]]] | None = None,
        record: bool = False,
    ) -> bool:
        """视频数字人链路：流式 TTS 逐句合成 -> 数字人引擎逐句渲染 -> 下发 ``ai_avatar``。

        ``speech`` 可为完整话术字符串（内部按句切分），也可为句子异步迭代器
        （LLM 流式答疑：句子边生成边到达，随到随合成，进一步压低首句延迟）。

        ``sink`` 非空时**不直接下发**，而是把 ``(事件名, 载荷)`` 依次收进 ``sink``：
        供 :meth:`_prepare_speech_events` 「备货」，让调用方能等到翻页那一刻
        再和页图一起下发（见 :meth:`_prepare_slide`），做到画面与声音同时到。

        返回 True 表示整段话术均成功渲染；任一句失败立即返回 False，
        调用方回退到「ai_audio + visemes」照片木偶链路，保证播报不中断。

        声音从哪来：
        * 默认（``AVATAR_VIDEO_HAS_AUDIO=true``）视频自带音轨，口型与语音天然同步；
        * 若引擎返回的是无声视频（如用示例 mp4 空跑），置 False 则额外下发 ``ai_audio``，
          让学生至少能听到 TTS 语音。
        """
        # 一段播报开始：让 TTS 重新锁定音色，保证本段内不换引擎、不换声
        begin_utterance = getattr(self.tts, "begin_utterance", None)
        if callable(begin_utterance):
            begin_utterance(self._teacher_id())

        prefetch: "asyncio.Task[Any] | None" = None
        try:
            epoch = self.session.speech_epoch    # 学生插话会自增，旧播报随即作废
            first = True
            # 句子流：每项 = ``(音频, 文本, 是否整段最后一句)``。
            # * 完整话术（str）：先切句 -> 末句位置已知，直接带 bool，零额外等待；
            # * 流式话术（LLM 逐句产出）：末句未知 -> 先带 None，由下面的 ``held`` 流水线补上。
            if isinstance(speech, str):
                marked: AsyncIterator[tuple[str, bool | None]] = _aiter_known_last(
                    split_sentences(speech) or [speech]
                )
            else:
                marked = _aiter_unknown_last(speech)
            # 安全网关：流式话术是**唯一不经过 `_speak` 的下行文本通道**
            # （句子边生成边直接送 TTS/渲染），因此必须在这里单独收口。
            iterator = self._tts_sentence_stream(self._screen_sentence_stream(marked)).__aiter__()

            async def _next_sentence() -> Any:
                """取下一句 ``(音频, 文本, 是否末句)``；句子流结束返回 None。"""
                try:
                    return await iterator.__anext__()
                except StopAsyncIteration:
                    return None

            # 预取流水线：把「下一句的 TTS 合成」放进后台任务，与当前句的引擎渲染真正并行。
            # 之前是「渲染完一句 → 才去取下一句 → 取的时候才合成 TTS」，两段完全串行；
            # 现在当前句渲染（GPU 耗时）期间，后台已在合成下一句，句间停顿与首响一并下降。
            prefetch = asyncio.create_task(_next_sentence())
            pending: tuple[bytes, str, "asyncio.Task[dict[str, Any] | None]"] | None = None
            # 「待定末句」：流式链路里取到一句时还不知道它是不是整段最后一句 —— 先挂起，
            # 等下一句（或句子流结束）到达再渲染。这样只有**整段最后一句**才补尾部静音，
            # 句与句之间不补：上一句的末帧嘴型直接接下一句首帧，口型衔接不断层。
            held: tuple[bytes, str] | None = None

            def _emit(event: str, payload: dict[str, Any]) -> None:
                """下发一条事件；``sink`` 非空时改为收进列表（备货，不立即下发）。

                ``record=True``（页讲解）时同步记入断点日志，供「继续讲」补发剩余。
                """
                if sink is None:
                    self.session.push_event(event, payload)
                    if record:
                        self._record_slide_event(event, payload)
                else:
                    sink.append((event, payload))

            async def _flush_prev() -> bool:
                """等上一句渲染完成并下发；返回 False 表示渲染失败（需回退木偶）。"""
                nonlocal pending
                if pending is None:
                    return True
                prev_audio, prev_sentence, task = pending
                pending = None
                result = await task
                if result is None:
                    return False
                _emit("ai_avatar", result)
                if not settings.avatar_video_has_audio and prev_audio:
                    _emit(
                        "ai_audio",
                        {
                            "audio": base64.b64encode(prev_audio).decode("ascii"),
                            "mime": "audio/mpeg",
                            "text": prev_sentence,
                        },
                    )
                return True

            async def _cancel_pending() -> None:
                """作废在途的预取与渲染任务（学生插话/回退时调用），并回收异常避免告警。"""
                nonlocal pending, prefetch
                tasks: list[asyncio.Task[Any]] = []
                if prefetch is not None:
                    prefetch.cancel()
                    tasks.append(prefetch)
                    prefetch = None
                if pending is not None:
                    pending[2].cancel()
                    tasks.append(pending[2])
                    pending = None
                if tasks:
                    # gather(return_exceptions=True)：取回被取消任务的异常，又不吞掉
                    # 自身的取消信号（见 _reap_task 的说明）
                    await asyncio.gather(*tasks, return_exceptions=True)

            def _launch(audio: bytes, sentence: str, *, tail: bool) -> None:
                """启动一句渲染（不等待）：``tail`` 决定尾部是否补静音（只有末句补）。"""
                nonlocal pending, first
                # 手势只在首句叠加（点头/摇头一次即可），后续句子回到纯讲解，避免反复做同一动作
                pending = (
                    audio,
                    sentence,
                    asyncio.create_task(
                        self._render_avatar_sentence_padded(
                            audio, sentence, gesture if first else None, tail=tail
                        )
                    ),
                )
                first = False

            while True:
                # 学生插话（epoch 变化）立刻收工：旧播报不再下发，老师马上闭嘴
                if self.session.speech_epoch != epoch:
                    log.info("avatar speech superseded by new student input; stop rendering")
                    await _cancel_pending()
                    return True
                # 先收上一句：渲染是瓶颈，一渲染完就下发 —— 不等下一句的 TTS，
                # 也不等「末句判定」（视频早到多久，学生那边就早播多久）
                if not await _flush_prev():
                    await _cancel_pending()
                    return False
                item = await prefetch
                prefetch = None
                if item is None:
                    # 句子流结束：挂起的那句就是整段最后一句 -> 补尾部静音（末帧闭嘴）
                    if held is not None:
                        _launch(held[0], held[1], tail=True)
                        held = None
                    break
                audio_chunk, sentence, is_last = item
                if not audio_chunk and not settings.avatar_allow_silent_audio:
                    # 无音频（如未配置 TTS 的 Mock）：交给调用方走文本回退链路
                    await _cancel_pending()
                    return False
                # 立刻启动下一句 TTS 预取，使其与「当前句渲染」重叠
                prefetch = asyncio.create_task(_next_sentence())
                if held is not None:
                    # 后一句已到 ⇒ 挂起的那句不是末句：不补静音、直接渲染（口型接上下一句）
                    _launch(held[0], held[1], tail=False)
                    held = None
                if is_last is None and not first:
                    # 流式链路：本句是不是末句得等下一句 -> 先挂起，下轮再渲染（首句例外，见下）
                    held = (audio_chunk, sentence)
                    continue
                # 末句位置已知（完整话术）；或这是流式链路的首句 —— 首句不等后继，
                # 否则首响要多等「LLM 出下一句 + 其 TTS」。万一整段话只有一句（流式），
                # 少补一次收尾静音，由前端「末帧 + 交叉淡化」兜住即可。
                _launch(audio_chunk, sentence, tail=bool(is_last))
            if self.session.speech_epoch != epoch:
                await _cancel_pending()
                return True
            if not await _flush_prev():
                return False
        except Exception as err:  # noqa: BLE001 - 渲染失败回退木偶模式
            log.warning("avatar render failed, fallback to puppet: {}", err)
            return False
        finally:
            if prefetch is not None:
                if not prefetch.done():
                    prefetch.cancel()
                # 取回预取任务的结果/异常（避免「never retrieved」告警），
                # 但不吞掉自身被取消的信号（见 _reap_task 的说明）
                await asyncio.gather(prefetch, return_exceptions=True)
        # ★ 本段话已全部下发：明确告诉前端「后面不会再有句子了」。
        # 前端据此取消末尾的「空窗缓冲」，话音落下就平滑回待机；否则队列一空就一律
        # 缓冲 AVATAR_GAP_HOLD_MS(900ms)，那 0.9s 的画面静止就是「最后停顿不自然」的来源。
        # 用 _emit 而不是直接 push_event：课件连讲走「备货」模式（sink），
        # 终点事件必须和句子事件一起入列，否则会提前下发（页面还没显示就说讲完了）。
        _emit("ai_speak_end", {})
        return True
