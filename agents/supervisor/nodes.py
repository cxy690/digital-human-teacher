"""总调度 Agent（``Supervisor``）负责的流程节点。

总控是唯一持有会话状态的一方，它自己的流程有两条（声明见 :data:`TURN_SPECS` /
:data:`SLIDE_SPECS` / :data:`PRESENTED_SPECS`）：

| 图 | 节点（按连接顺序） | 说明 |
|---|---|---|
| ``supervisor.turn`` | ``route`` → ``answer_check`` / ``multi_intent`` / ``resume_lecture`` / ``control`` / ``chitchat`` / ``teach`` | 一次学生输入的主流程：课件续讲指令 → 待答追问优先 → 意图分流 → 按分流结果只走一个分支 |
| ``supervisor.slide`` | ``present_page`` → ``explain`` → ``annotate`` → ``speak`` → ``push_audio`` | 单页呈现（首屏 / 手动翻页 / 预览）：先亮页面，再讲解、标注、出声 |
| ``supervisor.slide_prepared`` | ``present_page`` → ``annotate`` → ``speak`` → ``flush_events`` | 预取页呈现（连讲翻页）：页图与已渲染好的语音、事件一起下发（页音同到） |

**与 LangGraph 主图的关系**：``supervisor.turn`` 是 ``Supervisor._handle_turn_builtin``
的实现，与 ``core/turn_graph.py`` 的 ``route``/``control``/``chitchat``/``teach``/
``answer_check``/``multi_intent``/``resume_lecture`` 节点同名同序，两者读同一份
``IntentDecision``，行为由 ``tests/test_graph_parity.py`` 强绑定一致。

**分支怎么表达**：自研图刻意不引入条件边 —— ``route`` 节点把选中的分支写进
``ctx["route"]``，其余节点不匹配时返回 ``{"__skip__": True}`` 跳过（且不写上下文）。
这样既保住了「只有一条分支真正执行」的语义，又让节点清单在两张编排之间可逐一对齐。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from config.constants import IntentType
from config.logging import get_logger
from core.graph import NodeSpec, TaskGraph
from core.state.models import ChatMessage

if TYPE_CHECKING:  # 仅类型标注：handler 挂在 Supervisor 自身，运行期无需导入
    from agents.supervisor import Supervisor

log = get_logger("agents.supervisor.nodes")

#: 回合主流程图名
TURN_GRAPH = "supervisor.turn"
#: 单页呈现图名
SLIDE_GRAPH = "supervisor.slide"
#: 预取页呈现图名
PRESENTED_GRAPH = "supervisor.slide_prepared"


# ------------------------------------------------------------------ 回合主流程
TURN_SPECS: tuple[NodeSpec, ...] = (
    NodeSpec(
        name="route",
        summary="待答追问优先判定 + 意图分流（写 turn.meta），并决定走哪条分支",
        impl="Supervisor._should_treat_as_answer + IntentAnalyzer.analyze_async/plan",
        inputs=("turn",),
        outputs=("intent", "route", "multi_plan"),
        deps=(),
        requires=("core.intent.IntentAnalyzer", "Supervisor.router"),
    ),
    NodeSpec(
        name="answer_check",
        summary="判题分支：学生对追问/模拟题的作答 -> 判定并反馈（不重新路由）",
        impl="Supervisor._turn_answer_check",
        inputs=("turn",),
        outputs=("response",),
        deps=("route",),
        requires=("settings.interaction_max_retries",),
    ),
    NodeSpec(
        name="multi_intent",
        summary="多意图分支：非控制诉求 ≥2 条时逐条独立作答（控制先办、只对最后一条追问）",
        impl="Supervisor._turn_multi_intent",
        inputs=("turn", "multi_plan"),
        outputs=("response",),
        deps=("route",),
        requires=("core.intent.IntentSegment",),
    ),
    NodeSpec(
        name="resume_lecture",
        summary="课件续讲/重讲分支：「继续/接着讲」从断点接着讲，「再讲一遍」重讲当前页（不出声回复）",
        impl="Supervisor.resume_lecture",
        inputs=("turn",),
        outputs=("response",),
        deps=("route",),
        requires=("Supervisor.handle_slide_play",),
    ),
    NodeSpec(
        name="control",
        summary="控制分支：当场执行设备/白板指令，不出声（返回 response=None）",
        impl="Supervisor._handle_control",
        inputs=("turn",),
        outputs=("response",),
        deps=("route",),
        requires=("agents.student.nodes（设备控制图）",),
    ),
    NodeSpec(
        name="chitchat",
        summary="闲聊分支：温柔回应，不进答疑链路",
        impl="Supervisor._turn_chitchat",
        inputs=("turn",),
        outputs=("response",),
        deps=("route",),
    ),
    NodeSpec(
        name="teach",
        summary="答疑分支：学科护栏 -> 答疑流水线 -> 播报 -> 追问/出题 -> 经验复盘",
        impl="Supervisor._turn_teach",
        inputs=("turn", "intent"),
        outputs=("response",),
        deps=("route",),
        requires=(
            "agents.teacher.nodes（答疑/判题/出题图）",
            "settings.interaction_enable / course_quiz_enable",
        ),
    ),
)


def build_turn_graph(supervisor: "Supervisor") -> TaskGraph:
    """构建回合主流程图（``Supervisor._handle_turn_builtin`` 的实现）。"""

    async def route(ctx: dict[str, Any]) -> dict[str, Any]:
        turn = ctx["turn"]
        # 0) 课件续讲/重讲指令最优先：「继续」「接着讲」「再讲一遍」……
        #    学生喊「继续」就是要接着讲课件，不能被当成寒暄/普通提问吞掉
        #    （仅课件已打开且是短指令时命中，见 Supervisor.lecture_command）。
        if supervisor.lecture_command(turn.text):  # noqa: SLF001
            turn.intent = IntentType.CONTROL
            supervisor.session.add_message(
                ChatMessage(role="student", text=turn.text, image_id=turn.image_id)
            )
            return {"intent": IntentType.CONTROL, "route": "resume_lecture"}
        # 正等待作答且输入像作答：不重新路由，直接标为作答路径
        if supervisor._should_treat_as_answer(turn):  # noqa: SLF001
            turn.intent = IntentType.TEXT_QUESTION
            supervisor.session.add_message(
                ChatMessage(role="student", text=turn.text, image_id=turn.image_id)
            )
            return {"intent": IntentType.TEXT_QUESTION, "route": "answer_check"}

        decision = await supervisor.router.analyze_async(turn)
        intent = decision.intent
        turn.intent = intent
        turn.meta["intent_decision"] = decision.to_meta()
        turn.meta["intent_notes"] = decision.delegation_notes()
        if decision.delegate:
            log.info(
                "意图分流：复杂/多意图 -> 转交下游智能体处理（{}；{}）",
                intent.value, decision.reason,
            )
        supervisor.session.add_message(
            ChatMessage(role="student", text=turn.text, image_id=turn.image_id)
        )
        # 多意图且含**两个及以上提问**：逐条独立作答（每条一次讲解 + 对应白板）。
        # 单条提问的多意图（如「这道题怎么做 + 打开麦克风」）走下面的常规路径即可。
        plan = supervisor.router.plan(decision)
        questions = [seg for seg in plan if seg.intent is not IntentType.CONTROL]
        if len(questions) >= 2:
            return {"intent": intent, "route": "multi_intent", "multi_plan": plan}
        # 多意图里夹带的控制指令：简单意图，入口层当场办
        if decision.delegate and decision.control_clauses and intent is not IntentType.CONTROL:
            for clause in decision.control_clauses:
                supervisor._handle_control(clause)  # noqa: SLF001 - 复用同一实现
        if intent is IntentType.CONTROL:
            return {"intent": intent, "route": "control"}
        if intent is IntentType.CHITCHAT:
            return {"intent": intent, "route": "chitchat"}
        return {"intent": intent, "route": "teach"}

    async def answer_check(ctx: dict[str, Any]) -> dict[str, Any]:
        if ctx.get("route") != "answer_check":
            return {"__skip__": True}
        return {"response": await supervisor._turn_answer_check(ctx["turn"])}  # noqa: SLF001

    async def multi_intent(ctx: dict[str, Any]) -> dict[str, Any]:
        if ctx.get("route") != "multi_intent":
            return {"__skip__": True}
        return {
            "response": await supervisor._turn_multi_intent(  # noqa: SLF001
                ctx["turn"], ctx.get("multi_plan") or []
            )
        }

    async def resume_lecture(ctx: dict[str, Any]) -> dict[str, Any]:
        if ctx.get("route") != "resume_lecture":
            return {"__skip__": True}
        # 「继续」= 断点续讲；「再讲一遍」= 重讲当前页（模式由 lecture_command 内部再判）。
        # 老师直接接着讲，不出额外的语音回复（response=None，与控制指令同语义）。
        await supervisor.resume_lecture(ctx["turn"].text)  # noqa: SLF001
        return {"response": None}

    def control(ctx: dict[str, Any]) -> dict[str, Any]:
        if ctx.get("route") != "control":
            return {"__skip__": True}
        supervisor._handle_control(ctx["turn"].text)  # noqa: SLF001 - 复用同一实现
        return {"response": None}

    async def chitchat(ctx: dict[str, Any]) -> dict[str, Any]:
        if ctx.get("route") != "chitchat":
            return {"__skip__": True}
        return {"response": await supervisor._turn_chitchat()}  # noqa: SLF001

    async def teach(ctx: dict[str, Any]) -> dict[str, Any]:
        if ctx.get("route") != "teach":
            return {"__skip__": True}
        return {
            "response": await supervisor._turn_teach(ctx["turn"], ctx["intent"])  # noqa: SLF001
        }

    return TaskGraph.from_specs(
        TURN_GRAPH,
        TURN_SPECS,
        {
            "route": route,
            "answer_check": answer_check,
            "multi_intent": multi_intent,
            "resume_lecture": resume_lecture,
            "control": control,
            "chitchat": chitchat,
            "teach": teach,
        },
        session=getattr(supervisor, "session", None),   # 活动日志按会话归组（见 core/activity_log.py）
    )


# ------------------------------------------------------------------ 单页呈现
SLIDE_SPECS: tuple[NodeSpec, ...] = (
    NodeSpec(
        name="present_page",
        summary="换页：把页图推上白板并记下当前页序（页序不存在则中止本页）",
        impl="Supervisor._present_slide_page",
        inputs=("deck", "index", "with_speech"),
        outputs=("slide",),
        deps=(),
        requires=("core.courseware.loader.get_store",),
    ),
    NodeSpec(
        name="explain",
        summary="讲课链路讲这一页（讲师 -> 回退答疑 -> 兜底，见 ``lecturer.slide`` 图）",
        impl="Supervisor._explain_slide",
        inputs=("deck", "index", "with_speech"),
        outputs=("response",),
        deps=("present_page",),
        requires=("agents.lecturer.nodes（讲课件图）",),
    ),
    NodeSpec(
        name="annotate",
        summary="把本页标注叠加到白板（append=True，不重放/清屏）",
        impl="SessionState.push_board(append=True)",
        inputs=("response",),
        outputs=(),
        deps=("explain",),
        requires=("agents.shared.whiteboard.Whiteboard",),
    ),
    NodeSpec(
        name="speak",
        summary="下发聊天文本与教师状态（白板已在 annotate 下发，不重复推送）",
        impl="Supervisor._speak(push_board=False)",
        inputs=("response",),
        outputs=(),
        deps=("annotate",),
    ),
    NodeSpec(
        name="push_audio",
        summary="流式合成并下发语音（数字人 -> 纯语音 -> 前端朗读三级降级）",
        impl="Supervisor._push_audio",
        inputs=("response",),
        outputs=(),
        deps=("speak",),
        requires=("settings.digital_human_enable", "models.tts.get_tts", "models.avatar.get_avatar"),
    ),
)


def build_slide_graph(supervisor: "Supervisor") -> TaskGraph:
    """构建单页呈现图（``Supervisor._present_slide`` 的实现）。"""

    async def present_page(ctx: dict[str, Any]) -> dict[str, Any]:
        deck, index = ctx["deck"], int(ctx["index"])
        slide = deck.page(index)
        if slide is None:
            return {"__stop__": True, "response": None}
        supervisor._present_slide_page(deck, index, slide)  # noqa: SLF001
        return {"slide": slide}

    async def explain(ctx: dict[str, Any]) -> dict[str, Any]:
        if not ctx.get("with_speech", True):
            return {"__stop__": True}      # 预览某页：只上页面不讲解
        return {
            "response": await supervisor._explain_slide(  # noqa: SLF001
                ctx["deck"], int(ctx["index"])
            )
        }

    def annotate(ctx: dict[str, Any]) -> dict[str, Any]:
        response = ctx.get("response")
        if response is None or not response.board_ops:
            return {"__skip__": True}
        supervisor.session.push_board(response.board_ops, append=True)
        return {}

    def speak(ctx: dict[str, Any]) -> dict[str, Any]:
        supervisor._speak(ctx["response"], push_board=False)  # noqa: SLF001
        return {}

    async def push_audio(ctx: dict[str, Any]) -> dict[str, Any]:
        # record=True：页讲解的语音事件逐句入账，供「继续讲」从断点补发
        # （见 Supervisor._open_slide_event_log / _record_slide_event）
        await supervisor._push_audio(ctx["response"], record=True)  # noqa: SLF001
        return {}

    return TaskGraph.from_specs(
        SLIDE_GRAPH,
        SLIDE_SPECS,
        {
            "present_page": present_page,
            "explain": explain,
            "annotate": annotate,
            "speak": speak,
            "push_audio": push_audio,
        },
        session=getattr(supervisor, "session", None),
    )


# --------------------------------------------------- 预取页呈现（连讲翻页，页音同到）
PRESENTED_SPECS: tuple[NodeSpec, ...] = (
    NodeSpec(
        name="present_page",
        summary="换页：推页图并记页序（页面内容取自预取产物，不再现做）",
        impl="Supervisor._present_slide_page",
        inputs=("deck", "index", "prepared"),
        outputs=("slide",),
        deps=(),
        requires=("core.courseware.loader.get_store",),
    ),
    NodeSpec(
        name="annotate",
        summary="叠加本页标注（append=True，页面刚上不能被清掉）",
        impl="SessionState.push_board(append=True)",
        inputs=("prepared",),
        outputs=(),
        deps=("present_page",),
        requires=("agents.shared.whiteboard.Whiteboard",),
    ),
    NodeSpec(
        name="speak",
        summary="下发字幕与教师状态（语音在 flush_events 里与页图一起到）",
        impl="Supervisor._speak(push_board=False)",
        inputs=("prepared",),
        outputs=(),
        deps=("annotate",),
    ),
    NodeSpec(
        name="flush_events",
        summary="把预取阶段备好的音视频事件一次性下发（师生两侧画面与声音同时到）",
        impl="SessionState.push_event（预取事件队列）",
        inputs=("prepared",),
        outputs=(),
        deps=("speak",),
        requires=("Supervisor._prepare_speech_events",),
    ),
)


def build_presented_graph(supervisor: "Supervisor") -> TaskGraph:
    """构建预取页呈现图（``Supervisor._present_prepared`` 的实现）。"""

    def present_page(ctx: dict[str, Any]) -> dict[str, Any]:
        slide, _response, _events = ctx["prepared"]
        supervisor._present_slide_page(ctx["deck"], int(ctx["index"]), slide)  # noqa: SLF001
        return {"slide": slide}

    def annotate(ctx: dict[str, Any]) -> dict[str, Any]:
        response = ctx["prepared"][1]
        if not response.board_ops:
            return {"__skip__": True}
        # append=True 只叠加，不重放/清屏（页面刚上，不能被清掉）
        supervisor.session.push_board(response.board_ops, append=True)
        return {}

    def speak(ctx: dict[str, Any]) -> dict[str, Any]:
        supervisor._speak(ctx["prepared"][1], push_board=False)  # noqa: SLF001
        return {}

    def flush_events(ctx: dict[str, Any]) -> dict[str, Any]:
        for event, payload in ctx["prepared"][2]:
            supervisor.session.push_event(event, payload)
        return {}

    return TaskGraph.from_specs(
        PRESENTED_GRAPH,
        PRESENTED_SPECS,
        {
            "present_page": present_page,
            "annotate": annotate,
            "speak": speak,
            "flush_events": flush_events,
        },
        session=getattr(supervisor, "session", None),
    )


#: 本 agent 的图名 → 节点声明（供 ``agents.nodes`` 注册表汇总）
GRAPH_SPECS: dict[str, tuple[NodeSpec, ...]] = {
    TURN_GRAPH: TURN_SPECS,
    SLIDE_GRAPH: SLIDE_SPECS,
    PRESENTED_GRAPH: PRESENTED_SPECS,
}
