"""Agent 自主性四条链路的单测（工具调用 / ReAct 闭环 / 回合规划 / 学情记忆）。

对应 ``docs/agent_autonomy.md``，钉住四件事：
1. **工具真的被模型调到了**：schema 是 function-calling 格式，调用结果回灌成「观察」，
   失败也只变成一段可读文本（绝不抛异常打断上课）；
2. **闭环真的是闭环**：思考 -> 行动 -> 观察 -> 收尾，步数用尽时会禁用工具逼模型作答，
   模型不支持工具时退化成一次普通生成；
3. **计划真的是模型定的**：模型给计划就用模型的，超时 / 关闭 / 输出不合法回退规则计划，
   且同一回合只规划一次；
4. **记忆真的跨会话**：按学生隔离、可去重、可物理删除，且判题判错时零成本落一条错题。
"""
from __future__ import annotations

from typing import Any

from agents.lecturer.agent import LecturerAgent
from agents.shared.teaching_layers import build_messages
from agents.shared.toolkit import AgentToolkit
from agents.supervisor import Supervisor
from agents.teacher.agent import TeacherAgent
from agents.teacher.react import STOP_ANSWER, STOP_MAX_STEPS, STOP_NO_TOOLS, ReActLoop
from config.constants import IntentType
from config.settings import settings
from core import math_solver
from core.intent import IntentSegment
from core.memory.store import MemoryStore, get_memory_store
from core.memory.writer import MemoryWriter
from core.planning.planner import TurnPlanner
from core.state.models import ChatMessage, StudentTurn
from core.state.session import SessionState
from models.base import BaseLLMClient, ChatMessages, ToolCall, ToolChatResult
from models.llm import MockLLM


# ---------------------------------------------------------------- 夹具
def _session(session_id: str = "auto", student_id: str = "stu") -> SessionState:
    state = SessionState(session_id=session_id, student_id=student_id)
    state.scratch.update(grade="初中", subject="数学", subject_id="math")
    state.subject = "数学"
    return state


def _supervisor(session_id: str = "auto") -> tuple[Supervisor, SessionState]:
    state = _session(session_id)
    supervisor = Supervisor(
        state,
        teacher=TeacherAgent(llm=MockLLM()),
        lecturer=LecturerAgent(llm=MockLLM()),
    )
    return supervisor, state


class _NoToolLLM(BaseLLMClient):
    """不支持原生函数调用的模型（老供应商 / 部分自建服务）。"""

    def __init__(self, text: str = "只给文本的答复") -> None:
        self.text = text

    async def chat(
        self,
        messages: ChatMessages,
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        return self.text


class _StubPlanLLM(BaseLLMClient):
    """吐出固定 JSON 的规划模型（用于验证「模型给了计划就用模型的」）。"""

    def __init__(self, payload: Any) -> None:
        self.payload = payload

    async def chat(
        self,
        messages: ChatMessages,
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        return ""

    async def chat_json(
        self,
        messages: ChatMessages,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        return self.payload if isinstance(self.payload, dict) else {}


# ---------------------------------------------------------------- 1. 工具包
def test_toolkit_exposes_function_calling_schemas() -> None:
    """工具清单必须是 OpenAI function-calling 结构，且含闭环最需要的几件工具。"""
    toolkit = AgentToolkit(_session())
    schemas = toolkit.schemas()

    assert schemas, "工具清单为空"
    for schema in schemas:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"

    names = set(toolkit.tool_names())
    assert {"knowledge_search", "math_solve", "math_verify", "ocr_parse", "read_board"} <= names
    # 刻意不暴露坐标级白板工具：版式由确定性布局器负责，见 agents/shared/toolkit.py
    assert "whiteboard_write" not in names


async def test_toolkit_call_returns_observation_and_never_raises() -> None:
    """工具调用返回可读观察文本；未知工具 / 缺参数 / 执行失败都是文本，不是异常。"""
    toolkit = AgentToolkit(_session())

    assert "没有名为" in await toolkit.call("no_such_tool", {})
    assert "缺少" in await toolkit.call("math_solve", {})
    assert "不是合法 JSON" in await toolkit.call("math_solve", {"__raw__": "{bad json"})

    observation = await toolkit.call("math_solve", {"equation": "3x+5=20"})
    if math_solver.available():
        assert "x=5" in observation.replace(" ", "")
    else:  # 未装 sympy：给出可读提示，而不是空结果
        assert "sympy" in observation

    assert toolkit.call_count >= 4
    assert any(not record.ok for record in toolkit.records)


async def test_toolkit_reads_board_and_queues_notes() -> None:
    """读白板取当前板书；whiteboard_note 只入队，由上板方决定版式。"""
    state = _session("board")
    supervisor, _ = _supervisor("board-b")
    state.board_ops = list(
        supervisor.teacher.board.build_explain(title="解方程", steps=["移项", "合并同类项"])
    )
    toolkit = AgentToolkit(state, board=supervisor.teacher.board)

    board = await toolkit.call("read_board", {})
    assert "移项" in board

    assert "已记下" in await toolkit.call("whiteboard_note", {"text": "注意变号"})
    assert toolkit.consume_board_notes() == ["注意变号"]
    assert toolkit.consume_board_notes() == [], "取走即清空，避免重复上板"


async def test_toolkit_honours_call_limit(monkeypatch) -> None:
    """工具调用次数上限生效：超限后给出可读提示而不是继续执行。"""
    monkeypatch.setattr(settings, "agent_tool_max_calls", 1)
    toolkit = AgentToolkit(_session("limit"))

    await toolkit.call("read_board", {})
    blocked = await toolkit.call("read_board", {})
    assert "上限" in blocked
    assert toolkit.call_count == 2


async def test_toolkit_disabled_by_switch(monkeypatch) -> None:
    """``AGENT_TOOLS_ENABLE=false`` 时工具不可用（调用方据此走老链路）。"""
    monkeypatch.setattr(settings, "agent_tools_enable", False)
    assert AgentToolkit(_session()).enabled is False


# ---------------------------------------------------------------- 2. ReAct 闭环
async def test_react_loop_runs_think_act_observe() -> None:
    """闭环真的跑通：模型先调工具，拿到观察后再给最终答复。"""
    toolkit = AgentToolkit(_session("react"))
    llm = MockLLM(tool_script=[("math_solve", {"equation": "3x+5=20"})])

    result = await ReActLoop(llm, toolkit, max_steps=3).run([{"role": "user", "content": "解方程"}])

    assert result.stopped == STOP_ANSWER
    assert result.used_tools and result.tool_count == 1
    assert len(result.steps) == 1
    step = result.steps[0]
    assert step.calls[0]["name"] == "math_solve"
    assert step.observations and step.observations[0].strip()
    assert result.text.strip(), "拿到观察后必须仍然给出最终答复"
    assert result.to_meta()["tool_calls"] == 1


async def test_react_loop_disables_tools_when_steps_exhausted() -> None:
    """步数用尽：不再允许调工具，逼模型用已有观察收尾（避免无限循环）。"""
    toolkit = AgentToolkit(_session("react-max"))
    llm = MockLLM(tool_script=[
        ("read_board", {}),
        ("read_board", {}),
        ("read_board", {}),
    ])

    result = await ReActLoop(llm, toolkit, max_steps=2).run([{"role": "user", "content": "讲题"}])

    assert result.stopped == STOP_MAX_STEPS
    assert result.tool_count == 2, "工具调用次数必须被 max_steps 截住"
    assert result.text.strip()


async def test_react_loop_degrades_without_tool_support() -> None:
    """模型不支持函数调用：退化成一次普通生成，行为与加装闭环之前一致。"""
    toolkit = AgentToolkit(_session("react-nosup"))

    result = await ReActLoop(_NoToolLLM("普通答复"), toolkit).run(
        [{"role": "user", "content": "讲题"}]
    )

    assert result.stopped == STOP_NO_TOOLS
    assert result.text == "普通答复"
    assert result.tool_count == 0


class _RecordingToolLLM(BaseLLMClient):
    """记录每轮有没有要求 JSON 模式，首轮返回工具调用、次轮返回最终答复。"""

    supports_tools = True

    def __init__(self) -> None:
        self.json_flags: list[bool] = []

    async def chat(self, messages: ChatMessages, **_kwargs: Any) -> str:
        return "纯文本"

    async def chat_with_tools(
        self,
        messages: ChatMessages,
        *,
        tools: list[dict[str, Any]],
        tool_choice: Any = "auto",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> ToolChatResult:
        self.json_flags.append(json_mode)
        if len(self.json_flags) == 1:
            call = ToolCall(id="call_1", name="read_board", arguments={})
            return ToolChatResult(content="我先看一眼白板。", tool_calls=[call])
        return ToolChatResult(content='{"speech": "讲完了"}', finish_reason="stop")


async def test_react_loop_forces_json_only_after_first_round() -> None:
    """第 1 轮不强制 JSON（要留给模型自由选工具），第 2 轮起强制（那轮是最终答复）。

    回归点：长讲解的 JSON 偶发结构损坏，而 ReAct 的答复走 ``chat_with_tools`` ——
    不带上 ``response_format`` 的话，这条路径完全没有保护（实测云端「原文 1200 字」损坏）。
    """
    llm = _RecordingToolLLM()
    toolkit = AgentToolkit(_session("react-json"))
    messages = [{"role": "user", "content": "解方程 3x+5=20"}]

    result = await ReActLoop(llm, toolkit, max_steps=3).run(messages)

    assert result.stopped == STOP_ANSWER
    assert result.used_tools, "第 1 轮必须仍能调工具"
    assert llm.json_flags == [False, True], f"实际: {llm.json_flags}"


async def test_teacher_answer_merges_tool_board_notes() -> None:
    """闭环里工具要求补写的板书行，会接在契约板书之后（版式仍由 Whiteboard 负责）。"""
    state = _session("teach-react")
    teacher = TeacherAgent(llm=MockLLM(tool_script=[
        ("whiteboard_note", {"text": "记得检验根"}),
    ]))

    response = await teacher.answer(StudentTurn(text="解方程 3x+5=20"), state)

    texts = [op.text or "" for op in response.board_ops]
    assert any("记得检验根" in text for text in texts), "工具补写的板书行必须出现在板书末尾"
    assert response.meta["react"]["tool_calls"] == 1
    assert response.meta["react"]["steps"] == 1


async def test_supervisor_end_to_end_react_turn() -> None:
    """端到端：答疑链路真的走通闭环 —— 模型调了工具，板书与轨迹都落到了会话上。"""
    state = _session("e2e")
    supervisor = Supervisor(
        state,
        teacher=TeacherAgent(llm=MockLLM(tool_script=[
            ("math_solve", {"equation": "3x+5=20"}),
            ("whiteboard_note", {"text": "记得检验根"}),
        ])),
        lecturer=LecturerAgent(llm=MockLLM()),
    )

    response = await supervisor.handle_turn(StudentTurn(text="解方程 3x+5=20"))

    assert response is not None and response.speech
    assert response.meta["react"]["tool_calls"] == 2
    assert any("记得检验根" in (op.text or "") for op in state.board_ops)
    assert state.history and state.history[-1].role == "ai"
    assert state.teaching is False and state.ai_speaking is False, "收尾必须复位状态"


def test_toolkit_pool_reuses_one_toolkit_per_session() -> None:
    """同一会话只装配一次工具包（工具带会话上下文，不能每回合重建）。"""
    supervisor, state = _supervisor("pool")

    assert supervisor.toolkit_for(state) is supervisor.toolkit_for(state)
    assert supervisor.toolkit_for(_session("pool-other")) is not supervisor.toolkit_for(state)


async def test_teacher_answer_falls_back_when_react_disabled(monkeypatch) -> None:
    """``AGENT_REACT_ENABLE=false`` 时回到单次生成：不产生 ReAct 痕迹。"""
    monkeypatch.setattr(settings, "agent_react_enable", False)
    state = _session("teach-plain")
    teacher = TeacherAgent(llm=MockLLM(tool_script=[("read_board", {})]))

    turn = StudentTurn(text="解方程 3x+5=20")
    response = await teacher.answer(turn, state)

    assert response.speech
    assert "react" not in response.meta
    assert turn.meta.get("react") is None


# ---------------------------------------------------------------- 3. 回合规划
async def test_planner_uses_model_plan_and_filters_unknown_tools() -> None:
    """模型给了计划就用模型的；不认识的动作 / 工具名一律被丢掉（不污染链路）。"""
    planner = TurnPlanner(llm=_StubPlanLLM({
        "goal": "讲清移项法则",
        "steps": ["先讲什么是移项", "再演示一道题"],
        "tools": ["math_solve", "rm_rf_everything"],
        "ask_followup": False,
        "reason": "方程题要先算再讲",
    }))

    plan = await planner.plan(StudentTurn(text="解方程 3x+5=20"))

    assert plan.source == "llm"
    assert plan.goal == "讲清移项法则"
    assert len(plan.steps) == 2
    assert plan.tools == ("math_solve",), "未登记的工具名必须被过滤"
    assert plan.ask_followup is False
    assert plan.notes() and "执行计划" in plan.notes()[0]


async def test_planner_falls_back_to_rule_plan() -> None:
    """模型输出为空 -> 回退规则计划；方程题应当建议求解与验算两件工具。"""
    planner = TurnPlanner(llm=_StubPlanLLM({}))

    plan = await planner.plan(StudentTurn(text="解方程 3x+5=20"))

    assert plan.source == "rule"
    assert "math_solve" in plan.tools and "math_verify" in plan.tools


def test_rule_plan_reads_intent_snapshot_dict() -> None:
    """规则计划要能吃下 ``turn.meta`` 里的意图快照字典（两套编排留下的都是它）。"""
    planner = TurnPlanner(llm=_StubPlanLLM({}))
    decision = {
        "multi": True,
        "complex": False,
        "clauses": ["打开麦克风", "这道题怎么做", "再帮我看看这题"],
        "control_clauses": ["打开麦克风"],
    }

    plan = planner.rule_plan(StudentTurn(text="打开麦克风，这道题怎么做，再帮我看看这题"), decision)

    assert len(plan.steps) == 2, "控制指令不该进计划（它在入口层当场办掉）"
    assert "2 个诉求" in plan.goal


def test_rule_plan_skips_model_when_planner_disabled(monkeypatch) -> None:
    """``AGENT_PLANNER_ENABLE=false``：即使模型可用也只走规则计划。"""
    monkeypatch.setattr(settings, "agent_planner_enable", False)
    planner = TurnPlanner(llm=_StubPlanLLM({"goal": "模型计划", "tools": ["math_solve"]}))

    plan = planner.rule_plan(StudentTurn(text="解方程 3x+5=20"))
    assert plan.source == "rule"


async def test_multi_intent_sub_turn_skips_model_planning() -> None:
    """多意图逐条作答时，子回合不请模型规划（一次输入不能付多次规划往返）。"""
    calls: list[str] = []

    class _CountingPlanner(TurnPlanner):
        async def plan(self, turn: StudentTurn, decision: Any = None) -> Any:  # type: ignore[override]
            calls.append(turn.text)
            return await super().plan(turn, decision)

    supervisor, _ = _supervisor("sub-turn-plan")
    supervisor.planner = _CountingPlanner(llm=_StubPlanLLM({"goal": "模型计划"}))

    answer = await supervisor._turn_multi_intent(  # noqa: SLF001 - 直接验证子回合路径
        StudentTurn(text="打开麦克风，这道题怎么做，另外帮我看看那题"),
        [
            IntentSegment(text="打开麦克风", intent=IntentType.CONTROL),
            IntentSegment(text="这道题怎么做", intent=IntentType.TEXT_QUESTION),
            IntentSegment(text="另外帮我看看那题", intent=IntentType.TEXT_QUESTION),
        ],
    )

    assert answer is not None
    assert calls == [], "子回合应走规则计划，不该触发模型规划"


async def test_supervisor_caches_plan_per_turn() -> None:
    """同一回合只规划一次：流式答疑的两条路要用同一份计划。"""
    supervisor, _ = _supervisor("plan-cache")
    turn = StudentTurn(text="解方程 3x+5=20")

    first = await supervisor._resolve_plan(turn)  # noqa: SLF001 - 直接验证缓存
    second = await supervisor._resolve_plan(turn)  # noqa: SLF001

    assert first is second
    assert turn.meta["turn_plan"]["source"] == "rule"
    assert turn.meta["turn_plan"]["goal"]


# ---------------------------------------------------------------- 4. 学情记忆
def test_memory_store_dedupes_and_isolates_students(tmp_path) -> None:
    """同一学生的相似记忆累加命中；不同学生的记忆物理隔离（绝不串号）。"""
    store = MemoryStore(root=tmp_path)

    assert store.observe("stu-a", "mistake", "移项忘记变号：3x+5=20", subject="数学") is not None
    again = store.observe("stu-a", "mistake", "移项忘记变号：3x+5=20", subject="数学")
    assert again is not None and again.hits == 2
    assert len(store.active("stu-a")) == 1, "相似记忆应累加而不是新增"

    store.observe("stu-b", "weak_point", "分式通分不熟练", subject="数学")
    assert len(store.active("stu-b")) == 1
    assert len(store.active("stu-a")) == 1
    assert store.active("stu-a")[0].text != store.active("stu-b")[0].text


def test_memory_subject_filter_and_forget(tmp_path) -> None:
    """学科过滤生效；``forget`` 是物理删除（合规要求，不能只标记停用）。"""
    store = MemoryStore(root=tmp_path)
    store.observe("stu-c", "weak_point", "文言实词掌握不牢", subject="语文")
    store.observe("stu-c", "fact", "习惯先算再看题", subject="")

    assert len(store.active("stu-c")) == 2
    assert len(store.active("stu-c", subject="语文")) == 2, "不分学科的通用记忆任何学科都该看到"
    assert len(store.active("stu-c", subject="数学")) == 1

    assert store.forget("stu-c") is True
    assert store.active("stu-c") == []
    assert not store.path_for("stu-c").exists()


def test_memory_search_ranks_relevant_first(tmp_path) -> None:
    """相关性检索：与问题相关的记忆排在前面（记忆多了才挑得出该看的那几条）。"""
    store = MemoryStore(root=tmp_path)
    store.observe("stu-d", "mistake", "移项时忘记变号，导致整题算错", subject="数学")
    store.observe("stu-d", "fact", "写字比较慢，需要多一点时间", subject="数学")

    ranked = store.search("stu-d", "移项变号", subject="数学")
    assert ranked and "移项" in ranked[0].text


#: 12 条互不相似的学情记忆（相似度阈值 0.85，措辞太像会被合并，测不出条数）
_DISTINCT_MEMORIES = [
    "移项时经常忘记变号",
    "分式通分不熟练",
    "读图题容易忽略单位",
    "化学方程式配平常出错",
    "英语时态判断不准",
    "文言实词积累偏少",
    "声调标注位置记不牢",
    "写应用题不先列等量关系",
    "历史年代容易混淆",
    "细胞结构名词记混",
    "计算时习惯跳步",
    "做完题不爱检查",
]


def test_memory_no_eviction_by_default(tmp_path) -> None:
    """默认**不限**：学情记忆学到多少留多少（与教师经验同一条原则）。"""
    store = MemoryStore(root=tmp_path)
    assert store.max_items == 0, "默认必须是「不限」"

    for text in _DISTINCT_MEMORIES:
        store.observe("stu", "weak_point", text, subject="数学")

    assert len(store.active("stu")) == len(_DISTINCT_MEMORIES), "默认不该淘汰任何记忆"
    assert store.stats()["max_items_per_student"] == 0


def test_memory_explicit_zero_means_unlimited(monkeypatch, tmp_path) -> None:
    """显式 ``max_items=0`` = 不限，不能被 ``or`` 吃成配置里的正数。"""
    monkeypatch.setattr(settings, "memory_max_items", 5)
    assert MemoryStore(root=tmp_path, max_items=0).max_items == 0
    assert MemoryStore(root=tmp_path).max_items == 5


def test_memory_writer_records_mistake(tmp_path) -> None:
    """判题判错时零成本记一条错题（含学生答案与正确答案）。"""
    writer = MemoryWriter(store=MemoryStore(root=tmp_path))
    state = _session("mw", student_id="stu-e")

    item = writer.record_mistake(state, question="三角形内角和是多少", expected="180度", student_answer="90度")

    assert item is not None and item.kind == "mistake"
    assert "180度" in item.text and "90度" in item.text
    assert item.subject == "数学"


def test_memory_reaches_system_prompt(monkeypatch, tmp_path) -> None:
    """端到端：写入的记忆必须真的出现在提示词里，否则等于没记住。"""
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path))
    store = get_memory_store()
    assert store.observe("stu-f", "weak_point", "移项时经常忘记变号", subject="数学") is not None

    state = SessionState(session_id="mem-prompt", student_id="stu-f")
    state.scratch.update(grade="初中", subject="数学", subject_id="math")
    state.subject = "数学"
    state.add_message(ChatMessage(role="student", text="这道方程题怎么解？"))

    messages = build_messages(state, contract="（本轮任务契约）")
    system_text = "\n".join(
        str(m.get("content") or "") for m in messages if m.get("role") == "system"
    )
    assert "学情记忆" in system_text
    assert "移项时经常忘记变号" in system_text


async def test_wrong_answer_writes_mistake_memory(monkeypatch, tmp_path) -> None:
    """端到端：学生在判题环节答错，记忆表里应当出现这道错题。"""
    monkeypatch.setattr(settings, "memory_dir", str(tmp_path))
    supervisor, state = _supervisor("judge-memory")
    state.pending_question = {
        "question": "三角形的内角和是多少？",
        "expected": "180度",
        "hint": "想想平角",
        "options": [],
        "attempts": 0,
    }

    await supervisor.handle_turn(StudentTurn(text="90度"))

    items = get_memory_store().active("stu")
    assert items, "答错后应当写入一条学情记忆"
    assert items[0].kind == "mistake"
    assert "三角形的内角和" in items[0].text
