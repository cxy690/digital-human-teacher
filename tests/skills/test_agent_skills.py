"""Agent 技能单测（core/skills/agent_skills）。

钉住七件事：
1. **技能真的暴露给模型**：注册表与 AgentToolkit 都以 function-calling 格式给出技能，
   且技能名不与原子工具重名（同一个命名空间，重名会让调用意图产生歧义）；
2. **调用真的走 toolkit**：失败降级成「观察」、战绩落进 scratch；
3. **未命中也要可读**：技能说清「为什么这次没用我」，而不是静默返回空；
4. **自我更新·学**：技能上下文附上该老师经验表里选出的实战补充（人写基底 + 自动补充），
   且改 SKILL.md 的「讲课方式」小节下一次调用即生效；
5. **自我更新·评**：回合裁决回写 pulls/wins；战绩差时按开关自动停用，否则只进抽检台；
6. **可关**：``AGENT_SKILLS_ENABLE=false`` 时模型看到的清单与加装前完全一致；
7. **端到端**：答疑回合里模型调一次技能，回合结束后战绩落盘（接线证明）。
"""
from __future__ import annotations

import asyncio
from typing import Any

from agents.shared.toolkit import AgentToolkit
from agents.supervisor import Supervisor
from agents.teacher.agent import TeacherAgent
from config.constants import IntentType
from config.settings import settings
from config.teacher_skills import SKILLS_DIR
from config.teacher_skills.md import reload_docs
from core.skills.agent_skills import (
    AgentSkillRegistry,
    PraiseExpressionSkill,
    SkillUsageStore,
    StudentMistakeRecallSkill,
    SubjectPlaybookSkill,
    TeacherExperienceSkill,
    consume_used_skills,
    get_skill_registry,
    mark_skill_used,
    record_skill_outcome,
)
from core.skills.agent_skills.base import AgentSkill, SkillContext, SkillSpec
from core.state.models import AIResponse, StudentTurn
from core.state.session import SessionState
from models.llm import MockLLM


# ---------------------------------------------------------------- 夹具
def _session(session_id: str = "skill", student_id: str = "stu") -> SessionState:
    state = SessionState(session_id=session_id, student_id=student_id)
    state.scratch.update(grade="初中", subject="数学", subject_id="math", teacher="math_strict")
    state.subject = "数学"
    return state


def _context(session: SessionState, learned: str = "") -> SkillContext:
    return SkillContext(session=session, teacher_key="math_strict", learned_notes=learned)


class _FakeDecision:
    def __init__(self, praise: str, kind: str = "praise") -> None:
        self.praise = praise
        self.kind = kind


class _FakeExpression:
    """只实现 judge 的替身（PraiseExpressionSkill 只依赖这一个方法）。"""

    def __init__(self, decision: Any | None) -> None:
        self._decision = decision
        self.calls: list[str] = []

    async def judge(self, text: str, session: SessionState) -> Any | None:
        self.calls.append(text)
        return self._decision


class _FakeMemoryItem:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeMemoryStore:
    def __init__(self, items: list[_FakeMemoryItem]) -> None:
        self._items = items
        self.queries: list[tuple[str, str]] = []

    def search(
        self, student_id: str, query: str = "", *, subject: str = "", limit: int | None = None
    ) -> list[_FakeMemoryItem]:
        self.queries.append((student_id, query))
        return self._items[: limit or 3]


class _BrokenSkill(AgentSkill):
    """一律抛异常的技能的替身（验证「失败降级成观察」）。"""

    def __init__(self) -> None:
        self.spec = SkillSpec(name="broken_skill", title="坏技能", description="总是失败")

    async def run(self, args: dict[str, Any], context: SkillContext) -> Any:
        raise RuntimeError("boom")


# ---------------------------------------------------------------- 1. 暴露给模型
def test_registry_exposes_four_builtin_skills() -> None:
    names = set(get_skill_registry().names())
    assert {
        "praise_expression",
        "student_mistake_recall",
        "teacher_experience",
        "subject_playbook",
    } <= names
    # 技能在工具包里就是以「工具」身份暴露的（同一个命名空间），
    # 因此这里要钉的是：**原子工具**那一侧没有与技能重名的条目。
    toolkit = AgentToolkit(_session("names"))
    assert names <= set(toolkit.tool_names())


def test_skill_schemas_are_function_calling_shape() -> None:
    schemas = get_skill_registry().schemas()
    assert schemas
    for schema in schemas:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"
    by_name = {s["function"]["name"]: s for s in schemas}
    praise_params = by_name["praise_expression"]["function"]["parameters"]
    assert "student_answer" in praise_params["properties"]
    assert praise_params["required"] == ["student_answer"]


# ---------------------------------------------------------------- 2. 调用走 toolkit
async def test_toolkit_runs_skill_and_records_usage() -> None:
    session = _session("toolkit-skill")
    toolkit = AgentToolkit(session)

    assert "subject_playbook" in toolkit.tool_names()
    observation = await toolkit.call("subject_playbook", {"topic": "一元一次方程"})

    assert "讲课方式" in observation and "建模" in observation
    record = toolkit.records[-1]
    assert record.name == "subject_playbook" and record.ok
    # 战绩记账：调用即写进 scratch（回合末才按裁决回写）
    book = session.scratch.get("_skills_used") or {}
    assert "subject_playbook" in (book.get("names") or [])
    assert consume_used_skills(session) == ["subject_playbook"]
    assert consume_used_skills(session) == [], "取走即清空，避免下一轮重复记账"


async def test_skill_failure_degrades_to_observation() -> None:
    session = _session("broken")
    toolkit = AgentToolkit(session)
    registry = AgentSkillRegistry(skills=[_BrokenSkill()], usage_store=SkillUsageStore())
    toolkit._handlers["broken_skill"] = (  # noqa: SLF001 - 直接验证降级路径
        {"description": "总是失败", "parameters": {"type": "object", "properties": {}}},
        toolkit._make_skill_handler(_BrokenSkill(), registry),
    )

    observation = await toolkit.call("broken_skill", {})

    assert "工具执行失败" in observation
    assert toolkit.records[-1].ok is False and toolkit.records[-1].reason == "exception"


async def test_learned_notes_reach_the_model_through_toolkit(monkeypatch, tmp_path) -> None:
    """「学」的最后一环：经验表选出的实战补充必须出现在模型看到的观察里。

    接线证明：注册表把 learned_notes 装进上下文只是半条路，toolkit 不回给模型，
    自我更新就只剩战绩那一半（模块写完、测试也在、就是没到模型眼前）。
    """
    monkeypatch.setattr(settings, "skill_experience_dir", str(tmp_path))
    from core.skills.experience import get_experience_store  # noqa: PLC0415

    get_experience_store().observe(
        "math_strict", "pitfall", "学生常把移项后的符号忘掉，导致整题算错", confidence=0.9
    )
    session = _session("learned")
    toolkit = AgentToolkit(session)

    first = await toolkit.call("student_mistake_recall", {"query": "移项"})
    assert "实战补充" in first and "移项后的符号忘掉" in first, "学习增强必须到模型眼前"

    # 同一回合第二次调用不重复贴（按教师 key 去重，省 token）
    second = await toolkit.call("student_mistake_recall", {"query": "移项"})
    assert "实战补充" not in second

    # 新回合重新贴
    toolkit.reset_turn()
    third = await toolkit.call("student_mistake_recall", {"query": "移项"})
    assert "实战补充" in third


# ---------------------------------------------------------------- 3. 未命中也要可读
async def test_praise_skill_hit_and_miss() -> None:
    session = _session("praise")
    skill = PraiseExpressionSkill(inner=_FakeExpression(_FakeDecision("这一步做对了，很稳")))

    hit = await skill.run({"student_answer": "x=5"}, _context(session))
    assert "这一步做对了" in hit.text

    miss = await PraiseExpressionSkill(inner=_FakeExpression(None)).run(
        {"student_answer": "x=5"}, _context(session)
    )
    assert miss.hit is False and "未命中" in miss.text


async def test_mistake_recall_reads_memory_and_stays_quiet_when_empty(monkeypatch) -> None:
    import core.memory as memory_pkg  # noqa: PLC0415 - 运行期替换，避免改导入结构

    session = _session("recall")
    skill = StudentMistakeRecallSkill()

    monkeypatch.setattr(
        memory_pkg, "get_memory_store", lambda: _FakeMemoryStore([_FakeMemoryItem("移项忘记变号")])
    )
    hit = await skill.run({"query": "移项"}, _context(session))
    assert "移项忘记变号" in hit.text

    monkeypatch.setattr(memory_pkg, "get_memory_store", lambda: _FakeMemoryStore([]))
    miss = await skill.run({"query": "移项"}, _context(session))
    assert miss.hit is False and "没有查到" in miss.text


async def test_teacher_experience_skill_reads_store(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "skill_experience_dir", str(tmp_path))
    from core.skills.experience import get_experience_store  # noqa: PLC0415

    get_experience_store().observe(
        "math_strict", "must_cover", "讲数轴必须强调三要素缺一不可", confidence=0.9
    )

    outcome = await TeacherExperienceSkill().run({}, _context(_session("exp")))

    assert "三要素" in outcome.text
    miss = await TeacherExperienceSkill().run(
        {"kind": "analogy"}, _context(_session("exp"))
    )
    assert miss.hit is False


# ---------------------------------------------------------------- 4. 自我更新·学
def test_skill_context_carries_learned_notes(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "skill_experience_dir", str(tmp_path))
    monkeypatch.setattr(settings, "agent_skill_learned_max", 2)
    from core.skills.experience import get_experience_store  # noqa: PLC0415

    get_experience_store().observe(
        "math_strict", "pitfall", "学生常把移项后的符号忘掉，导致整题算错", confidence=0.9
    )
    registry = get_skill_registry()

    context = registry.context_for(StudentMistakeRecallSkill(), _session("learn"))
    assert "移项" in context.learned_notes, "技能应附上该老师的实战补充"

    # 不关心经验的技能（如表扬）不该被硬塞教学经验
    assert registry.context_for(PraiseExpressionSkill(), _session("learn")).learned_notes == ""


def test_playbook_skill_reads_subject_md_live() -> None:
    """改 SKILL.md 的「讲课方式」小节，下一次调用即生效（不需要重启）。"""
    path = SKILLS_DIR / "math" / "SKILL.md"
    original = path.read_text(encoding="utf-8")
    try:
        path.write_text(original.replace("建模五步闭环", "【自更新探针】建模探针闭环"), encoding="utf-8")
        reload_docs()
        outcome = asyncio.run(SubjectPlaybookSkill().run({}, _context(_session("md-live"))))
        assert "【自更新探针】建模探针闭环" in outcome.text
    finally:
        path.write_text(original, encoding="utf-8")
        reload_docs()


# ---------------------------------------------------------------- 5. 自我更新·评
def test_usage_store_accumulates_and_reviews(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "agent_skill_usage_dir", str(tmp_path))
    store = SkillUsageStore()

    assert store.record_usage(["subject_playbook"], gain=1.0) == 1
    store.record_usage(["subject_playbook", "praise_expression"], gain=1.0)
    store.record_usage(["subject_playbook"], gain=0.0)

    usage = store.load("subject_playbook")
    assert usage.pulls == 3 and usage.wins == 2 and usage.gain_rate == round(2 / 3, 3)

    # 战绩健康的技能**不进**抽检台（那里只列「最该人工看一眼」的）
    assert not any(row["name"] == "subject_playbook" for row in store.review_queue())

    # 战绩差的才进，且排在前面
    for _ in range(5):
        store.record_usage(["weak_skill"], gain=0.0)
    rows = store.review_queue()
    assert rows and rows[0]["name"] == "weak_skill"
    assert "有效率" in rows[0]["reasons"][0]

    assert store.set_status("subject_playbook", "disabled") is True
    assert store.load("subject_playbook").status == "disabled"


def test_auto_retire_only_when_configured(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "agent_skill_usage_dir", str(tmp_path))
    store = SkillUsageStore()
    store.record_usage(["subject_playbook"], gain=0.0)
    store.record_usage(["subject_playbook"], gain=0.0)
    registry = AgentSkillRegistry(skills=[SubjectPlaybookSkill()], usage_store=store)

    # 默认（关）：战绩差也照样暴露，只进抽检台 —— 自动学到的东西不静默丢弃
    assert [s.spec.name for s in registry.enabled()] == ["subject_playbook"]

    monkeypatch.setattr(settings, "agent_skill_auto_retire", True)
    monkeypatch.setattr(settings, "agent_skill_min_pulls", 2)
    monkeypatch.setattr(settings, "agent_skill_min_gain_rate", 0.5)
    assert registry.enabled() == [], "样本够且有效率过低时才自动停用"


def test_record_skill_outcome_follows_verdict(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "agent_skill_usage_dir", str(tmp_path))
    session = _session("verdict")

    mark_skill_used(session, "subject_playbook")
    assert record_skill_outcome(session, {"verdict": "pass"}) == 1
    assert session.scratch.get("_skills_used") is None, "记完就清"

    mark_skill_used(session, "subject_playbook")
    assert record_skill_outcome(session, {"verdict": "degraded"}) == 1

    mark_skill_used(session, "subject_playbook")
    assert record_skill_outcome(session, {"error": "TimeoutError"}) == 0, "异常回合不记账"

    usage = SkillUsageStore().load("subject_playbook")
    assert usage.pulls == 2 and usage.wins == 1


# ---------------------------------------------------------------- 6. 可关
def test_skills_disabled_by_switch(monkeypatch) -> None:
    monkeypatch.setattr(settings, "agent_skills_enable", False)
    assert get_skill_registry().enabled() == []
    assert get_skill_registry().schemas() == []
    assert not (set(AgentToolkit(_session("off")).tool_names()) & set(get_skill_registry().names()))


# ---------------------------------------------------------------- 7. 端到端
async def test_supervisor_records_skill_usage_end_to_end(monkeypatch, tmp_path) -> None:
    """接线证明：答疑回合里模型调一次技能，回合结束后战绩按裁决落盘。"""
    monkeypatch.setattr(settings, "agent_skill_usage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "skill_experience_dir", str(tmp_path))
    state = _session("e2e", student_id="stu-e2e")
    supervisor = Supervisor(
        state,
        teacher=TeacherAgent(llm=MockLLM(tool_script=[("subject_playbook", {"topic": "数轴"})])),
    )

    response = await supervisor.handle_turn(StudentTurn(text="什么是数轴"))

    assert response is not None and response.speech
    usage = SkillUsageStore().load("subject_playbook")
    assert usage.pulls >= 1, "技能被调用过，回合末必须回写战绩"
    assert usage.wins in (0, 1)


async def test_board_first_single_call_reroutes_when_react_on(monkeypatch) -> None:
    """single_call 走流式单次调用（没有工具集）→ ReAct 开启时必须改走 two_call。

    否则技能与工具在 ``BOARD_FIRST_MODE=single_call`` 的部署里会被静默跳过 ——
    功能在、链路不经过它，正是本项目栽过的那种跟头。
    """
    seen: list[str] = []
    state = _session("reroute", student_id="stu-rr")
    supervisor = Supervisor(state, teacher=TeacherAgent(llm=MockLLM()))

    async def spy_board_first(turn, intent, *, hint_level=None, ocr_text="", mode="off"):
        seen.append(mode)
        return AIResponse(speech="（桩）")

    monkeypatch.setattr(supervisor, "_turn_teach_board_first", spy_board_first)
    monkeypatch.setattr(settings, "board_first_mode", "single_call")
    monkeypatch.setattr(settings, "agent_react_enable", True)
    monkeypatch.setattr(settings, "agent_tools_enable", True)

    await supervisor._turn_teach(  # noqa: SLF001 - 直接验证分派决策
        StudentTurn(text="什么是数轴"), IntentType.TEXT_QUESTION, ask_followup=False
    )
    assert seen == ["two_call"], f"ReAct 开启时应改走 two_call，实际：{seen}"

    # 关掉 ReAct：保持用户配置的 single_call（不被本功能悄悄改掉）
    monkeypatch.setattr(settings, "agent_react_enable", False)
    await supervisor._turn_teach(  # noqa: SLF001
        StudentTurn(text="什么是数轴"), IntentType.TEXT_QUESTION, ask_followup=False
    )
    assert seen == ["two_call", "single_call"], f"关掉后应保持原配置，实际：{seen}"
