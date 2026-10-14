"""教师人格技能测试（core/skills/teacher_personality）。

覆盖：
1. 11 个学科各有独立人格（含合并时补齐的「政治」），且不多不少；
2. 注册表按 subject_id 精确查找，绝不跨学科顶替；
3. 平台底线规则存在；
4. TeacherAgent 组装提示词时确实带上了学科人格。
"""
from __future__ import annotations

from config.subjects import SUBJECTS
from core.skills.teacher_personality import (
    GLOBAL_TEACHER_RULES,
    SKILLS,
    SKILL_ORDER,
    TeacherPersonalitySkill,
    get_skill_registry,
)


class _FakeSession:
    """只带 scratch 的极简会话（get_for_session 只读 scratch）。"""

    def __init__(self, **scratch: object) -> None:
        self.scratch = dict(scratch)


def test_skill_count_matches_subject_count() -> None:
    """人格数量必须等于学科数量（政治为合并时补齐的第 11 个）。"""
    assert len(SKILLS) == len(SUBJECTS) == 11


def test_every_subject_has_personality() -> None:
    """每个学科的 key 都能查到独立人格，且查到的正是本学科。"""
    registry = get_skill_registry()
    for item in SUBJECTS:
        skill = registry.get_for_subject(item["key"])
        assert skill is not None, f"{item['name']}({item['key']}) 缺少教师人格"
        assert skill.subject_id == item["key"]
        assert skill.subject_name == item["name"]
        assert skill.skill_id.startswith(f"teacher.{item['key']}.")


def test_unknown_subject_returns_none() -> None:
    """查不到就返回 None（调用方回退 persona），绝不跨学科顶替。"""
    registry = get_skill_registry()
    assert registry.get_for_subject("") is None
    assert registry.get_for_subject("不存在的学科") is None
    assert registry.get_for_subject(None) is None


def test_subject_map_is_one_persona_per_subject() -> None:
    """一科一人格：subject_id -> skill_id 不允许一科多值或串科。"""
    mapping = get_skill_registry().subject_map()
    assert len(mapping) == len(SKILLS) == 11
    for subject_id, skill_id in mapping.items():
        assert skill_id.startswith(f"teacher.{subject_id}.")


def test_each_skill_has_independent_prompt() -> None:
    """每科提示词必须非空且彼此不同（不允许复制粘贴式共用）。"""
    prompts: dict[str, str] = {}
    for skill in SKILLS.values():
        assert isinstance(skill, TeacherPersonalitySkill)
        text = skill.build_system_prompt()
        assert text and len(text) > 100, skill.skill_id
        prompts[skill.skill_id] = text
    assert len(set(prompts.values())) == len(prompts), "存在完全相同的多科人格提示词"


def test_behaviors_all_declared() -> None:
    """九个行为维度都要有具体内容（不能是空串）。"""
    for skill in SKILLS.values():
        for name, value in skill.behaviors().items():
            assert str(value).strip(), f"{skill.skill_id}.{name} 为空"


def test_each_skill_has_identity_meta() -> None:
    """身份元信息齐全，且能结构化成前端/日志可用的摘要。"""
    for skill in SKILLS.values():
        described = skill.describe()
        for field in ("skill_id", "subject_id", "subject_name", "display_name", "persona", "tone"):
            assert str(described.get(field) or "").strip(), f"{skill.skill_id}.{field} 为空"


def test_platform_rules_exist() -> None:
    assert "通用教师规则" in GLOBAL_TEACHER_RULES
    assert "不编造事实" in GLOBAL_TEACHER_RULES


def test_get_for_session_uses_subject_id() -> None:
    """会话里写了 subject_id 时应命中学科人格。"""
    skill = get_skill_registry().get_for_session(_FakeSession(subject_id="math"))
    assert skill is not None and skill.subject_id == "math"


def test_get_for_session_unknown_returns_none() -> None:
    """未知学科返回 None，而不是乱加载别的学科人格。"""
    assert get_skill_registry().get_for_session(_FakeSession(subject_id="unknown-x")) is None


def test_skill_order_matches_registry() -> None:
    assert len(SKILL_ORDER) == len(SKILLS) == 11
    assert [cls().skill_id for cls in SKILL_ORDER] == list(SKILLS)


async def test_teacher_agent_injects_personality() -> None:
    """TeacherAgent 组装提示词时应带上平台底线 + 学科人格。"""
    from agents.teacher.agent import TeacherAgent
    from core.state.models import StudentTurn
    from core.state.session import SessionState

    state = SessionState(session_id="tp1", student_id="s1")
    state.scratch.update(grade="初中", subject="数学", subject_id="math")
    agent = TeacherAgent()
    messages = agent._build_messages(
        StudentTurn(text="老师，什么是数轴？"), state, hint_level=None, context_notes=None
    )
    joined = "\n".join(str(m.get("content", "")) for m in messages)
    assert "通用教师规则" in joined
    assert "严谨、清晰、耐心的数学老师" in joined


async def test_personality_can_be_disabled() -> None:
    """TEACHER_PERSONALITY_ENABLE=false 时应回到旧行为（不注入学科人格）。"""
    from config.settings import settings

    from agents.teacher.agent import TeacherAgent
    from core.state.models import StudentTurn
    from core.state.session import SessionState

    state = SessionState(session_id="tp2", student_id="s2")
    state.scratch.update(grade="初中", subject="数学", subject_id="math")
    original = settings.teacher_personality_enable
    settings.teacher_personality_enable = False
    try:
        agent = TeacherAgent()
        messages = agent._build_messages(
            StudentTurn(text="什么是数轴"), state, hint_level=None, context_notes=None
        )
    finally:
        settings.teacher_personality_enable = original
    joined = "\n".join(str(m.get("content", "")) for m in messages)
    assert "通用教师规则" not in joined


# ---------------------------------------------------------------- 接口下发
async def test_api_subjects_carries_personality() -> None:
    """/api/subjects 的每个学科都要带上该学科的人格摘要。"""
    from server.routes import list_subjects

    data = await list_subjects()
    assert len(data["subjects"]) == 11
    for item in data["subjects"]:
        p = item.get("personality")
        assert p is not None, f"{item['name']} 未下发 personality"
        assert p["subject_id"] == item["key"]
        assert p["display_name"] and p["persona"]


async def test_api_personality_lookup_and_detail() -> None:
    """/api/personality 支持中文名与英文 key，detail 时带行为维度与提示词。"""
    from server.routes import personality_info

    brief = await personality_info(subject="数学")
    assert brief["enabled"] is True
    assert brief["personality"]["skill_id"] == "teacher.math.yanli"

    by_key = await personality_info(subject="math")
    assert by_key["personality"]["subject_id"] == "math"

    full = await personality_info(subject="politics", detail=True)
    assert full["personality"]["subject_id"] == "politics"
    assert len(full["personality"]["behaviors"]) == 9
    assert "政治老师" in full["personality"]["system_prompt"]

    unknown = await personality_info(subject="不存在的学科")
    assert unknown["personality"] is None


async def test_api_personality_returns_none_when_disabled() -> None:
    """技能关闭时接口不报错，只是不下发人格。"""
    from config.settings import settings

    from server.routes import personality_info

    original = settings.teacher_personality_enable
    settings.teacher_personality_enable = False
    try:
        data = await personality_info(subject="math")
    finally:
        settings.teacher_personality_enable = original
    assert data["enabled"] is False
    assert data["personality"] is None


# ---------------------------------------------------------------- 防串台
def _head(subject_id: str) -> str:
    """取某科人格提示词的开头（用作「这科人格被注入了吗」的判据）。"""
    skill = get_skill_registry().get_for_subject(subject_id)
    assert skill is not None
    return skill.build_system_prompt()[:24]


async def test_personality_not_crossed_when_subject_changes() -> None:
    """会话里残留数学 subject_id、但当前学科是物理时，绝不能注入数学人格。"""
    from agents.teacher.agent import TeacherAgent
    from core.state.models import StudentTurn
    from core.state.session import SessionState

    state = SessionState(session_id="x1", student_id="s1")
    # 故意制造不一致：学科名是物理，残留的 subject_id 还是数学
    state.scratch.update(grade="初中", subject="物理", subject_id="math", teacher="")
    messages = TeacherAgent()._build_messages(
        StudentTurn(text="老师，牛顿第一定律是什么？"), state, hint_level=None, context_notes=None
    )
    joined = "\n".join(str(m.get("content", "")) for m in messages)

    assert _head("physics") in joined, "应注入物理人格"
    assert _head("math") not in joined, "绝不允许多余注入数学人格（串台）"


async def test_personality_follows_current_subject_without_scratch_id() -> None:
    """scratch 里没有 subject_id 时，应按当前学科名兜底解析。"""
    from agents.teacher.agent import TeacherAgent
    from core.state.models import StudentTurn
    from core.state.session import SessionState

    state = SessionState(session_id="x2", student_id="s2")
    state.scratch.update(grade="初中", subject="化学")
    messages = TeacherAgent()._build_messages(
        StudentTurn(text="什么是氧化反应？"), state, hint_level=None, context_notes=None
    )
    joined = "\n".join(str(m.get("content", "")) for m in messages)
    assert _head("chemistry") in joined


def test_sessions_do_not_share_personality() -> None:
    """同一个进程里两个不同学科的会话，各自拿到的必须是本学科人格。"""
    from agents.teacher.agent import TeacherAgent
    from core.state.models import StudentTurn
    from core.state.session import SessionState

    agent = TeacherAgent()
    heads: dict[str, str] = {}
    for subject, subject_id in (("数学", "math"), ("生物", "biology"), ("政治", "politics")):
        state = SessionState(session_id=f"s-{subject_id}", student_id="s")
        state.scratch.update(grade="初中", subject=subject, subject_id=subject_id)
        messages = agent._build_messages(
            StudentTurn(text="讲讲这个知识点"), state, hint_level=None, context_notes=None
        )
        joined = "\n".join(str(m.get("content", "")) for m in messages)
        heads[subject_id] = joined

    for subject_id in ("math", "biology", "politics"):
        mine = _head(subject_id)
        assert mine in heads[subject_id], f"{subject_id} 未拿到本学科人格"
        for other in ("math", "biology", "politics"):
            if other != subject_id:
                assert _head(other) not in heads[subject_id], f"{subject_id} 串到了 {other}"


def test_teacher_for_subject_rejects_cross_subject_key() -> None:
    """传错学科的老师 key 时应丢弃（形象/音色/人格一起防串台）。"""
    from config.teachers import teacher_for_subject

    # 物理课传了数学的老师 key -> 不能把数学老师整套带进来
    prof = teacher_for_subject("物理", "math_strict")
    assert prof is None or prof.subject == "物理"
    assert prof is None or prof.key != "math_strict"


def test_teacher_for_subject_accepts_same_subject_variant() -> None:
    """同科变体（math_strict）必须正常命中，不能被防串台逻辑误伤。"""
    from config.teachers import teacher_for_subject

    prof = teacher_for_subject("数学", "math_strict")
    assert prof is not None and prof.key == "math_strict"
    default = teacher_for_subject("数学")
    assert default is not None and default.subject == "数学"


# ---------------------------------------------------------------- 开课问候多样化
def test_every_subject_has_multiple_greetings() -> None:
    """11 科都要有多条问候（产品要求：开场白不能一直同一句）。"""
    from config.teachers import DEFAULT_GREETINGS, GENERIC_GREETINGS

    for item in SUBJECTS:
        pool = DEFAULT_GREETINGS.get(item["key"], ())
        assert len(pool) >= 4, f"{item['name']} 的问候池过少（{len(pool)} 条）"
        assert len(set(pool)) == len(pool), f"{item['name']} 的问候池有重复条目"
    assert len(GENERIC_GREETINGS) >= 3, "通用兜底池也要多条"


def test_teacher_greeting_pool_merges_variant_and_subject() -> None:
    """问候池 = 变体专属 greeting + 学科内置池（去重合并）。"""
    from config.teachers import DEFAULT_GREETINGS, teacher_for_subject

    prof = teacher_for_subject("数学", "math_strict")
    assert prof is not None
    pool = prof.greeting_pool()
    if prof.greeting.strip():
        assert prof.greeting.strip() in pool, "变体专属问候仍要参与随机（风格不被抹掉）"
    for item in DEFAULT_GREETINGS["math"]:
        assert item in pool, "学科内置池应并入"
    assert len(pool) == len(set(pool)) >= 5, "合并后应去重且足够多"


async def test_course_greeting_varies_across_sessions() -> None:
    """连续开课从池里随机取问候：多次调用不应总是同一句。"""
    from agents.teacher.agent import TeacherAgent
    from models.llm import MockLLM

    teacher = TeacherAgent(llm=MockLLM())
    speeches = {teacher.compose_course_greeting("数学", "math_strict").speech for _ in range(40)}
    assert len(speeches) >= 2, "问候语必须多样化（池内随机）"


def test_generic_greeting_pool_varies() -> None:
    """无档案时的通用池同样多条随机（每次开课都是同一句模板的问题一并解决）。"""
    from core.course_session import greeting_text

    outs = {greeting_text("语文") for _ in range(40)}
    assert len(outs) >= 2
