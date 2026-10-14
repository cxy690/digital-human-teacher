"""教师 Skill 自我更新（经验沉淀）单测：存储 / 复盘 / 注入 / 回合触发。"""
from __future__ import annotations

import asyncio
import random

# 生产链路那份总控（core/harness.py 导入的就是它；见 test_supervisor.py 的哨兵用例）
from agents.supervisor import Supervisor
from config.settings import settings
from core.skills.experience import (
    ExperienceStore,
    SkillReflector,
    build_experience_prompt,
    gain_from_verdict,
    record_turn_outcome,
    render_experience,
)
from agents.teacher.agent import TeacherAgent
from core.state.models import AIResponse, StudentTurn
from core.state.session import SessionState
from models.llm import MockLLM


# --------------------------------------------------------------- 存储
def test_observe_creates_and_merges_similar(tmp_path) -> None:
    store = ExperienceStore(root=tmp_path)
    first = store.observe("math", "pitfall", "学生常把移项符号搞错", confidence=0.8)
    assert first is not None and first.hits == 1

    # 标点/空白不同、措辞几乎一样 -> 视为同一条，累加命中而不是新增
    again = store.observe("math", "pitfall", "学生常把移项符号搞错。", confidence=0.9)
    assert again is not None and again.id == first.id
    assert again.hits == 2
    assert again.confidence == 0.9
    assert len(store.all_items("math")) == 1


def test_observe_rejects_noise(tmp_path) -> None:
    store = ExperienceStore(root=tmp_path)
    assert store.observe("math", "pitfall", "短") is None                  # 太短
    assert store.observe("math", "pitfall", "字" * 200) is None            # 太长
    assert store.observe("math", "pitfall", "   ") is None                 # 空白
    assert store.observe("", "pitfall", "学生常把移项符号搞错") is None      # 没有老师 key
    assert store.all_items("math") == []


def test_retire_and_restore_is_rollback(tmp_path) -> None:
    store = ExperienceStore(root=tmp_path)
    item = store.observe("math", "pitfall", "去分母时容易漏乘不含分母的项", confidence=0.9)
    assert item is not None and len(store.active("math")) == 1

    assert store.retire("math", item.id) is True
    assert store.active("math") == []
    assert len(store.all_items("math")) == 1, "停用只是不再注入，条目仍可追溯"

    assert store.restore("math", item.id) is True
    assert len(store.active("math")) == 1


#: 12 条**互不相似**的经验（相似度阈值 0.85，措辞太像会被合并成同一条，测不出条数）
_DISTINCT_NOTES = [
    "移项必须变号这是最常见的错处",
    "去分母时容易漏乘不含分母的项",
    "讲浮力时用船为什么能漂这个类比效果好",
    "合并同类项前要先确认字母部分完全相同",
    "化学方程式配平先配单质再配化合物",
    "英语时态题先找时间状语再定时态",
    "文言实词要结合上下文推断词义",
    "读图题先看图名图例再看坐标轴",
    "细胞结构用类比工厂车间来讲更直观",
    "历史事件要按时间线串成因果链",
    "声调标注要按韵母顺序判断标在哪",
    "解应用题先把等量关系写出来再列式",
]


def test_no_eviction_by_default(tmp_path) -> None:
    """默认**不限**：学到多少留多少 —— 系统不会自己把学过的经验停掉。

    为什么这条值得钉住：自动学习产物一旦被「封顶 + 淘汰」，就等于系统替你决定
    「以前学的可以忘了」；而「这一次该注入哪几条」另有 ``INJECT_LIMIT`` + 打分排序来解决，
    不需要靠丢内容来腾位置。
    """
    store = ExperienceStore(root=tmp_path)
    assert store.max_items == 0, "默认必须是「不限」"

    for note in _DISTINCT_NOTES:
        store.observe("math", "pitfall", note, confidence=0.9)

    assert len(store.active("math")) == len(_DISTINCT_NOTES), "默认不该淘汰任何条目"
    assert len(store.all_items("math")) == len(_DISTINCT_NOTES)
    assert store.stats()["max_items_per_teacher"] == 0


def test_explicit_zero_means_unlimited(tmp_path, monkeypatch) -> None:
    """显式传 ``max_items=0`` 必须解释为「不限」，不能被 ``or`` 回落成配置里的正数。

    （旧写法 ``self._max_items or 配置值`` 会把 0 当成「没传」，于是「不限」变「限额」。）
    """
    monkeypatch.setattr(settings, "skill_experience_max_items", 5)
    assert ExperienceStore(root=tmp_path, max_items=0).max_items == 0
    assert ExperienceStore(root=tmp_path).max_items == 5, "不传时才读配置"


def test_max_items_evicts_lowest_score(tmp_path) -> None:
    store = ExperienceStore(root=tmp_path, max_items=2)
    store.observe("math", "pitfall", "低分经验甲", confidence=0.3)
    store.observe("math", "pitfall", "低分经验乙", confidence=0.4)
    store.observe("math", "analogy", "高分经验丙", confidence=0.9)

    active = {it.text for it in store.active("math")}
    assert len(active) == 2
    assert "高分经验丙" in active
    # 被停用的那条仍在文件里，可人工恢复
    assert len(store.all_items("math")) == 3


def test_jsonl_is_append_only_and_survives_bad_lines(tmp_path) -> None:
    store = ExperienceStore(root=tmp_path)
    store.observe("math", "pitfall", "学生常把移项符号搞错", confidence=0.8)
    store.observe("math", "pitfall", "学生常把移项符号搞错", confidence=0.8)
    path = store.path_for("math")

    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2, "每次变更追加一行快照（同 id 后写的生效）"

    path.write_text(path.read_text(encoding="utf-8") + "{坏行\n", encoding="utf-8")
    fresh = ExperienceStore(root=tmp_path)
    assert len(fresh.all_items("math")) == 1, "坏行应被跳过而不是让整份经验失效"


# --------------------------------------------------------------- 注入
def test_experience_prompt_groups_by_kind() -> None:
    store = ExperienceStore()
    from core.skills.experience.models import ExperienceItem

    items = [
        ExperienceItem(id="a", teacher_key="math", kind="pitfall", text="移项要变号"),
        ExperienceItem(id="b", teacher_key="math", kind="analogy", text="移项像天平两边搬东西"),
    ]
    text = render_experience(items, name="数学老师")
    assert "数学老师经验" in text
    assert "学生易错点" in text and "有效类比" in text
    assert "- 移项要变号" in text and "- 移项像天平两边搬东西" in text
    assert render_experience([], name="数学老师") == ""


def test_injection_is_read_only_and_bounded(monkeypatch, tmp_path) -> None:
    """注入只读经验，且受 inject_limit 限制；人格本体不受任何影响。

    这里**刻意关掉 bandit**（``SKILL_EXPERIENCE_BANDIT_ENABLE=false``）：
    静态排序是关掉它之后的**回退契约**，得有测试守着；
    开着时的行为（按实测收益采样）见本文件「L2 bandit」那一组。
    """
    monkeypatch.setattr(settings, "skill_experience_bandit_enable", False)
    store = ExperienceStore(root=tmp_path)
    notes = ["移项忘记变号", "去分母漏乘常数项", "负号看错导致整题全错"]
    for i, note in enumerate(notes, start=1):
        store.observe("math", "pitfall", note, confidence=0.5 + i * 0.1)
    assert len(store.active("math")) == 3, "三条经验彼此差异明显，不应被去重合并"

    monkeypatch.setattr(settings, "skill_experience_inject_limit", 2)
    text = build_experience_prompt("math", store=store)
    assert text.count("\n- ") == 2, "只注入配置允许的条数"
    # 关闭 bandit 时按 score 降序取（置信度高的在后两条之上）
    assert "负号看错导致整题全错" in text
    assert "移项忘记变号" not in text


# --------------------------------------------------------------- L2 bandit（按实测收益定序）
def test_bandit_prefers_proven_experience(monkeypatch, tmp_path) -> None:
    """战绩好的经验**显著**更容易被注入（利用）。

    这是 L2 要解决的那件事：旧排序按 ``hits × confidence``（被提到几次、模型多自信），
    两条经验在这两项上完全一样，谁上全看运气 —— 而它们一个每次都好用、一个从没管用过。
    """
    monkeypatch.setattr(settings, "skill_experience_bandit_enable", True)
    store = ExperienceStore(root=tmp_path)
    good = store.observe("math", "pitfall", "移项一定要记得变号", confidence=0.5)
    bad = store.observe("math", "pitfall", "这个说法其实没什么用", confidence=0.5)
    assert good is not None and bad is not None
    for _ in range(20):
        store.record_outcome("math", [good.id], gain=1.0)
        store.record_outcome("math", [bad.id], gain=0.0)

    rng = random.Random(7)
    picked = [store.select("math", limit=1, rng=rng)[0].id for _ in range(60)]
    wins = picked.count(good.id)
    assert wins >= 55, f"战绩好的应稳定胜出，实际 {wins}/60"


def test_bandit_explores_untried_experience(monkeypatch, tmp_path) -> None:
    """从没被注入过的经验**也有机会被试到**（探索）—— 静态排序永远做不到这件事。

    一条新经验 ``pulls=0``，后验很宽：它偶尔会盖过老兵，于是拿到一次「证明自己」的机会。
    少了这个，经验表就会退化成「先到的几条一直霸占名额」。
    """
    monkeypatch.setattr(settings, "skill_experience_bandit_enable", True)
    store = ExperienceStore(root=tmp_path)
    veteran = store.observe("math", "pitfall", "老兵经验：已经用了很多次", confidence=0.5)
    rookie = store.observe("math", "pitfall", "新来的经验：还没上过场", confidence=0.5)
    assert veteran is not None and rookie is not None
    # 老兵战绩**不完美**（12 胜 8 负）：新兵仍应偶尔盖过它
    for _ in range(12):
        store.record_outcome("math", [veteran.id], gain=1.0)
    for _ in range(8):
        store.record_outcome("math", [veteran.id], gain=0.0)

    rng = random.Random(11)
    picked = [store.select("math", limit=1, rng=rng)[0].id for _ in range(120)]
    trials = picked.count(rookie.id)
    assert trials >= 5, f"新经验应有机会被试到（探索），实际只上了 {trials}/120"
    assert picked.count(veteran.id) > trials, "老兵总体仍应更常出场（利用）"


def test_same_turn_injects_the_same_experience(monkeypatch, tmp_path) -> None:
    """同一个回合里装配两遍提示词，必须注入**同一批**经验。

    为什么这条重要：流式答疑会把「话术路」与「结构路」并行跑，两条路各装配一次提示词。
    采样是随机的 —— 不按轮次缓存的话，同一位老师、同一道题，两条路会拿到不同的经验，
    于是「说出去的话」和「写到黑板上的内容」对不上。
    """
    monkeypatch.setattr(settings, "skill_experience_bandit_enable", True)
    monkeypatch.setattr(settings, "skill_experience_inject_limit", 1)
    store = ExperienceStore(root=tmp_path)
    for i in range(6):
        store.observe("math", "pitfall", f"第 {i} 条互不相同的经验内容", confidence=0.5)

    state = SessionState(session_id="same-turn", student_id="s")
    first = build_experience_prompt("math", store=store, session=state)
    book = state.scratch["_experience_injected"]
    assert first and book["ids"], "第一次装配就该选中并记账"

    second = build_experience_prompt("math", store=store, session=state)
    assert second == first, "同回合第二次装配必须复用同一批（不能重新采样）"
    assert state.scratch["_experience_injected"]["ids"] == book["ids"]

    # 换一轮（speech_epoch 自增）：这才该重新选
    state.speech_epoch += 1
    build_experience_prompt("math", store=store, session=state)
    assert state.scratch["_experience_injected"]["epoch"] == state.speech_epoch


def test_record_outcome_counts_pulls_and_wins(monkeypatch, tmp_path) -> None:
    """收益回写：记的是「用了没有/管用没有」，而不是「被提到几次」。"""
    store = ExperienceStore(root=tmp_path)
    item = store.observe("math", "pitfall", "移项要变号，否则整题错", confidence=0.5)
    assert item is not None and item.pulls == 0 and item.wins == 0

    assert store.record_outcome("math", [item.id], gain=1.0) == 1
    assert store.record_outcome("math", [item.id], gain=0.0) == 1
    fresh = [it for it in store.load("math") if it.id == item.id][0]
    assert (fresh.pulls, fresh.wins) == (2, 1)
    assert fresh.last_gain == 0.0
    assert store.record_outcome("math", [], gain=1.0) == 0, "没注入过就没什么可记的"


def test_gain_from_verdict_skips_infra_failures() -> None:
    """裁决 -> 收益：**抛异常不记账**，否则机器抽风会让经验背黑锅。"""
    assert gain_from_verdict({"verdict": "pass"}) == 1.0
    assert gain_from_verdict({"verdict": "degraded"}) == 0.0
    assert gain_from_verdict({"verdict": "failed"}) == 0.0
    # 基础设施抖动（TTS 挂了 / 网络断了）：与经验好坏无关，不该给反馈
    assert gain_from_verdict({"verdict": "failed", "error": "ConnectionError: boom"}) is None
    assert gain_from_verdict({"verdict": "?"}) is None
    assert gain_from_verdict(None) is None


async def test_turn_verdict_feedback_reaches_experience_file(monkeypatch, tmp_path) -> None:
    """端到端：一次真实回合结束后，注入过的经验在**文件里**留下实测战绩。

    这条是「闭环真的闭上了」的唯一证据 —— 上面那些单测只能证明模块自己没问题，
    证明不了 ``handle_turn`` 真的调了它（本项目已经栽过这种跟头：经验层写好了、
    测试也在，但就是没接进链路，一直没人发现）。
    """
    monkeypatch.setattr(settings, "skill_experience_dir", str(tmp_path))
    monkeypatch.setattr(settings, "verdict_dir", str(tmp_path / "verdicts"))
    monkeypatch.setattr(settings, "skill_experience_bandit_enable", True)
    monkeypatch.setattr(settings, "skill_experience_feedback_enable", True)

    store = ExperienceStore(root=tmp_path)
    item = store.observe("math", "pitfall", "移项要变号，这是最常见的错", confidence=0.9)
    assert item is not None

    state = SessionState(session_id="l2-e2e", student_id="s1")
    state.scratch.update(grade="初中", subject="数学", subject_id="math", teacher="math")
    state.subject = "数学"
    supervisor = Supervisor(state, teacher=TeacherAgent(llm=MockLLM()))
    await supervisor.handle_turn(StudentTurn(text="解方程 3x + 5 = 20"))

    # 本回合确实注入了经验（否则这条用例等于什么都没测）
    assert state.scratch.get("_experience_injected") is None, "记完就该清账，避免下一轮重复计"
    fresh = [it for it in store.load("math") if it.id == item.id][0]
    assert fresh.pulls == 1, f"回合结束后应回写一次实测战绩，实际 pulls={fresh.pulls}"
    assert fresh.last_gain in (0.0, 1.0), f"last_gain 应是本回合的收益，实际 {fresh.last_gain}"
    assert fresh.wins in (0, 1), "wins 只该是「这一次管没管用」"


# --------------------------------------------------------------- 复盘
def test_reflector_parses_and_filters(tmp_path) -> None:
    class _JsonLLM(MockLLM):
        async def chat(self, messages, *, temperature: float = 0.3, max_tokens: int = 1024) -> str:
            return (
                '{"items": ['
                '{"kind": "pitfall", "text": "学生常把移项符号搞错", "confidence": 0.9},'
                '{"kind": "analogy", "text": "太低置信度被过滤", "confidence": 0.2},'
                '{"kind": "不认识的类型", "text": "类型非法也要过滤", "confidence": 0.9},'
                '{"kind": "must_cover", "text": "短", "confidence": 0.9}]}'
            )

    store = ExperienceStore(root=tmp_path)      # 落 tmp，绝不污染项目里的真实经验
    items = asyncio.run(
        SkillReflector(llm=_JsonLLM(), store=store).reflect(
            teacher_key="math", subject="数学", question="3x+5=20", answer="先移项再求解"
        )
    )
    # 低置信 / 非法类型 / 过短 都被过滤，只剩 1 条
    assert [i.text for i in items] == ["学生常把移项符号搞错"]
    assert len(store.active("math")) == 1


def test_reflector_never_raises_and_handles_empty() -> None:
    class _BrokenLLM(MockLLM):
        async def chat(self, messages, *, temperature: float = 0.3, max_tokens: int = 1024) -> str:
            raise RuntimeError("llm down")

    reflector = SkillReflector(llm=_BrokenLLM(), store=ExperienceStore())
    assert asyncio.run(reflector.reflect(teacher_key="math", answer="讲解内容")) == []
    # 非 JSON 输出 -> 空结果，不抛
    assert asyncio.run(
        SkillReflector(llm=MockLLM(), store=ExperienceStore()).reflect(
            teacher_key="math", answer="讲解内容"
        )
    ) == []
    # 没有 answer -> 直接跳过（不浪费一次模型调用）
    assert asyncio.run(reflector.reflect(teacher_key="math", answer="")) == []


# --------------------------------------------------------------- 回合触发
class _SpyReflector:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def reflect(self, **kwargs):
        self.calls.append(kwargs)
        return []


async def test_turn_end_triggers_background_reflection() -> None:
    """答疑结束应触发一次**后台**复盘（不阻塞返回，且带上该老师 key）。"""
    state = SessionState(session_id="e1", student_id="s1")
    state.subject = "数学"
    state.scratch["subject"] = "数学"
    supervisor = Supervisor(state)
    await supervisor.on_start()
    spy = _SpyReflector()
    supervisor.skill_reflector = spy  # type: ignore[assignment]

    await supervisor.handle_turn(StudentTurn(text="老师，这道方程题怎么解？"))
    for _ in range(20):
        if spy.calls:
            break
        await asyncio.sleep(0)

    assert len(spy.calls) == 1
    assert spy.calls[0]["teacher_key"] == "math"


async def test_reflection_disabled_by_switch() -> None:
    state = SessionState(session_id="e2", student_id="s2")
    state.subject = "数学"
    supervisor = Supervisor(state)
    await supervisor.on_start()
    spy = _SpyReflector()
    supervisor.skill_reflector = spy  # type: ignore[assignment]

    original = settings.skill_self_update_enable
    settings.skill_self_update_enable = False
    try:
        await supervisor.handle_turn(StudentTurn(text="老师，这道题怎么解？"))
        for _ in range(10):
            await asyncio.sleep(0)
        assert spy.calls == []
    finally:
        settings.skill_self_update_enable = original


def test_supervisor_exposes_ai_response_shape() -> None:
    """护栏：经验层不得改变答疑产物结构。"""
    response = AIResponse(speech="讲解", knowledge_points=["移项法则"], pitfalls=["漏乘"])
    assert set(response.model_dump()) == {"speech", "board_ops", "knowledge_points", "pitfalls", "meta"}


def test_experience_reaches_teacher_system_prompt(monkeypatch, tmp_path) -> None:
    """端到端：沉淀的经验必须真的出现在该老师的 system prompt 里。"""
    from agents.teacher.agent import TeacherAgent
    from core.skills.experience import get_experience_store

    monkeypatch.setattr(settings, "skill_experience_dir", str(tmp_path))
    store = get_experience_store()
    note = "移项一定要记得变号，这是这类题最常错的一步"
    assert store.observe("math", "pitfall", note, confidence=0.9) is not None

    state = SessionState(session_id="p1", student_id="s1")
    state.subject = "数学"
    state.scratch["subject"] = "数学"
    teacher = TeacherAgent()
    messages = teacher._build_messages(  # noqa: SLF001 - 直接验证提示词拼装
        StudentTurn(text="这道方程题怎么解？"),
        state,
        hint_level=None,
        context_notes=None,
    )
    system_text = "\n".join(
        str(m.get("content") or "") for m in messages if m.get("role") == "system"
    )
    assert note in system_text
    assert "学生易错点" in system_text
