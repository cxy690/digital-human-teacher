"""出题链路（teacher.quiz）：**真题库优先** + LLM 降级 的单测。

钉住四件事：
1. 年级归一化 —— 登录用户 ``users.grade`` / 会话 ``scratch["grade"]`` 的写法差异
   （"初一" / "八年级下册" / "7" / "初中" / 范围表达），归一错会导致按错年级选题；
2. 真题行 → QuizItem 字段转换 —— 库里实测两种选项形态（options JSON / 选项内嵌题干）；
3. ``generate_quiz`` 三级降级 —— 真题库 → LLM → 规则兜底，任一环故障都不中断；
4. ``teacher.quiz`` 图把会话年级传给题源（真题按学生年级选）。

测试隔离：conftest 全局 ``EXERCISE_BANK_ENABLE=false``（不连真库）；
本文件自行开启并 ``exercise_bank.set_picker(fake)`` 注入假题源。
"""
from __future__ import annotations

import pytest

from agents.lecturer.agent import LecturerAgent
from agents.supervisor import Supervisor
from agents.teacher.agent import TeacherAgent
from config.settings import settings
from core import exercise_bank
from core.state.models import QuizItem
from core.state.session import SessionState
from models.asr import MockASR
from models.llm import MockLLM
from models.tts import MockTTS


# ---------------------------------------------------------------- 夹具
@pytest.fixture()
def bank(monkeypatch: pytest.MonkeyPatch):
    """开启题库 + 注入「查不到题」的假题源；yield 记录每次调用的参数。"""

    def _teardown() -> None:
        exercise_bank.set_picker(None)

    monkeypatch.setattr(settings, "exercise_bank_enable", True)
    calls: list[dict] = []

    def fake(**kw):
        calls.append(kw)
        return None

    exercise_bank.set_picker(fake)
    try:
        yield calls
    finally:
        _teardown()


def _row(**over) -> dict:
    """一条 ``exercise_item`` 行的典型形态（选项内嵌题干，实数库样本）。"""
    row = {
        "exercise_id": "EX-7-0031263107-0012",
        "grade": 7,
        "question_type": "choice",
        "stem_plain": "下列事件不可能发生的是(   )\nA．打开电视机\nB．明天会下雨\nC．光比声音快\nD．若实数，则",
        "options": None,
        "answer_text": "D",
    }
    row.update(over)
    return row


def _supervisor(grade: str = "") -> tuple[Supervisor, SessionState]:
    state = SessionState(session_id="quiz", student_id="stu")
    state.subject = "数学"
    state.scratch.update(subject="数学", subject_id="math")
    state.course_active = True
    if grade:
        state.scratch["grade"] = grade
    supervisor = Supervisor(
        state,
        teacher=TeacherAgent(llm=MockLLM()),
        lecturer=LecturerAgent(llm=MockLLM()),
    )
    supervisor.tts = MockTTS()
    supervisor.asr = MockASR()
    return supervisor, state


# ---------------------------------------------------------------- 1. 年级归一
def test_normalize_grade_variants() -> None:
    """各写法都要落到 7/8/9；范围表达与其它学段不臆测（None）。"""
    assert exercise_bank.normalize_grade("初一") == 7
    assert exercise_bank.normalize_grade("初二") == 8
    assert exercise_bank.normalize_grade("初三") == 9
    assert exercise_bank.normalize_grade("七年级") == 7
    assert exercise_bank.normalize_grade("八年级下册") == 8
    assert exercise_bank.normalize_grade("九年级") == 9
    assert exercise_bank.normalize_grade("7") == 7

    assert exercise_bank.normalize_grade("") is None
    assert exercise_bank.normalize_grade("初中") is None                 # 无具体年级
    assert exercise_bank.normalize_grade("小学一年级到初三") is None     # 范围不猜
    assert exercise_bank.normalize_grade("一年级") is None               # 小学不进初中题库


def test_supported_subject_only_math() -> None:
    assert exercise_bank.is_supported_subject("数学")
    assert exercise_bank.is_supported_subject("math")
    assert not exercise_bank.is_supported_subject("物理")


def test_pick_skips_unsupported_subject(bank: list[dict]) -> None:
    """非数学学科：直接返回 None 且**不问假题源**（题库只覆盖数学的契约）。"""
    assert exercise_bank.pick(subject="语文", grade="初一") is None
    assert exercise_bank.pick(subject="物理") is None
    assert bank == []


def test_pick_disabled_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """总开关关闭时一律不选题（等价旧行为：纯 LLM 出题）。"""
    monkeypatch.setattr(settings, "exercise_bank_enable", False)
    assert exercise_bank.pick(subject="数学", grade="初一") is None


# ---------------------------------------------------------------- 2. 行 → 字段转换
def test_to_quiz_fields_choice_with_options_json() -> None:
    """options JSON（[{"k","v"}]）→ 展示用选项文本（不带字母前缀，前端自动加）；答案/提示齐备。"""
    fields = exercise_bank.to_quiz_fields(
        {
            "question_type": "choice",
            "stem_plain": "在此案中肯定的作案对象是(   )",
            "options": '[{"k": "A", "v": "嫌疑犯A"}, {"k": "B", "v": "嫌疑犯B"}]',
            "answer_text": "A",
        },
        topic="逻辑推理",
    )
    assert fields["type"] == "choice"
    assert fields["options"] == ["嫌疑犯A", "嫌疑犯B"]   # 前端渲染时加 "A. " 前缀
    assert fields["answer"] == "A"
    assert "逻辑推理" in fields["hint"]


def test_to_quiz_fields_fill_strips_newlines() -> None:
    """填空题 → blank；题干换行折叠为空格（播报/白板都不吃换行）。"""
    fields = exercise_bank.to_quiz_fields(
        {
            "question_type": "fill",
            "stem_plain": "两直线平行，同旁内角相等，\n这个事件是__________事件。",
            "options": None,
            "answer_text": "可能",
        }
    )
    assert fields["type"] == "blank"
    assert "\n" not in fields["question"]
    assert fields["answer"] == "可能"
    assert fields["options"] == []


def test_to_quiz_fields_solve_maps_to_short() -> None:
    fields = exercise_bank.to_quiz_fields(
        {"question_type": "solve", "stem_plain": "解方程组……", "options": None, "answer_text": "x=2"}
    )
    assert fields["type"] == "short"


# ---------------------------------------------------------------- 2b. 答案质检
def test_answer_ok_rejects_formula_placeholder() -> None:
    """答案含公式占位符（⟦Fxx⟧）：判分无意义，必须拒（实测 choice 364 / fill 936 条）。"""
    assert exercise_bank._answer_ok(
        {"question_type": "fill", "answer_text": "⟦F44⟧；"}
    ) is None


def test_answer_ok_normalizes_choice_letter() -> None:
    """选择题答案「（D）」→「D」；解析段错位（非字母）拒。"""
    row = {"question_type": "choice", "answer_text": "（D）"}
    ok = exercise_bank._answer_ok(row)
    assert ok is not None and ok["answer_text"] == "D"
    assert exercise_bank._answer_ok({"question_type": "choice", "answer_text": "C，"}) is None
    assert exercise_bank._answer_ok(
        {"question_type": "choice", "answer_text": "BD=4."}
    ) is None


def test_answer_ok_fill_rules() -> None:
    """填空题：解析段（（9分）…）/超长/换行 拒；短短语放行。"""
    assert exercise_bank._answer_ok(
        {"question_type": "fill", "answer_text": "（9分）\n（1）5.1499×106m"}
    ) is None
    assert exercise_bank._answer_ok(
        {"question_type": "fill", "answer_text": "必然"}
    ) is not None
    assert exercise_bank._answer_ok(
        {"question_type": "fill", "answer_text": "解：(1) 由题意得，设该直线为 y=kx+b，代入两点..."[:60]}
    ) is None


def test_pick_walks_candidates_until_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """候选逐条质检：前两条是噪声题（解析错位/占位符）→ 取第三条干净题。"""
    monkeypatch.setattr(settings, "exercise_bank_enable", True)
    exercise_bank.set_picker(None)
    dirty1 = _row(exercise_id="EX-bad-1", answer_text="（9分）见解析全文……", question_type="fill")
    dirty2 = _row(exercise_id="EX-bad-2", answer_text="⟦F44⟧；", question_type="fill")
    good = _row(exercise_id="EX-ok-3", answer_text="D")
    monkeypatch.setattr(exercise_bank, "_query_candidates", lambda g, t: [dirty1, dirty2, good])
    pick = exercise_bank.pick(subject="数学", grade="初一", topic="随机事件")
    assert pick is not None and pick["exercise_id"] == "EX-ok-3"


# ---------------------------------------------------------------- 2c. 选择题判分
async def test_evaluate_quiz_letter_vs_option_text() -> None:
    """真题库答案=字母，学生点按钮提交=选项原文：两侧归一到选项字母，判对。"""
    teacher = TeacherAgent(llm=MockLLM())
    quiz = QuizItem(
        qid="EX-test",
        type="choice",
        question="在此案中肯定的作案对象是(   )",
        options=["嫌疑犯A", "嫌疑犯B", "嫌疑犯C"],
        answer="A",
    )
    # 学生点按钮提交选项原文 / 手输字母 / 输「A. 嫌疑犯A」——三种都要判对
    for submitted in ("嫌疑犯A", "A", "A．嫌疑犯A", "（A）"):
        verdict = await teacher.evaluate_quiz(quiz, submitted)
        assert verdict["correct"] is True, submitted
    # 选错：字母与原文都要判错
    for submitted in ("嫌疑犯B", "B"):
        verdict = await teacher.evaluate_quiz(quiz, submitted)
        assert verdict["correct"] is False, submitted


async def test_evaluate_answer_without_options_unchanged() -> None:
    """非选择题（无 options）不进入选项归一链，维持既有判分行为。"""
    teacher = TeacherAgent(llm=MockLLM())
    verdict = await teacher.evaluate_answer("这个事件是什么类型？", "必然", "必然")
    assert verdict["correct"] is True


# ---------------------------------------------------------------- 3. 三级降级
async def test_generate_quiz_prefers_exercise_bank(bank: list[dict]) -> None:
    """真题库有题：直接用真题（qid=exercise_id），不走 LLM。"""
    exercise_bank.set_picker(lambda **kw: _row())
    teacher = TeacherAgent(llm=MockLLM())
    quiz = await teacher.generate_quiz("数学", "随机事件", grade="初一")
    assert quiz.qid == "EX-7-0031263107-0012"
    assert quiz.type == "choice"
    assert quiz.answer == "D"
    assert quiz.options == []          # 选项内嵌题干：不重复拆出
    assert "随机事件" in quiz.topic


async def test_generate_quiz_falls_back_to_llm_when_bank_empty(bank: list[dict]) -> None:
    """题库无匹配（假题源返回 None）：回退 LLM 路径（MockLLM → 规则兜底题）。"""
    teacher = TeacherAgent(llm=MockLLM())
    quiz = await teacher.generate_quiz("数学", "三角形内角和", grade="初一")
    assert quiz.qid.startswith("quiz-")     # LLM / 兜底路径的时间戳 qid
    assert quiz.question
    assert bank and bank[0]["grade"] == "初一"   # 题库确实被问过（带年级）


async def test_generate_quiz_survives_bank_error(bank: list[dict]) -> None:
    """题库故障（题源抛异常）：降级不中断，回退 LLM。"""
    def boom(**kw):
        raise RuntimeError("mysql down")

    exercise_bank.set_picker(boom)
    teacher = TeacherAgent(llm=MockLLM())
    quiz = await teacher.generate_quiz("数学", "随机事件", grade="初一")
    assert quiz.qid.startswith("quiz-")


async def test_generate_quiz_skips_bank_for_other_subject(bank: list[dict]) -> None:
    """非数学学科：不问题库，直接走 LLM。"""
    teacher = TeacherAgent(llm=MockLLM())
    quiz = await teacher.generate_quiz("语文", "阅读理解")
    assert quiz.qid.startswith("quiz-")
    assert bank == []


# ---------------------------------------------------------------- 4. 图传年级
async def test_quiz_graph_passes_session_grade(bank: list[dict]) -> None:
    """会话年级（scratch["grade"]）要随出题链路传给题源（真题按年级选）。"""
    supervisor, state = _supervisor(grade="初一")
    exercise_bank.set_picker(lambda **kw: (bank.append(kw), _row())[1])

    response = await supervisor.on_quiz_request("随机事件")
    assert response is not None and response.speech
    assert bank and bank[0]["grade"] == "初一"
    assert bank[0]["topic"] == "随机事件"
    assert state.quiz is not None and state.quiz["qid"] == "EX-7-0031263107-0012"
    assert (state.pending_question or {}).get("kind") == "quiz"


async def test_quiz_graph_without_grade_still_works(bank: list[dict]) -> None:
    """未登录 / 未填年级：grade 传空串，题源按「不限年级」处理，出题照常。"""
    supervisor, state = _supervisor()
    exercise_bank.set_picker(lambda **kw: (bank.append(kw), _row())[1])

    response = await supervisor.on_quiz_request("随机事件")
    assert response is not None
    assert bank and bank[0]["grade"] == ""
    assert state.pending_question is not None


# ---------------------------------------------------------------- 5. 出题体验
def test_topic_hits_rejects_unrelated_random_pick() -> None:
    """有主题词时只保留命中行；一道都没命中 → 空（上层回退 LLM，不再随机抽无关题）。

    回归点：讲「配方法」时因随机抽样出到「同位角」的真题 = 驴唇不对马嘴。
    """
    rows = [
        {"exercise_id": "hit", "topic_hit": 1},
        {"exercise_id": "miss", "topic_hit": 0},
    ]
    assert [r["exercise_id"] for r in exercise_bank._topic_hits(rows, "配方法")] == ["hit"]
    assert exercise_bank._topic_hits([{"exercise_id": "miss", "topic_hit": 0}], "配方法") == []
    # 无主题词：原样返回（泛化练习随机选题是合理的）
    assert len(exercise_bank._topic_hits(rows, "")) == 2


async def test_quiz_announce_has_transition_lead(bank: list[dict]) -> None:
    """出题播报要带过渡语（先说清围绕什么、为什么做题），不能突然甩题。"""
    supervisor, state = _supervisor()
    exercise_bank.set_picker(lambda **kw: (bank.append(kw), _row())[1])
    state.scratch["slide_page_title"] = "配方法"       # 课件场景：刚讲完这一页

    response = await supervisor.on_quiz_request("配方法")
    assert response is not None
    assert "配方法" in response.speech, "过渡语应带上刚讲的页标题"
    assert "检验" in response.speech
    assert response.speech.index("检验") < response.speech.index("下列事件"), "过渡语应在题目前面"
