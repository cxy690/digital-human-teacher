"""任务1 专项测试：学科元数据 + 辅导课阶段/计时配置。

覆盖：
* ``config/subjects.py``：学科清单完整性、key/名称唯一、覆盖小学到初中、key/中文名归一化；
* ``config/constants.py``：阶段顺序/中文名/时长占比（和为 1）、教师形象状态与校验函数；
* ``config/settings.py``：``course_*`` 默认值与 ``core/course_session`` 的时间轴联动。
"""
from __future__ import annotations

from config.constants import (
    TUTOR_MOODS,
    TUTOR_PHASE_LABELS,
    TUTOR_PHASE_ORDER,
    TUTOR_PHASE_RATIOS,
    ClassPhase,
    is_valid_mood,
)
from config.settings import Settings, settings
from config.subjects import SUBJECT_BY_KEY, SUBJECT_BY_NAME, SUBJECTS, resolve_subject
from core.course_session import duration_seconds, phase_at, tutorial_plan

REQUIRED_FIELDS = ("key", "name", "emoji", "desc", "grades", "color")


# ---------------------------------------------------------------- 学科元数据
def test_subjects_metadata_is_complete() -> None:
    assert len(SUBJECTS) >= 10
    keys = [s["key"] for s in SUBJECTS]
    names = [s["name"] for s in SUBJECTS]
    assert len(set(keys)) == len(keys), "学科 key 不能重复"
    assert len(set(names)) == len(names), "学科名不能重复"
    for subject in SUBJECTS:
        for field in REQUIRED_FIELDS:
            assert str(subject.get(field) or "").strip(), f"{subject.get('name')} 缺字段 {field}"
        assert any(g in subject["grades"] for g in ("小学", "初中")), f"{subject['name']} 年级段异常"


def test_subjects_cover_primary_and_junior() -> None:
    all_grades = " ".join(s["grades"] for s in SUBJECTS)
    assert "小学" in all_grades and "初中" in all_grades
    for name in ("语文", "数学", "英语", "物理", "化学", "生物", "政治", "道德与法治", "地理", "历史"):
        assert name in SUBJECT_BY_NAME, f"缺学科：{name}"


def test_resolve_subject_normalizes_key_and_name() -> None:
    assert resolve_subject("math") == "数学"        # key -> 中文名
    assert resolve_subject("数学") == "数学"        # 中文名 -> 原样
    assert resolve_subject("  数学 ") == "数学"      # 去首尾空格
    assert resolve_subject("") == "综合"            # 空 -> 默认学科
    assert resolve_subject(None) == "综合"
    assert resolve_subject("编程") == "编程"         # 未知值原样透传，不报错
    assert SUBJECT_BY_KEY["math"]["name"] == "数学"


# ---------------------------------------------------------------- 阶段配置
def test_tutor_phase_config_consistent() -> None:
    assert TUTOR_PHASE_ORDER == [
        ClassPhase.OPENING,
        ClassPhase.LECTURE,
        ClassPhase.INTERACTION,
        ClassPhase.SUMMARY,
    ]
    for phase in TUTOR_PHASE_ORDER:
        assert phase in TUTOR_PHASE_LABELS, f"{phase} 缺中文名"
    # 时长占比之和为 1（允许浮点误差），保证阶段条能铺满
    assert abs(sum(TUTOR_PHASE_RATIOS.values()) - 1.0) < 1e-9
    # 本仓库的课堂节奏（与 COURSE_QUIZ_INTERVAL_MINUTES=30 对齐）：
    # 40 分钟 -> 讲解 30 分钟 + 互动出题 10 分钟；问候热身并入开课招呼、
    # 总结答疑并入出题回合的收尾对话，两者不再单独占阶段时长（故占比为 0）。
    assert TUTOR_PHASE_RATIOS[ClassPhase.LECTURE] == 0.75
    assert TUTOR_PHASE_RATIOS[ClassPhase.INTERACTION] == 0.25
    assert TUTOR_PHASE_RATIOS[ClassPhase.OPENING] == 0.0
    assert TUTOR_PHASE_RATIOS[ClassPhase.SUMMARY] == 0.0
    # 出题节奏与阶段占比必须一致：「讲解满多少分钟才出第一题」就是讲解阶段的时长
    # （40 分钟 * 0.75 = 30 分钟，与 COURSE_QUIZ_INTERVAL_MINUTES 对齐）。
    assert (
        settings.course_duration_minutes * TUTOR_PHASE_RATIOS[ClassPhase.LECTURE]
        == settings.course_quiz_interval_minutes
    )
    assert TUTOR_PHASE_LABELS[ClassPhase.INTERACTION] == "互动出题"
    assert TUTOR_PHASE_LABELS[ClassPhase.ENDED] == "已结束"


def test_tutor_moods_and_validator() -> None:
    assert set(TUTOR_MOODS) == {"idle", "greeting", "speaking", "thinking", "praising", "waiting"}
    for mood in TUTOR_MOODS:
        assert is_valid_mood(mood) is True
    # 非法状态名必须被拦下，避免前后端状态漂移
    assert is_valid_mood("dancing") is False
    assert is_valid_mood("") is False
    assert is_valid_mood(None) is False


# ---------------------------------------------------------------- 设置与时间轴
def test_course_setting_defaults() -> None:
    """忽略 .env，校验代码里写死的默认值。"""
    defaults = Settings(_env_file=None)  # type: ignore[call-arg]
    assert defaults.course_enable is True
    assert defaults.course_duration_minutes == 40
    assert defaults.course_quiz_enable is True


def test_course_settings_wire_into_timeline() -> None:
    assert settings.course_duration_minutes >= 1
    total = duration_seconds()
    assert total == settings.course_duration_minutes * 60

    plan = tutorial_plan()
    assert [p["phase"] for p in plan] == [p.value for p in TUTOR_PHASE_ORDER]
    assert plan[0]["start_seconds"] == 0
    assert plan[-1]["end_seconds"] == total
    # 阶段首尾相接、时长之和等于总时长
    for prev, cur in zip(plan, plan[1:]):
        assert prev["end_seconds"] == cur["start_seconds"]
    assert sum(p["duration_seconds"] for p in plan) == total
    # 时间 -> 阶段映射。
    # 本仓库 OPENING 占比为 0（问候热身并入开课招呼），因此开课即进入讲解阶段；
    # ENDED 是「超出课时」的哨兵阶段。
    assert phase_at(0) is ClassPhase.LECTURE
    assert phase_at(total + 1) is ClassPhase.ENDED
