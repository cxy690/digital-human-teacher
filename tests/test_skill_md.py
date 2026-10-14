"""教师技能 Markdown 事实源的契约测试（``.agents/skills/<学科>/SKILL.md``）。

钉住四件事：
1. **一科一份**：11 个学科各有 SKILL.md，且 key 与 ``config/subjects.py`` 对齐；
2. **内容齐全**：frontmatter 必填字段 + 九项行为小节都在，解析器能完整读回；
3. **md 是唯一事实源**：旧的 ``config/teacher_skills/<key>.py`` 内容模块不再存在
   （否则「改一处漏一处」的双份问题会悄悄回来）；
4. **两层同源**：风格层（``TeacherSkill``）与人格层（``TeacherPersonalitySkill``）
   从同一份 md 取值，且改 md 即生效（不必重启）。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from config.subjects import SUBJECTS
from config.teacher_skills import available_skills, load_skill
from config.teacher_skills.base import TeacherSkill
from config.teacher_skills.md import (
    BEHAVIOR_SECTIONS,
    REQUIRED_FRONT,
    SKILLS_DIR,
    SkillDoc,
    load_doc,
    parse_skill_md,
    reload_docs,
)
from core.skills.teacher_personality import SKILLS, SKILL_ORDER, get_skill_registry
from core.skills.teacher_personality.md_teacher import build_skill_class

# ---------------------------------------------------------------- 目录与数量


def test_one_skill_md_per_subject() -> None:
    """一科一份 SKILL.md，目录名 = 学科 key，与 SUBJECTS 完全对齐。"""
    keys = set(available_skills())
    assert keys == {str(s["key"]) for s in SUBJECTS}, "技能目录与学科清单不一致"
    for key in keys:
        assert (SKILLS_DIR / key / "SKILL.md").is_file()


def test_skills_dir_is_agents_folder() -> None:
    """事实源必须落在 .agents/skills/ 下（而不是散在 docs / config 里）。"""
    assert SKILLS_DIR.name == "skills"
    assert SKILLS_DIR.parent.name == ".agents"


def test_old_python_content_modules_are_gone() -> None:
    """旧的两层内容模块已清理：md 是唯一事实源，不允许双份并存。"""
    root = Path(__file__).resolve().parents[1]
    legacy_style = root / "config" / "teacher_skills"
    legacy_persona = root / "core" / "skills" / "teacher_personality"
    keys = {str(s["key"]) for s in SUBJECTS}

    assert not (legacy_style / "math.py").exists()
    for key in keys:
        assert not (legacy_style / f"{key}.py").exists(), f"config/teacher_skills/{key}.py 应已迁入 md"
        assert not (legacy_persona / f"{key}_teacher.py").exists(), f"{key}_teacher.py 应已迁入 md"


# ---------------------------------------------------------------- 字段与小节


def test_frontmatter_required_fields_present() -> None:
    for key in available_skills():
        doc = load_doc(key)
        assert doc is not None, f"{key}: SKILL.md 解析失败"
        for field in REQUIRED_FRONT:
            assert doc.text(field), f"{key}: frontmatter 缺 {field}"
        assert doc.skill_id.startswith(f"teacher.{key}.")
        assert doc.class_name.endswith("TeacherSkill")
        assert doc.depth in ("elementary", "standard", "advanced")
        assert 0.5 <= doc.speech_rate <= 2.0


def test_behavior_sections_all_present() -> None:
    """九项行为小节必须逐字齐全（解析器按 ``## 标题`` 精确匹配）。"""
    for key in available_skills():
        doc = load_doc(key)
        assert doc is not None
        for title in BEHAVIOR_SECTIONS:
            assert doc.section(title), f"{key}: 缺少小节「{title}」"
        assert doc.system_prompt(), f"{key}: 缺少「人格与教学」"
        assert "{where}" in doc.section("进场问候"), f"{key}: 进场问候缺少 {{where}} 占位"


def test_notes_are_not_injected() -> None:
    """给人看的注解（``>`` 引用块 / HTML 注释）不参与注入。"""
    doc = load_doc("math")
    assert doc is not None
    for text in [doc.system_prompt(), *doc.sections.values()]:
        for line in text.splitlines():
            assert not line.strip().startswith(">")
            assert "<!--" not in line


# ---------------------------------------------------------------- 两层同源


def test_style_layer_comes_from_same_md() -> None:
    """风格层字段与 md 完全一致（同源，不是各写一份）。"""
    for key in available_skills():
        doc = load_doc(key)
        skill = load_skill(key)
        assert doc is not None and skill is not None
        assert isinstance(skill, TeacherSkill)
        assert skill.key == key
        assert skill.depth == doc.depth
        assert skill.speech_rate == doc.speech_rate
        assert skill.teaching_style == doc.teaching_style
        assert skill.persona == doc.style_field("style_persona")
        assert skill.teaching_way == doc.style_field("teaching_way")
        assert skill.catchphrases == doc.items("catchphrases")
        assert skill.do == doc.items("do")
        assert skill.dont == doc.items("dont")
        assert "【说话风格】" in skill.prompt_text()


def test_personality_classes_come_from_md() -> None:
    """人格层：类名/身份/提示词都来自 md。"""
    for key in available_skills():
        doc = load_doc(key)
        assert doc is not None
        cls = get_skill_registry().get_for_subject(key)
        assert cls is not None and type(cls).__name__ == doc.class_name
        assert cls.skill_id == doc.skill_id
        assert cls.display_name == doc.display_name
        assert cls.build_system_prompt() == doc.system_prompt()
        assert cls.greeting() == doc.greeting()
        assert cls.greeting("分数例题") == doc.greeting("分数例题")


def test_skill_md_unknown_key_returns_none() -> None:
    assert load_doc("") is None
    assert load_doc("不存在的学科") is None
    assert load_skill("不存在的学科") is None


# ---------------------------------------------------------------- 解析器行为


def test_parse_requires_frontmatter() -> None:
    with pytest.raises(ValueError):
        parse_skill_md("# 只有正文\n")


def test_missing_section_falls_back_not_empty() -> None:
    """某个行为小节没写时，接口仍返回非空兜底文本（behaviors 的契约）。"""
    text = (
        "---\n"
        'subject_key: "demo"\nsubject_name: "示例"\nskill_id: "teacher.demo.x"\n'
        'class_name: "DemoTeacherSkill"\ndisplay_name: "示例老师"\npersona: "示例"\n'
        'tone: "示例"\ndepth: "standard"\nspeech_rate: 1.0\nteaching_style: "plain"\n'
        'style_persona: |-\n  示例说话风格。\n'
        "---\n\n## 人格与教学\n\n你是一位示例老师。\n"
    )
    front, sections = parse_skill_md(text)
    doc = SkillDoc(key="demo", path=Path("demo/SKILL.md"), front=front, sections=sections)
    assert doc.is_complete()

    skill = build_skill_class(doc)()
    assert skill.opening_style().strip(), "缺失小节时应回退，而不是返回空串"
    assert all(str(v).strip() for v in skill.behaviors().values())


def test_incomplete_frontmatter_is_rejected(tmp_path, monkeypatch) -> None:
    """必填字段不全的 md 视为不可用（宁可回退 persona，也不注入半份人格）。"""
    import config.teacher_skills.md as mdmod

    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "SKILL.md").write_text(
        '---\nsubject_key: "demo"\n---\n\n## 人格与教学\n\n半份人格\n', encoding="utf-8"
    )
    monkeypatch.setattr(mdmod, "SKILLS_DIR", tmp_path)
    mdmod.reload_docs()
    try:
        assert mdmod.load_doc("demo") is None
    finally:
        mdmod.reload_docs()


# ---------------------------------------------------------------- 热更新


def test_editing_md_takes_effect_without_restart(tmp_path, monkeypatch) -> None:
    """改一份 SKILL.md 后再次读取即用新内容（缓存按 mtime 失效，不用重启）。"""
    import config.teacher_skills.md as mdmod

    src = SKILLS_DIR / "math" / "SKILL.md"
    target = tmp_path / "math"
    target.mkdir()
    md_file = target / "SKILL.md"
    md_file.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(mdmod, "SKILLS_DIR", tmp_path)
    mdmod.reload_docs()
    try:
        assert mdmod.load_doc("math").text("display_name") == "严谨的数学老师"

        changed = md_file.read_text(encoding="utf-8").replace(
            'display_name: "严谨的数学老师"', 'display_name: "改了名字的数学老师"'
        )
        md_file.write_text(changed, encoding="utf-8")
        # 显式把 mtime 拨到未来，避免同秒内写入导致缓存判定为「没变」
        stamp = md_file.stat().st_mtime + 10
        os.utime(md_file, (stamp, stamp))

        assert mdmod.load_doc("math").text("display_name") == "改了名字的数学老师"
    finally:
        mdmod.reload_docs()


# ---------------------------------------------------------------- 与既有契约一致


def test_counts_stay_consistent() -> None:
    """技能目录数 = 学科数 = 人格数 = 人格类数（多一层少一层都算配置漏了）。"""
    assert len(available_skills()) == len(SUBJECTS) == len(SKILLS) == len(SKILL_ORDER) == 11
