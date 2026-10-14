"""教师人格 Skill 注册表。

注册表只负责「查找与加载」，不承担任何人格定义：
* :class:`TeacherSkillRegistry` 按 ``skill_id`` 或 ``subject_id`` 返回对应 Skill；
* :data:`SKILLS` 是各学科教师 Skill 实例的唯一索引（key = ``skill_id``）；
* 找不到 Skill 时返回 ``None``，由调用方回退到 ``persona`` / ``style``，
  但 **绝不会** 把其他学科的 Skill 错误加载过来。

**人格内容从哪来**：``.agents/skills/<学科key>/SKILL.md``（见
:mod:`config.teacher_skills.md`）。本模块在导入时按 md 目录生成一科一个类
（``MathTeacherSkill`` / ``ChineseTeacherSkill`` …，见
:func:`~core.skills.teacher_personality.md_teacher.build_skill_class`），
人格文本则在**每次调用**时从 md 现读 —— 所以在线上改一份 md，
下一次答疑即生效，不需要重启进程。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from config.logging import get_logger
from config.subjects import SUBJECTS
from config.teacher_skills.md import available_keys, load_doc
from core.skills.teacher_personality.base import TeacherPersonalitySkill
from core.skills.teacher_personality.md_teacher import (
    MarkdownTeacherSkill,
    build_skill_class,
)

if TYPE_CHECKING:  # pragma: no cover - 仅类型标注
    from core.state.session import SessionState

log = get_logger("core.skills.teacher_personality.registry")


def _ordered_keys() -> tuple[str, ...]:
    """技能顺序：与 ``config/subjects.py`` 的 ``SUBJECTS`` 对齐，多出来的排后面。

    顺序只影响注册与展示；查表一律按 key 精确匹配，不依赖次序。
    """
    keys = set(available_keys())
    known = [str(s["key"]) for s in SUBJECTS]
    ordered = [k for k in known if k in keys]
    ordered += sorted(keys - set(known))
    return tuple(ordered)


def _build_order() -> tuple[type[MarkdownTeacherSkill], ...]:
    """按 md 目录生成各学科人格 Skill 类（缺 md 的学科直接跳过，由调用方回退）。"""
    classes: list[type[MarkdownTeacherSkill]] = []
    for key in _ordered_keys():
        doc = load_doc(key)
        if doc is None:
            log.warning("学科 {} 的 SKILL.md 不可用，本学科人格将回退到 persona", key)
            continue
        classes.append(build_skill_class(doc))
    return tuple(classes)


#: 各学科教师 Skill 类（顺序 = SUBJECTS 顺序），共 11 科
SKILL_ORDER: tuple[type[MarkdownTeacherSkill], ...] = _build_order()


class TeacherSkillRegistry:
    """各学科教师人格 Skill 的查找器（同一接口，不同人格）。"""

    def __init__(self, skills: Iterable[TeacherPersonalitySkill] | None = None) -> None:
        self._skills: dict[str, TeacherPersonalitySkill] = {}
        self._by_subject: dict[str, str] = {}
        for skill in skills if skills is not None else _default_skills():
            self.register(skill)

    # ------------------------------------------------------------ 注册
    def register(self, skill: TeacherPersonalitySkill) -> TeacherPersonalitySkill:
        """注册一个教师 Skill（同一学科多次注册时，先注册者作为该学科默认人格）。"""
        if not isinstance(skill, TeacherPersonalitySkill):
            raise TypeError("skill must be a TeacherPersonalitySkill")
        if not skill.skill_id:
            raise ValueError("skill_id is required")
        if skill.skill_id in self._skills:
            log.warning("personality skill overridden: {}", skill.skill_id)
        self._skills[skill.skill_id] = skill
        if skill.subject_id:
            self._by_subject.setdefault(skill.subject_id.strip().lower(), skill.skill_id)
        return skill

    # ------------------------------------------------------------ 查找
    def get(self, skill_id: str | None) -> TeacherPersonalitySkill | None:
        """按 ``skill_id`` 精确查找（找不到返回 ``None``）。"""
        return self._skills.get(str(skill_id or "").strip())

    def get_for_subject(self, subject_id: str | None) -> TeacherPersonalitySkill | None:
        """按 ``subject_id``（或直接给 ``skill_id``）查找当前学科的默认人格。"""
        key = str(subject_id or "").strip().lower()
        if not key:
            return None
        if key in self._skills:
            return self._skills[key]
        skill_id = self._by_subject.get(key)
        return self._skills.get(skill_id) if skill_id else None

    def get_for_session(self, session: "SessionState") -> TeacherPersonalitySkill | None:
        """从会话上下文中解析本会话应加载的教师 Skill。

        优先级：``subject_ctx.personality_skill``（学科目录声明）→ 会话 scratch 中的
        ``personality_skill`` → 按 ``subject_id`` 兜底；都没有则返回 ``None``，
        由调用方回退到 ``persona`` / ``style``，不做跨学科顶替。
        """
        scratch = getattr(session, "scratch", None) or {}
        ctx = scratch.get("subject_ctx") or {}
        skill_id = str(ctx.get("personality_skill") or scratch.get("personality_skill") or "").strip()
        if skill_id:
            skill = self.get(skill_id)
            if skill is None:
                log.warning("personality skill not found, fallback to persona: {}", skill_id)
                return None
            return skill
        return self.get_for_subject(ctx.get("subject_id") or scratch.get("subject_id"))

    # ------------------------------------------------------------ 元信息
    def skill_ids(self) -> list[str]:
        return list(self._skills)

    def subject_map(self) -> dict[str, str]:
        """学科 id → 默认 skill_id（当前各科均为「一科一人格」）。"""
        return dict(self._by_subject)

    def all(self) -> list[TeacherPersonalitySkill]:
        return list(self._skills.values())

    def __contains__(self, skill_id: object) -> bool:
        return str(skill_id) in self._skills

    def __len__(self) -> int:
        return len(self._skills)


def _default_skills() -> list[TeacherPersonalitySkill]:
    return [cls() for cls in SKILL_ORDER]


#: 各学科教师 Skill 索引（key = skill_id）
SKILLS: dict[str, TeacherPersonalitySkill] = {s.skill_id: s for s in _default_skills()}

#: 学科 key -> 人格 Skill 类（定位「这一科用的是哪个类 / 哪份 md」时用）
SKILL_CLASSES: dict[str, type[MarkdownTeacherSkill]] = {
    cls.subject_id: cls for cls in SKILL_ORDER
}

#: 进程级默认注册表（只做查找，不定义人格）
_REGISTRY = TeacherSkillRegistry(SKILLS.values())


def get_skill_registry() -> TeacherSkillRegistry:
    """返回进程级默认注册表。"""
    return _REGISTRY


def get_skill(skill_id: str | None) -> TeacherPersonalitySkill | None:
    """按 ``skill_id`` 取教师 Skill（找不到返回 ``None``）。"""
    return _REGISTRY.get(skill_id)
