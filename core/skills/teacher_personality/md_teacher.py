"""由 Markdown 事实源驱动的人格 Skill（``.agents/skills/<学科>/SKILL.md``）。

设计：**人格内容全部来自 md**，Python 这里只有一个「适配器」——
:class:`MarkdownTeacherSkill` 把 :class:`~config.teacher_skills.md.SkillDoc` 的
frontmatter 与正文小节映射到 :class:`TeacherPersonalitySkill` 的接口上，
并为每个学科生成一个具名子类（``MathTeacherSkill`` / ``ChineseTeacherSkill`` …），
这样 ``SKILL_ORDER``、``cls().skill_id``、``from ... import MathTeacherSkill``
这些既有用法全部保持不变。

内容改动一律改 md：改完下一次调用即生效（:func:`~config.teacher_skills.md.load_doc`
按文件修改时间失效缓存），不需要动任何 Python。
"""
from __future__ import annotations

from typing import Any

from config.logging import get_logger
from config.teacher_skills.md import BEHAVIOR_SECTIONS, SkillDoc, load_doc
from core.skills.teacher_personality.base import TeacherPersonalitySkill

log = get_logger("core.skills.teacher_personality.md_teacher")

#: 兜底文本：**只在 md 缺失或某个小节没写**时兜住，保证接口永不返回空串
#: （``behaviors()`` 的契约是九项皆有内容）。语义与 :class:`TeacherPersonalitySkill`
#: 的默认实现一致 —— 它是「平台最低要求」，不是某位老师的人格。
_FALLBACK: dict[str, str] = {
    "build_system_prompt": (
        "你是一位耐心、友好的中小学 AI 一对一教师。"
        "讲清「是什么、为什么、怎么用」，不跳步、不训斥，学生听不懂就换一种说法再讲一遍。"
    ),
    "opening_style": "先看清题目给了什么、问的是什么，再一起决定从哪里入手。",
    "questioning_style": "用一个学生能直接回答的小问题推进，一次只问一件事。",
    "encouragement_style": "先肯定学生已经做对或想对的部分，再指出下一步可以怎么走。",
    "praise_style": "具体指出学生答对的部分，再顺势推进到下一步。",
    "correction_style": "指出出错的具体步骤，给出可以直接执行的修正办法。",
    "confused_student_strategy": "回到最基础的定义，换一个更小的例子，示范一步再让学生跟着做。",
    "repeated_error_strategy": "降低难度，改为示范一个小步骤，先让学生重获成功体验。",
    "whiteboard_style": "按步骤分行板书，每行是一句完整、适合朗读的话。",
    "tone_rhythm": "语速平稳，一句一个意思，长句拆短。",
}


class MarkdownTeacherSkill(TeacherPersonalitySkill):
    """一位老师的人格 Skill：内容取自 ``.agents/skills/<学科key>/SKILL.md``。

    子类由 :func:`build_skill_class` 按学科动态生成，并把 frontmatter 里的身份字段
    （``skill_id`` / ``subject_id`` / …）写成**类属性** —— 与迁移前手写类的形状一致。
    """

    #: 该子类对应的学科 key（= md 目录名）；子类创建时写入
    _doc_key: str = ""

    # ------------------------------------------------------------ md 取用
    @classmethod
    def doc(cls) -> SkillDoc | None:
        """本学科当前的 SKILL.md（文件被改过时会自动重读）。"""
        return load_doc(cls._doc_key)

    def _text(self, name: str) -> str:
        """读一个行为小节；md 缺失时返回兜底文本（绝不返回空串，见测试契约）。"""
        doc = type(self).doc()
        value = doc.behavior(name) if doc is not None else ""
        return value.strip() or _FALLBACK.get(name, "")

    # ------------------------------------------------------------ 人格行为
    def build_system_prompt(self) -> str:
        """该教师独立的人格与教学行为提示词（md 的「人格与教学」整段）。"""
        doc = type(self).doc()
        text = doc.system_prompt() if doc is not None else ""
        return text.strip() or _FALLBACK["build_system_prompt"]

    def opening_style(self) -> str:
        return self._text("opening_style")

    def questioning_style(self) -> str:
        return self._text("questioning_style")

    def encouragement_style(self) -> str:
        return self._text("encouragement_style")

    def praise_style(self) -> str:
        return self._text("praise_style")

    def correction_style(self) -> str:
        return self._text("correction_style")

    def confused_student_strategy(self) -> str:
        return self._text("confused_student_strategy")

    def repeated_error_strategy(self) -> str:
        return self._text("repeated_error_strategy")

    def whiteboard_style(self) -> str:
        return self._text("whiteboard_style")

    def tone_rhythm(self) -> str:
        return self._text("tone_rhythm")

    # ------------------------------------------------------------ 开场语
    def greeting(self, where: str = "") -> str:
        doc = type(self).doc()
        text = doc.greeting(where) if doc is not None else ""
        return text or super().greeting(where)


def build_skill_class(doc: SkillDoc) -> type[MarkdownTeacherSkill]:
    """按一份 SKILL.md 生成该学科的人格 Skill 类。

    类名取自 frontmatter 的 ``class_name``（缺失时用 ``<Key>TeacherSkill``），
    身份字段写成类属性 —— 于是 ``MathTeacherSkill().skill_id`` 这类旧写法不变。
    """
    class_name = doc.class_name or f"{doc.subject_key.capitalize()}TeacherSkill"
    namespace: dict[str, Any] = {
        "_doc_key": doc.subject_key,
        "skill_id": doc.skill_id,
        "subject_id": doc.subject_key,
        "subject_name": doc.subject_name,
        "display_name": doc.display_name,
        "persona": doc.persona,
        "tone": doc.tone,
        "__module__": __name__,
        "__doc__": (
            f"{doc.subject_name}教师人格 Skill：{doc.display_name}（{doc.skill_id}）。"
            f"内容来自 .agents/skills/{doc.subject_key}/SKILL.md。"
        ),
    }
    return type(class_name, (MarkdownTeacherSkill,), namespace)


#: 九项行为 + 整段提示词在 md 里的落点说明（排障时对照用）
SECTION_HINT: dict[str, str] = dict(BEHAVIOR_SECTIONS)
