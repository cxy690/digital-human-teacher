"""独立教师人格 Skill 包。

每个学科（语文 / 数学 / 英语 / 科学 / 物理 / 化学 / 生物 / 政治 / 历史 / 地理 /
道德与法治）各自拥有 **独立** 的教师人格 Skill：独立身份、独立语言风格、
独立教学行为和互动边界，彼此不共享人格内容，只共享
:class:`TeacherPersonalitySkill` 接口与 :data:`GLOBAL_TEACHER_RULES` 平台底线。

**人格内容的事实源是 Markdown**：``.agents/skills/<学科key>/SKILL.md``
（frontmatter 给身份与参数，正文小节给「人格与教学」及各行为维度）。
本包不再手写人格 Python 模块，只按 md 生成一科一个类：

    .agents/skills/math/SKILL.md  ->  MathTeacherSkill（frontmatter.class_name）

使用方式（TeacherAgent 只做通用调用，不写学科分支）::

    from core.skills.teacher_personality import GLOBAL_TEACHER_RULES, get_skill_registry

    skill = get_skill_registry().get_for_session(session)
    if skill is not None:
        messages.append({"role": "system", "content": GLOBAL_TEACHER_RULES})
        messages.append({"role": "system", "content": skill.build_system_prompt()})

维护提示：改一位老师的说话方式 / 教学行为，改对应的 SKILL.md 即可，
下一次调用即生效（md 解析按文件修改时间失效缓存），无需重启、无需改代码。
"""
from core.skills.teacher_personality.base import (
    GLOBAL_TEACHER_RULES,
    TeacherPersonalitySkill,
)
from core.skills.teacher_personality.md_teacher import (
    MarkdownTeacherSkill,
    build_skill_class,
)
from core.skills.teacher_personality.registry import (
    SKILL_CLASSES,
    SKILL_ORDER,
    SKILLS,
    TeacherSkillRegistry,
    get_skill,
    get_skill_registry,
)

#: 各学科人格类（``MathTeacherSkill`` 等）按**类名**注入本模块命名空间 ——
#: 类是按 md 动态生成的，这样 ``from core.skills.teacher_personality import MathTeacherSkill``
#: 与迁移前的写法完全一致。
for _cls in SKILL_ORDER:
    globals()[_cls.__name__] = _cls

__all__ = [
    "GLOBAL_TEACHER_RULES",
    "SKILL_CLASSES",
    "SKILL_ORDER",
    "SKILLS",
    "MarkdownTeacherSkill",
    "TeacherPersonalitySkill",
    "TeacherSkillRegistry",
    "build_skill_class",
    "get_skill",
    "get_skill_registry",
    *[c.__name__ for c in SKILL_ORDER],
]
