"""教师技能（Skill）基础结构：说话风格 + 讲课方式 + 深度 + 语速。

**事实源已迁到 Markdown**：``.agents/skills/<学科key>/SKILL.md``。
本模块只保留 :class:`TeacherSkill` 这个「风格层」数据结构与 ``prompt_text()`` 装配逻辑，
内容来源由 :mod:`config.teacher_skills.md` 读取的 :class:`~config.teacher_skills.md.SkillDoc` 提供
（见 :meth:`TeacherSkill.from_doc`）。

为什么用 md 而不是 teacher.json：
风格、讲课方式、口头禅是成段的中文文本，md 里可以写标题、注释、示例，读得懂、改得动，
也便于版本管理与 review —— 而且**同一份 md 同时供风格层与人格层消费**，不再各写一份。
teacher.json 继续管「形象/音色/标定」，本层管「这个人怎么说话、怎么讲课」。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - 仅类型标注
    from config.teacher_skills.md import SkillDoc

# 讲解深度指令：depth 字段对应的提示词（管"讲到多细"）。
DEPTH_GUIDE: dict[str, str] = {
    "elementary": (
        "【讲解深度 · 浅显】面向小学或基础较弱的学生：多打比方，概念一出现立刻用生活例子解释；"
        "不引入超出课标的概念；术语能不用就不用，必须用时马上用大白话解释。"
    ),
    "standard": (
        "【讲解深度 · 标准】按学生所在学段的正常要求讲：概念讲到位，步骤给依据；"
        "可以在学生跟得上的前提下适度拓展，但不炫技、不超纲。"
    ),
    "advanced": (
        "【讲解深度 · 深入】在讲懂的基础上带学生往深处走一层：讲来龙去脉、因果与脉络，"
        "适度联系已学知识；仍以学生能消化为限，不堆砌超纲名词。"
    ),
}


@dataclass(frozen=True)
class TeacherSkill:
    """一位老师的技能档案：说话风格 + 讲课方式 + 深度 + 语速。"""

    key: str                       # 学科 key（= 文件名 = data/teachers/<key>）
    name: str                      # 显示名，如「语文老师」
    persona: str                   # 说话风格：这个人开口是什么感觉
    teaching_way: str              # 讲课方式：拿到一道题按什么顺序/习惯讲
    depth: str = "standard"        # 讲解深度：elementary / standard / advanced
    speech_rate: float = 1.0       # 语速倍率：0.9 慢 / 1.0 正常 / 1.1 快（TTS 接入后生效）
    teaching_style: str = "plain"  # plain 大白话 / normal 按学段 / academic 规范术语
    catchphrases: tuple[str, ...] = ()   # 常挂嘴边的话（让老师有"人味"）
    do: tuple[str, ...] = ()       # 讲课时务必做到的
    dont: tuple[str, ...] = ()     # 不要做的

    @classmethod
    def from_doc(cls, doc: "SkillDoc") -> "TeacherSkill":
        """由一份 ``.agents/skills/<key>/SKILL.md`` 构建风格层档案。

        frontmatter 给参数与短文本，正文小节给成段的「说话风格 / 讲课方式」。
        深度不在 md 里写提示词正文：档位含义由 :data:`DEPTH_GUIDE` 统一维护，
        md 只声明档位（改档位含义才需要动代码，改某位老师讲到多细只改 md）。
        """
        return cls(
            key=doc.subject_key,
            name=f"{doc.subject_name}老师",
            persona=doc.style_field("style_persona"),
            teaching_way=doc.style_field("teaching_way"),
            depth=doc.depth,
            speech_rate=doc.speech_rate,
            teaching_style=doc.teaching_style,
            catchphrases=doc.items("catchphrases"),
            do=doc.items("do"),
            dont=doc.items("dont"),
        )

    def prompt_text(self) -> str:
        """拼装成注入 LLM 的完整提示词（由 config/teachers.py 调用）。"""
        parts: list[str] = [
            f"【说话风格】{self.persona}",
            f"【讲课方式】{self.teaching_way}",
        ]
        depth = DEPTH_GUIDE.get(self.depth)
        if depth:
            parts.append(depth)
        if self.catchphrases:
            parts.append("【常挂在嘴边的话（自然穿插，别每句都用）】" + "；".join(self.catchphrases))
        if self.do:
            parts.append("【务必做到】\n" + "\n".join(f"- {x}" for x in self.do))
        if self.dont:
            parts.append("【不要】\n" + "\n".join(f"- {x}" for x in self.dont))
        return "\n".join(parts)
