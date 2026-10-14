"""Agent 技能：模型可**自主调用**的业务能力单元（含自我更新）。

与另外两个 skill 目录的分工
--------------------------
===========================  ==========================  ==================================
目录                        是什么                      谁来决定用不用
===========================  ==========================  ==================================
``teacher_personality``      人格提示词（人写，永不自动改） 学科 / 老师档案（按 key 查表）
``experience``               自动学到的经验条目             bandit 采样 + 提示词注入
``agent_skills``（本目录）   可执行业务技能                **模型自己**（function calling）
===========================  ==========================  ==================================

自我更新（对应经验闭环）
------------------------
* **学**：注册表把经验表里选出的「实战补充」附加到技能上下文（人写基底 + 自动补充）；
* **用**：模型在 ReAct 闭环里自主决定调不调、传什么参数；
* **评**：回合裁决回写每个被调用技能的 ``pulls/wins``（append-only），
  战绩进抽检台（:meth:`usage.SkillUsageStore.review_queue`），
  打开 ``AGENT_SKILL_AUTO_RETIRE`` 后可自动停用战绩过差的技能。

新增一项技能：在 :mod:`core.skills.agent_skills.builtin` 里加一个
:class:`~core.skills.agent_skills.base.AgentSkill` 子类并注册即可 ——
工具暴露、学习增强、战绩记账都由基座负责，技能只写「什么时候用我 + 怎么执行」。
"""
from core.skills.agent_skills.base import (
    AgentSkill,
    SkillContext,
    SkillOutcome,
    SkillSpec,
    render_learned,
)
from core.skills.agent_skills.builtin import (
    PraiseExpressionSkill,
    StudentMistakeRecallSkill,
    SubjectPlaybookSkill,
    TeacherExperienceSkill,
    builtin_skills,
    current_subject_key,
    current_teacher_key,
)
from core.skills.agent_skills.feedback import (
    USED_KEY,
    consume_used_skills,
    mark_skill_used,
    record_skill_outcome,
)
from core.skills.agent_skills.registry import AgentSkillRegistry, get_skill_registry
from core.skills.agent_skills.usage import SkillUsage, SkillUsageStore, get_skill_usage_store

__all__ = [
    "AgentSkill",
    "AgentSkillRegistry",
    "PraiseExpressionSkill",
    "SkillContext",
    "SkillOutcome",
    "SkillSpec",
    "SkillUsage",
    "SkillUsageStore",
    "StudentMistakeRecallSkill",
    "SubjectPlaybookSkill",
    "TeacherExperienceSkill",
    "USED_KEY",
    "builtin_skills",
    "consume_used_skills",
    "current_subject_key",
    "current_teacher_key",
    "get_skill_registry",
    "get_skill_usage_store",
    "mark_skill_used",
    "record_skill_outcome",
    "render_learned",
]
