"""模型技能包：驱动数字人/客户端表现的技能集合。

当前包含：
* 表情技能（expression_skill）：学生作答短句被判定答对时，
  表扬播报并让前端弹出表情包（praise → 👍 / celebrate → 🎉）；
* 教师人格技能（teacher_personality）：各学科各自独立的教师人格 Skill，
  由注册表按 ``subject_id`` / ``skill_id`` 选择加载。
"""
from core.skills.expression_skill import (
    EXPRESSION_MATERIALS,
    ExpressionDecision,
    ExpressionSkill,
)

__all__ = [
    "EXPRESSION_MATERIALS",
    "ExpressionDecision",
    "ExpressionSkill",
]
