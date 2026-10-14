"""core.skills.experience：教师 Skill 的「自我更新」经验层。

设计边界（重要）：

* **只沉淀经验**：本包只往 ``data/teacher_experience/<老师key>.jsonl`` 追加经验条目，
  并把它注入提示词；**永远不会改写** ``core/skills/teacher_personality`` 里的人格
  提示词本体 —— 人格与平台底线永远由人写、由人改，自我更新只在其之上生长；
* **可回滚**：文件是 append-only 的 JSONL，每条经验可 ``retire``（停用）／``restore``
  （恢复），坏经验出问题能立刻止损，且历史快照都在；
* **按环境隔离**：经验是运行时学习产物，已被 ``scripts/sync_cloud.py`` 排除在
  代码同步之外，本地与云端各自学习、互不覆盖。
"""
from core.skills.experience.feedback import gain_from_verdict, record_turn_outcome
from core.skills.experience.models import KIND_LABELS, ExperienceItem
from core.skills.experience.prompt import build_experience_prompt, render_experience
from core.skills.experience.dedupe import find_duplicate_groups
from core.skills.experience.reflector import SkillReflector
from core.skills.experience.review import review_queue, risk_of
from core.skills.experience.store import ExperienceStore, get_experience_store

__all__ = [
    "KIND_LABELS",
    "ExperienceItem",
    "ExperienceStore",
    "SkillReflector",
    "build_experience_prompt",
    "find_duplicate_groups",
    "gain_from_verdict",
    "get_experience_store",
    "record_turn_outcome",
    "render_experience",
    "review_queue",
    "risk_of",
]
