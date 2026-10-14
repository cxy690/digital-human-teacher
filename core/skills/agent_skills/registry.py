"""Agent 技能注册表：装配、暴露给模型、按战绩开关。

三件事
------
1. **装配**：进程级单例，默认装四个内置技能（:func:`~core.skills.agent_skills.builtin.builtin_skills`）；
2. **暴露**：:meth:`schemas` 产出 function-calling 格式清单（``AgentToolkit`` 直接拿去给模型），
   :meth:`context_for` 装配每次执行的上下文（含**学习增强**）；
3. **按战绩开关**：:meth:`enabled` 过滤掉「人工停用」与「战绩过差且打开了自动停用」的技能。

自我更新闭环在本模块的落点
--------------------------
* **学**：:meth:`learned_notes` 从该老师的经验表按 ``spec.experience_kinds`` 选出
  实战补充，附加到技能上下文（人写基底 + 自动补充，与人格层同构）；
* **评**：:meth:`enabled` 消费 :mod:`core.skills.agent_skills.usage` 的战绩
  （默认只进抽检台、不阻断；``AGENT_SKILL_AUTO_RETIRE=true`` 才自动停用）。
"""
from __future__ import annotations

from typing import Any, Iterable

from config.logging import get_logger
from config.settings import settings
from core.skills.agent_skills.base import AgentSkill, SkillContext, render_learned
from core.skills.agent_skills.builtin import builtin_skills, current_teacher_key
from core.skills.agent_skills.usage import SkillUsageStore

log = get_logger("core.skills.agent_skills.registry")


class AgentSkillRegistry:
    """技能注册表（进程级单例，见 :func:`get_skill_registry`）。"""

    def __init__(
        self,
        skills: Iterable[AgentSkill] | None = None,
        *,
        usage_store: SkillUsageStore | None = None,
    ) -> None:
        self._skills: dict[str, AgentSkill] = {}
        self._usage = usage_store
        for skill in skills if skills is not None else builtin_skills():
            self.register(skill)

    # ---------------------------------------------------------------- 装配
    def register(self, skill: AgentSkill) -> AgentSkill:
        name = str(getattr(skill, "spec", None) and skill.spec.name or "").strip()
        if not name:
            raise ValueError("技能必须有 spec.name")
        if name in self._skills:
            log.warning("agent skill overridden: {}", name)
        self._skills[name] = skill
        return skill

    @property
    def usage_store(self) -> SkillUsageStore:
        if self._usage is None:
            from core.skills.agent_skills.usage import skill_usage_store  # noqa: PLC0415

            self._usage = skill_usage_store
        return self._usage

    # ---------------------------------------------------------------- 查找
    def get(self, name: str) -> AgentSkill | None:
        return self._skills.get(str(name or "").strip())

    def all(self) -> list[AgentSkill]:
        return list(self._skills.values())

    def names(self) -> list[str]:
        return list(self._skills)

    # ---------------------------------------------------------------- 暴露给模型
    def enabled(self) -> list[AgentSkill]:
        """当前应暴露给模型的技能（总开关 + 人工停用 + 可选自动停用）。"""
        if not settings.agent_skills_enable:
            return []
        out: list[AgentSkill] = []
        for skill in self.all():
            usage = self.usage_store.load(skill.spec.name)
            if usage.status == "disabled":
                continue
            if settings.agent_skill_auto_retire and self._too_weak(usage):
                log.info("技能战绩过差，已自动停用：{}（有效率 {:.2f}）", skill.spec.name, usage.gain_rate)
                continue
            out.append(skill)
        return out

    def schemas(self) -> list[dict[str, Any]]:
        """function-calling 格式的技能清单（与工具同一命名空间，name 不得重名）。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": skill.spec.name,
                    "description": skill.spec.description,
                    "parameters": {
                        "type": "object",
                        "properties": dict(skill.spec.inputs),
                        "required": list(skill.spec.required),
                    },
                },
            }
            for skill in self.enabled()
        ]

    def specs(self) -> list[dict[str, Any]]:
        """全部技能的声明（抽检台 / 前端展示用，不受开关影响）。"""
        return [skill.spec_dict() for skill in self.all()]

    # ---------------------------------------------------------------- 学习增强
    def learned_notes(self, skill: AgentSkill, teacher_key: str) -> str:
        """该技能这一轮可以附加的「实战补充」（来自经验表，按 kind 过滤）。

        * 技能自己不关心经验（``experience_kinds=()``）时直接返回空串；
        * bandit 关闭时退回静态排序（与经验注入同一取舍）；
        * 经验表不可用 / 为空时返回空串 —— **学习增强是加法，缺了不该影响技能本身**。
        """
        kinds = tuple(skill.spec.experience_kinds)
        key = (teacher_key or "").strip()
        if not kinds or not key:
            return ""
        limit = int(settings.agent_skill_learned_max or 0)
        if limit <= 0:
            return ""
        try:
            from core.skills.experience import get_experience_store  # noqa: PLC0415

            items = get_experience_store().select(key, limit=limit * 2)
        except Exception as err:  # noqa: BLE001 - 读经验失败绝不能影响技能
            log.warning("技能学习增强读取失败（{}）：{}", skill.spec.name, err)
            return ""
        picked = [it for it in items if it.kind in kinds][:limit]
        if not picked:
            return ""
        return "\n".join(f"- {it.text}" for it in picked)

    def context_for(self, skill: AgentSkill, session: Any) -> SkillContext:
        """装配一次技能执行的上下文（含学习增强）。"""
        teacher_key = current_teacher_key(session)
        notes = self.learned_notes(skill, teacher_key)
        return SkillContext(
            session=session,
            learned_notes=render_learned(notes) if notes else "",
            teacher_key=teacher_key,
        )

    # ---------------------------------------------------------------- 战绩判定
    def _too_weak(self, usage: Any) -> bool:
        """战绩是否差到该自动停用（样本不足时不下结论 —— 与经验抽检同一立场）。"""
        pulls = int(getattr(usage, "pulls", 0) or 0)
        if pulls < int(settings.agent_skill_min_pulls):
            return False
        return float(getattr(usage, "gain_rate", 0.0) or 0.0) < float(
            settings.agent_skill_min_gain_rate
        )


#: 进程级默认注册表（与 ``core.skills.teacher_personality`` 的注册表同一用法）
_REGISTRY = AgentSkillRegistry()


def get_skill_registry() -> AgentSkillRegistry:
    """返回进程级技能注册表。"""
    return _REGISTRY


__all__ = ["AgentSkillRegistry", "get_skill_registry"]
