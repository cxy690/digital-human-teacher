"""Agent 技能基座：把「一项可被模型自主调用的业务能力」定义成技能。

与工具（``agents/shared/toolkit.py`` 的原子工具）的分工
------------------------------------------------------
工具是**原子能力**：查教材、解方程、读白板 —— 输入输出确定、无需判断「该不该用」；
技能是**一段带判断的业务能力**：什么时候表扬、要不要翻这位学生的旧账、
本科目按什么套路讲 —— 它内部可能再调 LLM 或其它模块，并自己决定「这题适不适合用我」。

两者对模型是**同一个接口**（function calling），差别只在语义与复杂度：
把「表扬学生」做成技能而不是工具，是因为它包含「先判断学生是不是真答对了」这一层 ——
这层判断不该由答疑模型在写讲解的中途分心去做。

为什么技能要能自我更新（对应经验闭环）
--------------------------------------
技能有两块内容，边界与人格 Skill 完全一致：

* ``instruction`` —— **人工写的基底**，永不被程序改写；
* 运行时附加的 **learned notes** —— 从该老师的经验表里按 kind 选出的「实战补充」
  （:meth:`~core.skills.agent_skills.registry.AgentSkillRegistry.context_for` 装配）。

于是技能正文 = 人写基底 + 自动学到的补充；技能的**调用战绩**（pulls/wins）由回合裁决
回写（:mod:`core.skills.agent_skills.feedback`），战绩进抽检台、可按开关自动停用。
学（复盘沉淀）→ 选（模型自主 + 经验增强）→ 用（执行）→ 评（裁决回写）因此闭合成环。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: 技能名命名规范：小写蛇形，且**不与工具重名**（工具与技能对模型是同一个命名空间，
#: 重名会让模型的调用意图产生歧义）。测试里钉住这条约束。
SKILL_NAME_HINT = "小写蛇形（如 praise_expression），不得与原子工具重名"


@dataclass(frozen=True)
class SkillSpec:
    """一项技能的静态声明（人工维护的部分）。

    ``description`` 与 ``instruction`` 的分工：
    ``description`` 是给**模型**看的「什么时候该调我」（进 function-calling schema，
    每轮都在上下文里，必须短）；``instruction`` 是给**技能自己**看的执行纪律
    （只在被调用时生效，可以长）。
    """

    name: str
    title: str                      # 中文名（日志 / 抽检台 / 前端展示）
    description: str                # 给模型：何时调用（进 schema，宜短）
    instruction: str = ""           # 技能自身的执行纪律（人工基底，不被自动改写）
    inputs: dict[str, Any] = field(default_factory=dict)   # JSON Schema properties
    required: tuple[str, ...] = ()  # 必填参数名
    #: 该技能关心哪几类经验（用于「学习增强」过滤）；空元组 = 不接经验增强。
    #: 取值见 :data:`core.skills.experience.models.KIND_LABELS`。
    experience_kinds: tuple[str, ...] = ("pitfall", "must_cover", "confusion")
    #: 技能内部是否需要 LLM（抽检台据此估算成本；本身不改变行为）
    needs_llm: bool = False


@dataclass
class SkillOutcome:
    """一次技能执行的产物。"""

    text: str                                  # 回给模型的「观察」文本
    instruction: str = ""                      # 本次实际使用的正文（含学习增强，排障/审计用）
    events: tuple[tuple[str, dict[str, Any]], ...] = ()   # 需要推给前端的事件
    board_notes: tuple[str, ...] = ()           # 要求补充上板的行（版式仍由 Whiteboard 负责）
    hit: bool = True                            # 是否「命中」（未命中也要给模型一句可读说明）


@dataclass
class SkillContext:
    """一次技能执行的上下文（由注册表装配，技能只读）。"""

    session: Any
    #: 从经验表自动学到的补充（已渲染成文本；空串表示这位老师还没有相关经验）
    learned_notes: str = ""
    #: 本会话的教师 key（经验 / 战绩都按它归档）
    teacher_key: str = ""
    llm: Any | None = None      # 需要 LLM 的技能用（默认由技能自建，测试可注入）


class AgentSkill:
    """一项可被模型调用的技能。

    实现约定（与原子工具一致）：
    * :meth:`run` **绝不抛异常** —— 失败要变成一句可读的「观察」，让模型自己决定
      换个方式还是直接作答。技能是增强，不是新的故障点；
    * 未命中也要返回 :class:`SkillOutcome`（``hit=False``），并说清「为什么这次没用我」，
      否则模型会把「静默失败」理解成「调用成功但没内容」。
    """

    spec: SkillSpec

    async def run(self, args: dict[str, Any], context: SkillContext) -> SkillOutcome:
        raise NotImplementedError

    # ---------------------------------------------------------------- 工具方法
    def spec_dict(self) -> dict[str, Any]:
        """技能的完整声明（抽检台 / 前端展示用）。"""
        spec = self.spec
        return {
            "name": spec.name,
            "title": spec.title,
            "description": spec.description,
            "instruction": spec.instruction,
            "inputs": dict(spec.inputs),
            "required": list(spec.required),
            "experience_kinds": list(spec.experience_kinds),
            "needs_llm": spec.needs_llm,
        }


def render_learned(notes: str, *, header: str = "实战补充（从这位老师的真实课堂沉淀，优先采用）") -> str:
    """把学到的经验渲染成附加段（空则返回空串，调用方直接跳过）。"""
    text = (notes or "").strip()
    return f"【{header}】\n{text}" if text else ""
