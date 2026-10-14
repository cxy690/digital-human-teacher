"""内置 Agent 技能。

四个技能覆盖四种典型形态，也是这套机制的最小可用集：

=========================  ==================  ======================================
技能                       形态                 复用自
=========================  ==================  ======================================
``praise_expression``      执行型（内部再判一次） ``core/skills/expression_skill``（原来写了没接线）
``student_mistake_recall`` 执行型（查历史）      ``core/memory``（按学生物理隔离）
``teacher_experience``     执行型（查经验）      ``core/skills/experience``
``subject_playbook``       参考型（无执行体）    ``.agents/skills/<学科>/SKILL.md`` 现读
=========================  ==================  ======================================

两个刻意的设计
--------------
1. **修复「写了没接线」**：``ExpressionSkill`` 此前只有单测引用它（死代码）。
   包成技能后由模型在 ReAct 闭环里自主调用，接线与测试一起补上。
2. **参考型技能证明「渐进披露」**：``subject_playbook`` 没有执行体，调用它就返回
   该科目的讲法正文（含学习增强）—— 学科讲法因此不必每轮全量注入，模型需要时才取。
   后续新增「提示词型技能」照它抄即可，不必写代码。

所有技能都遵守基座约定：不抛异常、未命中也要给一句可读说明。
"""
from __future__ import annotations

from typing import Any

from config.logging import get_logger
from config.teacher_skills import load_skill
from config.teachers import teacher_for_subject
from core.skills.agent_skills.base import AgentSkill, SkillContext, SkillOutcome, SkillSpec

log = get_logger("core.skills.agent_skills.builtin")


# ---------------------------------------------------------------- 会话工具
def current_teacher_key(session: Any) -> str:
    """本会话的教师 key（= 教师档案 key；经验与战绩都按它归档）。

    与 ``agents/shared/teaching_layers.build_profile_layers`` 取同一位老师：
    学生选中的变体优先，为空回退学科默认 —— 两边不一致会出现「经验学在 A 身上、
    注入到 B 身上」的漂移。
    """
    scratch = getattr(session, "scratch", None) or {}
    selected = str(scratch.get("teacher") or getattr(session, "teacher", "") or "").strip()
    subject = str(scratch.get("subject") or getattr(session, "subject", "") or "")
    profile = teacher_for_subject(subject, selected)
    return profile.key if profile else ""


def current_subject_key(session: Any) -> str:
    """本会话的学科 key（``.agents/skills/`` 的目录名）。"""
    scratch = getattr(session, "scratch", None) or {}
    subject = str(scratch.get("subject") or getattr(session, "subject", "") or "")
    profile = teacher_for_subject(subject, str(scratch.get("teacher") or ""))
    if profile is not None:
        own = load_skill(profile.key)
        if own is not None:
            return profile.key
    from config.subjects import SUBJECT_BY_KEY, SUBJECT_BY_NAME  # noqa: PLC0415 - 避免导入期耦合

    meta = SUBJECT_BY_NAME.get(subject) or SUBJECT_BY_KEY.get(subject)
    return str((meta or {}).get("key") or "")


# ---------------------------------------------------------------- 1. 答对表扬
class PraiseExpressionSkill(AgentSkill):
    """学生答对时生成一句口头表扬（修复 expression_skill 的「写了没接线」）。

    为什么是技能而不是原子工具：它内部要先判断「学生这句话是不是真答对了」
    （规则预筛 + LLM 判定），这层判断不该由正在写讲解的模型分心去做。
    """

    def __init__(self, inner: Any | None = None) -> None:
        self._inner = inner

    spec = SkillSpec(
        name="praise_expression",
        title="答对表扬",
        description=(
            "学生刚刚答对（或答得干脆出色）时，生成一句具体、口语化、可直接朗读的表扬。"
            "只有确实答对才调用；答错、提问、求助、寒暄一律不要用。"
        ),
        instruction=(
            "表扬要落到「他做对了哪一步」，一句话说完；不夸「聪明」，只夸「这一步做对了」；"
            "拿到话术后用自然口语带出，不要照搬原文，也不要每答一题都夸。"
        ),
        inputs={"student_answer": {"type": "string", "description": "学生刚才说出的答案或结论原文"}},
        required=("student_answer",),
        experience_kinds=(),      # 表扬是情境技能，不接教学经验
        needs_llm=True,
    )

    async def run(self, args: dict[str, Any], context: SkillContext) -> SkillOutcome:
        answer = str(args.get("student_answer") or "").strip()
        if not answer:
            return SkillOutcome(
                text="未提供学生原话，无法判断是否答对。不要刻意表扬，继续正常讲解。",
                hit=False,
            )
        try:
            inner = self._inner_or_default(context)
            decision = await inner.judge(answer, context.session)
        except Exception as err:  # noqa: BLE001 - 技能失败必须降级成「观察」
            log.warning("praise_expression 判定失败：{}", err)
            return SkillOutcome(text="表扬判定失败（忽略即可），继续正常讲解。", hit=False)
        if decision is None:
            return SkillOutcome(
                text="未命中：学生这句话不像答对（或当前不像在答题）。不要刻意表扬，继续正常讲解。",
                hit=False,
            )
        praise = str(getattr(decision, "praise", "") or "").strip()
        return SkillOutcome(
            text=(
                f"已生成表扬（请用自然口语带出，不要照搬）：{praise}\n"
                "若这句话与当前讲解衔接生硬，可以不说 —— 表扬是加分项，不是必选项。"
            ),
            events=(("agent_skill", {"skill": self.spec.name, "kind": getattr(decision, "kind", "praise")}),),
        )

    def _inner_or_default(self, context: SkillContext):
        if self._inner is not None:
            return self._inner
        from core.skills.expression_skill import ExpressionSkill  # noqa: PLC0415

        return ExpressionSkill(llm=context.llm) if context.llm else ExpressionSkill()


# ---------------------------------------------------------------- 2. 学生旧账
class StudentMistakeRecallSkill(AgentSkill):
    """开讲前查这位学生历史上在同类点上的错题 / 薄弱点。

    数据来自 :mod:`core.memory` —— **按学生物理分文件**，因此这个技能天然不会串学生；
    查询用字符 bigram 相关性（与 RAG 同一套零依赖做法）。
    """

    spec = SkillSpec(
        name="student_mistake_recall",
        title="学生旧账回顾",
        description=(
            "讲解（尤其是易错点、学生反复出错类）之前，查这位学生以前在类似地方出过的错。"
            "第一次给某位学生上课、或与历史明显无关时不要用。"
        ),
        instruction=(
            "翻旧账是为了「顺带提醒」，不是点名批评：用「这类题以前容易在…上马虎」带过即可；"
            "查不到就按第一次接触来讲，不要编造历史。"
        ),
        inputs={
            "query": {"type": "string", "description": "本题相关的关键词，如「移项变号」「通分」"},
            "top_k": {"type": "integer", "description": "返回几条，默认 3"},
        },
        required=("query",),
        experience_kinds=("pitfall", "confusion"),
        needs_llm=False,
    )

    async def run(self, args: dict[str, Any], context: SkillContext) -> SkillOutcome:
        query = str(args.get("query") or "").strip()
        if not query:
            return SkillOutcome(text="缺少 query 参数（要查的关键词）。", hit=False)
        session = context.session
        student_id = str(getattr(session, "student_id", "") or "").strip()
        if not student_id:
            return SkillOutcome(text="当前会话没有学生标识，无从查历史。按第一次接触来讲。", hit=False)
        try:
            from core.memory import get_memory_store  # noqa: PLC0415

            store = get_memory_store()
            subject = str(getattr(session, "subject", "") or "")
            hits = store.search(student_id, query, subject=subject, limit=_as_int(args.get("top_k"), 3, 1, 6))
        except Exception as err:  # noqa: BLE001
            log.warning("student_mistake_recall 查询失败：{}", err)
            return SkillOutcome(text="历史记录查询失败（忽略即可），按第一次接触来讲。", hit=False)
        if not hits:
            return SkillOutcome(
                text=f"这位学生没有查到与「{query}」相关的历史错题/薄弱点。按第一次接触来讲，"
                     "讲完提醒他记下要点即可。",
                hit=False,
            )
        lines = [f"- {getattr(item, 'text', '')}" for item in hits]
        return SkillOutcome(
            text=(
                f"这位学生以前在类似地方出过错（讲解时顺带提醒，不要点名翻旧账）：\n"
                + "\n".join(lines)
            ),
        )


# ---------------------------------------------------------------- 3. 教师经验
class TeacherExperienceSkill(AgentSkill):
    """查这位老师教这类题攒下的经验（:mod:`core.skills.experience`）。

    与「自动注入」的分工：自动注入按 bandit 每轮选几条进 system prompt；
    本技能让模型在**需要时主动查**（例如讲到某个具体易错点时想看看老师以前怎么处理的）。
    """

    spec = SkillSpec(
        name="teacher_experience",
        title="教师经验查询",
        description=(
            "不确定「这类题以前是怎么讲的更有效」时，查这位老师从真实课堂沉淀下来的经验。"
            "常规讲解不需要每次都查。"
        ),
        instruction="经验是「这位老师的手感」，可以用来定讲解重点与易错点；与教材冲突时以教材为准。",
        inputs={
            "kind": {
                "type": "string",
                "description": "可选：pitfall 易错点 / analogy 类比 / must_cover 必讲点 / confusion 常见困惑；留空取全部",
            },
            "top_k": {"type": "integer", "description": "返回几条，默认 4"},
        },
        required=(),
        experience_kinds=(),      # 它就是经验本身，不需要再叠一层
        needs_llm=False,
    )

    async def run(self, args: dict[str, Any], context: SkillContext) -> SkillOutcome:
        teacher_key = context.teacher_key or current_teacher_key(context.session)
        if not teacher_key:
            return SkillOutcome(text="当前没有可归档的教师标识，跳过经验查询。", hit=False)
        kind = str(args.get("kind") or "").strip()
        limit = _as_int(args.get("top_k"), 4, 1, 8)
        try:
            from core.skills.experience import get_experience_store  # noqa: PLC0415

            items = get_experience_store().active(teacher_key)
        except Exception as err:  # noqa: BLE001
            log.warning("teacher_experience 查询失败：{}", err)
            return SkillOutcome(text="经验查询失败（忽略即可），按你自己的判断讲。", hit=False)
        if kind:
            items = [it for it in items if it.kind == kind]
        items = items[:limit]
        if not items:
            return SkillOutcome(
                text=f"这位老师（{teacher_key}）还没有相关经验。按你自己的判断讲，讲完经验会自行沉淀。",
                hit=False,
            )
        lines = [f"- [{getattr(it, 'kind', '')}] {getattr(it, 'text', '')}" for it in items]
        return SkillOutcome(text=f"这位老师沉淀的经验（优先采用）：\n" + "\n".join(lines))


# ---------------------------------------------------------------- 4. 学科讲法
class SubjectPlaybookSkill(AgentSkill):
    """本科目的讲课套路（参考型技能：无执行体，调用即返回正文）。

    正文现读 ``.agents/skills/<学科key>/SKILL.md`` 的「讲课方式」小节 ——
    改 md 即生效，不需要重启；也**不必每轮全量注入**，模型需要时才来取。
    """

    spec = SkillSpec(
        name="subject_playbook",
        title="学科讲法",
        description=(
            "要按本学科的规范讲法组织讲解（如数学的建模五步、化学的探究七环）时调用，"
            "拿到讲法骨架后再动笔。闲聊、简单概念题不必用。"
        ),
        instruction="拿到讲法后按其骨架组织白板步骤；骨架是参考，题目本身不适合套用时以题目为准。",
        inputs={"topic": {"type": "string", "description": "本题主题，如「一元一次方程」「金属活动性」"}},
        required=(),
        experience_kinds=("must_cover", "analogy"),
        needs_llm=False,
    )

    async def run(self, args: dict[str, Any], context: SkillContext) -> SkillOutcome:
        key = current_subject_key(context.session)
        if not key:
            return SkillOutcome(text="当前学科无法识别，跳过讲法查询。", hit=False)
        try:
            skill = load_skill(key)
        except Exception as err:  # noqa: BLE001
            log.warning("subject_playbook 读取失败：{}", err)
            return SkillOutcome(text="学科讲法读取失败（忽略即可），按常规思路讲。", hit=False)
        if skill is None:
            return SkillOutcome(text=f"本科目（{key}）没有独立的讲法技能，按常规思路讲。", hit=False)
        way = str(getattr(skill, "teaching_way", "") or "").strip()
        if not way:
            return SkillOutcome(text=f"本科目（{key}）的讲法小节为空，按常规思路讲。", hit=False)
        topic = str(args.get("topic") or "").strip()
        head = f"本题主题：{topic}\n" if topic else ""
        # 学习增强由 AgentToolkit 统一附在观察末尾（见 toolkit._with_learned），
        # 这里不重复拼一遍，否则模型会看到两份相同的「实战补充」。
        return SkillOutcome(text=f"{head}{key} 的讲课方式（骨架，供组织白板步骤）：\n{way}")


def _as_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


#: 内置技能工厂（注册表默认装配这四个）
def builtin_skills() -> list[AgentSkill]:
    return [
        PraiseExpressionSkill(),
        StudentMistakeRecallSkill(),
        TeacherExperienceSkill(),
        SubjectPlaybookSkill(),
    ]


__all__ = [
    "PraiseExpressionSkill",
    "StudentMistakeRecallSkill",
    "SubjectPlaybookSkill",
    "TeacherExperienceSkill",
    "builtin_skills",
    "current_subject_key",
    "current_teacher_key",
]
