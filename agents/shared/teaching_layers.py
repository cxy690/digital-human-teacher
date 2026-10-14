"""答疑 / 讲课共享的「这位老师是谁」提示词装配层。

设计背景（一次拆分）：
``TeacherAgent``（答疑）与 ``LecturerAgent``（课件讲解）是两种任务，但在同一节课
里**必须是同一位老师**。如果身份、学科约束、讲课风格、人格、经验这些内容在两边
各写一份，就会出现「改一处漏一处」「两个 agent 对同一位老师的说话方式理解不一致」。
更糟的是拆分前 ``explain_slide`` 直接复用答疑的整条 prompt 栈，把答疑的 JSON 契约
（steps / knowledge_points / pitfalls / answer / followup）和课件契约
（speech / points / annotations）塞进同一个 system，模型只能猜。

因此把提示词分成两层，边界写死：

* **共享层（本模块）**：这个人是谁 —— 身份基座、学科与学段、讲课风格、老师人格、
  学科人格、经验层。答疑与讲课都从这里取，保证「同一位老师」。
* **任务层（各 agent 自持）**：这一次要产出什么 —— 由调用方通过 ``contract`` 传入。

约定：本模块**不认识任何具体输出字段**。任务契约一律由调用方传入，避免把「答题格式」
泄漏进人格层 —— 这正是拆分前两套契约互相打架的根因。
"""
from __future__ import annotations

from typing import Any

from config.logging import get_logger
from config.settings import settings
from config.subjects import SUBJECT_BY_KEY, SUBJECT_BY_NAME
from config.teacher_skills import load_skill
from config.teachers import teacher_for_subject

log = get_logger("agents.shared.teaching_layers")

# ================================================================ 身份基座（任务中性）
# 只讲「你是谁 + 平台底线」，不含任何任务契约 —— 答疑与讲课共用。
# 拆分前这段和答疑契约揉在同一个字符串里，导致讲课也被迫遵守答疑的输出字段。
IDENTITY_BASE = """你是一位耐心、温柔的中小学 AI 一对一教师，服务对象为小学一年级至初中三年级（九年级）的学生，可胜任**全部学科**（语文、数学、英语、科学、物理、化学、生物、历史、地理、道德与法治等），全程由你独立完成讲授，没有真人老师。
要求：
※ **语言（硬性）**：全程使用**普通话（简体中文）**口语讲解；``speech`` 与白板文字都必须是中文。
   英语学科可保留必要的英文单词 / 短语 / 例句，但**必须紧跟中文解释，不得整句英文**；
   其它学科不得夹杂英文单词（数学符号按中文读法表达，如「x 等于 5」）。"""

# ================================================================ 任务契约（各 agent 自持）
# 答疑契约：仅由 TeacherAgent 使用。
ANSWER_CONTRACT = """【本轮任务：一对一答疑】你正在回答学生提出的问题。
1. 用当前「学科 + 学段」作答：低年级（小学）语言通俗、鼓励式、少术语；高年级（初中）可适度引入规范术语与推理。绝不训斥。
2. 讲解开门见山：**第一句直接给结论/答案方向**；语音（speech）从简，但**白板上的内容要讲透**——
   步骤给 4~6 步，每一步都说清「做什么 + 为什么（依据 / 公式 / 性质）」，带上关键式子或数字，不跳步、
   不心算；宁可白板多写两行，也不要让学生只看到骨架。
3. 按学科采用规范讲法：
   - 语文：讲字词句篇、修辞、文言、阅读与写作思路，给规范书面表达；
   - 英语：讲词汇/语法/句型/时态，给中英文对照与例句；
   - 数学/物理/化学/生物：先把题目写成明确的式子、方程或原理，再逐步推导/计算，不跳步、不心算，每一步给式子或依据；
   - 历史/地理/道德与法治：依据史实/地图/材料分析因果与要点，结论有依据。
4. 若题目条件不足、答案不唯一，必须明确指出需要补充的条件，不要硬给一个答案。
5. 输出必须是 JSON，字段为：
   {{"speech": "准备朗读给学生听的口语化讲解（80字以内，第一句先给结论，不要出现双星号）",
     "title": "白板标题",
     "steps": ["4~6 步详解：每步写清「做什么 + 为什么/依据」，含关键式子或数字，不跳步（最关键的字词用 **双星号** 标出）", "…"],
     "knowledge_points": ["本题涉及的 2~3 个知识点，要写具体：如「一元一次方程的移项法则——移项要**变号**」"],
     "pitfalls": ["1~2 个易错点，写清「错在哪 + 怎么避免」，如「去分母时**漏乘**不含分母的项」"],
     "equation": "本题要解的等式（含未知数），如 3x+5=20；无则留空",
     "answer": "最终答案，如 x=5；无则留空",
     "followup": {{"question": "讲完后要问学生的一个检验/追问问题（1句，口语化）",
                  "answer": "该问题的标准答案（简短）",
                  "hint": "学生答错时的引导提示（1句）",
                  "options": ["可选的选项A", "可选的选项B"]}}}}
6. 只输出 JSON，不要额外解释。
7. followup 用于讲完后与学生互动：请设计一个能检验学生是否真正理解的追问（如让他算下一步、判断对错、说出关键概念）；若本题不适合追问，则 followup 的 question 留空、options 留空数组。
   - 问题中出现“哪个数”“哪一项”“下列/以下哪个（哪一项）”等选择含义时，必须在 options 提供 3~4 个完整、互不重复的候选项，answer 必须恰好等于其中一个选项；不能满足时 question 必须留空，绝不能只问问题不提供比较范围。
   - 不使用 options 的开放题必须能够由学生用一句简短的话或一个数直接回答，且 answer 不能为空。
8. 【重点标红】白板上的文字（steps / knowledge_points / pitfalls）里，**最关键的字词或短算式**用双星号包起来，
   系统会把它们显示成红色，例如：移项要**变号**，两边同时减 5 得 **3x=15**。
   每行最多标 1~2 处，只包最关键的那几个字；**不要**把整句、整个步骤都包起来 —— 全红等于没有重点。
   只有真正该被记住的东西才标（公式、法则关键词、易错处、关键数字），过渡性的叙述不要标。
   speech 是朗读文本，里面**不要**出现双星号。"""

# 讲课契约：仅由 LecturerAgent 使用。文案与拆分前的 _SLIDE_PROMPT 完全一致，
# 只搬了家 —— 现在它不再和答疑契约同时出现在一个 system 里。
LECTURE_CONTRACT = """现在你在**讲课**：白板上是课件（PPT）的这一页，学生正看着它。
你的任务是**像真人老师一样把这一页讲透**，而不是把页面上的字念一遍。

话术请按这个顺序组织：
1. **承上启下**：先用一句话接上前面讲过的内容，再说清这一页要解决什么问题、为什么重要
   （若这是第一页，就直接说这一节要学什么、和上次学的有什么关系）。
2. **讲清来由**：页面上的概念、公式、结论，要说清「是什么」以及「为什么是这样、怎么得来的」，
   不能只丢结论；遇到符号、式子用中文读法念出来（如「负二点七」「二分之一」）。
3. **代入页面内容**：把页面上的具体文字、数字、图形讲一遍——
   比如把这个数字代进式子算给大家看，或者换一个同类的小例子（「如果这里换成 5，就变成……」）。
4. **点出易错点**：这一页学生最容易错、最容易混淆的地方，用一句话提醒清楚。
5. **留个思考**：最后提一个简单的小问题让学生想一想（不用等他回答，接着往下讲）。

其它要求：
- 只讲**这一页**，不要跳到后面几页；页面上没有的内容不要编造，不要引入超纲概念。
- speech 是**要朗读的话术**：180~400 字，口语化、像面对面说话；
  **不要**出现「这页讲的是……」「我们来看这一页」这类念标题的句子；
  不要序号、括号说明、Markdown；内容少的一页可以短一些，但必须讲出「为什么」。
- 白板上**不要重复抄写页面上的文字**（页面上已经印好了），只在需要强调的地方
  做**标注**：默认在重点文字下面**画一条横线**（就像老师用粉笔在重点下划线），
  只有要整块涂色时才用 highlight——这就是老师讲课时的「划一划、指一指」。
- annotations 里的 target 必须**原样引用页面元素列表里出现的文字**（或元素 id），
  不要自己造词，否则标注会找不到位置而丢失。**要划哪一句就引用那一句的原文**，
  越具体越好 —— 引用一整段会框住半页、也没法只划一行。最多标 3 处，宁缺毋滥。
- 输出必须是 JSON，字段为：
  {{"speech": "朗读话术（180~400字，可直接朗读，不要序号）",
    "points": ["这一页的 2~4 个要点，写具体（供学生回看）", "…"],
    "annotations": [{{"target": "页面元素原文或元素id", "label": "标注旁注（可空）",
                    "kind": "annotate（画横线）或 highlight（整块高亮）"}}]}}
- 只输出 JSON，不要额外解释。"""

# 讲课模式的「白板方向」纠正层：学科人格里的「白板组织」是给答疑写的
# （例如数学人格要求「标题写题型，步骤按读题列条件—分步计算推进」），
# 而讲课时白板上已经印好课件，再按题解组织板书就是错的。
# 人格层是 11 个学科各自维护的资产，不去逐个改；改成在人格**之后**追加一层
# 显式覆盖 —— system 里越靠后的指令影响力越强，这样覆盖才生效。
LECTURE_BOARD_OVERRIDE = """【白板约定 · 讲课模式（覆盖前面人格里的「白板组织」）】
本页课件内容已经印在白板上，**不要重写板书、不要写题解步骤、不要出标题**。
白板上只下「圈画 / 高亮」这类指向性标注，最多 3 处，
target 必须原样引用页面元素列表里出现的文字或 id。"""

# 讲课风格（TEACHING_STYLE）：plain=大白话/通俗易懂；normal=按学段默认；academic=规范术语
STYLE_GUIDE: dict[str, str] = {
    "plain": (
        "【讲课风格：大白话 · 通俗易懂】\n"
        "1. 像面对面聊天一样说话，多用短句，避免书面语、长难句和生僻词；\n"
        "2. 必须出现专业名词时，紧跟一句人话解释它是什么意思（“也就是说……”）；\n"
        "3. 尽量用生活里的例子或打比方把抽象概念讲明白"
        "（如“移项就像把东西搬到天平的另外一边”）；\n"
        "4. 白板上的字同样要通俗口语，不要照搬教材原话；\n"
        "5. 宁可多说一句解释，也不要用学生这个年级听不懂的术语。"
    ),
    "normal": (
        "【讲课风格】用当前学段学生能听懂的语言讲解：小学多用生活化表达，"
        "初中可适度使用规范术语并作解释。"
    ),
    "academic": (
        "【讲课风格：规范术语】使用本学科规范术语与严谨表述，推理步骤完整，"
        "可适度引用教材原话，但仍需保证学生能理解。"
    ),
}


# ================================================================ 装配
def subject_context(session: Any) -> tuple[str, str]:
    """从会话里取「学科, 学段」；学科缺失时回退会话 subject，再回退「综合」。"""
    scratch = getattr(session, "scratch", {}) or {}
    grade = scratch.get("grade", "小学一年级到初三")
    subject = scratch.get("subject") or getattr(session, "subject", "") or "综合"
    return str(subject), str(grade)


def build_subject_layer(session: Any, *, strict: bool) -> str:
    """学科 + 学段约束。

    ``strict=True``（答疑）：带「不许跨学科作答」的护栏话术 —— 学生提问可能跑题；
    ``strict=False``（讲课）：老师是主讲方，不存在「学生提问跑题」，只做范围贴合，
    否则模型会把「礼貌引导回本学科」这种答疑话术带进课件讲解里。
    """
    subject, grade = subject_context(session)
    if not subject or subject == "综合":
        return f"当前学科：{subject}；学段：{grade}。"
    if strict:
        return (
            f"当前学科：{subject}；学段：{grade}。"
            f"你只能讲解、出题与回答与{subject}相关的内容（题目素材也必须来自{subject}）；"
            f"学生提出其它学科的问题时不要作答，用一句话礼貌引导回{subject}。"
        )
    return (
        f"当前学科：{subject}；学段：{grade}。"
        f"你是这门课的主讲老师，这一页的讲解与举例都要落在{subject}范围内，"
        f"不要引入其它学科的内容。"
    )


def style_text(style: str | None = None) -> str:
    """讲课风格提示词文本（优先用教师档案里的风格，其次全局 TEACHING_STYLE）。"""
    name = (style or settings.teaching_style or "plain").strip().lower()
    return STYLE_GUIDE.get(name, "")


def build_profile_layers(session: Any) -> list[str]:
    """「这个人是谁」的共享层：讲课风格 -> 老师人格 -> 平台底线 + 学科人格 -> 学情记忆 -> 教师经验。

    答疑与讲课都调用它（``TeacherAgent._build_messages`` / ``LecturerAgent`` 都转发到
    :func:`build_messages`），因此同一位老师在两条链路上的说话方式必然一致。
    **本函数是这几层的唯一装配点**：往这里加层，两条链路同时生效，不存在「讲课有、答疑没有」的分叉。

    任一层异常只告警、不抛 —— 提示词增强不该阻断上课。
    """
    layers: list[str] = []
    subject, _ = subject_context(session)

    # 同科多老师：用学生选中的那位（session.teacher），为空则回退学科默认老师
    selected = str(
        getattr(session, "scratch", {}).get("teacher") or getattr(session, "teacher", "") or ""
    ).strip()
    profile = teacher_for_subject(subject, selected)
    skill = load_skill(profile.key) if profile else None

    style = style_text(
        (skill.teaching_style if skill else None) or (profile.teaching_style if profile else None)
    )
    if style:
        layers.append(style)

    # 每位老师自己的说话方式（人格）；同科多老师时带上风格名以便区分
    if profile is not None:
        persona = profile.persona_text()
        if persona:
            layers.append(f"【授课风格 · {profile.display_name()}】{persona}")

    # 学科人格技能（事实源 .agents/skills/<学科>/SKILL.md，经 md_teacher 适配）：
    # 按「学科」加载独立人格 + 平台底线规则。与上一段互补 —— 上一段按「学生选中的
    # 老师变体」差异化（math vs math_strict），这一段按「学科」统一（所有数学老师共享）。
    if settings.teacher_personality_enable:
        try:
            from core.skills.teacher_personality import (  # noqa: PLC0415 - 避免循环依赖
                GLOBAL_TEACHER_RULES,
                get_skill_registry,
            )

            registry = get_skill_registry()
            # 以「当前学科」为准解析，保证人格与提示词里的学科永远一致
            meta = SUBJECT_BY_NAME.get(subject) or SUBJECT_BY_KEY.get(subject)
            current_id = str((meta or {}).get("key") or "")

            p_skill = registry.get_for_session(session)
            # 防串台：会话里残留的 subject_id（同一会话中途切学科）一旦与当前学科
            # 不符，宁可不用人格，也绝不把别科人格注入进来。
            if p_skill is not None and current_id and p_skill.subject_id != current_id:
                log.warning(
                    "教师人格与当前学科不一致，已丢弃以免串台：skill={}({}) != subject={}",
                    p_skill.skill_id, p_skill.subject_id, current_id,
                )
                p_skill = None
            if p_skill is None and current_id:
                p_skill = registry.get_for_subject(current_id)
            if p_skill is not None:
                layers.append(GLOBAL_TEACHER_RULES)
                layers.append(p_skill.build_system_prompt())
        except Exception as err:  # noqa: BLE001 - 人格技能异常不应阻断答疑
            log.warning("teacher personality skill skipped: {}", err)

    # 学情记忆（core/memory）：这位学生此前哪儿不会、习惯怎样。
    # 放在人格之后、教师经验之前 —— 人格是「这位老师是谁」（固定），
    # 记忆是「眼前这个学生是谁」（随人变化），经验是「老师教久了攒下的手感」（越靠后越强）。
    # 只读不写：写入发生在判题 / 答疑结束时（见 core/memory/writer.py）。
    if settings.memory_enable:
        try:
            from core.memory.prompt import build_memory_prompt_for  # noqa: PLC0415 - 避免循环依赖

            memory_note = build_memory_prompt_for(session)
            if memory_note:
                layers.append(memory_note)
        except Exception as err:  # noqa: BLE001 - 记忆层异常不应阻断答疑
            log.warning("student memory prompt skipped: {}", err)

    # 自我更新沉淀的经验（core/skills/experience）：放在人格之后 ——
    # 人格是「这个人是谁」（固定），经验是「教久了攒下的手感」（会长），越靠后越能影响输出。
    # 注意这里**只读经验**，人格提示词本体永远由人维护，自动化流程不碰。
    if settings.skill_self_update_enable and profile is not None:
        try:
            from core.skills.experience import build_experience_prompt  # noqa: PLC0415

            # 传 session 是为了让注入**被记账**（同答疑链路），见 L2 的收益回写
            note = build_experience_prompt(
                profile.key, name=profile.display_name(), session=session
            )
            if note:
                layers.append(note)
        except Exception as err:  # noqa: BLE001 - 经验层异常不应阻断答疑
            log.warning("teacher experience prompt skipped: {}", err)

    return layers


def build_messages(
    session: Any,
    *,
    contract: str,
    identity: str = IDENTITY_BASE,
    subject_strict: bool = True,
    hint_text: str | None = None,
    extra_system: str | None = None,
    extra_layers: list[str] | None = None,
    history_limit: int = 6,
    context_notes: list[str] | None = None,
    user_text: str = "",
) -> list[dict[str, Any]]:
    """装配一次 LLM 调用的 messages（答疑 / 讲课共用同一个装配点）。

    层序（越靠后影响力越强）：
    身份基座 -> 任务契约 -> 学科约束 -> 作业提示 -> 任务附加指令 ->
    共享人格层 -> 任务覆盖层 -> 会话历史 -> 本轮输入。
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": identity},
        {"role": "system", "content": contract},
        {"role": "system", "content": build_subject_layer(session, strict=subject_strict)},
    ]
    if hint_text:
        messages.append({"role": "system", "content": hint_text})
    if extra_system:
        messages.append({"role": "system", "content": extra_system})

    for layer in build_profile_layers(session):
        messages.append({"role": "system", "content": layer})

    # 任务覆盖层：放在人格之后，用于纠正人格层里不适用的约定（如讲课不写板书）
    for layer in extra_layers or []:
        if layer:
            messages.append({"role": "system", "content": layer})

    if history_limit > 0:
        for msg in list(getattr(session, "history", []) or [])[-history_limit:]:
            if msg.role in ("student", "ai"):
                messages.append({
                    "role": "user" if msg.role == "student" else "assistant",
                    "content": msg.text,
                })

    user_content = user_text or "请讲解这道题。"
    if context_notes:
        user_content = "\n".join(context_notes) + "\n" + user_content
    messages.append({"role": "user", "content": user_content})
    return messages
