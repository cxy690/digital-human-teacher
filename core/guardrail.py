"""教学护栏与内容安全（G5）：面向未成年人的大班直播课硬性约束。

覆盖 LIVE_CLASS_PIVOT_PLAN.md 第四章 G5 与第十章合规要求：
* **范围护栏**：回答限定在本学科、本学段；超纲问题礼貌引导回课程主线；
* **内容安全**：全量输出经过敏感内容过滤（真实环境应接入内容安全服务）；
* **AI 生成内容标识**：下行事件统一附加水印字段，满足《生成式人工智能服务管理暂行办法》。

设计要点：与 :class:`core.validator.ResponseValidator` 分工——validator 校验「单条回复的
格式与答案泄露」，guardrail 校验「教学范围合规 + 内容安全 + 合规标识」，两者互补。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from config.constants import AI_CONTENT_WATERMARK_TEXT
from config.logging import get_logger
from config.settings import settings

log = get_logger("core.guardrail")

# 敏感词分三级——**分级是这套护栏的核心设计**，不是分类洁癖。接线后每段 AI 话术
# 都会过一遍词库，所以「一刀切遮蔽」会直接损坏教学内容：
#
# 1) 危机信号（CRISIS）：学生表达自伤/自杀意念。绝不能「打码然后继续讲课」，
#    必须走专线——关怀话术 + 求助引导 + 上报。因此它**不在**遮蔽词表里。
# 2) 不当用语（ABUSE）：面向未成年人的输出里绝不该出现，直接遮蔽。
#    注意用「滚开/滚蛋/滚出去」而不是裸「滚」：后者会把物理课的「滚动摩擦」
#    打成「**动摩擦」——这是接线时最容易踩的误伤坑。
# 3) 教学内容敏感（SENSITIVE）：**只标记不遮蔽**。道法课讲「赌博的危害」、
#    生物/健康教育课讲「毒品」「色情信息」都是正当教学内容，
#    一刀切遮蔽等于把合规课程打码。这类只上报，交人工/抽检复核。
#: 危机信号词表：``(标签, 正则)``。**标签是唯一会外泄的部分**——
#: 它会被写进审计文件、显示在教师端，所以必须是能直接读懂的中文短语，
#: 不能是正则源码（``不想活(?!动)`` 这种东西漏到界面上只会让人以为系统坏了）。
_CRISIS_PATTERNS: list[tuple[str, str]] = [
    # —— 明确的自我伤害表述（几乎无歧义，直接命中）——
    ("自杀", r"自杀"), ("自残", r"自残"), ("自伤", r"自伤"), ("轻生", r"轻生"),
    ("割腕", r"割腕"), ("上吊", r"上吊"), ("跳楼", r"跳楼"), ("遗书", r"遗书"),
    ("结束生命", r"结束生命"), ("结束自己", r"结束自己"),
    ("结束这一切", r"结束这一切"), ("伤害自己", r"伤害自己"),
    # —— 死亡意愿 ——
    # ``(?!我了|你了)``：避开「老师我想死你了」「这道题想死我了」这类日常/亲昵说法，
    # 这是本词表里**最容易误伤**的一条，必须有负向断言。
    ("想死", r"想死(?!我了|你了)"),
    ("不如死了", r"不如死了"), ("死了算了", r"死了算了"), ("一了百了", r"一了百了"),
    # —— 生存意愿否定 ——
    ("不想活", r"不想活(?!动)"),          # ``(?!动)``：避开「下雨了不想活动」
    ("活不下去", r"活不下去"), ("活够了", r"活够了"),
    ("撑不下去", r"撑不下去了"), ("熬不下去", r"熬不下去了"),
    # —— 生存意义否定 ——
    # 必须成句：单留「没意思」会把「这道题没意思」判成危机。
    ("活着没意思", r"活着没意思"), ("活着没意义", r"活着没意义"),
    ("活着没劲", r"活着没劲"), ("活着好累", r"活着好累"),
    ("活得没意思", r"活得没意思"),
    # —— 被抛弃感 ——
    ("没人在乎我", r"没人在乎我"), ("没人需要我", r"没人需要我"), ("没人喜欢我", r"没人喜欢我"),
    # —— 寻短见的惯用说法 ——
    ("想不开", r"想不开"),
]

#: 已知误伤（保留记录，便于后续接语义判定时对照）：
#: 「想不开」在极少数语境下可能指思路（如「这道题我想不开」），
#: 但它在中文里作为寻短见的惯用说法更常见，因此**宁可误报**——
#: 误报的代价是一句关怀话术 + 一次人工复核，漏报的代价不可挽回。

_ABUSE_PATTERNS = [
    r"去死", r"笨蛋", r"蠢货", r"蠢蛋", r"白痴", r"傻子",
    r"滚开", r"滚蛋", r"滚出去", r"没用的东西",
]

_SENSITIVE_PATTERNS = [
    r"色情", r"暴力", r"赌博", r"毒品", r"枪支", r"血腥",
]

#: 危机专线话术：不追问细节、不评判，先接住情绪再给求助渠道。
#: 注意它自身**不能命中任何一级词表**（否则过滤会把关怀话术自己打码）。
_CRISIS_SPEECH = (
    "同学，你刚才说的话我很在意，也想认真听你说。你现在还好吗？"
    "如果心里很难受，请一定告诉身边信任的老师或家人，"
    "也可以打 12355 青少年服务热线，那里有专门的人愿意听你说。"
    "我会一直在这里陪着你。"
)

#: 兼容旧名（历史文档/脚本引用过 ``_UNSAFE_PATTERNS``）。
_UNSAFE_PATTERNS = _ABUSE_PATTERNS

# 超纲引导话术：礼貌地把问题拉回课程主线
_REDIRECT_SPEECH = (
    "这个问题很有意思，不过它超出了我们这节课的范围。"
    "我们先把今天的重点弄扎实，课后再一起探讨它，好吗？"
)


@dataclass
class GuardrailVerdict:
    """护栏判定结果。"""

    allowed: bool = True
    in_scope: bool = True                 # 是否落在本学科本学段范围内
    redirect_speech: str = ""             # 超纲时的引导话术
    unsafe: list[str] = field(default_factory=list)   # 命中的不当用语（已遮蔽）
    crisis: list[str] = field(default_factory=list)   # 命中的危机信号（走专线）
    flagged: list[str] = field(default_factory=list)  # 命中的教学内容敏感词（只标记）
    speech: str = ""                      # 需要顶替原话术时的安全话术

    def block(self, reason: str) -> None:
        self.allowed = False
        self.unsafe.append(reason)

    @property
    def hit(self) -> bool:
        """是否命中任何一级（调用方据此决定要不要记 meta / 上报）。"""
        return bool(self.crisis or self.unsafe or self.flagged)


class TeachingGuardrail:
    """教学护栏：范围限定 + 内容安全过滤 + AI 内容标识。"""

    def __init__(
        self,
        *,
        subject: str | None = None,
        grade: str | None = None,
        enable: bool | None = None,
        safety_enable: bool | None = None,
    ) -> None:
        self.subject = (settings.guardrail_subject if subject is None else subject).strip()
        self.grade = (settings.guardrail_grade if grade is None else grade).strip()
        self.enable = settings.guardrail_enable if enable is None else enable
        self.safety_enable = (
            settings.content_safety_enable if safety_enable is None else safety_enable
        )

    # ------------------------------------------------------------------ 范围护栏
    def screen_subject(self, text: str, subject: str) -> GuardrailVerdict:
        """学科范围护栏：学生提问点名了其它学科时，礼貌引导回本学科。

        检测规则（低误伤优先）见 :func:`config.subjects.detect_other_subject`——
        只有当问题**明确指向其它学科**且不涉及本学科关键词时才拦截；
        「这道题怎么解」这类不含学科词的正常提问不会被误拦。
        """
        verdict = GuardrailVerdict()
        if not self.enable:
            return verdict
        subject = (subject or "").strip()
        if not subject or subject == "综合":      # 未选学科（综合课）不做范围限制
            return verdict
        from config.subjects import detect_other_subject  # 局部导入避免循环依赖

        other = detect_other_subject(text or "", subject)
        if other:
            verdict.in_scope = False
            verdict.redirect_speech = (
                f"我们这节课是{subject}课哦，{other}的问题可以先记在小本本上，"
                f"下节{other}课再请老师讲，好吗？现在我们把{subject}的这道题弄明白～"
            )
            log.info("question out of subject scope ({} -> {}): {}", subject, other, (text or "")[:30])
        return verdict

    def screen_question(self, text: str, *, topic_keywords: list[str] | None = None) -> GuardrailVerdict:
        """判定学生提问是否落在本节课范围内。

        ``topic_keywords`` 为本节课/本学科的关键词；命中任一即视为在范围内。
        未提供关键词时不做范围限制（避免误伤），仅返回 in_scope=True。
        """
        verdict = GuardrailVerdict()
        if not self.enable:
            return verdict
        question = (text or "").strip()
        if not question:
            return verdict
        if topic_keywords:
            if not any(kw and kw in question for kw in topic_keywords):
                verdict.in_scope = False
                verdict.redirect_speech = _REDIRECT_SPEECH
                log.info("question out of scope, redirect: {}", question[:30])
        return verdict

    # ------------------------------------------------------------------ 内容安全
    #: 危机专线话术（供上层直接取用，避免各处硬编码同一句话）。
    CRISIS_SPEECH = _CRISIS_SPEECH

    def screen_crisis(self, text: str) -> GuardrailVerdict:
        """危机信号识别（自伤/自杀意念）。

        与 :meth:`filter_output` 的区别在**处置方式**：危机信号不遮蔽、不答疑，
        而是返回 ``allowed=False`` + :attr:`CRISIS_SPEECH`，由调用方走专线
        （播报关怀话术 + 上报 + 转人工）。
        """
        verdict = GuardrailVerdict()
        if not self.safety_enable:
            return verdict
        # 只记**可读标签**，不把原始正则漏给上层：
        # ``hits`` 会一路进入审计文件与教师端界面，露出 ``不想活(?!动)`` 这种
        # 正则源码对复核的老师毫无意义，还容易让人以为系统出了 bug。
        for label, pattern in _CRISIS_PATTERNS:
            if re.search(pattern, text or ""):
                verdict.crisis.append(label)
        if verdict.crisis:
            verdict.allowed = False
            verdict.speech = _CRISIS_SPEECH
            log.warning("危机信号命中：{} | {}", verdict.crisis, (text or "")[:40])
        return verdict

    def flag_sensitive(self, text: str) -> GuardrailVerdict:
        """教学内容敏感词：**只标记不遮蔽**（避免把正当课程内容打码）。"""
        verdict = GuardrailVerdict()
        if not self.safety_enable:
            return verdict
        for pat in _SENSITIVE_PATTERNS:
            if re.search(pat, text or ""):
                verdict.flagged.append(pat)
        return verdict

    def filter_output(self, text: str) -> tuple[str, GuardrailVerdict]:
        """对 AI 输出做内容安全过滤，返回 (安全文本, 判定)。

        三级处置口径：

        * **不当用语** —— 以「*」遮蔽，保证面向未成年人的输出零漏放；
        * **危机信号** —— 不遮蔽，整段顶替为 :attr:`CRISIS_SPEECH`
          （把危机内容「打码后照常讲课」是错误处置）；
        * **教学内容敏感** —— 不改文本，只在 ``flagged`` 里标记，交人工复核。
        """
        verdict = GuardrailVerdict()
        if not self.safety_enable:
            return text, verdict
        safe = text or ""
        for pat in _ABUSE_PATTERNS:
            if re.search(pat, safe):
                verdict.block(f"unsafe:{pat}")
                safe = re.sub(pat, lambda m: "*" * len(m.group(0)), safe)
        # 危机与敏感判定放在遮蔽之后：既避免危机词被 ``*`` 掉后失去专线语义，
        # 也保证关怀话术自身（不含任何词表命中）不会被二次处理——即幂等。
        crisis = self.screen_crisis(safe)
        if crisis.crisis:
            verdict.crisis.extend(crisis.crisis)
            verdict.allowed = False
            verdict.speech = crisis.speech
            safe = crisis.speech
        verdict.flagged.extend(self.flag_sensitive(safe).flagged)
        return safe, verdict

    # ------------------------------------------------------------------ 合规标识
    def tag_ai_content(self, payload: dict) -> dict:
        """给下行事件附加 AI 生成内容标识（合规要求）。"""
        if not settings.ai_content_watermark_enable:
            return payload
        tagged = dict(payload)
        tagged["ai_generated"] = True
        tagged["watermark"] = AI_CONTENT_WATERMARK_TEXT
        return tagged
