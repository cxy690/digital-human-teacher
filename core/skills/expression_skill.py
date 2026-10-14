"""表情技能：模型判断学生情境（表扬场景），驱动前端弹出对应「表情包」。

链路：学生短回答（非疑问/求助/寒暄）→ 最近有 AI 讲解上下文时调用 LLM 判定
（是否作答 / 是否正确 / 出色程度 / 生成表扬话术）→ 命中返回 ExpressionDecision
（表情键 + 表扬话术），由 Supervisor 直接表扬播报，并把表情键随 ai_speech 事件下发，
前端在数字人面板上弹出对应贴纸；未命中返回 None，正常走答疑链路（技能不拦截）。

表情键（对应前端 EMOTES 贴纸表；数字人视频素材互动已下线，数字人固定用讲课素材）：
- praise   一般答对，肯定鼓励 → 👍「答对啦」
- celebrate 表现出色，情绪热烈 → 🎉「太棒了」
"""
from __future__ import annotations

from dataclasses import dataclass

from config.logging import get_logger
from core.state.session import SessionState
from models.base import BaseLLMClient
from models.llm import get_llm

log = get_logger("core.skills.expression_skill")

# 判定档位 → 前端表情包键（经 ai_speech 事件下发，前端弹对应贴纸）
# 注：不再切换数字人视频素材（ai3_2/ai3_3 已下线），数字人固定用默认讲课素材。
EXPRESSION_MATERIALS = {
    "praise": "praise",        # 👍 答对啦
    "celebrate": "celebrate",  # 🎉 太棒了
}

_MAX_ANSWER_LEN = 40  # 作答通常很短；超长按讲解/提问处理

# 疑问/求助/寒暄类词：出现即不进入判定（这些场景由答疑/闲聊链路负责）
_SKIP_HINTS = (
    "吗", "呢", "?", "？", "怎么", "为什么", "什么", "多少", "哪个", "哪道", "谁",
    "帮我", "讲讲", "讲一下", "教我", "解释", "不懂", "不会", "没听懂", "再说一遍",
    "重新讲", "你好", "谢谢", "再见",
)

_SYSTEM_PROMPT = """你是一位中小学 AI 教师的课堂观察助手。根据最近的课堂对话，判断学生最后一条消息
是否在“回答题目”（给出答案/结论/表达弄懂了），以及答得对不对、有多好。

规则：
1. is_answer：只有学生给出答案、算式结果或明确表达“做出来了/我会了”才算；提问、闲聊都不算；
2. correct：必须结合老师最近的讲解判断；背景不足、无法确认对错时，correct 一律填 false；
3. level：普通答对填 normal；回答干脆利落、题目较难或学生自己发现并纠正了错误，填 great；
4. praise：学生答对时老师口吻的表扬（1 句话，30 字以内，口语化、适合直接朗读；
   不用 emoji、引号、星号）；答错或不作答时留空。

只输出 JSON，不要多余文字：
{"is_answer": false, "correct": false, "level": "normal", "praise": ""}"""


@dataclass
class ExpressionDecision:
    """一次表情技能判定结果（仅答对时产生）。"""

    expression: str   # 表情包键（praise / celebrate），前端据此弹贴纸
    praise: str       # 表扬话术（数字人播报文本）
    kind: str         # 判定档位：praise / celebrate
    reason: str = ""  # 判定依据摘要（日志/调试用）


class ExpressionSkill:
    """答对表扬表情技能（规则预筛 + LLM 判定 → 表情素材 + 表扬话术）。"""

    def __init__(self, llm: BaseLLMClient | None = None) -> None:
        self.llm = llm or get_llm()

    # ---------------------------------------------------------------- 预筛
    def should_try(self, text: str, session: SessionState) -> bool:
        """规则预筛：短回答 + 无疑问/求助/寒暄词 + 最近有 AI 讲解上下文。"""
        t = (text or "").strip()
        if not t or len(t) > _MAX_ANSWER_LEN:
            return False
        if any(kw in t for kw in _SKIP_HINTS):
            return False
        return self._has_teaching_context(session)

    @staticmethod
    def _has_teaching_context(session: SessionState) -> bool:
        """最近对话里需要有 AI 的实质讲解（短句提醒不算），否则无从判断对错。"""
        return any(
            m.role == "ai" and len((m.text or "").strip()) >= 12
            for m in session.history[-8:]
        )

    # ---------------------------------------------------------------- 判定
    async def judge(self, text: str, session: SessionState) -> ExpressionDecision | None:
        """判定学生最新输入是否为“答对”；命中返回表情决策，否则 None。"""
        if not self.should_try(text, session):
            return None
        try:
            data = await self.llm.chat_json(
                self._build_messages(text, session), temperature=0.2, max_tokens=200
            )
        except Exception as err:  # noqa: BLE001 - 判定失败不阻塞答疑
            log.debug("expression judge skipped: {}", err)
            return None
        if not data or not data.get("is_answer") or not data.get("correct"):
            return None
        kind = "celebrate" if str(data.get("level")) == "great" else "praise"
        praise = " ".join(str(data.get("praise") or "").split())[:60]
        if not praise:
            praise = "太棒了，回答正确！" if kind == "praise" else "太出色了，给你一个大大的赞！"
        decision = ExpressionDecision(
            expression=EXPRESSION_MATERIALS[kind],
            praise=praise,
            kind=kind,
            reason=str(data.get("reason") or "")[:80],
        )
        log.info(
            "expression hit: kind={} emote={} praise={}",
            decision.kind, decision.expression, decision.praise,
        )
        return decision

    # ---------------------------------------------------------------- 内部
    @staticmethod
    def _build_messages(text: str, session: SessionState) -> list[dict[str, str]]:
        lines = []
        for m in session.history[-8:]:
            content = (m.text or "").strip()
            if m.role in ("student", "ai") and content:
                lines.append(f"{'老师' if m.role == 'ai' else '学生'}：{content[:160]}")
        context = "\n".join(lines) or "（无历史）"
        user = (
            f"最近的课堂对话（按时间顺序，最后一条为学生最新消息）：\n{context}\n\n"
            f"学生最新一句话：{text[:120]}\n请判断并输出 JSON。"
        )
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
