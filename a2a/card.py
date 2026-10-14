"""Agent Card：对外声明本系统的能力与调用入口。

卡片内容刻意**与实际实现对齐**，不吹牛：
* ``capabilities.streaming = False`` —— 当前只实现阻塞式 ``message/send``，
  流式（``message/stream`` + SSE）与 push notification 还没做；
* 三个 skill 分别对应已经落地的三块能力：答疑讲解、意图识别/多意图拆解、白板板书。
  改动能力时**同时改这里**，否则外部 client 会按卡片调用不存在的东西。
"""
from __future__ import annotations

from a2a.models import AgentCapabilities, AgentCard, AgentSkill

_SKILLS: tuple[AgentSkill, ...] = (
    AgentSkill(
        id="tutor_qa",
        name="学科答疑讲解",
        description=(
            "面向小学一年级至初三的全学科一对一讲解：第一句先给结论，"
            "再分步推导（每步含依据与关键式子），并给出知识点与易错点。"
        ),
        tags=["教育", "答疑", "全学科", "K12"],
        examples=[
            "老师，这道方程题怎么解？",
            "为什么天空是蓝色的？",
            "帮我讲讲这个英语时态",
        ],
    ),
    AgentSkill(
        id="intent_triage",
        name="意图识别与多意图拆解",
        description=(
            "入口先判简单意图（控制指令 / 闲聊 / 单一提问），"
            "复杂或多意图交由小模型复核并拆成子问题清单，逐条独立作答。"
        ),
        tags=["意图识别", "多意图", "分流"],
        examples=[
            "这道题怎么解？另外帮我打开麦克风",
            "这道方程题怎么解？另外这道函数题怎么解？",
        ],
    ),
    AgentSkill(
        id="whiteboard",
        name="白板板书",
        description=(
            "把讲解以结构化白板指令返回（标题 / 分步文字 / 圈画 / 高亮），"
            "调用方可自行渲染；同时返回本回合的追问题目（若有）。"
        ),
        tags=["白板", "板书", "结构化输出"],
    ),
)


def build_agent_card(base_url: str = "") -> AgentCard:
    """构造 Agent Card；``base_url`` 一般传请求的根地址（网关 / 隧道域名）。"""
    root = (base_url or "").rstrip("/")
    return AgentCard(
        name="education-tutor",
        description="AI 一对一实时答疑教师（全学科讲解 + 白板板书 + 多意图拆解）",
        url=f"{root}/a2a" if root else "/a2a",
        version="0.1.0",
        capabilities=AgentCapabilities(streaming=False, pushNotifications=False),
        skills=list(_SKILLS),
    )
