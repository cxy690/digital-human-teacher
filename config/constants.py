"""全局常量：事件类型、麦控状态、白板指令类型与课堂编排参数。

这里的数值是策略层的唯一事实来源，避免阈值散落在业务代码里。
"""
from __future__ import annotations

from enum import Enum


class MicState(str, Enum):
    """智能麦控对话状态机。"""

    IDLE = "idle"                    # 空闲，等待学生开口
    STUDENT_SPEAKING = "student_speaking"  # 学生正在说话：AI 静默倾听
    STUDENT_SILENT = "student_silent"      # 学生静默，AI 可触发答疑
    AI_SPEAKING = "ai_speaking"            # AI 正在讲解
    BARGED_IN = "barged_in"                # 学生插话，AI 需让出话语权


class BoardOp(str, Enum):
    """白板操作指令类型（AI 自主答疑 / 教师批注 / 课件讲解）。"""

    TEXT = "text"
    DRAW = "draw"
    SHAPE = "shape"
    ANNOTATE = "annotate"   # 在图片上层圈画
    ERASE = "erase"
    HIGHLIGHT = "highlight"
    CLEAR = "clear"
    SLIDE = "slide"         # 切换课件页（白板背景换成该页，并清空上一页的标注）


class IntentType(str, Enum):
    """学生输入意图路由类型。"""

    VOICE_QUESTION = "voice_question"
    TEXT_QUESTION = "text_question"
    IMAGE_QUESTION = "image_question"
    CONTROL = "control"
    CHITCHAT = "chitchat"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# 麦控 / 语音交互
# ---------------------------------------------------------------------------

VAD_SILENCE_MS = 700           # 静音判定窗口：超过该时长无语音视为学生停止提问
VAD_SPEECH_MS = 180            # 连续语音达到该时长视为学生开口
BARGE_IN_GRACE_MS = 120        # 学生插话时 AI 让出话语权的宽限时长
AI_MIN_RESPONSE_GAP_MS = 400   # AI 两次播报之间最小间隔

# ---------------------------------------------------------------------------
# 音视频 / 白板
# ---------------------------------------------------------------------------

TARGET_AV_LATENCY_MS = 300          # 音视频端到端延迟目标
TARGET_WHITEBOARD_SYNC_MS = 120     # 白板同步延迟目标
TARGET_VOICE_RESPONSE_MS = 1200     # AI 语音交互首响目标

# 上传图片：前端会先等比压缩，这里做服务端兜底上限（解码后字节数）
MAX_IMAGE_BYTES = 8 * 1024 * 1024


# ---------------------------------------------------------------------------
# 大班直播课 · 分层混合架构（对应 LIVE_CLASS_PIVOT_PLAN.md）
# ---------------------------------------------------------------------------


class ContentLayer(str, Enum):
    """内容层：一节课按生成方式与延迟要求分三层。"""

    L1_LECTURE = "l1_lecture"        # 主讲层：离线预生成，按知识点切片入库（≈80% 时长）
    L2_INTERACTION = "l2_interaction"  # 程序交互层：预制动作片段 + 变量插槽（<1s）
    L3_QA = "l3_qa"                  # 自由问答层：ASR→LLM(RAG)→流式TTS→实时数字人（<3s）


class ClassPhase(str, Enum):
    """直播编排引擎的课堂流程节点（状态机状态）。"""

    OPENING = "opening"        # 开场寒暄
    REVIEW = "review"          # 复习上节
    LECTURE = "lecture"        # 新知讲授（播放 L1 切片）
    INTERACTION = "interaction"  # 提问互动（L2 点名/答题）
    QUIZ = "quiz"              # 随堂测验
    SUMMARY = "summary"        # 总结与作业布置
    ENDED = "ended"            # 课堂结束


# 课堂流程的默认线性顺序（编排引擎按此推进，节点可超时降级）
CLASS_PHASE_ORDER: list[ClassPhase] = [
    ClassPhase.OPENING,
    ClassPhase.REVIEW,
    ClassPhase.LECTURE,
    ClassPhase.INTERACTION,
    ClassPhase.QUIZ,
    ClassPhase.SUMMARY,
    ClassPhase.ENDED,
]


class InteractionType(str, Enum):
    """L2 程序交互类型。"""

    ROLL_CALL = "roll_call"      # 点名
    ANSWER = "answer"            # 答题
    LOTTERY = "lottery"          # 抽奖
    AWARD = "award"              # 发奖状
    READ_ALONG = "read_along"    # 跟读评分


# 节点超时降级：学生无响应达到该秒数时，编排引擎自动继续下一环节
PHASE_TIMEOUT_SECONDS: dict[ClassPhase, float] = {
    ClassPhase.INTERACTION: 15.0,
    ClassPhase.QUIZ: 30.0,
}

# 承接口播话术库（默认值；可被 data/interaction_phrases/*.json 覆盖）。
# 让节点切换「像真人」——直播观感七成靠编排话术而非模型。
TRANSITION_PHRASES: dict[str, list[str]] = {
    "opening": ["同学们好，欢迎回到我们的课堂，今天也要一起加油哦！"],
    "review": ["上节课的内容大家都掌握了吗？我们先快速回顾一下。"],
    "lecture": ["接下来进入今天的新知识，请大家集中注意力。"],
    "interaction": ["我看到有同学已经答对了，非常棒！我们请下一位。"],
    "quiz": ["现在是随堂小测时间，看看谁掌握得最好。"],
    "summary": ["今天我们学习了这些重点，课后记得完成作业巩固哦。"],
    "answer_correct": ["答对啦，真棒！", "完全正确，给你点个赞！"],
    "answer_wrong": ["再想想看，别着急，我们一起理清思路。"],
    "timeout": ["没关系，我们先继续往下讲，这道题课后可以再思考。"],
}

# L2 真人动作基元（预录片段，肢体动作不用扩散模型实时生成）
ACTION_PRIMITIVES: tuple[str, ...] = (
    "point_to_courseware",  # 指向课件
    "write_board",          # 板书
    "applaud",              # 鼓掌
    "thumbs_up",            # 竖大拇指
    "turn_page",            # 翻页
    "thinking",             # 思考
)

# AI 生成内容标识（合规要求：全程加水印/标识）
AI_CONTENT_WATERMARK_TEXT = "AI 生成内容"


# ---------------------------------------------------------------------------
# 1v1 学科辅导课流程（学科选择页 -> 打招呼 -> 讲解+出题 -> 总结答疑）
# 复用上面 ClassPhase 枚举语义，避免再造一套状态机
# ---------------------------------------------------------------------------

# 一次辅导的阶段顺序
TUTOR_PHASE_ORDER: list[ClassPhase] = [
    ClassPhase.OPENING,      # 问候热身
    ClassPhase.LECTURE,      # 知识点讲解（学生提问驱动）
    ClassPhase.INTERACTION,  # 互动出题（模拟题 + 判题反馈）
    ClassPhase.SUMMARY,      # 总结答疑
]

# 阶段中文名（前端阶段条与播报共用）
TUTOR_PHASE_LABELS: dict[ClassPhase, str] = {
    ClassPhase.OPENING: "问候热身",
    ClassPhase.LECTURE: "知识点讲解",
    ClassPhase.INTERACTION: "互动出题",
    ClassPhase.SUMMARY: "总结答疑",
    ClassPhase.ENDED: "已结束",
}

# 阶段时长占比（按总时长切分，各项之和为 1.0）。
# 40 分钟课时 -> 讲解 30 分钟 + 出题 10 分钟（与 COURSE_QUIZ_INTERVAL_MINUTES=30 对齐）：
# 问候热身并入开课招呼（start_course 实时播报），总结答疑并入出题回合的收尾对话，
# 两者不再单独占阶段时长。
TUTOR_PHASE_RATIOS: dict[ClassPhase, float] = {
    ClassPhase.OPENING: 0.0,
    ClassPhase.LECTURE: 0.75,
    ClassPhase.INTERACTION: 0.25,
    ClassPhase.SUMMARY: 0.0,
}

# 教师形象状态（前端据此切换表情/动画）
TUTOR_MOODS: tuple[str, ...] = (
    "idle",       # 待机
    "greeting",   # 打招呼
    "speaking",   # 讲解中
    "thinking",   # 思考中
    "praising",   # 表扬
    "waiting",    # 等待回答
)


def is_valid_mood(mood: str | None) -> bool:
    """校验教师形象状态是否合法（防止前后端状态名拼写漂移）。"""
    return bool(mood) and mood in TUTOR_MOODS
