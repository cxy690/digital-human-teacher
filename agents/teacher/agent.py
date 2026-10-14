"""AI 智能教师（答疑）：讲题、图片题、白板书写、追问判定、出题判题与语音话术。

职责边界（协作关系见 :mod:`agents.protocol`）：
* 本模块负责**答疑**这条链路；
* **课件逐页讲解**已拆给 :class:`~agents.lecturer.agent.LecturerAgent`，
  本模块只保留 ``explain_slide`` 的兼容转发与回退入口；
* 提示词按「共享层 / 任务层」分离：身份、人格、风格、学科约束走
  :mod:`agents.shared.teaching_layers`（与讲课链路共享），
  答疑契约（``ANSWER_CONTRACT``）只属于本模块。

对应 PRD「AI 智能教师核心能力」：
* 实时语音 Agent + 智能麦控对话（话术侧）；
* AI 自主白板答疑、图片题目智能讲解、全维度交互响应。
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import time
from typing import TYPE_CHECKING, Any, AsyncIterator

from agents.base import BaseAgent
from agents.shared.emphasis import strip_emphasis
from agents.shared.reply_parser import (
    looks_like_json,
    parse_json_reply,
    salvage_object,
    salvage_text_field,
)
from agents.shared.teaching_layers import (
    ANSWER_CONTRACT,
    build_messages,
    style_text,
)
from agents.shared.toolkit import AgentToolkit
from agents.shared.whiteboard import Whiteboard
from agents.teacher.react import ReActLoop
from config.constants import BoardOp
from config.logging import get_logger
from config.settings import settings
from config.teachers import teacher_for_subject
from core import exercise_bank, math_solver
from core.course_session import greeting_text, phase_intro_text
from core.hint_homework import HintLevel
from core.state.models import AIResponse, QuizItem, StudentTurn, WhiteboardOp
from core.state.session import SessionState
from models.base import BaseLLMClient
from models.llm import get_llm
from models.tts import split_sentences

if TYPE_CHECKING:  # 仅类型标注，避免运行时循环导入
    from agents.lecturer.agent import LecturerAgent
    from core.courseware.models import Slide, SlideDeck

# 提示词已按「共享层 / 任务层」拆分到 agents/shared/teaching_layers.py：
#   * 身份基座（IDENTITY_BASE）与老师人格、风格、学科约束由两个 agent 共享；
#   * 答疑契约（ANSWER_CONTRACT）只属于本 agent，讲课件契约（LECTURE_CONTRACT）
#     只属于 LecturerAgent —— 拆分前两者被塞进同一个 system，模型只能猜，这才是
#     「讲 PPT 变成念 PPT」的根因。
# 本模块不再自持任何提示词常量，只负责「答疑」这个任务本身。

# 流式讲解专用：只产出朗读话术（无 JSON），让第一句尽快到达 TTS/数字人
_SPEECH_ONLY_PROMPT = (
    "现在只要朗读话术，不要输出 JSON。"
    "第一句先给结论/答案方向，然后用不超过 3 步把思路讲完（把关键步骤内容都讲到，让学生只听语音也能跟上）。"
    "总长不超过 150 字，口语化、温柔鼓励，不要标题、序号、Markdown 或任何标记，直接输出要朗读的正文。"
)

# 白板先行：先只做「读题与题意理解」，用口语讲出来（不写白板、不给最终答案）
_UNDERSTAND_PROMPT = """现在请先只做「读题与题意理解」，并把它用**口语**讲出来（这一步**只朗读、不写白板、不给最终答案**）。
只输出 JSON，字段为：
{{"title": "本题主题（不超过12字）",
  "understanding": ["对题意的理解要点", "..."],
  "steps": ["初步解题思路", "..."],
  "speech": "把上述理解用大白话讲给学生听（3~5 句：题目给了什么条件、要求什么、涉及哪个知识点，以及初步思路的方向）"}}
要求：
1. understanding 写 2~4 条：题目给了哪些已知条件、要求什么、涉及哪个知识点或方法；
2. steps 写 2~4 条：初步的解题方向（可以不含最终计算结果）；
3. speech 必须是能直接朗读的口语，不要出现序号、括号、Markdown 等符号；
4. 只输出 JSON，不要任何多余解释。"""

# 方案 B（单次调用·两段式）：一次 LLM 同时产出「理解」与「解答」两段，
# 用哨兵分隔，流式消费时在段落边界先把「理解」上板，再上「理解 + 解答」。
_UNDERSTANDING_SENTINEL = "###UNDERSTANDING###"
_SOLUTION_SENTINEL = "###SOLUTION###"

_BOARD_FIRST_PROMPT = """请**一次性**按下面顺序输出两段内容，两段之间与前后都不要有任何多余文字：

###UNDERSTANDING###
{"title": "本题主题（不超过12字）",
 "understanding": ["对题意的理解要点1", "理解要点2"],
 "steps": ["初步解题思路1", "思路2"],
 "speech": "把上述理解用大白话讲给学生听（3~5句，可直接朗读，不要序号/括号）"}
###SOLUTION###
{"title": "白板标题",
 "steps": ["4~6 步详解：每步写清「做什么 + 为什么/依据」，含关键式子或数字，不跳步", "…"],
 "equation": "本题要解的等式（含未知数），如 3x+5=20；无则留空",
 "answer": "最终答案，如 x=5；无则留空",
 "knowledge_points": ["2~3 个具体知识点（不是笼统章节名），如「移项法则：移项要变号」"],
 "pitfalls": ["1~2 个易错点：「错在哪 + 怎么避免」"],
 "speech": "准备朗读给学生听的口语化讲解（80字以内，第一句先给结论）",
 "followup": {"question": "讲完后要问学生的一个检验/追问问题（1句，口语化）",
              "answer": "该问题的标准答案（简短）",
              "hint": "学生答错时的引导提示（1句）",
              "options": ["可选的选项A", "可选的选项B"]}}

要求：
1. 第一段（###UNDERSTANDING###）只写「题意理解 + 初步思路」：已知条件、要求什么、涉及哪个知识点；**不要给最终答案**；
   其中 speech 是老师接下来要**朗读**的话（不写白板），必须是口语、可直接朗读；
2. 第二段（###SOLUTION###）**在第一段理解的基础上继续推导**，给出结论、4~6 步详解
   （每步含依据与关键式子）、2~3 个具体知识点、1~2 个易错点；**不要重复照抄第一段的文字**；
3. **必须严格按上面给出的字段顺序输出**：`title` 和 `steps` 一定要最先写、`speech` 放在后面
   （白板要先于语音出现在学生眼前，所以板书内容必须先输出）；
4. 两段都必须是合法 JSON；除这两段外不要输出任何其它文字、解释或 markdown 代码块标记。"""

# ReAct 闭环：允许模型在作答前先调工具把事实弄清楚（见 agents/teacher/react.py）。
# 与其它提示词一样，只加「怎么用工具」这一件事，输出契约仍由 ANSWER_CONTRACT 决定。
_REACT_PROMPT = """写讲解之前，你也可以**先调用工具**把事实弄清楚，再动笔：
1. 概念、公式、教材原文不确定 -> 先 knowledge_search 查证，不要凭印象编；
2. 题目含方程 / 等式 -> 先用 math_solve 求出结果，再用 math_verify 复核，然后写进讲解；
3. 图片题条件看不清 -> 用 ocr_parse 取回识别到的题干；
4. 学生提到「刚才那道题 / 上面那一步」-> 用 read_board 确认白板上写了什么。
要求：
* 需要用工具就直接发起调用，**这一轮不要输出最终 JSON**；拿到结果后再按下面的 JSON 契约作答；
* 工具结果与你的直觉冲突时，**以工具结果为准**，并在讲解里点明学生可能错在哪；
* 不需要工具（闲聊式问答、概念解释）就直接作答，不要为了用工具而用工具；
* 无论用没用工具，最终答复必须是下面规定的 JSON 结构。"""

_SENT_END = "。！？!?；;\n"


def _segment(buf: str, start: str, end: str | None = None) -> str:
    """取 ``start`` 之后、``end`` 之前（不含）的片段；找不到 ``start`` 返回空串。"""
    i = buf.find(start)
    if i < 0:
        return ""
    i += len(start)
    if end is None:
        return buf[i:]
    j = buf.find(end, i)
    return buf[i:j] if j >= 0 else buf[i:]


def _drain_sentences(buf: str) -> tuple[list[str], str]:
    """从流式缓冲里切出完整句子，返回 ``(完整句列表, 剩余缓冲)``。"""
    sentences: list[str] = []
    start = 0
    for i, ch in enumerate(buf):
        if ch in _SENT_END:
            piece = buf[start:i + 1].strip()
            if piece:
                sentences.append(piece)
            start = i + 1
    return sentences, buf[start:]


_STEPS_ARRAY_RE = re.compile(r'"steps"\s*:\s*\[')
_TITLE_FIELD_RE = re.compile(r'"title"\s*:\s*"((?:[^"\\]|\\.)*)"')

# JSON 标准转义（\uXXXX 单独处理）；不解码会让流式板书里的换行/引号显示成字面量，
# 与最终 json.loads 出来的文本不一致
_JSON_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


def _drain_title(buf: str) -> str:
    """从流式缓冲里取 ``"title"`` 字段（未出现或未闭合时返回空串）。"""
    m = _TITLE_FIELD_RE.search(buf or "")
    return m.group(1).strip() if m else ""


def drain_steps(buf: str) -> list[str]:
    """从**流式**缓冲里取出 ``"steps"`` 数组里已经写成完整字符串的步骤。

    为什么需要：白板要「边生成边上屏」。模型按字段顺序先输出 ``title``/``steps``，
    这里的轻量扫描就能在第 1 步刚写完时立刻把它交给白板，而不是等整段 JSON 结束
    （后者正是「问了半天白板才出来」的原因）。

    实现要点：只扫描 steps 数组内部；遇到引号就整段消费（含 ``\\`` 转义并**解码**），
    因此步骤文本里的 ``,``/``]`` 不会破坏数组边界；未闭合的末项直接丢弃，
    等下一批增量再取。多次调用对同一缓冲是幂等的（调用方按数量去重）。
    """
    m = _STEPS_ARRAY_RE.search(buf)
    if not m:
        return []
    seg = buf[m.end():]
    items: list[str] = []
    i = 0
    n = len(seg)
    while i < n:
        if seg[i] == "]":          # 数组已闭合
            break
        if seg[i] != '"':
            i += 1
            continue
        i += 1                      # 跨过起始引号
        out: list[str] = []
        closed = False
        while i < n:
            ch = seg[i]
            if ch == "\\" and i + 1 < n:
                nxt = seg[i + 1]
                if nxt == "u" and i + 5 < n:          # \uXXXX
                    try:
                        out.append(chr(int(seg[i + 2:i + 6], 16)))
                        i += 6
                        continue
                    except ValueError:
                        pass
                out.append(_JSON_ESCAPES.get(nxt, nxt))
                i += 2
                continue
            if ch == '"':
                closed = True
                i += 1
                break
            out.append(ch)
            i += 1
        if not closed:              # 这一项还没写完，留到下一批
            break
        text = "".join(out).strip()
        if text:
            items.append(text)
    return items


# 课件（PPT）讲解已整体迁出到 agents/lecturer/agent.py（LecturerAgent）：
# 契约常量见 agents/shared/teaching_layers.py 的 LECTURE_CONTRACT，
# 标注映射（_slide_annotations / _deck_context / _element_catalog）随之迁移。
# 本模块保留同名静态方法作为**兼容转发**，方法体见文件末尾。

# 作业模式下的分层提示指令会替换默认讲解风格
_HINT_STYLE: dict[HintLevel, str] = {
    HintLevel.CONCEPT: "只引导学生回忆知识点，不要给出方法与答案。",
    HintLevel.METHOD: "只给出解题思路方向，不要写具体步骤与答案。",
    HintLevel.STEP: "给出解题步骤，但最后一步留白，让学生自己算。",
    HintLevel.ANSWER: "给出完整步骤与标准答案。",
}

# 讲课风格（TEACHING_STYLE）已并入共享层：agents/shared/teaching_layers.py 的
# STYLE_GUIDE / style_text() —— 答疑与讲课必须用同一套风格档位。

# 步骤自带的前缀（“第一步：”“步骤一”“1.”“（1）”等）：白板会再加“1.”、播报会再加“第1步，”，
# 且前缀可能叠加（如“1. 第一步：”），故在源头循环剥离，避免重复编号
_STEP_PREFIX = re.compile(
    r"^\s*(?:(?:第\s*[0-9一二三四五六七八九十百]+\s*步|步骤\s*[0-9一二三四五六七八九十百]+"
    r"|[（(]\s*[0-9一二三四五六七八九十百]+\s*[)）])\s*[:：、,，.。．]?\s*"
    r"|[0-9]+\s*[.、．)）]\s*)"
)

log = get_logger("agents.teacher")


class TeacherAgent(BaseAgent):
    """AI 智能教师智能体（答疑任务）。

    职责边界：讲题 / 图片题 / 白板书写 / 追问判定 / 出题判题 / 课程问候与阶段过渡。
    **课件逐页讲解不在这里** —— 那是 :class:`~agents.lecturer.agent.LecturerAgent`
    的职责；本类只保留 ``explain_slide`` 的兼容转发（见文件后段）。
    """

    role = "teacher"

    def __init__(
        self,
        llm: BaseLLMClient | None = None,
        board: Whiteboard | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        self.llm = llm or get_llm()
        self.board = board or Whiteboard()
        # 讲课件 agent 懒建（兼容转发 / 回退用），首次调用才实例化
        self._lecturer_agent: "LecturerAgent | None" = None

    # ---------------------------------------------------------------- 文本答疑
    async def answer(
        self,
        turn: StudentTurn,
        session: SessionState,
        *,
        hint_level: HintLevel | None = None,
        context_notes: list[str] | None = None,
        toolkit: AgentToolkit | None = None,
    ) -> AIResponse:
        """对文字/语音提问生成讲解与白板内容。

        加装 ReAct（``AGENT_REACT_ENABLE``）后先在闭环里跑「思考 -> 行动 -> 观察」：
        模型可自行检索教材、解方程、复核答案，再把结论写进讲解；
        闭环拿到空答复（不支持工具 / 超时 / 异常）时**自动退回**单次生成链路 ——
        也就是加装之前的行为，不会有「因为闭环失败而整题不答」的情况。
        """
        if settings.agent_react_enable and settings.agent_tools_enable:
            react_response = await self.answer_with_react(
                turn, session,
                hint_level=hint_level, context_notes=context_notes, toolkit=toolkit,
            )
            if react_response is not None:
                return react_response
        return await self._answer_plain(
            turn, session, hint_level=hint_level, context_notes=context_notes
        )

    async def _answer_plain(
        self,
        turn: StudentTurn,
        session: SessionState,
        *,
        hint_level: HintLevel | None = None,
        context_notes: list[str] | None = None,
    ) -> AIResponse:
        """单次生成（加装工具之前的原始链路；闭环失败时的回退目标）。"""
        messages = self._build_messages(turn, session, hint_level=hint_level, context_notes=context_notes)
        # json_mode：这一步产出的是上千 token 的 JSON 契约，交给厂商保证结构合法
        # （实测长输出偶发多写一个 '['，一旦非法整段讲解就废了 —— compose_answer 会先修复）
        text, llm_error = await self._safe_chat(messages, json_mode=True)
        response = await self.compose_answer(text)
        if llm_error:
            # 兜底是为了学生，不是为了把问题藏起来：故障必须可见，
            # 由 core.verdict.judge 判为降级（收口见 tests/test_failure_cluster.py）。
            response.meta["llm_error"] = llm_error
        return response

    async def _safe_chat(self, messages: Any, *, json_mode: bool = False) -> tuple[str, str]:
        """调模型，**失败不抛**：返回 ``(正文, 失败摘要)``，失败时正文为空串。

        为什么要兜这一层：答疑链路的生成节点原本没有兜底，**上游 LLM 一次 HTTP 抖动
        （429/5xx）就让整个回合直接失败** —— 学生那边老师干脆不说话了。这与本项目
        到处写着的「绝不因外部服务抖动把课卡死」是矛盾的。

        关键取舍：**兜底是为了学生，不是为了把问题藏起来**。失败摘要会写进
        ``response.meta["llm_error"]``，由 ``core.verdict.judge`` 判为降级 ——
        所以学生听到兜底话术的同时，L0/L3 依然看得见这次故障。
        悄悄吞掉它（不标 meta）会让可观测层只剩「一切正常」，比不兜底更糟。

        顺带把输出上限统一到 ``ANSWER_MAX_TOKENS``：原来各条链路各写各的默认值，
        长讲解会被截断成不合法的 JSON（整块降级、白板全空）。
        """
        try:
            text = await self.llm.chat(
                messages, max_tokens=int(settings.answer_max_tokens), json_mode=json_mode
            )
            return str(text or ""), ""
        except Exception as err:  # noqa: BLE001 - 模型/网络抖动绝不能把整回合打死
            log.warning("答疑模型调用失败，降级为兜底话术：{}", err)
            return "", f"{type(err).__name__}: {str(err)[:80]}"

    async def answer_with_react(
        self,
        turn: StudentTurn,
        session: SessionState,
        *,
        hint_level: HintLevel | None = None,
        context_notes: list[str] | None = None,
        toolkit: AgentToolkit | None = None,
    ) -> AIResponse | None:
        """走 ReAct 闭环作答；闭环没能产出文本时返回 ``None``（交由调用方回退）。

        ``toolkit`` 由装配方（``Supervisor``）注入 —— 它带着本会话的检索器 / OCR /
        白板，因此工具看到的就是这节课的真实上下文；未注入时退化为只带白板的工具集。
        """
        kit = toolkit if toolkit is not None else AgentToolkit(session, board=self.board)
        if not kit.enabled:
            return None
        kit.reset_turn()
        messages = self._build_messages(
            turn, session, hint_level=hint_level, context_notes=context_notes,
            extra_system=_REACT_PROMPT,
        )
        result = await ReActLoop(self.llm, kit).run(messages)
        turn.meta["react"] = result.to_meta()
        turn.meta["tool_calls"] = kit.trace()
        if not result.text.strip():
            return None
        response = await self.compose_answer(result.text)
        self._apply_tool_board_notes(response, kit)
        response.meta["react"] = {
            "steps": len(result.steps),
            "tool_calls": result.tool_count,
            "stopped": result.stopped,
            "elapsed_ms": result.elapsed_ms,
        }
        return response

    def _apply_tool_board_notes(self, response: AIResponse, toolkit: AgentToolkit) -> None:
        """把工具侧要求补充的板书行接到契约板书**之后**（版式仍由 Whiteboard 负责）。

        为什么要接在后面而不是让工具直接下发坐标：白板的行距 / 防重叠 / 归一化坐标
        由布局器统一计算，模型给坐标只会把版面写乱。这里按「已有内容的最后一行再往下
        一个行距」定位，因此补充行总是落在板书末尾，且不会盖住知识点 / 易错点区块。
        """
        notes = toolkit.consume_board_notes()
        if not notes:
            return
        ops = list(response.board_ops)
        gap = self.board.EXPLAIN_LINE_GAP
        ys = [float(op.y) for op in ops if op.op is BoardOp.TEXT and op.y is not None]
        y = min(0.94 - gap, (max(ys) if ys else self.board.EXPLAIN_START_Y - gap) + gap)
        for offset, note in enumerate(notes):
            ops.append(
                self.board.write_text(
                    note, self.board.EXPLAIN_X, round(min(0.94, y + offset * gap), 3),
                    color="#0958d9",
                )
            )
        response.board_ops = ops

    async def compose_answer(self, text: str) -> AIResponse:
        """组装答疑产物，四层：正常解析 -> 就地抢救 -> 让模型重发 -> 兜底话术。

        为什么要有「就地抢救」这一层：``response_format=json_object`` 只是**请求**厂商
        保证结构，实测 DeepSeek 在长输出（1000+ 字）下仍会偶发把数组括号写坏（3 次里 2 次）；
        而只靠兜底的话，那段讲解的**板书会整块丢掉** —— 对教学是可感知的损失。
        让模型重发一遍也不行：同样是 1000+ 字长输出，实测照样写坏（日志出现过
        「修复后 1023 字仍然损坏」）。所以先走**本地确定性抢救**（按字段名扫描正文），
        救不出来才请模型重发，最后才退兜底话术。
        """
        if not self._parse_reply(text) and looks_like_json(text):
            # ① 本地按字段抢救：免费、且正对症（坏法就是数组括号错位）
            salvaged = salvage_object(text)
            if salvaged.get("speech") or salvaged.get("steps"):
                log.info(
                    "答疑 JSON 结构损坏，已就地抢救（原文 {} 字 -> {} 个字段）",
                    len(text or ""), len(salvaged),
                )
                text = json.dumps(salvaged, ensure_ascii=False)
            else:
                # ② 本地救不出来（不是括号错位那种坏法）：请模型照原文重发一次
                repaired = await self._repair_json(text)
                if repaired:
                    log.info("答疑 JSON 已让模型重发（原文 {} 字）", len(text or ""))
                    # 重发回来的也可能坏 —— 再免费抢救一遍，别白花这次调用
                    again = salvage_object(repaired)
                    text = (
                        json.dumps(again, ensure_ascii=False)
                        if again.get("speech") or again.get("steps")
                        else repaired
                    )
        return self._compose_response(text)

    async def _repair_json(self, broken: str) -> str:
        """把写坏的 JSON 交给模型「只修语法、不改内容」重发一次；失败返回空串。"""
        prompt = (
            "下面这段 JSON 是你刚才的回复，**内容是对的，但括号/引号有语法错误**。\n"
            "请把它重新输出为**合法 JSON**：只修语法（补齐或去掉多余的 [ ] { } \" ,），"
            "一个字都不要改写、不要增删内容，不要输出任何解释或代码块标记。\n\n"
            f"{broken}"
        )
        try:
            return await self.llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=min(4096, max(600, len(broken) // 2)),
                json_mode=True,
            )
        except Exception as err:  # noqa: BLE001 - 修不好就走兜底，绝不能因此中断答疑
            log.warning("JSON 修复调用失败：{}", err)
            return ""

    def _compose_response(self, text: str) -> AIResponse:
        """把 LLM 返回的 JSON 文本归一化为 ``AIResponse``（含白板、话术、追问、验算字段）。"""
        data = self._parse_reply(text)
        if not data and looks_like_json(text):
            # 模型想给 JSON 但结构写坏了（实测长讲解里偶发多一个 '['）：
            # **绝不能**把原文当话术 —— 学生会听到一串 {"speech": … 的 JSON，
            # 板书也会被 _split_steps 切成乱码。捞回 speech 那一句，其余老实兜底。
            salvaged = salvage_text_field(text, "speech")
            log.warning(
                "答疑 JSON 结构损坏，已兜底（原文 {} 字，抢救到话术 {} 字）",
                len(text or ""), len(salvaged),
            )
            return AIResponse(
                speech=strip_emphasis(salvaged)
                or "这道题我再理一下思路，你告诉我卡在哪一步，我们接着看。",
                meta={"mood": "thinking", "json_degraded": True},
            )

        # 重点标记（``**…**``）只给白板用：话术是朗读文本，星号会被念出来；
        # 而且它还会被写进聊天记录与学情记忆，留着就是两个碍眼的符号。
        speech = strip_emphasis(data.get("speech") or self._first_sentence(text))
        steps = [self._strip_step_prefix(s) for s in (data.get("steps") or self._split_steps(text))]
        title = data.get("title") or "题目讲解"

        ops = self.board.build_explain(
            title=title,
            steps=steps,
            knowledge_points=list(data.get("knowledge_points") or []),
            pitfalls=list(data.get("pitfalls") or []),
        )
        followup = self._parse_followup(data.get("followup"))
        return AIResponse(
            speech=speech,
            board_ops=ops,
            knowledge_points=list(data.get("knowledge_points") or []),
            pitfalls=list(data.get("pitfalls") or []),
            meta={
                "steps": steps,
                "title": title,
                "equation": data.get("equation") or "",
                "answer": data.get("answer") or "",
                "followup": followup,
                "mood": "speaking",   # 教师形象状态：讲解中（前端据此切表情/动画）
            },
        )

    async def draft_understanding(self, turn: StudentTurn, session: SessionState) -> AIResponse:
        """阶段一（理解先行）：只生成「题意理解」的**口语讲解**，不写白板、不给最终答案。

        供 Supervisor 两阶段答疑使用：老师先用口语把理解讲出来（带口型），
        再由 :meth:`answer`（把这份理解作为 ``context_notes`` 回流）给出结合后的解答。
        LLM 不可用或返回不合规时，给出通用理解话术兜底。
        """
        messages = self._build_messages(
            turn, session, hint_level=None, context_notes=None, extra_system=_UNDERSTAND_PROMPT
        )
        data: dict[str, Any] = {}
        try:
            raw = await self.llm.chat(messages, temperature=0.3, max_tokens=500, json_mode=True)
            parsed = self._parse_reply(raw)
            if isinstance(parsed, dict):
                data = parsed
        except Exception as err:  # noqa: BLE001 - 理解阶段失败不应中断答疑
            log.warning("draft_understanding llm failed: {}", err)

        title = str(data.get("title") or "").strip() or "题目理解"
        points = [str(p).strip() for p in (data.get("understanding") or []) if str(p).strip()]
        steps = [self._strip_step_prefix(s) for s in (data.get("steps") or []) if str(s).strip()]
        if not points:
            points = ["先读清题目给出的已知条件", "明确要求解的问题", "回忆涉及的知识点或方法"]
        speech = str(data.get("speech") or "").strip()

        return self._understanding_response(title=title, points=points, steps=steps, speech=speech)

    def _understanding_response(
        self, *, title: str, points: list[str], steps: list[str], speech: str = ""
    ) -> AIResponse:
        """把「理解」组装为**只念不写**的 ``AIResponse``：有话术、无白板指令。

        ``meta`` 里刻意不放 ``steps`` 键——``_spoken_text`` 会把 ``steps`` 追加进播报，
        而这里的话术已由 ``speech`` 完整覆盖，避免重复朗读。
        """
        spoken = speech.strip()
        if not spoken:
            spoken = "我先把题意理一下。" + "".join(f"{p}。" for p in points)
        return AIResponse(
            speech=spoken,
            board_ops=[],
            meta={
                "title": title,
                "understanding": points,
                "understand_steps": steps,   # 供「结合理解」回流上下文，不参与朗读
                "mood": "thinking",          # 教师形象：思考中（先讲理解）
            },
        )

    def _parse_understanding(self, text: str) -> AIResponse | None:
        """解析「理解」段 JSON；缺关键字段返回 ``None``（交由调用方兜底）。"""
        data = self._parse_reply(text)
        if not data:
            return None
        points = [str(p).strip() for p in (data.get("understanding") or []) if str(p).strip()]
        if not points:
            return None
        steps = [self._strip_step_prefix(s) for s in (data.get("steps") or []) if str(s).strip()]
        title = str(data.get("title") or "").strip() or "题目理解"
        speech = str(data.get("speech") or "").strip()
        return self._understanding_response(title=title, points=points, steps=steps, speech=speech)

    def _fallback_understanding(self, final: AIResponse) -> AIResponse:
        """模型未按两段输出时的兜底：用解答的标题/步骤反推一段「理解」话术。"""
        title = str(final.meta.get("title") or "").strip() or "题目理解"
        steps = [str(s).strip() for s in (final.meta.get("steps") or []) if str(s).strip()][:3]
        points = ["先读清题目给出的已知条件", "明确要求解的问题", "回忆涉及的知识点或方法"]
        return self._understanding_response(title=title, points=points, steps=steps)

    def _board_snapshot(self, title: str, steps: list[str]) -> AIResponse:
        """「边流边上板」用的**累计**板书快照（标题 + 目前已生成的步骤）。

        用累计而非增量，是因为前端收到任何一批 ``board_ops`` 都会先清板再渲染
        （见 student_tablet/index.html 的 board_ops 分支）：推全量天然得到正确画面，
        也就不需要动前端。最后 ``solution`` 阶段还会再推一次完整板书覆盖它。
        """
        ops: list[WhiteboardOp] = [self.board.explain_title(title)]
        ops.extend(self.board.explain_step(i, s) for i, s in enumerate(steps, start=1))
        return AIResponse(speech="", board_ops=ops, meta={"board_snapshot": True})

    async def stream_board_first(
        self,
        turn: StudentTurn,
        session: SessionState,
        *,
        hint_level: HintLevel | None = None,
        context_notes: list[str] | None = None,
    ) -> AsyncIterator[tuple[str, AIResponse]]:
        """方案 B：**单次** LLM 调用产出「理解 + 解答」两段，边流式边在边界产出。

        产出顺序：
        * ``("understanding", 理解AIResponse)`` —— 模型写完第一段即产出，供**立刻语音讲出**（不上板）；
        * ``("board", 累计板书快照)`` —— 解答段的 ``title``/``steps`` **一写出来就上板**（可多次）；
        * ``("solution", 解答AIResponse)`` —— 流结束后产出，含完整结构化产物（话术/步骤/答案/追问）。

        「board」是本次改造的核心：以前白板要等**整段 JSON（含 >100 字的 speech、
        知识点、易错点、追问）全部生成完**才出现；现在模型按提示词先吐 ``title``/``steps``，
        我们边流边扫描、一有完整步骤就推一次白板，首次上板时间从「全文完成」提前到
        「标题 + 第一步完成」，通常能早 5~10 秒。

        每次 board 都是**累计快照**（标题 + 目前已生成的步骤），因此前端无需改动 ——
        它本来就会「收到一批 board_ops 先清板再渲染」，推全量天然正确。

        模型未按哨兵分段时降级：整段当作解答，并反推一段理解话术兜底。
        """
        messages = self._build_messages(
            turn, session, hint_level=hint_level, context_notes=context_notes,
            extra_system=_BOARD_FIRST_PROMPT,
        )

        buf = ""
        understanding: AIResponse | None = None
        boundary = False
        streamed_title = ""
        streamed_steps: list[str] = []
        try:
            async for delta in self.llm.chat_stream(messages, temperature=0.4, max_tokens=900):
                buf += delta
                if not boundary and _SOLUTION_SENTINEL in buf:
                    boundary = True
                    draft = self._parse_understanding(
                        _segment(buf, _UNDERSTANDING_SENTINEL, _SOLUTION_SENTINEL)
                    )
                    if draft is not None:
                        understanding = draft
                        yield ("understanding", draft)
                if not boundary:
                    continue

                # ---- 白板先行：解答段的 title / steps 一写出来就立刻上板 ----
                solution_so_far = _segment(buf, _SOLUTION_SENTINEL)
                if not streamed_title:
                    title_now = _drain_title(solution_so_far)
                    if title_now:
                        streamed_title = title_now
                        yield ("board", self._board_snapshot(streamed_title, streamed_steps))
                        continue
                if not streamed_title:
                    continue                      # 还没有标题：先不上板，等下一批增量
                steps_now = [self._strip_step_prefix(s) for s in drain_steps(solution_so_far)]
                if len(steps_now) > len(streamed_steps):
                    streamed_steps = steps_now
                    yield ("board", self._board_snapshot(streamed_title, streamed_steps))
        except Exception as err:  # noqa: BLE001 - 流失败也要尽量给出解答
            log.warning("stream_board_first failed: {}", err)

        # 流结束后再补推一次：最后一批增量里往往还有新步骤（循环已退出，上面来不及推）。
        # 补完这里，板书在「生成一结束」就是完整的，不必等下面的解答解析与语音准备。
        if streamed_title:
            solution_all = _segment(buf, _SOLUTION_SENTINEL) if boundary else buf
            steps_final = [self._strip_step_prefix(s) for s in drain_steps(solution_all)]
            if len(steps_final) > len(streamed_steps):
                streamed_steps = steps_final
                yield ("board", self._board_snapshot(streamed_title, streamed_steps))

        solution_text = _segment(buf, _SOLUTION_SENTINEL) if boundary else buf
        if not solution_text.strip():
            solution_text = buf
        final = self._compose_response(solution_text)
        if understanding is None:
            understanding = self._fallback_understanding(final)
            yield ("understanding", understanding)
        yield ("solution", final)

    async def speech_stream(
        self,
        turn: StudentTurn,
        session: SessionState,
        *,
        hint_level: HintLevel | None = None,
        context_notes: list[str] | None = None,
        sentence_queue: asyncio.Queue[str | None],
    ) -> str:
        """流式产出讲解话术：LLM 增量文本逐句推 ``sentence_queue``（None 结束）。

        与 :meth:`answer`（结构化 JSON）并行调用：本方法只负责"要朗读的正文"，
        让 supervisor 拿到一句就立刻 TTS/渲染，首句延迟从「全文生成完」降到
        「首句生成完」。返回拼接后的完整话术（供覆盖 ``AIResponse.speech``）。

        流式失败时返回空串，调用方回退 JSON 里的 ``speech``，保证总有可播报内容。
        """
        speech_messages = self._build_messages(
            turn, session, hint_level=hint_level, context_notes=context_notes,
            extra_system=_SPEECH_ONLY_PROMPT,
        )
        spoken: list[str] = []
        try:
            buf = ""
            async for delta in self.llm.chat_stream(speech_messages, temperature=0.4, max_tokens=400):
                buf += delta
                sentences, buf = _drain_sentences(buf)
                for sentence in sentences:
                    clean = strip_emphasis(sentence)      # 队列直通 TTS，星号必须先摘掉
                    spoken.append(clean)
                    await sentence_queue.put(clean)
            tail = strip_emphasis(buf.strip())
            if tail:
                spoken.append(tail)
                await sentence_queue.put(tail)
        except Exception as err:  # noqa: BLE001 - 话术流失败回退 JSON speech
            log.warning("speech_stream failed: {}", err)
        await sentence_queue.put(None)
        return "".join(spoken).strip()

    async def answer_image(
        self,
        turn: StudentTurn,
        session: SessionState,
        *,
        ocr_text: str = "",
        hint_level: HintLevel | None = None,
        extra_notes: list[str] | None = None,
        toolkit: AgentToolkit | None = None,
    ) -> AIResponse:
        """图片题目讲解：识别题干 -> 圈画题目区域 -> 分步讲解。

        ``extra_notes``：额外上下文（如「白板先行」阶段已写出的题意理解），
        会置于 OCR 说明之前一并喂给 LLM。
        ``toolkit``：ReAct 闭环用（图片题尤其受益 —— 模型可用 ``ocr_parse`` 复核题干）。
        """
        notes = [f"题目OCR识别内容：{ocr_text}"] if ocr_text else ["题目来自学生上传的图片"]
        if extra_notes:
            notes = list(extra_notes) + notes
        response = await self.answer(
            turn, session, hint_level=hint_level, context_notes=notes, toolkit=toolkit
        )

        # 在图片上层圈画题目区域：用 OCR 行框算出的**真实**区域（Supervisor 在
        # handle_image 里写入 turn.meta，算法见 document_parsing/region.py）。
        # 定位不出来就不画 —— 画错比不画更糟：学生盯着错的地方，反而听不懂。
        ops: list[WhiteboardOp] = []
        region = turn.image_region
        if turn.image_id and region:
            x1, y1, x2, y2 = region
            ops.append(
                self.board.annotate_region(
                    x1, y1, x2 - x1, y2 - y1, image_id=turn.image_id, label="题目区域"
                )
            )
        response.board_ops = ops + response.board_ops
        return response

    # ---------------------------------------------------------------- 课件讲解（兼容转发）
    # 讲课件已拆为独立的 LecturerAgent（agents/lecturer/agent.py）：它有自己的任务契约
    # （LECTURE_CONTRACT）与装配路径，不再借用答疑契约，因此不会再出现「模型在 steps
    # 与 points 两套字段之间猜」的问题。
    #
    # 这里保留同名方法作为**兼容与回退**入口：
    #   * 兼容：脚本 / 旧对接仍可直接调 teacher.explain_slide；
    #   * 回退：Supervisor 走 lecturer 失败时可退到这条等价实现，保证不哑火。
    # 它复用同一个 llm 与同一个白板工厂，因此结果与 LecturerAgent 完全一致。
    def _lecturer(self) -> "LecturerAgent":
        """懒建讲课 agent（复用本 agent 的 llm / board，避免重复加载模型）。"""
        if self._lecturer_agent is None:
            from agents.lecturer.agent import LecturerAgent  # noqa: PLC0415 - 避免循环依赖

            self._lecturer_agent = LecturerAgent(llm=self.llm, board=self.board)
        return self._lecturer_agent

    async def explain_slide(
        self,
        session: SessionState,
        deck: "SlideDeck",
        index: int,
        *,
        extra_notes: list[str] | None = None,
    ) -> AIResponse:
        """讲解课件某一页（兼容转发到 :class:`~agents.lecturer.agent.LecturerAgent`）。"""
        return await self._lecturer().explain_slide(
            session, deck, index, extra_notes=extra_notes
        )

    @staticmethod
    def _element_catalog(slide: "Slide") -> str:
        """页面元素目录（兼容转发）。"""
        from agents.lecturer.agent import LecturerAgent  # noqa: PLC0415

        return LecturerAgent._element_catalog(slide)

    @staticmethod
    def _deck_context(deck: "SlideDeck", index: int, *, around: int = 3) -> str:
        """教材位置上下文（兼容转发）。"""
        from agents.lecturer.agent import LecturerAgent  # noqa: PLC0415

        return LecturerAgent._deck_context(deck, index, around=around)

    def _slide_annotations(
        self,
        deck: "SlideDeck",
        index: int,
        slide: "Slide",
        annotations: list[dict[str, Any]],
    ) -> list[WhiteboardOp]:
        """标注意图 -> 白板指令（兼容转发，复用同一白板工厂）。"""
        from agents.lecturer.agent import LecturerAgent  # noqa: PLC0415

        return LecturerAgent(board=self.board)._slide_annotations(
            deck, index, slide, annotations
        )

    # ---------------------------------------------------------------- 1v1 辅导课
    def compose_course_greeting(self, subject: str, teacher_key: str = "") -> AIResponse:
        """进入辅导页时教师主动打招呼（语言 + 挥手动作 greet）。

        同科多老师：从该老师档案的**问候池**（专属问候 + 学科内置池，见
        :meth:`config.teachers.TeacherProfile.greeting_pool`）里随机取一条 ——
        每次开课的第一句话不重样（此前固定念档案里那一句，听久了很单调）；
        档案缺失时用学科通用池兜底（见 ``course_session.greeting_text``）。
        """
        opening = ""
        try:
            profile = teacher_for_subject(subject, teacher_key)
            pool = profile.greeting_pool() if profile else ()
            if pool:
                opening = random.choice(pool)
        except Exception as err:  # noqa: BLE001 - 档案异常不应阻断打招呼
            log.warning("greeting profile lookup failed: {}", err)
        return AIResponse(
            speech=greeting_text(subject, opening),
            meta={
                "gesture": "greet",
                "mood": "greeting",
                "subject": subject,
                "teacher": teacher_key or "",
            },
        )

    def compose_phase_intro(self, phase: Any, subject: str) -> AIResponse:
        """阶段切换的过渡话术（讲解中/思考中由上层拦截，不在此判断）。"""
        return AIResponse(
            speech=phase_intro_text(phase, subject),
            meta={"mood": "speaking", "phase": getattr(phase, "value", str(phase))},
        )

    async def generate_quiz(
        self,
        subject: str,
        topic: str = "",
        *,
        level: str = "基础",
        grade: str = "",
    ) -> QuizItem:
        """出一道模拟题（选择/填空/简答）。三级降级，保证流程不中断：

        1. **真题库优先**（:mod:`core.exercise_bank`，``ai_teacher.exercise_item``）：
           按学生年级 + 当前知识点选一道教研真题（有标准答案、可溯源，判分更可靠）；
        2. **LLM 即兴生成**：题库未部署 / 连不上 / 无匹配时的常规路径；
        3. **规则兜底**：LLM 也不可用 / 返回不合规时给一道可判定的选择题（见 ``_build_quiz``）。
        """
        focus = (topic or subject).strip()
        # ---- 1) 真题库优先 ----
        if settings.exercise_bank_enable:
            try:
                row = exercise_bank.pick(subject=subject, grade=grade, topic=focus)
            except Exception as err:  # noqa: BLE001 - 题库只读层已全降级，这里再兜一层
                row = None
                log.warning("exercise bank pick failed: {}", err)
            if row:
                quiz = self._build_quiz(
                    exercise_bank.to_quiz_fields(row, topic=focus),
                    subject,
                    focus,
                    qid=str(row.get("exercise_id") or ""),
                    keep_choice=True,   # 真题选择题的选项常内嵌题干（options=null 是正常形态）
                )
                log.info(
                    "quiz from exercise bank: {}（grade={}, type={}）",
                    quiz.qid, row.get("grade"), quiz.type,
                )
                return quiz
        # ---- 2) LLM 即兴生成（题库不可用/无匹配）----
        prompt = (
            f"你是一位{subject}老师，请围绕知识点「{focus}」出 1 道{level}难度的模拟题，用来检验学生是否听懂。\n"
            "只输出 JSON，字段为：\n"
            '{"type": "choice|blank|short", "question": "题干", '
            '"options": ["选项A", "选项B", "选项C", "选项D"], '
            '"answer": "标准答案（选择题填正确选项原文；填空/简答填简短答案）", '
            '"hint": "学生答错时的一句引导提示"}\n'
            "要求：题干口语化、指向明确；选择题必须给出 3-4 个选项且只有一个正确；"
            "填空/简答题答案要简短（不超过15字）；不要输出多余解释。"
        )
        style = self._style_text()
        if style:
            prompt += "\n" + style
        data: dict[str, Any] = {}
        try:
            raw = await self.llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.8, max_tokens=400, json_mode=True,
            )
            parsed = self._parse_reply(raw)
            if isinstance(parsed, dict):
                data = parsed
        except Exception as err:  # noqa: BLE001 - 出题失败回退规则题
            log.warning("generate_quiz llm failed: {}", err)
        return self._build_quiz(data, subject, focus)

    async def evaluate_quiz(self, quiz: QuizItem, student_answer: str) -> dict[str, Any]:
        """判定模拟题作答（复用 ``evaluate_answer``：sympy + LLM + 字符串兜底）。"""
        return await self.evaluate_answer(
            quiz.question,
            quiz.answer,
            student_answer,
            options=list(quiz.options) or None,
        )

    @staticmethod
    def _build_quiz(
        data: dict[str, Any], subject: str, topic: str, *,
        qid: str = "", keep_choice: bool = False,
    ) -> QuizItem:
        """归一化出题结果（LLM 或真题库）；缺失关键字段时回退为可判定的选择题。

        ``qid`` 显式传入时沿用（真题库路径用 ``exercise_id``，便于溯源）；
        否则按时间戳生成（LLM / 兜底路径）。

        ``keep_choice``：真题库里大量选择题的选项**内嵌在题干里**（``options`` 字段为
        null，如 ``"…的是(   )\\nA．…\\nB．…"``），此时 ``options`` 列表为空是正常形态，
        不能让「选项不足 2 个 → 降级为简答」的规则把它改型（LLM 路径忘给选项才该降级）。
        """
        question = str(data.get("question") or "").strip()
        options = [str(o).strip() for o in (data.get("options") or []) if str(o).strip()]
        answer = str(data.get("answer") or "").strip()
        qtype = str(data.get("type") or "").strip().lower()
        hint = str(data.get("hint") or "").strip()

        if qtype not in ("choice", "blank", "short"):
            qtype = "choice" if len(options) >= 2 else "short"

        if not question:
            # 兜底：无 LLM（MockLLM）时给一道与知识点相关的选择题，保证流程可演示
            question = f"关于刚才讲的「{topic}」，下面哪种做法最有助于真正学会它？"
            options = ["先弄清概念和条件，再按步骤推导", "直接凭感觉写一个答案", "跳过条件只看结论"]
            answer = options[0]
            qtype = "choice"
            hint = "回想一下我们讲的第一步是什么。"

        if qtype == "choice":
            if len(options) < 2 and not keep_choice:
                qtype = "short"
                options = []
            elif len(options) >= 2 and not answer:
                answer = options[0]
        else:
            options = []

        return QuizItem(
            qid=qid or f"quiz-{int(time.time() * 1000)}",
            type=qtype,  # type: ignore[arg-type]
            question=question,
            options=options,
            answer=answer,
            hint=hint,
            topic=topic or subject,
        )

    # ---------------------------------------------------------------- 互动追问
    async def evaluate_answer(
        self,
        question: str,
        expected: str,
        student_answer: str,
        *,
        options: list[str] | None = None,
    ) -> dict[str, Any]:
        """判定学生对追问的回答是否正确，返回 ``{correct, feedback, hint}``。

        选择题（传了 ``options`` 且答案可归一为选项字母）先做确定性比对：
        学生可能点选项按钮提交**选项原文**（"嫌疑犯A"）、也可能手输**字母**（"A"），
        标准答案则来自真题库（字母形态）——两侧都归一到选项字母再比，不依赖 LLM。
        其余：等式类用 sympy 精确判定；再其余交给 LLM；LLM 异常时降级为字符串比较。
        """
        answer = (student_answer or "").strip()
        expected = (expected or "").strip()
        # 0) 选择题：字母 ↔ 选项原文 ↔「A．xxx」双向归一（真题库答案为字母时必需）
        if options:
            got = self._option_token(answer, options)
            want = self._option_token(expected, options)
            if got and want:
                ok = got == want
                return {
                    "correct": ok,
                    "feedback": "选对了，很棒！" if ok else "选项再斟酌一下。",
                    "hint": "" if ok else "把每个选项代回题目条件里试一试。",
                }
        # 1) 等式型标准答案（如 x=5）：用 sympy 精确比对
        if math_solver.available() and "=" in expected:
            verdict = math_solver.verify_answer(expected, answer)
            if verdict is not None:
                return {
                    "correct": bool(verdict),
                    "feedback": "完全正确！" if verdict else "答案还不对哦。",
                    "hint": "" if verdict else "再看看等式两边要怎么变形。",
                }
        # 2) 其余：让 LLM 判定（严格 JSON）
        try:
            prompt = (
                "你是批改老师，请判断学生的回答是否正确。只输出 JSON："
                '{"correct": true/false, "feedback": "一句鼓励式反馈", '
                '"hint": "若错误，给一句引导提示"}。\n'
                f"问题：{question}\n标准答案：{expected}\n学生回答：{answer}"
            )
            text = await self.llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=256, json_mode=True,
            )
            data = self._parse_reply(text)
            if isinstance(data, dict) and "correct" in data:
                return {
                    "correct": bool(data.get("correct")),
                    "feedback": str(data.get("feedback") or "").strip(),
                    "hint": str(data.get("hint") or "").strip(),
                }
        except Exception as err:  # noqa: BLE001 - 判定失败不应中断会话
            log.warning("evaluate_answer llm failed: {}", err)
        # 3) 降级：归一化字符串比较
        return self._fallback_judge(expected, answer)

    # ---------------------------------------------------------------- 内部（提示词装配）
    @staticmethod
    def _style_text(style: str | None = None) -> str:
        """讲课风格提示词文本（优先用教师档案里的风格，其次全局 TEACHING_STYLE）。

        实体已迁到共享层 ``agents.shared.teaching_layers.style_text``——
        答疑与讲课必须用同一套风格档位，否则同一位老师在两条链路上的说话方式会分叉。
        """
        return style_text(style)

    def _build_messages(
        self,
        turn: StudentTurn,
        session: SessionState,
        *,
        hint_level: HintLevel | None,
        context_notes: list[str] | None,
        extra_system: str | None = None,
    ) -> list[dict[str, Any]]:
        """装配**答疑**用的 messages。

        身份基座、老师人格、讲课风格、学科人格、经验层全部委托给共享层
        （``agents.shared.teaching_layers.build_messages``）——
        答疑与讲课共用同一份「这个人是谁」，保证同一位老师两条链路一致。

        本方法只决定**答疑专属**的三件事：学科护栏从严、作业模式提示层级、
        本轮任务的附加指令（理解先行 / 白板先行 / 只出话术）。
        课件的任务契约不在这里 —— 它属于 ``LecturerAgent``。
        """
        hint_text = (
            f"作业模式提示层级：{_HINT_STYLE[hint_level]}" if hint_level is not None else None
        )
        return build_messages(
            session,
            contract=ANSWER_CONTRACT,
            subject_strict=True,          # 答疑从严：学生可能问到别的学科
            hint_text=hint_text,          # 作业模式分层提示是答疑专属
            extra_system=extra_system,    # 本轮任务附加指令（理解先行/白板先行/只出话术）
            history_limit=6,
            context_notes=context_notes,
            user_text=turn.text or "请讲解这道题。",
        )

    @staticmethod
    def _parse_reply(text: str) -> dict[str, Any]:
        """解析模型返回的 JSON（与讲课链路共用同一个解析口，见 ``reply_parser``）。"""
        return parse_json_reply(text)

    @staticmethod
    def _strip_step_prefix(step: str) -> str:
        """去除步骤自带的“第一步：”/“1.”/“（1）”等前缀（可叠加），避免白板与播报重复编号。"""
        text = str(step).strip()
        # 循环剥离，直到不再变化（处理“1. 第一步：…”这类叠加前缀）
        while True:
            stripped = _STEP_PREFIX.sub("", text)
            if stripped == text:
                return text
            text = stripped

    @staticmethod
    def _parse_followup(raw: Any) -> dict[str, Any] | None:
        """解析讲完后的追问：问题/答案/提示/选项；问题为空则视为不追问。"""
        if not isinstance(raw, dict):
            return None

        def _clean(value: Any) -> str:
            return str(value).strip() if value is not None else ""

        question = _clean(raw.get("question"))
        if not question:
            return None
        options = [str(o).strip() for o in (raw.get("options") or []) if str(o).strip()]
        return {
            "question": question,
            "answer": _clean(raw.get("answer")),
            "hint": _clean(raw.get("hint")),
            "options": options,
        }

    @staticmethod
    def _option_token(text: str, options: list[str]) -> str:
        """选择题答案归一：选项原文 /「A．xxx」/「（A）」/ 字母 → 统一为选项字母。

        无法归一（如开放式回答）返回 ``""``，调用方据此退回 LLM 判分链。
        仅当选项存在且字母在选项范围内才归一——防把「D」这类普通文本误当选项号。
        """
        t = (text or "").strip()
        if not t or not options:
            return ""
        # 选项原文 → 字母（学生点按钮提交的就是原文）
        for i, opt in enumerate(options):
            if t == str(opt).strip():
                return chr(65 + i)
        # 「A．xxx」/「A. xxx」/「A、xxx」/「（A）」/「A」→ 提取首字母
        m = re.match(r"^[（(]?\s*([A-Ha-h])\s*[）)。．、,，:：]?", t)
        if m:
            idx = ord(m.group(1).upper()) - 65
            if 0 <= idx < len(options):
                return chr(65 + idx)
        return ""

    @staticmethod
    def _fallback_judge(expected: str, student_answer: str) -> dict[str, Any]:
        """LLM 不可用时的兜底判定：归一化后比较包含关系。"""
        def _norm(value: str) -> str:
            return re.sub(r"[\s。，,．.！!？?；;:：]", "", (value or "")).lower()

        exp, ans = _norm(expected), _norm(student_answer)
        correct = bool(exp) and bool(ans) and (exp == ans or exp in ans or ans in exp)
        return {
            "correct": correct,
            "feedback": "完全正确！" if correct else "再想想看～",
            "hint": "" if correct else "可以回顾一下刚才的讲解步骤。",
        }

    @staticmethod
    def _first_sentence(text: str) -> str:
        cleaned = (text or "").strip()
        parts = re.split(r"[。！？\n]", cleaned)
        return next((p.strip() for p in parts if p.strip()), "我们一起来看看这道题。")

    @staticmethod
    def _split_steps(text: str, limit: int = 6) -> list[str]:
        cleaned = re.sub(r"\{[\s\S]*\}", "", text or "").strip()
        parts = [p.strip() for p in re.split(r"[。；\n]", cleaned) if p.strip()]
        return parts[:limit] or ["第一步，读题提取条件", "第二步，找到对应知识点", "第三步，代入求解"]
