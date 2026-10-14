"""AI 讲课智能体（LecturerAgent）：按页讲解课件，只做「讲 + 指」。

职责边界（与 TeacherAgent / Supervisor 的分工，必须守住）：

| 谁 | 负责 | 不负责 |
| --- | --- | --- |
| LecturerAgent | 课件逐页讲解话术、本页要点、页面元素圈画/高亮标注 | 答疑、出题判题、写正文板书、作业模式分层提示、追问判定 |
| TeacherAgent | 答疑全流程（讲题、图片题、出题判题、追问） | 课件讲解（保留兼容转发，正常链路不再经过它） |
| Supervisor | 调用顺序、抢占判定、异常回退、共享上下文写入 | 具体讲解内容 |

协作约定：

* 本 agent **不直接调用** TeacherAgent，也**不改会话状态**（翻页、白板 push、
  事件下发都由 Supervisor 负责）—— 它只吃 ``(session, deck, index)``，
  吐出一个 ``AIResponse``（话术 + 标注指令）。纯函数式，可预测、可丢弃、可回退。
* 讲课时**不注入答疑契约与答疑历史**：契约由 :mod:`agents.shared.teaching_layers`
  单独传入 ``LECTURE_CONTRACT``，人格/风格/学科仍与答疑共享，
  保证「同一位老师」在不同任务下的说话方式一致。
"""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from agents.base import BaseAgent
from agents.protocol import (
    KIND_RESPONSE,
    ROLE_LECTURER,
    ROLE_SUPERVISOR,
    AgentBus,
)
from agents.shared.reply_parser import parse_json_reply
from agents.shared.teaching_layers import (
    LECTURE_BOARD_OVERRIDE,
    LECTURE_CONTRACT,
    build_messages,
)
from agents.shared.whiteboard import Whiteboard
from config.logging import get_logger
from core.state.models import AIResponse, WhiteboardOp
from core.state.session import SessionState
from models.base import BaseLLMClient
from models.llm import get_llm

if TYPE_CHECKING:  # 仅类型标注（课件模型），避免运行时循环导入
    from core.courseware.models import Slide, SlideDeck

log = get_logger("agents.lecturer")

#: 课件标注框面积上限（归一化）：超过就说明「罩住大半页」，没有指向性，索性不标。
#: 行级框一般 0.05 左右；整页大文本框约 0.45，整页图 1.0。
_MAX_ANNOTATE_AREA = 0.45

#: 逐页讲解的 token 预算：要「讲透」（180~400 字 + 要点 + 标注），
#: 给少了会在话术中途被截断，学生听到一半就没了。
_SLIDE_MAX_TOKENS = 1400

#: 兜底口播模板（有要点 / 无要点两套）。**必须多样化**：LLM 不可用时（实测
#: 2026-10-12 API key 失效会连续触发）单一模板会让每页、每位老师听起来一模一样，
#: 非常"人机"。这里按「学科 + 页码」稳定选模板：同一位老师不同页换说法、
#: 不同老师也不同，同一页可复现（便于测试与排查）。
_FALLBACK_TEMPLATES: tuple[str, ...] = (
    "这一页讲的是{title}。{points}",
    "好，我们看这一页——{title}。{points}",
    "翻到「{title}」这一页了，{points}",
    "接下来这一页是{title}。{points}",
    "我们接着看{title}。{points}",
    "来，看{title}这一页。{points}",
    "这一页是{title}，{points}",
    "好，往下走，看{title}。{points}",
)
_FALLBACK_NO_POINTS: tuple[str, ...] = (
    "这一页讲的是{title}，我们一起来看一下。",
    "好，我们看这一页——{title}，重点我们慢慢讲。",
    "翻到「{title}」这一页了，一起看看它在讲什么。",
    "接下来是{title}，我们一页一页来。",
    "来看{title}这一页，先看它主要说了什么。",
    "这一页是{title}，跟着老师一起看。",
    "好，我们翻到{title}，这页挺有意思。",
    "接着看{title}，别急，我们慢慢来。",
)


def _fallback_pick(slide: Any, session: Any) -> int:
    """按「学科 + 页码」稳定挑一个兜底模板下标（同页可复现、跨页/跨老师尽量不同）。

    用 md5 前 4 字节而不是内置 ``hash()``：后者受 PYTHONHASHSEED 随机化影响，
    跨进程/跨重启会变，同一页的话术就无法复现（测试与排查都会抖）。
    """
    subject = ""
    if session is not None:
        subject = str(getattr(session, "subject_id", "") or "")
        if not subject:
            try:
                subject = str((getattr(session, "scratch", {}) or {}).get("subject_id") or "")
            except Exception:  # noqa: BLE001 - 兜底路径不允许再抛错
                subject = ""
    index = int(getattr(slide, "index", 0) or 0)
    # 学科决定「起始模板」，页码在其上顺序轮转 —— 保证同一位老师连续翻页
    # 不会撞同一句（用户最直观的"人机感"来源）；不同老师起点也基本不同。
    span = min(len(_FALLBACK_TEMPLATES), len(_FALLBACK_NO_POINTS))
    offset = int.from_bytes(hashlib.md5(subject.encode("utf-8")).digest()[:4], "big") % span
    return offset + index


class LecturerAgent(BaseAgent):
    """课件讲解智能体：讲透一页 PPT，并在页面上做指向性标注。"""

    role = ROLE_LECTURER

    def __init__(
        self,
        llm: BaseLLMClient | None = None,
        board: Whiteboard | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        self.llm = llm or get_llm()
        self.board = board or Whiteboard()

    # ---------------------------------------------------------------- 主入口
    async def explain_slide(
        self,
        session: SessionState,
        deck: "SlideDeck",
        index: int,
        *,
        extra_notes: list[str] | None = None,
        budget_sec: float = 0.0,
    ) -> AIResponse:
        """讲解课件某一页：产出朗读话术 + 该页的圈画/高亮标注。

        ``budget_sec`` > 0 时表示「这一页大约讲多少秒」的预算（连讲时由总控逐页给出）；
        0 = 不限额（前端单页预览）。

        与答疑（``TeacherAgent.answer``）最大的不同：**不重写板书**。
        页面本身已经印好了内容，老师只需「讲 + 指」，
        所以 ``board_ops`` 里只有标注指令，没有 text 指令。

        失败策略（保证「绝不哑巴翻页」）：
        1. LLM 正常 -> 用 speech/points/annotations；
        2. LLM 抛错或返回非法 JSON -> 降级为「本页要点口播」；
        3. 契约漂移（模型按答疑格式返回了 steps）-> 从 steps 里自愈出要点。
        """
        slide = deck.page(index)
        if slide is None:
            AgentBus.record(
                session, sender=ROLE_LECTURER, recipient=ROLE_SUPERVISOR,
                kind=KIND_RESPONSE, ok=False, detail="课件页不存在", page=index,
            )
            return AIResponse(speech="这一页不存在哦，我们翻回前面看看。")

        messages = self._build_messages(session, deck, index, slide, extra_notes, budget_sec=budget_sec)

        speech = ""
        points: list[str] = []
        annotations: list[dict[str, Any]] = []
        degraded = False
        try:
            # json_mode：一页讲解也是长 JSON（180~400 字话术 + 要点 + 标注），
            # 结构写坏同样会退化成「本页要点口播」——交给厂商保证结构合法
            raw = await self.llm.chat(
                messages, temperature=0.5, max_tokens=_SLIDE_MAX_TOKENS, json_mode=True
            )
            data = parse_json_reply(raw)
            speech = str(data.get("speech") or "").strip()
            points = [str(p).strip() for p in (data.get("points") or []) if str(p).strip()]
            annotations = [a for a in (data.get("annotations") or []) if isinstance(a, dict)]
            points, healed = self._heal_points(points, data)
            degraded = healed
        except Exception as err:  # noqa: BLE001 - LLM 不可用不应阻断翻页讲解
            log.warning("slide explanation llm failed: {}", err)
            degraded = True

        if not speech:
            # 降级：模型没给话术时，至少把页面要点念出来，不能哑巴翻页
            # （模板按 学科+页码 挑，避免每页/每位老师都是同一句"人机腔"）
            speech = self._fallback_speech(slide, points, session)
            degraded = True

        ops = self._slide_annotations(deck, index, slide, annotations)
        AgentBus.record(
            session,
            sender=ROLE_LECTURER,
            recipient=ROLE_SUPERVISOR,
            kind=KIND_RESPONSE,
            ok=not degraded,
            detail=("课件讲解完成（已降级兜底）" if degraded else "课件讲解完成"),
            page=index,
            deck_id=deck.deck_id,
            points=len(points),
            annotations=len(ops),
        )
        return AIResponse(
            speech=speech,
            board_ops=ops,
            meta={
                "slide": {"deck_id": deck.deck_id, "index": index, "total": deck.total},
                "points": points,
                "mood": "speaking",
                "degraded": degraded,
            },
        )

    # ---------------------------------------------------------------- 提示词装配
    def _build_messages(
        self,
        session: SessionState,
        deck: "SlideDeck",
        index: int,
        slide: "Slide",
        extra_notes: list[str] | None = None,
        budget_sec: float = 0.0,
    ) -> list[dict[str, Any]]:
        """装配**讲课**用的 messages。

        与答疑装配（``TeacherAgent._build_messages``）共用共享层（身份、老师人格、
        风格、学科人格、经验），但任务侧有三处刻意的不同：

        1. ``contract=LECTURE_CONTRACT`` —— 只讲 ``speech / points / annotations``，
           不再出现答疑的 ``steps / knowledge_points / answer / followup``；
        2. ``subject_strict=False`` —— 老师是主讲方，不做「其它学科不许作答」的拒绝；
        3. ``hint_text=None`` + ``history_limit=0`` —— 作业分层提示与答疑对话历史
           对讲课都是噪音，讲课上下文改由 :meth:`_build_notes` 提供。
        另外追加 ``LECTURE_BOARD_OVERRIDE`` 覆盖学科人格里的「白板组织」约定。
        """
        notes = self._build_notes(deck, index, slide, extra_notes, budget_sec=budget_sec)
        user_text = (
            f"请讲解课件《{deck.title}》第 {index + 1} 页"
            + (f"「{slide.title}」" if slide.title else "")
        )
        return build_messages(
            session,
            contract=LECTURE_CONTRACT,
            subject_strict=False,
            hint_text=None,
            history_limit=0,
            extra_layers=[LECTURE_BOARD_OVERRIDE],
            context_notes=notes,
            user_text=user_text,
        )

    # ---------------------------------------------------------------- 上下文
    def _build_notes(
        self,
        deck: "SlideDeck",
        index: int,
        slide: "Slide",
        extra_notes: list[str] | None,
        budget_sec: float = 0.0,
    ) -> list[str]:
        """拼装喂给模型的「课件上下文」（替代答疑用的对话历史）。

        ``budget_sec`` > 0 时按「**这一页有多少内容**」给出讲解长度 —— 连讲一整份
        PPT 时由 ``Supervisor`` 逐页计算并传入：内容多的页留足字数讲透，
        导航页只点一两句。它**不再**由「整节课剩余时间 ÷ 剩余页数」反推，所以不会
        出现「页数一多，每页只剩几十个字、赶着翻页」的压缩式讲解。
        """
        notes: list[str] = []
        # 先给「教材位置」：这一页在整份/整章里的位置、前面讲过什么、后面要讲什么。
        # 有了前后页，模型才能「承上启下」讲出连贯的课，而不是把每页当成孤立的一页念过去。
        notes.append(self._deck_context(deck, index))
        if slide.text:
            notes.append(f"【本页文字】\n{slide.text}")
        if slide.notes:
            notes.append(f"【讲稿备注】\n{slide.notes}")
        notes.append(self._element_catalog(slide))

        # 没有可提取文字的页：模型**看不到页面图片**（只收到文字），硬讲就只能编造
        # 图里的结论/数据，所以明确禁止它瞎猜。
        if not (slide.text or "").strip():
            notes.append(
                "【本页内容】这一页没有可提取的文字（通常是**纯图片 / 图表 / 公式截图**）。"
                "只根据标题、图注这类能读到的线索**简单点一句**就好；"
                "**不要凭想象编造图里的结论、数据或题目**。"
            )

        # 页面性质：按信息量告诉模型这页该「一句话带过」还是「展开讲」。
        # 这是「讲 PPT 聪明一点」的关键 —— 同一句提示词套所有页时，模型会把
        # 章节封面也讲得像重点页；明确告知性质后它才知道该长该短。
        from core.courseware.models import LIGHT_WEIGHT, RICH_WEIGHT, slide_weight  # noqa: PLC0415

        weight = slide_weight(slide)
        if weight < LIGHT_WEIGHT:
            notes.append(
                "【本页性质】**导航页**（章节封面 / 目录 / 过渡页 / 谢谢页），信息量很小。"
                "**只允许说一到两句话**，点明「这页是什么、接下来学什么」就翻页，例如："
                "「这一章是第八章，接下来学第三节——环境保护与绿色化学。」"
                "**不要展开讲解、不要总结、不要提问、不要举例、不要逐条念页面上的字**。"
            )
        elif weight >= RICH_WEIGHT:
            notes.append(
                "【本页性质】**重点页**，内容较多：可以把层次讲清楚"
                "（这页在讲什么 → 关键结论是什么），必要时补一个简短例子；"
                "但**仍然不要照读页面文字**。"
            )

        if budget_sec and budget_sec > 0:
            # 中文 TTS 约 4~5 字/秒；留 20% 余量给数字人渲染与翻页停顿。
            words = max(16, int(budget_sec * 4.2 * 0.8))
            # 该短该长**看页面性质**，不看秒数：导航页本来就是一两句话的页，
            # 而内容页的预算是按内容量给的（足够讲透），不该再被按秒数压回去。
            if weight < LIGHT_WEIGHT:
                # 导航页：只说「20~40 字」时模型实测会写到 70 字，所以把上限说死
                tips = (f"**硬性上限 {words} 字**（一到两句话），写完请自己数一遍；"
                        f"只点明这页讲什么就够，不要展开、不要举例、不要提问。")
            else:
                tips = (f"这一页请讲**充分**：话术总长控制在 {words} 字以内，"
                        f"把「这页在讲什么 → 为什么 → 结论是什么」讲清楚，"
                        f"可以分 3~4 个小点展开、必要时补一个简短例子；"
                        f"但不要照读页面文字、不要跑题、不要提问。")
            notes.append(
                f"【本页时长】这一页大约讲 {budget_sec:.0f} 秒（按这页的内容量给的预算，"
                f"讲到这里就够了，不必刻意压缩）。{tips}"
            )

        if extra_notes:
            notes.extend(extra_notes)
        return notes

    # ---------------------------------------------------------------- 契约自愈
    @staticmethod
    def _heal_points(
        points: list[str], data: dict[str, Any]
    ) -> tuple[list[str], bool]:
        """契约漂移自愈：模型按答疑格式返回 ``steps`` 时，改从 steps 提炼本页要点。

        正常情况下不该发生（契约已隔离），但只要发生，宁可要点稍稍变形，
        也不要让学生看到「要点」栏空空如也。返回 ``(要点, 是否降级)``。
        """
        if points:
            return points, False
        steps = [str(s).strip() for s in (data.get("steps") or []) if str(s).strip()]
        if not steps:
            return [], False
        log.info("讲课契约漂移：模型返回了 steps，已从中提炼本页要点")
        return steps[:4], True

    @staticmethod
    def _fallback_speech(slide: "Slide", points: list[str], session: Any = None) -> str:
        """话术缺失时的兜底口播：把本页主题与要点念出来（模板按学科/页码多样化）。"""
        title = (slide.title or "这一页").strip()
        body = "".join(f"{p}。" for p in points[:2])
        pick = _fallback_pick(slide, session)
        if body:
            template = _FALLBACK_TEMPLATES[pick % len(_FALLBACK_TEMPLATES)]
            return template.format(title=title, points=body)
        template = _FALLBACK_NO_POINTS[pick % len(_FALLBACK_NO_POINTS)]
        return template.format(title=title)

    # ---------------------------------------------------------------- 元素与标注
    @staticmethod
    def _element_catalog(slide: "Slide") -> str:
        """把页面元素编成目录喂给 LLM：模型只能圈这里出现过的东西。

        这是「标注不跑偏」的关键——不给模型凭空编坐标的机会。
        """
        if not slide.elements:
            return "【页面元素】本页没有可定位的元素，不要输出 annotations。"
        lines = ["【页面元素】（圈画时只能引用这里的 id 或文字，原样给出）"]
        for el in slide.elements:
            label = el.text.strip().replace("\n", " / ") or f"（{el.kind}）"
            lines.append(f"- id={el.id} | {label}")
        return "\n".join(lines)

    @staticmethod
    def _deck_context(deck: "SlideDeck", index: int, *, around: int = 3) -> str:
        """给模型的「教材位置」上下文：本页在整份课件中的位置 + 前后几页的标题。

        为什么需要：只喂当前页文字时，模型只能复述这一页（学生反馈「像在念 PPT」）。
        告诉它「这是第11章三角形的第 3 页，前面讲了三角形的高、后面讲内角和」，
        它才讲得出承上启下的过渡、也才敢点出「这页容易错在哪」。

        页数多的大课件只取邻近几页标题，避免把上下文撑爆（其余靠 deck.title 的交待）。
        """
        total = max(1, int(getattr(deck, "total", 0) or 0))
        lines = [f"【教材位置】《{deck.title or '课件'}》共 {total} 页，现在讲第 {index + 1} 页"]
        if index > 0:
            lines[0] += f"（进度约 {int((index + 1) * 100 / total)}%）"

        prev_titles: list[str] = []
        for i in range(max(0, index - around), index):
            title = str((deck.page(i).title if deck.page(i) else "") or "").strip()
            if title:
                prev_titles.append(f"第{i + 1}页「{title}」")
        if prev_titles:
            lines.append("【前面讲过】" + "、".join(prev_titles))

        next_titles: list[str] = []
        for i in range(index + 1, min(total, index + 1 + around)):
            title = str((deck.page(i).title if deck.page(i) else "") or "").strip()
            if title:
                next_titles.append(f"第{i + 1}页「{title}」")
        if next_titles:
            lines.append("【后面要讲】" + "、".join(next_titles) + "（不要提前讲）")
        return "\n".join(lines)

    def _slide_annotations(
        self,
        deck: "SlideDeck",
        index: int,
        slide: "Slide",
        annotations: list[dict[str, Any]],
    ) -> list[WhiteboardOp]:
        """把 LLM 给的标注意图映射到真实元素框上（找不到就丢弃，绝不猜坐标）。"""
        ops: list[WhiteboardOp] = []
        seen: set[str] = set()
        for ann in annotations[:3]:                      # 最多 3 处，避免满屏红框
            target = str(ann.get("target") or "").strip()
            if not target:
                continue
            el = slide.element_by_id(target) or slide.match_element(target)
            if el is None or el.id in seen:
                continue
            # 框太大 = 罩住大半页，起不到「指向某处」的作用，学生只会觉得画糊了。
            # 宁可不标：切分后的行级框面积通常只有 0.05 左右。
            if el.width * el.height > _MAX_ANNOTATE_AREA:
                log.info("跳过过大的标注框：{} ({}x{})", el.id, round(el.width, 2), round(el.height, 2))
                continue
            seen.add(el.id)
            ops.append(self.board.annotate_element(
                el,
                deck_id=deck.deck_id,
                page_index=index,
                kind="highlight" if str(ann.get("kind") or "") == "highlight" else "annotate",
                label=str(ann.get("label") or "").strip() or None,
            ))
        return ops
