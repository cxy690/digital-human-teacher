"""Agent 协作协议：角色、消息、共享上下文与事件触发的统一约定。

一个会话里的 agent 分工是**按职责**切的，而不是按学科：

* ``router``（``core.intent.IntentAnalyzer``）—— 会话入口闸门的**决策层**（不是 Agent，
  不参与消息总线）：判意图、决定转交谁；
* ``teacher``（TeacherAgent）—— 答疑：讲题、图片题、白板书写、追问判定、出题判题；
* ``lecturer``（LecturerAgent）—— 讲课：课件逐页讲解、页面标注，**不写正文板书**；
* ``supervisor``（``agents.supervisor.Supervisor``）—— 总调度：唯一持有会话状态、决定调用顺序
  与抢占的一方。四个角色都在 ``agents`` 包内（``ROLE_*`` 与实现类一一对应，见
  ``agents/__init__.py``）；总调度是被 ``core/loop.py`` / ``core/harness.py`` 装配与分发的入口，
  反向引用只在 ``TYPE_CHECKING`` 下，无运行期循环依赖。

协作依靠三种机制，按可靠性从高到低：

1. **共享上下文** —— ``SessionState`` 是本会话唯一的共享黑板（白板指令、课件页序、
   history、scratch）。所有 agent 读写同一个实例，不互相拷贝，它是唯一事实来源。
2. **消息传递** —— agent 之间**不直接互相调用**，而是把「谁请求谁、结果如何、失败后
   回退给了谁」记成 :class:`AgentMessage` 投递到 :class:`AgentBus`。
3. **事件触发** —— ``AgentBus`` 把消息同步扇出成客户端事件 ``agent_message``（供 UI 与
   排障），同时写入 ``session.scratch`` 的环形缓冲，支持事后回溯。

为什么要有独立的协议层：拆分出 LecturerAgent 之后，课件链路是
「Supervisor → LecturerAgent → 白板」，答疑链路是「Supervisor → TeacherAgent → 白板」。
两条链路谁被谁触发、异常时回退到了谁，必须有**统一记录点**，否则出问题只能翻日志猜。

**总控（Supervisor）的调度策略** —— 它同样继承 :class:`~agents.base.BaseAgent`
（``role = supervisor``），位置在 ``agents/supervisor/``（唯一有状态的 agent，
被 ``core`` 侧装配与分发）。策略全貌见 ``docs/supervisor_orchestration.md``，要点：

1. **状态所有权**：``SessionState`` 由它独占；其它 agent 只通过参数拿同一实例，不持有、不缓存；
2. **回合决策顺序**（``Supervisor._handle_turn_builtin``）：待答追问优先 → 意图分流 →
   非控制诉求 ≥2 条则逐条作答 → 混装控制指令当场执行 → 控制 / 闲聊 / 答疑；两套编排
   （内建 TaskGraph 与 LangGraph 主图）读同一份 ``IntentDecision``，行为由单测强绑定一致；
3. **抢占**：学生新输入**入队瞬间**即自增 ``speech_epoch`` 并下发 ``avatar_interrupt``
   （见 ``core/loop.py`` 的 ``_barge_in``）；讲课中学生提问记一次 ``handoff`` 让位给 TeacherAgent；
4. **回退**：讲课链路失败 → ``TeacherAgent.explain_slide``（等价实现）→ 兜底话术；
   意图小模型超时/输出不合法 → 保留规则结果；RAG 失败静默忽略；
5. **不记逐回合调度**：本总线只记 ``request`` / ``handoff`` / ``fallback`` / ``error``
   这些**转交与异常路径**（环形缓冲 80 条，逐回合记录会被刷满）；常规判定痕迹走
   ``turn.meta["intent_decision"]``。
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

# ---------------------------------------------------------------- 角色
ROLE_SUPERVISOR = "supervisor"
ROLE_ROUTER = "router"
ROLE_TEACHER = "teacher"
ROLE_LECTURER = "lecturer"
ROLE_STUDENT = "student"

# ---------------------------------------------------------------- 消息类型
KIND_REQUEST = "request"      # 请求某 agent 干活
KIND_RESPONSE = "response"    # 该 agent 正常返回
KIND_HANDOFF = "handoff"      # 转交（如：讲课中学生提问 -> 转答疑）
KIND_FALLBACK = "fallback"    # 主链路失败，改走备用实现
KIND_ERROR = "error"          # 失败且无可回退


@dataclass
class AgentMessage:
    """一次 agent 间交互的可追溯记录（不是业务数据，只用于协作与排障）。"""

    sender: str
    recipient: str
    kind: str
    #: 人类可读的一句话说明（如「第 3 页课件讲解」「LLM 失败，回退答疑链路」）
    detail: str = ""
    ok: bool = True
    #: 附加信息（页序、耗时、失败原因等），保持小体积
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentBus:
    """会话级消息总线：记录协作过程，并把消息扇出为客户端事件。

    刻意做成无状态工具类（全部 classmethod）—— 消息落点是 ``SessionState``，
    总线本身不持有会话，避免多会话之间串台。
    """

    #: 追踪记录落在共享上下文的哪个 key
    TRACE_KEY = "agent_trace"
    #: 环形缓冲上限：只留最近若干条，防止长会话把 scratch 撑大
    TRACE_LIMIT = 80

    @classmethod
    def publish(cls, session: Any, message: AgentMessage) -> AgentMessage:
        """投递一条消息：写环形缓冲 + 推客户端事件。任何异常都不外抛。"""
        try:
            trace = session.scratch.setdefault(cls.TRACE_KEY, [])
            trace.append(message.to_dict())
            if len(trace) > cls.TRACE_LIMIT:
                del trace[: len(trace) - cls.TRACE_LIMIT]
        except Exception:  # noqa: BLE001 - 追踪失败绝不能影响上课
            return message
        try:
            session.push_event("agent_message", message.to_dict())
        except Exception:  # noqa: BLE001 - 客户端未接 emit 时静默
            pass
        return message

    @classmethod
    def record(
        cls,
        session: Any,
        *,
        sender: str,
        recipient: str,
        kind: str,
        detail: str = "",
        ok: bool = True,
        **payload: Any,
    ) -> AgentMessage:
        """便捷投递：``AgentBus.record(session, sender=..., recipient=..., kind=...)``。"""
        return cls.publish(session, AgentMessage(
            sender=sender,
            recipient=recipient,
            kind=kind,
            detail=detail,
            ok=ok,
            payload=payload,
        ))

    @classmethod
    def trace(cls, session: Any, limit: int = 20) -> list[dict[str, Any]]:
        """最近 ``limit`` 条协作记录（排障 / 前端调试面板用）。"""
        try:
            return list(session.scratch.get(cls.TRACE_KEY, []))[-max(0, limit):]
        except Exception:  # noqa: BLE001
            return []
