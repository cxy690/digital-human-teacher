"""学生端 Agent（``StudentAgent``）负责的流程节点。

学生端只出「状态与回合」，不出内容，因此它的节点只有两条链路：

| 图 | 节点（按连接顺序） | 说明 |
|---|---|---|
| ``student.voice`` | ``guard_mic`` → ``transcribe`` → ``guard_transcript`` → ``build_turn`` → ``dispatch`` | 一次语音提问：麦控校验 → ASR → 空结果反馈 → 构造回合 → 交回总控 |
| ``student.control`` | ``camera_off`` / ``camera_on`` / ``mic_off`` / ``mic_on`` / ``clear_board`` | 设备与白板控制指令（互斥命中，命中即收工） |

两条链路都由图驱动，调用方是总控：

* ``Supervisor.on_voice`` 跑 ``student.voice``（``dispatch`` 节点回到 ``Supervisor.handle_turn``）；
* ``Supervisor._handle_control`` 跑 ``student.control``（同步图，见 :meth:`core.graph.TaskGraph.run_sync`）。

节点声明（功能 / 输入输出 / 前置依赖 / 所需配置）见 :data:`VOICE_SPECS` 与
:data:`CONTROL_SPECS`；它们同时是构建图的输入，避免「说明与实现各写一份」。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from config.logging import get_logger
from core.graph import NodeSpec, TaskGraph
from core.state.models import ChatMessage

if TYPE_CHECKING:  # 仅类型标注：节点 handler 都挂在总控实例上，运行期不需要导入
    from agents.supervisor import Supervisor

log = get_logger("agents.student.nodes")

#: 语音回合图名
VOICE_GRAPH = "student.voice"
#: 设备控制图名
CONTROL_GRAPH = "student.control"


# ------------------------------------------------------------------ 语音回合
VOICE_SPECS: tuple[NodeSpec, ...] = (
    NodeSpec(
        name="guard_mic",
        summary="麦克风开关校验：关麦时提示学生开麦并中止本回合（不进 ASR）",
        impl="StudentAgent.microphone_on + SessionState.add_message",
        inputs=("audio",),
        deps=(),
        requires=("StudentAgent.set_microphone",),
    ),
    NodeSpec(
        name="transcribe",
        summary="ASR 转写学生语音；异常与空结果都降级为空转写（绝不因识别失败中断会话）",
        impl="Supervisor.asr.transcribe",
        inputs=("audio",),
        outputs=("transcript",),
        deps=("guard_mic",),
        requires=("models.asr.get_asr",),
    ),
    NodeSpec(
        name="guard_transcript",
        summary="空转写反馈：下发「没听清」提示与 ai_speech 事件，然后中止本回合",
        impl="SessionState.add_message + SessionState.push_event",
        inputs=("audio", "transcript"),
        outputs=("transcript",),
        deps=("transcribe",),
    ),
    NodeSpec(
        name="build_turn",
        summary="把转写文本构造成学生回合 StudentTurn（meta.from_voice=True）",
        impl="StudentAgent.build_turn_from_voice",
        inputs=("transcript",),
        outputs=("turn",),
        deps=("guard_transcript",),
    ),
    NodeSpec(
        name="dispatch",
        summary="把回合交回总控 handle_turn：意图分流 / 答疑 / 控制",
        impl="Supervisor.handle_turn",
        inputs=("turn",),
        outputs=("response",),
        deps=("build_turn",),
        requires=("Supervisor.handle_turn",),
    ),
)


def build_voice_graph(supervisor: "Supervisor") -> TaskGraph:
    """构建「一次语音提问」的节点图（无 IO 建图，调用方可按需缓存）。"""

    def guard_mic(ctx: dict[str, Any]) -> dict[str, Any]:
        if supervisor.student.microphone_on:
            return {}
        supervisor.session.add_message(
            ChatMessage(role="ai", text="麦克风好像关着，先点一下「打开麦克风」再提问哦。")
        )
        return {"__stop__": True}

    async def transcribe(ctx: dict[str, Any]) -> dict[str, Any]:
        audio: bytes = ctx["audio"]
        try:
            transcript = await supervisor.asr.transcribe(audio)
        except Exception as err:  # noqa: BLE001 - ASR 异常不能静默，否则前端一直停在「识别中…」
            log.exception("asr transcribe failed: {}", err)
            transcript = ""
        text = (transcript or "").strip()
        log.info("asr transcript: {!r} (audio {} bytes)", text, len(audio))
        return {"transcript": text}

    def guard_transcript(ctx: dict[str, Any]) -> dict[str, Any]:
        text = str(ctx.get("transcript") or "").strip()
        if text:
            return {"transcript": text}
        # 识别为空（静音 / 太短 / 网络抖动）也要给可见反馈，避免前端「识别中…」卡住
        msg = "我这边没有听清，可以再靠近麦克风说一遍吗？也可以直接打字告诉我～"
        supervisor.session.add_message(ChatMessage(role="ai", text=msg))
        supervisor.session.push_event("ai_speech", {"text": msg, "mood": "thinking"})
        return {"transcript": "", "__stop__": True}

    def build_turn(ctx: dict[str, Any]) -> dict[str, Any]:
        return {"turn": supervisor.student.build_turn_from_voice(str(ctx["transcript"]))}

    async def dispatch(ctx: dict[str, Any]) -> dict[str, Any]:
        return {"response": await supervisor.handle_turn(ctx["turn"])}

    return TaskGraph.from_specs(
        VOICE_GRAPH,
        VOICE_SPECS,
        {
            "guard_mic": guard_mic,
            "transcribe": transcribe,
            "guard_transcript": guard_transcript,
            "build_turn": build_turn,
            "dispatch": dispatch,
        },
        session=getattr(supervisor, "session", None),   # 活动日志按会话归组（见 core/activity_log.py）
    )


# ------------------------------------------------------------------ 设备控制
CONTROL_SPECS: tuple[NodeSpec, ...] = (
    NodeSpec(
        name="camera_off",
        summary='命中「关闭摄像头」：关闭摄像头并同步设备状态',
        impl="StudentAgent.set_camera(False)",
        inputs=("text",),
        deps=(),
        requires=("StudentAgent.set_camera",),
    ),
    NodeSpec(
        name="camera_on",
        summary='命中「打开摄像头」：打开摄像头并同步设备状态',
        impl="StudentAgent.set_camera(True)",
        inputs=("text",),
        deps=(),
        requires=("StudentAgent.set_camera",),
    ),
    NodeSpec(
        name="mic_off",
        summary='命中「关闭麦克风」：关麦（停止上行语音）',
        impl="StudentAgent.set_microphone(False)",
        inputs=("text",),
        deps=(),
        requires=("StudentAgent.set_microphone",),
    ),
    NodeSpec(
        name="mic_on",
        summary='命中「打开麦克风」：开麦',
        impl="StudentAgent.set_microphone(True)",
        inputs=("text",),
        deps=(),
        requires=("StudentAgent.set_microphone",),
    ),
    NodeSpec(
        name="clear_board",
        summary='命中「清空白板」：下发白板清屏指令',
        impl="Whiteboard.clear + SessionState.push_board",
        inputs=("text",),
        deps=(),
        requires=("shared.whiteboard.Whiteboard",),
    ),
)

#: 控制指令 → 节点名（与原有 if/elif 的判定顺序一致：先匹配先执行）
_CONTROL_MATCH: dict[str, tuple[str, ...]] = {
    "camera_off": ("关闭摄像头",),
    "camera_on": ("打开摄像头",),
    "mic_off": ("关闭麦克风",),
    "mic_on": ("打开麦克风",),
    "clear_board": ("清空白板",),
}


def build_control_graph(supervisor: "Supervisor") -> TaskGraph:
    """构建设备控制图（同步图，见 :meth:`core.graph.TaskGraph.run_sync`）。

    语义与原来的 ``if/elif`` 链一致：按顺序找**第一个**命中的指令执行，然后收工；
    一条都不命中则什么都不做（各节点返回 ``__skip__``）。
    """

    def _guard(ctx: dict[str, Any], name: str) -> str:
        text = str(ctx.get("text") or "")
        for keyword in _CONTROL_MATCH[name]:
            if keyword in text:
                return keyword
        return ""

    def camera_off(ctx: dict[str, Any]) -> dict[str, Any]:
        if not _guard(ctx, "camera_off"):
            return {"__skip__": True}
        supervisor.student.set_camera(False, supervisor.session)
        return {"__stop__": True}

    def camera_on(ctx: dict[str, Any]) -> dict[str, Any]:
        if not _guard(ctx, "camera_on"):
            return {"__skip__": True}
        supervisor.student.set_camera(True, supervisor.session)
        return {"__stop__": True}

    def mic_off(ctx: dict[str, Any]) -> dict[str, Any]:
        if not _guard(ctx, "mic_off"):
            return {"__skip__": True}
        supervisor.student.set_microphone(False, supervisor.session)
        return {"__stop__": True}

    def mic_on(ctx: dict[str, Any]) -> dict[str, Any]:
        if not _guard(ctx, "mic_on"):
            return {"__skip__": True}
        supervisor.student.set_microphone(True, supervisor.session)
        return {"__stop__": True}

    def clear_board(ctx: dict[str, Any]) -> dict[str, Any]:
        if not _guard(ctx, "clear_board"):
            return {"__skip__": True}
        supervisor.session.push_board([supervisor.teacher.board.clear()])
        return {"__stop__": True}

    return TaskGraph.from_specs(
        CONTROL_GRAPH,
        CONTROL_SPECS,
        {
            "camera_off": camera_off,
            "camera_on": camera_on,
            "mic_off": mic_off,
            "mic_on": mic_on,
            "clear_board": clear_board,
        },
        session=getattr(supervisor, "session", None),
    )


#: 本 agent 的图名 → 节点声明（供 ``agents.nodes`` 注册表汇总）
GRAPH_SPECS: dict[str, tuple[NodeSpec, ...]] = {
    VOICE_GRAPH: VOICE_SPECS,
    CONTROL_GRAPH: CONTROL_SPECS,
}
