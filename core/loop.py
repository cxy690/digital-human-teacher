"""实时事件循环：消费客户端上行事件并分发到 Supervisor。

每条 WebSocket 连接对应一个 :class:`SessionLoop`。
"""
from __future__ import annotations

import asyncio
import base64
import time
from typing import Any

from agents.supervisor import Supervisor
from config import constants as C
from config.logging import get_logger
from config.settings import settings
from core.state.models import ChatMessage, StudentTurn
from core.state.session import SessionState

log = get_logger("core.loop")


class SessionLoop:
    """单会话实时循环。"""

    def __init__(
        self,
        session: SessionState,
        supervisor: Supervisor,
    ) -> None:
        self.session = session
        self.supervisor = supervisor
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._stop = asyncio.Event()
        self._last_quiz_tick = 0.0              # 上次「出题节奏」巡检时间（monotonic）

    # ---------------------------------------------------------------- 控制
    def submit(self, event: dict[str, Any]) -> None:
        """线程/协程安全地向循环投递一个事件。

        打断类事件（学生新输入）在**入队瞬间**就作废旧播报，而不是等轮到它
        被分发时才打断：主循环是串行 ``await`` 分发，旧回合的 ``handle_turn``
        （LLM 生成 + 白板，可达数十秒）不结束，新消息就一直排在队尾 ——
        表现就是「发了新问题，老师还在讲上一题」。入队即自增 ``speech_epoch``
        并下发 ``avatar_interrupt``：旧播报的逐句渲染会在下一个句边界立即中止，
        前端同步清空播放队列，老师马上闭嘴开始听新问题。

        这里是**唯一**的打断点：``_dispatch`` 不再重复打断，否则 ``speech_epoch``
        每次学生输入会跳 2，基于它做的「轮次去重」与中断判定都会被污染。
        """
        # 连讲配速回报：**就地处理、不入队**。主循环此刻往往正卡在
        # Supervisor._await_page_played 里等这个信号 —— 若排进队列就永远轮不到
        # 它被消费（只能干等到超时），连讲就会「每页卡一个超时时长」。
        if event.get("type") == "slide_ready":
            # 注意：不能写 `int(raw or -1)` —— 页序 0 是合法值，`0 or -1` 会变成 -1，
            # 回报就永远匹配不上，连讲只能干等超时（表现为「每页都卡一个超时时长」）。
            raw = event.get("index")
            try:
                idx = int(raw)
            except (TypeError, ValueError):
                idx = -1
            self.supervisor.mark_slide_ready(idx)
            return
        # 课件翻页 / 暂停 / 继续也算「学生要改课堂节奏」，必须与「新提问」同样打断。
        #
        # 为什么**必须**在这里打断（而不是等分发时再处理）：主循环是串行 await 分发，
        # 课件连讲期间它正卡在 ``Supervisor._await_page_played`` 里等前端回报「本页播完」，
        # 最长 ``COURSEWARE_PAGE_WAIT_SECONDS``（默认 5 分钟）—— 这期间队列里的
        # ``slide_stop`` / ``slide_step`` 永远轮不到执行。只有靠这里的 ``speech_epoch``
        # 自增，才能让 ``_await_page_played`` 立刻收工、连讲循环随即退出，
        # 队列里那条事件才取得出来。
        # 历史现象：按「暂停讲解」「下一页」看着毫无反应，要等本页讲完（甚至 5 分钟超时）才生效。
        if event.get("type") in (
            "text", "voice", "image", "interrupt", "courseware",
            "slide_goto", "slide_step", "slide_play", "slide_stop",
        ):
            self._barge_in()
        self._queue.put_nowait(event)

    def stop(self) -> None:
        self._stop.set()

    # ---------------------------------------------------------------- 主循环
    async def run(self) -> None:
        # 走 Agent 生命周期契约（Supervisor 是 BaseAgent 子类）
        await self.supervisor.on_session_start(self.session)
        try:
            while not self._stop.is_set():
                try:
                    event = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    await self._course_quiz_tick()
                    continue
                try:
                    await self._dispatch(event)
                except Exception as err:  # noqa: BLE001 - 单事件失败不影响会话
                    log.exception("dispatch error: {}", err)
        finally:
            log.info("session loop stopped: {}", self.session.session_id)

    async def _course_quiz_tick(self) -> None:
        """空闲 tick 巡检：辅导课「每 N 分钟一道模拟题」。

        放在会话主循环 ``wait_for`` 的超时分支里，而不是另起长驻定时任务：
        与「学生插话打断 / 连讲配速」等既有机制同源——队列一有事件就让位给新回合，
        只有在老师空闲（没在讲、没有待答的题）时才会真的出题。
        间隔由 ``settings.course_quiz_interval_minutes`` 决定（默认 30 分钟，讲解满 30 分钟后才出题）。
        """
        if not settings.course_quiz_enable or not self.session.course_active:
            return
        now = time.monotonic()
        if now - self._last_quiz_tick < 2.0:     # 0.5s 一跳，2s 检查一次足够
            return
        self._last_quiz_tick = now
        try:
            await self.supervisor.maybe_course_quiz()
        except Exception as err:  # noqa: BLE001 - 巡检失败不影响会话
            log.warning("course quiz tick failed: {}", err)

    # ---------------------------------------------------------------- 分发
    def _barge_in(self) -> None:
        """学生插话：作废上一段未播完的播报，并中止课件连讲。

        ``speech_epoch`` 自增后，正在逐句渲染数字人的旧播报会在下一句前中止；
        同时下发 ``avatar_interrupt`` 让前端立刻清空视频队列（老师马上闭嘴）。
        ``abort_auto_play`` 顺带把「连讲中」状态归零并同步前端按钮（幂等）——
        文字/语音/图片/翻页/暂停等一切打断场景都经过这里。
        """
        self.session.speech_epoch += 1
        self.session.push_event("avatar_interrupt", {})
        self.supervisor.abort_auto_play()

    async def _dispatch(self, event: dict[str, Any]) -> None:
        etype = event.get("type")
        if etype == "ping":
            self.session.push_event("pong", {"ts": event.get("ts")})
        elif etype == "vad":
            self.supervisor.on_vad(bool(event.get("has_voice")), int(event.get("ts_ms", 0)))
            # 学生开口抢话（持续 ≥ VAD_SPEECH_MS）：老师正在播报/连讲时立即让出话语权。
            # 这是「按住提问」的兜底路径——前端按住瞬间还会发 interrupt（零延迟），
            # 这里保证其它客户端 / 帧丢失场景同样能打断。
            if self.supervisor.voice_barge_in_due():
                self._barge_in()
        elif etype == "text":
            # 打断已在 submit() 入队瞬间完成（见 _barge_in），这里不再重复自增 speech_epoch
            await self.supervisor.handle_turn(StudentTurn(text=_text(event.get("text"))))
        elif etype == "interrupt":
            # 课程到点 / 前端要求立刻静音：只作废当前播报，不生成新回合。
            # 打断同样已在 submit() 完成，这里只需吞掉事件。
            pass
        elif etype == "idle":
            await self.supervisor.on_idle()
        elif etype == "course_start":
            await self.supervisor.on_course_start(
                str(event.get("subject") or "") or None,
                str(event.get("teacher") or "") or None,
            )
        elif etype == "course_phase":
            await self.supervisor.on_course_phase(event.get("phase"))
        elif etype == "quiz_request":
            await self.supervisor.on_quiz_request(str(event.get("topic") or ""))
        elif etype == "voice":
            audio = _decode(event.get("data"))
            if audio:
                log.info("voice received: {} bytes", len(audio))
                await self.supervisor.on_voice(audio)
            else:
                log.warning("voice payload decode failed (empty/invalid base64)")
        elif etype == "courseware":
            # 打开课件（可选 index 直接翻到某一页）
            await self.supervisor.handle_courseware(
                str(event.get("deck_id") or ""), int(event.get("index", 0) or 0)
            )
        elif etype == "slide_goto":
            await self.supervisor.handle_slide_goto(int(event.get("index", 0) or 0))
        elif etype == "slide_step":
            await self.supervisor.handle_slide_step(int(event.get("step", 1) or 1))
        elif etype == "slide_play":
            # 从某页开始连讲（前端「继续讲」按钮）
            await self.supervisor.handle_slide_play(int(event.get("index", -1)))
        elif etype == "slide_stop":
            # 暂停连讲（前端「暂停」按钮）
            await self.supervisor.handle_slide_stop()
        elif etype == "slide_pos":
            # 前端上报「本轮已播语音事件数」（暂停/插话瞬间发一次）：供课件断点续讲裁剪。
            # 只记录进度，不触发打断、不生成回合（因此不在 submit 的打断名单里）。
            self.supervisor.on_slide_pos(int(event.get("index", -1)), int(event.get("pos", 0)))
        elif etype == "slide_ready":
            # 前端回报：某一页的音频已播完（连讲配速，见 Supervisor._await_page_played）
            # 页序 0 是合法值，故不能用 `or -1` 兜底（0 会被当假值丢掉）。
            try:
                self.supervisor.mark_slide_ready(int(event.get("index")))
            except (TypeError, ValueError):
                pass
        elif etype == "image":
            data = _decode(event.get("data"))
            if data is None:
                return
            if len(data) > C.MAX_IMAGE_BYTES:
                log.warning("image too large: {} bytes", len(data))
                self.session.add_message(
                    ChatMessage(
                        role="ai",
                        text=f"图片有点大（约 {len(data) // (1024 * 1024)}MB），请换一张更小的题目图片哦。",
                    )
                )
                return
            await self.supervisor.handle_image(
                _text(event.get("image_id")) or "img", data, text=_text(event.get("text"))
            )
        else:
            log.debug("unknown event type: {}", etype)


def _text(value: Any) -> str:
    """把客户端字段归一成字符串 —— ``None`` 归一成**空串**。

    注意不能用 ``str(value)``：``str(None)`` 拿到的是字面量 ``"None"``，
    会被当成学生的问题原样发给大模型（前端漏传字段时就会这样）。
    """
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _decode(data: Any) -> bytes | None:
    """解码 base64（兼容 data URL 前缀）。"""
    if not data:
        return None
    if isinstance(data, str) and "," in data and data.strip().startswith("data:"):
        data = data.split(",", 1)[1]
    try:
        return base64.b64decode(data)
    except (ValueError, TypeError):
        return None
