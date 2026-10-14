"""Supervisor 端到端单测（使用 Mock LLM/OCR，无需外部依赖）。"""
from __future__ import annotations

import asyncio
import time

from agents.supervisor import Supervisor
from agents.base import BaseAgent
from config.constants import BoardOp
from config.settings import settings
from core.courseware.models import Slide, SlideDeck
from core.loop import SessionLoop
from core.state.models import StudentTurn
from core.state.session import SessionState
from document_parsing.ocr import MockOCRClient
from document_parsing.pipeline import QuestionParser
from models.llm import MockLLM


async def test_handle_turn_pushes_board_ops() -> None:
    state = SessionState(session_id="t1", student_id="s1")
    supervisor = Supervisor(state)
    await supervisor.on_start()

    response = await supervisor.handle_turn(StudentTurn(text="老师，这道方程题怎么解？"))
    assert response is not None
    assert response.speech
    assert len(state.board_ops) >= 1
    # 答疑结束应恢复非教学态
    assert state.ai_speaking is False


async def test_handle_image_annotates_region() -> None:
    state = SessionState(session_id="t2", student_id="s2")
    # 显式注入 Mock OCR：环境装有 PaddleOCR 时，假图片字节会触发真实解析并抛错
    parser = QuestionParser(ocr=MockOCRClient(), llm=MockLLM())
    supervisor = Supervisor(state, parser=parser)
    await supervisor.on_start()

    response = await supervisor.handle_image("img-1", b"fake-image-bytes")
    assert response is not None
    assert any(op.op is BoardOp.ANNOTATE for op in state.board_ops)
    assert "img-1" in state.board_images


async def test_control_message_toggles_microphone() -> None:
    state = SessionState(session_id="t4", student_id="s4")
    supervisor = Supervisor(state)
    await supervisor.on_start()

    await supervisor.handle_turn(StudentTurn(text="关闭麦克风"))
    assert supervisor.student.microphone_on is False


async def test_idle_greeting_only_when_student_silent() -> None:
    """空闲主动问候：静默时产出一次问候；讲话中不插话；达到上限后不再反复打扰。

    注：本用例原为 education_fusion 的 ``supervisor.greet()``，本项目的对应能力是
    ``on_idle()``（异步、带 idle_nudge_max 上限），已按本项目 API 改写。
    """
    state = SessionState(session_id="t5", student_id="s5")
    state.scratch.update(grade="中小学", subject="全科")
    supervisor = Supervisor(state)
    await supervisor.on_start()

    first = await supervisor.on_idle()
    assert first is not None and first.speech
    assert any(m.role == "ai" and m.text for m in state.history)

    # AI 正在讲话时不插话
    state.ai_speaking = True
    assert await supervisor.on_idle() is None
    state.ai_speaking = False

    # 达到 idle_nudge_max 后停止，避免反复打扰学生
    for _ in range(settings.idle_nudge_max + 2):
        await supervisor.on_idle()
    assert await supervisor.on_idle() is None


async def test_idle_greeting_is_disabled_for_active_course() -> None:
    """课堂进行期间不能用“你还在吗”打断讲解或重复刚完成的小测。"""
    state = SessionState(session_id="t5-course", student_id="s5")
    state.course_active = True
    state.pending_question = {"question": "3 加 5 等于几？", "expected": "8"}
    supervisor = Supervisor(state)

    assert await supervisor.on_idle() is None


async def test_idle_greeting_only_when_student_silent() -> None:
    """总控已加装成 Agent：继承 BaseAgent、角色为 supervisor。"""
    state = SessionState(session_id="t6", student_id="s6")
    supervisor = Supervisor(state)

    assert isinstance(supervisor, BaseAgent)
    assert supervisor.role == "supervisor"
    assert supervisor.name == "supervisor"


async def test_session_lifecycle_hooks() -> None:
    """生命周期钩子可用：on_session_start 推运行态；on_session_end 停后台任务。"""
    state = SessionState(session_id="t7", student_id="s7")
    pushed: list[dict] = []
    state.emit = pushed.append
    supervisor = Supervisor(state)

    await supervisor.on_session_start(state)
    assert any(p.get("payload", {}).get("ai_running") is True for p in pushed), \
        "on_session_start 应下发 ai_running=True"

    state.auto_playing = True
    state.ai_speaking = True
    await supervisor.on_session_end(state)
    assert state.auto_playing is False
    assert state.ai_speaking is False


def test_agent_roles_are_wired() -> None:
    """四个角色的「对接关系」落到实处：每个 ``ROLE_*`` 都有可导入的实现类。

    总控已迁到 ``agents/supervisor/``（总调度 agent），本用例同时钉住：
    1. ``from agents import Supervisor`` 可用（包的惰性导出没被打断）；
    2. ``from core import Supervisor`` 这条历史写法仍可用，且是**同一个类**；
    3. 总控默认会装配三个执行者（student / teacher / lecturer）。
    """
    from agents import LecturerAgent, StudentAgent, Supervisor as PackagedSupervisor, TeacherAgent
    from agents.protocol import (
        ROLE_LECTURER,
        ROLE_STUDENT,
        ROLE_SUPERVISOR,
        ROLE_TEACHER,
        AgentBus,
    )
    from core import Supervisor as LegacySupervisor

    assert PackagedSupervisor is Supervisor
    assert LegacySupervisor is Supervisor

    impls = {
        ROLE_SUPERVISOR: Supervisor,
        ROLE_STUDENT: StudentAgent,
        ROLE_TEACHER: TeacherAgent,
        ROLE_LECTURER: LecturerAgent,
    }
    assert {cls.role for cls in impls.values()} == set(impls)

    state = SessionState(session_id="t8", student_id="s8")
    supervisor = Supervisor(state)
    assert isinstance(supervisor.student, StudentAgent)
    assert isinstance(supervisor.teacher, TeacherAgent)
    assert isinstance(supervisor.lecturer, LecturerAgent)

    # 协作总线：总控与讲课/答疑之间的转交挂在本会话的共享上下文上（可回溯）
    AgentBus.record(
        state,
        sender=ROLE_SUPERVISOR,
        recipient=ROLE_LECTURER,
        kind="request",
        detail="装配自检",
    )
    assert AgentBus.trace(state)[-1]["recipient"] == ROLE_LECTURER


async def test_barge_in_happens_once_per_turn() -> None:
    """一次学生输入只打断一次：``speech_epoch`` 是中断判定与轮次去重的唯一标识。

    回归点：``SessionLoop.submit`` 入队时打断后，``_dispatch`` 曾对同一批事件
    （text / voice / image / interrupt）再打断一次，导致每次输入 ``speech_epoch`` 跳 2。
    """
    state = SessionState(session_id="t9", student_id="s9")
    supervisor = Supervisor(state)
    loop = SessionLoop(state, supervisor)
    event = {"type": "text", "text": "老师，这道题怎么解？"}

    loop.submit(event)
    assert state.speech_epoch == 1, "入队瞬间应打断一次"

    await loop._dispatch(event)  # noqa: SLF001 - 直接验证分发阶段不再重复打断
    assert state.speech_epoch == 1, "分发阶段不应重复打断"


async def test_slide_control_events_also_barge_in() -> None:
    """课件控制事件（翻页 / 暂停 / 继续）必须与「新提问」一样：入队即打断。

    回归点：``slide_stop`` / ``slide_play`` 曾不在打断名单里。主循环是串行 await 分发，
    课件连讲期间它正卡在 ``_await_page_played`` 等前端回报「本页播完」（上限默认 5 分钟），
    队列里的暂停事件根本轮不到执行 —— 现象就是「点暂停讲解毫无反应」。
    """
    for event in (
        {"type": "slide_step", "step": 1},
        {"type": "slide_goto", "index": 2},
        {"type": "slide_play", "index": -1},
        {"type": "slide_stop"},
        {"type": "courseware", "deck_id": "demo"},
    ):
        state = SessionState(session_id=f"barge-{event['type']}", student_id="s")
        loop = SessionLoop(state, Supervisor(state))

        loop.submit(dict(event))

        assert state.speech_epoch == 1, f"{event['type']} 未触发打断（连讲会一直占着分发循环）"


async def test_await_page_played_detects_barge_in_before_entry(monkeypatch) -> None:
    """等待入口必须用「连讲开始时」的 epoch，而不是入口处重取的当前值。

    回归点：一页话术要经过 LLM + TTS + 数字人渲染（实测首屏 20s+），学生完全可以
    在这段时间里点翻页 —— 那时 ``speech_epoch`` 已经自增，入口重取的「当前值」等于新值、
    比对不出差异，等待只能一路耗到 ``COURSEWARE_PAGE_WAIT_SECONDS``（默认 300s）。
    """
    monkeypatch.setattr(settings, "courseware_page_wait_seconds", 2.0)
    state = SessionState(session_id="t10", student_id="s10")
    supervisor = Supervisor(state)
    state.auto_playing = True

    epoch_at_lecture_start = state.speech_epoch     # 连讲开始时捕获
    state.speech_epoch += 1                          # 讲解期间学生点了下一页（= _barge_in）

    started = time.monotonic()
    await supervisor._await_page_played(0, epoch=epoch_at_lecture_start)  # noqa: SLF001
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"应立即返回，实际等了 {elapsed:.1f}s（说明 epoch 没传对，退化成了等超时）"


async def test_pause_during_lecture_stops_loop_immediately(monkeypatch) -> None:
    """连讲期间按「暂停讲解」：连讲循环必须立刻收工，而不是讲完这一页（或超时）才停。"""
    monkeypatch.setattr(settings, "courseware_page_wait_seconds", 30.0)
    state = SessionState(session_id="t11", student_id="s11")
    supervisor = Supervisor(state)

    async def fake_present(deck, index, *, speak=True):
        """模拟「本页讲解」：只占时间，不消费 epoch（学生正是在这段时间里点的暂停）。"""
        await asyncio.sleep(0.05)
        state.page_index = index
        return None

    monkeypatch.setattr(supervisor, "_present_slide", fake_present)
    loop = SessionLoop(state, supervisor)
    deck = SlideDeck(
        deck_id="demo",
        title="演示课件",
        slides=[Slide(index=i, title=f"第 {i + 1} 页") for i in range(5)],
    )

    task = asyncio.create_task(supervisor._play_slides(deck, 0))  # noqa: SLF001
    await asyncio.sleep(0.15)
    assert state.auto_playing is True, "连讲应已开始"

    loop.submit({"type": "slide_stop"})            # 学生按「暂停讲解」
    await asyncio.wait_for(task, timeout=8)         # 修好前会一直等到 30s 超时

    assert state.auto_playing is False
    assert state.page_index == 0, "暂停不应继续往后翻页"


async def test_play_slides_resets_stale_ready_marker(monkeypatch) -> None:
    """新一轮连讲不能拿上一页的完成回执当作当前页已经播完。"""
    state = SessionState(session_id="fresh-ready", student_id="s")
    supervisor = Supervisor(state)
    state.scratch["slide_ready_index"] = 8

    async def fake_present(deck, index, *, speak=True):
        state.page_index = index
        return None

    async def fake_wait(index, *, epoch=None):
        assert supervisor._slide_ready_index() == index - 1  # noqa: SLF001
        state.auto_playing = False

    monkeypatch.setattr(supervisor, "_present_slide", fake_present)
    monkeypatch.setattr(supervisor, "_await_page_played", fake_wait)
    deck = SlideDeck(deck_id="demo", title="演示", slides=[Slide(index=0, title="第一页")])

    await supervisor._play_slides(deck, 0)  # noqa: SLF001


async def test_handle_slide_stop_notifies_frontend() -> None:
    """暂停要下发停止状态，让前端按钮切回继续讲。"""
    state = SessionState(session_id="t12", student_id="s12")
    events: list[dict] = []
    state.emit = events.append
    supervisor = Supervisor(state)
    state.auto_playing = True

    await supervisor.handle_slide_stop()

    assert state.auto_playing is False
    assert any(
        e["type"] == "slide_play_state" and e["payload"]["playing"] is False for e in events
    ), "必须下发停止状态，否则按钮看起来没反应"
    assert any(e["type"] == "avatar_interrupt" for e in events), "还要立刻让老师闭嘴"


async def test_voice_barge_in_due_when_ai_speaking() -> None:
    """学生按住说话（vad 帧）：开口满阈值且在老师播报中 → 让出话语权（同一轮只打断一次）。"""
    state = SessionState(session_id="vb1", student_id="s1")
    supervisor = Supervisor(state)
    state.ai_speaking = True                        # 老师正在播报/连讲

    base = 10_000
    supervisor.on_vad(True, base)                   # 第 1 帧：只记语音起点
    supervisor.on_vad(True, base + 100)             # 100ms：未过 VAD_SPEECH_MS
    assert supervisor.voice_barge_in_due() is False, "刚开口（<180ms）不算抢话"

    supervisor.on_vad(True, base + 250)             # 250ms ≥ 180ms：学生在说
    assert supervisor.voice_barge_in_due() is True
    # vad 帧每 100ms 一帧：同一轮说话只打断一次，否则事件会刷屏
    assert supervisor.voice_barge_in_due() is False

    # 松手静默（> VAD_SILENCE_MS）后复位：下一轮开口应能再次打断。
    # 真实链路里 _dispatch 对**每一帧** vad 都会调一次判定（用于复位去重标记），
    # 所以这里也逐帧调用，保持与生产一致的时序。
    supervisor.on_vad(False, base + 400)
    assert supervisor.voice_barge_in_due() is False, "静默未满阈值：仍在说话，不重复打断"
    supervisor.on_vad(False, base + 1300)
    assert supervisor.voice_barge_in_due() is False, "静默达标：标记复位，无需打断"
    supervisor.on_vad(True, base + 2000)
    supervisor.on_vad(True, base + 2250)
    assert supervisor.voice_barge_in_due() is True, "松手后再开口应能再次让出话语权"


async def test_voice_barge_in_due_requires_ai_speaking() -> None:
    """老师空闲时学生开口不算抢话（没人可打断，也不该自增 epoch）。"""
    state = SessionState(session_id="vb2", student_id="s2")
    supervisor = Supervisor(state)

    supervisor.on_vad(True, 1000)
    supervisor.on_vad(True, 1300)
    assert supervisor.voice_barge_in_due() is False


async def test_abort_auto_play_stops_and_notifies() -> None:
    """学生插话：立即中止连讲、同步前端按钮（幂等，连讲未跑时不推事件）。"""
    state = SessionState(session_id="ab1", student_id="s1")
    events: list[dict] = []
    state.emit = events.append
    supervisor = Supervisor(state)

    supervisor.abort_auto_play()                    # 连讲没在跑：什么都不做
    assert events == []

    state.auto_playing = True
    supervisor.abort_auto_play()
    assert state.auto_playing is False
    assert any(
        e["type"] == "slide_play_state"
        and e["payload"]["playing"] is False
        and e["payload"]["finished"] is False
        for e in events
    ), "中止后要推 playing=False，前端按钮才能切回「继续讲」"

    events.clear()
    supervisor.abort_auto_play()                    # 幂等：不会重复推
    assert events == []


async def test_play_slides_aborted_notifies_stop(monkeypatch) -> None:
    """连讲被学生插话中止（epoch 变化）时：也要推 playing=False。

    回归点：此前只有「讲完」和「点暂停」会推状态，插话中止（break）不推，
    前端按钮会一直停在「暂停讲解」（实际已停）。
    """
    state = SessionState(session_id="ab2", student_id="s2")
    events: list[dict] = []
    state.emit = events.append
    supervisor = Supervisor(state)

    async def fake_present(deck, index, *, speak=True):
        state.page_index = index
        return None

    async def fake_wait(index, *, epoch=None):
        state.speech_epoch += 1                     # 学生插话（= SessionLoop._barge_in）
        state.auto_playing = False                  # abort_auto_play 的副作用

    monkeypatch.setattr(supervisor, "_present_slide", fake_present)
    monkeypatch.setattr(supervisor, "_await_page_played", fake_wait)
    deck = SlideDeck(
        deck_id="demo",
        title="演示",
        slides=[Slide(index=0, title="第一页"), Slide(index=1, title="第二页")],
    )

    await supervisor._play_slides(deck, 0)  # noqa: SLF001

    assert state.auto_playing is False
    assert any(
        e["type"] == "slide_play_state"
        and e["payload"]["playing"] is False
        and e["payload"]["finished"] is False
        for e in events
    ), "被插话中止也要同步按钮状态，否则一直显示「暂停讲解」"


# ---------------------------------------------------------------- 课件断点续讲
async def test_slide_event_log_records_and_trims() -> None:
    """断点日志：流式记录 + 进度上报只记进度 + 裁剪只看「继续讲」那一刻。"""
    state = SessionState(session_id="r4", student_id="s4")
    state.deck_id = "d1"
    supervisor = Supervisor(state)

    supervisor._open_slide_event_log("d1", 0)               # noqa: SLF001 - 流式路径：空列表起步
    supervisor._record_slide_event("ai_avatar", {"n": 0})   # noqa: SLF001
    supervisor._record_slide_event("ai_avatar", {"n": 1})   # noqa: SLF001
    assert supervisor._slide_resume_events(9) is None, "页不匹配不可续讲"  # noqa: SLF001

    supervisor.on_slide_pos(9, 5)                           # 页不匹配的上报：忽略
    supervisor.on_slide_pos(0, 1)
    assert [p["n"] for _, p in supervisor._slide_resume_events(0)] == [1]  # noqa: SLF001


async def test_slide_resume_trims_to_reported_pos() -> None:
    """按前端上报的已播句数裁剪剩余；重复上报 / 重复取用都不会重复裁剪（幂等）。"""
    state = SessionState(session_id="r1", student_id="s1")
    state.deck_id = "d1"
    state.page_index = 2
    supervisor = Supervisor(state)

    events = [("ai_avatar", {"n": i}) for i in range(5)]
    supervisor._open_slide_event_log("d1", 2, events=events)   # noqa: SLF001

    supervisor.on_slide_pos(2, 2)      # bargeIn 被本地与 avatar_interrupt 各触发一次 = 重复上报
    supervisor.on_slide_pos(2, 2)
    remaining = supervisor._slide_resume_events(2)             # noqa: SLF001
    assert [p["n"] for _, p in remaining] == [2, 3, 4]

    again = supervisor._slide_resume_events(2)                 # noqa: SLF001 - 未再播放：不丢内容
    assert [p["n"] for _, p in again] == [2, 3, 4]

    supervisor.on_slide_pos(2, 1)      # 续讲里又播了 1 句
    tail = supervisor._slide_resume_events(2)                  # noqa: SLF001
    assert [p["n"] for _, p in tail] == [3, 4]


async def test_handle_slide_play_prefers_resume(monkeypatch) -> None:
    """「继续讲」优先断点续讲；本页已讲完则自动接到下一页。"""
    state = SessionState(session_id="r2", student_id="s2")
    state.deck_id = "d1"
    state.page_index = 1
    supervisor = Supervisor(state)
    deck = SlideDeck(
        deck_id="d1", title="演示",
        slides=[Slide(index=i, title=f"p{i}") for i in range(3)],
    )

    async def fake_deck():
        return deck

    captured: dict = {}

    async def fake_schedule(d, start, resume_events=None):
        captured["start"] = start
        captured["resume"] = resume_events

    monkeypatch.setattr(supervisor, "_current_deck", fake_deck)
    monkeypatch.setattr(supervisor, "_schedule_slide_playback", fake_schedule)

    events = [("ai_avatar", {"n": i}) for i in range(4)]
    supervisor._open_slide_event_log("d1", 1, events=events)   # noqa: SLF001
    supervisor.on_slide_pos(1, 3)                              # 前端：本页已播 3 句

    await supervisor.handle_slide_play(-1)
    assert captured["start"] == 1, "有断点时仍从本页开始（补发剩余）"
    assert [p["n"] for _, p in captured["resume"]] == [3]

    supervisor.on_slide_pos(1, 1)                              # 续讲里把最后一句也播完了
    await supervisor.handle_slide_play(-1)
    assert captured["start"] == 2, "本页已讲完：继续 = 下一页（不再重讲本页）"
    assert captured["resume"] is None


async def test_play_slides_replays_resume_events(monkeypatch) -> None:
    """断点续讲：直接补发剩余语音事件，本页不重新生成/渲染（_present_slide 不被调用）。"""
    state = SessionState(session_id="r3", student_id="s3")
    events: list[dict] = []
    state.emit = events.append
    supervisor = Supervisor(state)
    called = {"present": 0}

    async def fake_present(deck, index, *, speak=True):
        called["present"] += 1
        return None

    async def fake_wait(index, *, epoch=None):
        state.auto_playing = False            # 播完即收工

    monkeypatch.setattr(supervisor, "_present_slide", fake_present)
    monkeypatch.setattr(supervisor, "_await_page_played", fake_wait)
    deck = SlideDeck(
        deck_id="d1", title="演示",
        slides=[Slide(index=0, title="p0"), Slide(index=1, title="p1")],
    )

    resume = [("ai_avatar", {"n": 1}), ("ai_avatar", {"n": 2})]
    await supervisor._play_slides(deck, 1, resume_events=resume)  # noqa: SLF001

    assert called["present"] == 0, "断点续讲不应重讲本页"
    pushed = [e["payload"]["n"] for e in events if e["type"] == "ai_avatar"]
    assert pushed == [1, 2], "剩余事件必须按原顺序补发"


# ---------------------------------------------------------------- 课件续讲/重讲指令
async def test_lecture_command_detection() -> None:
    """「继续/接着讲/再讲一遍」判定：课件打开才生效、只认短指令、长句不劫持。"""
    state = SessionState(session_id="lc1", student_id="s1")
    supervisor = Supervisor(state)

    # 课件未打开：一律不劫持（这些词回落常规链路）
    assert supervisor.lecture_command("请继续") == ""
    assert supervisor.lecture_command("再讲一遍") == ""

    state.deck_id = "d1"
    assert supervisor.lecture_command("请继续") == "resume"
    assert supervisor.lecture_command("继续") == "resume"
    assert supervisor.lecture_command("继续啊") == "resume"
    assert supervisor.lecture_command("那我们继续") == "resume"
    assert supervisor.lecture_command("接着讲吧") == "resume"
    assert supervisor.lecture_command("往下讲") == "resume"
    assert supervisor.lecture_command("再讲一遍") == "replay"
    assert supervisor.lecture_command("重新讲") == "replay"

    # 长句 / 非指令不劫持（应是提问，走答疑链路）
    assert supervisor.lecture_command("继续讲一下这道题为什么选B，我还有点不明白") == ""
    assert supervisor.lecture_command("继续做题") == ""
    assert supervisor.lecture_command("这道题怎么做") == ""
    assert supervisor.lecture_command("") == ""


async def test_resume_lecture_command_triggers_slide_play(monkeypatch) -> None:
    """学生输入「请继续」：走 resume_lecture 分支 → 断点续讲（handle_slide_play(-1)）。"""
    state = SessionState(session_id="lc2", student_id="s2")
    supervisor = Supervisor(state)
    state.deck_id = "d1"
    state.page_index = 3
    calls: list[int] = []

    async def fake_play(index: int = -1) -> None:
        calls.append(index)

    monkeypatch.setattr(supervisor, "handle_slide_play", fake_play)

    response = await supervisor.handle_turn(StudentTurn(text="请继续"))
    assert response is None, "续讲指令不出额外语音（老师直接接着讲）"
    assert calls == [-1], "「继续」应触发断点续讲"

    await supervisor.handle_turn(StudentTurn(text="再讲一遍"))
    assert calls == [-1, 3], "「再讲一遍」应重讲当前页"


# ---------------------------------------------------------------- 出题过渡语
async def test_quiz_lead_prefers_readable_page_title() -> None:
    """出题过渡语：优先带当前页标题（学生听得懂）；含公式/过长则退回通用说法。"""
    state = SessionState(session_id="ql1", student_id="s1")
    supervisor = Supervisor(state)
    state.subject = "数学"

    class _Q:  # _quiz_lead 只读 .topic
        topic = "一元二次方程"

    assert "检验" in supervisor._quiz_lead(_Q()), "无上下文：通用过渡语"

    state.scratch["slide_page_title"] = "配方法"
    assert "配方法" in supervisor._quiz_lead(_Q()), "短页标题应带上"

    state.scratch["slide_page_title"] = "x²+6x=-4 怎么解"
    assert "x²" not in supervisor._quiz_lead(_Q()), "含公式的标题不能念（TTS 会读怪）"
    assert "一元二次方程" in supervisor._quiz_lead(_Q()), "应退回上一题知识点"

    state.scratch["slide_page_title"] = "这一页我们学习配方法的完整推导过程与典型例题讲解"
    assert "这一页" not in supervisor._quiz_lead(_Q()), "过长标题不念"
