"""回归：一堂课里老师的音色必须恒定，不能「讲着讲着换了个声音」。

背景（线上真实问题）
--------------------
云端 ``TTS_PROVIDER=cosyvoice``，装配出来的是
``TeacherTTS(clone=CosyVoice@:8090, edge=EdgeTTS)``，而 20 位老师的档案里
``voice_engine`` 全是 ``edge``。旧实现里 edge **只要失败一次**就会：

1. 给该老师打上 300 秒熔断（``_edge_dead``）；
2. 这 300 秒内每一句都改走克隆引擎 —— 音色与 edge 音色完全不同；
3. 而且 ``begin_utterance`` 每段都清锁，熔断期一过又跳回 edge。

学生听到的就是「一开始一个音色讲课，后面又是另一个音色」。

现在的约定
----------
* 引擎**在一位老师的一堂课内定死**，``begin_utterance`` 不再清锁；
* edge 失败先**用同一音色重试**，吸收偶发抖动；
* 重试仍失败：未达阈值就**本句静音但保持音色**；只有「本课从未成功过」或
  「连续失败达到 ``EDGE_SWITCH_AFTER_FAILS``」才切一次克隆并永久锁定。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from models.base import BaseTTSClient
from models.tts import TeacherTTS

EDGE_AUDIO = b"EDGE"
CLONE_AUDIO = b"CLONE"


class _FakeEdge(BaseTTSClient):
    """可开关的假 edge：``down=True`` 时每次调用都失败。"""

    def __init__(self) -> None:
        self.calls = 0
        self.down = False

    async def synthesize(self, text: str, *, teacher_id: str | None = None) -> bytes:
        self.calls += 1
        if self.down:
            raise RuntimeError("edge down")
        return EDGE_AUDIO


class _FakeClone(BaseTTSClient):
    def __init__(self) -> None:
        self.calls = 0

    async def synthesize(self, text: str, *, teacher_id: str | None = None) -> bytes:
        self.calls += 1
        return CLONE_AUDIO


@pytest.fixture()
def tts(monkeypatch: pytest.MonkeyPatch) -> tuple[TeacherTTS, _FakeEdge, _FakeClone]:
    """构造 TeacherTTS：假 edge + 假克隆，并把教师档案固定为「用 edge 音色」。"""
    monkeypatch.setattr(
        "config.teachers.load_teacher",
        lambda key: SimpleNamespace(key=key, voice_engine="edge"),
    )
    edge, clone = _FakeEdge(), _FakeClone()
    engine = TeacherTTS(clone=clone, edge=edge)
    engine.EDGE_RETRY_SLEEP = 0.0        # 测试里不要真的 sleep
    return engine, edge, clone


async def test_edge_normal_keeps_one_voice(tts) -> None:
    """edge 正常：整段都用 edge，克隆引擎一次都不碰。"""
    engine, edge, clone = tts
    for _ in range(4):
        assert await engine.synthesize("你好", teacher_id="english") == EDGE_AUDIO
    assert edge.calls == 4
    assert clone.calls == 0
    assert engine._lock["english"] == "edge"


async def test_transient_failure_is_absorbed_by_retry(tts) -> None:
    """抖动：第一次尝试失败、重试成功 -> 音色不变，也不切引擎。"""
    engine, edge, clone = tts
    assert await engine.synthesize("第一句", teacher_id="english") == EDGE_AUDIO
    # 让「第 1 次尝试失败、第 2 次成功」：用一个只失败一次的包装
    # （注意不要同时置 edge.down —— 那会让重试也失败，测的就不是「抖动」了）
    real = edge.synthesize
    state = {"n": 0}

    async def flaky(text: str, *, teacher_id: str | None = None) -> bytes:
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("blip")
        return await real(text, teacher_id=teacher_id)

    edge.synthesize = flaky  # type: ignore[method-assign]
    assert await engine.synthesize("第二句", teacher_id="english") == EDGE_AUDIO
    assert clone.calls == 0
    assert engine._lock["english"] == "edge"


async def test_flapping_does_not_change_voice(tts) -> None:
    """**核心回归**：已用 edge 发声后，整句失败也不许换音色。

    旧实现在这里会切到克隆引擎（音色立刻变），并在 300 秒内一直用克隆。
    """
    engine, edge, clone = tts
    assert await engine.synthesize("开场", teacher_id="english") == EDGE_AUDIO

    edge.down = True                     # 一整句都失败（含全部重试）
    got = await engine.synthesize("这一句会失败", teacher_id="english")
    assert got == b"", "未达阈值时应保持音色、本句不下发音频"
    assert clone.calls == 0, "不许因为一次抖动就换音色"
    assert engine._lock["english"] == "edge"

    edge.down = False                    # edge 恢复
    assert await engine.synthesize("又回来了", teacher_id="english") == EDGE_AUDIO
    assert clone.calls == 0, "恢复后仍必须是同一个音色"


async def test_never_spoke_switches_once(tts) -> None:
    """本课一次都没成功过（开课就断网）：切一次克隆并永久锁定，避免整堂哑巴。"""
    engine, edge, clone = tts
    edge.down = True
    assert await engine.synthesize("开场白", teacher_id="english") == CLONE_AUDIO
    assert engine._lock["english"] == "clone"
    assert edge.calls == engine.EDGE_ATTEMPTS, "切走前应先把重试机会用完"

    edge.down = False                    # 即使 edge 恢复也不再切回（音色已定）
    assert await engine.synthesize("第二句", teacher_id="english") == CLONE_AUDIO
    assert engine._lock["english"] == "clone"


async def test_persistent_failure_switches_once_then_sticky(tts) -> None:
    """已发声但持续故障：连续失败达阈值才切一次，之后永久锁定。"""
    engine, edge, clone = tts
    assert await engine.synthesize("开场", teacher_id="english") == EDGE_AUDIO

    edge.down = True
    # 前 EDGE_SWITCH_AFTER_FAILS-1 句：保持音色、不下发音频
    for i in range(engine.EDGE_SWITCH_AFTER_FAILS - 1):
        assert await engine.synthesize(f"失败{i}", teacher_id="english") == b""
        assert clone.calls == 0

    # 达到阈值那一句：切克隆
    assert await engine.synthesize("阈值句", teacher_id="english") == CLONE_AUDIO
    assert engine._lock["english"] == "clone"

    # 永久粘住：edge 恢复也不回头（否则又是一次换声）
    edge.down = False
    assert await engine.synthesize("之后", teacher_id="english") == CLONE_AUDIO
    assert engine._lock["english"] == "clone"


async def test_begin_utterance_does_not_reset_voice(tts) -> None:
    """begin_utterance 由编排层每段调用 —— 它必须**不能**清掉音色锁定。"""
    engine, edge, clone = tts
    assert await engine.synthesize("第一段", teacher_id="english") == EDGE_AUDIO
    engine.begin_utterance("english")
    assert engine._lock.get("english") == "edge", "begin_utterance 不应解除音色锁定"
    assert await engine.synthesize("第二段", teacher_id="english") == EDGE_AUDIO


async def test_reset_voice_allows_reshuffle_for_new_lesson(tts) -> None:
    """换老师 / 重新开课时，reset_voice 允许重新选一次引擎。"""
    engine, edge, clone = tts
    edge.down = True
    assert await engine.synthesize("开场", teacher_id="english") == CLONE_AUDIO
    engine.reset_voice("english")
    edge.down = False
    assert await engine.synthesize("新课第一句", teacher_id="english") == EDGE_AUDIO
    assert engine._lock["english"] == "edge"
