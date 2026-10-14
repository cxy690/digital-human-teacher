"""LLM 输出健壮性：JSON 模式 + 坏 JSON 兜底。

为什么单独钉这一组：答疑产物是**上千 token 的 JSON**，而模型偶发把它写坏。
实测（云端 deepseek-v4-flash，长讲解）：

    "steps": ["**读题列条件**：…", ["**先去掉多余的 5**：…   ← 多写了一个 '['

``json.loads`` 直接失败 → 话术退化成原始 JSON 片段（**会被 TTS 念出来**）、
板书被切成乱码。两层防护要能各自独立生效，所以分开测：

1. 源头：JSON 契约的调用带 ``response_format``，交给厂商保证结构合法（不支持则自动降级且只试一次）；
2. 兜底：真坏了也不能把原文当话术，至少把 ``speech`` 那一句捞回来。

另外钉住「白板重点标红」所依赖的前置条件：**JSON 必须能解析出来**，
否则 ``**…**`` 标记根本到不了前端。
"""
from __future__ import annotations

import json
from typing import Any

from agents.shared.reply_parser import (
    looks_like_json,
    parse_json_reply,
    salvage_object,
    salvage_text_field,
)
from agents.teacher.agent import TeacherAgent
from models.base import BaseLLMClient
from models.llm import MockLLM, OpenAICompatLLM

#: 真实坏样本，按云端日志（deepseek-v4-flash）还原：模型在 steps 里多开了一个 `[`
#: 又**忘了闭合**，接着就去写 `"knowledge_points"` —— 报错正是
#: ``Expecting ',' delimiter``（与日志里的 char 577 同型）。
#: 注意别把样本写成 `["a", ["b","c"]]` —— 那是**合法**的嵌套数组，测不出问题。
BROKEN = (
    '{"speech": "这道题的解是 x 等于 5。我们把它想成一架天平。",'
    ' "title": "解一元一次方程",'
    ' "steps": ["**读题列条件**：题目是 3x+5=20。",'
    ' ["**先去掉多余的 5**：两边同时减 5。", "**为什么能这么减**：两边同减同一个数。",'
    ' "knowledge_points": ["等式的性质"]}'
)


# ---------------------------------------------------------------- 兜底：坏 JSON
def test_broken_json_is_detected_and_speech_salvaged() -> None:
    assert parse_json_reply(BROKEN) == {}, "结构已坏，必须判为解析失败"
    assert looks_like_json(BROKEN) is True
    assert salvage_text_field(BROKEN, "speech") == "这道题的解是 x 等于 5。我们把它想成一架天平。"


def test_looks_like_json_separates_prose_from_broken_json() -> None:
    """散文回复不该被当成「坏 JSON」，否则正常兜底路径会被抢走。"""
    assert looks_like_json("我们先看这道题的已知条件。") is False
    assert looks_like_json('  {"speech": "…"}') is True


def test_salvage_handles_escapes_and_missing_field() -> None:
    assert salvage_text_field('{"speech": "第一行\\n第二行"}') == "第一行\n第二行"
    assert salvage_text_field('{"title": "没有 speech 字段"}') == ""


def test_compose_response_never_reads_raw_json_aloud() -> None:
    """坏 JSON 也不能让老师把 ``{"speech": …`` 念出来。"""
    teacher = TeacherAgent(llm=MockLLM())

    response = teacher._compose_response(BROKEN)  # noqa: SLF001 - 直接验证产物组装

    assert "{" not in response.speech and '"speech"' not in response.speech
    assert response.speech.startswith("这道题的解是 x 等于 5"), "能救回来的话术必须救回来"
    assert response.meta.get("json_degraded") is True
    assert response.board_ops == [], "结构坏了就别硬写板书（切出来的分步是乱的）"


def test_compose_response_keeps_markers_when_json_is_fine() -> None:
    """JSON 正常时：标记留在板书（供标红），话术里没有星号。"""
    teacher = TeacherAgent(llm=MockLLM())
    good = (
        '{"speech": "移项要**变号**。", "title": "解方程",'
        ' "steps": ["两边同时减 5，得 **3x=15**"], "knowledge_points": [], "pitfalls": []}'
    )

    response = teacher._compose_response(good)  # noqa: SLF001

    assert response.meta.get("json_degraded") is None
    assert "**" not in response.speech
    assert any("**3x=15**" in (op.text or "") for op in response.board_ops)


def test_compose_response_still_handles_plain_prose() -> None:
    """模型给的是散文（不是 JSON）：走原有兜底，不进 json_degraded 分支。"""
    teacher = TeacherAgent(llm=MockLLM())

    response = teacher._compose_response("我们先把已知条件列出来。第二步再代入求值。")

    assert response.speech
    assert response.meta.get("json_degraded") is None


# ---------------------------------------------------------------- 修复重发
VALID = (
    '{"speech": "移项要变号。", "title": "解方程",'
    ' "steps": ["两边同时减 5，得 **3x=15**"], "knowledge_points": [], "pitfalls": []}'
)


class _RepairLLM(BaseLLMClient):
    """固定返回一段文本（修复调用用的替身），并记录被调用了几次。"""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[str] = []

    async def chat(
        self,
        messages: Any,
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        self.calls.append(str(messages[-1].get("content", "")))
        return self.reply


class _BoomLLM(BaseLLMClient):
    """一旦被调用就抛错：用来证明「就地抢救」没有多花模型调用。"""

    def __init__(self) -> None:
        self.calls = 0

    async def chat(
        self,
        messages: Any,
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        self.calls += 1
        raise AssertionError("本地抢救不该调用模型")


def test_salvage_recovers_arrays_split_by_broken_brackets() -> None:
    """就地抢救：把「多开了一个 `[` 又没闭合」的数组按字段名抠回来（嵌套数组摊平）。"""
    salvaged = salvage_object(BROKEN)

    assert salvaged["speech"] == "这道题的解是 x 等于 5。我们把它想成一架天平。"
    assert salvaged["steps"] == [
        "**读题列条件**：题目是 3x+5=20。",
        "**先去掉多余的 5**：两边同时减 5。",
        "**为什么能这么减**：两边同减同一个数。",
    ]
    assert salvaged["knowledge_points"] == ["等式的性质"], "坏在 steps 里也要能把后面的字段抓出来"
    # 抢救结果必须是**可再次解析**的合法 JSON（后续走正常组装链路）
    assert parse_json_reply(json.dumps(salvaged, ensure_ascii=False)) != {}


async def test_broken_json_is_salvaged_locally_without_extra_llm_call() -> None:
    """坏 JSON 先本地抢救：板书要保住，而且**不多花一次模型调用**。

    对比：让模型「照原文重发一遍合法 JSON」也是 1000+ 字长输出，实测照样会写坏
    （云端日志出现过「重发后 1023 字仍然损坏」），所以本地扫描才是主力。
    """
    llm = _BoomLLM()
    teacher = TeacherAgent(llm=llm)

    response = await teacher.compose_answer(BROKEN)

    assert llm.calls == 0, "本地能救就不该再调模型"
    assert response.meta.get("json_degraded") is None
    texts = [op.text or "" for op in response.board_ops]
    assert any("**读题列条件**" in t for t in texts), "抢救回来的步骤要上板书"
    assert any("等式的性质" in t for t in texts), "知识点也要上板书"


async def test_repair_is_skipped_for_valid_json() -> None:
    llm = _RepairLLM("不应被使用")
    teacher = TeacherAgent(llm=llm)

    response = await teacher.compose_answer(VALID)

    assert llm.calls == [], "合法 JSON 不该多花一次模型调用"
    assert response.speech == "移项要变号。"


async def test_model_repair_used_when_local_salvage_finds_nothing() -> None:
    """本地抠不出字段（不是数组括号那种坏法）时，才请模型照原文重发。"""
    llm = _RepairLLM(VALID)
    teacher = TeacherAgent(llm=llm)

    response = await teacher.compose_answer('{"foo": 1, "bar": ')   # 没有 speech/steps 可救

    assert len(llm.calls) == 1, "本地救不出来时应当重发一次"
    assert "只修语法" in llm.calls[0], "修复提示词必须强调「只改语法、不改内容」"
    assert '"foo"' in llm.calls[0], "原文要原样带进修复提示词（否则等于让模型重写）"
    assert response.speech == "移项要变号。"


async def test_fallback_when_everything_fails() -> None:
    """本地救不出、重发也还是救不出：老实走兜底（话术能救就救，板书宁可不写）。"""
    llm = _RepairLLM('{"bar": ')
    teacher = TeacherAgent(llm=llm)

    response = await teacher.compose_answer('{"foo": 1, "bar": ')

    assert response.meta.get("json_degraded") is True
    assert response.board_ops == []
    assert response.speech and "{" not in response.speech


# ---------------------------------------------------------------- 源头：JSON 模式
class _FakeResponse:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx

            request = httpx.Request("POST", "http://fake/chat/completions")
            raise httpx.HTTPStatusError(
                "boom",
                request=request,
                response=httpx.Response(self.status_code, request=request),
            )

    def json(self) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "ok"}}]}


def _patch_httpx(monkeypatch: Any, statuses: list[int], captured: list[dict[str, Any]]) -> None:
    """把 httpx.AsyncClient 换成假的：记录每次请求 payload，按给定状态码依次返回。"""
    queue = list(statuses)

    class _FakeClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "_FakeClient":
            return self

        async def __aexit__(self, *_exc: Any) -> bool:
            return False

        async def post(self, _url: str, *, json: dict[str, Any], headers: Any) -> _FakeResponse:
            captured.append(json)
            return _FakeResponse(queue.pop(0) if queue else 200)

    monkeypatch.setattr("models.llm.httpx.AsyncClient", _FakeClient)


async def test_json_mode_sends_response_format(monkeypatch: Any) -> None:
    captured: list[dict[str, Any]] = []
    _patch_httpx(monkeypatch, [200], captured)
    llm = OpenAICompatLLM(api_base="http://fake", api_key="k", model="m")

    assert await llm.chat([{"role": "user", "content": "hi"}], json_mode=True) == "ok"
    assert captured[0]["response_format"] == {"type": "json_object"}


async def test_plain_call_has_no_response_format(monkeypatch: Any) -> None:
    captured: list[dict[str, Any]] = []
    _patch_httpx(monkeypatch, [200], captured)
    llm = OpenAICompatLLM(api_base="http://fake", api_key="k", model="m")

    await llm.chat([{"role": "user", "content": "hi"}])

    assert "response_format" not in captured[0]


async def test_json_mode_falls_back_and_remembers(monkeypatch: Any) -> None:
    """厂商不认 response_format：当次退回普通请求，且**后续不再尝试**（省一次 4xx 往返）。"""
    captured: list[dict[str, Any]] = []
    _patch_httpx(monkeypatch, [400, 200, 200], captured)
    monkeypatch.setattr(OpenAICompatLLM, "supports_json_mode", True)
    llm = OpenAICompatLLM(api_base="http://fake", api_key="k", model="m")

    assert await llm.chat([{"role": "user", "content": "hi"}], json_mode=True) == "ok"
    assert "response_format" in captured[0], "第一次应当尝试 JSON 模式"
    assert "response_format" not in captured[1], "被拒绝后当次请求要退回普通模式"
    assert OpenAICompatLLM.supports_json_mode is False, "要记住厂商不支持"

    assert await llm.chat([{"role": "user", "content": "hi"}], json_mode=True) == "ok"
    assert "response_format" not in captured[2], "后续请求不该再白撞一次 4xx"


async def test_mock_llm_accepts_json_mode() -> None:
    """Mock 必须能吃下 json_mode（离线链路与单测都靠它）。"""
    text = await MockLLM().chat([{"role": "user", "content": "讲题"}], json_mode=True)
    assert text
