"""模型层抽象基类：统一的 LLM / ASR / TTS / 视觉 / 数字人接口，便于 mock 与替换供应商。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

# OpenAI 风格消息：{"role": "...", "content": "..."} 或含多模态 content 列表
ChatMessages = list[dict[str, Any]]


@dataclass
class ToolCall:
    """模型请求调用的一次工具（对应 OpenAI ``message.tool_calls`` 里的一项）。

    ``raw`` 保留服务端原样返回的字典：ReAct 闭环要把 assistant 消息（含 ``tool_calls``）
    原封不动回填进下一轮 messages，自己拼一遍容易漏字段（``type`` / ``index`` 等）。
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": dict(self.arguments)}


@dataclass
class ToolChatResult:
    """一次「可带工具」的对话补全结果。"""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""

    @property
    def wants_tools(self) -> bool:
        """模型是否要求先调工具（True = 还没给出最终答复）。"""
        return bool(self.tool_calls)


class BaseLLMClient(ABC):
    """大模型客户端接口（OpenAI 兼容协议为默认实现）。"""

    is_mock: bool = False
    #: 是否真的支持原生函数调用。Mock / 老供应商为 False，编排层据此决定是否走 ReAct。
    supports_tools: bool = False

    @abstractmethod
    async def chat(
        self,
        messages: ChatMessages,
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        """普通对话补全，返回文本。

        ``json_mode=True``：要求模型返回**合法 JSON**（OpenAI 兼容的
        ``response_format={"type": "json_object"}``）。

        为什么需要它：答疑/讲课的产物是一段上千 token 的 JSON，长输出里模型
        偶发写坏结构（实测 ``"steps": ["a", ["b", …`` 多一个 ``[``），一旦非法，
        整段解析失败 —— 话术退化成原始 JSON 片段（会被念出来）、板书变成乱切的分步。
        厂商不支持时自动退回普通请求（见 ``OpenAICompatLLM.chat``），不支持也不报错。
        """

    async def chat_stream(
        self,
        messages: ChatMessages,
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """流式对话补全，逐段产出文本增量（SSE）。

        默认实现降级为一次性产出完整文本（Mock / 不支持流式的供应商），
        真流式由 OpenAI 兼容子类覆写。
        """
        yield await self.chat(messages, temperature=temperature, max_tokens=max_tokens)

    async def chat_json(
        self,
        messages: ChatMessages,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """要求模型输出 JSON。默认实现直接从文本中解析，子类可覆写为结构化输出。"""
        import json
        import re

        text = await self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    async def chat_with_tools(
        self,
        messages: ChatMessages,
        *,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> ToolChatResult:
        """带工具的对话补全（原生 function calling）。

        默认实现**退化为普通对话**（``tool_calls`` 恒为空）：不支持函数调用的供应商
        （Mock / 老 OpenAI 兼容实现）不必实现本方法，编排层可以无条件调用它 ——
        退化成「纯文本生成」正好等价于加装工具之前的行为，调用处不需要写 if。

        ``json_mode``：要求**最终答复**是合法 JSON（ReAct 闭环的最后一句必须是 JSON 契约）。
        """
        text = await self.chat(
            messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
        )
        return ToolChatResult(content=text)


class BaseASRClient(ABC):
    """语音识别客户端接口。"""

    is_mock: bool = False

    @abstractmethod
    async def transcribe(self, audio: bytes, *, sample_rate: int = 16000) -> str:
        ...


class BaseTTSClient(ABC):
    """语音合成客户端接口。"""

    is_mock: bool = False

    @abstractmethod
    async def synthesize(self, text: str, *, teacher_id: str | None = None) -> bytes:
        """合成一句话；``teacher_id`` 指定用哪位老师的音色（None=默认音色）。"""

    async def synthesize_stream(
        self, text: str, *, teacher_id: str | None = None
    ) -> AsyncIterator[tuple[bytes, str]]:
        """按句切片流式合成，产出 ``(音频片段, 句子文本)``。

        默认实现：逐句调用 :meth:`synthesize`（多数 HTTP TTS 本就是按请求合成，
        按句切分即可获得首句低延迟）；支持字节流协议的子类可覆写为真流式。
        """
        from models.tts import split_sentences  # 局部导入避免循环依赖

        for sentence in split_sentences(text):
            yield await self.synthesize(sentence, teacher_id=teacher_id), sentence


class BaseAvatarClient(ABC):
    """数字人引擎接口：把 TTS 音频渲染成教师形象视频。

    * ``produces_video=False``（如 MockAvatar）：服务端不渲染，
      前端走「照片木偶」模式（能量驱动口型 + 视素调制 + 眨眼）；
    * ``produces_video=True``（如 MuseTalk 类引擎）：服务端逐句渲染视频，
      通过 ``ai_avatar`` 事件下发给学生端播放。
    """

    is_mock: bool = False
    produces_video: bool = False

    @abstractmethod
    async def render(
        self, audio: bytes, *, text: str = "", gesture: str | None = None,
        teacher_id: str | None = None,
    ) -> dict[str, Any]:
        """渲染一句音频对应的数字人画面，返回可 JSON 序列化的结果字典。

        ``gesture`` 为可选的手势基元名（如 nod/shake/shrug/point），由编排层根据
        对话状态（答对/答错/指向课件等）下发；不支持的引擎可忽略。
        ``teacher_id`` 为多老师场景下的教师标识（= 学科 key），引擎据此切换底片；
        为空表示沿用引擎默认底片（向后兼容）。
        """

